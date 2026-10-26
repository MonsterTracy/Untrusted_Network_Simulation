"""Guarded vLLM service for an execution permit, never the synthetic 38-call service."""

import argparse
import inspect
import os
from pathlib import Path
import sys
import time

from werewolf.tom_llm.common import digest, require
from werewolf.tom_llm.experiment import open_run
from werewolf.tom_llm.model_spec import ModelSpec, load_cpu_tokenizer, render
from werewolf.tom_llm.live_admission import open_permit
from werewolf.tom_llm.deployment import require_idle_gpu, validate_cache_paths, validate_process_environment, driver_version
from werewolf.tom_llm.gpu_probe import JsonlSink, lifetime, snapshot_sampling, validate_launch_args
from werewolf.tom_llm.qualification_evidence import validate_sampling


def install_live_probe(engine, sink, *, tasks, spec, tokenizer, permit_digest, process_lifetime):
    require(not getattr(engine,'_mm_live_probe',False),'live probe already installed')
    add,generate=engine._add_request,engine.generate
    require(inspect.iscoroutinefunction(add),'unsupported final input seam')
    common={'permit_digest':permit_digest,'process_lifetime_digest':digest(process_lifetime)}
    state={'ordinal':0,'inflight':{},'finished':set()}
    sink({**common,'kind':'startup','task_digests':[digest(t) for t in tasks]})
    async def audited_add(core,*args,**kwargs):
        i=state['ordinal'];require(i<len(tasks) and not state['inflight'],'unexpected/concurrent engine request')
        task=tasks[i];text=render(task,spec)['text'];ids=tokenizer.encode(text,add_special_tokens=False)
        require(ids==tokenizer.apply_chat_template(task['messages'],tokenize=True,return_dict=False,add_generation_prompt=True,
            **spec.record()['chat_template']['kwargs'])==list(core.prompt_token_ids), 'final native input_ids drift')
        require(core.prompt_embeds is None and not core.mm_features and core.pooling_params is None and core.lora_request is None
            and core.cache_salt is None and core.reasoning_ended is None and core.reasoning_parser_kwargs is None
            and not core.resumable and core.priority==0,'undeclared engine modifiers')
        sampling=snapshot_sampling(core.sampling_params);validate_sampling(sampling,task['response_schema'],spec)
        key=core.external_req_id;require(type(key) is str and key and key not in state['finished'],'engine request ID reuse')
        sink({**common,'kind':'input','ordinal':i,'task_digest':digest(task),
              'engine_request_id':key,'input_ids':ids,'sampling':sampling})
        state['ordinal']+=1;state['inflight'][key]=(i,time.perf_counter())
        return await add(core,*args,**kwargs)
    async def audited_generate(*args,**kwargs):
        async for output in generate(*args,**kwargs):
            if output.finished:
                key=output.request_id;require(key in state['inflight'] and len(output.outputs)==1,'unexpected engine output')
                i,start=state['inflight'].pop(key);choice=output.outputs[0]
                tokens=list(choice.token_ids);require(len(tokens)<=512 and all(type(v) is int and 0<=v<len(tokenizer) for v in tokens),'invalid output tokens')
                if spec.key=='qwen3':require(not {151667,151668}&set(tokens),'Non-thinking output drift')
                require(choice.stop_reason is None or choice.stop_reason in set(_eos(spec)), 'unexpected stop reason')
                sink({**common,'kind':'output','ordinal':i,'task_digest':digest(tasks[i]),
                    'engine_request_id':key,'output_token_ids':tokens,'content':choice.text,'finish_reason':choice.finish_reason,
                    'stop_reason':choice.stop_reason,'engine_wall_seconds':time.perf_counter()-start})
                state['finished'].add(key)
            yield output
    engine._add_request,engine.generate=audited_add,audited_generate;engine._mm_live_probe=True
    return state


def _eos(spec):
    value=spec.record()['termination']['generation_eos_token_id']
    return (value if type(value) is list else [value])+[spec.record()['tokenizer']['eos_token_id']]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--model',required=True,choices=['qwen3','yi','mistral'])
    p.add_argument('--expected-permit',required=True);p.add_argument('--execute-live',action='store_true')
    a=p.parse_args();require(a.execute_live and sys.platform=='linux','explicit authorized Ubuntu GPU launch required')
    permit,r=open_permit(a.run,a.model,a.expected_permit)
    _,run,tasks=open_run(a.run)
    from werewolf.tom_llm.live_runner import pending_tasks
    tasks=pending_tasks(a.run,a.model,tasks)  # unknown outcomes fail BEFORE loading vLLM
    profile=r['profile'];validate_process_environment(profile,os.environ);validate_cache_paths(profile,require_existing=True)
    require_idle_gpu(profile)
    require(r['qualified_runtime']=={'driver':driver_version(),'kernel':os.uname().release},'driver/kernel changed since qualification')
    tokenizer=load_cpu_tokenizer(ModelSpec(a.model),r['tokenizer_assets'])
    from vllm.entrypoints.openai import api_server
    import uvloop
    api_server.cli_env_setup()
    args=api_server.make_arg_parser(api_server.FlexibleArgumentParser()).parse_args(['--config',str(permit.path/'vllm.yaml')])
    api_server.validate_parsed_serve_args(args);validate_launch_args(args,profile)
    process=lifetime();sink=JsonlSink(Path(profile['engine_log_root'])/digest(process)/'engine-events.jsonl')
    original=api_server.init_app_state
    async def init(engine,state,args,*rest,**kwargs):
        result=await original(engine,state,args,*rest,**kwargs);validate_launch_args(args,profile)
        install_live_probe(engine,sink,tasks=tasks,spec=ModelSpec(a.model),tokenizer=tokenizer,permit_digest=permit.manifest_digest,process_lifetime=process)
        return result
    api_server.init_app_state=init
    try:uvloop.run(api_server.run_server(args))
    finally:sink.close()


if __name__=='__main__':main()
