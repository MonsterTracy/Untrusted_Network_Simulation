"""Explicit opt-in vLLM synthetic service; importing never starts GPU or HTTP work.

Ported selectively from W5 probe seams; all identities/artifact domains are new.
"""

import argparse
import inspect
import os
from pathlib import Path
import sys

from werewolf.artifact_io import canonical_json_bytes
from werewolf.tom_llm.common import digest, require, strict_json
from werewolf.tom_llm.deployment import open_deployment, require_idle_gpu, validate_process_environment, validate_cache_paths
from werewolf.tom_llm.model_spec import ROOT, ModelSpec
from werewolf.tom_llm.qualification_evidence import validate_sampling, CAPTURE_SITE
from werewolf.tom_llm.qualification_inputs import SCHEDULE, validate_suite


def snapshot_sampling(params):
    keys=['n','temperature','top_p','top_k','min_p','max_tokens','seed','presence_penalty',
          'frequency_penalty','repetition_penalty','min_tokens','stop','stop_token_ids','ignore_eos']
    result={k:getattr(params,k) for k in keys}
    result['eos_token_ids']=sorted(params._all_stop_token_ids)
    result['primary_eos_token_id']=params._eos_token_id
    structured=getattr(params,'structured_outputs',None)
    require(structured is not None,'missing actual structured-output SamplingParams')
    schema=structured.json
    result['json_schema']=strict_json(schema) if type(schema) is str else schema
    for k in ('logit_bias','allowed_token_ids','bad_words','logits_processors'):result[k]=getattr(params,k,None)
    return result


def lifetime():
    pid=os.getpid();proc=Path('/proc')
    return {'pid':pid,'boot_id':(proc/'sys/kernel/random/boot_id').read_text().strip(),
            'start_ticks':(proc/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19]}


class JsonlSink:
    def __init__(self,path):
        p=Path(path)
        require(not any(q.is_symlink() for q in (p,*p.parents)),'probe log path uses symlink')
        p.parent.mkdir(parents=True,exist_ok=True)
        require(not any(q.is_symlink() for q in (p,*p.parents)),'probe log path changed to symlink')
        self.fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)

    def __call__(self,event):
        raw=canonical_json_bytes(event)+b'\n'
        while raw:
            n=os.write(self.fd,raw);require(n>0,'audit write failed');raw=raw[n:]
        os.fsync(self.fd)

    def close(self):os.close(self.fd)


def install_probe(engine,sink,*,deployment,process_lifetime):
    spec=ModelSpec(deployment['key']);suite=validate_suite(deployment['suite'],spec)
    require(not getattr(engine,'_mm_qualification_probe',False),'probe already installed')
    add,generate=engine._add_request,engine.generate
    require(inspect.iscoroutinefunction(add),'unsupported vLLM final-input seam')
    inputs={};outputs=set()
    common={'deployment_digest':digest(deployment),'model_manifest_digest':digest(deployment['model']),
            'process_lifetime_digest':digest(process_lifetime)}
    sink({**common,'kind':'startup','argv_options':deployment['profile']['argv_options'],
          'frontend':deployment['profile']['frontend'],'sdk_hidden_retries':False,'engine_request_retries':0})

    async def audited_add(core,*args,**kwargs):
        ordinal=len(inputs);require(ordinal<38,'synthetic engine dispatch budget exhausted')
        case=suite['cases'][SCHEDULE[ordinal][1]]
        require(core.prompt_embeds is None and not core.mm_features and core.pooling_params is None
                and core.lora_request is None and core.cache_salt is None and core.reasoning_ended is None
                and core.reasoning_parser_kwargs is None and not core.resumable and core.priority==0,
                'undeclared engine request modifiers')
        ids=list(core.prompt_token_ids)
        require(ids==case['input_ids'],'final engine input_ids/synthetic schedule drift')
        sampling=snapshot_sampling(core.sampling_params)
        validate_sampling(sampling,case['task']['response_schema'],spec)
        external=core.external_req_id
        require(type(external) is str and external and external not in inputs,'duplicate/absent request ID')
        record={**common,'kind':'input','case_index':case['case_index'],'engine_request_id':external,
                'internal_request_id':core.request_id,'capture_site':CAPTURE_SITE,'input_ids':ids,'sampling':sampling}
        sink(record);inputs[external]=record  # Durable BEFORE engine enqueue. No replay API.
        return await add(core,*args,**kwargs)

    async def audited_generate(*args,**kwargs):
        try:
            async for output in generate(*args,**kwargs):
                if output.finished:
                    key=output.request_id
                    require(key in inputs and key not in outputs and len(output.outputs)==1,'engine output identity/count mismatch')
                    choice=output.outputs[0]
                    sink({**common,'kind':'output','case_index':inputs[key]['case_index'],'engine_request_id':key,
                          'output_token_ids':list(choice.token_ids),'content':choice.text,
                          'finish_reason':choice.finish_reason,'stop_reason':choice.stop_reason})
                    outputs.add(key)
                yield output
        except Exception as error:
            sink({**common,'kind':'infrastructure_error','exception_type':type(error).__name__})
            raise
    engine._add_request,engine.generate=audited_add,audited_generate
    engine._mm_qualification_probe=True


def validate_launch_args(parsed,profile):
    for key,want in profile['argv_options'].items():
        actual=getattr(parsed,key.replace('-','_'))
        if key=='structured-outputs-config':
            actual={'backend':actual['backend'] if type(actual) is dict else actual.backend}
        require(actual==want,'effective server argument drift: '+key)
    require(parsed.quantization is None and parsed.reasoning_parser is None,'undeclared quantization/reasoning parser')


def start_deployment(root, expected, *, execute=False):
    """Foreground exec of the existing probe, with its exact scoped environment."""
    require(execute and sys.platform=='linux','separately authorized Linux GPU operation required')
    artifact,record=open_deployment(root,expected)
    p=record['profile']
    require(artifact.path==Path(p['root'])/'deployment' and record['source']['source_inventory_clean'],
            'committed Source and isolated deployment path required')
    validate_cache_paths(p)
    from werewolf.tom_llm.deployment import CACHE_VARIABLES
    for key in CACHE_VARIABLES:Path(p['environment_requirements'][key]).mkdir(parents=True,exist_ok=True)
    validate_cache_paths(p,require_existing=True)
    env={key:os.environ[key] for key in ('HOME','USER','LANG','LC_ALL') if key in os.environ}
    env.update(p['environment_requirements'])
    python=p['environment_prefix']+'/bin/python'
    os.chdir(ROOT)
    os.execve(python,[python,'-m','werewolf.tom_llm.gpu_probe','--deployment',str(artifact.path),
        '--expected-deployment',expected,'--execute-synthetic-gpu-qualification'],env)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment',required=True);parser.add_argument('--expected-deployment',required=True)
    parser.add_argument('--execute-synthetic-gpu-qualification',action='store_true')
    args=parser.parse_args()
    require(args.execute_synthetic_gpu_qualification and sys.platform=='linux','separately authorized Linux GPU operation required')
    artifact,r=open_deployment(args.deployment,args.expected_deployment,physical_recheck=True)
    p=r['profile'];root=Path(p['root'])
    require(artifact.path==root/'deployment' and r['source']['source_inventory_clean'],'committed Source and isolated deployment path required')
    validate_process_environment(p,os.environ)
    validate_cache_paths(p,require_existing=True)  # Prevent tempfile/JIT fallback to system /tmp.
    require(Path(p['argv_options']['chat-template']).read_bytes()==(artifact.path/'chat_template.jinja').read_bytes(),'fixed service template drift')
    require_idle_gpu(p)
    # No vLLM import or server start before all explicit gates above.
    from vllm.entrypoints.openai import api_server
    import uvloop
    api_server.cli_env_setup()
    parsed=api_server.make_arg_parser(api_server.FlexibleArgumentParser()).parse_args(['--config',str(artifact.path/'vllm.yaml')])
    api_server.validate_parsed_serve_args(parsed);validate_launch_args(parsed,p)
    process=lifetime();sink=JsonlSink(Path(p['engine_log_root'])/digest(process)/'engine-events.jsonl')
    original=api_server.init_app_state
    async def init(engine_client,state,args,*rest,**kwargs):
        result=await original(engine_client,state,args,*rest,**kwargs)
        validate_launch_args(args,p)
        install_probe(engine_client,sink,deployment=r,process_lifetime=process)
        return result
    api_server.init_app_state=init
    try:uvloop.run(api_server.run_server(parsed))
    finally:sink.close()


if __name__=='__main__':main()
