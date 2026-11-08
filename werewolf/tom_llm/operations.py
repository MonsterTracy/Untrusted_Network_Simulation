"""轻量配置编排；仅调用单一执行器和服务所有权模块。"""

import json
from pathlib import Path
import re
import sys

from werewolf.tom_llm.common import require,strict_json
from werewolf.tom_llm.experiment import safe_path,prepare_run,open_run
from werewolf.tom_llm.model_spec import ModelSpec


def load_config(path):
    path = safe_path(path)
    if path.suffix=='.json':
        c = strict_json(path.read_bytes())
    else:
        import yaml
        class Loader(yaml.SafeLoader):
            def construct_mapping(self,node,deep=False):
                keys = [self.construct_object(key,deep=deep) for key,_ in node.value]
                require(all(type(k) is str for k in keys) and len(set(keys))==len(keys),'duplicate/non-string YAML key')
                return super().construct_mapping(node,deep=deep)
        require(path.suffix in ('.yaml','.yml'),'JSON/YAML configuration required')
        c = yaml.load(path.read_bytes(),Loader=Loader)
    fields = {'output_root','cohort','publication','evaluation','expected_reference','gpu_uuid','models'}
    require(type(c) is dict and set(c)==fields,'lightweight execution config required; old qualification settings are not inherited')
    require(type(c['models']) is dict and c['models'],'configured models required')
    for key,item in c['models'].items():
        ModelSpec(key)
        require(type(item) is dict and set(item)=={'model_directory','environment_prefix','tokenizer_assets'},
                'model paths only; decoding/qualification overrides forbidden')
    require(type(c['cohort']) is dict and set(c['cohort'])=={'path','digest'},'Cohort path/digest required')
    return c


def run_root(c, run_id):
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',run_id) is not None,'invalid run-id')
    return safe_path(c['output_root'])/run_id


def keyed(values):
    result = {}
    for item in values:
        key,value = item.split('=',1)
        require(key not in result,'duplicate model argument')
        result[key] = value
    return result


def prepare(c, run_id, key):
    from werewolf.tom_llm.deployment import profile
    root = run_root(c,run_id);item = c['models'][key]
    runtime = profile(ModelSpec(key),root=root,model_directory=item['model_directory'],
                      environment_prefix=item['environment_prefix'],gpu_uuid=c['gpu_uuid'])
    return prepare_run(root,run_id=run_id,model=key,cohort_root=c['cohort']['path'],
        expected_cohort=c['cohort']['digest'],publication=c['publication'],evaluation=c['evaluation'],
        expected_reference=c['expected_reference'],runtime=runtime,tokenizer_assets=item['tokenizer_assets'])


def execute(c, args):
    from werewolf.tom_llm.live_runner import Journal,run_lock,run_model,recover_received,status,read_predictions
    from werewolf.tom_llm.service_manager import Service
    require(args.execute_live and sys.platform=='linux','explicit --execute-live on Ubuntu required')
    require(args.model in c['models'],'unconfigured model')
    root = run_root(c,args.run_id)
    run = open_run(root,check_source=True) if root.exists() else prepare(c,args.run_id,args.model)
    require(run.key==args.model,'Run/model mismatch')
    require(run.record['mode']=='formal','CPU/Mock Run cannot start a real model')
    with run_lock(root):
        if (root/'seal').exists():
            read_predictions(run)
            return status(run)
        journal = Journal(run)
        require(args.resume or journal.sequence==0,'existing journal requires explicit --resume')
        recover_received(run,journal) if args.resume else None
        require(not journal.summary()['blocked'],'unknown result; no restart/resend')
        if len(journal.rows)==run.record['request_count']:
            return run_model(run,None,resume=True,journal=journal)
        manager = Service(run)
        from werewolf.tom_llm.deployment import profile
        runtime = run.record['runtime']
        require(runtime==profile(run.spec,root=run.root,model_directory=runtime['argv_options']['model'],
            environment_prefix=runtime['environment_prefix'],gpu_uuid=runtime['gpu_uuid']),'runtime settings drift')
        manager.start(execute=True)
        from werewolf.tom_llm.live_runner import HttpTransport
        transport = HttpTransport('http://127.0.0.1:'+str(run.record['runtime']['argv_options']['port']),
                                  alive_check=manager.guard)
        try:
            result = run_model(run,transport,resume=args.resume,journal=journal)
        except BaseException as error:
            try:
                manager.stop(execute=True)
            except Exception as stop_error:
                error.add_note('Owned service stop needs review: '+str(stop_error))
            raise
        manager.stop(execute=True)
        return result


def dispatch(args):
    c = load_config(args.config)
    if args.operation=='report':
        from werewolf.tom_llm.baseline_analysis import analyze_cross_runs
        roots = keyed(args.sealed_run);digests = keyed(args.sealed_run_digest)
        require(set(roots)==set(digests),'result paths/digests differ')
        return analyze_cross_runs({k:{'path':v,'digest':digests[k]} for k,v in roots.items()},args.output)
    if args.operation=='prepare':
        require(args.model in c['models'],'unconfigured model')
        return prepare(c,args.run_id,args.model).artifact
    if args.operation=='run':
        return execute(c,args)
    run = open_run(run_root(c,args.run_id),with_tasks=False)
    from werewolf.tom_llm.service_manager import Service
    if args.operation=='stop':
        return Service(run).stop(execute=args.execute_live)
    from werewolf.tom_llm.live_runner import status
    result = status(run)
    if run.record['mode']=='formal':
        result['service'] = Service(run).status()
    return result


def invoke(args):
    result = dispatch(args)
    if hasattr(result,'manifest_digest'):
        result = {'path':str(result.path),'digest':result.manifest_digest}
    print(json.dumps(result,ensure_ascii=False,sort_keys=True))
    return 0
