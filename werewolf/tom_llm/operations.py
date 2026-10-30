"""Config-driven operations over the existing ToM experiment implementations."""

import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

from werewolf.artifact_io import verify_artifact
from werewolf.tom_llm.common import require,strict_json
from werewolf.tom_llm.experiment import safe_path,VERSION as EXPERIMENT_VERSION
from werewolf.tom_llm.qualification_evidence import VERSION as QUALIFICATION_VERSION
from werewolf.tom_llm.model_spec import ModelSpec


def load_config(path, *, raw=None):
    path=safe_path(path);raw=path.read_bytes() if raw is None else raw
    if path.suffix=='.json':result=strict_json(raw)
    else:
        require(path.suffix in ('.yaml','.yml'),'configuration must be JSON or YAML')
        import yaml
        class Loader(yaml.SafeLoader):
            def construct_mapping(self,node,deep=False):
                keys=[self.construct_object(key,deep=deep) for key,_ in node.value]
                require(all(type(k) is str for k in keys) and len(set(keys))==len(keys),'duplicate/non-string YAML key')
                return super().construct_mapping(node,deep=deep)
        result=yaml.load(raw,Loader=Loader)
    require(type(result) is dict and type(result.get('models')) is dict and result['models'],'models mapping required')
    for key,item in result['models'].items():
        ModelSpec(key).record()
        require(type(item) is dict,'model operations must be a mapping')
        require(not set(item)&{'pid','service_pid','revision','temperature','top_k','chat_template','model_spec'},
                'runtime PID or frozen ModelSpec override is not an operations setting')
        require(set(item)<={'candidate_root','cpu_root','model_directory','environment_prefix','tokenizer_assets','prior_audit_digest',
            'gpu_uuid','assets','environment','suite','budget','deployment','admission'},'unknown model operations setting')
    return result


def anchor(ref):
    require(type(ref) is dict and set(ref)=={'path','digest'},'input Artifact requires path and digest in YAML')
    path=safe_path(ref['path']);m=strict_json((path/'manifest.json').read_bytes())
    require(m['manifest_digest']==ref['digest'],'configured Artifact digest mismatch')
    a=verify_artifact(path,expected_artifact_type=m['artifact_type'],expected_schema_version=m['schema_version'])
    require(a.manifest_digest==ref['digest'],'configured Artifact digest mismatch')
    return a


def ref(artifact):return {'path':str(artifact.path),'digest':artifact.manifest_digest}


def save_configuration(path,original,configuration):
    """Explicit operations-config update, never a rewrite of an Artifact.

    Atomic replacement; detect intervening edits before replacement. YAML is
    re-serialized (comments are not retained). No implicit updates or registry.
    """
    import os,tempfile
    path=safe_path(path)
    if sys.platform=='linux':
        from werewolf.tom_llm.deployment import baseline_root
        baseline_root(path)
    require(not any((p/'manifest.json').exists() for p in path.parents),'configuration is inside a sealed Artifact')
    require(path.read_bytes()==original,'configuration changed; generated Artifacts retained, configuration not replaced')
    if path.suffix=='.json':
        from werewolf.artifact_io import canonical_json_bytes
        raw=canonical_json_bytes(configuration)
    else:
        import yaml
        raw=yaml.safe_dump(configuration,allow_unicode=True,sort_keys=False).encode()
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    temporary=Path(name)
    try:
        with os.fdopen(fd,'wb') as stream:
            os.fchmod(stream.fileno(),path.stat().st_mode & 0o777)
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        require(safe_path(path).read_bytes()==original,'configuration changed; generated Artifacts retained, configuration not replaced')
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def _save_refs(args,original,c,*,models=None,cohort=None):
    if not args.save_config:return
    from copy import deepcopy
    updated=deepcopy(c)
    if cohort is not None:updated['cohort']=cohort
    for key,refs in (models or {}).items():updated['models'][key].update(refs)
    save_configuration(args.config,original,updated)


def prepare_inputs(c,key,stage):
    """Official small assets and full-cohort CPU inputs; no weights/prefix/GPU."""
    from werewolf.tom_llm.model_spec import check_cpu_parity
    from werewolf.tom_llm.qualification_inputs import synthetic_tasks,publish_suite,open_suite,publish_budget,open_budget
    item=c['models'][key];spec=ModelSpec(key)
    if stage=='tokenizer':
        return {'tokenizer':check_cpu_parity(synthetic_tasks(),spec,item['tokenizer_assets'])}
    def destination():
        root=safe_path(item['cpu_root']) if item.get('cpu_root') else safe_path(c['output_root'])/'cpu'/key
        if sys.platform=='linux':
            from werewolf.tom_llm.deployment import baseline_root
            baseline_root(root)
        require(not any((p/'manifest.json').exists() for p in (root,*root.parents)),'CPU inputs inside sealed Artifact')
        for field in ('model_directory','environment_prefix','tokenizer_assets'):
            if item.get(field):require(not root.is_relative_to(safe_path(item[field])), 'CPU outputs inside model/tokenizer/environment assets')
        if item.get('deployment'):
            require(not root.is_relative_to(safe_path(item['deployment']['path']).parent),
                    'CPU inputs must not write inside an existing candidate')
        return root/stage
    if stage=='suite':
        a=anchor(item['suite']) if item.get('suite') else publish_suite(destination(),spec,item['tokenizer_assets'])
        open_suite(a.path,spec,a.manifest_digest)
        return {'suite':ref(a)}
    require(stage=='budget','unknown CPU preparation stage')
    cohort=anchor(c['cohort'])
    a=anchor(item['budget']) if item.get('budget') else publish_budget(destination(),cohort.path,cohort.manifest_digest,spec,item['tokenizer_assets'])
    budget=open_budget(a.path,spec,a.manifest_digest)
    require(budget['cohort_digest']==cohort.manifest_digest,'configured budget belongs to another common cohort')
    summary={k:budget[k] for k in ('request_count','row_identity_digest','input_tokens_total','maximum_input_tokens','p50','p95','p99','context_length','output_cap')}
    return {'budget':ref(a),'token_budget':summary}


def gpu_status():
    """Read-only device/process occupancy, never an inference or service action."""
    if sys.platform!='linux':return {'state':'not_linux'}
    import subprocess
    try:
        device=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True,timeout=10).strip()
        pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
            '--format=csv,noheader,nounits'],text=True,timeout=10).strip().splitlines()
        return {'state':'observed','device_columns':['uuid','used_mib','free_mib','utilization_percent'],
                'devices_csv':device,'compute_pids':pids}
    except (OSError,subprocess.SubprocessError) as error:return {'state':'unavailable','error':str(error)}


def run_root(args,c):
    require(args.run_id and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',args.run_id),'valid --run-id required')
    return safe_path(c['output_root'])/args.run_id


def check_environment_binding(item,profile):
    # Editing YAML cannot move old physical GPU evidence to a different prefix.
    if item.get('environment_prefix'):
        require(str(safe_path(item['environment_prefix']))==profile['environment_prefix'],
                'configured environment differs from sealed binding; new Deployment and GPU Admission required')


def service(args,c,key):
    from werewolf.tom_llm.service_manager import Service
    if args.purpose=='qualification':
        d=anchor(c['models'][key]['deployment'])
        manager=Service(d.path,key,d.manifest_digest,qualification=True)
    else:
        root=run_root(args,c)
        p=verify_artifact(root/'models'/key/'permit',expected_artifact_type='mm_live_execution_permit',expected_schema_version=EXPERIMENT_VERSION)
        manager=Service(root,key,p.manifest_digest)
    # Status/logs/stop must remain usable for an already owned historical service.
    if getattr(args,'operation',None)=='qualify' or getattr(args,'service_action',None)=='start':
        check_environment_binding(c['models'][key],manager.binding()[1]['profile'])
    return manager


def prepare_model(c,key):
    """CPU-only preparation; no installs/downloads, Selection publishing, or GPU."""
    from werewolf.tom_llm.physical import verify_model,publish_receipt,read_receipt
    from werewolf.tom_llm.live_admission import capture_bound_environment
    from werewolf.tom_llm.qualification_inputs import publish_suite,publish_budget,open_suite,open_budget
    from werewolf.tom_llm.deployment import prepare_deployment,profile
    item=c['models'][key];spec=ModelSpec(key);root=safe_path(item['candidate_root'])
    require(not (root/'deployment').exists(),'deployment exists; select a NEW candidate_root, never overwrite')
    profile(spec,gpu_uuid=item.get('gpu_uuid',c.get('gpu_uuid')),candidate_root=root,
        model_directory=item['model_directory'],environment_prefix=item['environment_prefix'])
    cohort=anchor(c['cohort'])
    def receipt(name,kind,build):
        if item.get(name):
            a=anchor(item[name]);read_receipt(a.path,kind,a.manifest_digest);return a
        return publish_receipt(root/'receipts'/name,kind,build())
    assets=receipt('assets','model',lambda:verify_model(spec,item['model_directory']))
    environment=receipt('environment','environment',lambda:capture_bound_environment(item['environment_prefix'],
        item['prior_audit_digest']))
    if item.get('suite'):
        suite=anchor(item['suite']);open_suite(suite.path,spec,suite.manifest_digest)
    else:suite=publish_suite(root/'receipts'/'suite',spec,item['tokenizer_assets'])
    if item.get('budget'):
        budget=anchor(item['budget']);open_budget(budget.path,spec,budget.manifest_digest)
    else:budget=publish_budget(root/'receipts'/'budget',cohort.path,cohort.manifest_digest,spec,item['tokenizer_assets'])
    deployment=prepare_deployment(root/'deployment',spec,gpu_uuid=item.get('gpu_uuid',c.get('gpu_uuid')),
        model_root=item['model_directory'],model_receipt=assets.path,expected_model=assets.manifest_digest,
        environment_root=item['environment_prefix'],environment_receipt=environment.path,expected_environment=environment.manifest_digest,
        suite_root=suite.path,expected_suite=suite.manifest_digest,budget_root=budget.path,expected_budget=budget.manifest_digest)
    return {name:ref(a) for name,a in [('assets',assets),('environment',environment),('suite',suite),('budget',budget),('deployment',deployment)]}


def dispatch(args):
    original=safe_path(args.config).read_bytes();c=load_config(args.config,raw=original)
    keys=getattr(args,'models',None) or list(c['models'])
    require(len(set(keys))==len(keys) and set(keys)<=set(c['models']),'duplicate/unconfigured models')
    op=args.operation
    if hasattr(args,'model'):require(args.model in c['models'],'model is not configured')
    if sys.platform=='linux' and op in ('prepare','admit','run','report') and not (op=='prepare' and args.stage=='tokenizer'):
        from werewolf.tom_llm.deployment import baseline_root
        baseline_root(safe_path(c['output_root']))
        for key in keys:
            if c['models'][key].get('candidate_root'):
                baseline_root(safe_path(c['models'][key]['candidate_root']))
    if op=='status':
        result={'models':{},'gpu':gpu_status()}
        for key in keys:
            spec=ModelSpec(key);item=c['models'][key];r={'model':spec.record()['model'],'revision':spec.record()['revision']}
            for name in ('model_directory','environment_prefix','tokenizer_assets'):
                if item.get(name):r[name]={'path':item[name],'exists':safe_path(item[name]).exists(),
                    'verification':'presence_only; use prepare for physical validation'}
            for name in ('assets','environment','suite','budget','deployment','admission'):
                if not item.get(name):r[name]={'state':'not_configured'};continue
                try:
                    a=anchor(item[name]);r[name]={'state':'artifact_verified',**ref(a)}
                    if name=='admission':
                        from werewolf.tom_llm.physical_admission import open_admission
                        d=anchor(item['deployment'])
                        open_admission(a.path,expected_manifest=a.manifest_digest,expected_deployment=d.manifest_digest,model_spec=spec)
                        r[name]['state']='historical_gpu_evidence_verified'
                except (OSError,ValueError,KeyError) as error:r[name]={'state':'unavailable','error':str(error)}
            result['models'][key]=r
        if args.run_id:
            from werewolf.tom_llm.live_runner import status
            result['run']=status(run_root(args,c))
        return result
    if op=='prepare':
        require(not args.save_config or args.stage in ('cohort','suite','budget','model'),
                '--save-config applies only to generated Artifact references')
        if args.stage in ('tokenizer','suite','budget'):
            if args.stage=='budget':require(args.games==150,'CPU budget always covers the full fixed 150-game cohort')
            result={key:prepare_inputs(c,key,args.stage) for key in keys}
            if args.stage!='tokenizer':
                _save_refs(args,original,c,models={key:{args.stage:result[key][args.stage]} for key in keys})
            return result
        if args.stage=='model':
            result={key:prepare_model(c,key) for key in keys}
            _save_refs(args,original,c,models=result)
            return result
        if args.stage=='cohort':
            if c.get('cohort'):
                from werewolf.tom_llm.multimodel import read_cohort
                a=anchor(c['cohort']);read_cohort(a.path,a.manifest_digest,current_source=False);return ref(a)
            from werewolf.tom_llm.qualification_inputs import export_cpu_cohort
            a=export_cpu_cohort(safe_path(c['output_root'])/'cohort',publication_root=c['publication'],evaluation_root=c['evaluation'],
                expected_reference=c['expected_reference'],selection_root=c['selection'],
                expected_selection=strict_json(safe_path(c['expected_selection_json']).read_bytes()),
                selection_provenance=strict_json(safe_path(c['selection_provenance_json']).read_bytes()),prepared_root=c['prepared'])
            _save_refs(args,original,c,cohort=ref(a))
            return ref(a)
        root=run_root(args,c)
        if args.stage=='run':
            from werewolf.tom_llm.baseline_cli import prepare
            return prepare(SimpleNamespace(config=args.config,models=keys,games=args.games,mode=args.mode,run_id=args.run_id),root)
        from werewolf.tom_llm.live_admission import publish_permit
        from werewolf.tom_llm.deployment import open_deployment
        result={}
        for key in keys:
            item=c['models'][key];admission=anchor(item['admission']);d=anchor(item['deployment'])
            claimed=strict_json((admission.path/'admission.json').read_bytes())
            require(claimed['deployment_manifest_digest']==d.manifest_digest,'configured Admission/Deployment mismatch')
            _,record=open_deployment(d.path,d.manifest_digest,current_source=False)
            check_environment_binding(item,record['profile'])
            budget=anchor(item['budget'])
            require(budget.manifest_digest==record['upstream_receipts']['budget'],'configured qualified budget drift')
            result[key]=ref(publish_permit(root,key,qualification=admission.path,expected_qualification=admission.manifest_digest,
                tokenizer_assets=item['tokenizer_assets'],budget_root=budget.path))
        return result
    if op=='serve':
        manager=service(args,c,args.model)
        if args.service_action=='start':return manager.start(execute=args.execute_live)
        if args.service_action=='stop':return manager.stop(execute=args.execute_live)
        if args.service_action=='logs':return {'log_tail':manager.logs(args.lines)}
        state=manager.status()
        state['gpu_occupancy']=gpu_status()
        if state['state']=='running':
            try:state.update(ready=True,instance=manager.capture())
            except (OSError,ValueError,KeyError) as error:state['readiness_error']=str(error)
        return state
    if op=='qualify':
        require(args.execute_live and sys.platform=='linux','explicit --execute-live on Ubuntu required')
        args.purpose='qualification';manager=service(args,c,args.model);instance=manager.capture()
        a,r,_=manager.binding()
        # Existing client audits its own prefix. Run it with the model's Python,
        # not with a potentially different controller/report environment.
        import os,subprocess,uuid
        from werewolf.tom_llm.model_spec import ROOT
        destination=Path(r['profile']['root'])/'qualification'/('round-'+str(args.round))
        require(not destination.exists(),'qualification round exists; no replay/overwrite')
        cmd=[r['profile']['environment_prefix']+'/bin/python',str(ROOT/'scripts/tom_multimodel_admission.py'),
            'round','--output',str(destination),'--deployment',str(a.path),'--expected-deployment',a.manifest_digest,
            '--pid',str(instance['pid']),'--run',str(args.round),'--execute-synthetic-gpu-qualification']
        env={k:os.environ[k] for k in ('HOME','USER','LANG','LC_ALL') if k in os.environ}
        env.update(r['profile']['environment_requirements'])
        log=manager.directory/('client-'+uuid.uuid4().hex+'.log')
        with log.open('xb') as stream:
            result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        require(result.returncode==0,'qualification stopped; no retry; inspect '+str(log))
        return verify_artifact(destination/'seal',expected_artifact_type='mm_probe_round_seal',expected_schema_version=QUALIFICATION_VERSION)
    if op=='admit':
        from werewolf.tom_llm.physical_admission import publish_admission
        from werewolf.tom_llm.deployment import open_deployment
        result={}
        for key in keys:
            d=anchor(c['models'][key]['deployment']);_,r=open_deployment(d.path,d.manifest_digest)
            q=Path(r['profile']['root'])/'qualification';rounds=[q/('round-'+str(i)) for i in (0,1)]
            seals=[verify_artifact(p/'seal',expected_artifact_type='mm_probe_round_seal',expected_schema_version=QUALIFICATION_VERSION) for p in rounds]
            result[key]=ref(publish_admission(q/'admission',d.path,d.manifest_digest,
                round_roots=rounds,expected_rounds=[a.manifest_digest for a in seals]))
        _save_refs(args,original,c,models={key:{'admission':result[key]} for key in keys})
        return result
    root=run_root(args,c)
    if op=='report':
        from werewolf.tom_llm.baseline_analysis import analyze
        return analyze(root,args.output or root/'analysis',expected_run_seal=args.expected_run_seal)
    from werewolf.tom_llm.baseline_cli import execute
    from werewolf.tom_llm.experiment import open_run
    _,run,_=open_run(root)
    keys=args.models or run['model_order']
    permits=[];pids=[]
    if run['mode']!='fake':
        require(args.execute_live and sys.platform=='linux','explicit --execute-live on Ubuntu required')
        for key in keys:
            p=verify_artifact(root/'models'/key/'permit',expected_artifact_type='mm_live_execution_permit',expected_schema_version=EXPERIMENT_VERSION)
            from werewolf.tom_llm.live_admission import open_permit
            _,record=open_permit(root,key,p.manifest_digest,physical=False)
            check_environment_binding(c['models'][key],record['profile'])
            permits.append(key+'='+p.manifest_digest)
            if not args.manage_services:
                manager=service(SimpleNamespace(purpose='run',run_id=args.run_id),c,key)
                pids.append(key+'='+str(manager.capture()['pid']))
    return execute(SimpleNamespace(models=keys,games=None,mode=None,permit=permits,fake_endpoint=[],service_pid=pids,
        execute_live=args.execute_live,manage_services=args.manage_services,resume=args.resume),root)


def invoke(args):
    result=dispatch(args)
    if hasattr(result,'manifest_digest'):result=ref(result)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True))
    return 0
