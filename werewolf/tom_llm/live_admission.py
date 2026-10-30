"""Bind a real execution to sealed physical qualification; no human-approval evidence."""

from pathlib import Path
import os
import sys
import subprocess

from werewolf.artifact_io import publish_artifact, verify_artifact, canonical_json_bytes
from werewolf.tom_llm.common import digest, require, strict_json
from werewolf.tom_llm.experiment import VERSION, open_run, source_v3, safe_path
from werewolf.tom_llm.model_spec import ROOT, ModelSpec, template_bytes, load_cpu_tokenizer
from werewolf.tom_llm.physical_admission import ADMISSION_VERSION
from werewolf.tom_llm.deployment import (baseline_root, open_deployment, profile, validate_process_environment,
    gpu_snapshot, driver_version, vllm_config_bytes, CACHE_VARIABLES)
from werewolf.tom_llm.qualification_inputs import open_budget


def qualified_offline(root, expected, spec):
    """Durable qualification validation after the old second instance is shut down.

    This is evidence of past synthetic qualification, never a live-service identity.
    """
    from werewolf.tom_llm.physical_admission import open_admission
    from werewolf.tom_llm.deployment import qualification_compatibility
    # The external admission digest anchors its Deployment digest too.
    anchor=verify_artifact(root,expected_artifact_type='mm_physical_gpu_admission',expected_schema_version=ADMISSION_VERSION)
    require(anchor.manifest_digest==expected,'qualification anchor/domain drift')
    claimed=strict_json((anchor.path/'admission.json').read_bytes())
    _,r=open_admission(root,expected_manifest=expected,expected_deployment=claimed['deployment_manifest_digest'],model_spec=spec)
    _,record=open_deployment(r['deployment_root'],r['deployment_manifest_digest'],current_source=False)
    qualification_compatibility(record['source'])
    from werewolf.tom_llm.physical import verify_model
    require(record['model']==verify_model(spec,record['model']['directory']),'qualified physical model bytes drift')
    require(record['environment']==capture_bound_environment(record['profile']['environment_prefix'],
            record['environment']['prior_audit_digest']),'qualified physical environment drift')
    require(record['key']==spec.key and digest(record)==r['deployment_digest'],'qualification Deployment drift')
    return record


def _cpu_child(prefix, code, arguments):
    prefix=safe_path(prefix)
    require((prefix/'bin/python').is_file(),'qualified environment interpreter missing')
    child={k:os.environ[k] for k in ('HOME','USER','LANG','LC_ALL') if k in os.environ}
    child.update(PATH=str(prefix/'bin')+':/usr/bin:/bin',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    try:
        result=subprocess.run([str(prefix/'bin/python'),'-I','-S','-B','-c',code,str(ROOT),
            str(prefix/'lib/python3.12/site-packages'),*map(str,arguments)],
            env=child,capture_output=True,text=True,timeout=300,check=True)
    except subprocess.CalledProcessError as error:
        raise ValueError(f'CPU environment check failed (exit {error.returncode})\n'
                         f'stdout:\n{error.stdout or ""}\nstderr:\n{error.stderr or ""}') from error
    return strict_json(result.stdout)


def capture_bound_environment(prefix, prior_audit_digest):
    """Revalidate using the target prefix, never the report/controller Python."""
    code="""import sys,json
sys.path[:0]=[sys.argv[1],sys.argv[2]]
from werewolf.tom_llm.physical import capture_environment
print(json.dumps(capture_environment(sys.argv[3],prior_audit_digest=sys.argv[4]),sort_keys=True))
"""
    return _cpu_child(prefix,code,[prefix,prior_audit_digest])


def service_profile(root, spec, qualified):
    old=qualified['profile']
    p=profile(spec,gpu_uuid=old['gpu']['uuid'],candidate_root=old['root'],model_directory=old['argv_options']['model'],environment_prefix=old['environment_prefix'])
    service=Path(root)/'models'/spec.key/'service'
    original=p['root'];p['root']=str(service)
    p['environment_requirements']={k:(v.replace(original,str(service)) if k in CACHE_VARIABLES else v)
        for k,v in p['environment_requirements'].items()}
    p['environment_requirements']['PYTHONPATH']=str(ROOT)
    p['argv_options']['chat-template']=str(Path(root)/'models'/spec.key/'permit/chat_template.jinja')
    p['engine_log_root']=str(service/'engine');p['qualification_only']=False
    require({k:v for k,v in p['argv_options'].items() if k!='chat-template'}==
            {k:v for k,v in old['argv_options'].items() if k!='chat-template'},'qualified inference arguments changed')
    ignored=set(CACHE_VARIABLES)|{'PYTHONPATH'}
    require({k:v for k,v in p['environment_requirements'].items() if k not in ignored}==
            {k:v for k,v in old['environment_requirements'].items() if k not in ignored},'qualified inference environment changed')
    return p


def publish_permit(root, key, *, qualification, expected_qualification, tokenizer_assets, budget_root):
    require(sys.platform=='linux','physical execution permits require Ubuntu; CPU tests cannot grant them')
    a,run,tasks=open_run(root)
    require(run['mode'] in ('formal','technical') and run['source']['source_inventory_clean'],'committed live Source required')
    baseline_root(safe_path(root))
    spec=ModelSpec(key);require(key in run['models'],'unplanned model')
    qualified=qualified_offline(qualification,expected_qualification,spec)
    budget=open_budget(budget_root,spec,qualified['upstream_receipts']['budget'])
    require(budget==qualified['budget'],'qualified CPU budget payload drift')
    require(budget['cohort_digest']==run['cohort_digest'],'qualified budget is for another common cohort')
    load_cpu_tokenizer(spec,tokenizer_assets)
    p=service_profile(root,spec,qualified)
    record={'run_root':str(Path(root).absolute()),'run_digest':a.manifest_digest,'model':key,'model_spec_digest':spec.digest,
        'source':source_v3(),'qualified_runtime':qualification_runtime(qualification),'profile':p,'qualification_root':str(Path(qualification).absolute()),
        'qualification_digest':expected_qualification,'budget_root':str(safe_path(budget_root)),
        'tokenizer_assets':str(safe_path(tokenizer_assets)),'evidence_kind':'physical_execution_binding'}
    from werewolf.tom_llm.deployment import qualification_compatibility
    record['qualification_compatibility']=qualification_compatibility(qualified['source'])
    dest=Path(root)/'models'/key/'permit';require(not dest.exists(),'permit exists')
    return publish_artifact(dest,manifest_fields={'artifact_type':'mm_live_execution_permit','schema_version':VERSION,
        'record_digest':digest(record)},files={'permit.json':canonical_json_bytes(record),
        'vllm.yaml':vllm_config_bytes(p['argv_options']),'chat_template.jinja':template_bytes(spec)})


def qualification_runtime(root):
    record=strict_json((Path(root)/'admission.json').read_bytes())
    last=strict_json((Path(record['round_roots'][1])/'seal/round.json').read_bytes())['instance']
    return {'driver':last['driver_version'],'kernel':last['kernel_release']}


def open_permit(root, key, expected, *, physical=True):
    a,run,tasks=open_run(root,current_source=physical);spec=ModelSpec(key)
    p=verify_artifact(Path(root)/'models'/key/'permit',expected_artifact_type='mm_live_execution_permit',expected_schema_version=VERSION)
    require(p.manifest_digest==expected and set(p.manifest['file_table'])=={'permit.json','vllm.yaml','chat_template.jinja'},'execution permit anchor/inventory drift')
    r=strict_json((p.path/'permit.json').read_bytes())
    require(digest(r)==p.manifest['record_digest'] and r['run_digest']==a.manifest_digest and r['model']==key
            and r['model_spec_digest']==spec.digest and r['source']==run['source']
            and (not physical or r['source']==source_v3())
            and r['evidence_kind']=='physical_execution_binding' and run['mode'] in ('formal','technical'), 'live permit/source/model drift')
    require((p.path/'chat_template.jinja').read_bytes()==template_bytes(spec)
            and (p.path/'vllm.yaml').read_bytes()==vllm_config_bytes(r['profile']['argv_options']), 'live template/config drift')
    if physical:
        require(sys.platform=='linux' and run['source']['source_inventory_clean'], 'live physical platform/source/path gate')
        baseline_root(safe_path(root))
        q=qualified_offline(r['qualification_root'],r['qualification_digest'],spec)
        from werewolf.tom_llm.deployment import qualification_compatibility
        compatibility=qualification_compatibility(q['source'])
        if 'qualification_compatibility' in r:
            require(r['qualification_compatibility']==compatibility,'qualification inheritance binding drift')
        else:
            from werewolf.tom_llm.multimodel import source_identity
            require(q['source']==source_identity(),
                    'legacy permit cannot inherit another Source')
        require(open_budget(r['budget_root'],spec,q['upstream_receipts']['budget'])==q['budget'],'live budget drift')
        require(r['profile']==service_profile(root,spec,q) and q['budget']['cohort_digest']==run['cohort_digest']
                and r['qualified_runtime']==qualification_runtime(r['qualification_root']),'live profile/budget/runtime drift')
    return p,r


def capture_live_instance(permit, record, pid):
    require(sys.platform=='linux' and type(pid) is int and pid>0,'physical service PID required')
    proc=Path('/proc')/str(pid);p=record['profile']
    expected=[p['environment_prefix']+'/bin/python','-m','werewolf.tom_llm.live_service','--run',record['run_root'],
              '--model',record['model'],'--expected-permit',permit.manifest_digest,'--execute-live']
    require([b.decode() for b in (proc/'cmdline').read_bytes().split(b'\0') if b]==expected
            and (proc/'exe').resolve()==Path(expected[0]).resolve() and (proc/'cwd').resolve()==ROOT,'live service executable/argv/cwd drift')
    environment=dict(x.split(b'=',1) for x in (proc/'environ').read_bytes().split(b'\0') if b'=' in x)
    validate_process_environment(p,{k.decode():v.decode() for k,v in environment.items()})
    life={'pid':pid,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
          'start_ticks':(proc/'stat').read_text().rsplit(')',1)[1].split()[19]}
    log=str(Path(p['engine_log_root'])/digest(life)/'engine-events.jsonl');inodes=set();owned=False
    for fd in (proc/'fd').iterdir():
        try:
            target=os.readlink(fd)
            if target.startswith('socket:['):inodes.add(target[8:-1])
            if target==log:owned=True
        except OSError:pass
    port=p['argv_options']['port']
    require(owned and any(v[1]==f'0100007F:{port:04X}' and v[3]=='0A' and v[9] in inodes
        for line in Path('/proc/net/tcp').read_text().splitlines()[1:] if (v:=line.split())), 'service socket/audit FD not owned by PID')
    require(gpu_snapshot()==p['gpu'],'physical GPU drift')
    require(record['qualified_runtime']=={'driver':driver_version(),'kernel':os.uname().release},'driver/kernel changed since qualification')
    return {**life,'permit_digest':permit.manifest_digest,'gpu':p['gpu'],'driver':driver_version(),
            'kernel':os.uname().release,'engine_log':log}
