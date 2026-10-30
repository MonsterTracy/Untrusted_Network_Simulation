"""CPU-only operation/lifecycle contracts; no physical Admission is granted."""

from copy import deepcopy
from contextlib import contextmanager
import json
import os
from pathlib import Path
from subprocess import Popen as CpuChild
from types import SimpleNamespace as NS

import pytest

from werewolf.artifact_io import canonical_json_bytes,publish_artifact
from werewolf.cli import build_parser,main
from werewolf.tom_llm import deployment as dep,operations as ops,physical_admission as admission
from werewolf.tom_llm.common import digest
from werewolf.tom_llm.model_spec import ModelSpec,ROOT
from werewolf.tom_llm.service_manager import Service
from tests.tom_llm import test_physical_tools as fixtures

native_suites=fixtures.native_suites
short_root=fixtures.short_root


def test_yaml_and_json_share_configuration_and_frozen_model_registry(tmp_path,capsys):
    y=tmp_path/'ops.yaml';y.write_text('models:\n  qwen3: {}\n')
    j=tmp_path/'ops.json';j.write_text(json.dumps({'models':{'qwen3':{}}}))
    assert ops.load_config(y)==ops.load_config(j)
    assert main(['tom-llm','--config',str(y),'status'])==0
    r=json.loads(capsys.readouterr().out)
    assert r['models']['qwen3']['revision']==ModelSpec('qwen3').record()['revision']
    assert r['models']['qwen3']['admission']['state']=='not_configured'
    y.write_text('models: {}\nmodels: {}\n')
    with pytest.raises(ValueError,match='duplicate'):ops.load_config(y)
    y.write_text('models:\n  qwen3:\n    temperature: 0.1\n')
    with pytest.raises(ValueError,match='override'):ops.load_config(y)


def test_three_models_share_explicit_environment_but_not_candidate_assets():
    c=ops.load_config(ROOT/'configs/paper/tom-llm-operations.example.yaml')
    shared='/data/yuxiao/envs/untrusted-network-simulation-tom-baseline'
    profiles=[]
    for key,item in c['models'].items():
        assert item['environment_prefix']==shared
        p=dep.profile(ModelSpec(key),gpu_uuid='GPU-UNIT',candidate_root=item['candidate_root'],
            model_directory=item['model_directory'],environment_prefix=shared)
        assert p['environment_prefix']==shared
        assert p['environment_requirements']['PATH'].startswith(shared+'/bin:')
        assert p['environment_requirements']['CUDA_HOME']==shared+'/lib/python3.12/site-packages/nvidia/cu13'
        dep.validate_cache_paths(p)
        profiles.append(p)
    for path in ('root','engine_log_root'):
        assert len({p[path] for p in profiles})==3
    assert len({p['argv_options']['model'] for p in profiles})==3
    assert len({p['argv_options']['chat-template'] for p in profiles})==3
    assert len({p['model_spec_digest'] for p in profiles})==3


@pytest.mark.parametrize('prefix',['/data/yuxiao/envs','/data/yuxiao/envs/nested/env',
    '/data/yuxiao/envs/../foreign','/tmp/foreign','relative-env'])
def test_shared_environment_does_not_allow_arbitrary_external_paths(prefix):
    with pytest.raises(ValueError,match='environment'):
        dep.profile(ModelSpec('qwen3'),gpu_uuid='GPU-UNIT',candidate_root=dep.BASE/'multimodel/qwen3/new',
            environment_prefix=prefix)


def test_shared_environment_symlink_is_rejected_and_legacy_profile_is_preserved(tmp_path,monkeypatch):
    monkeypatch.setattr(dep,'ENV_BASE',tmp_path)
    (tmp_path/'real').mkdir();(tmp_path/'alias').symlink_to(tmp_path/'real',target_is_directory=True)
    with pytest.raises(ValueError,match='symlinks'):
        dep.profile(ModelSpec('qwen3'),gpu_uuid='GPU-UNIT',candidate_root=dep.BASE/'multimodel/qwen3/new',
            environment_prefix=tmp_path/'alias')
    root=dep.BASE/'multimodel/qwen3/candidate-v4'
    default=dep.profile(ModelSpec('qwen3'),gpu_uuid='GPU-UNIT',candidate_root=root)
    assert default==dep.profile(ModelSpec('qwen3'),gpu_uuid='GPU-UNIT',candidate_root=root,
        model_directory=root/'model',environment_prefix=root/'env')
    assert default['environment_prefix']==str(root/'env')


def test_shared_prefix_still_requires_exact_physical_environment_receipt(native_suites,monkeypatch):
    from werewolf.tom_llm import live_admission as live
    record=fixtures.model_record('qwen3',native_suites)
    shared='/data/yuxiao/envs/untrusted-network-simulation-tom-baseline'
    record['profile']=dep.profile(ModelSpec('qwen3'),gpu_uuid='GPU-unit-only',
        candidate_root=record['profile']['root'],environment_prefix=shared)
    record['model']['directory']=record['profile']['argv_options']['model']
    record['environment']={'prefix':shared,'prior_audit_digest':'a'*64,'UNIT_version':'original'}
    monkeypatch.setattr(dep,'verify_model',lambda *a:record['model'])
    observed=[]
    def capture(prefix,audit):observed.append((prefix,audit));return deepcopy(record['environment'])
    monkeypatch.setattr(live,'capture_bound_environment',capture)
    dep.verify_cpu_binding(record,ModelSpec('qwen3'))
    assert observed==[(shared,'a'*64)]
    monkeypatch.setattr(live,'capture_bound_environment',lambda *a:{**record['environment'],'UNIT_version':'changed'})
    with pytest.raises(ValueError,match='physical environment changed'):
        dep.verify_cpu_binding(record,ModelSpec('qwen3'))


@pytest.mark.parametrize('operation,action,refused',[
    ('serve','start',True),('qualify',None,True),('serve','status',False),
    ('serve','logs',False),('serve','stop',False)])
def test_new_yaml_environment_cannot_start_or_qualify_old_candidate_but_allows_inspection(monkeypatch,operation,action,refused):
    from werewolf.tom_llm import service_manager
    old='/data/yuxiao/tom-llm-reasoner-b2/env'
    manager=NS(binding=lambda:(None,{'profile':{'environment_prefix':old}},None))
    monkeypatch.setattr(service_manager,'Service',lambda *a,**kw:manager)
    monkeypatch.setattr(ops,'anchor',lambda r:NS(path=Path(r['path']),manifest_digest=r['digest']))
    c={'models':{'qwen3':{'environment_prefix':'/data/yuxiao/envs/untrusted-network-simulation-tom-baseline',
        'deployment':{'path':'/UNIT/deployment','digest':'a'*64}}}}
    args=NS(operation=operation,service_action=action,purpose='qualification')
    if refused:
        with pytest.raises(ValueError,match='new Deployment and GPU Admission'):ops.service(args,c,'qwen3')
    else:assert ops.service(args,c,'qwen3') is manager


@pytest.mark.parametrize('operation',['permit','run'])
def test_new_environment_refuses_old_admission_or_permit_before_execution(tmp_path,monkeypatch,operation):
    from werewolf.tom_llm import live_admission as live,experiment,baseline_cli
    monkeypatch.setattr(ops,'sys',NS(platform='linux'));monkeypatch.setattr(dep,'BASE',tmp_path)
    root=tmp_path/'deployment';root.mkdir()
    (root/'admission.json').write_text(json.dumps({'deployment_manifest_digest':'a'*64}))
    old_profile={'environment_prefix':'/data/yuxiao/tom-llm-reasoner-b2/env'}
    monkeypatch.setattr(ops,'anchor',lambda r:NS(path=Path(r['path']),manifest_digest=r['digest']))
    monkeypatch.setattr(dep,'open_deployment',lambda *a,**kw:(None,{'profile':old_profile}))
    monkeypatch.setattr(live,'publish_permit',lambda *a,**kw:pytest.fail('old environment must not grant Permit'))
    monkeypatch.setattr(live,'open_permit',lambda *a,**kw:(None,{'profile':old_profile}))
    monkeypatch.setattr(experiment,'open_run',lambda *a:(None,{'mode':'formal','model_order':['qwen3']},[]))
    monkeypatch.setattr(ops,'verify_artifact',lambda *a,**kw:NS(manifest_digest='p'*64))
    monkeypatch.setattr(baseline_cli,'execute',lambda *a:pytest.fail('old environment must not execute'))
    item={'environment_prefix':'/data/yuxiao/envs/untrusted-network-simulation-tom-baseline',
        'deployment':{'path':str(root),'digest':'a'*64},'admission':{'path':str(root),'digest':'b'*64}}
    cfg=tmp_path/'ops.json';cfg.write_text(json.dumps({'models':{'qwen3':item},'output_root':str(tmp_path/'runs')}))
    command=['prepare','--stage','permit'] if operation=='permit' else ['run','--execute-live','--manage-services']
    with pytest.raises(ValueError,match='new Deployment and GPU Admission'):
        main(['tom-llm','--config',str(cfg),*command,'--run-id','unit','--models','qwen3'])


def test_shared_environment_lock_has_exact_official_binary_hashes_and_runtime_pins():
    import re
    from urllib.parse import unquote,urlsplit
    from packaging.utils import parse_wheel_filename
    raw=(ROOT/'configs/paper/tom-llm-environment-linux-py312-v1.txt').read_text()
    versions={}
    for line in raw.splitlines():
        if not line or line.startswith('#'):continue
        name,url=line.split(' @ ');parsed=urlsplit(url)
        assert parsed.scheme=='https' and parsed.hostname in ('files.pythonhosted.org','github.com')
        assert re.fullmatch('sha256=[0-9a-f]{64}',parsed.fragment)
        filename=unquote(parsed.path.rsplit('/',1)[-1])
        _,version,_,tags=parse_wheel_filename(filename)
        assert name not in versions
        versions[name]=str(version)
        assert any(t.platform=='any' or 'x86_64' in t.platform for t in tags)
    assert len(versions)==213
    assert {name:versions[name] for name in ('vllm','torch','transformers','tokenizers','triton','flashinfer-python',
        'apache-tvm-ffi','nvidia-cuda-runtime','nvidia-cuda-nvcc')}=={
        'vllm':'0.27.0','torch':'2.13.0','transformers':'5.14.1','tokenizers':'0.22.2','triton':'3.7.1',
        'flashinfer-python':'0.6.16.post3','apache-tvm-ffi':'0.1.11','nvidia-cuda-runtime':'13.0.96','nvidia-cuda-nvcc':'13.0.88'}


def test_qualification_reuse_checks_bytes_not_head_and_unknown_files_fail(monkeypatch):
    now=deepcopy(dep.source_identity());now['source_inventory_clean']=True
    monkeypatch.setattr(dep,'source_identity',lambda:now)
    old=deepcopy(now);old['source_revision']='a'*40
    old['file_digests']['werewolf/tom_llm/baseline_cli.py']='b'*64
    # An old Source predates the new control modules.
    for name in ('operations','service_manager'):old['file_digests'].pop('werewolf/tom_llm/'+name+'.py',None)
    old['implementation_digest']=digest(old['file_digests'])
    evidence=dep.qualification_compatibility(old)
    assert evidence['qualified_source_digest']==digest(old)
    assert evidence['execution_source_digest']==digest(now)
    changed=deepcopy(old);changed['file_digests']['werewolf/tom_llm/gpu_probe.py']='c'*64
    changed['implementation_digest']=digest(changed['file_digests'])
    with pytest.raises(ValueError,match='new GPU qualification'):dep.qualification_compatibility(changed)
    changed=deepcopy(old);changed['file_digests']['werewolf/tom_llm/unknown_runtime.py']='d'*64
    changed['implementation_digest']=digest(changed['file_digests'])
    with pytest.raises(ValueError,match='new GPU qualification'):dep.qualification_compatibility(changed)
    changed=deepcopy(old);changed['source_inventory_clean']=False
    with pytest.raises(ValueError,match='committed'):dep.qualification_compatibility(changed)
    changed=deepcopy(old);changed['implementation_digest']='0'*64
    with pytest.raises(ValueError,match='digest drift'):dep.qualification_compatibility(changed)


def test_old_v1_admission_reads_offline_after_source_update_and_stopped_service(tmp_path,native_suites,monkeypatch):
    record=fixtures.model_record('qwen3',native_suites)
    record['source']['source_inventory_clean']=True
    rounds=[];roots=[];digests=[];deployment=None
    for run in (0,1):
        client,events,count=fixtures.fake_candidate(tmp_path/str(run),record,run)
        if deployment is None:deployment=client.deployment
        # Both physical unit instances must refer to the same immutable Deployment.
        client.deployment=deployment;client.instance['deployment_manifest_digest']=deployment.manifest_digest
        client.instance['instance_digest']=digest({k:v for k,v in client.instance.items() if k!='instance_digest'})
        root=tmp_path/str(run)/'round'
        seal=fixtures.collect_round(root,client,run=run,event_reader=lambda:events,memory_monitor=fixtures.Memory)
        roots.append(str(root));digests.append(seal.manifest_digest)
        rounds.append(admission.open_round(root,seal.manifest_digest,record,deployment_manifest_digest=deployment.manifest_digest)[1])
    payload={'schema_version':admission.ADMISSION_VERSION,'evidence_kind':'physical_gpu_synthetic',
        'deployment_root':str(deployment.path),'deployment_manifest_digest':deployment.manifest_digest,
        'deployment_digest':digest(record),'model_spec_digest':ModelSpec('qwen3').digest,
        'round_roots':roots,'round_manifest_digests':digests,'qualification':fixtures.validate_qualification(record,rounds),
        'formal_execution_authorized':False}
    artifact=publish_artifact(tmp_path/'admission',manifest_fields={'artifact_type':'mm_physical_gpu_admission',
        'schema_version':admission.ADMISSION_VERSION,'record_digest':digest(payload),'qualification_scope':'synthetic_only',
        'formal_execution_authorized':False},files={'admission.json':canonical_json_bytes(payload)})
    before=(artifact.path/'admission.json').read_bytes()
    monkeypatch.setattr(dep,'source_identity',lambda:{'source_revision':'f'*40})
    def forbidden(*a,**kw):raise AssertionError('historical reader accessed physical process/assets')
    monkeypatch.setattr(dep,'capture_instance',forbidden)
    monkeypatch.setattr(dep,'verify_model',forbidden)
    from werewolf.tom_llm import physical
    monkeypatch.setattr(physical,'capture_environment',forbidden)
    a,r=admission.open_admission(artifact.path,expected_manifest=artifact.manifest_digest,
        expected_deployment=deployment.manifest_digest,model_spec=ModelSpec('qwen3'))
    assert r==payload and r['formal_execution_authorized'] is False
    assert (a.path/'admission.json').read_bytes()==before
    with pytest.raises(ValueError,match='identity/domain'):dep.open_deployment(deployment.path,deployment.manifest_digest)
    with pytest.raises(ValueError):admission.open_admission(artifact.path,expected_manifest='0'*64,
        expected_deployment=deployment.manifest_digest,model_spec=ModelSpec('qwen3'))


@pytest.fixture(params=['historical','new'])
def managed(short_root,monkeypatch,request):
    from werewolf.tom_llm import service_manager as sm
    tmp_path=short_root
    base=tmp_path/'historical';base.mkdir()
    new=tmp_path/'TomBaseline-data'
    monkeypatch.setattr(sm.sys,'platform','linux');monkeypatch.setattr(dep,'BASE',base)
    monkeypatch.setattr(dep,'EXPERIMENT_BASE',new)
    monkeypatch.setattr(dep,'require_idle_gpu',lambda p:None)
    service=(base if request.param=='historical' else new)/'service'
    rpc=(base if request.param=='historical' else new)/'multimodel/qwen3'
    rpc.mkdir(parents=True,exist_ok=True)
    profile={'key':'qwen3','root':str(service),'argv_options':{'port':0},'gpu':{'uuid':'GPU-UNIT'},
        'environment_prefix':str(tmp_path/'env'),
        'environment_requirements':{**{k:str(service/k) for k in dep.CACHE_VARIABLES},'VLLM_RPC_BASE_PATH':str(rpc)}}
    spec=Service(tmp_path/'run','qwen3','a'*64)
    cmd=[str(tmp_path/'env/bin/python'),'-m','werewolf.tom_llm.live_service','--execute-live']
    monkeypatch.setattr(spec,'binding',lambda **kw:(NS(manifest_digest='a'*64),{'profile':profile},cmd))
    child=NS(pid=987654,done=False)
    child.poll=lambda:0 if child.done else None
    child.wait=lambda **kw:0
    identity={'pid':child.pid,'pgid':child.pid,'argv':cmd,'uid':os.getuid(),'cwd':str(ROOT),
        'executable':str(Path(cmd[0]).resolve()),'boot_id':'UNIT_BOOT','start_ticks':'1'}
    monkeypatch.setattr(spec,'identity',lambda pid:None if child.done else deepcopy(identity))
    calls=[]
    def launch(*args,**kw):
        calls.append(('launch',args,kw));assert kw['pass_fds'];return child
    monkeypatch.setattr(sm.subprocess,'Popen',launch)
    monkeypatch.setattr(spec,'capture',lambda:{'pid':child.pid})
    @contextmanager
    def pidfd(pid):
        if pid!=os.getpid():calls.append(('pidfd',pid))
        def send(sig):
            if sig:setattr(child,'done',True);calls.append(('signal',7654,sig))
        try:yield send
        finally:
            if pid!=os.getpid():calls.append(('close',7654))
    monkeypatch.setattr(sm,'_pidfd',pidfd)
    return spec,child,identity,calls


def test_explicit_start_stop_persistent_ownership_and_duplicate_refusal(managed):
    manager,child,identity,calls=managed
    with pytest.raises(ValueError,match='execute-live'):manager.start()
    assert manager.start(execute=True)['ready'] is True
    assert manager.status()['state']=='running'
    with pytest.raises(ValueError,match='already running'):manager.start(execute=True)
    assert sum(c[0]=='launch' for c in calls)==1
    assert manager.stop(execute=True)['state']=='exited'
    assert any(c[0]=='signal' for c in calls)
    # An exited launch can be inspected/stopped again without signalling.
    n=len(calls);manager.stop(execute=True);assert len(calls)==n


def test_managed_launch_binds_short_rpc_parent_without_inheriting_override(managed,monkeypatch):
    manager,child,identity,calls=managed
    p=manager.binding()[1]['profile'];rpc=p['environment_requirements']['VLLM_RPC_BASE_PATH']
    monkeypatch.setenv('VLLM_RPC_BASE_PATH','/UNIT/undeclared-parent')
    manager.start(execute=True)
    env=next(c[2]['env'] for c in calls if c[0]=='launch')
    assert env['VLLM_RPC_BASE_PATH']==rpc
    assert Path(rpc).is_dir() and not Path(rpc).is_relative_to(p['root'])
    for var in dep.CACHE_VARIABLES:assert Path(env[var]).is_relative_to(p['root'])
    assert manager.stop(execute=True)['state']=='exited'


def test_managed_launch_refuses_unwritable_rpc_before_popen(managed,monkeypatch):
    manager,child,identity,calls=managed
    rpc=Path(manager.binding()[1]['profile']['environment_requirements']['VLLM_RPC_BASE_PATH'])
    access=dep.os.access
    monkeypatch.setattr(dep.os,'access',lambda path,mode:False if Path(path)==rpc else access(path,mode))
    with pytest.raises(ValueError,match='existing writable RPC'):manager.start(execute=True)
    assert not calls and manager.child is None and not manager.pending_launch.exists()


def test_exited_launch_state_cannot_be_reused_for_another_deployment_anchor(managed,monkeypatch):
    manager,child,identity,calls=managed
    manager.start(execute=True);child.done=True
    assert manager.status()['state']=='exited'
    state=(manager.directory/'current.json').read_bytes()
    _,record,command=manager.binding()
    fresh=Service(manager.root,manager.key,'b'*64)
    monkeypatch.setattr(fresh,'binding',lambda **kw:(NS(manifest_digest='b'*64),record,command))
    with pytest.raises(ValueError,match='ownership drift'):fresh.status()
    with pytest.raises(ValueError,match='ownership drift'):fresh.start(execute=True)
    assert (manager.directory/'current.json').read_bytes()==state
    assert sum(c[0]=='launch' for c in calls)==1 and not any(c[0]=='signal' for c in calls)


@pytest.mark.parametrize('drift',['pid','group','uid','cwd','executable','argv','boot','state'])
def test_stop_never_signals_changed_identity_or_tampered_state(managed,drift):
    manager,child,identity,calls=managed;manager.start(execute=True)
    if drift=='pid':identity['start_ticks']='reused'
    elif drift=='group':identity['pgid']+=1
    elif drift=='uid':identity['uid']+=1
    elif drift=='cwd':identity['cwd']+='-changed'
    elif drift=='executable':identity['executable']+='-changed'
    elif drift=='argv':identity['argv']=[*identity['argv'],'--changed']
    elif drift=='boot':identity['boot_id']='changed'
    else:
        path=manager.directory/'current.json';r=json.loads(path.read_text());r['anchor']='x'*64;path.write_text(json.dumps(r))
    with pytest.raises(ValueError):manager.stop(execute=True)
    assert not any(c[0]=='signal' for c in calls)


def test_port_busy_and_early_exit_do_not_adopt_or_restart(managed,monkeypatch):
    manager,child,identity,calls=managed
    from werewolf.tom_llm import service_manager as sm
    import socket
    with socket.socket() as busy:
        busy.bind(('127.0.0.1',0));port=busy.getsockname()[1]
        a,r,cmd=manager.binding();r['profile']['argv_options']['port']=port
        monkeypatch.setattr(manager,'binding',lambda **kw:(a,r,cmd))
        with pytest.raises(OSError):manager.start(execute=True)
    assert not calls
    r['profile']['argv_options']['port']=0
    monkeypatch.setattr(child,'poll',lambda:1)
    with pytest.raises(ValueError,match='owned service exited'):manager.start(execute=True)
    assert sum(c[0]=='launch' for c in calls)==1


def test_qualification_requires_explicit_flag_before_any_capture(tmp_path,monkeypatch):
    cfg=tmp_path/'ops.yaml';cfg.write_text('models:\n  qwen3: {}\n')
    monkeypatch.setattr(ops,'service',lambda *a:pytest.fail('no capture without execute-live'))
    with pytest.raises(ValueError,match='execute-live'):
        main(['tom-llm','--config',str(cfg),'qualify','--model','qwen3','--round','0'])
    with pytest.raises(SystemExit) as error:build_parser().parse_args(['tom-llm','serve','start','--model','qwen3','--pid','123'])
    assert error.value.code==2


def test_source_inventory_includes_all_new_operation_code():
    from werewolf.tom_llm.experiment import source_v3
    source=source_v3()
    assert {'werewolf/tom_llm/operations.py','werewolf/tom_llm/service_manager.py','werewolf/cli.py'}<=set(source['file_digests'])
    assert source['qualification']=='SOFTWARE_IDENTITY_ONLY_NOT_ADMISSION'


def test_runner_readout_changes_do_not_hide_transport_or_recovery_changes():
    raw=b'def read_predictions():\n return 1\nclass HttpTransport:\n def send(self):\n  return 2\n'
    assert dep._runner_inference_body(raw)!=dep._runner_inference_body(raw.replace(b'return 1',b'return 100'))
    assert dep._runner_inference_body(raw)!=dep._runner_inference_body(raw.replace(b'return 2',b'return 100'))


def test_runner_inheritance_uses_anchored_blob_and_refuses_transport_changes(monkeypatch):
    import hashlib
    now=deepcopy(dep.source_identity());now['source_inventory_clean']=True
    old=deepcopy(now);old['source_revision']='a'*40
    name='werewolf/tom_llm/live_runner.py'
    raw=(ROOT/name).read_bytes().replace(b'open_run(root,current_source=False)',b'open_run(root)')
    previous=[raw]
    monkeypatch.setattr(dep,'source_identity',lambda:now)
    monkeypatch.setattr(dep.subprocess,'check_output',lambda *a,**kw:previous[0])
    old['file_digests'][name]=hashlib.sha256(raw).hexdigest();old['implementation_digest']=digest(old['file_digests'])
    assert dep.qualification_compatibility(old)['policy']=='classic7_physical_inference_compatibility_v2'
    previous[0]=raw.replace(b"c.request('POST'",b"c.request('DELETE'")
    with pytest.raises(ValueError,match='bytes drift'):dep.qualification_compatibility(old)
    old['file_digests'][name]=hashlib.sha256(previous[0]).hexdigest();old['implementation_digest']=digest(old['file_digests'])
    with pytest.raises(ValueError,match='new GPU qualification'):dep.qualification_compatibility(old)


@pytest.mark.parametrize('before,after',[
    (b"sealed['rows']==rows",b'True'),
    (b"sealed['task_digests']==run['task_digests']",b'True'),
    (b"request_root.is_dir() and",b'True or'),
    (b"record['row']==row",b'True'),
    (b'rows.append(row)',b'rows.append({})'),
    (b"require(received is not None,'incomplete task cannot seal')",b"require(True,'incomplete task cannot seal')"),
    (b"done+=int((p/'result').exists())",b'done+=100'),
    (b'current_source=False',b'current_source=True'),
    (b'physical=False',b'physical=True'),
])
def test_known_read_mode_compatibility_does_not_exclude_validation_or_sealing(monkeypatch,before,after):
    import hashlib
    now=deepcopy(dep.source_identity());now['source_inventory_clean']=True
    old=deepcopy(now);old['source_revision']='a'*40
    name='werewolf/tom_llm/live_runner.py';raw=(ROOT/name).read_bytes()
    assert before in raw
    changed=raw.replace(before,after)
    old['file_digests'][name]=hashlib.sha256(changed).hexdigest();old['implementation_digest']=digest(old['file_digests'])
    monkeypatch.setattr(dep,'source_identity',lambda:now)
    monkeypatch.setattr(dep.subprocess,'check_output',lambda *a,**kw:changed)
    with pytest.raises(ValueError,match='new GPU qualification'):dep.qualification_compatibility(old)


def test_only_the_four_known_calls_are_normalized():
    from werewolf.tom_llm import live_runner
    import inspect
    raw=inspect.getsource(live_runner).encode()
    previous=raw.replace(b'open_run(root,current_source=False)',b'open_run(root)')
    previous=previous.replace(b'open_permit(root,key,digest_anchor,physical=False)',b'open_permit(root,key,digest_anchor)')
    assert dep._runner_inference_body(raw)==dep._runner_inference_body(previous)
    # False at another call site is not a compatible rewrite.
    other=b'def recover(root):\n return open_run(root,current_source=False)\n'
    assert dep._runner_inference_body(other)!=dep._runner_inference_body(other.replace(b',current_source=False',b''))


def test_qualified_service_profile_preserves_effective_arguments_and_environment(monkeypatch):
    from werewolf.tom_llm import live_admission as live
    spec=ModelSpec('qwen3');old=dep.profile(spec,gpu_uuid='GPU-UNIT',candidate_root=dep.BASE/'multimodel/qwen3/unit-candidate')
    p=live.service_profile(dep.BASE/'experiments/test',spec,{'profile':old})
    assert {k:v for k,v in p['argv_options'].items() if k!='chat-template'}=={k:v for k,v in old['argv_options'].items() if k!='chat-template'}
    wrong=deepcopy(old);wrong['argv_options']['max-model-len']=4096
    monkeypatch.setattr(live,'profile',lambda *a,**kw:wrong)
    with pytest.raises(ValueError,match='inference arguments'):live.service_profile(dep.BASE/'experiments/test',spec,{'profile':old})


@pytest.mark.parametrize('reuse',[False,True])
def test_prepare_model_reuses_existing_pipeline_and_refuses_old_candidate(short_root,monkeypatch,reuse):
    from werewolf.tom_llm import physical,qualification_inputs as qi,live_admission as live
    tmp_path=short_root
    monkeypatch.setattr(dep,'BASE',tmp_path)
    root=tmp_path/'multimodel/yi/new';calls=[]
    def artifact(path):return NS(path=Path(path),manifest_digest='a'*64)
    def publish(path,kind,payload):calls.append((kind,payload));return artifact(path)
    monkeypatch.setattr(ops,'anchor',lambda r:artifact(r['path']))
    monkeypatch.setattr(physical,'publish_receipt',publish)
    monkeypatch.setattr(physical,'verify_model',lambda s,p:{'directory':p})
    monkeypatch.setattr(physical,'read_receipt',lambda *a:calls.append(('reuse',a)))
    monkeypatch.setattr(live,'capture_bound_environment',lambda p,d:{'prefix':p,'prior_audit_digest':d})
    monkeypatch.setattr(qi,'publish_suite',lambda path,*a:artifact(path))
    monkeypatch.setattr(qi,'publish_budget',lambda path,*a:artifact(path))
    monkeypatch.setattr(qi,'open_suite',lambda *a:None)
    monkeypatch.setattr(qi,'open_budget',lambda *a:None)
    def deployment(path,spec,**kw):calls.append(('deploy',kw));return artifact(path)
    monkeypatch.setattr(dep,'prepare_deployment',deployment)
    item={'candidate_root':str(root),'model_directory':str(tmp_path/'weights'),'environment_prefix':str(tmp_path/'env'),
        'tokenizer_assets':str(tmp_path/'tokens'),'prior_audit_digest':'b'*64}
    if reuse:
        for name in ('assets','environment','suite','budget'):item[name]={'path':str(tmp_path/name),'digest':'a'*64}
    c={'models':{'yi':item},'gpu_uuid':'GPU-UNIT','cohort':{'path':str(tmp_path/'cohort'),'digest':'a'*64}}
    result=ops.prepare_model(c,'yi')
    assert set(result)=={'assets','environment','suite','budget','deployment'}
    assert calls[-1][0]=='deploy' and calls[-1][1]['model_root']==item['model_directory']
    assert sum(v[0]=='reuse' for v in calls)==(2 if reuse else 0)
    (root/'deployment').mkdir(parents=True)
    n=len(calls)
    with pytest.raises(ValueError,match='deployment exists'):ops.prepare_model(c,'yi')
    assert len(calls)==n


def test_qualification_uses_model_interpreter_and_never_retries_failure(tmp_path,monkeypatch):
    import subprocess
    cfg=tmp_path/'ops.yaml';cfg.write_text('models:\n  qwen3: {}\n')
    monkeypatch.setattr(ops,'sys',NS(platform='linux'))
    p={'root':str(tmp_path/'candidate'),'environment_prefix':str(tmp_path/'model-env'),'environment_requirements':{}}
    manager=NS(directory=tmp_path,capture=lambda:{'pid':987654},binding=lambda:(NS(path=tmp_path/'deployment',manifest_digest='a'*64),{'profile':p},[]))
    monkeypatch.setattr(ops,'service',lambda *a:manager)
    calls=[]
    monkeypatch.setattr(subprocess,'run',lambda command,**kw:calls.append((command,kw)) or NS(returncode=1))
    with pytest.raises(ValueError,match='no retry'):main(['tom-llm','--config',str(cfg),'qualify','--model','qwen3','--round','0','--execute-live'])
    assert len(calls)==1
    cmd=calls[0][0]
    assert cmd[0]==p['environment_prefix']+'/bin/python' and cmd[1]==str(ROOT/'scripts/tom_multimodel_admission.py')
    assert cmd[cmd.index('--pid')+1]=='987654' and cmd[-1]=='--execute-synthetic-gpu-qualification'
    assert list(tmp_path.glob('client-*.log'))


def test_input_anchor_and_parent_traversal_are_not_silently_accepted(tmp_path):
    a=publish_artifact(tmp_path/'a',manifest_fields={'artifact_type':'unit','schema_version':'unit'},files={})
    with pytest.raises(ValueError,match='digest mismatch'):ops.anchor({'path':str(a.path),'digest':'0'*64})
    from werewolf.tom_llm.experiment import safe_path
    with pytest.raises(ValueError,match='parent traversal'):safe_path(tmp_path/'..'/'outside')


@pytest.mark.parametrize('key,variable',[('yi','UNS_LLM_YI_ASSETS'),('mistral','UNS_LLM_MISTRAL_ASSETS')])
def test_native_cpu_preparation_needs_no_weights_environment_or_gpu(tmp_path,monkeypatch,key,variable):
    from werewolf.tom_llm import physical,qualification_inputs as qi,multimodel as mm,live_admission as live
    assets=os.environ.get(variable)
    if not assets:pytest.skip('NOT_RUN: official local tokenizer assets required')
    def forbidden(*a,**kw):pytest.fail('CPU preparation accessed weights, environment or GPU')
    monkeypatch.setattr(physical,'verify_model',forbidden)
    monkeypatch.setattr(live,'capture_bound_environment',forbidden)
    monkeypatch.setattr(dep,'profile',forbidden)
    monkeypatch.setattr(dep,'prepare_deployment',forbidden)
    monkeypatch.setattr(dep,'gpu_snapshot',forbidden)
    c={'output_root':str(tmp_path/'outputs'),'models':{key:{'tokenizer_assets':assets}}}
    parity=ops.prepare_inputs(c,key,'tokenizer')['tokenizer']
    assert parity['status']=='CPU_PARITY_PASS' and parity['gpu_admission'] is False
    result=ops.prepare_inputs(c,key,'suite')
    suite=qi.open_suite(result['suite']['path'],ModelSpec(key),result['suite']['digest'])
    assert len(suite['cases'])==12 and len(suite['schedule'])==38
    c['models'][key].update(result)
    assert ops.prepare_inputs(c,key,'suite')==result
    # Synthetic full-150 population with two public requests/game. Real tokenizer,
    # real budget writer/reader; no claim that these are the server's 10,714 rows.
    cohort=publish_artifact(tmp_path/'cohort',manifest_fields={'artifact_type':'unit','schema_version':'unit'},files={})
    c['cohort']=ops.ref(cohort)
    games=[f'UNIT-{i:03}' for i in range(150)]
    public=qi.synthetic_tasks()[:2]
    tasks=[{**deepcopy(task),'game_id':g,'fold':i//30} for i,g in enumerate(games) for task in public]
    m=NS(path=cohort.path,manifest_digest=cohort.manifest_digest,
        manifest={'selection':{'game_ids':games},'reference':{'seal_digest':'b'*64}})
    monkeypatch.setattr(mm,'read_cohort',lambda *a,**kw:(m,tasks))
    report=ops.prepare_inputs(c,key,'budget')
    assert report['token_budget']['request_count']==len(tasks)==300
    assert report['token_budget']['context_length']==8192 and report['token_budget']['output_cap']==512
    assert report['token_budget']['maximum_input_tokens']<=7680
    c['models'][key]['budget']=report['budget']
    assert ops.prepare_inputs(c,key,'budget')==report


def test_cpu_stage_cli_saves_only_generated_references_and_never_defaults_to_download(tmp_path,monkeypatch):
    monkeypatch.setattr(dep,'BASE',tmp_path)
    monkeypatch.setattr(ops,'sys',NS(platform='linux'))
    cfg=tmp_path/'ops.yaml';cfg.write_text('output_root: '+str(tmp_path/'outputs')+'\nmodels:\n  yi:\n    tokenizer_assets: /UNIT/small-assets\n')
    calls=[]
    def prepare(c,key,stage):
        calls.append((key,stage,deepcopy(c)))
        return {stage:{'path':str(tmp_path/stage),'digest':'a'*64}}
    monkeypatch.setattr(ops,'prepare_inputs',prepare)
    argv=['tom-llm','--config',str(cfg),'prepare','--models','yi']
    original=cfg.read_bytes()
    assert main([*argv,'--stage','suite'])==0
    assert cfg.read_bytes()==original
    assert main([*argv,'--stage','suite','--save-config'])==0
    assert ops.load_config(cfg)['models']['yi']['suite']=={'path':str(tmp_path/'suite'),'digest':'a'*64}
    assert main([*argv,'--stage','budget','--save-config'])==0
    assert calls[-1][2]['models']['yi']['suite']['digest']=='a'*64
    assert ops.load_config(cfg)['models']['yi']['budget']['digest']=='a'*64
    with pytest.raises(ValueError,match='full fixed 150'):main([*argv,'--stage','budget','--games','10'])
    with pytest.raises(ValueError,match='generated Artifact'):main([*argv,'--stage','tokenizer','--save-config'])
    assert not list(tmp_path.glob('*.tmp'))


def test_explicit_config_update_refuses_intervening_edits_and_sealed_directories(tmp_path,monkeypatch):
    monkeypatch.setattr(dep,'BASE',tmp_path)
    monkeypatch.setattr(ops,'sys',NS(platform='linux'))
    cfg=tmp_path/'ops.yaml';original=b'models: {yi: {}}\n';cfg.write_bytes(original)
    cfg.write_bytes(original+b'# user edit\n')
    with pytest.raises(ValueError,match='configuration changed'):ops.save_configuration(cfg,original,{'models':{'yi':{}}})
    assert cfg.read_bytes()==original+b'# user edit\n'
    (tmp_path/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='sealed Artifact'):ops.save_configuration(cfg,cfg.read_bytes(),{'models':{'yi':{}}})


def test_preparing_a_new_deployment_cannot_silently_use_an_old_admission(tmp_path,monkeypatch):
    monkeypatch.setattr(dep,'BASE',tmp_path)
    monkeypatch.setattr(ops,'sys',NS(platform='linux'))
    admission_root=tmp_path/'old-admission';admission_root.mkdir()
    (admission_root/'admission.json').write_text(json.dumps({'deployment_manifest_digest':'a'*64}))
    cfg=tmp_path/'ops.json'
    c={'output_root':str(tmp_path/'runs'),'models':{'yi':{'admission':{'path':str(admission_root),'digest':'c'*64},
        'deployment':{'path':str(tmp_path/'new-deployment'),'digest':'b'*64}}}}
    cfg.write_text(json.dumps(c))
    monkeypatch.setattr(ops,'anchor',lambda r:NS(path=Path(r['path']),manifest_digest=r['digest']))
    from werewolf.tom_llm import live_admission as live
    monkeypatch.setattr(live,'publish_permit',lambda *a,**kw:pytest.fail('mismatched Admission must not issue Permit'))
    with pytest.raises(ValueError,match='Admission/Deployment mismatch'):
        main(['tom-llm','--config',str(cfg),'prepare','--stage','permit','--models','yi','--run-id','unit'])


@pytest.mark.parametrize('kind',['model_directory','environment_prefix','tokenizer_assets','deployment'])
def test_cpu_output_paths_cannot_modify_existing_assets_or_candidates(tmp_path,monkeypatch,kind):
    monkeypatch.setattr(dep,'BASE',tmp_path)
    monkeypatch.setattr(ops,'sys',NS(platform='linux'))
    root=tmp_path/'protected'
    item={'cpu_root':str(root/'nested'),'tokenizer_assets':str(tmp_path/'tokens')}
    if kind=='deployment':item[kind]={'path':str(root/'deployment'),'digest':'a'*64}
    else:item[kind]=str(root)
    c={'models':{'yi':item}}
    with pytest.raises(ValueError,match='inside'):
        ops.prepare_inputs(c,'yi','suite')
    assert not root.exists()


def test_linux_cpu_cli_still_rejects_outputs_outside_independent_root(tmp_path,monkeypatch):
    monkeypatch.setattr(dep,'BASE',tmp_path/'baseline')
    monkeypatch.setattr(ops,'sys',NS(platform='linux'))
    outside=tmp_path/'outside';cfg=tmp_path/'ops.json'
    cfg.write_text(json.dumps({'output_root':str(outside),'models':{'yi':{}}}))
    monkeypatch.setattr(ops,'prepare_inputs',lambda *a:pytest.fail('outside path reached CPU publisher'))
    with pytest.raises(ValueError,match='writes outside independent baseline root'):
        main(['tom-llm','--config',str(cfg),'prepare','--stage','suite','--models','yi'])
    assert not outside.exists()


def test_spawn_failure_preserves_intent_and_refuses_an_automatic_reset(managed,monkeypatch):
    manager,child,identity,calls=managed
    from werewolf.tom_llm import service_manager as sm
    def failure(*a,**kw):calls.append(('launch_error',));raise OSError('UNIT exec failure')
    monkeypatch.setattr(sm.subprocess,'Popen',failure)
    with pytest.raises(ValueError,match='launch incomplete'):manager.start(execute=True)
    assert calls==[('launch_error',)] and manager.status()['state']=='launch_incomplete'
    assert list(manager.directory.glob('stdout-*.log'))
    events=[json.loads(line) for line in manager.pending_launch.read_text().splitlines()]
    assert [e['kind'] for e in events]==['launch_intent']
    with pytest.raises(ValueError,match='no restart'):manager.start(execute=True)
    with pytest.raises(ValueError,match='no automatic restart or stop'):manager.stop(execute=True)
    assert calls==[('launch_error',)]


def test_ready_timeout_preserves_complete_ownership_and_process_exit_is_not_retried(managed,monkeypatch):
    manager,child,identity,calls=managed
    monkeypatch.setattr(manager,'capture',lambda:(_ for _ in ()).throw(ValueError('UNIT not ready')))
    with pytest.raises(ValueError,match='retain service and log'):manager.start(execute=True,timeout=0)
    assert manager.status()['state']=='running' and not any(c[0]=='signal' for c in calls)
    assert not manager.pending_launch.exists()
    archive=list(manager.directory.glob('launch-*.jsonl'))
    assert len(archive)==1
    assert [json.loads(line)['kind'] for line in archive[0].read_text().splitlines()]==[
        'launch_intent','popen_returned','ownership_committed']
    child.done=True
    assert manager.status()['state']=='exited'
    manager.capture=Service.capture.__get__(manager)
    with pytest.raises(ValueError,match='owned service exited'):manager.capture()
    assert not any(c[0]=='signal' for c in calls)


@pytest.mark.parametrize('failure',['identity_error','identity_missing','state_write','state_replace','state_directory_sync','archive'])
def test_post_popen_failure_cannot_adopt_signal_or_restart_unknown_process(managed,monkeypatch,failure):
    manager,child,identity,calls=managed
    from werewolf.tom_llm import service_manager as sm
    if failure=='identity_error':
        monkeypatch.setattr(manager,'identity',lambda pid:(_ for _ in ()).throw(PermissionError('UNIT /proc inaccessible')))
    elif failure=='identity_missing':monkeypatch.setattr(manager,'identity',lambda pid:None)
    elif failure=='state_write':
        serialize=sm.canonical_json_bytes
        def write(record):
            if 'identity' in record:raise OSError('UNIT state write failed')
            return serialize(record)
        monkeypatch.setattr(sm,'canonical_json_bytes',write)
    elif failure=='state_replace':
        monkeypatch.setattr(sm.os,'replace',lambda *a:(_ for _ in ()).throw(OSError('UNIT state replace failed')))
    elif failure=='state_directory_sync':
        sync=sm._fsync_directory;count=[0]
        def directory_sync(path):
            count[0]+=1
            if count[0]==2:raise OSError('UNIT state directory sync failed')
            sync(path)
        monkeypatch.setattr(sm,'_fsync_directory',directory_sync)
    else:monkeypatch.setattr(sm.os,'rename',lambda *a:(_ for _ in ()).throw(OSError('UNIT journal archive failed')))
    with pytest.raises((ValueError,OSError)):manager.start(execute=True)
    assert sum(c[0]=='launch' for c in calls)==1 and child.done is False
    journal=[json.loads(line) for line in manager.pending_launch.read_text().splitlines()]
    assert journal[0]['kind']=='launch_intent' and journal[0]['argv']==identity['argv']
    assert journal[1]=={'kind':'popen_returned','pid_hint':child.pid}
    if failure in ('state_directory_sync','archive'):assert (manager.directory/'current.json').exists()
    # A new CLI has no Popen object. It must not infer ownership from a PID hint,
    # or trust even a complete-looking current.json while the intent is pending.
    fresh=Service(manager.root,manager.key,manager.expected)
    monkeypatch.setattr(fresh,'binding',manager.binding)
    monkeypatch.setattr(fresh,'identity',lambda pid:pytest.fail('pending launch must not adopt a PID'))
    assert fresh.status()['state']=='launch_incomplete' and fresh.status()['ready'] is False
    assert 'pid' not in fresh.status()
    assert fresh.logs()==''
    with pytest.raises(ValueError,match='incomplete launch'):fresh.start(execute=True)
    with pytest.raises(ValueError,match='incomplete launch'):fresh.capture()
    with pytest.raises(ValueError,match='incomplete launch'):fresh.stop(execute=True)
    assert not any(c[0]=='signal' for c in calls) and sum(c[0]=='launch' for c in calls)==1
    child.done=True
    assert fresh.status()['state']=='launch_incomplete'  # No PID-based automatic reset.


def test_intent_must_be_durable_before_popen(managed,monkeypatch):
    manager,child,identity,calls=managed
    from werewolf.tom_llm import service_manager as sm
    monkeypatch.setattr(sm,'_fsync_directory',lambda path:(_ for _ in ()).throw(OSError('UNIT intent not durable')))
    with pytest.raises(OSError,match='intent not durable'):manager.start(execute=True)
    assert not calls and manager.status()['state']=='launch_incomplete'
    with pytest.raises(ValueError,match='incomplete launch'):manager.start(execute=True)


def test_interrupt_after_popen_keeps_pending_ownership_fence(managed,monkeypatch):
    manager,child,identity,calls=managed
    monkeypatch.setattr(manager,'identity',lambda pid:(_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):manager.start(execute=True)
    assert manager.status()['state']=='launch_incomplete' and child.done is False
    with pytest.raises(ValueError,match='incomplete launch'):manager.stop(execute=True)
    assert sum(c[0]=='launch' for c in calls)==1 and not any(c[0]=='signal' for c in calls)


def test_pending_marker_appearing_during_state_read_also_refuses_stop(managed,monkeypatch):
    manager,child,identity,calls=managed;manager.start(execute=True)
    from werewolf.tom_llm import service_manager as sm
    decode=sm.strict_json
    def interrupted_read(raw):
        record=decode(raw)
        manager.pending_launch.write_bytes(b'{}\n')
        return record
    monkeypatch.setattr(sm,'strict_json',interrupted_read)
    with pytest.raises(ValueError,match='incomplete launch'):manager.stop(execute=True)
    assert not any(c[0]=='signal' for c in calls)


def test_pidfd_race_and_stop_timeout_never_signal_another_pid_or_force_kill(managed,monkeypatch):
    manager,child,identity,calls=managed;manager.start(execute=True)
    from werewolf.tom_llm import service_manager as sm
    original=sm._pidfd
    @contextmanager
    def raced(pid):
        with original(pid) as send:
            identity['start_ticks']='reused'
            yield send
    monkeypatch.setattr(sm,'_pidfd',raced)
    with pytest.raises(ValueError,match='PID identity changed'):manager.stop(execute=True)
    assert not any(c[0]=='signal' for c in calls) and calls[-1]==('close',7654)
    identity['start_ticks']='1'
    @contextmanager
    def unresponsive(pid):
        with original(pid):yield lambda sig:calls.append(('signal',7654,sig))
    monkeypatch.setattr(sm,'_pidfd',unresponsive)
    with pytest.raises(ValueError,match='no force kill'):manager.stop(execute=True,timeout=0)
    assert [(c[0],c[2]) for c in calls if c[0]=='signal']==[('signal',sm.signal.SIGTERM)]


def test_missing_python_pidfd_bindings_use_kernel_probe_before_launch(managed,monkeypatch):
    manager,child,identity,calls=managed
    from werewolf.tom_llm import service_manager as sm
    monkeypatch.delattr(sm.os,'pidfd_open',raising=False)
    monkeypatch.delattr(sm.signal,'pidfd_send_signal',raising=False)
    probe=[]
    @contextmanager
    def pidfd(pid):
        probe.append(('open',pid))
        try:yield lambda sig:probe.append(('send',sig))
        finally:probe.append(('close',pid))
    monkeypatch.setattr(sm,'_pidfd',pidfd,raising=False)
    monkeypatch.setattr(dep,'require_idle_gpu',lambda p:(_ for _ in ()).throw(ValueError('UNIT reached GPU gate')))
    with pytest.raises(ValueError,match='UNIT reached GPU gate'):manager.start(execute=True)
    assert probe==[('open',os.getpid()),('send',0),('close',os.getpid())]
    assert not calls and not manager.pending_launch.exists()


def test_gpu_lock_fd_survives_controller_close_until_cpu_child_exits(managed,monkeypatch,tmp_path):
    """Real POSIX FD/flock test, isolated CPU sleep child; not a vLLM/Linux test."""
    import fcntl,sys
    manager,child,identity,calls=managed
    from werewolf.tom_llm import service_manager as sm
    a,r,_=manager.binding()
    cmd=[sys.executable,'-I','-S','-B','-c','import time; time.sleep(30)']
    monkeypatch.setattr(manager,'binding',lambda **kw:(a,r,cmd))
    monkeypatch.setattr(sm.subprocess,'Popen',CpuChild)
    def actual_identity(pid):
        if manager.child is None or manager.child.poll() is not None:return None
        return {**identity,'pid':pid,'pgid':pid,'argv':cmd,'executable':str(Path(cmd[0]).resolve())}
    monkeypatch.setattr(manager,'identity',actual_identity)
    monkeypatch.setattr(manager,'capture',lambda:{'pid':manager.child.pid})
    try:
        state=manager.start(execute=True)
        assert state['pid']==manager.child.pid
        with (dep.BASE/'baseline-gpu.lock').open('a+b') as other:
            with pytest.raises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
            # The other output root/model must contend on the same inherited FD.
            other_base=dep.EXPERIMENT_BASE if Path(r['profile']['root']).is_relative_to(dep.BASE) else dep.BASE
            other_profile=deepcopy(r['profile']);other_profile['root']=str(other_base/'other-service')
            second=Service(tmp_path/'other-run','yi','b'*64)
            monkeypatch.setattr(second,'binding',lambda **kw:(a,{'profile':other_profile},cmd))
            with pytest.raises(BlockingIOError):second.start(execute=True)
            assert second.child is None and not second.pending_launch.exists()
            manager.child.terminate();manager.child.wait(timeout=5)
            fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (dep.EXPERIMENT_BASE/'baseline-gpu.lock').exists()
    finally:
        if manager.child is not None and manager.child.poll() is None:
            manager.child.terminate();manager.child.wait(timeout=5)


@pytest.mark.parametrize('key',['qwen3','yi','mistral'])
def test_new_candidate_profiles_keep_historical_asset_and_model_protocol_contracts(key):
    spec=ModelSpec(key);shared=dep.ENV_BASE/'untrusted-network-simulation-tom-baseline'
    model=dep.BASE/'models'/key
    old=dep.profile(spec,gpu_uuid='GPU-UNIT',candidate_root=dep.BASE/'multimodel'/key/'unit-candidate',
        model_directory=model,environment_prefix=shared)
    root=dep.EXPERIMENT_BASE/'multimodel'/key/'unit-candidate'
    new=dep.profile(spec,gpu_uuid='GPU-UNIT',candidate_root=root,model_directory=model,environment_prefix=shared)
    assert dep.baseline_root(root)==dep.EXPERIMENT_BASE
    assert new['argv_options']=={**old['argv_options'],'chat-template':str(root/'deployment/chat_template.jinja')}
    ignored=set(dep.CACHE_VARIABLES)|{'PYTHONPATH','VLLM_RPC_BASE_PATH'}
    assert {k:v for k,v in new['environment_requirements'].items() if k not in ignored}=={
        k:v for k,v in old['environment_requirements'].items() if k not in ignored}
    assert old['environment_requirements']['VLLM_RPC_BASE_PATH']==str(dep.BASE/'multimodel'/key)
    assert new['environment_requirements']['VLLM_RPC_BASE_PATH']==str(dep.EXPERIMENT_BASE/'multimodel'/key)
    for k in dep.CACHE_VARIABLES:assert Path(new['environment_requirements'][k]).is_relative_to(root)
    assert new['model_spec_digest']==old['model_spec_digest']==spec.digest
    # Only the output contract changed, never the allowed physical asset roots.
    with pytest.raises(ValueError,match='physical model path'):
        dep.profile(spec,gpu_uuid='GPU-UNIT',candidate_root=root,model_directory=root/'model',environment_prefix=shared)
    with pytest.raises(ValueError,match='environment'):
        dep.profile(spec,gpu_uuid='GPU-UNIT',candidate_root=root,model_directory=model,environment_prefix=root/'env')
    with pytest.raises(ValueError,match='isolated candidate'):
        dep.profile(spec,gpu_uuid='GPU-UNIT',candidate_root=dep.EXPERIMENT_BASE/'multimodel'/key,
            model_directory=model,environment_prefix=shared)


@pytest.mark.parametrize('key',['qwen3','yi','mistral'])
def test_new_yaml_saves_anchored_historical_references_without_rewriting_them(tmp_path,monkeypatch,key):
    old=tmp_path/'historical';new=tmp_path/'TomBaseline-data';new.mkdir()
    monkeypatch.setattr(dep,'BASE',old);monkeypatch.setattr(dep,'EXPERIMENT_BASE',new)
    monkeypatch.setattr(ops,'sys',NS(platform='linux'))
    a=publish_artifact(old/'receipt',manifest_fields={'artifact_type':'unit','schema_version':'unit'},files={'record.json':b'{}'})
    protected={p:p.read_bytes() for p in a.path.iterdir()}
    cfg=new/'Qwen3-8b.yaml';cfg.write_text('models: {'+key+': {}}\n')
    original=cfg.read_bytes();updated=ops.load_config(cfg)
    updated['models'][key]['assets']=ops.ref(a)
    ops.save_configuration(cfg,original,updated)
    assert ops.load_config(cfg)==updated
    assert ops.anchor(updated['models'][key]['assets']).manifest_digest==a.manifest_digest
    assert all(p.read_bytes()==raw for p,raw in protected.items())
    wrong={**ops.ref(a),'digest':'0'*64}
    with pytest.raises(ValueError,match='digest mismatch'):ops.anchor(wrong)
    alias=new/'alias.yaml';alias.symlink_to(cfg)
    with pytest.raises(ValueError,match='symlink'):ops.load_config(alias)
    with pytest.raises(ValueError,match='symlink'):ops.save_configuration(alias,cfg.read_bytes(),updated)
    with pytest.raises(ValueError,match='configuration changed'):ops.save_configuration(cfg,original,updated)
    for invalid in (tmp_path/'outside',Path(str(new)+'-other')/'outputs',new/'..'/'outside'):
        with pytest.raises(ValueError):dep.baseline_root(invalid)
    (new/'alias').symlink_to(old,target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):dep.baseline_root(new/'alias'/'outputs')
    assert not list(new.glob('*.tmp')) and all(p.read_bytes()==raw for p,raw in protected.items())


@pytest.mark.parametrize('key',['qwen3','yi','mistral'])
def test_linux_cli_new_output_root_and_cpu_inputs_share_the_same_path_gate(tmp_path,monkeypatch,key):
    from werewolf.tom_llm import qualification_inputs as qi
    old=tmp_path/'historical';new=tmp_path/'TomBaseline-data';new.mkdir()
    monkeypatch.setattr(dep,'BASE',old);monkeypatch.setattr(dep,'EXPERIMENT_BASE',new)
    monkeypatch.setattr(ops,'sys',NS(platform='linux'))
    outputs=[]
    monkeypatch.setattr(qi,'publish_suite',lambda path,*a:outputs.append(path) or NS(path=path,manifest_digest='a'*64))
    monkeypatch.setattr(qi,'open_suite',lambda *a:None)
    cfg=new/'Qwen3-8b.yaml'
    c={'output_root':str(new/'experiments'),'models':{key:{'tokenizer_assets':str(old/'tokens'/key)}}}
    cfg.write_text(json.dumps(c))
    argv=['tom-llm','--config',str(cfg),'prepare','--stage','suite','--models',key,'--save-config']
    assert main(argv)==0 and outputs==[new/'experiments/cpu'/key/'suite']
    assert ops.load_config(cfg)['models'][key]['suite']['path']==str(outputs[0])
    c['models'][key]['cpu_root']=str(tmp_path/'outside');cfg.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='outside independent'):main(argv)
    assert len(outputs)==1
    c['models'][key]['cpu_root']=str(new/'cpu')
    c['models'][key]['candidate_root']=str(tmp_path/'foreign-candidate');cfg.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='outside independent'):main(argv)
    assert len(outputs)==1


@pytest.mark.parametrize('key',['qwen3','yi','mistral'])
def test_manual_cpu_cli_new_root_rejects_symlink_and_traversal_before_asset_reads(tmp_path,monkeypatch,key):
    from scripts import tom_multimodel_admission as cli
    from werewolf.tom_llm import physical
    old=tmp_path/'historical';new=tmp_path/'TomBaseline-data';new.mkdir()
    monkeypatch.setattr(dep,'BASE',old);monkeypatch.setattr(dep,'EXPERIMENT_BASE',new)
    monkeypatch.setattr(cli,'sys',NS(platform='linux'))
    reads=[]
    def verify(spec,path):
        reads.append((spec.key,path))
        return {'schema_version':physical.VERSION+'_model','evidence_kind':'physical_cpu','UNIT_ONLY':True}
    monkeypatch.setattr(physical,'verify_model',verify)
    def invoke(target):
        monkeypatch.setattr('sys.argv',['tool','assets','--model',key,'--directory',str(old/'models'/key),'--output',str(target)])
        return cli.main()
    invoke(new/'receipt')
    assert ops.anchor({'path':str(new/'receipt'),'digest':json.loads((new/'receipt/manifest.json').read_text())['manifest_digest']})
    (new/'alias').symlink_to(new/'receipt',target_is_directory=True)
    for invalid in (new/'alias'/'nested',new/'..'/'historical',tmp_path/'outside'):
        with pytest.raises(SystemExit) as error:invoke(invalid)
        assert error.value.code==2
    assert reads==[(key,str(old/'models'/key))]


def test_direct_baseline_cli_defaults_to_new_experiment_root(monkeypatch):
    from werewolf.tom_llm import baseline_cli,baseline_analysis
    observed=[]
    monkeypatch.setattr(baseline_cli,'prepare',lambda args,root:observed.append(root) or {})
    monkeypatch.setattr(baseline_analysis,'analyze',lambda root,destination,**kw:observed.append((root,destination)) or {})
    baseline_cli.main(['--action','prepare','--run-id','unit-run'])
    baseline_cli.main(['--action','analyze','--run-id','unit-run'])
    root=dep.EXPERIMENT_BASE/'experiments/unit-run'
    assert observed==[root,(root,root/'analysis')]


@pytest.fixture
def pidfd_libc(monkeypatch):
    """Inject only libc/syscall and platform; exercise the real pidfd adapter on Mac."""
    import ctypes
    from werewolf.tom_llm import service_manager as sm
    observed=NS(calls=[],closed=[],loads=[],errors={},fd=7654)
    def syscall(number,*args):
        observed.calls.append((number,*args))
        if number.value in observed.errors:
            ctypes.set_errno(observed.errors[number.value]);return -1
        assert number.value in (434,424)
        return observed.fd if number.value==434 else 0
    def load(name,*,use_errno):
        observed.loads.append((name,use_errno));return NS(syscall=syscall)
    monkeypatch.setattr(sm,'sys',NS(platform='linux'))
    monkeypatch.setattr(sm.os,'uname',lambda:NS(machine='x86_64'))
    monkeypatch.setattr(sm.ctypes,'CDLL',load)
    close=os.close
    def release(fd):
        if fd==observed.fd:observed.closed.append(fd)
        else:close(fd)
    monkeypatch.setattr(sm.os,'close',release)
    observed.syscall=syscall
    return observed


@pytest.mark.parametrize('fd',[0,7654])
def test_pidfd_syscall_abi_flags_and_descriptor_release(pidfd_libc,monkeypatch,fd):
    import ctypes
    from werewolf.tom_llm import service_manager as sm
    monkeypatch.delattr(sm.os,'pidfd_open',raising=False)
    monkeypatch.delattr(sm.signal,'pidfd_send_signal',raising=False)
    pidfd_libc.fd=fd
    with sm._pidfd(123) as send:
        send(0);send(sm.signal.SIGTERM)
    assert pidfd_libc.loads==[(None,True)]
    assert pidfd_libc.syscall.restype is ctypes.c_long
    assert pidfd_libc.syscall.argtypes==(ctypes.c_long,)
    calls=pidfd_libc.calls
    assert [[a.value for a in call] for call in calls]==[
        [434,123,0],[424,fd,0,None,0],[424,fd,int(sm.signal.SIGTERM),None,0]]
    assert [type(a) for a in calls[0]]==[ctypes.c_long,ctypes.c_int,ctypes.c_uint]
    assert [type(a) for a in calls[1]]==[ctypes.c_long,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_uint]
    assert pidfd_libc.closed==[fd]


@pytest.mark.parametrize('operation',[434,424])
@pytest.mark.parametrize('error_name',['ENOSYS','EPERM','ESRCH','EINVAL','EBADF','EMFILE'])
def test_pidfd_syscall_errors_propagate_errno_without_retry(pidfd_libc,operation,error_name):
    import errno
    from werewolf.tom_llm import service_manager as sm
    code=getattr(errno,error_name);pidfd_libc.errors[operation]=code
    with pytest.raises(OSError) as error:
        with sm._pidfd(123) as send:send(sm.signal.SIGTERM)
    assert error.value.errno==code and str(operation) in str(error.value)
    assert [call[0].value for call in pidfd_libc.calls]==([434] if operation==434 else [434,424])
    assert pidfd_libc.closed==([] if operation==434 else [pidfd_libc.fd])


@pytest.mark.parametrize('failure',['body','interrupt'])
def test_pidfd_descriptor_closes_on_body_error_or_interrupt(pidfd_libc,failure):
    from werewolf.tom_llm import service_manager as sm
    kind=ValueError if failure=='body' else KeyboardInterrupt
    with pytest.raises(kind):
        with sm._pidfd(123):raise kind()
    assert pidfd_libc.closed==[pidfd_libc.fd]
    assert [call[0].value for call in pidfd_libc.calls]==[434]


@pytest.mark.parametrize('platform,machine,small_type',[
    ('darwin','x86_64',None),('linux','aarch64',None),('linux','i686',None),
    ('linux','x86_64','c_void_p'),('linux','x86_64','c_long')])
def test_pidfd_rejects_unreviewed_platform_or_abi_before_libc(pidfd_libc,monkeypatch,platform,machine,small_type):
    from werewolf.tom_llm import service_manager as sm
    sm.sys.platform=platform
    monkeypatch.setattr(sm.os,'uname',lambda:NS(machine=machine))
    if small_type:
        size=sm.ctypes.sizeof;small=getattr(sm.ctypes,small_type)
        monkeypatch.setattr(sm.ctypes,'sizeof',lambda t:4 if t is small else size(t))
    with pytest.raises(ValueError,match='Linux x86-64 LP64'):
        with sm._pidfd(123):pytest.fail('unsupported ABI reached body')
    assert not pidfd_libc.loads and not pidfd_libc.calls and not pidfd_libc.closed


@pytest.mark.parametrize('pid',[0,-1,2**31,True])
def test_pidfd_rejects_invalid_pid_without_c_integer_wrap(pidfd_libc,pid):
    from werewolf.tom_llm import service_manager as sm
    with pytest.raises(ValueError,match='pid_t'):
        with sm._pidfd(pid):pytest.fail('invalid pid reached body')
    assert not pidfd_libc.calls


@pytest.mark.parametrize('operation',[434,424])
def test_kernel_pidfd_probe_failure_refuses_before_binding_or_launch(tmp_path,monkeypatch,pidfd_libc,operation):
    import errno
    from werewolf.tom_llm import service_manager as sm
    manager=Service(tmp_path/'run','qwen3','a'*64)
    monkeypatch.setattr(manager,'binding',lambda **kw:pytest.fail('failed probe reached Deployment/GPU/launch state'))
    monkeypatch.setattr(sm.subprocess,'Popen',lambda *a,**kw:pytest.fail('failed probe launched service'))
    pidfd_libc.errors[operation]=errno.ENOSYS
    with pytest.raises(OSError) as error:manager.start(execute=True)
    assert error.value.errno==errno.ENOSYS
    assert [[a.value for a in call] for call in pidfd_libc.calls]==(
        [[434,os.getpid(),0]] if operation==434 else [[434,os.getpid(),0],[424,pidfd_libc.fd,0,None,0]])
    assert pidfd_libc.closed==([] if operation==434 else [pidfd_libc.fd])
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('operation',['open','send'])
def test_stop_pidfd_failure_preserves_ownership_and_does_not_retry(managed,monkeypatch,operation):
    import errno
    from werewolf.tom_llm import service_manager as sm
    manager,child,identity,calls=managed;manager.start(execute=True)
    state=(manager.directory/'current.json').read_bytes();closed=[];attempts=[]
    @contextmanager
    def failing(pid):
        attempts.append(('open',pid))
        if operation=='open':raise OSError(errno.ESRCH,'UNIT target exited before pidfd_open')
        def send(sig):
            attempts.append(('send',sig));raise OSError(errno.EPERM,'UNIT kernel signal rejected')
        try:yield send
        finally:closed.append(7654)
    monkeypatch.setattr(sm,'_pidfd',failing)
    monkeypatch.setattr(sm.os,'kill',lambda *a:pytest.fail('ordinary PID signalling is forbidden'))
    with pytest.raises(OSError):manager.stop(execute=True)
    assert (manager.directory/'current.json').read_bytes()==state
    assert attempts==([('open',child.pid)] if operation=='open' else [('open',child.pid),('send',sm.signal.SIGTERM)])
    assert closed==([] if operation=='open' else [7654])
    assert not any(c[0]=='signal' for c in calls) and child.done is False
