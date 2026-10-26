"""CPU/native-tokenizer and injected-engine tests; never a physical GPU qualification."""

import asyncio
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import openai
import pytest

from werewolf.artifact_io import publish_artifact,verify_artifact,canonical_json_bytes
from werewolf.tom_llm.common import digest
from werewolf.tom_llm import multimodel as mm, physical, deployment as dep, qualification_inputs as qi
from werewolf.tom_llm.model_spec import ModelSpec
from werewolf.tom_llm.qualification_evidence import NUMERIC,MEMORY,validate_sampling,validate_qualification
from werewolf.tom_llm.gpu_probe import install_probe,snapshot_sampling,validate_launch_args,JsonlSink
from werewolf.tom_llm.candidate_client import SyntheticCandidateClient,collect_round
from werewolf.tom_llm.physical_admission import open_round,publish_admission,open_admission
from werewolf.tom_llm.provider_semantics import NotExecuted,OutcomeUnknown

KEYS=['qwen3','yi','mistral']


@pytest.fixture(scope='module')
def native_suites():
    variables={'qwen3':'UNS_LLM_TOKENIZER_ASSETS','yi':'UNS_LLM_YI_ASSETS','mistral':'UNS_LLM_MISTRAL_ASSETS'}
    result={}
    for key,name in variables.items():
        assets=os.environ.get(name)
        if not assets:pytest.skip('NOT_RUN: supplied official tokenizer assets required: '+name)
        result[key]=qi.build_suite(ModelSpec(key),assets)
    return result


def model_record(key,suites):
    spec=ModelSpec(key)
    return {'schema_version':dep.VERSION,'key':key,'model_spec':spec.record(),'source':mm.source_identity(),
        'profile':dep.profile(spec,gpu_uuid='GPU-unit-only',candidate_root=dep.BASE/'multimodel'/spec.key/'unit-candidate'),'model':{'evidence_kind':'SYNTHETIC_UNIT_FIXTURE'},
        'environment':{'evidence_kind':'SYNTHETIC_UNIT_FIXTURE'},'suite':suites[key],'budget':{},
        'upstream_receipts':{},'purpose':'synthetic_gpu_qualification_only','physical_gpu_admission':False}


def candidate_artifact(tmp_path,record):
    spec=ModelSpec(record['key'])
    return publish_artifact(tmp_path/'deployment',manifest_fields={'artifact_type':'mm_deployment_candidate','schema_version':dep.VERSION,
        'deployment_digest':digest(record),'model_spec_digest':spec.digest,'gpu_qualification_passed':False},
        files={'deployment.json':canonical_json_bytes(record),'vllm.yaml':dep.vllm_config_bytes(record['profile']['argv_options']),
               'chat_template.jinja':dep.template_bytes(spec)})


def sampling(spec,schema):
    eos=spec.record()['termination']['generation_eos_token_id'];eos=eos if isinstance(eos,list) else [eos]
    eos=sorted(set(eos)|{spec.record()['tokenizer']['eos_token_id']})
    return {**NUMERIC,'stop':[],'stop_token_ids':spec.record()['termination']['extra_stop_token_ids'],
        'eos_token_ids':eos,'primary_eos_token_id':spec.record()['tokenizer']['eos_token_id'],'ignore_eos':False,
        'json_schema':schema,**{k:None for k in ('logit_bias','allowed_token_ids','bad_words','logits_processors')}}


def params_object(record,case):
    p=sampling(ModelSpec(record['key']),case['task']['response_schema']);kwargs={k:p[k] for k in NUMERIC}
    kwargs.update({k:p[k] for k in ('stop','stop_token_ids','ignore_eos')})
    return NS(**kwargs,_all_stop_token_ids=set(p['eos_token_ids']),_eos_token_id=p['primary_eos_token_id'],
        structured_outputs=NS(json=p['json_schema']))


class Memory:
    measurement_identity=MEMORY
    def __enter__(self):
        self.samples=[{'monotonic_seconds':1.,'used_bytes':100,'gpu_uuid':'GPU-unit-only'},
                      {'monotonic_seconds':1.1,'used_bytes':200,'gpu_uuid':'GPU-unit-only'}]
        self.peak_bytes=200;return self
    def __exit__(self,*_):return False


def fake_candidate(tmp_path,record,run,*,failure=None,after_response_failure=False):
    artifact=candidate_artifact(tmp_path,record)
    life={'pid':100+run,'boot_id':'UNIT-BOOT','start_ticks':str(10+run)}
    instance={**life,'deployment_manifest_digest':artifact.manifest_digest,'gpu':record['profile']['gpu'],
              'argv':['SYNTHETIC_ONLY'],'driver_version':'UNIT','kernel_release':'UNIT','environment_requirements':record['profile']['environment_requirements']}
    instance['instance_digest']=digest(instance)
    events=[];count={'calls':0,'returned':False}
    class Engine:
        async def _add_request(self,core):return core.request_id
        async def generate(self,output):yield output
    engine=Engine();install_probe(engine,events.append,deployment=record,process_lifetime=life)
    def create(**wire):
        count['calls']+=1
        if failure:raise failure
        item=record['suite']['cases'][qi.SCHEDULE[count['calls']-1][1]]
        seats=item['task']['response_schema']['properties']['probabilities']['required']
        content=json.dumps({'probabilities':{p:1/6 for p in seats}})
        external=f'UNIT-{run}-{count["calls"]}'
        core=NS(prompt_embeds=None,mm_features=[],pooling_params=None,lora_request=None,cache_salt=None,
            reasoning_ended=None,reasoning_parser_kwargs=None,resumable=False,priority=0,
            prompt_token_ids=item['input_ids'],sampling_params=params_object(record,item),external_req_id=external,request_id=external)
        choice=NS(token_ids=[10,11],text=content,finish_reason='stop',stop_reason=None)
        async def execute():
            await engine._add_request(core)
            async for _ in engine.generate(NS(finished=True,request_id=external,outputs=[choice])):pass
        asyncio.run(execute())
        raw={'model':record['model_spec']['model'],'choices':[{'message':{'content':content,'refusal':None},'finish_reason':'stop'}],
             'usage':{'prompt_tokens':len(item['input_ids']),'completion_tokens':2,'total_tokens':len(item['input_ids'])+2}}
        count['returned']=True
        return NS(model=raw['model'],choices=[NS(message=NS(content=content,refusal=None),finish_reason='stop')],
            usage=NS(**raw['usage']),model_dump=lambda **_:raw)
    sdk=NS(base_url=f"http://127.0.0.1:{record['profile']['argv_options']['port']}/v1",max_retries=0,
        _client=NS(_trust_env=False,_transport=NS(_pool=NS(_retries=0))),chat=NS(completions=NS(create=create)),close=lambda:None)
    def guard():
        if after_response_failure and count['returned']:raise ValueError('UNIT service lifetime drift')
    return SyntheticCandidateClient(sdk,deployment=artifact,record=record,instance=instance,guard=guard),events,count


def test_source_v2_git_status_covers_all_extended_runtime_paths(monkeypatch):
    original=mm.development_source_identity()
    monkeypatch.setattr(mm,'development_source_identity',lambda:{**original,'source_inventory_clean':True})
    observed=[]
    def status(command,**kwargs):observed.append(command);return b'?? configs/paper/tom-llm-models-v1.json\n'
    monkeypatch.setattr(mm.subprocess,'check_output',status)
    source=mm.source_identity()
    assert source['schema_version']=='classic7_multimodel_source_v2'
    assert source['source_inventory_clean'] is False
    paths=observed[0][observed[0].index('--')+1:]
    assert set(paths)==set(source['file_digests'])
    assert {'scripts/tom_multimodel_admission.py','configs/paper/tom-llm-physical-assets-v1.json',
            'configs/paper/tom-llm-qualification-recipe-v1.json','werewolf/tom_llm/qualification_assets/qwen3-suite-v1.json'}<=set(paths)
    assert source['qualification']=='CPU_MOCK_ONLY_NOT_ADMITTED'


@pytest.mark.parametrize('key',KEYS)
def test_official_physical_inventory_is_closed_and_pinned(key):
    spec=ModelSpec(key);assets=physical.asset_spec(spec)
    assert assets['revision']==spec.record()['revision'] and assets['model']==spec.record()['model']
    assert assets['license']=='apache-2.0'
    shards=[r for n,r in assets['files'].items() if n.endswith('.safetensors')]
    assert len(shards)=={'qwen3':5,'yi':4,'mistral':3}[key]
    assert all(len(r['sha256'])==64 and r['bytes']>0 for r in shards)


@pytest.mark.parametrize('change',['bytes','revision','extra','symlink','index'])
def test_streamed_model_physical_mismatch_rejected(tmp_path,monkeypatch,change):
    files={'config.json':b'{}','model-00001.safetensors':b'UNIT_NOT_REAL_WEIGHTS',
           'model.safetensors.index.json':canonical_json_bytes({'weight_map':{'unit':'model-00001.safetensors'}})}
    for name,raw in files.items():(tmp_path/name).write_bytes(raw)
    rev=ModelSpec('qwen3').record()['revision'];(tmp_path/'HF_REVISION').write_text(rev+'\n')
    manifest={n:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'git_blob_oid':None} for n,b in files.items()}
    monkeypatch.setattr(physical,'asset_spec',lambda _: {'model':'Qwen/Qwen3-8B','revision':rev,'files':manifest})
    assert physical.verify_model(ModelSpec('qwen3'),tmp_path)['weights_loaded'] is False
    if change=='bytes':(tmp_path/'model-00001.safetensors').write_bytes(b'changed')
    elif change=='revision':(tmp_path/'HF_REVISION').write_text('0'*40+'\n')
    elif change=='extra':(tmp_path/'unbound.json').write_text('{}')
    elif change=='symlink':(tmp_path/'link').symlink_to(tmp_path/'config.json')
    else:
        raw=canonical_json_bytes({'weight_map':{'unit':'foreign.safetensors'}})
        (tmp_path/'model.safetensors.index.json').write_bytes(raw)
        manifest['model.safetensors.index.json']={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'git_blob_oid':None}
    with pytest.raises(ValueError):physical.verify_model(ModelSpec('qwen3'),tmp_path)


@pytest.mark.parametrize('key',KEYS)
def test_native_suite_preserves_shared_cases_and_qwen_76_protocol(native_suites,key):
    suite=qi.validate_suite(native_suites[key],ModelSpec(key))
    assert len(suite['schedule'])*2==76 and len(suite['cases'])==12
    assert [c['task'] for c in suite['cases']]==qi.synthetic_tasks()
    if key=='qwen3':
        for c,old in zip(suite['cases'],qi.legacy_suite()['cases'],strict=True):
            assert c['input_ids']==old['input_ids'] and c['wire_digest']==old['wire_digest']
    other=deepcopy(suite);other['cases'][0]['task']['q']=[1]
    with pytest.raises(ValueError):qi.validate_suite(other,ModelSpec(key))
    with pytest.raises(ValueError):qi.validate_suite(suite,ModelSpec('yi' if key!='yi' else 'mistral'))


@pytest.mark.parametrize('key',KEYS)
def test_native_sampling_eos_schema_and_parameters_are_strict(native_suites,key):
    spec=ModelSpec(key);schema=native_suites[key]['cases'][0]['task']['response_schema'];p=sampling(spec,schema)
    validate_sampling(p,schema,spec)
    if key=='yi':assert p['eos_token_ids']==[2,7] and p['stop_token_ids']==[7]
    for field,value in [('top_k',-1),('max_tokens',513),('temperature',True),('json_schema',{}),('ignore_eos',True),('eos_token_ids',[999])]:
        wrong=deepcopy(p);wrong[field]=value
        with pytest.raises(ValueError):validate_sampling(wrong,schema,spec)
    assert snapshot_sampling(params_object(model_record(key,native_suites),native_suites[key]['cases'][0]))==p


def test_profiles_are_disjoint_and_cache_scoped():
    profiles=[dep.profile(ModelSpec(k),gpu_uuid='GPU-unit-only',candidate_root=dep.BASE/'multimodel'/k/'unit-candidate') for k in KEYS]
    assert len({p['root'] for p in profiles})==len({p['argv_options']['port'] for p in profiles})==3
    for p in profiles:
        for k,v in p['environment_requirements'].items():
            if k.endswith('_DIR') or k in ('HF_HOME','XDG_CACHE_HOME','TORCH_HOME','VLLM_CACHE_ROOT','FLASHINFER_WORKSPACE_BASE','TMPDIR'):
                assert Path(v).is_relative_to(Path(p['root']))
        assert p['argv_options']['trust-request-chat-template'] is False
        with pytest.raises(ValueError):dep.profile(ModelSpec(p['key']),gpu_uuid='GPU-unit-only',candidate_root=p['root'],model_directory='/outside/model')


@pytest.mark.parametrize('key',KEYS)
def test_two_native_rounds_join_durable_engine_and_reply_evidence(tmp_path,native_suites,key,monkeypatch):
    record=model_record(key,native_suites);rounds=[]
    for run in (0,1):
        client,events,count=fake_candidate(tmp_path/str(run),record,run)
        root=tmp_path/str(run)/'round'
        seal=collect_round(root,client,run=run,event_reader=lambda:events,memory_monitor=Memory)
        _,r=open_round(root,seal.manifest_digest,record,deployment_manifest_digest=client.deployment.manifest_digest);rounds.append(r)
        assert count['calls']==38 and len(events)==77
        with pytest.raises(ValueError):collect_round(root,client,run=run,event_reader=lambda:events,memory_monitor=Memory)
    report=validate_qualification(record,rounds)
    assert report['dispatches']==76 and report['measurement_dispatches']==72
    assert report['conversion_status_counts']=={'valid':76}
    assert report['formal_execution_authorized'] is False
    changed=deepcopy(rounds);changed[1]['instance']=deepcopy(changed[0]['instance'])
    with pytest.raises(ValueError):validate_qualification(record,changed)
    changed=deepcopy(rounds);changed[1]['events'][2]['content']='{}'
    with pytest.raises(ValueError):validate_qualification(record,changed)
    changed=deepcopy(rounds);changed[1]['events'][1]['input_ids']=[1]
    with pytest.raises(ValueError):validate_qualification(record,changed)
    changed=deepcopy(rounds);changed[1]['calls'][0]['gpu_peak_bytes']=999
    with pytest.raises(ValueError):validate_qualification(record,changed)
    assert record['source']['qualification']=='CPU_MOCK_ONLY_NOT_ADMITTED'
    monkeypatch.setattr('sys.platform','darwin')
    with pytest.raises(ValueError,match='Mac/Mock'):publish_admission(tmp_path/'never',tmp_path,'0'*64,round_roots=[],expected_rounds=[])


@pytest.mark.parametrize('failure',['timeout','not_executed','after_reply'])
def test_uncertain_or_received_failures_never_retry_or_resume(tmp_path,native_suites,failure):
    record=model_record('qwen3',native_suites)
    error=openai.APITimeoutError(request=object()) if failure=='timeout' else NotExecuted('UNIT confirmed rejection') if failure=='not_executed' else None
    client,events,count=fake_candidate(tmp_path,record,0,failure=error,after_response_failure=failure=='after_reply')
    root=tmp_path/'round'
    with pytest.raises(Exception):collect_round(root,client,run=0,event_reader=lambda:events,memory_monitor=Memory)
    with pytest.raises(ValueError):collect_round(root,client,run=0,event_reader=lambda:events,memory_monitor=Memory)
    assert count['calls']==1 and not (root/'seal').exists()
    event=verify_artifact(root/'failures/0',expected_artifact_type='mm_probe_failure',expected_schema_version='classic7_multimodel_gpu_evidence_v1')
    assert event.manifest['outcome']=={'timeout':'outcome_unknown','not_executed':'confirmed_not_executed','after_reply':'reply_received_invalid'}[failure]
    assert event.manifest['automatic_retry'] is False
    assert (root/'responses/0').exists()==(failure=='after_reply')


@pytest.mark.parametrize('field',['model','endpoint','sdk','http','proxy'])
def test_candidate_gate_rejects_wire_service_and_hidden_retries(tmp_path,native_suites,field):
    record=model_record('qwen3',native_suites);client,events,count=fake_candidate(tmp_path,record,0)
    request=__import__('werewolf.tom_llm.model_spec',fromlist=['wire']).wire(record['suite']['cases'][0]['task'],ModelSpec('qwen3'))
    if field=='model':request['model']='unapproved'
    elif field=='endpoint':client.client.base_url='http://127.0.0.1:1/v1'
    elif field=='sdk':client.client.max_retries=1
    elif field=='http':client.client._client._transport._pool._retries=1
    else:client.client._client._trust_env=True
    with pytest.raises(NotExecuted):client.dispatch(request)
    assert count['calls']==0


@pytest.mark.parametrize('key',KEYS)
def test_deployment_reader_rejects_template_source_model_drift(tmp_path,native_suites,key,monkeypatch):
    r=model_record(key,native_suites);a=candidate_artifact(tmp_path,r)
    assert dep.open_deployment(a.path,a.manifest_digest)[1]==r
    with pytest.raises(ValueError):dep.open_deployment(a.path,'0'*64)
    original=dep.source_identity()
    monkeypatch.setattr(dep,'source_identity',lambda:{**original,'source_revision':'0'*40})
    with pytest.raises(ValueError,match='identity/domain'):dep.open_deployment(a.path,a.manifest_digest)


def test_full_queue_budget_no_truncation_or_model_filtering(native_suites):
    task=qi.synthetic_tasks()[0]
    tasks=[{**deepcopy(task),'game_id':f'UNIT-{i:03}','fold':i//30} for i in range(150)]
    class Tokens:
        def encode(self,*_,**__):return [1]*100
        def apply_chat_template(self,*_,**__):return [1]*100
    results=[qi.budget_record(tasks,ModelSpec(k),Tokens()) for k in KEYS]
    assert len({r['row_identity_digest'] for r in results})==1
    assert all(r['request_count']==150 and r['input_tokens_total']==15000 for r in results)
    records={k:{**r,'schema_version':qi.VERSION+'_budget','model_spec_digest':ModelSpec(k).digest,
        'cohort_digest':'a'*64,'selection':{'game_ids':[f'UNIT-{i:03}' for i in range(150)]},'reference':{'seal':'b'*64}}
        for k,r in zip(KEYS,results,strict=True)}
    assert qi.budget_matrix(records)['common']['request_count']==150
    bad=deepcopy(records);bad['yi']['cohort_digest']='c'*64
    with pytest.raises(ValueError):qi.budget_matrix(bad)
    class Overflow(Tokens):
        def encode(self,*_,**__):return [1]*7681
        def apply_chat_template(self,*_,**__):return [1]*7681
    with pytest.raises(ValueError,match='no row filtering'):qi.budget_record(tasks,ModelSpec('yi'),Overflow())


def test_probe_launch_effective_options_and_mac_gpu_guard(tmp_path,monkeypatch):
    p=dep.profile(ModelSpec('qwen3'),gpu_uuid='GPU-unit-only',candidate_root=dep.BASE/'multimodel/qwen3/unit-candidate')
    args=NS(**{k.replace('-','_'):v for k,v in p['argv_options'].items()},quantization=None,reasoning_parser=None)
    validate_launch_args(args,p);args.max_model_len=4096
    with pytest.raises(ValueError):validate_launch_args(args,p)
    from werewolf.tom_llm.gpu_probe import main
    monkeypatch.setattr('sys.argv',['probe','--deployment','missing','--expected-deployment','0'*64])
    with pytest.raises(ValueError,match='authorized Linux'):main()
    log=tmp_path/'audit.jsonl';sink=JsonlSink(log);sink({'unit':1});sink.close()
    assert json.loads(log.read_bytes())=={'unit':1}
    with pytest.raises(FileExistsError):JsonlSink(log)


def test_cpu_env_audit_wrong_interpreter_does_not_install_or_import_gpu(tmp_path):
    with pytest.raises(ValueError,match='own isolated interpreter'):physical.capture_environment(tmp_path,prior_audit_digest='a'*64)


def test_linux_service_process_listener_log_and_environment_identity(tmp_path,native_suites,monkeypatch):
    record=model_record('qwen3',native_suites);record['source']['source_inventory_clean']=True
    prefix=tmp_path/'env';(prefix/'bin').mkdir(parents=True);(prefix/'bin/python').write_bytes(b'UNIT_INTERPRETER')
    record['profile']['environment_prefix']=str(prefix)
    record['profile']['engine_log_root']=str(tmp_path/'engine')
    installed=tmp_path/'installed-template.jinja';installed.write_bytes(dep.template_bytes(ModelSpec('qwen3')))
    record['profile']['argv_options']['chat-template']=str(installed)
    a=candidate_artifact(tmp_path,record);proc=tmp_path/'proc';pid=100
    process=proc/str(pid);(process/'fd').mkdir(parents=True);(proc/'sys/kernel/random').mkdir(parents=True);(proc/'net').mkdir()
    (proc/'sys/kernel/random/boot_id').write_text('UNIT')
    (process/'stat').write_text('100 (unit process) '+' '.join(['S']+['0']*18+['10']))
    argv=[str(prefix/'bin/python'),'-m','werewolf.tom_llm.gpu_probe','--deployment',str(a.path),
          '--expected-deployment',a.manifest_digest,'--execute-synthetic-gpu-qualification']
    (process/'cmdline').write_bytes(b'\0'.join(s.encode() for s in argv)+b'\0')
    (process/'exe').symlink_to(prefix/'bin/python')
    env=record['profile']['environment_requirements']
    (process/'environ').write_bytes(b'\0'.join((k+'='+v).encode() for k,v in env.items()))
    (process/'fd/1').symlink_to('socket:[123]')
    life={'pid':100,'boot_id':'UNIT','start_ticks':'10'}
    log=Path(record['profile']['engine_log_root'])/digest(life)/'engine-events.jsonl';log.parent.mkdir(parents=True);log.write_text('')
    (process/'fd/2').symlink_to(log)
    port=record['profile']['argv_options']['port']
    (proc/'net/tcp').write_text('header\n'+' '.join(['0:',f'0100007F:{port:04X}','0','0A','0','0','0','0','0','123'])+'\n')
    monkeypatch.setattr(dep,'Path',lambda p:proc if str(p)=='/proc' else Path(p))
    monkeypatch.setattr('sys.platform','linux');monkeypatch.setattr(dep,'gpu_snapshot',lambda:record['profile']['gpu']);monkeypatch.setattr(dep,'driver_version',lambda:'UNIT')
    instance=dep.capture_instance(a,record,100)
    assert instance['start_ticks']=='10' and instance['instance_digest']==digest({k:v for k,v in instance.items() if k!='instance_digest'})
    (process/'environ').write_bytes((process/'environ').read_bytes()+b'\0HTTPS_PROXY=http://invalid')
    with pytest.raises(ValueError,match='proxies'):dep.capture_instance(a,record,100)
    (process/'environ').write_bytes(b'\0'.join((k+'='+v).encode() for k,v in env.items()))
    (process/'fd/2').unlink()
    with pytest.raises(ValueError,match='audit log'):dep.capture_instance(a,record,100)


def test_environment_snapshot_retains_record_deviations_and_pins_probe_code(tmp_path,monkeypatch):
    import sys
    from pathlib import PurePosixPath
    prefix=tmp_path/'env';binary=prefix/'bin/python';binary.parent.mkdir(parents=True);binary.write_bytes(b'UNIT')
    site=prefix/'lib/python3.12/site-packages';site.mkdir(parents=True)
    class Distribution:
        def __init__(self,name):
            self.version='0.27.0' if name=='vllm' else 'UNIT'
            self.files=[PurePosixPath(name+'.dist-info/'+n) for n in ('METADATA','RECORD')]
            for name in self.files:
                file=site/str(name);file.parent.mkdir(parents=True,exist_ok=True);file.write_text('UNIT KNOWN MISMATCH PRESERVED')
        def locate_file(self,path):return site/str(path)
    distributions={n:Distribution(n) for n in physical.CORE_PACKAGES}
    original=dep.recipe();critical={}
    for name in original['vllm_source_sha256']:
        file=site/'vllm'/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(b'UNIT CODE')
        critical[name]=hashlib.sha256(file.read_bytes()).hexdigest()
    for name in ['openai/_base_client.py','torch/__init__.py','flashinfer/__init__.py','tvm_ffi/__init__.py']:
        file=site/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(b'UNIT CODE')
    monkeypatch.setattr(sys,'prefix',str(prefix));monkeypatch.setattr(sys,'executable',str(binary))
    monkeypatch.setattr(physical.importlib.metadata,'distribution',lambda n:distributions[n])
    monkeypatch.setattr(dep,'recipe',lambda:{**original,'vllm_source_sha256':critical})
    monkeypatch.setattr(physical.subprocess,'run',lambda *a,**k:NS(returncode=0,stdout='UNIT pip check PASS'))
    result=physical.capture_environment(prefix,prior_audit_digest='a'*64)
    assert result['gpu_runtime_imported'] is False and result['record_integrity']=='PRIOR_STRICT_FAILURES_RETAINED_NOT_REPAIRED'
    assert (site/'vllm.dist-info/RECORD').read_text()=='UNIT KNOWN MISMATCH PRESERVED'
    (site/'vllm/v1/engine/async_llm.py').write_bytes(b'drift')
    with pytest.raises(ValueError,match='seam code drift'):physical.capture_environment(prefix,prior_audit_digest='a'*64)


def test_old_w5_or_mock_admission_domains_are_never_upgraded(tmp_path):
    old=publish_artifact(tmp_path/'old',manifest_fields={'artifact_type':'llm_physical_service_admission',
        'schema_version':'classic7_llm_physical_service_admission_v1'},files={})
    with pytest.raises(ValueError):open_admission(old.path,expected_manifest=old.manifest_digest,
        expected_deployment='a'*64,model_spec=ModelSpec('qwen3'))


def test_budget_artifact_full_150_rows_and_model_binding(tmp_path,monkeypatch):
    task=qi.synthetic_tasks()[0]
    tasks=[{**deepcopy(task),'game_id':f'UNIT-{i:03}','fold':i//30} for i in range(150)]
    manifest={'selection':{'game_ids':[t['game_id'] for t in tasks]},'reference':{'seal_digest':'b'*64}}
    cohort=NS(manifest=manifest,manifest_digest='a'*64,path=tmp_path/'cohort')
    monkeypatch.setattr(mm,'read_cohort',lambda *_:(cohort,tasks))
    class Tokens:
        def encode(self,*_,**__):return [1]*100
        def apply_chat_template(self,*_,**__):return [1]*100
    monkeypatch.setattr(qi,'load_cpu_tokenizer',lambda *_:Tokens())
    artifact=qi.publish_budget(tmp_path/'budget',cohort.path,'a'*64,ModelSpec('qwen3'),tmp_path)
    report=qi.open_budget(artifact.path,ModelSpec('qwen3'),artifact.manifest_digest)
    assert report['request_count']==150 and report['input_tokens_total']==15000
    with pytest.raises(ValueError):qi.open_budget(artifact.path,ModelSpec('yi'),artifact.manifest_digest)
    tasks.pop();manifest['selection']['game_ids'].pop()
    with pytest.raises(ValueError,match='150-game'):qi.publish_budget(tmp_path/'never',cohort.path,'a'*64,ModelSpec('qwen3'),tmp_path)


def test_cohort_export_builds_public_tasks_before_opening_reference(tmp_path,monkeypatch):
    import werewolf.development_publication as publication
    import werewolf.tom_llm.selection_binding as selection_binding
    import werewolf.tom_llm.reference as reference
    import werewolf.tom_llm.task as task_module
    import werewolf.tom_llm.multimodel_evaluation as evaluation
    games=[f'UNIT-{i:03}' for i in range(150)]
    folds=[{'fold':i,'held_out_game_ids':games[i*30:(i+1)*30]} for i in range(5)]
    monkeypatch.setattr(qi,'verify_artifact',lambda *_a,**_k:NS(manifest={'folds':folds}))
    calls=[]
    monkeypatch.setattr(selection_binding,'open_selection_binding',lambda *_a,**_k:calls.append('selection'))
    (tmp_path/'selection.json').write_bytes(canonical_json_bytes({'game_ids':games}))
    public=NS(public_view=NS(games=[NS(game_id=g,authoritative_pre_prefixes=[g]) for g in games+['EXCLUDED']]))
    monkeypatch.setattr(publication,'open_publication',lambda _:public)
    def build(prefixes,membership):
        calls.append('public');assert prefixes==games and len(membership)==150
        return ['UNIT_PUBLIC_TASKS']
    monkeypatch.setattr(task_module,'public_tasks',build)
    def open_ref(*_a,**_k):
        assert calls==['selection','public'];calls.append('reference');return 'UNIT_REF'
    monkeypatch.setattr(reference,'open_reference',open_ref)
    def publish(*args,**kwargs):
        assert args[1:]==(['UNIT_PUBLIC_TASKS'],'UNIT_REF');return 'UNIT_COHORT'
    monkeypatch.setattr(evaluation,'publish_mock_cohort',publish)
    assert qi.export_cpu_cohort(tmp_path/'out',publication_root=tmp_path,evaluation_root=tmp_path,
        expected_reference='a'*64,selection_root=tmp_path,expected_selection={},selection_provenance={},prepared_root=tmp_path)=='UNIT_COHORT'
    assert calls==['selection','public','reference']


def test_cache_and_engine_log_symlinks_cannot_escape_model_directory(tmp_path):
    profile=dep.profile(ModelSpec('qwen3'),gpu_uuid='GPU-unit-only',candidate_root=dep.BASE/'multimodel/qwen3/unit-candidate')
    profile['root']=str(tmp_path/'model')
    for name in dep.CACHE_VARIABLES:profile['environment_requirements'][name]=str(tmp_path/'model/cache'/name)
    dep.validate_cache_paths(profile)
    with pytest.raises(ValueError,match='missing'):dep.validate_cache_paths(profile,require_existing=True)
    for name in dep.CACHE_VARIABLES:Path(profile['environment_requirements'][name]).mkdir(parents=True,exist_ok=True)
    dep.validate_cache_paths(profile,require_existing=True)
    for name in dep.CACHE_VARIABLES:Path(profile['environment_requirements'][name]).rmdir()
    (tmp_path/'model/cache').rmdir();(tmp_path/'model').rmdir()
    outside=tmp_path/'outside';outside.mkdir();(tmp_path/'model').mkdir()
    (tmp_path/'model/cache').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match='cache path'):dep.validate_cache_paths(profile)
    with pytest.raises(ValueError,match='symlink'):JsonlSink(tmp_path/'model/cache/new/events.jsonl')
    assert not (outside/'new').exists()


@pytest.mark.parametrize('key',['CUDA_LAUNCH_BLOCKING','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','FLASHINFER_UNDECLARED','LD_PRELOAD'])
def test_process_environment_rejects_undeclared_runtime_overrides(key):
    p=dep.profile(ModelSpec('yi'),gpu_uuid='GPU-unit-only',candidate_root=dep.BASE/'multimodel/yi/unit-candidate')
    values=dict(p['environment_requirements']);dep.validate_process_environment(p,values)
    values[key]='UNIT_OVERRIDE'
    with pytest.raises(ValueError):dep.validate_process_environment(p,values)


@pytest.fixture
def vllm027_parser(monkeypatch):
    """Exact official CPU parser module; no vLLM engine/package import."""
    import importlib.util,logging,sys,types
    location=os.environ.get('UNS_VLLM027_ARGPARSE_SOURCE')
    if not location:pytest.skip('NOT_RUN: official vLLM0.27 argparse source not supplied')
    raw=Path(location).read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='e00efebd4e4f459391769ab8a46af9401e830e3be5226abcacfdfd09a77c6a4d'
    logger=types.ModuleType('vllm.logger');logger.init_logger=lambda _:logging.getLogger('unit-vllm-parser')
    package=types.ModuleType('vllm');package.__path__=[]
    monkeypatch.setitem(sys.modules,'vllm',package);monkeypatch.setitem(sys.modules,'vllm.logger',logger)
    spec=importlib.util.spec_from_file_location('unit_vllm027_argparse',location)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.FlexibleArgumentParser


@pytest.mark.parametrize('key',KEYS)
@pytest.mark.parametrize('stage',['qualification','formal'])
def test_native_vllm_yaml_config_keeps_effective_parameters(tmp_path,vllm027_parser,key,stage):
    import argparse
    from werewolf.tom_llm.live_admission import service_profile
    spec=ModelSpec(key)
    p=dep.profile(spec,gpu_uuid='GPU-unit-only',candidate_root=dep.BASE/'multimodel'/key/'candidate-yaml-unit')
    if stage=='formal':
        qualified=p;p=service_profile(dep.BASE/'experiments/unit-run',spec,{'profile':qualified})
        assert p['environment_prefix']==qualified['environment_prefix']
        for name in ('PATH','CUDA_HOME'):assert p['environment_requirements'][name]==qualified['environment_requirements'][name]
        for name in dep.CACHE_VARIABLES:assert Path(p['environment_requirements'][name]).is_relative_to(p['root'])
    parser=vllm027_parser();parser.add_argument('--config')
    for name,value in p['argv_options'].items():
        if type(value) is bool:parser.add_argument('--'+name,action=argparse.BooleanOptionalAction,default=None)
        elif type(value) is dict:parser.add_argument('--'+name,type=json.loads)
        elif type(value) is list:parser.add_argument('--'+name,nargs='+')
        else:parser.add_argument('--'+name,type=type(value))
    parser.set_defaults(quantization=None,reasoning_parser=None)
    config=tmp_path/'vllm.yaml';config.write_bytes(dep.vllm_config_bytes(p['argv_options']))
    parsed=parser.parse_args(['--config',str(config)]);validate_launch_args(parsed,p)
    assert parsed.served_model_name==[spec.record()['model']]
    assert parsed.enable_prefix_caching is False and parsed.trust_request_chat_template is False
    assert parsed.structured_outputs_config=={'backend':'xgrammar'}
    bad=tmp_path/'vllm.json';bad.write_bytes(config.read_bytes())
    with pytest.raises(ValueError,match='yaml/yml type'):parser.parse_args(['--config',str(bad)])
    plain=tmp_path/'false-omitted.yaml';plain.write_bytes(canonical_json_bytes(p['argv_options']))
    assert parser.parse_args(['--config',str(plain)]).enable_prefix_caching is None
    with pytest.raises(ValueError):validate_launch_args(parser.parse_args(['--config',str(plain)]),p)
    with pytest.raises(ValueError):validate_launch_args(parser.parse_args(['--config',str(config),'--enable-prefix-caching']),p)


def test_deployment_uses_new_output_root_and_keeps_old_assets(tmp_path,native_suites,monkeypatch):
    base=tmp_path/'baseline';monkeypatch.setattr(dep,'BASE',base)
    spec=ModelSpec('qwen3');old=base/'multimodel/qwen3/candidate-v1'
    sealed=old/'deployment/sealed-original';sealed.parent.mkdir(parents=True);sealed.write_bytes(b'UNCHANGED_OLD_CANDIDATE')
    model=old/'model';env=old/'env';destination=base/'multimodel/qwen3/candidate-yaml-v1/deployment'
    monkeypatch.setattr(dep,'read_receipt',lambda path,kind,expected:{'directory':str(model)} if kind=='model' else {'prefix':str(env)})
    monkeypatch.setattr(dep,'open_suite',lambda *a:native_suites['qwen3'])
    monkeypatch.setattr(dep,'open_budget',lambda *a:{'UNIT_ONLY':'budget'})
    checked=[];monkeypatch.setattr(dep,'verify_cpu_binding',lambda record,spec:checked.append(record))
    args=dict(gpu_uuid='GPU-unit-only',model_root=model,model_receipt=tmp_path,expected_model='a'*64,
        environment_root=env,environment_receipt=tmp_path,expected_environment='b'*64,
        suite_root=tmp_path,expected_suite='c'*64,budget_root=tmp_path,expected_budget='d'*64)
    artifact=dep.prepare_deployment(destination,spec,**args)
    _,record=dep.open_deployment(artifact.path,artifact.manifest_digest)
    p=record['profile'];assert p['root']==str(destination.parent) and len(checked)==1
    assert p['argv_options']['model']==str(model) and p['environment_prefix']==str(env)
    assert Path(p['argv_options']['chat-template'])==artifact.path/'chat_template.jinja'
    assert set(artifact.manifest['file_table'])=={'deployment.json','vllm.yaml','chat_template.jinja'}
    assert sealed.read_bytes()==b'UNCHANGED_OLD_CANDIDATE'
    assert not (old/'cache').exists()
    for key in dep.CACHE_VARIABLES:assert Path(p['environment_requirements'][key]).is_relative_to(destination.parent)
    with pytest.raises(ValueError,match='new ROOT/deployment'):dep.prepare_deployment(destination,spec,**args)
    with pytest.raises(ValueError,match='new ROOT/deployment'):dep.prepare_deployment(old/'deployment',spec,**args)
    assert sealed.read_bytes()==b'UNCHANGED_OLD_CANDIDATE'


def test_qualification_start_is_foreground_exec_with_bound_environment(tmp_path,monkeypatch):
    from werewolf.tom_llm import gpu_probe
    base=tmp_path/'base';monkeypatch.setattr(dep,'BASE',base)
    p=dep.profile(ModelSpec('qwen3'),gpu_uuid='GPU-unit-only',candidate_root=base/'multimodel/qwen3/new-candidate')
    artifact=NS(path=Path(p['root'])/'deployment');seen={}
    monkeypatch.setattr(gpu_probe,'sys',NS(platform='linux'))
    monkeypatch.setattr(gpu_probe,'open_deployment',lambda *a,**kw:(artifact,{'profile':p,'source':{'source_inventory_clean':True}}))
    monkeypatch.setattr(gpu_probe.os,'chdir',lambda path:seen.update(cwd=str(path)))
    class Executed(BaseException):pass
    def execute(binary,argv,env):seen.update(binary=binary,argv=argv,env=env);raise Executed()
    monkeypatch.setattr(gpu_probe.os,'execve',execute)
    with pytest.raises(ValueError,match='authorized Linux'):gpu_probe.start_deployment(artifact.path,'a'*64)
    assert not Path(p['root']).exists()
    monkeypatch.setenv('HTTP_PROXY','UNIT_PROXY_MUST_NOT_LEAK');monkeypatch.setenv('VLLM_UNIT_OVERRIDE','MUST_NOT_LEAK')
    with pytest.raises(Executed):gpu_probe.start_deployment(artifact.path,'a'*64,execute=True)
    assert seen['binary']==p['environment_prefix']+'/bin/python'
    assert seen['argv']==[seen['binary'],'-m','werewolf.tom_llm.gpu_probe','--deployment',str(artifact.path),
        '--expected-deployment','a'*64,'--execute-synthetic-gpu-qualification']
    assert seen['cwd']==str(dep.ROOT)
    dep.validate_process_environment(p,seen['env']);dep.validate_cache_paths(p,require_existing=True)
    assert 'HTTP_PROXY' not in seen['env'] and 'VLLM_UNIT_OVERRIDE' not in seen['env']
    from scripts import tom_multimodel_admission as cli
    monkeypatch.setattr(cli.sys,'argv',['tool','start','--deployment',str(artifact.path),
        '--expected-deployment','a'*64,'--execute-synthetic-gpu-qualification'])
    with pytest.raises(Executed):cli.main()
