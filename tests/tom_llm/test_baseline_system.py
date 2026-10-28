"""Real loopback HTTP engineering tests. No model, GPU, Admission or real study."""

import asyncio
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace as NS
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import xml.etree.ElementTree as ET

import pytest

from tests.tom_llm.test_multimodel import cohort
from werewolf.artifact_io import verify_artifact,canonical_json_bytes
from werewolf.tom_llm import experiment as ex
from werewolf.tom_llm import live_runner as runner
from werewolf.tom_llm.baseline_cli import main
from werewolf.tom_llm.common import digest,strict_json
from werewolf.tom_llm.evaluation import compare
from werewolf.tom_llm.model_spec import ModelSpec,wire,load_cpu_tokenizer,render
from werewolf.tom_llm.multimodel_evaluation import paired_extras
from werewolf.tom_llm.inference import InfrastructureFailure
from werewolf.tom_llm.provider_semantics import NotExecuted


@pytest.fixture
def study(tmp_path,cohort,sealed_reference):
    ref,pub,evaluation,_=sealed_reference
    root=tmp_path/'output/test-run'
    artifact=ex.prepare_run(root,run_id='test-run',models=['qwen3','yi','mistral'],games=5,mode='fake',
        cohort_root=cohort[0].path,expected_cohort=cohort[0].manifest_digest,
        publication=pub.path,evaluation=evaluation,expected_reference=ref.seal_digest)
    return root,cohort[1],ref


@pytest.fixture
def fake_http():
    state={'requests':[],'style':'valid','delay':0.,'reject_first':False}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*a):pass
        def do_POST(self):
            data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            state['requests'].append(data)
            if state['reject_first'] and len(state['requests'])==1:
                self.send_response(429);self.end_headers();return
            if state['delay']:time.sleep(state['delay'])
            seats=data['response_format']['json_schema']['schema']['properties']['probabilities']['required']
            content=json.dumps({'probabilities':{s:(.2 if state['style']=='invalid_sum' else 1/6) for s in seats}})
            body={'id':'fake-'+str(len(state['requests'])),'model':data['model'],
                'choices':[{'message':{'role':'assistant','content':content,'refusal':None},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':100,'completion_tokens':10,'total_tokens':110}}
            if state['style']=='bad_usage':body['usage']['total_tokens']=0
            raw=json.dumps(body).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers()
            try:self.wfile.write(raw)
            except (BrokenPipeError,ConnectionResetError):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:yield 'http://127.0.0.1:'+str(server.server_port),state
    finally:server.shutdown();server.server_close();thread.join(timeout=2)


def test_source_v3_covers_both_clis_and_frozen_policy():
    s=ex.source_v3()
    assert s['schema_version']=='classic7_multimodel_source_v3'
    assert {'scripts/run_tom_multimodel_baseline.py','scripts/analyze_tom_multimodel_baseline.py',
            'configs/paper/tom-llm-experiment-v1.json'}<=set(s['file_digests'])
    assert s['qualification']=='SOFTWARE_IDENTITY_ONLY_NOT_ADMISSION'


@pytest.mark.parametrize('count',[1,10,149,150])
def test_technical_whole_game_selection_is_deterministic(count):
    selection={'scope':'subset150','game_ids':[f'g{i:03}' for i in range(150)],
               'folds':[{'game_ids':[str(j) for j in range(30)]} for _ in range(5)]}
    a=ex.choose_games(selection,count,'technical')
    assert len(a)==count and a==ex.choose_games(selection,count,'technical')
    assert set(a)<=set(selection['game_ids'])
    if count==150:assert a==ex.choose_games(selection,150,'formal')


@pytest.mark.parametrize('count',[0,10,149,151,True])
def test_formal_requires_exact_150(count):
    with pytest.raises(ValueError):ex.choose_games({'scope':'subset150','game_ids':list(range(150))},count,'formal')


@pytest.mark.parametrize('key',['qwen3','yi','mistral'])
def test_real_http_wire_and_durable_model_seal(study,fake_http,key):
    root,tasks,ref=study;endpoint,state=fake_http
    seal=runner.run_model(root,key,runner.HttpTransport(endpoint))
    a,rows=runner.read_predictions(root,key,expected=seal.manifest_digest)
    assert len(rows)==len(tasks)==len(state['requests'])
    assert all(r['prediction']['status']=='valid' for r in rows)
    w=wire(tasks[0],ModelSpec(key));body={k:v for k,v in w.items() if k!='extra_body'};body.update(w['extra_body'])
    assert state['requests'][0]==body
    assert 'chat_template' not in body
    assert strict_json((a.path/'record.json').read_bytes())['formal_result'] is False
    result=compare(ref,rows,ref.game_ids)
    assert result['failures']['legal_uniform_rows']>0


def test_complete_cli_three_model_analysis_reloads_seals(study,fake_http,tmp_path,monkeypatch):
    root,tasks,ref=study;endpoint,state=fake_http
    args=['--run-id','test-run','--output-root',str(root.parent)]
    for k in ('qwen3','yi','mistral'):args+=['--fake-endpoint',k+'='+endpoint]
    response=main(args)
    assert len(state['requests'])==3*len(tasks)
    original=ex.source_v3()
    monkeypatch.setattr(ex,'source_v3',lambda:{**original,'source_revision':'f'*40})
    with pytest.raises(ValueError,match='Source v3'):ex.open_run(root)
    report=main(['--run-id','test-run','--output-root',str(root.parent)],analysis=True)
    artifact=verify_artifact(report['path'],expected_artifact_type='mm_baseline_analysis',expected_schema_version=ex.VERSION)
    data=strict_json((artifact.path/'analysis.json').read_bytes())
    assert data['source']==original and data['analysis_source']['source_revision']=='f'*40
    assert data['formal_result'] is False and len(data['llm_pairwise'])==3
    assert set(data['models'])=={'qwen3','yi','mistral'}
    for key in data['models']:
        _,rows=runner.read_predictions(root,key)
        old=compare(ref,rows,ref.game_ids);old.update(paired_extras(ref,rows,ref.game_ids,old))
        assert all(data['models'][key][n]==old[n] for n in ('Delta_KL','Delta_TV','Delta_Top1_NU','cells'))
        assert data['models'][key]['performance']['http_attempts']==len(tasks)
    for pair in data['llm_pairwise']:
        assert pair['Delta_TV']['point']==0 and pair['Delta_KL']['point']==0
        assert pair['Delta_Top1_NU'] is None or pair['Delta_Top1_NU']['point']==0
    assert len(list(csv.DictReader((artifact.path/'metrics.csv').open())))==15
    assert 'ENGINEERING ONLY' in (artifact.path/'report.md').read_text()
    for name in ('metrics.svg','efficiency.svg'):ET.parse(artifact.path/name)
    with pytest.raises(ValueError):main(['--run-id','test-run','--output-root',str(root.parent)],analysis=True)


def test_semantic_failure_is_one_reply_fallback_not_retry(study,fake_http):
    root,tasks,_=study;endpoint,state=fake_http;state['style']='invalid_sum'
    runner.run_model(root,'qwen3',runner.HttpTransport(endpoint))
    _,rows=runner.read_predictions(root,'qwen3')
    assert len(state['requests'])==len(tasks)
    assert all(r['prediction']['failure_type']=='invalid_sum' and len(r['attempt_digests'])==1 for r in rows)


def test_confirmed_rejection_has_exact_two_dispatch_budget(study,fake_http):
    root,tasks,_=study;endpoint,state=fake_http;state['reject_first']=True
    runner.run_model(root,'qwen3',runner.HttpTransport(endpoint))
    _,rows=runner.read_predictions(root,'qwen3')
    assert len(state['requests'])==len(tasks)+1
    assert len(rows[0]['attempt_digests'])==2
    assert len(rows[0]['dispatch_elapsed_seconds'])==2
    from werewolf.tom_llm.baseline_analysis import performance
    summary=compare(study[2],rows,study[2].game_ids)['failures']
    measured=performance(root,'qwen3',tasks,rows)
    assert summary['dispatch_attempt_count']==summary['reported_dispatch_elapsed_seconds']['attempts_with_timing']==len(tasks)+1
    assert summary['reported_dispatch_elapsed_seconds']['sum']==pytest.approx(measured['active_client_wall_seconds'])


def test_timeout_unknown_never_resends_on_resume(study,fake_http):
    root,tasks,_=study;endpoint,state=fake_http;state['delay']=.15
    with pytest.raises(InfrastructureFailure):runner.run_model(root,'qwen3',runner.HttpTransport(endpoint,timeout=.02))
    assert len(state['requests'])==1
    with pytest.raises(ValueError,match='quarantine'):runner.run_model(root,'qwen3',runner.HttpTransport(endpoint))
    assert len(state['requests'])==1
    assert runner.status(root)['models']['qwen3']['blocked_requests']==1
    assert not (root/'models/qwen3/seal').exists()


def test_reply_durable_but_guard_rejected_is_quarantined(study,fake_http):
    root,tasks,_=study;endpoint,state=fake_http
    class Drift(runner.HttpTransport):
        calls=0
        def guard(self):
            self.calls+=1
            if self.calls==3:raise ValueError('test identity drift after reply')
            return super().guard()
    with pytest.raises(InfrastructureFailure):runner.run_model(root,'qwen3',Drift(endpoint))
    p=root/'models/qwen3/requests'/digest(tasks[0])
    assert (p/'response-1').exists() and not (p/'result').exists()
    with pytest.raises(ValueError):runner.run_model(root,'qwen3',runner.HttpTransport(endpoint))
    assert len(state['requests'])==1


def test_interruption_after_accepted_response_recovers_without_duplicate(study,fake_http):
    root,tasks,_=study;endpoint,state=fake_http
    def interrupt(_):raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):runner.run_model(root,'qwen3',runner.HttpTransport(endpoint),after_accepted=interrupt)
    assert len(state['requests'])==1
    assert tasks[0] not in runner.pending_tasks(root,'qwen3',tasks)
    runner.run_model(root,'qwen3',runner.HttpTransport(endpoint))
    assert len(state['requests'])==len(tasks)
    runner.read_predictions(root,'qwen3')


def test_two_not_executed_failures_never_fabricate_prediction(study):
    root,tasks,_=study
    class Reject:
        evidence_kind='fake_http';calls=0
        def guard(self):return {'kind':'fake_http','endpoint':'unit'}
        def send(self,_):self.calls+=1;raise NotExecuted('explicit local rejection')
    transport=Reject()
    with pytest.raises(InfrastructureFailure):runner.run_model(root,'qwen3',transport)
    assert transport.calls==2
    with pytest.raises(ValueError,match='exhausted'):runner.run_model(root,'qwen3',transport)
    assert transport.calls==2 and not (root/'models/qwen3/seal').exists()


def test_bad_usage_is_infrastructure_failure_not_uniform_prediction(study,fake_http):
    root,tasks,_=study;endpoint,state=fake_http;state['style']='bad_usage'
    with pytest.raises(InfrastructureFailure):runner.run_model(root,'qwen3',runner.HttpTransport(endpoint))
    assert not (root/'models/qwen3/requests'/digest(tasks[0])/'result').exists()


def test_cross_model_event_and_foreign_task_rejected(study,fake_http):
    root,tasks,_=study;endpoint,state=fake_http
    runner.run_model(root,'qwen3',runner.HttpTransport(endpoint))
    path=root/'models/qwen3/requests'/digest(tasks[0])/'result'
    wrong=root/'models/yi/requests'/digest(tasks[0])/'result';wrong.parent.mkdir(parents=True)
    shutil.copytree(path,wrong)
    with pytest.raises(ValueError):runner.pending_tasks(root,'yi',tasks)
    (root/'models/qwen3/requests/foreign').mkdir()
    with pytest.raises(ValueError,match='extra'):runner.read_predictions(root,'qwen3')


def test_incomplete_experiment_and_wrong_seal_anchor_cannot_analyze(study,fake_http):
    root,tasks,_=study;endpoint,_=fake_http
    runner.run_model(root,'qwen3',runner.HttpTransport(endpoint))
    with pytest.raises((ValueError,OSError)):runner.seal_run(root)
    with pytest.raises(ValueError):runner.read_predictions(root,'qwen3',expected='0'*64)


def test_physical_permit_not_created_from_cpu_or_mock(study,tmp_path,monkeypatch):
    from werewolf.tom_llm.live_admission import publish_permit
    root,_,_=study
    with pytest.raises(ValueError):publish_permit(root,'qwen3',qualification=tmp_path,expected_qualification='0'*64,
        tokenizer_assets=tmp_path,budget_root=tmp_path)
    assert not (root/'models/qwen3/permit').exists()


def test_wrong_mode_and_game_count_refused_before_http(study,fake_http):
    root,_,_=study;endpoint,state=fake_http
    for opts in (['--mode','formal'],['--games','150'],['--execute-live']):
        with pytest.raises(ValueError):main(['--run-id','test-run','--output-root',str(root.parent),*opts])
    assert not state['requests']


@pytest.mark.parametrize('script',['run_tom_multimodel_baseline.py','analyze_tom_multimodel_baseline.py'])
def test_cli_help_is_cpu_only(script):
    from werewolf.tom_llm.model_spec import ROOT
    result=subprocess.run([os.sys.executable,'-B',str(ROOT/'scripts'/script),'--help'],capture_output=True,text=True)
    assert result.returncode==0 and '--run-id' in result.stdout


@pytest.mark.parametrize('key',['qwen3','yi','mistral'])
def test_native_live_probe_and_proof_match_effective_engine_inputs(key):
    from tests.tom_llm.test_physical_tools import model_record,params_object
    from werewolf.tom_llm.qualification_inputs import build_suite
    from werewolf.tom_llm.live_service import install_live_probe
    assets=os.environ[{'qwen3':'UNS_LLM_TOKENIZER_ASSETS','yi':'UNS_LLM_YI_ASSETS','mistral':'UNS_LLM_MISTRAL_ASSETS'}[key]]
    spec=ModelSpec(key);suite=build_suite(spec,assets);task=suite['cases'][0]['task'];tokenizer=load_cpu_tokenizer(spec,assets)
    record=model_record(key,{key:suite});events=[];called=[]
    class Engine:
        async def _add_request(self,core):called.append(core);return None
        async def generate(self,output):yield output
    engine=Engine();permit='a'*64
    life={'pid':123,'boot_id':'UNIT_BOOT','start_ticks':'1'}
    install_live_probe(engine,events.append,tasks=[task],spec=spec,tokenizer=tokenizer,permit_digest=permit,process_lifetime=life)
    core=NS(prompt_embeds=None,mm_features=[],pooling_params=None,lora_request=None,cache_salt=None,
        reasoning_ended=None,reasoning_parser_kwargs=None,resumable=False,priority=0,
        prompt_token_ids=suite['cases'][0]['input_ids'],sampling_params=params_object(record,suite['cases'][0]),external_req_id='real-format-id')
    seats=task['response_schema']['properties']['probabilities']['required']
    content=json.dumps({'probabilities':{s:1/6 for s in seats}})
    output=NS(finished=True,request_id='real-format-id',outputs=[NS(token_ids=[10,11],text=content,finish_reason='stop',stop_reason=None)])
    async def execute():
        await engine._add_request(core)
        async for _ in engine.generate(output):pass
    asyncio.run(execute());assert len(called)==1 and len(events)==3
    proof={'input':events[1],'output':events[2]}
    n=len(core.prompt_token_ids)
    raw={'status':200,'headers':[],'body':json.dumps({'id':'real-format-id','model':spec.record()['model'],
        'choices':[{'message':{'content':content,'refusal':None},'finish_reason':'stop'}],
        'usage':{'prompt_tokens':n,'completion_tokens':2,'total_tokens':n+2}})}
    session={'identity':{'permit_digest':permit,'instance':life,'engine_startup':events[0]},'remaining_task_digests':[digest(task)]}
    prediction,usage=runner.validate_reply(raw,task,spec,proof,tokenizer=tokenizer,session=session)
    assert prediction['status']=='valid' and usage['input_tokens']==n
    for modifier in ('ids','sampling','permit','tokens','reply_id','process'):
        wrong=deepcopy(proof);response=deepcopy(raw)
        if modifier=='ids':wrong['input']['input_ids']=[1]
        elif modifier=='sampling':wrong['input']['sampling']['temperature']=1
        elif modifier=='permit':wrong['output']['permit_digest']='b'*64
        elif modifier=='tokens':wrong['output']['output_token_ids']=[10]
        elif modifier=='process':wrong['output']['process_lifetime_digest']='b'*64
        else:
            body=json.loads(response['body']);body['id']='foreign';response['body']=json.dumps(body)
        with pytest.raises(ValueError):runner.validate_reply(response,task,spec,wrong,tokenizer=tokenizer,session=session)


def test_source_or_model_drift_refuses_before_http(study,fake_http,monkeypatch):
    root,tasks,_=study;endpoint,state=fake_http
    actual=ex.source_v3()
    monkeypatch.setattr(ex,'source_v3',lambda:{**actual,'source_revision':'f'*40})
    with pytest.raises(ValueError,match='Source'):runner.run_model(root,'qwen3',runner.HttpTransport(endpoint))
    assert not state['requests']


def test_result_corruption_is_detected_before_resume_dispatch(study,fake_http):
    root,tasks,_=study;endpoint,state=fake_http
    runner.run_model(root,'qwen3',runner.HttpTransport(endpoint))
    target=root/'models/qwen3/requests'/digest(tasks[-1])/'result/record.json'
    target.write_bytes(target.read_bytes()+b' ')
    before=len(state['requests'])
    with pytest.raises(ValueError):runner.pending_tasks(root,'qwen3',tasks)
    assert len(state['requests'])==before


def test_run_lock_rejects_concurrent_execution(study):
    root,_,_=study
    with runner.run_lock(root):
        with pytest.raises(BlockingIOError):
            with runner.run_lock(root):pass


def test_analysis_pairwise_signs_are_left_minus_right(study,fake_http):
    from werewolf.tom_llm.baseline_analysis import compute
    root,tasks,ref=study;endpoint,_=fake_http
    predictions={}
    for key in ('qwen3','yi','mistral'):
        runner.run_model(root,key,runner.HttpTransport(endpoint));predictions[key]=runner.read_predictions(root,key)[1]
    contract,run,_=ex.open_run(root)
    # Pure scoring test: replace Yi predictions with nonuniform probabilities, leaving task identity fixed.
    changed=deepcopy(predictions)
    for row in changed['yi']:
        seats=[f'player{i+1}' for i in range(7) if i!=row['observer']]
        raw={'probabilities':{s:(.75 if i==0 else .05) for i,s in enumerate(seats)}}
        row['prediction']=runner.parse_output(json.dumps(raw),row['observer']).to_record()
    result=compute(ref,run,tasks,changed)
    pair=next(p for p in result['llm_pairwise'] if p['left']=='qwen3' and p['right']=='yi')
    left=result['models']['qwen3']['cells']['llm']['tv']['point'];right=result['models']['yi']['cells']['llm']['tv']['point']
    assert pair['Delta_TV']['point']==pytest.approx(left-right)
    assert pair['Delta_KL']['point']==pytest.approx(result['models']['qwen3']['cells']['llm']['kl']['point']-result['models']['yi']['cells']['llm']['kl']['point'])
    # Independent row formulas and bootstrap arithmetic, rather than file generation only.
    import numpy as np
    from werewolf.tom.protocol import bootstrap_indices
    from werewolf.tom_llm.common import identity
    from werewolf.tom_llm.policy import POLICY
    refs=ref.selected(run['games']);indexed={key:{identity(r):r for r in rows} for key,rows in changed.items()}
    def vectors(key,label):
        grouped={}
        for row in refs:
            q=np.asarray(row['q'],dtype=np.float64)
            if label=='Delta_Top1_NU' and (q>0).sum()==6:continue
            pred=row if key=='tom' else indexed[key][identity(row)]['prediction']
            p=np.zeros(7);p[[j for j in range(7) if j!=row['observer']]]=np.exp(pred['non_self_log_probability'])
            support=q>0
            value=(q[support]*(np.log(q[support])-np.log(p[support]))).sum() if label=='Delta_KL' else (
                .5*np.abs(q-p).sum() if label=='Delta_TV' else float(q[p.argmax()]>0))
            grouped.setdefault(row['game_id'],[]).append(value)
        games=sorted(grouped)
        return games,np.asarray([np.mean(grouped[g]) for g in games])
    for label in ('Delta_KL','Delta_TV','Delta_Top1_NU'):
        for left_key,right_key,actual in [('qwen3','yi',pair[label]),
                *[(key,'tom',result['models'][key][label]) for key in changed]]:
            games,left_values=vectors(left_key,label);other_games,right_values=vectors(right_key,label)
            assert games==other_games
            if not games:
                assert actual is None;continue
            delta=left_values-right_values
            indices=bootstrap_indices(games,POLICY['bootstrap']['replicates'],POLICY['bootstrap']['seed'])
            assert actual['point']==pytest.approx(delta.mean(),abs=1e-12)
            assert actual['interval']==pytest.approx(np.quantile(delta[indices].mean(1),[.025,.975],method='linear'),abs=1e-12)



def test_cli_prepare_reuses_real_readers_and_existing_selection(study,cohort,sealed_reference,tmp_path):
    from werewolf.tom_llm.multimodel_evaluation import publish_mock_cohort
    from werewolf.artifact_io import sha256_bytes
    root,tasks,ref=study;_,publication,evaluation,_=sealed_reference
    original=cohort[0].manifest['selection']
    expected={k:original[k] for k in ('parent_folds_digest','selection_digest','game_ids_digest','selection_payload_sha256')}
    provenance={k:original[k] for k in ('publication_digest','fold_manifest_digest','prepared_manifest_digest','pre_reference_selection_evidence_digest','reconciliation_evidence_digest')}
    pre=tmp_path/'pre-evidence';rec=tmp_path/'reconcile-evidence';pre.write_bytes(b'UNIT PRE');rec.write_bytes(b'UNIT RECONCILIATION')
    provenance.update(pre_reference_selection_evidence_digest=sha256_bytes(pre.read_bytes()),reconciliation_evidence_digest=sha256_bytes(rec.read_bytes()))
    prepared=evaluation.parent.parent/'prepared'
    pm=json.loads((prepared/'manifest.json').read_bytes())
    shared=publish_mock_cohort(tmp_path/'cli-cohort',tasks,ref,selection_root=cohort[0].manifest['selection_root'],
        expected=expected,provenance=provenance,prepared_manifest=pm)
    exp=tmp_path/'expected';prov=tmp_path/'provenance';exp.write_bytes(canonical_json_bytes(expected));prov.write_bytes(canonical_json_bytes(provenance))
    config={'models':{'qwen3':{}},'publication':str(publication.path),'evaluation':str(evaluation),
        'expected_reference':ref.seal_digest,'prepared':str(prepared),'selection':cohort[0].manifest['selection_root'],
        'expected_selection_json':str(exp),'selection_provenance_json':str(prov),'pre_reference_evidence':str(pre),
        'reconciliation_evidence':str(rec),'cohort':{'path':str(shared.path),'digest':shared.manifest_digest}}
    cfg=tmp_path/'config.json';cfg.write_bytes(canonical_json_bytes(config))
    result=main(['--action','prepare','--config',str(cfg),'--mode','fake','--games','5','--models','qwen3',
        '--run-id','cli-prepared','--output-root',str(root.parent)])
    _,run,rows=ex.open_run(root.parent/'cli-prepared')
    assert run['request_count']==len(tasks) and run['configuration']['input_sha256']==sha256_bytes(cfg.read_bytes())
    with pytest.raises(ValueError):main(['--action','prepare','--config',str(cfg),'--mode','formal','--games','10',
        '--run-id','bad-prepare','--output-root',str(root.parent)])
    assert not (root.parent/'bad-prepare').exists()
    wrong=tmp_path/'wrong-expected';wrong.write_bytes(canonical_json_bytes({**expected,'selection_digest':'0'*64}))
    config['expected_selection_json']=str(wrong);cfg.write_bytes(canonical_json_bytes(config))
    with pytest.raises(ValueError):main(['--action','prepare','--config',str(cfg),'--mode','fake','--games','5',
        '--run-id','bad-expected','--output-root',str(root.parent)])
    assert not (root.parent/'bad-expected').exists()
    config['expected_selection_json']=str(exp);config['prepared']=str(tmp_path/'missing-prepared');cfg.write_bytes(canonical_json_bytes(config))
    with pytest.raises((ValueError,OSError)):main(['--action','prepare','--config',str(cfg),'--mode','fake','--games','5',
        '--run-id','bad-prepared','--output-root',str(root.parent)])
    assert not (root.parent/'bad-prepared').exists()
    config['prepared']=str(prepared);cfg.write_bytes(canonical_json_bytes(config))
    pre.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='evidence bytes'):main(['--action','prepare','--config',str(cfg),'--mode','fake','--games','5',
        '--run-id','bad-evidence','--output-root',str(root.parent)])
    assert not (root.parent/'bad-evidence').exists()


@pytest.mark.parametrize('mode',['formal','technical'])
@pytest.mark.parametrize('key',['qwen3','yi','mistral'])
def test_permit_binds_sealed_qualification_without_previous_run_or_human_file(tmp_path,monkeypatch,mode,key):
    """Unit-only physical boundary stubs: this artifact is not real Admission."""
    from werewolf.tom_llm import live_admission as admission
    root=tmp_path/'unit-run';spec=ModelSpec(key);source={**ex.source_v3(),'source_inventory_clean':True}
    contract=NS(manifest_digest='a'*64)
    run={'mode':mode,'source':source,'models':{key:spec.digest},'cohort_digest':'c'*64}
    budget={'cohort_digest':run['cohort_digest']};qualified={'budget':budget,'upstream_receipts':{'budget':'b'*64},'source':source}
    from werewolf.tom_llm import deployment
    monkeypatch.setattr(deployment,'qualification_compatibility',lambda s:{'UNIT_ONLY':digest(s)})
    profile={'argv_options':{'UNIT_ONLY':'not_a_real_vllm_config'}};observed=[]
    monkeypatch.setattr(admission,'sys',NS(platform='linux'))
    monkeypatch.setattr(admission,'BASE',tmp_path)
    monkeypatch.setattr(admission,'open_run',lambda *a,**kw:(contract,run,[]))
    monkeypatch.setattr(admission,'source_v3',lambda:source)
    def physical_check(path,expected,model):
        observed.append((path,expected,model.key));return qualified
    monkeypatch.setattr(admission,'qualified_offline',physical_check)
    monkeypatch.setattr(admission,'open_budget',lambda path,model,expected:budget if expected=='b'*64 else None)
    monkeypatch.setattr(admission,'load_cpu_tokenizer',lambda *a:object())
    monkeypatch.setattr(admission,'service_profile',lambda *a:profile)
    monkeypatch.setattr(admission,'qualification_runtime',lambda *a:{'driver':'UNIT_DRIVER','kernel':'UNIT_KERNEL'})
    args={'qualification':tmp_path/'sealed-qualification','expected_qualification':'d'*64,
        'tokenizer_assets':tmp_path/'small-assets','budget_root':tmp_path/'budget'}
    artifact=admission.publish_permit(root,key,**args)
    assert set(artifact.manifest['file_table'])=={'permit.json','vllm.yaml','chat_template.jinja'}
    _,record=admission.open_permit(root,key,artifact.manifest_digest)
    assert set(record)=={'run_root','run_digest','model','model_spec_digest','source','qualified_runtime','profile',
        'qualification_root','qualification_digest','budget_root','tokenizer_assets','evidence_kind','qualification_compatibility'}
    assert record['run_digest']==contract.manifest_digest and record['model_spec_digest']==spec.digest
    assert record['source']==source and record['qualification_digest']=='d'*64 and record['profile']==profile
    assert record['evidence_kind']=='physical_execution_binding'
    assert observed==[(args['qualification'],'d'*64,key),(str(args['qualification']),'d'*64,key)]
    assert not any(hasattr(admission,name) for name in ('approval_fields','active_qualification','technical_evidence'))
    with pytest.raises(ValueError,match='anchor'):
        admission.open_permit(root,key,'e'*64)
    run['cohort_digest']='foreign-cohort'
    with pytest.raises(ValueError,match='profile/budget'):
        admission.open_permit(root,key,artifact.manifest_digest)
    run['cohort_digest']=budget['cohort_digest'];source['source_inventory_clean']=False
    with pytest.raises(ValueError,match='committed'):
        admission.publish_permit(tmp_path/'dirty-run',key,**args)
    source['source_inventory_clean']=True
    def invalid_qualification(*args):raise ValueError('sealed physical qualification rejected')
    monkeypatch.setattr(admission,'qualified_offline',invalid_qualification)
    with pytest.raises(ValueError,match='sealed physical'):
        admission.publish_permit(tmp_path/'invalid-physical-run',key,**args)
    assert not (tmp_path/'invalid-physical-run/models'/key/'permit').exists()


def test_cli_removes_old_approval_and_predecessor_options(tmp_path,monkeypatch):
    from werewolf.tom_llm import live_admission as admission
    args=['--run-id','unit','--output-root',str(tmp_path)]
    for option in ('--approval','--expected-approval','--technical-run','--expected-technical-seal'):
        with pytest.raises(SystemExit) as error:main([*args,option,'removed'])
        assert error.value.code==2
    with pytest.raises(SystemExit) as error:main([*args,'--action','approval-template'])
    assert error.value.code==2
    observed={}
    def publish(root,key,**kwargs):
        observed.update(root=root,key=key,kwargs=kwargs)
        return NS(path=root/'models'/key/'permit',manifest_digest='a'*64)
    monkeypatch.setattr(admission,'publish_permit',publish)
    main([*args,'--action','permit','--models','qwen3','--qualification','UNIT_QUALIFICATION',
        '--expected-qualification','d'*64,'--budget','UNIT_BUDGET','--tokenizer-assets','UNIT_SMALL_ASSETS'])
    assert observed['key']=='qwen3' and observed['kwargs']=={
        'qualification':'UNIT_QUALIFICATION','expected_qualification':'d'*64,'budget_root':'UNIT_BUDGET',
        'tokenizer_assets':'UNIT_SMALL_ASSETS'}


def test_managed_cli_reuses_persistent_service_manager(monkeypatch):
    from werewolf.tom_llm import baseline_cli as cli,service_manager
    calls=[]
    class Managed:
        def __init__(self,*args):calls.append(('bind',args))
        def start(self,**kw):calls.append(('start',kw));return {'pid':123}
        def stop(self,**kw):calls.append(('stop',kw))
    monkeypatch.setattr(service_manager,'Service',Managed)
    with cli.owned_service('/UNIT/run','qwen3','d'*64) as pid:assert pid==123
    assert calls==[('bind',('/UNIT/run','qwen3','d'*64)),('start',{'execute':True}),('stop',{'execute':True})]



def test_environment_recheck_uses_target_prefix_not_controller_python(tmp_path,monkeypatch):
    from werewolf.tom_llm import live_admission as admission
    prefix=tmp_path/'model-env';(prefix/'bin').mkdir(parents=True)
    (prefix/'bin/python').write_bytes(b'UNIT INTERPRETER NOT EXECUTED')
    observed={}
    def run(command,**kwargs):
        observed.update(command=command,kwargs=kwargs)
        return NS(stdout=json.dumps({'evidence_kind':'UNIT_CPU_CHILD_RESULT','prefix':str(prefix)}))
    monkeypatch.setattr(admission.subprocess,'run',run)
    result=admission.capture_bound_environment(prefix,'e'*64)
    assert result['evidence_kind']=='UNIT_CPU_CHILD_RESULT'
    assert observed['command'][0]==str(prefix/'bin/python')
    assert observed['command'][1:4]==['-I','-S','-B']
    assert observed['kwargs']['env']['CUDA_VISIBLE_DEVICES']==''
    assert 'capture_environment' in observed['command'][5]
    assert observed['kwargs']['check'] is True


def test_environment_recheck_surfaces_child_dependency_diagnostics(tmp_path,monkeypatch):
    from werewolf.tom_llm import live_admission as admission
    prefix=tmp_path/'model-env';(prefix/'bin').mkdir(parents=True)
    (prefix/'bin/python').write_bytes(b'UNIT INTERPRETER NOT EXECUTED')
    def run(command,**kwargs):
        raise subprocess.CalledProcessError(1,command,output='UNIT child stdout',
            stderr='ValueError: dependency integrity failure\nuns-local-broken requires uns-local-missing')
    monkeypatch.setattr(admission.subprocess,'run',run)
    with pytest.raises(ValueError,match='uns-local-missing') as error:
        admission.capture_bound_environment(prefix,'e'*64)
    assert 'UNIT child stdout' in str(error.value)
    assert isinstance(error.value.__cause__,subprocess.CalledProcessError)



def test_cross_session_response_is_rejected_before_recovery(study,fake_http,monkeypatch):
    root,tasks,_=study;endpoint,_=fake_http
    runner.run_model(root,'qwen3',runner.HttpTransport(endpoint))
    real=runner.owned
    def mismatched(path,*args):
        artifact,record=real(path,*args)
        if path.name.startswith('response-'):record={**record,'session_digest':'f'*64}
        return artifact,record
    monkeypatch.setattr(runner,'owned',mismatched)
    with pytest.raises(ValueError,match='session mismatch'):
        runner.pending_tasks(root,'qwen3',tasks)


def test_foreign_request_directory_blocks_before_any_http(study,fake_http):
    root,tasks,_=study;endpoint,state=fake_http
    (root/'models/qwen3/requests/foreign').mkdir(parents=True)
    with pytest.raises(ValueError,match='extra'):
        runner.run_model(root,'qwen3',runner.HttpTransport(endpoint))
    assert not state['requests']


@pytest.mark.parametrize('target',['failure','session_complete'])
def test_prediction_seal_binds_performance_records(study,fake_http,target):
    root,tasks,_=study;endpoint,state=fake_http;state['reject_first']=True
    seal=runner.run_model(root,'qwen3',runner.HttpTransport(endpoint))
    if target=='failure':
        path=root/'models/qwen3/requests'/digest(tasks[0])/'failure-1';kind='mm_baseline_event'
    else:
        path=next((root/'models/qwen3/sessions').glob('*-complete'));kind='mm_baseline_session_complete'
    _,record=ex.read_record(path,kind)
    record['elapsed_seconds']+=1000
    shutil.rmtree(path)  # Test-owned synthetic artifact; simulate self-consistent replacement.
    ex.publish_record(path,kind,record)
    with pytest.raises(ValueError):runner.read_predictions(root,'qwen3',expected=seal.manifest_digest)


def test_status_rejects_empty_seal_directory(study):
    root,_,_=study;(root/'models/qwen3/seal').mkdir(parents=True)
    result=runner.status(root)['models']['qwen3']
    assert result['sealed'] is False
    assert result['errors']


def test_live_transport_reuses_physical_check_from_permit(tmp_path,monkeypatch):
    from werewolf.tom_llm import live_admission as admission,candidate_client,physical
    log=tmp_path/'engine.jsonl';permit='a'*64
    log.write_text(json.dumps({'kind':'startup','permit_digest':permit})+'\n')
    record={'profile':{'argv_options':{'port':18581,'model':'/UNIT/QWEN'}},'tokenizer_assets':'/UNIT/SMALL'}
    checked=[];inventories=[]
    def open_permit(*args,**kwargs):
        assert kwargs.get('physical',True) is True
        checked.append(args);return NS(manifest_digest=permit),record
    monkeypatch.setattr(admission,'open_permit',open_permit)
    monkeypatch.setattr(admission,'qualified_offline',lambda *a,**kw:pytest.fail('duplicate full weight/environment scan'))
    monkeypatch.setattr(admission,'capture_live_instance',lambda *a,**kw:{'engine_log':str(log)})
    def stats(r):inventories.append(r['model']);return {'UNIT':'STATS'}
    monkeypatch.setattr(candidate_client,'model_stats',stats)
    monkeypatch.setattr(runner,'load_cpu_tokenizer',lambda *a:object())
    transport=runner.LiveTransport(tmp_path,'qwen3',permit,123)
    assert len(checked)==1 and transport.stats=={'UNIT':'STATS'}
    assert inventories[0]['directory']=='/UNIT/QWEN'
    assert set(inventories[0]['files'])==set(physical.asset_spec(ModelSpec('qwen3'))['files'])|{'HF_REVISION'}
