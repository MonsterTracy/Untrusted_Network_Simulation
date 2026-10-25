"""Synthetic shared cohorts and offline tokenizers. No live calls or physical admission."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil

import httpx2 as httpx
import openai
import pytest

from tests.canonical_collection.test_structured_token_plan import _prefix
from tests.tom_llm.test_selection_inference import uniform
from tests.tom_llm.test_server_fixed import PARITY_CASES
from werewolf.artifact_io import publish_artifact, verify_artifact, canonical_json_bytes, sha256_bytes
from werewolf.tom_llm import model_spec as ms, multimodel as mm
from werewolf.tom_llm.common import digest, identity
from werewolf.tom_llm.inference import MockBackend, Reply, InfrastructureFailure
from werewolf.tom_llm.model_spec import ModelSpec, render, wire, check_cpu_parity, token_preflight
from werewolf.tom_llm.provider_semantics import NotExecuted, OutcomeUnknown, classify_provider_error, require_live_admission
from werewolf.tom_llm.task import from_legacy, to_legacy, public_tasks, task_binding
from werewolf.tom_llm.study import make_request
from werewolf.tom_llm.server_fixed import wire_request_v2
from werewolf.tom_llm.multimodel_evaluation import publish_mock_cohort, compare_models, paired_extras
from werewolf.tom_llm.evaluation import compare

KEYS = ('qwen3','yi','mistral')


@pytest.fixture
def task():
    return from_legacy(make_request(_prefix(),0,fold=0))


@pytest.fixture
def cohort(tmp_path,monkeypatch,sealed_reference):
    # Like the historical 5-game Reference fixture, shrink only fixture selection counts.
    # Production select_games remains fixed at 1500 and 30 games/fold.
    from werewolf.tom_llm import selection
    ref,publication,evaluation,report = sealed_reference
    def tiny(folds,*,scope):
        games = selection.validate_folds(folds)
        result = {'scope':scope,'rule':dict(selection.POLICY['subset']),
            'folds':[{'fold':r['fold'],'game_ids':sorted(r['held_out_game_ids'])} for r in folds],
            'game_ids':games,'parent_folds_digest':digest(folds)}
        return {**result,'selection_digest':digest(result)}
    monkeypatch.setattr(selection,'select_games',tiny)
    selected = selection.publish_selection(tmp_path/'selection',ref.folds,scope='full1500')
    payload = json.loads((selected.path/'selection.json').read_bytes())
    expected = {'parent_folds_digest':payload['parent_folds_digest'],'selection_digest':payload['selection_digest'],
        'game_ids_digest':digest(payload['game_ids']),'selection_payload_sha256':sha256_bytes(canonical_json_bytes(payload))}
    prepared = verify_artifact(evaluation.parent.parent/'prepared',expected_artifact_type='backbone_tom_study',
        expected_schema_version='classic7_backbone_study_v1')
    provenance = {'publication_digest':ref.publication_digest,'fold_manifest_digest':ref.fold_manifest_digest,
        'prepared_manifest_digest':prepared.manifest_digest,'pre_reference_selection_evidence_digest':'e'*64,
        'reconciliation_evidence_digest':'f'*64}
    membership = {g:r['fold'] for r in ref.folds for g in r['held_out_game_ids']}
    tasks = public_tasks([p for g in publication.public_view.games for p in g.authoritative_pre_prefixes],membership)
    path = tmp_path/'cohort'
    artifact = publish_mock_cohort(path,tasks,ref,selection_root=selected.path,expected=expected,
        provenance=provenance,prepared_manifest=prepared.manifest)
    return artifact,tasks,ref


def prepare(tmp_path,cohort,key='qwen3',service='synthetic:service-1'):
    execution = mm.ModelExecution(ModelSpec(key),service)
    root = tmp_path/key
    mm.prepare_model_mock(root,cohort[0].path,expected_cohort_digest=cohort[0].manifest_digest,
        execution=execution,tokenizer_identity='synthetic:count-v1',count_tokens=lambda _:100)
    return root,execution


@pytest.mark.parametrize('key',KEYS)
def test_shared_public_task_schema_and_model_native_wrapping(task,key):
    spec = ModelSpec(key)
    rendered = render(task,spec)
    assert wire(task,spec)['messages'] == task['messages']
    assert wire(task,spec)['response_format']['json_schema']['schema'] == task['response_schema']
    assert rendered['task_digest'] == digest(task)
    assert all(message['content'] in rendered['text'] for message in task['messages'])
    assert 'model' not in task_binding()['policy']
    assert 'rendered_prompt' not in task
    if key == 'qwen3':
        assert wire(task,spec) == wire_request_v2(to_legacy(task))
        assert rendered['text'] == to_legacy(task)['rendered_prompt']
    elif key == 'yi':
        assert rendered['text'].startswith(task['messages'][0]['content']+'<|im_start|>user\n')
        assert rendered['text'].endswith('<|im_end|>\n<|im_start|>assistant\n')
        assert wire(task,spec)['extra_body']['stop_token_ids'] == [7]
        assert spec.record()['termination']['generation_eos_token_id'] == 2
    else:
        assert rendered['text'].startswith('<s>[INST] ')
        assert rendered['text'].endswith('[/INST]') and rendered['text'].count('<s>') == 1
    if key != 'qwen3':
        assert '<think>' not in rendered['text']
        assert 'enable_thinking' not in wire(task,spec)['extra_body']['chat_template_kwargs']


@pytest.mark.parametrize('key',KEYS)
def test_registry_is_pinned_and_cannot_authorize_arbitrary_models(key,monkeypatch,tmp_path):
    row = ModelSpec(key).record()
    assert len(row['revision']) == 40 and row['revision'] != 'main'
    assert row['source']['file_url_prefix'].endswith(row['revision']+'/')
    assert row['source']['weights_verified'] is False
    assert row['context_length'] == 8192 and row['quantization'] is None
    row['revision'] = '0'*40
    assert ModelSpec(key).record()['revision'] != row['revision']
    bad = tmp_path/'models.json';bad.write_bytes(canonical_json_bytes({key:row}))
    monkeypatch.setattr(ms,'REGISTRY',bad)
    with pytest.raises(ValueError,match='registry identity'):ModelSpec(key).record()


@pytest.mark.parametrize('key',['Llama','unknown','Qwen/Qwen3-8B',True])
def test_unauthorized_registry_model_rejected(key):
    with pytest.raises(ValueError):ModelSpec(key).record()


@pytest.mark.parametrize('field',['q','role','label_observed','raw_text','primary_eligible','future_speech'])
def test_public_task_forbids_private_or_supervised_payload(task,field):
    task[field] = 'forbidden'
    for key in KEYS:
        with pytest.raises(ValueError):render(task,ModelSpec(key))


@pytest.mark.parametrize('key',KEYS)
def test_template_drift_and_complete_context_overflow_rejected(task,key,monkeypatch):
    spec = ModelSpec(key)
    target = Path(ms.__file__).parent/spec.record()['chat_template']['path']
    read = Path.read_bytes
    with monkeypatch.context() as patch:
        patch.setattr(Path,'read_bytes',lambda p:read(p)+b'changed' if p==target else read(p))
        with pytest.raises(ValueError,match='template.*drift'):render(task,spec)
    for count in [0,True,7681]:
        with pytest.raises(ValueError,match='context'):token_preflight([task],spec,count_tokens=lambda _:count)
    assert token_preflight([task],spec,count_tokens=lambda _:7680)['maximum_input_tokens']==7680


@pytest.mark.parametrize('key',KEYS)
def test_real_offline_native_template_and_input_ids_parity(key):
    env = {'qwen3':'UNS_LLM_TOKENIZER_ASSETS','yi':'UNS_LLM_YI_ASSETS','mistral':'UNS_LLM_MISTRAL_ASSETS'}[key]
    directory = os.environ.get(env)
    if not directory:pytest.skip('NOT_RUN: explicitly supplied official small assets required: '+env)
    tasks = [from_legacy(make_request(factory(),observer,fold=0)) for factory,observer in PARITY_CASES]
    result = check_cpu_parity(tasks,ModelSpec(key),directory)
    assert result['status']=='CPU_PARITY_PASS' and result['request_count']==11
    assert result['weights_verified'] is result['gpu_admission'] is False


def test_wrong_official_asset_directory_rejected(tmp_path):
    with pytest.raises(ValueError,match='missing'):ms.verify_cpu_assets(ModelSpec('yi'),tmp_path)


@pytest.mark.parametrize('key',KEYS)
def test_durable_model_mock_resume_preserves_reply_and_uniform_status(tmp_path,cohort,key):
    root,execution = prepare(tmp_path,cohort,key)
    calls=[]
    def respond(request):
        calls.append(request)
        return uniform(request)
    class Crash(Exception):pass
    with pytest.raises(Crash):
        mm.run_model_mock(root,execution,MockBackend(respond),after_response=lambda:(_ for _ in ()).throw(Crash()))
    assert len(calls)==1
    mm.run_model_mock(root,execution,MockBackend(respond))
    assert len(calls)==len(cohort[1])
    seal=mm.seal_model_mock(root,execution)
    opened,rows=mm.open_model_predictions(root,execution)
    assert opened.manifest_digest==seal.manifest_digest
    assert all(r['prediction']['status']=='valid' for r in rows)
    assert [r['task_digest'] for r in rows]==[digest(t) for t in cohort[1]]
    assert all(r['model_spec_digest']==execution.spec.digest for r in rows)
    assert b'label_observed' not in (root/'contract/manifest.json').read_bytes()
    with pytest.raises(ValueError,match='closed'):mm.run_model_mock(root,execution,MockBackend(respond))


@pytest.mark.parametrize('change',['model','service','revision'])
def test_cross_model_revision_service_resume_rejected_before_call(tmp_path,cohort,change):
    root,execution = prepare(tmp_path,cohort)
    backend = MockBackend(uniform)
    if change=='model':wrong=mm.ModelExecution(ModelSpec('yi'),execution.service_id)
    elif change=='service':wrong=mm.ModelExecution(execution.spec,'synthetic:different')
    else:
        artifact=verify_artifact(root/'contract',expected_artifact_type='llm_multimodel_mock_contract',expected_schema_version=mm.VERSION)
        fields={k:v for k,v in artifact.manifest.items() if k not in ('manifest_digest','file_table')}
        fields['execution']['model_spec']['revision']='0'*40
        fields['execution_digest']=digest(fields['execution'])
        alternative=tmp_path/'wrong-revision'
        publish_artifact(alternative/'contract',manifest_fields=fields,files={})
        root=alternative;wrong=execution
    with pytest.raises(ValueError,match='execution drift'):mm.run_model_mock(root,wrong,backend)
    assert backend.calls==[]


def test_unknown_dispatch_blocks_retry_and_recovery_without_prediction(tmp_path,cohort):
    root,execution = prepare(tmp_path,cohort)
    calls=[]
    def lost(request):
        calls.append(request)
        raise classify_provider_error(openai.APITimeoutError(request=object()))
    with pytest.raises(OutcomeUnknown):mm.run_model_mock(root,execution,MockBackend(lost))
    with pytest.raises(ValueError,match='resend forbidden'):mm.run_model_mock(root,execution,MockBackend(lost))
    assert len(calls)==1 and not list((root/'requests').glob('*/result'))
    with pytest.raises((ValueError,FileNotFoundError)):mm.seal_model_mock(root,execution)


def test_only_definitely_not_executed_is_retried_with_original_budget(tmp_path,cohort):
    root,execution = prepare(tmp_path,cohort)
    count=0
    def retry(request):
        nonlocal count
        count+=1
        if count==1:raise NotExecuted('synthetic explicit rejection')
        return uniform(request)
    mm.run_model_mock(root,execution,MockBackend(retry))
    mm.seal_model_mock(root,execution);_,rows=mm.open_model_predictions(root,execution)
    assert len(rows[0]['attempt_digests'])==2 and count==len(rows)+1
    other,ex=prepare(tmp_path/'other',cohort)
    def reject(_):raise NotExecuted('synthetic rejection')
    backend=MockBackend(reject)
    with pytest.raises(InfrastructureFailure):mm.run_model_mock(other,ex,backend)
    assert len(backend.calls)==2
    with pytest.raises(InfrastructureFailure):mm.run_model_mock(other,ex,backend)
    assert len(backend.calls)==2


def test_provider_failure_classifier_is_conservative_and_live_remains_blocked():
    for error in [openai.APITimeoutError(request=object()),openai.APIConnectionError(request=object()),
                  httpx.ReadTimeout('accepted?'),httpx.WriteError('partial?'),httpx.RemoteProtocolError('lost')]:
        assert isinstance(classify_provider_error(error),OutcomeUnknown)
    error=openai.APIConnectionError(request=object());error.__cause__=httpx.ConnectError('no connection')
    assert isinstance(classify_provider_error(error),NotExecuted)
    request=httpx.Request('POST','http://127.0.0.1/v1')
    response=httpx.Response(429,request=request)
    assert isinstance(classify_provider_error(openai.RateLimitError('rejected',response=response,body=None)),NotExecuted)
    assert isinstance(classify_provider_error(RuntimeError('OOM')),InfrastructureFailure)
    with pytest.raises(NotImplementedError):require_live_admission({'pretend':'CPU_PASS'})
    with pytest.raises(ValueError):mm.ModelExecution(ModelSpec('qwen3'),'real-service').binding()


def test_three_model_end_to_end_uses_exact_selection_reference_and_math(tmp_path,cohort):
    runs={};anchors={}
    for key in KEYS:
        root,ex=prepare(tmp_path,cohort,key)
        mm.run_model_mock(root,ex,MockBackend(uniform))
        seal=mm.seal_model_mock(root,ex);runs[key]=(root,ex);anchors[key]=seal.manifest_digest
    result=compare_models(cohort[2],cohort[0].path,expected_cohort_digest=cohort[0].manifest_digest,
        runs=runs,expected_seal_digests=anchors)
    for key,(root,ex) in runs.items():
        _,rows=mm.open_model_predictions(root,ex)
        legacy=compare(cohort[2],rows,cohort[2].game_ids)
        current=result['comparisons'][key]
        for field in ['cells','Delta_KL','bootstrap_indices_digest','nu_bootstrap_indices_digest','primary_row_identity_digest']:
            assert current[field]==legacy[field]
        assert current['Delta_TV']['point']==pytest.approx(0,abs=1e-12)
        if current['Delta_Top1_NU'] is not None:
            assert current['Delta_Top1_NU']['point']==pytest.approx(0,abs=1e-12)
    assert result['formal_result'] is False
    wrong=deepcopy(anchors);wrong['yi']=anchors['qwen3']
    with pytest.raises(ValueError,match='anchor'):compare_models(cohort[2],cohort[0].path,
        expected_cohort_digest=cohort[0].manifest_digest,runs=runs,expected_seal_digests=wrong)
    swapped=dict(runs);swapped['yi']=runs['qwen3']
    with pytest.raises(ValueError,match='label/spec'):compare_models(cohort[2],cohort[0].path,
        expected_cohort_digest=cohort[0].manifest_digest,runs=swapped,expected_seal_digests=anchors)
    with pytest.raises(ValueError,match='three-model'):compare_models(cohort[2],cohort[0].path,
        expected_cohort_digest=cohort[0].manifest_digest,runs={'yi':runs['yi']},expected_seal_digests=anchors)


def test_cross_model_copied_result_and_incomplete_cohort_are_rejected(tmp_path,cohort):
    q,qe=prepare(tmp_path,cohort,'qwen3');yi,ye=prepare(tmp_path,cohort,'yi')
    mm.run_model_mock(q,qe,MockBackend(uniform))
    lineage=next((q/'requests').iterdir())
    target=yi/'requests'/lineage.name
    shutil.copytree(lineage,target)
    backend=MockBackend(uniform)
    with pytest.raises(ValueError,match='ownership'):mm.run_model_mock(yi,ye,backend)
    assert backend.calls==[]
    from werewolf.tom_llm.multimodel_evaluation import validate_public_coverage
    for rows in [cohort[1][:-1],cohort[1]+cohort[1][:1]]:
        with pytest.raises(ValueError):validate_public_coverage(rows,cohort[2],cohort[2].game_ids)


def test_shared_reference_or_selection_drift_is_rejected(tmp_path,cohort):
    from dataclasses import replace
    with pytest.raises(ValueError,match='Reference drift'):
        compare_models(replace(cohort[2],seal_digest='0'*64),cohort[0].path,
            expected_cohort_digest=cohort[0].manifest_digest,runs={},expected_seal_digests={})
    root=Path(cohort[0].manifest['selection_root'])
    (root/'selection.json').write_bytes(b'{}')
    with pytest.raises(ValueError):mm.read_cohort(cohort[0].path,cohort[0].manifest_digest)


@pytest.mark.parametrize('key',KEYS)
def test_raw_speech_private_links_and_future_are_causally_isolated(key,monkeypatch):
    from tests.tom_llm.test_prompt_output import rebound_prefix
    import werewolf.development_publication as publication
    monkeypatch.setattr(publication,'open_role_sidecar',lambda *_a,**_k:pytest.fail('private role access'))
    first=from_legacy(make_request(rebound_prefix(),0,fold=0))
    second=from_legacy(make_request(rebound_prefix(raw_text='SECRET_ROLE_AND_LABEL',
        observation_marker='PRIVATE_LABEL_LINK'),0,fold=0))
    assert first['plan_digest']!=second['plan_digest']
    assert render(first,ModelSpec(key))['text']==render(second,ModelSpec(key))['text']
    assert wire(first,ModelSpec(key))==wire(second,ModelSpec(key))
    with pytest.raises(ValueError,match='turn_start'):rebound_prefix(include_current_speech=True)
    wrong=deepcopy(first);wrong['messages'][1]['content']+='\nFUTURE_OR_PRIVATE_PAYLOAD'
    with pytest.raises(ValueError):render(wrong,ModelSpec(key))


def test_resume_template_or_source_drift_fails_before_any_calls(tmp_path,cohort,monkeypatch):
    root,execution=prepare(tmp_path,cohort,'yi')
    target=Path(ms.__file__).parent/execution.spec.record()['chat_template']['path']
    read=Path.read_bytes
    monkeypatch.setattr(Path,'read_bytes',lambda p:read(p)+b'drift' if p==target else read(p))
    backend=MockBackend(uniform)
    with pytest.raises(ValueError):mm.run_model_mock(root,execution,backend)
    assert backend.calls==[]


def test_foreign_and_noncontiguous_pending_events_block_whole_run(tmp_path,cohort):
    root,execution=prepare(tmp_path,cohort)
    contract,_,tasks=mm.open_model_mock(root,execution)
    # Last task is deliberately contaminated: no earlier clean task may dispatch.
    task=tasks[-1];lineage=root/'requests'/digest(identity(task))
    mm.event(lineage/'attempt-2',contract,task,{'number':2,'outcome':'not_executed'})
    backend=MockBackend(uniform)
    with pytest.raises(ValueError,match='noncontiguous'):mm.run_model_mock(root,execution,backend)
    assert backend.calls==[]


@pytest.mark.parametrize('key',KEYS)
def test_semantic_failure_is_terminal_and_distinct_from_legal_uniform(tmp_path,cohort,key):
    root,execution=prepare(tmp_path,cohort,key)
    backend=MockBackend(lambda _:Reply('not JSON','stop',{}))
    mm.run_model_mock(root,execution,backend)
    mm.seal_model_mock(root,execution)
    _,rows=mm.open_model_predictions(root,execution)
    assert len(backend.calls)==len(cohort[1])
    assert all(r['prediction']['status']=='fallback_uniform' and
               r['prediction']['failure_type']=='invalid_json' and len(r['attempt_digests'])==1 for r in rows)


def test_paired_tv_and_nu_nonzero_mixed_games_and_no_nu(cohort):
    # Pure synthetic mathematics fixture; no re-sealing or changing real Reference data.
    from dataclasses import replace
    import numpy as np
    from werewolf.tom.protocol import bootstrap_indices
    from werewolf.tom.scoring import paired_summary
    from werewolf.tom_llm.output import parse_output
    from werewolf.tom_llm.policy import POLICY
    reference=cohort[2];games=reference.game_ids
    nu_games=set(games[:2]);modified=[];predictions=[]
    for row in reference.rows:
        q=[0. if i==row['observer'] else 1/6 for i in range(7)]
        target=[i for i in range(7) if i!=row['observer']][1]
        if row['game_id'] in nu_games:q=[float(i==target) for i in range(7)]
        modified.append({**row,'q':q})
    by_id={identity(r):r for r in modified}
    reference=replace(reference,rows=tuple(modified),
        primary_rows=tuple(by_id[identity(r)] for r in reference.primary_rows))
    for task in cohort[1]:
        target=[i for i in range(7) if i!=task['observer']][1]
        content=json.dumps({'probabilities':{f'player{i+1}':int(i==target) for i in range(7) if i!=task['observer']}})
        predictions.append({k:task[k] for k in ('game_id','boundary_id','prefix_digest','plan_digest','observer','fold')} |
            {'canonical_shift':0,'temporal_condition':'explicit_day_phase',
             'prediction':parse_output(content,task['observer']).to_record()})
    legacy=compare(reference,predictions,games)
    extra=paired_extras(reference,predictions,games,legacy)
    assert set(legacy['nu_game_ids'])==nu_games
    assert extra['Delta_Top1_NU']['point']==pytest.approx(1.)
    left,right=[],[]
    for g in games:
        rs=[r for r in reference.primary_rows if r['game_id']==g]
        pred={identity(p):p['prediction']['probability'] for p in predictions}
        left.append(np.mean([sum(abs(a-b) for a,b in zip(r['q'],pred[identity(r)]))/2 for r in rs]))
        right.append(np.mean([sum(abs(a-b) for a,b in zip(r['q'],r['probability']))/2 for r in rs]))
    expected_tv=paired_summary(left,right,reference.bootstrap_indices,.95)
    assert extra['Delta_TV']['point']==pytest.approx(expected_tv['point'],abs=1e-12)
    assert extra['Delta_TV']['interval']==pytest.approx(expected_tv['interval'],abs=1e-12)
    assert {k:v for k,v in extra['Delta_TV'].items() if k not in ('point','interval')}=={
        k:v for k,v in expected_tv.items() if k not in ('point','interval')}
    expected_nu=paired_summary([1.,1.],[0.,0.],bootstrap_indices(sorted(nu_games),
        POLICY['bootstrap']['replicates'],POLICY['bootstrap']['seed']),.95)
    assert extra['Delta_Top1_NU']==expected_nu
    uniform=replace(reference,primary_rows=tuple({**r,'q':[0. if i==r['observer'] else 1/6 for i in range(7)]}
        for r in reference.primary_rows))
    assert paired_extras(uniform,predictions,games,compare(uniform,predictions,games))['Delta_Top1_NU'] is None


def test_unbound_tokenizer_file_cannot_be_implicitly_loaded(tmp_path):
    directory=os.environ.get('UNS_LLM_MISTRAL_ASSETS')
    if not directory:pytest.skip('NOT_RUN: official offline Mistral assets required')
    shutil.copytree(directory,tmp_path/'assets')
    (tmp_path/'assets/chat_template.jinja').write_text('unauthorized override')
    with pytest.raises(ValueError,match='unbound'):ms.load_cpu_tokenizer(ModelSpec('mistral'),tmp_path/'assets')


def test_cpu_cli_missing_assets_reports_not_run_without_network(tmp_path,cohort,capsys,monkeypatch):
    import sys
    import socket
    from scripts.check_tom_multimodel_cpu import main
    from werewolf.artifact_io import canonical_jsonl_bytes
    tasks=tmp_path/'public.jsonl';tasks.write_bytes(canonical_jsonl_bytes(cohort[1]))
    monkeypatch.setattr(sys,'argv',['check','--requests',str(tasks)])
    monkeypatch.setattr(socket,'create_connection',lambda *_a,**_k:pytest.fail('network forbidden'))
    main();result=json.loads(capsys.readouterr().out)
    assert result['physical_admission'] is False
    assert {r['status'] for r in result['models'].values()}=={'NOT_RUN'}
