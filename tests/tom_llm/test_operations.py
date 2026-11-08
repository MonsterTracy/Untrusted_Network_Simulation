"""统一 CLI、一次性准备和主路径编排。"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from tests.tom_llm.conftest import Replies
from werewolf.tom_llm import operations as ops,experiment as ex
from werewolf.tom_llm import cli
from werewolf.tom_llm.common import digest


def test_only_one_public_cli_without_qualification_permit_or_parallel_entrypoints():
    parser = argparse.ArgumentParser()
    cli.register(parser.add_subparsers(dest='command'))
    args = parser.parse_args(['tom-llm','--config','x.yaml','run','--model','yi','--run-id','r','--execute-live'])
    assert args.operation=='run' and args.model=='yi' and args.execute_live
    for removed in ('qualify','admit','permit','serve','open','fake','check-config'):
        with pytest.raises(SystemExit):
            parser.parse_args(['tom-llm',removed])
    root = Path(__file__).resolve().parents[2]
    for name in ('execution','inference','baseline_cli','server_fixed','transport','live_service','gpu_probe',
                 'candidate_client','physical_admission','live_admission','qualification_inputs','qualification_evidence'):
        assert not (root/'werewolf/tom_llm'/ (name+'.py')).exists()
    assert not list((root/'scripts').glob('*tom_multimodel*'))


def config_for(run):
    return {'output_root':str(run.root.parent),'cohort':{'path':run.record['cohort_root'],'digest':run.record['cohort_digest']},
        'publication':run.record['publication'],'evaluation':run.record['evaluation'],
        'expected_reference':run.record['reference']['seal_digest'],'gpu_uuid':'GPU-example',
        'models':{run.key:{'model_directory':'not-loaded','environment_prefix':'not-loaded','tokenizer_assets':'not-loaded'}}}


def test_config_rejects_old_execution_settings_and_duplicate_keys(make_run,tmp_path):
    c = config_for(make_run())
    path = tmp_path/'config.json'
    path.write_text(json.dumps(c))
    assert ops.load_config(path)==c
    c['permit'] = 'old'
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='old qualification'):
        ops.load_config(path)
    path.write_text('{"models":{},"models":{}}')
    with pytest.raises(ValueError):
        ops.load_config(path)


def test_prepare_has_fixed_task_and_no_rounds_or_admission(make_run):
    run = make_run()
    protocol = run.record['protocol']
    assert protocol['research_protocol']=='classic7_suspicion_set_research_protocol_v2'
    assert protocol['schema_version']=='classic7_suspicion_set_execution_v2'
    assert protocol['task']['schema_version']=='classic7_llm_suspicion_set_task_v2'
    assert protocol['population']['public_requests']==10714
    assert protocol['qualification_rounds']==0
    assert protocol['physical_admission_required'] is protocol['execution_permit_required'] is False


@pytest.mark.parametrize('change',(None,'count','fold','rows','reference'))
def test_formal_partition_uses_actual_task_folds_not_nonexistent_binding_field(change):
    from werewolf.tom_llm.reference import fixed_oof_contract
    games = [str(i) for i in range(150)]
    tasks = [{'game_id':games[i%150],'fold':i%150//30} for i in range(10714)]
    manifest = {'selection':{'game_ids':games,'scope':'subset150'},
                'row_identity_digest':ex.FORMAL_PUBLIC_ROWS_DIGEST,
                'reference':fixed_oof_contract()['reference']}
    if change=='count':
        tasks.pop()
    elif change=='fold':
        for row in tasks:
            row['fold'] = 0
    elif change=='rows':
        manifest['row_identity_digest'] = '0'*64
    elif change=='reference':
        manifest['reference'] = {}
    if change is None:
        ex.require_formal_cohort(manifest,tasks)
    else:
        with pytest.raises(ValueError):
            ex.require_formal_cohort(manifest,tasks)


def test_over_budget_preparation_writes_nothing(make_run,tmp_path):
    with pytest.raises(ValueError,match='budget'):
        make_run(name='too-long',count=7681)
    assert not (tmp_path/'too-long').exists()


def test_incomplete_formal_population_or_dirty_source_rejected_without_gpu(make_run,monkeypatch,tmp_path):
    mock = make_run()
    kwargs = dict(run_id='formal',model='qwen3',cohort_root=mock.record['cohort_root'],
        expected_cohort=mock.record['cohort_digest'],publication=mock.record['publication'],
        evaluation=mock.record['evaluation'],expected_reference=mock.record['reference']['seal_digest'])
    source = deepcopy(mock.record['source'])
    monkeypatch.setattr(ex,'source_identity',lambda:{**source,'source_inventory_clean':False})
    with pytest.raises(ValueError,match='committed Source'):
        ex.prepare_run(tmp_path/'formal',**kwargs)
    monkeypatch.setattr(ex,'source_identity',lambda:{**source,'source_inventory_clean':True})
    with pytest.raises(ValueError,match='150/10714'):
        ex.prepare_run(tmp_path/'formal',**kwargs)
    assert not (tmp_path/'formal').exists()


def test_source_changed_blocks_resume(make_run,monkeypatch):
    run = make_run()
    monkeypatch.setattr(ex,'source_identity',lambda:{'different':'source'})
    with pytest.raises(ValueError,match='Source changed'):
        ex.open_run(run.root,check_source=True)


def test_mock_record_cannot_masquerade_as_formal(make_run):
    run = make_run()
    r = deepcopy(run.record)
    r['mode'] = 'formal'
    replacement = run.root.parent/'forged'
    ex.publish_record(replacement/'contract','tom_baseline_run',r)
    with pytest.raises(ValueError,match='formal'):
        ex.open_run(replacement)


def test_single_service_start_and_stop_for_whole_model(make_run,monkeypatch):
    from werewolf.tom_llm import service_manager as sm,live_runner as lr,deployment as dep
    run = make_run()
    runtime = {'argv_options':{'port':18581,'model':'m'},'environment_prefix':'e','gpu_uuid':'GPU-test'}
    run.record['runtime'] = runtime
    run.record['mode'] = 'formal'
    args = SimpleNamespace(execute_live=True,model=run.key,run_id=run.record['run_id'],resume=False)
    c = config_for(run)
    monkeypatch.setattr(ops.sys,'platform','linux')
    monkeypatch.setattr(ops,'open_run',lambda *a,**kw:run)
    monkeypatch.setattr(dep,'profile',lambda *a,**kw:runtime)
    transport = Replies()
    monkeypatch.setattr(lr,'HttpTransport',lambda *a,**kw:transport)
    events = []
    class Manager:
        def __init__(self,r):
            assert r is run
        def start(self,**kw):
            events.append('start')
        def stop(self,**kw):
            events.append('stop')
        def guard(self):
            pass
    monkeypatch.setattr(sm,'Service',Manager)
    result = ops.execute(c,args)
    assert result.manifest['artifact_type']=='tom_baseline_predictions'
    assert len(transport.calls)==len(run.tasks) and events==['start','stop']
    args.resume = True
    ops.execute(c,args)
    assert events==['start','stop']


def test_unknown_blocks_before_model_start(make_run,monkeypatch):
    from werewolf.tom_llm.live_runner import Journal
    from werewolf.tom_llm import service_manager as sm
    run = make_run()
    from werewolf.tom_llm.model_spec import wire
    Journal(run).append('dispatch',{'wire_digest':digest(wire(run.tasks[0],run.spec))})
    run.record['mode'] = 'formal'
    monkeypatch.setattr(ops.sys,'platform','linux')
    monkeypatch.setattr(ops,'open_run',lambda *a,**kw:run)
    monkeypatch.setattr(sm,'Service',lambda *a:pytest.fail('unknown request must block before loading'))
    args = SimpleNamespace(execute_live=True,model=run.key,run_id=run.record['run_id'],resume=True)
    with pytest.raises(ValueError,match='unknown'):
        ops.execute(config_for(run),args)


def test_status_reads_no_cohort_source_or_model(make_run,monkeypatch):
    from werewolf.tom_llm import multimodel as mm,physical
    run = make_run()
    monkeypatch.setattr(ex,'read_cohort',lambda *a:pytest.fail('Cohort in status'))
    monkeypatch.setattr(ex,'source_identity',lambda:pytest.fail('Source scan in status'))
    monkeypatch.setattr(physical,'verify_model',lambda *a:pytest.fail('weights in status'))
    args = SimpleNamespace(config='unused',operation='status',run_id=run.record['run_id'])
    monkeypatch.setattr(ops,'load_config',lambda _:config_for(run))
    result = ops.dispatch(args)
    assert result['completed_requests']==0 and result['request_count']==len(run.tasks)


def test_old_execution_artifact_never_upgraded(tmp_path):
    from werewolf.artifact_io import publish_artifact
    root = tmp_path/'historic'
    artifact = publish_artifact(root/'contract',manifest_fields={'artifact_type':'mm_baseline_run',
        'schema_version':'classic7_multimodel_experiment_v3'},files={})
    before = {str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError):
        ex.open_run(root)
    assert before=={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_mock_cannot_reach_service_constructor(make_run,monkeypatch):
    from werewolf.tom_llm import service_manager as sm
    run = make_run()
    monkeypatch.setattr(ops.sys,'platform','linux')
    monkeypatch.setattr(ops,'open_run',lambda *a,**kw:run)
    monkeypatch.setattr(sm,'Service',lambda *a:pytest.fail('Mock must not load a real model'))
    args = SimpleNamespace(execute_live=True,model=run.key,run_id=run.record['run_id'],resume=False)
    with pytest.raises(ValueError,match='CPU/Mock'):
        ops.execute(config_for(run),args)
