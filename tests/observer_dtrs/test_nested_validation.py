"""Bounded end-to-end engineering acceptance on explicitly synthetic games."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from werewolf.artifact_io import ArtifactValidationError
from werewolf.observer_dtrs.d2_protocol import frozen_protocol
from werewolf.observer_dtrs.validation_data import controlled_fixture
from werewolf.observer_dtrs.nested_validation import execute, candidates
from werewolf.observer_dtrs import nested_validation as runner
from werewolf.observer_dtrs.splits import partition_games


@pytest.fixture
def data():
    return controlled_fixture(frozen_protocol())


def test_outer_targets_open_only_after_both_weighting_predictions_and_do_not_change_selection(tmp_path,data):
    root=tmp_path/'first'
    train,test=partition_games(data.split,tom_fold=0,outer_fold=0)
    original=data.targets
    accessed=[]
    def guarded(games):
        accessed.append(tuple(games))
        if set(games)==set(test):
            assert (root/'blocks/0-0/game_equal/outer_predictions/manifest.json').exists()
            assert (root/'blocks/0-0/record_equal/outer_predictions/manifest.json').exists()
        else:
            assert set(games)<=set(train)
        return original(games)
    data.targets=guarded
    result=execute(root,data=data,block=(0,0))
    assert result['status']=='PARTIAL_NOT_FULL_EXPERIMENT'
    assert not (root/'complete').exists()
    first=json.loads((root/'blocks/0-0/game_equal/outer_predictions/record.json').read_bytes())
    for row in data.legal(test):
        data.target_values[row.sample_id]={'y':1-data.target_values[row.sample_id]['y'],'successful_empty':False}
    data.targets=original
    other=tmp_path/'counterfactual'
    execute(other,data=data,block=(0,0))
    assert first==json.loads((other/'blocks/0-0/game_equal/outer_predictions/record.json').read_bytes())
    assert (root/'blocks/0-0/game_equal/selection/record.json').read_bytes()==(other/'blocks/0-0/game_equal/selection/record.json').read_bytes()


def test_interruption_persists_failure_and_resume_does_not_refit_completed_candidate(tmp_path,data,monkeypatch):
    root=tmp_path/'resume'
    counter=0
    def interrupt(stage,path):
        nonlocal counter
        counter+=1
        if counter==3:
            raise KeyboardInterrupt('controlled interruption')
    with pytest.raises(KeyboardInterrupt):
        execute(root,data=data,block=(0,0),hook=interrupt)
    assert (root/'failures/000000/manifest.json').exists()
    assert not (root/'blocks/0-0/complete').exists()
    original=runner._fit
    fitted=[]
    def traced(*args,**kwargs):
        fitted.append((args[1],args[2]))
        return original(*args,**kwargs)
    monkeypatch.setattr(runner,'_fit',traced)
    execute(root,data=data,block=(0,0))
    assert len(fitted)==2*(3*16+3)-3
    fitted.clear()
    again=execute(root,data=data,block=(0,0))
    assert fitted==[]
    assert again['status']=='PARTIAL_NOT_FULL_EXPERIMENT'
    pred=root/'blocks/0-0/game_equal/outer_predictions/record.json'
    pred.write_bytes(b'truncated')
    with pytest.raises(ArtifactValidationError):
        execute(root,data=data,block=(0,0))
    assert pred.read_bytes()==b'truncated'  # stop, no repair/overwrite


def test_full_controlled_25_blocks_75_inner_and_2000_bootstrap(tmp_path,data,monkeypatch):
    root=tmp_path/'controlled-full'
    result=execute(root,data=data)
    assert result['scope']=='controlled_fixture' and result['status']=='ALL_BLOCKS_COMPLETE'
    assert result['games']==75 and result['rows']==150
    record=json.loads((root/'complete/record.json').read_bytes())
    assert len(record['block_digests'])==25
    assert record['summary']['bootstrap']['replicates']==2000
    assert len(list((root/'blocks').glob('*/*/inner/*/*/manifest.json')))==25*2*3*16
    assert len(list((root/'blocks').glob('*/*/selection/manifest.json')))==50
    # Every frozen candidate has exactly the same 12 held-out inner games.
    for path in (root/'blocks').glob('*/*/selection/record.json'):
        selection=json.loads(path.read_bytes())
        assert len(selection['candidate_game_brier'])==16
        population=[set(v['games']) for v in selection['candidate_game_brier']]
        assert all(v==population[0] for v in population) and len(population[0])==12
    def forbid(*args,**kwargs):
        raise AssertionError('completed run must not fit again')
    monkeypatch.setattr(runner,'_fit',forbid)
    replay=execute(root,data=data)
    assert replay==result
    from werewolf.observer_dtrs.nested_validation import verify_result
    audited=verify_result(root)
    assert audited['manifest_digest']==result['manifest_digest']
    (root/'complete/bootstrap_indices.int64le').write_bytes(b'corrupted')
    with pytest.raises(ArtifactValidationError):
        verify_result(root)


def test_readonly_partial_verify(tmp_path,data):
    from werewolf.observer_dtrs.nested_validation import verify_result
    execute(tmp_path/'partial',data=data,block=(0,0))
    verified=verify_result(tmp_path/'partial')
    assert verified=={'scope':'controlled_fixture','status':'PARTIAL_NOT_FULL_EXPERIMENT','verified_blocks':1}


@pytest.fixture(params=('a'*40,'b'*40))
def source_guard_fixture(tmp_path,monkeypatch,request):
    """Controlled Git states over a tiny source tree, never the real repository."""
    from werewolf.observer_dtrs import validation_artifacts
    root=tmp_path/'source-fixture'
    files={'werewolf/controlled.py':'VALUE = 1\n',
           'scripts/run_observer_dtrs_validation.py':'# controlled entry point\n',
           'configs/observer_dtrs/d2-execution-v1.json':frozen_protocol().payload.decode()}
    for name,text in files.items():
        path=root/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(text)
    state=SimpleNamespace(head=request.param,tracked=set(files),dirty='',calls=[],files=files)
    def controlled_git(command,**kwargs):
        assert kwargs=={'cwd':root,'capture_output':True,'text':True,'check':True}
        state.calls.append(command)
        if command==['git','rev-parse','HEAD']:
            output=state.head+'\n'
        elif command==['git','ls-files']:
            output='\n'.join(sorted(state.tracked))+'\n'
        elif command==['git','diff','HEAD','--','werewolf',
                       'scripts/run_observer_dtrs_validation.py','configs/observer_dtrs/d2-execution-v1.json']:
            output=state.dirty
        else:
            pytest.fail(f'uncontrolled subprocess: {command}')
        return SimpleNamespace(stdout=output)
    monkeypatch.setattr(validation_artifacts,'__file__',str(root/'werewolf/observer_dtrs/validation_artifacts.py'))
    monkeypatch.setattr(validation_artifacts.subprocess,'run',controlled_git)
    state.snapshot=validation_artifacts.source_snapshot
    return state


def test_source_guard_rejects_mismatched_oid_before_source_state_checks(source_guard_fixture):
    state=source_guard_fixture
    with pytest.raises(ValueError,match='explicitly pinned current source commit'):
        state.snapshot(formal=True,expected_revision='0'*40)
    assert state.calls==[['git','rev-parse','HEAD']]


@pytest.mark.parametrize('condition',('untracked','modified'))
def test_source_guard_rejects_uncommitted_or_modified_production(source_guard_fixture,condition):
    state=source_guard_fixture
    if condition=='untracked':
        state.tracked.remove('scripts/run_observer_dtrs_validation.py')
    else:
        state.dirty='controlled production diff\n'
    with pytest.raises(ValueError,match='committed, unchanged production sources'):
        state.snapshot(formal=True,expected_revision=state.head)
    assert [command[:2] for command in state.calls]==[['git','rev-parse'],['git','ls-files'],['git','diff']]


def test_source_guard_accepts_matching_committed_clean_production(source_guard_fixture):
    state=source_guard_fixture
    snapshot=state.snapshot(formal=True,expected_revision=state.head)
    assert snapshot['head']==state.head
    assert snapshot['files']==state.files
    assert [command[:2] for command in state.calls]==[['git','rev-parse'],['git','ls-files'],['git','diff']]


def test_duplicate_prediction_rehashed_checkpoint_stops_without_overwrite(tmp_path,data):
    from werewolf.artifact_io import canonical_json_bytes, publish_artifact
    root=tmp_path/'rehashed'
    execute(root,data=data,block=(0,0))
    # Replace only our controlled test artifact using a new path; production never overwrites.
    path=root/'blocks/0-0/game_equal/outer_predictions'
    manifest=json.loads((path/'manifest.json').read_bytes())
    value=json.loads((path/'record.json').read_bytes())
    value['predictions']['OBS-EQ0'].append(deepcopy(value['predictions']['OBS-EQ0'][0]))
    wrong=publish_artifact(tmp_path/'different-controlled-unit',manifest_fields={
        k:v for k,v in manifest.items() if k not in ('file_table','manifest_digest')},
        files={'record.json':canonical_json_bytes(value)})
    import shutil
    shutil.rmtree(path)
    shutil.copytree(wrong.path,path)
    with pytest.raises(ValueError,match='predictions'):
        execute(root,data=data,block=(0,0))
    assert len(json.loads((path/'record.json').read_bytes())['predictions']['OBS-EQ0'])==len(value['predictions']['OBS-EQ0'])
