"""Synthetic/tiny evaluation only; no real publication or held-out results."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from tests.tom.test_paper_study_execution import prepared, synthetic_contract, rewritten
from tests.tom.test_paper_tom_metrics import prediction
from werewolf.artifact_io import canonical_json_bytes, publish_artifact, ArtifactConflictError
from werewolf.tom import paper_study_execution as execution
from werewolf.tom import paper_study_evaluation as m
from werewolf.tom.protocol import load_bootstrap_plan
from werewolf.tom.scoring import paired_summary


def rows():
    full = [prediction(game='g', observer=i, support=(2,)) for i in (0,1)]
    agnostic = deepcopy(full)
    for row in agnostic: row['checkpoint_digest'] = 'agnostic-checkpoint'
    masks = {m._identity(row): True for row in full}
    return full, agnostic, masks


def test_exact_rows_pair_without_probability_or_checkpoint_equality():
    full, agnostic, masks = rows()
    agnostic[0]['probability'][1] += .05
    agnostic[0]['probability'][2] -= .05
    agnostic[0]['non_self_log_probability'] = np.log(agnostic[0]['probability'][1:]).tolist()
    assert len(m._pair_rows(full, agnostic, masks, masks, fold=0, condition='implicit', game_ids=['g'])) == 64
    assert m._pair_rows(full, agnostic[::-1], masks, masks, fold=0, condition='implicit', game_ids=['g']) == m._pair_rows(
        full, agnostic, masks, masks, fold=0, condition='implicit', game_ids=['g'])


@pytest.mark.parametrize('change', ['missing','extra','duplicate','q','label','alive','PRE','plan',
                                   'observer','eligibility','temporal','fold'])
def test_pairing_fail_closed(change):
    full, agnostic, masks = rows()
    other_masks = dict(masks); fold = 0
    if change == 'missing': agnostic.pop()
    elif change == 'extra':
        extra = deepcopy(agnostic[0]); extra['boundary_id'] = 'extra'; agnostic.append(extra)
    elif change == 'duplicate': agnostic.append(deepcopy(agnostic[0]))
    elif change == 'q': agnostic[0]['q'] = [0,1,0,0,0,0,0]
    elif change == 'label': agnostic[0]['label_observed'] = False
    elif change == 'alive': agnostic[0]['observer_alive'] = False
    elif change == 'PRE': agnostic[0]['prefix_digest'] = 'different-prefix'
    elif change == 'plan': agnostic[0]['plan_digest'] = 'different-plan'
    elif change == 'observer': agnostic[0]['observer'] = 6
    elif change == 'eligibility': other_masks[m._identity(agnostic[0])] = False
    elif change == 'temporal': agnostic[0]['temporal_condition'] = 'explicit_day_phase'
    else: fold = 5
    with pytest.raises(ValueError):
        m._pair_rows(full, agnostic, masks, other_masks, fold=fold, condition='implicit', game_ids=['g'])


def test_fixed_effect_directions_and_formal_paired_ci():
    games = ['a','b','c']
    full = {g:dict(kl=v, uniform_kl=1.) for g,v in zip(games,[.2,.4,.8])}
    agnostic = {g:dict(kl=v, uniform_kl=1.) for g,v in zip(games,[.5,.6,.7])}
    indices = np.array([[0,0,1],[1,2,2],[2,1,0]], dtype='<i4')
    result = m._comparison(full, agnostic, games, indices, .95)
    assert result['Delta_ToM'] == paired_summary([.5,.6,.7],[.2,.4,.8],indices,.95)
    assert result['Delta_ToM']['point'] == pytest.approx((.3+.2-.1)/3)
    for family, scores in zip(m.FAMILIES, ([.2,.4,.8],[.5,.6,.7])):
        assert result['Delta_uniform'][family] == paired_summary([1.,1.,1.],scores,indices,.95)
        assert result['Delta_uniform'][family]['point'] > 0
    with pytest.raises(ValueError, match='coverage'):
        m._comparison(full, {'a':agnostic['a']}, games, indices,.95)


def test_support_strata_counts_and_game_macro():
    selected = [prediction(game='a',support=(1,)), prediction(game='a',support=(1,)),
        prediction(game='b',support=(1,)), prediction(game='c',support=(1,2)),
        prediction(game='d',support=(1,2,3)), prediction(game='e',support=tuple(range(1,7)))]
    summary = m._support_summary(selected)
    assert [summary[s]['row_count'] for s in m.STRATA] == [3,1,1,1]
    assert [summary[s]['game_count'] for s in m.STRATA] == [2,1,1,1]
    for label, size in zip(m.STRATA,(1,2,3,6)):
        assert summary[label]['kl'] == pytest.approx(np.log(6/size))
        assert summary[label]['uniform_kl'] == pytest.approx(np.log(6/size))
        assert set(summary[label]) == {'kl','uniform_kl','row_count','game_count'}
    assert m._support_summary([])['1'] == dict(kl=None,uniform_kl=None,row_count=0,game_count=0)


@pytest.fixture(scope='module')
def evaluated(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(execution, 'open_contract', synthetic_contract)
        study = prepared(tmp_path_factory.mktemp('paper-evaluation'))
        for lineage in execution.EXPECTED_LINEAGES:
            execution._train_lineage(study, *lineage)
        execution.seal_study(study.artifact.path)
        from werewolf.tom import protocol
        patch.setattr(protocol, 'bootstrap_indices', lambda *a: pytest.fail('must not regenerate draws'))
        paired = set()
        pair, score = m._pair_rows, m.evaluation.score_prediction_rows
        def tracked_pair(*args, **kwargs):
            result = pair(*args, **kwargs)
            paired.add((kwargs['condition'], kwargs['fold']))
            return result
        def gated_score(*args, **kwargs):
            assert paired == {(c, f) for c in m.TEMPORAL_CONDITIONS for f in range(5)}
            return score(*args, **kwargs)
        patch.setattr(m, '_pair_rows', tracked_pair)
        patch.setattr(m.evaluation, 'score_prediction_rows', gated_score)
        artifact = m.evaluate_study(study.artifact.path)
        return study, artifact


@pytest.fixture(autouse=True)
def synthetic_source(monkeypatch):
    monkeypatch.setattr(execution, 'open_contract', synthetic_contract)


@pytest.mark.parametrize('invalid', [False, True])
def test_preseal_fails_before_prediction_inputs(evaluated, monkeypatch, invalid):
    study, artifact = evaluated
    seal = study.seal_path.read_bytes(); study.seal_path.unlink()
    if invalid:
        study.seal_path.write_bytes(b'corrupt')
    monkeypatch.setattr(m.evaluation, 'open_predictions', lambda *a: pytest.fail('held-out access before gate'))
    monkeypatch.setattr(m.evaluation, '_held_out_data', lambda *a: pytest.fail('held-out access before gate'))
    read = Path.read_bytes
    def guarded(path):
        assert not any(part.startswith('held_out_') for part in path.parts)
        assert not ('public' in path.parts and 'games' in path.parts)
        return read(path)
    monkeypatch.setattr(Path, 'read_bytes', guarded)
    try:
        with pytest.raises(ValueError):
            m.evaluate_study(study.artifact.path)
    finally:
        study.seal_path.write_bytes(seal)


def test_four_cells_references_effects_and_no_extra_metrics(evaluated):
    study, artifact = evaluated
    report = artifact.manifest
    assert len(report['cells']) == 4 and len(report['predictions']) == 10
    assert set(report['Delta_ToM']) == set(m.TEMPORAL_CONDITIONS)
    assert len(report['Delta_uniform']) == 4
    formal = m.read_record(study.full.runs_path/'reports/oof_report_set.json')
    assert report['full_report_set_digest'] == formal['record_digest']
    assert report['Delta_temp']['record_digest'] == report['Delta_stress']['record_digest'] == formal['record_digest']
    indices, digest = load_bootstrap_plan(study.full)
    assert digest == report['bootstrap']['indices_digest']
    assert report['bootstrap']['replicates'] == 10000
    assert report['bootstrap']['game_ids'] == formal['game_ids']
    assert indices.shape == (10000,5)
    for condition in m.TEMPORAL_CONDITIONS:
        a, b = [report['cells'][f'{family}/{condition}'] for family in m.FAMILIES]
        owner = formal['cells'][f'{condition}/{m.PRIMARY}']
        assert a['kl'] == {'record_digest':owner['record_digest'],'field':'headline'}
        assert a['uniform_kl']['field'] == 'uniform_non_self_reference'
        assert a['row_count'] == b['row_count'] and a['game_count'] == b['game_count'] == 5
        assert a['total_variation']['point'] == owner['secondary_game_macro_diagnostics']['total_variation']
        assert set(a) == set(b) == {'kl','uniform_kl','total_variation','row_count','game_count','support_size'}
    text = canonical_json_bytes(report).decode()
    for prohibited in ('cross_entropy','mean_absolute_error','soft_brier','top1','mrr','normalized_gap',
                       'interaction','winner','significance'):
        assert prohibited not in text
    assert not study.agnostic.runs_path.joinpath('reports').exists()
    assert not list(study.agnostic.runs_path.rglob('all_alive_identifiability_stress'))


def test_revalidate_without_training_or_baseline_forward_and_immutable(evaluated, monkeypatch):
    study, artifact = evaluated
    before = study.seal_path.read_bytes()
    monkeypatch.setattr(m, '_predict_agnostic', lambda *a: pytest.fail('existing prediction regenerated'))
    monkeypatch.setattr(m.training, '_train_worker', lambda *a: pytest.fail('evaluation trained'))
    assert m.evaluate_study(study.artifact.path).manifest_digest == artifact.manifest_digest
    assert study.seal_path.read_bytes() == before
    with pytest.raises(ArtifactConflictError):
        publish_artifact(artifact.path, manifest_fields={
            'artifact_type':'paper_study_evaluation', 'schema_version':m.EVALUATION_VERSION}, files={})


@pytest.mark.parametrize('change',['parent','fold','bootstrap','metric','selection'])
def test_rehashed_evaluation_binding_corruption(evaluated, change):
    study, artifact = evaluated
    def corrupt(value):
        if change == 'parent': value['study_seal_digest'] = '0'*64
        elif change == 'fold': value['predictions']['implicit/0']['agnostic']['fold'] = 1
        elif change == 'bootstrap': value['bootstrap']['indices_digest'] = '0'*64
        elif change == 'selection': value['predictions']['implicit/0']['agnostic']['primary_row_identity_digest'] = '0'*64
        else: value['Delta_ToM']['implicit']['point'] += 1
    with rewritten(artifact.path/'manifest.json', corrupt, 'manifest_digest'):
        with pytest.raises(ValueError, match='binding mismatch'):
            m.evaluate_study(study.artifact.path)


def test_bootstrap_bytes_corruption_rejected(evaluated):
    study, artifact = evaluated
    path = study.full.path/'bootstrap/game_cluster_bootstrap_indices.bin'
    before = path.read_bytes(); path.write_bytes(before[:-4]+b'\xff'*4)
    try:
        with pytest.raises(ValueError): m.evaluate_study(study.artifact.path)
    finally:
        path.write_bytes(before)


@pytest.mark.parametrize('kind',['prediction','checkpoint'])
def test_corrupt_payload_rejected(evaluated, kind, monkeypatch):
    study, artifact = evaluated
    path = (artifact.path/'agnostic/implicit/0/held_out_predictions.jsonl' if kind=='prediction'
            else study.agnostic.runs_path/'explicit_day_phase/4/terminal_checkpoint/tensors.bin')
    original = path.read_bytes(); path.write_bytes(b'corrupt')
    if kind == 'checkpoint':
        monkeypatch.setattr(m.evaluation, 'open_predictions', lambda *a: pytest.fail('early held-out access'))
    try:
        with pytest.raises(ValueError): m.evaluate_study(study.artifact.path)
    finally:
        path.write_bytes(original)


def test_cli_only_evaluation_dispatch(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    from werewolf.cli import main, build_parser
    profile=tmp_path/'profile.json'; profile.write_text(json.dumps({'artifact_root':str(tmp_path)}))
    calls=[]
    def evaluate(path):
        calls.append(path)
        return SimpleNamespace(manifest_digest='a'*64)
    monkeypatch.setattr(m, 'evaluate_study', evaluate)
    assert main(['--storage-profile',str(profile),'evaluate-paper-tom-study','--study','study/execution']) == 0
    assert calls == [tmp_path/'study/execution']
    assert capsys.readouterr().out.strip() == 'a'*64
    options = {o for a in build_parser()._subparsers._group_actions[0].choices['evaluate-paper-tom-study']._actions
               for o in a.option_strings}
    assert options == {'-h','--help','--study'}
