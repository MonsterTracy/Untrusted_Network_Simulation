"""Synthetic schema fixtures ONLY; no experiment result or model execution."""
from copy import deepcopy
import math
from pathlib import Path

import pytest

from scripts import paper_tom_metrics as m


def prediction(game='g', observer=0, support=(1,), p=None):
    q = [0.] * 7
    for j in support: q[j] = 1 / len(support)
    if p is None:
        p = [1 / 6] * 7
        p[observer] = 0
    return dict(game_id=game, boundary_id=game+'-b', observer=observer,
        prefix_digest='prefix', plan_digest='plan', probability=p,
        non_self_log_probability=[math.log(p[j]) for j in range(7) if j != observer],
        q=q, label_observed=True, observer_alive=True,
        checkpoint_digest='checkpoint', temporal_condition='implicit')


def score(row):
    # Fixture values are schema witnesses, not real scientific results.
    return {**{k: row[k] for k in ('game_id', 'boundary_id', 'observer', 'prefix_digest')},
        'kl': .2, 'uniform_kl': .3, 'cross_entropy': .4, 'total_variation': .1,
        'mean_absolute_error': .02, 'argmax_in_target_support': True,
        'target_support_probability': 1. if sum(x > 0 for x in row['q']) == 6 else 1/6,
        'normalized_reducible_gap_improvement': None if sum(x > 0 for x in row['q']) == 6 else .3}


def fold_rows(rows):
    scores = [score(r) for r in rows]
    return {'row_scores': scores,
        'row_identity_digest': m.digest(m.canonical([list(m.identity(r)) for r in rows])),
        'game_scores': {g: {'row_count': sum(r['game_id'] == g for r in rows)} for g in {r['game_id'] for r in rows}}}


def test_exact_join_without_population_and_trivial_support():
    row = prediction(support=tuple(range(1, 7)))
    assert 'population' not in row
    result = m.join_rows(fold_rows([row]), {m.identity(row): row})[0]
    assert result['support_size'] == 6
    assert result['top1_nu'] is result['mrr_nu'] is result['target_support_probability_nu'] is None
    assert sum(row['probability'][j] for j in range(1, 7)) == pytest.approx(1)
    assert score(row)['argmax_in_target_support'] is True
    assert score(row)['target_support_probability'] == 1


def test_top_support_mrr_brier_and_tie():
    row = prediction(support=(2,), p=[0, .3, .3, .1, .1, .1, .1])
    result = m.row_diagnostics(row)
    assert result['top1_nu'] == 0  # seat1 precedes seat2 at the exact tie
    assert result['mrr_nu'] == .5
    assert result['target_support_probability_nu'] == .3
    assert result['soft_brier_sum'] == pytest.approx(.3**2 + .7**2 + 4*.1**2)


def test_game_macro_not_row_micro_and_no_zero_filling():
    rows = [{'game_id': 'a', 'v': 1}, {'game_id': 'a', 'v': 1},
            {'game_id': 'b', 'v': 0}, {'game_id': 'c', 'v': None}]
    result = m.macro(rows, 'v')
    assert result['point'] == .5 and result['point'] != pytest.approx(2/3)
    assert result['valid_row_count'] == 3 and result['valid_game_count'] == 2
    assert m.macro([], 'v')['point'] is None


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'prefix', 'identity_digest'])
def test_bad_join(change):
    row = prediction()
    report = fold_rows([row])
    predictions = {m.identity(row): row}
    if change == 'missing': predictions.clear()
    elif change == 'duplicate': report = fold_rows([row, row])
    elif change == 'prefix': report['row_scores'][0]['prefix_digest'] = 'wrong'
    else: report['row_identity_digest'] = 'wrong'
    with pytest.raises(ValueError): m.join_rows(report, predictions)


@pytest.mark.parametrize('change', ['shape', 'negative', 'nan', 'self', 'q_empty', 'q_self', 'q_nonuniform', 'observer'])
def test_malformed_predictions(change):
    row = prediction()
    if change == 'shape': row['probability'].pop()
    elif change == 'negative': row['probability'][1] = -.1
    elif change == 'nan': row['probability'][1] = float('nan')
    elif change == 'self': row['probability'][0] = .1
    elif change == 'q_empty': row['q'] = [0]*7
    elif change == 'q_self': row['q'] = [1,0,0,0,0,0,0]
    elif change == 'q_nonuniform': row['q'] = [0,.3,.7,0,0,0,0]
    else: row['observer'] = True
    with pytest.raises(ValueError): m.row_diagnostics(row)


def write_record(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {**record, 'record_digest': m.digest(m.canonical(record))}
    path.write_bytes(m.canonical(value))
    return value


def interval():
    return dict(point=.2, interval=[.1,.3], aggregation='game_macro', sampling_unit='game',
                interval_method='percentile_linear', confidence=.95)


def fixture(tmp_path):
    root = tmp_path / 'runs' / ('a'*64)
    base = dict(experiment_digest=root.name, checkpoint_set_digest='seal')
    cells = {}
    for condition in m.CONDITIONS:
        reports = {p: [] for p in m.POPULATIONS}
        for f in range(5):
            game = f'g{f}'
            rows = []
            for observer in range(7):
                seats = [j for j in range(7) if j != observer]
                size = (1,2,3,6,6)[f]
                row = prediction(game, observer, tuple(seats[:size]))
                row['temporal_condition'] = condition
                rows.append(row)
            payload = b''.join(m.canonical(r)+b'\n' for r in rows)
            stem = root / condition / str(f)
            stem.mkdir(parents=True)
            (stem / 'held_out_predictions.jsonl').write_bytes(payload)
            manifest = write_record(stem / 'prediction_manifest.json', dict(**base,
                artifact_type='held_out_predictions', schema_version=m.SCHEMAS['prediction'], fold=f,
                temporal_condition=condition, checkpoint_digest='checkpoint', prediction_digest=m.digest(payload),
                row_count=7, canonical_shift=0))
            for population in m.POPULATIONS:
                # Different membership despite every prediction having identical flags.
                selected = rows[:2] if population == m.POPULATIONS[0] else rows
                report = fold_rows(selected)
                report.update({**base, 'schema_version': m.SCHEMAS['fold'],
                    'artifact_type': population+'_fold', 'fold': f, 'temporal_condition': condition,
                    'checkpoint_digest': 'checkpoint', 'prediction_digest': manifest['prediction_digest'],
                    'population_identity': 'non_wolf_alive' if population == m.POPULATIONS[0] else 'all_alive',
                    'population_selector_version': 'classic7_primary_population_v1' if population == m.POPULATIONS[0] else 'classic7_all_alive_eligibility_v1'})
                if population == m.POPULATIONS[0]: report['role_sidecar_digest'] = 'role-digest'
                report['game_scores'][game].update({key: .2 for key in ('kl','uniform_kl','cross_entropy','total_variation','mean_absolute_error','argmax_in_target_support','target_support_probability')})
                reports[population].append(write_record(stem / population / 'fold_report.json', report))
        for population in m.POPULATIONS:
            cell = {**base, 'artifact_type': population+'_aggregate', 'schema_version': m.SCHEMAS['aggregate'],
                'temporal_condition': condition, 'bootstrap_digest': 'bootstrap',
                'fold_report_digests': [r['record_digest'] for r in reports[population]],
                'game_scores': {g: v for r in reports[population] for g,v in r['game_scores'].items()},
                'headline': interval(), 'uniform_non_self_reference': interval(),
                'secondary_game_macro_diagnostics': {k: .2 for k in ('cross_entropy','total_variation','mean_absolute_error','argmax_in_target_support','target_support_probability')},
                'observer_row_weighted_diagnostic': dict(kl=.2,row_count=10 if population == m.POPULATIONS[0] else 35,
                    normalized_reducible_gap_improvement_nonzero_gap_only=.3, nonzero_gap_row_count=6 if population == m.POPULATIONS[0] else 21)}
            cells[condition+'/'+population] = write_record(root / 'reports' / condition / population / 'aggregate_report.json', cell)
    write_record(root / 'reports/oof_report_set.json', {**base, 'artifact_type':'development_oof_report_set',
        'schema_version':m.SCHEMAS['report_set'], 'bootstrap_digest':'bootstrap', 'game_ids':[f'g{i}' for i in range(5)],
        'cells':cells, 'paired_headlines':dict(primary_temporal_information_effect=interval(),
            identifiability_stress_penalty={c:interval() for c in m.CONDITIONS})})
    return root


def test_complete_synthetic_analysis_readonly_and_strata(tmp_path):
    root = fixture(tmp_path)
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    result = m.analyze(root, tmp_path / 'paper')
    assert {p: p.read_bytes() for p in root.rglob('*') if p.is_file()} == before
    key = 'implicit/primary_development_oof'
    assert result['derived_paper_diagnostics'][key]['soft_brier_sum']['valid_row_count'] == 10
    assert result['derived_paper_diagnostics'][key]['top1_nu']['valid_game_count'] == 3
    strata = result['support_size_diagnostics'][key]
    assert [strata[s]['game_count'] for s in m.STRATA] == [1,1,1,2]
    assert strata['6']['mrr_nu']['point'] is None
    assert result['formal_metrics'][key]['headline'] == interval()
    assert set(p.name for p in (tmp_path/'paper').iterdir()) == {'summary.json','README.txt','main_metrics.csv','support_size_metrics.csv'}


def test_output_safety_and_final_rejection(tmp_path, monkeypatch):
    root = fixture(tmp_path)
    existing = tmp_path / 'existing'; existing.mkdir()
    for out in (existing, root / 'paper'):
        with pytest.raises(ValueError): m.analyze(root, out)
    (root / 'final_evaluation').mkdir()
    monkeypatch.setattr(m.Inputs, 'read', lambda *a, **k: pytest.fail('must not read Final/input'))
    with pytest.raises(ValueError, match='Final'): m.analyze(root, tmp_path/'paper')


@pytest.mark.parametrize('change', ['digest', 'duplicate'])
def test_prediction_file_failure(tmp_path, change):
    root = fixture(tmp_path)
    path = root/'implicit/0/held_out_predictions.jsonl'
    if change == 'digest': path.write_bytes(path.read_bytes()+b'\n')
    else:
        rows = [m.strict_json(line) for line in path.read_bytes().splitlines()]
        rows[1] = rows[0]
        payload = b''.join(m.canonical(r)+b'\n' for r in rows)
        path.write_bytes(payload)
        manifest_path = path.parent/'prediction_manifest.json'
        manifest = m.strict_json(manifest_path.read_bytes()); del manifest['record_digest']
        manifest['prediction_digest'] = m.digest(payload)
        write_record(manifest_path, manifest)
    with pytest.raises(ValueError): m.analyze(root, tmp_path/'paper')
    assert not (tmp_path/'paper').exists()
