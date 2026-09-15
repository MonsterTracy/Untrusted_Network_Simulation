"""Read-only Development OOF analysis. DERIVED PAPER DIAGNOSTICS only."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

HISTORICAL_REVISION = '33d580cdf0700f6e38ae5b14648fcfdaa9be2b9a'
CONDITIONS = ('implicit', 'explicit_day_phase')
POPULATIONS = ('primary_development_oof', 'all_alive_identifiability_stress')
METRICS = ('top1_nu', 'target_support_probability_nu', 'soft_brier_sum', 'mrr_nu')
STRATA = ('1', '2', '3-5', '6')
FORMAL = ('headline', 'uniform_non_self_reference', 'secondary_game_macro_diagnostics',
          'observer_row_weighted_diagnostic', 'game_scores')
SCHEMAS = {'prediction': 'classic7_predictions_v1', 'fold': 'classic7_named_fold_report_v1',
           'aggregate': 'classic7_oof_cell_v1', 'report_set': 'classic7_oof_report_set_v1'}
PRED_FIELDS = set('game_id boundary_id observer prefix_digest plan_digest probability non_self_log_probability q label_observed observer_alive checkpoint_digest temporal_condition'.split())
ROW_FIELDS = set('game_id boundary_id observer prefix_digest kl uniform_kl cross_entropy total_variation mean_absolute_error argmax_in_target_support target_support_probability normalized_reducible_gap_improvement'.split())


def canonical(value):
    # Exact historical artifact_io canonical JSON convention, without importing runtime.
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def strict_json(data):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, 'duplicate JSON key')
            value[key] = item
        return value
    value = json.loads(data, object_pairs_hook=pairs,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    require(canonical(value) == data, 'noncanonical JSON bytes')
    return value


class Inputs:
    def __init__(self, root):
        self.root = root
        self.manifest = {}

    def read(self, relative, record=True):
        path = self.root / relative
        require(not any(p.is_symlink() for p in (path, *path.parents)), 'input symlink prohibited')
        data = path.read_bytes()
        self.manifest[relative] = {'sha256': digest(data)}
        if not record:
            return data
        value = strict_json(data)
        require(type(value) is dict and 'record_digest' in value, 'missing record digest')
        require(digest(canonical({k: v for k, v in value.items() if k != 'record_digest'})) == value['record_digest'],
                'record digest mismatch')
        self.manifest[relative]['record_digest'] = value['record_digest']
        if 'prediction_digest' in value:
            self.manifest[relative]['prediction_digest'] = value['prediction_digest']
        return value


def identity(row):
    require(type(row['observer']) is int and 0 <= row['observer'] < 7, 'invalid observer')
    require(all(type(row[k]) is str and row[k] for k in ('game_id', 'boundary_id')), 'invalid row identity')
    return row['game_id'], row['boundary_id'], row['observer']


def finite_vector(value, length):
    require(type(value) is list and len(value) == length, 'invalid vector shape')
    require(all(type(x) in (int, float) and math.isfinite(x) for x in value), 'nonfinite vector')
    return value


def validate_prediction(row, selected=False):
    require(set(row) == PRED_FIELDS, 'prediction schema mismatch (no population field allowed)')
    observer = identity(row)[2]
    p = finite_vector(row['probability'], 7)
    q = finite_vector(row['q'], 7)
    logs = finite_vector(row['non_self_log_probability'], 6)
    require(all(x >= 0 for x in p) and p[observer] == 0 and math.isclose(sum(p), 1, abs_tol=1e-6, rel_tol=1e-5), 'invalid probability simplex')
    nonself = [j for j in range(7) if j != observer]
    require(all(math.isclose(p[j], math.exp(log), abs_tol=1e-7, rel_tol=1e-5) for j, log in zip(nonself, logs)), 'p/log mismatch')
    require(type(row['label_observed']) is bool and type(row['observer_alive']) is bool, 'invalid observation flags')
    require(all(x >= 0 for x in q) and q[observer] == 0, 'invalid target')
    if selected:
        require(row['label_observed'] and row['observer_alive'], 'selected unsuccessful/dead row')
    if row['label_observed']:
        support = [j for j in nonself if q[j] > 0]
        require(bool(support) and math.isclose(sum(q), 1, abs_tol=1e-6), 'empty/nonunit successful target')
        require(all(math.isclose(q[j], 1 / len(support), abs_tol=1e-7) for j in support), 'target not uniform on support')
    else:
        require(all(x == 0 for x in q), 'unobserved target must be zero')


def row_diagnostics(row):
    validate_prediction(row, selected=True)
    observer, p, q = row['observer'], row['probability'], row['q']
    seats = [j for j in range(7) if j != observer]
    support = {j for j in seats if q[j] > 0}
    ranking = sorted(seats, key=lambda j: (-p[j], j))
    size = len(support)
    return {'game_id': row['game_id'], 'support_size': size,
        'top1_nu': float(ranking[0] in support) if size < 6 else None,
        'target_support_probability_nu': sum(p[j] for j in support) if size < 6 else None,
        'soft_brier_sum': sum((p[j] - q[j])**2 for j in seats),
        'mrr_nu': 1 / min(rank for rank, j in enumerate(ranking, 1) if j in support) if size < 6 else None}


def macro(rows, name):
    games = defaultdict(list)
    for row in rows:
        if row[name] is not None:
            games[row['game_id']].append(row[name])
    means = [sum(values) / len(values) for values in games.values()]
    return {'point': sum(means) / len(means) if means else None,
            'valid_row_count': sum(map(len, games.values())), 'valid_game_count': len(games),
            'aggregation': 'game_macro', 'sampling_unit': 'game'}


def join_rows(report, predictions):
    joined, seen = [], set()
    rows = report['row_scores']
    ids = [list(identity(row)) for row in rows]
    require(digest(canonical(ids)) == report['row_identity_digest'], 'row identity digest mismatch')
    for score in rows:
        require(set(score) == ROW_FIELDS, 'formal row score schema mismatch')
        key = identity(score)
        require(key not in seen and key in predictions, 'duplicate/missing selected prediction identity')
        seen.add(key)
        row = predictions[key]
        require(row['prefix_digest'] == score['prefix_digest'], 'row prefix mismatch')
        derived = row_diagnostics(row)
        for name in ('kl', 'total_variation'):
            require(type(score[name]) in (int, float) and math.isfinite(score[name]), 'invalid formal row score')
            derived[name] = score[name]  # Reuse formal scores; never recompute KL/TV.
        joined.append(derived)
    require({r['game_id'] for r in joined} == set(report['game_scores']), 'fold game/row coverage mismatch')
    for game, values in report['game_scores'].items():
        require(values['row_count'] == sum(r['game_id'] == game for r in joined), 'fold row count mismatch')
    return joined


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = f'{prefix}_{key}' if prefix else key
        if isinstance(item, dict): result.update(flatten(item, name))
        elif isinstance(item, list): result[name] = json.dumps(item)
        else: result[name] = item
    return result


def validate_summary(value):
    require(set(value) == {'point', 'interval', 'aggregation', 'sampling_unit',
                          'interval_method', 'confidence'}, 'formal summary schema mismatch')
    require(value['aggregation'] == 'game_macro' and value['sampling_unit'] == 'game'
            and value['interval_method'] == 'percentile_linear' and value['confidence'] == .95,
            'expected formal 95% game-cluster summary')
    finite_vector(value['interval'], 2)
    require(type(value['point']) in (int, float) and math.isfinite(value['point']), 'invalid formal point')


def analyze(runs_root, output_dir):
    raw_root, raw_out = Path(runs_root).absolute(), Path(output_dir).absolute()
    require(not raw_out.exists() and not raw_out.is_symlink(), 'output directory already exists')
    root, out = raw_root.resolve(), raw_out.resolve()
    require(out != root and root not in out.parents, 'output cannot be inside runs root')
    if root.parent.name == 'runs':
        require(root.parent != out and root.parent not in out.parents, 'output cannot be inside formal runs tree')
    require(not any(p.is_symlink() for p in (raw_root, *raw_root.parents)), 'input symlink prohibited')
    require(root.is_dir(), 'missing runs root')
    # Inspect names only, never open Final artifacts or discover alternative layouts.
    require(not any(p.name.startswith('final') for p in root.iterdir()), 'Final layout prohibited')
    require({p.name for p in root.iterdir()} >= {*CONDITIONS, 'reports'}, 'not Development OOF layout')
    inputs = Inputs(root)
    report_set = inputs.read('reports/oof_report_set.json')
    require(set(report_set) == set('artifact_type schema_version experiment_digest checkpoint_set_digest bootstrap_digest game_ids cells paired_headlines record_digest'.split()), 'report set schema mismatch')
    require(report_set['artifact_type'] == 'development_oof_report_set' and report_set['schema_version'] == SCHEMAS['report_set'], 'not Development OOF report set')
    games = report_set['game_ids']
    require(type(games) is list and games and games == sorted(set(games)), 'invalid OOF game coverage')
    require(root.name == report_set['experiment_digest'], 'runs root experiment identity mismatch')
    expected_cells = {f'{c}/{p}' for c in CONDITIONS for p in POPULATIONS}
    require(set(report_set['cells']) == expected_cells, 'four OOF cells required')
    paired = report_set['paired_headlines']
    require(set(paired) == {'primary_temporal_information_effect', 'identifiability_stress_penalty'} and set(paired['identifiability_stress_penalty']) == set(CONDITIONS), 'paired headline schema mismatch')
    validate_summary(paired['primary_temporal_information_effect'])
    for value in paired['identifiability_stress_penalty'].values():
        validate_summary(value)
    formal, derived, strata, main_csv, strata_csv = {}, {}, {}, [], []
    for condition in CONDITIONS:
        folds = []
        prediction_games = set()
        for fold in range(5):
            stem = f'{condition}/{fold}'
            manifest = inputs.read(f'{stem}/prediction_manifest.json')
            require(set(manifest) == set('artifact_type schema_version experiment_digest fold temporal_condition checkpoint_digest checkpoint_set_digest prediction_digest row_count canonical_shift record_digest'.split()), 'prediction manifest schema mismatch')
            require(manifest['artifact_type'] == 'held_out_predictions' and manifest['schema_version'] == SCHEMAS['prediction'] and manifest['canonical_shift'] == 0, 'prediction manifest type mismatch')
            require(manifest['fold'] == fold and manifest['temporal_condition'] == condition, 'prediction fold/condition mismatch')
            for key in ('experiment_digest', 'checkpoint_set_digest'):
                require(manifest[key] == report_set[key], 'prediction provenance mismatch')
            payload = inputs.read(f'{stem}/held_out_predictions.jsonl', record=False)
            require(digest(payload) == manifest['prediction_digest'], 'prediction digest mismatch')
            rows = [strict_json(line) for line in payload.splitlines()]
            require(payload == b''.join(canonical(r) + b'\n' for r in rows), 'noncanonical prediction JSONL')
            require(len(rows) == manifest['row_count'], 'prediction count mismatch')
            by_id = {}
            for row in rows:
                validate_prediction(row)
                key = identity(row)
                require(key not in by_id, 'duplicate prediction identity')
                require(row['temporal_condition'] == condition and row['checkpoint_digest'] == manifest['checkpoint_digest'], 'prediction row provenance mismatch')
                by_id[key] = row
            boundaries = defaultdict(list)
            for row in rows:
                boundaries[row['game_id'], row['boundary_id']].append(row)
            for batch in boundaries.values():
                require(sorted(r['observer'] for r in batch) == list(range(7)), 'prediction boundary needs seven observers')
                require(len({r['prefix_digest'] for r in batch}) == len({r['plan_digest'] for r in batch}) == 1,
                        'inconsistent prediction boundary identity')
            fg = {key[0] for key in by_id}
            require(not prediction_games & fg, 'duplicate game across folds')
            prediction_games.update(fg)
            folds.append((manifest, by_id))
        require(prediction_games == set(games), 'prediction game coverage mismatch')
        for population in POPULATIONS:
            cell_key = f'{condition}/{population}'
            aggregate = inputs.read(f'reports/{cell_key}/aggregate_report.json')
            require(set(aggregate) == set('artifact_type schema_version experiment_digest checkpoint_set_digest temporal_condition bootstrap_digest fold_report_digests record_digest'.split()) | set(FORMAL), 'aggregate schema mismatch')
            require(aggregate == report_set['cells'][cell_key], 'aggregate/report set cell mismatch')
            require(aggregate['artifact_type'] == population + '_aggregate' and aggregate['schema_version'] == SCHEMAS['aggregate'], 'aggregate type mismatch')
            require(aggregate['temporal_condition'] == condition, 'aggregate condition mismatch')
            for key in ('experiment_digest', 'checkpoint_set_digest', 'bootstrap_digest'):
                require(aggregate[key] == report_set[key], 'aggregate provenance mismatch')
            validate_summary(aggregate['headline'])
            validate_summary(aggregate['uniform_non_self_reference'])
            require(set(aggregate['secondary_game_macro_diagnostics']) == set('cross_entropy total_variation mean_absolute_error argmax_in_target_support target_support_probability'.split()), 'secondary metric schema mismatch')
            require(set(aggregate['observer_row_weighted_diagnostic']) == set('kl row_count normalized_reducible_gap_improvement_nonzero_gap_only nonzero_gap_row_count'.split()), 'row-weighted metric schema mismatch')
            all_rows, fold_digests, fold_games = [], [], {}
            for fold, (manifest, predictions) in enumerate(folds):
                report = inputs.read(f'{condition}/{fold}/{population}/fold_report.json')
                required = set('schema_version artifact_type population_identity population_selector_version experiment_digest fold temporal_condition checkpoint_digest prediction_digest checkpoint_set_digest row_identity_digest game_scores row_scores record_digest'.split())
                if population == POPULATIONS[0]: required.add('role_sidecar_digest')
                require(set(report) == required and report['schema_version'] == SCHEMAS['fold'], 'fold schema mismatch')
                require(report['artifact_type'] == population + '_fold', 'fold artifact type mismatch')
                primary = population == POPULATIONS[0]
                require(report['population_identity'] == ('non_wolf_alive' if primary else 'all_alive'), 'population mismatch')
                require(report['population_selector_version'] == ('classic7_primary_population_v1' if primary else 'classic7_all_alive_eligibility_v1'), 'population version mismatch')
                for key in ('experiment_digest', 'checkpoint_set_digest', 'checkpoint_digest', 'prediction_digest', 'fold', 'temporal_condition'):
                    require(report[key] == manifest[key], 'fold prediction binding mismatch')
                joined = join_rows(report, predictions)
                require(set(report['game_scores']) == {k[0] for k in predictions}, 'fold held-out coverage mismatch')
                all_rows.extend(joined)
                fold_digests.append(report['record_digest'])
                fold_games.update(report['game_scores'])
            require(fold_digests == aggregate['fold_report_digests'] and fold_games == aggregate['game_scores'] and set(fold_games) == set(games), 'aggregate fold coverage/binding mismatch')
            require(aggregate['observer_row_weighted_diagnostic']['row_count'] == len(all_rows)
                    and aggregate['observer_row_weighted_diagnostic']['nonzero_gap_row_count'] == sum(r['support_size'] < 6 for r in all_rows),
                    'aggregate diagnostic row counts mismatch')
            formal[cell_key] = {key: aggregate[key] for key in FORMAL}
            derived[cell_key] = {name: macro(all_rows, name) for name in METRICS}
            strata[cell_key] = {}
            main_csv.append({'condition': condition, 'population': population,
                **flatten({k: aggregate[k] for k in FORMAL if k != 'game_scores'}, 'formal'),
                **flatten(derived[cell_key], 'derived')})
            for label in STRATA:
                subset = [r for r in all_rows if ('3-5' if 3 <= r['support_size'] <= 5 else str(r['support_size'])) == label]
                value = {'row_count': len(subset), 'game_count': len({r['game_id'] for r in subset}),
                         **{name: macro(subset, name) for name in ('kl', 'total_variation', *METRICS)}}
                strata[cell_key][label] = value
                strata_csv.append({'condition': condition, 'population': population, 'support_stratum': label, **flatten(value)})
    try:
        head = subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[1]), 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError): head = None
    provenance = {'runs_root': str(root), 'artifact_schema_versions': SCHEMAS,
        **{k: report_set[k] for k in ('experiment_digest', 'checkpoint_set_digest', 'bootstrap_digest')},
        'game_count': len(games), 'game_ids': games, 'conditions': CONDITIONS, 'populations': POPULATIONS,
        'analyzer_git_head': head, 'analyzer_source_sha256': digest(Path(__file__).read_bytes()),
        'utc_execution_time': datetime.now(timezone.utc).isoformat(), 'historical_schema_revision': HISTORICAL_REVISION,
        'consumed_inputs': inputs.manifest}
    summary = {'classification': 'DERIVED PAPER DIAGNOSTICS', 'formal_artifacts_were_not_modified': True,
        'formal_metrics': formal, 'paired_formal_headlines': paired,
        'derived_paper_diagnostics': derived, 'support_size_diagnostics': strata, 'input_provenance': provenance}
    out.mkdir(parents=True, exist_ok=False)
    (out / 'summary.json').write_bytes(canonical(summary))
    for filename, rows in (('main_metrics.csv', main_csv), ('support_size_metrics.csv', strata_csv)):
        with (out / filename).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (out / 'README.txt').write_text(README + '\nInput provenance:\n' + json.dumps(provenance, indent=2), encoding='utf-8')
    return summary


README = '''DERIVED PAPER DIAGNOSTICS — DERIVED-ONLY analysis, not a formal OOF artifact.
Formal artifacts were not modified. No training, prediction, Final Test,
calibration, selection, tuning or derived bootstrap CI is performed.
Formal headline/CI, uniform reference, secondary and observer-row-weighted
normalized reducible-gap diagnostics are copied unchanged. Paired effects are
copied: temporal = implicit minus explicit Primary KL; stress = All-Alive minus
Primary KL per condition. Positive temporal difference favors explicit.
Population is selected only by fold_report.row_scores identities, never inferred
from flags; prediction rows have NO population field.
Top1_NU = top non-self seat in support; target_support_probability_nu = support
probability sum; MRR_NU = reciprocal best support rank. All three exclude support
size 6. Ties use ascending canonical seat index (formal first-argmax convention).
Soft Brier (soft_brier_sum) = sum of six non-self squared p-q differences, NOT /6.
All derived metrics: row -> within-game mean -> valid-game mean. Games without
qualifying rows are omitted, never zero-filled. Counts accompany every metric.
Support strata: 1, 2, 3-5, 6. KL/TV reuse formal fold row scores. Size 6 has uniform
six-seat q: formal support hit/probability mechanically equal 1/1; they are not
Top-1 accuracy. Size-6 derived support metrics are null and have zero valid counts.
Historical sources (33d580cdf0700f6e38ae5b14648fcfdaa9be2b9a):
werewolf/tom/evaluation.py: predict_held_out_fold/open_predictions,
_fold_report_value/primary_fold_report_value/all_alive_fold_report_value,
score_prediction_rows; reporting.py: _aggregate_cell/_report_set_value/summarize_scores;
scoring.py: game_macro_summary/paired_summary (classic7_game_macro_belief_kl_v1);
training.py: lineage_path; run_records.py: read_record; dataset.py: belief_target.
Exact input record keys are retained in source schema checks. The analysis checks
consumed record bindings but does not revalidate original publication/model/seal
artifacts or recompute formal scores. It is not formal qualification validation.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    analyze(args.runs_root, args.output_dir)


if __name__ == '__main__':
    main()
