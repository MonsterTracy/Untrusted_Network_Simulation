"""Read-only, post-hoc Top1_NU points from sealed paper OOF predictions. No CI."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.paper_tom_metrics import Inputs, require, strict_json, identity, validate_prediction, join_rows
from werewolf.artifact_io import canonical_json_bytes, canonical_jsonl_bytes, sha256_bytes, verify_artifact
from werewolf.tom.evaluation import score_prediction_rows
from werewolf.tom.paper_study_evaluation import _pair_rows, FAMILIES, TEMPORAL_CONDITIONS, PRIMARY

# External trust anchors supplied for the completed study; never derived from the inputs being checked.
STUDY = 'added53e6e9c62ca5b54e79962cebcf1fce1545ec823f40a91fea822fa03fbe5'
SEAL = '7e54c97dd392f220b6ff4c46c947b3611687a7b8ac051aa1e066b7dcf53d1bc8'
EVALUATION = '8ebd80295bb2517d20212ed0f641109097b2bb86774c926c99dea32592bb9a14'


def summarize(rows, hits):
    require(len(rows) == len(hits), 'hit coverage mismatch')
    games = defaultdict(list)
    excluded = 0
    for row, hit in zip(rows, hits):
        validate_prediction(row, selected=True)
        require(type(hit) is bool, 'invalid formal support hit')
        size = sum(q > 0 for q in row['q'])
        require(1 <= size <= 6, 'invalid support size')
        if size == 6:
            excluded += 1
        else:
            games[row['game_id']].append(hit)
    count = sum(len(values) for values in games.values())
    require(count + excluded == len(rows), 'NU coverage mismatch')
    means = [sum(games[g]) / len(games[g]) for g in sorted(games)]
    return {'top1_nu': sum(means) / len(means) if means else None,
        'total_primary_row_count': len(rows), 'valid_nu_row_count': count,
        'excluded_s6_row_count': excluded,
        'total_primary_game_count': len({r['game_id'] for r in rows}),
        'valid_nu_game_count': len(games)}


def predictions(inputs, relative, descriptor, condition):
    payload = inputs.read(relative, record=False)
    require(sha256_bytes(payload) == descriptor['prediction_digest'], 'prediction digest mismatch')
    rows = [strict_json(line) for line in payload.splitlines()]
    require(payload == canonical_jsonl_bytes(rows) and len(rows) == descriptor['row_count'],
            'prediction serialization/count mismatch')
    indexed = {}
    for row in rows:
        validate_prediction(row)
        key = identity(row)
        require(key not in indexed, 'duplicate prediction row')
        require(row['temporal_condition'] == condition and row['checkpoint_digest'] == descriptor['checkpoint_digest'],
                'prediction condition/checkpoint mismatch')
        indexed[key] = row
    return rows, indexed


def analyze(study_root, output, *, expected_study=STUDY, expected_seal=SEAL, expected_evaluation=EVALUATION):
    root, out = Path(study_root).absolute(), Path(output).absolute()
    require(not any(p.is_symlink() for p in (root, *root.parents, out, *out.parents)), 'symlink prohibited')
    root, out = root.resolve(), out.resolve()
    require(root.is_dir() and root != out and root not in out.parents, 'output must be outside study tree')
    require(not out.exists(), 'output already exists')
    execution = verify_artifact(root/'execution', expected_artifact_type='paper_study_execution',
                                expected_schema_version='classic7_paper_study_execution_v1')
    artifact = verify_artifact(root/'evaluation', expected_artifact_type='paper_study_evaluation',
                               expected_schema_version='classic7_paper_study_evaluation_v1')
    m, e = artifact.manifest, execution.manifest
    require(execution.manifest_digest == m['study_digest'] == expected_study, 'study identity mismatch')
    require(artifact.manifest_digest == expected_evaluation, 'evaluation identity mismatch')
    inputs = Inputs(root)
    seal = inputs.read('study_seal.json')
    require(seal['record_digest'] == m['study_seal_digest'] == expected_seal
            and seal['study_digest'] == expected_study, 'seal identity mismatch')
    expected = {(family,c,f) for family in FAMILIES for c in TEMPORAL_CONDITIONS for f in range(5)}
    require(len(e['expected_lineages']) == 20 and {tuple(v) for v in e['expected_lineages']} == expected,
            'execution lineage mismatch')
    terminals = {}
    for entry in seal['checkpoints']:
        key = entry['model_family'], entry['temporal_condition'], entry['fold']
        require(key not in terminals and entry['study_digest'] == expected_study, 'duplicate/foreign terminal binding')
        terminals[key] = entry['checkpoint_digest']
    require(set(terminals) == expected, 'seal lineage coverage mismatch')
    full_digest = e['full_experiment_digest']
    runs = f'runs/{full_digest}'
    formal = inputs.read(f'{runs}/reports/oof_report_set.json')
    require(formal['record_digest'] == m['full_report_set_digest']
            and formal['experiment_digest'] == full_digest
            and formal['checkpoint_set_digest'] == m['full_checkpoint_set_digest'], 'formal report binding mismatch')
    games = formal['game_ids']
    require(games and games == sorted(set(games)), 'invalid game ordering')
    require(games == m['bootstrap']['game_ids'], 'frozen game ordering mismatch')
    keys = {f'{c}/{f}' for c in TEMPORAL_CONDITIONS for f in range(5)}
    require(set(m['predictions']) == keys, 'prediction fold coverage mismatch')
    require(set(m['file_table']) == {f'agnostic/{k}/held_out_predictions.jsonl' for k in keys},
            'agnostic payload coverage mismatch')
    require(set(m['cells']) == {f'{family}/{c}' for family in FAMILIES for c in TEMPORAL_CONDITIONS},
            'four Primary cells required')
    cells, bindings = {}, {}
    for condition in TEMPORAL_CONDITIONS:
        selected = {family: [] for family in FAMILIES}
        hits = {family: [] for family in FAMILIES}
        seen_games = set()
        aggregate = formal['cells'][f'{condition}/{PRIMARY}']
        for fold in range(5):
            key = f'{condition}/{fold}'
            binding = m['predictions'][key]
            descriptor = inputs.read(f'{runs}/{key}/prediction_manifest.json')
            require(descriptor['record_digest'] == binding['full_prediction_digest']
                    and descriptor['experiment_digest'] == full_digest
                    and descriptor['checkpoint_set_digest'] == m['full_checkpoint_set_digest'], 'Full prediction binding mismatch')
            baseline = binding['agnostic']
            require(baseline['experiment_digest'] == e['agnostic_experiment_digest']
                    and baseline['checkpoint_set_digest'] == expected_seal, 'agnostic parent mismatch')
            for family, desc in zip(FAMILIES, (descriptor, baseline)):
                require(type(desc['fold']) is int and desc['fold'] == fold
                        and desc['temporal_condition'] == condition and desc['canonical_shift'] == 0
                        and desc['checkpoint_digest'] == terminals[family,condition,fold], 'fold/terminal mismatch')
            full_rows, full_map = predictions(inputs, f'{runs}/{key}/held_out_predictions.jsonl', descriptor, condition)
            base_rows, base_map = predictions(inputs, f'evaluation/agnostic/{key}/held_out_predictions.jsonl', baseline, condition)
            report = inputs.read(f'{runs}/{key}/{PRIMARY}/fold_report.json')
            require(report['record_digest'] == binding['full_primary_report_digest'] == aggregate['fold_report_digests'][fold],
                    'Primary report digest mismatch')
            require(report['population_identity'] == 'non_wolf_alive'
                    and report['population_selector_version'] == 'classic7_primary_population_v1', 'Primary population mismatch')
            for name in ('experiment_digest','checkpoint_set_digest','checkpoint_digest','prediction_digest','fold','temporal_condition'):
                require(report[name] == descriptor[name], 'Primary prediction provenance mismatch')
            join_rows(report, full_map)  # Existing strict selected-row/identity validation.
            require(report['row_identity_digest'] == baseline['primary_row_identity_digest'], 'Primary identity mismatch')
            selected_ids = [identity(r) for r in report['row_scores']]
            selected_set = set(selected_ids)
            masks = {identity(r): identity(r) in selected_set for r in full_rows}
            fold_games = sorted({r['game_id'] for r in full_rows})
            require(not seen_games.intersection(fold_games), 'duplicate game across folds')
            seen_games.update(fold_games)
            bindings[key] = _pair_rows(full_rows, base_rows, masks, masks, fold=fold,
                                       condition=condition, game_ids=fold_games)
            for family, indexed in zip(FAMILIES, (full_map, base_map)):
                rows = [indexed[k] for k in selected_ids]
                scores, _ = score_prediction_rows(rows, fold_games)
                row_hits = [r['argmax_in_target_support'] for r in scores]
                if family == FAMILIES[0]:
                    saved = [r['argmax_in_target_support'] for r in report['row_scores']]
                    require(all(type(v) is bool for v in saved) and saved == row_hits, 'formal hit mismatch')
                    row_hits = saved
                selected[family].extend(rows)
                hits[family].extend(row_hits)
        require(seen_games == set(games), 'OOF game coverage mismatch')
        for family in FAMILIES:
            key = f'{family}/{condition}'
            cells[key] = summarize(selected[family], hits[family])
            require(cells[key]['total_primary_row_count'] == m['cells'][key]['row_count']
                    and cells[key]['total_primary_game_count'] == m['cells'][key]['game_count'], 'sealed coverage mismatch')
    require(sha256_bytes(canonical_json_bytes(bindings)) == m['row_identity_binding_digest'], 'sealed row binding mismatch')
    result = {'schema_version':'paper_top1_nu_point_v1', 'status':'post_hoc_supplementary_diagnostic',
        'source':{'evaluation_manifest_digest':expected_evaluation,'study_digest':expected_study,'study_seal_digest':expected_seal},
        'metric_definition':{'name':'top1_nu','aggregation':'game_macro_over_nonuniform_targets',
            'support_rule':'1 <= support_size < 6','tie_rule':'lowest_canonical_seat_index'}, 'cells':cells}
    with out.open('xb') as stream:
        stream.write(canonical_json_bytes(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    analyze(args.study_root, args.output)


if __name__ == '__main__':
    main()
