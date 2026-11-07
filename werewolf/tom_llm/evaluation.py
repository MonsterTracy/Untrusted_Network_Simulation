"""Strict public-alive joins using the unchanged game-macro mathematics."""

from collections import Counter, defaultdict
import math

import numpy as np

from werewolf.artifact_io import sha256_bytes
from werewolf.tom.evaluation import score_prediction_rows
from werewolf.tom.protocol import bootstrap_indices
from werewolf.tom.scoring import game_macro_summary, paired_summary
from werewolf.tom_llm.common import digest, identity, require
from werewolf.tom_llm.output import validate_prediction
from werewolf.tom_llm.policy import POLICY


def _aligned_predictions(reference, predictions, game_ids):
    """所有指标共用完整公开请求核对，不按预测是否成功取交集。"""
    games = tuple(sorted(game_ids))
    expected = reference.selected(games)
    required_keys = {key for key in reference.alive_keys if key[0] in set(games)}
    indexed = {}
    for row in predictions:
        require(not {"q", "label_observed", "primary_eligible", "observer_alive", "role", "suspicion_support"} & set(row),
                "LLM rows must not supply supervision/population semantics")
        key = identity(row)
        require(key not in indexed and key in required_keys, "duplicate/foreign LLM row; no intersection permitted")
        validate_prediction(row["prediction"])
        require(row["prediction"]["observer"] == row["observer"], "prediction observer mismatch")
        indexed[key] = row
    require(set(indexed) == required_keys, "missing public-alive LLM row; no dropping permitted")
    all_reference = {identity(r): r for r in reference.rows}
    for key, row in indexed.items():
        ref = all_reference[key]
        require(type(row["fold"]) is int and row["fold"] == ref["fold"]
                and row["plan_digest"] == ref["plan_digest"] and type(row["canonical_shift"]) is int and row["canonical_shift"] == 0
                and row["temporal_condition"] == ref["temporal_condition"], "LLM/reference semantics mismatch")
    return games, expected, indexed


def compare(reference, predictions, game_ids):
    """All public-alive requests must match exactly; Primary is selected afterwards."""
    games, expected, indexed = _aligned_predictions(reference, predictions, game_ids)
    selected, statuses = [], []
    for ref in expected:
        prediction = indexed[identity(ref)]["prediction"]
        selected.append({**ref, "probability": prediction["probability"],
                         "non_self_log_probability": prediction["non_self_log_probability"]})
        statuses.append((ref, prediction))
    indices = (reference.bootstrap_indices if games == reference.game_ids else
        bootstrap_indices(list(games), POLICY["bootstrap"]["replicates"], POLICY["bootstrap"]["seed"]))
    cells, row_scores = {}, {}
    uniform = [{**r, "probability": [0. if j == r["observer"] else 1/6 for j in range(7)],
                "non_self_log_probability": [-math.log(6)] * 6} for r in expected]
    for name, rows in (("llm", selected), ("qwen3_oof", expected), ("uniform_non_self", uniform)):
        scores, game_scores = score_prediction_rows(rows, games)
        row_scores[name] = scores
        cells[name] = {"kl": game_macro_summary([game_scores[g]["kl"] for g in games], indices, .95),
            "tv": game_macro_summary([game_scores[g]["total_variation"] for g in games], indices, .95),
            "game_scores": game_scores, "row_count": len(rows), "game_count": len(games)}
    delta = paired_summary([cells["llm"]["game_scores"][g]["kl"] for g in games],
        [cells["qwen3_oof"]["game_scores"][g]["kl"] for g in games], indices, .95)
    nu_games = sorted({r["game_id"] for r in expected if sum(q > 0 for q in r["q"]) < 6})
    nu_indices = bootstrap_indices(nu_games, POLICY["bootstrap"]["replicates"], POLICY["bootstrap"]["seed"]) if nu_games else None
    nu_rows = sum(sum(q > 0 for q in r["q"]) < 6 for r in expected)
    for name, scores in row_scores.items():
        counts = defaultdict(list)
        for ref, score in zip(expected, scores, strict=True):
            if sum(q > 0 for q in ref["q"]) < 6:
                counts[ref["game_id"]].append(int(score["argmax_in_target_support"]))
        cells[name]["top1_nu"] = (game_macro_summary([np.mean(counts[g]) for g in nu_games], nu_indices, .95)
                                   if nu_games else None)
        cells[name]["nu_row_count"] = nu_rows
        cells[name]["nu_game_count"] = len(nu_games)
    failures = Counter(p["failure_type"] for _, p in statuses if p["status"] == "fallback_uniform")
    per_game = {g: np.mean([p["status"] == "fallback_uniform" for r, p in statuses if r["game_id"] == g]) for g in games}
    failure_summary = {"primary_failure_types": dict(failures), "primary_row_failure_rate": sum(failures.values()) / len(expected),
        "primary_game_macro_failure_rate": game_macro_summary([per_game[g] for g in games], indices, .95),
        "scheduled_public_alive_rows": len(predictions), "primary_rows": len(expected),
        "legal_uniform_rows": sum(p["status"] == "valid" and len(set(p["raw_probability"])) == 1 for _, p in statuses),
        "normalized_for_rounding_rows": sum(p["normalized_for_rounding"] for _, p in statuses),
        "raw_zero_rows": sum(p["raw_zero_count"] > 0 for _, p in statuses)}
    failure_summary["completed_request_count"] = len(predictions)
    failure_summary["dispatch_attempt_count"] = (sum(len(r["attempt_digests"]) for r in predictions)
        if all("attempt_digests" in r for r in predictions) else None)
    durations = [duration for r in predictions for duration in r.get("dispatch_elapsed_seconds", [])]
    failure_summary["reported_dispatch_elapsed_seconds"] = {
        "sum": math.fsum(durations), "attempts_with_timing": len(durations)}
    usages = [r.get("usage", {}) for r in predictions]
    failure_summary["reported_token_usage"] = {}
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        values = [u[name] for u in usages if type(u) is dict and type(u.get(name)) is int and u[name] >= 0]
        failure_summary["reported_token_usage"][name] = {"sum": sum(values), "rows_with_usage": len(values)}
    public_failures = Counter(r["prediction"]["failure_type"] for r in predictions
                              if r["prediction"]["status"] == "fallback_uniform")
    failure_summary["public_alive_failure_types"] = dict(public_failures)
    failure_summary["public_alive_row_failure_rate"] = sum(public_failures.values()) / len(predictions)
    failure_summary["primary_by_observer"] = {}
    failure_summary["primary_by_phase"] = {}
    failure_summary["primary_by_length"] = {}
    for ref, prediction in statuses:
        groups = (("primary_by_observer", str(ref["observer"])),
                  ("primary_by_phase", ref["public_phase"]),
                  ("primary_by_length", "short" if ref["public_token_count"] <= 256 else
                                        "medium" if ref["public_token_count"] <= 512 else "long"))
        for group, key in groups:
            cell = failure_summary[group].setdefault(key, {"row_count": 0, "failure_count": 0})
            cell["row_count"] += 1
            cell["failure_count"] += int(prediction["status"] == "fallback_uniform")
    return {"game_ids": list(games), "cells": cells, "Delta_KL": delta,
            "Delta_KL_definition": "KL_llm_minus_KL_qwen3_oof", "failures": failure_summary,
            "reference_seal_digest": reference.seal_digest, "reference_prediction_digests": reference.prediction_digests,
            "primary_row_identity_digest": digest([identity(r) for r in expected]),
            "bootstrap_indices_digest": sha256_bytes(indices.tobytes()), "nu_game_ids": nu_games,
            "nu_bootstrap_indices_digest": None if nu_indices is None else sha256_bytes(nu_indices.tobytes()),
            "limitations": POLICY["limitations"]}


SET_METRICS = ("set_f1", "set_precision", "set_recall", "jaccard", "exact_set_match", "valid_output_rate")


def set_scores(prediction, gold):
    """空空记一，单边空记零；非法输出在全部集合指标中记零。"""
    from werewolf.canonical_collection.public_history import PLAYER_IDS
    from werewolf.tom_llm.output import SET_OUTPUT_VERSION
    validate_prediction(prediction)
    require(prediction.get('schema_version') == SET_OUTPUT_VERSION, 'set metrics require set output protocol')
    require(type(gold) is list and all(p in PLAYER_IDS for p in gold) and len(set(gold)) == len(gold)
            and PLAYER_IDS[prediction['observer']] not in gold, 'invalid raw gold set')
    result = dict.fromkeys(SET_METRICS, 0.)
    if prediction['status'] != 'valid':
        return result
    result['valid_output_rate'] = 1.
    actual, target = set(prediction['suspected_players']), set(gold)
    if not actual and not target:
        return dict.fromkeys(SET_METRICS, 1.)
    if not actual or not target:
        return result
    overlap = len(actual & target)
    result.update(set_precision=overlap / len(actual), set_recall=overlap / len(target),
                  set_f1=2 * overlap / (len(actual) + len(target)),
                  jaccard=overlap / len(actual | target), exact_set_match=float(actual == target))
    return result


def compare_sets(reference, predictions, game_ids, raw_labels, *, probability_to_set_rule=None,
                 formal_game_ids, approved=False, engineering_only=False):
    """共用原始集合人口、游戏级统计与完整预测分母的多模型评价。"""
    from itertools import combinations
    from werewolf.canonical_collection.public_history import PLAYER_IDS
    from werewolf.tom_llm.output import SET_OUTPUT_VERSION, parse_set_output, probability_to_set
    from werewolf.artifact_io import canonical_json_bytes

    games = tuple(sorted(game_ids))
    require(raw_labels['publication_digest'] == reference.publication_digest
            and raw_labels['reference_seal_digest'] == reference.seal_digest
            and digest(raw_labels['labels']) == raw_labels['labels_digest'], 'raw-set source binding mismatch')
    refs = {identity(r):r for r in reference.rows
            if r['game_id'] in set(games) and r['primary_eligible'] and r['observer_alive']}
    labels = {identity(r):r for r in raw_labels['labels']}
    require(len(labels) == len(raw_labels['labels']) and set(labels) == set(refs), 'raw-set Primary coverage mismatch')
    indexed = {}
    for model, rows in predictions.items():
        _, _, indexed[model] = _aligned_predictions(reference, rows, games)
        require(all(r['prediction'].get('schema_version') == SET_OUTPUT_VERSION for r in rows),
                'mixed probability/set result protocols')
    require(indexed, 'set evaluation requires model predictions')
    derived_methods = {}
    if approved:
        require(set(indexed)<={'qwen3','yi','mistral'} and probability_to_set_rule is None,
                'approved set evaluation forbids legacy threshold or extra model runs')
        original = next(iter(indexed.values()))
        for name in ('Always Empty','Always All-Six'):
            derived = {}
            for key, row in original.items():
                players = [] if name=='Always Empty' else [p for j,p in enumerate(PLAYER_IDS) if j!=row['observer']]
                fields = ('game_id','boundary_id','prefix_digest','plan_digest','observer','fold','canonical_shift','temporal_condition')
                derived[key] = {k:row[k] for k in fields}
                derived[key]['prediction'] = parse_set_output(
                    canonical_json_bytes({'suspected_players':players}).decode(),row['observer']).to_record()
            indexed[name] = derived
            derived_methods[name] = {'kind':'deterministic_reference', 'model_run_seal':None,
                'parent_public_row_identity_digest':digest(list(original)),
                'predictions_digest':digest([[list(k),r['prediction']['suspected_players']] for k,r in derived.items()])}
    groups = {name:[] for name in ('full_primary_raw', 'primary_comparable_nu', 'empty', 'nonempty_proper', 'all_six')}
    unavailable = []
    label_statuses = Counter()
    for key, label in labels.items():
        ref = refs[key]
        require(label['plan_digest'] == ref['plan_digest'] and label['fold'] == ref['fold'], 'raw-set plan/fold drift')
        require(label['status'] in ('observed','unobservable','not_recoverable'), 'unknown raw-set label status')
        label_statuses[label['status']] += 1
        if label['status'] != 'observed':
            require(label['suspected_players'] is None, 'unobservable report cannot supply a gold set')
            unavailable.append(list(key))
            continue
        gold = label['suspected_players']
        require(type(gold) is list and len(set(gold)) == len(gold)
                and all(p in PLAYER_IDS for p in gold) and PLAYER_IDS[ref['observer']] not in gold,
                'invalid raw report support')
        require(ref['label_observed'], 'raw-set/Reference observability conflict')
        expected = set(gold) or {p for j,p in enumerate(PLAYER_IDS) if j != ref['observer']}
        require(expected == {p for p,q in zip(PLAYER_IDS,ref['q'],strict=True) if q > 0},
                'raw-set/Reference support conflict')
        groups['full_primary_raw'].append(key)
        group = 'nonempty_proper'
        if not gold:
            group = 'empty'
        elif len(gold) == 6:
            group = 'all_six'
        groups[group].append(key)
        if group == 'nonempty_proper':
            groups['primary_comparable_nu'].append(key)
    mapper = {'status':'BLOCKED_INDEPENDENT_DEVELOPMENT_NOT_VERIFIED', 'rule':None}
    tom_sets = {}
    if approved:
        from werewolf.tom_llm.output import uniform_reference_projection, fixed_projection_rule
        from werewolf.tom_llm.reference import projection_source
        source = projection_source(reference,engineering_only=engineering_only)
        tom_sets = {key:uniform_reference_projection(ref['probability'],ref['observer']) for key,ref in refs.items()}
        rule = fixed_projection_rule()
        mapper = {'status':source['status'], 'rule':rule, 'rule_digest':digest(rule), 'source':source,
                  'predictions_digest':digest([[list(k),v] for k,v in tom_sets.items()])}
        derived_methods['Observer-ToM'] = {'kind':'fixed_oof_projection', 'model_run_seal':None,
                                          'source_digest':source['actual_digest'], 'rule_digest':digest(rule)}
    elif probability_to_set_rule is not None:
        # Mock 只证明接口，摘要及分离名单不提升为真实开发来源资格。
        for key, ref in refs.items():
            tom_sets[key] = probability_to_set(ref['probability'], ref['observer'], probability_to_set_rule,
                                                formal_game_ids=formal_game_ids)
        mapper = {'status':'ENGINEERING_ONLY_NOT_CALIBRATED', 'rule':probability_to_set_rule,
                  'rule_digest':digest(probability_to_set_rule)}
    populations = {}
    for name, keys in groups.items():
        hit_games = sorted({key[0] for key in keys})
        indices = bootstrap_indices(hit_games, POLICY['bootstrap']['replicates'], POLICY['bootstrap']['seed']) if hit_games else None
        models = list(indexed)
        if name == 'primary_comparable_nu' and tom_sets:
            models.append('Observer-ToM')
        cells, row_values = {}, {}
        for model in models:
            scores = []
            for key in keys:
                if model == 'Observer-ToM':
                    content = canonical_json_bytes({'suspected_players':tom_sets[key]}).decode('utf-8')
                    prediction = parse_set_output(content, refs[key]['observer']).to_record()
                else:
                    prediction = indexed[model][key]['prediction']
                scores.append({'identity':list(key), **set_scores(prediction, labels[key]['suspected_players'])})
            per_game = {g:{metric:float(np.mean([s[metric] for s in scores if s['identity'][0] == g]))
                           for metric in SET_METRICS} for g in hit_games}
            metrics = {metric:game_macro_summary([per_game[g][metric] for g in hit_games], indices, .95)
                       for metric in SET_METRICS} if hit_games else {metric:None for metric in SET_METRICS}
            cells[model] = {'row_count':len(keys), 'game_count':len(hit_games), 'metrics':metrics,
                            'game_scores':per_game, 'row_scores':scores}
            row_values[model] = per_game
        pairs = []
        for left, right in combinations(models, 2):
            pair = {'left':left, 'right':right, 'direction':'left_minus_right'}
            if approved:
                simple = {'Always Empty','Always All-Six'}
                if name not in ('full_primary_raw','primary_comparable_nu'):
                    pair['tier'] = 'diagnostic'
                elif 'Observer-ToM' in (left,right) and not ({left,right}&simple):
                    pair['tier'] = 'key_method'
                elif len({left,right}&simple)==1:
                    pair['tier'] = 'simple_baseline'
                else:
                    pair['tier'] = 'exploratory'
            pair['metrics'] = {metric:paired_summary([row_values[left][g][metric] for g in hit_games],
                [row_values[right][g][metric] for g in hit_games], indices, .95) if hit_games else None
                for metric in SET_METRICS}
            pairs.append(pair)
        populations[name] = {'row_count':len(keys), 'game_ids':hit_games,
            'zero_sample_games':sorted(set(games) - set(hit_games)), 'row_identity_digest':digest(keys),
            'bootstrap_replicates':POLICY['bootstrap']['replicates'],
            'bootstrap_indices_digest':None if indices is None else sha256_bytes(indices.tobytes()),
            'cells':cells, 'paired':pairs}
        if approved:
            populations[name]['reporting_tier'] = ('primary_descriptive' if name=='full_primary_raw' else
                'key_secondary' if name=='primary_comparable_nu' else 'diagnostic')
    coverage = {}
    for model, rows in predictions.items():
        valid = [r['prediction'] for r in rows if r['prediction']['status'] == 'valid']
        failures = Counter(r['prediction']['failure_type'] for r in rows if r['prediction']['status'] != 'valid')
        coverage[model] = {'rows':len(rows), 'valid':len(valid), 'invalid':sum(failures.values()),
            'valid_output_rate':len(valid)/len(rows), 'failure_types':dict(failures),
            'valid_empty_predictions':sum(p['suspected_players'] == [] for p in valid),
            'valid_all_six_predictions':sum(len(p['suspected_players']) == 6 for p in valid),
            'valid_nonempty_proper_predictions':sum(0 < len(p['suspected_players']) < 6 for p in valid)}
    result = {'schema_version':'classic7_llm_suspicion_set_evaluation_v2', 'formal_result':False,
            'raw_label_sources':{k:v for k,v in raw_labels.items() if k != 'labels'},
            'primary_candidate_rows':len(refs), 'unobservable_or_unrecoverable_rows':len(unavailable),
            'unobservable_row_identities':unavailable, 'label_status_counts':dict(label_statuses),
            'raw_reports':raw_labels['labels'], 'populations':populations,
            'probability_to_set':mapper,
            'public_output_coverage':coverage,
            'interval_scope':'fixed_predictions_game_bootstrap_not_simultaneous',
            'uniform_reference_limitation':'empty_and_all_six_raw_reports_share_q; Observer-ToM set comparison is NU only'}
    if approved:
        result.update(research_protocol_approved=True, physical_qualification=False,
            derived_methods=derived_methods, comparison_complete=set(predictions)=={'qwen3','yi','mistral'},
            primary_endpoint='full_primary_raw/set_f1',
            valid_output_definitions={'public':'row_micro_all_public_alive', 'populations':'game_macro_same_gold_denominator'})
    return result


def analyze_mock(reference, study_root, game_ids, destination):
    """Publish only engineering comparisons from a verified mock prediction seal."""
    from pathlib import Path
    from werewolf.artifact_io import publish_artifact, canonical_json_bytes
    from werewolf.tom_llm.inference import open_mock_predictions
    seal, rows = open_mock_predictions(study_root)
    output = Path(destination).absolute()
    require(not output.exists() and not output.is_relative_to(Path(study_root).absolute()),
            "analysis output must be new and outside the sealed study")
    result = compare(reference, rows, game_ids)
    return publish_artifact(output, manifest_fields={
        "artifact_type": "llm_mock_comparison", "schema_version": "classic7_llm_mainline_mock_comparison_v1",
        "engineering_only": True, "llm_prediction_seal_digest": seal.manifest_digest,
        "reference_seal_digest": reference.seal_digest, "game_ids": result["game_ids"],
        "bootstrap_indices_digest": result["bootstrap_indices_digest"],
        "nu_bootstrap_indices_digest": result["nu_bootstrap_indices_digest"],
    }, files={"comparison.json": canonical_json_bytes(result)})
