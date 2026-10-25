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


def compare(reference, predictions, game_ids):
    """All public-alive requests must match exactly; Primary is selected afterwards."""
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
