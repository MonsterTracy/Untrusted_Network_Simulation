"""All-twenty-gated OOF comparison for the two frozen paper model families."""

import numpy as np
import torch

from werewolf.artifact_io import (canonical_json_bytes, canonical_jsonl_bytes,
    sha256_bytes, publish_artifact, verify_artifact)
from werewolf.tom import evaluation, reporting, training
from werewolf.tom.dataset import PublicTensors
from werewolf.tom.experiment import TEMPORAL_CONDITIONS, read_json
from werewolf.tom.paper_study_execution import (open_study, evaluation_eligibility,
    _verify_terminal, _baseline_model, FAMILIES)
from werewolf.tom.population import load_held_out_primary, PRIMARY_SELECTOR_VERSION
from werewolf.tom.protocol import load_bootstrap_plan, BOOTSTRAP_VERSION
from werewolf.tom.run_records import read_record, record_with_digest
from werewolf.tom.scoring import game_macro_summary, paired_summary
from werewolf.tom.state import load_model_state

EVALUATION_VERSION = "classic7_paper_study_evaluation_v1"
PRIMARY = "primary_development_oof"
STRATA = ("1", "2", "3-5", "6")


def _identity(row):
    if (type(row["observer"]) is not int or not 0 <= row["observer"] < 7
            or any(type(row[k]) is not str or not row[k] for k in ("game_id", "boundary_id"))):
        raise ValueError("invalid prediction identity")
    return row["game_id"], row["boundary_id"], row["observer"]


def _pair_rows(full, agnostic, full_masks, agnostic_masks, *, fold, condition, game_ids):
    """Exact identity/target/population equality, never intersection or partial pairing."""
    if type(fold) is not int or fold not in range(5) or condition not in TEMPORAL_CONDITIONS:
        raise ValueError("invalid fold/temporal identity")
    maps = []
    for rows, masks in ((full, full_masks), (agnostic, agnostic_masks)):
        indexed = {_identity(row): row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError("duplicate prediction row")
        if {key[0] for key in indexed} != set(game_ids):
            raise ValueError("prediction game/fold coverage mismatch")
        if set(masks) != set(indexed) or any(type(v) is not bool for v in masks.values()):
            raise ValueError("Primary eligibility coverage/type mismatch")
        maps.append(indexed)
    a, b = maps
    if set(a) != set(b):
        raise ValueError("Full/agnostic row identity sets differ")
    if full_masks != agnostic_masks:
        raise ValueError("Primary eligibility mismatch")
    bindings = []
    for key in sorted(a):
        left, right = a[key], b[key]
        if set(left) != set(right):
            raise ValueError("prediction row schema mismatch")
        identity = {k: v for k, v in left.items() if k not in
                    ("probability", "non_self_log_probability", "checkpoint_digest")}
        other = {k: right[k] for k in identity}
        if canonical_json_bytes(identity) != canonical_json_bytes(other) or identity["temporal_condition"] != condition:
            raise ValueError("paired PRE/target/observation/temporal mismatch")
        observer = key[2]
        p, logs = np.asarray(right["probability"]), np.asarray(right["non_self_log_probability"])
        if (p.shape != (7,) or logs.shape != (6,) or not np.isfinite(p).all()
                or not np.isfinite(logs).all() or p[observer] != 0 or np.any(p < 0)
                or not np.isclose(p.sum(), 1, atol=1e-6)
                or not np.allclose(np.delete(p, observer), np.exp(logs), atol=1e-7)):
            raise ValueError("agnostic prediction simplex/log mismatch")
        bindings.append({"fold": fold, **identity, "primary_eligible": full_masks[key],
                         "population_selector_version": PRIMARY_SELECTOR_VERSION})
    return sha256_bytes(canonical_json_bytes(bindings))


def _predict_agnostic(study, dataset, condition, checkpoint):
    model = _baseline_model(study.agnostic, condition)
    load_model_state(checkpoint.path, model, checkpoint.manifest_digest)
    model.eval()
    rows = []
    with torch.inference_mode():
        for sample in dataset:
            public = PublicTensors.stack([sample.public])
            logp = model(**{k: v.to(study.full.config.device) for k, v in public.kwargs().items()})[0].cpu()
            for observer in range(7):
                rows.append({"game_id": sample.game_id, "boundary_id": sample.boundary_id,
                    "observer": observer, "prefix_digest": sample.prefix_digest, "plan_digest": sample.plan_digest,
                    "probability": logp[observer].exp().tolist(),
                    "non_self_log_probability": [logp[observer, j].item() for j in range(7) if j != observer],
                    "q": sample.q[observer].tolist(), "label_observed": bool(sample.label_observed[observer]),
                    "observer_alive": bool(sample.observer_alive[observer]),
                    "checkpoint_digest": checkpoint.manifest_digest, "temporal_condition": condition})
    return rows


def _support_summary(selected):
    result = {}
    for label in STRATA:
        subset = []
        for row in selected:
            size = sum(q > 0 for q in row["q"])
            if ("3-5" if 3 <= size <= 5 else str(size)) == label:
                subset.append(row)
        ids = sorted({r["game_id"] for r in subset})
        if ids:
            _, games = evaluation.score_prediction_rows(subset, ids)
            means = {name: float(np.mean([games[g][name] for g in ids])) for name in ("kl", "uniform_kl")}
        else:
            means = {"kl": None, "uniform_kl": None}
        result[label] = {**means, "row_count": len(subset), "game_count": len(ids)}
    return result


def _comparison(full_games, agnostic_games, game_ids, indices, confidence):
    """The only new effects: same games, same frozen draws, fixed directions."""
    if set(full_games) != set(game_ids) or set(agnostic_games) != set(game_ids):
        raise ValueError("paired game coverage mismatch")
    def values(games, name):
        return [games[g][name] for g in game_ids]
    if values(full_games, "uniform_kl") != values(agnostic_games, "uniform_kl"):
        raise ValueError("paired Uniform KL mismatch")
    return {
        "Delta_ToM": paired_summary(values(agnostic_games, "kl"), values(full_games, "kl"), indices, confidence),
        "Delta_uniform": {family: paired_summary(values(games, "uniform_kl"), values(games, "kl"), indices, confidence)
                          for family, games in zip(FAMILIES, (full_games, agnostic_games), strict=True)}}


def evaluate_study(path):
    # This must precede opening any prediction, population, or held-out dataset.
    seal = evaluation_eligibility(path)
    study = open_study(path)
    full = study.full
    destination = study.artifact.path.parent / "evaluation"
    existing = (verify_artifact(destination, expected_artifact_type="paper_study_evaluation",
        expected_schema_version=EVALUATION_VERSION) if destination.exists() else None)
    expected_files = {f"agnostic/{c}/{f}/held_out_predictions.jsonl" for c in TEMPORAL_CONDITIONS for f in range(5)}
    if existing is not None and set(existing.manifest["file_table"]) != expected_files:
        raise ValueError("paper prediction file coverage mismatch")
    if (full.runs_path / "checkpoint_set_manifest.json").exists():
        full_seal = training.verify_checkpoint_set(full)
    else:
        full_seal = training.seal_checkpoint_set(full)
    for condition in TEMPORAL_CONDITIONS:
        for fold in range(5):
            evaluation.predict_held_out_fold(full, fold, condition)
    indices, bootstrap_digest = load_bootstrap_plan(full)
    game_ids = full.manifest["game_ids"]
    confidence = full.config.confidence_level
    if indices.shape != (10000, len(game_ids)) or confidence != .95:
        raise ValueError("paper bootstrap binding mismatch")
    predictions, pair_bindings, files, cells, uniform_effects, tom_effects = {}, {}, {}, {}, {}, {}
    # No paired effect is computed until every fold in both conditions has paired exactly.
    selected_cells = {}
    for condition in TEMPORAL_CONDITIONS:
        selected_full, selected_agnostic = [], []
        for fold in range(5):
            full_prediction, full_rows, publication = evaluation.open_predictions(full, fold, condition)
            masks_by_game = {g.game_id: load_held_out_primary(full, fold, g)["rows"] for g in publication.public_view.games}
            masks = {_identity(r): masks_by_game[r["game_id"]][r["boundary_id"]][r["observer"]] for r in full_rows}
            checkpoint = _verify_terminal(study, FAMILIES[1], condition, fold)
            relative = f"agnostic/{condition}/{fold}/held_out_predictions.jsonl"
            if existing is None:
                _, dataset = evaluation._held_out_data(full, fold)
                rows = _predict_agnostic(study, dataset, condition, checkpoint)
                payload = canonical_jsonl_bytes(rows)
            else:
                payload = (existing.path / relative).read_bytes()
                rows = [read_json(line) for line in payload.splitlines()]
                if payload != canonical_jsonl_bytes(rows):
                    raise ValueError("noncanonical agnostic prediction JSONL")
            if any(row["checkpoint_digest"] != checkpoint.manifest_digest for row in rows):
                raise ValueError("agnostic checkpoint binding mismatch")
            key = f"{condition}/{fold}"
            pair_bindings[key] = _pair_rows(full_rows, rows, masks, masks, fold=fold,
                condition=condition, game_ids=full.fold(fold)["held_out_game_ids"])
            primary_full = [r for r in full_rows if r["label_observed"] and masks[_identity(r)]]
            primary_agnostic = [r for r in rows if r["label_observed"] and masks[_identity(r)]]
            primary_digest = sha256_bytes(canonical_json_bytes([list(_identity(r)) for r in primary_full]))
            descriptor = record_with_digest({"artifact_type": "paper_agnostic_held_out_predictions",
                "schema_version": "classic7_paper_agnostic_predictions_v1",
                "experiment_digest": study.agnostic.digest, "fold": fold, "temporal_condition": condition,
                "checkpoint_digest": checkpoint.manifest_digest, "checkpoint_set_digest": seal["record_digest"],
                "prediction_digest": sha256_bytes(payload), "row_count": len(rows), "canonical_shift": 0,
                "primary_row_identity_digest": primary_digest})
            predictions[key] = {"full_prediction_digest": full_prediction["record_digest"],
                "agnostic": descriptor}
            files[relative] = payload
            selected_full.extend(primary_full)
            selected_agnostic.extend(primary_agnostic)
        selected_cells[condition] = (selected_full, selected_agnostic)
    reporting.validate_existing_evaluation(full)
    formal = reporting.aggregate_development_oof(full)
    if formal["bootstrap_digest"] != bootstrap_digest or formal["game_ids"] != game_ids:
        raise ValueError("paper bootstrap binding mismatch")
    for condition in TEMPORAL_CONDITIONS:
        for fold in range(5):
            prediction = predictions[f"{condition}/{fold}"]
            full_report = read_record(training.lineage_path(full, fold, condition) / PRIMARY / "fold_report.json")
            if full_report["row_identity_digest"] != prediction["agnostic"]["primary_row_identity_digest"]:
                raise ValueError("formal Primary selection mismatch")
            prediction["full_primary_report_digest"] = full_report["record_digest"]
        full_selected, agnostic_selected = selected_cells[condition]
        formal_cell = formal["cells"][f"{condition}/{PRIMARY}"]
        full_games = formal_cell["game_scores"]
        _, agnostic_games = evaluation.score_prediction_rows(agnostic_selected, game_ids)
        effects = _comparison(full_games, agnostic_games, game_ids, indices, confidence)
        tom_effects[condition] = effects["Delta_ToM"]
        for family, selected, games in zip(FAMILIES, (full_selected, agnostic_selected), (full_games, agnostic_games), strict=True):
            key = f"{family}/{condition}"
            tv = game_macro_summary([games[g]["total_variation"] for g in game_ids], indices, confidence)
            if family == FAMILIES[0]:
                if tv["point"] != formal_cell["secondary_game_macro_diagnostics"]["total_variation"]:
                    raise ValueError("formal TV point mismatch")
                metrics = {name: {"record_digest": formal_cell["record_digest"], "field": field}
                           for name, field in (("kl", "headline"), ("uniform_kl", "uniform_non_self_reference"))}
            else:
                metrics = {name: game_macro_summary([games[g][name] for g in game_ids], indices, confidence)
                           for name in ("kl", "uniform_kl")}
            cells[key] = {**metrics, "total_variation": tv, "row_count": len(selected), "game_count": len(game_ids),
                          "support_size": _support_summary(selected)}
            uniform_effects[key] = effects["Delta_uniform"][family]
    fields = {"artifact_type": "paper_study_evaluation", "schema_version": EVALUATION_VERSION,
        "study_digest": study.artifact.manifest_digest, "study_seal_digest": seal["record_digest"],
        "source": {k: full.manifest["runtime"][k] for k in ("source_revision", "implementation_digest")},
        "full_checkpoint_set_digest": full_seal["record_digest"],
        "full_report_set_digest": formal["record_digest"], "predictions": predictions,
        "row_identity_binding_digest": sha256_bytes(canonical_json_bytes(pair_bindings)),
        "bootstrap": {"version": BOOTSTRAP_VERSION, "full_experiment_digest": full.digest,
            "game_ids": game_ids, "indices_digest": bootstrap_digest, "replicates": len(indices),
            "confidence": confidence, "interval_method": "percentile_linear"},
        "cells": cells, "Delta_uniform": uniform_effects, "Delta_ToM": tom_effects,
        "Delta_temp": {"record_digest": formal["record_digest"], "field": "paired_headlines.primary_temporal_information_effect"},
        "Delta_stress": {"record_digest": formal["record_digest"], "field": "paired_headlines.identifiability_stress_penalty"}}
    if existing is not None:
        actual = {k: v for k, v in existing.manifest.items() if k not in ("file_table", "manifest_digest")}
        if canonical_json_bytes(actual) != canonical_json_bytes(fields):
            raise ValueError("paper evaluation parent/prediction/metric binding mismatch")
        return existing
    return publish_artifact(destination, manifest_fields=fields, files=files)
