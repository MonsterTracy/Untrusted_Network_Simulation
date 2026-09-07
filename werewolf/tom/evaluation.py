"""Sealed shift-zero inference and two named population evaluations."""

import math

import numpy as np
import torch

from werewolf.artifact_io import canonical_json_bytes, canonical_jsonl_bytes, sha256_bytes
from werewolf.development_publication import open_publication
from werewolf.tom.dataset import CanonicalToMDataset, ExperimentCapacity, PublicTensors
from werewolf.tom.experiment import read_json, validate_runtime
from werewolf.tom.run_records import publish_record, publish_run_bytes, read_record
from werewolf.tom.scoring import row_kl, row_cross_entropy
from werewolf.tom.state import load_model_state
from werewolf.tom.training import verify_checkpoint_set, verify_terminal, lineage_path, build_model
from werewolf.tom.population import load_held_out_primary, load_held_out_all_alive


def _held_out_data(experiment, fold):
    verify_checkpoint_set(experiment)
    validate_runtime(experiment)
    publication = open_publication(experiment.manifest["publication_path"], game_ids=experiment.fold(fold)["held_out_game_ids"])
    if publication.manifest_digest != experiment.manifest["publication_digest"]:
        raise ValueError("held-out publication parent mismatch")
    data = CanonicalToMDataset(publication.public_view, publication.public_view.game_ids, ExperimentCapacity(experiment.config.max_seq_len))
    return publication, data


def predict_held_out_fold(experiment, fold, temporal_condition):
    seal = verify_checkpoint_set(experiment)
    checkpoint = verify_terminal(experiment, fold, temporal_condition)
    lineage = lineage_path(experiment, fold, temporal_condition)
    if (lineage / "prediction_manifest.json").exists():
        return open_predictions(experiment, fold, temporal_condition)[0]
    _, dataset = _held_out_data(experiment, fold)
    model = build_model(experiment, temporal_condition)
    load_model_state(checkpoint.path, model, checkpoint.manifest_digest)
    model.eval()
    rows = []
    with torch.inference_mode():
        for sample in dataset:
            public = PublicTensors.stack([sample.public])
            logp = model(**{k: v.to(experiment.config.device) for k, v in public.kwargs().items()})[0].cpu()
            for observer in range(7):
                rows.append({"game_id": sample.game_id, "boundary_id": sample.boundary_id,
                    "observer": observer, "prefix_digest": sample.prefix_digest, "plan_digest": sample.plan_digest,
                    "probability": logp[observer].exp().tolist(),
                    "non_self_log_probability": [logp[observer, j].item() for j in range(7) if j != observer],
                    "q": sample.q[observer].tolist(), "label_observed": bool(sample.label_observed[observer]),
                    "observer_alive": bool(sample.observer_alive[observer]),
                    "checkpoint_digest": checkpoint.manifest_digest, "temporal_condition": temporal_condition})
    payload = canonical_jsonl_bytes(rows)
    publish_run_bytes(lineage / "held_out_predictions.jsonl", payload)
    manifest = publish_record(lineage / "prediction_manifest.json", {
        "artifact_type": "held_out_predictions", "schema_version": "classic7_predictions_v1",
        "experiment_digest": experiment.digest, "fold": fold, "temporal_condition": temporal_condition,
        "checkpoint_digest": checkpoint.manifest_digest, "checkpoint_set_digest": seal["record_digest"],
        "prediction_digest": sha256_bytes(payload), "row_count": len(rows), "canonical_shift": 0})
    open_predictions(experiment, fold, temporal_condition)
    return manifest


def open_predictions(experiment, fold, condition):
    seal = verify_checkpoint_set(experiment)
    checkpoint = verify_terminal(experiment, fold, condition)
    lineage = lineage_path(experiment, fold, condition)
    manifest = read_record(lineage / "prediction_manifest.json")
    expected = {"artifact_type": "held_out_predictions", "schema_version": "classic7_predictions_v1",
        "experiment_digest": experiment.digest, "fold": fold, "temporal_condition": condition,
        "checkpoint_digest": checkpoint.manifest_digest, "checkpoint_set_digest": seal["record_digest"], "canonical_shift": 0}
    if set(manifest) != set(expected) | {"prediction_digest", "row_count", "record_digest"} or any(manifest[k] != v for k, v in expected.items()):
        raise ValueError("prediction provenance mismatch")
    payload = (lineage / "held_out_predictions.jsonl").read_bytes()
    if sha256_bytes(payload) != manifest["prediction_digest"]:
        raise ValueError("prediction digest mismatch")
    rows = [read_json(line) for line in payload.splitlines()]
    publication, dataset = _held_out_data(experiment, fold)
    if len(rows) != len(dataset) * 7 or manifest["row_count"] != len(rows):
        raise ValueError("prediction row coverage mismatch")
    for sample, batch in zip(dataset, [rows[i:i+7] for i in range(0, len(rows), 7)], strict=True):
        for observer, row in enumerate(batch):
            expected_row = {"game_id": sample.game_id, "boundary_id": sample.boundary_id, "observer": observer,
                "prefix_digest": sample.prefix_digest, "plan_digest": sample.plan_digest,
                "q": sample.q[observer].tolist(), "label_observed": bool(sample.label_observed[observer]),
                "observer_alive": bool(sample.observer_alive[observer]), "checkpoint_digest": checkpoint.manifest_digest,
                "temporal_condition": condition}
            if set(row) != set(expected_row) | {"probability", "non_self_log_probability"} or any(row[k] != v for k, v in expected_row.items()):
                raise ValueError("prediction row identity/target mismatch")
            p, logs = np.asarray(row["probability"]), np.asarray(row["non_self_log_probability"])
            if p.shape != (7,) or logs.shape != (6,) or not np.isfinite(p).all() or not np.isfinite(logs).all() or p[observer] != 0 or np.any(p < 0) or not np.isclose(p.sum(), 1, atol=1e-6):
                raise ValueError("prediction violates Non-Self Suspicion Simplex")
            if not np.allclose(np.delete(p, observer), np.exp(logs), atol=1e-7):
                raise ValueError("prediction probability/log-probability mismatch")
    return manifest, rows, publication


def _fold_report_value(experiment, fold, condition, fields, load_masks):
    prediction, rows, publication = open_predictions(experiment, fold, condition)
    masks = {g.game_id: load_masks(experiment, fold, g)["rows"] for g in publication.public_view.games}
    selected = [r for r in rows if r["label_observed"] and masks[r["game_id"]][r["boundary_id"]][r["observer"]]]
    scores = []
    for row in selected:
        q = torch.tensor(row["q"], dtype=torch.float64)
        logp = torch.empty(7, dtype=torch.float64)
        logp[row["observer"]] = -torch.inf
        logp[[j for j in range(7) if j != row["observer"]]] = torch.tensor(row["non_self_log_probability"], dtype=torch.float64)
        uniform = torch.full((7,), -math.log(6), dtype=torch.float64)
        uniform[row["observer"]] = -torch.inf
        p = logp.exp()
        scores.append({"game_id": row["game_id"], "boundary_id": row["boundary_id"], "observer": row["observer"],
            "prefix_digest": row["prefix_digest"], "kl": row_kl(logp, q).item(), "uniform_kl": row_kl(uniform, q).item(),
            "cross_entropy": row_cross_entropy(logp, q).item(), "total_variation": (.5 * (q-p).abs().sum()).item(),
            "mean_absolute_error": (q-p).abs().mean().item(), "argmax_in_target_support": bool(q[p.argmax()] > 0),
            "target_support_probability": p[q > 0].sum().item()})
        # All-six support has exactly zero theoretical reference KL. Do not
        # manufacture a denominator from float32 round-off on uniform targets.
        scores[-1]["normalized_reducible_gap_improvement"] = (
            (scores[-1]["uniform_kl"] - scores[-1]["kl"]) / scores[-1]["uniform_kl"]
            if int((q > 0).sum()) < 6 else None)
    game_scores = {}
    for game_id in publication.public_view.game_ids:
        values = [r for r in scores if r["game_id"] == game_id]
        if not values:
            raise ValueError("game has zero effective evaluation rows")
        game_scores[game_id] = {"row_count": len(values), **{name: float(np.mean([r[name] for r in values]))
            for name in ("kl", "uniform_kl", "cross_entropy", "total_variation", "mean_absolute_error", "argmax_in_target_support", "target_support_probability")}}
    return {
        "schema_version": "classic7_named_fold_report_v1", **fields, "experiment_digest": experiment.digest,
        "fold": fold, "temporal_condition": condition, "checkpoint_digest": prediction["checkpoint_digest"],
        "prediction_digest": prediction["prediction_digest"], "checkpoint_set_digest": prediction["checkpoint_set_digest"],
        "row_identity_digest": sha256_bytes(canonical_json_bytes([[r["game_id"], r["boundary_id"], r["observer"]] for r in scores])),
        "game_scores": game_scores, "row_scores": scores}


def primary_fold_report_value(experiment, fold, temporal_condition):
    from werewolf.tom.population import PRIMARY_SELECTOR_VERSION
    return _fold_report_value(experiment, fold, temporal_condition, {
        "artifact_type": "primary_development_oof_fold", "population_identity": "non_wolf_alive",
        "role_sidecar_digest": experiment.manifest["primary_sidecar_digest"], "population_selector_version": PRIMARY_SELECTOR_VERSION}, load_held_out_primary)


def all_alive_fold_report_value(experiment, fold, temporal_condition):
    from werewolf.tom.population import ALL_ALIVE_VERSION
    return _fold_report_value(experiment, fold, temporal_condition, {
        "artifact_type": "all_alive_identifiability_stress_fold", "population_identity": "all_alive",
        "population_selector_version": ALL_ALIVE_VERSION}, load_held_out_all_alive)


def evaluate_primary(experiment, fold, temporal_condition):
    return publish_record(lineage_path(experiment, fold, temporal_condition) / "primary_development_oof" / "fold_report.json",
        primary_fold_report_value(experiment, fold, temporal_condition))


def evaluate_all_alive(experiment, fold, temporal_condition):
    return publish_record(lineage_path(experiment, fold, temporal_condition) / "all_alive_identifiability_stress" / "fold_report.json",
        all_alive_fold_report_value(experiment, fold, temporal_condition))
