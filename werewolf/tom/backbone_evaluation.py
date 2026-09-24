"""Read-only historical backbone training validation and independent OOF evaluation."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import torch

from werewolf.artifact_io import canonical_json_bytes, canonical_jsonl_bytes, publish_artifact, sha256_bytes, verify_artifact
from werewolf.development_publication import open_publication
from werewolf.tom import backbone_execution as training, evaluation
from werewolf.tom.backbone_model import ARCHITECTURES
from werewolf.tom.dataset import CanonicalToMDataset, ExperimentCapacity, PublicTensors
from werewolf.tom.experiment import ExperimentConfig
from werewolf.tom.protocol import BOOTSTRAP_VERSION, bootstrap_indices
from werewolf.tom.scoring import game_macro_summary, paired_summary

VERSION = "classic7_backbone_oof_evaluation_v1"
PREDICTION_VERSION = "classic7_backbone_oof_predictions_v1"
REPORT_VERSION = "classic7_backbone_oof_report_v1"
CONDITION = "explicit_day_phase"
TRAINING_REVISION = "ebf98b7e3249f78d3f1fa1fd46c20affe0486913"
INFERENCE_SOURCES = (
    "werewolf/artifact_io/__init__.py",
    "werewolf/artifact_io/canonical.py",
    "werewolf/artifact_io/tensor_state.py",
    "werewolf/canonical_collection/__init__.py",
    "werewolf/canonical_collection/attempt_ledger.py",
    "werewolf/canonical_collection/failure_evidence.py",
    "werewolf/canonical_collection/frozen_json.py",
    "werewolf/canonical_collection/game_bundle.py",
    "werewolf/canonical_collection/pre.py",
    "werewolf/canonical_collection/public_history.py",
    "werewolf/canonical_collection/speech.py",
    "werewolf/canonical_collection/trajectory_evidence.py",
    "werewolf/development_publication.py",
    "werewolf/structured_history.py",
    "werewolf/tom/backbone_execution.py",
    "werewolf/tom/backbone_model.py",
    "werewolf/tom/backbone_study.py",
    "werewolf/tom/dataset.py",
    "werewolf/tom/experiment.py",
    "werewolf/tom/model.py",
    "werewolf/tom/population.py",
    "werewolf/tom/protocol.py",
    "werewolf/tom/state.py",
    "werewolf/tom/temporal.py",
    "werewolf/tom/training.py",
)


def _same(a, b):
    return canonical_json_bytes(a) == canonical_json_bytes(b)


@dataclass(frozen=True)
class HistoricalTraining:
    execution: training.Execution
    seal: object


def open_sealed_training(root, execution_digest, seal_digest, training_revision):
    """Read the exact historical training artifacts without current-source attestation.

    These three externally supplied identities are mandatory trust anchors. This
    reader does not call the source-strict training open/terminal entry points.
    """
    root = Path(root).resolve(strict=True)
    contract = verify_artifact(root / "contract", expected_artifact_type="backbone_execution",
                               expected_schema_version=training.EXECUTION_VERSION)
    m = contract.manifest
    if (contract.manifest_digest != execution_digest
            or set(m) != {"artifact_type", "schema_version", "engineering_only", "prepared_path", "preparation",
                          "execution_source", "runtime", "lineages", "training_rng_version", "recovery_cadence",
                          "expected_optimizer_steps", "file_table", "manifest_digest"}
            or m["engineering_only"] is not False or m["file_table"]
            or m["lineages"] != [list(x) for x in training.LINEAGES]
            or m["training_rng_version"] != training.RNG_VERSION
            or type(m["expected_optimizer_steps"]) is not int or m["expected_optimizer_steps"] != 25200
            or type(m["recovery_cadence"]) is not int or m["recovery_cadence"] != 1000
            or set(m["execution_source"]) != {"source_revision", "implementation_digest"}
            or training_revision != TRAINING_REVISION
            or m["execution_source"]["source_revision"] != training_revision
            or m["execution_source"]["implementation_digest"] != m["runtime"]["implementation_digest"]):
        raise ValueError("historical training contract identity/control mismatch")
    snapshot = training.validate_frozen_snapshot(m["prepared_path"], m["preparation"])
    expected_root = snapshot.path.parent / "executions" / snapshot.manifest_digest
    if root != expected_root or m["runtime"]["source_revision"] != snapshot.manifest["runtime"]["source_revision"]:
        raise ValueError("historical training ownership/runtime mismatch")
    if not _same({k: v for k, v in m["runtime"].items() if k != "implementation_digest"},
                 {k: v for k, v in snapshot.manifest["runtime"].items() if k != "implementation_digest"}):
        raise ValueError("historical preparation/execution runtime mismatch")
    config = ExperimentConfig(**snapshot.manifest["protocol_inputs"]["design"]["reference_protocol"])
    if config.recovery_cadence != 1000 or any(f["optimizer_steps"] != 25200 for f in snapshot.manifest["folds"]):
        raise ValueError("historical formal budget mismatch")
    seal = verify_artifact(root / "seal", expected_artifact_type="backbone_execution_seal",
                           expected_schema_version=training.SEAL_VERSION)
    s = seal.manifest
    expected = {"artifact_type": "backbone_execution_seal", "schema_version": training.SEAL_VERSION,
                "execution_digest": execution_digest, "preparation": m["preparation"],
                "execution_source": m["execution_source"],
                "graphs": snapshot.manifest["protocol_inputs"]["graphs"],
                "initializations": snapshot.manifest["initializations"]}
    if (seal.manifest_digest != seal_digest or s["file_table"]
            or set(s) != set(expected) | {"terminals", "file_table", "manifest_digest"}
            or any(not _same(s[k], v) for k, v in expected.items())
            or len(s["terminals"]) != 20
            or [(v["architecture"], v["fold"], v["temporal_condition"]) for v in s["terminals"]] != list(training.LINEAGES)
            or any(set(v) != {"architecture", "fold", "temporal_condition", "terminal_digest"}
                   for v in s["terminals"])):
        raise ValueError("historical seal identity/lineage mismatch")
    runs = root / "runs"
    if (not runs.is_dir() or {p.name for p in runs.iterdir()} != set(ARCHITECTURES)
            or any({p.name for p in (runs / a).iterdir()} != {str(f) for f in range(5)}
                   for a in ARCHITECTURES)):
        raise ValueError("historical sealed run inventory mismatch")
    return HistoricalTraining(training.Execution(contract, snapshot, config), seal)


def validate_historical_terminal(owner, architecture, fold):
    """Verify the full recovery chain and terminal without training's HEAD gate."""
    execution = owner.execution
    lineage = training._lineage(execution, architecture, fold)
    if {p.name for p in lineage.iterdir()} != {"recovery", "terminal"}:
        raise ValueError("historical terminal ownership mismatch")
    model, optimizer = training._model_optimizer(execution, architecture, fold)
    logs, ancestry = training._recovery_chain(execution, architecture, fold, model, optimizer)
    if len(logs) != 25200 or not ancestry or ancestry[-1]["step"] != 25200:
        raise ValueError("historical terminal incomplete")
    checkpoint = verify_artifact(lineage / "recovery" / "25200", expected_artifact_type="backbone_recovery",
                                 expected_schema_version=training.CHECKPOINT_VERSION)
    expected = training._terminal_fields(execution, architecture, fold, checkpoint, ancestry)
    terminal = verify_artifact(lineage / "terminal", expected_artifact_type="backbone_terminal",
                               expected_schema_version=training.TERMINAL_VERSION)
    actual = {k: v for k, v in terminal.manifest.items() if k not in {"manifest_digest", "file_table"}}
    row = owner.seal.manifest["terminals"][training.LINEAGES.index((architecture, fold, CONDITION))]
    if terminal.manifest["file_table"] or not _same(actual, expected) or terminal.manifest_digest != row["terminal_digest"]:
        raise ValueError("historical terminal/seal mismatch")
    return terminal, model


def _evaluation_source(owner):
    source, runtime = training._runtime(owner.execution.config)
    if source["implementation_digest"] != runtime["implementation_digest"]:
        raise ValueError("evaluation source/runtime mismatch")
    recorded = owner.execution.manifest["runtime"]
    if not _same({k: v for k, v in runtime.items() if k != "implementation_digest"},
                 {k: v for k, v in recorded.items() if k != "implementation_digest"}):
        raise ValueError("evaluation runtime differs from frozen training runtime")
    revision = owner.execution.manifest["execution_source"]["source_revision"]
    if revision != TRAINING_REVISION or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("invalid historical training revision")
    project = Path(__file__).resolve().parents[2]
    for relative in INFERENCE_SOURCES:
        try:
            frozen = subprocess.check_output(["git", "-C", str(project), "show", f"{revision}:{relative}"],
                                              env={k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
                                              stderr=subprocess.PIPE)
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError("historical inference source unavailable") from error
        if frozen != (project / relative).read_bytes():
            raise ValueError("historical inference source changed")
    return source, runtime


def prepare_evaluation(root, execution_digest, seal_digest, training_revision):
    owner = open_sealed_training(root, execution_digest, seal_digest, training_revision)
    source, runtime = _evaluation_source(owner)
    for architecture, fold, _ in training.LINEAGES:
        validate_historical_terminal(owner, architecture, fold)
    destination = owner.execution.snapshot.path.parent / "evaluations" / execution_digest / "contract"
    artifact = publish_artifact(destination, manifest_fields={
        "artifact_type": "backbone_oof_evaluation_contract", "schema_version": VERSION,
        "training_path": str(owner.execution.root), "training_execution_digest": execution_digest,
        "training_seal_digest": seal_digest, "training_revision": training_revision,
        "preparation": owner.execution.manifest["preparation"],
        "training_source": owner.execution.manifest["execution_source"],
        "evaluation_source": source, "evaluation_runtime": runtime,
    }, files={})
    return artifact


def open_evaluation(root):
    root = Path(root).resolve(strict=True)
    artifact = verify_artifact(root / "contract", expected_artifact_type="backbone_oof_evaluation_contract",
                               expected_schema_version=VERSION)
    m = artifact.manifest
    if (set(m) != {"artifact_type", "schema_version", "training_path", "training_execution_digest",
                  "training_seal_digest", "training_revision", "preparation", "training_source",
                  "evaluation_source", "evaluation_runtime", "file_table", "manifest_digest"} or m["file_table"]):
        raise ValueError("evaluation contract schema mismatch")
    owner = open_sealed_training(m["training_path"], m["training_execution_digest"],
                                 m["training_seal_digest"], m["training_revision"])
    expected_root = owner.execution.snapshot.path.parent / "evaluations" / owner.execution.artifact.manifest_digest
    source, runtime = _evaluation_source(owner)
    if (root != expected_root or not _same(m["preparation"], owner.execution.manifest["preparation"])
            or not _same(m["training_source"], owner.execution.manifest["execution_source"])
            or not _same(m["evaluation_source"], source) or not _same(m["evaluation_runtime"], runtime)):
        raise ValueError("evaluation provenance changed")
    return artifact, owner


def _held_out(owner, fold):
    execution = owner.execution
    record = execution.snapshot.manifest["folds"][fold]
    ids = record["held_out_game_ids"]
    publication = open_publication(execution.snapshot.manifest["publication_path"], game_ids=ids)
    if publication.manifest_digest != execution.snapshot.manifest["protocol_inputs"]["publication_digest"]:
        raise ValueError("held-out publication binding mismatch")
    dataset = CanonicalToMDataset(publication.public_view, ids, ExperimentCapacity(execution.config.max_seq_len))
    masks = json.loads((execution.snapshot.path / "primary.json").read_bytes())["rows"]
    return dataset, {g: masks[g] for g in ids}


def _prediction_payload(owner, terminal, model, dataset, masks):
    """Canonical held-out inference from the terminal-restored model."""
    model.eval()
    rows, primary = [], []
    with torch.inference_mode():
        for sample in dataset:
            public = PublicTensors.stack([sample.public])
            logp = model(**{k: v.to(owner.execution.config.device) for k, v in public.kwargs().items()})[0].cpu()
            for observer in range(7):
                row = {"game_id": sample.game_id, "boundary_id": sample.boundary_id,
                       "observer": observer, "prefix_digest": sample.prefix_digest, "plan_digest": sample.plan_digest,
                       "probability": logp[observer].exp().tolist(),
                       "non_self_log_probability": [logp[observer, j].item() for j in range(7) if j != observer],
                       "q": sample.q[observer].tolist(), "label_observed": bool(sample.label_observed[observer]),
                       "observer_alive": bool(sample.observer_alive[observer]),
                       "checkpoint_digest": terminal.manifest["checkpoint_ancestry"][-1]["digest"],
                       "temporal_condition": CONDITION}
                rows.append(row)
                if row["label_observed"] and masks[sample.game_id][sample.boundary_id][observer]:
                    primary.append([sample.game_id, sample.boundary_id, observer])
    return canonical_jsonl_bytes(rows), sha256_bytes(canonical_json_bytes(primary)), len(rows)


def evaluate_fold(root, architecture, fold):
    contract, owner = open_evaluation(root)
    if type(fold) is not int or (architecture, fold, CONDITION) not in training.LINEAGES:
        raise ValueError("unknown backbone OOF lineage")
    terminal, model = validate_historical_terminal(owner, architecture, fold)
    dataset, masks = _held_out(owner, fold)
    payload, primary_digest, row_count = _prediction_payload(owner, terminal, model, dataset, masks)
    return publish_artifact(Path(root) / "folds" / architecture / str(fold), manifest_fields={
        "artifact_type": "backbone_oof_fold_predictions", "schema_version": PREDICTION_VERSION,
        "evaluation_digest": contract.manifest_digest, "training_seal_digest": owner.seal.manifest_digest,
        "architecture": architecture, "fold": fold, "temporal_condition": CONDITION,
        "terminal_digest": terminal.manifest_digest,
        "checkpoint_digest": terminal.manifest["checkpoint_ancestry"][-1]["digest"],
        "row_count": row_count, "primary_row_identity_digest": primary_digest,
    }, files={"predictions.jsonl": payload})


def _verify_model_predictions(owner, artifact, terminal, model, dataset, masks):
    payload, primary_digest, row_count = _prediction_payload(owner, terminal, model, dataset, masks)
    if (sha256_bytes(payload) != artifact.manifest["file_table"]["predictions.jsonl"]["sha256"]
            or primary_digest != artifact.manifest["primary_row_identity_digest"]
            or row_count != artifact.manifest["row_count"]):
        raise ValueError("OOF prediction differs from terminal model inference")


def _validated_fold(owner, contract, architecture, fold, terminal, expected_dataset, masks):
    artifact = verify_artifact(contract.path.parent / "folds" / architecture / str(fold),
        expected_artifact_type="backbone_oof_fold_predictions", expected_schema_version=PREDICTION_VERSION)
    m = artifact.manifest
    checkpoint_digest = terminal.manifest["checkpoint_ancestry"][-1]["digest"]
    expected = {"artifact_type": "backbone_oof_fold_predictions", "schema_version": PREDICTION_VERSION,
        "evaluation_digest": contract.manifest_digest, "training_seal_digest": owner.seal.manifest_digest,
        "architecture": architecture, "fold": fold, "temporal_condition": CONDITION,
        "terminal_digest": terminal.manifest_digest, "checkpoint_digest": checkpoint_digest}
    if (set(m) != set(expected) | {"row_count", "primary_row_identity_digest", "file_table", "manifest_digest"}
            or any(m[k] != v for k, v in expected.items()) or set(m["file_table"]) != {"predictions.jsonl"}):
        raise ValueError("OOF fold prediction provenance mismatch")
    raw = (artifact.path / "predictions.jsonl").read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    if raw != canonical_jsonl_bytes(rows) or len(rows) != m["row_count"] or len(rows) != len(expected_dataset) * 7:
        raise ValueError("OOF fold prediction coverage/serialization mismatch")
    selected, selected_ids = [], []
    for sample, batch in zip(expected_dataset, (rows[i:i+7] for i in range(0, len(rows), 7)), strict=True):
        for observer, row in enumerate(batch):
            expected_row = {"game_id": sample.game_id, "boundary_id": sample.boundary_id,
                "observer": observer, "prefix_digest": sample.prefix_digest, "plan_digest": sample.plan_digest,
                "q": sample.q[observer].tolist(), "label_observed": bool(sample.label_observed[observer]),
                "observer_alive": bool(sample.observer_alive[observer]),
                "checkpoint_digest": checkpoint_digest, "temporal_condition": CONDITION}
            if set(row) != set(expected_row) | {"probability", "non_self_log_probability"} or any(row[k] != v for k, v in expected_row.items()):
                raise ValueError("OOF PRE/target/observer mismatch")
            p, logs = np.asarray(row["probability"]), np.asarray(row["non_self_log_probability"])
            if (p.shape != (7,) or logs.shape != (6,) or not np.isfinite(p).all() or not np.isfinite(logs).all()
                    or p[observer] != 0 or np.any(p < 0) or not np.isclose(p.sum(), 1, atol=1e-6)
                    or not np.allclose(np.delete(p, observer), np.exp(logs), atol=1e-7)):
                raise ValueError("OOF non-self prediction invalid")
            if row["label_observed"] and masks[sample.game_id][sample.boundary_id][observer]:
                selected.append(row)
                selected_ids.append([sample.game_id, sample.boundary_id, observer])
    if sha256_bytes(canonical_json_bytes(selected_ids)) != m["primary_row_identity_digest"]:
        raise ValueError("OOF Primary identity mismatch")
    return artifact, selected


def _top1_nu(counts, excluded, total_rows, total_games):
    means = [counts[g][0] / counts[g][1] for g in sorted(counts)]
    return {"top1_nu": sum(means) / len(means) if means else None,
        "total_primary_row_count": total_rows,
        "valid_nu_row_count": sum(total for _, total in counts.values()),
        "excluded_s6_row_count": excluded,
        "total_primary_game_count": total_games, "valid_nu_game_count": len(counts)}


def _accumulate_top1_nu(counts, rows, scores):
    excluded = 0
    for row, score in zip(rows, scores, strict=True):
        support = sum(q > 0 for q in row["q"])
        if not 1 <= support <= 6:
            raise ValueError("invalid Top1_NU support")
        if support == 6:
            excluded += 1
        else:
            hits, total = counts.get(row["game_id"], (0, 0))
            counts[row["game_id"]] = (hits + int(score["argmax_in_target_support"]), total + 1)
    return excluded


def _summarize_cells(scores_by_arch, game_ids, indices, confidence):
    if any(set(scores_by_arch[a]) != set(game_ids) for a in ARCHITECTURES):
        raise ValueError("incomplete OOF game coverage")
    uniform = [scores_by_arch[ARCHITECTURES[0]][g]["uniform_kl"] for g in game_ids]
    cells = {}
    for architecture in ARCHITECTURES:
        scores = scores_by_arch[architecture]
        if [scores[g]["uniform_kl"] for g in game_ids] != uniform:
            raise ValueError("architecture uniform reference differs")
        kl = [scores[g]["kl"] for g in game_ids]
        tv = [scores[g]["total_variation"] for g in game_ids]
        cells[architecture] = {"kl": game_macro_summary(kl, indices, confidence),
            "total_variation": game_macro_summary(tv, indices, confidence),
            "uniform_kl": game_macro_summary(uniform, indices, confidence),
            "Delta_uniform": paired_summary(uniform, kl, indices, confidence),
            "row_count": sum(scores[g]["row_count"] for g in game_ids), "game_count": len(game_ids)}
    return cells


def seal_evaluation(root):
    contract, owner = open_evaluation(root)
    execution = owner.execution
    fold_root = contract.path.parent / "folds"
    if {p.name for p in fold_root.iterdir()} != set(ARCHITECTURES) or any(
            {p.name for p in (fold_root / a).iterdir()} != {str(f) for f in range(5)} for a in ARCHITECTURES):
        raise ValueError("OOF evaluation requires exactly twenty fold artifacts")
    game_ids = sorted({g for row in execution.snapshot.manifest["folds"] for g in row["held_out_game_ids"]})
    if len(game_ids) != sum(len(row["held_out_game_ids"]) for row in execution.snapshot.manifest["folds"]):
        raise ValueError("OOF held-out game overlap")
    config = execution.config
    if config.bootstrap_replicates != 10000 or config.confidence_level != .95:
        raise ValueError("frozen bootstrap controls mismatch")
    indices = bootstrap_indices(game_ids, config.bootstrap_replicates, config.bootstrap_seed)
    draw_bytes = indices.tobytes()
    scores_by_arch = {a: {} for a in ARCHITECTURES}
    top_counts = {a: {} for a in ARCHITECTURES}
    top_excluded = {a: 0 for a in ARCHITECTURES}
    descriptors, row_bindings = [], {}
    for fold in range(5):
        dataset, masks = _held_out(owner, fold)
        fold_ids = execution.snapshot.manifest["folds"][fold]["held_out_game_ids"]
        reference = None
        for architecture in ARCHITECTURES:
            terminal, model = validate_historical_terminal(owner, architecture, fold)
            artifact, selected = _validated_fold(owner, contract, architecture, fold, terminal, dataset, masks)
            _verify_model_predictions(owner, artifact, terminal, model, dataset, masks)
            binding = artifact.manifest["primary_row_identity_digest"]
            if reference is not None and binding != reference:
                raise ValueError("architecture Primary row identities differ")
            reference = binding
            row_scores, scores = evaluation.score_prediction_rows(selected, fold_ids)
            if set(scores_by_arch[architecture]).intersection(scores):
                raise ValueError("duplicate OOF game")
            scores_by_arch[architecture].update(scores)
            if len(selected) != sum(v["row_count"] for v in scores.values()):
                raise ValueError("OOF score row count mismatch")
            top_excluded[architecture] += _accumulate_top1_nu(top_counts[architecture], selected, row_scores)
            descriptors.append({"architecture": architecture, "fold": fold,
                                "prediction_digest": artifact.manifest_digest,
                                "terminal_digest": terminal.manifest_digest})
        row_bindings[str(fold)] = reference
    cells = _summarize_cells(scores_by_arch, game_ids, indices, .95)
    supplementary = {a: _top1_nu(top_counts[a], top_excluded[a], cells[a]["row_count"], len(game_ids))
                     for a in ARCHITECTURES}
    return publish_artifact(contract.path.parent / "seal", manifest_fields={
        "artifact_type": "backbone_oof_evaluation_report", "schema_version": REPORT_VERSION,
        "evaluation_digest": contract.manifest_digest,
        "training_execution_digest": execution.artifact.manifest_digest,
        "training_seal_digest": owner.seal.manifest_digest,
        "preparation": execution.manifest["preparation"], "training_source": execution.manifest["execution_source"],
        "evaluation_source": contract.manifest["evaluation_source"],
        "predictions": descriptors, "primary_row_bindings": row_bindings,
        "bootstrap": {"version": BOOTSTRAP_VERSION, "seed": config.bootstrap_seed,
                      "game_ids": game_ids, "indices_digest": sha256_bytes(draw_bytes),
                      "replicates": config.bootstrap_replicates, "confidence": .95,
                      "sampling_unit": "game", "interval_method": "percentile_linear"},
        "cells": cells, "supplementary_top1_nu": supplementary,
    }, files={"bootstrap_indices.bin": draw_bytes})
