"""Synthetic public games and sealed manifests; no models or real artifacts."""

import json
import math
from pathlib import Path

import pytest

from tests.development_publication.test_development_publication import _publication
from werewolf.artifact_io import canonical_json_bytes, canonical_jsonl_bytes, publish_artifact
from werewolf.tom.dataset import CanonicalToMDataset, ExperimentCapacity
from werewolf.tom.population import select_primary_population
from werewolf.tom.protocol import bootstrap_indices, BOOTSTRAP_VERSION
from werewolf.tom_llm.common import digest
from werewolf.tom_llm import reference as adapter
from werewolf.tom_llm.reference import open_reference


@pytest.fixture(autouse=True)
def tiny_historical_counts(monkeypatch):
    # Only this synthetic fixture changes the existing reader's size assertions.
    monkeypatch.setattr(adapter.historical, "GAME_COUNT", 5)
    monkeypatch.setattr(adapter.historical, "REPLICATES", 20)


@pytest.fixture
def sealed_reference(tmp_path):
    _, handles = _publication(tmp_path / "fixture")
    publication = handles.public
    view = publication.public_view
    population = select_primary_population(publication)
    masks = {"metadata": population.metadata, "rows": population.rows}
    games = sorted(view.game_ids)
    folds = [{"fold": f.fold_index, "held_out_game_ids": list(f.game_ids),
              "training_game_ids": [g for g in games if g not in f.game_ids]} for f in view.fold_manifest.folds]
    source = {"source_revision": "c" * 40, "implementation_digest": "d" * 64}
    base = tmp_path / "historical"
    prepared = publish_artifact(base / "prepared", manifest_fields={
        "artifact_type": "backbone_tom_study", "schema_version": adapter.historical.STUDY_VERSION,
        "study_digest": "e" * 64,
        "protocol_inputs": {"source": source, "primary_digest": digest(masks), "primary_identity": population.metadata,
                           "publication_digest": publication.manifest_digest,
                           "fold_manifest_digest": view.fold_manifest.manifest_digest},
        "folds": folds}, files={"primary.json": canonical_json_bytes(masks)})
    preparation = {"manifest_digest": prepared.manifest_digest, "study_digest": "e" * 64, **source}
    training_source = {"source_revision": "f" * 40, "implementation_digest": "0" * 64}
    training_root = base / "executions" / prepared.manifest_digest
    execution = publish_artifact(training_root / "contract", manifest_fields={
        "artifact_type": "backbone_execution", "schema_version": adapter.historical.EXECUTION_VERSION,
        "prepared_path": str(prepared.path), "preparation": preparation,
        "execution_source": training_source}, files={})
    terminals = []
    for architecture in adapter.historical.ARCHITECTURES:
        for fold in range(5):
            checkpoint = digest([architecture, fold, "checkpoint"])
            terminal = publish_artifact(training_root / "runs" / architecture / str(fold) / "terminal",
                manifest_fields={"artifact_type": "backbone_terminal", "schema_version": adapter.historical.TERMINAL_VERSION,
                                 "checkpoint_ancestry": [{"step": 25200, "digest": checkpoint}]}, files={})
            terminals.append({"architecture": architecture, "fold": fold, "temporal_condition": "explicit_day_phase",
                              "terminal_digest": terminal.manifest_digest})
    train_seal = publish_artifact(training_root / "seal", manifest_fields={
        "artifact_type": "backbone_execution_seal", "schema_version": adapter.historical.SEAL_VERSION,
        "execution_digest": execution.manifest_digest, "preparation": preparation,
        "execution_source": training_source, "terminals": terminals}, files={})
    evaluation_source = {"source_revision": "1" * 40, "implementation_digest": "2" * 64}
    root = base / "evaluations" / execution.manifest_digest
    contract = publish_artifact(root / "contract", manifest_fields={
        "artifact_type": "backbone_oof_evaluation_contract", "schema_version": adapter.historical.EVALUATION_VERSION,
        "training_path": str(training_root), "training_execution_digest": execution.manifest_digest,
        "training_seal_digest": train_seal.manifest_digest, "preparation": preparation,
        "training_source": training_source, "evaluation_source": evaluation_source}, files={})
    descriptors, bindings = [], {}
    for fold, entry in enumerate(folds):
        data = CanonicalToMDataset(view, entry["held_out_game_ids"], ExperimentCapacity(1024))
        ids = [[s.game_id, s.boundary_id, observer] for s in data for observer in range(7)
               if s.label_observed[observer] and population.rows[s.game_id][s.boundary_id][observer]]
        bindings[str(fold)] = digest(ids)
        for architecture in adapter.historical.ARCHITECTURES:
            rows = []
            checkpoint = digest([architecture, fold, "checkpoint"])
            for sample in data:
                for observer in range(7):
                    rows.append({"game_id": sample.game_id, "boundary_id": sample.boundary_id, "observer": observer,
                        "prefix_digest": sample.prefix_digest, "plan_digest": sample.plan_digest,
                        "q": sample.q[observer].tolist(), "label_observed": bool(sample.label_observed[observer]),
                        "observer_alive": bool(sample.observer_alive[observer]), "checkpoint_digest": checkpoint,
                        "temporal_condition": "explicit_day_phase",
                        "probability": [0. if j == observer else 1/6 for j in range(7)],
                        "non_self_log_probability": [-math.log(6)] * 6})
            terminal = next(t for t in terminals if t["architecture"] == architecture and t["fold"] == fold)
            artifact = publish_artifact(root / "folds" / architecture / str(fold), manifest_fields={
                "artifact_type": "backbone_oof_fold_predictions", "schema_version": adapter.historical.PREDICTION_VERSION,
                "evaluation_digest": contract.manifest_digest, "training_seal_digest": train_seal.manifest_digest,
                "architecture": architecture, "fold": fold, "temporal_condition": "explicit_day_phase",
                "terminal_digest": terminal["terminal_digest"], "checkpoint_digest": checkpoint,
                "row_count": len(rows), "primary_row_identity_digest": digest(ids)},
                files={"predictions.jsonl": canonical_jsonl_bytes(rows)})
            descriptors.append({"architecture": architecture, "fold": fold, "prediction_digest": artifact.manifest_digest,
                                "terminal_digest": terminal["terminal_digest"]})
    indices = bootstrap_indices(games, 20, 7)
    from werewolf.artifact_io import sha256_bytes
    report = publish_artifact(root / "seal", manifest_fields={
        "artifact_type": "backbone_oof_evaluation_report", "schema_version": adapter.historical.REPORT_VERSION,
        "evaluation_digest": contract.manifest_digest, "training_execution_digest": execution.manifest_digest,
        "training_seal_digest": train_seal.manifest_digest, "preparation": preparation,
        "training_source": training_source, "evaluation_source": evaluation_source,
        "predictions": descriptors, "primary_row_bindings": bindings,
        "bootstrap": {"version": BOOTSTRAP_VERSION, "seed": 7, "game_ids": games,
                      "indices_digest": sha256_bytes(indices.tobytes()), "replicates": 20, "confidence": .95,
                      "sampling_unit": "game", "interval_method": "percentile_linear"},
        "cells": {}, "supplementary_top1_nu": {}}, files={"bootstrap_indices.bin": indices.tobytes()})
    reference = open_reference(publication.path, root, expected_seal_digest=report.manifest_digest)
    return reference, publication, root, report


def predictions_for(reference, *, failure=False):
    from werewolf.tom_llm.output import parse_output
    rows = []
    for row in reference.rows:
        if not row["observer_alive"]:
            continue
        content = "invalid" if failure else json.dumps({"probabilities": {
            f"player{j+1}": 1/6 for j in range(7) if j != row["observer"]}})
        prediction = parse_output(content, row["observer"]).to_record()
        rows.append({key: row[key] for key in ("game_id", "boundary_id", "prefix_digest", "plan_digest",
            "observer", "fold", "canonical_shift", "temporal_condition")} | {"prediction": prediction})
    return rows
