"""Synthetic sealed backbone OOF contrasts; no model execution or real artifacts."""

from copy import deepcopy
import json
import math
from pathlib import Path
import shutil

import pytest
import torch

from scripts import paper_backbone_contrasts as m
from werewolf.artifact_io import canonical_json_bytes, canonical_jsonl_bytes, publish_artifact, sha256_bytes
from werewolf.tom.evaluation import score_prediction_rows
from werewolf.tom.protocol import BOOTSTRAP_VERSION, bootstrap_indices
from werewolf.tom.scoring import game_macro_summary, paired_summary


def _prediction(game, observer, architecture, fold):
    target = (observer + 1) % 7
    strength = {"gpt2": .2, "qwen2": .3, "qwen3": .4, "gemma3_text": .35}[architecture] + fold * .01
    probability = [0. if seat == observer else strength if seat == target else (1 - strength) / 5
                   for seat in range(7)]
    return {"game_id": game, "boundary_id": f"{game}-PRE", "observer": observer,
            "prefix_digest": f"{game}-prefix", "plan_digest": f"{game}-plan",
            "probability": probability,
            "non_self_log_probability": [math.log(probability[seat]) for seat in range(7) if seat != observer],
            "q": [float(seat == target) for seat in range(7)], "label_observed": True,
            "observer_alive": True, "checkpoint_digest": f"{architecture}-{fold}-checkpoint",
            "temporal_condition": "explicit_day_phase"}


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "GAME_COUNT", 5)
    monkeypatch.setattr(m, "REPLICATES", 20)
    monkeypatch.setattr(m, "_analysis_source", lambda: {"source_revision": "a" * 40,
                                                        "implementation_digest": "b" * 64})
    base = tmp_path / "study"
    games = [f"g{fold}" for fold in range(5)]
    primary = {"metadata": {"population_identity": "non_wolf_alive"},
               "rows": {game: {f"{game}-PRE": [True] * 5 + [False] * 2} for game in games}}
    preparation_source = {"source_revision": "c" * 40, "implementation_digest": "d" * 64}
    prepared = publish_artifact(base / "prepared", manifest_fields={
        "artifact_type": "backbone_tom_study", "schema_version": m.STUDY_VERSION,
        "study_digest": "e" * 64,
        "protocol_inputs": {"source": preparation_source,
                            "primary_digest": sha256_bytes(canonical_json_bytes(primary)),
                            "primary_identity": primary["metadata"]},
        "folds": [{"fold": fold, "held_out_game_ids": [games[fold]],
                   "training_game_ids": [game for game in games if game != games[fold]]} for fold in range(5)]
    }, files={"primary.json": canonical_json_bytes(primary)})
    preparation = {"manifest_digest": prepared.manifest_digest, "study_digest": "e" * 64,
                   **preparation_source}
    training_source = {"source_revision": "f" * 40, "implementation_digest": "0" * 64}
    training_root = base / "executions" / prepared.manifest_digest
    execution = publish_artifact(training_root / "contract", manifest_fields={
        "artifact_type": "backbone_execution", "schema_version": m.EXECUTION_VERSION,
        "prepared_path": str(prepared.path), "preparation": preparation,
        "execution_source": training_source}, files={})
    terminals = []
    for architecture in m.ARCHITECTURES:
        for fold in range(5):
            terminal = publish_artifact(training_root / "runs" / architecture / str(fold) / "terminal",
                manifest_fields={"artifact_type": "backbone_terminal", "schema_version": m.TERMINAL_VERSION,
                                 "checkpoint_ancestry": [{"step": 25200,
                                                          "digest": f"{architecture}-{fold}-checkpoint"}]}, files={})
            terminals.append({"architecture": architecture, "fold": fold,
                              "temporal_condition": "explicit_day_phase",
                              "terminal_digest": terminal.manifest_digest})
    train_seal = publish_artifact(training_root / "seal", manifest_fields={
        "artifact_type": "backbone_execution_seal", "schema_version": m.SEAL_VERSION,
        "execution_digest": execution.manifest_digest, "preparation": preparation,
        "execution_source": training_source, "terminals": terminals}, files={})
    root = base / "evaluations" / execution.manifest_digest
    evaluation_source = {"source_revision": "1" * 40, "implementation_digest": "2" * 64}
    contract = publish_artifact(root / "contract", manifest_fields={
        "artifact_type": "backbone_oof_evaluation_contract", "schema_version": m.EVALUATION_VERSION,
        "training_path": str(training_root), "training_execution_digest": execution.manifest_digest,
        "training_seal_digest": train_seal.manifest_digest, "preparation": preparation,
        "training_source": training_source, "evaluation_source": evaluation_source}, files={})
    descriptors, bindings = [], {}
    scores = {architecture: {} for architecture in m.ARCHITECTURES}
    for fold, game in enumerate(games):
        for architecture in m.ARCHITECTURES:
            rows = [_prediction(game, observer, architecture, fold) for observer in range(7)]
            selected = rows[:5]
            _, game_scores = score_prediction_rows(selected, [game])
            scores[architecture].update(game_scores)
            ids = [[game, f"{game}-PRE", observer] for observer in range(5)]
            binding = sha256_bytes(canonical_json_bytes(ids))
            bindings[str(fold)] = binding
            terminal = next(value for value in terminals
                            if value["architecture"] == architecture and value["fold"] == fold)
            artifact = publish_artifact(root / "folds" / architecture / str(fold), manifest_fields={
                "artifact_type": "backbone_oof_fold_predictions", "schema_version": m.PREDICTION_VERSION,
                "evaluation_digest": contract.manifest_digest, "training_seal_digest": train_seal.manifest_digest,
                "architecture": architecture, "fold": fold, "temporal_condition": "explicit_day_phase",
                "terminal_digest": terminal["terminal_digest"],
                "checkpoint_digest": rows[0]["checkpoint_digest"], "row_count": len(rows),
                "primary_row_identity_digest": binding}, files={"predictions.jsonl": canonical_jsonl_bytes(rows)})
            descriptors.append({"architecture": architecture, "fold": fold,
                                "prediction_digest": artifact.manifest_digest,
                                "terminal_digest": terminal["terminal_digest"]})
    indices = bootstrap_indices(games, 20, 7)
    cells = {architecture: {"kl": game_macro_summary([scores[architecture][game]["kl"] for game in games],
                                                   indices, .95)} for architecture in m.ARCHITECTURES}
    raw = indices.tobytes()
    report = publish_artifact(root / "seal", manifest_fields={
        "artifact_type": "backbone_oof_evaluation_report", "schema_version": m.REPORT_VERSION,
        "evaluation_digest": contract.manifest_digest, "training_execution_digest": execution.manifest_digest,
        "training_seal_digest": train_seal.manifest_digest, "preparation": preparation,
        "training_source": training_source, "evaluation_source": evaluation_source,
        "predictions": descriptors, "primary_row_bindings": bindings,
        "bootstrap": {"version": BOOTSTRAP_VERSION, "seed": 7, "game_ids": games,
                      "indices_digest": sha256_bytes(raw), "replicates": 20, "confidence": .95,
                      "sampling_unit": "game", "interval_method": "percentile_linear"},
        "cells": cells, "supplementary_top1_nu": {}}, files={"bootstrap_indices.bin": raw})
    return root, report, scores, indices, games


def test_four_paired_contrasts_use_sealed_draws_and_preserve_inputs(sealed, tmp_path, monkeypatch):
    root, report, scores, indices, games = sealed
    before = {str(path): path.read_bytes() for path in root.parent.parent.rglob("*") if path.is_file()}
    monkeypatch.setattr(torch.nn.Module, "_call_impl", lambda *_args, **_kwargs: pytest.fail("model inference forbidden"))
    artifact = m.analyze(root, report.manifest_digest, tmp_path / "contrasts")
    assert before == {str(path): path.read_bytes() for path in root.parent.parent.rglob("*") if path.is_file()}
    assert artifact.manifest["evaluation_seal_digest"] == report.manifest_digest
    assert artifact.manifest["bootstrap_digest"] == report.manifest["bootstrap"]["indices_digest"]
    assert artifact.manifest["analysis_source"] == {"source_revision": "a" * 40, "implementation_digest": "b" * 64}
    assert [(row["left"], row["right"]) for row in artifact.manifest["contrasts"]] == list(m.CONTRASTS)
    for row in artifact.manifest["contrasts"]:
        left = [scores[row["left"]][game]["kl"] for game in games]
        right = [scores[row["right"]][game]["kl"] for game in games]
        assert row["Delta_KL"] == paired_summary(left, right, indices, .95)
        assert row["Delta_KL"]["point"] < 0
    with pytest.raises(ValueError, match="outside sealed evaluation"):
        m.analyze(root, report.manifest_digest, root / "derived")
    with pytest.raises(ValueError, match="outside sealed evaluation"):
        m.analyze(root, report.manifest_digest, root / "folds/../derived")


def test_analysis_source_binds_committed_script_bytes(monkeypatch):
    script = Path(m.__file__).read_bytes()
    monkeypatch.setattr(m, "attest_source", lambda: {"source_revision": "a" * 40,
                                                      "implementation_digest": "b" * 64})
    monkeypatch.setattr(m.subprocess, "check_output", lambda *_args, **_kwargs: script)
    source = m._analysis_source()
    assert source == {"source_revision": "a" * 40,
                      "implementation_digest": sha256_bytes(canonical_json_bytes({
                          "werewolf_implementation_digest": "b" * 64,
                          "analysis_script_sha256": sha256_bytes(script)}))}
    monkeypatch.setattr(m.subprocess, "check_output", lambda *_args, **_kwargs: b"drifted")
    with pytest.raises(ValueError, match="analysis source differs"):
        m._analysis_source()


@pytest.mark.parametrize("corruption", ("seal_anchor", "bootstrap", "prediction", "primary", "kl_cell", "lineage"))
def test_corrupt_or_rebound_sealed_input_fails_without_output(sealed, tmp_path, corruption):
    original, report, _, _, _ = sealed
    root = tmp_path / "changed"
    shutil.copytree(original, root)
    digest = report.manifest_digest
    if corruption == "seal_anchor":
        digest = "0" * 64
    elif corruption == "bootstrap":
        path = root / "seal/bootstrap_indices.bin"
        path.write_bytes(b"\0" * len(path.read_bytes()))
    elif corruption == "prediction":
        path = root / "folds/qwen3/0/predictions.jsonl"
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        rows[0]["probability"][1] = .9
        path.write_bytes(canonical_jsonl_bytes(rows))
    elif corruption == "primary":
        path = root / "folds/qwen3/0/manifest.json"
        value = json.loads(path.read_bytes())
        value["primary_row_identity_digest"] = "0" * 64
        path.write_bytes(canonical_json_bytes(value))
    else:
        fields = {key: deepcopy(value) for key, value in report.manifest.items()
                  if key not in {"file_table", "manifest_digest"}}
        if corruption == "kl_cell":
            fields["cells"]["qwen3"]["kl"]["point"] += .1
        else:
            fields["predictions"].pop()
        shutil.rmtree(root / "seal")
        changed = publish_artifact(root / "seal", manifest_fields=fields,
                                   files={"bootstrap_indices.bin": (original / "seal/bootstrap_indices.bin").read_bytes()})
        digest = changed.manifest_digest
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        m.analyze(root, digest, output)
    assert not output.exists()
