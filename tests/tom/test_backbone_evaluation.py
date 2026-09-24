"""Synthetic validation of the independent backbone OOF evaluation contract."""

import math
from types import SimpleNamespace

import pytest
import torch

from tests.tom.test_backbone_execution import fixture  # Real synthetic prepared publication/artifact.
from werewolf.artifact_io import canonical_json_bytes, canonical_jsonl_bytes, publish_artifact, sha256_bytes
from werewolf.tom import backbone_evaluation as m
from werewolf.tom import backbone_execution as training
from werewolf.tom.dataset import PublicTensors
from werewolf.tom.protocol import bootstrap_indices


def test_historical_reader_rejects_engineering_execution(fixture):
    execution = fixture[2]
    with pytest.raises(ValueError):
        m.open_sealed_training(execution.root, execution.artifact.manifest_digest,
                               "0" * 64, "b" * 40)
    assert not (execution.snapshot.path.parent / "evaluations").exists()


def test_evaluation_source_is_independent_but_inference_bytes_are_frozen(monkeypatch):
    from pathlib import Path
    import subprocess
    project = Path(m.__file__).resolve().parents[2]
    revision = m.TRAINING_REVISION
    assert subprocess.check_output(["git", "-C", str(project), "cat-file", "-t", revision], text=True).strip() == "commit"
    assert {"werewolf/canonical_collection/pre.py", "werewolf/canonical_collection/public_history.py",
            "werewolf/development_publication.py"} <= set(m.INFERENCE_SOURCES)
    old_runtime = {"source_revision": "reference-scientific-revision", "implementation_digest": "a" * 64,
                   "torch": "frozen"}
    owner = SimpleNamespace(execution=SimpleNamespace(config=object(), manifest={
        "runtime": old_runtime, "execution_source": {"source_revision": revision,
        "implementation_digest": "a" * 64}}))
    current = {**old_runtime, "implementation_digest": "b" * 64}
    monkeypatch.setattr(training, "_runtime", lambda _: ({"source_revision": "c" * 40,
        "implementation_digest": "b" * 64}, current))
    source, runtime = m._evaluation_source(owner)
    assert source["source_revision"] != revision
    assert runtime["implementation_digest"] != old_runtime["implementation_digest"]
    with monkeypatch.context() as patch:
        patch.setattr(m, "INFERENCE_SOURCES", ("werewolf/tom/nonexistent_archived_source.py",))
        with pytest.raises(ValueError, match="historical inference source unavailable"):
            m._evaluation_source(owner)
    with monkeypatch.context() as patch:
        original_read = Path.read_bytes
        changed = project / "werewolf/canonical_collection/public_history.py"
        patch.setattr(Path, "read_bytes", lambda path: b"changed-public-history" if path == changed else original_read(path))
        patch.setattr(m, "open_sealed_training", lambda *_: owner)
        with pytest.raises(ValueError, match="historical inference source changed"):
            m.prepare_evaluation(project / "unused-training", "e" * 64, "f" * 64, revision)
    with monkeypatch.context() as patch:
        patch.setattr(training, "_runtime", lambda _: ({"source_revision": "c" * 40,
            "implementation_digest": "b" * 64}, {**current, "torch": "drift"}))
        with pytest.raises(ValueError, match="runtime differs"):
            m._evaluation_source(owner)


def test_shared_whole_game_draws_and_fixed_uniform_direction():
    games = ["g0", "g1", "g2"]
    draws = bootstrap_indices(games, 100, 31)
    scores = {}
    for architecture, kl in zip(m.ARCHITECTURES,
                                ([.2, .3, .4], [.4, .5, .6], [.1, .4, .7], [.3, .3, .3]), strict=True):
        scores[architecture] = {g: {"kl": value, "uniform_kl": 1., "total_variation": value / 2,
                                    "row_count": 2} for g, value in zip(games, kl, strict=True)}
    cells = m._summarize_cells(scores, games, draws, .95)
    assert len(cells) == 4
    for cell in cells.values():
        assert cell["Delta_uniform"]["point"] == pytest.approx(1 - cell["kl"]["point"])
        assert cell["kl"]["confidence"] == cell["total_variation"]["confidence"] == .95
        assert cell["row_count"] == 6 and cell["game_count"] == 3
    scores["qwen3"]["g0"]["uniform_kl"] = 2.
    with pytest.raises(ValueError, match="uniform reference"):
        m._summarize_cells(scores, games, draws, .95)


def test_top1_nu_is_game_macro_without_ci_and_excludes_six_support():
    rows = [{"game_id": "a", "q": [0, 1, 0, 0, 0, 0, 0]},
            {"game_id": "a", "q": [0, 1, 0, 0, 0, 0, 0]},
            {"game_id": "b", "q": [0, 1, 0, 0, 0, 0, 0]},
            {"game_id": "b", "q": [0, 1, 1, 1, 1, 1, 1]}]
    scores = [{"argmax_in_target_support": x} for x in (True, False, True, True)]
    counts = {}
    excluded = m._accumulate_top1_nu(counts, rows, scores)
    assert m._top1_nu(counts, excluded, 4, 2) == {
        "top1_nu": .75, "total_primary_row_count": 4, "valid_nu_row_count": 3,
        "excluded_s6_row_count": 1, "total_primary_game_count": 2, "valid_nu_game_count": 2}


def test_fold_validator_uses_real_synthetic_pre_targets_and_primary(fixture, tmp_path):
    execution = fixture[2]
    owner = SimpleNamespace(execution=execution, seal=SimpleNamespace(manifest_digest="e" * 64))
    dataset, masks = m._held_out(owner, 0)
    model, _ = training._model_optimizer(execution, "qwen2", 0)
    model.eval()
    rows, selected = [], []
    with torch.inference_mode():
        for sample in dataset:
            public = PublicTensors.stack([sample.public])
            logp = model(**public.kwargs())[0]
            for observer in range(7):
                row = {"game_id": sample.game_id, "boundary_id": sample.boundary_id,
                    "observer": observer, "prefix_digest": sample.prefix_digest, "plan_digest": sample.plan_digest,
                    "probability": logp[observer].exp().tolist(),
                    "non_self_log_probability": [logp[observer, j].item() for j in range(7) if j != observer],
                    "q": sample.q[observer].tolist(), "label_observed": bool(sample.label_observed[observer]),
                    "observer_alive": bool(sample.observer_alive[observer]),
                    "checkpoint_digest": "c" * 64, "temporal_condition": m.CONDITION}
                rows.append(row)
                if row["label_observed"] and masks[sample.game_id][sample.boundary_id][observer]:
                    selected.append([sample.game_id, sample.boundary_id, observer])
    terminal = SimpleNamespace(manifest={"checkpoint_ancestry": [{"digest": "c" * 64}]},
                               manifest_digest="f" * 64)
    contract = SimpleNamespace(path=tmp_path / "contract", manifest_digest="d" * 64)
    def publish(root, predictions):
        publish_artifact(root / "folds/qwen2/0", manifest_fields={
            "artifact_type": "backbone_oof_fold_predictions", "schema_version": m.PREDICTION_VERSION,
            "evaluation_digest": contract.manifest_digest, "training_seal_digest": owner.seal.manifest_digest,
            "architecture": "qwen2", "fold": 0, "temporal_condition": m.CONDITION,
            "terminal_digest": terminal.manifest_digest, "checkpoint_digest": "c" * 64,
            "row_count": len(predictions),
            "primary_row_identity_digest": sha256_bytes(canonical_json_bytes(selected)),
        }, files={"predictions.jsonl": canonical_jsonl_bytes(predictions)})
    publish(tmp_path, rows)
    artifact, effective = m._validated_fold(owner, contract, "qwen2", 0, terminal, dataset, masks)
    assert artifact.manifest["row_count"] == len(dataset) * 7
    assert len(effective) == len(selected) > 0
    m._verify_model_predictions(owner, artifact, terminal, model, dataset, masks)
    coherent_wrong = [dict(r) for r in rows]
    coherent_wrong[0]["probability"] = list(rows[0]["probability"])
    coherent_wrong[0]["non_self_log_probability"] = list(rows[0]["non_self_log_probability"])
    coherent_wrong[0]["probability"][1] += .005
    coherent_wrong[0]["probability"][2] -= .005
    for seat in (1, 2):
        coherent_wrong[0]["non_self_log_probability"][seat - 1] = math.log(coherent_wrong[0]["probability"][seat])
    wrong_model = SimpleNamespace(path=tmp_path / "wrong-model/contract", manifest_digest=contract.manifest_digest)
    publish(tmp_path / "wrong-model", coherent_wrong)
    republished, _ = m._validated_fold(owner, wrong_model, "qwen2", 0, terminal, dataset, masks)
    assert republished.manifest_digest != artifact.manifest_digest
    assert republished.manifest["file_table"]["predictions.jsonl"]["sha256"] != artifact.manifest["file_table"]["predictions.jsonl"]["sha256"]
    with pytest.raises(ValueError, match="differs from terminal model inference"):
        m._verify_model_predictions(owner, republished, terminal, model, dataset, masks)
    modified = [dict(r) for r in rows]
    modified[0]["prefix_digest"] = "wrong-PRE"
    wrong = SimpleNamespace(path=tmp_path / "corrupt/contract", manifest_digest=contract.manifest_digest)
    publish(tmp_path / "corrupt", modified)
    with pytest.raises(ValueError, match="PRE/target"):
        m._validated_fold(owner, wrong, "qwen2", 0, terminal, dataset, masks)
