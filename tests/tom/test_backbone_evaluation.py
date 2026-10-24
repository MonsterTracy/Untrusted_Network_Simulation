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


# Independently pin the complete source inventory; dropping a production entry
# must not silently remove a historical source from this contract test.
HISTORICAL_INFERENCE_SOURCES = (
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


@pytest.fixture
def historical_evaluation_source(monkeypatch):
    """Supply genuine archived bytes as the local source, keeping all checks real."""
    from pathlib import Path
    import subprocess
    project = Path(m.__file__).resolve().parents[2]
    revision = "ebf98b7e3249f78d3f1fa1fd46c20affe0486913"
    assert m.TRAINING_REVISION == revision
    assert m.INFERENCE_SOURCES == HISTORICAL_INFERENCE_SOURCES
    assert subprocess.check_output(["git", "-C", str(project), "cat-file", "-t", revision], text=True).strip() == "commit"
    assert subprocess.check_output(["git", "-C", str(project), "rev-parse", f"{revision}^{{commit}}"], text=True).strip() == revision
    frozen = {project / relative: subprocess.check_output(
        ["git", "-C", str(project), "show", f"{revision}:{relative}"])
        for relative in HISTORICAL_INFERENCE_SOURCES}
    original_read = Path.read_bytes
    def read_historical(path):
        return frozen[path] if path in frozen else original_read(path)
    monkeypatch.setattr(Path, "read_bytes", read_historical)
    old_runtime = {"source_revision": "reference-scientific-revision", "implementation_digest": "a" * 64,
                   "torch": "frozen"}
    owner = SimpleNamespace(execution=SimpleNamespace(config=object(), paired=False, manifest={
        "runtime": old_runtime, "execution_source": {"source_revision": revision,
        "implementation_digest": "a" * 64}}))
    source = {"source_revision": "c" * 40, "implementation_digest": "b" * 64}
    current = {**old_runtime, "implementation_digest": source["implementation_digest"]}
    monkeypatch.setattr(training, "_runtime", lambda _: (source, current))
    return SimpleNamespace(project=project, owner=owner, source=source, runtime=current,
                           frozen=frozen, original_read=original_read)


def test_evaluation_source_is_independent_but_inference_bytes_are_frozen(historical_evaluation_source):
    case = historical_evaluation_source
    source, runtime = m._evaluation_source(case.owner)
    assert source == case.source and runtime == case.runtime
    assert source["source_revision"] != m.TRAINING_REVISION
    assert runtime["implementation_digest"] != case.owner.execution.manifest["runtime"]["implementation_digest"]


@pytest.mark.parametrize("relative", HISTORICAL_INFERENCE_SOURCES)
def test_evaluation_rejects_each_changed_inference_source(historical_evaluation_source, monkeypatch, relative):
    from pathlib import Path
    case = historical_evaluation_source
    changed = case.project / relative
    original_read = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda path:
                        case.frozen[path] + b"\n# changed source\n" if path == changed else original_read(path))
    with pytest.raises(ValueError, match="historical inference source changed"):
        m._evaluation_source(case.owner)


def test_evaluation_rejects_unavailable_historical_source(historical_evaluation_source, monkeypatch):
    case = historical_evaluation_source
    original_check_output = m.subprocess.check_output
    command = ["git", "-C", str(case.project), "show",
               f"{m.TRAINING_REVISION}:{HISTORICAL_INFERENCE_SOURCES[0]}"]
    def unavailable(args, **kwargs):
        if args == command:
            raise m.subprocess.CalledProcessError(128, args)
        return original_check_output(args, **kwargs)
    monkeypatch.setattr(m.subprocess, "check_output", unavailable)
    with pytest.raises(ValueError, match="historical inference source unavailable"):
        m._evaluation_source(case.owner)


def test_evaluation_rejects_runtime_drift(historical_evaluation_source, monkeypatch):
    case = historical_evaluation_source
    monkeypatch.setattr(training, "_runtime", lambda _: (case.source, {**case.runtime, "torch": "drift"}))
    with pytest.raises(ValueError, match="runtime differs"):
        m._evaluation_source(case.owner)


def test_evaluation_rejects_source_runtime_mismatch(historical_evaluation_source, monkeypatch):
    case = historical_evaluation_source
    monkeypatch.setattr(training, "_runtime", lambda _: (case.source,
                        {**case.runtime, "implementation_digest": "d" * 64}))
    with pytest.raises(ValueError, match="evaluation source/runtime mismatch"):
        m._evaluation_source(case.owner)


@pytest.mark.parametrize("revision", ("0" * 40, "invalid-revision"))
def test_evaluation_rejects_wrong_training_revision(historical_evaluation_source, revision):
    case = historical_evaluation_source
    case.owner.execution.manifest["execution_source"]["source_revision"] = revision
    with pytest.raises(ValueError, match="invalid historical training revision"):
        m._evaluation_source(case.owner)


def test_evaluation_checks_actual_checkout_bytes(historical_evaluation_source, monkeypatch):
    from pathlib import Path
    case = historical_evaluation_source
    changed = any(case.original_read(path) != frozen for path, frozen in case.frozen.items())
    monkeypatch.setattr(Path, "read_bytes", case.original_read)
    if changed:
        with pytest.raises(ValueError, match="historical inference source changed"):
            m._evaluation_source(case.owner)
    else:
        assert m._evaluation_source(case.owner) == (case.source, case.runtime)


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
