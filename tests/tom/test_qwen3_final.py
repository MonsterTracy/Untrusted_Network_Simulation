"""Small synthetic Qwen3 final contracts; no formal training is started."""

import json
from collections import Counter

import pytest
import torch

from tests.tom.test_backbone_execution import fixture as backbone_fixture
from tests.tom.test_paper_study_execution import rewritten
from werewolf.artifact_io import canonical_json_bytes, publish_artifact, sha256_bytes
from werewolf.development_publication import open_publication
from werewolf.tom import backbone_execution, qwen3_final as q, training
from werewolf.tom.dataset import CanonicalToMDataset, ExperimentCapacity
from werewolf.tom.protocol import FIELD_ENCODING, ORDER_VERSION, SCHEDULE_VERSION
from werewolf.cli import build_parser


@pytest.fixture
def fit(backbone_fixture, monkeypatch, tmp_path):
    _, study, _, design = backbone_fixture
    count = design["game_count"]
    monkeypatch.setattr(q, "GAMES", count)
    monkeypatch.setattr(q, "STEPS", count * 3 * 7)
    monkeypatch.setattr(q, "attest_source", backbone_execution.attest_source)
    destination = tmp_path / "experiments" / "experiment"
    result = q.prepare(study.manifest["publication_path"], study.path,
                       study.manifest_digest, destination)
    return result


def test_frozen_formal_controls_and_all_development_schedule():
    design, config = q._protocol()
    assert design["publication_id"] == "paper-development-qwen35-9b-1500-v1"
    schedule = q._schedule(config, [f"g{i:04}" for i in range(1500)])
    game_ids = schedule["game_ids"]
    assert len(game_ids) == len(set(game_ids)) == 1500
    assert schedule["lifecycle_version"] == q.VERSION
    assert schedule["schedule_version"] == schedule["rotation_version"] == SCHEDULE_VERSION
    assert schedule["order_version"] == ORDER_VERSION
    assert schedule["field_encoding"] == FIELD_ENCODING
    assert schedule["optimizer_steps"] == 31500
    assert len(schedule["batches"]) == 31500
    assert all(len(batch) == 1 for batch in schedule["batches"])
    rows = [batch[0] for batch in schedule["batches"]]
    assert Counter(row["game_id"] for row in rows) == {game: 21 for game in game_ids}
    assert Counter((row["game_id"], row["shift"]) for row in rows) == {
        (game, shift): 3 for game in game_ids for shift in range(7)}
    with pytest.raises(ValueError, match="1500"):
        q._schedule(config, [f"g{i:04}" for i in range(1499)])


def test_cli_has_separate_qwen3_lifecycle_commands():
    parser = build_parser()
    assert parser.parse_args(["prepare-qwen3-final-fit", "--study", "study",
        "--study-digest", "a" * 64, "--publication", "publication",
        "--destination", "fit"]).command == "prepare-qwen3-final-fit"
    for command in ("run-qwen3-final-fit", "seal-qwen3-final-fit", "validate-qwen3-final-fit"):
        assert parser.parse_args([command, "--experiment", "fit"]).command == command


def test_prepare_binds_exact_initial_and_every_development_game(fit):
    opened = q.open_fit(fit.path)
    assert opened.digest == fit.digest
    assert fit.manifest["initialization"]["architecture"] == "qwen3"
    assert fit.manifest["initialization"]["fold"] == 0
    assert fit.manifest["initialization"]["state_digest"] == fit.manifest["protocol_inputs"]["initial_state_digest"]
    assert fit.manifest["protocol_inputs"]["partition"] == "all_development"
    assert fit.schedule()["game_ids"] == fit.manifest["game_ids"]
    assert fit.schedule()["optimizer_steps"] == q.STEPS
    assert "fold" not in fit.schedule()
    assert len(fit.manifest["game_ids"]) == q.GAMES
    assert not fit.runs_path.exists()


@pytest.mark.parametrize("change", [
    lambda m: m["protocol_inputs"].update(initial_state_digest="0" * 64),
    lambda m: m["protocol_inputs"].update(partition="fold_0"),
    lambda m: m["protocol_inputs"].update(destination_path="/tmp/cloned"),
    lambda m: m["initialization"].update(fold=1),
    lambda m: m.update(optimizer_steps=25200),
    lambda m: m.update(schedule_digest="0" * 64),
])
def test_rehashed_contract_corruption_fails_closed(fit, change):
    with rewritten(fit.path / q.MANIFEST, change, "manifest_digest"):
        with pytest.raises(ValueError):
            q.open_fit(fit.path)


def test_fold_schedule_injection_fails_even_with_rehashed_artifact(fit):
    schedule_path = fit.path / "schedule.json"
    manifest_path = fit.path / q.MANIFEST
    original_schedule, original_manifest = schedule_path.read_bytes(), manifest_path.read_bytes()
    fold_schedule = (fit.snapshot.path / "folds/0/schedule.json").read_bytes()
    assert fold_schedule != original_schedule
    manifest = json.loads(original_manifest)
    manifest["file_table"]["schedule.json"] = {
        "byte_size": len(fold_schedule), "sha256": sha256_bytes(fold_schedule)}
    manifest["schedule_digest"] = sha256_bytes(fold_schedule)
    del manifest["manifest_digest"]
    manifest["manifest_digest"] = sha256_bytes(canonical_json_bytes(manifest))
    try:
        schedule_path.write_bytes(fold_schedule)
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        with pytest.raises(ValueError, match="schedule"):
            q.open_fit(fit.path)
    finally:
        schedule_path.write_bytes(original_schedule)
        manifest_path.write_bytes(original_manifest)


def test_no_terminal_or_seal_cannot_predict_or_resume(fit):
    with pytest.raises(ValueError):
        q.verify_terminal(fit)
    with pytest.raises(ValueError):
        q.verify_seal(fit)
    with pytest.raises(ValueError):
        q.SealedQwen3Predictor(fit)
    with pytest.raises(ValueError, match="no Qwen3 recovery"):
        q.run(fit, resume=True)


def test_empty_recovery_directory_cannot_resume_or_start_worker(fit, monkeypatch):
    recovery = fit.runs_path / "recovery"
    recovery.mkdir(parents=True)
    monkeypatch.setattr(q.multiprocessing, "get_context",
        lambda *_: pytest.fail("empty recovery must not launch a training worker"))
    with pytest.raises(ValueError, match="at least one verified"):
        q.run(fit, resume=True)
    with pytest.raises(ValueError, match="explicit resume"):
        q.run(fit)
    assert list(recovery.iterdir()) == []


def test_one_real_step_recovery_roundtrip_and_no_early_seal(fit, monkeypatch):
    publication = open_publication(fit.manifest["publication_path"])
    dataset = CanonicalToMDataset(publication.public_view, fit.manifest["game_ids"], ExperimentCapacity(1024))
    first = fit.schedule()["batches"][0]
    game = first[0]["game_id"]
    by_game = {game: [sample for sample in dataset if sample.game_id == game]}
    masks = json.loads((fit.path / "primary.json").read_bytes())
    model, optimizer = q._model_optimizer(fit)
    logs = []
    training.fit_steps(model, optimizer, by_game, masks,
        {"optimizer_steps": 1, "batches": [first]}, device="cpu",
        learning_rate=fit.config.learning_rate, start=0, logs=logs,
        gate=lambda: None, on_step=lambda step, current: None)
    assert logs[0]["step"] == 1 and torch.isfinite(torch.tensor(logs[0]["loss"]))
    q._checkpoint(fit, 1, model, optimizer, logs, None)
    monkeypatch.setattr(q, "_points", lambda _fit: [1, q.STEPS])
    restored, restored_optimizer = q._model_optimizer(fit)
    rows, ancestry = q._chain(fit, restored, restored_optimizer)
    assert rows == logs and [row["step"] for row in ancestry] == [1]
    with pytest.raises(ValueError, match="contiguous"):
        q.seal(fit)
    with rewritten(fit.runs_path / "recovery/1/manifest.json",
                   lambda m: m["binding"].update(initial_state_digest="0" * 64), "manifest_digest"):
        with pytest.raises(ValueError, match="binding"):
            q._chain(fit, restored, restored_optimizer)

    # Exercise the terminal/seal/predictor mechanics with a one-step synthetic
    # budget. The formal 31500-step contract is checked separately above.
    full_schedule = fit.schedule()
    monkeypatch.setattr(q, "STEPS", 1)
    monkeypatch.setattr(q, "_points", lambda _fit: [1])
    monkeypatch.setattr(q.FinalFit, "schedule", lambda _fit: full_schedule)
    checkpoint = q.verify_artifact(fit.runs_path / "recovery/1",
        expected_artifact_type="qwen3_final_recovery", expected_schema_version=q.RECOVERY_VERSION)
    publish_artifact(fit.runs_path / "terminal", manifest_fields={
        "artifact_type": "qwen3_final_terminal", "schema_version": q.TERMINAL_VERSION,
        "binding": q._binding(fit), "terminal_step": 1,
        "last_recovery_digest": checkpoint.manifest_digest,
        "model_digest": q._hash(checkpoint.manifest["containers"]["model"]),
        "checkpoint_ancestry": [{"step": 1, "digest": checkpoint.manifest_digest}]}, files={})
    sealed = q.seal(fit)
    assert q.verify_seal(fit).manifest_digest == sealed.manifest_digest
    predictor = q.SealedQwen3Predictor(fit)
    prefix = publication.public_view.load_game(game).authoritative_pre_prefixes[0]
    probabilities = predictor.predict(prefix, 0)
    assert probabilities.shape == (7,)
    assert probabilities[0] == 0
    assert torch.isclose(probabilities.sum(), torch.tensor(1.), atol=1e-6)
    with pytest.raises(ValueError, match="completed"):
        q.run(fit, resume=True)
