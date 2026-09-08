"""Content mutations traverse real artifact writers without changing control."""

from collections import Counter

import pytest

from tests.development_publication.test_development_publication import _publication
from tests.tom.test_experiment import experiment_config


def _changed_evidence(evidence, change):
    from dataclasses import replace
    from werewolf.canonical_collection.trajectory_evidence import (
        construct_belief_observation, construct_private_replay_evidence,
    )

    if change == "belief":
        # A non-speaker report: the existing speaker handoff stays exact.
        observation = evidence.belief_observations[3]
        record = observation.to_record()
        for key in ("schema_version", "observation_digest", "label_observed"):
            record.pop(key)
        record.update(status=observation.status, suspicion_support=["player5"])
        observations = list(evidence.belief_observations)
        observations[3] = construct_belief_observation(**record)
        return replace(evidence, belief_observations=tuple(observations))
    record = evidence.private_replay_evidence.to_record()
    for key in ("schema_version", "evidence_digest"):
        record.pop(key)
    if change == "role":
        roles = record["role_assignment"]
        roles["player2"], roles["player4"] = roles["player4"], roles["player2"]
    else:
        assert change == "private"
        # The fixture replay executor deliberately emits the same public events.
        record["initial_runtime_state"]["private_fixture_marker"] = "changed"
    return replace(evidence, private_replay_evidence=construct_private_replay_evidence(**record))


@pytest.mark.parametrize("change", ["belief", "role", "private"])
def test_held_out_content_changes_provenance_not_control_or_tensors(tmp_path, change):
    from werewolf.artifact_io import publish_artifact
    from werewolf.tom.experiment import prepare_experiment, open_experiment, validate_experiment_preflight
    from werewolf.tom.training import train_primary_fold

    original_collection, original = _publication(tmp_path / "original")
    fold = 0
    held_out = original.public.public_view.fold_manifest.folds[fold].game_ids[0]
    changed_collection, changed = _publication(tmp_path / "changed", transform_evidence=lambda e:
        _changed_evidence(e, change) if e.game_id == held_out else e)
    before, after = original.public.public_view, changed.public.public_view
    assert before.game_ids == after.game_ids
    assert before.development_game_set_digest != after.development_game_set_digest
    assert original.public.manifest_digest != changed.public.manifest_digest
    assert original.restricted.sidecar_digest != changed.restricted.sidecar_digest
    assert [f.game_ids for f in before.fold_manifest.folds] == [f.game_ids for f in after.fold_manifest.folds]
    assert before.fold_manifest.manifest_digest != after.fold_manifest.manifest_digest
    for left, right in zip(original_collection.games, changed_collection.games, strict=True):
        assert (left.manifest_digest == right.manifest_digest) == (left.game_id != held_out)

    # The same predeclared configuration is used for both complete publications.
    config = experiment_config(16)
    first = prepare_experiment(original.public, config, tmp_path / "original" / "experiments" / "run")
    second = prepare_experiment(changed.public, config, tmp_path / "changed" / "experiments" / "run")
    assert first.manifest["protocol_digest"] != second.manifest["protocol_digest"]
    assert first.digest != second.digest
    for index in range(5):
        assert first.schedule(index) == second.schedule(index)
        for name in ("schedule_manifest.json", "training_schedule.jsonl", "initial_state/tensors.bin"):
            assert first.file(f"folds/{index}/{name}") == second.file(f"folds/{index}/{name}")
        assert first.fold(index)["paired_initial_state_digest"] != second.fold(index)["paired_initial_state_digest"]

    if change == "belief":
        assert before.load_game(held_out).belief_observations != after.load_game(held_out).belief_observations
    if change == "role":
        assert first.file(f"folds/{fold}/population/held_out_primary/{held_out}.json") != second.file(f"folds/{fold}/population/held_out_primary/{held_out}.json")

    # Valid outer hashes cannot make another publication satisfy the old binding.
    fields = {k: v for k, v in first.manifest.items() if k not in {"file_table", "manifest_digest"}}
    fields["publication_path"] = str(changed.public.path.resolve())
    wrong_path = tmp_path / "wrong-binding" / "experiments" / "run"
    publish_artifact(wrong_path, manifest_name="experiment_manifest.json", manifest_fields=fields,
        files={p: first.file(p) for p in first.manifest["file_table"]})
    with pytest.raises(ValueError, match="publication preflight"):
        validate_experiment_preflight(open_experiment(wrong_path))

    # Exercise production optimization, including dropout and both temporal codes.
    for condition in first.manifest["temporal_conditions"]:
        left = train_primary_fold(first, fold, condition)
        right = train_primary_fold(second, fold, condition)
        assert (left.path / "tensors.bin").read_bytes() == (right.path / "tensors.bin").read_bytes()
        assert left.manifest_digest != right.manifest_digest
        assert (left.path.parent / "training_manifest.json").read_bytes() != (right.path.parent / "training_manifest.json").read_bytes()


def test_training_identity_and_frozen_seed_control_balanced_schedule():
    from werewolf.tom.protocol import training_schedule

    games = ("a", "b", "c", "d", "e")
    base = training_schedule(303, 0, games, 2, 3)
    assert base == training_schedule(303, 0, tuple(reversed(games)), 2, 3)
    renamed = training_schedule(303, 0, ("a", "b", "c", "d", "renamed"), 2, 3)
    assert base["batches"] != renamed["batches"]
    assert base["batches"] != training_schedule(304, 0, games, 2, 3)["batches"]
    assert base["batches"] != training_schedule(303, 1, games, 2, 3)["batches"]
    for schedule in (base, renamed):
        counts = Counter((r["game_id"], r["shift"]) for b in schedule["batches"] for r in b)
        ids = {g for g, _ in counts}
        assert counts == Counter({(g, s): 2 for g in ids for s in range(7)})
        assert all(len({r["game_id"] for r in b}) == len(b) for b in schedule["batches"])


@pytest.mark.parametrize("seed", ["a" * 64, True, -1, 2**63])
def test_schedule_rejects_content_digest_or_invalid_seed(seed):
    from werewolf.tom.protocol import training_schedule

    with pytest.raises(ValueError, match="schedule seed"):
        training_schedule(seed, 0, ["game"], 1, 1)


def test_validly_rehashed_schedule_still_must_match_frozen_seed(tmp_path):
    from werewolf.artifact_io import canonical_json_bytes, canonical_jsonl_bytes, publish_artifact, sha256_bytes
    from werewolf.tom.experiment import open_experiment
    from werewolf.tom.protocol import training_schedule
    from tests.tom.test_experiment import prepared_experiment

    experiment, _ = prepared_experiment(tmp_path)
    files = {p: experiment.file(p) for p in experiment.manifest["file_table"]}
    changed = training_schedule(experiment.config.schedule_seed + 1, 0,
        experiment.fold(0)["training_game_ids"], experiment.config.rotation_cycles, experiment.config.game_batch_size)
    files["folds/0/schedule_manifest.json"] = canonical_json_bytes(changed)
    files["folds/0/training_schedule.jsonl"] = canonical_jsonl_bytes(changed["batches"])
    from copy import deepcopy
    fields = deepcopy({k: v for k, v in experiment.manifest.items() if k not in {"file_table", "manifest_digest"}})
    fields["folds"][0]["schedule_digest"] = sha256_bytes(canonical_json_bytes(changed))
    destination = tmp_path / "wrong-schedule"
    publish_artifact(destination, manifest_name="experiment_manifest.json", manifest_fields=fields, files=files)
    with pytest.raises(ValueError, match="schedule/protocol"):
        open_experiment(destination).schedule(0)
