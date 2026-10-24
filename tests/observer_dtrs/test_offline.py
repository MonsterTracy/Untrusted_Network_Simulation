from copy import deepcopy
import json

import pytest

from scripts.inspect_observer_dtrs_samples import main
from tests.development_publication.test_development_publication import _republish
from werewolf.artifact_io import ArtifactValidationError, canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection import BeliefObservationStatus, construct_belief_observation
from werewolf.observer_dtrs.offline import build_observer_samples


def test_verified_publication_samples_labels_and_lineage(publication_factory):
    handles = publication_factory(supports={"player3": (), "player4": ("player2",),
        "player5": ("player1",), "player6": ("player7",), "player7": ("player6",)})
    rows = build_observer_samples(handles.public.path)
    assert len(rows) == len({s.identity for s in rows}) == 25
    for game in handles.public.public_view.games:
        selected = [s for s in rows if s.context.game_id == game.game_id]
        assert [s.context.observer_id for s in selected] == [f"player{i}" for i in range(3, 8)]
        assert [s.supervision.event_label_y for s in selected] == [1, 1, 1, 0, 0]
        for row in selected:
            assert row.source.publication_digest == handles.public.manifest_digest
            assert row.source.role_sidecar_digest == handles.restricted.sidecar_digest
            assert row.source.bundle_digest == game.bundle_digest
            assert row.source.published_game_record_digest == game.record_digest
            assert row.source.fold_manifest_digest == handles.public.public_view.fold_manifest.manifest_digest
            fold = handles.public.public_view.fold_manifest.folds[row.source.fold_index]
            assert row.context.game_id in fold.game_ids
            observation = next(o for o in game.belief_observations if o.observer_id == row.context.observer_id)
            assert row.supervision.observation_id == observation.observation_id
            assert row.supervision.observation_digest == observation.observation_digest
            assert row.supervision.attempt_id == observation.attempt_id
            record = row.to_record()
            assert set(record) == {"schema_version", "context", "supervision", "source"}
            assert "role_assignment" not in json.dumps(record)
            assert "event_label_y" not in record["context"]
    assert build_observer_samples(handles.public.path) == rows


@pytest.mark.parametrize("phase", ("discussion", "pk_discussion"))
def test_publication_single_wolf_dead_observer_and_dead_wolf_only_label(publication_factory, phase):
    handles = publication_factory(phase=phase, dead=("player2", "player7"),
        supports={"player3": ("player2",), "player4": (),
                  "player5": ("player1", "player7"), "player6": ("player7",)})
    rows = build_observer_samples(handles.public.path)
    assert len(rows) == 20
    first = rows[:4]
    assert [s.context.observer_id for s in first] == [f"player{i}" for i in range(3, 7)]
    assert [s.supervision.event_label_y for s in first] == [0, 1, 1, 0]
    assert all(s.context.wolf_count == 1 and s.context.phase == phase for s in rows)
    assert first[0].supervision.effective_suspicion_support == ("player2",)
    assert "player7" in first[1].supervision.effective_suspicion_support


def test_pk_with_only_wolves_in_tied_targets_keeps_all_nonwolf_observers(publication_factory):
    handles = publication_factory(phase="pk_discussion")
    history = handles.public.public_view.games[0].public_event_stream
    votes = next(e.votes for e in history.events if e.event_type == "vote_result")
    assert {v.target for v in votes} == {"player1", "player2", None}
    rows = build_observer_samples(handles.public.path)
    assert len(rows) == 25 and all(s.context.phase == "pk_discussion" for s in rows)
    assert [s.context.observer_id for s in rows[:5]] == [f"player{i}" for i in range(3, 8)]


def test_nonwolf_pre_has_zero_samples_but_still_validates_all_reports(publication_factory):
    handles = publication_factory(actor="player3")
    assert build_observer_samples(handles.public.path) == ()


def test_actual_sidecar_roles_determine_team_not_seat_numbers(publication_factory):
    handles = publication_factory(actor="player5", wolves=("player3", "player5"),
                                  supports={"player1": ("player5",), "player2": ("player1",)})
    rows = build_observer_samples(handles.public.path)
    assert len(rows) == 25
    assert [s.context.observer_id for s in rows[:5]] == ["player1", "player2", "player4", "player6", "player7"]
    assert [s.supervision.event_label_y for s in rows[:5]] == [1, 0, 1, 1, 1]
    assert all(s.context.acting_wolf == "player5" and s.context.known_wolf_ids == ("player3", "player5")
               for s in rows)


def test_changing_reports_changes_labels_but_not_legal_context(publication_factory):
    positive = publication_factory(name="positive", supports={"player3": ("player1",)})
    negative = publication_factory(name="negative", supports={"player3": ("player4",)})
    left, right = build_observer_samples(positive.public.path), build_observer_samples(negative.public.path)
    assert [s.context for s in left] == [s.context for s in right]
    assert sum(a.supervision.event_label_y != b.supervision.event_label_y
               for a, b in zip(left, right, strict=True)) == 5
    assert left[0].source.publication_digest != right[0].source.publication_digest


def test_explicit_whole_game_partition(publication_factory):
    handles = publication_factory()
    game_id = handles.public.public_view.game_ids[2]
    rows = build_observer_samples(handles.public.path, game_ids=(game_id,))
    assert len(rows) == 5 and {s.context.game_id for s in rows} == {game_id}
    for ids in ((), (game_id, game_id), ("unknown-game",)):
        with pytest.raises(ArtifactValidationError):
            build_observer_samples(handles.public.path, game_ids=ids)


def corrupt_report_publication(handles, destination, mutation):
    game = handles.public.public_view.games[0]
    record = deepcopy(game.to_record())
    if mutation == "missing":
        record["belief_observations"].pop(2)
    elif mutation == "duplicate":
        record["belief_observations"].append(deepcopy(record["belief_observations"][2]))
    else:
        observation = game.belief_observations[2]
        raw = observation.to_record()
        for key in ("schema_version", "label_observed", "observation_digest"):
            raw.pop(key)
        raw["status"] = observation.status
        if mutation == "failed":
            raw.update(status=BeliefObservationStatus.ERROR, suspicion_support=(), attempts=({
                "attempt_index": 1, "call_id": "controlled-failure", "status": "error",
                "response_digest": None, "error_category": "fixture", "error_message": "failed"},))
        else:
            raw[mutation] = {"prefix_digest": "a" * 64, "boundary_id": "wrong-pre",
                             "game_id": "wrong-game", "observer_id": "player4",
                             "observation_id": "wrong-report"}[mutation]
        record["belief_observations"][2] = construct_belief_observation(**raw).to_record()
    without = {k: v for k, v in record.items() if k != "record_digest"}
    record["record_digest"] = sha256_bytes(canonical_json_bytes(without))
    return _republish(handles.public.path, destination / "observer-controlled",
        files={f"public/games/{game.game_id}.json": canonical_json_bytes(record)})


@pytest.mark.parametrize("actor", ("player1", "player3"))
@pytest.mark.parametrize("mutation", ("failed", "missing", "duplicate", "prefix_digest",
                                      "boundary_id", "game_id", "observer_id", "observation_id"))
def test_complete_publication_errors_are_never_silently_filtered(publication_factory, tmp_path, actor, mutation):
    handles = publication_factory(actor=actor)
    artifact = corrupt_report_publication(handles, tmp_path / "corrupt", mutation)
    message = {"missing": "missing Belief Observation", "observation_id": "missing Belief Observation",
               "duplicate": "identities are not unique",
               "game_id": "game identity mismatch"}.get(mutation, "PRE binding mismatch")
    with pytest.raises(ArtifactValidationError, match=message):
        build_observer_samples(artifact.path)


@pytest.mark.parametrize("field", ("publication_id", "collection_identity_digest",
                                   "development_game_set_digest", "bundle_digest"))
def test_rehashed_sidecar_with_wrong_lineage_is_rejected(publication_factory, tmp_path, field):
    handles = publication_factory()
    record = json.loads((handles.public.path / "restricted/role_sidecar.json").read_bytes())
    if field == "bundle_digest":
        record["games"][0][field] = "a" * 64
    else:
        record[field] = "wrong-publication" if field == "publication_id" else "a" * 64
    record["sidecar_digest"] = sha256_bytes(canonical_json_bytes(
        {k: v for k, v in record.items() if k != "sidecar_digest"}))
    artifact = _republish(handles.public.path, tmp_path / "wrong-sidecar/observer-controlled",
        manifest_updates={"role_sidecar_digest": record["sidecar_digest"]},
        files={"restricted/role_sidecar.json": canonical_json_bytes(record)})
    with pytest.raises(ArtifactValidationError, match="Sidecar"):
        build_observer_samples(artifact.path)


def test_read_only_inspection_entrypoint(publication_factory, capsys):
    handles = publication_factory()
    before = {p.relative_to(handles.public.path): p.read_bytes()
              for p in handles.public.path.rglob("*") if p.is_file()}
    assert main(["--publication", str(handles.public.path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["sample_count"] == report["positive_count"] == 25
    assert report["games_with_samples"] == report["wolf_pre_count"] == 5
    assert report["phase_sample_counts"] == {"discussion": 25}
    assert "raw_suspicion_support" not in report
    assert before == {p.relative_to(handles.public.path): p.read_bytes()
                      for p in handles.public.path.rglob("*") if p.is_file()}
