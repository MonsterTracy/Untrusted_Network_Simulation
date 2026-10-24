from dataclasses import fields, replace
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tests.observer_dtrs.conftest import observation_for, prefix_for
from werewolf.artifact_io import ArtifactValidationError
from werewolf.canonical_collection import BeliefObservationStatus, construct_belief_observation
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.observer_dtrs.context import ObserverContext, contexts_for_pre
from werewolf.observer_dtrs.labels import label_from_observation


WOLVES = ("player1", "player2")


@pytest.mark.parametrize("phase", ("discussion", "pk_discussion"))
def test_canonical_population_and_event_step_are_preserved(phase):
    pre = prefix_for(phase=phase)
    contexts = contexts_for_pre(pre, WOLVES)
    assert [c.observer_id for c in contexts] == list(PLAYER_IDS[2:])
    assert all(c.phase == phase and c.acting_wolf == "player1" for c in contexts)
    assert all(c.step_index == pre.step_index == pre.public_event_history.events[-1].event_index
               and c.prefix_digest == pre.prefix_digest for c in contexts)
    assert all(c.wolf_count == 2 and c.alive_wolf_ids == WOLVES for c in contexts)


def test_nonwolf_speaker_is_not_a_sample_and_dead_observers_are_excluded():
    assert contexts_for_pre(prefix_for(actor="player3"), WOLVES) == ()
    contexts = contexts_for_pre(prefix_for(dead=("player2", "player7")), WOLVES)
    assert [c.observer_id for c in contexts] == list(PLAYER_IDS[2:6])
    assert all(c.wolf_count == 1 and c.alive_wolf_ids == ("player1",) for c in contexts)
    assert all(c.known_wolf_ids == WOLVES for c in contexts)


@pytest.mark.parametrize("team", (("player1",), ("player1", "player1"), ("player2", "player1"),
                                  ("P1", "P2")))
def test_noncanonical_team_is_rejected(team):
    with pytest.raises(ValueError):
        contexts_for_pre(prefix_for(), team)


@pytest.mark.parametrize(("support", "expected"), (((), 1), (("player1",), 1),
    (("player2",), 0), (("player4",), 0), (("player1", "player2", "player7"), 1)))
def test_effective_support_uses_live_wolves_and_keeps_dead_targets(support, expected):
    context = contexts_for_pre(prefix_for(dead=("player2", "player7")), WOLVES)[0]
    observation = observation_for(context, support)
    label = label_from_observation(context, observation)
    assert label.event_label_y == expected and label.label_observed is True
    assert label.raw_suspicion_support == support
    effective = support or tuple(p for p in PLAYER_IDS if p != context.observer_id)
    assert label.effective_suspicion_support == effective
    assert observation.suspicion_support == support
    if not support:
        assert len(effective) == 6 and "player2" in effective and "player7" in effective


def test_failed_and_missing_reports_are_neither_empty_success_nor_negative():
    context = contexts_for_pre(prefix_for(), WOLVES)[0]
    error = observation_for(context, status=BeliefObservationStatus.ERROR)
    for observation, status in ((None, "missing"), (error, "error")):
        label = label_from_observation(context, observation)
        assert label.observation_status == status and label.label_observed is False
        assert label.event_label_y is None and label.effective_suspicion_support is None
    assert label_from_observation(context, None).raw_suspicion_support is None
    assert label_from_observation(context, error).raw_suspicion_support == ()


@pytest.mark.parametrize(("field", "value"), (("game_id", "another-game"),
    ("boundary_id", "another-pre"), ("prefix_digest", "a" * 64),
    ("observer_id", "player4"), ("day", 2), ("phase", "pk_discussion"),
    ("observer_alive", False)))
def test_valid_but_misbound_reports_fail(field, value):
    context = contexts_for_pre(prefix_for(), WOLVES)[0]
    observation = observation_for(context)
    record = observation.to_record()
    for key in ("schema_version", "label_observed", "observation_digest"):
        record.pop(key)
    record.update(status=observation.status, **{field: value})
    changed = construct_belief_observation(**record)
    with pytest.raises(ValueError, match="bind"):
        label_from_observation(context, changed)


@pytest.mark.parametrize("change", ({"label_observed": False}, {"observation_digest": "a" * 64},
                                   {"suspicion_support": ("player3",)}))
def test_forged_canonical_report_uses_existing_validation(change):
    context = contexts_for_pre(prefix_for(), WOLVES)[0]
    with pytest.raises((ArtifactValidationError, ValueError)):
        label_from_observation(context, replace(observation_for(context), **change))


def test_legal_schema_and_imports_exclude_supervision():
    import werewolf.observer_dtrs as package
    import werewolf.observer_dtrs.context as legal
    names = {f.name for f in fields(ObserverContext)}
    assert names == {"game_id", "boundary_id", "prefix_digest", "step_index", "acting_wolf",
                     "observer_id", "day", "phase", "alive_player_ids", "known_wolf_ids"}
    assert not any(token in inspect.getsource(legal) for token in
                   ("open_role_sidecar", "BeliefObservation", "event_label_y", "phase2_", "tom.dataset"))
    assert "import" not in inspect.getsource(package).split('"""')[-1]
    context = contexts_for_pre(prefix_for(), WOLVES)[0]
    before = context.to_record()
    positive = label_from_observation(context, observation_for(context, ("player1",)))
    negative = label_from_observation(context, observation_for(context, ("player4",)))
    assert positive.event_label_y == 1 and negative.event_label_y == 0
    assert context.to_record() == before


def test_importing_legal_context_does_not_load_offline_supervision_or_models():
    script = """
import json, sys
from werewolf.observer_dtrs.context import contexts_for_pre
print(json.dumps([name for name in sys.modules if name in (
    'werewolf.observer_dtrs.labels', 'werewolf.observer_dtrs.offline',
    'werewolf.development_publication') or name.startswith((
    'werewolf.tom', 'werewolf.phase2_', 'torch', 'transformers'))]))
"""
    result = subprocess.run([sys.executable, "-B", "-c", script], check=True,
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
    assert json.loads(result.stdout) == []
