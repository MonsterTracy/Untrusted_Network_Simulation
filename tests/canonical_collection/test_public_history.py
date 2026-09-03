from dataclasses import replace

import pytest

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection import (
    PUBLIC_PHASES,
    PublicPhase,
    PublicTemporalState,
    freeze_public_event_history,
    validate_public_event_history,
)


def _phase(event_index, day, phase):
    return {
        "event_id": f"event-{event_index}",
        "event_index": event_index,
        "event_type": "phase_change",
        "day": day,
        "phase": phase,
    }


def _event(event_index, event_type, **fields):
    return {
        "event_id": f"event-{event_index}",
        "event_index": event_index,
        "event_type": event_type,
        **fields,
    }


def test_night0_and_all_five_public_phases_are_causally_inherited():
    history = freeze_public_event_history(
        [
            _phase(0, 0, "night"),
            _event(1, "death_announcement", dead_players=["player7"]),
            _phase(2, 1, "discussion"),
            _phase(3, 1, "vote"),
            _event(
                4,
                "vote_result",
                votes=[{"voter": "player1", "target": "player2"}],
            ),
            _phase(5, 1, "pk_discussion"),
            _phase(6, 1, "pk_vote"),
            _event(7, "exile_result", exiled_players=["player2"]),
            _phase(8, 1, "night"),
            _event(9, "death_announcement", dead_players=["player6"]),
            _phase(10, 2, "discussion"),
        ]
    )

    assert PUBLIC_PHASES == (
        "night",
        "discussion",
        "vote",
        "pk_discussion",
        "pk_vote",
    )
    assert history.events[0].temporal_state.day == 0
    assert history.events[0].temporal_state.phase is PublicPhase.NIGHT
    assert history.events[1].temporal_state == history.events[0].temporal_state
    assert history.events[2].temporal_state.phase is PublicPhase.DISCUSSION
    assert history.events[4].temporal_state.phase is PublicPhase.VOTE
    assert history.events[5].temporal_state.phase is PublicPhase.PK_DISCUSSION
    assert history.events[7].temporal_state.phase is PublicPhase.PK_VOTE
    assert history.events[9].temporal_state.phase is PublicPhase.NIGHT
    assert history.current_state.day == 2
    assert history.current_state.phase is PublicPhase.DISCUSSION
    assert history.max_public_day == 2


@pytest.mark.parametrize("private_phase", ["skill_wolf", "skill_seer", "skill_witch"])
def test_private_scheduler_phase_is_rejected(private_phase):
    with pytest.raises(ValueError, match="Public Phase"):
        freeze_public_event_history([_phase(0, 0, private_phase)])


def test_missing_explicit_night0_initial_state_is_rejected():
    with pytest.raises(ValueError, match="day=0, phase=night"):
        freeze_public_event_history([_phase(0, 1, "discussion")])


def test_day1_cannot_begin_before_the_public_night0_result():
    with pytest.raises(ValueError, match="death_announcement"):
        freeze_public_event_history(
            [
                _phase(0, 0, "night"),
                _phase(1, 1, "discussion"),
            ]
        )


def test_ordinary_event_cannot_receive_state_from_a_future_phase_change():
    with pytest.raises(ValueError, match="future phase_change"):
        freeze_public_event_history(
            [
                _event(0, "death_announcement", dead_players=[]),
                _phase(1, 0, "night"),
            ]
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (
            "current_state",
            PublicTemporalState(day=99, phase=PublicPhase.DISCUSSION),
            "current_state",
        ),
        ("max_public_day", 99, "max_public_day"),
    ],
)
def test_public_history_validation_rederives_frozen_summary_fields(
    field,
    value,
    match,
):
    history = freeze_public_event_history(
        [
            _phase(0, 0, "night"),
            _event(1, "death_announcement", dead_players=[]),
            _phase(2, 1, "discussion"),
        ]
    )

    with pytest.raises(ValueError, match=match):
        validate_public_event_history(replace(history, **{field: value}))


def test_public_history_validation_rejects_forged_typed_transition():
    history = freeze_public_event_history(
        [
            _phase(0, 0, "night"),
            _event(1, "death_announcement", dead_players=[]),
            _phase(2, 1, "discussion"),
        ]
    )
    forged_state = PublicTemporalState(day=3, phase=PublicPhase.DISCUSSION)
    forged_events = (
        *history.events[:2],
        replace(history.events[2], temporal_state=forged_state),
    )
    forged = replace(
        history,
        events=forged_events,
        current_state=forged_state,
        max_public_day=3,
    )
    forged = replace(
        forged,
        digest=sha256_bytes(canonical_json_bytes(forged.to_records())),
    )

    with pytest.raises(ValueError, match="invalid causal Public Phase transition"):
        validate_public_event_history(forged)
