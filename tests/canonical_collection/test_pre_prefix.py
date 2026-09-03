from dataclasses import replace

import pytest

from werewolf.artifact_io import sha256_bytes
from werewolf.canonical_collection import (
    V1_SPEECH_PARSER_VERSION,
    V1_SPEECH_PROMPT_VERSION,
    V1AnnotationStatus,
    V1PerceptionAttempt,
    V1SpeechAction,
    construct_authoritative_pre_prefix,
    construct_speaker_pre_belief_handoff,
    construct_v1_speech_annotation,
    freeze_public_event_history,
    validate_authoritative_pre_prefix,
)


def _events(*, include_current_speech=False):
    events = [
        {
            "event_id": "event-0",
            "event_index": 0,
            "event_type": "phase_change",
            "day": 0,
            "phase": "night",
        },
        {
            "event_id": "event-1",
            "event_index": 1,
            "event_type": "death_announcement",
            "dead_players": ["player7"],
        },
        {
            "event_id": "event-2",
            "event_index": 2,
            "event_type": "phase_change",
            "day": 1,
            "phase": "discussion",
        },
        {
            "event_id": "event-3",
            "event_index": 3,
            "event_type": "turn_start",
            "speaker": "player1",
        },
        {
            "event_id": "event-4",
            "event_index": 4,
            "event_type": "public_speech",
            "speaker": "player1",
            "raw_text": "player3 is suspicious",
        },
        {
            "event_id": "event-5",
            "event_index": 5,
            "event_type": "turn_start",
            "speaker": "player2",
        },
    ]
    if include_current_speech:
        events.append(
            {
                "event_id": "event-6",
                "event_index": 6,
                "event_type": "public_speech",
                "speaker": "player2",
                "raw_text": "this must not be visible at PRE",
            }
        )
    return events


def _attempt():
    return V1PerceptionAttempt(
        attempt_index=1,
        call_id="parser-call-1",
        backend_id="parser-backend",
        model_id="parser-model",
        prompt_version=V1_SPEECH_PROMPT_VERSION,
        parser_version=V1_SPEECH_PARSER_VERSION,
        status=V1AnnotationStatus.OK,
        raw_response="player1,point_as_werewolf,player3",
        error_category=None,
        error_message=None,
    )


def _annotation():
    return construct_v1_speech_annotation(
        freeze_public_event_history(_events()[:5]),
        status=V1AnnotationStatus.OK,
        actions=(
            V1SpeechAction(
                "player1",
                "point_as_werewolf",
                "player3",
            ),
        ),
        attempts=(_attempt(),),
    )


def _observation_ids():
    return {
        f"player{seat}": f"belief-observation-player{seat}"
        for seat in range(1, 7)
    }


def _prefix(**overrides):
    arguments = {
        "game_id": "game-001",
        "boundary_id": "boundary-005",
        "step_index": 5,
        "report_trigger_id": "pre-public-speech-005",
        "current_speaker": "player2",
        "alive_observer_ids": tuple(f"player{seat}" for seat in range(1, 7)),
        "public_event_history": freeze_public_event_history(_events()),
        "v1_annotations": (_annotation(),),
        "belief_observation_ids_by_observer": _observation_ids(),
    }
    arguments.update(overrides)
    return construct_authoritative_pre_prefix(**arguments)


def test_authoritative_pre_prefix_is_full_cumulative_and_ends_in_turn_start():
    prefix = _prefix()

    assert prefix.public_event_history.events[0].temporal_state.day == 0
    assert prefix.public_event_history.events[-1].event_type == "turn_start"
    assert prefix.public_event_history.events[-1].speaker == "player2"
    assert prefix.current_speaker == "player2"
    assert prefix.speaker_pre_belief_observation_id == (
        "belief-observation-player2"
    )
    assert prefix.to_record()["public_events"][-1]["event_type"] == "turn_start"
    assert prefix.to_record()["prefix_digest"] == prefix.prefix_digest


def test_pre_prefix_rejects_r0_terminal_turn_removal():
    with pytest.raises(ValueError, match="end with matching turn_start"):
        _prefix(
            public_event_history=freeze_public_event_history(_events()[:5]),
        )


def test_pre_prefix_rejects_current_speech_event_leakage():
    with pytest.raises(ValueError, match="end with matching turn_start"):
        _prefix(
            public_event_history=freeze_public_event_history(
                _events(include_current_speech=True)
            ),
        )


def test_pre_prefix_rejects_current_speech_action_annotation_leakage():
    current_annotation = construct_v1_speech_annotation(
        freeze_public_event_history(_events(include_current_speech=True)),
        status=V1AnnotationStatus.NO_ACTION,
        actions=(),
        attempts=(replace(_attempt(), status=V1AnnotationStatus.NO_ACTION),),
    )

    with pytest.raises(ValueError, match="upcoming speech annotation"):
        _prefix(v1_annotations=(_annotation(), current_annotation))


def test_pre_prefix_requires_one_unique_observation_link_per_alive_observer():
    duplicate_ids = _observation_ids()
    duplicate_ids["player3"] = duplicate_ids["player2"]

    with pytest.raises(ValueError, match="observation IDs must be unique"):
        _prefix(belief_observation_ids_by_observer=duplicate_ids)


def test_pre_prefix_validation_reapplies_canonical_alive_observer_order():
    prefix = _prefix()

    with pytest.raises(ValueError, match="canonical seat order"):
        validate_authoritative_pre_prefix(
            replace(
                prefix,
                alive_observer_ids=(
                    "player2",
                    "player1",
                    "player3",
                    "player4",
                    "player5",
                    "player6",
                ),
            )
        )


def test_speaker_pre_belief_handoff_binds_the_one_collected_speaker_row():
    prefix = _prefix()
    observation_digest = sha256_bytes(b"speaker observation")

    handoff = construct_speaker_pre_belief_handoff(
        prefix,
        observation_id="belief-observation-player2",
        observation_digest=observation_digest,
        observer_id="player2",
        observation_status="success",
        suspicion_support=("player3", "player6"),
    )

    assert handoff.observation_id == prefix.speaker_pre_belief_observation_id
    assert handoff.prefix_digest == prefix.prefix_digest
    assert handoff.suspicion_support == ("player3", "player6")
    assert handoff.to_record()["source"] == "realized_pre_belief_observation"


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("observation_id", "second-belief", "one linked speaker observation"),
        ("observer_id", "player3", "same current speaker"),
        ("observation_status", "error", "successful observation"),
    ],
)
def test_speaker_pre_belief_handoff_rejects_a_second_or_failed_source(
    field,
    value,
    match,
):
    arguments = {
        "observation_id": "belief-observation-player2",
        "observation_digest": sha256_bytes(b"speaker observation"),
        "observer_id": "player2",
        "observation_status": "success",
        "suspicion_support": ("player3",),
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=match):
        construct_speaker_pre_belief_handoff(_prefix(), **arguments)
