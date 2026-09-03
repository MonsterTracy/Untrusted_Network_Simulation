import pytest

from werewolf.canonical_collection import (
    V1_SPEECH_PARSER_VERSION,
    V1_SPEECH_PROMPT_VERSION,
    V1AnnotationStatus,
    V1PerceptionAttempt,
    V1SpeechAction,
    construct_authoritative_pre_prefix,
    construct_v1_speech_annotation,
    freeze_public_event_history,
)
from werewolf.structured_history import plan_structured_history


def _events():
    return [
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
            "raw_text": "player3 is suspicious; I support player4",
        },
        {
            "event_id": "event-5",
            "event_index": 5,
            "event_type": "phase_change",
            "day": 1,
            "phase": "vote",
        },
        {
            "event_id": "event-6",
            "event_index": 6,
            "event_type": "vote_result",
            "votes": [{"voter": "player1", "target": "player3"}],
        },
        {
            "event_id": "event-7",
            "event_index": 7,
            "event_type": "exile_result",
            "exiled_players": ["player3"],
        },
        {
            "event_id": "event-8",
            "event_index": 8,
            "event_type": "phase_change",
            "day": 1,
            "phase": "night",
        },
        {
            "event_id": "event-9",
            "event_index": 9,
            "event_type": "death_announcement",
            "dead_players": ["player6"],
        },
        {
            "event_id": "event-10",
            "event_index": 10,
            "event_type": "phase_change",
            "day": 2,
            "phase": "discussion",
        },
        {
            "event_id": "event-11",
            "event_index": 11,
            "event_type": "turn_start",
            "speaker": "player2",
        },
    ]


def _prefix():
    events = _events()
    speech_history = freeze_public_event_history(events[:5])
    annotation = construct_v1_speech_annotation(
        speech_history,
        status=V1AnnotationStatus.OK,
        actions=(
            V1SpeechAction(
                "player1",
                "point_as_werewolf",
                "player3",
            ),
            V1SpeechAction("player1", "support", "player4"),
        ),
        attempts=(
            V1PerceptionAttempt(
                attempt_index=1,
                call_id="parser-call-1",
                backend_id="parser-backend",
                model_id="parser-model",
                prompt_version=V1_SPEECH_PROMPT_VERSION,
                parser_version=V1_SPEECH_PARSER_VERSION,
                status=V1AnnotationStatus.OK,
                raw_response="two actions",
                error_category=None,
                error_message=None,
            ),
        ),
    )
    alive = ("player1", "player2", "player4", "player5")
    return construct_authoritative_pre_prefix(
        game_id="game-001",
        boundary_id="boundary-011",
        step_index=11,
        report_trigger_id="pre-public-speech-011",
        current_speaker="player2",
        alive_observer_ids=alive,
        public_event_history=freeze_public_event_history(events),
        v1_annotations=(annotation,),
        belief_observation_ids_by_observer={
            observer: f"belief-{observer}" for observer in alive
        },
    )


def test_structured_token_plan_has_frozen_order_state_count_and_digest():
    prefix = _prefix()
    first = plan_structured_history(prefix)
    second = plan_structured_history(prefix)

    assert [token.token_type for token in first.tokens] == [
        "phase_change",
        "death_announcement",
        "dead_player",
        "phase_change",
        "turn_start",
        "public_speech",
        "speech_action",
        "speech_action",
        "phase_change",
        "vote_result",
        "vote",
        "exile_result",
        "exiled_player",
        "phase_change",
        "death_announcement",
        "dead_player",
        "phase_change",
        "turn_start",
    ]
    assert [(token.day, token.phase) for token in first.tokens] == [
        (0, "night"),
        (0, "night"),
        (0, "night"),
        (1, "discussion"),
        (1, "discussion"),
        (1, "discussion"),
        (1, "discussion"),
        (1, "discussion"),
        (1, "vote"),
        (1, "vote"),
        (1, "vote"),
        (1, "vote"),
        (1, "vote"),
        (1, "night"),
        (1, "night"),
        (1, "night"),
        (2, "discussion"),
        (2, "discussion"),
    ]
    assert first.token_count == 18
    assert first.plan_digest == (
        "6cee336c77528d5f9924592d8b4b2879979d13127427f32c289ecfaf0a8d92d4"
    )
    assert first.tokens[-1].source == "player2"
    assert first.tokens[-1].to_record() == {
        "token_index": 17,
        "token_type": "turn_start",
        "source": "player2",
        "action": None,
        "target": None,
        "day": 2,
        "phase": "discussion",
        "event_id": "event-11",
        "event_index": 11,
        "semantic_index": 0,
    }
    assert all("raw_text" not in token.to_record() for token in first.tokens)
    assert first == second


def test_phase_change_and_speech_actions_carry_their_causal_event_state():
    plan = plan_structured_history(_prefix())

    day_one_transition = plan.tokens[3]
    speech_boundary = plan.tokens[5]
    speech_actions = plan.tokens[6:8]

    assert (day_one_transition.day, day_one_transition.phase) == (
        1,
        "discussion",
    )
    assert all(
        (token.day, token.phase)
        == (speech_boundary.day, speech_boundary.phase)
        for token in speech_actions
    )


def test_planner_accepts_only_an_authoritative_pre_prefix():
    prefix = _prefix()

    with pytest.raises(TypeError, match="AuthoritativePREPrefix"):
        plan_structured_history(prefix.to_record())


def test_token_count_is_available_without_dataset_or_tensorization():
    plan = plan_structured_history(_prefix())

    assert plan.token_count == len(plan.tokens)
    assert all(isinstance(token.token_type, str) for token in plan.tokens)
    assert not any("tensor" in key for key in plan.to_record())
