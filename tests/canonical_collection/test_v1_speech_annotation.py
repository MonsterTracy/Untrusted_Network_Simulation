from dataclasses import replace

import pytest

from werewolf.canonical_collection import (
    V1_ANNOTATION_STATUSES,
    V1_SPEECH_ONTOLOGY_VERSION,
    V1_SPEECH_PARSER_VERSION,
    V1_SPEECH_PROMPT_VERSION,
    V1AnnotationStatus,
    V1PerceptionAttempt,
    V1SpeechAction,
    construct_v1_speech_annotation,
    freeze_public_event_history,
    validate_v1_annotation_binding,
)


def _speech_history(
    raw_text="player2 looks suspicious",
    *,
    night0_dead_players=(),
):
    return freeze_public_event_history(
        [
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
                "dead_players": list(night0_dead_players),
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
                "raw_text": raw_text,
            },
        ]
    )


def _attempt(status, *, raw_response, error_category=None, error_message=None):
    return V1PerceptionAttempt(
        attempt_index=1,
        call_id="parser-call-1",
        backend_id="parser-backend",
        model_id="parser-model",
        prompt_version=V1_SPEECH_PROMPT_VERSION,
        parser_version=V1_SPEECH_PARSER_VERSION,
        status=status,
        raw_response=raw_response,
        error_category=error_category,
        error_message=error_message,
    )


def test_v1_ok_annotation_binds_ordered_actions_and_full_provenance():
    history = _speech_history()
    actions = (
        V1SpeechAction("player1", "point_as_werewolf", "player2"),
        V1SpeechAction("player1", "support", "player3"),
    )

    annotation = construct_v1_speech_annotation(
        history,
        status=V1AnnotationStatus.OK,
        actions=actions,
        attempts=(
            _attempt(
                V1AnnotationStatus.OK,
                raw_response="player1,point_as_werewolf,player2",
            ),
        ),
    )

    assert V1_ANNOTATION_STATUSES == ("ok", "no_action", "error")
    assert annotation.is_successful
    assert annotation.actions == actions
    assert annotation.event_id == "event-4"
    assert annotation.public_event_stream_digest == history.digest
    assert annotation.to_record()["ontology_version"] == V1_SPEECH_ONTOLOGY_VERSION
    assert validate_v1_annotation_binding(annotation, history) is annotation


def test_v1_no_action_is_a_successful_empty_semantic_observation():
    annotation = construct_v1_speech_annotation(
        _speech_history("I have no read yet"),
        status=V1AnnotationStatus.NO_ACTION,
        actions=(),
        attempts=(
            _attempt(V1AnnotationStatus.NO_ACTION, raw_response="NONE"),
        ),
    )

    assert annotation.is_successful
    assert annotation.status is V1AnnotationStatus.NO_ACTION
    assert annotation.actions == ()


def test_v1_error_is_exhausted_failure_not_no_action():
    annotation = construct_v1_speech_annotation(
        _speech_history(),
        status=V1AnnotationStatus.ERROR,
        actions=(),
        attempts=(
            _attempt(
                V1AnnotationStatus.ERROR,
                raw_response=None,
                error_category="backend_timeout",
                error_message="bounded attempts exhausted",
            ),
        ),
    )

    assert not annotation.is_successful
    assert annotation.status is V1AnnotationStatus.ERROR
    assert annotation.actions == ()
    assert annotation.to_record()["attempts"][0]["error_category"] == (
        "backend_timeout"
    )


def test_v1_status_and_action_semantics_fail_closed():
    with pytest.raises(ValueError, match="no_action requires an empty"):
        construct_v1_speech_annotation(
            _speech_history(),
            status=V1AnnotationStatus.NO_ACTION,
            actions=(
                V1SpeechAction(
                    "player1",
                    "point_as_werewolf",
                    "player2",
                ),
            ),
            attempts=(
                _attempt(V1AnnotationStatus.NO_ACTION, raw_response="NONE"),
            ),
        )


def test_v1_binding_rejects_changed_raw_text_or_event_stream():
    history = _speech_history()
    annotation = construct_v1_speech_annotation(
        history,
        status=V1AnnotationStatus.NO_ACTION,
        actions=(),
        attempts=(
            _attempt(V1AnnotationStatus.NO_ACTION, raw_response="NONE"),
        ),
    )

    with pytest.raises(ValueError, match="raw-text digest"):
        validate_v1_annotation_binding(
            annotation,
            _speech_history("changed speech"),
        )

    with pytest.raises(ValueError, match="public-event-stream digest"):
        validate_v1_annotation_binding(
            annotation,
            _speech_history(night0_dead_players=("player7",)),
        )

    with pytest.raises(ValueError, match="annotation digest"):
        validate_v1_annotation_binding(
            replace(annotation, annotation_digest="0" * 64),
            history,
        )
