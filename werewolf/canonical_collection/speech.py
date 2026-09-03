"""The sole V1 public-speech semantic annotation contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection.public_history import (
    PLAYER_IDS,
    PublicEventHistory,
)


V1_ANNOTATION_SCHEMA_VERSION = "classic7_v1_speech_annotation_v1"
V1_SPEECH_ONTOLOGY_VERSION = "classic7_speech_action_v1"
V1_SPEECH_PROMPT_VERSION = "classic7_v1_speech_perception_prompt_v1"
V1_SPEECH_PARSER_VERSION = "classic7_v1_speech_parser_v1"
V1_ANNOTATION_STATUSES = ("ok", "no_action", "error")
V1_ACTIONS = (
    "point_as_werewolf",
    "point_as_non_werewolf",
    "point_as_villager",
    "point_as_seer",
    "point_as_witch",
    "support",
    "oppose",
    "check_as_non_werewolf",
    "check_as_werewolf",
    "save",
    "poison",
    "vote_intent",
    "abstain_intent",
    "no_commitment",
)
_TARGETLESS_ACTIONS = frozenset({"abstain_intent", "no_commitment"})


class V1AnnotationStatus(str, Enum):
    OK = "ok"
    NO_ACTION = "no_action"
    ERROR = "error"


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return value


@dataclass(frozen=True)
class V1SpeechAction:
    subject: str
    action: str
    object: str | None

    def __post_init__(self) -> None:
        if self.subject not in PLAYER_IDS:
            raise ValueError("V1 action subject must be a canonical player")
        if self.action not in V1_ACTIONS:
            raise ValueError(f"unsupported V1 speech action: {self.action!r}")
        if self.action in _TARGETLESS_ACTIONS:
            if self.object is not None:
                raise ValueError(f"{self.action} requires object=None")
        elif self.object not in PLAYER_IDS:
            raise ValueError(f"{self.action} requires a canonical player object")

    def to_record(self) -> list[str | None]:
        return [self.subject, self.action, self.object]


@dataclass(frozen=True)
class V1PerceptionAttempt:
    attempt_index: int
    call_id: str
    backend_id: str
    model_id: str
    prompt_version: str
    parser_version: str
    status: V1AnnotationStatus
    raw_response: str | None
    error_category: str | None
    error_message: str | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.attempt_index, bool)
            or not isinstance(self.attempt_index, int)
            or self.attempt_index < 1
        ):
            raise ValueError("V1 attempt_index must be a positive integer")
        for field_name in ("call_id", "backend_id", "model_id"):
            _required_text(getattr(self, field_name), field_name)
        if self.prompt_version != V1_SPEECH_PROMPT_VERSION:
            raise ValueError("unsupported V1 prompt_version")
        if self.parser_version != V1_SPEECH_PARSER_VERSION:
            raise ValueError("unsupported V1 parser_version")
        if not isinstance(self.status, V1AnnotationStatus):
            raise TypeError("V1 attempt status must be V1AnnotationStatus")
        if self.status is V1AnnotationStatus.ERROR:
            _required_text(self.error_category, "V1 attempt error_category")
            _required_text(self.error_message, "V1 attempt error_message")
            if self.raw_response is not None and not isinstance(
                self.raw_response,
                str,
            ):
                raise TypeError("V1 attempt raw_response must be text or None")
        else:
            if not isinstance(self.raw_response, str):
                raise TypeError("successful V1 attempt requires raw_response text")
            if self.error_category is not None or self.error_message is not None:
                raise ValueError("successful V1 attempt cannot contain an error")

    def to_record(self) -> dict[str, Any]:
        return {
            "attempt_index": self.attempt_index,
            "call_id": self.call_id,
            "backend_id": self.backend_id,
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "parser_version": self.parser_version,
            "status": self.status.value,
            "raw_response": self.raw_response,
            "error_category": self.error_category,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class V1SpeechAnnotation:
    schema_version: str
    event_id: str
    event_index: int
    speaker: str
    raw_text_digest: str
    public_event_stream_digest: str
    ontology_version: str
    status: V1AnnotationStatus
    actions: tuple[V1SpeechAction, ...]
    attempts: tuple[V1PerceptionAttempt, ...]
    annotation_digest: str

    @property
    def is_successful(self) -> bool:
        return self.status in {
            V1AnnotationStatus.OK,
            V1AnnotationStatus.NO_ACTION,
        }

    def to_record(self) -> dict[str, Any]:
        return {
            **_annotation_record_without_digest(self),
            "annotation_digest": self.annotation_digest,
        }


def _validate_attempts(
    attempts: Sequence[V1PerceptionAttempt],
    final_status: V1AnnotationStatus,
) -> tuple[V1PerceptionAttempt, ...]:
    if not attempts:
        raise ValueError("V1 annotation requires at least one perception attempt")
    frozen = tuple(attempts)
    for expected_index, attempt in enumerate(frozen, start=1):
        if not isinstance(attempt, V1PerceptionAttempt):
            raise TypeError("V1 attempts must be V1PerceptionAttempt values")
        if attempt.attempt_index != expected_index:
            raise ValueError("V1 attempt indices must be contiguous from one")
        if (
            expected_index < len(frozen)
            and attempt.status is not V1AnnotationStatus.ERROR
        ):
            raise ValueError("V1 perception must stop after its first success")
    if frozen[-1].status is not final_status:
        raise ValueError("V1 final status must match the final attempt")
    return frozen


def _validate_actions(
    actions: Sequence[V1SpeechAction],
    *,
    speaker: str,
    status: V1AnnotationStatus,
) -> tuple[V1SpeechAction, ...]:
    frozen = tuple(actions)
    if any(not isinstance(action, V1SpeechAction) for action in frozen):
        raise TypeError("V1 actions must be V1SpeechAction values")
    if any(action.subject != speaker for action in frozen):
        raise ValueError("V1 action subject must equal the public speaker")
    if status is V1AnnotationStatus.OK and not frozen:
        raise ValueError("V1 status ok requires at least one action")
    if status is V1AnnotationStatus.NO_ACTION and frozen:
        raise ValueError("V1 status no_action requires an empty action list")
    if status is V1AnnotationStatus.ERROR and frozen:
        raise ValueError("V1 status error requires an empty action list")
    return frozen


def _annotation_record_without_digest(
    annotation: V1SpeechAnnotation,
) -> dict[str, Any]:
    return {
        "schema_version": annotation.schema_version,
        "event_id": annotation.event_id,
        "event_index": annotation.event_index,
        "speaker": annotation.speaker,
        "raw_text_digest": annotation.raw_text_digest,
        "public_event_stream_digest": annotation.public_event_stream_digest,
        "ontology_version": annotation.ontology_version,
        "status": annotation.status.value,
        "actions": [action.to_record() for action in annotation.actions],
        "attempts": [attempt.to_record() for attempt in annotation.attempts],
    }


def _digest_history_through(
    history: PublicEventHistory,
    event_index: int,
) -> str:
    records = [
        event.to_record()
        for event in history.events[: event_index + 1]
    ]
    return sha256_bytes(canonical_json_bytes(records))


def construct_v1_speech_annotation(
    public_event_history: PublicEventHistory,
    *,
    status: V1AnnotationStatus,
    actions: Sequence[V1SpeechAction],
    attempts: Sequence[V1PerceptionAttempt],
) -> V1SpeechAnnotation:
    """Freeze one V1 result for the terminal public speech in a history."""

    if not isinstance(public_event_history, PublicEventHistory):
        raise TypeError("V1 perception requires a frozen PublicEventHistory")
    if not public_event_history.events:
        raise ValueError("V1 perception requires one public_speech event")
    public_speech = public_event_history.events[-1]
    if public_speech.event_type != "public_speech":
        raise ValueError("V1 perception history must end with public_speech")
    if not isinstance(status, V1AnnotationStatus):
        raise TypeError("V1 annotation status must be V1AnnotationStatus")
    assert public_speech.speaker is not None
    assert public_speech.raw_text is not None
    frozen_actions = _validate_actions(
        actions,
        speaker=public_speech.speaker,
        status=status,
    )
    frozen_attempts = _validate_attempts(attempts, status)
    provisional = V1SpeechAnnotation(
        schema_version=V1_ANNOTATION_SCHEMA_VERSION,
        event_id=public_speech.event_id,
        event_index=public_speech.event_index,
        speaker=public_speech.speaker,
        raw_text_digest=sha256_bytes(public_speech.raw_text.encode("utf-8")),
        public_event_stream_digest=public_event_history.digest,
        ontology_version=V1_SPEECH_ONTOLOGY_VERSION,
        status=status,
        actions=frozen_actions,
        attempts=frozen_attempts,
        annotation_digest="",
    )
    digest = sha256_bytes(
        canonical_json_bytes(_annotation_record_without_digest(provisional))
    )
    return V1SpeechAnnotation(
        **{
            **provisional.__dict__,
            "annotation_digest": digest,
        }
    )


def validate_v1_annotation_binding(
    annotation: V1SpeechAnnotation,
    public_event_history: PublicEventHistory,
) -> V1SpeechAnnotation:
    """Validate one frozen annotation against its exact public-speech prefix."""

    if not isinstance(annotation, V1SpeechAnnotation):
        raise TypeError("annotation must be a V1SpeechAnnotation")
    if not isinstance(public_event_history, PublicEventHistory):
        raise TypeError("binding requires a frozen PublicEventHistory")
    if annotation.schema_version != V1_ANNOTATION_SCHEMA_VERSION:
        raise ValueError("unsupported V1 annotation schema_version")
    if annotation.ontology_version != V1_SPEECH_ONTOLOGY_VERSION:
        raise ValueError("unsupported V1 speech ontology_version")
    if not 0 <= annotation.event_index < len(public_event_history.events):
        raise ValueError("V1 annotation event_index is outside public history")
    event = public_event_history.events[annotation.event_index]
    if event.event_type != "public_speech" or event.event_id != annotation.event_id:
        raise ValueError("V1 annotation is not bound to its public_speech event")
    if event.speaker != annotation.speaker:
        raise ValueError("V1 annotation speaker binding mismatch")
    assert event.raw_text is not None
    if sha256_bytes(event.raw_text.encode("utf-8")) != annotation.raw_text_digest:
        raise ValueError("V1 annotation raw-text digest mismatch")
    if (
        _digest_history_through(public_event_history, annotation.event_index)
        != annotation.public_event_stream_digest
    ):
        raise ValueError("V1 annotation public-event-stream digest mismatch")
    _validate_actions(
        annotation.actions,
        speaker=annotation.speaker,
        status=annotation.status,
    )
    _validate_attempts(annotation.attempts, annotation.status)
    expected_digest = sha256_bytes(
        canonical_json_bytes(_annotation_record_without_digest(annotation))
    )
    if annotation.annotation_digest != expected_digest:
        raise ValueError("V1 annotation digest mismatch")
    return annotation
