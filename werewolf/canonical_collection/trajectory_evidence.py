"""Frozen per-game evidence values for Canonical Game Bundles."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection.frozen_json import (
    FrozenJSONValue,
    freeze_json_object,
    freeze_json_value,
)
from werewolf.canonical_collection.pre import (
    AuthoritativePREPrefix,
    SpeakerPREBeliefHandoff,
    construct_speaker_pre_belief_handoff,
    validate_authoritative_pre_prefix,
)
from werewolf.canonical_collection.public_history import (
    PLAYER_IDS,
    PUBLIC_PHASES,
    PublicEventHistory,
    validate_public_event_history,
)
from werewolf.canonical_collection.speech import (
    V1AnnotationStatus,
    V1SpeechAnnotation,
    validate_v1_annotation_binding,
)


BELIEF_OBSERVATION_SCHEMA_VERSION = "classic7_belief_observation_v1"
SUBMITTED_GAMEPLAY_ACTION_SCHEMA_VERSION = (
    "classic7_submitted_gameplay_action_v1"
)
PRIVATE_REPLAY_EVIDENCE_SCHEMA_VERSION = "classic7_private_replay_evidence_v1"
BACKEND_CALL_EVIDENCE_SCHEMA_VERSION = "classic7_backend_call_evidence_v1"
CALL_BUDGET_SUMMARY_SCHEMA_VERSION = "classic7_call_budget_summary_v1"
PARSER_SUMMARY_SCHEMA_VERSION = "classic7_parser_summary_v1"
CANONICAL_GAME_EVIDENCE_SCHEMA_VERSION = "classic7_game_evidence_v1"
_PLAYER_ORDER = {player_id: seat for seat, player_id in enumerate(PLAYER_IDS)}
_CLASSIC7_ROLE_COUNTS = {
    "Werewolf": 2,
    "Villager": 3,
    "Seer": 1,
    "Witch": 1,
}


class BeliefObservationStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"


class BackendCallPurpose(str, Enum):
    BELIEF_OBSERVATION = "belief_observation"
    SPEECH_PERCEPTION = "speech_perception"
    SPEECH_GENERATION = "speech_generation"
    GAMEPLAY_ACTION = "gameplay_action"
    RUNTIME = "runtime"


class BackendCallStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"


@dataclass(frozen=True)
class BeliefObservationAttempt:
    attempt_index: int
    call_id: str
    status: BeliefObservationStatus
    response_digest: str | None
    error_category: str | None
    error_message: str | None

    def to_record(self) -> dict[str, Any]:
        return {
            "attempt_index": self.attempt_index,
            "call_id": self.call_id,
            "status": self.status.value,
            "response_digest": self.response_digest,
            "error_category": self.error_category,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class BeliefObservation:
    schema_version: str
    game_id: str
    attempt_id: str
    boundary_id: str
    prefix_digest: str
    observation_id: str
    observer_id: str
    observer_alive: bool
    day: int
    phase: str
    status: BeliefObservationStatus
    label_observed: bool
    suspicion_support: tuple[str, ...]
    attempts: tuple[BeliefObservationAttempt, ...]
    observation_digest: str

    def to_record(self) -> dict[str, Any]:
        return {
            **_belief_record_without_digest(self),
            "observation_digest": self.observation_digest,
        }


@dataclass(frozen=True)
class BackendCallEvidence:
    schema_version: str
    call_id: str
    operation_id: str
    purpose: BackendCallPurpose
    status: BackendCallStatus
    boundary_id: str | None
    observer_id: str | None
    attempt_index: int
    backend_identity: str
    model_identity: str
    parser_identity: str
    prompt_identity: str
    retry_policy_identity: str
    call_budget_identity: str
    private_payload: FrozenJSONValue
    call_digest: str

    def to_record(self) -> dict[str, Any]:
        return {
            **_backend_call_record_without_digest(self),
            "call_digest": self.call_digest,
        }


@dataclass(frozen=True)
class CallBudgetSummary:
    schema_version: str
    configured_call_limit: int
    used_calls: int
    retry_calls: int
    fallback_action_count: int
    second_speaker_belief_count: int
    opaque_call_digests: tuple[str, ...]
    summary_digest: str

    def to_record(self) -> dict[str, Any]:
        return {
            **_call_budget_record_without_digest(self),
            "summary_digest": self.summary_digest,
        }


@dataclass(frozen=True)
class ParserSummary:
    schema_version: str
    public_speech_count: int
    ok_count: int
    no_action_count: int
    error_count: int
    parser_versions: tuple[str, ...]
    prompt_versions: tuple[str, ...]
    summary_digest: str

    def to_record(self) -> dict[str, Any]:
        return {
            **_parser_summary_record_without_digest(self),
            "summary_digest": self.summary_digest,
        }


@dataclass(frozen=True)
class SubmittedGameplayAction:
    schema_version: str
    action_id: str
    step_index: int
    actor_id: str
    action_type: str
    action_payload: FrozenJSONValue
    resulting_public_event_ids: tuple[str, ...]
    boundary_id: str | None
    speaker_pre_belief_handoff: SpeakerPREBeliefHandoff | None
    fallback_used: bool
    action_digest: str

    def to_record(self) -> dict[str, Any]:
        return {
            **_submitted_action_record_without_digest(self),
            "action_digest": self.action_digest,
        }


@dataclass(frozen=True)
class PrivateReplayEvidence:
    schema_version: str
    game_id: str
    seed: int
    role_assignment: tuple[tuple[str, str], ...]
    runtime_configuration: FrozenJSONValue
    initial_runtime_state: FrozenJSONValue
    replay_inputs: FrozenJSONValue
    expected_public_event_digest: str
    evidence_digest: str

    def to_record(self) -> dict[str, Any]:
        return {
            **_private_replay_record_without_digest(self),
            "evidence_digest": self.evidence_digest,
        }


@dataclass(frozen=True)
class CanonicalGameEvidence:
    schema_version: str
    game_id: str
    public_event_stream: PublicEventHistory
    authoritative_pre_prefixes: tuple[AuthoritativePREPrefix, ...]
    belief_observations: tuple[BeliefObservation, ...]
    speech_annotations_v1: tuple[V1SpeechAnnotation, ...]
    call_budget_summary: CallBudgetSummary
    parser_summary: ParserSummary
    submitted_gameplay_actions: tuple[SubmittedGameplayAction, ...]
    backend_call_evidence: tuple[BackendCallEvidence, ...]
    private_replay_evidence: PrivateReplayEvidence


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return value


def _sha256(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _support(values: Sequence[str], observer_id: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("suspicion_support must be a sequence")
    support = tuple(values)
    if any(player not in PLAYER_IDS for player in support):
        raise ValueError("suspicion_support must use exact player IDs")
    if observer_id in support:
        raise ValueError("belief observation cannot contain self-suspicion")
    if len(support) != len(set(support)):
        raise ValueError("suspicion_support cannot contain duplicates")
    if support != tuple(sorted(support, key=_PLAYER_ORDER.__getitem__)):
        raise ValueError("suspicion_support must use canonical seat order")
    return support


def _belief_attempt(value: Mapping[str, Any]) -> BeliefObservationAttempt:
    expected = {
        "attempt_index",
        "call_id",
        "status",
        "response_digest",
        "error_category",
        "error_message",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("belief attempt has an invalid field set")
    attempt_index = _nonnegative_integer(value["attempt_index"], "attempt_index")
    if attempt_index == 0:
        raise ValueError("belief attempt indices must start at one")
    try:
        status = BeliefObservationStatus(value["status"])
    except (TypeError, ValueError) as error:
        raise ValueError("unsupported belief observation attempt status") from error
    response_digest = value["response_digest"]
    error_category = value["error_category"]
    error_message = value["error_message"]
    if status is BeliefObservationStatus.SUCCESS:
        _sha256(response_digest, "belief response_digest")
        if error_category is not None or error_message is not None:
            raise ValueError("successful belief attempt cannot contain an error")
    else:
        if response_digest is not None:
            _sha256(response_digest, "belief response_digest")
        _required_text(error_category, "belief error_category")
        _required_text(error_message, "belief error_message")
    return BeliefObservationAttempt(
        attempt_index=attempt_index,
        call_id=_required_text(value["call_id"], "belief call_id"),
        status=status,
        response_digest=response_digest,
        error_category=error_category,
        error_message=error_message,
    )


def _belief_record_without_digest(
    observation: BeliefObservation,
) -> dict[str, Any]:
    return {
        "schema_version": observation.schema_version,
        "game_id": observation.game_id,
        "attempt_id": observation.attempt_id,
        "boundary_id": observation.boundary_id,
        "prefix_digest": observation.prefix_digest,
        "observation_id": observation.observation_id,
        "observer_id": observation.observer_id,
        "observer_alive": observation.observer_alive,
        "day": observation.day,
        "phase": observation.phase,
        "status": observation.status.value,
        "label_observed": observation.label_observed,
        "suspicion_support": list(observation.suspicion_support),
        "attempts": [attempt.to_record() for attempt in observation.attempts],
    }


def _backend_call_record_without_digest(
    evidence: BackendCallEvidence,
) -> dict[str, Any]:
    return {
        "schema_version": evidence.schema_version,
        "call_id": evidence.call_id,
        "operation_id": evidence.operation_id,
        "purpose": evidence.purpose.value,
        "status": evidence.status.value,
        "boundary_id": evidence.boundary_id,
        "observer_id": evidence.observer_id,
        "attempt_index": evidence.attempt_index,
        "backend_identity": evidence.backend_identity,
        "model_identity": evidence.model_identity,
        "parser_identity": evidence.parser_identity,
        "prompt_identity": evidence.prompt_identity,
        "retry_policy_identity": evidence.retry_policy_identity,
        "call_budget_identity": evidence.call_budget_identity,
        "private_payload": evidence.private_payload.to_value(),
    }


def construct_backend_call_evidence(
    *,
    call_id: str,
    operation_id: str,
    purpose: BackendCallPurpose,
    status: BackendCallStatus,
    boundary_id: str | None,
    observer_id: str | None,
    attempt_index: int,
    backend_identity: str,
    model_identity: str,
    parser_identity: str,
    prompt_identity: str,
    retry_policy_identity: str,
    call_budget_identity: str,
    private_payload: Mapping[str, Any] | FrozenJSONValue,
) -> BackendCallEvidence:
    if not isinstance(purpose, BackendCallPurpose):
        raise TypeError("purpose must be BackendCallPurpose")
    if not isinstance(status, BackendCallStatus):
        raise TypeError("status must be BackendCallStatus")
    if boundary_id is not None:
        boundary_id = _required_text(boundary_id, "backend call boundary_id")
    if observer_id is not None and observer_id not in PLAYER_IDS:
        raise ValueError("backend call observer_id must be an exact player ID")
    if purpose in {
        BackendCallPurpose.BELIEF_OBSERVATION,
        BackendCallPurpose.SPEECH_PERCEPTION,
        BackendCallPurpose.SPEECH_GENERATION,
    } and boundary_id is None:
        raise ValueError("cognition backend call requires a PRE Boundary")
    if purpose in {
        BackendCallPurpose.BELIEF_OBSERVATION,
        BackendCallPurpose.SPEECH_GENERATION,
    } and observer_id is None:
        raise ValueError("observer cognition backend call requires an observer")
    attempt_index = _nonnegative_integer(attempt_index, "attempt_index")
    if attempt_index == 0:
        raise ValueError("backend call attempt indices must start at one")
    provisional = BackendCallEvidence(
        schema_version=BACKEND_CALL_EVIDENCE_SCHEMA_VERSION,
        call_id=_required_text(call_id, "backend call_id"),
        operation_id=_required_text(operation_id, "backend operation_id"),
        purpose=purpose,
        status=status,
        boundary_id=boundary_id,
        observer_id=observer_id,
        attempt_index=attempt_index,
        backend_identity=_required_text(backend_identity, "backend_identity"),
        model_identity=_required_text(model_identity, "model_identity"),
        parser_identity=_required_text(parser_identity, "parser_identity"),
        prompt_identity=_required_text(prompt_identity, "prompt_identity"),
        retry_policy_identity=_required_text(
            retry_policy_identity,
            "retry_policy_identity",
        ),
        call_budget_identity=_required_text(
            call_budget_identity,
            "call_budget_identity",
        ),
        private_payload=freeze_json_object(
            private_payload,
            "backend private_payload",
        ),
        call_digest="",
    )
    return replace(
        provisional,
        call_digest=sha256_bytes(
            canonical_json_bytes(_backend_call_record_without_digest(provisional))
        ),
    )


def construct_belief_observation(
    *,
    game_id: str,
    attempt_id: str,
    boundary_id: str,
    prefix_digest: str,
    observation_id: str,
    observer_id: str,
    observer_alive: bool,
    day: int,
    phase: str,
    status: BeliefObservationStatus,
    suspicion_support: Sequence[str],
    attempts: Sequence[Mapping[str, Any]],
) -> BeliefObservation:
    if observer_id not in PLAYER_IDS:
        raise ValueError("observer_id must be an exact player ID")
    if not isinstance(observer_alive, bool):
        raise TypeError("observer_alive must be boolean")
    if phase not in PUBLIC_PHASES:
        raise ValueError("belief observation has an unknown Public Phase")
    if not isinstance(status, BeliefObservationStatus):
        raise TypeError("status must be BeliefObservationStatus")
    frozen_attempts = tuple(_belief_attempt(value) for value in attempts)
    if not frozen_attempts:
        raise ValueError("belief observation requires bounded-attempt evidence")
    if tuple(item.attempt_index for item in frozen_attempts) != tuple(
        range(1, len(frozen_attempts) + 1)
    ):
        raise ValueError("belief attempt indices must be contiguous from one")
    if frozen_attempts[-1].status is not status:
        raise ValueError("belief final status must match the final attempt")
    if any(
        item.status is not BeliefObservationStatus.ERROR
        for item in frozen_attempts[:-1]
    ):
        raise ValueError("belief collection must stop after its first success")
    support = _support(suspicion_support, observer_id)
    if status is BeliefObservationStatus.ERROR and support:
        raise ValueError("failed belief observation cannot contain support")
    provisional = BeliefObservation(
        schema_version=BELIEF_OBSERVATION_SCHEMA_VERSION,
        game_id=_required_text(game_id, "game_id"),
        attempt_id=_required_text(attempt_id, "attempt_id"),
        boundary_id=_required_text(boundary_id, "boundary_id"),
        prefix_digest=_sha256(prefix_digest, "prefix_digest"),
        observation_id=_required_text(observation_id, "observation_id"),
        observer_id=observer_id,
        observer_alive=observer_alive,
        day=_nonnegative_integer(day, "day"),
        phase=phase,
        status=status,
        label_observed=status is BeliefObservationStatus.SUCCESS,
        suspicion_support=support,
        attempts=frozen_attempts,
        observation_digest="",
    )
    return replace(
        provisional,
        observation_digest=sha256_bytes(
            canonical_json_bytes(_belief_record_without_digest(provisional))
        ),
    )


def _call_budget_record_without_digest(summary: CallBudgetSummary) -> dict[str, Any]:
    return {
        "schema_version": summary.schema_version,
        "configured_call_limit": summary.configured_call_limit,
        "used_calls": summary.used_calls,
        "retry_calls": summary.retry_calls,
        "fallback_action_count": summary.fallback_action_count,
        "second_speaker_belief_count": summary.second_speaker_belief_count,
        "opaque_call_digests": list(summary.opaque_call_digests),
    }


def construct_call_budget_summary(
    *,
    configured_call_limit: int,
    used_calls: int,
    retry_calls: int,
    fallback_action_count: int,
    second_speaker_belief_count: int,
    opaque_call_digests: Sequence[str],
) -> CallBudgetSummary:
    values = {
        name: _nonnegative_integer(value, name)
        for name, value in {
            "configured_call_limit": configured_call_limit,
            "used_calls": used_calls,
            "retry_calls": retry_calls,
            "fallback_action_count": fallback_action_count,
            "second_speaker_belief_count": second_speaker_belief_count,
        }.items()
    }
    if values["used_calls"] > values["configured_call_limit"]:
        raise ValueError("used_calls cannot exceed configured_call_limit")
    call_digests = tuple(
        _sha256(value, "opaque call digest") for value in opaque_call_digests
    )
    if len(call_digests) != values["used_calls"]:
        raise ValueError("opaque call digests must cover every used call")
    provisional = CallBudgetSummary(
        schema_version=CALL_BUDGET_SUMMARY_SCHEMA_VERSION,
        **values,
        opaque_call_digests=call_digests,
        summary_digest="",
    )
    return replace(
        provisional,
        summary_digest=sha256_bytes(
            canonical_json_bytes(_call_budget_record_without_digest(provisional))
        ),
    )


def _parser_summary_record_without_digest(summary: ParserSummary) -> dict[str, Any]:
    return {
        "schema_version": summary.schema_version,
        "public_speech_count": summary.public_speech_count,
        "ok_count": summary.ok_count,
        "no_action_count": summary.no_action_count,
        "error_count": summary.error_count,
        "parser_versions": list(summary.parser_versions),
        "prompt_versions": list(summary.prompt_versions),
    }


def construct_parser_summary(
    annotations: Sequence[V1SpeechAnnotation],
) -> ParserSummary:
    counts = Counter(annotation.status for annotation in annotations)
    parser_versions = tuple(
        sorted(
            {
                attempt.parser_version
                for item in annotations
                for attempt in item.attempts
            }
        )
    )
    prompt_versions = tuple(
        sorted(
            {
                attempt.prompt_version
                for item in annotations
                for attempt in item.attempts
            }
        )
    )
    provisional = ParserSummary(
        schema_version=PARSER_SUMMARY_SCHEMA_VERSION,
        public_speech_count=len(annotations),
        ok_count=counts[V1AnnotationStatus.OK],
        no_action_count=counts[V1AnnotationStatus.NO_ACTION],
        error_count=counts[V1AnnotationStatus.ERROR],
        parser_versions=parser_versions,
        prompt_versions=prompt_versions,
        summary_digest="",
    )
    return replace(
        provisional,
        summary_digest=sha256_bytes(
            canonical_json_bytes(_parser_summary_record_without_digest(provisional))
        ),
    )


def _submitted_action_record_without_digest(
    action: SubmittedGameplayAction,
) -> dict[str, Any]:
    return {
        "schema_version": action.schema_version,
        "action_id": action.action_id,
        "step_index": action.step_index,
        "actor_id": action.actor_id,
        "action_type": action.action_type,
        "action_payload": action.action_payload.to_value(),
        "resulting_public_event_ids": list(action.resulting_public_event_ids),
        "boundary_id": action.boundary_id,
        "speaker_pre_belief_handoff": (
            None
            if action.speaker_pre_belief_handoff is None
            else action.speaker_pre_belief_handoff.to_record()
        ),
        "fallback_used": action.fallback_used,
    }


def construct_submitted_gameplay_action(
    *,
    action_id: str,
    step_index: int,
    actor_id: str,
    action_type: str,
    action_payload: Any,
    resulting_public_event_ids: Sequence[str],
    boundary_id: str | None,
    speaker_pre_belief_handoff: SpeakerPREBeliefHandoff | None,
    fallback_used: bool,
) -> SubmittedGameplayAction:
    if actor_id not in PLAYER_IDS:
        raise ValueError("actor_id must be an exact player ID")
    if not isinstance(fallback_used, bool):
        raise TypeError("fallback_used must be boolean")
    frozen_payload = freeze_json_value(action_payload)
    event_ids = tuple(
        _required_text(event_id, "resulting public event ID")
        for event_id in resulting_public_event_ids
    )
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("resulting public event IDs must be unique")
    action_type = _required_text(action_type, "action_type")
    if action_type == "public_speech":
        if boundary_id is None or speaker_pre_belief_handoff is None:
            raise ValueError("public speech action requires one Speaker PRE handoff")
    elif speaker_pre_belief_handoff is not None:
        raise ValueError("only public speech actions may carry a Speaker PRE handoff")
    provisional = SubmittedGameplayAction(
        schema_version=SUBMITTED_GAMEPLAY_ACTION_SCHEMA_VERSION,
        action_id=_required_text(action_id, "action_id"),
        step_index=_nonnegative_integer(step_index, "step_index"),
        actor_id=actor_id,
        action_type=action_type,
        action_payload=frozen_payload,
        resulting_public_event_ids=event_ids,
        boundary_id=(
            None if boundary_id is None else _required_text(boundary_id, "boundary_id")
        ),
        speaker_pre_belief_handoff=speaker_pre_belief_handoff,
        fallback_used=fallback_used,
        action_digest="",
    )
    return replace(
        provisional,
        action_digest=sha256_bytes(
            canonical_json_bytes(_submitted_action_record_without_digest(provisional))
        ),
    )


def _private_replay_record_without_digest(
    evidence: PrivateReplayEvidence,
) -> dict[str, Any]:
    return {
        "schema_version": evidence.schema_version,
        "game_id": evidence.game_id,
        "seed": evidence.seed,
        "role_assignment": dict(evidence.role_assignment),
        "runtime_configuration": evidence.runtime_configuration.to_value(),
        "initial_runtime_state": evidence.initial_runtime_state.to_value(),
        "replay_inputs": evidence.replay_inputs.to_value(),
        "expected_public_event_digest": evidence.expected_public_event_digest,
    }


def construct_private_replay_evidence(
    *,
    game_id: str,
    seed: int,
    role_assignment: Mapping[str, str],
    runtime_configuration: Mapping[str, Any] | FrozenJSONValue,
    initial_runtime_state: Mapping[str, Any] | FrozenJSONValue,
    replay_inputs: Mapping[str, Any] | FrozenJSONValue,
    expected_public_event_digest: str,
) -> PrivateReplayEvidence:
    if not isinstance(role_assignment, Mapping) or set(role_assignment) != set(
        PLAYER_IDS
    ):
        raise ValueError("Private Replay Evidence requires all seven role assignments")
    roles = tuple((player, role_assignment[player]) for player in PLAYER_IDS)
    if Counter(role for _, role in roles) != Counter(_CLASSIC7_ROLE_COUNTS):
        raise ValueError("role assignment does not satisfy Classic7 multiplicities")
    provisional = PrivateReplayEvidence(
        schema_version=PRIVATE_REPLAY_EVIDENCE_SCHEMA_VERSION,
        game_id=_required_text(game_id, "game_id"),
        seed=_nonnegative_integer(seed, "seed"),
        role_assignment=roles,
        runtime_configuration=freeze_json_object(
            runtime_configuration,
            "runtime_configuration",
        ),
        initial_runtime_state=freeze_json_object(
            initial_runtime_state,
            "initial_runtime_state",
        ),
        replay_inputs=freeze_json_object(replay_inputs, "replay_inputs"),
        expected_public_event_digest=_sha256(
            expected_public_event_digest,
            "expected_public_event_digest",
        ),
        evidence_digest="",
    )
    return replace(
        provisional,
        evidence_digest=sha256_bytes(
            canonical_json_bytes(_private_replay_record_without_digest(provisional))
        ),
    )


def _validate_handoffs(
    prefixes: tuple[AuthoritativePREPrefix, ...],
    observations: tuple[BeliefObservation, ...],
    actions: tuple[SubmittedGameplayAction, ...],
) -> None:
    observation_by_id = {item.observation_id: item for item in observations}
    speech_actions = [item for item in actions if item.action_type == "public_speech"]
    for prefix in prefixes:
        matches = [
            item
            for item in speech_actions
            if item.boundary_id == prefix.boundary_id
        ]
        if len(matches) != 1:
            raise ValueError("each PRE Boundary requires exactly one speech handoff")
        action = matches[0]
        handoff = action.speaker_pre_belief_handoff
        assert handoff is not None
        source = observation_by_id.get(prefix.speaker_pre_belief_observation_id)
        if source is None or source.status is not BeliefObservationStatus.SUCCESS:
            raise ValueError("Speaker PRE handoff requires its successful observation")
        if (
            handoff.boundary_id != prefix.boundary_id
            or handoff.prefix_digest != prefix.prefix_digest
            or handoff.observation_id != source.observation_id
            or handoff.observation_digest != source.observation_digest
            or handoff.observer_id != prefix.current_speaker
            or handoff.suspicion_support != source.suspicion_support
        ):
            raise ValueError("Speaker PRE Belief Handoff binding mismatch")
        expected_handoff = construct_speaker_pre_belief_handoff(
            prefix,
            observation_id=source.observation_id,
            observation_digest=source.observation_digest,
            observer_id=source.observer_id,
            observation_status=source.status.value,
            suspicion_support=source.suspicion_support,
        )
        if handoff != expected_handoff:
            raise ValueError("Speaker PRE Belief Handoff digest mismatch")


def construct_canonical_game_evidence(
    *,
    game_id: str,
    public_event_stream: PublicEventHistory,
    authoritative_pre_prefixes: Sequence[AuthoritativePREPrefix],
    belief_observations: Sequence[BeliefObservation],
    speech_annotations_v1: Sequence[V1SpeechAnnotation],
    call_budget_summary: CallBudgetSummary,
    submitted_gameplay_actions: Sequence[SubmittedGameplayAction],
    backend_call_evidence: Sequence[BackendCallEvidence],
    private_replay_evidence: PrivateReplayEvidence,
) -> CanonicalGameEvidence:
    game_id = _required_text(game_id, "game_id")
    validate_public_event_history(public_event_stream)
    prefixes = tuple(authoritative_pre_prefixes)
    if not prefixes:
        raise ValueError("Canonical Game Evidence requires at least one PRE Prefix")
    if any(
        validate_authoritative_pre_prefix(prefix).game_id != game_id
        for prefix in prefixes
    ):
        raise ValueError("PRE Prefix game identity mismatch")
    if len({prefix.boundary_id for prefix in prefixes}) != len(prefixes):
        raise ValueError("PRE Boundary identities must be unique")
    if tuple(prefix.step_index for prefix in prefixes) != tuple(
        sorted(prefix.step_index for prefix in prefixes)
    ):
        raise ValueError("PRE Prefixes must use chronological step order")
    full_records = public_event_stream.to_records()
    for prefix in prefixes:
        records = prefix.public_event_history.to_records()
        if records != full_records[: len(records)]:
            raise ValueError("Authoritative PRE Prefix is not an exact public prefix")

    annotations = tuple(speech_annotations_v1)
    public_speeches = [
        event
        for event in public_event_stream.events
        if event.event_type == "public_speech"
    ]
    if tuple(item.event_id for item in annotations) != tuple(
        event.event_id for event in public_speeches
    ):
        raise ValueError("V1 annotations must exactly cover public speeches")
    expected_turn_event_ids = tuple(
        public_event_stream.events[event.event_index - 1].event_id
        for event in public_speeches
    )
    actual_turn_event_ids = tuple(
        prefix.public_event_history.events[-1].event_id for prefix in prefixes
    )
    if actual_turn_event_ids != expected_turn_event_ids:
        raise ValueError(
            "PRE Prefix coverage must exactly match every public speech turn"
        )
    for annotation in annotations:
        validate_v1_annotation_binding(annotation, public_event_stream)
        if not annotation.is_successful:
            raise ValueError("V1 perception failure makes a game ineligible")

    observations = tuple(belief_observations)
    if len({item.observation_id for item in observations}) != len(observations):
        raise ValueError("Belief Observation identities must be unique")
    observation_by_id = {item.observation_id: item for item in observations}
    linked_ids: list[str] = []
    for prefix in prefixes:
        for link in prefix.belief_observation_links:
            linked_ids.append(link.observation_id)
            observation = observation_by_id.get(link.observation_id)
            if observation is None:
                raise ValueError("missing alive-observer Belief Observation")
            if (
                observation.game_id != game_id
                or observation.boundary_id != prefix.boundary_id
                or observation.prefix_digest != prefix.prefix_digest
                or observation.observer_id != link.observer_id
                or observation.day != prefix.public_temporal_state.day
                or observation.phase != prefix.public_temporal_state.phase.value
            ):
                raise ValueError("Belief Observation PRE binding mismatch")
            if not observation.observer_alive:
                raise ValueError("linked PRE observer must be alive")
            if observation.status is not BeliefObservationStatus.SUCCESS:
                raise ValueError("failed Belief Observation makes a game ineligible")
    if len(linked_ids) != len(set(linked_ids)) or set(linked_ids) != set(
        observation_by_id
    ):
        raise ValueError("Belief Observations must exactly equal PRE links")

    if not isinstance(call_budget_summary, CallBudgetSummary):
        raise TypeError("call_budget_summary must be CallBudgetSummary")
    if call_budget_summary.fallback_action_count:
        raise ValueError("fallback action makes a game ineligible")
    if call_budget_summary.second_speaker_belief_count:
        raise ValueError("second speaker belief makes a game ineligible")
    actions = tuple(submitted_gameplay_actions)
    if any(not isinstance(item, SubmittedGameplayAction) for item in actions):
        raise TypeError("submitted_gameplay_actions contains an invalid value")
    if any(item.fallback_used for item in actions):
        raise ValueError("fallback gameplay action makes a game ineligible")
    if len({item.action_id for item in actions}) != len(actions):
        raise ValueError("submitted gameplay action identities must be unique")
    event_by_id = {
        item.event_id: item for item in public_event_stream.events
    }
    if any(
        event_id not in event_by_id
        for action in actions
        for event_id in action.resulting_public_event_ids
    ):
        raise ValueError("submitted action references an unknown public event")
    prefixes_by_boundary = {
        prefix.boundary_id: prefix for prefix in prefixes
    }
    for action in actions:
        if action.action_type != "public_speech":
            continue
        assert action.boundary_id is not None
        prefix = prefixes_by_boundary.get(action.boundary_id)
        if prefix is None:
            raise ValueError("public speech action references an unknown PRE Boundary")
        if (
            action.actor_id != prefix.current_speaker
            or action.step_index != prefix.step_index
        ):
            raise ValueError("public speech action does not match its PRE speaker")
        if len(action.resulting_public_event_ids) != 1:
            raise ValueError("public speech action requires one resulting public event")
        public_speech = event_by_id[action.resulting_public_event_ids[0]]
        if (
            public_speech.event_type != "public_speech"
            or public_speech.event_index != prefix.step_index + 1
            or public_speech.speaker != action.actor_id
            or public_speech.raw_text != action.action_payload.to_value()
        ):
            raise ValueError("submitted speech diverges from its public event")
    _validate_handoffs(prefixes, observations, actions)

    private_records = tuple(backend_call_evidence)
    if any(not isinstance(item, BackendCallEvidence) for item in private_records):
        raise TypeError("backend_call_evidence requires typed call records")
    for item in private_records:
        expected = construct_backend_call_evidence(
            call_id=item.call_id,
            operation_id=item.operation_id,
            purpose=item.purpose,
            status=item.status,
            boundary_id=item.boundary_id,
            observer_id=item.observer_id,
            attempt_index=item.attempt_index,
            backend_identity=item.backend_identity,
            model_identity=item.model_identity,
            parser_identity=item.parser_identity,
            prompt_identity=item.prompt_identity,
            retry_policy_identity=item.retry_policy_identity,
            call_budget_identity=item.call_budget_identity,
            private_payload=item.private_payload,
        )
        if item != expected:
            raise ValueError("backend call evidence digest mismatch")
    call_ids = [item.call_id for item in private_records]
    if len(call_ids) != len(set(call_ids)):
        raise ValueError("backend call evidence call IDs must be unique")
    if len(private_records) != call_budget_summary.used_calls:
        raise ValueError("call-budget usage disagrees with backend call evidence")
    if call_budget_summary.opaque_call_digests != tuple(
        record.call_digest for record in private_records
    ):
        raise ValueError("opaque call digests disagree with private call evidence")

    call_groups: dict[
        tuple[BackendCallPurpose, str],
        list[BackendCallEvidence],
    ] = defaultdict(list)
    for item in private_records:
        call_groups[(item.purpose, item.operation_id)].append(item)
    for group in call_groups.values():
        if tuple(item.attempt_index for item in group) != tuple(
            range(1, len(group) + 1)
        ):
            raise ValueError(
                "backend operation attempt indices must be contiguous from one"
            )
        if any(item.status is BackendCallStatus.SUCCESS for item in group[:-1]):
            raise ValueError("backend operation must stop after its first success")
        if group[-1].status is not BackendCallStatus.SUCCESS:
            raise ValueError("failed backend operation makes a game ineligible")

    expected_belief_calls = {
        (
            attempt.call_id,
            observation.observation_id,
            observation.boundary_id,
            observation.observer_id,
            attempt.attempt_index,
            attempt.status.value,
        )
        for observation in observations
        for attempt in observation.attempts
    }
    actual_belief_calls = {
        (
            item.call_id,
            item.operation_id,
            item.boundary_id,
            item.observer_id,
            item.attempt_index,
            item.status.value,
        )
        for item in private_records
        if item.purpose is BackendCallPurpose.BELIEF_OBSERVATION
    }
    if actual_belief_calls != expected_belief_calls:
        raise ValueError(
            "belief backend calls must exactly equal collected observations"
        )

    prefix_by_turn_index = {item.step_index: item for item in prefixes}
    expected_perception_calls = {
        (
            attempt.call_id,
            annotation.event_id,
            prefix_by_turn_index[annotation.event_index - 1].boundary_id,
            annotation.speaker,
            attempt.attempt_index,
            (
                BackendCallStatus.ERROR.value
                if attempt.status is V1AnnotationStatus.ERROR
                else BackendCallStatus.SUCCESS.value
            ),
        )
        for annotation in annotations
        for attempt in annotation.attempts
    }
    actual_perception_calls = {
        (
            item.call_id,
            item.operation_id,
            item.boundary_id,
            item.observer_id,
            item.attempt_index,
            item.status.value,
        )
        for item in private_records
        if item.purpose is BackendCallPurpose.SPEECH_PERCEPTION
    }
    if actual_perception_calls != expected_perception_calls:
        raise ValueError(
            "perception backend calls must exactly equal V1 attempts"
        )
    retry_count = sum(len(group) - 1 for group in call_groups.values())
    if call_budget_summary.retry_calls != retry_count:
        raise ValueError("call-budget retry count disagrees with attempt evidence")
    if call_budget_summary.second_speaker_belief_count != 0:
        raise ValueError("second speaker belief makes a game ineligible")
    if not isinstance(private_replay_evidence, PrivateReplayEvidence):
        raise TypeError("private_replay_evidence must be PrivateReplayEvidence")
    if private_replay_evidence.game_id != game_id:
        raise ValueError("Private Replay Evidence game identity mismatch")
    if (
        private_replay_evidence.expected_public_event_digest
        != public_event_stream.digest
    ):
        raise ValueError("Private Replay Evidence public digest mismatch")

    return CanonicalGameEvidence(
        schema_version=CANONICAL_GAME_EVIDENCE_SCHEMA_VERSION,
        game_id=game_id,
        public_event_stream=public_event_stream,
        authoritative_pre_prefixes=prefixes,
        belief_observations=observations,
        speech_annotations_v1=annotations,
        call_budget_summary=call_budget_summary,
        parser_summary=construct_parser_summary(annotations),
        submitted_gameplay_actions=actions,
        backend_call_evidence=private_records,
        private_replay_evidence=private_replay_evidence,
    )


def validate_canonical_game_evidence(
    evidence: CanonicalGameEvidence,
) -> CanonicalGameEvidence:
    """Recompute every nested digest before an immutable Bundle is published."""

    if not isinstance(evidence, CanonicalGameEvidence):
        raise TypeError("evidence must be CanonicalGameEvidence")
    observations = tuple(
        construct_belief_observation(
            game_id=item.game_id,
            attempt_id=item.attempt_id,
            boundary_id=item.boundary_id,
            prefix_digest=item.prefix_digest,
            observation_id=item.observation_id,
            observer_id=item.observer_id,
            observer_alive=item.observer_alive,
            day=item.day,
            phase=item.phase,
            status=item.status,
            suspicion_support=item.suspicion_support,
            attempts=tuple(attempt.to_record() for attempt in item.attempts),
        )
        for item in evidence.belief_observations
    )
    call_budget = construct_call_budget_summary(
        configured_call_limit=evidence.call_budget_summary.configured_call_limit,
        used_calls=evidence.call_budget_summary.used_calls,
        retry_calls=evidence.call_budget_summary.retry_calls,
        fallback_action_count=evidence.call_budget_summary.fallback_action_count,
        second_speaker_belief_count=(
            evidence.call_budget_summary.second_speaker_belief_count
        ),
        opaque_call_digests=evidence.call_budget_summary.opaque_call_digests,
    )
    private = construct_private_replay_evidence(
        game_id=evidence.private_replay_evidence.game_id,
        seed=evidence.private_replay_evidence.seed,
        role_assignment=dict(evidence.private_replay_evidence.role_assignment),
        runtime_configuration=(
            evidence.private_replay_evidence.runtime_configuration
        ),
        initial_runtime_state=evidence.private_replay_evidence.initial_runtime_state,
        replay_inputs=evidence.private_replay_evidence.replay_inputs,
        expected_public_event_digest=(
            evidence.private_replay_evidence.expected_public_event_digest
        ),
    )
    actions = tuple(
        construct_submitted_gameplay_action(
            action_id=item.action_id,
            step_index=item.step_index,
            actor_id=item.actor_id,
            action_type=item.action_type,
            action_payload=item.action_payload,
            resulting_public_event_ids=item.resulting_public_event_ids,
            boundary_id=item.boundary_id,
            speaker_pre_belief_handoff=item.speaker_pre_belief_handoff,
            fallback_used=item.fallback_used,
        )
        for item in evidence.submitted_gameplay_actions
    )
    expected = construct_canonical_game_evidence(
        game_id=evidence.game_id,
        public_event_stream=evidence.public_event_stream,
        authoritative_pre_prefixes=evidence.authoritative_pre_prefixes,
        belief_observations=observations,
        speech_annotations_v1=evidence.speech_annotations_v1,
        call_budget_summary=call_budget,
        submitted_gameplay_actions=actions,
        backend_call_evidence=tuple(
            construct_backend_call_evidence(
                call_id=item.call_id,
                operation_id=item.operation_id,
                purpose=item.purpose,
                status=item.status,
                boundary_id=item.boundary_id,
                observer_id=item.observer_id,
                attempt_index=item.attempt_index,
                backend_identity=item.backend_identity,
                model_identity=item.model_identity,
                parser_identity=item.parser_identity,
                prompt_identity=item.prompt_identity,
                retry_policy_identity=item.retry_policy_identity,
                call_budget_identity=item.call_budget_identity,
                private_payload=item.private_payload,
            )
            for item in evidence.backend_call_evidence
        ),
        private_replay_evidence=private,
    )
    if evidence != expected:
        raise ValueError("Canonical Game Evidence contains a stale nested digest")
    return evidence
