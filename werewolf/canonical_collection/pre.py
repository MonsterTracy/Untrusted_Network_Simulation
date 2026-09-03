"""Authoritative PRE Prefix and Speaker PRE Belief Handoff contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection.public_history import (
    PLAYER_IDS,
    PUBLIC_EVENT_SCHEMA_VERSION,
    PublicEventHistory,
    PublicTemporalState,
    validate_public_event_history,
)
from werewolf.canonical_collection.speech import (
    V1_ANNOTATION_SCHEMA_VERSION,
    V1AnnotationStatus,
    V1SpeechAnnotation,
    validate_v1_annotation_binding,
)


AUTHORITATIVE_PRE_PREFIX_SCHEMA_VERSION = (
    "classic7_authoritative_pre_prefix_v1"
)
SPEAKER_PRE_BELIEF_HANDOFF_SCHEMA_VERSION = (
    "classic7_speaker_pre_belief_handoff_v1"
)
_HANDOFF_SOURCE = "realized_pre_belief_observation"
_PLAYER_ORDER = {player_id: seat for seat, player_id in enumerate(PLAYER_IDS)}


@dataclass(frozen=True)
class BeliefObservationLink:
    observer_id: str
    observation_id: str

    def to_record(self) -> dict[str, str]:
        return {
            "observer_id": self.observer_id,
            "observation_id": self.observation_id,
        }


@dataclass(frozen=True)
class AuthoritativePREPrefix:
    schema_version: str
    game_id: str
    boundary_id: str
    step_index: int
    report_trigger_id: str
    current_speaker: str
    alive_observer_ids: tuple[str, ...]
    public_event_history: PublicEventHistory
    public_event_schema_version: str
    public_event_digest: str
    v1_annotations: tuple[V1SpeechAnnotation, ...]
    v1_annotation_schema_version: str
    v1_annotation_digest: str
    maximum_public_day: int
    belief_observation_links: tuple[BeliefObservationLink, ...]
    speaker_pre_belief_observation_id: str
    prefix_digest: str

    @property
    def public_temporal_state(self) -> PublicTemporalState:
        return self.public_event_history.current_state

    def to_record(self) -> dict[str, Any]:
        return {
            **_prefix_record_without_digest(self),
            "prefix_digest": self.prefix_digest,
        }


@dataclass(frozen=True)
class SpeakerPREBeliefHandoff:
    schema_version: str
    boundary_id: str
    prefix_digest: str
    observation_id: str
    observation_digest: str
    observer_id: str
    suspicion_support: tuple[str, ...]
    source: str
    handoff_digest: str

    def to_record(self) -> dict[str, Any]:
        return {
            **_handoff_record_without_digest(self),
            "handoff_digest": self.handoff_digest,
        }


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


def _alive_observers(values: Sequence[str], current_speaker: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("alive_observer_ids must be a sequence")
    observers = tuple(values)
    if any(observer not in PLAYER_IDS for observer in observers):
        raise ValueError("alive observers must use exact player1...player7 IDs")
    if len(observers) != len(set(observers)):
        raise ValueError("alive_observer_ids cannot contain duplicates")
    if observers != tuple(sorted(observers, key=_PLAYER_ORDER.__getitem__)):
        raise ValueError("alive_observer_ids must use canonical seat order")
    if current_speaker not in observers:
        raise ValueError("current speaker must be an alive observer")
    return observers


def _observation_links(
    value: Mapping[str, str],
    alive_observers: tuple[str, ...],
) -> tuple[BeliefObservationLink, ...]:
    if not isinstance(value, Mapping):
        raise TypeError("belief observation links must be a mapping")
    if set(value) != set(alive_observers):
        raise ValueError(
            "belief observation links must cover every alive observer exactly once"
        )
    links = tuple(
        BeliefObservationLink(
            observer_id=observer,
            observation_id=_required_text(
                value[observer],
                f"belief observation ID for {observer}",
            ),
        )
        for observer in alive_observers
    )
    observation_ids = [link.observation_id for link in links]
    if len(observation_ids) != len(set(observation_ids)):
        raise ValueError("belief observation IDs must be unique")
    return links


def _annotation_digest(annotations: Sequence[V1SpeechAnnotation]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            [annotation.to_record() for annotation in annotations]
        )
    )


def _prefix_record_without_digest(prefix: AuthoritativePREPrefix) -> dict[str, Any]:
    return {
        "schema_version": prefix.schema_version,
        "game_id": prefix.game_id,
        "boundary_id": prefix.boundary_id,
        "step_index": prefix.step_index,
        "report_trigger_id": prefix.report_trigger_id,
        "current_speaker": prefix.current_speaker,
        "day": prefix.public_temporal_state.day,
        "phase": prefix.public_temporal_state.phase.value,
        "alive_observer_ids": list(prefix.alive_observer_ids),
        "public_event_schema_version": prefix.public_event_schema_version,
        "public_events": prefix.public_event_history.to_records(),
        "public_event_digest": prefix.public_event_digest,
        "v1_annotation_schema_version": prefix.v1_annotation_schema_version,
        "v1_annotations": [
            annotation.to_record() for annotation in prefix.v1_annotations
        ],
        "v1_annotation_digest": prefix.v1_annotation_digest,
        "maximum_public_day": prefix.maximum_public_day,
        "belief_observation_links": [
            link.to_record() for link in prefix.belief_observation_links
        ],
        "speaker_pre_belief_observation_id": (
            prefix.speaker_pre_belief_observation_id
        ),
    }


def _validate_annotations(
    history: PublicEventHistory,
    annotations: Sequence[V1SpeechAnnotation],
) -> tuple[V1SpeechAnnotation, ...]:
    frozen = tuple(annotations)
    if any(not isinstance(item, V1SpeechAnnotation) for item in frozen):
        raise TypeError("v1_annotations must contain V1SpeechAnnotation values")
    if any(item.event_index >= len(history.events) for item in frozen):
        raise ValueError("upcoming speech annotation is forbidden at PRE")
    public_speeches = tuple(
        event for event in history.events if event.event_type == "public_speech"
    )
    if tuple(item.event_id for item in frozen) != tuple(
        event.event_id for event in public_speeches
    ):
        raise ValueError(
            "V1 annotation coverage must exactly match prior public speeches"
        )
    for annotation in frozen:
        validate_v1_annotation_binding(annotation, history)
        if annotation.status is V1AnnotationStatus.ERROR:
            raise ValueError("an error V1 annotation cannot enter a PRE Prefix")
    return frozen


def construct_authoritative_pre_prefix(
    *,
    game_id: str,
    boundary_id: str,
    step_index: int,
    report_trigger_id: str,
    current_speaker: str,
    alive_observer_ids: Sequence[str],
    public_event_history: PublicEventHistory,
    v1_annotations: Sequence[V1SpeechAnnotation],
    belief_observation_ids_by_observer: Mapping[str, str],
) -> AuthoritativePREPrefix:
    """Freeze the sole cumulative history value at one strict PRE Boundary."""

    game_id = _required_text(game_id, "game_id")
    boundary_id = _required_text(boundary_id, "boundary_id")
    report_trigger_id = _required_text(report_trigger_id, "report_trigger_id")
    step_index = _nonnegative_integer(step_index, "step_index")
    if current_speaker not in PLAYER_IDS:
        raise ValueError("current_speaker must be an exact player ID")
    if not isinstance(public_event_history, PublicEventHistory):
        raise TypeError("PRE construction requires a frozen PublicEventHistory")
    validate_public_event_history(public_event_history)
    if not public_event_history.events:
        raise ValueError("PRE public history cannot be empty")
    terminal = public_event_history.events[-1]
    if terminal.event_type != "turn_start" or terminal.speaker != current_speaker:
        raise ValueError(
            "Authoritative PRE Prefix must end with matching turn_start"
        )

    observers = _alive_observers(alive_observer_ids, current_speaker)
    annotations = _validate_annotations(public_event_history, v1_annotations)
    links = _observation_links(
        belief_observation_ids_by_observer,
        observers,
    )
    speaker_observation_id = next(
        link.observation_id
        for link in links
        if link.observer_id == current_speaker
    )
    provisional = AuthoritativePREPrefix(
        schema_version=AUTHORITATIVE_PRE_PREFIX_SCHEMA_VERSION,
        game_id=game_id,
        boundary_id=boundary_id,
        step_index=step_index,
        report_trigger_id=report_trigger_id,
        current_speaker=current_speaker,
        alive_observer_ids=observers,
        public_event_history=public_event_history,
        public_event_schema_version=PUBLIC_EVENT_SCHEMA_VERSION,
        public_event_digest=public_event_history.digest,
        v1_annotations=annotations,
        v1_annotation_schema_version=V1_ANNOTATION_SCHEMA_VERSION,
        v1_annotation_digest=_annotation_digest(annotations),
        maximum_public_day=public_event_history.max_public_day,
        belief_observation_links=links,
        speaker_pre_belief_observation_id=speaker_observation_id,
        prefix_digest="",
    )
    prefix_digest = sha256_bytes(
        canonical_json_bytes(_prefix_record_without_digest(provisional))
    )
    return replace(provisional, prefix_digest=prefix_digest)


def validate_authoritative_pre_prefix(
    prefix: AuthoritativePREPrefix,
) -> AuthoritativePREPrefix:
    """Defensively validate a frozen prefix without transforming it."""

    if not isinstance(prefix, AuthoritativePREPrefix):
        raise TypeError("prefix must be an AuthoritativePREPrefix")
    if prefix.schema_version != AUTHORITATIVE_PRE_PREFIX_SCHEMA_VERSION:
        raise ValueError("unsupported Authoritative PRE Prefix schema_version")
    if prefix.public_event_schema_version != PUBLIC_EVENT_SCHEMA_VERSION:
        raise ValueError("unsupported public event schema_version")
    if prefix.v1_annotation_schema_version != V1_ANNOTATION_SCHEMA_VERSION:
        raise ValueError("unsupported V1 annotation schema_version")
    _required_text(prefix.game_id, "game_id")
    _required_text(prefix.boundary_id, "boundary_id")
    _required_text(prefix.report_trigger_id, "report_trigger_id")
    _nonnegative_integer(prefix.step_index, "step_index")
    if prefix.current_speaker not in PLAYER_IDS:
        raise ValueError("current_speaker must be an exact player ID")
    validate_public_event_history(prefix.public_event_history)
    if prefix.public_event_digest != prefix.public_event_history.digest:
        raise ValueError("PRE public event digest mismatch")
    if prefix.maximum_public_day != prefix.public_event_history.max_public_day:
        raise ValueError("PRE maximum_public_day mismatch")
    terminal = prefix.public_event_history.events[-1]
    if (
        terminal.event_type != "turn_start"
        or terminal.speaker != prefix.current_speaker
    ):
        raise ValueError("PRE Prefix does not end with matching turn_start")
    annotations = _validate_annotations(
        prefix.public_event_history,
        prefix.v1_annotations,
    )
    if prefix.v1_annotation_digest != _annotation_digest(annotations):
        raise ValueError("PRE V1 annotation digest mismatch")
    observers = _alive_observers(
        prefix.alive_observer_ids,
        prefix.current_speaker,
    )
    if any(
        not isinstance(link, BeliefObservationLink)
        for link in prefix.belief_observation_links
    ):
        raise TypeError(
            "belief_observation_links must contain BeliefObservationLink values"
        )
    if tuple(
        link.observer_id for link in prefix.belief_observation_links
    ) != observers:
        raise ValueError(
            "belief observation links must follow canonical alive-observer order"
        )
    link_mapping = {
        link.observer_id: link.observation_id
        for link in prefix.belief_observation_links
    }
    links = _observation_links(link_mapping, observers)
    if links != prefix.belief_observation_links:
        raise ValueError("belief observation links are not canonical")
    speaker_links = [
        link
        for link in links
        if link.observer_id == prefix.current_speaker
    ]
    if (
        len(speaker_links) != 1
        or speaker_links[0].observation_id
        != prefix.speaker_pre_belief_observation_id
    ):
        raise ValueError("PRE Prefix must bind exactly one speaker observation")
    expected_digest = sha256_bytes(
        canonical_json_bytes(_prefix_record_without_digest(prefix))
    )
    if prefix.prefix_digest != expected_digest:
        raise ValueError("Authoritative PRE Prefix digest mismatch")
    return prefix


def _suspicion_support(
    values: Sequence[str],
    observer_id: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("suspicion_support must be a sequence")
    support = tuple(values)
    if any(player not in PLAYER_IDS for player in support):
        raise ValueError("suspicion_support must use exact player IDs")
    if len(support) != len(set(support)):
        raise ValueError("suspicion_support cannot contain duplicates")
    if observer_id in support:
        raise ValueError("speaker cannot report self-suspicion")
    if support != tuple(sorted(support, key=_PLAYER_ORDER.__getitem__)):
        raise ValueError("suspicion_support must use canonical seat order")
    return support


def _handoff_record_without_digest(
    handoff: SpeakerPREBeliefHandoff,
) -> dict[str, Any]:
    return {
        "schema_version": handoff.schema_version,
        "boundary_id": handoff.boundary_id,
        "prefix_digest": handoff.prefix_digest,
        "observation_id": handoff.observation_id,
        "observation_digest": handoff.observation_digest,
        "observer_id": handoff.observer_id,
        "suspicion_support": list(handoff.suspicion_support),
        "source": handoff.source,
    }


def construct_speaker_pre_belief_handoff(
    prefix: AuthoritativePREPrefix,
    *,
    observation_id: str,
    observation_digest: str,
    observer_id: str,
    observation_status: str,
    suspicion_support: Sequence[str],
) -> SpeakerPREBeliefHandoff:
    """Freeze the existing speaker observation for the next speech cognition."""

    validate_authoritative_pre_prefix(prefix)
    if observation_id != prefix.speaker_pre_belief_observation_id:
        raise ValueError("handoff must use the one linked speaker observation")
    if observer_id != prefix.current_speaker:
        raise ValueError("handoff observer must be the same current speaker")
    if observation_status != "success":
        raise ValueError("handoff requires a successful observation")
    observation_digest = _sha256(observation_digest, "observation_digest")
    support = _suspicion_support(suspicion_support, observer_id)
    provisional = SpeakerPREBeliefHandoff(
        schema_version=SPEAKER_PRE_BELIEF_HANDOFF_SCHEMA_VERSION,
        boundary_id=prefix.boundary_id,
        prefix_digest=prefix.prefix_digest,
        observation_id=observation_id,
        observation_digest=observation_digest,
        observer_id=observer_id,
        suspicion_support=support,
        source=_HANDOFF_SOURCE,
        handoff_digest="",
    )
    digest = sha256_bytes(
        canonical_json_bytes(_handoff_record_without_digest(provisional))
    )
    return replace(provisional, handoff_digest=digest)
