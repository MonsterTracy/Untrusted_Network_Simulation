"""Pure semantic planning over an Authoritative PRE Prefix."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection.pre import (
    AuthoritativePREPrefix,
    validate_authoritative_pre_prefix,
)
from werewolf.canonical_collection.speech import V1AnnotationStatus


STRUCTURED_TOKEN_PLANNER_VERSION = "classic7_structured_token_planner_v1"
STRUCTURED_TOKEN_TYPES = (
    "phase_change",
    "turn_start",
    "public_speech",
    "speech_action",
    "vote_result",
    "vote",
    "exile_result",
    "exiled_player",
    "death_announcement",
    "dead_player",
)


@dataclass(frozen=True)
class StructuredTokenDescriptor:
    token_index: int
    token_type: str
    source: str | None
    action: str | None
    target: str | None
    day: int
    phase: str
    event_id: str
    event_index: int
    semantic_index: int

    def to_record(self) -> dict[str, Any]:
        return {
            "token_index": self.token_index,
            "token_type": self.token_type,
            "source": self.source,
            "action": self.action,
            "target": self.target,
            "day": self.day,
            "phase": self.phase,
            "event_id": self.event_id,
            "event_index": self.event_index,
            "semantic_index": self.semantic_index,
        }


@dataclass(frozen=True)
class StructuredTokenPlan:
    planner_version: str
    prefix_digest: str
    tokens: tuple[StructuredTokenDescriptor, ...]
    token_count: int
    plan_digest: str

    def to_record(self) -> dict[str, Any]:
        return {
            **_plan_record_without_digest(self),
            "plan_digest": self.plan_digest,
        }


def _plan_record_without_digest(plan: StructuredTokenPlan) -> dict[str, Any]:
    return {
        "planner_version": plan.planner_version,
        "prefix_digest": plan.prefix_digest,
        "tokens": [token.to_record() for token in plan.tokens],
        "token_count": plan.token_count,
    }


def plan_structured_history(
    prefix: AuthoritativePREPrefix,
) -> StructuredTokenPlan:
    """Return the one deterministic semantic token plan for a frozen prefix."""

    if not isinstance(prefix, AuthoritativePREPrefix):
        raise TypeError("planner input must be an AuthoritativePREPrefix")
    validate_authoritative_pre_prefix(prefix)
    annotations_by_event = {
        annotation.event_id: annotation
        for annotation in prefix.v1_annotations
    }
    tokens: list[StructuredTokenDescriptor] = []

    def append(
        token_type: str,
        event_id: str,
        event_index: int,
        day: int,
        phase: str,
        *,
        semantic_index: int = 0,
        source: str | None = None,
        action: str | None = None,
        target: str | None = None,
    ) -> None:
        tokens.append(
            StructuredTokenDescriptor(
                token_index=len(tokens),
                token_type=token_type,
                source=source,
                action=action,
                target=target,
                day=day,
                phase=phase,
                event_id=event_id,
                event_index=event_index,
                semantic_index=semantic_index,
            )
        )

    for event in prefix.public_event_history.events:
        day = event.temporal_state.day
        phase = event.temporal_state.phase.value
        source = (
            event.speaker
            if event.event_type in {"turn_start", "public_speech"}
            else None
        )
        append(
            event.event_type,
            event.event_id,
            event.event_index,
            day,
            phase,
            source=source,
        )
        if event.event_type == "public_speech":
            annotation = annotations_by_event[event.event_id]
            if annotation.status is V1AnnotationStatus.OK:
                for semantic_index, speech_action in enumerate(
                    annotation.actions,
                    start=1,
                ):
                    append(
                        "speech_action",
                        event.event_id,
                        event.event_index,
                        day,
                        phase,
                        semantic_index=semantic_index,
                        source=speech_action.subject,
                        action=speech_action.action,
                        target=speech_action.object,
                    )
        elif event.event_type == "vote_result":
            for semantic_index, vote in enumerate(event.votes, start=1):
                append(
                    "vote",
                    event.event_id,
                    event.event_index,
                    day,
                    phase,
                    semantic_index=semantic_index,
                    source=vote.voter,
                    target=vote.target,
                )
        elif event.event_type == "exile_result":
            for semantic_index, player in enumerate(
                event.affected_players,
                start=1,
            ):
                append(
                    "exiled_player",
                    event.event_id,
                    event.event_index,
                    day,
                    phase,
                    semantic_index=semantic_index,
                    target=player,
                )
        elif event.event_type == "death_announcement":
            for semantic_index, player in enumerate(
                event.affected_players,
                start=1,
            ):
                append(
                    "dead_player",
                    event.event_id,
                    event.event_index,
                    day,
                    phase,
                    semantic_index=semantic_index,
                    target=player,
                )

    frozen_tokens = tuple(tokens)
    provisional = StructuredTokenPlan(
        planner_version=STRUCTURED_TOKEN_PLANNER_VERSION,
        prefix_digest=prefix.prefix_digest,
        tokens=frozen_tokens,
        token_count=len(frozen_tokens),
        plan_digest="",
    )
    plan_digest = sha256_bytes(
        canonical_json_bytes(_plan_record_without_digest(provisional))
    )
    return replace(provisional, plan_digest=plan_digest)


__all__ = [
    "STRUCTURED_TOKEN_PLANNER_VERSION",
    "STRUCTURED_TOKEN_TYPES",
    "StructuredTokenDescriptor",
    "StructuredTokenPlan",
    "plan_structured_history",
]
