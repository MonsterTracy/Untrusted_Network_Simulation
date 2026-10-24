"""Restricted offline supervision; never import this module for online features."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from werewolf.canonical_collection.game_bundle import _observation_from_record
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.canonical_collection.trajectory_evidence import (
    BeliefObservation, BeliefObservationStatus,
)
from werewolf.observer_dtrs.context import ObserverContext


LABEL_VERSION = "observer_dtrs_effective_support_live_wolf_v1"


@dataclass(frozen=True)
class ObserverLabel:
    observation_id: str | None
    observation_digest: str | None
    attempt_id: str | None
    observation_status: str
    label_observed: bool
    raw_suspicion_support: tuple[str, ...] | None
    effective_suspicion_support: tuple[str, ...] | None
    event_label_y: int | None

    def to_record(self) -> dict:
        return {"label_construction_version": LABEL_VERSION, **asdict(self)}


def label_from_observation(context: ObserverContext,
                           observation: BeliefObservation | None) -> ObserverLabel:
    """Bind the raw canonical report and compute 1[effective support intersects W].

    A missing/failed report has no effective support or Y. A formal publication
    must reject such missing supervision rather than use this as row filtering.
    """
    if not isinstance(context, ObserverContext):
        raise TypeError("label construction requires ObserverContext")
    if observation is None:
        return ObserverLabel(None, None, None, "missing", False, None, None, None)
    if not isinstance(observation, BeliefObservation):
        raise TypeError("observation must be a canonical BeliefObservation or None")
    # Reuse the canonical decoder's schema, support, status and digest checks.
    observation = _observation_from_record(observation.to_record())
    if ((observation.game_id, observation.boundary_id, observation.prefix_digest,
         observation.observer_id, observation.day, observation.phase)
            != (context.game_id, context.boundary_id, context.prefix_digest,
                context.observer_id, context.day, context.phase)
            or not observation.observer_alive):
        raise ValueError("BeliefObservation does not bind to this observer PRE context")
    observed = observation.status is BeliefObservationStatus.SUCCESS
    raw = observation.suspicion_support
    effective = (raw or tuple(p for p in PLAYER_IDS if p != context.observer_id)) if observed else None
    y = int(bool(set(effective) & set(context.alive_wolf_ids))) if observed else None
    return ObserverLabel(observation.observation_id, observation.observation_digest,
        observation.attempt_id, observation.status.value, observed, raw, effective, y)
