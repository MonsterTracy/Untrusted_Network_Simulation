"""Phase-2 Action Contract V1: structured semantics, without gameplay effects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping

from werewolf.canonical_collection.pre import validate_authoritative_pre_prefix
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.phase2_offline import phase_context


CONTRACT_VERSION = "phase2_action_contract_v1"
NO_STANCE = "NO_STANCE"
REQUEST_TYPE = "CURRENT_SUSPICION_BASIS"


class ActionContractError(ValueError):
    """A current-PRE action context or structured request is invalid."""


class Action(str, Enum):
    PUSH = "PUSH"
    REDIRECT = "REDIRECT"
    PROBE = "PROBE"


@dataclass(frozen=True)
class InformationRequestV1:
    target_j: str
    addressee_j: str
    request_type: str = REQUEST_TYPE

    def to_record(self) -> dict:
        return {"target_j": self.target_j, "addressee_j": self.addressee_j,
                "request_type": self.request_type}


@dataclass(frozen=True)
class ObservationWindowV1:
    """Expected public speech turns, not future observed event IDs or answers."""

    after_boundary_id: str
    before_actor: str
    phase: str
    expected_speakers: tuple[str, ...]

    def to_record(self) -> dict:
        return {"after_boundary_id": self.after_boundary_id,
                "before_actor": self.before_actor, "phase": self.phase,
                "event_type": "public_speech",
                "expected_speakers": list(self.expected_speakers)}


@dataclass(frozen=True)
class ActionContextV1:
    game_id: str
    boundary_id: str
    prefix_digest: str
    public_history_digest: str
    phase: str
    acting_wolf: str
    alive: tuple[str, ...]
    known_wolves: frozenset[str]
    competition: tuple[str, ...]
    public_speaker_queue: tuple[str, ...]

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value for value in
               (self.game_id, self.boundary_id, self.prefix_digest,
                self.public_history_digest)):
            raise ActionContractError("missing PRE identity")
        if self.phase not in ("speech", "speech_pk"):
            raise ActionContractError("action phase must be speech or speech_pk")
        if (not isinstance(self.known_wolves, frozenset)
                or len(self.known_wolves) != 2
                or not self.known_wolves <= set(PLAYER_IDS)
                or self.acting_wolf not in self.known_wolves):
            raise ActionContractError("invalid legally known wolf team")
        for name, values in (("alive", self.alive), ("competition", self.competition),
                             ("public_speaker_queue", self.public_speaker_queue)):
            if (not isinstance(values, tuple) or not values or
                    len(values) != len(set(values)) or not set(values) <= set(PLAYER_IDS)):
                raise ActionContractError(f"invalid {name}")
        if self.acting_wolf not in self.alive or self.acting_wolf not in self.public_speaker_queue:
            raise ActionContractError("acting wolf is not an alive phase speaker")
        if (not set(self.competition) <= set(self.alive)
                or (self.phase == "speech" and set(self.competition) != set(self.alive))
                or set(self.public_speaker_queue) !=
                (set(self.alive) if self.phase == "speech" else set(self.competition))):
            raise ActionContractError("competition or queue differs from phase rules")
        if self.alive != tuple(p for p in PLAYER_IDS if p in self.alive) or self.competition != tuple(
                p for p in PLAYER_IDS if p in self.competition):
            raise ActionContractError("alive and competition require canonical seat order")
        members = self.alive if self.phase == "speech" else self.competition
        start = members.index(self.public_speaker_queue[0])
        if self.public_speaker_queue != members[start:] + members[:start]:
            raise ActionContractError("public queue must be a cyclic seat-order rotation")

    @property
    def legal_targets(self) -> tuple[str, ...]:
        return tuple(p for p in PLAYER_IDS
                     if p in self.competition and p in self.alive and p not in self.known_wolves)


def context_from_pre(pre, known_wolves: frozenset[str]) -> ActionContextV1:
    """Read only a current authoritative PRE plus the acting wolf's legal team."""
    validate_authoritative_pre_prefix(pre)
    phase, competition, _remaining = phase_context(pre)
    events = pre.public_event_history.events
    start = max(i for i, event in enumerate(events) if event.event_type == "phase_change")
    observed = tuple(event.speaker for event in events[start + 1:]
                     if event.event_type == "turn_start")
    members = tuple(pre.alive_observer_ids) if phase == "speech" else competition
    first = members.index(observed[0])
    queue = members[first:] + members[:first]
    return ActionContextV1(pre.game_id, pre.boundary_id, pre.prefix_digest,
                           pre.public_event_history.digest, phase, pre.current_speaker,
                           tuple(pre.alive_observer_ids), known_wolves,
                           competition, queue)


def probe_continuation(context: ActionContextV1, candidate_j: str) -> tuple[str, ObservationWindowV1] | None:
    """Direct-to-j request; j must speak before the next same-phase wolf."""
    if candidate_j not in context.legal_targets:
        return None
    queue = context.public_speaker_queue
    current = queue.index(context.acting_wolf)
    try:
        j_pos = queue.index(candidate_j)
    except ValueError:
        return None
    if j_pos <= current:
        return None
    later = next((p for p in range(j_pos + 1, len(queue))
                  if queue[p] in context.known_wolves and queue[p] in context.alive), None)
    if later is None:
        return None
    actor = queue[later]
    return actor, ObservationWindowV1(context.boundary_id, actor, context.phase,
                                      queue[current + 1:later])


def select_redirect_target(context: ActionContextV1, candidate_j: str,
                           p_tilde: Mapping[str, float]) -> str | None:
    """Max frozen p_tilde over alternatives; canonical seat order breaks exact ties."""
    if candidate_j not in context.legal_targets:
        raise ActionContractError("illegal candidate j")
    alternatives = tuple(k for k in context.legal_targets if k != candidate_j)
    if not alternatives:
        return None
    if (not isinstance(p_tilde, Mapping) or
            not set(alternatives) <= set(p_tilde) <= set(context.legal_targets)):
        raise ActionContractError("p_tilde must cover all alternatives and no illegal target")
    if any(type(p_tilde[k]) not in (int, float) or not math.isfinite(p_tilde[k])
           or not 0.0 <= p_tilde[k] <= 1.0 for k in p_tilde):
        raise ActionContractError("invalid p_tilde value")
    return max(alternatives, key=lambda k: p_tilde[k])


@dataclass(frozen=True)
class Phase2SemanticPlanV1:
    action: Action
    candidate_j: str
    commitment_target: str | None
    rejected_target: str | None
    redirect_target: str | None
    information_request: InformationRequestV1 | None
    vote_intent: str
    phase: str
    acting_wolf: str
    legal_targets: tuple[str, ...]
    continuation_actor: str | None
    expected_observation_window: ObservationWindowV1 | None

    def to_record(self) -> dict:
        return {"schema_version": CONTRACT_VERSION,
                "action": self.action.value if isinstance(self.action, Action) else self.action,
                "candidate_j": self.candidate_j,
                "commitment_target": self.commitment_target,
                "rejected_target": self.rejected_target,
                "redirect_target": self.redirect_target,
                "information_request": None if self.information_request is None else self.information_request.to_record(),
                "vote_intent": self.vote_intent, "phase": self.phase,
                "acting_wolf": self.acting_wolf, "legal_targets": list(self.legal_targets),
                "continuation_actor": self.continuation_actor,
                "expected_observation_window": None if self.expected_observation_window is None
                else self.expected_observation_window.to_record()}


@dataclass(frozen=True)
class VerificationV1:
    valid: bool
    invalid_reason: str | None


def verify_plan(context: ActionContextV1, plan: Phase2SemanticPlanV1) -> VerificationV1:
    """Validate structured intent only; no language or outcome is inspected."""
    if not isinstance(context, ActionContextV1) or not isinstance(plan, Phase2SemanticPlanV1):
        raise TypeError("expected V1 context and plan")
    def fail(reason: str) -> VerificationV1:
        return VerificationV1(False, reason)
    if plan.phase != context.phase or plan.acting_wolf != context.acting_wolf:
        return fail("OPPORTUNITY_MISMATCH")
    if plan.legal_targets != context.legal_targets or plan.candidate_j not in context.legal_targets:
        return fail("ILLEGAL_CANDIDATE")
    if plan.action is Action.PUSH:
        if (plan.commitment_target != plan.candidate_j or plan.vote_intent != plan.candidate_j
                or any(x is not None for x in (plan.rejected_target, plan.redirect_target,
                                               plan.information_request, plan.continuation_actor,
                                               plan.expected_observation_window))):
            return fail("PUSH_SEMANTICS_MISMATCH")
    elif plan.action is Action.REDIRECT:
        k = plan.redirect_target
        if k not in context.legal_targets or k == plan.candidate_j:
            return fail("ILLEGAL_REDIRECT_TARGET")
        if (plan.commitment_target != k or plan.rejected_target != plan.candidate_j
                or plan.vote_intent != k or any(x is not None for x in
                (plan.information_request, plan.continuation_actor, plan.expected_observation_window))):
            return fail("REDIRECT_SEMANTICS_MISMATCH")
    elif plan.action is Action.PROBE:
        continuation = probe_continuation(context, plan.candidate_j)
        if continuation is None:
            return fail("NO_PROBE_CONTINUATION")
        request = plan.information_request
        if (not isinstance(request, InformationRequestV1)
                or request.target_j != plan.candidate_j
                or request.addressee_j != plan.candidate_j
                or request.request_type != REQUEST_TYPE):
            return fail("INVALID_INFORMATION_REQUEST")
        if (any(x is not None for x in (plan.commitment_target, plan.rejected_target,
                                       plan.redirect_target)) or plan.vote_intent != NO_STANCE
                or (plan.continuation_actor, plan.expected_observation_window) != continuation):
            return fail("PROBE_SEMANTICS_MISMATCH")
    else:
        return fail("UNKNOWN_ACTION")
    return VerificationV1(True, None)


def push_plan(context: ActionContextV1, j: str) -> Phase2SemanticPlanV1:
    plan = Phase2SemanticPlanV1(Action.PUSH, j, j, None, None, None, j,
                                context.phase, context.acting_wolf, context.legal_targets,
                                None, None)
    result = verify_plan(context, plan)
    if not result.valid:
        raise ActionContractError(result.invalid_reason)
    return plan


def redirect_plan(context: ActionContextV1, j: str,
                  p_tilde: Mapping[str, float]) -> Phase2SemanticPlanV1:
    k = select_redirect_target(context, j, p_tilde)
    if k is None:
        raise ActionContractError("NO_REDIRECT_ALTERNATIVE")
    plan = Phase2SemanticPlanV1(Action.REDIRECT, j, k, j, k, None, k,
                                context.phase, context.acting_wolf, context.legal_targets,
                                None, None)
    result = verify_plan(context, plan)
    if not result.valid:
        raise ActionContractError(result.invalid_reason)
    return plan


def probe_plan(context: ActionContextV1, j: str) -> Phase2SemanticPlanV1:
    continuation = probe_continuation(context, j)
    if continuation is None:
        raise ActionContractError("NO_PROBE_CONTINUATION")
    actor, window = continuation
    plan = Phase2SemanticPlanV1(Action.PROBE, j, None, None, None,
                                InformationRequestV1(j, j), NO_STANCE,
                                context.phase, context.acting_wolf, context.legal_targets,
                                actor, window)
    result = verify_plan(context, plan)
    if not result.valid:
        raise ActionContractError(result.invalid_reason)
    return plan
