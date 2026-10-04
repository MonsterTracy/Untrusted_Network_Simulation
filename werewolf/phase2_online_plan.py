"""Frozen Pilot-T V1 candidate sampling, treatment assignment, and stopping."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.phase2_actions import Action, ActionContextV1, ObservationWindowV1, probe_continuation
from werewolf.phase2_decision_opportunity import CandidateEvidenceV1, Phase2DecisionOpportunityV1
from werewolf.phase2_treatment import Phase2TreatmentV1, build_phase2_treatment


PLAN_VERSION = "phase2_online_terminal_pilot_plan_v1"
SELECTION_RULE = "first_pn_eligible_pre_uniform_candidate_v1"
PROBE_PLAN_VERSION = "phase2_online_probe_pilot_plan_v1"
PROBE_SELECTION_RULE = "first_probe_redirect_eligible_pre_uniform_candidate_v1"
FROZEN = "FROZEN"
QUALIFICATION = "qualification"
FORMAL_PILOT = "pilot"
TARGET_ASSIGNMENTS = {QUALIFICATION: 10, FORMAL_PILOT: 120}


class OnlinePilotPlanError(ValueError):
    """Pilot-T selection or assignment contract was violated."""


class ProbeStrategy(str, Enum):
    IMMEDIATE_REDIRECT = "IMMEDIATE_REDIRECT"
    PROBE_THEN_REDIRECT = "PROBE_THEN_REDIRECT"


@dataclass(frozen=True)
class Phase2OnlineProbePilotPlanV1:
    pilot_id: str
    assignment_seed: int
    target_assignment_count: int
    max_games_attempted: int
    campaign_purpose: str = QUALIFICATION
    candidate_selection_rule: str = PROBE_SELECTION_RULE
    selection_status: str = FROZEN

    def __post_init__(self):
        if (not isinstance(self.pilot_id, str) or not self.pilot_id
                or type(self.assignment_seed) is not int or self.assignment_seed < 0
                or type(self.target_assignment_count) is not int
                or self.target_assignment_count < 1
                or type(self.max_games_attempted) is not int
                or self.max_games_attempted < self.target_assignment_count
                or self.campaign_purpose not in (QUALIFICATION, FORMAL_PILOT)
                or self.candidate_selection_rule != PROBE_SELECTION_RULE
                or self.selection_status != FROZEN):
            raise OnlinePilotPlanError("invalid frozen PROBE strategy plan")

    def to_record(self) -> dict:
        return {"schema_version": PROBE_PLAN_VERSION, "pilot_id": self.pilot_id,
                "assignment_seed": self.assignment_seed,
                "campaign_purpose": self.campaign_purpose,
                "target_assignment_count": self.target_assignment_count,
                "max_games_attempted": self.max_games_attempted,
                "immediate_redirect_probability": 0.5,
                "probe_then_redirect_probability": 0.5,
                "candidate_selection_rule": self.candidate_selection_rule,
                "selection_status": self.selection_status,
                "max_assignments_per_game": 1,
                "eligible_strategies": [strategy.value for strategy in ProbeStrategy],
                "eligible_actions": [Action.REDIRECT.value, Action.PROBE.value]}

    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_record()))


@dataclass(frozen=True)
class Phase2OnlineTerminalPilotPlanV1:
    pilot_id: str
    assignment_seed: int
    campaign_purpose: str = FORMAL_PILOT
    push_probability: float = 0.5
    candidate_selection_rule: str = SELECTION_RULE
    selection_status: str = FROZEN
    max_games_attempted: int | None = None

    def __post_init__(self):
        if (not isinstance(self.pilot_id, str) or not self.pilot_id
                or type(self.assignment_seed) is not int or self.assignment_seed < 0
                or self.campaign_purpose not in TARGET_ASSIGNMENTS
                or type(self.push_probability) not in (int, float)
                or not math.isfinite(self.push_probability)
                or self.push_probability != 0.5
                or self.candidate_selection_rule != SELECTION_RULE
                or self.selection_status != FROZEN
                or (self.max_games_attempted is not None and
                    (type(self.max_games_attempted) is not int or
                     self.max_games_attempted < TARGET_ASSIGNMENTS[self.campaign_purpose]))):
            raise OnlinePilotPlanError("invalid frozen Pilot-T V1 plan")

    @property
    def target_assignment_count(self) -> int:
        return TARGET_ASSIGNMENTS[self.campaign_purpose]

    def to_record(self) -> dict:
        return {"schema_version": PLAN_VERSION, "pilot_id": self.pilot_id,
                "assignment_seed": self.assignment_seed,
                "campaign_purpose": self.campaign_purpose,
                "target_assignment_count": self.target_assignment_count,
                "max_games_attempted": self.max_games_attempted,
                "push_probability": self.push_probability,
                "redirect_probability": 1 - self.push_probability,
                "candidate_selection_rule": self.candidate_selection_rule,
                "selection_status": self.selection_status,
                "max_assignments_per_game": 1,
                "eligible_actions": [Action.PUSH.value, Action.REDIRECT.value]}

    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_record()))


@dataclass(frozen=True)
class CandidateSelectionV1:
    rule: str
    candidate_j: str
    candidate_pool: tuple[str, ...]
    candidate_selection_probability: float
    candidate_selection_seed: int
    candidate_selection_key: str
    game_id: str
    boundary_id: str
    prefix_digest: str
    plan_digest: str
    selection_digest: str

    def __post_init__(self):
        if (self.rule not in (SELECTION_RULE, PROBE_SELECTION_RULE)
                or not isinstance(self.candidate_pool, tuple)
                or len(self.candidate_pool) < (1 if self.rule == PROBE_SELECTION_RULE else 2)
                or len(set(self.candidate_pool)) != len(self.candidate_pool)
                or self.candidate_j not in self.candidate_pool
                or self.candidate_selection_probability != 1 / len(self.candidate_pool)
                or type(self.candidate_selection_seed) is not int
                or len(self.candidate_selection_key) != 64):
            raise OnlinePilotPlanError("invalid uniform candidate selection")

    def to_record(self) -> dict:
        return {"schema_version": "phase2_online_candidate_selection_v1",
                "rule": self.rule, "candidate_j": self.candidate_j,
                "candidate_pool": list(self.candidate_pool),
                "candidate_selection_probability": self.candidate_selection_probability,
                "candidate_selection_seed": self.candidate_selection_seed,
                "candidate_selection_key": self.candidate_selection_key,
                "game_id": self.game_id, "boundary_id": self.boundary_id,
                "prefix_digest": self.prefix_digest, "plan_digest": self.plan_digest,
                "selection_digest": self.selection_digest}


def _uniform_index(key_parts: list, size: int) -> tuple[int, str]:
    """Rejection sampling avoids modulo bias over SHA256 output."""
    if size < 1:
        raise OnlinePilotPlanError("empty candidate pool")
    limit = (1 << 256) - ((1 << 256) % size)
    counter = 0
    while True:
        key = sha256_bytes(canonical_json_bytes([*key_parts, counter]))
        value = int(key, 16)
        if value < limit:
            return value % size, key
        counter += 1


def select_candidate(plan: Phase2OnlineTerminalPilotPlanV1,
                     context: ActionContextV1) -> CandidateSelectionV1 | None:
    """Uniform j over the first PRE with at least two legal non-wolf targets."""
    if not isinstance(plan, Phase2OnlineTerminalPilotPlanV1) or not isinstance(context, ActionContextV1):
        raise OnlinePilotPlanError("plan and current public PRE context are required")
    pool = context.legal_targets
    if len(pool) < 2:
        return None
    index, key = _uniform_index([PLAN_VERSION, "candidate", plan.digest(),
                                 plan.assignment_seed, context.game_id,
                                 context.boundary_id, context.prefix_digest, list(pool)], len(pool))
    candidate = pool[index]
    identity = [plan.digest(), context.game_id, context.boundary_id,
                context.prefix_digest, list(pool), candidate, key, SELECTION_RULE]
    return CandidateSelectionV1(SELECTION_RULE, candidate, pool, 1 / len(pool),
                                plan.assignment_seed, key, context.game_id,
                                context.boundary_id, context.prefix_digest,
                                plan.digest(), sha256_bytes(canonical_json_bytes(identity)))


def probe_candidate_pool(context: ActionContextV1) -> tuple[str, ...]:
    """Joint legal PROBE/REDIRECT targets, independent of frozen scores."""
    if not isinstance(context, ActionContextV1):
        raise OnlinePilotPlanError("current public PRE context is required")
    if len(context.legal_targets) < 2:
        return ()
    return tuple(j for j in context.legal_targets if probe_continuation(context, j) is not None)


def select_probe_candidate(plan: Phase2OnlineProbePilotPlanV1,
                           context: ActionContextV1) -> CandidateSelectionV1 | None:
    if not isinstance(plan, Phase2OnlineProbePilotPlanV1):
        raise OnlinePilotPlanError("frozen PROBE strategy plan is required")
    pool = probe_candidate_pool(context)
    if not pool:
        return None
    index, key = _uniform_index([PROBE_PLAN_VERSION, "candidate", plan.digest(),
                                 plan.assignment_seed, context.game_id,
                                 context.boundary_id, context.prefix_digest, list(pool)], len(pool))
    candidate = pool[index]
    identity = [plan.digest(), context.game_id, context.boundary_id,
                context.prefix_digest, list(pool), candidate, key, PROBE_SELECTION_RULE]
    return CandidateSelectionV1(PROBE_SELECTION_RULE, candidate, pool, 1 / len(pool),
                                plan.assignment_seed, key, context.game_id,
                                context.boundary_id, context.prefix_digest,
                                plan.digest(), sha256_bytes(canonical_json_bytes(identity)))


@dataclass(frozen=True)
class OnlineTerminalAssignmentV1:
    selection: CandidateSelectionV1
    opportunity: Phase2DecisionOpportunityV1
    treatment: Phase2TreatmentV1
    assignment_seed: int
    assignment_key: str
    legal_action_set: tuple[Action, ...]

    def __post_init__(self):
        context = self.opportunity.legal_context
        if (self.selection.rule != SELECTION_RULE
                or self.selection.candidate_j != self.opportunity.candidate_j
                or self.selection.candidate_pool != context.legal_targets
                or (self.selection.game_id, self.selection.boundary_id,
                    self.selection.prefix_digest) !=
                   (context.game_id, context.boundary_id, context.prefix_digest)
                or self.treatment.opportunity_digest != self.opportunity.digest()
                or self.treatment.assignment_source != "randomized_pilot"
                or self.treatment.action not in (Action.PUSH, Action.REDIRECT)
                or self.treatment.action not in self.legal_action_set
                or self.legal_action_set != self.opportunity.legal_actions
                or not {Action.PUSH, Action.REDIRECT}.issubset(self.legal_action_set)
                or self.treatment.randomization_key != self.assignment_key):
            raise OnlinePilotPlanError("Pilot-T assignment differs from selected PRE")

    def to_record(self) -> dict:
        return {"schema_version": "phase2_online_terminal_assignment_v1",
                "selection": self.selection.to_record(),
                "opportunity": self.opportunity.to_record(),
                "legal_action_set": [action.value for action in self.legal_action_set],
                "assigned_action": self.treatment.action.value,
                "assignment_probability": self.treatment.assignment_probability,
                "assignment_seed": self.assignment_seed,
                "assignment_key": self.assignment_key,
                "treatment": self.treatment.to_record()}

    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_record()))


def assign_terminal_action(plan: Phase2OnlineTerminalPilotPlanV1,
                           selection: CandidateSelectionV1,
                           opportunity: Phase2DecisionOpportunityV1,
                           ) -> OnlineTerminalAssignmentV1:
    """SHA256 seed rule, independent of all post-PRE observations."""
    if (not isinstance(plan, Phase2OnlineTerminalPilotPlanV1)
            or not isinstance(selection, CandidateSelectionV1)
            or not isinstance(opportunity, Phase2DecisionOpportunityV1)
            or selection.plan_digest != plan.digest()
            or select_candidate(plan, opportunity.legal_context) != selection
            or selection.candidate_j != opportunity.candidate_j
            or not {Action.PUSH, Action.REDIRECT}.issubset(opportunity.legal_actions)):
        raise OnlinePilotPlanError("Pilot-T requires selected candidate with legal P and N")
    index, key = _uniform_index([PLAN_VERSION, "treatment", plan.digest(),
                                 plan.assignment_seed, opportunity.legal_context.game_id,
                                 opportunity.legal_context.boundary_id,
                                 opportunity.legal_context.prefix_digest], 2)
    action = Action.PUSH if index == 0 else Action.REDIRECT
    probability = (plan.push_probability if action is Action.PUSH
                   else 1 - plan.push_probability)
    treatment = build_phase2_treatment(
        opportunity, action, assignment_source="randomized_pilot",
        assignment_probability=probability, randomization_key=key)
    return OnlineTerminalAssignmentV1(selection, opportunity, treatment,
                                      plan.assignment_seed, key, opportunity.legal_actions)


@dataclass(frozen=True)
class OnlineProbeAssignmentV1:
    selection: CandidateSelectionV1
    opportunity: Phase2DecisionOpportunityV1
    treatment: Phase2TreatmentV1
    assignment_seed: int
    assignment_key: str
    strategy: ProbeStrategy

    def __post_init__(self):
        if (not isinstance(self.selection, CandidateSelectionV1)
                or not isinstance(self.opportunity, Phase2DecisionOpportunityV1)
                or not isinstance(self.treatment, Phase2TreatmentV1)
                or not isinstance(self.strategy, ProbeStrategy)):
            raise OnlinePilotPlanError("typed PROBE strategy assignment is required")
        context = self.opportunity.legal_context
        action = (Action.REDIRECT if self.strategy is ProbeStrategy.IMMEDIATE_REDIRECT
                  else Action.PROBE)
        if (self.selection.rule != PROBE_SELECTION_RULE
                or self.selection.candidate_j != self.opportunity.candidate_j
                or self.selection.candidate_pool != probe_candidate_pool(context)
                or (self.selection.game_id, self.selection.boundary_id,
                    self.selection.prefix_digest) !=
                   (context.game_id, context.boundary_id, context.prefix_digest)
                or type(self.assignment_seed) is not int or self.assignment_seed < 0
                or self.selection.candidate_selection_seed != self.assignment_seed
                or not isinstance(self.assignment_key, str)
                or len(self.assignment_key) != 64
                or any(c not in "0123456789abcdef" for c in self.assignment_key)
                or not {Action.REDIRECT, Action.PROBE}.issubset(self.opportunity.legal_actions)
                or self.treatment != build_phase2_treatment(
                    self.opportunity, action, assignment_source="randomized_pilot",
                    assignment_probability=0.5, randomization_key=self.assignment_key)):
            raise OnlinePilotPlanError("PROBE strategy assignment differs from selected PRE")

    @property
    def legal_action_set(self) -> tuple[Action, ...]:
        return self.opportunity.legal_actions

    @property
    def continuation_schedule(self) -> dict | None:
        if self.strategy is ProbeStrategy.IMMEDIATE_REDIRECT:
            return None
        return {"candidate_j": self.opportunity.candidate_j,
                "continuation_actor": self.opportunity.continuation_actor,
                "phase": self.opportunity.legal_context.phase,
                "observation_window": self.opportunity.observation_window.to_record()}

    def to_record(self) -> dict:
        return {"schema_version": "phase2_online_probe_assignment_v1",
                "selection": self.selection.to_record(),
                "opportunity": self.opportunity.to_record(),
                "legal_action_set": [action.value for action in self.legal_action_set],
                "assigned_strategy": self.strategy.value,
                "assigned_action": self.treatment.action.value,
                "assignment_probability": self.treatment.assignment_probability,
                "assignment_seed": self.assignment_seed,
                "assignment_key": self.assignment_key,
                "treatment": self.treatment.to_record(),
                "continuation_schedule": self.continuation_schedule}

    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_record()))


def assign_probe_strategy(plan: Phase2OnlineProbePilotPlanV1,
                          selection: CandidateSelectionV1,
                          opportunity: Phase2DecisionOpportunityV1) -> OnlineProbeAssignmentV1:
    """Randomize one whole strategy, independently of selection and score evidence."""
    if (not isinstance(plan, Phase2OnlineProbePilotPlanV1)
            or not isinstance(selection, CandidateSelectionV1)
            or not isinstance(opportunity, Phase2DecisionOpportunityV1)
            or selection.plan_digest != plan.digest()
            or select_probe_candidate(plan, opportunity.legal_context) != selection
            or selection.candidate_j != opportunity.candidate_j
            or not {Action.REDIRECT, Action.PROBE}.issubset(opportunity.legal_actions)):
        raise OnlinePilotPlanError("PROBE policy requires the selected joint legal candidate")
    context = opportunity.legal_context
    index, key = _uniform_index([PROBE_PLAN_VERSION, "strategy", plan.digest(),
                                 plan.assignment_seed, context.game_id,
                                 context.boundary_id, context.prefix_digest], 2)
    strategy = (ProbeStrategy.IMMEDIATE_REDIRECT if index == 0
                else ProbeStrategy.PROBE_THEN_REDIRECT)
    action = Action.REDIRECT if strategy is ProbeStrategy.IMMEDIATE_REDIRECT else Action.PROBE
    treatment = build_phase2_treatment(
        opportunity, action, assignment_source="randomized_pilot",
        assignment_probability=0.5, randomization_key=key)
    return OnlineProbeAssignmentV1(selection, opportunity, treatment,
                                   plan.assignment_seed, key, strategy)


def probe_opportunity_from_record(raw: dict) -> Phase2DecisionOpportunityV1:
    """Recover a typed current-PRE panel and verify its complete canonical record."""
    if not isinstance(raw, dict):
        raise OnlinePilotPlanError("PROBE opportunity record is required")
    try:
        identity = raw["identity"]
        legal = raw["public_legal"]
        evidence = raw["evidence"]
        context = ActionContextV1(
            identity["game_id"], identity["boundary_id"], identity["prefix_digest"],
            legal["public_history_digest"], identity["phase"], identity["acting_wolf"],
            tuple(legal["alive"]), frozenset(legal["known_wolves"]),
            tuple(legal["C"]), tuple(legal["speaker_queue"]))
        probe = raw["probe"]
        window = None
        if probe is not None:
            raw_window = probe["observation_window"]
            window = ObservationWindowV1(
                raw_window["after_boundary_id"], raw_window["before_actor"],
                raw_window["phase"], tuple(raw_window["expected_speakers"]))
        opportunity = Phase2DecisionOpportunityV1(
            context, identity["candidate_j"], tuple(raw["s_pre"]),
            legal["current_position"], legal["remaining_speakers"],
            evidence["q_source_digest"], evidence["q_matrix_digest"],
            evidence["mapper_artifact_digest"],
            tuple(CandidateEvidenceV1(**row) for row in evidence["panel"]),
            tuple(Action(action) for action in raw["legal_actions"]),
            raw["redirect_target"], None if probe is None else probe["request_type"],
            None if probe is None else probe["continuation_actor"], window)
        if (opportunity.to_record() != raw
                or canonical_json_bytes(opportunity.to_record()) != canonical_json_bytes(raw)):
            raise OnlinePilotPlanError("stored PROBE opportunity does not reproduce")
        return opportunity
    except (KeyError, TypeError, ValueError) as exc:
        raise OnlinePilotPlanError("invalid or altered PROBE opportunity record") from exc


def probe_assignment_from_record(record: dict,
                                 plan: Phase2OnlineProbePilotPlanV1) -> OnlineProbeAssignmentV1:
    """Reconstruct and reproduce the frozen selection, assignment, and treatment."""
    if not isinstance(record, dict) or not isinstance(plan, Phase2OnlineProbePilotPlanV1):
        raise OnlinePilotPlanError("PROBE assignment record and frozen plan are required")
    try:
        selected = record["selection"]
        selection = CandidateSelectionV1(
            selected["rule"], selected["candidate_j"], tuple(selected["candidate_pool"]),
            selected["candidate_selection_probability"], selected["candidate_selection_seed"],
            selected["candidate_selection_key"], selected["game_id"], selected["boundary_id"],
            selected["prefix_digest"], selected["plan_digest"], selected["selection_digest"])
        opportunity = probe_opportunity_from_record(record["opportunity"])
        expected = assign_probe_strategy(plan, selection, opportunity)
        expected_record = expected.to_record()
        if (expected_record != record
                or canonical_json_bytes(expected_record) != canonical_json_bytes(record)):
            raise OnlinePilotPlanError("stored PROBE assignment does not reproduce")
        return expected
    except (KeyError, TypeError, ValueError) as exc:
        raise OnlinePilotPlanError("invalid or altered PROBE assignment record") from exc


class OnlinePilotGameStateV1:
    """One assignment at most, including failed language execution."""

    def __init__(self):
        self.games_seen: set[str] = set()
        self.eligible_opportunities = 0
        self.assignments: dict[str, OnlineTerminalAssignmentV1 | OnlineProbeAssignmentV1] = {}

    def mark_seen(self, game_id: str) -> None:
        self.games_seen.add(game_id)

    def try_assign(self, plan, selection, opportunity, *, persist=None):
        game_id = opportunity.legal_context.game_id
        self.mark_seen(game_id)
        if game_id in self.assignments:
            return None
        if len(self.assignments) >= plan.target_assignment_count:
            return None
        self.eligible_opportunities += 1
        assignment = (assign_probe_strategy(plan, selection, opportunity)
                      if isinstance(plan, Phase2OnlineProbePilotPlanV1)
                      else assign_terminal_action(plan, selection, opportunity))
        if persist is not None:
            persist(assignment)  # durable write must succeed before runtime state or LLM
        self.assignments[game_id] = assignment
        return assignment


# Existing integration imports keep working while the formal contract uses the new name.
Phase2OnlinePilotPlanV1 = Phase2OnlineTerminalPilotPlanV1
