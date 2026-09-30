"""Preregistered controlled-intervention assignment, never observational votes."""

from __future__ import annotations

from dataclasses import dataclass
import math

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.phase2_actions import (
    Action, Phase2SemanticPlanV1, probe_plan, push_plan, verify_plan,
)
from werewolf.phase2_decision_opportunity import Phase2DecisionOpportunityV1


VERSION = "phase2_treatment_v1"
ASSIGNMENT_SOURCES = frozenset(("paired_branch", "randomized_pilot"))


class TreatmentError(ValueError):
    """A treatment is not an assigned, legal current-PRE intervention."""


@dataclass(frozen=True)
class Phase2TreatmentV1:
    treatment_id: str
    opportunity_digest: str
    opportunity_identity: tuple[str, str, str, str, str, str]
    action: Action
    candidate_j: str
    plan: Phase2SemanticPlanV1
    assignment_source: str
    assignment_probability: float
    randomization_key: str

    def __post_init__(self):
        if (not self.treatment_id or not self.opportunity_digest
                or not self.randomization_key or self.assignment_source not in ASSIGNMENT_SOURCES
                or type(self.assignment_probability) not in (float, int)
                or not math.isfinite(self.assignment_probability)
                or not 0 < self.assignment_probability <= 1
                or (self.assignment_source == "paired_branch"
                    and self.assignment_probability != 1)
                or not verify_plan_from_identity(self)
                or self.treatment_id != sha256_bytes(canonical_json_bytes([
                    VERSION, self.opportunity_digest, self.action.value,
                    self.assignment_source, self.assignment_probability,
                    self.randomization_key]))):
            raise TreatmentError("invalid controlled treatment")

    def to_record(self) -> dict:
        return {"schema_version": VERSION, "treatment_id": self.treatment_id,
                "opportunity_digest": self.opportunity_digest,
                "opportunity_identity": list(self.opportunity_identity),
                "requested_action": self.action.value, "candidate_j": self.candidate_j,
                "semantic_plan": self.plan.to_record(),
                "assignment_source": self.assignment_source,
                "assignment_probability": self.assignment_probability,
                "randomization_key": self.randomization_key,
                "observational_vote_intent_is_treatment": False}


def verify_plan_from_identity(treatment: Phase2TreatmentV1) -> bool:
    plan = treatment.plan
    return (len(treatment.opportunity_identity) == 6
            and isinstance(treatment.action, Action)
            and isinstance(plan, Phase2SemanticPlanV1)
            and plan.action is treatment.action
            and plan.candidate_j == treatment.candidate_j
            and treatment.opportunity_identity[3:] == (
                plan.acting_wolf, plan.phase, plan.candidate_j))


def build_phase2_treatment(opportunity: Phase2DecisionOpportunityV1, action: Action,
                           *, assignment_source: str, assignment_probability: float,
                           randomization_key: str) -> Phase2TreatmentV1:
    if not isinstance(opportunity, Phase2DecisionOpportunityV1) or action not in opportunity.legal_actions:
        raise TreatmentError("action is unavailable at this opportunity")
    if (not isinstance(assignment_source, str) or not assignment_source
            or assignment_source not in ASSIGNMENT_SOURCES):
        raise TreatmentError("controlled assignment source is required")
    context, j = opportunity.legal_context, opportunity.candidate_j
    if action is Action.PUSH:
        plan = push_plan(context, j)
    elif action is Action.PROBE:
        plan = probe_plan(context, j)
    elif action is Action.REDIRECT:
        k = opportunity.redirect_target
        # The k was already frozen by the mapper panel; never reselect it.
        plan = Phase2SemanticPlanV1(Action.REDIRECT, j, k, j, k, None, k,
                                    context.phase, context.acting_wolf, context.legal_targets,
                                    None, None)
    else:
        raise TreatmentError("unknown action")
    if not verify_plan(context, plan).valid:
        raise TreatmentError("treatment plan violates frozen Action Contract")
    key = [VERSION, opportunity.digest(), action.value, assignment_source,
           assignment_probability, randomization_key]
    treatment_id = sha256_bytes(canonical_json_bytes(key))
    return Phase2TreatmentV1(treatment_id, opportunity.digest(), opportunity.identity,
                             action, j, plan, assignment_source,
                             assignment_probability, randomization_key)
