"""Versioned, structured Phase-2 action audit; no speech text or gameplay calls."""

from __future__ import annotations

from dataclasses import dataclass

from werewolf.artifact_io import canonical_json_bytes
from werewolf.phase2_actions import (
    ActionContextV1, Phase2SemanticPlanV1, VerificationV1, verify_plan,
)


AUDIT_VERSION = "phase2_action_audit_v1"


@dataclass(frozen=True)
class Phase2ActionAuditV1:
    game_id: str
    boundary_id: str
    prefix_digest: str
    public_history_digest: str
    phase: str
    acting_wolf: str
    candidate_j: str
    requested_action: str
    legal_J: tuple[str, ...]
    commitment_target: str | None
    rejected_target: str | None
    redirect_target: str | None
    probe_request: dict | None
    continuation_actor: str | None
    expected_observation_window: dict | None
    execution_valid: bool
    invalid_reason: str | None
    # Deliberately unfilled until a separate language/gameplay execution stage.
    generated_text_digest: str | None = None
    perceived_actions: tuple | None = None
    observed_information_request: bool | None = None
    observation_event_ids: tuple[str, ...] | None = None
    reconsideration_boundary: str | None = None
    reconsideration_executed: bool | None = None
    request_executed: bool | None = None
    response_opportunity_reached: bool | None = None
    public_observation_realized: bool | None = None
    information_gain: bool | None = None

    def to_record(self) -> dict:
        return {"schema_version": AUDIT_VERSION,
                "game_id": self.game_id, "boundary_id": self.boundary_id,
                "prefix_digest": self.prefix_digest,
                "public_history_digest": self.public_history_digest,
                "phase": self.phase, "acting_wolf": self.acting_wolf,
                "candidate_j": self.candidate_j,
                "requested_action": self.requested_action,
                "legal_J": list(self.legal_J),
                "commitment_target": self.commitment_target,
                "rejected_target": self.rejected_target,
                "redirect_target": self.redirect_target,
                "probe_request": self.probe_request,
                "continuation_actor": self.continuation_actor,
                "expected_observation_window": self.expected_observation_window,
                "execution_valid": self.execution_valid,
                "invalid_reason": self.invalid_reason,
                "generated_text_digest": self.generated_text_digest,
                "perceived_actions": self.perceived_actions,
                "observed_information_request": self.observed_information_request,
                "observation_event_ids": self.observation_event_ids,
                "reconsideration_boundary": self.reconsideration_boundary,
                "reconsideration_executed": self.reconsideration_executed,
                "request_executed": self.request_executed,
                "response_opportunity_reached": self.response_opportunity_reached,
                "public_observation_realized": self.public_observation_realized,
                "information_gain": self.information_gain}

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_record())


def audit_structured_plan(context: ActionContextV1,
                          plan: Phase2SemanticPlanV1) -> Phase2ActionAuditV1:
    """Only structural validity. No future execution field is inferred."""
    result: VerificationV1 = verify_plan(context, plan)
    return Phase2ActionAuditV1(
        context.game_id, context.boundary_id, context.prefix_digest,
        context.public_history_digest, context.phase, context.acting_wolf,
        plan.candidate_j,
        plan.action.value if hasattr(plan.action, "value") else str(plan.action),
        context.legal_targets, plan.commitment_target, plan.rejected_target,
        plan.redirect_target,
        None if plan.information_request is None else plan.information_request.to_record(),
        plan.continuation_actor,
        None if plan.expected_observation_window is None else plan.expected_observation_window.to_record(),
        result.valid, result.invalid_reason)
