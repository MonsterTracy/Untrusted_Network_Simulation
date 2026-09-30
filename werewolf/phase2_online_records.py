"""Online Pilot-T assignment, execution, and day consequence in separate stages."""

from __future__ import annotations

from dataclasses import dataclass, replace

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.phase2_backend_audit import Phase2BackendCallV1
from werewolf.phase2_online_plan import OnlineTerminalAssignmentV1
from werewolf.phase2_outcome import Phase2DayOutcomeV1


RECORD_VERSION = "phase2_online_intervention_record_v1"


class OnlinePilotRecordError(ValueError):
    """Assignment/execution/consequence evidence is contradictory."""


@dataclass(frozen=True)
class Phase2OnlineInterventionRecordV1:
    assignment: OnlineTerminalAssignmentV1
    execution_success: bool | None = None
    execution_failure_reason: str | None = None
    language_attempt_count: int | None = None
    language_audit_digest: str | None = None
    backend_calls: tuple[Phase2BackendCallV1, ...] = ()
    canonical_commit_success: bool | None = None
    canonical_event_digest: str | None = None
    canonical_event_id: str | None = None
    generated_text_digest: str | None = None
    day_outcome: Phase2DayOutcomeV1 | None = None
    reference_artifact_digest: str | None = None
    theta_audit_label: bool | None = None
    final_game_result: str | None = None

    def __post_init__(self):
        if self.execution_success is None:
            if any(value is not None for value in (
                    self.execution_failure_reason, self.language_attempt_count,
                    self.language_audit_digest, self.canonical_commit_success,
                    self.canonical_event_digest, self.canonical_event_id,
                    self.generated_text_digest, self.day_outcome)) or self.backend_calls:
                raise OnlinePilotRecordError("execution cannot precede assignment")
            return
        if (type(self.execution_success) is not bool
                or type(self.canonical_commit_success) is not bool
                or type(self.language_attempt_count) is not int
                or not 0 <= self.language_attempt_count <= 2
                or (self.language_audit_digest is not None and
                    (not isinstance(self.language_audit_digest, str)
                     or not self.language_audit_digest))
                or not isinstance(self.backend_calls, tuple)
                or any(not isinstance(call, Phase2BackendCallV1)
                       or call.treatment_id != self.assignment.treatment.treatment_id
                       or call.opportunity_digest != self.assignment.opportunity.digest()
                       or call.assignment_id != self.assignment.digest()
                       for call in self.backend_calls)
                or self.execution_success != self.canonical_commit_success):
            raise OnlinePilotRecordError("language and canonical execution proof disagree")
        if self.execution_success:
            if (not self.language_audit_digest
                    or self.execution_failure_reason is not None
                    or not self.canonical_event_digest or not self.canonical_event_id
                    or not self.generated_text_digest):
                raise OnlinePilotRecordError("successful treatment lacks exact commit proof")
        elif (not self.execution_failure_reason or any(value is not None for value in (
                self.canonical_event_digest, self.canonical_event_id,
                self.generated_text_digest))):
            raise OnlinePilotRecordError("failed treatment cannot claim a canonical event")
        if self.day_outcome is not None and (
                self.execution_success is not True
                or not isinstance(self.day_outcome, Phase2DayOutcomeV1)
                or self.day_outcome.reference.s_pre != self.assignment.opportunity.s_pre
                or not self.reference_artifact_digest):
            raise OnlinePilotRecordError("day outcome lacks frozen reference lineage")
        if type(self.theta_audit_label) not in (bool, type(None)):
            raise OnlinePilotRecordError("Theta is offline audit only")

    @property
    def game_id(self) -> str:
        return self.assignment.opportunity.legal_context.game_id

    def to_record(self) -> dict:
        return {"schema_version": RECORD_VERSION,
                "assignment": self.assignment.to_record(),
                "execution": {
                    "success": self.execution_success,
                    "failure_reason": self.execution_failure_reason,
                    "language_attempt_count": self.language_attempt_count,
                    "language_audit_digest": self.language_audit_digest,
                    "backend_call_audit_digest": sha256_bytes(canonical_json_bytes([
                        call.to_record() for call in self.backend_calls])),
                    "canonical_commit_success": self.canonical_commit_success,
                    "canonical_event_id": self.canonical_event_id,
                    "canonical_event_digest": self.canonical_event_digest,
                    "generated_text_digest": self.generated_text_digest},
                "day_consequence": None if self.day_outcome is None else {
                    "exiled_player": self.day_outcome.exiled_player,
                    "Y": self.day_outcome.category.value,
                    "s_plus": list(self.day_outcome.s_plus),
                    "v_ref": self.day_outcome.v_ref,
                    "l_ref": self.day_outcome.l_ref,
                    "reference_artifact_digest": self.reference_artifact_digest},
                "offline_audit": {"theta_ac": self.theta_audit_label,
                                  "final_game_result": self.final_game_result}}

    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_record()))


def record_online_execution(record: Phase2OnlineInterventionRecordV1, *,
                            language_audit_digest: str | None, attempt_count: int,
                            backend_calls: tuple[Phase2BackendCallV1, ...],
                            canonical_link=None, failure_reason: str | None = None):
    if record.execution_success is not None:
        raise OnlinePilotRecordError("treatment execution already recorded")
    success = canonical_link is not None
    if success == (failure_reason is not None):
        raise OnlinePilotRecordError("execution requires exactly one success or failure")
    return replace(record, execution_success=success,
                   execution_failure_reason=failure_reason,
                   language_attempt_count=attempt_count,
                   language_audit_digest=language_audit_digest,
                   backend_calls=backend_calls,
                   canonical_commit_success=success,
                   canonical_event_id=(canonical_link.canonical_event_id if success else None),
                   canonical_event_digest=(canonical_link.canonical_event_digest if success else None),
                   generated_text_digest=(canonical_link.public_text_digest if success else None))


def record_online_day_outcome(record: Phase2OnlineInterventionRecordV1,
                              outcome: Phase2DayOutcomeV1, *,
                              reference_artifact_digest: str):
    if record.execution_success is not True or record.day_outcome is not None:
        raise OnlinePilotRecordError("day resolution requires successful execution")
    return replace(record, day_outcome=outcome,
                   reference_artifact_digest=reference_artifact_digest)
