"""Online Pilot-T assignment, execution, and day consequence in separate stages."""

from __future__ import annotations

from dataclasses import dataclass, replace

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.phase2_backend_audit import Phase2BackendCallV1
from werewolf.phase2_online_plan import OnlineTerminalAssignmentV1, OnlineProbeAssignmentV1
from werewolf.phase2_decision_opportunity import Phase2DecisionOpportunityV1
from werewolf.phase2_treatment import Phase2TreatmentV1
from werewolf.phase2_pilot_records import PublicProbeObservationV1
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


PROBE_RECORD_VERSION = "phase2_online_probe_record_v1"

# Concrete two-stage protocol, not a general workflow language.
_PROBE_NEXT = {
    "ASSIGNED": {"T1_ATTEMPTED"},
    "T1_ATTEMPTED": {"T1_COMMITTED", "T1_LANGUAGE_INVALID"},
    "T1_COMMITTED": {"T3_SCHEDULED", "EXECUTION_RECORDED"},
    "T1_LANGUAGE_INVALID": {"T3_CANCELLED", "EXECUTION_RECORDED"},
    "T3_CANCELLED": {"EXECUTION_RECORDED"},
    "T3_SCHEDULED": {"T3_REACHED"},
    "T3_REACHED": {"T3_PREPARED"},
    "T3_PREPARED": {"T3_COMMITTED", "T3_LANGUAGE_INVALID"},
    "T3_COMMITTED": {"EXECUTION_RECORDED"},
    "T3_LANGUAGE_INVALID": {"EXECUTION_RECORDED"},
    "EXECUTION_RECORDED": {"DAY_CONSEQUENCE_RECORDED"},
    "DAY_CONSEQUENCE_RECORDED": {"GAME_RESULT_RECORDED"},
    "GAME_RESULT_RECORDED": set(), "STRUCTURAL_FAILURE": set(),
}


@dataclass(frozen=True)
class Phase2StrategyStageExecutionV1:
    opportunity: Phase2DecisionOpportunityV1
    treatment: Phase2TreatmentV1
    language_audit_digest: str
    attempt_count: int
    backend_calls: tuple[Phase2BackendCallV1, ...]
    canonical_link: object | None = None
    failure_reason: str | None = None

    def to_record(self):
        link = self.canonical_link
        return {"opportunity": self.opportunity.to_record(),
                "treatment": self.treatment.to_record(),
                "language_audit_digest": self.language_audit_digest,
                "attempt_count": self.attempt_count,
                "backend_calls": [call.to_record() for call in self.backend_calls],
                "success": link is not None, "failure_reason": self.failure_reason,
                "canonical_event_id": link.canonical_event_id if link else None,
                "canonical_event_digest": link.canonical_event_digest if link else None,
                "generated_text_digest": link.public_text_digest if link else None}


@dataclass(frozen=True)
class Phase2OnlineProbeRecordV1:
    assignment: OnlineProbeAssignmentV1
    lifecycle: tuple[str, ...] = ("ASSIGNED",)
    t1_execution: Phase2StrategyStageExecutionV1 | None = None
    t3_execution: Phase2StrategyStageExecutionV1 | None = None
    t3_opportunity: Phase2DecisionOpportunityV1 | None = None
    t3_treatment: Phase2TreatmentV1 | None = None
    observations: tuple[PublicProbeObservationV1, ...] = ()
    structural_failure_reason: str | None = None
    day_outcome: Phase2DayOutcomeV1 | None = None
    reference_artifact_digest: str | None = None
    final_game_result: str | None = None

    def __post_init__(self):
        if not isinstance(self.assignment, OnlineProbeAssignmentV1):
            raise OnlinePilotRecordError("whole-strategy assignment required")
        validate_probe_lifecycle_record(self.to_record())

    @property
    def game_id(self):
        return self.assignment.opportunity.legal_context.game_id

    @property
    def backend_calls(self):
        return tuple(call for stage in (self.t1_execution, self.t3_execution)
                     if stage is not None for call in stage.backend_calls)

    @property
    def execution_success(self):
        if "EXECUTION_RECORDED" not in self.lifecycle:
            return None
        stage = self.t3_execution or self.t1_execution
        return stage is not None and stage.canonical_link is not None

    @property
    def theta_audit_label(self):
        return None

    def to_record(self):
        last = self.t3_execution or self.t1_execution
        completed = "EXECUTION_RECORDED" in self.lifecycle
        link = last.canonical_link if completed and last else None
        return {"schema_version": PROBE_RECORD_VERSION,
                "assignment": self.assignment.to_record(),
                "lifecycle": list(self.lifecycle),
                "stages": {"T1": self.t1_execution.to_record() if self.t1_execution else None,
                           "T3": self.t3_execution.to_record() if self.t3_execution else None},
                "continuation": {
                    "opportunity": self.t3_opportunity.to_record() if self.t3_opportunity else None,
                    "treatment": self.t3_treatment.to_record() if self.t3_treatment else None},
                "execution": {
                    "success": self.execution_success,
                    "failure_reason": last.failure_reason if completed and last else None,
                    "language_attempt_count": sum(stage.attempt_count for stage in
                                                  (self.t1_execution, self.t3_execution)
                                                  if stage) if completed else None,
                    "language_audit_digest": last.language_audit_digest if completed and last else None,
                    "backend_call_audit_digest": sha256_bytes(canonical_json_bytes([
                        call.to_record() for call in self.backend_calls])),
                    "canonical_commit_success": self.execution_success,
                    "canonical_event_id": link.canonical_event_id if link else None,
                    "canonical_event_digest": link.canonical_event_digest if link else None,
                    "generated_text_digest": link.public_text_digest if link else None},
                "observations": [event.to_record() for event in self.observations],
                "structural_failure_reason": self.structural_failure_reason,
                "day_consequence": None if self.day_outcome is None else {
                    "exiled_player": self.day_outcome.exiled_player,
                    "Y": self.day_outcome.category.value,
                    "s_plus": list(self.day_outcome.s_plus),
                    "v_ref": self.day_outcome.v_ref, "l_ref": self.day_outcome.l_ref,
                    "reference_artifact_digest": self.reference_artifact_digest},
                "offline_audit": {"theta_ac": None, "final_game_result": self.final_game_result}}

    def digest(self):
        return sha256_bytes(canonical_json_bytes(self.to_record()))


def validate_probe_lifecycle_record(raw):
    """Validate persisted stage semantics independently of runner state."""
    def require(condition, message):
        if not condition:
            raise OnlinePilotRecordError(message)

    require(raw.get("schema_version") == PROBE_RECORD_VERSION, "unknown Probe record")
    events = raw["lifecycle"]
    require(isinstance(events, list) and events and events[0] == "ASSIGNED"
            and len(events) == len(set(events)), "invalid Probe lifecycle")
    for before, after in zip(events, events[1:]):
        require(before in _PROBE_NEXT and (after in _PROBE_NEXT[before] or
                (after == "STRUCTURAL_FAILURE" and before not in
                 ("STRUCTURAL_FAILURE", "GAME_RESULT_RECORDED"))), "illegal Probe transition")
    assignment = raw["assignment"]
    strategy = assignment["assigned_strategy"]
    probe = strategy == "PROBE_THEN_REDIRECT"
    require(strategy in ("IMMEDIATE_REDIRECT", "PROBE_THEN_REDIRECT"), "unknown strategy")
    require(probe or not any(event.startswith("T3_") for event in events), "control acquired T3")
    require("T3_SCHEDULED" not in events or "T1_COMMITTED" in events, "uncommitted T1 scheduled")
    require("T3_CANCELLED" not in events or "T1_LANGUAGE_INVALID" in events, "invalid cancellation")
    require(not ("T3_SCHEDULED" in events and "T1_LANGUAGE_INVALID" in events), "invalid Probe retained T3")
    original = assignment["opportunity"]
    continuation = raw["continuation"]
    for name in ("T1", "T3"):
        stage = raw["stages"][name]
        finished = f"{name}_COMMITTED" in events or f"{name}_LANGUAGE_INVALID" in events
        require((stage is not None) == finished, "stage evidence/lifecycle mismatch")
        if stage is None:
            continue
        require(type(stage["attempt_count"]) is int and 1 <= stage["attempt_count"] <= 2
                and bool(stage["language_audit_digest"]), "not a recognized language result")
        success = f"{name}_COMMITTED" in events
        require(stage["success"] is success and
                (stage["failure_reason"] is None if success else bool(stage["failure_reason"]))
                and all(bool(stage[field]) if success else stage[field] is None
                        for field in ("canonical_event_id", "canonical_event_digest", "generated_text_digest")),
                "invalid canonical stage proof")
        expected_op = original if name == "T1" else continuation["opportunity"]
        expected_treatment = assignment["treatment"] if name == "T1" else continuation["treatment"]
        require(stage["opportunity"] == expected_op and stage["treatment"] == expected_treatment,
                "stage plan was replaced")
        require(all(call["assignment_id"] == sha256_bytes(canonical_json_bytes(assignment))
                    and call["opportunity_digest"] == sha256_bytes(canonical_json_bytes(expected_op))
                    and call["treatment_id"] == expected_treatment["treatment_id"]
                    and call["boundary_id"] == expected_op["identity"]["boundary_id"]
                    and call["game_id"] == expected_op["identity"]["game_id"]
                    for call in stage["backend_calls"]), "stage backend binding differs")
    prepared = "T3_PREPARED" in events
    require((continuation["opportunity"] is not None) == prepared and
            (continuation["treatment"] is not None) == prepared, "continuation plan not write-ahead")
    window = original["probe"]["observation_window"]
    observations = raw["observations"]
    require(not observations or "T3_SCHEDULED" in events, "observations lack a committed Probe")
    require([row["speaker"] for row in observations] == window["expected_speakers"][:len(observations)]
            and all(row["event_type"] == "public_speech" and row["text"].strip() for row in observations)
            and len({row["event_id"] for row in observations}) == len(observations), "observation window mismatch")
    if "T3_REACHED" in events:
        require(len(observations) == len(window["expected_speakers"]), "incomplete observation window")
    if prepared:
        op, treatment = continuation["opportunity"], continuation["treatment"]
        from werewolf.phase2_online_plan import probe_opportunity_from_record
        from werewolf.phase2_actions import Action
        typed_op = probe_opportunity_from_record(op)
        identity, origin = op["identity"], original["identity"]
        require(identity["candidate_j"] == origin["candidate_j"] and
                identity["game_id"] == origin["game_id"] and identity["phase"] == origin["phase"] and
                identity["acting_wolf"] == original["probe"]["continuation_actor"] and
                identity["boundary_id"] != origin["boundary_id"] and
                op["public_legal"]["alive"] == original["public_legal"]["alive"] and
                op["public_legal"]["C"] == original["public_legal"]["C"] and
                op["public_legal"]["known_wolves"] == original["public_legal"]["known_wolves"] and
                op["public_legal"]["speaker_queue"] == original["public_legal"]["speaker_queue"] and
                op["evidence"]["q_source_digest"] == original["evidence"]["q_source_digest"] and
                op["evidence"]["mapper_artifact_digest"] == original["evidence"]["mapper_artifact_digest"] and
                treatment["requested_action"] == "REDIRECT" and
                treatment["assignment_source"] == "strategy_continuation" and
                treatment["assignment_probability"] == 1, "illegal T3 context/plan")
        parent_digest = sha256_bytes(canonical_json_bytes(assignment))
        op_digest = sha256_bytes(canonical_json_bytes(op))
        require(treatment["opportunity_digest"] == op_digest and
                treatment["randomization_key"] == sha256_bytes(canonical_json_bytes([
                    "phase2_probe_continuation_v1", parent_digest, op_digest])),
                "T3 plan lost its parent assignment binding")
        from werewolf.phase2_treatment import build_phase2_treatment
        require(treatment == build_phase2_treatment(
            typed_op, Action.REDIRECT, assignment_source="strategy_continuation",
            assignment_probability=1, randomization_key=treatment["randomization_key"]).to_record(),
            "T3 treatment differs from current full-panel selector")
    completed = "EXECUTION_RECORDED" in events
    require(not completed or ("T1_COMMITTED" in events or "T1_LANGUAGE_INVALID" in events), "unfinished T1")
    require(not completed or not probe or any(event in events for event in
            ("T3_CANCELLED", "T3_COMMITTED", "T3_LANGUAGE_INVALID")), "unfinished strategy")
    last = raw["stages"]["T3"] or raw["stages"]["T1"]
    execution = raw["execution"]
    require(execution["success"] is (last["success"] if completed else None)
            and execution["canonical_commit_success"] is execution["success"], "strategy execution mismatch")
    calls = [call for name in ("T1", "T3") if raw["stages"][name] is not None
             for call in raw["stages"][name]["backend_calls"]]
    require(execution["backend_call_audit_digest"] == sha256_bytes(canonical_json_bytes(calls)),
            "strategy backend summary changed")
    for field in ("failure_reason", "language_audit_digest", "canonical_event_id",
                  "canonical_event_digest", "generated_text_digest"):
        require(execution[field] == (last[field] if completed else None),
                "strategy execution summary changed")
    require(execution["language_attempt_count"] == (sum(raw["stages"][name]["attempt_count"]
            for name in ("T1", "T3") if raw["stages"][name] is not None) if completed else None),
            "strategy attempt summary changed")
    require((raw["structural_failure_reason"] is not None) == (events[-1] == "STRUCTURAL_FAILURE"),
            "structural failure lacks evidence")
    require(raw["structural_failure_reason"] is None or
            (isinstance(raw["structural_failure_reason"], str) and raw["structural_failure_reason"].strip()),
            "empty structural failure evidence")
    require((raw["day_consequence"] is not None) == ("DAY_CONSEQUENCE_RECORDED" in events),
            "day outcome lifecycle mismatch")
    require((raw["offline_audit"]["final_game_result"] is not None) == ("GAME_RESULT_RECORDED" in events)
            and raw["offline_audit"]["theta_ac"] is None, "invalid offline audit")
