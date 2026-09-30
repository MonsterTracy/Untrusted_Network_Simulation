"""Versioned intervention outcome and sequential Probe audit records."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.phase2_actions import Action
from werewolf.phase2_decision_opportunity import Phase2DecisionOpportunityV1
from werewolf.phase2_outcome import Phase2DayOutcomeV1
from werewolf.phase2_treatment import Phase2TreatmentV1


CONSEQUENCE_VERSION = "phase2_action_consequence_v1"
PROBE_VERSION = "phase2_probe_sequence_v1"


class PilotRecordError(ValueError):
    """A controlled branch audit lacks a required stage or lineage."""


def _lineage(opportunity, treatment, checkpoint_identity, branch_identity,
             branch_seed, execution_proof_digest):
    if (not isinstance(opportunity, Phase2DecisionOpportunityV1)
            or not isinstance(treatment, Phase2TreatmentV1)
            or treatment.opportunity_digest != opportunity.digest()
            or treatment.opportunity_identity != opportunity.identity
            or not all(isinstance(value, str) and value for value in (
                checkpoint_identity, branch_identity, execution_proof_digest))
            or type(branch_seed) is not int or branch_seed < 0
            or branch_identity != sha256_bytes(canonical_json_bytes([
                checkpoint_identity, treatment.treatment_id, branch_seed]))):
        raise PilotRecordError("intervention opportunity or branch lineage differs")


@dataclass(frozen=True)
class Phase2ActionConsequenceRecordV1:
    opportunity: Phase2DecisionOpportunityV1
    treatment: Phase2TreatmentV1
    checkpoint_identity: str
    branch_identity: str
    branch_seed: int
    execution_proof_digest: str
    language_audit_digest: str
    language_execution_valid: bool
    outcome: Phase2DayOutcomeV1
    reference_artifact_digest: str
    theta_audit_label: bool | None = None

    def __post_init__(self):
        _lineage(self.opportunity, self.treatment, self.checkpoint_identity,
                 self.branch_identity, self.branch_seed, self.execution_proof_digest)
        if (self.treatment.action not in (Action.PUSH, Action.REDIRECT)
                or not self.language_audit_digest or not self.language_execution_valid
                or not isinstance(self.outcome, Phase2DayOutcomeV1)
                or self.outcome.reference.s_pre != self.opportunity.s_pre
                or not self.reference_artifact_digest
                or type(self.theta_audit_label) not in (bool, type(None))):
            raise PilotRecordError("terminal action consequence is incomplete")

    def to_record(self) -> dict:
        return {"schema_version": CONSEQUENCE_VERSION,
                "runtime_safe": {"opportunity": self.opportunity.to_record(),
                                 "treatment": self.treatment.to_record(),
                                 "p_tilde_at_pre": self.opportunity.p_tilde_j},
                "offline_audit": {"theta_ac": self.theta_audit_label,
                                  "checkpoint_identity": self.checkpoint_identity,
                                  "branch_identity": self.branch_identity,
                                  "branch_seed": self.branch_seed,
                                  "execution_proof_digest": self.execution_proof_digest,
                                  "language_audit_digest": self.language_audit_digest,
                                  "language_execution_valid": self.language_execution_valid,
                                  "actual_exiled_player": self.outcome.exiled_player,
                                  "Y": self.outcome.category.value,
                                  "s_plus": list(self.outcome.s_plus),
                                  "v_ref": self.outcome.v_ref,
                                  "l_ref": self.outcome.l_ref,
                                  "reference_artifact_digest": self.reference_artifact_digest}}

    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_record()))


@dataclass(frozen=True)
class PublicProbeObservationV1:
    event_id: str
    speaker: str
    text: str
    event_type: str = "public_speech"

    def __post_init__(self):
        if (not isinstance(self.event_id, str) or not self.event_id
                or not isinstance(self.speaker, str) or not self.speaker
                or not isinstance(self.text, str) or not self.text.strip()
                or self.event_type != "public_speech"):
            raise PilotRecordError("Probe O may contain public speech events only")

    def to_record(self) -> dict:
        return {"event_id": self.event_id, "event_type": self.event_type,
                "speaker": self.speaker, "text": self.text}


@dataclass(frozen=True)
class Phase2ProbeSequenceRecordV1:
    opportunity: Phase2DecisionOpportunityV1
    treatment: Phase2TreatmentV1
    checkpoint_identity: str
    branch_identity: str
    branch_seed: int
    execution_proof_digest: str
    request_executed: bool | None = None
    response_opportunity_reached: bool | None = None
    public_events: tuple[PublicProbeObservationV1, ...] = ()
    public_observation_realized: bool | None = None
    observation_digest: str | None = None
    reconsideration_reached: bool | None = None
    updated_legal_targets: tuple[str, ...] | None = None
    updated_q_digest: str | None = None
    updated_p_panel: tuple[tuple[str, float], ...] | None = None
    original_j_still_legal: bool | None = None
    updated_action_universe: tuple[Action, ...] | None = None
    day_outcome: Phase2DayOutcomeV1 | None = None
    information_gain: None = None

    def __post_init__(self):
        _lineage(self.opportunity, self.treatment, self.checkpoint_identity,
                 self.branch_identity, self.branch_seed, self.execution_proof_digest)
        if (self.treatment.action is not Action.PROBE
                or Action.PROBE not in self.opportunity.legal_actions
                or self.information_gain is not None):
            raise PilotRecordError("invalid Probe treatment or information gain")
        if self.request_executed is None:
            if any(value is not None for value in (self.response_opportunity_reached,
                    self.public_observation_realized, self.observation_digest,
                    self.reconsideration_reached, self.day_outcome)) or self.public_events:
                raise PilotRecordError("Probe stages must be recorded in order")
            return
        if type(self.request_executed) is not bool:
            raise PilotRecordError("request execution must be observed")
        if self.response_opportunity_reached is not None:
            if type(self.response_opportunity_reached) is not bool or type(self.public_observation_realized) is not bool:
                raise PilotRecordError("observation status must be explicit")
            window = self.opportunity.observation_window
            if (window is None or not isinstance(self.public_events, tuple)
                    or any(not isinstance(event, PublicProbeObservationV1) for event in self.public_events)
                    or len(self.public_events) > len(window.expected_speakers)
                    or tuple(event.speaker for event in self.public_events) !=
                    window.expected_speakers[:len(self.public_events)]
                    or len({event.event_id for event in self.public_events}) != len(self.public_events)
                    or self.public_observation_realized != bool(self.public_events)
                    or self.observation_digest != sha256_bytes(canonical_json_bytes([
                        event.to_record() for event in self.public_events]))):
                raise PilotRecordError("Probe O differs from frozen public speech window")
        elif self.public_events or self.public_observation_realized is not None or self.observation_digest is not None:
            raise PilotRecordError("observation window not yet reached")
        if self.reconsideration_reached is not None and type(self.reconsideration_reached) is not bool:
            raise PilotRecordError("continuation status must be boolean")
        if self.reconsideration_reached is None:
            if any(value is not None for value in (self.updated_legal_targets,
                    self.updated_q_digest, self.updated_p_panel,
                    self.original_j_still_legal, self.updated_action_universe)):
                raise PilotRecordError("no continuation state without T3")
        elif self.reconsideration_reached:
            if (self.response_opportunity_reached is not True
                    or tuple(event.speaker for event in self.public_events) !=
                    self.opportunity.observation_window.expected_speakers
                    or not self.updated_q_digest or self.updated_legal_targets is None
                    or self.updated_p_panel is None or self.updated_action_universe is None
                    or not isinstance(self.updated_legal_targets, tuple)
                    or not isinstance(self.updated_p_panel, tuple)
                    or not isinstance(self.updated_action_universe, tuple)
                    or any(not isinstance(action, Action) for action in self.updated_action_universe)
                    or self.original_j_still_legal != (self.opportunity.candidate_j in self.updated_legal_targets)
                    or tuple(k for k, _ in self.updated_p_panel) != self.updated_legal_targets
                    or any(not math.isfinite(p) or not 0 <= p <= 1 for _, p in self.updated_p_panel)):
                raise PilotRecordError("incomplete continuation PRE evidence")
        elif any(value is not None for value in (self.updated_legal_targets,
                self.updated_q_digest, self.updated_p_panel,
                self.original_j_still_legal, self.updated_action_universe)):
            raise PilotRecordError("continuation not reached")
        if self.day_outcome is not None and (
                self.response_opportunity_reached is None
                or self.day_outcome.reference.s_pre != self.opportunity.s_pre):
            raise PilotRecordError("Probe day outcome requires T2 and the same S_pre")

    def to_record(self) -> dict:
        return {"schema_version": PROBE_VERSION,
                "T0": {"opportunity": self.opportunity.to_record(),
                       "treatment": self.treatment.to_record(),
                       "p_tilde_before": self.opportunity.p_tilde_j,
                       "legal_candidates": list(self.opportunity.legal_context.legal_targets),
                       "request": self.treatment.plan.information_request.to_record()},
                "T1": {"request_executed": self.request_executed},
                "T2": {"response_opportunity_reached": self.response_opportunity_reached,
                       "public_events": [event.to_record() for event in self.public_events],
                       "public_observation_realized": self.public_observation_realized,
                       "observation_digest": self.observation_digest},
                "T3": {"reconsideration_reached": self.reconsideration_reached,
                       "updated_legal_targets": self.updated_legal_targets,
                       "updated_q_digest": self.updated_q_digest,
                       "updated_p_panel": self.updated_p_panel,
                       "original_j_still_legal": self.original_j_still_legal,
                       "updated_action_universe": None if self.updated_action_universe is None else [
                           action.value for action in self.updated_action_universe]},
                "T4": {"Y": None if self.day_outcome is None else self.day_outcome.category.value,
                       "l_ref": None if self.day_outcome is None else self.day_outcome.l_ref},
                "audit": {"checkpoint_identity": self.checkpoint_identity,
                          "branch_identity": self.branch_identity,
                          "branch_seed": self.branch_seed,
                          "execution_proof_digest": self.execution_proof_digest,
                          "information_gain": None}}

    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_record()))


def record_probe_execution(record: Phase2ProbeSequenceRecordV1, executed: bool) -> Phase2ProbeSequenceRecordV1:
    if record.request_executed is not None or type(executed) is not bool:
        raise PilotRecordError("Probe T1 already recorded or invalid")
    return replace(record, request_executed=executed)


def record_probe_observation(record: Phase2ProbeSequenceRecordV1, *, reached: bool,
                             events: tuple[PublicProbeObservationV1, ...]) -> Phase2ProbeSequenceRecordV1:
    if record.request_executed is None or record.response_opportunity_reached is not None or type(reached) is not bool:
        raise PilotRecordError("Probe T2 requires completed T1")
    return replace(record, response_opportunity_reached=reached, public_events=events,
                   public_observation_realized=bool(events),
                   observation_digest=sha256_bytes(canonical_json_bytes([
                       event.to_record() for event in events])))


def record_probe_reconsideration(record: Phase2ProbeSequenceRecordV1, *, reached: bool,
                                 legal_targets: tuple[str, ...] | None = None,
                                 q_digest: str | None = None,
                                 p_panel: tuple[tuple[str, float], ...] | None = None,
                                 actions: tuple[Action, ...] | None = None) -> Phase2ProbeSequenceRecordV1:
    if record.response_opportunity_reached is None or record.reconsideration_reached is not None:
        raise PilotRecordError("Probe T3 requires completed T2")
    return replace(record, reconsideration_reached=reached,
                   updated_legal_targets=legal_targets if reached else None,
                   updated_q_digest=q_digest if reached else None,
                   updated_p_panel=p_panel if reached else None,
                   original_j_still_legal=(record.opportunity.candidate_j in legal_targets)
                   if reached and legal_targets is not None else None,
                   updated_action_universe=actions if reached else None)


def record_probe_day_outcome(record: Phase2ProbeSequenceRecordV1,
                             outcome: Phase2DayOutcomeV1) -> Phase2ProbeSequenceRecordV1:
    if (record.response_opportunity_reached is None or record.day_outcome is not None
            or not isinstance(outcome, Phase2DayOutcomeV1)):
        raise PilotRecordError("Probe T4 requires completed T2 and a real day outcome")
    return replace(record, day_outcome=outcome)
