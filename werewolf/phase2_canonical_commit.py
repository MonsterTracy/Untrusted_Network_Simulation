"""Opt-in Phase-2 text binding to the existing canonical recorder transaction.

The canonical V1 parser owns its annotation. The Phase-2 audit remains a
separate sidecar and cannot supply V1 actions or alter the actual vote.
"""

from __future__ import annotations

from dataclasses import dataclass

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection.public_history import freeze_public_event_history
from werewolf.phase2_action_audit import audit_structured_plan
from werewolf.phase2_actions import InformationRequestV1
from werewolf.phase2_decision_opportunity import Phase2DecisionOpportunityV1
from werewolf.phase2_intervention import (
    Phase2CheckpointClosureV1, local_env_state_digest,
    validate_treatment_checkpoint_binding,
)
from werewolf.phase2_language import Phase2SpeechSemanticV1, verify_language_execution
from werewolf.phase2_language_audit import plan_digest, semantic_digest
from werewolf.phase2_treatment import Phase2TreatmentV1
from werewolf.phase2_verified_speech import Phase2VerifiedSpeechResultV1
from werewolf.speech.verified_commit import bind_audited_public_speech


class Phase2CanonicalCommitError(ValueError):
    """A verified Phase-2 speech cannot be bound to this live PRE."""


@dataclass(frozen=True)
class Phase2CanonicalSpeechLinkV1:
    """Audit-only linkage; never inserted into V1 speech annotation."""

    canonical_event_id: str
    canonical_event_digest: str
    phase2_language_audit_digest: str
    treatment_id: str
    checkpoint_closure_digest: str
    semantic_plan_digest: str
    public_text_digest: str

    def to_record(self) -> dict:
        return {"schema_version": "phase2_canonical_speech_link_v1",
                "canonical_event_id": self.canonical_event_id,
                "canonical_event_digest": self.canonical_event_digest,
                "phase2_language_audit_digest": self.phase2_language_audit_digest,
                "treatment_id": self.treatment_id,
                "checkpoint_closure_digest": self.checkpoint_closure_digest,
                "semantic_plan_digest": self.semantic_plan_digest,
                "public_text_digest": self.public_text_digest}

    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_record()))


def _validated_text(opportunity, treatment, verified):
    if (not isinstance(verified, Phase2VerifiedSpeechResultV1)
            or not verified.success or verified.semantic_plan != treatment.plan
            or verified.language_audit.structured_action_audit !=
               audit_structured_plan(opportunity.legal_context, treatment.plan)
            or verified.language_audit.language_execution_valid is not True
            or verified.language_audit.structured_execution_valid is not True
            or verified.language_audit.requested_action != treatment.action.value
            or verified.language_audit.requested_plan_digest != plan_digest(treatment.plan)
            or not verified.language_audit.attempts):
        raise Phase2CanonicalCommitError("Phase-2 language audit or semantic plan mismatch")
    text = verified.public_text
    audit = verified.language_audit
    last = audit.attempts[-1]
    text_digest = sha256_bytes(text.encode("utf-8")) if isinstance(text, str) else None
    if (not text or not text.strip() or audit.generated_text != text
            or last.generated_text != text or not last.language_execution_valid
            or audit.generated_text_digest != text_digest
            or last.generated_text_digest != text_digest
            or audit.perceived_semantics != last.perceived_semantics
            or audit.perceived_semantics_digest != last.perceived_semantics_digest):
        raise Phase2CanonicalCommitError("Phase-2 verified text digest differs from final audit")
    raw = audit.perceived_semantics
    try:
        perceived = Phase2SpeechSemanticV1(
            raw["speaker"], raw["phase"], tuple(raw["commitment_targets"]),
            tuple(raw["rejected_targets"]), tuple(raw["vote_intent_targets"]),
            tuple(InformationRequestV1(**item) for item in raw["information_requests"]),
            raw["abstain_intent"], raw["private_fact_claim"])
    except (KeyError, TypeError, ValueError) as error:
        raise Phase2CanonicalCommitError("Phase-2 perceived semantics are invalid") from error
    if (raw != perceived.to_record()
            or semantic_digest(perceived) != audit.perceived_semantics_digest
            or not verify_language_execution(treatment.plan, perceived,
                                             opportunity.legal_context).valid):
        raise Phase2CanonicalCommitError("Phase-2 audit no longer verifies the frozen plan")
    return text, text_digest


def commit_phase2_verified_speech(*, env, recorder,
                                  opportunity: Phase2DecisionOpportunityV1,
                                  treatment: Phase2TreatmentV1,
                                  verified: Phase2VerifiedSpeechResultV1,
                                  closure: Phase2CheckpointClosureV1):
    """Canonical parser → existing staged recorder commit; no regeneration.

    This is a live single-PRE seam, not a branch executor. `closure` proves
    PRE identity only; it explicitly does not prove a restorable checkpoint.
    """
    validate_treatment_checkpoint_binding(opportunity, treatment, closure)
    context = opportunity.legal_context
    pending = getattr(recorder, "_pending", None)
    prefix = getattr(pending, "prefix", None)
    handoff = getattr(pending, "handoff", None)
    if (getattr(recorder, "_env", None) is not env or prefix is None
            or handoff is None
            or getattr(pending, "raw_action", None) is not None
            or (prefix.game_id, prefix.boundary_id, prefix.prefix_digest,
                prefix.current_speaker) != (context.game_id, context.boundary_id,
                                            context.prefix_digest, context.acting_wolf)
            or (getattr(handoff, "boundary_id", None),
                getattr(handoff, "prefix_digest", None),
                getattr(handoff, "observer_id", None)) !=
               (context.boundary_id, context.prefix_digest, context.acting_wolf)):
        raise Phase2CanonicalCommitError("canonical recorder PRE mismatch")
    history = freeze_public_event_history(env.public_events)
    if (history.digest != context.public_history_digest
            or prefix.public_event_history.digest != history.digest
            or env.phase != context.phase
            or f"player{env.current_act_idx + 1}" != context.acting_wolf
            or pending.event_count_before != len(env.public_events)
            or local_env_state_digest(env) != closure.local_env_digest):
        raise Phase2CanonicalCommitError("environment PRE drift before canonical parse")
    speech, text_digest = _validated_text(opportunity, treatment, verified)
    audit_digest = sha256_bytes(verified.language_audit.canonical_bytes())
    closure_digest = closure.digest()
    frozen_plan_digest = plan_digest(treatment.plan)
    speaker_number = env.current_act_idx + 1
    expected_event = {
        "event_id": f"event-{len(env.public_events):06d}",
        "event_index": len(env.public_events),
        "event_type": "public_speech",
        "speaker": context.acting_wolf,
        "raw_text": speech,
    }
    audit_context = recorder.call_audit.speech_perception_context(
        event_id=expected_event["event_id"],
        boundary_id=context.boundary_id, speaker_id=speaker_number)
    with audit_context:
        canonical_audit = env.speech_perceiver.parse_with_audit(
            speaker=speaker_number, speech=speech, day=env.day, phase=env.phase)
    if local_env_state_digest(env) != closure.local_env_digest:
        raise Phase2CanonicalCommitError("environment changed during canonical parse")
    envelope = bind_audited_public_speech(
        speech=speech, speaker=context.acting_wolf, perception=canonical_audit,
        day=env.day, phase=env.phase, public_history_digest=history.digest,
        perceiver=env.speech_perceiver)
    result = recorder.commit_verified_speech(env, envelope,
                                             expected_event=expected_event)
    link = Phase2CanonicalSpeechLinkV1(
        expected_event["event_id"], sha256_bytes(canonical_json_bytes(expected_event)),
        audit_digest, treatment.treatment_id, closure_digest, frozen_plan_digest,
        text_digest)
    return result, link
