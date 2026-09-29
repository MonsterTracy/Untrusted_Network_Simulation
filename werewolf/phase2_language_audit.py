"""Private Phase-2 language execution audit, separate from frozen action V1."""

from __future__ import annotations

from dataclasses import dataclass

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.phase2_action_audit import Phase2ActionAuditV1
from werewolf.phase2_actions import Phase2SemanticPlanV1
from werewolf.phase2_language import Phase2SpeechSemanticV1


LANGUAGE_AUDIT_VERSION = "phase2_language_execution_audit_v1"


def plan_digest(plan: Phase2SemanticPlanV1) -> str:
    return sha256_bytes(canonical_json_bytes(plan.to_record()))


def semantic_digest(semantic: Phase2SpeechSemanticV1) -> str:
    return sha256_bytes(canonical_json_bytes(semantic.to_record()))


@dataclass(frozen=True)
class LanguageAttemptV1:
    attempt: int
    generated_text: str | None
    generated_text_digest: str | None
    perceived_semantics: dict | None
    perceived_semantics_digest: str | None
    language_execution_valid: bool
    language_invalid_reason: str | None

    def to_record(self) -> dict:
        return {"attempt": self.attempt,
                "generated_text": self.generated_text,
                "generated_text_digest": self.generated_text_digest,
                "perceived_semantics": self.perceived_semantics,
                "perceived_semantics_digest": self.perceived_semantics_digest,
                "language_execution_valid": self.language_execution_valid,
                "language_invalid_reason": self.language_invalid_reason}


@dataclass(frozen=True)
class Phase2LanguageAuditV1:
    structured_action_audit: Phase2ActionAuditV1
    requested_action: str
    requested_plan_digest: str
    structured_execution_valid: bool
    structured_invalid_reason: str | None
    attempts: tuple[LanguageAttemptV1, ...]
    generated_text: str | None
    generated_text_digest: str | None
    perceived_semantics: dict | None
    perceived_semantics_digest: str | None
    language_execution_valid: bool
    language_invalid_reason: str | None
    # Later gameplay/observation layers alone may fill these fields.
    response_opportunity_reached: bool | None = None
    public_observation_realized: bool | None = None
    reconsideration_reached: bool | None = None
    information_gain: bool | None = None

    def to_record(self) -> dict:
        return {"schema_version": LANGUAGE_AUDIT_VERSION,
                "structured_action_audit": self.structured_action_audit.to_record(),
                "requested_action": self.requested_action,
                "requested_plan_digest": self.requested_plan_digest,
                "structured_execution_valid": self.structured_execution_valid,
                "structured_invalid_reason": self.structured_invalid_reason,
                "attempts": [attempt.to_record() for attempt in self.attempts],
                "generated_text": self.generated_text,
                "generated_text_digest": self.generated_text_digest,
                "perceived_semantics": self.perceived_semantics,
                "perceived_semantics_digest": self.perceived_semantics_digest,
                "language_execution_valid": self.language_execution_valid,
                "language_invalid_reason": self.language_invalid_reason,
                "response_opportunity_reached": self.response_opportunity_reached,
                "public_observation_realized": self.public_observation_realized,
                "reconsideration_reached": self.reconsideration_reached,
                "information_gain": self.information_gain}

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_record())
