"""Opt-in Phase-2 language result; never commits speech or executes a vote."""

from __future__ import annotations

from dataclasses import dataclass

from scripts.phase2_language_realization import realize_verify_action
from werewolf.phase2_actions import ActionContextV1, Phase2SemanticPlanV1
from werewolf.phase2_language import PublicLanguageContextV1
from werewolf.phase2_language_audit import Phase2LanguageAuditV1


@dataclass(frozen=True)
class Phase2VerifiedSpeechResultV1:
    success: bool
    public_text: str | None
    semantic_plan: Phase2SemanticPlanV1
    language_audit: Phase2LanguageAuditV1
    attempt_count: int
    failure_reason: str | None

    def __post_init__(self):
        if (self.success != self.language_audit.language_execution_valid
                or self.attempt_count != len(self.language_audit.attempts)
                or self.success != (self.public_text is not None)
                or self.success != (self.failure_reason is None)):
            raise ValueError("Phase-2 commit-ready speech and audit disagree")


def verified_phase2_speech(plan: Phase2SemanticPlanV1, legal: ActionContextV1,
                           public: PublicLanguageContextV1, *, actor, perceiver,
                           ) -> Phase2VerifiedSpeechResultV1:
    """Reuse the frozen verifier and one-repair path, with no fallback."""
    result = realize_verify_action(plan, legal, public, actor=actor, perceiver=perceiver)
    audit = result.audit
    return Phase2VerifiedSpeechResultV1(
        audit.language_execution_valid, result.speech, plan, audit,
        len(audit.attempts), audit.language_invalid_reason)
