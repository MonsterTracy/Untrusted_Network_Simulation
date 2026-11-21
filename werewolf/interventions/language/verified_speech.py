"""生成并校验 Phase-2 发言，供 Canonical Commit 使用。"""

from __future__ import annotations

from dataclasses import dataclass

from werewolf.artifact_io import sha256_bytes
from werewolf.interventions.action_audit import audit_structured_plan
from werewolf.interventions.actions import ActionContextV1, Phase2SemanticPlanV1
from werewolf.interventions.language.semantics import (
    Phase2LanguageError, Phase2SpeechSemanticV1, PublicLanguageContextV1,
    verify_language_execution,
)
from werewolf.interventions.language.audit import (
    LanguageAttemptV1, Phase2LanguageAuditV1, plan_digest, semantic_digest,
)


MAX_REALIZATION_ATTEMPTS = 2


@dataclass(frozen=True)
class Phase2LanguageExecutionV1:
    """Only a verified speech is exposed for a future commit adapter."""

    speech: str | None
    perceived_semantics: Phase2SpeechSemanticV1 | None
    audit: Phase2LanguageAuditV1


def _audit(structured, plan, attempts: tuple[LanguageAttemptV1, ...]) -> Phase2LanguageAuditV1:
    last = attempts[-1] if attempts else None
    return Phase2LanguageAuditV1(
        structured, plan.action.value if hasattr(plan.action, "value") else str(plan.action),
        plan_digest(plan), structured.execution_valid, structured.invalid_reason,
        attempts,
        None if last is None else last.generated_text,
        None if last is None else last.generated_text_digest,
        None if last is None else last.perceived_semantics,
        None if last is None else last.perceived_semantics_digest,
        bool(last is not None and last.language_execution_valid),
        ("STRUCTURED_PLAN_INVALID:" + structured.invalid_reason)
        if not structured.execution_valid else
        (None if last is None else last.language_invalid_reason))


def _check_public_context(legal: ActionContextV1, public: PublicLanguageContextV1) -> None:
    if (not isinstance(legal, ActionContextV1)
            or not isinstance(public, PublicLanguageContextV1)
            or public.phase != legal.phase or public.speaker != legal.acting_wolf
            or public.alive != legal.alive or public.competition != legal.competition
            or public.public_history_digest != legal.public_history_digest):
        raise Phase2LanguageError("public language context differs from legal PRE")


def realize_verify_action(plan: Phase2SemanticPlanV1, legal_context: ActionContextV1,
                          public_context: PublicLanguageContextV1, *, actor, perceiver,
                          ) -> Phase2LanguageExecutionV1:
    """At most one repair. Perceiver sees only text/public context, never plan."""
    _check_public_context(legal_context, public_context)
    if not callable(getattr(actor, "realize", None)) or not callable(getattr(perceiver, "perceive", None)):
        raise TypeError("actor.realize and perceiver.perceive are required")
    structured = audit_structured_plan(legal_context, plan)
    if not structured.execution_valid:
        audit = _audit(structured, plan, ())
        return Phase2LanguageExecutionV1(None, None, audit)

    attempts = []
    failure_reason = None
    for number in range(1, MAX_REALIZATION_ATTEMPTS + 1):
        speech = None
        perceived = None
        reason = None
        try:
            speech = actor.realize(plan, public_context, failure_reason=failure_reason)
            if not isinstance(speech, str) or not speech.strip():
                raise Phase2LanguageError("EMPTY_SPEECH")
            speech = speech.strip()
            perceived = perceiver.perceive(speech, public_context)
            if not isinstance(perceived, Phase2SpeechSemanticV1):
                raise Phase2LanguageError("PERCEPTION_TYPE_INVALID")
            result = verify_language_execution(plan, perceived, legal_context)
            reason = result.invalid_reason
        except Phase2LanguageError as error:
            reason = str(error) or type(error).__name__
        valid = reason is None
        attempts.append(LanguageAttemptV1(
            number, speech,
            None if speech is None else sha256_bytes(speech.encode("utf-8")),
            None if perceived is None else perceived.to_record(),
            None if perceived is None else semantic_digest(perceived),
            valid, reason))
        if valid:
            audit = _audit(structured, plan, tuple(attempts))
            return Phase2LanguageExecutionV1(speech, perceived, audit)
        failure_reason = reason
    audit = _audit(structured, plan, tuple(attempts))
    return Phase2LanguageExecutionV1(None, None, audit)


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
