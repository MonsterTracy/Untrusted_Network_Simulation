"""Fail-closed backend call budget and immutable call-evidence capture."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from werewolf.canonical_collection.attempt_ledger import CollectionPlan
from werewolf.canonical_collection.trajectory_evidence import (
    BackendCallEvidence,
    BackendCallPurpose,
    BackendCallStatus,
    CallBudgetSummary,
    construct_backend_call_evidence,
    construct_call_budget_summary,
)
from werewolf.speech.validation import normalize_player


class CollectionCallBudgetExceeded(RuntimeError):
    """The predeclared per-game backend-call budget was exhausted."""


@dataclass(frozen=True)
class _CallContext:
    purpose: BackendCallPurpose
    operation_id: str
    boundary_id: str | None
    observer_id: str | None
    isolate_dispatches: bool


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise TypeError(
        "backend call evidence supports only canonical JSON request/response values"
    )


class CanonicalCallAudit:
    """Own every production backend call record for one claimed game."""

    def __init__(self, *, plan: CollectionPlan, configured_call_limit: int):
        if isinstance(configured_call_limit, bool) or not isinstance(
            configured_call_limit,
            int,
        ) or configured_call_limit <= 0:
            raise ValueError("configured_call_limit must be a positive integer")
        self.plan = plan
        self.configured_call_limit = configured_call_limit
        self._active: _CallContext | None = None
        self._operation_attempts: Counter[str] = Counter()
        self._dispatch_sequence = 0
        self._records: list[BackendCallEvidence] = []
        self._runtime_action_sequence = 0

    @property
    def records(self) -> tuple[BackendCallEvidence, ...]:
        return tuple(self._records)

    @contextmanager
    def context(
        self,
        *,
        purpose: BackendCallPurpose,
        operation_id: str,
        boundary_id: str | None,
        observer_id: str | None,
        isolate_dispatches: bool = False,
    ):
        if self._active is not None:
            raise RuntimeError("backend call contexts cannot be nested")
        self._active = _CallContext(
            purpose=purpose,
            operation_id=operation_id,
            boundary_id=boundary_id,
            observer_id=observer_id,
            isolate_dispatches=isolate_dispatches,
        )
        try:
            yield
        finally:
            self._active = None

    def belief_context(self, report_id: str):
        observer_id = report_id.rsplit("-belief-", 1)[-1]
        boundary_id = report_id.rsplit("-belief-", 1)[0]
        return self.context(
            purpose=BackendCallPurpose.BELIEF_OBSERVATION,
            operation_id=report_id,
            boundary_id=boundary_id,
            observer_id=observer_id,
        )

    def action_context(
        self,
        *,
        acting_player_id: int,
        boundary_id: str | None,
        is_public_speech: bool,
    ):
        self._runtime_action_sequence += 1
        purpose = (
            BackendCallPurpose.SPEECH_GENERATION
            if is_public_speech
            else BackendCallPurpose.GAMEPLAY_ACTION
        )
        return self.context(
            purpose=purpose,
            operation_id=(
                f"runtime-action-{self._runtime_action_sequence:06d}"
            ),
            boundary_id=boundary_id,
            observer_id=normalize_player(acting_player_id),
            isolate_dispatches=True,
        )

    def speech_perception_context(
        self,
        *,
        event_id: str,
        boundary_id: str,
        speaker_id: int,
    ):
        return self.context(
            purpose=BackendCallPurpose.SPEECH_PERCEPTION,
            operation_id=event_id,
            boundary_id=boundary_id,
            observer_id=normalize_player(speaker_id),
        )

    def prepare_report(
        self,
        *,
        observation_id: str,
        report_prompt: str,
        **_context,
    ) -> tuple[str, str]:
        return observation_id, report_prompt

    def complete_report(self, _report_id: str, _raw_response: Any) -> None:
        return None

    def record_agent_state(self, **_state) -> None:
        return None

    def record_label_generation_attempt(
        self,
        *,
        report_id: str,
        generation_attempt: int,
        status: str,
        error: str | None,
        **_event,
    ) -> None:
        call_id = f"{report_id}-attempt-{generation_attempt:02d}"
        self.mark_semantic_attempt(
            call_id,
            success=status == "ok",
            error_category=None if status == "ok" else status,
            error_message=error,
        )

    def mark_semantic_attempt(
        self,
        call_id: str,
        *,
        success: bool,
        error_category: str | None,
        error_message: str | None,
    ) -> None:
        for index, record in enumerate(self._records):
            if record.call_id != call_id:
                continue
            private_payload = record.private_payload.to_value()
            private_payload["semantic_status"] = "success" if success else "error"
            private_payload["semantic_error_category"] = error_category
            private_payload["semantic_error_message"] = error_message
            self._records[index] = construct_backend_call_evidence(
                call_id=record.call_id,
                operation_id=record.operation_id,
                purpose=record.purpose,
                status=(
                    BackendCallStatus.SUCCESS
                    if success
                    else BackendCallStatus.ERROR
                ),
                boundary_id=record.boundary_id,
                observer_id=record.observer_id,
                attempt_index=record.attempt_index,
                backend_identity=record.backend_identity,
                model_identity=record.model_identity,
                parser_identity=record.parser_identity,
                prompt_identity=record.prompt_identity,
                retry_policy_identity=record.retry_policy_identity,
                call_budget_identity=record.call_budget_identity,
                private_payload=private_payload,
            )
            return
        raise ValueError(f"semantic attempt has no backend call evidence: {call_id}")

    def dispatch(self, method_name: str, call, args, kwargs):
        if self._active is None:
            raise RuntimeError("backend call occurred outside canonical call context")
        if len(self._records) >= self.configured_call_limit:
            raise CollectionCallBudgetExceeded(
                f"configured call limit exhausted: {self.configured_call_limit}"
            )
        context = self._active
        operation_id = context.operation_id
        if context.isolate_dispatches:
            self._dispatch_sequence += 1
            operation_id = f"{operation_id}-dispatch-{self._dispatch_sequence:06d}"
        self._operation_attempts[operation_id] += 1
        attempt_index = self._operation_attempts[operation_id]
        call_id = f"{operation_id}-attempt-{attempt_index:02d}"
        payload = {
            "method": method_name,
            "args": _json_value(args),
            "kwargs": _json_value(kwargs),
        }
        try:
            response = call(*args, **kwargs)
        except Exception as error:
            payload.update(
                {
                    "response": None,
                    "error_category": type(error).__name__,
                    "error_message": str(error) or type(error).__name__,
                }
            )
            status = BackendCallStatus.ERROR
            raised = error
        else:
            payload.update(
                {
                    "response": _json_value(response),
                    "error_category": None,
                    "error_message": None,
                }
            )
            status = BackendCallStatus.SUCCESS
            raised = None
        self._records.append(
            construct_backend_call_evidence(
                call_id=call_id,
                operation_id=operation_id,
                purpose=context.purpose,
                status=status,
                boundary_id=context.boundary_id,
                observer_id=context.observer_id,
                attempt_index=attempt_index,
                backend_identity=self.plan.backend_identity,
                model_identity=self.plan.model_identity,
                parser_identity=self.plan.parser_identity,
                prompt_identity=self.plan.prompt_identity,
                retry_policy_identity=self.plan.retry_policy_identity,
                call_budget_identity=self.plan.call_budget_identity,
                private_payload=payload,
            )
        )
        if raised is not None:
            raise raised
        return response

    def summary(self) -> CallBudgetSummary:
        counts = Counter(record.operation_id for record in self._records)
        return construct_call_budget_summary(
            configured_call_limit=self.configured_call_limit,
            used_calls=len(self._records),
            retry_calls=sum(count - 1 for count in counts.values()),
            fallback_action_count=0,
            second_speaker_belief_count=0,
            opaque_call_digests=tuple(record.call_digest for record in self._records),
        )


class AuditedBackend:
    def __init__(self, backend: Any, audit: CanonicalCallAudit):
        self._backend = backend
        self._audit = audit
        self.canonical_backend_identity = audit.plan.backend_identity

    def __getattr__(self, name: str):
        return getattr(self._backend, name)

    def mark_semantic_attempt(self, *args, **kwargs):
        return self._audit.mark_semantic_attempt(*args, **kwargs)

    def chat(self, *args, **kwargs):
        return self._audit.dispatch(
            "chat",
            self._backend.chat,
            args,
            kwargs,
        )

    def chat_with_metadata(self, *args, **kwargs):
        return self._audit.dispatch(
            "chat_with_metadata",
            self._backend.chat_with_metadata,
            args,
            kwargs,
        )


def audited_backends(
    backends: Mapping[str, Any],
    audit: CanonicalCallAudit,
) -> dict[str, AuditedBackend]:
    return {
        name: AuditedBackend(backend, audit)
        for name, backend in backends.items()
    }


__all__ = [
    "AuditedBackend",
    "CanonicalCallAudit",
    "CollectionCallBudgetExceeded",
    "audited_backends",
]
