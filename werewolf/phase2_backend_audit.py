"""Phase-2 call sidecar bound to the existing canonical backend call budget.

Only hashes and identities enter this sidecar. CanonicalCallAudit retains its
own private dispatch evidence under RUNTIME; Phase-2 perception must never be
misclassified as a canonical V1 SPEECH_PERCEPTION annotation attempt.
"""

from __future__ import annotations

from dataclasses import dataclass

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection.trajectory_evidence import BackendCallPurpose


CALL_VERSION = "phase2_backend_call_v1"


class Phase2BackendAuditError(ValueError):
    """Call provenance or public-only perception contract is invalid."""


@dataclass(frozen=True)
class Phase2BackendCallV1:
    sequence: int
    role: str
    attempt_index: int
    pilot_id: str | None
    assignment_id: str | None
    game_id: str
    boundary_id: str
    prefix_digest: str
    opportunity_digest: str
    treatment_id: str
    backend_identity: str
    model_identity: str
    request_digest: str
    response_digest: str | None
    canonical_call_id: str | None
    error_category: str | None
    perception_public_only: bool

    def to_record(self) -> dict:
        return {"schema_version": CALL_VERSION, **vars(self)}

    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_record()))


class Phase2BackendCallAuditV1:
    """Shared by actor and perceiver for one frozen treatment."""

    def __init__(self, opportunity, treatment, canonical_audit, *,
                 pilot_id: str | None = None, assignment_id: str | None = None,
                 record_sink=None, sequence_offset: int = 0):
        if (treatment.opportunity_digest != opportunity.digest()
                or treatment.opportunity_identity != opportunity.identity
                or canonical_audit is None
                or not callable(getattr(canonical_audit, "context", None))):
            raise Phase2BackendAuditError("canonical call audit and treatment binding required")
        self.opportunity = opportunity
        self.treatment = treatment
        self.canonical_audit = canonical_audit
        if (pilot_id is None) != (assignment_id is None):
            raise Phase2BackendAuditError("pilot and assignment binding must travel together")
        self.pilot_id = pilot_id
        self.assignment_id = assignment_id
        if record_sink is not None and not callable(record_sink):
            raise Phase2BackendAuditError("backend record sink must be callable")
        self.record_sink = record_sink
        self.records: list[Phase2BackendCallV1] = []
        if type(sequence_offset) is not int or sequence_offset < 0:
            raise Phase2BackendAuditError("invalid assignment call sequence offset")
        self.sequence_offset = sequence_offset
        self._attempt = 0

    def _dispatch(self, *, role, backend, model, messages, temperature,
                  max_tokens, response_format=None, expected_public_prompt=None):
        if role not in ("realization", "repair", "perception"):
            raise Phase2BackendAuditError("unknown Phase-2 call role")
        if (getattr(backend, "_audit", None) is not self.canonical_audit
                or not isinstance(model, str) or not model
                or not isinstance(getattr(backend, "canonical_backend_identity", None), str)):
            raise Phase2BackendAuditError("Phase-2 backend must use this canonical audit")
        if role == "realization":
            self._attempt += 1
            if self._attempt != 1:
                raise Phase2BackendAuditError("first actor call must be realization")
        elif role == "repair":
            self._attempt += 1
            if self._attempt != 2:
                raise Phase2BackendAuditError("at most one repair is permitted")
        elif self._attempt not in (1, 2):
            raise Phase2BackendAuditError("perception requires preceding realization")
        if role == "perception":
            if (not isinstance(expected_public_prompt, str)
                    or messages != [{"role": "user", "content": expected_public_prompt}]):
                raise Phase2BackendAuditError("perception request differs from public-only prompt")
        elif expected_public_prompt is not None:
            raise Phase2BackendAuditError("actor may not claim a perception proof")
        kwargs = {"messages": messages, "model": model,
                  "temperature": temperature, "max_tokens": max_tokens}
        if response_format is not None:
            kwargs["response_format"] = response_format
        request_digest = sha256_bytes(canonical_json_bytes(kwargs))
        context = self.opportunity.legal_context
        sequence = self.sequence_offset + len(self.records) + 1
        operation = f"phase2-{self.treatment.treatment_id[:16]}-{sequence:03d}"
        before = len(self.canonical_audit.records)
        response = None
        category = None
        try:
            with self.canonical_audit.context(
                    purpose=BackendCallPurpose.RUNTIME,
                    operation_id=operation,
                    boundary_id=context.boundary_id,
                    observer_id=context.acting_wolf):
                response = backend.chat_with_metadata(**kwargs)
            return response
        except Exception as error:
            category = type(error).__name__
            raise
        finally:
            calls = self.canonical_audit.records[before:]
            if len(calls) != 1 or calls[0].operation_id != operation:
                raise Phase2BackendAuditError("Phase-2 canonical dispatch evidence missing")
            canonical_call = calls[0]
            if hasattr(canonical_call, "private_payload"):
                payload = canonical_call.private_payload.to_value()
                if (payload.get("method") != "chat_with_metadata"
                        or canonical_json_bytes(payload.get("kwargs")) !=
                           canonical_json_bytes(kwargs)
                        or (response is not None and
                            canonical_json_bytes(payload.get("response")) !=
                            canonical_json_bytes(response))
                        or getattr(canonical_call, "model_identity", model) != model):
                    raise Phase2BackendAuditError(
                        "Phase-2 sidecar differs from canonical backend dispatch")
            call = Phase2BackendCallV1(
                sequence, role, self._attempt, self.pilot_id, self.assignment_id,
                context.game_id,
                context.boundary_id, context.prefix_digest,
                self.opportunity.digest(), self.treatment.treatment_id,
                backend.canonical_backend_identity, model, request_digest,
                None if response is None else sha256_bytes(canonical_json_bytes(response)),
                canonical_call.call_id, category, role == "perception")
            self.records.append(call)
            if self.record_sink is not None:
                self.record_sink(call)

    def actor_call(self, *, backend, model, prompt, failure_reason,
                   temperature, max_tokens):
        role = "realization" if failure_reason is None else "repair"
        return self._dispatch(role=role, backend=backend, model=model,
                              messages=[{"role": "user", "content": prompt}],
                              temperature=temperature, max_tokens=max_tokens)

    def perception_call(self, *, backend, model, prompt, text, public_context,
                        temperature, max_tokens, response_format):
        # Reconstruct solely from generated text and trusted public context.
        from werewolf.phase2_language import build_perception_prompt
        expected = build_perception_prompt(text, public_context)
        return self._dispatch(role="perception", backend=backend, model=model,
                              messages=[{"role": "user", "content": prompt}],
                              temperature=temperature, max_tokens=max_tokens,
                              response_format=response_format,
                              expected_public_prompt=expected)

    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes([record.to_record()
                                                  for record in self.records]))
