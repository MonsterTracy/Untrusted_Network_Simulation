"""Pilot execution envelope; no executor or synthetic success path."""

from __future__ import annotations

from dataclasses import dataclass

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes


VERSION = "phase2_pilot_execution_v1"


def _digest(value: str | None) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value)


def _source_commit(value: str) -> bool:
    return isinstance(value, str) and len(value) in (40, 64) and all(
        character in "0123456789abcdef" for character in value)


@dataclass(frozen=True)
class Phase2PilotExecutionEnvelopeV1:
    source_commit: str
    checkpoint_schema: str
    checkpoint_digest: str
    opportunity_digest: str
    treatment_digest: str
    branch_id: str
    paired_seed: int
    action: str
    execution_status: str
    abort_reason: str | None
    language_audit_digest: str | None = None
    canonical_event_digest: str | None = None
    execution_proof_digest: str | None = None
    terminal_consequence_digest: str | None = None
    probe_sequence_digest: str | None = None

    def __post_init__(self):
        if (not _source_commit(self.source_commit)
                or any(not _digest(value) for value in (
                    self.checkpoint_digest, self.opportunity_digest,
                    self.treatment_digest, self.branch_id))
                or not isinstance(self.checkpoint_schema, str)
                or not self.checkpoint_schema
                or type(self.paired_seed) is not int or self.paired_seed < 0
                or self.action not in ("PUSH", "REDIRECT", "PROBE")
                or self.execution_status not in ("ABORTED", "COMPLETED")
                or any(value is not None and not _digest(value) for value in (
                    self.language_audit_digest, self.canonical_event_digest,
                    self.execution_proof_digest, self.terminal_consequence_digest,
                    self.probe_sequence_digest))):
            raise ValueError("invalid pilot execution lineage")
        if self.execution_status == "ABORTED":
            if (not self.abort_reason or self.terminal_consequence_digest is not None
                    or self.probe_sequence_digest is not None):
                raise ValueError("aborted branch cannot publish an outcome")
        elif (self.abort_reason is not None
              or self.checkpoint_schema == "phase2_checkpoint_closure_v1"
              or self.execution_proof_digest is None
              or self.language_audit_digest is None
              or self.canonical_event_digest is None
              or (self.action == "PROBE") != (self.probe_sequence_digest is not None)
              or (self.action != "PROBE") != (self.terminal_consequence_digest is not None)):
            raise ValueError("completed branch requires verified speech and action-specific result")

    def to_record(self) -> dict:
        return {"schema_version": VERSION, "source_commit": self.source_commit,
                "checkpoint_schema": self.checkpoint_schema,
                "checkpoint_digest": self.checkpoint_digest,
                "opportunity_digest": self.opportunity_digest,
                "treatment_digest": self.treatment_digest,
                "branch_id": self.branch_id, "paired_seed": self.paired_seed,
                "action": self.action, "language_audit_digest": self.language_audit_digest,
                "canonical_event_digest": self.canonical_event_digest,
                "execution_proof_digest": self.execution_proof_digest,
                "execution_status": self.execution_status,
                "abort_reason": self.abort_reason,
                "terminal_consequence_digest": self.terminal_consequence_digest,
                "probe_sequence_digest": self.probe_sequence_digest}

    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_record()))
