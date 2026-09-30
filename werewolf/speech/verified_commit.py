"""Intervention commit binding; not a new canonical annotation schema."""
from dataclasses import asdict, dataclass
import json

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection.speech import V1SpeechAction
from werewolf.speech.speech_perceiver import SpeechParseAuditResult


def perception_identity(perceiver):
    return (
        getattr(perceiver, "backend_id", None)
        or getattr(getattr(perceiver, "backend", None), "canonical_backend_identity", None)
        or type(perceiver).__name__,
        getattr(perceiver, "model_name", None) or type(perceiver).__name__,
    )


@dataclass(frozen=True)
class VerifiedSpeechCommit:
    speech: str
    expected: V1SpeechAction
    day: int
    phase: str
    public_history_digest: str
    backend_id: str
    model_id: str
    perception: SpeechParseAuditResult
    snapshot: bytes
    digest: str

    def _record(self):
        return {"speech": self.speech, "expected": self.expected.to_record(),
                "day": self.day, "phase": self.phase,
                "public_history_digest": self.public_history_digest,
                "backend_id": self.backend_id, "model_id": self.model_id,
                "perception": asdict(self.perception)}

    def validated_perception(self):
        if (canonical_json_bytes(self._record()) != self.snapshot
                or sha256_bytes(self.snapshot) != self.digest):
            raise ValueError("verified speech commit binding mismatch")
        # Materialize an isolated copy of exactly the verified audit content.
        record = json.loads(self.snapshot)["perception"]
        record["generation_attempts"] = tuple(record["generation_attempts"])
        audit = SpeechParseAuditResult(**record)
        _validate_success(self.speech, self.expected, audit)
        return audit


def _validate_success(speech, expected, audit):
    if not isinstance(speech, str) or not speech.strip():
        raise ValueError("verified speech must be nonempty")
    if not isinstance(expected, V1SpeechAction) or not isinstance(audit, SpeechParseAuditResult):
        raise TypeError("expected canonical action and real perception audit")
    if (audit.parse_status != "ok" or audit.error_type is not None
            or audit.error_message is not None
            or audit.normalized_actions != [expected.to_record()]
            or not audit.generation_attempts):
        raise ValueError("verified speech requires exact successful perception")


def bind_verified_speech(*, speech, expected, perception, day, phase,
                         public_history_digest, perceiver):
    _validate_success(speech, expected, perception)
    if type(day) is not int or day < 0 or phase not in ("speech", "speech_pk"):
        raise ValueError("invalid verified speech opportunity")
    backend_id, model_id = perception_identity(perceiver)
    provisional = VerifiedSpeechCommit(speech, expected, day, phase,
        public_history_digest, backend_id, model_id, perception, b"", "")
    snapshot = canonical_json_bytes(provisional._record())
    return VerifiedSpeechCommit(speech, expected, day, phase, public_history_digest,
        backend_id, model_id, perception, snapshot, sha256_bytes(snapshot))


@dataclass(frozen=True)
class AuditedPublicSpeechCommit:
    """Same canonical parser proof, allowing its genuine 0..N V1 actions.

    Phase-2 does not map its three action types into V1. The independently
    produced canonical audit is retained unchanged in the V1 annotation.
    """

    speech: str
    speaker: str
    day: int
    phase: str
    public_history_digest: str
    backend_id: str
    model_id: str
    perception: SpeechParseAuditResult
    snapshot: bytes
    digest: str

    def _record(self):
        return {"speech": self.speech, "speaker": self.speaker,
                "day": self.day, "phase": self.phase,
                "public_history_digest": self.public_history_digest,
                "backend_id": self.backend_id, "model_id": self.model_id,
                "perception": asdict(self.perception)}

    def validated_perception(self):
        if (canonical_json_bytes(self._record()) != self.snapshot
                or sha256_bytes(self.snapshot) != self.digest):
            raise ValueError("audited public speech binding mismatch")
        record = json.loads(self.snapshot)["perception"]
        record["generation_attempts"] = tuple(record["generation_attempts"])
        audit = SpeechParseAuditResult(**record)
        _validate_audited_public_speech(self.speech, self.speaker, audit)
        return audit


def _validate_audited_public_speech(speech, speaker, audit):
    if (not isinstance(speech, str) or not speech.strip()
            or not isinstance(speaker, str) or speaker not in
            {f"player{i}" for i in range(1, 8)}
            or not isinstance(audit, SpeechParseAuditResult)
            or audit.parse_status != "ok" or audit.error_type is not None
            or audit.error_message is not None or not audit.generation_attempts
            or not isinstance(audit.normalized_actions, list)):
        raise ValueError("canonical public speech perception is incomplete")
    for action in audit.normalized_actions:
        if (not isinstance(action, list) or len(action) != 3
                or V1SpeechAction(*action).subject != speaker):
            raise ValueError("canonical public speech action has wrong speaker")


def bind_audited_public_speech(*, speech, speaker, perception, day, phase,
                               public_history_digest, perceiver):
    """Bind exactly one existing V1 parser response to the original text."""
    _validate_audited_public_speech(speech, speaker, perception)
    if (type(day) is not int or day < 0 or phase not in ("speech", "speech_pk")
            or not isinstance(public_history_digest, str) or not public_history_digest):
        raise ValueError("invalid audited public speech opportunity")
    backend_id, model_id = perception_identity(perceiver)
    provisional = AuditedPublicSpeechCommit(
        speech, speaker, day, phase, public_history_digest,
        backend_id, model_id, perception, b"", "")
    snapshot = canonical_json_bytes(provisional._record())
    return AuditedPublicSpeechCommit(
        speech, speaker, day, phase, public_history_digest,
        backend_id, model_id, perception, snapshot, sha256_bytes(snapshot))
