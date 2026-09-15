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
