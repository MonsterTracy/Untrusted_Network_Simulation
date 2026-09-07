"""Fail-closed primitives for immutable Phase-1 artifacts."""

from werewolf.artifact_io.canonical import (
    ArtifactConflictError,
    ArtifactPublishUnsupportedError,
    ArtifactValidationError,
    ArtifactEnvelope,
    VerifiedArtifact,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    publish_artifact,
    sha256_bytes,
    verify_artifact,
    open_artifact_envelope,
    read_artifact_file,
)
from werewolf.artifact_io.tensor_state import (
    LoadedTensor,
    TensorValue,
    VerifiedTensorState,
    publish_tensor_state,
    verify_tensor_state,
)

__all__ = [
    "ArtifactConflictError",
    "ArtifactPublishUnsupportedError",
    "ArtifactValidationError",
    "ArtifactEnvelope",
    "open_artifact_envelope",
    "read_artifact_file",
    "LoadedTensor",
    "TensorValue",
    "VerifiedArtifact",
    "VerifiedTensorState",
    "canonical_json_bytes",
    "canonical_jsonl_bytes",
    "publish_artifact",
    "publish_tensor_state",
    "sha256_bytes",
    "verify_artifact",
    "verify_tensor_state",
]
