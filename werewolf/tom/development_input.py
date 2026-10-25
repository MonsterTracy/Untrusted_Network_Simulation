"""One versioned Development reader for preparation, workers and Baselines.

Initial V2 preparation requires both independent custody digests. Later workers
use their immutable parent's Publication digest, which transitively pins schema,
Split digest, public projections and the Development-only role sidecar.
V1 open_publication retains its original meaning and signature.
"""

from pathlib import Path

from werewolf.artifact_io import ArtifactValidationError, open_artifact_envelope
from werewolf.development_publication import (
    _load_canonical_json, _require_digest, open_publication,
    DEVELOPMENT_PUBLICATION_ARTIFACT_TYPE, DEVELOPMENT_PUBLICATION_SCHEMA_VERSION,
)
from werewolf.development_publication_v2 import (
    PUBLICATION_TYPE, PUBLICATION_VERSION, open_development_publication_v2,
)


def open_development_input(path, *, expected_publication_digest=None, expected_split_digest=None, game_ids=None):
    """Fail before game/role reads on unknown schema, missing anchors or drift."""
    root = Path(path)
    if ".." in root.parts or str(root) == ".":
        raise ArtifactValidationError("invalid Development input path")
    # Reject linked ancestors as well as links inside the artifact inventory.
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ArtifactValidationError("Development input cannot traverse symbolic links")
    if (root / "manifest.json").is_symlink():
        raise ArtifactValidationError("Development manifest cannot be a symbolic link")
    m = _load_canonical_json(root / "manifest.json")
    if not isinstance(m, dict):
        raise ArtifactValidationError("Development manifest must be an object")
    kind, version = m.get("artifact_type"), m.get("schema_version")
    if (kind, version) == (DEVELOPMENT_PUBLICATION_ARTIFACT_TYPE, DEVELOPMENT_PUBLICATION_SCHEMA_VERSION):
        if expected_split_digest is not None:
            raise ArtifactValidationError("V1 Development has no Split binding")
    elif (kind, version) == (PUBLICATION_TYPE, PUBLICATION_VERSION):
        if expected_publication_digest is None:
            raise ArtifactValidationError("V2 training input requires a trusted Publication digest")
    else:
        raise ArtifactValidationError("unsupported Development input schema; Test is not a training input")
    if expected_publication_digest is not None:
        _require_digest(expected_publication_digest, "expected_publication_digest")
        envelope = open_artifact_envelope(root, expected_artifact_type=kind, expected_schema_version=version)
        if envelope.manifest_digest != expected_publication_digest:
            raise ArtifactValidationError("Development input differs from trusted Publication digest")
        m = envelope.manifest
    if kind == DEVELOPMENT_PUBLICATION_ARTIFACT_TYPE:
        return open_publication(root, game_ids=game_ids)
    split_digest = m["split_manifest_digest"]
    if expected_split_digest is not None:
        _require_digest(expected_split_digest, "expected_split_digest")
        if expected_split_digest != split_digest:
            raise ArtifactValidationError("Development input differs from trusted Split digest")
    return open_development_publication_v2(root, expected_split_digest=split_digest, game_ids=game_ids)
