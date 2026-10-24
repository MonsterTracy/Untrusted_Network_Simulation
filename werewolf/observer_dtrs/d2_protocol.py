"""Versioned D2 execution contract; historical proposal remains unchanged."""
from dataclasses import dataclass
import json
from pathlib import Path

from werewolf.artifact_io import (
    canonical_json_bytes, publish_artifact, read_artifact_file, sha256_bytes, verify_artifact,
)

VERSION = "observer_dtrs_d2_execution_protocol_v1"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/observer_dtrs/d2-execution-v1.json"
# Pin the reviewed configuration, not a mutable path or an uncommitted Git HEAD.
CONFIG_DIGEST = "c76940b446ffaa3dc9e873620fe775a3b64db8544d0c5190a49778a2e48a6c20"


@dataclass(frozen=True)
class FrozenProtocol:
    payload: bytes

    def __post_init__(self):
        record = json.loads(self.payload)
        if (canonical_json_bytes(record) != self.payload
                or sha256_bytes(self.payload) != CONFIG_DIGEST
                or record["status"] != "FROZEN_NOT_RUN"):
            raise ValueError("unsupported or changed frozen D2 execution protocol")

    @property
    def record(self):
        return json.loads(self.payload)

    @property
    def digest(self):
        return sha256_bytes(self.payload)


def frozen_protocol(path=CONFIG_PATH):
    return FrozenProtocol(canonical_json_bytes(json.loads(Path(path).read_bytes())))


def implementation_sources():
    root = Path(__file__).resolve().parents[2]
    paths = ["werewolf/observer_dtrs/" + name for name in
             ("d2_protocol.py", "splits.py", "estimation.py", "estimation_offline.py", "scoring.py")]
    paths += ["scripts/manage_observer_dtrs_d2.py", "scripts/check_observer_dtrs_estimator.py", "configs/observer_dtrs/d2-execution-v1.json",
              "tests/observer_dtrs/test_estimation.py", "tests/observer_dtrs/test_d2_splits.py"]
    return {"base_revision": "d8397ce56c65d6d46b24a6b53f0b71431c04267b",
            "status": "uncommitted_sources_not_a_commit",
            "files": {p: (root / p).read_text() for p in paths}}


def publish_protocol(destination, protocol, *, sources=None):
    sources = implementation_sources() if sources is None else sources
    return publish_artifact(destination, manifest_fields={
        "artifact_type": "observer_dtrs_d2_protocol", "schema_version": VERSION,
        "protocol_digest": protocol.digest,
        "implementation_digest": sha256_bytes(canonical_json_bytes(sources))},
        files={"protocol.json": protocol.payload, "implementation_sources.json": canonical_json_bytes(sources)})


def read_protocol(path, *, expected_manifest_digest):
    artifact = verify_artifact(path, expected_artifact_type="observer_dtrs_d2_protocol",
                               expected_schema_version=VERSION)
    if (artifact.manifest_digest != expected_manifest_digest
            or set(artifact.manifest["file_table"]) != {"protocol.json", "implementation_sources.json"}):
        raise ValueError("D2 protocol artifact identity mismatch")
    protocol = FrozenProtocol(read_artifact_file(artifact, "protocol.json"))
    if artifact.manifest["protocol_digest"] != protocol.digest:
        raise ValueError("D2 protocol payload identity mismatch")
    if sha256_bytes(read_artifact_file(artifact, "implementation_sources.json")) != artifact.manifest["implementation_digest"]:
        raise ValueError("D2 implementation snapshot identity mismatch")
    return protocol
