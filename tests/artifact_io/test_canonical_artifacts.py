import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from werewolf.artifact_io import (
    ArtifactConflictError,
    ArtifactValidationError,
    TensorValue,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    publish_artifact,
    publish_tensor_state,
    verify_artifact,
    verify_tensor_state,
)


ARTIFACT_TYPE = "artifact_fixture"
SCHEMA_VERSION = "artifact_fixture_v1"


def _manifest_fields(*, identity: str = "fixture-001") -> dict[str, object]:
    return {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "artifact_identity": identity,
        "implementation_version": "test_v1",
    }


def _verify_fixture(path: Path):
    return verify_artifact(
        path,
        expected_artifact_type=ARTIFACT_TYPE,
        expected_schema_version=SCHEMA_VERSION,
    )


def _rewrite_manifest_with_current_digest(manifest_path: Path, manifest):
    manifest_without_digest = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_digest"
    }
    manifest["manifest_digest"] = hashlib.sha256(
        canonical_json_bytes(manifest_without_digest)
    ).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(manifest))


def test_canonical_json_and_jsonl_are_byte_stable_without_string_rewriting():
    decomposed = "e\u0301"
    value = {"z": [2, 1], "a": decomposed, "unicode": "狼人"}

    assert canonical_json_bytes(value) == (
        '{"a":"e\u0301","unicode":"狼人","z":[2,1]}'.encode("utf-8")
    )
    assert canonical_jsonl_bytes([value, {"b": True}]) == (
        canonical_json_bytes(value) + b'\n{"b":true}\n'
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_canonical_json_rejects_non_finite_numbers(value):
    with pytest.raises(ValueError, match="finite"):
        canonical_json_bytes({"value": value})


def test_artifact_round_trip_has_exact_file_table_and_stable_manifest(tmp_path):
    destination = tmp_path / "fixture"
    published = publish_artifact(
        destination,
        manifest_fields=_manifest_fields(),
        files={
            "public/events.jsonl": canonical_jsonl_bytes([{"event": 1}]),
            "audit/summary.json": canonical_json_bytes({"ok": True}),
        },
    )

    verified = _verify_fixture(destination)

    assert verified.manifest_digest == published.manifest_digest
    assert list(verified.manifest["file_table"]) == [
        "audit/summary.json",
        "public/events.jsonl",
    ]
    assert (destination / "manifest.json").read_bytes() == canonical_json_bytes(
        verified.manifest
    )


@pytest.mark.parametrize(
    ("expected_artifact_type", "expected_schema_version"),
    [
        ("other_artifact", SCHEMA_VERSION),
        (ARTIFACT_TYPE, "artifact_fixture_v2"),
    ],
)
def test_artifact_validation_rejects_unsupported_identity_without_fallback(
    tmp_path,
    expected_artifact_type,
    expected_schema_version,
):
    destination = tmp_path / "unsupported"
    publish_artifact(
        destination,
        manifest_fields=_manifest_fields(),
        files={"payload.bin": b"canonical-payload"},
    )

    with pytest.raises(ArtifactValidationError, match="unsupported"):
        verify_artifact(
            destination,
            expected_artifact_type=expected_artifact_type,
            expected_schema_version=expected_schema_version,
        )


@pytest.mark.parametrize("mutation", ["missing", "extra", "renamed", "digest"])
def test_artifact_validation_rejects_every_file_table_mismatch(tmp_path, mutation):
    destination = tmp_path / mutation
    publish_artifact(
        destination,
        manifest_fields=_manifest_fields(identity=mutation),
        files={"payload.bin": b"canonical-payload"},
    )
    payload = destination / "payload.bin"

    if mutation == "missing":
        payload.unlink()
    elif mutation == "extra":
        (destination / "extra.bin").write_bytes(b"extra")
    elif mutation == "renamed":
        payload.rename(destination / "renamed.bin")
    else:
        payload.write_bytes(b"canonical-payloae")

    with pytest.raises(ArtifactValidationError):
        _verify_fixture(destination)


def test_artifact_publication_reuses_identical_identity_and_never_overwrites(tmp_path):
    destination = tmp_path / "immutable"
    first = publish_artifact(
        destination,
        manifest_fields=_manifest_fields(),
        files={"payload.bin": b"first"},
    )
    before = {path.name: path.read_bytes() for path in destination.iterdir()}

    reused = publish_artifact(
        destination,
        manifest_fields=_manifest_fields(),
        files={"payload.bin": b"first"},
    )
    assert reused.manifest_digest == first.manifest_digest

    with pytest.raises(ArtifactConflictError):
        publish_artifact(
            destination,
            manifest_fields=_manifest_fields(),
            files={"payload.bin": b"second"},
        )

    assert {path.name: path.read_bytes() for path in destination.iterdir()} == before
    assert _verify_fixture(destination).manifest_digest == first.manifest_digest


def test_every_single_byte_artifact_mutation_fails_validation(tmp_path):
    destination = tmp_path / "mutation"
    publish_artifact(
        destination,
        manifest_fields=_manifest_fields(),
        files={"payload.bin": b"abc"},
    )

    for path in (destination / "manifest.json", destination / "payload.bin"):
        original = path.read_bytes()
        for offset in range(len(original)):
            mutated = bytearray(original)
            mutated[offset] ^= 1
            path.write_bytes(mutated)
            with pytest.raises(ArtifactValidationError):
                _verify_fixture(destination)
        path.write_bytes(original)


def test_tensor_state_round_trip_is_lexicographic_c_order_and_little_endian(tmp_path):
    destination = tmp_path / "tensor-state"
    z_weight = np.arange(6, dtype="<f4").reshape(2, 3)[:, ::-1]
    a_counter = np.array([3, 5], dtype="<i8")

    publish_tensor_state(
        destination,
        manifest_fields=_manifest_fields(identity="tensor-state-001"),
        tensors={
            "z.weight": TensorValue(z_weight, trainable=True),
            "a.counter": TensorValue(a_counter, trainable=False),
        },
    )
    verified = verify_tensor_state(
        destination,
        expected_artifact_type=ARTIFACT_TYPE,
        expected_schema_version=SCHEMA_VERSION,
    )

    tensor_manifest = verified.artifact.manifest["tensor_container"]
    assert [entry["name"] for entry in tensor_manifest["tensors"]] == [
        "a.counter",
        "z.weight",
    ]
    assert [entry["dtype"] for entry in tensor_manifest["tensors"]] == [
        "<i8",
        "<f4",
    ]
    assert tensor_manifest["tensors"][1]["order"] == "C"
    assert verified.tensors["z.weight"].array.flags.c_contiguous
    np.testing.assert_array_equal(verified.tensors["z.weight"].array, z_weight)
    np.testing.assert_array_equal(verified.tensors["a.counter"].array, a_counter)


@pytest.mark.parametrize(
    "array",
    [
        np.array([1.0], dtype="<f8"),
        np.array([1.0], dtype=">f4"),
    ],
)
def test_tensor_state_rejects_noncanonical_trainable_dtype(tmp_path, array):
    with pytest.raises(ValueError, match="trainable tensor"):
        publish_tensor_state(
            tmp_path / "bad-dtype",
            manifest_fields=_manifest_fields(),
            tensors={"weight": TensorValue(array, trainable=True)},
        )


def test_tensor_state_rejects_payload_digest_corruption(tmp_path):
    destination = tmp_path / "corrupt-tensor"
    publish_tensor_state(
        destination,
        manifest_fields=_manifest_fields(),
        tensors={
            "weight": TensorValue(
                np.array([[1.0, 2.0]], dtype="<f4"),
                trainable=True,
            )
        },
    )
    payload = destination / "tensors.bin"
    corrupted = bytearray(payload.read_bytes())
    corrupted[-1] ^= 1
    payload.write_bytes(corrupted)

    with pytest.raises(ArtifactValidationError):
        verify_tensor_state(
            destination,
            expected_artifact_type=ARTIFACT_TYPE,
            expected_schema_version=SCHEMA_VERSION,
        )


def test_tensor_state_rejects_nonlexicographic_manifest_even_if_redigested(tmp_path):
    destination = tmp_path / "bad-order"
    publish_tensor_state(
        destination,
        manifest_fields=_manifest_fields(),
        tensors={
            "a": TensorValue(np.array([1.0], dtype="<f4"), trainable=True),
            "b": TensorValue(np.array([2.0], dtype="<f4"), trainable=True),
        },
    )

    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tensor_container"]["tensors"].reverse()
    _rewrite_manifest_with_current_digest(manifest_path, manifest)

    with pytest.raises(ArtifactValidationError, match="lexicographic"):
        verify_tensor_state(
            destination,
            expected_artifact_type=ARTIFACT_TYPE,
            expected_schema_version=SCHEMA_VERSION,
        )


def test_tensor_state_rejects_per_tensor_digest_corruption_after_outer_redigest(
    tmp_path,
):
    destination = tmp_path / "bad-tensor-digest"
    publish_tensor_state(
        destination,
        manifest_fields=_manifest_fields(),
        tensors={
            "weight": TensorValue(
                np.array([1.0], dtype="<f4"),
                trainable=True,
            )
        },
    )

    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tensor_container"]["tensors"][0]["sha256"] = "0" * 64
    _rewrite_manifest_with_current_digest(manifest_path, manifest)

    with pytest.raises(ArtifactValidationError, match="tensor digest mismatch"):
        verify_tensor_state(
            destination,
            expected_artifact_type=ARTIFACT_TYPE,
            expected_schema_version=SCHEMA_VERSION,
        )
