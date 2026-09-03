"""Canonical raw tensor-container publication and validation."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from werewolf.artifact_io.canonical import (
    ArtifactValidationError,
    VerifiedArtifact,
    publish_artifact,
    sha256_bytes,
    verify_artifact,
)


_TENSOR_CONTAINER_VERSION = "raw_tensor_container_v1"
_TENSOR_PAYLOAD_PATH = "tensors.bin"
_CONTAINER_FIELDS = frozenset(
    {
        "format_version",
        "payload_path",
        "payload_byte_length",
        "payload_sha256",
        "tensors",
    }
)
_TENSOR_FIELDS = frozenset(
    {
        "name",
        "shape",
        "dtype",
        "order",
        "byte_offset",
        "byte_length",
        "trainable",
        "sha256",
    }
)


@dataclass(frozen=True)
class TensorValue:
    """One tensor and whether it belongs to the trainable parameter graph."""

    array: np.ndarray
    trainable: bool


@dataclass(frozen=True)
class LoadedTensor:
    """One validated tensor view loaded from a canonical payload."""

    array: np.ndarray
    trainable: bool
    sha256: str


@dataclass(frozen=True)
class VerifiedTensorState:
    """A verified artifact plus its read-only named tensor views."""

    artifact: VerifiedArtifact
    tensors: Mapping[str, LoadedTensor]


def _canonical_dtype(dtype: np.dtype[Any]) -> str:
    if dtype.fields is not None or dtype.subdtype is not None or dtype.hasobject:
        raise ValueError(f"unsupported tensor dtype: {dtype}")
    if dtype.kind not in "biufc":
        raise ValueError(f"unsupported tensor dtype: {dtype}")
    if dtype.byteorder == ">" or (
        dtype.byteorder == "=" and sys.byteorder != "little"
    ):
        raise ValueError(f"tensor dtype must be little-endian: {dtype}")
    if dtype.byteorder == "|":
        return dtype.str
    return dtype.newbyteorder("<").str


def _prepare_tensor(name: str, value: TensorValue) -> np.ndarray:
    if not isinstance(name, str) or not name:
        raise ValueError("tensor names must be non-empty strings")
    if not isinstance(value, TensorValue):
        raise TypeError(f"tensor {name!r} must be a TensorValue")
    if not isinstance(value.trainable, bool):
        raise TypeError(f"tensor {name!r} trainable flag must be bool")

    array = np.asarray(value.array)
    try:
        dtype = _canonical_dtype(array.dtype)
    except ValueError as error:
        if value.trainable:
            raise ValueError(
                f"trainable tensor {name!r} must use little-endian float32"
            ) from error
        raise
    if value.trainable and dtype != "<f4":
        raise ValueError(
            f"trainable tensor {name!r} must use little-endian float32"
        )
    return np.ascontiguousarray(array)


def publish_tensor_state(
    destination: Path | str,
    *,
    manifest_fields: Mapping[str, Any],
    tensors: Mapping[str, TensorValue],
    manifest_name: str = "manifest.json",
) -> VerifiedTensorState:
    """Publish tensors as one lexicographically ordered raw byte payload."""

    if "tensor_container" in manifest_fields:
        raise ValueError("tensor_container is owned by the tensor-state writer")
    if not tensors:
        raise ValueError("tensor state must contain at least one tensor")

    entries: list[dict[str, Any]] = []
    payload_parts: list[bytes] = []
    byte_offset = 0
    for name in sorted(tensors):
        value = tensors[name]
        array = _prepare_tensor(name, value)
        tensor_bytes = array.tobytes(order="C")
        entries.append(
            {
                "name": name,
                "shape": list(array.shape),
                "dtype": _canonical_dtype(array.dtype),
                "order": "C",
                "byte_offset": byte_offset,
                "byte_length": len(tensor_bytes),
                "trainable": value.trainable,
                "sha256": sha256_bytes(tensor_bytes),
            }
        )
        payload_parts.append(tensor_bytes)
        byte_offset += len(tensor_bytes)

    payload = b"".join(payload_parts)
    container = {
        "format_version": _TENSOR_CONTAINER_VERSION,
        "payload_path": _TENSOR_PAYLOAD_PATH,
        "payload_byte_length": len(payload),
        "payload_sha256": sha256_bytes(payload),
        "tensors": entries,
    }
    artifact = publish_artifact(
        destination,
        manifest_fields={**manifest_fields, "tensor_container": container},
        files={_TENSOR_PAYLOAD_PATH: payload},
        manifest_name=manifest_name,
    )
    return verify_tensor_state(
        artifact.path,
        expected_artifact_type=artifact.manifest["artifact_type"],
        expected_schema_version=artifact.manifest["schema_version"],
        manifest_name=manifest_name,
    )


def _require_nonnegative_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArtifactValidationError(
            f"tensor {field} must be a non-negative integer"
        )
    return value


def _validated_shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ArtifactValidationError("tensor shape must be a list")
    return tuple(
        _require_nonnegative_integer(dimension, "shape dimension")
        for dimension in value
    )


def _verify_tensor_entries(
    entries: Any,
    payload: bytes,
) -> Mapping[str, LoadedTensor]:
    if not isinstance(entries, list) or not entries:
        raise ArtifactValidationError("tensor list must be non-empty")

    names = [
        entry.get("name") if isinstance(entry, dict) else None
        for entry in entries
    ]
    if any(not isinstance(name, str) or not name for name in names):
        raise ArtifactValidationError("tensor names must be non-empty strings")
    if names != sorted(names) or len(names) != len(set(names)):
        raise ArtifactValidationError(
            "tensor names must be unique and in lexicographic order"
        )

    loaded: dict[str, LoadedTensor] = {}
    expected_offset = 0
    for entry in entries:
        if set(entry) != _TENSOR_FIELDS:
            raise ArtifactValidationError("tensor manifest entry has wrong fields")
        name = entry["name"]
        shape = _validated_shape(entry["shape"])
        dtype_text = entry["dtype"]
        if not isinstance(dtype_text, str):
            raise ArtifactValidationError("tensor dtype must be a string")
        try:
            dtype = np.dtype(dtype_text)
            canonical_dtype = _canonical_dtype(dtype)
        except (TypeError, ValueError) as error:
            raise ArtifactValidationError(
                f"invalid tensor dtype for {name!r}: {dtype_text!r}"
            ) from error
        if canonical_dtype != dtype_text:
            raise ArtifactValidationError("tensor dtype is not canonical")

        trainable = entry["trainable"]
        if not isinstance(trainable, bool):
            raise ArtifactValidationError("tensor trainable flag must be bool")
        if trainable and dtype_text != "<f4":
            raise ArtifactValidationError(
                "trainable tensor dtype must be little-endian float32"
            )
        if entry["order"] != "C":
            raise ArtifactValidationError("tensor order must be C")

        byte_offset = _require_nonnegative_integer(
            entry["byte_offset"],
            "byte_offset",
        )
        byte_length = _require_nonnegative_integer(
            entry["byte_length"],
            "byte_length",
        )
        expected_length = math.prod(shape) * dtype.itemsize
        if byte_offset != expected_offset or byte_length != expected_length:
            raise ArtifactValidationError(
                "tensor byte ranges must be contiguous and shape-consistent"
            )
        end = byte_offset + byte_length
        if end > len(payload):
            raise ArtifactValidationError("tensor byte range exceeds payload")
        tensor_bytes = payload[byte_offset:end]
        digest = entry["sha256"]
        if not isinstance(digest, str) or sha256_bytes(tensor_bytes) != digest:
            raise ArtifactValidationError(f"tensor digest mismatch: {name}")

        array = np.frombuffer(tensor_bytes, dtype=dtype).reshape(shape, order="C")
        array.flags.writeable = False
        loaded[name] = LoadedTensor(
            array=array,
            trainable=trainable,
            sha256=digest,
        )
        expected_offset = end

    if expected_offset != len(payload):
        raise ArtifactValidationError("tensor payload contains trailing bytes")
    return MappingProxyType(loaded)


def verify_tensor_state(
    path: Path | str,
    *,
    expected_artifact_type: str,
    expected_schema_version: str,
    manifest_name: str = "manifest.json",
) -> VerifiedTensorState:
    """Verify the artifact envelope and every tensor range and digest."""

    artifact = verify_artifact(
        path,
        expected_artifact_type=expected_artifact_type,
        expected_schema_version=expected_schema_version,
        manifest_name=manifest_name,
    )
    container = artifact.manifest.get("tensor_container")
    if not isinstance(container, dict) or set(container) != _CONTAINER_FIELDS:
        raise ArtifactValidationError("invalid tensor_container manifest")
    if container["format_version"] != _TENSOR_CONTAINER_VERSION:
        raise ArtifactValidationError("unsupported tensor container version")
    if container["payload_path"] != _TENSOR_PAYLOAD_PATH:
        raise ArtifactValidationError("unexpected tensor payload path")

    payload = (artifact.path / _TENSOR_PAYLOAD_PATH).read_bytes()
    payload_byte_length = _require_nonnegative_integer(
        container["payload_byte_length"],
        "payload_byte_length",
    )
    if payload_byte_length != len(payload):
        raise ArtifactValidationError("tensor payload byte length mismatch")
    payload_digest = container["payload_sha256"]
    if not isinstance(payload_digest, str) or sha256_bytes(payload) != payload_digest:
        raise ArtifactValidationError("tensor payload digest mismatch")

    return VerifiedTensorState(
        artifact=artifact,
        tensors=_verify_tensor_entries(container["tensors"], payload),
    )
