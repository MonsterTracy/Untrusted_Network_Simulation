"""Canonical-bytes-backed immutable JSON values for evidence objects."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from werewolf.artifact_io import canonical_json_bytes


@dataclass(frozen=True)
class FrozenJSONValue:
    canonical_bytes: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_bytes, bytes):
            raise TypeError("FrozenJSONValue requires bytes")
        try:
            value = json.loads(self.canonical_bytes.decode("utf-8"))
        except (UnicodeError, ValueError) as error:
            raise ValueError("FrozenJSONValue requires canonical JSON") from error
        if canonical_json_bytes(value) != self.canonical_bytes:
            raise ValueError("FrozenJSONValue bytes are not canonical JSON")

    def to_value(self) -> Any:
        """Return a fresh mutable decoding, never the frozen stored value."""

        return json.loads(self.canonical_bytes.decode("utf-8"))


def freeze_json_value(value: Any) -> FrozenJSONValue:
    if isinstance(value, FrozenJSONValue):
        return value
    return FrozenJSONValue(canonical_json_bytes(value))


def freeze_json_object(value: Any, field_name: str) -> FrozenJSONValue:
    frozen = freeze_json_value(value)
    if not isinstance(frozen.to_value(), dict):
        raise TypeError(f"{field_name} must be a mapping")
    return frozen
