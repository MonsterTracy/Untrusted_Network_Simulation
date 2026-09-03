"""Strict JSON helpers retained by pre-reconstruction ToM modules."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

from werewolf.helper.log_utils import Log

_LOG_FIELDS = (
    "viewer",
    "source",
    "target",
    "content",
    "day",
    "time",
    "event",
)


def serialize_json_value(value: Any) -> Any:
    """Convert one supported runtime value into strict JSON data."""

    if isinstance(value, Log):
        fields = (
            _LOG_FIELDS
            if hasattr(value, "viewer")
            else _LOG_FIELDS[1:]
        )
        return {
            field: serialize_json_value(getattr(value, field))
            for field in fields
        }
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floats are not valid canonical JSON")
        return value
    if isinstance(value, (tuple, list)):
        return [serialize_json_value(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical JSON object keys must be strings")
        return {
            key: serialize_json_value(item)
            for key, item in value.items()
        }
    raise TypeError(
        f"unsupported canonical JSON runtime type: {type(value).__name__}"
    )


def canonical_json(value: Any) -> str:
    return json.dumps(
        serialize_json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


__all__ = [
    "canonical_digest",
    "canonical_json",
    "serialize_json_value",
]
