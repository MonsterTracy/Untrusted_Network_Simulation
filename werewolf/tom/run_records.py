"""Crash-atomic immutable run records (mutable run directories are not artifacts)."""

import os
import tempfile
from pathlib import Path

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.artifact_io.canonical import _fsync_directory, ensure_durable_directory


def record_with_digest(value):
    return {**value, "record_digest": sha256_bytes(canonical_json_bytes(value))}


def publish_record(path, value):
    record = record_with_digest(value)
    publish_run_bytes(path, canonical_json_bytes(record))
    return record


def publish_run_bytes(path, data):
    """Publish one complete immutable run payload, or verify exact reuse."""
    path = Path(path)
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("run publication cannot traverse symbolic links")
    ensure_durable_directory(path.parent)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("immutable run record identity mismatch")
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        _fsync_directory(path.parent)
        return
    descriptor, staging = tempfile.mkstemp(prefix=f".{path.name}.staging-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staging, path)  # no-replace; never check-then-overwrite
        _fsync_directory(path.parent)
    finally:
        os.unlink(staging)
        _fsync_directory(path.parent)


def read_record(path):
    import json
    path = Path(path)
    if path.is_symlink():
        raise ValueError("run record cannot be a symbolic link")
    data = path.read_bytes()
    value = json.loads(data)
    digest = value.pop("record_digest")
    if sha256_bytes(canonical_json_bytes(value)) != digest:
        raise ValueError("run record digest mismatch")
    record = {**value, "record_digest": digest}
    if canonical_json_bytes(record) != data:
        raise ValueError("noncanonical run record bytes")
    return record
