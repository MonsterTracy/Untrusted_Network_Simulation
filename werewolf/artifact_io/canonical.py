"""Canonical serialization and crash-safe immutable artifact I/O."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Callable


_MANIFEST_RESERVED_FIELDS = frozenset({"file_table", "manifest_digest"})
_FILE_ENTRY_FIELDS = frozenset({"byte_size", "sha256"})
_SHA256_HEX_LENGTH = 64


class ArtifactValidationError(ValueError):
    """An artifact does not exactly satisfy its declared envelope."""


class ArtifactConflictError(FileExistsError):
    """A different artifact already occupies an immutable destination."""


class ArtifactPublishUnsupportedError(RuntimeError):
    """The host cannot provide atomic no-replace directory publication."""


@dataclass(frozen=True)
class ArtifactEnvelope:
    """Verified manifest and inventory; payload reads still require digest gates."""

    path: Path
    manifest_name: str
    manifest: dict[str, Any]
    manifest_digest: str


@dataclass(frozen=True)
class VerifiedArtifact(ArtifactEnvelope):
    """An immutable artifact whose complete payload has also been verified."""


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite numbers are not valid canonical JSON")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical JSON object keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    raise TypeError(
        f"unsupported canonical JSON value: {type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Encode one JSON value with the frozen canonical byte contract."""

    text = json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return text.encode("utf-8")


def canonical_jsonl_bytes(records: Iterable[Any]) -> bytes:
    """Encode canonical JSON records with exactly one newline per record."""

    return b"".join(canonical_json_bytes(record) + b"\n" for record in records)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _load_canonical_json(data: bytes) -> Any:
    value = json.loads(
        data.decode("utf-8"),
        parse_constant=_reject_json_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )
    if canonical_json_bytes(value) != data:
        raise ValueError("JSON bytes are not in canonical form")
    return value


def _validate_manifest_name(manifest_name: str) -> None:
    if (
        not manifest_name
        or manifest_name in {".", ".."}
        or "/" in manifest_name
        or "\\" in manifest_name
    ):
        raise ValueError("manifest_name must be one root-level filename")


def _validate_relative_path(relative_path: str, manifest_name: str) -> None:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("artifact file paths must be non-empty strings")
    if "\\" in relative_path:
        raise ValueError("artifact file paths must use POSIX separators")
    path = PurePosixPath(relative_path)
    if (
        path.is_absolute()
        or relative_path != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
        or relative_path == manifest_name
    ):
        raise ValueError(f"invalid artifact relative path: {relative_path!r}")


def _validated_manifest_fields(
    manifest_fields: Mapping[str, Any],
) -> dict[str, Any]:
    fields = _json_value(manifest_fields)
    if not isinstance(fields, dict):
        raise TypeError("manifest_fields must be a mapping")
    reserved = _MANIFEST_RESERVED_FIELDS.intersection(fields)
    if reserved:
        raise ValueError(
            "manifest_fields may not set envelope-owned fields: "
            + ", ".join(sorted(reserved))
        )
    for required in ("artifact_type", "schema_version"):
        if not isinstance(fields.get(required), str) or not fields[required]:
            raise ValueError(f"manifest field {required!r} must be a string")
    return fields


def _build_manifest(
    manifest_fields: Mapping[str, Any],
    files: Mapping[str, bytes],
    manifest_name: str,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    _validate_manifest_name(manifest_name)
    fields = _validated_manifest_fields(manifest_fields)
    normalized_files: dict[str, bytes] = {}
    for relative_path, data in files.items():
        _validate_relative_path(relative_path, manifest_name)
        if not isinstance(data, bytes):
            raise TypeError(
                f"artifact payload {relative_path!r} must be immutable bytes"
            )
        normalized_files[relative_path] = data

    file_table = {
        relative_path: {
            "byte_size": len(normalized_files[relative_path]),
            "sha256": sha256_bytes(normalized_files[relative_path]),
        }
        for relative_path in sorted(normalized_files)
    }
    manifest_without_digest = {**fields, "file_table": file_table}
    manifest_digest = sha256_bytes(
        canonical_json_bytes(manifest_without_digest)
    )
    manifest = {
        **manifest_without_digest,
        "manifest_digest": manifest_digest,
    }
    return manifest, normalized_files


def _write_fsynced(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def ensure_durable_directory(path: Path) -> None:
    """Create/confirm a directory chain and persist every containing entry."""
    path = Path(path).absolute()
    chain = [*reversed(path.parents), path]
    for directory in chain:
        if directory.is_symlink():
            raise ArtifactValidationError("symbolic link in durable directory chain")
        directory.mkdir(exist_ok=True)
        if not directory.is_dir():
            raise ArtifactValidationError("durable parent is not a directory")
        _fsync_directory(directory)
        if directory.parent != directory:
            _fsync_directory(directory.parent)


def _fsync_tree(root: Path) -> None:
    directories = [path for path in root.rglob("*") if path.is_dir()]
    directories.sort(key=lambda path: len(path.parts), reverse=True)
    for directory in directories:
        _fsync_directory(directory)
    _fsync_directory(root)


@lru_cache(maxsize=1)
def _atomic_noreplace_rename() -> Callable[[bytes, bytes], int]:
    libc = ctypes.CDLL(None, use_errno=True)

    if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        renamex_np = libc.renamex_np
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        rename_excl = 0x00000004

        def rename(source: bytes, destination: bytes) -> int:
            return renamex_np(source, destination, rename_excl)

        return rename

    if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        renameat2 = libc.renameat2
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        at_fdcwd = -100
        rename_noreplace = 1

        def rename(source: bytes, destination: bytes) -> int:
            return renameat2(
                at_fdcwd,
                source,
                at_fdcwd,
                destination,
                rename_noreplace,
            )

        return rename

    raise ArtifactPublishUnsupportedError(
        "atomic no-replace directory publication is unavailable"
    )


def _publish_directory_noreplace(source: Path, destination: Path) -> None:
    rename = _atomic_noreplace_rename()
    ctypes.set_errno(0)
    result = rename(os.fsencode(source), os.fsencode(destination))
    if result == 0:
        return

    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(destination)
    if error_number in {errno.ENOSYS, errno.ENOTSUP}:
        raise ArtifactPublishUnsupportedError(
            "filesystem does not support atomic no-replace publication"
        )
    raise OSError(error_number, os.strerror(error_number), destination)


def _reuse_or_conflict(
    destination: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_name: str,
) -> VerifiedArtifact:
    verified = verify_artifact(
        destination,
        expected_artifact_type=manifest["artifact_type"],
        expected_schema_version=manifest["schema_version"],
        manifest_name=manifest_name,
    )
    expected_digest = manifest["manifest_digest"]
    if verified.manifest_digest != expected_digest:
        raise ArtifactConflictError(
            f"immutable artifact destination has a different identity: "
            f"{destination}"
        )
    ensure_durable_directory(destination.parent)
    _fsync_directory(destination)
    _fsync_directory(destination.parent)
    return verified


def publish_artifact(
    destination: Path | str,
    *,
    manifest_fields: Mapping[str, Any],
    files: Mapping[str, bytes],
    manifest_name: str = "manifest.json",
) -> VerifiedArtifact:
    """Durably publish one immutable artifact without an overwrite race."""

    _atomic_noreplace_rename()
    destination = Path(destination)
    manifest, normalized_files = _build_manifest(
        manifest_fields,
        files,
        manifest_name,
    )

    if os.path.lexists(destination):
        return _reuse_or_conflict(
            destination,
            manifest=manifest,
            manifest_name=manifest_name,
        )

    ensure_durable_directory(destination.parent)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    published = False
    try:
        for relative_path in sorted(normalized_files):
            _write_fsynced(staging / relative_path, normalized_files[relative_path])
        _write_fsynced(
            staging / manifest_name,
            canonical_json_bytes(manifest),
        )
        _fsync_tree(staging)
        try:
            _publish_directory_noreplace(staging, destination)
        except FileExistsError:
            return _reuse_or_conflict(
                destination,
                manifest=manifest,
                manifest_name=manifest_name,
            )
        published = True
        _fsync_directory(destination.parent)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)

    return verify_artifact(
        destination,
        expected_artifact_type=manifest["artifact_type"],
        expected_schema_version=manifest["schema_version"],
        manifest_name=manifest_name,
    )


def _validated_file_table(
    value: Any,
    manifest_name: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise ArtifactValidationError("manifest file_table must be an object")
    result: dict[str, dict[str, Any]] = {}
    for relative_path, entry in value.items():
        _validate_relative_path(relative_path, manifest_name)
        if not isinstance(entry, dict) or set(entry) != _FILE_ENTRY_FIELDS:
            raise ArtifactValidationError(
                f"invalid file table entry for {relative_path!r}"
            )
        byte_size = entry["byte_size"]
        digest = entry["sha256"]
        if isinstance(byte_size, bool) or not isinstance(byte_size, int):
            raise ArtifactValidationError("file byte_size must be an integer")
        if byte_size < 0:
            raise ArtifactValidationError("file byte_size must be non-negative")
        if (
            not isinstance(digest, str)
            or len(digest) != _SHA256_HEX_LENGTH
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ArtifactValidationError("file sha256 must be lowercase hex")
        result[relative_path] = entry
    return result


def _actual_artifact_tree(
    root: Path,
    manifest_name: str,
) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    for path in root.rglob("*"):
        relative_path = path.relative_to(root).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ArtifactValidationError(
                f"artifact contains a symbolic link: {relative_path}"
            )
        if stat.S_ISREG(mode):
            if relative_path != manifest_name:
                files.add(relative_path)
        elif stat.S_ISDIR(mode):
            directories.add(relative_path)
        else:
            raise ArtifactValidationError(
                f"artifact contains an unsupported filesystem node: {relative_path}"
            )
    return files, directories


def _expected_directories(relative_paths: Iterable[str]) -> set[str]:
    directories: set[str] = set()
    for relative_path in relative_paths:
        parent = PurePosixPath(relative_path).parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def _open_artifact_envelope(
    path: Path,
    *,
    expected_artifact_type: str,
    expected_schema_version: str,
    manifest_name: str,
) -> ArtifactEnvelope:
    _validate_manifest_name(manifest_name)
    if path.is_symlink() or not path.is_dir():
        raise ArtifactValidationError(f"artifact is not a directory: {path}")

    manifest_path = path / manifest_name
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ArtifactValidationError(
            f"artifact manifest is missing: {manifest_path}"
        )
    manifest = _load_canonical_json(manifest_path.read_bytes())
    if not isinstance(manifest, dict):
        raise ArtifactValidationError("artifact manifest must be an object")
    if manifest.get("artifact_type") != expected_artifact_type:
        raise ArtifactValidationError("unsupported artifact_type")
    if manifest.get("schema_version") != expected_schema_version:
        raise ArtifactValidationError("unsupported schema_version")

    manifest_digest = manifest.get("manifest_digest")
    if not isinstance(manifest_digest, str):
        raise ArtifactValidationError("manifest_digest is missing")
    manifest_without_digest = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_digest"
    }
    if sha256_bytes(canonical_json_bytes(manifest_without_digest)) != manifest_digest:
        raise ArtifactValidationError("manifest digest mismatch")

    file_table = _validated_file_table(
        manifest.get("file_table"),
        manifest_name,
    )
    actual_files, actual_directories = _actual_artifact_tree(
        path,
        manifest_name,
    )
    expected_files = set(file_table)
    if actual_files != expected_files:
        raise ArtifactValidationError(
            "artifact files do not exactly match the manifest file table"
        )
    if actual_directories != _expected_directories(expected_files):
        raise ArtifactValidationError(
            "artifact directories do not exactly match the manifest file table"
        )

    return ArtifactEnvelope(
        path=path,
        manifest_name=manifest_name,
        manifest=manifest,
        manifest_digest=manifest_digest,
    )


def open_artifact_envelope(
    path: Path | str,
    *,
    expected_artifact_type: str,
    expected_schema_version: str,
    manifest_name: str = "manifest.json",
) -> ArtifactEnvelope:
    """Validate manifest and inventory without opening gated data partitions."""

    try:
        return _open_artifact_envelope(
            Path(path),
            expected_artifact_type=expected_artifact_type,
            expected_schema_version=expected_schema_version,
            manifest_name=manifest_name,
        )
    except ArtifactValidationError:
        raise
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        raise ArtifactValidationError(f"invalid artifact: {error}") from error


def read_artifact_file(envelope: ArtifactEnvelope, relative_path: str) -> bytes:
    """Open exactly one listed file and verify its complete canonical bytes."""
    if relative_path not in envelope.manifest["file_table"]:
        raise ArtifactValidationError("unlisted artifact payload")
    path = envelope.path / relative_path
    if path.is_symlink():
        raise ArtifactValidationError("symbolic link in artifact payload")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ArtifactValidationError("missing artifact payload") from error
    entry = envelope.manifest["file_table"][relative_path]
    if len(payload) != entry["byte_size"]:
        raise ArtifactValidationError(f"artifact file size mismatch: {relative_path}")
    if sha256_bytes(payload) != entry["sha256"]:
        raise ArtifactValidationError(f"artifact file digest mismatch: {relative_path}")
    return payload


def verify_artifact(path, *, expected_artifact_type, expected_schema_version, manifest_name="manifest.json") -> VerifiedArtifact:
    """Validate the exact schema, canonical bytes, tree, sizes, and digests."""
    envelope = open_artifact_envelope(path, expected_artifact_type=expected_artifact_type,
        expected_schema_version=expected_schema_version, manifest_name=manifest_name)
    for relative_path in envelope.manifest["file_table"]:
        read_artifact_file(envelope, relative_path)
    return VerifiedArtifact(**vars(envelope))
