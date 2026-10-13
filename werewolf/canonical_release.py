"""Audited collection release; same-account process policy, no Test consumer."""

from pathlib import Path

from werewolf.artifact_io import ArtifactValidationError, canonical_json_bytes, publish_artifact, verify_artifact
from werewolf.canonical_split import open_split_protocol, open_canonical_split, publish_canonical_split, validate_split_collection, validate_protocol_plan
from werewolf.canonical_terminal_audit import publish_collection_terminal_audit, verify_collection_terminal_audit
from werewolf.development_publication import (
    _SAFE_IDENTITY, _load_canonical_json, _validate_verified_collection,
    _published_game_from_bundle, _role_assignment, open_verified_collection, open_role_sidecar,
)
from werewolf.development_publication_v2 import publish_development_v2, open_development_publication_v2
from werewolf.runtime_config import normalize_runtime_config

RELEASE_TYPE = "canonical_collection_release"
RELEASE_VERSION = "classic7_canonical_collection_release_v1"
ACCESS_POLICY = "same_account_process_isolation_v1"


def _inside(root, path):
    path = Path(path)
    if (".." in path.parts or any(p.is_symlink() for p in (path, *path.parents))
            or not path.is_absolute() or not path.is_relative_to(root) or path == root):
        raise ArtifactValidationError("release path must be inside artifact root without symlinks")
    return path


def _layout(root, release_id):
    if not isinstance(release_id, str) or not _SAFE_IDENTITY.fullmatch(release_id):
        raise ArtifactValidationError("invalid release identity")
    return {"terminal_audit": _inside(root, root / "terminal_audits" / release_id),
            "split": _inside(root, root / "splits" / release_id),
            "development": _inside(root, root / "publications" / (release_id + "-development"))}


def _fields(collection, protocol, split, development, release_id):
    return {"artifact_type": RELEASE_TYPE, "schema_version": RELEASE_VERSION,
            "artifact_identity": release_id, "test_access_policy": ACCESS_POLICY,
            "collection_plan_digest": collection.plan.plan_digest,
            "collection_identity_digest": collection.collection_identity_digest,
            "source_revision": collection.plan.source_revision,
            "split_protocol_digest": protocol.digest,
            "split_manifest_digest": split.digest,
            "development_publication_digest": development.manifest_digest,
            "independent_test_identity_digest": split.manifest["independent_test_identity_digest"],
            "success_count": len(collection.games), "development_count": development.manifest["game_count"],
            "test_count": split.manifest["test_count"]}


def _check_development(collection, split, development):
    validate_split_collection(split, collection)
    ids = {g["game_id"] for g in split.members("development")}
    roles = open_role_sidecar(development).games
    if (set(development.public_view.game_ids) != ids
            or development.manifest["collection_success_count"] != len(collection.games)
            or {g.game_id for g in roles} != ids):
        raise ArtifactValidationError("release Development membership mismatch")
    # Membership hashes alone do not prove a derived projection or role truth.
    # Source-side verification compares only Development contents with their
    # authoritative Bundles; Test payloads are not part of this comparison.
    bundles = {b.game_id: b for b in collection.games if b.game_id in ids}
    for game in development.public_view.games:
        expected = _published_game_from_bundle(bundles[game.game_id])
        if game.record_digest != expected.record_digest:
            raise ArtifactValidationError("release Development public record differs from Canonical source")
    for game in roles:
        expected = _role_assignment(bundles[game.game_id].private_replay_evidence.role_assignment)
        if game.role_assignment != expected:
            raise ArtifactValidationError("release Development roles differ from Canonical source")
    sizes = sorted(len(f.game_ids) for f in development.public_view.fold_manifest.folds)
    count = len(ids)
    if sizes != sorted(count // 5 + (i < count % 5) for i in range(5)):
        raise ArtifactValidationError("release five-fold game coverage mismatch")


def publish_canonical_release(collection, protocol, destination, *, runtime_config, artifact_root):
    """Audit all successes before emitting Split/Development; retain any partial artifacts."""
    root = Path(artifact_root)
    if not root.is_dir():
        raise ArtifactValidationError("release artifact root is missing")
    destination = _inside(root, destination)
    if destination.parent != root / "releases":
        raise ArtifactValidationError("release receipt must use releases/ID layout")
    paths = _layout(root, destination.name)
    _validate_verified_collection(collection)
    _inside(root, collection.path)
    _inside(root, protocol.artifact.path)
    protocol = open_split_protocol(protocol.artifact.path, expected_digest=protocol.digest)
    validate_protocol_plan(protocol.manifest, collection.plan)
    config = normalize_runtime_config(runtime_config)
    # This is the release gate, not a change to Canonical success qualification.
    audit = publish_collection_terminal_audit(collection, paths["terminal_audit"], runtime_config=config)
    audit = verify_collection_terminal_audit(audit.path, collection, runtime_config=config)
    split = publish_canonical_split(collection, protocol, paths["split"], split_id=destination.name)
    split = open_canonical_split(split.artifact.path, expected_digest=split.digest)
    handles = publish_development_v2(collection, split, paths["development"], publication_id=paths["development"].name)
    development = open_development_publication_v2(handles.public.path, expected_split_digest=split.digest)
    _check_development(collection, split, development)
    record = {"collection_path": collection.path.relative_to(root).as_posix(),
              "protocol_path": protocol.artifact.path.relative_to(root).as_posix(),
              "terminal_audit_digest": audit.manifest_digest,
              "artifact_paths": {k: p.relative_to(root).as_posix() for k, p in paths.items()}}
    return publish_artifact(destination, manifest_fields={**_fields(collection, protocol, split, development, destination.name),
        "terminal_audit_digest": audit.manifest_digest}, files={
            "release.json": canonical_json_bytes(record), "runtime_config.json": canonical_json_bytes(config)})


def verify_canonical_release(path, *, expected_digest, artifact_root):
    """Explicit source-side verification; opens the full source, never a training input."""
    root = Path(artifact_root)
    path = _inside(root, path)
    if path.parent != root / "releases":
        raise ArtifactValidationError("release receipt must use releases/ID layout")
    artifact = verify_artifact(path, expected_artifact_type=RELEASE_TYPE, expected_schema_version=RELEASE_VERSION)
    if artifact.manifest_digest != expected_digest:
        raise ArtifactValidationError("release differs from trusted digest")
    if set(artifact.manifest["file_table"]) != {"release.json", "runtime_config.json"}:
        raise ArtifactValidationError("release inventory mismatch")
    record = _load_canonical_json(path / "release.json")
    if not isinstance(record, dict) or set(record) != {"collection_path", "protocol_path", "terminal_audit_digest", "artifact_paths"}:
        raise ArtifactValidationError("release record fields mismatch")
    paths = _layout(root, path.name)
    if record["artifact_paths"] != {k: p.relative_to(root).as_posix() for k, p in paths.items()}:
        raise ArtifactValidationError("release artifact paths mismatch")
    source_paths = []
    for key in ("collection_path", "protocol_path"):
        value = record[key]
        if not isinstance(value, str) or Path(value).is_absolute() or Path(value).as_posix() != value:
            raise ArtifactValidationError("release source reference must be relative")
        source_paths.append(_inside(root, root / value))
    config = normalize_runtime_config(_load_canonical_json(path / "runtime_config.json"))
    from werewolf.canonical_collection.production_runtime import classic7_replay_executor
    collection = open_verified_collection(source_paths[0], replay_executor=classic7_replay_executor(config))
    protocol = open_split_protocol(source_paths[1], expected_digest=artifact.manifest["split_protocol_digest"])
    validate_protocol_plan(protocol.manifest, collection.plan)
    audit = verify_collection_terminal_audit(paths["terminal_audit"], collection, runtime_config=config)
    split = open_canonical_split(paths["split"], expected_digest=artifact.manifest["split_manifest_digest"])
    development = open_development_publication_v2(paths["development"], expected_split_digest=split.digest)
    _check_development(collection, split, development)
    expected = {**_fields(collection, protocol, split, development, path.name), "terminal_audit_digest": audit.manifest_digest}
    if ({k: v for k, v in artifact.manifest.items() if k not in {"file_table", "manifest_digest"}} != expected
            or record["terminal_audit_digest"] != audit.manifest_digest):
        raise ArtifactValidationError("release parent binding mismatch")
    return artifact
