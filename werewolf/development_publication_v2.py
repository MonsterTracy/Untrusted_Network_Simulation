"""Explicit Development subset publication; V1 remains plan-closed and unchanged."""

from dataclasses import dataclass
from pathlib import Path

from werewolf.artifact_io import ArtifactValidationError, canonical_json_bytes, publish_artifact, open_artifact_envelope
from werewolf.canonical_split import (
    VerifiedCanonicalSplit, validate_split_collection,
    validate_split_header, verify_development_membership,
)
from werewolf.development_publication import (
    VerifiedCollection, VerifiedDevelopmentPublication, DevelopmentPublicationHandles,
    PublicationPublicView, RoleSidecarGame, assign_development_folds, open_role_sidecar,
    _SAFE_IDENTITY, _require_digest, _open_public_envelope, _load_canonical_json,
    _published_game_from_bundle, _published_game_from_record, _role_assignment,
    _role_sidecar_record, _development_game_set_digest, _validate_fold_manifest_record,
    _publication_statistics, FOLD_ASSIGNMENT_VERSION,
)
from werewolf.structured_history import STRUCTURED_TOKEN_PLANNER_VERSION

PUBLICATION_TYPE = "development_publication_v2"
PUBLICATION_VERSION = "classic7_development_publication_v2"
_FIELDS = {
    "artifact_type", "schema_version", "artifact_identity", "publication_id",
    "collection_id", "collection_plan_digest", "collection_identity_digest",
    "split_id", "split_manifest_digest", "split_protocol_digest",
    "independent_test_identity_digest", "development_game_set_digest",
    "development_games", "game_count", "collection_success_count",
    "statistics", "structured_token_planner_version", "fold_assignment_version",
    "development_fold_manifest_digest", "role_sidecar_digest", "source_revision",
}


@dataclass(frozen=True)
class VerifiedDevelopmentPublicationV2(VerifiedDevelopmentPublication):
    """Shares the Dataset and Development-only Role Sidecar boundary with V1."""
    split_header: dict


def publish_development_v2(collection, split, destination, *, publication_id):
    if not isinstance(collection, VerifiedCollection) or not isinstance(split, VerifiedCanonicalSplit):
        raise TypeError("Development V2 requires verified Collection and Split handles")
    split = validate_split_collection(split, collection)
    if not isinstance(publication_id, str) or not _SAFE_IDENTITY.fullmatch(publication_id) or Path(destination).name != publication_id:
        raise ValueError("invalid Development V2 publication identity")
    ids = {g["game_id"] for g in split.members("development")}
    # Filtering happens before any public projection or role extraction.
    bundles = tuple(g for g in collection.games if g.game_id in ids)
    games = tuple(_published_game_from_bundle(g) for g in bundles)
    pairs = tuple((g.game_id, g.bundle_digest) for g in games)
    game_set_digest = _development_game_set_digest(pairs)
    folds = assign_development_folds(pairs, development_game_set_digest=game_set_digest)
    sidecar = _role_sidecar_record(
        publication_id=publication_id, collection_identity_digest=collection.collection_identity_digest,
        development_game_set_digest=game_set_digest,
        games=tuple(RoleSidecarGame(g.game_id, g.manifest_digest,
                                  _role_assignment(g.private_replay_evidence.role_assignment)) for g in bundles),
    )
    descriptors = {g["game_id"]: g for g in split.members("development")}
    entries = [{**{k: v for k, v in descriptors[g.game_id].items()
                    if k not in {"ranking_digest", "partition"}},
                "published_record_digest": g.record_digest} for g in games]
    files = {f"public/games/{g.game_id}.json": canonical_json_bytes(g.to_record()) for g in games}
    files.update({
        "public/development_fold_manifest.json": canonical_json_bytes(folds.to_record()),
        "restricted/role_sidecar.json": canonical_json_bytes(sidecar),
        "lineage/split_header.json": canonical_json_bytes(split.manifest),
        "lineage/development_membership.json": canonical_json_bytes({"members": list(split.members("development"))}),
    })
    artifact = publish_artifact(destination, manifest_fields={
        "artifact_type": PUBLICATION_TYPE, "schema_version": PUBLICATION_VERSION,
        "artifact_identity": publication_id, "publication_id": publication_id,
        "collection_id": collection.plan.collection_id, "collection_plan_digest": collection.plan.plan_digest,
        "collection_identity_digest": collection.collection_identity_digest,
        "split_id": split.manifest["artifact_identity"], "split_manifest_digest": split.digest,
        "split_protocol_digest": split.manifest["split_protocol_digest"],
        "independent_test_identity_digest": split.manifest["independent_test_identity_digest"],
        "development_game_set_digest": game_set_digest, "development_games": entries,
        "game_count": len(games), "collection_success_count": len(collection.games),
        "statistics": _publication_statistics(games),
        "structured_token_planner_version": STRUCTURED_TOKEN_PLANNER_VERSION,
        "fold_assignment_version": FOLD_ASSIGNMENT_VERSION,
        "development_fold_manifest_digest": folds.manifest_digest,
        "role_sidecar_digest": sidecar["sidecar_digest"], "source_revision": collection.plan.source_revision,
    }, files=files)
    public = open_development_publication_v2(destination, expected_split_digest=split.digest)
    assert public.manifest_digest == artifact.manifest_digest
    return DevelopmentPublicationHandles(public, open_role_sidecar(public))


def open_development_publication_v2(path, *, expected_split_digest, game_ids=None):
    """Read only Development records; the trusted split digest is mandatory.

    The embedded commitment and Development membership disclose no Test identities,
    seeds, histories, beliefs, replay evidence or roles. Full split evidence stays
    with the custodian. Partition reads retain V1 laziness.
    """
    _require_digest(expected_split_digest, "expected_split_digest")
    root_path = Path(path)
    if any(p.is_symlink() for p in (root_path, *root_path.parents)):
        raise ArtifactValidationError("Development V2 cannot traverse symbolic links")
    # Inventory-only validation rejects extra restricted/lineage files and links
    # before any payload opens. It never parses or reads restricted role bytes.
    open_artifact_envelope(path, expected_artifact_type=PUBLICATION_TYPE,
                           expected_schema_version=PUBLICATION_VERSION)
    if game_ids is not None:
        game_ids = tuple(game_ids)
        if not game_ids or len(set(game_ids)) != len(game_ids):
            raise ArtifactValidationError("publication partition must be nonempty and unique")
    root, m, digest = _open_public_envelope(path, game_ids)
    if (set(m) != _FIELDS | {"file_table", "manifest_digest"}
            or m["artifact_type"] != PUBLICATION_TYPE or m["schema_version"] != PUBLICATION_VERSION):
        raise ArtifactValidationError("Development V2 schema/field mismatch")
    if (not isinstance(m["publication_id"], str) or not _SAFE_IDENTITY.fullmatch(m["publication_id"])
            or m["artifact_identity"] != m["publication_id"] or root.name != m["publication_id"]):
        raise ArtifactValidationError("Development V2 publication identity mismatch")
    header = validate_split_header(_load_canonical_json(root / "lineage/split_header.json"),
                                   expected_digest=expected_split_digest)
    for key in ("collection_id", "collection_plan_digest", "collection_identity_digest",
                "split_protocol_digest", "independent_test_identity_digest"):
        if m[key] != header[key]:
            raise ArtifactValidationError("Development V2 split lineage mismatch")
    if (m["split_manifest_digest"] != header["manifest_digest"] or m["split_id"] != header["artifact_identity"]
            or m["source_revision"] != header["source_revision"]
            or m["structured_token_planner_version"] != STRUCTURED_TOKEN_PLANNER_VERSION
            or m["fold_assignment_version"] != FOLD_ASSIGNMENT_VERSION):
        raise ArtifactValidationError("Development V2 version/lineage mismatch")
    membership = _load_canonical_json(root / "lineage/development_membership.json")
    if set(membership) != {"members"}:
        raise ArtifactValidationError("invalid Development membership certificate")
    expected_members = sorted(verify_development_membership(header, membership["members"]), key=lambda g: g["ordinal"])
    listed = m["development_games"]
    if not isinstance(listed, list) or len(listed) != len(expected_members):
        raise ArtifactValidationError("Development V2 membership mismatch")
    for row, member in zip(listed, expected_members, strict=True):
        expected = {k: v for k, v in member.items() if k not in {"ranking_digest", "partition"}}
        if (not isinstance(row, dict) or set(row) != set(expected) | {"published_record_digest"}
                or canonical_json_bytes({k: v for k, v in row.items() if k != "published_record_digest"}) != canonical_json_bytes(expected)):
            raise ArtifactValidationError("Development V2 membership mismatch")
        _require_digest(row["published_record_digest"], "published_record_digest")
    ids = {g["game_id"] for g in listed}
    if game_ids is not None and not set(game_ids) <= ids:
        raise ArtifactValidationError("game is outside Development partition")
    expected_files = {
        "public/development_fold_manifest.json", "restricted/role_sidecar.json",
        *(f"public/games/{g}.json" for g in ids),
        "lineage/split_header.json", "lineage/development_membership.json",
    }
    if set(m["file_table"]) != expected_files:
        raise ArtifactValidationError("Development V2 file partition mismatch")
    # Verify the outer file table also binds the nested metadata, not just its envelope.
    from werewolf.artifact_io import sha256_bytes
    for name in ("lineage/split_header.json", "lineage/development_membership.json"):
        data = (root / name).read_bytes()
        if m["file_table"][name] != {"byte_size": len(data), "sha256": sha256_bytes(data)}:
            raise ArtifactValidationError("Development V2 lineage file digest mismatch")
    games = []
    for row in listed:
        if game_ids is not None and row["game_id"] not in game_ids:
            continue
        game = _published_game_from_record(_load_canonical_json(root / f"public/games/{row['game_id']}.json"))
        if (game.game_id != row["game_id"] or game.bundle_digest != row["bundle_digest"]
                or game.record_digest != row["published_record_digest"]):
            raise ArtifactValidationError("Development V2 published game lineage mismatch")
        games.append(game)
    pairs = tuple((g["game_id"], g["bundle_digest"]) for g in listed)
    game_set_digest = _development_game_set_digest(pairs)
    folds = _validate_fold_manifest_record(
        _load_canonical_json(root / "public/development_fold_manifest.json"), pairs, game_set_digest)
    if (m["development_game_set_digest"] != game_set_digest
            or m["development_fold_manifest_digest"] != folds.manifest_digest):
        raise ArtifactValidationError("Development V2 fold binding mismatch")
    statistics = _publication_statistics(games)
    if not isinstance(m["statistics"], dict) or set(m["statistics"]) != set(statistics):
        raise ArtifactValidationError("Development V2 statistics mismatch")
    for key, value in statistics.items():
        stored = m["statistics"][key]
        if type(stored) is not int or stored < value or (game_ids is None and stored != value):
            raise ArtifactValidationError("Development V2 statistics mismatch")
    if (type(m["game_count"]) is not int or m["game_count"] != len(listed)
            or type(m["collection_success_count"]) is not int
            or m["collection_success_count"] != header["development_count"] + header["test_count"]):
        raise ArtifactValidationError("Development V2 count mismatch")
    _require_digest(m["role_sidecar_digest"], "role_sidecar_digest")
    return VerifiedDevelopmentPublicationV2(
        root, m, digest, PublicationPublicView(
            m["publication_id"], m["collection_id"], m["collection_plan_digest"],
            m["collection_identity_digest"], game_set_digest, tuple(games), folds,
            (), m["statistics"]["max_observed_day"], m["statistics"]["max_structured_token_count"],
            STRUCTURED_TOKEN_PLANNER_VERSION,
        ), header,
    )
