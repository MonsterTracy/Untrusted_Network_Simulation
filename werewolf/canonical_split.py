"""Precommitted, label-independent partition of a completed Canonical Collection.

These artifacts contain identities and ledger metadata only, never game records
or supervision. They are not a test-data access capability.
"""

from dataclasses import dataclass
from pathlib import Path

from werewolf.artifact_io import (
    ArtifactValidationError, VerifiedArtifact, canonical_json_bytes,
    publish_artifact, sha256_bytes, verify_artifact,
)
from werewolf.canonical_collection.attempt_ledger import (
    CollectionPlan, AttemptLedgerState, TerminalOutcome, collection_plan_from_record,
    _claim_from_record, _terminal_from_record, _state_from_records,
)
from werewolf.development_publication import (
    VerifiedCollection, _collection_identity, _load_canonical_json,
    _require_digest, _SAFE_IDENTITY, _validate_verified_collection,
)

PROTOCOL_TYPE = "canonical_split_protocol"
PROTOCOL_VERSION = "classic7_canonical_split_protocol_v1"
SPLIT_TYPE = "canonical_game_split"
SPLIT_VERSION = "classic7_canonical_game_split_v1"
ASSIGNMENT_VERSION = "classic7_sha256_plan_seed_identity_split_v1"
TEST_IDENTITY_VERSION = "classic7_independent_test_identity_v1"
DEVELOPMENT_MEMBERSHIP_VERSION = "classic7_development_split_membership_v1"
PROTOCOL_BINDING = "canonical_split_protocol_digest"
_ENVELOPE = {"file_table", "manifest_digest"}
_PROTOCOL_FIELDS = {
    "artifact_type", "schema_version", "artifact_identity", "collection_id",
    "target_success_count", "development_count", "test_count", "split_seed",
    "assignment_version",
}
_SPLIT_FIELDS = {
    "artifact_type", "schema_version", "artifact_identity", "collection_id",
    "collection_plan_digest", "collection_identity_digest", "split_protocol_digest",
    "assignment_version", "development_membership_digest", "development_count", "test_count",
    "independent_test_identity_digest",
    "source_revision",
}
SPLIT_FILES = {"split_protocol.json", "collection_plan.json", "attempt_ledger.json", "split_membership.json"}


@dataclass(frozen=True)
class VerifiedSplitProtocol:
    artifact: VerifiedArtifact

    @property
    def digest(self):
        return self.artifact.manifest_digest

    @property
    def manifest(self):
        return self.artifact.manifest


@dataclass(frozen=True)
class VerifiedCanonicalSplit:
    artifact: VerifiedArtifact
    protocol: dict
    plan: CollectionPlan
    ledger_state: AttemptLedgerState
    ranked_games: tuple[dict, ...]

    @property
    def digest(self):
        return self.artifact.manifest_digest

    @property
    def manifest(self):
        return self.artifact.manifest

    def members(self, partition):
        if partition not in {"development", "independent_test"}:
            raise ValueError("unknown split partition")
        return tuple(g for g in self.ranked_games if g["partition"] == partition)


def _development_membership_digest(members):
    return sha256_bytes(canonical_json_bytes([DEVELOPMENT_MEMBERSHIP_VERSION, members]))


def validate_split_header(record, *, expected_digest):
    """Verify the portable commitment only; no Test metadata file is opened."""
    _require_digest(expected_digest, "expected_split_digest")
    if not isinstance(record, dict) or set(record) != _SPLIT_FIELDS | _ENVELOPE:
        raise ArtifactValidationError("Split Manifest fields mismatch")
    if (record["artifact_type"] != SPLIT_TYPE or record["schema_version"] != SPLIT_VERSION
            or record["assignment_version"] != ASSIGNMENT_VERSION or set(record["file_table"]) != SPLIT_FILES):
        raise ArtifactValidationError("Split Manifest version/inventory mismatch")
    for key in ("artifact_identity", "collection_id"):
        if not isinstance(record[key], str) or not _SAFE_IDENTITY.fullmatch(record[key]):
            raise ArtifactValidationError("invalid split identity")
    if not isinstance(record["source_revision"], str) or not record["source_revision"].strip():
        raise ArtifactValidationError("invalid split source revision")
    for key in ("collection_plan_digest", "collection_identity_digest", "split_protocol_digest",
                "development_membership_digest", "independent_test_identity_digest", "manifest_digest"):
        _require_digest(record[key], key)
    if (type(record["development_count"]) is not int or record["development_count"] < 5
            or type(record["test_count"]) is not int or record["test_count"] < 1):
        raise ArtifactValidationError("invalid split counts")
    for entry in record["file_table"].values():
        if (not isinstance(entry, dict) or set(entry) != {"byte_size", "sha256"}
                or type(entry["byte_size"]) is not int or entry["byte_size"] < 0):
            raise ArtifactValidationError("invalid split file table")
        _require_digest(entry["sha256"], "file sha256")
    digest = sha256_bytes(canonical_json_bytes({k: v for k, v in record.items() if k != "manifest_digest"}))
    if digest != record["manifest_digest"] or digest != expected_digest:
        raise ArtifactValidationError("Split Manifest differs from trusted digest")
    return record


def verify_development_membership(header, members):
    """Validate the complete Development partition against the trusted header.

    Whole-partition publication needs one content commitment, not per-game tree
    proofs. The custodian's full Split reader separately recomputes both partitions.
    """
    if not isinstance(members, list) or len(members) != header["development_count"]:
        raise ArtifactValidationError("Development membership coverage mismatch")
    if any(not isinstance(member, dict) or member.get("partition") != "development" for member in members):
        raise ArtifactValidationError("Test game in Development membership")
    if _development_membership_digest(members) != header["development_membership_digest"]:
        raise ArtifactValidationError("Development membership digest mismatch")
    return tuple(members)


def _validate_protocol(record):
    if not isinstance(record, dict) or set(record) != _PROTOCOL_FIELDS | _ENVELOPE:
        raise ArtifactValidationError("Split Protocol field mismatch")
    if (record["artifact_type"] != PROTOCOL_TYPE or record["schema_version"] != PROTOCOL_VERSION
            or record["assignment_version"] != ASSIGNMENT_VERSION or record["file_table"] != {}):
        raise ArtifactValidationError("Split Protocol version/inventory mismatch")
    for key in ("artifact_identity", "collection_id"):
        if not isinstance(record[key], str) or not _SAFE_IDENTITY.fullmatch(record[key]):
            raise ArtifactValidationError("invalid Split Protocol identity")
    for key in ("target_success_count", "development_count", "test_count", "split_seed"):
        if type(record[key]) is not int or record[key] < 0:
            raise ArtifactValidationError("invalid Split Protocol integer")
    if (record["development_count"] < 5 or record["test_count"] < 1
            or record["target_success_count"] != record["development_count"] + record["test_count"]):
        raise ArtifactValidationError("Split Protocol counts must cover the target with at least five development games")
    _require_digest(record["manifest_digest"], "split_protocol_digest")
    payload = {k: v for k, v in record.items() if k != "manifest_digest"}
    if sha256_bytes(canonical_json_bytes(payload)) != record["manifest_digest"]:
        raise ArtifactValidationError("Split Protocol digest mismatch")
    return record


def publish_split_protocol(destination, *, protocol_id, collection_id,
                           development_count, test_count, split_seed):
    """Freeze rules before creating the Collection Plan; no source data is read."""
    fields = dict(artifact_type=PROTOCOL_TYPE, schema_version=PROTOCOL_VERSION,
                  artifact_identity=protocol_id, collection_id=collection_id,
                  target_success_count=development_count + test_count,
                  development_count=development_count, test_count=test_count,
                  split_seed=split_seed, assignment_version=ASSIGNMENT_VERSION)
    proposed = {**fields, "file_table": {}}
    proposed["manifest_digest"] = sha256_bytes(canonical_json_bytes(proposed))
    _validate_protocol(proposed)
    if Path(destination).name != protocol_id:
        raise ValueError("protocol destination must end with protocol_id")
    publish_artifact(destination, manifest_fields=fields, files={})
    return open_split_protocol(destination, expected_digest=proposed["manifest_digest"])


def open_split_protocol(path, *, expected_digest):
    _require_digest(expected_digest, "expected_protocol_digest")
    artifact = verify_artifact(path, expected_artifact_type=PROTOCOL_TYPE,
                               expected_schema_version=PROTOCOL_VERSION)
    _validate_protocol(artifact.manifest)
    if artifact.path.name != artifact.manifest["artifact_identity"]:
        raise ArtifactValidationError("Split Protocol path identity mismatch")
    if artifact.manifest_digest != expected_digest:
        raise ArtifactValidationError("Split Protocol differs from trusted digest")
    return VerifiedSplitProtocol(artifact)


def validate_protocol_plan(protocol, plan):
    """No new Plan schema: the existing provenance map binds the frozen rule."""
    _validate_protocol(protocol)
    if (plan.collection_id != protocol["collection_id"]
            or plan.target_canonical_success_count != protocol["target_success_count"]
            or dict(plan.environment_provenance).get(PROTOCOL_BINDING) != protocol["manifest_digest"]):
        raise ArtifactValidationError("Collection Plan does not bind this Split Protocol")


def _closed_state(plan, ledger):
    if not isinstance(ledger, dict) or set(ledger) != {"claims", "terminals"}:
        raise ArtifactValidationError("invalid split ledger metadata")
    try:
        claims = tuple(_claim_from_record(r, plan) for r in ledger["claims"])
        terminals = tuple(_terminal_from_record(r, plan) for r in ledger["terminals"])
        state = _state_from_records(plan, claims, terminals)
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactValidationError("invalid split ledger metadata") from error
    if (claims != state.claims or terminals != state.terminals or state.open_claim is not None
            or not state.target_reached or state.canonical_success_count != plan.target_canonical_success_count
            or not terminals or terminals[-1].outcome is not TerminalOutcome.CANONICAL_SUCCESS):
        raise ArtifactValidationError("split requires a complete ordered Collection")
    return state


def _split_fields(split_id, protocol, plan, state):
    validate_protocol_plan(protocol, plan)
    claims = {c.ordinal: c for c in state.claims}
    ranked = []
    for terminal in state.terminals:
        if terminal.outcome is not TerminalOutcome.CANONICAL_SUCCESS:
            continue
        # Bundle digests and all game contents are deliberately excluded from ranking.
        identity = [ASSIGNMENT_VERSION, protocol["split_seed"], plan.collection_id,
                    terminal.ordinal, terminal.seed]
        ranked.append(dict(
            ranking_digest=sha256_bytes(canonical_json_bytes(identity)),
            ordinal=terminal.ordinal, seed=terminal.seed, attempt_id=terminal.attempt_id,
            claim_record_digest=claims[terminal.ordinal].record_digest,
            terminal_record_digest=terminal.record_digest,
            game_id=terminal.canonical_game_bundle_id,
            bundle_digest=terminal.canonical_game_bundle_digest,
        ))
    ranked.sort(key=lambda g: (g["ranking_digest"], g["ordinal"]))
    if len(ranked) != protocol["target_success_count"]:
        raise ArtifactValidationError("split requires all target successes")
    if len({g["game_id"] for g in ranked}) != len(ranked):
        raise ArtifactValidationError("split game identities are not unique")
    for index, game in enumerate(ranked):
        game["partition"] = "development" if index < protocol["development_count"] else "independent_test"
    identity_digest = _collection_identity(plan, state)
    test_digest = sha256_bytes(canonical_json_bytes({
        "schema_version": TEST_IDENTITY_VERSION, "collection_identity_digest": identity_digest,
        "split_protocol_digest": protocol["manifest_digest"],
        "games": [g for g in ranked if g["partition"] == "independent_test"],
    }))
    fields = dict(artifact_type=SPLIT_TYPE, schema_version=SPLIT_VERSION,
                artifact_identity=split_id, collection_id=plan.collection_id,
                collection_plan_digest=plan.plan_digest, collection_identity_digest=identity_digest,
                split_protocol_digest=protocol["manifest_digest"], assignment_version=ASSIGNMENT_VERSION,
                development_membership_digest=_development_membership_digest(ranked[:protocol["development_count"]]),
                development_count=protocol["development_count"],
                test_count=protocol["test_count"], independent_test_identity_digest=test_digest,
                source_revision=plan.source_revision)
    return fields, ranked


def publish_canonical_split(collection, protocol, destination, *, split_id):
    if not isinstance(collection, VerifiedCollection) or not isinstance(protocol, VerifiedSplitProtocol):
        raise TypeError("split requires verified Collection and Split Protocol handles")
    _validate_verified_collection(collection)
    # Reopen handles to detect mutable dictionaries or changed on-disk artifacts.
    protocol = open_split_protocol(protocol.artifact.path, expected_digest=protocol.digest)
    if not isinstance(split_id, str) or not _SAFE_IDENTITY.fullmatch(split_id) or Path(destination).name != split_id:
        raise ValueError("invalid split identity")
    state = collection.ledger_state
    fields, ranked = _split_fields(split_id, protocol.manifest, collection.plan, state)
    files = {
        "split_protocol.json": canonical_json_bytes(protocol.manifest),
        "collection_plan.json": canonical_json_bytes(collection.plan.to_record()),
        "split_membership.json": canonical_json_bytes({"ranked_games": ranked}),
        "attempt_ledger.json": canonical_json_bytes({
            "claims": [c.to_record() for c in state.claims],
            "terminals": [t.to_record() for t in state.terminals],
        }),
    }
    artifact = publish_artifact(destination, manifest_fields=fields, files=files)
    return open_canonical_split(destination, expected_digest=artifact.manifest_digest)


def open_canonical_split(path, *, expected_digest):
    """Recompute membership from bound Plan/ledger metadata, not JSON membership claims.

    The expected digest is a trust anchor, recorded outside this artifact. Hashes
    cannot authenticate an attacker who can replace every parent and that anchor.
    This portable check does not replace replay verification of the source games.
    """
    _require_digest(expected_digest, "expected_split_digest")
    artifact = verify_artifact(path, expected_artifact_type=SPLIT_TYPE,
                               expected_schema_version=SPLIT_VERSION)
    m = artifact.manifest
    validate_split_header(m, expected_digest=expected_digest)
    if artifact.path.name != m["artifact_identity"]:
        raise ArtifactValidationError("Split Manifest path identity mismatch")
    protocol = _validate_protocol(_load_canonical_json(artifact.path / "split_protocol.json"))
    plan = collection_plan_from_record(_load_canonical_json(artifact.path / "collection_plan.json"))
    state = _closed_state(plan, _load_canonical_json(artifact.path / "attempt_ledger.json"))
    expected, ranked = _split_fields(m["artifact_identity"], protocol, plan, state)
    membership = _load_canonical_json(artifact.path / "split_membership.json")
    if (canonical_json_bytes({k: v for k, v in m.items() if k not in _ENVELOPE}) != canonical_json_bytes(expected)
            or canonical_json_bytes(membership) != canonical_json_bytes({"ranked_games": ranked})):
        raise ArtifactValidationError("Split Manifest membership or collection binding mismatch")
    if artifact.manifest_digest != expected_digest:
        raise ArtifactValidationError("Split Manifest differs from trusted digest")
    return VerifiedCanonicalSplit(artifact, protocol, plan, state, tuple(ranked))


def validate_split_collection(split, collection):
    _validate_verified_collection(collection)
    split = open_canonical_split(split.artifact.path, expected_digest=split.digest)
    expected, _ = _split_fields(split.manifest["artifact_identity"], split.protocol,
                                collection.plan, collection.ledger_state)
    if {k: v for k, v in split.manifest.items() if k not in _ENVELOPE} != expected:
        raise ArtifactValidationError("Split Manifest belongs to another verified Collection")
    return split
