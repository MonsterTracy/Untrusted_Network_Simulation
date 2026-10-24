"""Legal partition reader for the offline dataset; never opens supervision."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import re

from werewolf.artifact_io import (
    ArtifactValidationError, canonical_json_bytes, open_artifact_envelope,
    read_artifact_file, sha256_bytes,
)
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.observer_dtrs.context import CONTEXT_VERSION, ObserverContext
from werewolf.observer_dtrs.features import FEATURE_VERSION, ObserverFeatures, observer_features
from werewolf.observer_dtrs.probabilities import ProbabilityRow


DATASET_VERSION = "observer_dtrs_offline_dataset_v1"
DATASET_TYPE = "observer_dtrs_offline_dataset"
ACCESS_POLICY = "legal_partition_and_explicit_restricted_offline_supervision_v1"


def _check(condition, message):
    if not condition:
        raise ArtifactValidationError(message)


def _equal(a, b):
    return canonical_json_bytes(a) == canonical_json_bytes(b)


def _digest(value):
    _check(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
           "invalid dataset SHA-256 identity")


def _json(raw):
    value = json.loads(raw)
    _check(raw == canonical_json_bytes(value), "noncanonical dataset JSON")
    return value


def _jsonl(raw):
    _check(not raw or raw.endswith(b"\n"), "truncated dataset JSONL")
    return tuple(_json(line) for line in raw.splitlines())


def sample_id(context: ObserverContext) -> str:
    return sha256_bytes(canonical_json_bytes([DATASET_VERSION, context.game_id,
        context.boundary_id, context.acting_wolf, context.observer_id, context.prefix_digest]))


@dataclass(frozen=True)
class LegalDatasetRow:
    sample_id: str
    context: ObserverContext
    probability: ProbabilityRow
    features: ObserverFeatures

    def to_record(self):
        return {"sample_id": self.sample_id, "context": self.context.to_record(),
                "probability": self.probability.values, "features": self.features.to_record()}


@dataclass(frozen=True)
class ObserverDatasetView:
    """Only selected games' legal payloads have been opened and verified."""

    artifact: object
    game_ids: tuple[str, ...]
    rows: tuple[LegalDatasetRow, ...]

    @property
    def manifest_digest(self):
        return self.artifact.manifest_digest


def _legal_row(record):
    _check(set(record) == {"sample_id", "context", "probability", "features"},
           "dataset legal row field mismatch")
    raw = dict(record["context"])
    for key in ("schema_version", "alive_wolf_ids", "wolf_count"):
        raw.pop(key)
    for key in ("alive_player_ids", "known_wolf_ids"):
        raw[key] = tuple(raw[key])
    context = ObserverContext(**raw)
    _check(_equal(context.to_record(), record["context"]), "dataset context schema mismatch")
    probability = ProbabilityRow(context.game_id, context.boundary_id, context.prefix_digest,
                                  context.observer_id, record["probability"])
    features = observer_features(context, probability)
    _check(_equal(features.to_record(), record["features"]), "dataset feature/Q mismatch")
    _check(record["sample_id"] == sample_id(context), "dataset sample identity mismatch")
    return LegalDatasetRow(record["sample_id"], context, probability, features)


def _validate_population(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row.context.boundary_id].append(row)
    for group in groups.values():
        first = group[0].context
        expected = [p for p in first.alive_player_ids if p not in first.known_wolf_ids]
        _check([r.context.observer_id for r in group] == expected,
               "dataset missing/duplicate/noncanonical PRE observer population")
        fixed = {k: v for k, v in first.to_record().items() if k != "observer_id"}
        _check(all(_equal(fixed, {k: v for k, v in r.context.to_record().items()
                                if k != "observer_id"}) for r in group),
               "dataset inconsistent contexts within PRE")
    _check([r.context.step_index for r in rows] == sorted(r.context.step_index for r in rows),
           "dataset canonical PRE order mismatch")
    return len(groups)


def open_dataset(path, *, expected_digest=None, game_ids=None) -> ObserverDatasetView:
    """Validate manifest and selected legal games, without opening restricted bytes.

    Explicit nonempty whole-game partitions are supported. Full payload/audit
    verification requires dataset_offline.verify_dataset; source completeness
    additionally requires dataset_offline.audit_dataset_sources.
    """
    try:
        return _open(path, expected_digest, game_ids)
    except ArtifactValidationError:
        raise
    except (OSError, KeyError, TypeError, ValueError, IndexError) as error:
        raise ArtifactValidationError(f"invalid Observer-DTRS dataset: {error}") from error


def _open(path, expected_digest, game_ids):
    artifact = open_artifact_envelope(path, expected_artifact_type=DATASET_TYPE,
                                      expected_schema_version=DATASET_VERSION)
    m = artifact.manifest
    if expected_digest is not None:
        _digest(expected_digest)
        _check(artifact.manifest_digest == expected_digest, "dataset trust anchor mismatch")
    _check(set(m) == {"artifact_type", "schema_version", "artifact_identity", "access_policy",
        "versions", "origins", "games", "coverage", "implementation", "runtime", "source_locations",
        "file_table", "manifest_digest"}, "dataset manifest field mismatch")
    _check(m["access_policy"] == ACCESS_POLICY and m["versions"]["context"] == CONTEXT_VERSION
           and m["versions"]["features"] == FEATURE_VERSION, "dataset version/access policy mismatch")
    games = m["games"]
    ids = tuple(g["game_id"] for g in games)
    _check(len(ids) >= 5 and len(set(ids)) == len(ids), "dataset game coverage mismatch")
    for game in games:
        _check(set(game) == {"game_id", "fold_index", "row_count", "pre_count", "sample_ids_digest"}
               and isinstance(game["game_id"], str)
               and re.fullmatch(r"[A-Za-z0-9_.-]+", game["game_id"]) is not None,
               "dataset game index schema mismatch")
        _check(type(game["fold_index"]) is int and game["fold_index"] in range(5)
               and all(type(game[k]) is int and game[k] >= 0 for k in ("row_count", "pre_count")),
               "dataset game counts/fold mismatch")
        _digest(game["sample_ids_digest"])
    _check({g["fold_index"] for g in games} == set(range(5)), "dataset missing game fold")
    expected_files = {f"{part}/{g}.jsonl" for g in ids
        for part in ("legal/games", "restricted/supervision", "restricted/provenance")}
    expected_files |= {"restricted/audit.json", "restricted/implementation_sources.json",
                       "restricted/upstream_manifests.json"}
    _check(set(m["file_table"]) == expected_files, "dataset file partition mismatch")
    _check(set(m["coverage"]) == {"game_count", "canonical_pre_count", "oof_prediction_row_count",
        "oof_fold_prediction_counts", "sample_count", "wolf_pre_count"}
        and all(type(v) is int and v >= 0 for k, v in m["coverage"].items() if k != "oof_fold_prediction_counts")
        and len(m["coverage"]["oof_fold_prediction_counts"]) == 5
        and all(type(v) is int and v >= 0 for v in m["coverage"]["oof_fold_prediction_counts"])
        and sum(m["coverage"]["oof_fold_prediction_counts"]) == m["coverage"]["oof_prediction_row_count"],
        "dataset coverage schema mismatch")
    _check(m["coverage"]["game_count"] == len(ids)
           and m["coverage"]["sample_count"] == sum(g["row_count"] for g in games)
           and m["coverage"]["wolf_pre_count"] == sum(g["pre_count"] for g in games)
           and m["coverage"]["oof_prediction_row_count"] == 7 * m["coverage"]["canonical_pre_count"],
           "dataset declared coverage mismatch")
    chosen = ids if game_ids is None else tuple(game_ids)
    _check(bool(chosen) and len(chosen) == len(set(chosen)) and set(chosen) <= set(ids),
           "dataset reader partition must contain unique known whole games")
    chosen = tuple(g for g in ids if g in set(chosen))
    rows = []
    for game in games:
        game_id = game["game_id"]
        if game_id not in chosen:
            continue
        selected = tuple(_legal_row(r) for r in _jsonl(read_artifact_file(artifact,
            f"legal/games/{game_id}.jsonl")))
        _check(all(r.context.game_id == game_id for r in selected), "dataset cross-game row mismatch")
        _check(len(selected) == game["row_count"] and _validate_population(selected) == game["pre_count"],
               "dataset game row/PRE coverage mismatch")
        _check(len({r.sample_id for r in selected}) == len(selected)
               and sha256_bytes(canonical_json_bytes([r.sample_id for r in selected])) == game["sample_ids_digest"],
               "dataset duplicate/missing sample identity")
        rows.extend(selected)
    return ObserverDatasetView(artifact, chosen, tuple(rows))
