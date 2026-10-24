"""Label-independent whole-game partitions bound to the frozen D2 contract."""
from collections import Counter
import json

from werewolf.artifact_io import (
    canonical_json_bytes, open_artifact_envelope, publish_artifact, read_artifact_file,
    sha256_bytes, verify_artifact,
)
from werewolf.observer_dtrs.dataset import DATASET_TYPE, DATASET_VERSION

VERSION = "observer_dtrs_d2_whole_game_splits_v1"


def game_partition(game_ids, *, protocol, tom_fold, folds, parent_outer=None):
    ids = tuple(game_ids)
    if (len(ids) != len(set(ids)) or not ids or any(not isinstance(g, str) or not g for g in ids)
            or type(tom_fold) is not int or tom_fold not in range(5)
            or folds not in (3, 5) or (folds == 3 and parent_outer not in range(5))
            or (folds == 5 and parent_outer is not None)):
        raise ValueError("invalid whole-game split request")
    spec = protocol.record["split"]
    def rank(game):
        return sha256_bytes(canonical_json_bytes([spec["hash_domain"], spec["protocol_seed"],
            tom_fold, parent_outer, game])), game
    return {game: index % folds for index, game in enumerate(sorted(ids, key=rank))}


def build_splits(dataset_manifest, protocol):
    """Only source manifest metadata, never labels/features, enters this function."""
    m, p = dataset_manifest, protocol.record
    games = m["games"]
    ids = [g["game_id"] for g in games]
    if (m["manifest_digest"] != p["dataset_digest"] or len(ids) != 1500
            or len(set(ids)) != 1500 or Counter(g["fold_index"] for g in games) != {k: 300 for k in range(5)}
            or any(type(g["fold_index"]) is not int or type(g["row_count"]) is not int
                   or g["row_count"] < 0 for g in games)
            or sum(g["row_count"] for g in games) != m["coverage"]["sample_count"]):
        raise ValueError("D2 source game manifest does not match frozen population")
    assignments, folds = {}, []
    for k in range(5):
        group = sorted(g["game_id"] for g in games if g["fold_index"] == k)
        outer = game_partition(group, protocol=protocol, tom_fold=k, folds=5)
        assignments[str(k)] = outer
        nested = []
        for j in range(5):
            train = sorted(g for g in group if outer[g] != j)
            inner = game_partition(train, protocol=protocol, tom_fold=k, folds=3, parent_outer=j)
            assert Counter(inner.values()) == {h: 80 for h in range(3)}
            nested.append({"outer_fold": j, "inner_assignment": inner})
        assert Counter(outer.values()) == {j: 60 for j in range(5)}
        folds.append({"tom_fold": k, "outer_assignment": outer, "nested": nested})
    return {"schema_version": VERSION, "protocol_digest": protocol.digest,
        "dataset_digest": m["manifest_digest"], "source_origins": m["origins"],
        "source_games": games, "folds": folds,
        "outer_assignment_digest": sha256_bytes(canonical_json_bytes(assignments)),
        "coverage": {"games": 1500, "samples": sum(g["row_count"] for g in games),
            "outer_blocks": 25, "inner_blocks": 75, "outer_test_appearances_per_game": 1}}


def partition_games(record, *, tom_fold, outer_fold, inner_fold=None):
    if tom_fold not in range(5) or outer_fold not in range(5) or (inner_fold is not None and inner_fold not in range(3)):
        raise ValueError("invalid partition fold")
    fold = record["folds"][tom_fold]
    outer = fold["outer_assignment"]
    test = tuple(sorted(g for g, j in outer.items() if j == outer_fold))
    if inner_fold is None:
        return tuple(sorted(set(outer) - set(test))), test
    inner = fold["nested"][outer_fold]["inner_assignment"]
    return (tuple(sorted(g for g, h in inner.items() if h != inner_fold)),
            tuple(sorted(g for g, h in inner.items() if h == inner_fold)))


def source_manifest(path, protocol):
    # Metadata-only: the already-published D1 source remains unchanged and gated.
    artifact = open_artifact_envelope(path, expected_artifact_type=DATASET_TYPE,
                                      expected_schema_version=DATASET_VERSION)
    if artifact.manifest_digest != protocol.record["dataset_digest"]:
        raise ValueError("D2 dataset trust anchor mismatch")
    return artifact.manifest


def publish_splits(destination, dataset_path, protocol):
    record = build_splits(source_manifest(dataset_path, protocol), protocol)
    return publish_artifact(destination, manifest_fields={
        "artifact_type": "observer_dtrs_d2_splits", "schema_version": VERSION,
        "protocol_digest": protocol.digest, "dataset_digest": record["dataset_digest"],
        "coverage": record["coverage"]}, files={"splits.json": canonical_json_bytes(record)})


def read_splits(path, *, dataset_manifest, protocol, expected_manifest_digest):
    artifact = verify_artifact(path, expected_artifact_type="observer_dtrs_d2_splits",
                               expected_schema_version=VERSION)
    if (artifact.manifest_digest != expected_manifest_digest
            or set(artifact.manifest["file_table"]) != {"splits.json"}):
        raise ValueError("D2 split artifact identity mismatch")
    expected = build_splits(dataset_manifest, protocol)
    if (read_artifact_file(artifact, "splits.json") != canonical_json_bytes(expected)
            or artifact.manifest["protocol_digest"] != protocol.digest
            or artifact.manifest["dataset_digest"] != expected["dataset_digest"]
            or artifact.manifest["coverage"] != expected["coverage"]):
        raise ValueError("D2 split reconstruction mismatch")
    return expected
