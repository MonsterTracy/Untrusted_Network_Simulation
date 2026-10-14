"""Synthetic game manifests exercise the production split/partition mechanisms."""
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import pytest

from scripts.diagnose_observer_dtrs_dataset import game_partition as preview_partition
from tests.observer_dtrs.test_estimation import features
from werewolf.artifact_io import ArtifactConflictError, ArtifactValidationError, canonical_json_bytes, publish_artifact
from werewolf.observer_dtrs.d2_protocol import FrozenProtocol, frozen_protocol, publish_protocol, read_protocol
from werewolf.observer_dtrs import estimation_offline, splits


@pytest.fixture
def protocol():
    return frozen_protocol()


@pytest.fixture
def manifest(protocol):
    # In-memory controlled metadata, never claimed as an actual D1 publication.
    return {"manifest_digest": protocol.record["dataset_digest"],
        "origins": {"fixture": "synthetic_1500_game_ids"},
        "coverage": {"sample_count": 1500},
        "games": [{"game_id": f"synthetic-{k}-{j:03}", "fold_index": k, "row_count": 1,
                   "pre_count": 1, "sample_ids_digest": "a" * 64} for k in range(5) for j in range(300)]}


def test_all_outer_and_inner_partitions_complete_exclusive_and_match_preview(protocol, manifest):
    record = splits.build_splits(manifest, protocol)
    for k in range(5):
        games = {g["game_id"] for g in manifest["games"] if g["fold_index"] == k}
        assert record["folds"][k]["outer_assignment"] == preview_partition(games, tom_fold=k, folds=5)
        tested = []
        for j in range(5):
            train, test = splits.partition_games(record, tom_fold=k, outer_fold=j)
            assert len(train) == 240 and len(test) == 60
            assert set(train).isdisjoint(test) and set(train) | set(test) == games
            tested.extend(test)
            validated = []
            for h in range(3):
                fit, val = splits.partition_games(record, tom_fold=k, outer_fold=j, inner_fold=h)
                assert len(fit) == 160 and len(val) == 80
                assert set(fit).isdisjoint(val) and set(fit) | set(val) == set(train)
                assert set(fit).isdisjoint(test) and set(val).isdisjoint(test)
                validated.extend(val)
            assert sorted(validated) == sorted(train)
        assert sorted(tested) == sorted(games)
    changed = deepcopy(manifest)
    changed["irrelevant_test_supervision"] = {"Y": [0, 1, 1]}
    assert splits.build_splits(changed, protocol) == record
    assert record["coverage"]["outer_test_appearances_per_game"] == 1


def test_protocol_and_splits_publish_reconstruct_and_reject_rehashed_tampering(protocol, manifest, tmp_path, monkeypatch):
    pa = publish_protocol(tmp_path / "protocol", protocol)
    assert read_protocol(pa.path, expected_manifest_digest=pa.manifest_digest) == protocol
    # Patch only the source-manifest transport; full split artifact I/O is real.
    monkeypatch.setattr(splits, "source_manifest", lambda path, protocol: manifest)
    sa = splits.publish_splits(tmp_path / "split", "controlled", protocol)
    record = splits.read_splits(sa.path, dataset_manifest=manifest, protocol=protocol,
                               expected_manifest_digest=sa.manifest_digest)
    assert splits.publish_splits(sa.path, "controlled", protocol).manifest_digest == sa.manifest_digest
    bad = deepcopy(record)
    bad["folds"][0]["outer_assignment"]["synthetic-0-000"] = 99
    fields = {k: v for k, v in sa.manifest.items() if k not in ("manifest_digest", "file_table")}
    altered = publish_artifact(tmp_path / "rehashed-bad", manifest_fields=fields,
                              files={"splits.json": canonical_json_bytes(bad)})
    with pytest.raises(ValueError):
        splits.read_splits(altered.path, dataset_manifest=manifest, protocol=protocol,
                          expected_manifest_digest=altered.manifest_digest)
    (sa.path / "splits.json").write_bytes(b"truncated")
    with pytest.raises(ArtifactValidationError):
        splits.read_splits(sa.path, dataset_manifest=manifest, protocol=protocol,
                          expected_manifest_digest=sa.manifest_digest)
    changed = protocol.record
    changed["smoothing"]["lambda_grid_game_equivalent_mass"] = [99]
    with pytest.raises(ValueError):
        FrozenProtocol(canonical_json_bytes(changed))


@pytest.mark.parametrize("mode", ("duplicate", "wrong_fold", "wrong_count", "wrong_anchor"))
def test_invalid_source_population_rejected(protocol, manifest, mode):
    m = deepcopy(manifest)
    if mode == "duplicate":
        m["games"][0] = m["games"][1]
    elif mode == "wrong_fold":
        m["games"][0]["fold_index"] = 1
    elif mode == "wrong_count":
        m["games"].pop()
    else:
        m["manifest_digest"] = "0" * 64
    with pytest.raises(ValueError):
        splits.build_splits(m, protocol)


def test_gated_fit_adapter_opens_only_training_games_and_ignores_external_targets(protocol, manifest, monkeypatch):
    record = splits.build_splits(manifest, protocol)
    opened = []
    labels = {g["game_id"]: 1 for g in manifest["games"]}
    monkeypatch.setattr(estimation_offline, "source_manifest", lambda path, protocol: manifest)
    monkeypatch.setattr(estimation_offline, "read_splits", lambda *a, **k: record)
    def open_legal(path, *, expected_digest, game_ids):
        opened.append(tuple(game_ids))
        return NS(game_ids=game_ids, manifest_digest=expected_digest)
    def open_labels(view):
        return tuple(NS(legal=NS(sample_id=g, context=NS(game_id=g), features=features(.2)),
                        supervision=NS(event_label_y=labels[g]), q_source=NS(fold_index=0)) for g in view.game_ids)
    monkeypatch.setattr(estimation_offline, "open_dataset", open_legal)
    monkeypatch.setattr(estimation_offline, "open_supervision", open_labels)
    args = dict(split_path="controlled", expected_split_digest="a" * 64, protocol=protocol,
                tom_fold=0, outer_fold=2, inner_fold=1, model_id="OBS-EQ1", bins=3, smoothing=1)
    a = estimation_offline.fit_partition("controlled", **args)
    train, validation = splits.partition_games(record, tom_fold=0, outer_fold=2, inner_fold=1)
    _, test = splits.partition_games(record, tom_fold=0, outer_fold=2)
    assert opened[-1] == train and len(train) == 160
    for game in (*validation, *test):
        labels[game] = 0
    b = estimation_offline.fit_partition("controlled", **args)
    assert a.payload == b.payload
    assert set(a.record["source"]["game_sizes"]) == set(train)
    assert a.record["source"]["inner_fold"] == 1
