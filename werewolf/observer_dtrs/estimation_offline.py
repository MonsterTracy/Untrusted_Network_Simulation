"""Explicit gated dataset-to-fit adapter and immutable estimator artifacts."""
import json
from pathlib import Path

from werewolf.artifact_io import (
    canonical_json_bytes, publish_artifact, read_artifact_file, sha256_bytes, verify_artifact,
)
from werewolf.observer_dtrs.dataset import open_dataset
from werewolf.observer_dtrs.dataset_offline import open_supervision
from werewolf.observer_dtrs.estimation import FittedModel, TrainingRow, VERSION, fit_model, validate_model
from werewolf.observer_dtrs.splits import partition_games, read_splits, source_manifest


def implementation_digest():
    root = Path(__file__).parent
    paths = ("d2_protocol.py", "splits.py", "estimation.py", "estimation_offline.py", "scoring.py")
    return sha256_bytes(canonical_json_bytes({name: sha256_bytes((root / name).read_bytes()) for name in paths}))


def fit_partition(dataset_path, *, split_path, expected_split_digest, protocol,
                  tom_fold, outer_fold, inner_fold, model_id, bins, smoothing, weighting="game_equal"):
    """Reads supervision ONLY for the declared complete training partition.

    inner_fold=None means outer training; otherwise inner training. There is no
    argument for external validation/test rows or their targets.
    """
    m = source_manifest(dataset_path, protocol)
    split = read_splits(split_path, dataset_manifest=m, protocol=protocol,
                        expected_manifest_digest=expected_split_digest)
    train, _ = partition_games(split, tom_fold=tom_fold, outer_fold=outer_fold, inner_fold=inner_fold)
    view = open_dataset(dataset_path, expected_digest=protocol.record["dataset_digest"], game_ids=train)
    offline = open_supervision(view)
    if any(r.q_source.fold_index != tom_fold for r in offline):
        raise ValueError("fit cannot mix different ToM prediction sources")
    source = {"scope": "formal_training_partition", "protocol_digest": protocol.digest,
        "dataset_digest": view.manifest_digest, "split_manifest_digest": expected_split_digest,
        "tom_fold": tom_fold, "outer_fold": outer_fold, "inner_fold": inner_fold,
        "game_sizes": {g["game_id"]: g["row_count"] for g in m["games"] if g["game_id"] in train},
        "origins": m["origins"], "implementation_digest": implementation_digest()}
    rows = tuple(TrainingRow(r.legal.sample_id, r.legal.context.game_id, r.legal.features,
                             r.supervision.event_label_y) for r in offline)
    return fit_model(rows, protocol=protocol, model_id=model_id, bins=bins, smoothing=smoothing,
                     source=source, weighting=weighting)


def publish_model(destination, model, *, protocol):
    validate_model(model, protocol)
    m = model.record
    return publish_artifact(destination, manifest_fields={
        "artifact_type": "observer_dtrs_eq_probability_model", "schema_version": VERSION,
        "protocol_digest": protocol.digest, "model_digest": model.digest, "source": m["source"],
        "model_id": m["model_id"], "parameters": m["parameters"]}, files={"model.json": model.payload})


def read_model(path, *, expected_manifest_digest, protocol):
    a = verify_artifact(path, expected_artifact_type="observer_dtrs_eq_probability_model",
                        expected_schema_version=VERSION)
    if a.manifest_digest != expected_manifest_digest or set(a.manifest["file_table"]) != {"model.json"}:
        raise ValueError("estimator artifact identity mismatch")
    model = FittedModel(read_artifact_file(a, "model.json"))
    validate_model(model, protocol)
    if any(canonical_json_bytes(a.manifest[key]) != canonical_json_bytes(model.record[key])
           for key in ("protocol_digest", "source", "model_id", "parameters")) or a.manifest["model_digest"] != model.digest:
        raise ValueError("estimator manifest/payload mismatch")
    return model
