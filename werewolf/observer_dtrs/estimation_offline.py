"""Explicit gated dataset-to-fit adapter and immutable estimator artifacts."""
import json
import math
from numbers import Real
from pathlib import Path
import subprocess

from werewolf.artifact_io import (
    canonical_json_bytes, open_artifact_envelope, publish_artifact, read_artifact_file, sha256_bytes, verify_artifact,
)
from werewolf.observer_dtrs.dataset import open_dataset
from werewolf.observer_dtrs.dataset_offline import open_supervision
from werewolf.observer_dtrs.estimation import (
    FINAL_SELECTION_RULE, FittedModel, TrainingRow, VERSION, fit_model, validate_model,
    _validate_source,
)
from werewolf.observer_dtrs.scoring import select_parameters
from werewolf.observer_dtrs.splits import partition_games, read_splits, source_manifest
from werewolf.observer_dtrs.validation_artifacts import VERSION as RUN_VERSION


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


def aggregate_inner_scores(selections, *, split_record, game_folds, protocol):
    """只归集EQ内层评分；每局四次先平均，不读取外层测试成绩。"""
    grids = {
        name: {(b, l) for b in ((1,) if name == "OBS-EQ0" else protocol.record["histogram"]["requested_bin_counts"])
               for l in protocol.record["smoothing"]["lambda_grid_game_equivalent_mass"]}
        for name in ("OBS-EQ1", "OBS-EQ0")
    }
    collected = {name: {key: {g: {} for g in game_folds} for key in grid}
                 for name, grid in grids.items()}
    seen = set()
    for identity, record in selections:
        k, j = identity["tom_fold"], identity["outer_fold"]
        if (type(k) is not int or type(j) is not int or k not in range(5) or j not in range(5)
                or (k, j) in seen or identity["weighting"] != "game_equal"):
            raise ValueError("duplicate/invalid/non-game-equal Selection")
        seen.add((k, j))
        train, test = partition_games(split_record, tom_fold=k, outer_fold=j)
        if (set(train) & set(test) or set(train) | set(test) != {g for g, f in game_folds.items() if f == k}):
            raise ValueError("Selection does not bind to its OOF source partition")
        found = set()
        for entry in record["candidate_game_brier"]:
            name = entry["model"]
            # 历史Selection还包含LR；其评分和已选参数均不进入本次选参。
            if name == "OBS-LR-r":
                continue
            if name not in grids or set(entry["parameters"]) != {"bins", "smoothing"}:
                raise ValueError("unexpected final EQ candidate")
            b, l = entry["parameters"]["bins"], entry["parameters"]["smoothing"]
            if (type(b) is not int or isinstance(l, bool) or not isinstance(l, Real)
                    or not math.isfinite(l) or (b, l) not in grids[name] or (name, b, l) in found):
                raise ValueError("duplicate/invalid final EQ parameters")
            found.add((name, b, l))
            scores = entry["games"]
            if (not isinstance(scores, dict) or set(scores) != set(train)
                    or any(isinstance(v, bool) or not isinstance(v, Real)
                           or not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values())):
                raise ValueError("missing/invalid inner validation game scores")
            for game, value in scores.items():
                collected[name][b, l][game][j] = value
        if found != {(name, b, l) for name, grid in grids.items() for b, l in grid}:
            raise ValueError("incomplete final EQ candidate grid")
    if seen != {(k, j) for k in range(5) for j in range(5)}:
        raise ValueError("final selection requires all 25 source/outer blocks")
    tables = {name: {} for name in grids}
    for name, candidates in collected.items():
        for key, games in candidates.items():
            if any(len(scores) != 4 for scores in games.values()):
                raise ValueError("each game requires exactly four inner validation scores")
            tables[name][key] = {g: math.fsum(scores[j] for j in sorted(scores)) / 4
                                 for g, scores in sorted(games.items())}
    return tables


def _selection_record(path, kind, expected_digest):
    """沿已钉住的Run读取必要记录，不重读Bootstrap或重算历史结果。"""
    artifact = open_artifact_envelope(path, expected_artifact_type="observer_dtrs_" + kind,
                                      expected_schema_version=RUN_VERSION)
    if (artifact.manifest_digest != expected_digest
            or artifact.manifest["identity_digest"] != sha256_bytes(canonical_json_bytes(artifact.manifest["identity"]))):
        raise ValueError("final selection artifact identity mismatch")
    raw = read_artifact_file(artifact, "record.json")
    record = json.loads(raw)
    if raw != canonical_json_bytes(record):
        raise ValueError("noncanonical final selection record")
    return artifact, record


def select_final_parameters(run_path, *, expected_complete_digest, expected_split_digest,
                            dataset_manifest, split_record, protocol):
    """从冻结Complete→Contract/Block→Selection引用链选择新候选参数。"""
    root = Path(run_path)
    complete, result = _selection_record(root / "complete", "validation_complete", expected_complete_digest)
    contract_digest = complete.manifest["identity"]["contract_digest"]
    contract, c = _selection_record(root / "contract", "validation_contract", contract_digest)
    sizes = {g["game_id"]: g["row_count"] for g in dataset_manifest["games"]}
    folds = {g["game_id"]: g["fold_index"] for g in dataset_manifest["games"]}
    if (c["scope"] != "formal_development_validation" or c["protocol_digest"] != protocol.digest
            or c["dataset_digest"] != protocol.record["dataset_digest"] or c["split_digest"] != expected_split_digest
            or c["dataset_digest"] != dataset_manifest["manifest_digest"]
            or c["origins"] != dataset_manifest["origins"] or c["game_sizes"] != sizes
            or c["game_folds"] != folds or c["split_record"] != split_record
            or contract.manifest["identity"] != {"run_identity": sha256_bytes(canonical_json_bytes(c))}
            or result["status"] != "ALL_BLOCKS_COMPLETE"
            or result["block_digests"] != complete.manifest["identity"]["block_digests"]
            or set(result["block_digests"]) != {f"{k}-{j}" for k in range(5) for j in range(5)}):
        raise ValueError("final selection does not bind to the formal training population")
    selections, refs = [], {}
    for k in range(5):
        for j in range(5):
            key = f"{k}-{j}"
            identity = {"contract_digest": contract_digest, "tom_fold": k, "outer_fold": j}
            base = root / "blocks" / key
            block, value = _selection_record(base / "complete", "validation_block_complete", result["block_digests"][key])
            train, test = partition_games(split_record, tom_fold=k, outer_fold=j)
            if (block.manifest["identity"] != identity or value["status"] != "PAIRED_BLOCK_COMPLETE"
                    or value["train_games"] != list(train) or value["test_games"] != list(test)):
                raise ValueError("final Selection block partition mismatch")
            refs[key] = value["references"]["game_equal"]["selection"]
            artifact, record = _selection_record(base / "game_equal" / "selection", "validation_selection", refs[key])
            expected = {**identity, "weighting": "game_equal",
                        "candidate_artifact_digests": record["candidate_artifact_digests"]}
            if artifact.manifest["identity"] != expected:
                raise ValueError("final Selection source/weighting mismatch")
            selections.append((expected, record))
    tables = aggregate_inner_scores(selections, split_record=split_record, game_folds=folds, protocol=protocol)
    parameters, scores = {}, []
    for name, table in tables.items():
        b, l = select_parameters(table, protocol=protocol, model_id=name, weighting="game_equal")
        parameters[name] = {"bins": b, "smoothing": l}
        scores.extend({"model": name, "bins": b, "smoothing": l, "games": games}
                      for (b, l), games in sorted(table.items()))
    return {"rule": FINAL_SELECTION_RULE, "run_path": str(root.resolve()),
            "contract_digest": contract_digest, "complete_digest": expected_complete_digest,
            "selection_digests": refs, "aggregated_scores_digest": sha256_bytes(canonical_json_bytes(scores)),
            "selected_parameters": parameters}


def fit_final_models(dataset_path, *, split_path, expected_split_digest, run_path,
                     expected_complete_digest, protocol):
    """在历史完整OOF人口上拟合新EQ1/EQ0候选；不发布产物或启用在线来源。"""
    m = source_manifest(dataset_path, protocol)
    split = read_splits(split_path, dataset_manifest=m, protocol=protocol,
                        expected_manifest_digest=expected_split_digest)
    selection = select_final_parameters(run_path, expected_complete_digest=expected_complete_digest,
                                        expected_split_digest=expected_split_digest,
                                        dataset_manifest=m, split_record=split, protocol=protocol)
    source = {"scope": "final_development_oof", "protocol_digest": protocol.digest,
        "dataset_digest": m["manifest_digest"], "split_manifest_digest": expected_split_digest,
        "tom_fold": None, "outer_fold": None, "inner_fold": None,
        "game_sizes": {g["game_id"]: g["row_count"] for g in m["games"]}, "origins": m["origins"],
        "game_folds": {g["game_id"]: g["fold_index"] for g in m["games"]}, "selection": selection,
        "implementation_digest": implementation_digest(),
        "fit_revision": subprocess.check_output(["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2], text=True).strip()}
    # 在读取全量监督前先验证人口及新来源；历史分区限制保持原样。
    _validate_source(source, protocol)
    view = open_dataset(dataset_path, expected_digest=m["manifest_digest"])
    offline = open_supervision(view)
    if any(r.q_source.fold_index != source["game_folds"][r.legal.context.game_id] for r in offline):
        raise ValueError("final fit OOF source/game mismatch")
    rows = tuple(TrainingRow(r.legal.sample_id, r.legal.context.game_id, r.legal.features,
                             r.supervision.event_label_y) for r in offline)
    return {name: fit_model(rows, protocol=protocol, model_id=name, source=source,
                            weighting="game_equal", **parameters)
            for name, parameters in selection["selected_parameters"].items()}


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
