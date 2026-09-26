"""Read-only paired KL contrasts from a sealed backbone OOF evaluation."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from werewolf.artifact_io import canonical_json_bytes, canonical_jsonl_bytes, publish_artifact, sha256_bytes, verify_artifact
from werewolf.tom.backbone_evaluation import PREDICTION_VERSION, REPORT_VERSION, VERSION as EVALUATION_VERSION
from werewolf.tom.backbone_execution import EXECUTION_VERSION, SEAL_VERSION, TERMINAL_VERSION
from werewolf.tom.backbone_model import ARCHITECTURES
from werewolf.tom.backbone_study import STUDY_VERSION
from werewolf.tom.evaluation import score_prediction_rows
from werewolf.tom.paper_study import attest_source
from werewolf.tom.protocol import BOOTSTRAP_VERSION
from werewolf.tom.scoring import game_macro_summary, paired_summary

VERSION = "classic7_backbone_paired_kl_contrasts_v1"
CONTRASTS = (("qwen3", "qwen2"), ("gemma3_text", "qwen2"),
             ("qwen2", "gpt2"), ("qwen3", "gemma3_text"))
GAME_COUNT = 1500
REPLICATES = 10000
PREDICTION_FIELDS = set("game_id boundary_id observer prefix_digest plan_digest probability non_self_log_probability q label_observed observer_alive checkpoint_digest temporal_condition".split())
SEMANTIC_FIELDS = ("game_id", "boundary_id", "observer", "prefix_digest", "plan_digest",
                   "q", "label_observed", "observer_alive", "temporal_condition")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _analysis_source():
    source = attest_source()
    root = Path(__file__).resolve().parents[1]
    relative = "scripts/paper_backbone_contrasts.py"
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    try:
        committed = subprocess.check_output(["git", "-C", str(root), "show",
                                             f"{source['source_revision']}:{relative}"],
                                            env=env, stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("analysis source is absent from the attested Git revision") from error
    script = Path(__file__).read_bytes()
    _require(committed == script, "analysis source differs from attested Git revision")
    digest = sha256_bytes(canonical_json_bytes({"werewolf_implementation_digest": source["implementation_digest"],
                                                "analysis_script_sha256": sha256_bytes(script)}))
    return {"source_revision": source["source_revision"], "implementation_digest": digest}


def _inputs(root, expected_seal_digest):
    report = verify_artifact(root / "seal", expected_artifact_type="backbone_oof_evaluation_report",
                             expected_schema_version=REPORT_VERSION)
    _require(report.manifest_digest == expected_seal_digest, "evaluation seal trust anchor mismatch")
    rm = report.manifest
    _require(set(rm) == {"artifact_type", "schema_version", "evaluation_digest", "training_execution_digest",
                         "training_seal_digest", "preparation", "training_source", "evaluation_source",
                         "predictions", "primary_row_bindings", "bootstrap", "cells",
                         "supplementary_top1_nu", "file_table", "manifest_digest"}
             and set(rm["file_table"]) == {"bootstrap_indices.bin"}, "evaluation seal schema mismatch")
    contract = verify_artifact(root / "contract", expected_artifact_type="backbone_oof_evaluation_contract",
                               expected_schema_version=EVALUATION_VERSION)
    cm = contract.manifest
    _require(contract.manifest_digest == rm["evaluation_digest"]
             and cm["evaluation_source"] == rm["evaluation_source"]
             and cm["training_execution_digest"] == rm["training_execution_digest"]
             and cm["training_seal_digest"] == rm["training_seal_digest"]
             and cm["preparation"] == rm["preparation"]
             and cm["training_source"] == rm["training_source"], "evaluation contract/seal binding mismatch")
    training_root = Path(cm["training_path"]).resolve(strict=True)
    execution = verify_artifact(training_root / "contract", expected_artifact_type="backbone_execution",
                                expected_schema_version=EXECUTION_VERSION)
    seal = verify_artifact(training_root / "seal", expected_artifact_type="backbone_execution_seal",
                           expected_schema_version=SEAL_VERSION)
    em, sm = execution.manifest, seal.manifest
    _require(execution.manifest_digest == rm["training_execution_digest"]
             and seal.manifest_digest == rm["training_seal_digest"]
             and em["preparation"] == rm["preparation"]
             and em["execution_source"] == rm["training_source"]
             and sm["execution_digest"] == execution.manifest_digest
             and sm["preparation"] == rm["preparation"]
             and sm["execution_source"] == rm["training_source"], "historical training binding mismatch")
    prepared = verify_artifact(em["prepared_path"], expected_artifact_type="backbone_tom_study",
                               expected_schema_version=STUDY_VERSION)
    pm = prepared.manifest
    _require(prepared.manifest_digest == rm["preparation"]["manifest_digest"]
             and pm["study_digest"] == rm["preparation"]["study_digest"]
             and pm["protocol_inputs"]["source"] == {key: rm["preparation"][key]
                                                        for key in ("source_revision", "implementation_digest")},
             "prepared study binding mismatch")
    primary = json.loads((prepared.path / "primary.json").read_bytes())
    _require(sha256_bytes(canonical_json_bytes(primary)) == pm["protocol_inputs"]["primary_digest"]
             and primary["metadata"] == pm["protocol_inputs"]["primary_identity"]
             and primary["metadata"]["population_identity"] == "non_wolf_alive",
             "prepared Primary population mismatch")
    return report, prepared, training_root, seal, primary["rows"]


def _bootstrap(report, folds):
    meta = report.manifest["bootstrap"]
    games = meta["game_ids"]
    _require(type(games) is list and len(games) == GAME_COUNT and games == sorted(set(games))
             and len(folds) == 5 and [row["fold"] for row in folds] == list(range(5)),
             "frozen game/fold coverage mismatch")
    fold_games = [row["held_out_game_ids"] for row in folds]
    _require(all(type(ids) is list and ids and len(ids) == len(set(ids))
                 and set(row["training_game_ids"]) == set(games) - set(ids)
                 for ids, row in zip(fold_games, folds, strict=True))
             and sorted(g for ids in fold_games for g in ids) == games, "held-out fold coverage mismatch")
    raw = (report.path / "bootstrap_indices.bin").read_bytes()
    _require(set(meta) == {"version", "seed", "game_ids", "indices_digest", "replicates", "confidence",
                           "sampling_unit", "interval_method"}
             and meta["version"] == BOOTSTRAP_VERSION and meta["replicates"] == REPLICATES
             and meta["confidence"] == .95 and meta["sampling_unit"] == "game"
             and meta["interval_method"] == "percentile_linear"
             and sha256_bytes(raw) == meta["indices_digest"]
             and len(raw) == REPLICATES * len(games) * 4, "sealed bootstrap binding mismatch")
    indices = np.frombuffer(raw, dtype="<i4").reshape(REPLICATES, len(games))
    _require(np.all(indices >= 0) and np.all(indices < len(games)), "bootstrap index out of range")
    return games, fold_games, indices, meta["indices_digest"]


def _fold_scores(report, training_root, training_seal, primary, fold, fold_ids):
    descriptors = {(r["architecture"], r["fold"]): r for r in report.manifest["predictions"]}
    terminals = {(r["architecture"], r["fold"]): r for r in training_seal.manifest["terminals"]}
    reference, scores = None, {}
    for architecture in ARCHITECTURES:
        key = architecture, fold
        descriptor = descriptors[key]
        terminal = verify_artifact(training_root / "runs" / architecture / str(fold) / "terminal",
                                   expected_artifact_type="backbone_terminal", expected_schema_version=TERMINAL_VERSION)
        artifact = verify_artifact(report.path.parent / "folds" / architecture / str(fold),
                                   expected_artifact_type="backbone_oof_fold_predictions",
                                   expected_schema_version=PREDICTION_VERSION)
        m = artifact.manifest
        _require(terminal.manifest_digest == descriptor["terminal_digest"] == terminals[key]["terminal_digest"]
                 and artifact.manifest_digest == descriptor["prediction_digest"]
                 and m["evaluation_digest"] == report.manifest["evaluation_digest"]
                 and m["training_seal_digest"] == report.manifest["training_seal_digest"]
                 and m["architecture"] == architecture and m["fold"] == fold
                 and m["temporal_condition"] == "explicit_day_phase"
                 and m["terminal_digest"] == terminal.manifest_digest
                 and m["checkpoint_digest"] == terminal.manifest["checkpoint_ancestry"][-1]["digest"]
                 and set(m["file_table"]) == {"predictions.jsonl"}, "fold prediction/terminal binding mismatch")
        raw = (artifact.path / "predictions.jsonl").read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        _require(raw == canonical_jsonl_bytes(rows) and len(rows) == m["row_count"],
                 "prediction serialization/count mismatch")
        seen, selected, ids, semantics = set(), [], [], []
        observers = {}
        for row in rows:
            _require(set(row) == PREDICTION_FIELDS and type(row["observer"]) is int
                     and 0 <= row["observer"] < 7 and type(row["label_observed"]) is bool
                     and type(row["observer_alive"]) is bool and row["game_id"] in fold_ids
                     and row["temporal_condition"] == "explicit_day_phase"
                     and row["checkpoint_digest"] == m["checkpoint_digest"], "prediction row identity/schema mismatch")
            key = row["game_id"], row["boundary_id"], row["observer"]
            _require(key not in seen, "duplicate prediction row")
            seen.add(key)
            observers.setdefault(key[:2], set()).add(key[2])
            semantics.append({field: row[field] for field in SEMANTIC_FIELDS})
            mask = primary[row["game_id"]][row["boundary_id"]]
            _require(type(mask) is list and len(mask) == 7 and all(type(value) is bool for value in mask),
                     "invalid frozen Primary mask")
            if row["label_observed"] and mask[row["observer"]]:
                _require(row["observer_alive"] and len(row["q"]) == len(row["probability"]) == 7
                         and len(row["non_self_log_probability"]) == 6
                         and all(type(value) in (int, float) and math.isfinite(value)
                                 for field in ("q", "probability", "non_self_log_probability") for value in row[field])
                         and row["q"][row["observer"]] == row["probability"][row["observer"]] == 0
                         and all(value >= 0 for value in row["q"] + row["probability"])
                         and math.isclose(sum(row["q"]), 1, abs_tol=1e-6)
                         and math.isclose(sum(row["probability"]), 1, abs_tol=1e-6)
                         and all(math.isclose(row["probability"][seat], math.exp(logp), abs_tol=1e-7)
                                 for seat, logp in zip((seat for seat in range(7) if seat != row["observer"]),
                                                       row["non_self_log_probability"], strict=True)),
                         "invalid Primary prediction/target")
                selected.append(row)
                ids.append(list(key))
        expected_boundaries = {(game, boundary) for game in fold_ids for boundary in primary[game]}
        _require(set(observers) == expected_boundaries
                 and all(value == set(range(7)) for value in observers.values()),
                 "prediction PRE/game coverage mismatch")
        binding = sha256_bytes(canonical_json_bytes(ids))
        _require(binding == m["primary_row_identity_digest"] == report.manifest["primary_row_bindings"][str(fold)],
                 "Primary row binding mismatch")
        semantic_digest = sha256_bytes(canonical_json_bytes(semantics))
        if reference is not None:
            _require(semantic_digest == reference, "architecture PRE/q/Primary semantics differ")
        reference = semantic_digest
        _, scores[architecture] = score_prediction_rows(selected, fold_ids)
        _require(sum(score["row_count"] for score in scores[architecture].values()) == len(selected),
                 "scored Primary row coverage mismatch")
    return scores


def analyze(evaluation_root, expected_seal_digest, output):
    root = Path(evaluation_root).resolve(strict=True)
    requested = Path(output).absolute()
    _require(not any(parent.is_symlink() for parent in (requested, *requested.parents)),
             "contrast output cannot traverse a symbolic link")
    destination = requested.resolve()
    _require(not destination.exists() and not destination.is_relative_to(root),
             "contrast output must be new and outside sealed evaluation")
    report, prepared, training_root, training_seal, primary = _inputs(root, expected_seal_digest)
    _require(not destination.is_relative_to(prepared.path) and not destination.is_relative_to(training_root),
             "contrast output cannot be inside sealed training artifacts")
    games, fold_games, indices, bootstrap_digest = _bootstrap(report, prepared.manifest["folds"])
    expected = {(architecture, fold) for architecture in ARCHITECTURES for fold in range(5)}
    predictions = report.manifest["predictions"]
    terminals = training_seal.manifest["terminals"]
    _require(len(predictions) == len(terminals) == 20
             and {(row["architecture"], row["fold"]) for row in predictions} == expected
             and {(row["architecture"], row["fold"]) for row in terminals} == expected
             and set(report.manifest["cells"]) == set(ARCHITECTURES)
             and set(report.manifest["primary_row_bindings"]) == {str(fold) for fold in range(5)}
             and set(primary) == set(games), "sealed lineage/Primary coverage mismatch")
    scores = {architecture: {} for architecture in ARCHITECTURES}
    for fold, ids in enumerate(fold_games):
        for architecture, values in _fold_scores(report, training_root, training_seal, primary, fold, ids).items():
            _require(not set(scores[architecture]).intersection(values), "duplicate OOF game")
            scores[architecture].update(values)
    for architecture in ARCHITECTURES:
        _require(set(scores[architecture]) == set(games), "incomplete architecture OOF")
        kl = [scores[architecture][game]["kl"] for game in games]
        _require(game_macro_summary(kl, indices, .95) == report.manifest["cells"][architecture]["kl"],
                 "recomputed KL differs from sealed report")
    contrasts = [{"left": left, "right": right,
                  "Delta_KL": paired_summary([scores[left][game]["kl"] for game in games],
                                             [scores[right][game]["kl"] for game in games], indices, .95)}
                 for left, right in CONTRASTS]
    source = _analysis_source()
    return publish_artifact(destination, manifest_fields={
        "artifact_type": "paper_backbone_paired_kl_contrasts", "schema_version": VERSION,
        "evaluation_seal_digest": report.manifest_digest,
        "evaluation_contract_digest": report.manifest["evaluation_digest"],
        "bootstrap_digest": bootstrap_digest, "analysis_source": source,
        "metric": {"name": "KL", "direction": "left_minus_right", "negative_means": "left_lower_KL",
                   "aggregation": "game_macro", "sampling_unit": "game", "confidence": .95,
                   "interval_method": "percentile_linear"},
        "contrasts": contrasts,
    }, files={})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--seal-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.evaluation, args.seal_digest, args.output)
    print(result.manifest_digest)


if __name__ == "__main__":
    main()
