"""Analyze sealed Experiment A mapper CV artifacts; never fit or generate evidence."""

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error

from werewolf.artifact_io import (
    canonical_json_bytes, canonical_jsonl_bytes, publish_artifact, sha256_bytes, verify_artifact,
)
from werewolf.cli import _artifact_path, _storage_root
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf import phase2_offline as offline
from werewolf.phase2_mapper import (
    FEATURE_CONTRACT_VERSION, FOLDS, MODEL_SPECS, OOFProvenance, MapperPrediction,
    PROBABILITY_EPSILON, _log_loss,
)
from werewolf.phase2_mapper_selection import (
    _GamePanel, _draw_game_multiplicities, _weighted_metrics, percentile_interval,
)
from scripts.run_phase2_mapper_study import EXPERIMENT_A_VERSION

EXECUTION_REVISION = "39f4ea25906808bf1ed28b05fdfb63865cc3ffb3"
VERSION = "phase2_experiment_a_mapper_contrasts_v1"
REPLICATES, SEED, CONFIDENCE = 100_000, 20260929, .95
CONDITIONS = {"A0": ("explicit_day_phase", "continuous"),
              "A1": ("implicit", "continuous"), "A2": ("explicit_day_phase", "hard_top1")}
METRICS = ("pooled_log_loss", "pooled_brier", "game_macro_log_loss", "game_macro_brier")
METRIC_CONTRACT = {"primary": METRICS[0], "secondary": list(METRICS[1:]),
                   "contrasts": {"A1": "implicit - explicit", "A2": "hard_top1 - continuous"}}
SOURCE_FILES = ("scripts/experiment_a_mapper_contrasts.py", "scripts/run_phase2_mapper_study.py",
                "werewolf/phase2_mapper_selection.py", "werewolf/phase2_mapper.py",
                "werewolf/phase2_offline.py", "werewolf/phase2_q_representation.py",
                "werewolf/tom/backbone_evaluation.py", "werewolf/artifact_io/canonical.py",
                "werewolf/cli.py")
KERNELS = ("_draw_game_multiplicities", "_weighted_metrics", "percentile_interval")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return sha256_bytes(canonical_json_bytes(value))


def _git(root, *args):
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    return subprocess.check_output(["git", "-C", str(root), *args], env=env, stderr=subprocess.PIPE)


def _analysis_source():
    root = Path(__file__).resolve().parents[1]
    _require(Path(_git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve() == root,
             "analysis repository root mismatch")
    revision = _git(root, "rev-parse", "HEAD").decode().strip()
    _require(len(revision) == 40 and all(c in "0123456789abcdef" for c in revision),
             "invalid analysis source revision")
    _require(not _git(root, "status", "--porcelain=v1", "--untracked-files=all",
                      "--ignore-submodules=none").strip(), "analysis source must be clean")
    inventory = {}
    for relative in SOURCE_FILES:
        content = (root / relative).read_bytes()
        _require(_git(root, "show", f"{revision}:{relative}") == content,
                 f"analysis source differs from committed inventory: {relative}")
        inventory[relative] = sha256_bytes(content)
    return {"source_revision": revision, "source_sha256": inventory,
            "implementation_digest": _digest(inventory),
            "bootstrap_implementation_digest": _digest({"helpers": KERNELS,
                "kernel_sha256": inventory["werewolf/phase2_mapper_selection.py"],
                "consumer_sha256": inventory[SOURCE_FILES[0]],
                "metric_sha256": inventory["werewolf/phase2_mapper.py"]})}


def _load(root, expected_digest, condition, storage_root):
    artifact = verify_artifact(root, expected_artifact_type="phase2_mapper_development_study",
                               expected_schema_version=EXPERIMENT_A_VERSION)
    m = artifact.manifest
    _require(artifact.manifest_digest == expected_digest, f"{condition}: mapper digest mismatch")
    _require(m["source"]["commit"] == EXECUTION_REVISION, "wrong execution source revision")
    temporal, representation = CONDITIONS[condition]
    identity = m["experiment_a"]
    _require(identity["condition"] == condition and identity["temporal_condition"] == temporal
             and identity["q_representation"] == representation
             and identity["source_revision"] == EXECUTION_REVISION, "wrong condition/representation/source")
    inputs = m["inputs"]
    publication = {key: inputs[key] for key in ("publication_path", "publication_digest",
                                               "role_sidecar_digest", "fold_manifest_digest")}
    _require(publication["publication_digest"] == offline.PUBLICATION_DIGEST
             and publication["role_sidecar_digest"] == offline.ROLE_SIDECAR_DIGEST
             and publication["fold_manifest_digest"] == offline.FOLD_MANIFEST_DIGEST,
             "wrong frozen publication/fold identity")
    evaluation_root = _artifact_path(storage_root, inputs["oof_evaluation_path"])
    oof = OOFProvenance.from_evaluation_root(evaluation_root,
        paired_experiment_digest=identity["paired_experiment_digest"], temporal_condition=temporal,
        q_representation=representation, expected_seal_digest=inputs["oof"]["evaluation_seal_digest"])
    _require(identity == oof.experiment_a and inputs["oof"] == oof.record(), "OOF lineage mismatch")
    # These manifests were verified by the OOF reader, which deliberately has no current-source gate.
    contract = json.loads((evaluation_root / "contract/manifest.json").read_bytes())
    seal = json.loads((evaluation_root / "seal/manifest.json").read_bytes())
    _require(contract["evaluation_source"] == seal["evaluation_source"] == contract["training_source"]
             and contract["evaluation_source"]["source_revision"] == EXECUTION_REVISION,
             "wrong OOF execution source")
    _require(m["metric_contract"] == METRIC_CONTRACT
             and m["protocol"]["feature_contract_version"] == FEATURE_CONTRACT_VERSION
             and canonical_json_bytes(m["protocol"]["model_specs"]) == canonical_json_bytes([MODEL_SPECS[3].record()])
             and m["protocol"]["bootstrap"] is None
             and m["selection"] == {"primary_model": MODEL_SPECS[3].name, "stopped_at": None,
                 "development_supported_alternatives": [], "method": "prespecified_fixed_M3"},
             "wrong fixed mapper spec/metric contract")
    payload = (root / f"cross_fitted_predictions/{MODEL_SPECS[3].name}.jsonl").read_bytes()
    records = [json.loads(line) for line in payload.splitlines()]
    _require(canonical_jsonl_bytes(records) == payload, "noncanonical mapper predictions")
    rows = [MapperPrediction(**record) for record in records]
    _require(rows and len({r.identity for r in rows}) == len(rows), "empty/duplicate candidate panel")
    model_keys = {f"{MODEL_SPECS[3].name}/fold-{fold}" for fold in FOLDS}
    _require(set(m["fit_model_digests"]) == model_keys, "incomplete fixed-M3 fits")
    for r in rows:
        _require(all(isinstance(value, str) and value for value in r.identity)
                 and r.acting_wolf in PLAYER_IDS and r.candidate_j in PLAYER_IDS
                 and r.acting_wolf != r.candidate_j and r.phase in ("speech", "speech_pk")
                 and type(r.theta_ac) is bool and type(r.mapper_fold) is int and r.mapper_fold in FOLDS
                 and type(r.game_oof_fold) is int and r.game_oof_fold == r.mapper_fold
                 and r.model_spec == MODEL_SPECS[3].name
                 and type(r.raw_probability) in (int, float) and math.isfinite(r.raw_probability)
                 and 0 <= r.raw_probability <= 1 and type(r.raw_score) in (int, float)
                 and math.isfinite(r.raw_score) and isinstance(r.prefix_digest, str)
                 and bool(r.prefix_digest), "invalid candidate/fold/spec/probability")
        _require(r.qwen3_prediction_digest == oof.prediction_digest_by_fold[r.mapper_fold]
                 and r.evaluation_seal_digest == oof.evaluation_seal_digest
                 and r.model_digest == m["fit_model_digests"][f"{r.model_spec}/fold-{r.mapper_fold}"],
                 "candidate Q/model lineage mismatch")
    candidate_digest = _digest(sorted([*r.identity, r.mapper_fold, r.theta_ac] for r in rows))
    _require(m["candidate_identity_digest"] == candidate_digest, "candidate identity digest mismatch")
    population = json.loads((root / "population.json").read_bytes())
    _require(population == {"counts": m["population"], "folds": m["folds"]}, "population artifact mismatch")
    games = sorted({r.game_id for r in rows})
    _require(m["population"]["games"] == len(games)
             and m["population"]["candidate_rows"] == len(rows)
             and m["population"]["theta_positive_rows"] == sum(r.theta_ac for r in rows)
             and m["population"]["candidate_pre"] == len({r.identity[:4] for r in rows})
             and set(m["folds"]) == {str(f) for f in FOLDS}, "candidate population/fold coverage mismatch")
    for fold in FOLDS:
        split = m["folds"][str(fold)]
        held = sorted({r.game_id for r in rows if r.mapper_fold == fold})
        train = sorted(set(games) - set(held))
        _require(held and split == {"held_out_game_ids": held, "held_out_game_digest": _digest(held),
            "training_game_ids": train, "training_game_digest": _digest(train),
            "held_out_candidate_rows": sum(r.mapper_fold == fold for r in rows)}, "fold definition mismatch")
    fits = json.loads((root / "fit_manifests.json").read_bytes())
    _require(len(fits) == len(FOLDS) and all(type(fit["mapper_fold"]) is int for fit in fits)
             and {fit["mapper_fold"] for fit in fits} == set(FOLDS),
             "fold fit manifest coverage mismatch")
    for fit in fits:
        split = m["folds"][str(fit["mapper_fold"])]
        _require(canonical_json_bytes(fit["model_spec"]) == canonical_json_bytes(MODEL_SPECS[3].record())
                 and fit["oof_q"] == oof.record() and fit["experiment_a"] == identity
                 and fit["training_game_ids"] == split["training_game_ids"]
                 and fit["training_game_digest"] == split["training_game_digest"]
                 and fit["model_digest"] == _digest({k: v for k, v in fit.items() if k != "model_digest"})
                 and fit["model_digest"] == m["fit_model_digests"][f"{MODEL_SPECS[3].name}/fold-{fit['mapper_fold']}"],
                 "fold fit identity mismatch")
    return artifact, {r.identity: r for r in rows}, oof, publication, evaluation_root


def _aligned_panel(loaded):
    base, reference, base_oof, publication, evaluation_root = loaded["A0"]
    for condition in CONDITIONS:
        artifact, rows, oof, pub, other_evaluation_root = loaded[condition]
        _require(set(rows) == set(reference), "candidate identity mismatch across conditions")
        _require(pub == publication and artifact.manifest["folds"] == base.manifest["folds"]
                 and artifact.manifest["population"] == base.manifest["population"]
                 and oof.paired_experiment_digest == base_oof.paired_experiment_digest
                 and oof.evaluation_seal_digest == base_oof.evaluation_seal_digest
                 and oof.evaluation_contract_digest == base_oof.evaluation_contract_digest,
                 "shared publication/preparation/seal/fold identity mismatch")
        _require(other_evaluation_root == evaluation_root, "shared resolved evaluation root mismatch")
        for identity, row in reference.items():
            other = rows[identity]
            _require((row.theta_ac, row.mapper_fold, row.game_oof_fold, row.prefix_digest)
                     == (other.theta_ac, other.mapper_fold, other.game_oof_fold, other.prefix_digest),
                     "candidate label/fold/prefix mismatch")
    _require(loaded["A2"][2].prediction_digest_by_fold == base_oof.prediction_digest_by_fold,
             "A0/A2 continuous Q prediction digests differ")
    games = tuple(sorted({r.game_id for r in reference.values()}))
    game_rows = {game: sorted(identity for identity in reference if identity[0] == game) for game in games}
    _require(all(len({reference[i].mapper_fold for i in ids}) == 1 for ids in game_rows.values()),
             "one game spans mapper folds")
    folds = np.array([reference[game_rows[g][0]].mapper_fold for g in games], dtype=np.int64)
    counts = np.array([len(game_rows[g]) for g in games], dtype=np.int64)
    log_sums, brier_sums = [], []
    for condition in CONDITIONS:
        rows = loaded[condition][1]
        log, brier = [], []
        for game in games:
            ids = game_rows[game]
            y = np.array([int(reference[i].theta_ac) for i in ids])
            p = np.array([rows[i].raw_probability for i in ids], dtype=np.float64)
            # Aggregate existing metric implementations; bootstrap mathematics stays in the kernels.
            log.append(_log_loss(y, p) * len(ids))
            brier.append(float(mean_squared_error(y, p)) * len(ids))
        log_sums.append(log)
        brier_sums.append(brier)
    return _GamePanel(games, folds, tuple(np.flatnonzero(folds == f) for f in FOLDS), counts,
                      np.asarray(log_sums), np.asarray(brier_sums))


def _bootstrap(panel):
    points = _weighted_metrics(panel, np.ones(len(panel.game_ids), dtype=np.int64))
    samples = np.empty((REPLICATES, len(CONDITIONS), len(METRICS)), dtype=np.float64)
    rng, draws = np.random.default_rng(SEED), sha256()
    for replicate in range(REPLICATES):
        multiplicities = _draw_game_multiplicities(panel, rng)
        draws.update(multiplicities.astype("<i8", copy=False).tobytes())
        samples[replicate] = _weighted_metrics(panel, multiplicities)
    def summary(point, values):
        return {metric: {"point": float(point[i]),
                         "ci95": list(percentile_interval(values[:, i], level=CONFIDENCE))}
                for i, metric in enumerate(METRICS)}
    return {"conditions": {name: summary(points[i], samples[:, i]) for i, name in enumerate(CONDITIONS)},
            "contrasts": {name: {"direction": f"{name} - A0",
                "metrics": summary(points[i] - points[0], samples[:, i] - samples[:, 0])}
                for i, name in enumerate(CONDITIONS) if i},
            "bootstrap": {"replicates": REPLICATES, "seed": SEED, "confidence": CONFIDENCE,
                "interval_method": "percentile_linear", "sampling_unit": "game",
                "stratification": "mapper_fold", "shared_across_conditions": True,
                "game_ids": list(panel.game_ids), "mapper_folds": panel.folds.tolist(),
                "draw_encoding": "game_multiplicities_int64_le_row_major",
                "draw_shape": [REPLICATES, len(panel.game_ids)], "shared_draw_digest": draws.hexdigest()}}


def analyze(storage_profile, paths, expected_digests, destination):
    _require(set(paths) == set(expected_digests) == set(CONDITIONS), "exactly A0/A1/A2 inputs required")
    source = _analysis_source()  # Dirty/uncommitted analysis never publishes a formal report.
    storage_root = _storage_root(storage_profile)
    output = _artifact_path(storage_root, destination)
    loaded = {name: _load(_artifact_path(storage_root, paths[name]), expected_digests[name], name, storage_root)
              for name in CONDITIONS}
    _require(all(not output.is_relative_to(path) for item in loaded.values()
                 for path in (item[0].path, item[4],
                              _artifact_path(storage_root, item[3]["publication_path"]))),
             "output cannot be inside immutable inputs")
    panel = _aligned_panel(loaded)
    statistics = _bootstrap(panel)
    _require(_analysis_source() == source, "analysis source changed during computation")
    base, _, oof, publication, evaluation_root = loaded["A0"]
    lines = ["# Experiment A downstream contrasts", "", f"Execution source: `{EXECUTION_REVISION}`.",
             f"Analysis source: `{source['source_revision']}`.", "",
             "Paired game-cluster bootstrap within mapper folds; conditional on fixed CV predictions.", "",
             "| Condition/contrast | Metric | Point | 95% CI |", "|---|---|---:|---|"]
    for group, results in (("conditions", statistics["conditions"]), ("contrasts", statistics["contrasts"])):
        for name, result in results.items():
            for metric, value in (result if group == "conditions" else result["metrics"]).items():
                lines.append(f"| {name if group == 'conditions' else result['direction']} | {metric} | "
                             f"{value['point']:.12g} | [{value['ci95'][0]:.12g}, {value['ci95'][1]:.12g}] |")
    return publish_artifact(output, manifest_fields={
        "artifact_type": "phase2_experiment_a_mapper_contrasts", "schema_version": VERSION,
        "inputs": {name: {"path": str(item[0].path), "mapper_artifact_digest": item[0].manifest_digest}
                   for name, item in loaded.items()},
        "conditions": {name: {
            "temporal_condition": item[0].manifest["experiment_a"]["temporal_condition"],
            "q_representation": item[0].manifest["experiment_a"]["q_representation"],
            "mapper_spec": item[0].manifest["experiment_a"]["mapper_spec"],
            "mapper_artifact_digest": item[0].manifest_digest} for name, item in loaded.items()},
        "evaluation_ownership": {"resolved_root": str(evaluation_root),
                                 "path_scope": "local_execution_provenance"},
        "execution_source_revision": EXECUTION_REVISION, "analysis_source": source,
        "paired_experiment_digest": oof.paired_experiment_digest,
        "common_oof_seal_digest": oof.evaluation_seal_digest,
        "oof_evaluation_contract_digest": oof.evaluation_contract_digest,
        "publication": publication, "candidate_identity_digest": base.manifest["candidate_identity_digest"],
        "candidate_count": len(loaded["A0"][1]), "folds": base.manifest["folds"],
        "q_prediction_digests": {name: item[2].record()["prediction_digest_by_fold"] for name, item in loaded.items()},
        "metric_contract": METRIC_CONTRACT,
        "metric_clipping": {"epsilon": PROBABILITY_EPSILON, "applies_to": "binary_log_loss_only"},
        "runtime": {"numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "statistics": statistics,
    }, files={"statistics.json": canonical_json_bytes(statistics),
              "report.md": ("\n".join(lines) + "\n").encode()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-profile", type=Path, required=True)
    for condition in CONDITIONS:
        parser.add_argument(f"--{condition.lower()}", type=Path, required=True)
        parser.add_argument(f"--{condition.lower()}-digest", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    artifact = analyze(args.storage_profile,
        {name: getattr(args, name.lower()) for name in CONDITIONS},
        {name: getattr(args, f"{name.lower()}_digest") for name in CONDITIONS}, args.destination)
    print(artifact.manifest_digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
