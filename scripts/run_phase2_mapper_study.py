"""Run the frozen Phase-2 mapper development study on the sealed server inputs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import sklearn

from werewolf.artifact_io import (
    canonical_json_bytes, canonical_jsonl_bytes, publish_artifact, sha256_bytes,
)
from werewolf.cli import _artifact_path, _storage_root
from werewolf import phase2_offline as offline
from werewolf.phase2_mapper import (
    FEATURE_CONTRACT_VERSION, FOLDS, MODEL_SPECS, OOFProvenance,
    run_mapper_cv, validate_rows,
)
from werewolf.phase2_mapper_selection import (
    CI_LEVEL, DEFAULT_REPLICATES, DEFAULT_SEED, MODEL_ORDER,
    bootstrap_compare, select_from_comparison,
)


STUDY_NAME = "paper-phase2-mapper-development-v1"
STUDY_VERSION = "phase2_mapper_development_study_v1"
SOURCE_BRANCH = "twd/mainline"
PUBLICATION = Path("/data/yuxiao/Untrusted_Network_Simulation/publications/"
                   "paper-development-qwen35-9b-1500-v1")
EVALUATION = Path("/data/yuxiao/Untrusted_Network_Simulation/paper-studies/"
                  "evaluations/0b8c1d7220aae6fcf577038f8d930c97d6453b7f45616ed0a19c817e7a80b956")
OUTPUT_RELATIVE = Path("paper-studies/mapper-development") / STUDY_NAME
EXPECTED_POSITIVES = 5412
SOURCE_FILES = (
    "scripts/run_phase2_mapper_study.py",
    "werewolf/phase2_offline.py",
    "werewolf/phase2_mapper.py",
    "werewolf/phase2_mapper_selection.py",
    "docs/research/phase2-offline-data-contract.md",
    "docs/research/phase2-mapper-protocol.md",
    "docs/research/phase2-mapper-study-runbook.md",
    "setup.py",
)


class MapperStudyError(ValueError):
    """The formal study is not bound to the frozen source or population."""


def _git(root: Path, *arguments: str) -> bytes:
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    try:
        return subprocess.check_output(["git", "-C", str(root), *arguments],
                                       env=env, stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as error:
        raise MapperStudyError(f"Git source attestation failed: {' '.join(arguments)}") from error


def attest_source() -> dict:
    root = Path(__file__).resolve().parents[1]
    if Path(_git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve() != root:
        raise MapperStudyError("runner does not belong to the Git repository root")
    commit = _git(root, "rev-parse", "HEAD").decode().strip()
    branch = _git(root, "branch", "--show-current").decode().strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None or branch != SOURCE_BRANCH:
        raise MapperStudyError("unexpected Git commit identity or branch")
    tracked_status = _git(root, "status", "--porcelain=v1", "--untracked-files=no",
                          "--ignore-submodules=none").decode().splitlines()
    if tracked_status:
        raise MapperStudyError("tracked Git files must be clean before formal study")
    _git(root, "ls-files", "--error-unmatch", "--", *SOURCE_FILES)
    source_hashes = {}
    for relative in SOURCE_FILES:
        content = (root / relative).read_bytes()
        if _git(root, "show", f"HEAD:{relative}") != content:
            raise MapperStudyError(f"study source differs from HEAD: {relative}")
        source_hashes[relative] = sha256_bytes(content)
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all",
                  "--ignore-submodules=none").decode().splitlines()
    return {"commit": commit, "branch": branch, "tracked_clean": True,
            "dirty": bool(status), "status_porcelain": status,
            "source_sha256": source_hashes}


def preflight(storage_profile: Path | str) -> tuple[Path, dict]:
    source = attest_source()
    root = _storage_root(storage_profile)
    destination = _artifact_path(root, OUTPUT_RELATIVE)
    if os.path.lexists(destination):
        raise MapperStudyError(f"study destination already exists: {destination}")
    return destination, source


def validate_population(layer: offline.OfflinePhase2, oof: OOFProvenance) -> dict:
    rows = layer.candidates
    game_folds = validate_rows(rows, oof)
    counts = {
        "games": len(game_folds),
        "candidate_pre": layer.pre_count,
        "candidate_rows": len(rows),
        "theta_positive_rows": sum(row.theta_ac for row in rows),
        "post_day_states": len(layer.post_days),
        "skipped_no_candidate_pk_pre": layer.skipped_no_candidate_pk_pre_count,
    }
    expected = {
        "games": offline.EXPECTED_GAMES,
        "candidate_pre": offline.EXPECTED_PRE,
        "candidate_rows": offline.EXPECTED_CANDIDATES,
        "theta_positive_rows": EXPECTED_POSITIVES,
        "post_day_states": offline.EXPECTED_POST_DAYS,
        "skipped_no_candidate_pk_pre": offline.EXPECTED_SKIPPED_NO_CANDIDATE_PK_PRE,
    }
    if counts != expected:
        raise MapperStudyError(f"formal population mismatch before fitting: {counts}")
    if len({(row.game_id, row.boundary_id, row.acting_wolf, row.phase)
            for row in rows}) != layer.pre_count:
        raise MapperStudyError("candidate PRE identity count mismatch")
    fold_games = {}
    for fold in FOLDS:
        held_out = sorted(game for game, assigned in game_folds.items() if assigned == fold)
        training = sorted(game for game, assigned in game_folds.items() if assigned != fold)
        if not held_out or not training or set(held_out) & set(training):
            raise MapperStudyError("invalid whole-game fold coverage")
        fold_games[str(fold)] = {"held_out_game_ids": held_out,
            "held_out_game_digest": sha256_bytes(canonical_json_bytes(held_out)),
            "training_game_ids": training,
            "training_game_digest": sha256_bytes(canonical_json_bytes(training)),
            "held_out_candidate_rows": sum(row.fold == fold for row in rows)}
    return {"counts": counts, "folds": fold_games}


def validate_cv(result, rows: tuple[offline.CandidateRow, ...], population: dict) -> None:
    expected = {row.game_id: row.fold for row in rows}
    identities = {(row.game_id, row.boundary_id, row.acting_wolf,
                   row.phase, row.candidate_j): row for row in rows}
    if len(identities) != len(rows) or len(result.fit_manifests) != len(MODEL_SPECS) * len(FOLDS):
        raise MapperStudyError("candidate identity or fit manifest count mismatch")
    by_model = {name: {} for name in MODEL_ORDER}
    for prediction in result.predictions:
        if prediction.model_spec not in by_model:
            raise MapperStudyError("unexpected fitted model prediction")
        selected = by_model[prediction.model_spec]
        if prediction.identity in selected or prediction.identity not in identities:
            raise MapperStudyError("duplicate or unexpected cross-fitted candidate identity")
        source_row = identities[prediction.identity]
        if (prediction.theta_ac != source_row.theta_ac
                or prediction.mapper_fold != expected[prediction.game_id]
                or prediction.game_oof_fold != prediction.mapper_fold):
            raise MapperStudyError("prediction label or held-out game fold mismatch")
        selected[prediction.identity] = prediction
    if any(set(selected) != set(identities) or len(selected) != offline.EXPECTED_CANDIDATES
           for selected in by_model.values()):
        raise MapperStudyError("cross-fitted prediction coverage is not 21638/21638")
    seen = set()
    for manifest in result.fit_manifests:
        spec = manifest["model_spec"]["name"]
        fold = manifest["mapper_fold"]
        key = (spec, fold)
        if spec not in MODEL_ORDER or fold not in FOLDS or key in seen:
            raise MapperStudyError("duplicate or unexpected fold fit manifest")
        seen.add(key)
        panel = population["folds"][str(fold)]
        if (manifest["training_game_ids"] != panel["training_game_ids"]
                or manifest["training_game_digest"] != panel["training_game_digest"]
                or set(manifest["training_game_ids"]) & set(panel["held_out_game_ids"])):
            raise MapperStudyError("held-out game entered mapper training")
        if any(prediction.model_digest != manifest["model_digest"]
               for prediction in by_model[spec].values()
               if prediction.mapper_fold == fold):
            raise MapperStudyError("prediction model digest differs from fold fit")
    if seen != {(name, fold) for name in MODEL_ORDER for fold in FOLDS}:
        raise MapperStudyError("missing model-fold fit manifest")
    if set(result.metrics_by_spec) != set(MODEL_ORDER):
        raise MapperStudyError("frozen metric coverage mismatch")


def _report(source: dict, population: dict, result, comparison, decision) -> bytes:
    lines = [f"# {STUDY_NAME}", "", f"Source: `{source['commit']}` on `{source['branch']}`.",
             f"Candidate population: {population['counts']['candidate_rows']} rows from "
             f"{population['counts']['games']} games.", "",
             "Development CV only; not an independent final test or deployment calibration.",
             "", "## Model metrics", "",
             "| Model | Pooled LL | Pooled Brier | Game-macro LL | Game-macro Brier | AUROC | AUPRC | Calibration intercept | Calibration slope |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def fmt(value):
        return "NA" if value is None else f"{value:.8g}"
    def interval(value):
        return "NA" if value is None else f"[{fmt(value[0])}, {fmt(value[1])}]"
    for name in MODEL_ORDER:
        metric = result.metrics_by_spec[name]
        lines.append("| " + " | ".join((name, fmt(metric.pooled_log_loss),
            fmt(metric.pooled_brier), fmt(metric.game_macro_log_loss),
            fmt(metric.game_macro_brier), fmt(metric.auroc), fmt(metric.auprc),
            fmt(metric.calibration_intercept), fmt(metric.calibration_slope))) + " |")
    lines += ["", "## Prespecified paired comparisons", "",
              "Delta = complex minus simple; negative pooled log loss favors complex.", "",
              "| Pair | Role | Delta pooled LL (95% CI) | Delta pooled Brier (95% CI) | Delta game-macro LL (95% CI) | Delta game-macro Brier (95% CI) | Gate |",
              "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for pair in comparison.pairs:
        lines.append("| " + " | ".join((
            f"{pair.complex_model} − {pair.simple_model}", pair.role,
            f"{fmt(pair.delta_pooled_log_loss)} {interval((pair.ci_lower, pair.ci_upper))}",
            f"{fmt(pair.delta_pooled_brier)} {interval(pair.ci_pooled_brier)}",
            f"{fmt(pair.delta_game_macro_log_loss)} {interval(pair.ci_game_macro_log_loss)}",
            f"{fmt(pair.delta_game_macro_brier)} {interval(pair.ci_game_macro_brier)}",
            "development-supported" if pair.development_supported else
            "not development-supported / inconclusive")) + " |")
    lines += ["", f"Primary selected specification: **{decision.primary_model}**.",
              f"Sequential stop: {decision.stopped_at or 'none'}.",
              "Development-supported alternatives: " +
              (", ".join(decision.development_supported_alternatives) or "none") + ".",
              "", f"Bootstrap: {comparison.replicates} paired fold-stratified "
              f"game-cluster replicates, seed {comparison.seed}, percentile "
              f"{comparison.ci_level:.0%} CI. No mapper refitting per replicate.",
              "Intervals describe evaluation uncertainty conditional on fitted CV models.",
              "See `metrics.json`, `bootstrap.json`, `selection.json`, "
              "`fit_manifests.json`, and the four prediction JSONL files for exact values.", ""]
    return ("\n".join(lines)).encode("utf-8")


def run_study(storage_profile: Path | str):
    destination, source = preflight(storage_profile)
    oof = OOFProvenance.from_evaluation_root(EVALUATION)
    layer = offline.build_development_layer(PUBLICATION, EVALUATION)
    population = validate_population(layer, oof)  # All six counts before fitting.
    result = run_mapper_cv(layer.candidates, oof)
    validate_cv(result, layer.candidates, population)
    comparison = bootstrap_compare(result.predictions,
                                   replicates=DEFAULT_REPLICATES, seed=DEFAULT_SEED)
    for name in MODEL_ORDER:
        evaluated, compared = result.metrics_by_spec[name], comparison.points[name]
        if (evaluated.candidate_count != compared.candidate_count
                or evaluated.game_count != compared.game_count
                or any(not math.isclose(getattr(evaluated, field), getattr(compared, field),
                                        rel_tol=1e-10, abs_tol=1e-12)
                       for field in ("pooled_log_loss", "pooled_brier",
                                     "game_macro_log_loss", "game_macro_brier"))):
            raise MapperStudyError("mapper evaluation and bootstrap point metrics differ")
    decision = select_from_comparison(comparison)
    files = {
        "population.json": canonical_json_bytes(population),
        "fit_manifests.json": canonical_json_bytes(result.fit_manifests),
        "metrics.json": canonical_json_bytes({name: asdict(result.metrics_by_spec[name])
                                               for name in MODEL_ORDER}),
        "bootstrap.json": canonical_json_bytes(asdict(comparison)),
        "selection.json": canonical_json_bytes({
            "primary_model": decision.primary_model,
            "stopped_at": decision.stopped_at,
            "development_supported_alternatives": decision.development_supported_alternatives,
        }),
        "report.md": _report(source, population, result, comparison, decision),
    }
    for name in MODEL_ORDER:
        files[f"cross_fitted_predictions/{name}.jsonl"] = canonical_jsonl_bytes(
            asdict(prediction) for prediction in result.predictions
            if prediction.model_spec == name)
    artifact = publish_artifact(destination, manifest_fields={
        "artifact_type": "phase2_mapper_development_study",
        "schema_version": STUDY_VERSION,
        "study_name": STUDY_NAME,
        "source": source,
        "inputs": {"publication_path": str(PUBLICATION),
                   "publication_digest": offline.PUBLICATION_DIGEST,
                   "role_sidecar_digest": offline.ROLE_SIDECAR_DIGEST,
                   "fold_manifest_digest": offline.FOLD_MANIFEST_DIGEST,
                   "storage_profile_path": str(Path(storage_profile).resolve()),
                   "storage_profile_sha256": sha256_bytes(Path(storage_profile).read_bytes()),
                   "oof_evaluation_path": str(EVALUATION),
                   "oof": oof.record()},
        "protocol": {"offline_contract_source_sha256": source["source_sha256"]["werewolf/phase2_offline.py"],
                     "offline_contract_document_sha256": source["source_sha256"]["docs/research/phase2-offline-data-contract.md"],
                     "feature_contract_version": FEATURE_CONTRACT_VERSION,
                     "mapper_protocol_document_sha256": source["source_sha256"]["docs/research/phase2-mapper-protocol.md"],
                     "mapper_source_sha256": source["source_sha256"]["werewolf/phase2_mapper.py"],
                     "selection_source_sha256": source["source_sha256"]["werewolf/phase2_mapper_selection.py"],
                     "model_specs": [spec.record() for spec in MODEL_SPECS],
                     "bootstrap": {"replicates": DEFAULT_REPLICATES,
                                   "seed": DEFAULT_SEED, "ci_level": CI_LEVEL,
                                   "method": "paired_fold_stratified_game_cluster_percentile_linear"}},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scikit_learn": sklearn.__version__},
        "population": population["counts"],
        "folds": population["folds"],
        "fit_model_digests": {f"{m['model_spec']['name']}/fold-{m['mapper_fold']}":
                              m["model_digest"] for m in result.fit_manifests},
        "selection": {"primary_model": decision.primary_model,
                      "stopped_at": decision.stopped_at,
                      "development_supported_alternatives":
                      decision.development_supported_alternatives},
    }, files=files)
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-profile", type=Path, required=True,
                        help="storage profile JSON, normally configs/server.json")
    parser.add_argument("--preflight", action="store_true",
                        help="check committed source and unused destination; do not read data or fit")
    args = parser.parse_args(argv)
    if args.preflight:
        destination, source = preflight(args.storage_profile)
        print(canonical_json_bytes({"destination": str(destination),
                                    "source": source}).decode())
        return 0
    artifact = run_study(args.storage_profile)
    print(canonical_json_bytes({"study_path": str(artifact.path),
                                "manifest_digest": artifact.manifest_digest}).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
