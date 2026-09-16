"""Frozen paper study contract and fail-closed Git attestation; no experiment runner."""

import json
import os
from pathlib import Path
import re
import subprocess

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes, publish_artifact, verify_artifact
from werewolf.tom.protocol import BOOTSTRAP_VERSION

STUDY_VERSION = "classic7_paper_tom_study_v1"
CONTRACT = {
    "schema_version": STUDY_VERSION,
    "source_binding": "clean_git_head_at_study_prepare_v1",
    "publication_id": "paper-development-qwen35-9b-1500-v1",
    "model_families": ["full_observer_conditioned", "observer_agnostic_public_history"],
    "temporal_conditions": ["implicit", "explicit_day_phase"], "fold_count": 5,
    "terminal_lineages": 20, "training_population": "non_wolf_alive",
    "training_loss": "game_balanced_cross_entropy", "primary_metric": "game_macro_kl",
    "secondary_metric": "game_macro_total_variation",
    "effects": {
        "Delta_uniform": {"left": "uniform_kl", "right": "model_kl", "population": "primary", "models": "both", "temporal_conditions": "both"},
        "Delta_ToM": {"left": "agnostic_kl", "right": "full_kl", "population": "primary", "temporal_conditions": "both_required"},
        "Delta_temp": {"left": "full_implicit_kl", "right": "full_explicit_kl", "population": "primary"},
        "Delta_stress": {"left": "full_all_alive_kl", "right": "full_primary_kl", "temporal_conditions": "both"}},
    "bootstrap": {"version": BOOTSTRAP_VERSION, "replicates": 10000, "confidence": .95,
        "sampling_unit": "game", "aggregation": "paired_game_macro_difference",
        "interval_method": "percentile_linear", "draw_policy": "reuse_same_frozen_indices"},
    "shared_controls": ["publication", "folds", "training_games", "game_schedule", "rotation_budget",
        "optimizer_hyperparameters", "primary_population", "q_targets", "max_seq_len", "bootstrap_plan"],
    "initialization": {"shared_parameters": "exact_full_fold_initial_tensor_copy",
        "query_seed_domain": "baseline_shared_query_v1", "query_distribution": "Normal(0,0.02)",
        "temporal_pairing": "same_baseline_initial_state"},
}


def attest_source():
    root = Path(__file__).resolve().parents[2]
    # Git environment overrides must not redirect attestation to a different repository/index.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(root), *args], env=env,
                                           stderr=subprocess.PIPE, text=True).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError("Git source identity unavailable") from error
    if Path(git("rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("Git repository root mismatch")
    head = git("rev-parse", "HEAD")
    if re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise ValueError("Git source identity is not a full commit hash")
    if git("status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"):
        raise ValueError("paper study requires a clean Git worktree and index")
    # Same implementation inventory/hash as experiment.runtime_provenance; neither replaces nor weakens it.
    sources = {p.relative_to(root).as_posix(): sha256_bytes(p.read_bytes())
               for p in sorted((root / "werewolf").rglob("*.py"))}
    sources["run_random.py"] = sha256_bytes((root / "run_random.py").read_bytes())
    git("ls-files", "--error-unmatch", "--", *sources)
    return {"source_revision": head, "implementation_digest": sha256_bytes(canonical_json_bytes(sources))}


def prepare_contract(config_path, destination):
    config = json.loads(Path(config_path).read_bytes())
    if canonical_json_bytes(config) != canonical_json_bytes(CONTRACT):
        raise ValueError("paper study contract differs from frozen design")
    source = attest_source()
    return publish_artifact(destination, manifest_fields={
        "artifact_type": "paper_tom_study_contract", "schema_version": STUDY_VERSION,
        "contract": config, "source": source}, files={})


def open_contract(path):
    artifact = verify_artifact(path, expected_artifact_type="paper_tom_study_contract",
                               expected_schema_version=STUDY_VERSION)
    m = artifact.manifest
    if (set(m) != {"artifact_type", "schema_version", "contract", "source", "file_table", "manifest_digest"}
            or canonical_json_bytes(m["contract"]) != canonical_json_bytes(CONTRACT) or m["file_table"]):
        raise ValueError("paper study contract schema/design mismatch")
    current = attest_source()
    if not isinstance(m["source"], dict) or set(m["source"]) != set(current):
        raise ValueError("paper study source schema mismatch")
    if m["source"]["source_revision"] != current["source_revision"]:
        raise ValueError("current HEAD differs from study source_revision")
    if m["source"]["implementation_digest"] != current["implementation_digest"]:
        raise ValueError("paper study implementation digest mismatch")
    return artifact
