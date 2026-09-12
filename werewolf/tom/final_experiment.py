"""All-development final-fit identity; no evaluation input or partition."""

import subprocess
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from werewolf.artifact_io import canonical_json_bytes, publish_artifact, verify_artifact, sha256_bytes
from werewolf.development_publication import open_publication
from werewolf.tom.dataset import ExperimentCapacity
from werewolf.tom.experiment import (ExperimentConfig, TEMPORAL_CONDITIONS, RECOVERY_POLICY,
    configure_runtime, runtime_provenance, validate_runtime, read_json)
from werewolf.tom.final_capacity import derived_capacity, validate_final_pre
from werewolf.tom.model import MODEL_GRAPH, ObserverConditionedToM
from werewolf.tom.population import select_primary_population, PRIMARY_SELECTOR_VERSION
from werewolf.tom.scoring import SCORING_VERSION
from werewolf.canonical_collection.trajectory_evidence import BELIEF_OBSERVATION_SCHEMA_VERSION
from werewolf.tom.protocol import digest_fields, final_training_schedule
from werewolf.tom.state import publish_model_state, validate_model_state
from werewolf.tom.temporal import publish_temporal, TemporalCodeProvider

FINAL_EXPERIMENT_VERSION = "classic7_final_experiment_v1"
MANIFEST_NAME = "final_experiment_manifest.json"
FINAL_SEMANTICS = {"belief_observation_version": BELIEF_OBSERVATION_SCHEMA_VERSION,
    "population_selector_version": PRIMARY_SELECTOR_VERSION, "scoring_version": SCORING_VERSION,
    "target": "uniform_successful_support_or_uniform_six_non_self_v1",
    "loss": "game_balanced_cross_entropy", "inference_shift": 0}


def actual_clean_revision():
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("formal final experiment requires a clean implementation checkout")
    return git("rev-parse", "HEAD")


def numerical_seeds(config):
    return {
        "initialization_seed": int.from_bytes(digest_fields("classic7_final_paired_initial_v1", config.initialization_seed), "big") % 2**63,
        "rng_seed": int.from_bytes(digest_fields("classic7_final_paired_rng_v1", config.rng_seed), "big") % 2**32,
    }


@dataclass(frozen=True)
class VerifiedFinalExperiment:
    artifact: object
    config: ExperimentConfig

    @property
    def path(self):
        return self.artifact.path

    @property
    def digest(self):
        return self.artifact.manifest_digest

    @property
    def manifest(self):
        return self.artifact.manifest

    @property
    def runs_path(self):
        return self.path.parent.parent / "runs" / self.digest

    def schedule(self):
        expected = final_training_schedule(self.config.schedule_seed, self.manifest["game_ids"],
            self.config.rotation_cycles, self.config.game_batch_size)
        if read_json((self.path / "schedule.json").read_bytes()) != expected:
            raise ValueError("final schedule mismatch")
        return expected


def prepare_final_experiment(publication, config, destination):
    revision = actual_clean_revision()
    configure_runtime(config)
    publication = open_publication(publication.path)
    capacity = derived_capacity(config.max_seq_len)
    for game in publication.public_view.games:
        for prefix in game.authoritative_pre_prefixes:
            validate_final_pre(prefix, capacity)
    primary = select_primary_population(publication)
    game_ids = sorted(publication.public_view.game_ids)
    schedule = final_training_schedule(config.schedule_seed, game_ids, config.rotation_cycles, config.game_batch_size)
    inputs = {"schema_version": FINAL_EXPERIMENT_VERSION, "config": asdict(config),
              "model_graph": MODEL_GRAPH, "semantics": FINAL_SEMANTICS, "capacity": capacity, "recovery_policy": RECOVERY_POLICY,
              "publication_digest": publication.manifest_digest, "actual_source_revision": revision}
    protocol_digest = sha256_bytes(canonical_json_bytes(inputs))
    seeds = numerical_seeds(config)
    files = {"schedule.json": canonical_json_bytes(schedule), "primary.json": canonical_json_bytes(primary.rows)}
    with tempfile.TemporaryDirectory(prefix="classic7-final-") as directory:
        root = Path(directory)
        temporal = publish_temporal(root / "temporal", capacity["derived_day_capacity"], protocol_digest)
        provider = TemporalCodeProvider.implicit(root / "temporal", protocol_digest)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seeds["initialization_seed"])
            model = ObserverConditionedToM(ExperimentCapacity(config.max_seq_len), provider)
            initial = publish_model_state(root / "initial_state", model, {
                "purpose": "final_initial", "protocol_digest": protocol_digest,
                "initialization_seed": seeds["initialization_seed"], "model_graph": MODEL_GRAPH})
        files.update({p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()})
    publish_artifact(destination, manifest_name=MANIFEST_NAME, manifest_fields={
        "artifact_type": "classic7_final_experiment", "schema_version": FINAL_EXPERIMENT_VERSION,
        "protocol_inputs": inputs, "protocol_digest": protocol_digest,
        "publication_path": str(publication.path.resolve()), "publication_digest": publication.manifest_digest,
        "publication_id": publication.public_view.publication_id, "game_ids": game_ids,
        "primary_sidecar_digest": primary.metadata["role_sidecar_digest"],
        "temporal_conditions": list(TEMPORAL_CONDITIONS), "temporal_artifact_digest": temporal.manifest_digest,
        "paired_initial_state_digest": initial.artifact.manifest_digest,
        "schedule_digest": sha256_bytes(files["schedule.json"]), "optimizer_steps": schedule["optimizer_steps"],
        "seeds": seeds, "runtime": runtime_provenance(config)}, files=files)
    return open_final_experiment(destination)


def open_final_experiment(path):
    artifact = verify_artifact(path, manifest_name=MANIFEST_NAME,
        expected_artifact_type="classic7_final_experiment", expected_schema_version=FINAL_EXPERIMENT_VERSION)
    m = artifact.manifest
    fields = {"artifact_type", "schema_version", "protocol_inputs", "protocol_digest", "publication_path",
        "publication_digest", "publication_id", "game_ids", "primary_sidecar_digest", "temporal_conditions",
        "temporal_artifact_digest", "paired_initial_state_digest", "schedule_digest", "optimizer_steps",
        "seeds", "runtime", "file_table", "manifest_digest"}
    if set(m) != fields:
        raise ValueError("final experiment schema mismatch")
    inputs = m["protocol_inputs"]
    config = ExperimentConfig(**inputs["config"])
    if not isinstance(inputs["actual_source_revision"], str) or re.fullmatch(r"[0-9a-f]{40}", inputs["actual_source_revision"]) is None:
        raise ValueError("final actual source revision must be a Git commit identity")
    expected_inputs = {"schema_version": FINAL_EXPERIMENT_VERSION, "config": asdict(config),
        "model_graph": MODEL_GRAPH, "semantics": FINAL_SEMANTICS, "capacity": derived_capacity(config.max_seq_len), "recovery_policy": RECOVERY_POLICY,
        "publication_digest": m["publication_digest"], "actual_source_revision": inputs["actual_source_revision"]}
    if (inputs != expected_inputs or sha256_bytes(canonical_json_bytes(inputs)) != m["protocol_digest"]
        or m["temporal_conditions"] != list(TEMPORAL_CONDITIONS) or m["seeds"] != numerical_seeds(config)
        or not m["game_ids"] or m["game_ids"] != sorted(set(m["game_ids"]))):
        raise ValueError("final protocol identity mismatch")
    experiment = VerifiedFinalExperiment(artifact, config)
    schedule = experiment.schedule()
    if m["optimizer_steps"] != schedule["optimizer_steps"] or m["schedule_digest"] != sha256_bytes(canonical_json_bytes(schedule)):
        raise ValueError("final budget mismatch")
    provider = TemporalCodeProvider.implicit(experiment.path / "temporal", m["protocol_digest"])
    if provider.artifact_digest != m["temporal_artifact_digest"] or len(provider.day) != config.max_seq_len // 4 + 1:
        raise ValueError("final temporal capacity mismatch")
    initial = validate_model_state(experiment.path / "initial_state")
    if (initial.artifact.manifest_digest != m["paired_initial_state_digest"] or initial.artifact.manifest["provenance"] != {
        "purpose": "final_initial", "protocol_digest": m["protocol_digest"],
        "initialization_seed": m["seeds"]["initialization_seed"], "model_graph": MODEL_GRAPH}):
        raise ValueError("final initial-state identity mismatch")
    expected_files = {"schedule.json", "primary.json", "initial_state/manifest.json", "initial_state/tensors.bin",
        "temporal/phase_codebook.manifest.json", "temporal/phase_codebook.bin",
        "temporal/canonical_day_code_table.bin", "temporal/canonical_day_code_table.manifest.json"}
    if set(m["file_table"]) != expected_files:
        raise ValueError("final experiment inventory mismatch")
    return experiment


def final_training_inputs(experiment):
    validate_runtime(experiment)
    if actual_clean_revision() != experiment.manifest["protocol_inputs"]["actual_source_revision"]:
        raise ValueError("final implementation revision changed")
    publication = open_publication(experiment.manifest["publication_path"])
    if (publication.manifest_digest != experiment.manifest["publication_digest"]
        or publication.public_view.publication_id != experiment.manifest["publication_id"]
        or sorted(publication.public_view.game_ids) != experiment.manifest["game_ids"]):
        raise ValueError("final development provenance mismatch")
    primary = select_primary_population(publication)
    if (primary.metadata["role_sidecar_digest"] != experiment.manifest["primary_sidecar_digest"]
        or primary.rows != read_json((experiment.path / "primary.json").read_bytes())):
        raise ValueError("final training Primary mismatch")
    for game in publication.public_view.games:
        for prefix in game.authoritative_pre_prefixes:
            validate_final_pre(prefix, experiment.manifest["protocol_inputs"]["capacity"])
    return publication, primary
