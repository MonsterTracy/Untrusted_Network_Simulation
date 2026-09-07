"""Immutable experiment preparation and stage-gated artifact access."""

import json
import math
import os
import platform
import tempfile
import importlib.util
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import transformers

from werewolf.artifact_io import (canonical_json_bytes, canonical_jsonl_bytes, open_artifact_envelope,
    publish_artifact, read_artifact_file, sha256_bytes)
from werewolf.tom.dataset import ExperimentCapacity
from werewolf.tom.model import MODEL_GRAPH, ObserverConditionedToM
from werewolf.tom.population import select_primary_population, build_all_alive_eligibility
from werewolf.tom.protocol import bootstrap_indices, digest_fields, training_schedule, BOOTSTRAP_VERSION
from werewolf.tom.scoring import SCORING_VERSION
from werewolf.tom.state import publish_model_state
from werewolf.tom.temporal import publish_temporal, TemporalCodeProvider

EXPERIMENT_VERSION = "classic7_experiment_v1"
RECOVERY_POLICY = "deterministic_step_boundary_resume_v1"
TEMPORAL_CONDITIONS = ("implicit", "explicit_day_phase")


@dataclass(frozen=True)
class ExperimentConfig:
    max_seq_len: int
    learning_rate: float
    weight_decay: float
    adam_betas: tuple[float, float]
    adam_eps: float
    game_batch_size: int
    rotation_cycles: int
    initialization_seed: int
    rng_seed: int
    schedule_seed: int
    bootstrap_seed: int
    bootstrap_replicates: int
    confidence_level: float
    recovery_cadence: int
    device: str
    torch_num_threads: int
    source_revision: str
    optimizer: str
    scheduler: str
    deterministic_algorithms: bool

    def __post_init__(self):
        for name in ("max_seq_len", "game_batch_size", "rotation_cycles", "bootstrap_replicates", "recovery_cadence", "torch_num_threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be an explicit positive integer")
        for name in ("initialization_seed", "rng_seed", "schedule_seed", "bootstrap_seed"):
            if type(getattr(self, name)) is not int or not 0 <= getattr(self, name) < 2**63:
                raise ValueError(f"invalid {name}")
        for name in ("learning_rate", "adam_eps", "weight_decay", "confidence_level"):
            if type(getattr(self, name)) not in (int, float) or not math.isfinite(getattr(self, name)):
                raise ValueError(f"invalid {name}")
        if self.learning_rate <= 0 or self.adam_eps <= 0 or self.weight_decay < 0 or not 0 < self.confidence_level < 1:
            raise ValueError("invalid numeric experiment configuration")
        if len(self.adam_betas) != 2 or any(not 0 <= b < 1 for b in self.adam_betas):
            raise ValueError("invalid Adam betas")
        if self.optimizer != "adamw" or self.scheduler != "constant" or self.deterministic_algorithms is not True:
            raise ValueError("only declared deterministic AdamW/constant-step protocol is supported")
        if self.device not in ("cpu", "cuda") or not self.source_revision:
            raise ValueError("unsupported device or missing source provenance")


def configure_runtime(config):
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("declared CUDA backend is unavailable")
    torch.set_num_threads(config.torch_num_threads)
    torch.use_deterministic_algorithms(True)


def runtime_provenance(config):
    root = Path(__file__).resolve().parents[2]
    sources = {path.relative_to(root).as_posix(): sha256_bytes(path.read_bytes())
               for path in sorted((root / "werewolf").rglob("*.py"))}
    sources["run_random.py"] = sha256_bytes(Path(importlib.util.find_spec("run_random").origin).read_bytes())
    implementation_digest = sha256_bytes(canonical_json_bytes(sources))
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "transformers": transformers.__version__, "platform": platform.platform(), "device": config.device,
            "device_name": torch.cuda.get_device_name(0) if config.device == "cuda" else platform.machine(),
            "torch_num_threads": config.torch_num_threads, "deterministic_algorithms": True,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "default_dtype": str(torch.get_default_dtype()),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "implementation_digest": implementation_digest,
            "source_revision": config.source_revision}


def validate_runtime(experiment):
    configure_runtime(experiment.config)
    if runtime_provenance(experiment.config) != experiment.manifest["runtime"]:
        raise ValueError("experiment runtime environment/source mismatch")


def read_json(data):
    value = json.loads(data)
    if canonical_json_bytes(value) != data:
        raise ValueError("noncanonical JSON payload")
    return value


@dataclass(frozen=True)
class VerifiedExperiment:
    envelope: object
    config: ExperimentConfig

    @property
    def path(self):
        return self.envelope.path

    @property
    def digest(self):
        return self.envelope.manifest_digest

    @property
    def manifest(self):
        return self.envelope.manifest

    @property
    def runs_path(self):
        return self.path.parent.parent / "runs" / self.digest

    def file(self, relative):
        return read_artifact_file(self.envelope, relative)

    def fold(self, index):
        if type(index) is not int or not 0 <= index < 5:
            raise ValueError("fold must be in 0..4")
        return self.manifest["folds"][index]

    def schedule(self, index):
        fold = self.fold(index)
        value = read_json(self.file(f"folds/{index}/schedule_manifest.json"))
        expected = training_schedule(self.manifest["protocol_digest"], index, fold["training_game_ids"],
            self.config.rotation_cycles, self.config.game_batch_size)
        if value != expected or self.file(f"folds/{index}/training_schedule.jsonl") != canonical_jsonl_bytes(expected["batches"]):
            raise ValueError("schedule/protocol mismatch")
        if sha256_bytes(canonical_json_bytes(value)) != fold["schedule_digest"]:
            raise ValueError("schedule digest mismatch")
        return value


def _mask_record(eligibility, game, parent_digest):
    rows = eligibility.rows[game.game_id]
    identity = [{"boundary_id": p.boundary_id, "prefix_digest": p.prefix_digest,
                 "observer_ids": list(range(7))} for p in game.authoritative_pre_prefixes]
    return {"schema_version": "classic7_eligibility_v1", **eligibility.metadata,
            "parent_digest": parent_digest, "game_id": game.game_id, "bundle_digest": game.bundle_digest,
            "row_identity": identity, "row_identity_digest": sha256_bytes(canonical_json_bytes(identity)), "rows": rows}


def prepare_experiment(publication, config: ExperimentConfig, destination):
    configure_runtime(config)
    view = publication.public_view
    if config.max_seq_len < view.max_structured_token_count:
        raise ValueError("experiment capacity is smaller than complete publication history")
    primary = select_primary_population(publication)
    all_alive = build_all_alive_eligibility(view)
    inputs = {"publication_digest": publication.manifest_digest,
              "fold_manifest_digest": view.fold_manifest.manifest_digest,
              "development_game_set_digest": view.development_game_set_digest,
              "model_graph": MODEL_GRAPH, "config": asdict(config), "scoring_version": SCORING_VERSION,
              "recovery_policy": RECOVERY_POLICY, "bootstrap_version": BOOTSTRAP_VERSION}
    protocol_digest = sha256_bytes(canonical_json_bytes(inputs))
    games = {g.game_id: g for g in view.games}
    files = {}
    folds = []
    with tempfile.TemporaryDirectory(prefix="classic7-experiment-") as temporary:
        root = Path(temporary).resolve()
        temporal = publish_temporal(root / "temporal", view.max_observed_day, protocol_digest)
        provider = TemporalCodeProvider.implicit(root / "temporal", protocol_digest)
        for fold in view.fold_manifest.folds:
            index = fold.fold_index
            training = [g for g in view.game_ids if g not in fold.game_ids]
            schedule = training_schedule(protocol_digest, index, training, config.rotation_cycles, config.game_batch_size)
            files[f"folds/{index}/schedule_manifest.json"] = canonical_json_bytes(schedule)
            files[f"folds/{index}/training_schedule.jsonl"] = canonical_jsonl_bytes(schedule["batches"])
            for partition, ids, masks in (("training_primary", training, primary),
                    ("held_out_primary", fold.game_ids, primary), ("held_out_all_alive", fold.game_ids, all_alive)):
                for game_id in ids:
                    files[f"folds/{index}/population/{partition}/{game_id}.json"] = canonical_json_bytes(_mask_record(masks, games[game_id], protocol_digest))
            initial_seed = int.from_bytes(digest_fields("paired_initial_v1", config.initialization_seed, index), "big") % 2**63
            torch.manual_seed(initial_seed)
            model = ObserverConditionedToM(ExperimentCapacity(config.max_seq_len), provider)
            state = publish_model_state(root / "folds" / str(index) / "initial_state", model,
                {"purpose": "initial", "protocol_digest": protocol_digest, "fold": index, "initialization_seed": initial_seed, "model_graph": MODEL_GRAPH})
            folds.append({"fold": index, "training_game_ids": training, "held_out_game_ids": list(fold.game_ids),
                "schedule_digest": sha256_bytes(canonical_json_bytes(schedule)), "optimizer_steps": schedule["optimizer_steps"],
                "paired_initial_state_digest": state.artifact.manifest_digest, "initialization_seed": initial_seed})
        for file in root.rglob("*"):
            if file.is_file():
                files[file.relative_to(root).as_posix()] = file.read_bytes()
    ordered_games = sorted(view.game_ids)
    indices = bootstrap_indices(ordered_games, config.bootstrap_replicates, config.bootstrap_seed)
    files["bootstrap/game_cluster_bootstrap_indices.bin"] = indices.tobytes()
    files["bootstrap/game_cluster_bootstrap_plan.manifest.json"] = canonical_json_bytes({
        "schema_version": BOOTSTRAP_VERSION, "parent_digest": protocol_digest, "game_ids": ordered_games,
        "shape": list(indices.shape), "dtype": "<i4", "layout": "C", "seed": config.bootstrap_seed,
        "digest": sha256_bytes(indices.tobytes())})
    publish_artifact(destination, manifest_name="experiment_manifest.json", manifest_fields={
        "artifact_type": "classic7_experiment", "schema_version": EXPERIMENT_VERSION,
        "protocol_inputs": inputs, "protocol_digest": protocol_digest, "publication_path": str(publication.path.resolve()),
        "publication_id": view.publication_id,
        "publication_digest": publication.manifest_digest, "fold_manifest_digest": view.fold_manifest.manifest_digest,
        "temporal_conditions": list(TEMPORAL_CONDITIONS), "temporal_artifact_digest": temporal.manifest_digest,
        "max_structured_token_count": view.max_structured_token_count, "max_observed_day": view.max_observed_day,
        "folds": folds, "recovery_policy": RECOVERY_POLICY, "runtime": runtime_provenance(config),
        "primary_sidecar_digest": primary.metadata["role_sidecar_digest"], "game_ids": ordered_games,
    }, files=files)
    experiment = open_experiment(destination)
    validate_experiment_preflight(experiment)
    return experiment


def open_experiment(path):
    envelope = open_artifact_envelope(path, expected_artifact_type="classic7_experiment",
        expected_schema_version=EXPERIMENT_VERSION, manifest_name="experiment_manifest.json")
    m = envelope.manifest
    expected_fields = {"artifact_type", "schema_version", "protocol_inputs", "protocol_digest", "publication_path",
        "publication_digest", "fold_manifest_digest", "temporal_conditions", "temporal_artifact_digest",
        "max_structured_token_count", "max_observed_day", "folds", "recovery_policy", "runtime",
        "primary_sidecar_digest", "game_ids", "file_table", "manifest_digest", "publication_id"}
    if set(m) != expected_fields:
        raise ValueError("experiment manifest fields mismatch")
    inputs = m["protocol_inputs"]
    if set(inputs) != {"publication_digest", "fold_manifest_digest", "development_game_set_digest", "model_graph", "config", "scoring_version", "recovery_policy", "bootstrap_version"}:
        raise ValueError("experiment protocol inputs mismatch")
    config = ExperimentConfig(**inputs["config"])
    if (sha256_bytes(canonical_json_bytes(inputs)) != m["protocol_digest"] or inputs["model_graph"] != MODEL_GRAPH
        or inputs["scoring_version"] != SCORING_VERSION or inputs["bootstrap_version"] != BOOTSTRAP_VERSION
        or m["temporal_conditions"] != list(TEMPORAL_CONDITIONS) or inputs["recovery_policy"] != RECOVERY_POLICY
        or m["recovery_policy"] != RECOVERY_POLICY or inputs["publication_digest"] != m["publication_digest"]
        or inputs["fold_manifest_digest"] != m["fold_manifest_digest"]):
        raise ValueError("experiment protocol/parent mismatch")
    if config.max_seq_len < m["max_structured_token_count"]:
        raise ValueError("experiment capacity violation")
    if len(m["folds"]) != 5 or [f["fold"] for f in m["folds"]] != list(range(5)):
        raise ValueError("experiment requires exactly five folds")
    held_out = [g for f in m["folds"] for g in f["held_out_game_ids"]]
    if sorted(held_out) != m["game_ids"] or len(set(held_out)) != len(held_out):
        raise ValueError("fold game coverage mismatch")
    for f in m["folds"]:
        if set(f) != {"fold", "training_game_ids", "held_out_game_ids", "schedule_digest", "optimizer_steps", "paired_initial_state_digest", "initialization_seed"}:
            raise ValueError("fold fields mismatch")
        if (not f["held_out_game_ids"] or len(f["training_game_ids"]) != len(set(f["training_game_ids"]))
            or set(f["training_game_ids"]) != set(m["game_ids"]) - set(f["held_out_game_ids"])):
            raise ValueError("fold partitions overlap or omit games")
        expected_steps = config.rotation_cycles * 7 * math.ceil(len(f["training_game_ids"]) / config.game_batch_size)
        if f["optimizer_steps"] != expected_steps:
            raise ValueError("fixed optimizer-step budget mismatch")
        expected_seed = int.from_bytes(digest_fields("paired_initial_v1", config.initialization_seed, f["fold"]), "big") % 2**63
        if f["initialization_seed"] != expected_seed:
            raise ValueError("paired initialization seed mismatch")
    expected_files = {"temporal/phase_codebook.manifest.json", "temporal/phase_codebook.bin",
        "temporal/canonical_day_code_table.manifest.json", "temporal/canonical_day_code_table.bin",
        "bootstrap/game_cluster_bootstrap_plan.manifest.json", "bootstrap/game_cluster_bootstrap_indices.bin"}
    for f in m["folds"]:
        base = f"folds/{f['fold']}"
        expected_files.update(f"{base}/{name}" for name in ("schedule_manifest.json", "training_schedule.jsonl", "initial_state/manifest.json", "initial_state/tensors.bin"))
        for partition, games in (("training_primary", f["training_game_ids"]),
                ("held_out_primary", f["held_out_game_ids"]), ("held_out_all_alive", f["held_out_game_ids"])):
            expected_files.update(f"{base}/population/{partition}/{g}.json" for g in games)
    if set(m["file_table"]) != expected_files:
        raise ValueError("experiment file inventory mismatch")
    return VerifiedExperiment(envelope, config)


def validate_experiment_preflight(experiment):
    """Preparation/explicit validation only; never called by a training worker."""
    if experiment.runs_path.exists():
        # Once formal execution starts, even an explicit audit must respect the
        # all-ten seal before it opens any held-out records or eligibility.
        from werewolf.tom.training import verify_checkpoint_set
        verify_checkpoint_set(experiment)
    from werewolf.development_publication import open_publication
    from werewolf.tom.population import load_training_primary, load_held_out_primary, load_held_out_all_alive
    from werewolf.tom.protocol import load_bootstrap_plan
    from werewolf.tom.state import load_model_state
    publication = open_publication(experiment.manifest["publication_path"])
    if (publication.manifest_digest != experiment.manifest["publication_digest"]
        or publication.public_view.publication_id != experiment.manifest["publication_id"]
        or publication.public_view.fold_manifest.manifest_digest != experiment.manifest["fold_manifest_digest"]
        or publication.public_view.development_game_set_digest != experiment.manifest["protocol_inputs"]["development_game_set_digest"]
        or publication.public_view.max_structured_token_count != experiment.manifest["max_structured_token_count"]
        or publication.public_view.max_observed_day != experiment.manifest["max_observed_day"]):
        raise ValueError("experiment publication preflight mismatch")
    provider = TemporalCodeProvider.implicit(experiment.path / "temporal", experiment.manifest["protocol_digest"])
    if provider.artifact_digest != experiment.manifest["temporal_artifact_digest"]:
        raise ValueError("temporal artifact parent mismatch")
    model = ObserverConditionedToM(ExperimentCapacity(experiment.config.max_seq_len), provider)
    for index in range(5):
        fold = experiment.fold(index)
        experiment.schedule(index)
        if set(fold["held_out_game_ids"]) != set(publication.public_view.fold_manifest.folds[index].game_ids):
            raise ValueError("experiment fold assignment mismatch")
        initial = load_model_state(experiment.path / "folds" / str(index) / "initial_state", model, fold["paired_initial_state_digest"])
        if initial.artifact.manifest["provenance"] != {"purpose": "initial", "protocol_digest": experiment.manifest["protocol_digest"],
            "fold": index, "initialization_seed": fold["initialization_seed"], "model_graph": MODEL_GRAPH}:
            raise ValueError("initial-state provenance mismatch")
        for game_id in fold["training_game_ids"]:
            load_training_primary(experiment, index, publication.public_view.load_game(game_id))
        for game_id in fold["held_out_game_ids"]:
            game = publication.public_view.load_game(game_id)
            load_held_out_primary(experiment, index, game)
            load_held_out_all_alive(experiment, index, game)
    load_bootstrap_plan(experiment)
