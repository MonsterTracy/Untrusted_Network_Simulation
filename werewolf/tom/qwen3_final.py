"""One Qwen3 explicit-temporal all-development fit, sealed before prediction."""

from dataclasses import dataclass, replace
import json
import math
import multiprocessing
from pathlib import Path
import random
import traceback

import numpy as np
import torch

from werewolf.artifact_io import (canonical_json_bytes, canonical_jsonl_bytes,
    publish_artifact, sha256_bytes, verify_artifact)
from werewolf.artifact_io.tensor_state import decode_tensor_container
from werewolf.development_publication import open_publication
from werewolf.tom import backbone_execution, backbone_study, training
from werewolf.tom.backbone_model import BackboneToM, graph_identity
from werewolf.tom.dataset import CanonicalToMDataset, ExperimentCapacity, PublicTensors, tensorize_public_pre
from werewolf.tom.experiment import ExperimentConfig, configure_runtime, runtime_provenance
from werewolf.tom.final_capacity import derived_capacity, validate_final_pre
from werewolf.tom.paper_study import attest_source
from werewolf.tom.population import select_primary_population
from werewolf.tom.protocol import ORDER_VERSION, SCHEDULE_VERSION, FIELD_ENCODING, _balanced_schedule, digest_fields
from werewolf.tom.state import restore_model_tensors
from werewolf.tom.temporal import TemporalCodeProvider, publish_temporal


VERSION = "classic7_qwen3_all_development_final_v1"
RECOVERY_VERSION = "classic7_qwen3_final_recovery_v1"
TERMINAL_VERSION = "classic7_qwen3_final_terminal_v1"
SEAL_VERSION = "classic7_qwen3_final_seal_v1"
MANIFEST = "qwen3_final_manifest.json"
CONDITION = "explicit_day_phase"
STEPS = 31500
GAMES = 1500
INITIAL_FOLD = 0  # Source tensor identity only; never a data partition.
RNG_VERSION = "classic7_qwen3_final_training_rng_v1"


def _same(a, b):
    return canonical_json_bytes(a) == canonical_json_bytes(b)


def _hash(value):
    return sha256_bytes(canonical_json_bytes(value))


def _protocol():
    design = backbone_study._frozen_design()
    config = ExperimentConfig(**design["reference_protocol"])
    if (design["game_count"] != GAMES or design["training_population"] != "non_wolf_alive"
            or design["training_loss"] != "game_balanced_cross_entropy"
            or design["temporal_conditions"] != [CONDITION]
            or config.max_seq_len != 1024 or config.game_batch_size != 1
            or config.rotation_cycles != 3 or config.recovery_cadence != 1000):
        raise ValueError("Qwen3 final protocol differs from frozen backbone controls")
    return design, config


def _schedule(config, game_ids):
    games = sorted(game_ids)
    if len(games) != GAMES or len(set(games)) != GAMES:
        raise ValueError("Qwen3 final requires exactly 1500 distinct development games")
    batches, coefficients = _balanced_schedule(config.schedule_seed,
        {"lifecycle": VERSION, "population": "all_development"}, games,
        config.rotation_cycles, config.game_batch_size)
    schedule = {"lifecycle_version": VERSION, "schedule_version": SCHEDULE_VERSION,
        "rotation_version": SCHEDULE_VERSION,
        "order_version": ORDER_VERSION, "field_encoding": FIELD_ENCODING,
        "schedule_seed": config.schedule_seed, "game_ids": games,
        "rotation_cycles": 3, "game_batch_size": 1,
        "optimizer_steps": len(batches), "batches": batches,
        "cumulative_game_coefficients": coefficients}
    if schedule["optimizer_steps"] != STEPS:
        raise ValueError("Qwen3 final must contain exactly 31500 optimizer steps")
    return schedule


def _initial(snapshot):
    rows = [row for row in snapshot.manifest["initializations"]
            if row["architecture"] == "qwen3" and row["fold"] == INITIAL_FOLD
            and row["temporal_condition"] == CONDITION]
    if len(rows) != 1:
        raise ValueError("unique Qwen3 fold-0 initial tensor is required")
    return rows[0]


def _snapshot(path, digest):
    snapshot = verify_artifact(path, expected_artifact_type="backbone_tom_study",
                               expected_schema_version=backbone_study.STUDY_VERSION)
    if snapshot.manifest_digest != digest:
        raise ValueError("Qwen3 initial study digest mismatch")
    identity = backbone_execution.snapshot_identity(snapshot)
    backbone_execution.validate_frozen_snapshot(snapshot.path, identity)
    if snapshot.manifest["protocol_inputs"]["design"] != _protocol()[0]:
        raise ValueError("Qwen3 initial study design mismatch")
    return snapshot


@dataclass(frozen=True)
class FinalFit:
    artifact: object
    snapshot: object
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
        schedule = _schedule(self.config, self.manifest["game_ids"])
        if (self.path / "schedule.json").read_bytes() != canonical_json_bytes(schedule):
            raise ValueError("Qwen3 final schedule changed")
        return schedule


def prepare(publication_path, study_path, study_digest, destination):
    source = attest_source()
    design, config = _protocol()
    configure_runtime(config)
    runtime = runtime_provenance(config)
    if runtime["implementation_digest"] != source["implementation_digest"]:
        raise ValueError("Qwen3 final implementation attestation mismatch")
    snapshot = _snapshot(study_path, study_digest)
    publication = open_publication(publication_path)
    if (publication.public_view.publication_id != design["publication_id"]
            or publication.manifest_digest != snapshot.manifest["protocol_inputs"]["publication_digest"]):
        raise ValueError("Qwen3 final publication differs from initial study")
    game_ids = sorted(publication.public_view.game_ids)
    schedule = _schedule(config, game_ids)
    capacity = derived_capacity(config.max_seq_len)
    for game in publication.public_view.games:
        for prefix in game.authoritative_pre_prefixes:
            validate_final_pre(prefix, capacity)
    primary = select_primary_population(publication)
    initial = _initial(snapshot)
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("Qwen3 final destination already exists")
    inputs = {"schema_version": VERSION, "architecture": "qwen3", "temporal_condition": CONDITION,
        "population": "non_wolf_alive", "partition": "all_development", "config": design["reference_protocol"],
        "graph": graph_identity("qwen3"), "capacity": capacity, "rng_version": RNG_VERSION,
        "publication_digest": publication.manifest_digest, "study_digest": snapshot.manifest_digest,
        "initial_state_digest": initial["state_digest"], "source": source, "runtime": runtime,
        "destination_path": str(destination)}
    protocol_digest = _hash(inputs)
    # The temporal artifact is small and bound to this new lifecycle, not the fold's table.
    temporal_path = destination.parent / f".{destination.name}-temporal-staging"
    if temporal_path.exists():
        raise ValueError("Qwen3 final temporal staging already exists")
    try:
        temporal = publish_temporal(temporal_path, capacity["derived_day_capacity"], protocol_digest)
        files = {"schedule.json": canonical_json_bytes(schedule),
            "primary.json": canonical_json_bytes(primary.rows)}
        files.update({f"temporal/{p.name}": p.read_bytes() for p in temporal_path.iterdir() if p.is_file()})
        artifact = publish_artifact(destination, manifest_name=MANIFEST, manifest_fields={
            "artifact_type": "qwen3_all_development_final", "schema_version": VERSION,
            "protocol_inputs": inputs, "protocol_digest": protocol_digest,
            "study_path": str(snapshot.path), "publication_path": str(publication.path),
            "publication_id": publication.public_view.publication_id, "game_ids": game_ids,
            "primary_sidecar_digest": primary.metadata["role_sidecar_digest"],
            "initialization": initial, "temporal_artifact_digest": temporal.manifest_digest,
            "schedule_digest": _hash(schedule), "optimizer_steps": STEPS}, files=files)
    finally:
        if temporal_path.exists():
            for child in temporal_path.iterdir():
                child.unlink()
            temporal_path.rmdir()
    return open_fit(artifact.path)


def open_fit(path):
    artifact = verify_artifact(path, manifest_name=MANIFEST,
        expected_artifact_type="qwen3_all_development_final", expected_schema_version=VERSION)
    m = artifact.manifest
    fields = {"artifact_type", "schema_version", "protocol_inputs", "protocol_digest", "study_path",
        "publication_path", "publication_id", "game_ids", "primary_sidecar_digest", "initialization",
        "temporal_artifact_digest", "schedule_digest", "optimizer_steps", "file_table", "manifest_digest"}
    if set(m) != fields:
        raise ValueError("Qwen3 final manifest schema mismatch")
    design, config = _protocol()
    inputs = m["protocol_inputs"]
    if (inputs != {"schema_version": VERSION, "architecture": "qwen3", "temporal_condition": CONDITION,
            "population": "non_wolf_alive", "partition": "all_development", "config": design["reference_protocol"],
            "graph": graph_identity("qwen3"), "capacity": derived_capacity(config.max_seq_len),
            "rng_version": RNG_VERSION,
            "publication_digest": inputs.get("publication_digest"), "study_digest": inputs.get("study_digest"),
            "initial_state_digest": inputs.get("initial_state_digest"), "source": inputs.get("source"),
            "runtime": inputs.get("runtime"), "destination_path": str(artifact.path)}
            or m["protocol_digest"] != _hash(inputs) or m["optimizer_steps"] != STEPS):
        raise ValueError("Qwen3 final protocol identity mismatch")
    snapshot = _snapshot(m["study_path"], inputs["study_digest"])
    if (m["initialization"] != _initial(snapshot)
            or inputs["initial_state_digest"] != m["initialization"]["state_digest"]
            or m["publication_path"] != snapshot.manifest["publication_path"]):
        raise ValueError("Qwen3 final initial/publication binding mismatch")
    publication = open_publication(m["publication_path"])
    if (publication.manifest_digest != inputs["publication_digest"]
            or publication.public_view.publication_id != m["publication_id"]
            or sorted(publication.public_view.game_ids) != m["game_ids"]):
        raise ValueError("Qwen3 final development publication mismatch")
    primary = select_primary_population(publication)
    if (primary.metadata["role_sidecar_digest"] != m["primary_sidecar_digest"]
            or (artifact.path / "primary.json").read_bytes() != canonical_json_bytes(primary.rows)):
        raise ValueError("Qwen3 final Primary mismatch")
    provider = TemporalCodeProvider.explicit(artifact.path / "temporal", m["protocol_digest"])
    if (provider.artifact_digest != m["temporal_artifact_digest"]
            or len(provider.day) != config.max_seq_len // 4 + 1):
        raise ValueError("Qwen3 final temporal artifact mismatch")
    expected_files = {"schedule.json", "primary.json"} | {f"temporal/{name}" for name in (
        "phase_codebook.manifest.json", "phase_codebook.bin", "canonical_day_code_table.manifest.json",
        "canonical_day_code_table.bin")}
    if set(m["file_table"]) != expected_files:
        raise ValueError("Qwen3 final file inventory mismatch")
    fit = FinalFit(artifact, snapshot, config)
    if m["schedule_digest"] != _hash(fit.schedule()):
        raise ValueError("Qwen3 final schedule digest mismatch")
    return fit


def _source_gate(fit):
    source = attest_source()
    configure_runtime(fit.config)
    if (source != fit.manifest["protocol_inputs"]["source"]
            or runtime_provenance(fit.config) != fit.manifest["protocol_inputs"]["runtime"]):
        raise ValueError("Qwen3 final source/runtime changed")


def _model_optimizer(fit):
    provider = TemporalCodeProvider.explicit(fit.path / "temporal", fit.manifest["protocol_digest"])
    model = BackboneToM("qwen3", provider, study_seed=fit.config.initialization_seed,
                        fold=INITIAL_FOLD).to(fit.config.device)
    backbone_study.load_initial_state(fit.snapshot.path / "initial/qwen3/0", model,
        study_digest=fit.snapshot.manifest["study_digest"], study_seed=fit.config.initialization_seed,
        fold=INITIAL_FOLD, expected_digest=fit.manifest["initialization"]["state_digest"])
    seed = int.from_bytes(digest_fields(RNG_VERSION, fit.config.rng_seed,
        fit.manifest["game_ids"], "qwen3", CONDITION), "big") % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    c = fit.config
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.learning_rate, betas=tuple(c.adam_betas),
                                   eps=c.adam_eps, weight_decay=c.weight_decay, foreach=False, fused=False)
    return model, optimizer


def _binding(fit):
    return {"fit_digest": fit.digest, "protocol_digest": fit.manifest["protocol_digest"],
        "architecture": "qwen3", "temporal_condition": CONDITION, "partition": "all_development",
        "initial_state_digest": fit.manifest["initialization"]["state_digest"],
        "schedule_digest": fit.manifest["schedule_digest"], "source": fit.manifest["protocol_inputs"]["source"],
        "runtime": fit.manifest["protocol_inputs"]["runtime"]}


def _points(fit):
    return training._recovery_points(STEPS, fit.config.recovery_cadence)


def _checkpoint(fit, step, model, optimizer, logs, previous):
    containers, files, scalars, rng = backbone_execution._capture(model, optimizer)
    files["loss.jsonl"] = canonical_jsonl_bytes(logs)
    return publish_artifact(fit.runs_path / "recovery" / str(step), manifest_fields={
        "artifact_type": "qwen3_final_recovery", "schema_version": RECOVERY_VERSION,
        "binding": _binding(fit), "optimizer_step": step, "schedule_position": step,
        "previous_checkpoint_digest": previous, "containers": containers,
        "optimizer_scalars": scalars, "rng_scalars": rng,
        "loss_digest": sha256_bytes(files["loss.jsonl"])}, files=files)


def _chain(fit, model, optimizer, *, complete=False):
    root = fit.runs_path / "recovery"
    points = _points(fit)
    names = sorted(root.iterdir()) if root.exists() else []
    if (any(not path.is_dir() or not path.name.isdecimal() for path in names)
            or [int(path.name) for path in sorted(names, key=lambda p: int(p.name))]
            != (points if complete else points[:len(names)])):
        raise ValueError("Qwen3 final recovery chain is not contiguous")
    previous, logs, ancestry = None, [], []
    for step in points[:len(names)]:
        artifact = verify_artifact(root / str(step), expected_artifact_type="qwen3_final_recovery",
                                   expected_schema_version=RECOVERY_VERSION)
        m = artifact.manifest
        expected = {"artifact_type": "qwen3_final_recovery", "schema_version": RECOVERY_VERSION,
            "binding": _binding(fit), "optimizer_step": step, "schedule_position": step,
            "previous_checkpoint_digest": previous}
        if (set(m) != set(expected) | {"containers", "optimizer_scalars", "rng_scalars", "loss_digest",
                "file_table", "manifest_digest"}
                or any(not _same(m[k], v) for k, v in expected.items())
                or set(m["containers"]) != {"model", "optimizer", "rng"}
                or set(m["file_table"]) != {"model.bin", "optimizer.bin", "rng.bin", "loss.jsonl"}):
            raise ValueError("Qwen3 final recovery binding mismatch")
        raw = (artifact.path / "loss.jsonl").read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        if (raw != canonical_jsonl_bytes(rows) or sha256_bytes(raw) != m["loss_digest"]
                or len(rows) != step or rows[:len(logs)] != logs):
            raise ValueError("Qwen3 final loss trace mismatch")
        batches = fit.schedule()["batches"]
        for i, row in enumerate(rows):
            if (set(row) != {"step", "game_ids", "loss", "learning_rate"}
                    or type(row["step"]) is not int or row["step"] != i + 1
                    or row["game_ids"] != [item["game_id"] for item in batches[i]]
                    or row["learning_rate"] != fit.config.learning_rate
                    or type(row["loss"]) not in (float, int) or not math.isfinite(row["loss"])):
                raise ValueError("Qwen3 final loss/schedule mismatch")
        decoded = {name: decode_tensor_container(container, (artifact.path / f"{name}.bin").read_bytes(),
            payload_path=f"{name}.bin") for name, container in m["containers"].items()}
        restore_model_tensors(model, decoded["model"])
        backbone_execution._validate_optimizer(model, optimizer, decoded["optimizer"], m["optimizer_scalars"], step)
        # Reuse the already audited byte-exact optimizer and RNG restore path.
        optimizer_tensors = {name: replace(value, array=value.array.reshape(())) if name.endswith(".step") else value
            for name, value in decoded["optimizer"].items()}
        training._optimizer_restore(optimizer, optimizer_tensors, m["optimizer_scalars"])
        training._rng_restore(decoded["rng"], m["rng_scalars"])
        containers, files, scalars, rng = backbone_execution._capture(model, optimizer)
        if (not _same(containers, m["containers"]) or not _same(scalars, m["optimizer_scalars"])
                or not _same(rng, m["rng_scalars"])
                or any(data != (artifact.path / name).read_bytes() for name, data in files.items())):
            raise ValueError("Qwen3 final recovery restore is not byte exact")
        previous, logs = artifact.manifest_digest, rows
        ancestry.append({"step": step, "digest": previous})
    return logs, ancestry


def verify_terminal(fit):
    _source_gate(fit)
    model, optimizer = _model_optimizer(fit)
    logs, ancestry = _chain(fit, model, optimizer, complete=True)
    if len(logs) != STEPS or ancestry[-1]["step"] != STEPS:
        raise ValueError("Qwen3 final terminal requires all 31500 steps")
    root = fit.runs_path
    inventory = {path.name for path in root.iterdir()}
    if inventory not in ({"recovery", "terminal"}, {"recovery", "terminal", "seal"}):
        raise ValueError("Qwen3 final run inventory mismatch")
    last = verify_artifact(root / "recovery" / str(STEPS), expected_artifact_type="qwen3_final_recovery",
                           expected_schema_version=RECOVERY_VERSION)
    expected = {"artifact_type": "qwen3_final_terminal", "schema_version": TERMINAL_VERSION,
        "binding": _binding(fit), "terminal_step": STEPS, "last_recovery_digest": last.manifest_digest,
        "model_digest": _hash(last.manifest["containers"]["model"]), "checkpoint_ancestry": ancestry}
    terminal = verify_artifact(root / "terminal", expected_artifact_type="qwen3_final_terminal",
                               expected_schema_version=TERMINAL_VERSION)
    if terminal.manifest["file_table"] or not _same(
            {k: v for k, v in terminal.manifest.items() if k not in {"file_table", "manifest_digest"}}, expected):
        raise ValueError("Qwen3 final terminal provenance mismatch")
    return terminal


def _train(fit):
    _source_gate(fit)
    if (fit.runs_path / "seal").exists() or (fit.runs_path / "terminal").exists():
        raise ValueError("Qwen3 final run is already terminal or sealed")
    publication = open_publication(fit.manifest["publication_path"])
    dataset = CanonicalToMDataset(publication.public_view, fit.manifest["game_ids"],
                                  ExperimentCapacity(fit.config.max_seq_len))
    by_game = {g: [sample for sample in dataset if sample.game_id == g] for g in fit.manifest["game_ids"]}
    masks = json.loads((fit.path / "primary.json").read_bytes())
    model, optimizer = _model_optimizer(fit)
    logs, ancestry = _chain(fit, model, optimizer)
    points = _points(fit)
    def on_step(step, current):
        if step in points:
            _source_gate(fit)
            checkpoint = _checkpoint(fit, step, model, optimizer, current,
                                     ancestry[-1]["digest"] if ancestry else None)
            ancestry.append({"step": step, "digest": checkpoint.manifest_digest})
    def gate():
        if (fit.runs_path / "seal").exists():
            raise ValueError("Qwen3 final seal permanently closes training")
    training.fit_steps(model, optimizer, by_game, masks, fit.schedule(), device=fit.config.device,
        learning_rate=fit.config.learning_rate, start=len(logs), logs=logs,
        gate=gate, on_step=on_step)
    _source_gate(fit)
    last = verify_artifact(fit.runs_path / "recovery" / str(STEPS), expected_artifact_type="qwen3_final_recovery",
                           expected_schema_version=RECOVERY_VERSION)
    publish_artifact(fit.runs_path / "terminal", manifest_fields={
        "artifact_type": "qwen3_final_terminal", "schema_version": TERMINAL_VERSION,
        "binding": _binding(fit), "terminal_step": STEPS, "last_recovery_digest": last.manifest_digest,
        "model_digest": _hash(last.manifest["containers"]["model"]), "checkpoint_ancestry": ancestry}, files={})
    return verify_terminal(fit)


def _worker(path, digest, connection):
    try:
        fit = open_fit(path)
        if fit.digest != digest:
            raise ValueError("Qwen3 final fit identity changed")
        connection.send((True, _train(fit).manifest_digest))
    except Exception:
        connection.send((False, traceback.format_exc()))
    finally:
        connection.close()


def run(fit, *, resume=False):
    with backbone_execution._lock(fit.runs_path.parent / f"{fit.digest}.lock"):
        _source_gate(fit)
        if (fit.runs_path / "terminal").exists() or (fit.runs_path / "seal").exists():
            raise ValueError("Qwen3 final fit already completed")
        recovery = fit.runs_path / "recovery"
        started = recovery.exists()
        if started and not resume:
            raise ValueError("existing Qwen3 recovery requires explicit resume")
        if resume and not started:
            raise ValueError("no Qwen3 recovery exists to resume")
        if resume:
            if not recovery.is_dir() or not any(recovery.iterdir()):
                raise ValueError("resume requires at least one verified Qwen3 recovery checkpoint")
            model, optimizer = _model_optimizer(fit)
            _, ancestry = _chain(fit, model, optimizer)
            if not ancestry:
                raise ValueError("resume requires at least one verified Qwen3 recovery checkpoint")
        context = multiprocessing.get_context("spawn")
        reader, writer = context.Pipe(duplex=False)
        process = context.Process(target=_worker, args=(str(fit.path), fit.digest, writer))
        process.start()
        writer.close()
        try:
            success, result = reader.recv()
        except EOFError as error:
            raise RuntimeError("Qwen3 final worker interrupted; resume only from verified recovery") from error
        finally:
            reader.close()
            process.join()
        if not success:
            raise ValueError(result)
        terminal = verify_terminal(fit)
        if terminal.manifest_digest != result or process.exitcode != 0:
            raise ValueError("Qwen3 final worker handoff mismatch")
        return terminal


def seal(fit):
    with backbone_execution._lock(fit.runs_path.parent / f"{fit.digest}.lock"):
        terminal = verify_terminal(fit)
        return publish_artifact(fit.runs_path / "seal", manifest_fields={
            "artifact_type": "qwen3_final_seal", "schema_version": SEAL_VERSION,
            "binding": _binding(fit), "terminal_digest": terminal.manifest_digest,
            "terminal_step": STEPS}, files={})


def verify_seal(fit):
    terminal = verify_terminal(fit)
    artifact = verify_artifact(fit.runs_path / "seal", expected_artifact_type="qwen3_final_seal",
                               expected_schema_version=SEAL_VERSION)
    expected = {"artifact_type": "qwen3_final_seal", "schema_version": SEAL_VERSION,
        "binding": _binding(fit), "terminal_digest": terminal.manifest_digest, "terminal_step": STEPS}
    if artifact.manifest["file_table"] or not _same(
            {k: v for k, v in artifact.manifest.items() if k not in {"file_table", "manifest_digest"}}, expected):
        raise ValueError("Qwen3 final seal mismatch")
    return artifact


class SealedQwen3Predictor:
    """Read-only 7x7 public-PRE prediction from the sole sealed final terminal."""

    def __init__(self, fit):
        self.seal = verify_seal(fit)
        self.fit = fit
        provider = TemporalCodeProvider.explicit(fit.path / "temporal", fit.manifest["protocol_digest"])
        self.model = BackboneToM("qwen3", provider, study_seed=fit.config.initialization_seed,
                                 fold=INITIAL_FOLD).to(fit.config.device)
        last = verify_artifact(fit.runs_path / "recovery" / str(STEPS),
                               expected_artifact_type="qwen3_final_recovery", expected_schema_version=RECOVERY_VERSION)
        values = decode_tensor_container(last.manifest["containers"]["model"],
            (last.path / "model.bin").read_bytes(), payload_path="model.bin")
        restore_model_tensors(self.model, values)
        self.model.eval()

    def log_probabilities(self, prefix):
        validate_final_pre(prefix, self.fit.manifest["protocol_inputs"]["capacity"])
        _, tensors = tensorize_public_pre(prefix, ExperimentCapacity(1024))
        public = PublicTensors.stack([tensors])
        with torch.inference_mode():
            result = self.model(**{k: v.to(self.fit.config.device) for k, v in public.kwargs().items()})[0].cpu()
        if result.shape != (7, 7):
            raise ValueError("Qwen3 final prediction shape mismatch")
        for observer in range(7):
            non_self = result[observer, [j for j in range(7) if j != observer]]
            if (not torch.isfinite(non_self).all() or result[observer, observer] != -torch.inf
                    or not torch.isclose(non_self.exp().sum(), torch.tensor(1.), atol=1e-6)):
                raise ValueError("Qwen3 final prediction violates non-self simplex")
        return result

    def predict(self, prefix, observer):
        if type(observer) is not int or not 0 <= observer < 7:
            raise ValueError("observer must be a zero-based seat")
        return self.log_probabilities(prefix)[observer].exp()
