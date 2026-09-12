"""Primary-only fixed-budget training with one deterministic recovery rule."""

import multiprocessing
import random
import traceback
from pathlib import Path

import numpy as np
import torch

from werewolf.artifact_io import TensorValue, canonical_json_bytes, canonical_jsonl_bytes, publish_artifact, sha256_bytes, verify_artifact
from werewolf.artifact_io.tensor_state import encode_tensor_container, decode_tensor_container
from werewolf.development_publication import open_publication
from werewolf.tom.dataset import CanonicalToMDataset, ExperimentCapacity, PublicTensors, rotate
from werewolf.tom.experiment import (TEMPORAL_CONDITIONS, RECOVERY_POLICY, open_experiment,
    validate_runtime, read_json, MODEL_GRAPH)
from werewolf.tom.model import ObserverConditionedToM
from werewolf.tom.population import load_training_primary
from werewolf.tom.protocol import digest_fields
from werewolf.tom.run_records import publish_record, publish_run_bytes, read_record
from werewolf.tom.scoring import game_balanced_cross_entropy
from werewolf.tom.state import load_model_state, publish_model_state, model_tensor_values, restore_model_tensors, validate_model_state, validate_model_tensors
from werewolf.tom.temporal import TemporalCodeProvider

RECOVERY_VERSION = "classic7_training_recovery_v1"


def lineage_path(experiment, fold, condition):
    experiment.fold(fold)
    if condition not in TEMPORAL_CONDITIONS:
        raise ValueError("unknown temporal condition")
    return experiment.runs_path / condition / str(fold)


def _training_gate(experiment):
    root = experiment.runs_path
    if (root / "checkpoint_set_manifest.json").exists():
        raise ValueError("checkpoint-set seal permanently closes training/recovery")
    if list(root.rglob("held_out_predictions.jsonl")) or list(root.rglob("fold_report.json")):
        raise ValueError("held-out access permanently closes training/recovery")




def build_model(experiment, condition):
    if condition not in TEMPORAL_CONDITIONS:
        raise ValueError("unknown temporal condition")
    path = experiment.path / "temporal"
    constructor = TemporalCodeProvider.implicit if condition == "implicit" else TemporalCodeProvider.explicit
    provider = constructor(path, experiment.manifest["protocol_digest"])
    if provider.artifact_digest != experiment.manifest["temporal_artifact_digest"]:
        raise ValueError("temporal artifact digest mismatch")
    return ObserverConditionedToM(ExperimentCapacity(experiment.config.max_seq_len), provider).to(experiment.config.device)


def _rng_capture():
    numpy_state = np.random.get_state()
    arrays = {"torch_cpu": TensorValue(torch.get_rng_state().numpy(), False),
              "numpy_state": TensorValue(numpy_state[1], False)}
    if torch.cuda.is_initialized():
        arrays.update({f"torch_cuda_{i}": TensorValue(v.cpu().numpy(), False) for i, v in enumerate(torch.cuda.get_rng_state_all())})
    scalar = {"python": random.getstate(), "numpy_algorithm": numpy_state[0],
              "numpy_position": numpy_state[2], "numpy_has_gauss": numpy_state[3], "numpy_cached_gaussian": numpy_state[4]}
    return arrays, scalar


def _rng_restore(tensors, scalar):
    version, state, gaussian = scalar["python"]
    random.setstate((version, tuple(state), gaussian))
    np.random.set_state((scalar["numpy_algorithm"], tensors["numpy_state"].array.copy(),
        scalar["numpy_position"], scalar["numpy_has_gauss"], scalar["numpy_cached_gaussian"]))
    torch.set_rng_state(torch.from_numpy(tensors["torch_cpu"].array.copy()))
    cuda = sorted(k for k in tensors if k.startswith("torch_cuda_"))
    if cuda:
        torch.cuda.set_rng_state_all([torch.from_numpy(tensors[k].array.copy()) for k in cuda])


def _optimizer_capture(optimizer):
    state = optimizer.state_dict()
    tensors, scalars = {}, {}
    for parameter, values in state["state"].items():
        scalars[str(parameter)] = {}
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                tensors[f"{parameter}.{key}"] = TensorValue(value.detach().cpu().numpy(), False)
            else:
                scalars[str(parameter)][key] = value
    return tensors, {"param_groups": state["param_groups"], "state": scalars}


def _optimizer_restore(optimizer, tensors, scalar):
    state = {int(k): dict(v) for k, v in scalar["state"].items()}
    for name, tensor in tensors.items():
        parameter, key = name.split(".", 1)
        state[int(parameter)][key] = torch.from_numpy(tensor.array.copy())
    optimizer.load_state_dict({"param_groups": scalar["param_groups"], "state": state})


def _recovery_points(budget, cadence):
    return sorted(set(range(cadence, budget + 1, cadence)) | {budget})


def _validate_training_log(experiment, fold, data, step, prefix):
    rows = [read_json(row) for row in data.splitlines()]
    if len(rows) != step or rows[:len(prefix)] != prefix:
        raise ValueError("training log prefix/step mismatch")
    schedule = experiment.schedule(fold)
    for cursor, row in enumerate(rows):
        if (set(row) != {"step", "game_ids", "loss", "learning_rate"}
            or row["step"] != cursor + 1
            or row["game_ids"] != [item["game_id"] for item in schedule["batches"][cursor]]
            or row["learning_rate"] != experiment.config.learning_rate
            or not np.isfinite(row["loss"])):
            raise ValueError("training log schedule/step mismatch")
    return rows


def _read_recovery(experiment, fold, condition, path, point, previous, logs):
    """One validator shared by deterministic resume and the pre-seal audit."""
    artifact = verify_artifact(path, expected_artifact_type="training_recovery", expected_schema_version=RECOVERY_VERSION)
    m = artifact.manifest
    required = {"artifact_type", "schema_version", "experiment_digest", "fold", "temporal_condition", "recovery_policy", "step", "next_schedule_cursor", "previous_recovery_digest", "runtime", "schedule_digest", "containers", "optimizer_scalars", "rng_scalars", "scheduler_state", "training_log_prefix_digest", "file_table", "manifest_digest"}
    if (set(m) != required or m["experiment_digest"] != experiment.digest or m["fold"] != fold or m["temporal_condition"] != condition
        or m["step"] != point or m["next_schedule_cursor"] != point or m["previous_recovery_digest"] != previous
        or m["runtime"] != experiment.manifest["runtime"] or m["schedule_digest"] != experiment.fold(fold)["schedule_digest"]
        or m["recovery_policy"] != RECOVERY_POLICY or m["scheduler_state"] != {"name": "constant", "step": point}):
        raise ValueError("recovery provenance/cursor mismatch")
    if set(m["file_table"]) != {"training_log.jsonl", "model_tensors.bin", "optimizer_tensors.bin", "rng_state.bin"}:
        raise ValueError("recovery file inventory mismatch")
    if set(m["containers"]) != {"model_tensors", "optimizer_tensors", "rng_state"}:
        raise ValueError("recovery state sections mismatch")
    data = (artifact.path / "training_log.jsonl").read_bytes()
    if sha256_bytes(data) != m["training_log_prefix_digest"]:
        raise ValueError("recovery log digest mismatch")
    rows = _validate_training_log(experiment, fold, data, point, logs)
    decoded = {name: decode_tensor_container(container, (artifact.path / f"{name}.bin").read_bytes(), payload_path=f"{name}.bin")
               for name, container in m["containers"].items()}
    return artifact, rows, decoded


def _publish_recovery(path, experiment, fold, condition, step, model, optimizer, logs, previous):
    optimizer_values, optimizer_scalars = _optimizer_capture(optimizer)
    rng_values, rng_scalars = _rng_capture()
    containers, files = {}, {}
    for name, values in (("model_tensors", model_tensor_values(model)), ("optimizer_tensors", optimizer_values), ("rng_state", rng_values)):
        container, payload = encode_tensor_container(values, payload_path=f"{name}.bin")
        containers[name] = container
        files[f"{name}.bin"] = payload
    files["training_log.jsonl"] = canonical_jsonl_bytes(logs)
    return publish_artifact(path, manifest_fields={
        "artifact_type": "training_recovery", "schema_version": RECOVERY_VERSION,
        "experiment_digest": experiment.digest, "fold": fold, "temporal_condition": condition,
        "recovery_policy": RECOVERY_POLICY, "step": step, "next_schedule_cursor": step,
        "previous_recovery_digest": previous, "runtime": experiment.manifest["runtime"],
        "schedule_digest": experiment.fold(fold)["schedule_digest"], "containers": containers,
        "optimizer_scalars": optimizer_scalars, "rng_scalars": rng_scalars,
        "scheduler_state": {"name": "constant", "step": step},
        "training_log_prefix_digest": sha256_bytes(files["training_log.jsonl"]),
    }, files=files)


def verify_terminal(experiment, fold, condition):
    path = lineage_path(experiment, fold, condition) / "terminal_checkpoint"
    if (path.parent / "failure.json").exists():
        raise ValueError("lineage permanently failed closed")
    state = validate_model_state(path)
    p = state.artifact.manifest["provenance"]
    expected = {"purpose": "terminal", "experiment_digest": experiment.digest, "fold": fold,
        "temporal_condition": condition, "terminal_step": experiment.fold(fold)["optimizer_steps"],
        "paired_initial_state_digest": experiment.fold(fold)["paired_initial_state_digest"],
        "schedule_digest": experiment.fold(fold)["schedule_digest"], "protocol_digest": experiment.manifest["protocol_digest"]}
    if set(p) != set(expected) | {"training_log_digest", "last_recovery_digest", "initial_rng_digest"} or any(p.get(k) != v for k, v in expected.items()):
        raise ValueError("terminal checkpoint provenance mismatch")
    log = (path.parent / "training_log.jsonl").read_bytes()
    if sha256_bytes(log) != p["training_log_digest"]:
        raise ValueError("terminal training log mismatch")
    training = read_record(path.parent / "training_manifest.json")
    expected_training = {"schema_version": "classic7_training_manifest_v1", "experiment_digest": experiment.digest,
        "fold": fold, "temporal_condition": condition, "initial_rng_digest": p["initial_rng_digest"],
        "paired_initial_state_digest": p["paired_initial_state_digest"], "schedule_digest": p["schedule_digest"],
        "runtime": experiment.manifest["runtime"], "recovery_policy": RECOVERY_POLICY}
    if {k: v for k, v in training.items() if k != "record_digest"} != expected_training:
        raise ValueError("terminal training-manifest/RNG provenance mismatch")
    run = read_record(path.parent / "run_provenance.json")
    if {k: v for k, v in run.items() if k != "record_digest"} != {
        "schema_version": "classic7_run_provenance_v1", "runtime": experiment.manifest["runtime"], "experiment_digest": experiment.digest}:
        raise ValueError("terminal run provenance mismatch")
    _validate_training_log(experiment, fold, log, expected["terminal_step"], [])
    return state.artifact


def _verify_terminal_recovery(experiment, fold, condition, terminal):
    """Pre-seal audit only; evaluation never opens optimizer/recovery state."""
    root = terminal.path.parent / "recovery"
    points = _recovery_points(experiment.fold(fold)["optimizer_steps"], experiment.config.recovery_cadence)
    actual = sorted(int(p.name) for p in root.iterdir() if not p.name.startswith("."))
    if actual != points:
        raise ValueError("terminal recovery coverage mismatch")
    previous = None
    logs = []
    for point in points:
        artifact, logs, decoded = _read_recovery(experiment, fold, condition, root / str(point), point, previous, logs)
        m = artifact.manifest
        previous = artifact.manifest_digest
    validate_model_tensors(decoded["model_tensors"])
    p = terminal.manifest["provenance"]
    if (previous != p["last_recovery_digest"] or m["training_log_prefix_digest"] != p["training_log_digest"]
        or (artifact.path / "model_tensors.bin").read_bytes() != (terminal.path / "tensors.bin").read_bytes()):
        raise ValueError("terminal/final-recovery digest or model mismatch")


def fit_steps(model, optimizer, by_game, masks, schedule, *, device, learning_rate, start, logs, gate, on_step):
    """Shared fixed-budget optimizer loop; no validation or checkpoint selection."""
    model.train()
    step = start
    budget = schedule["optimizer_steps"]
    for cursor in range(start, budget):
        gate()
        optimizer.zero_grad(set_to_none=True)
        game_losses = []
        for item in schedule["batches"][cursor]:
            samples, eligibility = [], []
            for sample in by_game[item["game_id"]]:
                rotated, mask = rotate(sample, item["shift"], torch.tensor(masks[sample.game_id][sample.boundary_id], dtype=torch.bool))
                samples.append(rotated)
                eligibility.append(mask & rotated.label_observed)
            public = PublicTensors.stack([s.public for s in samples])
            logp = model(**{k: v.to(device) for k, v in public.kwargs().items()})
            q = torch.stack([s.q for s in samples]).to(device)
            mask = torch.stack(eligibility).to(device)
            game_losses.append((logp, q, mask))
        loss = game_balanced_cross_entropy(game_losses)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite Primary training loss")
        loss.backward()
        optimizer.step()
        step = cursor + 1
        logs.append({"step": step, "game_ids": [i["game_id"] for i in schedule["batches"][cursor]], "loss": float(loss.detach().cpu()), "learning_rate": learning_rate})
        on_step(step, logs)
    return step


def _train_worker(path, expected_digest, fold, condition):
    experiment = open_experiment(path)
    if experiment.digest != expected_digest:
        raise ValueError("experiment identity changed")
    _training_gate(experiment)
    lineage = lineage_path(experiment, fold, condition)
    if (lineage / "failure.json").exists():
        raise ValueError("lineage permanently failed closed")
    validate_runtime(experiment)
    if (lineage / "terminal_checkpoint").exists():
        publish_record(lineage / "run_provenance.json", {"schema_version": "classic7_run_provenance_v1", "runtime": experiment.manifest["runtime"], "experiment_digest": experiment.digest})
        return verify_terminal(experiment, fold, condition)
    schedule = experiment.schedule(fold)
    fold_record = experiment.fold(fold)
    publication = open_publication(experiment.manifest["publication_path"], game_ids=fold_record["training_game_ids"])
    if publication.manifest_digest != experiment.manifest["publication_digest"] or publication.public_view.fold_manifest.manifest_digest != experiment.manifest["fold_manifest_digest"]:
        raise ValueError("publication/fold parent mismatch")
    if set(publication.public_view.fold_manifest.folds[fold].game_ids) != set(fold_record["held_out_game_ids"]):
        raise ValueError("held-out fold assignment mismatch")
    dataset = CanonicalToMDataset(publication.public_view, fold_record["training_game_ids"], ExperimentCapacity(experiment.config.max_seq_len))
    by_game = {g: [s for s in dataset if s.game_id == g] for g in fold_record["training_game_ids"]}
    masks = {g.game_id: load_training_primary(experiment, fold, g)["rows"] for g in publication.public_view.games}
    model = build_model(experiment, condition)
    initial = load_model_state(experiment.path / "folds" / str(fold) / "initial_state", model, fold_record["paired_initial_state_digest"])
    if initial.artifact.manifest["provenance"] != {"purpose": "initial",
        "protocol_digest": experiment.manifest["protocol_digest"], "fold": fold,
        "initialization_seed": fold_record["initialization_seed"], "model_graph": MODEL_GRAPH}:
        raise ValueError("paired initial parent mismatch")
    seed = int.from_bytes(digest_fields("paired_rng_v1", experiment.config.rng_seed, fold), "big") % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if experiment.config.device == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng_values, rng_scalar = _rng_capture()
    rng_container, rng_payload = encode_tensor_container(rng_values, payload_path="rng_state.bin")
    initial_rng_digest = sha256_bytes(canonical_json_bytes({"container": rng_container, "scalar": rng_scalar}) + rng_payload)
    publish_record(lineage / "training_manifest.json", {"schema_version": "classic7_training_manifest_v1",
        "experiment_digest": experiment.digest, "fold": fold, "temporal_condition": condition,
        "initial_rng_digest": initial_rng_digest, "paired_initial_state_digest": fold_record["paired_initial_state_digest"],
        "schedule_digest": fold_record["schedule_digest"], "runtime": experiment.manifest["runtime"], "recovery_policy": RECOVERY_POLICY})
    optimizer = torch.optim.AdamW(model.parameters(), lr=experiment.config.learning_rate,
        betas=tuple(experiment.config.adam_betas), eps=experiment.config.adam_eps, weight_decay=experiment.config.weight_decay,
        foreach=False, fused=False)
    budget = fold_record["optimizer_steps"]
    points = _recovery_points(budget, experiment.config.recovery_cadence)
    recovery_root = lineage / "recovery"
    existing = sorted(int(p.name) for p in recovery_root.iterdir() if not p.name.startswith(".")) if recovery_root.exists() else []
    if existing != points[:len(existing)]:
        raise ValueError("non-contiguous recovery chain")
    logs, step, previous = [], 0, None
    for point in existing:
        artifact, candidate_logs, decoded = _read_recovery(experiment, fold, condition, recovery_root / str(point), point, previous, logs)
        m = artifact.manifest
        restore_model_tensors(model, decoded["model_tensors"])
        _optimizer_restore(optimizer, decoded["optimizer_tensors"], m["optimizer_scalars"])
        _rng_restore(decoded["rng_state"], m["rng_scalars"])
        logs, step, previous = candidate_logs, point, artifact.manifest_digest
    def on_step(step, logs):
        nonlocal previous
        if step in points:
            recovery = _publish_recovery(recovery_root / str(step), experiment, fold, condition, step, model, optimizer, logs, previous)
            previous = recovery.manifest_digest
    step = fit_steps(model, optimizer, by_game, masks, schedule, device=experiment.config.device,
        learning_rate=experiment.config.learning_rate, start=step, logs=logs,
        gate=lambda: _training_gate(experiment), on_step=on_step)
    log_bytes = canonical_jsonl_bytes(logs)
    publish_run_bytes(lineage / "training_log.jsonl", log_bytes)
    publish_model_state(lineage / "terminal_checkpoint", model, {
        "purpose": "terminal", "experiment_digest": experiment.digest, "fold": fold, "temporal_condition": condition,
        "terminal_step": step, "paired_initial_state_digest": fold_record["paired_initial_state_digest"],
        "schedule_digest": fold_record["schedule_digest"], "protocol_digest": experiment.manifest["protocol_digest"],
        "training_log_digest": sha256_bytes(log_bytes), "last_recovery_digest": previous, "initial_rng_digest": initial_rng_digest})
    publish_record(lineage / "run_provenance.json", {"schema_version": "classic7_run_provenance_v1", "runtime": experiment.manifest["runtime"], "experiment_digest": experiment.digest})
    return verify_terminal(experiment, fold, condition)


def _process_entry(path, digest, fold, condition, connection):
    try:
        artifact = _train_worker(path, digest, fold, condition)
        connection.send((True, artifact.manifest_digest))
    except Exception as error:
        experiment = open_experiment(path)
        target = lineage_path(experiment, fold, condition) / "failure.json"
        if not target.exists():
            publish_record(target, {"schema_version": "classic7_training_failure_v1", "experiment_digest": digest, "error": type(error).__name__, "message": str(error)})
        connection.send((False, traceback.format_exc()))
    finally:
        connection.close()


def train_primary_fold(experiment, fold, temporal_condition):
    """Run each formal lineage in a fresh process; there is no failure-time mode."""
    lineage_path(experiment, fold, temporal_condition)
    _training_gate(experiment)
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_process_entry, args=(str(experiment.path), experiment.digest, fold, temporal_condition, writer))
    process.start()
    writer.close()
    try:
        success, result = reader.recv()
    except EOFError as error:
        raise RuntimeError("training process interrupted; only deterministic step-boundary recovery is permitted") from error
    finally:
        reader.close()
        process.join()
    if not success:
        raise ValueError(result)
    artifact = verify_terminal(experiment, fold, temporal_condition)
    if artifact.manifest_digest != result:
        raise ValueError("terminal process handoff digest mismatch")
    return artifact


def seal_checkpoint_set(experiment):
    _training_gate(experiment)
    validate_runtime(experiment)
    checkpoints = []
    for condition in TEMPORAL_CONDITIONS:
        for fold in range(5):
            artifact = verify_terminal(experiment, fold, condition)
            _verify_terminal_recovery(experiment, fold, condition, artifact)
            checkpoints.append({"fold": fold, "temporal_condition": condition, "checkpoint_digest": artifact.manifest_digest})
    return publish_record(experiment.runs_path / "checkpoint_set_manifest.json", {
        "schema_version": "classic7_checkpoint_set_v1", "experiment_digest": experiment.digest, "checkpoints": checkpoints})


def verify_checkpoint_set(experiment):
    path = experiment.runs_path / "checkpoint_set_manifest.json"
    if not path.is_file():
        raise ValueError("all-ten checkpoint-set seal is required before held-out access")
    record = read_record(path)
    if set(record) != {"schema_version", "experiment_digest", "checkpoints", "record_digest"} or record["schema_version"] != "classic7_checkpoint_set_v1" or record["experiment_digest"] != experiment.digest:
        raise ValueError("checkpoint-set seal identity mismatch")
    expected = [{"fold": f, "temporal_condition": c, "checkpoint_digest": verify_terminal(experiment, f, c).manifest_digest}
                for c in TEMPORAL_CONDITIONS for f in range(5)]
    if record["checkpoints"] != expected:
        raise ValueError("checkpoint-set seal coverage/digest mismatch")
    return record
