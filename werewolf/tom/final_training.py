"""Final-fit orchestration over the shared fixed-budget optimizer loop."""

import multiprocessing
import random
import traceback

import numpy as np
import torch

from werewolf.artifact_io import canonical_json_bytes, canonical_jsonl_bytes, publish_artifact, verify_artifact, sha256_bytes
from werewolf.artifact_io.tensor_state import encode_tensor_container, decode_tensor_container
from werewolf.tom.dataset import CanonicalToMDataset, ExperimentCapacity
from werewolf.tom.experiment import TEMPORAL_CONDITIONS, RECOVERY_POLICY, read_json
from werewolf.tom.final_experiment import open_final_experiment, final_training_inputs
from werewolf.tom.run_records import publish_record, read_record, publish_run_bytes
from werewolf.tom.state import (load_model_state, publish_model_state, validate_model_state,
    model_tensor_values, restore_model_tensors, validate_model_tensors)
from werewolf.tom.training import (build_model, fit_steps, _rng_capture, _rng_restore,
    _optimizer_capture, _optimizer_restore, _recovery_points)


def final_lineage(experiment, condition):
    if condition not in TEMPORAL_CONDITIONS:
        raise ValueError("unknown final temporal condition")
    return experiment.runs_path / condition


def final_training_gate(experiment):
    if (experiment.runs_path / "final_model_seal.json").exists() or (experiment.runs_path / "final_evaluation_consumption.json").exists():
        raise ValueError("final model seal permanently closes training/recovery")


def _binding(experiment, condition):
    m = experiment.manifest
    return {"experiment_digest": experiment.digest, "temporal_condition": condition,
            "schedule_digest": m["schedule_digest"], "paired_initial_state_digest": m["paired_initial_state_digest"],
            "runtime": m["runtime"], "recovery_policy": RECOVERY_POLICY}


def _logs(experiment, data, step, prefix):
    rows = [read_json(line) for line in data.splitlines()]
    if len(rows) != step or rows[:len(prefix)] != prefix:
        raise ValueError("final recovery log prefix mismatch")
    for index, (row, batch) in enumerate(zip(rows, experiment.schedule()["batches"])):
        if (set(row) != {"step", "game_ids", "loss", "learning_rate"} or row["step"] != index + 1
            or row["game_ids"] != [r["game_id"] for r in batch]
            or row["learning_rate"] != experiment.config.learning_rate or not np.isfinite(row["loss"])):
            raise ValueError("final training log/schedule mismatch")
    return rows


def _publish_point(experiment, condition, step, model, optimizer, logs, previous):
    optimizer_values, optimizer_scalars = _optimizer_capture(optimizer)
    rng_values, rng_scalars = _rng_capture()
    containers, files = {}, {}
    for name, values in (("model_tensors", model_tensor_values(model)), ("optimizer_tensors", optimizer_values), ("rng_state", rng_values)):
        containers[name], files[f"{name}.bin"] = encode_tensor_container(values, payload_path=f"{name}.bin")
    files["training_log.jsonl"] = canonical_jsonl_bytes(logs)
    return publish_artifact(final_lineage(experiment, condition) / "recovery" / str(step), manifest_fields={
        "artifact_type": "final_training_recovery", "schema_version": "classic7_final_recovery_v1",
        **_binding(experiment, condition), "step": step, "next_schedule_cursor": step,
        "previous_recovery_digest": previous, "containers": containers, "optimizer_scalars": optimizer_scalars,
        "rng_scalars": rng_scalars, "scheduler_state": {"name": "constant", "step": step},
        "training_log_digest": sha256_bytes(files["training_log.jsonl"])}, files=files)


def _chain(experiment, condition, *, complete=False):
    root = final_lineage(experiment, condition) / "recovery"
    points = _recovery_points(experiment.manifest["optimizer_steps"], experiment.config.recovery_cadence)
    existing = sorted(int(p.name) for p in root.iterdir() if not p.name.startswith(".")) if root.exists() else []
    if existing != (points if complete else points[:len(existing)]):
        raise ValueError("final recovery must be a contiguous scheduled chain")
    previous, logs, last = None, [], None
    for step in existing:
        artifact = verify_artifact(root / str(step), expected_artifact_type="final_training_recovery", expected_schema_version="classic7_final_recovery_v1")
        m = artifact.manifest
        expected = {"artifact_type": "final_training_recovery", "schema_version": "classic7_final_recovery_v1",
            **_binding(experiment, condition), "step": step, "next_schedule_cursor": step,
            "previous_recovery_digest": previous, "scheduler_state": {"name": "constant", "step": step}}
        if set(m) != set(expected) | {"containers", "optimizer_scalars", "rng_scalars", "training_log_digest", "file_table", "manifest_digest"} or any(m[k] != v for k, v in expected.items()):
            raise ValueError("final recovery identity mismatch")
        if set(m["containers"]) != {"model_tensors", "optimizer_tensors", "rng_state"} or set(m["file_table"]) != {"model_tensors.bin", "optimizer_tensors.bin", "rng_state.bin", "training_log.jsonl"}:
            raise ValueError("final recovery inventory mismatch")
        data = (artifact.path / "training_log.jsonl").read_bytes()
        if sha256_bytes(data) != m["training_log_digest"]:
            raise ValueError("final recovery log digest mismatch")
        logs = _logs(experiment, data, step, logs)
        decoded = {name: decode_tensor_container(container, (artifact.path / f"{name}.bin").read_bytes(), payload_path=f"{name}.bin")
                   for name, container in m["containers"].items()}
        validate_model_tensors(decoded["model_tensors"])
        previous, last = artifact.manifest_digest, (artifact, decoded)
    return logs, last


def verify_final_terminal(experiment, condition):
    root = final_lineage(experiment, condition)
    if (root / "failure.json").exists():
        raise ValueError("final lineage permanently failed")
    state = validate_model_state(root / "terminal_checkpoint")
    p = state.artifact.manifest["provenance"]
    expected = {"purpose": "final_terminal", **_binding(experiment, condition),
        "terminal_step": experiment.manifest["optimizer_steps"], "protocol_digest": experiment.manifest["protocol_digest"]}
    if set(p) != set(expected) | {"initial_rng_digest", "last_recovery_digest", "training_log_digest"} or any(p[k] != v for k, v in expected.items()):
        raise ValueError("final terminal provenance mismatch")
    training = read_record(root / "training_manifest.json")
    if {k: v for k, v in training.items() if k != "record_digest"} != {
        "schema_version": "classic7_final_training_v1", **_binding(experiment, condition), "initial_rng_digest": p["initial_rng_digest"]}:
        raise ValueError("final training manifest mismatch")
    data = (root / "training_log.jsonl").read_bytes()
    if sha256_bytes(data) != p["training_log_digest"]:
        raise ValueError("final terminal log mismatch")
    _logs(experiment, data, p["terminal_step"], [])
    return state.artifact


def _final_worker(path, digest, condition, resume):
    experiment = open_final_experiment(path)
    if experiment.digest != digest:
        raise ValueError("final experiment changed")
    final_training_gate(experiment)
    root = final_lineage(experiment, condition)
    if (root / "failure.json").exists():
        raise ValueError("failed final lineage cannot resume")
    if root.exists() and not resume:
        raise ValueError("existing final lineage requires explicit resume")
    publication, primary = final_training_inputs(experiment)
    if (root / "terminal_checkpoint").exists():
        return verify_final_terminal(experiment, condition)
    dataset = CanonicalToMDataset(publication.public_view, experiment.manifest["game_ids"], ExperimentCapacity(experiment.config.max_seq_len))
    by_game = {g: [s for s in dataset if s.game_id == g] for g in experiment.manifest["game_ids"]}
    model = build_model(experiment, condition)
    load_model_state(experiment.path / "initial_state", model, experiment.manifest["paired_initial_state_digest"])
    seed = experiment.manifest["seeds"]["rng_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if experiment.config.device == "cuda":
        torch.cuda.manual_seed_all(seed)
    values, scalar = _rng_capture()
    container, payload = encode_tensor_container(values, payload_path="rng_state.bin")
    initial_rng_digest = sha256_bytes(canonical_json_bytes({"container": container, "scalar": scalar}) + payload)
    publish_record(root / "training_manifest.json", {"schema_version": "classic7_final_training_v1",
        **_binding(experiment, condition), "initial_rng_digest": initial_rng_digest})
    config = experiment.config
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=tuple(config.adam_betas),
        eps=config.adam_eps, weight_decay=config.weight_decay, foreach=False, fused=False)
    logs, last = _chain(experiment, condition)
    previous = None
    if last:
        artifact, decoded = last
        restore_model_tensors(model, decoded["model_tensors"])
        _optimizer_restore(optimizer, decoded["optimizer_tensors"], artifact.manifest["optimizer_scalars"])
        _rng_restore(decoded["rng_state"], artifact.manifest["rng_scalars"])
        previous = artifact.manifest_digest
    points = _recovery_points(experiment.manifest["optimizer_steps"], config.recovery_cadence)
    def on_step(step, logs):
        nonlocal previous
        if step in points:
            previous = _publish_point(experiment, condition, step, model, optimizer, logs, previous).manifest_digest
    step = fit_steps(model, optimizer, by_game, primary.rows, experiment.schedule(), device=config.device,
        learning_rate=config.learning_rate, start=len(logs), logs=logs, gate=lambda: final_training_gate(experiment), on_step=on_step)
    final_training_gate(experiment)
    data = canonical_jsonl_bytes(logs)
    publish_run_bytes(root / "training_log.jsonl", data)
    publish_model_state(root / "terminal_checkpoint", model, {"purpose": "final_terminal", **_binding(experiment, condition),
        "terminal_step": step, "protocol_digest": experiment.manifest["protocol_digest"], "initial_rng_digest": initial_rng_digest,
        "last_recovery_digest": previous, "training_log_digest": sha256_bytes(data)})
    return verify_final_terminal(experiment, condition)


def _entry(path, digest, condition, resume, connection):
    try:
        connection.send((True, _final_worker(path, digest, condition, resume).manifest_digest))
    except Exception as error:
        experiment = open_final_experiment(path)
        publish_record(final_lineage(experiment, condition) / "failure.json", {
            "schema_version": "classic7_final_training_failure_v1", "experiment_digest": digest,
            "error": type(error).__name__, "message": str(error)})
        connection.send((False, traceback.format_exc()))
    finally:
        connection.close()


def train_final_condition(experiment, condition, *, resume=False):
    final_training_gate(experiment)
    root = final_lineage(experiment, condition)
    if root.exists() and not resume:
        raise ValueError("existing final lineage requires explicit resume")
    if (root / "failure.json").exists():
        raise ValueError("failed final lineage cannot resume")
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_entry, args=(str(experiment.path), experiment.digest, condition, resume, writer))
    process.start()
    writer.close()
    try:
        success, result = reader.recv()
    except EOFError as error:
        raise RuntimeError("final worker interrupted; explicit deterministic recovery is required") from error
    finally:
        reader.close()
        process.join()
    if not success:
        raise ValueError(result)
    artifact = verify_final_terminal(experiment, condition)
    if artifact.manifest_digest != result:
        raise ValueError("final worker handoff mismatch")
    return artifact


def seal_final_models(experiment):
    final_training_gate(experiment)
    final_training_inputs(experiment)
    checkpoints, rng = [], []
    for condition in TEMPORAL_CONDITIONS:
        terminal = verify_final_terminal(experiment, condition)
        _, last = _chain(experiment, condition, complete=True)
        artifact, _ = last
        p = terminal.manifest["provenance"]
        if (p["last_recovery_digest"] != artifact.manifest_digest
            or (artifact.path / "model_tensors.bin").read_bytes() != (terminal.path / "tensors.bin").read_bytes()
            or p["training_log_digest"] != artifact.manifest["training_log_digest"]):
            raise ValueError("final terminal/recovery mismatch")
        checkpoints.append({"temporal_condition": condition, "checkpoint_digest": terminal.manifest_digest})
        rng.append(p["initial_rng_digest"])
    if len(set(rng)) != 1:
        raise ValueError("final conditions must share initial RNG state")
    return publish_record(experiment.runs_path / "final_model_seal.json", {
        "schema_version": "classic7_final_model_seal_v1", "experiment_digest": experiment.digest,
        "checkpoints": checkpoints, "initial_rng_digest": rng[0]})


def verify_final_seal(experiment):
    path = experiment.runs_path / "final_model_seal.json"
    if not path.is_file():
        raise ValueError("paired final model seal required before final data access")
    seal = read_record(path)
    terminals = [verify_final_terminal(experiment, c) for c in TEMPORAL_CONDITIONS]
    expected = {"schema_version": "classic7_final_model_seal_v1", "experiment_digest": experiment.digest,
        "checkpoints": [{"temporal_condition": c, "checkpoint_digest": t.manifest_digest} for c, t in zip(TEMPORAL_CONDITIONS, terminals)],
        "initial_rng_digest": terminals[0].manifest["provenance"]["initial_rng_digest"]}
    if {k: v for k, v in seal.items() if k != "record_digest"} != expected or any(t.manifest["provenance"]["initial_rng_digest"] != expected["initial_rng_digest"] for t in terminals):
        raise ValueError("final seal binding mismatch")
    return seal
