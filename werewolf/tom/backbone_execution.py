"""Two-stage backbone provenance, fixed-budget training, recovery and all-twenty seal."""

from contextlib import contextmanager
from dataclasses import dataclass, replace
import fcntl
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import re
import traceback

import numpy as np
import torch

from werewolf.artifact_io import (canonical_json_bytes, canonical_jsonl_bytes, sha256_bytes,
    publish_artifact, verify_artifact)
from werewolf.artifact_io.tensor_state import encode_tensor_container, decode_tensor_container
from werewolf.artifact_io.canonical import ensure_durable_directory
from werewolf.development_publication import open_publication
from werewolf.tom import backbone_study as prepared, training
from werewolf.tom.backbone_model import ARCHITECTURES, BackboneToM, graph_identity
from werewolf.tom.dataset import CanonicalToMDataset, ExperimentCapacity
from werewolf.tom.experiment import ExperimentConfig, configure_runtime, runtime_provenance
from werewolf.tom.paper_study import attest_source
from werewolf.tom.protocol import digest_fields
from werewolf.tom.state import model_tensor_values, restore_model_tensors
from werewolf.tom.temporal import TemporalCodeProvider

EXECUTION_VERSION = "classic7_backbone_execution_v1"
CHECKPOINT_VERSION = "classic7_backbone_recovery_v1"
TERMINAL_VERSION = "classic7_backbone_terminal_v1"
SEAL_VERSION = "classic7_backbone_all_twenty_v1"
RNG_VERSION = "backbone_training_rng_v1"
LINEAGES = tuple((a, f, "explicit_day_phase") for a in ARCHITECTURES for f in range(5))


def _same(left, right):
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def _hash(value):
    return sha256_bytes(canonical_json_bytes(value))


def snapshot_identity(artifact):
    m = artifact.manifest
    return {"manifest_digest": artifact.manifest_digest, "study_digest": m["study_digest"],
            **m["protocol_inputs"]["source"]}


def validate_frozen_snapshot(path, identity):
    """Validate pinned preparation bytes, never reinterpret them as current source.

    The caller's exact manifest digest is the trust anchor. All other preparation
    identities remain covered by that digest. No call to source-strict open_study,
    no caught source mismatch, and no mutation of the preparation artifact.
    """
    artifact = verify_artifact(path, expected_artifact_type="backbone_tom_study",
                               expected_schema_version=prepared.STUDY_VERSION)
    m = artifact.manifest
    if (set(identity) != {"manifest_digest", "study_digest", "source_revision", "implementation_digest"}
            or any(re.fullmatch(r"[0-9a-f]{40}" if k == "source_revision" else r"[0-9a-f]{64}", v) is None
                   for k, v in identity.items()) or not _same(snapshot_identity(artifact), identity)):
        raise ValueError("frozen preparation identity mismatch")
    if set(m) != {"artifact_type", "schema_version", "protocol_inputs", "study_digest", "publication_path",
                  "runtime", "temporal_artifact_digest", "folds", "initializations", "file_table", "manifest_digest"}:
        raise ValueError("frozen preparation schema mismatch")
    publication = open_publication(m["publication_path"])
    inputs, population = prepared._inputs(prepared._frozen_design(), publication, m["protocol_inputs"]["source"])
    if (not _same(inputs, m["protocol_inputs"]) or _hash(inputs) != identity["study_digest"]
            or m["runtime"]["implementation_digest"] != identity["implementation_digest"]):
        raise ValueError("frozen preparation design/source mismatch")
    config = ExperimentConfig(**inputs["design"]["reference_protocol"])
    folds, files = prepared._folds(publication, config)
    files["primary.json"] = canonical_json_bytes(population)
    if not _same(folds, m["folds"]) or any((artifact.path / n).read_bytes() != b for n, b in files.items()):
        raise ValueError("frozen preparation population/schedule mismatch")
    provider = TemporalCodeProvider.explicit(artifact.path / "temporal", identity["study_digest"])
    if provider.artifact_digest != m["temporal_artifact_digest"] or len(provider.day) != publication.public_view.max_observed_day + 1:
        raise ValueError("frozen preparation temporal mismatch")
    names = set(files) | {f"temporal/{n}" for n in ("phase_codebook.manifest.json", "phase_codebook.bin",
             "canonical_day_code_table.manifest.json", "canonical_day_code_table.bin")}
    if [(r["architecture"], r["fold"], r["temporal_condition"]) for r in m["initializations"]] != list(LINEAGES):
        raise ValueError("frozen preparation requires exactly twenty lineages")
    shells = {}
    for row in m["initializations"]:
        a, f = row["architecture"], row["fold"]
        base = f"initial/{a}/{f}"
        names.update(f"{base}/{n}" for n in ("manifest.json", "tensors.bin"))
        model = BackboneToM(a, provider, study_seed=config.initialization_seed, fold=f)
        state = prepared.load_initial_state(artifact.path / base, model, study_digest=identity["study_digest"],
            study_seed=config.initialization_seed, fold=f, expected_digest=row["state_digest"])
        expected = {"architecture": a, "fold": f, "temporal_condition": "explicit_day_phase",
            "graph_digest": model.graph["graph_digest"], "state_digest": state.artifact.manifest_digest,
            **state.artifact.manifest["initialization"]}
        if not _same(row, expected) or (f in shells and shells[f] != row["shell_initialization_digest"]):
            raise ValueError("frozen preparation initial graph/pairing mismatch")
        shells[f] = row["shell_initialization_digest"]
    if set(m["file_table"]) != names:
        raise ValueError("frozen preparation file inventory mismatch")
    return artifact


@dataclass(frozen=True)
class Execution:
    artifact: object
    snapshot: object
    config: ExperimentConfig

    @property
    def root(self):
        return self.artifact.path.parent

    @property
    def manifest(self):
        return self.artifact.manifest

    def schedule(self, fold):
        schedule = json.loads((self.snapshot.path / f"folds/{fold}/schedule.json").read_bytes())
        if self.manifest["engineering_only"]:
            schedule = {**schedule, "optimizer_steps": 2, "batches": schedule["batches"][:2]}
        return schedule


def _runtime(config):
    configure_runtime(config)
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("backbone execution requires FP32")
    if config.device == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("backbone CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    source = attest_source()
    runtime = runtime_provenance(config)
    if runtime["implementation_digest"] != source["implementation_digest"]:
        raise ValueError("execution source/runtime mismatch")
    return source, runtime


def create_execution(study_path, expected_manifest_digest, *, engineering=False):
    if type(engineering) is not bool:
        raise ValueError("engineering marker must be boolean")
    snapshot = verify_artifact(study_path, expected_artifact_type="backbone_tom_study",
                               expected_schema_version=prepared.STUDY_VERSION)
    if snapshot.manifest_digest != expected_manifest_digest:
        raise ValueError("prepared manifest trust anchor mismatch")
    identity = snapshot_identity(snapshot)
    snapshot = validate_frozen_snapshot(snapshot.path, identity)
    config = ExperimentConfig(**snapshot.manifest["protocol_inputs"]["design"]["reference_protocol"])
    if not engineering and (config.recovery_cadence != 1000 or any(f["optimizer_steps"] != 25200 for f in snapshot.manifest["folds"])):
        raise ValueError("formal execution requires 25200 steps and cadence 1000")
    source, runtime = _runtime(config)
    if not _same({k: v for k, v in runtime.items() if k != "implementation_digest"},
                 {k: v for k, v in snapshot.manifest["runtime"].items() if k != "implementation_digest"}):
        raise ValueError("prepared/execution runtime differs beyond source implementation")
    root = snapshot.path.parent / ("engineering-executions" if engineering else "executions") / snapshot.manifest_digest
    artifact = publish_artifact(root / "contract", manifest_fields={
        "artifact_type": "backbone_engineering_execution" if engineering else "backbone_execution",
        "schema_version": EXECUTION_VERSION, "engineering_only": engineering,
        "prepared_path": str(snapshot.path), "preparation": identity,
        "execution_source": source, "runtime": runtime,
        "lineages": [list(x) for x in LINEAGES], "training_rng_version": RNG_VERSION,
        "recovery_cadence": 1 if engineering else 1000,
        "expected_optimizer_steps": 2 if engineering else 25200,
    }, files={})
    return Execution(artifact, snapshot, config)


def open_execution(root):
    path = Path(root) / "contract"
    # The explicit namespace selects one type; there is no type fallback.
    engineering = path.parent.parent.name == "engineering-executions"
    artifact = verify_artifact(path, expected_artifact_type=("backbone_engineering_execution" if engineering else "backbone_execution"),
                               expected_schema_version=EXECUTION_VERSION)
    m = artifact.manifest
    if (set(m) != {"artifact_type", "schema_version", "engineering_only", "prepared_path", "preparation",
                  "execution_source", "runtime", "lineages", "training_rng_version", "recovery_cadence",
                  "expected_optimizer_steps", "file_table", "manifest_digest"}
            or m["file_table"] or m["engineering_only"] is not engineering
            or not _same(m["lineages"], [list(x) for x in LINEAGES]) or m["training_rng_version"] != RNG_VERSION
            or type(m["recovery_cadence"]) is not int or m["recovery_cadence"] != (1 if engineering else 1000)
            or type(m["expected_optimizer_steps"]) is not int or m["expected_optimizer_steps"] != (2 if engineering else 25200)):
        raise ValueError("execution schema/lineage/control mismatch")
    snapshot = validate_frozen_snapshot(m["prepared_path"], m["preparation"])
    expected_root = snapshot.path.parent / ("engineering-executions" if engineering else "executions") / snapshot.manifest_digest
    if artifact.path.parent != expected_root:
        raise ValueError("execution ownership path mismatch")
    config = ExperimentConfig(**snapshot.manifest["protocol_inputs"]["design"]["reference_protocol"])
    source, runtime = _runtime(config)
    if (not _same(source, m["execution_source"]) or not _same(runtime, m["runtime"])
            or not _same({k: v for k, v in runtime.items() if k != "implementation_digest"},
                         {k: v for k, v in snapshot.manifest["runtime"].items() if k != "implementation_digest"})):
        raise ValueError("execution source identity changed")
    if not engineering and (config.recovery_cadence != 1000 or any(f["optimizer_steps"] != 25200 for f in snapshot.manifest["folds"])):
        raise ValueError("formal execution budget mismatch")
    return Execution(artifact, snapshot, config)


def _lineage(execution, architecture, fold, temporal="explicit_day_phase"):
    if type(fold) is not int or (architecture, fold, temporal) not in LINEAGES:
        raise ValueError("unknown backbone lineage")
    return execution.root / "runs" / architecture / str(fold)


def _binding(execution, architecture, fold):
    _lineage(execution, architecture, fold)
    initial = next(r for r in execution.snapshot.manifest["initializations"] if r["architecture"] == architecture and r["fold"] == fold)
    return {"execution_digest": execution.artifact.manifest_digest, "preparation": execution.manifest["preparation"],
        "architecture": architecture, "fold": fold, "temporal_condition": "explicit_day_phase",
        "graph": graph_identity(architecture), "initialization": initial,
        "optimizer_protocol": execution.snapshot.manifest["protocol_inputs"]["design"]["reference_protocol"],
        "schedule_digest": _hash(execution.schedule(fold)), "execution_source": execution.manifest["execution_source"],
        "training_rng": {"version": RNG_VERSION, "protocol_seed": execution.config.rng_seed,
                         "study": execution.snapshot.manifest_digest, "architecture": architecture, "fold": fold},
        "engineering_only": execution.manifest["engineering_only"]}


def _reset_rng(binding):
    seed = int.from_bytes(digest_fields(RNG_VERSION, binding["training_rng"]), "big") % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_optimizer(execution, architecture, fold):
    provider = TemporalCodeProvider.explicit(execution.snapshot.path / "temporal", execution.snapshot.manifest["study_digest"])
    model = BackboneToM(architecture, provider, study_seed=execution.config.initialization_seed, fold=fold).to(execution.config.device)
    binding = _binding(execution, architecture, fold)
    prepared.load_initial_state(execution.snapshot.path / "initial" / architecture / str(fold), model,
        study_digest=execution.snapshot.manifest["study_digest"], study_seed=execution.config.initialization_seed,
        fold=fold, expected_digest=binding["initialization"]["state_digest"])
    # Last construction/restore operation precedes the frozen training RNG reset.
    _reset_rng(binding)
    c = execution.config
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.learning_rate, betas=tuple(c.adam_betas),
                                   eps=c.adam_eps, weight_decay=c.weight_decay, foreach=False, fused=False)
    return model, optimizer


def _source_gate(execution):
    source, runtime = _runtime(execution.config)
    if not _same(source, execution.manifest["execution_source"]) or not _same(runtime, execution.manifest["runtime"]):
        raise ValueError("execution source identity changed")


def _training_gate(execution):
    if (execution.root / "seal").exists():
        raise ValueError("sealed execution permanently closes training/recovery")


@contextmanager
def _lock(path, *, shared=False):
    ensure_durable_directory(path.parent)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def _capture(model, optimizer):
    values, scalars = training._optimizer_capture(optimizer)
    rng, rng_scalars = training._rng_capture()
    containers, files = {}, {}
    for name, tensors in (("model", model_tensor_values(model)), ("optimizer", values), ("rng", rng)):
        containers[name], files[f"{name}.bin"] = encode_tensor_container(tensors, payload_path=f"{name}.bin")
    return containers, files, scalars, rng_scalars


def _checkpoint(execution, architecture, fold, step, model, optimizer, logs, previous):
    _source_gate(execution)
    _training_gate(execution)
    containers, files, scalar, rng = _capture(model, optimizer)
    files["loss.jsonl"] = canonical_jsonl_bytes(logs)
    return publish_artifact(_lineage(execution, architecture, fold) / "recovery" / str(step), manifest_fields={
        "artifact_type": "backbone_recovery", "schema_version": CHECKPOINT_VERSION,
        "binding": _binding(execution, architecture, fold), "optimizer_step": step, "schedule_position": step,
        "previous_checkpoint_digest": previous, "containers": containers,
        "optimizer_scalars": scalar, "rng_scalars": rng, "loss_digest": sha256_bytes(files["loss.jsonl"])}, files=files)


def _validate_optimizer(model, optimizer, tensors, scalars, step):
    expected_group = optimizer.state_dict()["param_groups"]
    if not _same(scalars.get("param_groups"), expected_group) or scalars.get("state") != {str(i): {} for i in range(len(list(model.parameters())))}:
        raise ValueError("checkpoint optimizer protocol/parameter order mismatch")
    expected = {}
    for i, parameter in enumerate(model.parameters()):
        expected.update({f"{i}.step": (1,), f"{i}.exp_avg": tuple(parameter.shape), f"{i}.exp_avg_sq": tuple(parameter.shape)})
    if set(tensors) != set(expected):
        raise ValueError("checkpoint optimizer state coverage mismatch")
    for name, tensor in tensors.items():
        a = tensor.array
        if tensor.trainable or a.dtype.str != '<f4' or a.shape != expected[name] or not np.isfinite(a).all():
            raise ValueError("checkpoint optimizer tensor mismatch")
        if name.endswith('.step') and a.item() != step:
            raise ValueError("checkpoint optimizer step mismatch")
        if name.endswith('.exp_avg_sq') and np.any(a < 0):
            raise ValueError("checkpoint optimizer variance mismatch")


def _read_checkpoint(execution, architecture, fold, point, previous, prefix, model, optimizer):
    artifact = verify_artifact(_lineage(execution, architecture, fold) / "recovery" / str(point),
        expected_artifact_type="backbone_recovery", expected_schema_version=CHECKPOINT_VERSION)
    m = artifact.manifest
    if (set(m) != {"artifact_type", "schema_version", "binding", "optimizer_step", "schedule_position",
                  "previous_checkpoint_digest", "containers", "optimizer_scalars", "rng_scalars", "loss_digest", "file_table", "manifest_digest"}
            or not _same(m["binding"], _binding(execution, architecture, fold))
            or type(m["optimizer_step"]) is not int or m["optimizer_step"] != point
            or type(m["schedule_position"]) is not int or m["schedule_position"] != point
            or m["previous_checkpoint_digest"] != previous or set(m["containers"]) != {"model", "optimizer", "rng"}
            or set(m["file_table"]) != {"model.bin", "optimizer.bin", "rng.bin", "loss.jsonl"}):
        raise ValueError("checkpoint binding/step/ancestry mismatch")
    raw = (artifact.path / "loss.jsonl").read_bytes()
    logs = [json.loads(row) for row in raw.splitlines()]
    if (raw != canonical_jsonl_bytes(logs) or sha256_bytes(raw) != m["loss_digest"]
            or len(logs) != point or not _same(logs[:len(prefix)], prefix)):
        raise ValueError("checkpoint loss trace mismatch")
    schedule = execution.schedule(fold)
    for i, row in enumerate(logs):
        if (set(row) != {"step", "game_ids", "loss", "learning_rate"} or type(row['step']) is not int
                or row['step'] != i + 1 or row['game_ids'] != [x['game_id'] for x in schedule['batches'][i]]
                or row['learning_rate'] != execution.config.learning_rate
                or type(row['loss']) not in (float, int) or not math.isfinite(row['loss'])):
            raise ValueError("checkpoint loss schedule mismatch")
    decoded = {n: decode_tensor_container(c, (artifact.path / f"{n}.bin").read_bytes(), payload_path=f"{n}.bin")
               for n, c in m["containers"].items()}
    restore_model_tensors(model, decoded['model'])
    _validate_optimizer(model, optimizer, decoded['optimizer'], m['optimizer_scalars'], point)
    # The shared codec encodes native scalar tensors as a one-element array.
    # AdamW step is the sole scalar field; restore its native zero-dimensional shape.
    optimizer_tensors = {n: replace(t, array=t.array.reshape(())) if n.endswith('.step') else t
                         for n, t in decoded['optimizer'].items()}
    training._optimizer_restore(optimizer, optimizer_tensors, m['optimizer_scalars'])
    expected_rng = {'torch_cpu', 'numpy_state'} | {f'torch_cuda_{i}' for i in range(torch.cuda.device_count()) if torch.cuda.is_initialized()}
    if set(decoded['rng']) != expected_rng or any(t.trainable for t in decoded['rng'].values()):
        raise ValueError("checkpoint RNG device coverage mismatch")
    training._rng_restore(decoded['rng'], m['rng_scalars'])
    # Codec roundtrip must be byte-exact, including optimizer and all RNG streams.
    containers, files, scalar, rng = _capture(model, optimizer)
    if (not _same(containers, m['containers']) or not _same(scalar, m['optimizer_scalars'])
            or not _same(rng, m['rng_scalars'])
            or any(data != (artifact.path / name).read_bytes() for name, data in files.items())):
        raise ValueError("checkpoint restore is not byte exact")
    return artifact, logs


def _recovery_chain(execution, architecture, fold, model, optimizer):
    root = _lineage(execution, architecture, fold) / 'recovery'
    points = training._recovery_points(execution.manifest['expected_optimizer_steps'], execution.manifest['recovery_cadence'])
    actual = sorted(p.name for p in root.iterdir()) if root.exists() else []
    expected = sorted(str(p) for p in points[:len(actual)])
    if actual != expected or any(not (root / name).is_dir() for name in actual):
        raise ValueError("noncontiguous or unexpected recovery checkpoint")
    previous, logs, ancestry = None, [], []
    for point in points[:len(actual)]:
        state, logs = _read_checkpoint(execution, architecture, fold, point, previous, logs, model, optimizer)
        previous = state.manifest_digest
        ancestry.append({'step': point, 'digest': previous})
    return logs, ancestry


def _terminal_fields(execution, architecture, fold, checkpoint, ancestry):
    m = checkpoint.manifest
    return {"artifact_type": "backbone_engineering_terminal" if execution.manifest['engineering_only'] else "backbone_terminal",
        "schema_version": TERMINAL_VERSION, "binding": _binding(execution, architecture, fold),
        "expected_optimizer_steps": execution.manifest['expected_optimizer_steps'], "actual_optimizer_steps": m['optimizer_step'],
        "terminal_model_digest": _hash(m['containers']['model']),
        "terminal_optimizer_digest": _hash({'container': m['containers']['optimizer'], 'scalars': m['optimizer_scalars']}),
        "terminal_rng_digest": _hash({'container': m['containers']['rng'], 'scalars': m['rng_scalars']}),
        "loss_digest": m['loss_digest'], "checkpoint_ancestry": ancestry}


def validate_terminal(execution, architecture, fold):
    _source_gate(execution)
    model, optimizer = _model_optimizer(execution, architecture, fold)
    logs, ancestry = _recovery_chain(execution, architecture, fold, model, optimizer)
    budget = execution.manifest['expected_optimizer_steps']
    if len(logs) != budget or not ancestry or ancestry[-1]['step'] != budget:
        raise ValueError("terminal requires complete recovery history")
    lineage = _lineage(execution, architecture, fold)
    if {p.name for p in lineage.iterdir()} != {'recovery', 'terminal'}:
        raise ValueError("unexpected or duplicate terminal entry")
    checkpoint = verify_artifact(lineage / 'recovery' / str(budget), expected_artifact_type='backbone_recovery', expected_schema_version=CHECKPOINT_VERSION)
    expected = _terminal_fields(execution, architecture, fold, checkpoint, ancestry)
    terminal = verify_artifact(lineage / 'terminal', expected_artifact_type=expected['artifact_type'], expected_schema_version=TERMINAL_VERSION)
    actual = {k: v for k, v in terminal.manifest.items() if k not in {'file_table', 'manifest_digest'}}
    if terminal.manifest['file_table'] or not _same(actual, expected):
        raise ValueError("terminal binding/state/ancestry mismatch")
    return terminal


def _materialize(execution, fold):
    record = execution.snapshot.manifest['folds'][fold]
    ids = record['training_game_ids']
    if execution.manifest['engineering_only']:
        ids = sorted({r['game_id'] for b in execution.schedule(fold)['batches'] for r in b})
    publication = open_publication(execution.snapshot.manifest['publication_path'], game_ids=ids)
    if publication.manifest_digest != execution.snapshot.manifest['protocol_inputs']['publication_digest']:
        raise ValueError("training publication mismatch")
    dataset = CanonicalToMDataset(publication.public_view, ids, ExperimentCapacity(execution.config.max_seq_len))
    by_game = {g: [s for s in dataset if s.game_id == g] for g in ids}
    masks = json.loads((execution.snapshot.path / 'primary.json').read_bytes())['rows']
    return by_game, {g: masks[g] for g in ids}


def _train(execution, architecture, fold, *, interrupt_after=None):
    # Interruption injection is only available to the explicit engineering namespace.
    if interrupt_after is not None and (not execution.manifest['engineering_only'] or type(interrupt_after) is not int or interrupt_after != 1):
        raise ValueError("only engineering step-one interruption is supported")
    _training_gate(execution)
    lineage = _lineage(execution, architecture, fold)
    if (lineage / 'terminal').exists():
        raise ValueError("duplicate completed lineage")
    by_game, masks = _materialize(execution, fold)
    model, optimizer = _model_optimizer(execution, architecture, fold)
    logs, ancestry = _recovery_chain(execution, architecture, fold, model, optimizer)
    points = training._recovery_points(execution.manifest['expected_optimizer_steps'], execution.manifest['recovery_cadence'])
    if execution.config.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    def on_step(step, current_logs):
        if step in points:
            checkpoint = _checkpoint(execution, architecture, fold, step, model, optimizer, current_logs,
                                     ancestry[-1]['digest'] if ancestry else None)
            ancestry.append({'step': step, 'digest': checkpoint.manifest_digest})
        if step == interrupt_after:
            raise InterruptedError('engineering checkpoint published; restart required')
    training.fit_steps(model, optimizer, by_game, masks, execution.schedule(fold), device=execution.config.device,
        learning_rate=execution.config.learning_rate, start=len(logs), logs=logs,
        gate=lambda: _training_gate(execution), on_step=on_step)
    _source_gate(execution)
    checkpoint = verify_artifact(lineage / 'recovery' / str(points[-1]), expected_artifact_type='backbone_recovery', expected_schema_version=CHECKPOINT_VERSION)
    publish_artifact(lineage / 'terminal', manifest_fields=_terminal_fields(execution, architecture, fold, checkpoint, ancestry), files={})
    terminal = validate_terminal(execution, architecture, fold)
    return {'terminal_digest': terminal.manifest_digest,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated() if execution.config.device == 'cuda' else None,
            'peak_reserved_bytes': torch.cuda.max_memory_reserved() if execution.config.device == 'cuda' else None}


def _process_entry(root, expected_digest, architecture, fold, interrupt_after, connection):
    try:
        execution = open_execution(root)
        if execution.artifact.manifest_digest != expected_digest:
            raise ValueError('execution changed before process start')
        result = _train(execution, architecture, fold, interrupt_after=interrupt_after)
        connection.send(('complete', result))
    except InterruptedError as error:
        connection.send(('interrupted', str(error)))
    except Exception:
        connection.send(('failed', traceback.format_exc()))
    finally:
        connection.close()


def run_lineage(root, architecture, fold, *, interrupt_after=None):
    execution = open_execution(root)
    _lineage(execution, architecture, fold)
    with _lock(execution.root / 'execution.lock', shared=True), _lock(execution.root / 'locks' / f'{architecture}-{fold}.lock'):
        _training_gate(execution)
        context = multiprocessing.get_context('spawn')
        reader, writer = context.Pipe(duplex=False)
        process = context.Process(target=_process_entry, args=(str(execution.root), execution.artifact.manifest_digest,
                                  architecture, fold, interrupt_after, writer))
        process.start()
        writer.close()
        try:
            status, result = reader.recv()
        finally:
            reader.close()
            process.join()
        if status == 'failed':
            raise ValueError(result)
        if status == 'interrupted':
            raise InterruptedError(result)
        if process.exitcode != 0:
            raise ValueError('lineage process failed')
        return result


def seal_execution(root):
    execution = open_execution(root)
    with _lock(execution.root / 'execution.lock'):
        runs = execution.root / 'runs'
        if not runs.is_dir() or {p.name for p in runs.iterdir()} != set(ARCHITECTURES):
            raise ValueError('seal requires all twenty lineages')
        for a in ARCHITECTURES:
            if {p.name for p in (runs / a).iterdir()} != {str(f) for f in range(5)}:
                raise ValueError('missing or duplicate lineage prevents seal')
        entries = []
        for a, f, t in LINEAGES:
            terminal = validate_terminal(execution, a, f)
            entries.append({'architecture': a, 'fold': f, 'temporal_condition': t, 'terminal_digest': terminal.manifest_digest})
        _source_gate(execution)
        return publish_artifact(execution.root / 'seal', manifest_fields={
            'artifact_type': 'backbone_engineering_seal' if execution.manifest['engineering_only'] else 'backbone_execution_seal',
            'schema_version': SEAL_VERSION, 'execution_digest': execution.artifact.manifest_digest,
            'preparation': execution.manifest['preparation'], 'execution_source': execution.manifest['execution_source'],
            'terminals': entries, 'graphs': execution.snapshot.manifest['protocol_inputs']['graphs'],
            'initializations': execution.snapshot.manifest['initializations']}, files={})
