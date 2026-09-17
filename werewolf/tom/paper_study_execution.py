"""The fixed two-model paper study: prepare, train twenty lineages, then seal."""

from dataclasses import dataclass
import multiprocessing
import random
from pathlib import Path
import tempfile
import traceback

import numpy as np
import torch

from werewolf.artifact_io import (canonical_json_bytes, sha256_bytes, publish_artifact,
    verify_artifact, open_artifact_envelope, canonical_jsonl_bytes)
from werewolf.artifact_io.tensor_state import encode_tensor_container
from werewolf.development_publication import open_publication
from werewolf.tom.population import load_training_primary
from werewolf.tom.protocol import digest_fields
from werewolf.tom.experiment import (prepare_experiment, open_experiment, validate_runtime,
    TEMPORAL_CONDITIONS, RECOVERY_POLICY)
from werewolf.tom.observer_agnostic import ObserverAgnosticToM, MODEL_GRAPH
from werewolf.tom.observer_agnostic_state import publish_initial_state, load_initial_state
from werewolf.tom.dataset import ExperimentCapacity, CanonicalToMDataset
from werewolf.tom.paper_study import open_contract
from werewolf.tom.run_records import publish_record, read_record, record_with_digest, publish_run_bytes
from werewolf.tom.state import load_model_state, restore_model_tensors, publish_model_state, MODEL_STATE_VERSION
from werewolf.tom.temporal import TemporalCodeProvider
from werewolf.tom import training

EXECUTION_VERSION = "classic7_paper_study_execution_v1"
AGNOSTIC_VERSION = "classic7_agnostic_experiment_v1"
SEAL_VERSION = "classic7_paper_all_twenty_v1"
FAMILIES = ("full_observer_conditioned", "observer_agnostic_public_history")
EXPECTED_LINEAGES = tuple((model, condition, fold) for model in FAMILIES
                          for condition in TEMPORAL_CONDITIONS for fold in range(5))


@dataclass(frozen=True)
class AgnosticExperiment:
    artifact: object
    full: object

    @property
    def path(self):
        return self.artifact.path

    @property
    def digest(self):
        return self.artifact.manifest_digest

    @property
    def config(self):
        return self.full.config

    @property
    def runs_path(self):
        return self.path.parent.parent / "runs" / self.digest

    @property
    def manifest(self):
        return {"runtime": self.full.manifest["runtime"],
                "protocol_digest": self.artifact.manifest["protocol_digest"]}

    def fold(self, index):
        return {**self.full.fold(index), "paired_initial_state_digest":
                self.artifact.manifest["initial_states"][index]}

    def schedule(self, index):
        return self.full.schedule(index)


def _agnostic_protocol(full, contract):
    return {"full_protocol_digest": full.manifest["protocol_digest"],
            "study_contract_digest": contract.manifest_digest, "model_graph": MODEL_GRAPH}


def _open_agnostic(path, full, contract):
    artifact = verify_artifact(path, expected_artifact_type="paper_agnostic_experiment",
                               expected_schema_version=AGNOSTIC_VERSION)
    m = artifact.manifest
    if set(m) != {"artifact_type", "schema_version", "full_experiment_digest", "protocol_inputs",
                 "protocol_digest", "initial_states", "file_table", "manifest_digest"}:
        raise ValueError("agnostic experiment schema mismatch")
    inputs = _agnostic_protocol(full, contract)
    if (m["full_experiment_digest"] != full.digest or m["protocol_inputs"] != inputs
            or m["protocol_digest"] != sha256_bytes(canonical_json_bytes(inputs))):
        raise ValueError("agnostic experiment parent/model protocol mismatch")
    expected_files = {f"folds/{i}/initial_state/{name}" for i in range(5)
                      for name in ("manifest.json", "tensors.bin")}
    if set(m["file_table"]) != expected_files or len(m["initial_states"]) != 5:
        raise ValueError("agnostic initial-state coverage mismatch")
    experiment = AgnosticExperiment(artifact, full)
    for fold in range(5):
        with torch.random.fork_rng(devices=[]):
            model = ObserverAgnosticToM(ExperimentCapacity(1), None)
        load_initial_state(path / "folds" / str(fold) / "initial_state", model,
            experiment.fold(fold)["paired_initial_state_digest"],
            full_initial_path=full.path / "folds" / str(fold) / "initial_state")
    return experiment


def _prepare_agnostic(full, contract, destination):
    files, initial_states = {}, []
    with tempfile.TemporaryDirectory(prefix="paper-initial-") as temp:
        for fold in range(5):
            state = publish_initial_state(full.path / "folds" / str(fold) / "initial_state",
                full.fold(fold)["paired_initial_state_digest"], Path(temp).resolve() / str(fold),
                initialization_seed=full.config.initialization_seed, fold=fold)
            initial_states.append(state.artifact.manifest_digest)
            for name in ("manifest.json", "tensors.bin"):
                files[f"folds/{fold}/initial_state/{name}"] = (state.artifact.path / name).read_bytes()
    inputs = _agnostic_protocol(full, contract)
    publish_artifact(destination, manifest_fields={"artifact_type": "paper_agnostic_experiment",
        "schema_version": AGNOSTIC_VERSION, "full_experiment_digest": full.digest,
        "protocol_inputs": inputs, "protocol_digest": sha256_bytes(canonical_json_bytes(inputs)),
        "initial_states": initial_states}, files=files)
    return _open_agnostic(destination, full, contract)


@dataclass(frozen=True)
class PaperStudy:
    artifact: object
    full: object
    agnostic: AgnosticExperiment

    @property
    def seal_path(self):
        return self.artifact.path.parent / "study_seal.json"


def prepare_study(contract_path, publication, config, destination):
    contract = open_contract(contract_path)
    c = contract.manifest["contract"]
    source = contract.manifest["source"]
    if (publication.public_view.publication_id != c["publication_id"]
            or config.source_revision != source["source_revision"]
            or config.bootstrap_replicates != c["bootstrap"]["replicates"]
            or config.confidence_level != c["bootstrap"]["confidence"]):
        raise ValueError("paper study publication/config binding mismatch")
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("paper study destination already exists")
    full = prepare_experiment(publication, config, destination / "artifacts" / "full")
    if full.manifest["runtime"]["implementation_digest"] != source["implementation_digest"]:
        raise ValueError("paper study implementation binding mismatch")
    agnostic = _prepare_agnostic(full, contract, destination / "artifacts" / "agnostic")
    publish_artifact(destination / "execution", manifest_fields={
        "artifact_type": "paper_study_execution", "schema_version": EXECUTION_VERSION,
        "contract_path": str(contract.path), "contract_digest": contract.manifest_digest,
        "full_experiment_digest": full.digest, "agnostic_experiment_digest": agnostic.digest,
        "expected_lineages": [list(row) for row in EXPECTED_LINEAGES]}, files={})
    return open_study(destination / "execution")


def open_study(path):
    artifact = verify_artifact(path, expected_artifact_type="paper_study_execution",
                               expected_schema_version=EXECUTION_VERSION)
    m = artifact.manifest
    if (set(m) != {"artifact_type", "schema_version", "contract_path", "contract_digest",
            "full_experiment_digest", "agnostic_experiment_digest", "expected_lineages",
            "file_table", "manifest_digest"} or m["file_table"]
            or canonical_json_bytes(m["expected_lineages"]) != canonical_json_bytes(EXPECTED_LINEAGES)):
        raise ValueError("paper study exact twenty-lineage schema mismatch")
    contract = open_contract(m["contract_path"])
    full = open_experiment(artifact.path.parent / "artifacts" / "full")
    validate_runtime(full)
    source = contract.manifest["source"]
    if (contract.manifest_digest != m["contract_digest"] or full.digest != m["full_experiment_digest"]
            or full.config.source_revision != source["source_revision"]
            or full.manifest["runtime"]["implementation_digest"] != source["implementation_digest"]
            or full.manifest["publication_id"] != contract.manifest["contract"]["publication_id"]
            or full.config.bootstrap_replicates != contract.manifest["contract"]["bootstrap"]["replicates"]
            or full.config.confidence_level != contract.manifest["contract"]["bootstrap"]["confidence"]):
        raise ValueError("paper study parent/source/control mismatch")
    agnostic = _open_agnostic(artifact.path.parent / "artifacts" / "agnostic", full, contract)
    if agnostic.digest != m["agnostic_experiment_digest"]:
        raise ValueError("paper study agnostic identity mismatch")
    return PaperStudy(artifact, full, agnostic)


def _lineage(study, family, condition, fold):
    if type(fold) is not int or (family, condition, fold) not in EXPECTED_LINEAGES:
        raise ValueError("unknown paper study lineage")
    experiment = study.full if family == FAMILIES[0] else study.agnostic
    return experiment, training.lineage_path(experiment, fold, condition)


def _binding(study, family, condition, fold):
    return {"schema_version": EXECUTION_VERSION, "study_digest": study.artifact.manifest_digest,
            "model_family": family, "temporal_condition": condition, "fold": fold}


def _training_gate(study):
    if study.seal_path.exists():
        raise ValueError("all-twenty study seal permanently closes training/recovery")
    for experiment in (study.full, study.agnostic):
        training._training_gate(experiment)


def _baseline_model(experiment, condition):
    constructor = TemporalCodeProvider.implicit if condition == "implicit" else TemporalCodeProvider.explicit
    full = experiment.full
    provider = constructor(full.path / "temporal", full.manifest["protocol_digest"])
    if provider.artifact_digest != full.manifest["temporal_artifact_digest"]:
        raise ValueError("shared temporal artifact mismatch")
    return ObserverAgnosticToM(ExperimentCapacity(full.config.max_seq_len), provider).to(full.config.device)


def _verify_baseline_records(experiment, fold, condition, state):
    """Baseline terminal records use the existing training artifact contract."""
    path = state.artifact.path
    if (path.parent / "failure.json").exists():
        raise ValueError("lineage permanently failed closed")
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
    training_record = read_record(path.parent / "training_manifest.json")
    expected_training = {"schema_version": "classic7_training_manifest_v1", "experiment_digest": experiment.digest,
        "fold": fold, "temporal_condition": condition, "initial_rng_digest": p["initial_rng_digest"],
        "paired_initial_state_digest": p["paired_initial_state_digest"], "schedule_digest": p["schedule_digest"],
        "runtime": experiment.manifest["runtime"], "recovery_policy": RECOVERY_POLICY}
    if {k: v for k, v in training_record.items() if k != "record_digest"} != expected_training:
        raise ValueError("terminal training-manifest/RNG provenance mismatch")
    run = read_record(path.parent / "run_provenance.json")
    if {k: v for k, v in run.items() if k != "record_digest"} != {
        "schema_version": "classic7_run_provenance_v1", "runtime": experiment.manifest["runtime"], "experiment_digest": experiment.digest}:
        raise ValueError("terminal run provenance mismatch")
    training._validate_training_log(experiment, fold, log, expected["terminal_step"], [])
    return state.artifact


def _baseline_terminal_recovery(experiment, fold, condition, terminal):
    """Validate the complete recovery chain; caller validates returned model graph."""
    root = terminal.path.parent / "recovery"
    points = training._recovery_points(experiment.fold(fold)["optimizer_steps"], experiment.config.recovery_cadence)
    actual = sorted(int(p.name) for p in root.iterdir() if not p.name.startswith("."))
    if actual != points:
        raise ValueError("terminal recovery coverage mismatch")
    previous = None
    logs = []
    for point in points:
        artifact, logs, decoded = training._read_recovery(experiment, fold, condition, root / str(point), point, previous, logs)
        m = artifact.manifest
        previous = artifact.manifest_digest
    p = terminal.manifest["provenance"]
    if (previous != p["last_recovery_digest"] or m["training_log_prefix_digest"] != p["training_log_digest"]
        or (artifact.path / "model_tensors.bin").read_bytes() != (terminal.path / "tensors.bin").read_bytes()):
        raise ValueError("terminal/final-recovery digest or model mismatch")
    return decoded["model_tensors"]


def _train_baseline(study, condition, fold):
    """Compose the frozen Dataset, optimizer loop and recovery primitives."""
    experiment = study.agnostic
    full = study.full
    lineage = training.lineage_path(experiment, fold, condition)
    fold_record = experiment.fold(fold)
    schedule = experiment.schedule(fold)
    publication = open_publication(full.manifest["publication_path"], game_ids=fold_record["training_game_ids"])
    if (publication.manifest_digest != full.manifest["publication_digest"]
            or publication.public_view.fold_manifest.manifest_digest != full.manifest["fold_manifest_digest"]
            or set(publication.public_view.fold_manifest.folds[fold].game_ids) != set(fold_record["held_out_game_ids"])):
        raise ValueError("publication/fold parent mismatch")
    dataset = CanonicalToMDataset(publication.public_view, fold_record["training_game_ids"],
                                 ExperimentCapacity(experiment.config.max_seq_len))
    by_game = {g: [sample for sample in dataset if sample.game_id == g] for g in fold_record["training_game_ids"]}
    masks = {g.game_id: load_training_primary(full, fold, g)["rows"] for g in publication.public_view.games}
    model = _baseline_model(experiment, condition)
    load_initial_state(experiment.path / "folds" / str(fold) / "initial_state", model,
        fold_record["paired_initial_state_digest"], full_initial_path=full.path / "folds" / str(fold) / "initial_state")
    seed = int.from_bytes(digest_fields("paired_rng_v1", experiment.config.rng_seed, fold), "big") % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if experiment.config.device == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng_values, rng_scalar = training._rng_capture()
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
    points = training._recovery_points(budget, experiment.config.recovery_cadence)
    recovery_root = lineage / "recovery"
    existing = sorted(int(p.name) for p in recovery_root.iterdir() if not p.name.startswith(".")) if recovery_root.exists() else []
    if existing != points[:len(existing)]:
        raise ValueError("non-contiguous recovery chain")
    logs, step, previous = [], 0, None
    for point in existing:
        artifact, candidate_logs, decoded = training._read_recovery(experiment, fold, condition, recovery_root / str(point), point, previous, logs)
        m = artifact.manifest
        restore_model_tensors(model, decoded["model_tensors"])
        training._optimizer_restore(optimizer, decoded["optimizer_tensors"], m["optimizer_scalars"])
        training._rng_restore(decoded["rng_state"], m["rng_scalars"])
        logs, step, previous = candidate_logs, point, artifact.manifest_digest
    def on_step(step, logs):
        nonlocal previous
        if step in points:
            recovery = training._publish_recovery(recovery_root / str(step), experiment, fold, condition, step, model, optimizer, logs, previous)
            previous = recovery.manifest_digest
    step = training.fit_steps(model, optimizer, by_game, masks, schedule, device=experiment.config.device,
        learning_rate=experiment.config.learning_rate, start=step, logs=logs,
        gate=lambda: _training_gate(study), on_step=on_step)
    log_bytes = canonical_jsonl_bytes(logs)
    publish_run_bytes(lineage / "training_log.jsonl", log_bytes)
    publish_model_state(lineage / "terminal_checkpoint", model, {
        "purpose": "terminal", "experiment_digest": experiment.digest, "fold": fold, "temporal_condition": condition,
        "terminal_step": step, "paired_initial_state_digest": fold_record["paired_initial_state_digest"],
        "schedule_digest": fold_record["schedule_digest"], "protocol_digest": experiment.manifest["protocol_digest"],
        "training_log_digest": sha256_bytes(log_bytes), "last_recovery_digest": previous, "initial_rng_digest": initial_rng_digest})
    publish_record(lineage / "run_provenance.json", {"schema_version": "classic7_run_provenance_v1", "runtime": experiment.manifest["runtime"], "experiment_digest": experiment.digest})


def _verify_terminal(study, family, condition, fold, *, recovery=False):
    experiment, path = _lineage(study, family, condition, fold)
    if read_record(path / "study_lineage.json") != record_with_digest(_binding(study, family, condition, fold)):
        raise ValueError("study lineage parent/model/fold/temporal binding mismatch")
    if family == FAMILIES[0]:
        terminal = training.verify_terminal(experiment, fold, condition)
        if recovery:
            training._verify_terminal_recovery(experiment, fold, condition, terminal)
    else:
        with torch.random.fork_rng(devices=[]):
            model = ObserverAgnosticToM(ExperimentCapacity(1), None)
        envelope = open_artifact_envelope(path / "terminal_checkpoint", expected_artifact_type="canonical_model_state",
                                         expected_schema_version=MODEL_STATE_VERSION)
        state = load_model_state(envelope.path, model, envelope.manifest_digest)
        terminal = _verify_baseline_records(experiment, fold, condition, state)
        if recovery:
            restore_model_tensors(model, _baseline_terminal_recovery(experiment, fold, condition, terminal))
    return terminal


def _train_lineage(study, family, condition, fold):
    _training_gate(study)
    experiment, path = _lineage(study, family, condition, fold)
    if (path / "failure.json").exists():
        raise ValueError("lineage permanently failed closed")
    if (path / "terminal_checkpoint").exists():
        return _verify_terminal(study, family, condition, fold)
    publish_record(path / "study_lineage.json", _binding(study, family, condition, fold))
    if family == FAMILIES[0]:
        training._train_worker(str(experiment.path), experiment.digest, fold, condition)
    else:
        _train_baseline(study, condition, fold)
    return _verify_terminal(study, family, condition, fold)


def _process_entry(path, expected_digest, family, condition, fold, connection):
    study = None
    try:
        study = open_study(path)
        if study.artifact.manifest_digest != expected_digest:
            raise ValueError("study identity changed")
        terminal = _train_lineage(study, family, condition, fold)
        connection.send((True, terminal.manifest_digest))
    except Exception as error:
        if study is not None and not study.seal_path.exists():
            experiment, lineage = _lineage(study, family, condition, fold)
            if not (lineage / "failure.json").exists():
                publish_record(lineage / "failure.json", {"schema_version": "classic7_training_failure_v1",
                    "experiment_digest": experiment.digest, "error": type(error).__name__, "message": str(error)})
        connection.send((False, traceback.format_exc()))
    finally:
        connection.close()


def _terminal_entries(study, *, recovery):
    # Reject extra run lineage directories as well as missing expected terminals.
    for experiment in (study.full, study.agnostic):
        if experiment.runs_path.exists():
            for path in experiment.runs_path.iterdir():
                if path.name not in TEMPORAL_CONDITIONS or not path.is_dir():
                    raise ValueError("unexpected study run entry")
                if any(p.name not in {str(f) for f in range(5)} or not p.is_dir() for p in path.iterdir()):
                    raise ValueError("unexpected study fold entry")
    entries = []
    for family, condition, fold in EXPECTED_LINEAGES:
        _, lineage = _lineage(study, family, condition, fold)
        if lineage.exists():
            allowed = {"study_lineage.json", "training_manifest.json", "training_log.jsonl",
                       "run_provenance.json", "terminal_checkpoint", "recovery", "failure.json"}
            if any(p.name not in allowed and not p.name.startswith(".") for p in lineage.iterdir()):
                raise ValueError("unexpected study lineage entry")
        terminal = _verify_terminal(study, family, condition, fold, recovery=recovery)
        experiment, _ = _lineage(study, family, condition, fold)
        entries.append({**_binding(study, family, condition, fold),
            "checkpoint_digest": terminal.manifest_digest,
            "schedule_digest": experiment.fold(fold)["schedule_digest"],
            "initial_state_digest": experiment.fold(fold)["paired_initial_state_digest"]})
    return entries


def seal_study(path):
    study = open_study(path)
    _training_gate(study)
    entries = _terminal_entries(study, recovery=True)
    return publish_record(study.seal_path, {"schema_version": SEAL_VERSION,
        "study_digest": study.artifact.manifest_digest, "checkpoints": entries})


def evaluation_eligibility(path):
    """The paper-study held-out barrier; does not load held-out data or score anything."""
    study = open_study(path)
    if not study.seal_path.is_file():
        raise ValueError("all-twenty study seal required before paper held-out access")
    seal = read_record(study.seal_path)
    expected = record_with_digest({"schema_version": SEAL_VERSION,
        "study_digest": study.artifact.manifest_digest,
        "checkpoints": _terminal_entries(study, recovery=False)})
    if seal != expected:
        raise ValueError("all-twenty study seal coverage/identity mismatch")
    return seal


def run_study(path):
    study = open_study(path)
    _training_gate(study)
    for family, condition, fold in EXPECTED_LINEAGES:
        _, lineage = _lineage(study, family, condition, fold)
        if (lineage / "terminal_checkpoint").exists():
            _verify_terminal(study, family, condition, fold)
            continue
        context = multiprocessing.get_context("spawn")
        reader, writer = context.Pipe(duplex=False)
        process = context.Process(target=_process_entry, args=(str(study.artifact.path),
            study.artifact.manifest_digest, family, condition, fold, writer))
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
        terminal = _verify_terminal(study, family, condition, fold)
        if terminal.manifest_digest != result:
            raise ValueError("study terminal process handoff mismatch")
    return seal_study(path)
