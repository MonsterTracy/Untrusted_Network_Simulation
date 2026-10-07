"""Immutable backbone-study preparation and opening, without a training lifecycle."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile

from werewolf.artifact_io import (canonical_json_bytes, sha256_bytes, publish_artifact,
    verify_artifact, publish_tensor_state, verify_tensor_state)
from werewolf.artifact_io.tensor_state import encode_tensor_container
from werewolf.development_publication import open_publication
from werewolf.tom.backbone_model import (ARCHITECTURES, BACKBONE_CONFIGS, BackboneToM,
    graph_identity, initialization_seeds)
from werewolf.tom.experiment import ExperimentConfig, configure_runtime, runtime_provenance
from werewolf.tom.paper_study import attest_source
from werewolf.tom.population import select_primary_population
from werewolf.tom.protocol import training_schedule, BOOTSTRAP_VERSION
from werewolf.tom.state import model_tensor_values, restore_model_tensors
from werewolf.tom.temporal import publish_temporal, TemporalCodeProvider

STUDY_VERSION = "classic7_backbone_study_v1"
INITIAL_VERSION = "classic7_backbone_initial_v1"
PAIRED_STUDY_VERSION = "classic7_qwen3_paired_temporal_study_v2"
PAIRED_INITIAL_VERSION = "classic7_qwen3_paired_initial_v1"
PAIRED_CONDITIONS = ("explicit_day_phase", "implicit")
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/paper/tom-backbone-study-v1.json"


def _digest(value):
    return sha256_bytes(canonical_json_bytes(value))


def _frozen_design():
    design = json.loads(CONFIG_PATH.read_bytes())
    if (design["schema_version"] != STUDY_VERSION or design["architectures"] != BACKBONE_CONFIGS
            or design["temporal_conditions"] != ["explicit_day_phase"]
            or design["fold_count"] != 5 or design["terminal_lineages"] != 20
            or design["reference_protocol"]["max_seq_len"] != 1024):
        raise ValueError("invalid frozen backbone design")
    return design


def _paired_design():
    design = deepcopy(_frozen_design())
    design.update(schema_version=PAIRED_STUDY_VERSION,
                  temporal_conditions=list(PAIRED_CONDITIONS), terminal_lineages=10,
                  architectures={"qwen3": design["architectures"]["qwen3"]})
    design["initialization"] = {"temporal_pairing": "exact_shared_fold_tensors",
                               "training_rng": "selection_blind_paired_v2"}
    return design


def _design(schema):
    if schema == STUDY_VERSION:
        return _frozen_design()
    if schema == PAIRED_STUDY_VERSION:
        return _paired_design()
    raise ValueError("unknown backbone study schema")


def _architectures(design):
    return ("qwen3",) if design["schema_version"] == PAIRED_STUDY_VERSION else ARCHITECTURES


def initialization_lineages(design):
    temporal = "shared" if design["schema_version"] == PAIRED_STUDY_VERSION else "explicit_day_phase"
    return [(a, f, temporal) for a in _architectures(design) for f in range(5)]


def _inputs(design, publication, source):
    view = publication.public_view
    if (view.publication_id != design["publication_id"] or len(view.game_ids) != design["game_count"]
            or view.max_structured_token_count > 1024):
        raise ValueError("backbone study publication/capacity mismatch")
    folds = view.fold_manifest.folds
    if ([f.fold_index for f in folds] != list(range(5))
            or sorted(g for f in folds for g in f.game_ids) != sorted(view.game_ids)):
        raise ValueError("backbone study fold coverage mismatch")
    primary = select_primary_population(publication)
    population = {"metadata": primary.metadata, "rows": primary.rows}
    return {
        "design": design, "config_digest": _digest(design), "source": source,
        "publication_digest": publication.manifest_digest,
        "fold_manifest_digest": view.fold_manifest.manifest_digest,
        "development_game_set_digest": view.development_game_set_digest,
        "primary_identity": primary.metadata, "primary_digest": _digest(population),
        "bootstrap_version": BOOTSTRAP_VERSION,
        "graphs": [graph_identity(a) for a in _architectures(design)],
    }, population


def _folds(publication, config):
    result, schedules = [], {}
    view = publication.public_view
    for fold in view.fold_manifest.folds:
        training = [g for g in view.game_ids if g not in fold.game_ids]
        schedule = training_schedule(config.schedule_seed, fold.fold_index, training,
                                     config.rotation_cycles, config.game_batch_size)
        schedules[f"folds/{fold.fold_index}/schedule.json"] = canonical_json_bytes(schedule)
        result.append({"fold": fold.fold_index, "training_game_ids": training,
            "held_out_game_ids": list(fold.game_ids), "schedule_digest": _digest(schedule),
            "optimizer_steps": schedule["optimizer_steps"]})
    return result, schedules


def _state_digests(container):
    # Offsets depend on surrounding tensors; exclude them from the shell identity.
    entries = [{k: v for k, v in t.items() if k != "byte_offset"} for t in container["tensors"]]
    return {
        "shell_initialization_digest": _digest([t for t in entries if not t["name"].startswith("transformer.")]),
        "backbone_initialization_digest": _digest([t for t in entries if t["name"].startswith("transformer.")]),
        "combined_initialization_digest": _digest(entries),
    }


def _initial_binding(model, study_digest, study_seed, fold, *, paired=False):
    return {"study_digest": study_digest, "graph": model.graph, "fold": fold,
        "temporal_condition": "shared" if paired else "explicit_day_phase",
        "seeds": initialization_seeds(study_seed, fold, model.graph["architecture"])}


def load_initial_state(path, model, *, study_digest, study_seed, fold, expected_digest, paired=False):
    """Only the frozen initial artifact type, graph, fold, seed and exact FP32 graph."""
    state = verify_tensor_state(path, expected_artifact_type="backbone_initial_state",
                                expected_schema_version=PAIRED_INITIAL_VERSION if paired else INITIAL_VERSION)
    m = state.artifact.manifest
    if (set(m) != {"artifact_type", "schema_version", "binding", "initialization", "tensor_container",
                   "file_table", "manifest_digest"}
            or state.artifact.manifest_digest != expected_digest
            or _digest(m["binding"]) != _digest(_initial_binding(model, study_digest, study_seed, fold, paired=paired))
            or _digest(model.graph) != _digest(graph_identity(model.graph["architecture"]))
            or m["initialization"] != _state_digests(m["tensor_container"])
            or set(m["file_table"]) != {"tensors.bin"}):
        raise ValueError("backbone initial state graph/config/binding mismatch")
    restore_model_tensors(model, state.tensors)
    return state


def prepare_study(config_path, publication_path, destination, *, paired_temporal=False):
    design = json.loads(Path(config_path).read_bytes())
    if canonical_json_bytes(design) != canonical_json_bytes(_frozen_design()):
        raise ValueError("backbone study differs from frozen design")
    if type(paired_temporal) is not bool:
        raise ValueError("paired temporal mode must be boolean")
    if paired_temporal:
        design = _paired_design()
    source = attest_source()
    publication = open_publication(publication_path)
    inputs, population = _inputs(design, publication, source)
    study_digest = _digest(inputs)
    config = ExperimentConfig(**design["reference_protocol"])
    configure_runtime(config)
    runtime = runtime_provenance(config)
    if runtime["implementation_digest"] != source["implementation_digest"]:
        raise ValueError("backbone implementation attestation mismatch")
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("backbone study destination already exists")
    folds, files = _folds(publication, config)
    files["primary.json"] = canonical_json_bytes(population)
    initializations = []
    # All nested artifacts are staged and validated before the public destination exists.
    with tempfile.TemporaryDirectory(prefix="backbone-study-") as directory:
        root = Path(directory).resolve()
        temporal = publish_temporal(root / "temporal", publication.public_view.max_observed_day, study_digest)
        provider = TemporalCodeProvider.explicit(root / "temporal", study_digest)
        for architecture in _architectures(design):
            for fold in range(5):
                model = BackboneToM(architecture, provider, study_seed=config.initialization_seed, fold=fold)
                tensors = model_tensor_values(model)
                container, _ = encode_tensor_container(tensors, payload_path="tensors.bin")
                state = publish_tensor_state(root / "initial" / architecture / str(fold), manifest_fields={
                    "artifact_type": "backbone_initial_state", "schema_version": PAIRED_INITIAL_VERSION if paired_temporal else INITIAL_VERSION,
                    "binding": _initial_binding(model, study_digest, config.initialization_seed, fold, paired=paired_temporal),
                    "initialization": _state_digests(container)}, tensors=tensors)
                initializations.append({"architecture": architecture, "fold": fold,
                    "temporal_condition": "shared" if paired_temporal else "explicit_day_phase", "graph_digest": model.graph["graph_digest"],
                    "state_digest": state.artifact.manifest_digest, **_state_digests(container)})
        files.update({p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()})
        fields = {"artifact_type": "backbone_tom_study", "schema_version": design["schema_version"],
            "protocol_inputs": inputs, "study_digest": study_digest,
            "publication_path": str(publication.path.resolve()), "runtime": runtime,
            "temporal_artifact_digest": temporal.manifest_digest,
            "folds": folds, "initializations": initializations}
        staged = publish_artifact(root / "validated", manifest_fields=fields, files=files)
        open_study(staged.path)
        return publish_artifact(destination, manifest_fields=fields, files=files)


def verify_study_artifact(path):
    schema = json.loads((Path(path) / "manifest.json").read_bytes())["schema_version"]
    _design(schema)  # Only the frozen historical and paired study contracts.
    return verify_artifact(path, expected_artifact_type="backbone_tom_study",
                           expected_schema_version=schema)


def open_study(path):
    artifact = verify_study_artifact(path)
    m = artifact.manifest
    if set(m) != {"artifact_type", "schema_version", "protocol_inputs", "study_digest", "publication_path",
                  "runtime", "temporal_artifact_digest", "folds", "initializations", "file_table", "manifest_digest"}:
        raise ValueError("backbone study schema mismatch")
    design = _design(m["schema_version"])
    paired_temporal = design["schema_version"] == PAIRED_STUDY_VERSION
    source = attest_source()
    publication = open_publication(m["publication_path"])
    inputs, population = _inputs(design, publication, source)
    if _digest(m["protocol_inputs"]) != _digest(inputs) or m["study_digest"] != _digest(inputs):
        raise ValueError("backbone study source/config/publication binding mismatch")
    config = ExperimentConfig(**design["reference_protocol"])
    configure_runtime(config)
    if (m["runtime"] != runtime_provenance(config)
            or m["runtime"]["implementation_digest"] != source["implementation_digest"]):
        raise ValueError("backbone study runtime/implementation mismatch")
    folds, expected_files = _folds(publication, config)
    expected_files["primary.json"] = canonical_json_bytes(population)
    if _digest(m["folds"]) != _digest(folds):
        raise ValueError("backbone study schedule/fold mismatch")
    for name, data in expected_files.items():
        if (artifact.path / name).read_bytes() != data:
            raise ValueError("backbone study population/schedule mismatch")
    provider = TemporalCodeProvider.explicit(artifact.path / "temporal", m["study_digest"])
    if (provider.artifact_digest != m["temporal_artifact_digest"]
            or len(provider.day) != publication.public_view.max_observed_day + 1):
        raise ValueError("backbone study temporal binding mismatch")
    expected_names = set(expected_files) | {f"temporal/{name}" for name in (
        "phase_codebook.manifest.json", "phase_codebook.bin",
        "canonical_day_code_table.manifest.json", "canonical_day_code_table.bin")}
    expected_rows, shells = [], {}
    for architecture in _architectures(design):
        for fold in range(5):
            base = f"initial/{architecture}/{fold}"
            expected_names.update(f"{base}/{name}" for name in ("manifest.json", "tensors.bin"))
            model = BackboneToM(architecture, provider, study_seed=config.initialization_seed, fold=fold)
            container, _ = encode_tensor_container(model_tensor_values(model), payload_path="tensors.bin")
            digests = _state_digests(container)
            state = verify_tensor_state(artifact.path / base, expected_artifact_type="backbone_initial_state",
                                        expected_schema_version=PAIRED_INITIAL_VERSION if paired_temporal else INITIAL_VERSION)
            load_initial_state(artifact.path / base, model, study_digest=m["study_digest"],
                study_seed=config.initialization_seed, fold=fold, expected_digest=state.artifact.manifest_digest, paired=paired_temporal)
            if state.artifact.manifest["initialization"] != digests:
                raise ValueError("backbone study seeded initialization mismatch")
            shell = digests["shell_initialization_digest"]
            if fold in shells and shells[fold] != shell:
                raise ValueError("backbone study shell pairing mismatch")
            shells[fold] = shell
            expected_rows.append({"architecture": architecture, "fold": fold,
                "temporal_condition": "shared" if paired_temporal else "explicit_day_phase", "graph_digest": model.graph["graph_digest"],
                "state_digest": state.artifact.manifest_digest, **digests})
    if _digest(m["initializations"]) != _digest(expected_rows) or set(m["file_table"]) != expected_names:
        raise ValueError("backbone study lineage/file coverage mismatch")
    return artifact
