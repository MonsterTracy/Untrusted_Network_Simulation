"""Exact, auditable projection of an untrained Full fold state into the baseline."""

import torch

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes, publish_tensor_state, verify_tensor_state
from werewolf.tom.dataset import ExperimentCapacity
from werewolf.tom.model import MODEL_GRAPH as FULL_GRAPH
from werewolf.tom.observer_agnostic import ObserverAgnosticToM, MODEL_GRAPH
from werewolf.tom.protocol import digest_fields
from werewolf.tom.state import validate_model_state, model_tensor_values, restore_model_tensors

INITIAL_VERSION = "classic7_observer_agnostic_initial_v1"
# This enumerates the fixed graph, not a prefix/wildcard selection from a checkpoint.
LAYER_KEYS = (
    "self_attn.q_proj.weight", "self_attn.q_proj.bias", "self_attn.k_proj.weight", "self_attn.k_proj.bias",
    "self_attn.v_proj.weight", "self_attn.v_proj.bias", "self_attn.o_proj.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    "input_layernorm.weight", "post_attention_layernorm.weight",
)
SHARED_KEYS = (
    "source_embedding.weight", "target_embedding.weight", "action_embedding.weight", "event_embedding.weight",
    *(f"transformer.layers.{layer}.{key}" for layer in range(4) for key in LAYER_KEYS),
    "transformer.norm.weight", "query_attention.in_proj_weight", "query_attention.in_proj_bias",
    "query_attention.out_proj.weight", "query_attention.out_proj.bias", "query_norm.weight", "query_norm.bias",
    "output.weight", "output.bias",
)
FULL_ONLY_KEYS = {"observer_embedding.weight", "source_relative_embedding.weight",
                  "target_relative_embedding.weight", "relation_projection.weight"}
PARAMETER_MAPPING = [{"full": key, "baseline": key} for key in SHARED_KEYS]


def query_initialization(initialization_seed, fold):
    if type(initialization_seed) is not int or not 0 <= initialization_seed < 2**63:
        raise ValueError("invalid initialization seed")
    if type(fold) is not int or not 0 <= fold < 5:
        raise ValueError("invalid fold")
    seed = int.from_bytes(digest_fields("baseline_shared_query_v1", initialization_seed, fold), "big") % 2**63
    generator = torch.Generator(device="cpu").manual_seed(seed)
    query = torch.empty(1, 256, dtype=torch.float32).normal_(0, .02, generator=generator)
    return seed, query


def _tensor_digest(tensor):
    # Shape/dtype are independently checked by the strict graph/container validators.
    return sha256_bytes(tensor.detach().cpu().numpy().tobytes(order="C"))


def _initial_model(full_initial_path, parent_digest, initialization_seed, fold):
    seed, query = query_initialization(initialization_seed, fold)
    parent = validate_model_state(full_initial_path)
    if parent.artifact.manifest_digest != parent_digest:
        raise ValueError("Full initial-state digest mismatch")
    provenance = parent.artifact.manifest["provenance"]
    full_seed = int.from_bytes(digest_fields("paired_initial_v1", initialization_seed, fold), "big") % 2**63
    expected = {"purpose": "initial", "protocol_digest": provenance.get("protocol_digest"),
                "fold": fold, "initialization_seed": full_seed, "model_graph": FULL_GRAPH}
    protocol = provenance.get("protocol_digest")
    if (canonical_json_bytes(provenance) != canonical_json_bytes(expected)
            or not isinstance(protocol, str) or len(protocol) != 64
            or any(c not in "0123456789abcdef" for c in protocol)):
        raise ValueError("requires corresponding untrained Full fold initial state")
    if set(parent.tensors) != set(SHARED_KEYS) | FULL_ONLY_KEYS:
        raise ValueError("Full parameter mapping is not exhaustive")
    with torch.random.fork_rng(devices=[]):
        model = ObserverAgnosticToM(ExperimentCapacity(1), None)
    target = model.state_dict()
    if set(target) != set(SHARED_KEYS) | {"shared_query"}:
        raise ValueError("baseline parameter mapping is not exhaustive")
    for key in SHARED_KEYS:
        value = parent.tensors[key]
        if value.array.shape != tuple(target[key].shape) or value.array.dtype.str != "<f4" or not value.trainable:
            raise ValueError("shared tensor shape/dtype mismatch")
        target[key] = torch.from_numpy(value.array.copy())
    target["shared_query"] = query
    model.load_state_dict(target, strict=True)
    if any(not torch.equal(model.state_dict()[key], target[key]) for key in target):
        raise ValueError("initial-state byte copy mismatch")
    binding = {"purpose": "initial", "model_graph": MODEL_GRAPH, "fold": fold,
        "initialization_seed": initialization_seed, "parent_full_initial_state_digest": parent_digest,
        "parent_full_protocol_digest": protocol,
        "parameter_mapping": PARAMETER_MAPPING,
        "parameter_mapping_digest": sha256_bytes(canonical_json_bytes(PARAMETER_MAPPING)),
        "shared_tensor_digests": {key: _tensor_digest(target[key]) for key in SHARED_KEYS},
        "shared_query_seed": seed, "shared_query_tensor_digest": _tensor_digest(query),
        "query_initialization": "baseline_shared_query_v1/CPU-float32-normal(0,0.02)",
        "temporal_conditions": ["implicit", "explicit_day_phase"]}
    return model, binding


def publish_initial_state(full_initial_path, parent_digest, destination, *, initialization_seed, fold):
    model, binding = _initial_model(full_initial_path, parent_digest, initialization_seed, fold)
    # Its manifest_digest binds all metadata and the complete baseline tensor payload.
    return publish_tensor_state(destination, manifest_fields={
        "artifact_type": "observer_agnostic_initial_state", "schema_version": INITIAL_VERSION,
        "provenance": binding}, tensors=model_tensor_values(model))


def load_initial_state(path, model, expected_digest, *, full_initial_path):
    state = verify_tensor_state(path, expected_artifact_type="observer_agnostic_initial_state",
                                expected_schema_version=INITIAL_VERSION)
    if state.artifact.manifest_digest != expected_digest or set(state.artifact.manifest) != {
            "artifact_type", "schema_version", "provenance", "tensor_container", "manifest_digest", "file_table"}:
        raise ValueError("baseline initial-state digest/schema mismatch")
    p = state.artifact.manifest["provenance"]
    reference, expected = _initial_model(full_initial_path, p["parent_full_initial_state_digest"],
                                         p["initialization_seed"], p["fold"])
    if p != expected:
        raise ValueError("baseline initialization provenance mismatch")
    restore_model_tensors(model, state.tensors)
    if any(not torch.equal(value.detach().cpu(), reference.state_dict()[key])
           for key, value in model.state_dict().items()):
        raise ValueError("baseline initialization tensor binding mismatch")
    return state
