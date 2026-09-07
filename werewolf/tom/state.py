"""Strict model state binding on the shared raw tensor-container codec."""

import torch

from werewolf.artifact_io import TensorValue, publish_tensor_state, verify_tensor_state

MODEL_STATE_VERSION = "classic7_model_state_v1"


def model_tensor_values(model):
    parameters = dict(model.named_parameters())
    state = model.state_dict()
    if set(state) != set(parameters):
        raise ValueError("model state must contain exactly the trainable graph")
    return {k: TensorValue(v.detach().cpu().numpy(), True) for k, v in state.items()}


def restore_model_tensors(model, tensors):
    expected = model.state_dict()
    if set(tensors) != set(expected):
        raise ValueError("model state key mismatch")
    state = {}
    for name, tensor in tensors.items():
        if not tensor.trainable or tensor.array.dtype.str != "<f4" or tensor.array.shape != tuple(expected[name].shape):
            raise ValueError("model tensor shape/dtype/trainable mismatch")
        value = torch.from_numpy(tensor.array.copy())
        if not torch.isfinite(value).all():
            raise ValueError("model state contains non-finite values")
        state[name] = value
    model.load_state_dict(state, strict=True)
    for name, value in model.state_dict().items():
        if not torch.equal(value.detach().cpu(), state[name]):
            raise ValueError("model state byte restore mismatch")


def publish_model_state(path, model, provenance):
    return publish_tensor_state(path, manifest_fields={
        "artifact_type": "canonical_model_state", "schema_version": MODEL_STATE_VERSION,
        "provenance": provenance}, tensors=model_tensor_values(model))


def load_model_state(path, model, expected_digest):
    state = verify_tensor_state(path, expected_artifact_type="canonical_model_state", expected_schema_version=MODEL_STATE_VERSION)
    if state.artifact.manifest_digest != expected_digest or set(state.artifact.manifest) != {
        "artifact_type", "schema_version", "provenance", "tensor_container", "manifest_digest", "file_table"}:
        raise ValueError("model state digest/schema mismatch")
    restore_model_tensors(model, state.tensors)
    return state


def _fixed_validation_model():
    from werewolf.tom.dataset import ExperimentCapacity
    from werewolf.tom.model import ObserverConditionedToM

    # Capacity and temporal inputs do not affect trainable keys or shapes.
    # No temporal table is generated or loaded, and this object cannot predict.
    with torch.random.fork_rng(devices=[]):
        return ObserverConditionedToM(ExperimentCapacity(1), temporal=None)


def validate_model_tensors(tensors):
    """Validate decoded tensors against the one fixed trainable graph."""
    model = _fixed_validation_model()
    restore_model_tensors(model, tensors)


def validate_model_state(path):
    """Validate a canonical state against the one graph, without running forward."""
    from werewolf.artifact_io import open_artifact_envelope

    envelope = open_artifact_envelope(path, expected_artifact_type="canonical_model_state",
        expected_schema_version=MODEL_STATE_VERSION)
    model = _fixed_validation_model()
    return load_model_state(path, model, envelope.manifest_digest)
