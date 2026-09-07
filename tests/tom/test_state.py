import pytest
import torch


def test_canonical_trainable_state_exact_keys_shapes_and_bytes(tmp_path):
    from werewolf.tom.state import publish_model_state, load_model_state
    from werewolf.tom.dataset import ExperimentCapacity
    from werewolf.tom.model import ObserverConditionedToM
    from werewolf.tom.temporal import publish_temporal, TemporalCodeProvider

    publish_temporal(tmp_path / "temporal", 0, "a" * 64)
    provider = TemporalCodeProvider.implicit(tmp_path / "temporal", "a" * 64)
    first = ObserverConditionedToM(ExperimentCapacity(10), provider)
    second = ObserverConditionedToM(ExperimentCapacity(10), provider)
    artifact = publish_model_state(tmp_path / "state", first, {"purpose": "initial", "parent_digest": "a" * 64})
    load_model_state(tmp_path / "state", second, artifact.artifact.manifest_digest)
    assert all(torch.equal(v, second.state_dict()[k]) for k, v in first.state_dict().items())
    with pytest.raises(ValueError, match="digest"):
        load_model_state(tmp_path / "state", second, "b" * 64)


@pytest.mark.parametrize("violation", ["shape", "keys", "version", "field", "byte"])
def test_model_state_rejects_wrong_graph_and_artifact_identity(tmp_path, violation):
    from werewolf.tom.state import publish_model_state, load_model_state
    from werewolf.artifact_io import publish_artifact
    model = torch.nn.Linear(2, 2)
    state = publish_model_state(tmp_path / "state", model, {"purpose": "fixture"})
    digest = state.artifact.manifest_digest
    path = state.artifact.path
    if violation == "shape":
        model = torch.nn.Linear(2, 3)
    elif violation == "keys":
        model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    elif violation == "byte":
        (path / "tensors.bin").write_bytes(b"corrupt")
    else:
        fields = {k: v for k, v in state.artifact.manifest.items() if k not in {"manifest_digest", "file_table"}}
        if violation == "version":
            fields["schema_version"] = "other"
        else:
            fields["private_conditioning"] = True
        artifact = publish_artifact(tmp_path / "wrong", manifest_fields=fields, files={"tensors.bin": (path / "tensors.bin").read_bytes()})
        path, digest = artifact.path, artifact.manifest_digest
    with pytest.raises(ValueError):
        load_model_state(path, model, digest)
