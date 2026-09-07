import inspect

import pytest
import torch

from tests.development_publication.test_development_publication import _publication


def test_public_only_qwen_parameter_parity_and_non_self_output(tmp_path):
    from werewolf.tom.dataset import CanonicalToMDataset, ExperimentCapacity, PublicTensors
    from werewolf.tom.model import ObserverConditionedToM
    from werewolf.tom.temporal import publish_temporal, TemporalCodeProvider

    _, handles = _publication(tmp_path)
    view = handles.public.public_view
    capacity = ExperimentCapacity(view.max_structured_token_count)
    sample = CanonicalToMDataset(view, view.game_ids[:1], capacity)[0]
    publish_temporal(tmp_path / "temporal", view.max_observed_day, "a" * 64)
    implicit = ObserverConditionedToM(capacity, TemporalCodeProvider.implicit(tmp_path / "temporal", "a" * 64))
    explicit = ObserverConditionedToM(capacity, TemporalCodeProvider.explicit(tmp_path / "temporal", "a" * 64))
    explicit.load_state_dict(implicit.state_dict(), strict=True)
    assert [(k, p.shape) for k, p in implicit.named_parameters()] == [(k, p.shape) for k, p in explicit.named_parameters()]
    assert not any("day" in k or "phase" in k or "speaker" in k for k, _ in implicit.named_parameters())
    for model in (implicit, explicit):
        model.eval()
        logp = model(**PublicTensors.stack([sample.public]).kwargs())
        assert logp.shape == (1, 7, 7)
        assert torch.all(logp.exp().diagonal(dim1=-2, dim2=-1) == 0)
        assert torch.allclose(logp.exp().sum(-1), torch.ones(1, 7))
        assert not torch.equal(logp[:, 0], logp[:, 1])
    signature = inspect.signature(implicit.forward)
    assert set(signature.parameters) == set(sample.public.kwargs())
    with pytest.raises(TypeError):
        implicit(**PublicTensors.stack([sample.public]).kwargs(), eligibility=torch.ones(7))
