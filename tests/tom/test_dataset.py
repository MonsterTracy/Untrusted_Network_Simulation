import inspect

import pytest
import torch

from tests.development_publication.test_development_publication import _publication


def test_full_prefix_targets_and_rotation(tmp_path):
    from werewolf.tom.dataset import CanonicalToMDataset, ExperimentCapacity, rotate

    _, handles = _publication(tmp_path)
    view = handles.public.public_view
    data = CanonicalToMDataset(view, view.game_ids, ExperimentCapacity(view.max_structured_token_count))
    sample = data[0]
    assert sample.public.event_ids[sample.token_count - 1].item() == 2  # turn_start
    assert sample.public.attention_mask.sum().item() == sample.token_count
    assert torch.all(sample.q.diagonal() == 0)
    assert torch.allclose(sample.q.sum(-1)[sample.label_observed], torch.ones(int(sample.label_observed.sum())))
    eligibility = sample.observer_alive.clone()
    eligibility[0] = False
    for shift in range(7):
        rotated, mask = rotate(sample, shift, eligibility)
        restored, original_mask = rotate(rotated, (-shift) % 7, mask)
        assert torch.equal(restored.q, sample.q)
        assert torch.equal(restored.public.source_ids, sample.public.source_ids)
        assert torch.equal(original_mask, eligibility)
        assert torch.equal(mask, torch.roll(eligibility, shift, -1))
    with pytest.raises(ValueError, match="capacity"):
        CanonicalToMDataset(view, view.game_ids, ExperimentCapacity(1))
    assert not {"scope", "population", "role_filter", "belief_source", "target_semantics"} & set(inspect.signature(CanonicalToMDataset).parameters)


def test_unique_target_semantics():
    from werewolf.tom.dataset import belief_target

    q, observed = belief_target(0, (), "success")
    assert observed and q[0] == 0
    assert torch.equal(q[1:], torch.full((6,), 1 / 6))
    q, observed = belief_target(1, ("player1", "player4"), "success")
    assert observed and q.tolist() == [0.5, 0, 0, 0.5, 0, 0, 0]
    for status in ("error", None):
        q, observed = belief_target(0, (), status)
        assert not observed and not q.any()
    with pytest.raises(ValueError, match="self"):
        belief_target(0, ("player1",), "success")
