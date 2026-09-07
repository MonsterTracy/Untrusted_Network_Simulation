from collections import Counter

import numpy as np
import pytest
import torch


def test_schedule_complete_seven_shifts_and_partial_batch_weights():
    from werewolf.tom.protocol import training_schedule

    schedule = training_schedule("a" * 64, 2, ("a", "b", "c", "d"), 2, 3)
    assert len(schedule["batches"]) == 28
    counts = Counter((item["game_id"], item["shift"]) for batch in schedule["batches"] for item in batch)
    assert counts == Counter({(g, shift): 2 for g in "abcd" for shift in range(7)})
    assert {len(b) for b in schedule["batches"]} == {1, 3}
    assert all(len({i["game_id"] for i in b}) == len(b) for b in schedule["batches"])
    for game in "abcd":
        actual = sum(1/len(b) for b in schedule["batches"] if any(i["game_id"] == game for i in b))
        assert schedule["cumulative_game_coefficients"][game] == actual
    assert schedule == training_schedule("a" * 64, 2, ("a", "b", "c", "d"), 2, 3)


def test_game_balanced_loss_not_pooled_rows():
    from werewolf.tom.scoring import game_balanced_cross_entropy

    first = torch.tensor([[[.0, .5, .5, 0, 0, 0, 0]]])
    second = first.repeat(1, 3, 1)
    log1 = torch.log(torch.tensor([[[0., .25, .75, 0, 0, 0, 0]]]))
    log2 = torch.log(torch.tensor([[[0., .5, .5, 0, 0, 0, 0]]])).repeat(1, 3, 1)
    loss = game_balanced_cross_entropy([(log1, first, torch.ones(1, 1, dtype=torch.bool)),
                                        (log2, second, torch.ones(1, 3, dtype=torch.bool))])
    expected = (-.5*np.log(.25)-.5*np.log(.75)+np.log(2))/2
    assert loss.item() == pytest.approx(expected)
    with pytest.raises(ValueError, match="zero"):
        game_balanced_cross_entropy([(log1, first, torch.zeros(1, 1, dtype=torch.bool))])


def test_paired_game_bootstrap_retains_negative_differences():
    from werewolf.tom.scoring import game_macro_summary, paired_summary

    indices = np.array([[0, 0], [0, 1], [1, 1]], dtype="<i4")
    summary = game_macro_summary([1., 3.], indices, .8)
    assert summary["point"] == 2.
    assert summary["interval"] == pytest.approx([1.2, 2.8])
    paired = paired_summary([1., 2.], [3., 1.], indices, .8)
    assert paired["point"] == -.5
    assert paired["interval"] == pytest.approx([-1.7, .7])
