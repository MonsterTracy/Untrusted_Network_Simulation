from collections import Counter
from itertools import count, islice

import numpy as np
import pytest
import torch


def test_schedule_complete_seven_shifts_and_partial_batch_weights():
    from werewolf.tom.protocol import training_schedule

    schedule = training_schedule(303, 2, ("a", "b", "c", "d"), 2, 3)
    assert len(schedule["batches"]) == 28
    counts = Counter((item["game_id"], item["shift"]) for batch in schedule["batches"] for item in batch)
    assert counts == Counter({(g, shift): 2 for g in "abcd" for shift in range(7)})
    assert {len(b) for b in schedule["batches"]} == {1, 3}
    assert all(len({i["game_id"] for i in b}) == len(b) for b in schedule["batches"])
    for game in "abcd":
        actual = sum(1/len(b) for b in schedule["batches"] if any(i["game_id"] == game for i in b))
        assert schedule["cumulative_game_coefficients"][game] == actual
    assert schedule == training_schedule(303, 2, ("a", "b", "c", "d"), 2, 3)


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


def test_bootstrap_reuses_prefix_without_changing_frozen_draws():
    from werewolf.tom.protocol import BOOTSTRAP_VERSION, bootstrap_indices, digest_fields

    games = ["game-a", "game-b", "\u6e2c\u8a66"]
    replicates, seed = 17, 2196221715671680944
    values = (int.from_bytes(digest_fields(BOOTSTRAP_VERSION, seed, games, counter), "big") for counter in count())
    limit = 2**256 - 2**256 % len(games)
    draws = list(islice((v % len(games) for v in values if v < limit), replicates * len(games)))
    original = np.array(draws, dtype="<i4").reshape(replicates, len(games))
    assert bootstrap_indices(games, replicates, seed).tobytes() == original.tobytes()


@pytest.fixture
def development_ids():
    return tuple(f"synthetic-development-{index:04d}" for index in range(4800))


def test_development_holdout_is_fixed_complete_and_order_independent(development_ids):
    from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
    from werewolf.tom.protocol import development_holdout_split

    original = list(development_ids)
    split = development_holdout_split(original, development_game_ids=development_ids)
    repeated = development_holdout_split(
        reversed(development_ids), development_game_ids=reversed(development_ids)
    )
    assert split == repeated
    assert original == list(development_ids)
    assert split["rule_version"] == "classic7_development_train4500_validation300_v1"
    assert split["seed"] == 20261115
    training, validation = split["training_game_ids"], split["validation_game_ids"]
    assert len(training) == 4500 and len(validation) == 300
    assert training == sorted(set(training)) and validation == sorted(set(validation))
    assert not set(training) & set(validation)
    assert set(training) | set(validation) == set(development_ids)
    for name, ids in (("development", sorted(development_ids)),
                      ("training", training), ("validation", validation)):
        assert split[f"{name}_game_ids_digest"] == sha256_bytes(canonical_json_bytes(ids))


def test_development_holdout_uses_only_identity_inputs(development_ids, monkeypatch):
    import inspect
    from pathlib import Path
    from werewolf.tom.protocol import development_holdout_split

    assert set(inspect.signature(development_holdout_split).parameters) == {
        "game_ids", "development_game_ids"
    }
    monkeypatch.setattr(Path, "open", lambda *a, **kw: pytest.fail("pure split opened a file"))
    split = development_holdout_split(development_ids, development_game_ids=development_ids)
    assert len(split["training_game_ids"]) + len(split["validation_game_ids"]) == 4800


def test_development_holdout_frozen_hash_order(development_ids):
    from werewolf.tom.protocol import development_holdout_split

    split = development_holdout_split(development_ids, development_game_ids=development_ids)
    # 固定的合成 ID 向量检测哈希域、字段编码或种子被静默改变。
    assert split["training_game_ids_digest"] == (
        "4301721b02615fc9efe0a5a7236277aa7aa676787ccbca099b5ac12f77e88e5d"
    )
    assert split["validation_game_ids_digest"] == (
        "8ca5fe3832b3af95f1887b2ac19f5954622d18924c808250d21870ed1fb49fce"
    )


@pytest.mark.parametrize("location", ["requested", "trusted"])
def test_development_holdout_rejects_duplicate_identity(development_ids, location):
    from werewolf.tom.protocol import development_holdout_split

    duplicate = (*development_ids[:-1], development_ids[0])
    requested = duplicate if location == "requested" else development_ids
    trusted = duplicate if location == "trusted" else development_ids
    with pytest.raises(ValueError, match="duplicate"):
        development_holdout_split(requested, development_game_ids=trusted)


def test_development_holdout_rejects_foreign_and_incomplete_games(development_ids):
    from werewolf.tom.protocol import development_holdout_split

    foreign = (*development_ids[:-1], "independent-test-game")
    with pytest.raises(ValueError, match="outside verified Development"):
        development_holdout_split(foreign, development_game_ids=development_ids)
    with pytest.raises(ValueError, match="every verified Development"):
        development_holdout_split(development_ids[:-1], development_game_ids=development_ids)


@pytest.mark.parametrize("size", [4799, 4801])
def test_development_holdout_requires_exact_development_size(size):
    from werewolf.tom.protocol import development_holdout_split

    ids = tuple(f"game-{index}" for index in range(size))
    with pytest.raises(ValueError, match="exactly 4800"):
        development_holdout_split(ids, development_game_ids=ids)


@pytest.mark.parametrize("invalid", [None, True, 1, "", "  ", {"role": "Werewolf"}])
def test_development_holdout_rejects_nonidentity_values(development_ids, invalid):
    from werewolf.tom.protocol import development_holdout_split

    bad = (*development_ids[:-1], invalid)
    with pytest.raises(ValueError, match="nonempty strings"):
        development_holdout_split(bad, development_game_ids=development_ids)
