from types import SimpleNamespace as NS

import pytest

from scripts.diagnose_observer_dtrs_dataset import (
    describe, game_partition, historical_rng_seed, weighted_quantile,
)


def row(game, y, r=.3):
    return NS(legal=NS(context=NS(game_id=game), features=NS(r=r)),
              supervision=NS(event_label_y=y, raw_suspicion_support=("player1",)))


def test_record_and_game_equal_are_different_targets():
    rows = [row("large", 1) for _ in range(9)] + [row("small", 0)]
    result = describe(rows, {"large": 9, "small": 1})
    assert result["record_prevalence"] == .9
    assert result["game_weighted_conditional_prevalence"] == pytest.approx(.5)
    assert result["effective_games_kish"] == pytest.approx(2)
    # A subgroup keeps FULL game sizes, not subgroup counts per game.
    sub = describe([rows[0], rows[-1]], {"large": 9, "small": 1})
    assert sub["game_weighted_conditional_prevalence"] == pytest.approx(.1)


def test_weighted_quantile_is_left_inverse_and_tie_stable():
    assert weighted_quantile([.2, .2, .8], [1, 1, 2], .5) == .2
    assert weighted_quantile([.2, .2, .8], [1, 1, 2], .75) == .8
    assert weighted_quantile([], [], .5) is None


def test_hash_preview_is_order_independent_balanced_and_parent_specific():
    games = [f"g-{i}" for i in range(300)]
    args = dict(tom_fold=2, folds=5)
    first = game_partition(games, **args)
    assert first == game_partition(reversed(games), **args)
    assert [sum(x == j for x in first.values()) for j in range(5)] == [60] * 5
    assert game_partition(games, **args, parent_outer=1) != first


def test_historical_provenance_field_is_a_numeric_rng_input():
    binding = {"architecture": "qwen3", "fold": 0,
               "protocol_seed": 8223107198166168390,
               "study": "f4ff88373da25caf21e8bc608fd049eba8cc378e886fb28a0ef0646d1a21907a",
               "version": "backbone_training_rng_v1"}
    assert historical_rng_seed(binding) == 3981651915
    assert historical_rng_seed(binding) == historical_rng_seed(dict(reversed(list(binding.items()))))
    # Hypothetical metadata substitution only; no claim about changed weights/scores.
    assert historical_rng_seed(binding) != historical_rng_seed({**binding, "study": "0" * 64})
