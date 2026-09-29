"""Synthetic checks of the preregistered game-cluster comparison rule."""

from dataclasses import replace
import math

import numpy as np
import pytest

from werewolf.phase2_mapper import MODEL_SPECS, MapperPrediction
from werewolf.phase2_mapper_selection import (
    ADJACENT_PAIRS, ALL_PAIRS, DEFAULT_REPLICATES, DEFAULT_SEED,
    MODEL_ORDER, BootstrapComparison, MapperSelectionError,
    ModelPointMetrics, PairComparison, _draw_game_multiplicities,
    _prepare_panel, _weighted_metrics, bootstrap_compare,
    passes_promotion, percentile_interval, select_from_comparison,
)


def predictions():
    result = []
    strengths = (.6, .7, .8, .9)
    for fold in range(5):
        for within_fold in range(2):
            game = f"game-{fold}-{within_fold}"
            count = 1 if within_fold == 0 else 3
            for candidate in range(count):
                label = (fold + within_fold + candidate) % 2 == 0
                for spec, strength in zip(MODEL_SPECS, strengths, strict=True):
                    probability = strength if label else 1 - strength
                    result.append(MapperPrediction(game, f"boundary-{game}",
                        "player1", "speech", f"player-{candidate + 2}", label,
                        fold, spec.name, math.log(probability / (1 - probability)),
                        probability, fold, f"prefix-{game}", "a" * 64,
                        "b" * 64, f"model-{spec.name}"))
    return tuple(result)


def comparison_with_support(*supported_pairs):
    points = {name: ModelPointMetrics(20, 10, .5, .25, .5, .25)
              for name in MODEL_ORDER}
    pairs = tuple(PairComparison(b, a,
        "adjacent" if (b, a) in ADJACENT_PAIRS else "rescue",
        -.1, -.2, -.01 if (b, a) in supported_pairs else .01,
        -.01, -.01, -.01, (b, a) in supported_pairs)
        for b, a in ALL_PAIRS)
    return BootstrapComparison(points, pairs, 10, DEFAULT_SEED, .95)


def test_default_settings_and_pair_order_are_frozen():
    assert DEFAULT_REPLICATES == 100_000
    assert DEFAULT_SEED == 20260929
    assert ADJACENT_PAIRS == tuple(zip(MODEL_ORDER[1:], MODEL_ORDER[:-1], strict=True))
    assert ALL_PAIRS[-2:] == ((MODEL_ORDER[2], MODEL_ORDER[0]),
                              (MODEL_ORDER[3], MODEL_ORDER[0]))


def test_draws_are_games_not_candidate_rows_and_keep_each_fold_size():
    panel = _prepare_panel(predictions())
    assert len(panel.game_ids) == 10 and panel.candidate_counts.tolist() == [1, 3] * 5
    rng = np.random.default_rng(7)
    for _ in range(30):
        weights = _draw_game_multiplicities(panel, rng)
        assert weights.shape == (10,)
        assert np.issubdtype(weights.dtype, np.integer)
        assert weights.sum() == 10
        assert all(weights[panel.folds == fold].sum() == 2 for fold in range(5))
        # The sampled unit is one game, regardless of its one or three rows.
        assert np.dot(weights, panel.candidate_counts) == sum(
            int(weights[g]) * int(panel.candidate_counts[g]) for g in range(10))


def test_repeated_game_multiplicity_preserves_pooled_and_macro_semantics():
    panel = _prepare_panel(predictions())
    weights = np.ones(10, dtype=int)
    weights[0], weights[1] = 2, 0
    metrics = _weighted_metrics(panel, weights)
    for model in range(4):
        pooled_ll = np.dot(weights, panel.log_loss_sums[model]) / np.dot(
            weights, panel.candidate_counts)
        pooled_brier = np.dot(weights, panel.brier_sums[model]) / np.dot(
            weights, panel.candidate_counts)
        macro_ll = np.dot(weights, panel.log_loss_sums[model] /
                          panel.candidate_counts) / weights.sum()
        macro_brier = np.dot(weights, panel.brier_sums[model] /
                             panel.candidate_counts) / weights.sum()
        np.testing.assert_allclose(metrics[model],
                                   (pooled_ll, pooled_brier, macro_ll, macro_brier))
    # Duplicating the one-row game changes pooled row count but not game count.
    assert np.dot(weights, panel.candidate_counts) == 18
    assert weights.sum() == 10
    rows = [row for row in predictions() if row.model_spec == MODEL_ORDER[0]]
    expanded = [row for row in rows for _ in range(int(weights[
        panel.game_ids.index(row.game_id)]))]
    observed_ll = np.mean([-math.log(row.raw_probability if row.theta_ac
                                     else 1 - row.raw_probability) for row in expanded])
    observed_brier = np.mean([(row.raw_probability - int(row.theta_ac)) ** 2
                              for row in expanded])
    assert metrics[0, 0] == pytest.approx(observed_ll)
    assert metrics[0, 1] == pytest.approx(observed_brier)


def test_every_model_uses_the_identical_cluster_draw(monkeypatch):
    panel = _prepare_panel(predictions())
    weights = np.ones(10, dtype=int)
    weights[0], weights[1] = 2, 0
    draws = []

    def fixed_draw(_panel, _rng):
        assert _panel.game_ids == panel.game_ids
        draws.append(1)
        return weights.copy()

    monkeypatch.setattr("werewolf.phase2_mapper_selection._draw_game_multiplicities",
                        fixed_draw)
    result = bootstrap_compare(predictions(), replicates=1)
    assert draws == [1]  # Not one independent draw per model or candidate.
    metrics = _weighted_metrics(panel, weights)
    for pair in result.pairs:
        expected = metrics[MODEL_ORDER.index(pair.complex_model)] - metrics[
            MODEL_ORDER.index(pair.simple_model)]
        assert pair.ci_lower == pytest.approx(expected[0])
        assert pair.ci_upper == pytest.approx(expected[0])
        assert pair.ci_pooled_brier == pytest.approx((expected[1], expected[1]))
        assert pair.ci_game_macro_log_loss == pytest.approx((expected[2], expected[2]))
        assert pair.ci_game_macro_brier == pytest.approx((expected[3], expected[3]))


def test_candidate_identity_mismatch_explicitly_fails():
    population = predictions()
    altered = tuple(replace(row, candidate_j="other") if
                    row.model_spec == MODEL_ORDER[3] and row.game_id == "game-0-0"
                    else row for row in population)
    with pytest.raises(MapperSelectionError, match="candidate identity mismatch"):
        bootstrap_compare(altered, replicates=2)


def test_label_fold_and_provenance_mismatch_explicitly_fail():
    population = predictions()
    for change in ({"theta_ac": not population[0].theta_ac},
                   {"mapper_fold": 1, "game_oof_fold": 1},
                   {"prefix_digest": "different"}):
        altered = (replace(population[0], **change), *population[1:])
        with pytest.raises(MapperSelectionError, match="mismatch"):
            bootstrap_compare(altered, replicates=2)


def test_delta_sign_is_complex_minus_simple():
    result = bootstrap_compare(predictions(), replicates=32, seed=13)
    for pair in result.pairs:
        assert pair.delta_pooled_log_loss == pytest.approx(
            result.points[pair.complex_model].pooled_log_loss
            - result.points[pair.simple_model].pooled_log_loss)
        assert pair.delta_pooled_log_loss < 0
        assert pair.ci_upper < 0


def test_percentile_ci_uses_linear_quantiles():
    assert percentile_interval(np.arange(5.), level=.5) == pytest.approx((1., 3.))
    assert percentile_interval(np.array([0., 10.]), level=.95) == pytest.approx((.25, 9.75))


def test_fixed_seed_reproduces_all_bootstrap_intervals():
    first = bootstrap_compare(predictions(), replicates=75, seed=20260929)
    second = bootstrap_compare(predictions(), replicates=75, seed=20260929)
    assert first == second


def test_promotion_gate_passes_only_when_all_three_conditions_hold():
    assert passes_promotion(-.001, 0., 0.)
    assert not passes_promotion(0., -.1, -.1)  # CI touches zero.
    assert not passes_promotion(.001, -.1, -.1)  # CI crosses zero.
    assert not passes_promotion(-.1, .001, -.1)  # Brier worsens.
    assert not passes_promotion(-.1, -.1, .001)  # Game-macro LL worsens.


def test_sequential_promotion_passes_all_adjacent_steps():
    result = select_from_comparison(comparison_with_support(*ADJACENT_PAIRS))
    assert result.primary_model == MODEL_ORDER[3]
    assert result.stopped_at is None
    assert result.development_supported_alternatives == ()


def test_sequential_promotion_stops_at_first_unsupported_step():
    result = select_from_comparison(comparison_with_support(ADJACENT_PAIRS[0],
                                                            ADJACENT_PAIRS[2]))
    assert result.primary_model == MODEL_ORDER[1]
    assert result.stopped_at == ADJACENT_PAIRS[1]


def test_rescue_is_alternative_and_never_changes_primary():
    result = select_from_comparison(comparison_with_support(
        (MODEL_ORDER[2], MODEL_ORDER[0]), (MODEL_ORDER[3], MODEL_ORDER[0])))
    assert result.primary_model == MODEL_ORDER[0]
    assert result.stopped_at == ADJACENT_PAIRS[0]
    assert result.development_supported_alternatives == MODEL_ORDER[2:]


def test_selection_rejects_inconsistent_gate_flag():
    comparison = comparison_with_support(*ADJACENT_PAIRS)
    altered = replace(comparison, pairs=(replace(comparison.pairs[0],
        development_supported=False), *comparison.pairs[1:]))
    with pytest.raises(MapperSelectionError, match="inconsistent"):
        select_from_comparison(altered)


def test_bootstrap_never_invokes_model_fitting(monkeypatch):
    def forbidden_fit(*_args, **_kwargs):
        raise AssertionError("bootstrap attempted mapper refit")

    monkeypatch.setattr("werewolf.phase2_mapper._fit_mapper", forbidden_fit)
    assert bootstrap_compare(predictions(), replicates=4).replicates == 4


def test_invalid_population_and_settings_fail():
    with pytest.raises(MapperSelectionError, match="empty"):
        bootstrap_compare((), replicates=2)
    with pytest.raises(MapperSelectionError, match="replicates"):
        bootstrap_compare(predictions(), replicates=0)
    with pytest.raises(MapperSelectionError, match="percentile"):
        percentile_interval(np.array([float("nan")]))
