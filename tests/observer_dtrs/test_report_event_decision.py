"""固定概率重建、三域支持和整局配对的定向测试。"""

from collections import Counter
import math

import numpy as np
import pytest

from scripts import analyze_observer_dtrs_report_event as study
from scripts import fit_observer_dtrs_report_event as fit
from werewolf.observer_dtrs.scoring import score_predictions


def fixture_rows():
    return [fit.Sample(game, f"pre-{i}", "player1", 1, "discussion", 2, 2,
                       (0.,) * len(fit.CONTEXT_FEATURES) + (x, x * x, .6, .2, .05), y, False)
            for i, (game, x, y) in enumerate((("short", .2, 1), ("long", .4, 0),
                                             ("long", .6, 1), ("long", .8, 1)))]


def test_rebuild_uses_saved_transform_and_both_logits():
    rows = fixture_rows()
    parameters = {"features": list(fit.FEATURES["M1"]), "standardization": {"mean": [.2, 0., .6], "std": [.2, 1., 0.]},
                  "base": {"intercept": -.4, "coefficients": [0.] * len(fit.CONTEXT_FEATURES) + [.5, 0., 2.]},
                  "calibrator": {"intercept": .1, "coefficients": [2.]}}
    p = study.rebuild_probabilities(rows, "M1", parameters)
    expected = [1 / (1 + math.exp(-(.1 + 2 * (-.4 + .5 * (r.features[-5] - .2) / .2)))) for r in rows]
    assert list(p.values()) == pytest.approx(expected)
    archived = score_predictions(rows, p)
    reproduction = study.verify_scores(rows, "M1", p, archived)
    assert reproduction["games_matched"] == 2 and reproduction["rows_matched"] == 4
    assert all(delta == 0 for scores in reproduction["differences"].values() for delta in scores.values())
    tampered = {**archived, "record_equal": {**archived["record_equal"], "brier": archived["record_equal"]["brier"] + 1e-8}}
    with pytest.raises(ValueError, match="record_equal brier does not reproduce"):
        study.verify_scores(rows, "M1", p, tampered)
    with pytest.raises(ValueError, match="by-game identities"):
        study.verify_scores(rows, "M1", p, {**archived, "by_game": {"different": {"rows": 4}}})


def test_boundaries_and_subgroup_use_full_game_weights():
    rows = fixture_rows()
    sizes = Counter(r.game_id for r in rows)
    games = tuple(sorted(sizes))
    p = fit.probability_map(rows, [.25, .75, .125, .875])
    draws = np.asarray([[0, 0], [0, 1], [1, 1]])
    result, totals = study.evaluate_model(rows, p, games, sizes, {"sub": np.array([True, False, True, False])}, draws)
    game = result["overall"]["game_equal"]
    record = result["overall"]["record_equal"]
    assert game["three_way"]["regions"]["BND"]["rows"] == 2
    assert game["three_way"]["regions"]["BND"]["proportion"] == pytest.approx(2 / 3)
    assert game["three_way"]["regions"]["NEG"]["error_games"] == 1
    assert game["three_way"]["POS_error_rate"] == 0 and game["three_way"]["NEG_error_rate"] == 1
    assert game["three_way"]["decided_coverage"] == pytest.approx(1 / 3)
    assert game["three_way"]["realized_risk"] == pytest.approx(1 / 6)
    assert record["three_way"]["realized_risk"] == pytest.approx(.1875)
    assert game["forced_binary"]["realized_risk"] == pytest.approx(5 / 12)
    assert game["forced_binary"]["decided_coverage"] == 1
    assert game["three_way"]["predicted_risk"] == pytest.approx(game["forced_binary"]["predicted_risk"])
    assert game["three_way"]["error_intervals"]["POS"] == {"percentile_95": None, "valid_replicates": 2, "replicates": 3}

    sub = result["subgroups"]["sub"]["game_equal"]["three_way"]
    assert sub["POS_error_rate"] is None
    assert sub["regions"]["BND"]["proportion"] == pytest.approx(.75)
    assert sub["realized_risk"] == pytest.approx(.21875)
    assert result["subgroups"]["sub"]["record_equal"]["three_way"]["realized_risk"] == pytest.approx(.3125)

    comparison = study.paired_losses({m: result["overall"] for m in study.MODELS},
                                     {m: totals for m in study.MODELS}, np.asarray([sizes[g] for g in games]), draws)
    assert comparison["game_equal"]["M1_three_minus_binary"]["difference"] == pytest.approx(-.25)
    assert comparison["record_equal"]["M1_three_minus_binary"]["difference"] == pytest.approx(-.1875)
    assert comparison["game_equal"]["M2_minus_M1_three_way"]["percentile_95"] == [0., 0.]


def test_cluster_ratio_reweights_by_selected_game_denominators():
    draws = np.asarray([[0, 1], [0, 0], [2, 2]])
    failures = np.asarray([0., 1., 0.])
    exposure = np.asarray([2., 1., 0.])
    risk = study.bootstrap_ratio(failures, exposure, np.asarray([.5, .25, .5]), draws)
    assert list(risk[:2]) == pytest.approx([.2, 0.])
    assert math.isnan(risk[-1])
    record = study.bootstrap_ratio(failures, exposure, np.ones(3), draws)
    assert record[0] == pytest.approx(1 / 3)
    assert study.interval(np.full(3, np.nan))["valid_replicates"] == 0
