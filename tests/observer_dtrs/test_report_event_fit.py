"""原始报告事件的联结、折隔离和配对评价定向测试。"""

from dataclasses import replace
import math

import numpy as np
import pytest

from scripts import fit_observer_dtrs_report_event as study
from tests.observer_dtrs.conftest import observation_for, prefix_for
from werewolf.canonical_collection import BeliefObservationStatus
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.observer_dtrs.baselines import objective
from werewolf.observer_dtrs.context import contexts_for_pre
from werewolf.observer_dtrs.labels import label_from_observation
from werewolf.observer_dtrs.scoring import score_predictions
from werewolf.tom.protocol import bootstrap_indices


WOLVES = ("player2", "player6")
ACTOR = "player6"
ACTOR_MASS = {p: mass for p, mass in zip(PLAYER_IDS, (.2, .9, .4, .8, .7, 0, .6), strict=True)}


def pre_fixture(dead=("player4", "player5")):
    prefix = prefix_for(actor=ACTOR, dead=dead)
    contexts = contexts_for_pre(prefix, WOLVES)
    reports = {c.observer_id: observation_for(c) for c in contexts}
    matrix = []
    for observer in PLAYER_IDS:
        mass = ACTOR_MASS[observer]
        matrix.append([0.0 if target == observer else mass if target == ACTOR
                       else (1 - mass) / (6 if observer == ACTOR else 5) for target in PLAYER_IDS])
    q = {"game_id": prefix.game_id, "boundary_id": prefix.boundary_id,
         "prefix_digest": prefix.prefix_digest, "step_index": prefix.step_index, "q": matrix}
    return prefix, q, contexts, reports


def samples_with_reports(prefix, q, reports):
    return list(study.pre_samples(prefix, WOLVES, {r.observation_id: r for r in reports.values()}, q))


def test_raw_report_target_and_leave_one_out_panel():
    prefix, q, contexts, empty = pre_fixture()
    reports = {**empty, "player3": observation_for(contexts[1], ("player2",)),
               "player7": observation_for(contexts[2], (ACTOR,))}
    rows = samples_with_reports(prefix, q, reports)
    assert [r.observer_id for r in rows] == ["player1", "player3", "player7"]
    assert [r.y for r in rows] == [0, 0, 1]
    assert [r.successful_empty for r in rows] == [True, False, False]
    assert all(label_from_observation(c, reports[c.observer_id]).event_label_y == 1 for c in contexts)

    for row, mean, std in zip(rows, (.5, .4, .3), (.1, .2, .1), strict=True):
        p = ACTOR_MASS[row.observer_id]
        entropy = -(p * math.log(p) + (1 - p) * math.log((1 - p) / 5)) / math.log(6)
        assert row.peer_count == 2
        assert row.features == pytest.approx((0, 0, 0, 0, 0, 0, 1, 0, 1, p, p * p, entropy, mean, std))
    assert [r.features for r in samples_with_reports(prefix, q, empty)] == [r.features for r in rows]


@pytest.mark.parametrize(("dead", "expected"), (
    (("player2", "player4", "player5", "player7"), (1, .4, 0.0)),
    (("player2", "player3", "player4", "player5", "player7"), (0, 0.0, 0.0)),
))
def test_small_panel_conventions(dead, expected):
    prefix, q, _, reports = pre_fixture(dead)
    row = samples_with_reports(prefix, q, reports)[0]
    assert (row.peer_count, *row.features[-2:]) == expected


@pytest.mark.parametrize(("field", "value"), (
    ("game_id", "other-game"), ("boundary_id", "other-pre"),
    ("prefix_digest", "b" * 64), ("step_index", -1),
))
def test_q_identity_mismatch_rejected(field, value):
    prefix, q, _, reports = pre_fixture()
    with pytest.raises(ValueError, match="Q does not bind"):
        samples_with_reports(prefix, {**q, field: value}, reports)


def test_missing_failed_and_misbound_reports_rejected():
    prefix, q, contexts, reports = pre_fixture()
    by_id = {r.observation_id: r for r in reports.values()}
    victim = reports["player1"].observation_id
    failed = observation_for(contexts[0], status=BeliefObservationStatus.ERROR)
    alternatives = (None, failed, reports["player3"])
    for observation in alternatives:
        with pytest.raises(ValueError, match="report|bind"):
            list(study.pre_samples(prefix, WOLVES, {**by_id, victim: observation}, q))


def training_fixture():
    folds, rows = {}, []
    for fold in range(5):
        games = (f"fold-{fold}-short", f"fold-{fold}-long")
        folds[fold] = games
        for game, size in zip(games, (2, 5), strict=True):
            for i in range(size):
                p = (3 + fold + i) / 12
                context = (float(i % 2), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
                rows.append(study.Sample(game, f"pre-{i}", "player1", 1 + i % 2, "discussion", 2, 4,
                                         (*context, p, p * p, .5, p / 2, 0.), (fold + i) % 2, False))
    return rows, folds


def test_game_equal_fitting_and_fold_isolation():
    rows, folds = training_fixture()
    parts = study.split_samples(rows, folds)
    assert all({r.game_id for r in parts[fold]} == set(games) for fold, games in folds.items())
    with pytest.raises(ValueError, match="more than one Fold"):
        study.split_samples(rows, {**folds, 4: (*folds[4], folds[0][0])})
    train = {fold: parts[fold] for fold in study.TRAIN_FOLDS}
    cal = parts[study.CAL_FOLD]
    fitted = study.fit_models(train, cal)
    changed = [replace(r, y=1-r.y, successful_empty=r.y == 1) if r.game_id in folds[4] else r for r in rows]
    changed_parts = study.split_samples(changed, folds)
    assert fitted == study.fit_models({f: changed_parts[f] for f in study.TRAIN_FOLDS}, changed_parts[3])
    assert np.array_equal(study.design(parts[4], "M2"), study.design(changed_parts[4], "M2"))

    base_rows = [r for fold in study.TRAIN_FOLDS for r in train[fold]]
    weights = study.game_weights(base_rows)
    assert sum(weights) == pytest.approx(1.0)
    for game in (*folds[0], *folds[1], *folds[2]):
        assert sum(w for r, w in zip(base_rows, weights, strict=True) if r.game_id == game) == pytest.approx(1 / 6)
    own = fitted["M1"]
    x = study.design(base_rows, "M1")
    prep = own["standardization"]
    assert prep["mean"][0] == pytest.approx(weights @ x[:, len(study.CONTEXT_FEATURES)])
    assert prep["std"][2] == 0.0
    scaled = study.standardize(x, prep)
    assert np.all(scaled[:, -1] == 0.0)
    base = own["base"]
    _, gradient, _ = objective(np.asarray([base["intercept"], *base["coefficients"]]),
        np.column_stack((np.ones(len(scaled)), scaled)), np.asarray([r.y for r in base_rows]), weights, base["lambda"])
    assert np.max(np.abs(gradient)) < 1e-6


def test_paired_evaluation_uses_same_games_and_correct_sign():
    rows, folds = training_fixture()
    parts = study.split_samples(rows, folds)
    model = study.fit_models({f: parts[f] for f in study.TRAIN_FOLDS}, parts[3])
    held = parts[4]
    evaluation, comparisons = study.evaluate_models(model, held)
    games = tuple(sorted(folds[4]))
    for version in ("raw", "calibrated"):
        for scores in evaluation[version].values():
            assert tuple(scores["by_game"]) == games
            assert sum(s["rows"] for s in scores["by_game"].values()) == len(held)
            assert sum(b["rows"] for b in scores["calibration"]["bins"]) == len(held)
        for name in ("M1-M0", "M2-M1"):
            assert set(comparisons[version][name]) == {"brier", "log_loss"}

    baseline = score_predictions(held, {r.sample_id: .5 for r in held})
    better = score_predictions(held, {r.sample_id: .9 if r.y else .1 for r in held})
    draws = bootstrap_indices(games, 20, study.BOOTSTRAP_SEED)
    delta = study.paired_intervals(baseline, better, games, draws)
    assert delta["brier"]["game_macro_difference"] == pytest.approx(-.24)
    assert delta["brier"]["percentile_95"] == pytest.approx([-.24, -.24])
    assert delta["log_loss"]["game_macro_difference"] < 0
    for loss in study.paired_intervals(better, better, games, draws).values():
        assert loss == {"game_macro_difference": 0.0, "percentile_95": [0., 0.], "record_equal_difference": 0.0}
