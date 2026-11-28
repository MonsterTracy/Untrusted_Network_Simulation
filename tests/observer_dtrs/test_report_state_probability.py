"""合成 PRE、原始报告与游戏折上的双层概率回归测试。"""

from dataclasses import replace
import json
import math
import sys

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from scripts import fit_observer_dtrs_report_state as study
from tests.observer_dtrs.conftest import observation_for, prefix_for
from tests.observer_dtrs.test_team_event import ATTEMPT, WOLVES, mocked_development, pre_fixture
from werewolf.canonical_collection import BeliefObservationStatus
from werewolf.observer_dtrs import report_state_probability as probability
from werewolf.observer_dtrs.context import contexts_for_pre


@pytest.mark.parametrize("dead", ((), ("player2", "player4")))
def test_raw_states_and_q_features_use_legal_absolute_seats(dead):
    pre, _, reports, q = pre_fixture(dead=dead, supports={"player5": ("player2",), "player6": WOLVES})
    rows = list(study.pre_samples(pre, WOLVES, reports, q, attempt_id=ATTEMPT))
    by_seat = {row.context.observer_id: row for row in rows}
    assert list(by_seat) == [seat for seat in pre.alive_observer_ids if seat not in WOLVES]
    assert by_seat["player3"].successful_empty and by_seat["player3"].state == 0
    assert by_seat["player5"].state == (0 if dead else 1)
    assert by_seat["player6"].state == 2
    s, m = .25, 0 if dead else .125
    entropy = -sum(value * math.log(value) for value in (.25, .125, .5, .125)) / math.log(6)
    distance = .5 * sum(abs(value - 1/6) for value in (.25, .125, .5, .125, 0, 0))
    dual = int(not dead)
    assert by_seat["player3"].features == pytest.approx((
        s, m, s * s, m * m, s * m, entropy, distance, 1, 0, 7 - len(dead), dual,
        dual * distance, dual * s,
    ))


def test_feature_roles_follow_the_speaker_in_pk():
    pre = prefix_for(phase="pk_discussion", actor="player2")
    _, _, _, q = pre_fixture()
    q.update({key: getattr(pre, key) for key in study.team_event.PRE_IDENTITY})
    matrix = study.team_event.bind_q(pre, q)
    context, x = probability.pre_features(pre, WOLVES, matrix)[0]
    assert context.observer_id == "player3" and context.acting_wolf == "player2"
    assert x[:2] == (.125, .25)
    assert x[probability.FEATURES.index("pk")] == 1


def test_misbound_feature_request_is_rejected():
    pre, _, _, q = pre_fixture()
    matrix = study.team_event.bind_q(pre, q)
    with pytest.raises(ValueError, match="identity mismatch"):
        probability.pre_features(prefix_for(game_id="another-game"), WOLVES, matrix)


@pytest.mark.parametrize(("case", "status"), (
    ("missing", "missing"), ("failed", BeliefObservationStatus.ERROR.value),
    ("wrong_observer", "invalid_or_misbound"), ("wrong_attempt", "invalid_or_misbound"),
))
def test_unsuccessful_or_unbound_report_cannot_become_e0(case, status):
    pre, contexts, reports, q = pre_fixture()
    report_id = "report-player3"
    attempt = ATTEMPT
    if case == "missing":
        reports.pop(report_id)
    elif case == "failed":
        reports[report_id] = observation_for(contexts[0], status=BeliefObservationStatus.ERROR)
    elif case == "wrong_observer":
        reports[report_id] = reports["report-player4"]
    else:
        attempt = "wrong-attempt"
    rows = list(study.pre_samples(pre, WOLVES, reports, q, attempt_id=attempt))
    assert rows[0].state is None and rows[0].report_status == status and not rows[0].successful_empty
    counts = study.population(rows)
    assert counts["unobserved_records"] == (5 if case == "wrong_attempt" else 1)
    assert counts["states"]["E0"] == counts["successful_reports"]


def test_real_loader_port_preserves_membership_and_unobserved_rows(mocked_development):
    mocked_development.games["game-1"].belief_observations = ()
    parts, lineage = study.team_event.load_development((1, 3), build_samples=study.pre_samples)
    assert list(parts) == [1, 3]
    assert mocked_development.opened == ["game-1", "game-3"]
    assert lineage["fold_manifest_digest"] == "fold-digest"
    assert all(row.attempt_id == ATTEMPT and row.state is None for row in parts[1])
    assert all(row.state == 0 and row.successful_empty for row in parts[3])


@pytest.fixture
def partitions():
    groups = {}
    for fold in range(5):
        rows = []
        for name, dead, supports in (
            ("dual", (), {"player3": ("player2",), "player5": WOLVES}),
            ("single", ("player2", "player4"), {"player5": ("player1",), "player6": ("player1",)}),
        ):
            pre, _, reports, q = pre_fixture(game_id=f"{fold}-{name}", dead=dead, supports=supports)
            if fold == 0 and name == "dual":
                reports.pop("report-player7")
            q["q"][5][0] += fold / 30
            q["q"][5][1] -= fold / 30
            rows.extend(study.pre_samples(pre, WOLVES, reports, q, attempt_id=ATTEMPT))
        groups[fold] = rows
    return groups


def test_hierarchical_uses_joint_likelihood_scale_and_per_head_standardization(partitions):
    rows, x, y, weights = study.supervised(partitions[1])
    for game in {row.game_id for row in rows}:
        assert sum(weight for row, weight in zip(rows, weights, strict=True) if row.game_id == game) == pytest.approx(.5)
    regularization = 0.01
    model = study.fit_method(probability.HIERARCHICAL, x, y, weights, regularization)
    speaker, teammate = (model["heads"][key] for key in ("speaker", "teammate_given_no_speaker"))
    conditional = (x[:, probability.DUAL_COLUMN] == 1) & (y != 2)
    assert speaker["training_weight"] == pytest.approx(1)
    assert teammate["training_weight"] == pytest.approx(weights[conditional].sum())
    assert teammate["training_weight"] < 1
    for head, features, target, weight in (
        (speaker, x, y == 2, weights), (teammate, x[conditional], y[conditional], weights[conditional]),
    ):
        assert head["training_records"] == len(target)
        np.testing.assert_allclose(head["standardization"]["mean"], np.average(features, axis=0, weights=weight))
        standardized = probability.standardize(features, head["standardization"])
        reference = LogisticRegression(C=1/regularization, solver="lbfgs", tol=1e-10).fit(
            standardized, target, sample_weight=weight,
        )
        ours = np.asarray(head["intercept"]) + standardized @ np.asarray(head["coefficients"])
        np.testing.assert_allclose(ours.ravel(), reference.decision_function(standardized), atol=2e-5)
    log_q = probability.log_probabilities(model, x)
    p_s = np.exp(log_q[:, 2])
    r = np.exp(log_q[conditional, 1]) / (1 - p_s[conditional])
    factorized_nll = -weights @ np.log(np.where(y == 2, p_s, 1 - p_s))
    factorized_nll -= weights[conditional] @ np.log(np.where(y[conditional] == 1, r, 1 - r))
    assert -weights @ log_q[np.arange(len(y)), y] == pytest.approx(factorized_nll)


@pytest.mark.parametrize("method", study.METHODS)
def test_fitted_models_predict_coherent_probabilities_and_single_wolf_zero(method, partitions):
    _, x, y, weights = study.supervised(partitions[1])
    model = study.fit_method(method, x, y, weights, .01)
    assert model["features"] == probability.METHOD_FEATURES[method]
    for dead in ((), ("player2", "player4")):
        pre, contexts, _, q = pre_fixture(dead=dead)
        panel = probability.predict_pre(pre, WOLVES, study.team_event.bind_q(pre, q), model)
        assert panel["prefix_digest"] == pre.prefix_digest and panel["step_index"] == pre.step_index
        assert panel["acting_wolf"] == pre.current_speaker
        assert panel["alive_wolf_ids"] == contexts[0].alive_wolf_ids
        assert list(panel["observers"]) == [context.observer_id for context in contexts]
        for record in panel["observers"].values():
            q0, q1, q2 = (record["q0"], record["q1"], record["q2"])
            assert min(q0, q1, q2) >= 0 and q0 + q1 + q2 == pytest.approx(1)
            assert record["ps"] == q2 <= q1 + q2 == record["pT"]
            if dead:
                assert q1 == 0 and record["ps"] == record["pT"]
            else:
                assert q1 > 0 and record["ps"] < record["pT"]


def test_multinomial_mask_removes_e1_before_normalization():
    log_q = probability.masked_log_softmax(np.asarray([[1000., math.log(3)], [math.log(2), math.log(3)]]),
                                            np.asarray([False, True]))
    np.testing.assert_allclose(np.exp(log_q), [[1/4, 0, 3/4], [1/6, 2/6, 3/6]])


def test_multinomial_unpenalized_intercepts_match_weighted_state_masses(partitions):
    _, x, y, weights = study.supervised(partitions[1])
    model = study.fit_method(probability.MULTINOMIAL, x, y, weights, 1.)
    q = probability.probabilities(probability.log_probabilities(model, x), x)
    np.testing.assert_allclose(weights @ q, weights @ np.eye(3)[y], atol=1e-6)


def test_unidentifiable_conditional_head_fails_instead_of_substituting_a_model(partitions):
    _, x, y, weights = study.supervised(partitions[1])
    with pytest.raises(ValueError, match="两个结果类别"):
        study.fit_method(probability.HIERARCHICAL, x, np.where(y == 1, 0, y), weights, .01)


def test_outer_holdout_labels_do_not_change_its_own_selection_or_predictions(partitions):
    original, original_selection = study.nested_predictions(partitions)
    changed = {**partitions, 0: [replace(row, state=0) for row in partitions[0]]}
    regenerated, changed_selection = study.nested_predictions(changed)
    assert original_selection["0"]["selection_folds"] == (1, 2, 3)
    assert original_selection["0"] == changed_selection["0"]
    held = [row for row in original if row["fold"] == 0]
    assert held == [row for row in regenerated if row["fold"] == 0]
    assert len(held) == len(partitions[0])
    assert len(original) == sum(len(partitions[i]) for i in range(4))
    assert all("state" not in row and "report_status" not in row for row in original)


def test_exact_ties_prefer_hierarchical_then_larger_lambda(monkeypatch, partitions):
    monkeypatch.setattr(study, "fit_method", lambda *args: None)
    monkeypatch.setattr(probability, "log_probabilities", lambda model, x: np.empty((len(x), 3)))
    monkeypatch.setattr(probability, "probabilities", lambda log_q, x: log_q)
    monkeypatch.setattr(study, "joint_losses", lambda log_q, q, y: (np.ones(len(y)), np.ones(len(y))))
    selected = study.select_models(partitions, (0, 1, 2, 3))
    assert selected["selected_method"] == probability.HIERARCHICAL
    assert selected["best_lambda"] == dict.fromkeys(study.METHODS, 1.)


@pytest.fixture
def m1():
    features = study.frozen.FEATURES["M1"]
    num_continuous = len(features) - len(study.frozen.CONTEXT_FEATURES)
    return {"features": features, "standardization": {"mean": [0]*num_continuous, "std": [1]*num_continuous},
            "base": {"intercept": 0, "coefficients": [0]*len(features)},
            "calibrator": {"intercept": 0, "coefficients": [1]}}


def test_fold4_changes_evaluation_only(monkeypatch, partitions, m1):
    monkeypatch.setattr(study, "BOOTSTRAP_REPLICATES", 10)
    models, results, oof = study.run_development(partitions, m1)
    shifted = {**partitions, 4: [replace(row, state=0) for row in partitions[4]]}
    other_models, other_results, other_oof = study.run_development(shifted, m1)
    assert models == other_models and oof == other_oof
    assert results["selection"] == other_results["selection"]
    assert results["fold4_evaluation"] != other_results["fold4_evaluation"]


def test_artifact_runs_nested_and_frozen_evaluation_round_trip(monkeypatch, tmp_path, partitions, m1):
    destination = tmp_path / "report-state-probability"
    monkeypatch.setattr(study, "DESTINATION", destination)
    monkeypatch.setattr(study, "BOOTSTRAP_REPLICATES", 40)
    monkeypatch.setattr(sys, "argv", ["fit_observer_dtrs_report_state"])
    lineage = {"fold_manifest_digest": "test-folds"}
    monkeypatch.setattr(study.report_event, "open_fitted_models", lambda: {
        "development_fold_manifest_digest": "test-folds", "models": {"M1": m1},
    })
    usages = []

    def load(folds, *, build_samples):
        usages.append((folds, build_samples))
        return partitions, lineage

    monkeypatch.setattr(study.team_event, "load_development", load)
    study.main()
    assert usages == [(study.team_event.DEVELOPMENT_FOLDS, study.pre_samples)]
    model = probability.load_development_model(destination)
    results = json.loads((destination / "results.json").read_text())
    records = [json.loads(line) for line in (destination / "oof_predictions.jsonl").read_text().splitlines()]
    assert model["method"] == results["selection"]["selected_method"]
    assert len(records) == results["oof_records"] == sum(len(partitions[i]) for i in range(4))
    assert all(row["method"] == results["outer_selection"][str(row["fold"])]["selected_method"] for row in records)
    assert results["fold_populations"]["0"]["unobserved_records"] == 1
    summary = results["fold4_evaluation"]
    assert summary["frozen_m1"]["overall"]["scores"] == {"speaker_brier": .25}
    assert set(summary["frozen_m1"]["overall"]["calibration"]) == {"speaker"}
    for method in study.METHODS:
        assert summary[method]["overall"]["population"]["games"] == 2
        assert summary[method]["single_wolf"]["population"]["states"]["E1"] == 0
        assert summary[method]["single_wolf"]["calibration"]["E1"]["bins"][0]["mean_probability"] == 0
        assert summary[method]["single_wolf"]["game_weight_mass"] == pytest.approx(.5)
    intervals = results["fold4_paired_bootstrap"]
    assert len(intervals) == 3
    difference = (summary[probability.MULTINOMIAL]["overall"]["scores"]["joint_brier"]
                  - summary[probability.HIERARCHICAL]["overall"]["scores"]["joint_brier"])
    assert intervals["multinomial_minus_hierarchical"]["joint_brier_difference"] == pytest.approx(difference)
    with pytest.raises(FileExistsError):
        study.main()
