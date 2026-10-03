"""Frozen terminal-estimator mathematics, using synthetic feature data only.

The sole recorded-data test checks unadjusted arm means from the existing
119-row loss fixture and its already documented recovered failure. It never
creates formal model features or fits a model to recorded gameplay data.
"""

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from werewolf import phase2_terminal_estimator as estimator
from terminal_estimator_test_support import synthetic_rows


@pytest.fixture
def rows():
    return synthetic_rows()


@pytest.fixture
def models(rows):
    return estimator.fit_models(rows)


def test_validate_population_is_sorted_copied_and_keeps_failed_and_sparse_rows(rows):
    original = deepcopy(rows)
    validated = estimator.validate_rows(list(reversed(rows)))
    assert len(validated) == 120
    assert [r["assignment_id"] for r in validated] == sorted(r["assignment_id"] for r in rows)
    assert sum(r["execution_success"] is False for r in validated) == 1
    assert {tuple(r["S_pre"]) for r in validated} == {(1, 4), (2, 3), (2, 4), (2, 5)}
    assert rows == original
    validated[0]["L_ref"] = .9
    assert rows == original


@pytest.mark.parametrize("fault", (
    "119_rows", "121_rows", "duplicate_assignment", "duplicate_game", "unknown_action",
    "wrong_arm_count", "missing_assignment", "missing_game", "non_ascii_assignment",
))
def test_population_faults_fail_closed(rows, fault):
    if fault == "119_rows":
        rows.pop()
    elif fault == "121_rows":
        rows.append(deepcopy(rows[0]))
    elif fault == "duplicate_assignment":
        rows[0]["assignment_id"] = rows[1]["assignment_id"]
    elif fault == "duplicate_game":
        rows[0]["game_id"] = rows[1]["game_id"]
    elif fault == "unknown_action":
        rows[0]["assigned_action"] = "PROBE"
    elif fault == "wrong_arm_count":
        rows[0]["assigned_action"] = "REDIRECT"
    elif fault == "missing_assignment":
        rows[0].pop("assignment_id")
    elif fault == "missing_game":
        rows[0].pop("game_id")
    else:
        rows[0]["assignment_id"] = "non-ascii-中文"
    with pytest.raises(ValueError):
        estimator.validate_rows(rows)


@pytest.mark.parametrize("field", ("L_ref", "p_tilde_j"))
@pytest.mark.parametrize("value", (None, float("nan"), float("inf"), -float("inf"), -.001, 1.001, True))
def test_missing_nonfinite_and_illegal_numeric_inputs_fail(rows, field, value):
    rows[0][field] = value
    with pytest.raises(ValueError):
        estimator.validate_rows(rows)


@pytest.mark.parametrize("field", ("L_ref", "p_tilde_j", "S_pre"))
def test_missing_required_fields_fail(rows, field):
    rows[0].pop(field)
    with pytest.raises(ValueError):
        estimator.validate_rows(rows)


@pytest.mark.parametrize("field", ("L_ref", "p_tilde_j"))
def test_integer_too_large_for_float64_is_rejected_as_numeric_input(rows, field):
    rows[0][field] = 10 ** 1000
    with pytest.raises(ValueError, match="finite number"):
        estimator.validate_rows(rows)


@pytest.mark.parametrize("state", (None, [2], [2, 5, 0], [3, 4], [True, 4], [2.5, 5], "2,5"))
def test_illegal_states_fail(rows, state):
    rows[0]["S_pre"] = state
    with pytest.raises(ValueError):
        estimator.validate_rows(rows)


def test_primary_uses_independent_raw_means_and_neyman_ddof_one(rows):
    result = estimator.primary_itt(rows)
    losses = {a: np.asarray([r["L_ref"] for r in rows if r["assigned_action"] == a])
              for a in ("PUSH", "REDIRECT")}
    means = {a: float(values.mean()) for a, values in losses.items()}
    variance = float(losses["PUSH"].var(ddof=1) / 58 + losses["REDIRECT"].var(ddof=1) / 62)
    difference = means["PUSH"] - means["REDIRECT"]
    assert result["arm_means"] == pytest.approx(means, abs=1e-14)
    assert result["difference_in_means"] == pytest.approx(difference, abs=1e-14)
    assert result["neyman_variance"] == pytest.approx(variance, abs=1e-14)
    assert result["standard_error"] == pytest.approx(np.sqrt(variance), abs=1e-14)
    assert result["interval"] == pytest.approx([
        difference - 1.959963984540054 * np.sqrt(variance),
        difference + 1.959963984540054 * np.sqrt(variance),
    ], abs=1e-14)


def test_primary_does_not_filter_success_or_state_or_score(rows):
    base = estimator.primary_itt(rows)
    for row in rows:
        row["execution_success"] = False
        row["S_pre"] = None
        row["p_tilde_j"] = None
    assert estimator.primary_itt(rows) == base


def _hand_computed_arm_rows(rows):
    """Embed two hand-checkable two-point panels in the frozen 58/62 rows.

    PUSH repeats (1/4, 3/4) 29 times; REDIRECT repeats (0, 1/2) 31
    times. Both panels have deviations +/-1/4 from their respective means.
    These are synthetic outcomes, with the original synthetic identities/features.
    """
    indices = {"PUSH": 0, "REDIRECT": 0}
    for row in rows:
        action = row["assigned_action"]
        low = Fraction(1, 4) if action == "PUSH" else Fraction(0)
        row["L_ref"] = float(low + Fraction(indices[action] % 2, 2))
        indices[action] += 1
    return rows


def test_neyman_matches_hand_computed_two_point_panels(rows):
    # Sample variance is n/(16*(n-1)), so its contribution to the mean
    # variance is 1/(16*(n-1)); use exact fractions rather than np.var.
    result = estimator.primary_itt(_hand_computed_arm_rows(rows))
    expected_variance = Fraction(1, 16 * 57) + Fraction(1, 16 * 61)
    expected_se = math.sqrt(float(expected_variance))
    assert result["arm_means"] == {"PUSH": .5, "REDIRECT": .25}
    assert result["difference_in_means"] == .25
    assert result["arm_sample_variances"] == pytest.approx({
        "PUSH": float(Fraction(58, 16 * 57)),
        "REDIRECT": float(Fraction(62, 16 * 61)),
    }, abs=1e-15)
    assert result["neyman_variance"] == pytest.approx(float(expected_variance), abs=1e-15)
    assert result["standard_error"] == pytest.approx(expected_se, abs=1e-15)
    assert result["interval"] == pytest.approx([
        .25 - 1.959963984540054 * expected_se,
        .25 + 1.959963984540054 * expected_se,
    ], abs=1e-15)


def test_m0_ols_hc1_and_coefficient_ci_match_hand_computed_two_by_two_matrix(rows):
    # X'X=[[120,58],[58,58]], X'y=[44.5,29], determinant=58*62.
    # Solving those two equations by hand gives beta=(1/4,1/4).
    # Residual SSEs are 58/16 and 62/16. In the arm-mean basis, HC1
    # variances are (120/118)*SSE_a/n_a^2; transforming to (alpha,tau)
    # gives Cov(alpha,tau)=-Var(mean_REDIRECT). No numerical solver is
    # used to construct the expected coefficient or sandwich matrix.
    model = estimator.fit_models(_hand_computed_arm_rows(rows))["M0"]
    correction = Fraction(120, 118)
    v_redirect = correction / (16 * 62)
    v_push = correction / (16 * 58)
    expected_covariance = [[float(v_redirect), float(-v_redirect)],
                           [float(-v_redirect), float(v_push + v_redirect)]]
    expected_se = [math.sqrt(float(v_redirect)), math.sqrt(float(v_push + v_redirect))]
    assert model["coefficients"] == pytest.approx([.25, .25], abs=1e-14)
    np.testing.assert_allclose(model["covariance"], expected_covariance, atol=1e-15, rtol=1e-12)
    assert model["standard_errors"] == pytest.approx(expected_se, abs=1e-14)
    for interval, se in zip(model["coefficient_intervals"], expected_se):
        assert interval == pytest.approx([
            .25 - 1.959963984540054 * se, .25 + 1.959963984540054 * se,
        ], abs=1e-14)
    assert model["diagnostics"]["df_residual"] == 118


def test_existing_recorded_losses_reproduce_frozen_raw_means_without_fitting():
    fixture = Path(__file__).parent / "fixtures/phase2-terminal-itt-recorded-arm-losses.json"
    observed = json.loads(fixture.read_bytes())
    assert observed["source_formal_manifest_digest"] == "ec1cc8730aefe8fce57ed83a3c84ca9231ae645ea6c9047cd588f8022d2223c2"
    assert len(observed["rows"]) == 119
    recovered_game = "paper-phase2-online-terminal-pilot-v1-game-000075-seed-7714212207331013538"
    recorded = [dict(r, assignment_id=r["game_id"]) for r in observed["rows"]]
    # This outcome is the previously sealed canonical recovery, not a new inference.
    recorded.append({"assignment_id": recovered_game, "game_id": recovered_game,
                     "assigned_action": "PUSH", "L_ref": .235})
    result = estimator.primary_itt(recorded)
    assert result["arm_means"] == pytest.approx({
        "PUSH": .3118433375828878, "REDIRECT": .23960506311607913,
    }, abs=1e-12)
    assert result["difference_in_means"] == pytest.approx(.07223827446680867, abs=1e-12)
    assert all("S_pre" not in r and "p_tilde_j" not in r for r in recorded)


@pytest.mark.parametrize("model_name,k", (("M0", 2), ("M1", 5), ("M2", 7)))
def test_fixed_design_columns_and_parameter_counts(rows, model_name, k):
    matrix = estimator.design_matrix(rows, model_name)
    assert matrix.shape == (120, k)
    expected = ["intercept", "I_A", "state_14", "state_23", "state_24", "p_tilde", "I_A:p_tilde"][:k]
    assert list(estimator.MODEL_COLUMNS[model_name]) == expected
    for index, row in enumerate(sorted(rows, key=lambda r: r["assignment_id"])):
        state = tuple(row["S_pre"])
        push = int(row["assigned_action"] == "PUSH")
        expected_values = [1, push, int(state == (1, 4)), int(state == (2, 3)),
                           int(state == (2, 4)), row["p_tilde_j"], push * row["p_tilde_j"]][:k]
        np.testing.assert_array_equal(matrix[index], expected_values)


def test_design_order_deterministic_and_post_treatment_fields_excluded(rows):
    before = deepcopy(rows)
    matrix = estimator.design_matrix(rows, "M2")
    for row in rows:
        row["execution_success"] = not row["execution_success"]
        row["phase"] = "post-treatment-future-phase"
        row["S_plus"] = [0, 99]
        row["Y"] = "injected_future_information"
        row["theta_audit_label"] = True
    np.testing.assert_array_equal(matrix, estimator.design_matrix(list(reversed(rows)), "M2"))
    assert all("phase" not in c and "execution" not in c for c in estimator.MODEL_COLUMNS["M2"])
    assert before != rows


@pytest.mark.parametrize("action,treatment", (("PUSH", 1), ("REDIRECT", 0)))
@pytest.mark.parametrize("state,dummies", (
    ((1, 4), [1, 0, 0]), ((2, 3), [0, 1, 0]),
    ((2, 4), [0, 0, 1]), ((2, 5), [0, 0, 0]),
))
def test_explicit_state_reference_treatment_and_interaction_coding(rows, action, treatment, state, dummies):
    row = next(r for r in rows if r["assigned_action"] == action)
    row["S_pre"] = list(state)
    row["p_tilde_j"] = .375
    ordered = sorted(rows, key=lambda r: r["assignment_id"])
    index = next(i for i, r in enumerate(ordered) if r["assignment_id"] == row["assignment_id"])
    expected = [1, treatment, *dummies, .375, .375 if treatment else 0]
    np.testing.assert_array_equal(estimator.design_matrix(rows, "M2")[index], expected)


@pytest.mark.parametrize("model_name,k", (("M0", 2), ("M1", 5), ("M2", 7)))
def test_ols_and_hc1_match_independent_formula_all_rows(rows, models, model_name, k):
    matrix = estimator.design_matrix(rows, model_name)
    outcome = np.asarray([r["L_ref"] for r in sorted(rows, key=lambda r: r["assignment_id"])])
    beta = np.linalg.lstsq(matrix, outcome, rcond=None)[0]
    residual = outcome - matrix @ beta
    bread = np.linalg.inv(matrix.T @ matrix)
    covariance = 120 / (120 - k) * bread @ (matrix.T @ (residual[:, None] ** 2 * matrix)) @ bread
    model = models[model_name]
    assert model["covariance_convention"] == "HC1"
    assert model["diagnostics"]["n"] == 120
    assert model["diagnostics"]["k"] == k
    assert model["diagnostics"]["rank"] == k
    assert model["diagnostics"]["df_residual"] == 120 - k
    assert model["diagnostics"]["passed"] is True
    np.testing.assert_allclose(model["coefficients"], beta, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(model["covariance"], covariance, atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(model["standard_errors"], np.sqrt(np.diag(covariance)), atol=1e-12, rtol=1e-10)


def test_fit_deterministic_without_mutation_and_without_model_selection(rows):
    original = deepcopy(rows)
    first = estimator.fit_models(rows)
    second = estimator.fit_models(list(reversed(rows)))
    assert first == second
    assert rows == original
    assert set(first) == {"M0", "M1", "M2"}


def test_rank_failure_is_reported_without_ridge_or_column_dropping(rows):
    for row in rows:
        row["p_tilde_j"] = .2
    models = estimator.fit_models(rows)
    assert models["M2"]["diagnostics"]["rank"] < 7
    assert models["M2"]["diagnostics"]["passed"] is False
    assert len(models["M2"]["columns"]) == 7
    result = estimator.predict_risk(models["M0"], "PUSH", (2, 5), .2,
                                    lineage_verified=True, m2_diagnostics_passed=False)
    assert result["supported"] is False and result["risk"] is None


def test_supported_mean_range_violation_fails_closed_without_clipping(rows):
    for index, row in enumerate(rows):
        row["p_tilde_j"] = .2 + .001 * (index % 11)
        row["L_ref"] = .4 + 5 * (row["p_tilde_j"] - .2) + .02 * (row["assigned_action"] == "PUSH")
    models = estimator.fit_models(rows)
    assert models["M2"]["diagnostics"]["rank"] == 7
    assert models["M2"]["diagnostics"]["passed"] is False
    raw = estimator.audit_prediction(models["M2"], "PUSH", (2, 4), .65)
    assert raw["risk"] > 1


@pytest.mark.parametrize("state", ((2, 4), (2, 5)))
@pytest.mark.parametrize("action", ("PUSH", "REDIRECT"))
def test_frozen_support_endpoints_inclusive_and_next_float_outside_rejected(state, action):
    low, high = estimator.SUPPORTED_INTERVALS[state]
    for score in (low, high, (low + high) / 2):
        result = estimator.support_gate(action, state, score, lineage_verified=True, diagnostics_passed=True)
        assert result["supported"] is True and result["status"] == "SUPPORTED"
    for score in (np.nextafter(low, -np.inf), np.nextafter(high, np.inf)):
        result = estimator.support_gate(action, state, score, lineage_verified=True, diagnostics_passed=True)
        assert result["supported"] is False and result["status"] == "UNSUPPORTED_EXTRAPOLATION"


@pytest.mark.parametrize("state,status", (((2, 3), "AUDIT_ONLY"), ((1, 4), "UNSUPPORTED"),
                                          ((9, 9), "UNSUPPORTED")))
def test_sparse_missing_arm_and_unknown_state_never_operational(models, state, status):
    gate = estimator.support_gate("PUSH", state, .2, lineage_verified=True, diagnostics_passed=True)
    assert gate["supported"] is False and gate["status"] == status
    for model in models.values():
        result = estimator.predict_risk(model, "PUSH", state, .2,
                                        lineage_verified=True, m2_diagnostics_passed=True)
        assert result["risk"] is None
        assert result["variance"] is None
        assert result["standard_error"] is None
        assert result["interval"] is None


@pytest.mark.parametrize("score", (None, float("nan"), float("inf"), -.1, 1.1, True))
def test_invalid_score_never_silently_operational(models, score):
    gate = estimator.support_gate("PUSH", (2, 5), score, lineage_verified=True, diagnostics_passed=True)
    assert gate["supported"] is False
    result = estimator.predict_risk(models["M2"], "PUSH", (2, 5), score,
                                    lineage_verified=True, m2_diagnostics_passed=True)
    assert result["risk"] is None


def test_overflowing_integer_score_fails_closed_in_gate_and_prediction(models):
    score = 10 ** 1000
    gate = estimator.support_gate("PUSH", (2, 5), score, lineage_verified=True, diagnostics_passed=True)
    assert gate == {"status": "INVALID_INPUT", "supported": False}
    audit = estimator.audit_prediction(models["M2"], "PUSH", (2, 5), score)
    assert audit["risk"] is None and audit["supported"] is False
    result = estimator.predict_risk(models["M2"], "PUSH", (2, 5), score,
                                    lineage_verified=True, m2_diagnostics_passed=True)
    assert result["status"] == "INVALID_INPUT" and result["risk"] is None


@pytest.mark.parametrize("field", ("coefficients", "covariance"))
def test_overflowing_integer_model_entries_fail_closed_in_audit_and_diagnostics(models, field):
    model = deepcopy(models["M2"])
    if field == "coefficients":
        model[field][0] = 10 ** 1000
    else:
        model[field][0][0] = 10 ** 1000
    assert estimator.model_diagnostics(model)["passed"] is False
    audit = estimator.audit_prediction(model, "PUSH", (2, 5), .2)
    assert audit["risk"] is None and audit["supported"] is False


@pytest.mark.parametrize("action", (None, "PROBE", "push", True))
def test_invalid_action_never_operational(action):
    assert estimator.support_gate(action, (2, 5), .2, lineage_verified=True,
                                  diagnostics_passed=True)["supported"] is False


@pytest.mark.parametrize("lineage,diagnostics", ((False, True), (True, False), (False, False)))
def test_lineage_or_m2_failure_blocks_all_models(models, lineage, diagnostics):
    for model in models.values():
        result = estimator.predict_risk(model, "PUSH", (2, 5), .2,
                                        lineage_verified=lineage, m2_diagnostics_passed=diagnostics)
        assert result["supported"] is False and result["risk"] is None


@pytest.mark.parametrize("model_name", ("M0", "M1", "M2"))
@pytest.mark.parametrize("action", ("PUSH", "REDIRECT"))
def test_predictions_and_mean_uncertainty_roundtrip(models, model_name, action):
    model = models[model_name]
    reloaded = json.loads(json.dumps(model, allow_nan=False))
    score = .2
    push = int(action == "PUSH")
    vector = np.asarray([1, push, 0, 0, 1, score, push * score][:len(model["columns"])])
    mean = float(vector @ model["coefficients"])
    variance = float(vector @ np.asarray(model["covariance"]) @ vector)
    result = estimator.predict_risk(model, action, (2, 4), score,
                                    lineage_verified=True, m2_diagnostics_passed=True)
    second = estimator.predict_risk(reloaded, action, (2, 4), score,
                                    lineage_verified=True, m2_diagnostics_passed=True)
    assert result["supported"] is True
    assert result["risk"] == pytest.approx(mean, abs=1e-12)
    assert result["variance"] == pytest.approx(variance, abs=1e-12)
    assert result["standard_error"] == pytest.approx(np.sqrt(variance), abs=1e-12)
    assert result["interval"] == pytest.approx([
        mean - estimator.NORMAL_CRITICAL * np.sqrt(variance),
        mean + estimator.NORMAL_CRITICAL * np.sqrt(variance),
    ], abs=1e-12)
    assert result == second


def test_m0_arm_predictions_match_primary_raw_means(rows, models):
    primary = estimator.primary_itt(rows)
    for action in ("PUSH", "REDIRECT"):
        prediction = estimator.audit_prediction(models["M0"], action, (2, 5), .2)
        assert prediction["risk"] == pytest.approx(primary["arm_means"][action], abs=1e-12)


@pytest.mark.parametrize("action,mean,variance", (
    ("PUSH", Fraction(69, 160), Fraction(1, 3200)),
    ("REDIRECT", Fraction(21, 80), Fraction(33, 160000)),
))
def test_m2_prediction_matches_hand_computed_affine_mean_and_quadratic_variance(models, action, mean, variance):
    model = deepcopy(models["M2"])
    model["coefficients"] = [.1, .2, -.05, .025, .1, .25, -.125]
    model["covariance"] = (np.eye(7) / 10000).tolist()
    # At state (2,4), p=1/4, PUSH x=(1,1,0,0,1,1/4,1/4)
    # and REDIRECT x=(1,0,0,0,1,1/4,0). Covariance=I/10000.
    result = estimator.audit_prediction(model, action, (2, 4), .25)
    expected_se = math.sqrt(float(variance))
    assert result["audit_only"] is True and result["supported"] is False
    assert result["risk"] == pytest.approx(float(mean), abs=1e-15)
    assert result["variance"] == pytest.approx(float(variance), abs=1e-15)
    assert result["standard_error"] == pytest.approx(expected_se, abs=1e-15)
    assert result["interval"] == pytest.approx([
        float(mean) - 1.959963984540054 * expected_se,
        float(mean) + 1.959963984540054 * expected_se,
    ], abs=1e-15)


def test_audit_prediction_is_explicit_and_cannot_override_support(models):
    audit = estimator.audit_prediction(models["M2"], "REDIRECT", (1, 4), .2)
    assert np.isfinite(audit["risk"])
    operational = estimator.predict_risk(models["M2"], "REDIRECT", (1, 4), .2,
                                         lineage_verified=True, m2_diagnostics_passed=True)
    assert operational["risk"] is None


def test_singleton_leverage_and_singular_hc1_do_not_delete_rows(rows):
    for row in rows:
        row["L_ref"] = .3
    models = estimator.fit_models(rows)
    for name in ("M1", "M2"):
        diagnostics = models[name]["diagnostics"]
        assert diagnostics["n"] == 120
        assert diagnostics["leverage"]["max"] == pytest.approx(1, abs=1e-12)
        assert diagnostics["leverage"]["near_one_count"] == 1
        assert diagnostics["passed"] is True
        covariance = np.asarray(models[name]["covariance"])
        assert np.isfinite(covariance).all()
        assert np.linalg.matrix_rank(covariance, tol=1e-12) == 0
        assert estimator.predict_risk(models[name], "REDIRECT", (1, 4), .2,
                                       lineage_verified=True, m2_diagnostics_passed=True)["risk"] is None


def test_variance_diagnostic_checks_interior_stationary_point(models):
    model = deepcopy(models["M2"])
    direction = np.asarray([-.2, 0, 0, 0, 0, 1, 0])
    model["covariance"] = np.outer(direction, direction).tolist()
    diagnostic = estimator.model_diagnostics(model)
    assert diagnostic["passed"] is True
    for domain in diagnostic["supported_domain_checks"]:
        assert len(domain["scores"]) == 3
        assert domain["scores"][2] == pytest.approx(.2, abs=1e-12)
        assert domain["variances"][2] == pytest.approx(0, abs=1e-12)


def test_only_tiny_negative_prediction_variance_is_labelled_roundoff(models):
    model = deepcopy(models["M0"])
    model["covariance"] = [[-5e-12, 0], [0, -5e-12]]
    result = estimator.predict_risk(model, "PUSH", (2, 5), .2,
                                    lineage_verified=True, m2_diagnostics_passed=True)
    assert result["supported"] is True
    assert result["variance"] == pytest.approx(-1e-11, abs=1e-20)
    assert result["variance_roundoff"] is True
    assert result["standard_error"] == 0
    assert result["interval"] == [result["risk"], result["risk"]]
    model["covariance"] = [[-.1, 0], [0, -.1]]
    rejected = estimator.predict_risk(model, "PUSH", (2, 5), .2,
                                      lineage_verified=True, m2_diagnostics_passed=True)
    assert rejected["risk"] is None and rejected["supported"] is False


def test_pointwise_interval_is_raw_and_not_clipped(models):
    model = deepcopy(models["M0"])
    model["covariance"] = [[1, 0], [0, 1]]
    result = estimator.predict_risk(model, "PUSH", (2, 5), .2,
                                    lineage_verified=True, m2_diagnostics_passed=True)
    assert result["supported"] is True
    assert result["interval"][0] < 0 and result["interval"][1] > 1


@pytest.mark.parametrize("fault", ("nan_coefficient", "nan_covariance", "wrong_columns", "HC3", "asymmetric",
                                  "null_diagnostics", "array_diagnostics"))
def test_model_tamper_is_rechecked_and_fail_closed(models, fault):
    model = deepcopy(models["M2"])
    if fault == "nan_coefficient":
        model["coefficients"][0] = float("nan")
    elif fault == "nan_covariance":
        model["covariance"][0][0] = float("nan")
    elif fault == "wrong_columns":
        model["columns"][0] = "execution_success"
    elif fault == "HC3":
        model["covariance_convention"] = "HC3"
    elif fault == "asymmetric":
        model["covariance"][0][1] += .1
    elif fault == "null_diagnostics":
        model["diagnostics"] = None
    else:
        model["diagnostics"] = []
    # Saved passed=True flags cannot authorize a malformed reloaded record.
    if isinstance(model["diagnostics"], dict):
        model["diagnostics"]["passed"] = True
    result = estimator.predict_risk(model, "PUSH", (2, 5), .2,
                                    lineage_verified=True, m2_diagnostics_passed=True)
    assert result["supported"] is False and result["risk"] is None


def test_fisher_frozen_seed_derivation_and_defaults():
    digest = hashlib.sha256(b"phase2-terminal-estimator-protocol-v1:fisher-sharp-null").digest()
    assert int.from_bytes(digest[:8], "big") & ((1 << 63) - 1) == 5027505451237408553
    assert estimator.FISHER_SEED == 5027505451237408553
    assert estimator.FISHER_REPETITIONS == 100000


def test_fisher_reproducible_fixed_counts_ascii_order_and_plus_one(rows):
    repetitions = 127  # Synthetic unit-test budget; the formal pipeline uses frozen 100000.
    first = estimator.fisher_sharp_null(rows, repetitions=repetitions, seed=estimator.FISHER_SEED)
    second = estimator.fisher_sharp_null(list(reversed(rows)), repetitions=repetitions, seed=estimator.FISHER_SEED)
    assert first == second
    sorted_rows = sorted(rows, key=lambda r: r["assignment_id"])
    loss = np.asarray([r["L_ref"] for r in sorted_rows])
    actual = abs(estimator.primary_itt(rows)["difference_in_means"])
    rng = np.random.Generator(np.random.PCG64(estimator.FISHER_SEED))
    tails = 0
    for _ in range(repetitions):
        allocation = rng.permutation(120)
        statistic = abs(float(loss[allocation[:58]].mean() - loss[allocation[58:]].mean()))
        tails += statistic >= actual - 1e-12
    assert first["tail_count"] == tails
    assert first["p_value"] == pytest.approx((tails + 1) / (repetitions + 1), abs=1e-15)


def test_fisher_sharp_null_zero_statistic_includes_ties(rows):
    for row in rows:
        row["L_ref"] = .25
    result = estimator.fisher_sharp_null(rows, repetitions=17, seed=estimator.FISHER_SEED)
    assert result["tail_count"] == 17 and result["p_value"] == 1


def test_fisher_fixed_pcg64_one_hot_reference_includes_exact_fixed_counts(rows, monkeypatch):
    # The first ASCII assignment is PUSH and has the only nonzero loss.
    # Observed |difference|=1/58. A draw is in the tail exactly when that
    # unit is in PUSH; otherwise |difference|=1/62. Frozen PCG64 seed's
    # first 11 allocations place it in PUSH on draws 3, 9 and 10.
    for row in rows:
        row["L_ref"] = float(row["assignment_id"] == "synthetic-assignment-000")
    generator = np.random.Generator
    memberships = []

    class RecordingGenerator:
        def __init__(self, bit_generator):
            assert isinstance(bit_generator, np.random.PCG64)
            self.generator = generator(bit_generator)

        def permutation(self, size):
            assert size == 120
            permutation = self.generator.permutation(size)
            # Permutation retains every unit once: no outcome bootstrap.
            assert set(permutation.tolist()) == set(range(120))
            assert len(permutation[:58]) == 58 and len(permutation[58:]) == 62
            memberships.append(0 in permutation[:58])
            return permutation

    monkeypatch.setattr(estimator.np.random, "Generator", RecordingGenerator)
    result = estimator.fisher_sharp_null(list(reversed(rows)), repetitions=11)
    assert memberships == [False, False, True, False, False, False, False, False, True, True, False]
    assert result["observed_statistic"] == pytest.approx(1 / 58, abs=1e-15)
    assert result["tail_count"] == 3
    assert result["p_value"] == pytest.approx(float(Fraction(4, 12)), abs=1e-15)


@pytest.mark.parametrize("repetitions", (0, -1, True, 1.5))
def test_fisher_invalid_repetition_count_fails(rows, repetitions):
    with pytest.raises(ValueError):
        estimator.fisher_sharp_null(rows, repetitions=repetitions, seed=estimator.FISHER_SEED)
