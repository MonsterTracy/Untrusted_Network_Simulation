"""Frozen Terminal Estimator Protocol V1 statistics, without gameplay imports.

These are reduced-form working models.  The support gate never turns an audit
prediction into an operational prediction, and no model selects an action.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np


NORMAL_CRITICAL = 1.959963984540054
FISHER_SEED = 5027505451237408553
FISHER_REPETITIONS = 100000
ARMS = ("PUSH", "REDIRECT")
STATE_LEVELS = ((1, 4), (2, 3), (2, 4), (2, 5))
SUPPORTED_INTERVALS = {
    (2, 4): (0.0077301424406220195, 0.6566524289627977),
    (2, 5): (0.00521034560968036, 0.5252706354667532),
}
MODEL_COLUMNS = {
    "M0": ("intercept", "I_A"),
    "M1": ("intercept", "I_A", "state_14", "state_23", "state_24"),
    "M2": ("intercept", "I_A", "state_14", "state_23", "state_24", "p_tilde", "I_A:p_tilde"),
}
NUMERICAL_TOLERANCE = 1e-10


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _state(value: Any) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("S_pre must contain two integer counts")
    if any(isinstance(x, bool) or not isinstance(x, Integral) for x in value):
        raise ValueError("S_pre must contain two integer counts")
    return int(value[0]), int(value[1])


def validate_rows(rows: Sequence[Mapping[str, Any]], *, require_features: bool = True) -> tuple[dict[str, Any], ...]:
    """Validate the complete population, then copy and sort by ASCII ID.

    Additional row fields are retained for audit but cannot enter any design.
    Primary estimation needs no conditional features and never filters a row.
    """
    if len(rows) != 120:
        raise ValueError("the frozen primary population requires exactly 120 rows")
    assignment_ids: set[str] = set()
    game_ids: set[str | int] = set()
    counts = {arm: 0 for arm in ARMS}
    copied = []
    for source in rows:
        if not isinstance(source, Mapping):
            raise ValueError("each row must be a mapping")
        row = dict(source)
        assignment_id = row.get("assignment_id")
        if not isinstance(assignment_id, str) or not assignment_id or not assignment_id.isascii():
            raise ValueError("assignment_id must be a nonempty ASCII string")
        if assignment_id in assignment_ids:
            raise ValueError("duplicate assignment_id")
        assignment_ids.add(assignment_id)
        game_id = row.get("game_id")
        if isinstance(game_id, bool) or not isinstance(game_id, (str, int)) or game_id == "":
            raise ValueError("game_id must be a nonempty string or integer")
        if game_id in game_ids:
            raise ValueError("duplicate game_id")
        game_ids.add(game_id)
        arm = row.get("assigned_action")
        if arm not in ARMS:
            raise ValueError("assigned_action must be PUSH or REDIRECT")
        counts[arm] += 1
        loss = _finite_number(row.get("L_ref"), "L_ref")
        if not 0.0 <= loss <= 1.0:
            raise ValueError("L_ref must lie in [0,1]")
        row["L_ref"] = loss
        if "assignment_probability" in row:
            if _finite_number(row["assignment_probability"], "assignment_probability") != 0.5:
                raise ValueError("assignment_probability must be 0.5")
        if require_features:
            state = _state(row.get("S_pre"))
            if state not in STATE_LEVELS:
                raise ValueError("S_pre must use a frozen categorical state")
            score = _finite_number(row.get("p_tilde_j"), "p_tilde_j")
            if not 0.0 <= score <= 1.0:
                raise ValueError("p_tilde_j must lie in [0,1]")
            row["S_pre"] = state
            row["p_tilde_j"] = score
        copied.append(row)
    if counts != {"PUSH": 58, "REDIRECT": 62}:
        raise ValueError("the frozen arm counts require PUSH=58 and REDIRECT=62")
    return tuple(sorted(copied, key=lambda row: row["assignment_id"]))


def primary_itt(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Independent raw arm means and the prespecified Neyman interval."""
    ordered = validate_rows(rows, require_features=False)
    losses = {arm: np.asarray([r["L_ref"] for r in ordered if r["assigned_action"] == arm], dtype=np.float64) for arm in ARMS}
    means = {arm: float(values.mean()) for arm, values in losses.items()}
    variances = {arm: float(values.var(ddof=1)) for arm, values in losses.items()}
    difference = means["PUSH"] - means["REDIRECT"]
    variance = variances["PUSH"] / 58 + variances["REDIRECT"] / 62
    standard_error = math.sqrt(variance)
    return {
        "population": "all_120_randomized_assignments",
        "row_count": 120,
        "arm_counts": {"PUSH": 58, "REDIRECT": 62},
        "arm_means": means,
        "arm_sample_variances": variances,
        "difference_in_means": difference,
        "neyman_variance": variance,
        "standard_error": standard_error,
        "interval": [difference - NORMAL_CRITICAL * standard_error, difference + NORMAL_CRITICAL * standard_error],
        "uncertainty_method": "Neyman randomization-based normal-approximation 95% interval",
        "normal_critical": NORMAL_CRITICAL,
        "variance_ddof": 1,
        "model_adjusted": False,
    }


def fisher_sharp_null(rows: Sequence[Mapping[str, Any]], *, repetitions: int = FISHER_REPETITIONS, seed: int = FISHER_SEED) -> dict[str, Any]:
    """Supplementary fixed-count sharp-null Monte Carlo randomization test.

    Small explicit repetitions/seeds are useful for synthetic unit tests.  The
    formal analysis adapter must enforce the exported frozen defaults.
    """
    ordered = validate_rows(rows, require_features=False)
    if isinstance(repetitions, bool) or not isinstance(repetitions, Integral) or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    loss = np.asarray([row["L_ref"] for row in ordered], dtype=np.float64)
    observed = abs(primary_itt(ordered)["difference_in_means"])
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    tail_count = 0
    for _ in range(int(repetitions)):
        permutation = rng.permutation(120)
        statistic = abs(float(loss[permutation[:58]].mean() - loss[permutation[58:]].mean()))
        tail_count += statistic >= observed - 1e-12
    return {
        "method": "Fisher sharp-null Monte Carlo fixed-count supplement",
        "null": "all_unit_treatment_effects_zero",
        "row_count": 120,
        "arm_counts": {"PUSH": 58, "REDIRECT": 62},
        "statistic": "absolute_raw_difference_in_means",
        "observed_statistic": observed,
        "repetitions": int(repetitions),
        "seed": int(seed),
        "prng": "NumPy Generator(PCG64)",
        "numpy_version": np.__version__,
        "row_order": "assignment_id_ASCII_lexicographic",
        "tail_tolerance": 1e-12,
        "tail_count": int(tail_count),
        "p_value": (int(tail_count) + 1) / (int(repetitions) + 1),
        "allocations_sampled_with_replacement": True,
        "weak_null_test": False,
        "primary_replacement": False,
    }


def _design_row(model_name: str, action: str, state: tuple[int, int], score: float) -> np.ndarray:
    if model_name not in MODEL_COLUMNS:
        raise ValueError("model_name must be M0, M1 or M2")
    treatment = float(action == "PUSH")
    values = [1.0, treatment]
    if model_name != "M0":
        values.extend(float(state == s) for s in STATE_LEVELS[:3])
    if model_name == "M2":
        values.extend((score, treatment * score))
    return np.asarray(values, dtype=np.float64)


def design_matrix(rows: Sequence[Mapping[str, Any]], model_name: str) -> np.ndarray:
    """The sole fixed column encoding, including all 120 rows."""
    ordered = validate_rows(rows)
    return np.asarray([_design_row(model_name, r["assigned_action"], r["S_pre"], r["p_tilde_j"]) for r in ordered], dtype=np.float64)


def support_gate(action: str, S_pre: Any, p_tilde: Any, *, lineage_verified: bool, diagnostics_passed: bool) -> dict[str, Any]:
    """State/range eligibility only; rejected calls contain no numeric risk."""
    if lineage_verified is not True:
        return {"status": "LINEAGE_MISMATCH", "supported": False}
    if not isinstance(action, str) or action not in ARMS:
        return {"status": "INVALID_ACTION", "supported": False}
    try:
        state = _state(S_pre)
        score = _finite_number(p_tilde, "p_tilde")
    except ValueError:
        return {"status": "INVALID_INPUT", "supported": False}
    if not 0.0 <= score <= 1.0:
        return {"status": "INVALID_INPUT", "supported": False}
    if state == (2, 3):
        return {"status": "AUDIT_ONLY", "supported": False}
    if state not in SUPPORTED_INTERVALS:
        return {"status": "UNSUPPORTED", "supported": False}
    lower, upper = SUPPORTED_INTERVALS[state]
    if not lower <= score <= upper:
        return {"status": "UNSUPPORTED_EXTRAPOLATION", "supported": False}
    if diagnostics_passed is not True:
        return {"status": "MODEL_SANITY_FAILED", "supported": False}
    return {"status": "SUPPORTED", "supported": True}


def _gate_self_check() -> bool:
    for state, (lower, upper) in SUPPORTED_INTERVALS.items():
        for action in ARMS:
            for score in (lower, upper):
                if not support_gate(action, state, score, lineage_verified=True, diagnostics_passed=True)["supported"]:
                    return False
            for score in (np.nextafter(lower, -math.inf), np.nextafter(upper, math.inf)):
                if support_gate(action, state, score, lineage_verified=True, diagnostics_passed=True)["supported"]:
                    return False
    for state in ((2, 3), (1, 4), (9, 9), None):
        if support_gate("PUSH", state, 0.1, lineage_verified=True, diagnostics_passed=True)["supported"]:
            return False
    for score in (None, math.nan, math.inf, -math.inf):
        if support_gate("PUSH", (2, 4), score, lineage_verified=True, diagnostics_passed=True)["supported"]:
            return False
    return not any((
        support_gate("PUSH", (2, 4), 0.1, lineage_verified=False, diagnostics_passed=True)["supported"],
        support_gate("PUSH", (2, 4), 0.1, lineage_verified=True, diagnostics_passed=False)["supported"],
    ))


def _arrays(model: Mapping[str, Any]) -> tuple[str, np.ndarray, np.ndarray]:
    if not isinstance(model, Mapping):
        raise ValueError("model must be a mapping")
    name = model.get("model_name")
    if name not in MODEL_COLUMNS or list(model.get("columns", ())) != list(MODEL_COLUMNS[name]):
        raise ValueError("model uses an invalid frozen design")
    if model.get("covariance_convention") != "HC1":
        raise ValueError("model covariance convention must be HC1")
    k = len(MODEL_COLUMNS[name])
    try:
        raw_coefficient = np.asarray(model.get("coefficients"), dtype=object)
        raw_covariance = np.asarray(model.get("covariance"), dtype=object)
        coefficient = np.asarray([_finite_number(value, "coefficient") for value in raw_coefficient.flat], dtype=np.float64).reshape(raw_coefficient.shape)
        covariance = np.asarray([_finite_number(value, "covariance") for value in raw_covariance.flat], dtype=np.float64).reshape(raw_covariance.shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("model coefficients/covariance must be numeric") from exc
    if coefficient.shape != (k,) or covariance.shape != (k, k):
        raise ValueError("model coefficient/covariance dimensions mismatch")
    if not np.isfinite(coefficient).all() or not np.isfinite(covariance).all():
        raise ValueError("model coefficients/covariance must be finite")
    return name, coefficient, covariance


def _prediction_values(x: np.ndarray, coefficient: np.ndarray, covariance: np.ndarray) -> dict[str, Any]:
    risk = float(x @ coefficient)
    variance = float(x @ covariance @ x)
    eigenvalues = np.linalg.eigvalsh((covariance + covariance.T) / 2)
    epsilon = NUMERICAL_TOLERANCE * max(1.0, float(x @ x) * float(np.max(np.abs(eigenvalues))))
    if not math.isfinite(risk) or not math.isfinite(variance) or variance < -epsilon:
        raise ValueError("nonfinite mean or materially negative/nonfinite prediction variance")
    roundoff = variance < 0.0
    standard_error = math.sqrt(max(0.0, variance))
    interval = [risk - NORMAL_CRITICAL * standard_error, risk + NORMAL_CRITICAL * standard_error]
    if not all(math.isfinite(value) for value in interval):
        raise ValueError("nonfinite prediction interval")
    return {
        "risk": risk,
        "variance": variance,
        "standard_error": standard_error,
        "interval": interval,
        "variance_roundoff": roundoff,
        "variance_tolerance": epsilon,
        "interval_kind": "normal_95_pointwise_model_mean_diagnostic",
    }


def _model_checks(model: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "design_full_rank": False,
        "finite_coefficients": False,
        "finite_covariance_uncertainty": False,
        "finite_supported_means": False,
        "supported_risk_range": False,
        "reproducible_arm_predictions": False,
        "support_gate_complete": _gate_self_check(),
    }
    reasons = []
    if not isinstance(model, Mapping):
        return {"passed": False, "checks": checks, "reasons": ["model must be a mapping"]}
    name = model.get("model_name")
    k = len(MODEL_COLUMNS.get(name, ()))
    design = model.get("diagnostics", {})
    if not isinstance(design, Mapping):
        return {"passed": False, "checks": checks, "reasons": ["model diagnostics must be a mapping"]}
    try:
        singular = np.asarray(design.get("singular_values"), dtype=np.float64)
        tolerance = 120 * np.finfo(np.float64).eps * float(singular.max())
        rank = int(np.count_nonzero(singular > tolerance))
        checks["design_full_rank"] = bool(
            k > 0 and singular.shape == (k,) and np.isfinite(singular).all()
            and (singular >= 0).all() and rank == k
            and design.get("n") == 120 and design.get("k") == k
            and design.get("rank") == rank and design.get("df_residual") == 120 - k
        )
    except (TypeError, ValueError):
        pass
    try:
        name, coefficient, covariance = _arrays(model)
    except (TypeError, ValueError) as exc:
        return {"passed": False, "checks": checks, "reasons": [str(exc)] + (["design is not full rank"] if not checks["design_full_rank"] else [])}
    checks["finite_coefficients"] = True
    maximum_entry = float(np.max(np.abs(covariance)))
    symmetry_tolerance = NUMERICAL_TOLERANCE * max(1.0, maximum_entry)
    symmetric = bool(np.max(np.abs(covariance - covariance.T)) <= symmetry_tolerance)
    try:
        eigenvalues = np.linalg.eigvalsh((covariance + covariance.T) / 2)
    except np.linalg.LinAlgError:
        return {"passed": False, "checks": checks, "reasons": ["covariance eigenvalue computation failed"]}
    psd_tolerance = NUMERICAL_TOLERANCE * max(1.0, float(np.max(np.abs(eigenvalues))))
    psd = bool(np.isfinite(eigenvalues).all() and eigenvalues.min() >= -psd_tolerance)
    finite_variance = True
    finite_means = True
    legal_means = True
    reproducible = True
    domain_checks = []
    # JSON float round-trip is the on-disk coefficient representation.
    reloaded = np.asarray(json.loads(json.dumps(coefficient.tolist(), allow_nan=False)), dtype=np.float64)
    for state, (lower, upper) in SUPPORTED_INTERVALS.items():
        for action in ARMS:
            x_lower = _design_row(name, action, state, lower)
            x_upper = _design_row(name, action, state, upper)
            direction = (x_upper - x_lower) / (upper - lower)
            base = x_lower - lower * direction
            # v(p)=a*p^2+b*p+c; include its interior extremum.
            a = float(direction @ covariance @ direction)
            b = float(base @ covariance @ direction + direction @ covariance @ base)
            c = float(base @ covariance @ base)
            finite_variance &= all(math.isfinite(value) for value in (a, b, c))
            points = [lower, upper]
            if math.isfinite(a) and math.isfinite(b) and a != 0.0:
                stationary = -b / (2.0 * a)
                if lower < stationary < upper:
                    points.append(stationary)
            means = []
            variances = []
            for score in points:
                x = _design_row(name, action, state, score)
                mean = float(x @ coefficient)
                variance = float(x @ covariance @ x)
                epsilon = NUMERICAL_TOLERANCE * max(1.0, float(x @ x) * float(np.max(np.abs(eigenvalues))))
                finite_variance &= math.isfinite(variance) and variance >= -epsilon
                finite_means &= math.isfinite(mean)
                legal_means &= math.isfinite(mean) and -NUMERICAL_TOLERANCE <= mean <= 1 + NUMERICAL_TOLERANCE
                reproducible &= math.isfinite(mean) and abs(mean - float(x @ reloaded)) <= NUMERICAL_TOLERANCE
                means.append(mean if math.isfinite(mean) else None)
                variances.append(variance if math.isfinite(variance) else None)
            domain_checks.append({"S_pre": list(state), "action": action, "scores": points, "means": means, "variances": variances})
    checks["finite_covariance_uncertainty"] = bool(symmetric and psd and finite_variance)
    checks["finite_supported_means"] = bool(finite_means)
    checks["supported_risk_range"] = bool(legal_means)
    checks["reproducible_arm_predictions"] = bool(reproducible)
    for check, passed in checks.items():
        if not passed:
            reasons.append(check)
    return {
        "passed": not reasons,
        "checks": checks,
        "reasons": reasons,
        "covariance_symmetric": symmetric,
        "covariance_psd": psd,
        "covariance_eigenvalues": eigenvalues.tolist(),
        "symmetry_tolerance": symmetry_tolerance,
        "psd_tolerance": psd_tolerance,
        "supported_domain_checks": domain_checks,
    }


def model_diagnostics(model: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute numerical eligibility rather than trusting stored PASS flags."""
    try:
        return _model_checks(model)
    except (ArithmeticError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return {"passed": False, "checks": {}, "reasons": [f"numerical diagnostic failure: {exc}"]}


def _nullable(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def fit_models(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Fit all three prespecified OLS models and raw HC1 covariance.

    A deficient design yields a failed record, never a reduced design or ridge.
    The primary ITT estimator is independent of all these records.
    """
    ordered = validate_rows(rows)
    y = np.asarray([row["L_ref"] for row in ordered], dtype=np.float64)
    records = {}
    for name, columns in MODEL_COLUMNS.items():
        X = np.asarray([_design_row(name, r["assigned_action"], r["S_pre"], r["p_tilde_j"]) for r in ordered], dtype=np.float64)
        n, k = X.shape
        record: dict[str, Any] = {
            "model_name": name,
            "columns": list(columns),
            "coefficients": None,
            "covariance": None,
            "standard_errors": None,
            "coefficient_intervals": None,
            "covariance_convention": "HC1",
            "estimator": "unweighted_unregularized_OLS_identity_link",
            "diagnostics": {"n": n, "k": k, "df_residual": n - k, "rank": None, "singular_values": None, "condition_number": None},
        }
        records[name] = record
        try:
            U, singular, Vt = np.linalg.svd(X, full_matrices=False)
            tolerance = max(n, k) * np.finfo(np.float64).eps * float(singular.max())
            rank = int(np.count_nonzero(singular > tolerance))
            record["diagnostics"].update({"rank": rank, "singular_values": singular.tolist(), "rank_tolerance": tolerance, "condition_number": _nullable(singular.max() / singular.min()) if singular.min() else None})
            if rank != k:
                record["diagnostics"].update({"passed": False, "checks": {"design_full_rank": False}, "reasons": ["design_full_rank"]})
                continue
            coefficient = Vt.T @ ((U.T @ y) / singular)
            inverse = (Vt.T / singular**2) @ Vt
            residual = y - X @ coefficient
            meat = (X * residual[:, None]).T @ (X * residual[:, None])
            covariance = (n / (n - k)) * inverse @ meat @ inverse
            if not np.isfinite(coefficient).all() or not np.isfinite(covariance).all():
                raise ValueError("nonfinite fitted coefficients or HC1 covariance")
            record["coefficients"] = coefficient.tolist()
            record["covariance"] = covariance.tolist()
            leverage = np.sum(U**2, axis=1)
            record["diagnostics"].update({
                "leverage": {"min": float(leverage.min()), "max": float(leverage.max()), "mean": float(leverage.mean()), "sum": float(leverage.sum()), "near_one_count": int(np.count_nonzero(np.abs(leverage - 1.0) <= NUMERICAL_TOLERANCE))},
                "residual": {"min": float(residual.min()), "max": float(residual.max()), "mean": float(residual.mean()), "sum_squares": float(residual @ residual), "max_absolute": float(np.max(np.abs(residual)))},
            })
            coefficient_values = [_prediction_values(np.eye(k)[index], coefficient, covariance) for index in range(k)]
            record["standard_errors"] = [value["standard_error"] for value in coefficient_values]
            record["coefficient_intervals"] = [value["interval"] for value in coefficient_values]
            record["coefficient_variance_roundoff"] = [value["variance_roundoff"] for value in coefficient_values]
            record["diagnostics"].update(model_diagnostics(record))
            if name == "M0":
                means = primary_itt(ordered)["arm_means"]
                consistent = all(
                    abs(float(_design_row(name, arm, (2, 5), 0.0) @ coefficient) - means[arm]) <= 1e-12
                    for arm in ARMS
                )
                record["diagnostics"]["raw_arm_means_consistent"] = consistent
                if not consistent:
                    record["diagnostics"]["passed"] = False
                    record["diagnostics"]["reasons"].append("raw_arm_means_consistent")
        except (ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
            record["diagnostics"].update({"passed": False, "checks": {}, "reasons": [f"fit failure: {exc}"]})
    return records


def audit_prediction(model: Mapping[str, Any], action: str, S_pre: Any, p_tilde: Any) -> dict[str, Any]:
    """A separately labelled raw diagnostic; it never grants eligibility."""
    empty = {"audit_only": True, "supported": False, "risk": None, "variance": None, "standard_error": None, "interval": None}
    if not isinstance(action, str) or action not in ARMS:
        return {**empty, "status": "INVALID_ACTION"}
    try:
        state = _state(S_pre)
        if state not in STATE_LEVELS:
            return {**empty, "status": "UNSUPPORTED_STATE_ENCODING"}
        score = _finite_number(p_tilde, "p_tilde")
        if not 0 <= score <= 1:
            return {**empty, "status": "INVALID_INPUT"}
        name, coefficient, covariance = _arrays(model)
        values = _prediction_values(_design_row(name, action, state, score), coefficient, covariance)
        return {**empty, **values, "status": "AUDIT_PREDICTION"}
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return {**empty, "status": "MODEL_OR_INPUT_INVALID"}


def predict_risk(model: Mapping[str, Any], action: str, S_pre: Any, p_tilde: Any, *, lineage_verified: bool, m2_diagnostics_passed: bool) -> dict[str, Any]:
    """Return an operational mean only after the frozen gate and fresh checks.

    The adapter supplies verified lineage and the complete analysis's M2 status;
    M0/M1 do not provide a fallback when M2 is ineligible.
    """
    empty = {"risk": None, "variance": None, "standard_error": None, "interval": None}
    gate = support_gate(action, S_pre, p_tilde, lineage_verified=lineage_verified, diagnostics_passed=m2_diagnostics_passed)
    if not gate["supported"]:
        return {**gate, **empty}
    if not model_diagnostics(model)["passed"]:
        return {"status": "MODEL_SANITY_FAILED", "supported": False, **empty}
    audit = audit_prediction(model, action, S_pre, p_tilde)
    if audit["risk"] is None or not -NUMERICAL_TOLERANCE <= audit["risk"] <= 1 + NUMERICAL_TOLERANCE:
        return {"status": "MODEL_SANITY_FAILED", "supported": False, **empty}
    return {"status": "SUPPORTED", "supported": True, **{key: value for key, value in audit.items() if key not in ("status", "supported", "audit_only")}}
