"""拟合双层报告概率，保存 Development 评价及嵌套折外预测。"""

import argparse
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import scipy
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from scripts import analyze_observer_dtrs_report_event as report_event  # noqa: E402
from scripts import analyze_observer_dtrs_team_event as team_event  # noqa: E402
from scripts import fit_observer_dtrs_report_event as frozen  # noqa: E402
from werewolf.artifact_io import canonical_json_bytes, canonical_jsonl_bytes, publish_artifact, sha256_bytes  # noqa: E402
from werewolf.observer_dtrs import report_state_probability as probability  # noqa: E402
from werewolf.observer_dtrs.baselines import sigmoid  # noqa: E402
from werewolf.observer_dtrs.context import ObserverContext  # noqa: E402
from werewolf.observer_dtrs.labels import label_from_observation  # noqa: E402
from werewolf.tom.protocol import BOOTSTRAP_VERSION, bootstrap_indices  # noqa: E402


DESTINATION = frozen.DATA / "dtrs/report-state-probability"
LAMBDA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.)
SOLVER_OPTIONS = {"maxiter": 1000, "gtol": 1e-9, "ftol": 1e-12}
BOOTSTRAP_REPLICATES = 10_000
STATE_NAMES = ("E0", "E1", "E2")
METHODS = tuple(probability.METHOD_FEATURES)


@dataclass(frozen=True, slots=True)
class Sample:
    context: ObserverContext
    attempt_id: str
    features: tuple[float, ...]
    state: int | None
    report_status: str
    successful_empty: bool
    m1_features: tuple[float, ...]

    @property
    def game_id(self):
        return self.context.game_id

    @property
    def sample_id(self):
        return self.game_id, self.context.boundary_id, self.context.observer_id


def pre_samples(pre, wolves, reports, q_record, *, attempt_id):
    matrix = team_event.bind_q(pre, q_record)
    panel = probability.pre_features(pre, wolves, matrix)
    legacy_features = frozen.legal_features([context for context, _ in panel], matrix)
    links = {link.observer_id: link.observation_id for link in pre.belief_observation_links}
    for (context, x), (_, _, m1_x) in zip(panel, legacy_features, strict=True):
        report_id = links[context.observer_id]
        observation = reports.get(report_id)
        state, empty = None, False
        try:
            label = label_from_observation(context, observation)
        except (TypeError, ValueError):
            status = "invalid_or_misbound"
        else:
            status = label.observation_status
            if observation is not None and (label.attempt_id != attempt_id or label.observation_id != report_id):
                status = "invalid_or_misbound"
            elif label.label_observed:
                raw = label.raw_suspicion_support
                empty = not raw
                if context.acting_wolf in raw:
                    state = 2
                else:
                    state = int(any(seat in raw for seat in context.alive_wolf_ids))
        yield Sample(context, attempt_id, x, state, status, empty, m1_x)


def population(rows):
    states = Counter(row.state for row in rows)
    return {
        "games": len({row.game_id for row in rows}),
        "pres": len({row.sample_id[:2] for row in rows}),
        "observer_records": len(rows),
        "observer_game_pairs": len({(row.game_id, row.context.observer_id) for row in rows}),
        "games_with_successful_reports": len({row.game_id for row in rows if row.state is not None}),
        "successful_reports": len(rows) - states[None],
        "unobserved_records": states[None],
        "report_statuses": dict(sorted(Counter(row.report_status for row in rows).items())),
        "states": {name: states[k] for k, name in enumerate(STATE_NAMES)},
        "successful_empty": sum(row.successful_empty for row in rows),
    }


def historical_population_check(actual):
    expected = team_event.HISTORICAL_POPULATION
    observed = {
        "games": actual["games"], "pres": actual["pres"], "rows": actual["successful_reports"],
        "speaker_hits": actual["states"]["E2"], "teammate_only_hits": actual["states"]["E1"],
        "team_misses": actual["states"]["E0"], "successful_empty": actual["successful_empty"],
    }
    return {key: {"expected": expected[key], "observed": value}
            for key, value in observed.items() if value != expected[key]}


def fit_head(x, y, weights, regularization, *, dual=None):
    """最小化原始游戏权重下的似然，掩码多项模型以 E0 为参考。"""
    if dual is None:
        if set(y) != {0, 1}:
            raise ValueError("二元 Logistic 的训练人口必须包含两个结果类别")
    elif set(y) != {0, 1, 2} or np.all(y[dual] == 1) or np.any((y == 1) & ~dual):
        raise ValueError("掩码多项 Logistic 缺少可识别的类别支持或包含单狼 E1 标签")

    # 标准化使用本头人口的加权矩。条件头的似然仍使用原始子集权重。
    normalized = weights / weights.sum()
    mean = normalized @ x
    std = np.sqrt(normalized @ (x - mean) ** 2)
    std[np.ptp(x, axis=0) == 0] = 0
    scaler = {"mean": mean.tolist(), "std": std.tolist()}
    design = np.column_stack((np.ones(len(x)), probability.standardize(x, scaler)))
    n_outputs = 1 if dual is None else 2
    target = y[:, None] if dual is None else np.eye(3)[y, 1:]
    indices = np.arange(len(y))

    def objective(flat):
        coefficients = flat.reshape(-1, n_outputs)
        logits = design @ coefficients
        slopes = coefficients[1:]
        if dual is None:
            nll = np.logaddexp(0, np.where(y == 1, -logits[:, 0], logits[:, 0]))
            fitted = sigmoid(logits)
            penalty_gradient = slopes
        else:
            log_q = probability.masked_log_softmax(logits, dual)
            nll = -log_q[indices, y]
            fitted = np.exp(log_q[:, 1:])
            # 对三个类的斜率中心化后施加 L2，使惩罚与参考类选择无关。
            penalty_gradient = slopes - slopes.sum(axis=1, keepdims=True) / 3
        gradient = design.T @ (weights[:, None] * (fitted - target))
        gradient[1:] += regularization * penalty_gradient
        loss = weights @ nll + .5 * regularization * np.sum(slopes * penalty_gradient)
        return float(loss), gradient.ravel()

    result = minimize(objective, np.zeros(design.shape[1] * n_outputs), jac=True,
                      method="L-BFGS-B", options=SOLVER_OPTIONS)
    if not result.success or not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise RuntimeError(f"联合报告概率拟合未收敛：{result.message}")
    coefficients = result.x.reshape(-1, n_outputs)
    return {"standardization": scaler, "intercept": coefficients[0].tolist(),
            "coefficients": coefficients[1:].tolist(), "iterations": int(result.nit),
            "training_records": len(y), "training_weight": float(weights.sum())}


def fit_method(method, x, y, weights, regularization):
    dual = x[:, probability.DUAL_COLUMN] == 1
    if np.any((y == 1) & ~dual):
        raise ValueError("E1 只能出现在双狼人口")
    design = probability.model_design(x, method)
    if method == probability.MULTINOMIAL:
        heads = {"state_logits": fit_head(design, y, weights, regularization, dual=dual)}
    else:
        teammate_population = dual & (y != 2)
        heads = {
            "speaker": fit_head(design, (y == 2).astype(int), weights, regularization),
            "teammate_given_no_speaker": fit_head(design[teammate_population],
                y[teammate_population], weights[teammate_population], regularization),
        }
    return {"method": method, "features": probability.METHOD_FEATURES[method],
            "lambda": regularization, "heads": heads}


def supervised(rows):
    valid = [row for row in rows if row.state is not None]
    if not valid:
        raise ValueError("训练或评价人口没有成功的原始报告")
    return valid, np.asarray([row.features for row in valid]), np.asarray([row.state for row in valid]), frozen.game_weights(valid)


def joint_losses(log_q, q, y):
    return (0.5 * np.sum((q - np.eye(3)[y]) ** 2, axis=1), -log_q[np.arange(len(y)), y])


def select_models(partitions, folds):
    """在传入折内比较模型及 λ，按验证游戏数合并各折的游戏等权分数。"""
    totals = {(method, value): np.zeros(2) for method in METHODS for value in LAMBDA_GRID}
    games = 0
    for held in folds:
        _, x, y, weights = supervised(row for fold in folds if fold != held for row in partitions[fold])
        valid, test_x, test_y, test_weights = supervised(partitions[held])
        game_count = len({row.game_id for row in valid})
        games += game_count
        for (method, value), scores in totals.items():
            model = fit_method(method, x, y, weights, value)
            log_q = probability.log_probabilities(model, test_x)
            q = probability.probabilities(log_q, test_x)
            scores += game_count * (np.asarray(joint_losses(log_q, q, test_y)) @ test_weights)

    best_lambda = {
        method: min(LAMBDA_GRID, key=lambda value: (*totals[method, value], -value))
        for method in METHODS
    }
    selected_method = min((probability.HIERARCHICAL, probability.MULTINOMIAL),
                          key=lambda method: (*totals[method, best_lambda[method]], method != probability.HIERARCHICAL))
    return {
        "selection_folds": folds, "validation_games": games,
        "scores": {method: {str(value): dict(zip(("joint_brier", "joint_log_loss"),
                                                 (totals[method, value] / games).tolist(), strict=True))
                            for value in LAMBDA_GRID} for method in METHODS},
        "best_lambda": best_lambda, "selected_method": selected_method,
    }


def nested_predictions(partitions):
    """每个外层留出折的模型与 λ 只由其余三个开发折确定。"""
    records, selections = [], {}
    for held in team_event.SELECTION_FOLDS:
        folds = tuple(fold for fold in team_event.SELECTION_FOLDS if fold != held)
        selection = selections[str(held)] = select_models(partitions, folds)
        method = selection["selected_method"]
        value = selection["best_lambda"][method]
        _, x, y, weights = supervised(row for fold in folds for row in partitions[fold])
        model = fit_method(method, x, y, weights, value)
        validation = partitions[held]
        features = np.asarray([row.features for row in validation])
        q = probability.probabilities(probability.log_probabilities(model, features), features)
        for row, joint in zip(validation, q, strict=True):
            ctx = row.context
            records.append({"fold": held, "method": method, "lambda": value, "attempt_id": row.attempt_id,
                            "game_id": ctx.game_id, "boundary_id": ctx.boundary_id, "prefix_digest": ctx.prefix_digest,
                            "step_index": ctx.step_index, "acting_wolf": ctx.acting_wolf,
                            "alive_wolf_ids": ctx.alive_wolf_ids, "observer_id": ctx.observer_id,
                            **probability.probability_record(joint)})
        print(f"完成外层 Fold {held}：{method}，λ={value}，{len(validation)} 条预测", flush=True)
    return records, selections


def evaluate_predictions(rows, *, model=None, m1_parameters=None):
    """以完整有效评价人口固定权重，狼数组内按条件质量计算指标。"""
    valid, x, y, weights = supervised(rows)
    if model is None:
        legacy = [SimpleNamespace(sample_id=row.sample_id, features=row.m1_features) for row in valid]
        p = report_event.rebuild_probabilities(legacy, "M1", m1_parameters)
        events = {"speaker": (np.asarray([p[row.sample_id] for row in valid]), y == 2)}
        losses = {}
    else:
        log_q = probability.log_probabilities(model, x)
        q = probability.probabilities(log_q, x)
        brier, log_loss = joint_losses(log_q, q, y)
        events = {"speaker": (q[:, 2], y == 2), "team": (q[:, 1] + q[:, 2], y != 0),
                  **{name: (q[:, k], y == k) for k, name in enumerate(STATE_NAMES)}}
        losses = {"joint_brier": brier, "joint_log_loss": log_loss}
    for event in ("speaker", "team"):
        if event in events:
            p, truth = events[event]
            losses[f"{event}_brier"] = (p - truth) ** 2

    summary = {}
    for group, count in (("overall", None), ("single_wolf", 1), ("two_wolves", 2)):
        group_rows = rows if count is None else [row for row in rows if row.context.wolf_count == count]
        mask = np.asarray([count is None or row.context.wolf_count == count for row in valid])
        selected = [row for row, keep in zip(valid, mask, strict=True) if keep]
        mass = float(weights[mask].sum())
        group_result = summary[group] = {"population": population(group_rows), "game_weight_mass": mass,
                                         "scores": None, "calibration": None}
        if not mass:
            continue
        weight_map = frozen.probability_map(selected, weights[mask] / mass)
        group_result["scores"] = {name: float(weights[mask] @ values[mask] / mass) for name, values in losses.items()}
        group_result["calibration"] = {}
        for event, (p, truth) in events.items():
            binary = [SimpleNamespace(game_id=row.game_id, sample_id=row.sample_id, y=int(value))
                      for row, value in zip(selected, truth[mask], strict=True)]
            group_result["calibration"][event] = frozen.reliability(binary, frozen.probability_map(binary, p[mask]), weight_map)
    return summary, losses


def paired_intervals(rows, evaluations):
    """对固定 Fold 4 预测，用相同整局抽样索引比较三组联合 Brier。"""
    game_ids = [row.game_id for row in rows if row.state is not None]
    games, game_index, counts = np.unique(game_ids, return_inverse=True, return_counts=True)
    draws = bootstrap_indices(tuple(games.tolist()), BOOTSTRAP_REPLICATES, frozen.BOOTSTRAP_SEED)
    scores = {method: np.bincount(game_index, weights=evaluations[method]["joint_brier"]) / counts
              for method in METHODS}
    intervals = {}
    for first, second in combinations(METHODS, 2):
        delta = scores[second] - scores[first]
        intervals[f"{second}_minus_{first}"] = {
            "joint_brier_difference": float(delta.mean()),
            "percentile_95": np.quantile(delta[draws].mean(axis=1), (0.025, 0.975)).tolist(),
        }
    return intervals


def run_development(partitions, m1_parameters):
    # 只将开发训练折交给选择和嵌套折外拟合。
    selection_folds = team_event.SELECTION_FOLDS
    train = {fold: partitions[fold] for fold in selection_folds}
    oof, outer_selection = nested_predictions(train)
    selection = select_models(train, selection_folds)
    _, x, y, weights = supervised(row for fold in selection_folds for row in train[fold])
    models = {method: fit_method(method, x, y, weights, selection["best_lambda"][method]) for method in METHODS}

    held = partitions[team_event.EVALUATION_FOLD]
    evaluation, scores = {}, {}
    for method, model in models.items():
        evaluation[method], scores[method] = evaluate_predictions(held, model=model)
    evaluation["frozen_m1"], _ = evaluate_predictions(held, m1_parameters=m1_parameters)
    results = {"selection": selection, "outer_selection": outer_selection, "oof_records": len(oof),
               "fold4_evaluation": evaluation, "fold4_paired_bootstrap": paired_intervals(held, scores)}
    return {"selected_method": selection["selected_method"], "models": models}, results, oof


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    if os.path.lexists(DESTINATION):
        raise FileExistsError(f"已有报告状态概率 Artifact：{DESTINATION}")
    m1 = report_event.open_fitted_models()
    partitions, lineage = team_event.load_development(team_event.DEVELOPMENT_FOLDS, build_samples=pre_samples)
    if lineage["fold_manifest_digest"] != m1["development_fold_manifest_digest"]:
        raise ValueError("Development Fold 名单与冻结 M1 来源不符")
    overall = population([row for rows in partitions.values() for row in rows])
    differences = historical_population_check(overall)
    print(json.dumps({"stage": "development_loaded", "population": overall,
                      "historical_population_differences": differences}, ensure_ascii=False), flush=True)
    with threadpool_limits(limits=1, user_api="blas"):
        models, results, oof = run_development(partitions, m1["models"]["M1"])

    source_files = (Path(__file__), Path(probability.__file__), Path(team_event.__file__))
    source = {"git_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
              "file_sha256": {str(path.resolve().relative_to(ROOT)): sha256_bytes(path.read_bytes())
                              for path in source_files}}
    lineage = {**lineage, "frozen_m1_digest": report_event.FIT_DIGEST, "source": source}
    results.update({
        **lineage, "scope": "development_internal_fixed_final_tom", "target": "successful_raw_report_state",
        "population": overall, "historical_population_differences": differences,
        "fold_populations": {str(fold): population(rows) for fold, rows in partitions.items()},
        "protocol": {"selection_folds": team_event.SELECTION_FOLDS, "evaluation_fold": team_event.EVALUATION_FOLD,
                     "lambda_grid": LAMBDA_GRID, "primary_score": "joint_brier", "tie_score": "joint_log_loss",
                     "ties": {"within_model": "larger_lambda", "between_models": probability.HIERARCHICAL},
                     "weights": "equal_game_then_equal_successful_observer_record", "calibrator": None,
                     "l2": "binary_slopes_or_centered_multinomial_slopes", "solver": "L-BFGS-B",
                     "solver_options": SOLVER_OPTIONS, "bin_edges": frozen.BIN_EDGES,
                     "bootstrap": {"version": BOOTSTRAP_VERSION, "unit": "game", "scope": "fixed_fold4_predictions",
                                   "seed": frozen.BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES}},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    })
    artifact = publish_artifact(DESTINATION, manifest_fields={
        "artifact_type": probability.ARTIFACT_TYPE, "schema_version": probability.ARTIFACT_TYPE,
        "artifact_identity": DESTINATION.name, **lineage,
    }, files={"models.json": canonical_json_bytes(models), "results.json": canonical_json_bytes(results),
              "oof_predictions.jsonl": canonical_jsonl_bytes(oof)})
    print(json.dumps({"path": str(DESTINATION), "manifest_digest": artifact.manifest_digest,
                      "selected_method": models["selected_method"], "oof_records": len(oof)}), flush=True)


if __name__ == "__main__":
    main()
