"""在固定 Final-ToM Q 上拟合并评价原始报告事件的三种概率模型。"""

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import warnings

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from werewolf.artifact_io import (  # noqa: E402
    canonical_json_bytes, open_artifact_envelope, publish_artifact, read_artifact_file, sha256_bytes,
)
from werewolf.canonical_collection.public_history import PLAYER_IDS  # noqa: E402
from werewolf.development_publication import open_role_sidecar  # noqa: E402
from werewolf.development_publication_v2 import PUBLICATION_TYPE, PUBLICATION_VERSION  # noqa: E402
from werewolf.observer_dtrs.baselines import sigmoid  # noqa: E402
from werewolf.observer_dtrs.context import SPEECH_PHASES, contexts_for_pre  # noqa: E402
from werewolf.observer_dtrs.features import observer_features  # noqa: E402
from werewolf.observer_dtrs.labels import label_from_observation  # noqa: E402
from werewolf.observer_dtrs.probabilities import ProbabilityMatrix  # noqa: E402
from werewolf.observer_dtrs.scoring import paired_difference, score_predictions  # noqa: E402
from werewolf.tom.development_input import open_development_input  # noqa: E402
from werewolf.tom.protocol import BOOTSTRAP_VERSION, bootstrap_indices  # noqa: E402


DATA = Path("/data/yuxiao/Untrusted_Network_Simulation")
DEVELOPMENT = DATA / "publications/paper-canonical-qwen35-9b-5800-v1-release-v1-development"
Q_PATH = DATA / "dtrs/final-tom-development-q"
DESTINATION = DATA / "dtrs/report-event-probability"
PUBLICATION_DIGEST = "28843a2f1cd79750c588682c376d66f6636c0bb955ab59ca6ee64ddbce4c641a"
SPLIT_DIGEST = "edc271676649d9b23e01f505f0f23deb38fdbd0b0ab0ad65753a90cd562ee6e8"
Q_DIGEST = "0ed5354a53952821dddce36bd7b72a9d098bcc43020b043bcc5c885251163289"
Q_SHA256 = "8425429cbd5e1af4ac09edee9d5a6b01c8726e4d119bf7399a351909da5ea9bc"
Q_TYPE = "observer_dtrs_development_q"
RESULT_TYPE = "observer_dtrs_report_event_probability"
TRAIN_FOLDS = (0, 1, 2)
CAL_FOLD = 3
EVAL_FOLD = 4
LAMBDA_GRID = (0.001, 0.01, 0.1, 1.0)
CAL_LAMBDA = 1e-4
LOGISTIC_OPTIONS = {"solver": "lbfgs", "fit_intercept": True, "max_iter": 1000, "tol": 1e-8}
BOOTSTRAP_SEED = 1729
BOOTSTRAP_REPLICATES = 5000
BIN_EDGES = tuple(i / 10 for i in range(11))
LOSS_NAMES = ("brier", "log_loss")
MODEL_NAMES = {"M0": "Context Baseline", "M1": "Own-Observer ToM", "M2": "Panel-Derived ToM"}
CONTEXT_FEATURES = ("day_2", "day_3_or_later", "pk_discussion",
                    *(f"alive_players_{n}" for n in range(2, 7)), "two_alive_wolves")
OWN_FEATURES = (*CONTEXT_FEATURES, "actor_mass", "actor_mass_squared", "row_entropy_over_log6")
FEATURES = {"M0": CONTEXT_FEATURES, "M1": OWN_FEATURES,
            "M2": (*OWN_FEATURES, "peer_actor_mass_mean", "peer_actor_mass_population_std")}


@dataclass(frozen=True)
class Sample:
    """离线记录。拟合设计矩阵只从已构造的特征列读取。"""

    game_id: str
    boundary_id: str
    observer_id: str
    day: int
    phase: str
    alive_wolves: int
    peer_count: int
    features: tuple[float, ...]
    y: int
    successful_empty: bool

    @property
    def sample_id(self):
        return self.game_id, self.boundary_id, self.observer_id


def legal_features(contexts, matrix):
    """由合法观察者及绝对座位 Q 构造嵌套设计。"""
    q_rows = {ctx.observer_id: matrix.row(ctx.observer_id) for ctx in contexts}
    masses = {ctx.observer_id: observer_features(ctx, q_rows[ctx.observer_id]).s for ctx in contexts}
    for ctx in contexts:
        observer = ctx.observer_id
        n_alive = len(ctx.alive_player_ids)
        peers = [mass for other, mass in masses.items() if other != observer]
        k = len(peers)
        if k != n_alive - ctx.wolf_count - 1:
            raise ValueError("incomplete legal observer panel")
        mean = math.fsum(peers) / k if k else 0.0
        std = math.sqrt(math.fsum((p - mean) ** 2 for p in peers) / k) if k else 0.0
        entropy = -math.fsum(v * math.log(v) for v in q_rows[observer].values if v) / math.log(6)
        s = masses[observer]
        yield ctx, k, (float(ctx.day == 2), float(ctx.day >= 3), float(ctx.phase == "pk_discussion"),
                       *(float(n_alive == n) for n in range(2, 7)), float(ctx.wolf_count == 2),
                       s, s * s, entropy, mean, std)


def pre_samples(prefix, wolves, observations, q_record):
    identity = (prefix.game_id, prefix.boundary_id, prefix.prefix_digest, prefix.step_index)
    if tuple(q_record[name] for name in ("game_id", "boundary_id", "prefix_digest", "step_index")) != identity:
        raise ValueError(f"Q does not bind to PRE {prefix.boundary_id}")
    matrix = ProbabilityMatrix(*identity[:3], q_record["q"])
    contexts = contexts_for_pre(prefix, wolves)
    if not contexts:
        raise ValueError(f"Q has no eligible observer at {prefix.boundary_id}")
    links = {link.observer_id: link.observation_id for link in prefix.belief_observation_links}
    for ctx, k, x in legal_features(contexts, matrix):
        observation_id = links[ctx.observer_id]
        label = label_from_observation(ctx, observations.get(observation_id))
        if not label.label_observed or label.observation_id != observation_id:
            raise ValueError(f"missing, failed or misbound report {observation_id}")
        raw = label.raw_suspicion_support
        yield Sample(ctx.game_id, ctx.boundary_id, ctx.observer_id, ctx.day, ctx.phase,
                     ctx.wolf_count, k, x, int(ctx.acting_wolf in raw), not raw)


def support(rows):
    positive = sum(row.y for row in rows)
    return {"rows": len(rows), "games": len({row.game_id for row in rows}),
            "positive": positive, "negative": len(rows) - positive}


def population(rows):
    return {**support(rows), "pres": len({(row.game_id, row.boundary_id) for row in rows}),
            "successful_empty": sum(row.successful_empty for row in rows),
            "peer_count_rows": dict(sorted(Counter(str(row.peer_count) for row in rows).items()))}


def load_development_samples():
    publication = open_artifact_envelope(DEVELOPMENT, expected_artifact_type=PUBLICATION_TYPE,
                                         expected_schema_version=PUBLICATION_VERSION)
    pm = publication.manifest
    if publication.manifest_digest != PUBLICATION_DIGEST or pm["split_manifest_digest"] != SPLIT_DIGEST:
        raise ValueError("unexpected Development Publication or Split")
    q_artifact = open_artifact_envelope(Q_PATH, expected_artifact_type=Q_TYPE, expected_schema_version=Q_TYPE)
    qm = q_artifact.manifest
    if (q_artifact.manifest_digest != Q_DIGEST or qm["publication_digest"] != PUBLICATION_DIGEST
            or qm["split_digest"] != SPLIT_DIGEST
            or qm["development_game_set_digest"] != pm["development_game_set_digest"]
            or qm["collection_identity_digest"] != pm["collection_identity_digest"]
            or tuple(qm["row_and_column_seat_order"]) != PLAYER_IDS
            or qm["file_table"]["predictions.jsonl"]["sha256"] != Q_SHA256):
        raise ValueError("Q archive differs from the approved Development source")
    by_pre = {}
    for line in read_artifact_file(q_artifact, "predictions.jsonl").splitlines():
        record = json.loads(line)
        key = record["game_id"], record["boundary_id"]
        if key in by_pre:
            raise ValueError(f"duplicate Q at PRE {key}")
        by_pre[key] = record
    if len(by_pre) != 18813:
        raise ValueError(f"expected 18,813 Q matrices, got {len(by_pre)}")

    game_ids = tuple(game["game_id"] for game in pm["development_games"])
    rows, roles = [], None
    # 分批读取完整游戏，避免让整个 Publication 常驻内存。
    for offset in range(0, len(game_ids), 120):
        batch_ids = game_ids[offset:offset + 120]
        batch = open_development_input(DEVELOPMENT, expected_publication_digest=PUBLICATION_DIGEST,
                                       expected_split_digest=SPLIT_DIGEST, game_ids=batch_ids)
        if roles is None:
            sidecar = open_role_sidecar(batch)
            roles = {(g.game_id, g.bundle_digest): tuple(p for p, r in g.role_assignment if r == "Werewolf")
                     for g in sidecar.games}
            fold_manifest = batch.public_view.fold_manifest
            fold_games = {fold.fold_index: tuple(sorted(fold.game_ids)) for fold in fold_manifest.folds}
            if {fold: len(games) for fold, games in fold_games.items()} != dict.fromkeys(range(5), 960):
                raise ValueError("expected five verified Development folds of 960 games")
        if tuple(game.game_id for game in batch.public_view.games) != batch_ids:
            raise ValueError("Development batch differs from published membership")
        for game in batch.public_view.games:
            wolves = roles[game.game_id, game.bundle_digest]
            observations = {o.observation_id: o for o in game.belief_observations}
            for prefix in game.authoritative_pre_prefixes:
                if prefix.current_speaker not in wolves or prefix.public_temporal_state.phase.value not in SPEECH_PHASES:
                    continue
                key = game.game_id, prefix.boundary_id
                record = by_pre.pop(key, None)
                if record is None:
                    raise ValueError(f"missing Q at PRE {key}")
                rows.extend(pre_samples(prefix, wolves, observations, record))
        if offset == 0 or (offset + len(batch_ids)) % 960 == 0:
            print(f"Development 联结 {offset + len(batch_ids)}/{len(game_ids)} 局，{len(rows)} 条记录", flush=True)
    observed = population(rows)
    expected = {"games": 4800, "pres": 18813, "rows": 73224, "positive": 37280,
                "negative": 35944, "successful_empty": 17547}
    if by_pre or any(observed[k] != v for k, v in expected.items()):
        raise ValueError(f"unexpected Development population: {observed}; unconsumed Q={len(by_pre)}")
    return split_samples(rows, fold_games), fold_games, fold_manifest.manifest_digest, observed


def split_samples(rows, fold_games):
    fold_by_game = {game: fold for fold, games in fold_games.items() for game in games}
    if len(fold_by_game) != sum(map(len, fold_games.values())):
        raise ValueError("a game belongs to more than one Fold")
    if ({row.game_id for row in rows} != set(fold_by_game)
            or len({row.sample_id for row in rows}) != len(rows)):
        raise ValueError("Fold membership or unique observation coverage differs")
    partitions = {fold: [] for fold in fold_games}
    for row in rows:
        partitions[fold_by_game[row.game_id]].append(row)
    return partitions


def game_weights(rows):
    counts = Counter(row.game_id for row in rows)
    return np.asarray([1 / (len(counts) * counts[row.game_id]) for row in rows], dtype=np.float64)


def design(rows, model):
    return np.asarray([row.features[:len(FEATURES[model])] for row in rows], dtype=np.float64)


def fit_standardizer(x, weights):
    continuous = x[:, len(CONTEXT_FEATURES):]
    mean = weights @ continuous
    std = np.sqrt(weights @ (continuous - mean) ** 2)
    std[np.ptp(continuous, axis=0) == 0] = 0
    return {"mean": mean.tolist(), "std": std.tolist()}


def standardize(x, parameters):
    continuous = x[:, len(CONTEXT_FEATURES):]
    std = np.asarray(parameters["std"])
    z = np.divide(continuous - parameters["mean"], std, out=np.zeros_like(continuous), where=std != 0)
    return np.column_stack((x[:, :len(CONTEXT_FEATURES)], z))


def fit_logistic(x, rows, weights, regularization):
    y = np.asarray([r.y for r in rows])
    if set(y) != {0, 1}:
        raise ValueError("Logistic fitting requires both outcome classes")
    # lbfgs 的有效 L2 系数为 1/(C*sum(sample_weight))，权重总和为 1。
    estimator = LogisticRegression(C=1 / regularization, **LOGISTIC_OPTIONS)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(x, y, sample_weight=weights)
    intercept = float(estimator.intercept_[0])
    coefficients = estimator.coef_[0]
    if not math.isfinite(intercept) or not np.isfinite(coefficients).all():
        raise ValueError("nonfinite fitted Logistic parameters")
    return {"intercept": intercept, "coefficients": coefficients.tolist(), "lambda": regularization,
            "iterations": int(estimator.n_iter_[0]), "converged": True}


def logits(x, fitted):
    return fitted["intercept"] + x @ np.asarray(fitted["coefficients"])


def probability_map(rows, values):
    return {row.sample_id: float(p) for row, p in zip(rows, values, strict=True)}


def fit_models(train, calibration):
    """训练和校准只接收 Fold 0–3。"""
    train_rows = [row for fold in TRAIN_FOLDS for row in train[fold]]
    weights, cal_weights = game_weights(train_rows), game_weights(calibration)
    models = {}
    for model in MODEL_NAMES:
        print(f"拟合 {model} · {MODEL_NAMES[model]}", flush=True)
        by_lambda = {value: [] for value in LAMBDA_GRID}
        for held in TRAIN_FOLDS:
            inner = [row for fold in TRAIN_FOLDS if fold != held for row in train[fold]]
            validation = train[held]
            inner_weights = game_weights(inner)
            x = design(inner, model)
            prep = fit_standardizer(x, inner_weights)
            x, v = standardize(x, prep), standardize(design(validation, model), prep)
            for value in LAMBDA_GRID:
                fitted = fit_logistic(x, inner, inner_weights, value)
                predictions = probability_map(validation, sigmoid(logits(v, fitted)))
                scores = score_predictions(validation, predictions)
                by_lambda[value].extend(game["brier"] for game in scores["by_game"].values())
        selection = {value: math.fsum(losses) / len(losses) for value, losses in by_lambda.items()}
        best = min(selection, key=lambda value: (selection[value], -value))
        x = design(train_rows, model)
        prep = fit_standardizer(x, weights)
        fitted = fit_logistic(standardize(x, prep), train_rows, weights, best)
        logit = logits(standardize(design(calibration, model), prep), fitted)
        calibrator = fit_logistic(logit[:, None], calibration, cal_weights, CAL_LAMBDA)
        models[model] = {"name": MODEL_NAMES[model], "features": FEATURES[model], "standardization": prep,
                         "inner_game_macro_brier": {str(value): score for value, score in selection.items()},
                         "base": fitted, "calibrator": calibrator}
    return models


def reliability(rows, probabilities, weights):
    bins = [[] for _ in range(10)]
    for row in rows:
        bins[bisect_right(BIN_EDGES[1:-1], probabilities[row.sample_id])].append(row)
    summary = []
    for lo, hi, group in zip(BIN_EDGES[:-1], BIN_EDGES[1:], bins, strict=True):
        mass = math.fsum(weights[row.sample_id] for row in group)
        predicted = math.fsum(weights[row.sample_id] * probabilities[row.sample_id] for row in group)
        observed = math.fsum(weights[row.sample_id] * row.y for row in group)
        summary.append({"lo": lo, "hi": hi, **support(group), "mass": mass,
                        "mean_probability": predicted / mass if mass else None,
                        "event_rate": observed / mass if mass else None})
    ece = math.fsum(b["mass"] * abs(b["mean_probability"] - b["event_rate"]) for b in summary if b["mass"])
    return {"bins": summary, "ece": ece}


def grouped_scores(groups, probabilities, game_sizes):
    result = {}
    for name, group in groups.items():
        if not group:
            result[name] = {**support(group), "game_mass": 0.0, "game_weighted_scores": None}
            continue
        local = score_predictions(group, {r.sample_id: probabilities[r.sample_id] for r in group})
        by_game = local["by_game"]
        masses = {game: values["rows"] / game_sizes[game] for game, values in by_game.items()}
        total_mass = math.fsum(masses.values())
        result[name] = {**support(group), "game_mass": total_mass / len(game_sizes), "game_weighted_scores": {
            metric: math.fsum(masses[g] * by_game[g][metric] for g in by_game) / total_mass
            for metric in (*LOSS_NAMES, "bias")}}
    return result


def paired_intervals(first, second, games, draws):
    point = paired_difference(first, second)
    if set(first["by_game"]) != set(games):
        raise ValueError("bootstrap game population differs from paired scores")
    comparisons = {}
    for loss in LOSS_NAMES:
        delta = np.asarray([second["by_game"][g][loss] - first["by_game"][g][loss] for g in games])
        bootstrap = delta[draws].mean(axis=1)
        comparisons[loss] = {"game_macro_difference": point[loss],
                             "percentile_95": np.quantile(bootstrap, (0.025, 0.975)).tolist(),
                             "record_equal_difference": second["record_equal"][loss] - first["record_equal"][loss]}
    return comparisons


def evaluate_models(models, rows):
    print("在 Fold 4 评价固定概率及配对差异", flush=True)
    game_sizes = Counter(row.game_id for row in rows)
    games = tuple(sorted(game_sizes))
    draws = bootstrap_indices(games, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
    weights = probability_map(rows, game_weights(rows))
    groups = {f"phase/{phase}": [r for r in rows if r.phase == phase] for phase in SPEECH_PHASES}
    groups.update({f"day/{day}": [r for r in rows if r.day == day] for day in range(1, 6)})
    groups.update({f"alive_wolves/{n}": [r for r in rows if r.alive_wolves == n] for n in (1, 2)})
    groups["report/successful_empty"] = [r for r in rows if r.successful_empty]
    groups["report/nonempty"] = [r for r in rows if not r.successful_empty]
    evaluation = {"raw": {}, "calibrated": {}}
    for model, fitted in models.items():
        z = logits(standardize(design(rows, model), fitted["standardization"]), fitted["base"])
        for name, values in (("raw", sigmoid(z)), ("calibrated", sigmoid(logits(z[:, None], fitted["calibrator"])))):
            p = probability_map(rows, values)
            evaluation[name][model] = {**score_predictions(rows, p),
                                       "calibration": reliability(rows, p, weights),
                                       "subgroups": grouped_scores(groups, p, game_sizes)}
    comparisons = {version: {f"{second}-{first}": paired_intervals(scores[first], scores[second], games, draws)
                             for first, second in (("M0", "M1"), ("M1", "M2"))}
                   for version, scores in evaluation.items()}
    return evaluation, comparisons


def main():
    started = time.monotonic()
    if os.path.lexists(DESTINATION) or DESTINATION.parent.resolve(strict=True) != DESTINATION.parent:
        raise ValueError("a new canonical result directory is required")
    source = {"git_head": subprocess.check_output(
        ["git", "--no-optional-locks", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": sha256_bytes(Path(__file__).read_bytes())}
    partitions, fold_games, fold_digest, overall = load_development_samples()
    print(json.dumps({"stage": "population_verified", **overall}), flush=True)
    models = fit_models({fold: partitions[fold] for fold in TRAIN_FOLDS}, partitions[CAL_FOLD])
    evaluation, comparisons = evaluate_models(models, partitions[EVAL_FOLD])
    lineage = {"publication_digest": PUBLICATION_DIGEST, "split_digest": SPLIT_DIGEST,
               "development_fold_manifest_digest": fold_digest, "q_manifest_digest": Q_DIGEST,
               "predictions_sha256": Q_SHA256, "source": source}
    results = {
        "scope": "development_internal_fixed_final_tom", "target": "acting_wolf_in_raw_suspicion_support",
        "input_population": overall,
        "folds": {str(fold): {"game_ids": games, **population(partitions[fold])} for fold, games in fold_games.items()},
        "protocol": {"train_folds": TRAIN_FOLDS, "calibration_fold": CAL_FOLD, "evaluation_fold": EVAL_FOLD,
                     "lambda_grid": LAMBDA_GRID, "selection": "three_fold_game_macro_brier_tie_larger_lambda",
                     "sample_weight": "1/(fitting_game_count*game_observer_row_count)",
                     "logistic": LOGISTIC_OPTIONS, "calibration_lambda": CAL_LAMBDA,
                     "primary_predictions": "calibrated", "score_log_loss_epsilon": 1e-12,
                     "feature_scaling": "train_game_weighted_continuous_only_zero_variance_z0",
                     "categorical_references": {"day": 1, "phase": "discussion", "alive_players": 7, "alive_wolves": 1},
                     "bootstrap": {"version": BOOTSTRAP_VERSION, "unit": "game", "seed": BOOTSTRAP_SEED,
                                   "replicates": BOOTSTRAP_REPLICATES, "scope": "fixed_predictions"}},
        "models": models, "evaluation": evaluation, "comparisons": comparisons, **lineage,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__,
                    "wall_seconds_before_publish": time.monotonic() - started,
                    "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        "limitations": ["explored Development; Fold 4 internal evaluation only",
                        "bootstrap holds preprocessing, fitting and calibration fixed",
                        "two prespecified panel summaries; no causal action or risk certification claim"],
    }
    contents = canonical_json_bytes(results)
    print("原子发布并回读结果", flush=True)
    artifact = publish_artifact(DESTINATION, manifest_fields={
        "artifact_type": RESULT_TYPE, "schema_version": RESULT_TYPE, "artifact_identity": DESTINATION.name,
        **lineage,
    }, files={"results.json": contents})
    verified = open_artifact_envelope(DESTINATION, expected_artifact_type=RESULT_TYPE, expected_schema_version=RESULT_TYPE)
    if artifact.manifest_digest != verified.manifest_digest or read_artifact_file(verified, "results.json") != contents:
        raise ValueError("result artifact readback differs")
    print(json.dumps({"status": "PROBABILITY_FIT_COMPLETE", "path": str(DESTINATION),
                      "manifest_digest": verified.manifest_digest, "results_sha256": sha256_bytes(contents),
                      "wall_seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    main()
