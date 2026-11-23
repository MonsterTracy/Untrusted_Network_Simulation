"""用封存概率参数评价新报告事件的固定损失三支决策。"""

from collections import Counter
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from scripts import fit_observer_dtrs_report_event as fit  # noqa: E402
from werewolf.artifact_io import (  # noqa: E402
    canonical_json_bytes, open_artifact_envelope, publish_artifact, read_artifact_file, sha256_bytes,
)
from werewolf.observer_dtrs.decision import (  # noqa: E402
    LossSpec, Region, TiePolicy, conditional_risks, forced_binary, thresholds,
)
from werewolf.observer_dtrs.decision_offline import ProbabilityInput, evaluate, realized_loss  # noqa: E402
from werewolf.observer_dtrs.scoring import score_predictions  # noqa: E402
from werewolf.tom.protocol import BOOTSTRAP_VERSION, bootstrap_indices  # noqa: E402


FIT_PATH = fit.DATA / "dtrs/report-event-probability"
FIT_DIGEST = "9a7e9b3eec31add91075d3c7b79ad84fbbaead8b1b46d23120e6df57dbd3777f"
FIT_RESULTS_SHA = "6eeb5bd9bf51d380fbde38b5bd8fdccc5b198a3a959d08865779e1fa61343988"
FIT_SOURCE_SHA = "62fed1de674fd1bdd59075b8607d3321789d30936ef06aeb051658976eb13129"
FIT_TEST_SHA = "0f73a273c3bef8312acd26e232bd159f4d0d4ea11525ea0b40bc7bb8efef64e6"
DESTINATION = fit.DATA / "dtrs/report-event-three-way"
RESULT_TYPE = "observer_dtrs_report_event_decision"
FOLD4 = {"games": 960, "pres": 3789, "rows": 14723}
MODELS = ("M1", "M2")
RULES = ("three_way", "forced_binary")
WEIGHTINGS = ("game_equal", "record_equal")
REGIONS = tuple(Region)
REGION_INDEX = {region: i for i, region in enumerate(REGIONS)}
ERROR_LABEL = {Region.POS: 0, Region.NEG: 1}
LOSS = LossSpec(Fraction(1, 2), Fraction(1, 2), Fraction(1, 8), Fraction(1, 8), TiePolicy.DEFER_FIRST)
BINARY_TIE_POLICY = TiePolicy.TERMINAL_NEG_FIRST
LOSS_MATRIX = np.asarray([[float(realized_loss(region, y, LOSS)) for y in (0, 1)] for region in REGIONS])
# JSON 浮点可往返还原。固定此绝对容差以容纳 float64 矩阵乘法的舍入差。
SCORE_ATOL = 1e-12
MIN_VALID_BOOTSTRAP = 0.95
METRICS = ("realized_risk", "decided_coverage", "POS_error_rate", "NEG_error_rate", "selective_error_rate",
           "false_positive_joint", "false_negative_joint", "false_positive_rate_given_Y0",
           "false_negative_rate_given_Y1")


def open_fitted_models():
    artifact = open_artifact_envelope(FIT_PATH, expected_artifact_type=fit.RESULT_TYPE,
                                     expected_schema_version=fit.RESULT_TYPE)
    raw = read_artifact_file(artifact, "results.json")
    if artifact.manifest_digest != FIT_DIGEST or sha256_bytes(raw) != FIT_RESULTS_SHA:
        raise ValueError("unexpected probability result identity")
    saved = json.loads(raw)
    if (saved["publication_digest"], saved["split_digest"], saved["q_manifest_digest"], saved["predictions_sha256"]) != (
            fit.PUBLICATION_DIGEST, fit.SPLIT_DIGEST, fit.Q_DIGEST, fit.Q_SHA256):
        raise ValueError("probability result differs from approved Development and Q sources")
    if (saved["source"]["script_sha256"] != FIT_SOURCE_SHA
            or sha256_bytes(Path(fit.__file__).read_bytes()) != FIT_SOURCE_SHA
            or sha256_bytes((ROOT / "tests/observer_dtrs/test_report_event_fit.py").read_bytes()) != FIT_TEST_SHA):
        raise ValueError("original fitting source bytes differ")
    return saved


def load_fold4(saved):
    held = saved["folds"]["4"]
    games = set(held["game_ids"])
    if len(games) != len(held["game_ids"]) or len(games) != FOLD4["games"]:
        raise ValueError("invalid saved Fold 4 membership")
    publication = open_artifact_envelope(fit.DEVELOPMENT, expected_artifact_type=fit.PUBLICATION_TYPE,
                                         expected_schema_version=fit.PUBLICATION_VERSION)
    if publication.manifest_digest != fit.PUBLICATION_DIGEST:
        raise ValueError("Development Publication identity differs")
    # 保持拟合时的游戏遍历顺序，只打开已登记的 Fold 4 游戏。
    game_ids = tuple(g["game_id"] for g in publication.manifest["development_games"] if g["game_id"] in games)
    archive = open_artifact_envelope(fit.Q_PATH, expected_artifact_type=fit.Q_TYPE, expected_schema_version=fit.Q_TYPE)
    if (archive.manifest_digest != fit.Q_DIGEST or tuple(archive.manifest["row_and_column_seat_order"]) != fit.PLAYER_IDS):
        raise ValueError("Q provenance or absolute seat order differs")
    by_pre = {}
    for line in read_artifact_file(archive, "predictions.jsonl").splitlines():
        record = json.loads(line)
        if record["game_id"] in games:
            key = record["game_id"], record["boundary_id"]
            if key in by_pre:
                raise ValueError(f"duplicate Fold 4 Q at {key}")
            by_pre[key] = record

    rows, roles = [], None
    for offset in range(0, len(game_ids), 120):
        batch_ids = game_ids[offset:offset + 120]
        batch = fit.open_development_input(fit.DEVELOPMENT, expected_publication_digest=fit.PUBLICATION_DIGEST,
                                           expected_split_digest=fit.SPLIT_DIGEST, game_ids=batch_ids)
        if roles is None:
            folds = batch.public_view.fold_manifest
            fold4 = next(f for f in folds.folds if f.fold_index == 4)
            if set(fold4.game_ids) != games or folds.manifest_digest != saved["development_fold_manifest_digest"]:
                raise ValueError("verified Fold 4 differs from the probability evaluation")
            roles = {(g.game_id, g.bundle_digest): tuple(p for p, r in g.role_assignment if r == "Werewolf")
                     for g in fit.open_role_sidecar(batch).games if g.game_id in games}
        if tuple(g.game_id for g in batch.public_view.games) != batch_ids:
            raise ValueError("Development batch differs from Fold 4 membership")
        for game in batch.public_view.games:
            wolves = roles[game.game_id, game.bundle_digest]
            observations = {o.observation_id: o for o in game.belief_observations}
            for prefix in game.authoritative_pre_prefixes:
                if prefix.current_speaker not in wolves or prefix.public_temporal_state.phase.value not in fit.SPEECH_PHASES:
                    continue
                key = game.game_id, prefix.boundary_id
                # PRE、观察者、Q 和新标签的联结由原脚本验证。
                rows.extend(fit.pre_samples(prefix, wolves, observations, by_pre.pop(key)))
        if offset == 0 or offset + len(batch_ids) == len(game_ids):
            print(f"Fold 4 联结 {offset + len(batch_ids)}/{len(game_ids)} 局，{len(rows)} 条记录", flush=True)
    population = fit.population(rows)
    if (by_pre or {r.game_id for r in rows} != games
            or any(population[k] != expected or held[k] != expected for k, expected in FOLD4.items())):
        raise ValueError(f"Fold 4 population mismatch: {population}; unused Q={len(by_pre)}")
    return rows, population


def rebuild_probabilities(rows, model, parameters):
    if tuple(parameters["features"]) != fit.FEATURES[model]:
        raise ValueError(f"feature order differs for {model}")
    x = fit.standardize(fit.design(rows, model), parameters["standardization"])
    z = fit.logits(x, parameters["base"])
    return fit.probability_map(rows, fit.sigmoid(fit.logits(z[:, None], parameters["calibrator"])))


def verify_scores(rows, model, probabilities, archived):
    actual = score_predictions(rows, probabilities)
    sizes = {g: s["rows"] for g, s in actual["by_game"].items()}
    if sizes != {g: s["rows"] for g, s in archived["by_game"].items()}:
        raise ValueError(f"{model} archived by-game identities or record counts differ")
    reconstructed = {w: {m: actual[w][m] for m in fit.LOSS_NAMES} for w in WEIGHTINGS}
    differences = {w: {m: actual[w][m] - archived[w][m] for m in fit.LOSS_NAMES} for w in WEIGHTINGS}
    for w in WEIGHTINGS:
        for m, delta in differences[w].items():
            if not math.isclose(delta, 0.0, rel_tol=0.0, abs_tol=SCORE_ATOL):
                raise ValueError(f"{model} {w} {m} does not reproduce: archived={archived[w][m]!r}, rebuilt={actual[w][m]!r}")
    return {"scores": reconstructed, "differences": differences, "games_matched": len(sizes), "rows_matched": sum(sizes.values())}


def game_totals(game_index, region_index, y, predicted, num_games):
    """补充现有评价器未返回的逐局区域类别计数和二支损失分子。"""
    counts = np.zeros((num_games, len(REGIONS), 2), dtype=np.int64)
    np.add.at(counts, (game_index, region_index, y), 1)
    return {"counts": counts, "realized_sum": (counts * LOSS_MATRIX).sum(axis=(1, 2)),
            "predicted_sum": np.bincount(game_index, weights=predicted, minlength=num_games)}


def summarize(evaluated, totals, weights):
    counts = totals["counts"]
    mass = weights @ counts.sum(axis=(1, 2))
    summary = {name: evaluated[name] for name in METRICS}
    summary["predicted_risk"] = float(weights @ totals["predicted_sum"] / mass) if mass else None
    summary["joint_error_mass"] = (summary["false_positive_joint"] + summary["false_negative_joint"]
                                   if mass else None)
    summary["regions"] = {}
    for j, region in enumerate(REGIONS):
        cell = counts[:, j, :]
        summary["regions"][region.value] = {
            **evaluated["regions"][region.value], "positive": int(cell[:, 1].sum()), "negative": int(cell[:, 0].sum()),
            "error_games": int(np.count_nonzero(cell[:, ERROR_LABEL[region]])) if region in ERROR_LABEL else None,
        }
    return summary


def bootstrap_ratio(numerator, denominator, weights, draws):
    num = (weights * numerator)[draws].sum(axis=1)
    den = (weights * denominator)[draws].sum(axis=1)
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)


def interval(values):
    finite = values[np.isfinite(values)]
    return {"percentile_95": np.quantile(finite, [0.025, 0.975]).tolist()
            if len(finite) >= MIN_VALID_BOOTSTRAP * len(values) and len(finite) else None,
            "valid_replicates": len(finite), "replicates": len(values)}


def evaluate_model(rows, probabilities, games, game_sizes, groups, draws):
    ids = [canonical_json_bytes(row.sample_id).decode() for row in rows]
    inputs = tuple(ProbabilityInput(sid, row.game_id, probabilities[row.sample_id]) for sid, row in zip(ids, rows, strict=True))
    targets = {sid: row.y for sid, row in zip(ids, rows, strict=True)}
    game_position = {game: j for j, game in enumerate(games)}
    game_index = np.asarray([game_position[row.game_id] for row in rows])
    y = np.asarray([row.y for row in rows])
    n = np.asarray([game_sizes[g] for g in games], dtype=np.float64)
    weights = {"game_equal": 1 / n, "record_equal": np.ones_like(n)}

    def evaluate_at(weighting, subgroup_ids=None):
        return evaluate(inputs, targets, loss=LOSS, game_sizes=game_sizes, weighting=weighting,
                        binary_tie_policy=BINARY_TIE_POLICY, subgroup_ids=subgroup_ids)

    reports = {w: evaluate_at(w) for w in WEIGHTINGS}
    decisions = {d["sample_id"]: d for d in reports["game_equal"]["decisions"]}
    three_actions = [Region(d["region"]) for d in (decisions[sid] for sid in ids)]
    binary_actions = [forced_binary(row.probability, LOSS, tie_policy=BINARY_TIE_POLICY) for row in inputs]
    chosen = {"three_way": three_actions, "forced_binary": binary_actions}
    predictions = {"three_way": [decisions[sid]["risks"][a.value] for sid, a in zip(ids, three_actions, strict=True)],
                   "forced_binary": [float(conditional_risks(row.probability, LOSS)[a])
                                     for row, a in zip(inputs, binary_actions, strict=True)]}
    indices = {rule: np.asarray([REGION_INDEX[a] for a in chosen[rule]]) for rule in RULES}
    predictions = {rule: np.asarray(value) for rule, value in predictions.items()}

    def totals_at(mask):
        return {rule: game_totals(game_index[mask], indices[rule][mask], y[mask], predictions[rule][mask], len(games))
                for rule in RULES}

    def summarize_at(report, totals, weighting):
        return {rule: summarize(report if rule == "three_way" else report["forced_binary"], totals[rule], weights[weighting])
                for rule in RULES}

    whole = totals_at(slice(None))
    overall = {w: summarize_at(reports[w], whole, w) for w in WEIGHTINGS}
    for w in WEIGHTINGS:
        for rule in RULES:
            counts = whole[rule]["counts"]
            error_intervals = {}
            for region, label in ERROR_LABEL.items():
                cell = counts[:, REGION_INDEX[region], :]
                error_intervals[region.value] = interval(bootstrap_ratio(cell[:, label], cell.sum(axis=1), weights[w], draws))
            overall[w][rule]["error_intervals"] = error_intervals

    subgroups = {}
    for name, mask in groups.items():
        group_ids = [sid for sid, keep in zip(ids, mask, strict=True) if keep]
        totals = totals_at(mask)
        subgroups[name] = {w: summarize_at(evaluate_at(w, group_ids), totals, w) for w in WEIGHTINGS}
    return {"overall": overall, "subgroups": subgroups}, whole


def paired_losses(overall, totals, n, draws):
    pairs = (("M1_three_minus_binary", ("M1", "three_way"), ("M1", "forced_binary")),
             ("M2_three_minus_binary", ("M2", "three_way"), ("M2", "forced_binary")),
             ("M2_minus_M1_three_way", ("M2", "three_way"), ("M1", "three_way")))
    result = {}
    for weighting, weights in (("game_equal", 1 / n), ("record_equal", np.ones_like(n))):
        result[weighting] = {}
        for name, (first, rule1), (second, rule2) in pairs:
            delta = totals[first][rule1]["realized_sum"] - totals[second][rule2]["realized_sum"]
            point = overall[first][weighting][rule1]["realized_risk"] - overall[second][weighting][rule2]["realized_risk"]
            result[weighting][name] = {"difference": point, **interval(bootstrap_ratio(delta, n, weights, draws))}
    return result


def main():
    started = time.monotonic()
    if os.path.lexists(DESTINATION) or DESTINATION.parent.resolve(strict=True) != DESTINATION.parent:
        raise ValueError("a new canonical result directory is required")
    saved = open_fitted_models()
    rows, population = load_fold4(saved)
    probabilities, reproduction = {}, {}
    for model in MODELS:
        probabilities[model] = rebuild_probabilities(rows, model, saved["models"][model])
        reproduction[model] = verify_scores(rows, model, probabilities[model], saved["evaluation"]["calibrated"][model])
    print(json.dumps({"stage": "fold4_scores_reproduced", "models": reproduction}), flush=True)

    sizes = Counter(row.game_id for row in rows)
    games = tuple(sorted(sizes))
    n = np.asarray([sizes[g] for g in games], dtype=np.float64)
    draws = bootstrap_indices(games, fit.BOOTSTRAP_REPLICATES, fit.BOOTSTRAP_SEED)
    groups = {f"{phase}/{wolves}": np.asarray([r.phase == phase and r.alive_wolves == wolves for r in rows])
              for phase in fit.SPEECH_PHASES for wolves in (1, 2)}
    evaluations, totals = {}, {}
    for model in MODELS:
        print(f"评价 {model} 三支、强制二支及四个预定子组", flush=True)
        evaluations[model], totals[model] = evaluate_model(rows, probabilities[model], games, sizes, groups, draws)
    paired = paired_losses({m: value["overall"] for m, value in evaluations.items()}, totals, n, draws)
    by_game = {model: {rule: {key: value.tolist() for key, value in t.items()} for rule, t in by_rule.items()}
               for model, by_rule in totals.items()}
    source = {"git_head": subprocess.check_output(
        ["git", "--no-optional-locks", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
              "analysis_sha256": sha256_bytes(Path(__file__).read_bytes()), "original_fit_sha256": FIT_SOURCE_SHA}
    lineage = {"publication_digest": fit.PUBLICATION_DIGEST, "split_digest": fit.SPLIT_DIGEST,
               "development_fold_manifest_digest": saved["development_fold_manifest_digest"],
               "q_manifest_digest": fit.Q_DIGEST, "probability_fit_manifest_digest": FIT_DIGEST,
               "probability_fit_results_sha256": FIT_RESULTS_SHA, "source": source}
    result = {
        "scope": "development_fold4_descriptive_fixed_loss", "target": saved["target"],
        "population": population, "subgroup_population": {name: fit.support([r for r, keep in zip(rows, mask, strict=True) if keep])
                                                           for name, mask in groups.items()},
        "probability_parameters": {"artifact": str(FIT_PATH), "reference": "models/{M1,M2}", "variant": "calibrated"},
        "score_reproduction": {"absolute_tolerance": SCORE_ATOL, "models": reproduction},
        "loss": LOSS.to_record(), "thresholds": thresholds(LOSS).to_record(), "binary_tie_policy": BINARY_TIE_POLICY.value,
        "primary_model": "M1", "comparison_model": "M2", "primary_weighting": "game_equal",
        "evaluation": evaluations, "paired_loss_differences": paired,
        "bootstrap": {"version": BOOTSTRAP_VERSION, "seed": fit.BOOTSTRAP_SEED, "replicates": fit.BOOTSTRAP_REPLICATES,
                      "minimum_valid_fraction_for_interval": MIN_VALID_BOOTSTRAP, "unit": "game",
                      "scope": "fixed_predictions_and_rules"},
        "by_game": {"game_ids": games, "rows": [sizes[g] for g in games], "region_order": [r.value for r in REGIONS],
                    "label_order": [0, 1], "models": by_game},
        "limitations": ["Fold 4 already used for development probability comparison and model choice",
                        "normative cognitive costs; no effect on actions or future information estimated",
                        "descriptive percentile intervals; no independent risk certification"],
        "wall_seconds_before_publish": time.monotonic() - started, **lineage,
    }
    payload = canonical_json_bytes(result)
    artifact = publish_artifact(DESTINATION, manifest_fields={"artifact_type": RESULT_TYPE, "schema_version": RESULT_TYPE,
                                                            "artifact_identity": DESTINATION.name, **lineage},
                                files={"results.json": payload})
    readback = open_artifact_envelope(DESTINATION, expected_artifact_type=RESULT_TYPE, expected_schema_version=RESULT_TYPE)
    if readback.manifest_digest != artifact.manifest_digest or read_artifact_file(readback, "results.json") != payload:
        raise ValueError("published decision results failed readback")
    print(json.dumps({"status": "DTRS_DEVELOPMENT_ANALYSIS_COMPLETE", "path": str(DESTINATION),
                      "manifest_digest": readback.manifest_digest, "results_sha256": sha256_bytes(payload),
                      "wall_seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    main()
