"""冻结双层报告风险规则并在 Development Fold 4 评价。"""

import argparse
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from scripts import analyze_observer_dtrs_report_event as report_event  # noqa: E402
from scripts import analyze_observer_dtrs_sequential as sequential  # noqa: E402
from scripts import analyze_observer_dtrs_team_event as development  # noqa: E402
from scripts import fit_observer_dtrs_report_state as probability_study  # noqa: E402
from werewolf.artifact_io import (  # noqa: E402
    canonical_json_bytes, open_artifact_envelope, publish_artifact, read_artifact_file, sha256_bytes,
)
from werewolf.observer_dtrs import report_state_decision as decision  # noqa: E402
from werewolf.observer_dtrs import report_state_probability as probability  # noqa: E402
from werewolf.observer_dtrs.report_state_decision import BND, NEG, POS, Thresholds  # noqa: E402
from werewolf.tom.protocol import BOOTSTRAP_VERSION, bootstrap_indices  # noqa: E402


DESTINATION = development.frozen.DATA / "dtrs/report-state-risk"
BOOTSTRAP_REPLICATES = probability_study.BOOTSTRAP_REPLICATES
BUDGETS = {
    "strict": {POS: "0.05", NEG: "0.05"},
    decision.MAIN_CONFIGURATION: {POS: "0.10", NEG: "0.10"},
    "moderate": {POS: "0.15", NEG: "0.15"},
    "loose": {POS: "0.20", NEG: "0.20"},
    "historical": {POS: "0.10", NEG: "0.15"},
    "reverse": {POS: "0.15", NEG: "0.10"},
}
DOUBLE_ONLY = ("shared_double", "constrained_double")


@dataclass(frozen=True, slots=True)
class Prediction:
    sample: probability_study.Sample
    ps: float
    pT: float

    @property
    def game_id(self):
        return self.sample.game_id

    @property
    def wolf_count(self):
        return self.sample.context.wolf_count


def match_oof(partitions, records, outer_selection):
    """严格按游戏尝试、PRE、观察者和外层折绑定已封存的概率。"""
    folds = development.SELECTION_FOLDS
    expected = {row.sample_id: (fold, row) for fold in folds for row in partitions[fold]}
    if len(expected) != sum(len(partitions[fold]) for fold in folds):
        raise ValueError("开发选择人口包含重复观察者 PRE")
    selections = {}
    for fold in folds:
        entry = outer_selection[str(fold)]
        method = entry["selected_method"]
        if tuple(entry["selection_folds"]) != tuple(other for other in folds if other != fold):
            raise ValueError(f"Fold {fold} 的折外选择来源不匹配")
        selections[fold] = {"method": method, "lambda": entry["best_lambda"][method]}

    joined = []
    for item in records:
        key = tuple(item[name] for name in ("game_id", "boundary_id", "observer_id"))
        if key not in expected:
            raise ValueError(f"折外预测重复或不属于开发选择人口：{key}")
        fold, sample = expected.pop(key)
        ctx = sample.context
        identity = (item["fold"], item["attempt_id"], item["prefix_digest"], item["step_index"],
                    item["acting_wolf"], tuple(item["alive_wolf_ids"]), item["method"], item["lambda"])
        actual = (fold, sample.attempt_id, ctx.prefix_digest, ctx.step_index, ctx.acting_wolf,
                  ctx.alive_wolf_ids, selections[fold]["method"], selections[fold]["lambda"])
        if identity != actual:
            raise ValueError(f"折外概率与 PRE 或外层拟合来源不匹配：{key}")
        joined.append(Prediction(sample, *decision.event_probabilities(item, ctx.wolf_count)))
    if expected:
        raise ValueError(f"折外概率缺少 {len(expected)} 条合法记录，首条为 {next(iter(expected))}")
    return joined, selections


def frozen_predictions(samples, model):
    features = np.asarray([sample.features for sample in samples])
    joint = probability.probabilities(probability.log_probabilities(model, features), features)
    return [Prediction(sample, *decision.event_probabilities(probability.probability_record(q), sample.context.wolf_count))
            for sample, q in zip(samples, joint, strict=True)]


def _supervised(predictions, *, double_only=False):
    return [row for row in predictions if row.sample.state is not None and (not double_only or row.wolf_count == 2)]


@dataclass(frozen=True, slots=True)
class Cut:
    threshold: float
    mass: tuple[int, int]
    errors: tuple[int, int]


def direction_cuts(rows):
    """将两层实际概率的并集排序，每个同分块整体进入对应方向。"""
    blocks = {NEG: {0.: [0, 0, 0, 0]}, POS: {1.: [0, 0, 0, 0]}}
    for row, weight in zip(rows, development.game_weight_units(rows), strict=True):
        state = row.sample.state
        for i, (p, hit) in enumerate(((row.ps, state == 2), (row.pT, state != 0))):
            if p == .5:
                continue
            direction = NEG if p < .5 else POS
            block = blocks[direction].setdefault(p, [0, 0, 0, 0])
            block[i] += weight
            block[2 + i] += weight * (hit if direction == NEG else not hit)

    cuts = {NEG: [], POS: []}
    for direction, scores in blocks.items():
        mass_s = mass_T = errors_s = errors_T = 0
        for threshold in sorted(scores, reverse=direction == POS):
            ms, mt, es, et = scores[threshold]
            mass_s, mass_T = mass_s + ms, mass_T + mt
            errors_s, errors_T = errors_s + es, errors_T + et
            cuts[direction].append(Cut(threshold, (mass_s, mass_T), (errors_s, errors_T)))
    return cuts


def choose_direction(cuts, direction, budget, *, shared):
    """共享阈值直接扫描，四阈值维护团队阈值不大于自身阈值的最优前缀。"""
    # 按阈值数值递增，正向切点的累计构造顺序恰好相反。
    ordered = cuts if direction == NEG else reversed(cuts)
    strictness = 1 if direction == NEG else -1
    prefix = best = best_key = None
    support = [0, 0]
    feasible = [0, 0]
    for cut in ordered:
        s_ok, T_ok = (development._risk_met(mass, errors, budget)
                      for mass, errors in zip(cut.mass, cut.errors, strict=True))
        for i, ok in enumerate((s_ok, T_ok)):
            support[i] += cut.mass[i] > 0
            feasible[i] += ok

        if shared:
            team = cut if T_ok else None
        else:
            if T_ok and (prefix is None or (-cut.mass[1], strictness * cut.threshold)
                         < (-prefix.mass[1], strictness * prefix.threshold)):
                prefix = cut
            team = prefix
        if s_ok and team is not None:
            key = (-cut.mass[0] - team.mass[1], strictness * cut.threshold, strictness * team.threshold)
            if best_key is None or key < best_key:
                best, best_key = (cut.threshold, team.threshold), key

    failures = {event: "no_positive_support" if not n_support else "risk_budget_unmet"
                for event, n_support, n_feasible in zip(decision.EVENTS, support, feasible, strict=True)
                if not n_feasible}
    reason = None
    if best is None:
        reason = ("event_infeasible" if failures else
                  "no_common_feasible_threshold" if shared else "threshold_order_infeasible")
    return {"thresholds": best, "reason": reason, "event_failures": failures,
            "supported_cuts": dict(zip(decision.EVENTS, support, strict=True)),
            "feasible_cuts": dict(zip(decision.EVENTS, feasible, strict=True))}


def select_thresholds(cuts, budgets, *, shared):
    directions = {d: choose_direction(cuts[d], d, Fraction(budgets[d]), shared=shared) for d in (NEG, POS)}
    betas, alphas = (directions[d]["thresholds"] for d in (NEG, POS))
    thresholds = Thresholds(betas[0], alphas[0], betas[1], alphas[1]).to_record() if betas and alphas else None
    return {"feasible": thresholds is not None, "thresholds": thresholds, "directions": directions}


def select_configurations(oof):
    all_cuts = direction_cuts(_supervised(oof))
    double_cuts = direction_cuts(_supervised(oof, double_only=True))
    rules = {}
    for name, budgets in BUDGETS.items():
        beta, alpha = float(Fraction(budgets[NEG])), float(1 - Fraction(budgets[POS]))
        rules[name] = {
            "budgets": budgets,
            "posterior": {"thresholds": Thresholds.shared(beta, alpha).to_record()},
            "shared": select_thresholds(all_cuts, budgets, shared=True),
            "shared_double": select_thresholds(double_cuts, budgets, shared=True),
            "constrained_double": select_thresholds(double_cuts, budgets, shared=False),
        }
    return rules


class Population:
    """在所研究的成功报告人口固定整局权重，所有子组均沿用基础权重。"""

    def __init__(self, rows):
        self.rows = rows
        self.units = development.game_weight_units(rows)
        self.total_units = sum(self.units)
        self.games = tuple(sorted({row.game_id for row in rows}))
        index = {game: i for i, game in enumerate(self.games)}
        self.row_games = np.asarray([index[row.game_id] for row in rows], dtype=int)
        self.states = np.asarray([row.sample.state for row in rows], dtype=int)
        self.wolf_count = np.asarray([row.wolf_count for row in rows], dtype=int)
        self.all = np.ones(len(rows), dtype=bool)
        self.game_weights = 1 / self.game_counts(self.all)

    def game_counts(self, mask):
        return np.bincount(self.row_games[mask], minlength=len(self.games))

    def weight(self, mask):
        return sum(self.units[i] for i in np.flatnonzero(mask))

    def support(self, mask):
        counts = self.game_counts(mask)
        mass = self.weight(mask)
        return {"records": int(counts.sum()), "games": int(np.count_nonzero(counts)), "weight_units": mass,
                "game_equal_mass": development._ratio(mass, self.total_units)}

    def rate(self, numerator, denominator):
        num, den = self.support(numerator), self.support(denominator)
        return {"value": development._ratio(num["weight_units"], den["weight_units"]),
                "numerator": num, "denominator": den}

    def interval(self, numerator_by_game, denominator_by_game, draws):
        values = report_event.bootstrap_ratio(numerator_by_game, denominator_by_game, self.game_weights, draws)
        ci = report_event.interval(values)
        return {**ci, "zero_denominator_replicates": ci["replicates"] - ci["valid_replicates"]}


def _regions(rows, thresholds):
    if thresholds is None:
        # Binary 对两个事件使用同一阈值，0.5 的并列固定判为 NEG。
        regions = [(POS if row.ps > .5 else NEG, POS if row.pT > .5 else NEG) for row in rows]
    else:
        regions = [thresholds.regions(row.ps, row.pT) for row in rows]
    return np.asarray(regions, dtype="U3").reshape(-1, 2)


def evaluate_rule(pop, thresholds, budgets, *, subgroups, draws=None):
    if thresholds is not None and not thresholds.is_shared and (pop.wolf_count != 2).any():
        raise ValueError("一致性四阈值对照只评价双狼人口")
    regions = _regions(pop.rows, thresholds)
    truth = (pop.states == 2, pop.states != 0)
    paired = np.char.add(regions[:, 0].astype("U1"), regions[:, 1].astype("U1"))
    if not set(paired) <= set(decision.JOINT_REGIONS):
        raise ValueError("双层判断出现不一致的联合区域")
    groups = {"overall": pop.all}
    if subgroups:
        groups.update(single_wolf=pop.wolf_count == 1, two_wolves=pop.wolf_count == 2)

    result = {}
    for group, mask in groups.items():
        population = pop.support(mask)
        units = population["weight_units"]
        events = {}
        for i, event in enumerate(decision.EVENTS):
            event_regions = {}
            for region in decision.REGIONS:
                selected = mask & (regions[:, i] == region)
                support = pop.support(selected)
                entry = event_regions[region] = {**support, "coverage": development._ratio(support["weight_units"], units)}
                if region != BND:
                    errors = selected & (truth[i] == (region == NEG))
                    error_support = pop.support(errors)
                    n, d = error_support["weight_units"], support["weight_units"]
                    entry.update(risk=development._ratio(n, d), error_support=error_support,
                                 empirical_budget_met=development._risk_met(d, n, Fraction(budgets[region])))
                    if draws is not None:
                        entry["risk_bootstrap"] = pop.interval(pop.game_counts(errors), pop.game_counts(selected), draws)
            coverage_units = event_regions[NEG]["weight_units"] + event_regions[POS]["weight_units"]
            events[event] = {"coverage": development._ratio(coverage_units, units), "regions": event_regions}

        joint = {}
        for region in decision.JOINT_REGIONS:
            support = pop.support(mask & (paired == region))
            joint[region] = {**support, "coverage": development._ratio(support["weight_units"], units)}
        np_mask = mask & (paired == "NP")
        np_hit = np_mask & (pop.states == 1)
        np_error = np_mask & ~np_hit
        e1 = pop.rate(np_hit, np_mask)
        np_risk = pop.rate(np_error, np_mask)
        covered = [mask & (regions[:, i] != BND) for i in range(2)]
        summary = result[group] = {
            "population": population, "events": events, "joint_regions": joint,
            "mean_coverage": development._ratio(sum(pop.weight(x) for x in covered), 2 * units),
            "all_four_budgets_met": all(events[event]["regions"][d]["empirical_budget_met"]
                                        for event in decision.EVENTS for d in (NEG, POS)),
            "np_e1_rate": e1, "np_error_risk": np_risk,
        }
        if draws is not None:
            summary["mean_coverage_bootstrap"] = pop.interval(sum(map(pop.game_counts, covered)),
                                                              2 * pop.game_counts(mask), draws)
            np_risk["bootstrap"] = pop.interval(pop.game_counts(np_error), pop.game_counts(np_mask), draws)
    return result


def evaluate_configurations(rules, oof, fold4):
    def populations(rows):
        return {"all": Population(_supervised(rows)), "double": Population(_supervised(rows, double_only=True))}

    selection, evaluation = populations(oof), populations(fold4)
    draws = {}
    for pop in evaluation.values():
        if pop.games and pop.games not in draws:
            draws[pop.games] = bootstrap_indices(pop.games, BOOTSTRAP_REPLICATES, development.frozen.BOOTSTRAP_SEED)

    def report(thresholds, budgets, *, double_only, bootstrap):
        scope = "double" if double_only else "all"
        train, test = selection[scope], evaluation[scope]
        return {"oof": evaluate_rule(train, thresholds, budgets, subgroups=not double_only),
                "fold4": evaluate_rule(test, thresholds, budgets, subgroups=not double_only,
                                       draws=draws.get(test.games) if bootstrap else None)}

    binary = {"threshold": .5, "tie": NEG,
              **report(None, BUDGETS[decision.MAIN_CONFIGURATION], double_only=False, bootstrap=True)}
    configurations = {}
    for name, configured in rules.items():
        methods = configurations[name] = {}
        for method, selected in configured.items():
            if method == "budgets":
                continue
            if selected["thresholds"] is None:
                methods[method] = {"status": "infeasible_on_oof", "oof": None, "fold4": None}
                continue
            thresholds = Thresholds(**selected["thresholds"])
            methods[method] = {"status": "evaluated", **report(thresholds, configured["budgets"],
                double_only=method in DOUBLE_ONLY, bootstrap=name == decision.MAIN_CONFIGURATION)}
    return {"binary": binary, "configurations": configurations}


def sequential_evaluation(rows, pairs, thresholds):
    """沿着 E4 的紧邻机会重新定域。报告缺失只影响后续错误率的分母。"""
    predictions = {row.sample.sample_id: row for row in rows}
    opportunities = {sequential.sample_id(pair.start): pair for pair in pairs}
    if len(opportunities) != len(pairs) or set(opportunities) != set(predictions):
        raise ValueError("E4 复判起点与 Fold 4 权威 PRE 人口不一致")
    for key, pair in opportunities.items():
        sample = predictions[key].sample
        if sample.context != pair.start or sample.attempt_id != pair.attempt_id:
            raise ValueError(f"E4 配对与 Fold 4 游戏尝试不一致：{key}")

    pop = Population(_supervised(rows))
    initial = _regions(pop.rows, thresholds)
    selected_pairs = [opportunities[row.sample.sample_id] for row in pop.rows]
    followups = [predictions[sequential.sample_id(pair.followup)] if pair.followup is not None else None
                 for pair in selected_pairs]
    opportunity = np.asarray([row is not None for row in followups], dtype=bool)
    valid = np.asarray([row is not None and row.sample.state is not None for row in followups], dtype=bool)
    changed = np.asarray([pair.followup is not None and pair.start.alive_wolf_ids != pair.followup.alive_wolf_ids
                          for pair in selected_pairs], dtype=bool)
    next_regions = np.asarray([thresholds.regions(row.ps, row.pT) if row is not None else ("", "")
                               for row in followups], dtype="U3").reshape(-1, 2)
    next_states = np.asarray([row.sample.state if keep else -1 for row, keep in zip(followups, valid, strict=True)])
    by_end = Counter(
        (pair.wolf_death is not None, pair.observer_death is not None, pair.terminal is not None)
        for pair in selected_pairs if pair.followup is None
    )
    population = {
        "initial_successful": pop.support(pop.all), "initial_report_unobserved_records": len(rows) - len(pop.rows),
        "next_opportunity": pop.support(opportunity), "next_prediction": pop.support(opportunity),
        "next_successful_report": pop.support(valid), "no_next": pop.support(~opportunity),
        "no_next_by_death_and_terminal": [
            {"wolf_dead": w, "observer_dead": o, "terminal_confirmed": t, "records": count}
            for (w, o, t), count in sorted(by_end.items())
        ],
        "next_report_unobserved": pop.support(opportunity & ~valid),
        "next_report_unobserved_statuses": dict(Counter(row.sample.report_status for row in followups
                                                        if row is not None and row.sample.state is None)),
    }
    groups = {}
    for name, mask in (("same_team", opportunity & ~changed), ("changed_team", opportunity & changed)):
        events = {}
        for i, event in enumerate(decision.EVENTS):
            start, followup = initial[:, i], next_regions[:, i]
            bnd = start == BND
            resolved = mask & bnd & (followup != BND)
            hit = next_states == 2 if i == 0 else next_states != 0
            errors = ((followup == NEG) & hit) | ((followup == POS) & ~hit)
            events[event] = {
                "transitions_given_next_prediction": {
                    a: {b: pop.rate(mask & (start == a) & (followup == b), mask & (start == a))
                        for b in decision.REGIONS} for a in decision.REGIONS},
                "bnd_resolution_given_initial_bnd": pop.rate(resolved, bnd),
                "bnd_resolution_given_next_prediction": pop.rate(resolved, mask & bnd),
                "next_commitment_error_by_direction": {
                    d: pop.rate(mask & valid & (followup == d) & errors, mask & valid & (followup == d))
                    for d in (NEG, POS)},
                "bnd_resolution_commitment_error": pop.rate(resolved & valid & errors, resolved & valid),
            }
        groups[name] = {"next_prediction": pop.support(mask), "next_successful_report": pop.support(mask & valid),
                        "events": events}
    return {"population": population, "by_team_change": groups}


def run_study(partitions, records, outer_selection, model, pairs):
    oof, selections = match_oof(partitions, records, outer_selection)
    rules = select_configurations(oof)
    print("已固定 Fold 0–3 阈值，开始 Fold 4 评价", flush=True)
    held = partitions[development.EVALUATION_FOLD]
    fold4 = frozen_predictions(held, model)
    evaluated = evaluate_configurations(rules, oof, fold4)
    main = rules[decision.MAIN_CONFIGURATION]["shared"]
    sequence = (sequential_evaluation(fold4, pairs, Thresholds(**main["thresholds"]))
                if main["feasible"] else {"status": "main_shared_infeasible"})
    results = {
        "population": {"oof": probability_study.population([row.sample for row in oof]),
                       "fold4": probability_study.population(held),
                       "by_fold": {str(fold): probability_study.population(rows) for fold, rows in partitions.items()}},
        "oof_probability_selections": {str(fold): source for fold, source in selections.items()},
        "fold4_probability_model": {"method": model["method"], "lambda": model["lambda"]},
        "evaluation": evaluated, "sequential_fold4_main_shared": sequence,
    }
    return rules, results


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    if os.path.lexists(DESTINATION):
        raise FileExistsError(f"已有双层风险规则 Artifact：{DESTINATION}")
    source_model = open_artifact_envelope(probability_study.DESTINATION, expected_artifact_type=probability.ARTIFACT_TYPE,
                                         expected_schema_version=probability.ARTIFACT_TYPE)
    previous = json.loads(read_artifact_file(source_model, "results.json"))
    model = probability.load_development_model(probability_study.DESTINATION)
    selection = previous["selection"]
    method = selection["selected_method"]
    if model["method"] != method or model["lambda"] != selection["best_lambda"][method]:
        raise ValueError("冻结概率模型参数与第一轮 Development 选择不一致")

    pairs = []

    def visit_game(fold, game, wolves, *, publication_digest, attempt_id):
        if fold == development.EVALUATION_FOLD:
            _, game_pairs = sequential.pair_game(game, wolves, publication_digest=publication_digest, attempt_id=attempt_id)
            pairs.extend(game_pairs)

    partitions, lineage = development.load_development(development.DEVELOPMENT_FOLDS,
                                                       build_samples=probability_study.pre_samples, visit_game=visit_game)
    if any(source_model.manifest.get(key) != value for key, value in lineage.items()):
        raise ValueError("第一轮概率 Archive 与当前 Development、Fold 或 Q 不匹配")
    records = (json.loads(line) for line in read_artifact_file(source_model, "oof_predictions.jsonl").splitlines())
    rules, results = run_study(partitions, records, previous["outer_selection"], model, pairs)

    source_files = (Path(__file__), Path(decision.__file__), Path(development.__file__))
    source = {"git_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
              "file_sha256": {str(path.resolve().relative_to(ROOT)): sha256_bytes(path.read_bytes()) for path in source_files}}
    identity = {**lineage, "probability_manifest_digest": source_model.manifest_digest, "source": source}
    results.update({**identity, "scope": "development_internal_frozen_report_state_risk",
                    "protocol": {"threshold_folds": development.SELECTION_FOLDS,
                                 "evaluation_fold": development.EVALUATION_FOLD, "budgets": BUDGETS,
                                 "boundaries": "p_le_beta_NEG_p_ge_alpha_POS",
                                 "optimization": "maximize_mean_coverage_subject_to_four_empirical_risk_budgets",
                                 "threshold_ties": "stricter_first_speaker_then_team",
                                 "weighting": "game_equal_within_each_study_population",
                                 "bootstrap": {"unit": "game", "version": BOOTSTRAP_VERSION,
                                               "replicates": BOOTSTRAP_REPLICATES, "seed": development.frozen.BOOTSTRAP_SEED,
                                               "scope": "fixed_fold4_predictions_and_rules_main_configuration",
                                               "minimum_valid_fraction_for_interval": report_event.MIN_VALID_BOOTSTRAP}}})
    artifact = publish_artifact(DESTINATION, manifest_fields={
        "artifact_type": decision.ARTIFACT_TYPE, "schema_version": decision.ARTIFACT_TYPE,
        "artifact_identity": DESTINATION.name, **identity,
    }, files={"rules.json": canonical_json_bytes(rules), "results.json": canonical_json_bytes(results)})
    print(json.dumps({"path": str(DESTINATION), "manifest_digest": artifact.manifest_digest,
                      "main_shared_feasible": rules[decision.MAIN_CONFIGURATION]["shared"]["feasible"]}), flush=True)


if __name__ == "__main__":
    main()
