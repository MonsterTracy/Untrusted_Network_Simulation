"""在冻结 Development 上评价狼队原始报告事件的 ToM 三支规则。"""

import argparse
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
from itertools import groupby
import json
from math import fsum, lcm, log
from operator import ge, lt
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import analyze_observer_dtrs_report_event as report_study
from scripts import fit_observer_dtrs_report_event as frozen
from werewolf.artifact_io import open_artifact_envelope, read_artifact_file
from werewolf.development_publication import open_role_sidecar
from werewolf.observer_dtrs.context import SPEECH_PHASES, contexts_for_pre
from werewolf.observer_dtrs.decision import Region
from werewolf.observer_dtrs.features import observer_features
from werewolf.observer_dtrs.labels import label_from_observation
from werewolf.observer_dtrs.probabilities import ProbabilityMatrix
from werewolf.tom.development_input import open_development_input
from werewolf.tom.protocol import BOOTSTRAP_VERSION, bootstrap_indices


SELECTION_FOLDS = (*frozen.TRAIN_FOLDS, frozen.CAL_FOLD)
EVALUATION_FOLD = frozen.EVAL_FOLD
DEVELOPMENT_FOLDS = (*SELECTION_FOLDS, EVALUATION_FOLD)
EVALUATION_POPULATION = {"games": 960, "pres": 3789, "rows": 14723}
HISTORICAL_POPULATION = {
    "games": 4800, "pres": 18813, "rows": 73224, "speaker_hits": 37280,
    "teammate_only_hits": 7650, "team_hits": 44930, "team_misses": 28294,
    "successful_empty": 17547, "nonempty_team_misses": 10747,
}
PRE_IDENTITY = ("game_id", "boundary_id", "prefix_digest", "step_index")
ZERO, ONE = Decimal(0), Decimal(1)

# 以下切点来自研究者提供的服务器 Fold 0–3 结果，保留完整十进制字面值。
DISTRIBUTION_ALPHA = Decimal("0.387551836669445037841796875")
DISTRIBUTION_POS_BUDGET = "0.10"
DISTRIBUTION_WINDOW = (0.3, 0.4)
DISTRIBUTION_GATES = {
    "Direct": None,
    "TV-0.05": ("uniform_tv", "<", 0.05),
    "TV-0.10": ("uniform_tv", "<", 0.10),
    "Entropy-0.99": ("normalized_entropy", ">=", 0.99),
}
DISTRIBUTION_BETAS = {
    "0.15": {
        "Direct": "0.079652510583400726318359375",
        "TV-0.05": "0.124301455914974212646484375",
        "TV-0.10": "0.1209847293794155120849609375",
        "Entropy-0.99": "0.12324838340282440185546875",
    },
    "0.20": {
        "Direct": "0.1065414138138294219970703125",
        "TV-0.05": "0.17849795520305633544921875",
        "TV-0.10": "0.183103973977267742156982421875",
        "Entropy-0.99": "0.18366961181163787841796875",
    },
}


@dataclass(frozen=True, slots=True)
class TeamSample:
    game_id: str
    boundary_id: str
    observer_id: str
    day: int
    phase: str
    alive_wolves: int
    score: float
    y: int
    speaker_hit: bool
    successful_empty: bool
    uniform_tv: float
    normalized_entropy: float

    @property
    def sample_id(self):
        return self.game_id, self.boundary_id, self.observer_id


def pre_samples(pre, wolves, reports, q_record, *, attempt_id):
    identity = tuple(getattr(pre, name) for name in PRE_IDENTITY)
    if tuple(q_record[name] for name in PRE_IDENTITY) != identity:
        raise ValueError(f"冻结 Q 与 PRE 身份不符：{identity[:2]}")
    matrix = ProbabilityMatrix(*identity[:3], q_record["q"])
    contexts = contexts_for_pre(pre, wolves)
    if not contexts:
        raise ValueError(f"狼人发言 PRE 缺少合格观察者：{identity[:2]}")
    links = {link.observer_id: link.observation_id for link in pre.belief_observation_links}
    for ctx in contexts:
        report_id = links[ctx.observer_id]
        label = label_from_observation(ctx, reports.get(report_id))
        if not label.label_observed or label.observation_id != report_id or label.attempt_id != attempt_id:
            raise ValueError(f"原始报告缺失、失败或身份错配：{identity[:2]} / {report_id}")
        raw = label.raw_suspicion_support
        q_row = matrix.row(ctx.observer_id)
        nonself = [value for seat, value in zip(q_row.seat_ids, q_row.values, strict=True)
                   if seat != ctx.observer_id]
        uniform_tv = 0.5 * fsum(abs(value - 1 / 6) for value in nonself)
        entropy = -fsum(value * log(value) for value in nonself if value) / log(6)
        yield TeamSample(
            ctx.game_id, ctx.boundary_id, ctx.observer_id, ctx.day, ctx.phase, ctx.wolf_count,
            observer_features(ctx, q_row).r, int(any(wolf in raw for wolf in ctx.alive_wolf_ids)),
            ctx.acting_wolf in raw, not raw, uniform_tv, entropy,
        )


def _frozen_q(publication, selected_games):
    archive = open_artifact_envelope(
        frozen.Q_PATH, expected_artifact_type=frozen.Q_TYPE, expected_schema_version=frozen.Q_TYPE,
    )
    manifest = archive.manifest
    if (archive.manifest_digest != frozen.Q_DIGEST
            or manifest["publication_digest"] != frozen.PUBLICATION_DIGEST
            or manifest["split_digest"] != frozen.SPLIT_DIGEST
            or manifest["development_game_set_digest"] != publication["development_game_set_digest"]
            or manifest["collection_identity_digest"] != publication["collection_identity_digest"]
            or tuple(manifest["row_and_column_seat_order"]) != frozen.PLAYER_IDS
            or manifest["file_table"]["predictions.jsonl"]["sha256"] != frozen.Q_SHA256):
        raise ValueError("Q 档案与冻结 Development 来源不符")
    all_games = {member["game_id"] for member in publication["development_games"]}
    seen, by_pre = set(), {}
    for line in read_artifact_file(archive, "predictions.jsonl").splitlines():
        record = json.loads(line)
        key = record["game_id"], record["boundary_id"]
        if key in seen or key[0] not in all_games:
            raise ValueError(f"Q 档案包含重复或非 Development 的 PRE：{key}")
        seen.add(key)
        if key[0] in selected_games:
            by_pre[key] = record
    if len(seen) != HISTORICAL_POPULATION["pres"]:
        raise ValueError(f"冻结 Q PRE 数量不符：实际 {len(seen)}，预期 {HISTORICAL_POPULATION['pres']}")
    return by_pre


def load_development(folds):
    """按冻结名单读完整游戏，仅消费所请求折的 PRE 和 Q。"""
    publication = open_artifact_envelope(
        frozen.DEVELOPMENT, expected_artifact_type=frozen.PUBLICATION_TYPE,
        expected_schema_version=frozen.PUBLICATION_VERSION,
    )
    pm = publication.manifest
    if publication.manifest_digest != frozen.PUBLICATION_DIGEST or pm["split_manifest_digest"] != frozen.SPLIT_DIGEST:
        raise ValueError("Development Publication 或 Split 身份不符")
    fold_record = json.loads(read_artifact_file(publication, "public/development_fold_manifest.json"))
    fold_games = {
        fold["fold_index"]: tuple(game["game_id"] for game in fold["held_out_games"])
        for fold in fold_record["folds"]
    }
    if ({fold: len(ids) for fold, ids in fold_games.items()}
            != dict.fromkeys(DEVELOPMENT_FOLDS, EVALUATION_POPULATION["games"])
            or fold_record["manifest_digest"] != pm["development_fold_manifest_digest"]):
        raise ValueError("冻结 Development Fold 名单不符")
    selected_folds = {fold: fold_games[fold] for fold in folds}
    selected_games = {game_id for ids in selected_folds.values() for game_id in ids}
    members = {game["game_id"]: game for game in pm["development_games"] if game["game_id"] in selected_games}
    by_pre = _frozen_q(pm, selected_games)
    game_ids = tuple(members)
    rows, teams = [], None
    for offset in range(0, len(game_ids), 120):
        batch_ids = game_ids[offset:offset + 120]
        batch = open_development_input(
            frozen.DEVELOPMENT, expected_publication_digest=frozen.PUBLICATION_DIGEST,
            expected_split_digest=frozen.SPLIT_DIGEST, game_ids=batch_ids,
        )
        if batch.public_view.game_ids != batch_ids:
            raise ValueError("Development 批次与冻结成员名单不符")
        if teams is None:
            teams = {
                (game.game_id, game.bundle_digest): tuple(seat for seat, role in game.role_assignment if role == "Werewolf")
                for game in open_role_sidecar(batch).games
            }
        for game in batch.public_view.games:
            member = members[game.game_id]
            if game.bundle_digest != member["bundle_digest"] or not member["attempt_id"]:
                raise ValueError(f"游戏尝试或 Bundle 身份不符：{game.game_id}")
            wolves = teams[game.game_id, game.bundle_digest]
            reports = {report.observation_id: report for report in game.belief_observations}
            if len(reports) != len(game.belief_observations):
                raise ValueError(f"原始报告重复：{game.game_id}")
            for pre in game.authoritative_pre_prefixes:
                if pre.current_speaker not in wolves or pre.public_temporal_state.phase.value not in SPEECH_PHASES:
                    continue
                key = game.game_id, pre.boundary_id
                q_record = by_pre.pop(key, None)
                if q_record is None:
                    raise ValueError(f"合法狼人发言 PRE 缺少冻结 Q：{key}")
                rows.extend(pre_samples(pre, wolves, reports, q_record, attempt_id=member["attempt_id"]))
    if by_pre:
        raise ValueError(f"所选折有 {len(by_pre)} 个 Q PRE 未匹配，首个为 {next(iter(by_pre))}")
    source = {
        "publication_digest": publication.manifest_digest, "split_digest": pm["split_manifest_digest"],
        "fold_manifest_digest": fold_record["manifest_digest"], "q_manifest_digest": frozen.Q_DIGEST,
        "q_predictions_sha256": frozen.Q_SHA256,
    }
    return frozen.split_samples(rows, selected_folds), source


def population(rows):
    counts = frozen.support(rows)
    speaker_hits = sum(row.speaker_hit for row in rows)
    empty = sum(row.successful_empty for row in rows)
    return {
        "games": counts["games"], "pres": len({(row.game_id, row.boundary_id) for row in rows}),
        "rows": counts["rows"], "speaker_hits": speaker_hits, "teammate_only_hits": counts["positive"] - speaker_hits,
        "team_hits": counts["positive"], "team_misses": counts["negative"], "successful_empty": empty,
        "nonempty_team_misses": counts["negative"] - empty,
    }


def population_by_wolf_count(rows):
    return {"overall": population(rows), **{
        str(count): population([row for row in rows if row.alive_wolves == count]) for count in (1, 2)
    }}


def _check_population(observed, expected, name):
    differences = {
        key: {"expected": value, "observed": observed[key]}
        for key, value in expected.items() if observed[key] != value
    }
    if differences:
        raise ValueError(f"{name} 冻结人口复现不一致：{json.dumps(differences, ensure_ascii=False)}")


def game_weight_units(rows):
    """用同分母的整数表示游戏等权，风险预算边界可精确比较。"""
    sizes = Counter(row.game_id for row in rows)
    common = lcm(*set(sizes.values()))
    return [common // sizes[row.game_id] for row in rows]


def score_blocks(rows, weights):
    ordered = sorted(zip(rows, weights, strict=True), key=lambda pair: pair[0].score)
    return [(score, list(block)) for score, block in groupby(ordered, key=lambda pair: pair[0].score)]


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def score_diagnostics(blocks, total_units):
    """用同分块计算加权 AUROC，区间内给出各标签质量和经验命中率。"""
    edges = frozen.BIN_EDGES
    histograms = [Counter() for _ in edges]
    positive_units = negative_units = twice_concordant = 0
    for score, block in blocks:
        counts = Counter(row.y for row, _ in block)
        hits = sum(weight for row, weight in block if row.y)
        misses = sum(weight for row, weight in block if not row.y)
        twice_concordant += hits * (2 * negative_units + misses)
        positive_units += hits
        negative_units += misses
        index = len(edges) - 1 if score > 1 else min(bisect_right(edges, score) - 1, len(edges) - 2)
        histograms[index].update({"team_hits": counts[1], "team_misses": counts[0],
                                  "positive_units": hits, "negative_units": misses})
    bins = []
    for index, histogram in enumerate(histograms):
        overflow = index == len(edges) - 1
        if overflow and not histogram:
            continue
        hits, misses = histogram["positive_units"], histogram["negative_units"]
        bins.append({
            "lower": edges[index], "upper": None if overflow else edges[index + 1],
            "lower_inclusive": not overflow, "upper_inclusive": index == len(edges) - 2,
            "team_hits": histogram["team_hits"], "team_misses": histogram["team_misses"],
            "mass": _ratio(hits + misses, total_units), "hit_rate": _ratio(hits, hits + misses),
            "share_of_positive_mass": _ratio(hits, positive_units),
            "share_of_negative_mass": _ratio(misses, negative_units),
        })
    return {
        "auroc": _ratio(twice_concordant, 2 * positive_units * negative_units),
        "game_equal_base_rate": _ratio(positive_units, total_units),
        "score_bins": bins,
    }


@dataclass(frozen=True, slots=True)
class Cut:
    split: int
    threshold: Decimal
    lower: Decimal
    upper: Decimal
    lower_inclusive: bool
    upper_inclusive: bool
    mass_units: int
    error_units: int
    records: int
    games: int
    error_games: int

    def to_record(self, total_units):
        return {
            "split": self.split, "threshold": str(self.threshold),
            "interval": {"lower": str(self.lower), "upper": str(self.upper),
                         "lower_inclusive": self.lower_inclusive, "upper_inclusive": self.upper_inclusive},
            "mass": _ratio(self.mass_units, total_units), "error_mass": _ratio(self.error_units, total_units),
            "error_rate": _ratio(self.error_units, self.mass_units), "records": self.records,
            "games": self.games, "error_games": self.error_games,
            "mass_units": self.mass_units, "error_units": self.error_units,
        }


def _exact_midpoint(lower, upper):
    # 两个端点是有限十进制。按其实际小数位设置精度，保留相邻 float 之间的中点。
    with localcontext() as ctx:
        ctx.prec = 2 + max(0, -lower.as_tuple().exponent, -upper.as_tuple().exponent)
        return (lower + upper) / 2


def direction_cuts(blocks, direction):
    """split 是 NEG 已纳入或 POS 已排除的低分块数。"""
    scores = [Decimal.from_float(score) for score, _ in blocks]
    size = len(scores)
    positive = direction is Region.POS
    splits = range(size, -1, -1) if positive else range(size + 1)
    first = size if positive else 0
    error_label = 0 if positive else 1
    mass = error = records = 0
    games, error_games, cuts = set(), set(), []
    for split in splits:
        if split != first:
            for row, weight in blocks[split if positive else split - 1][1]:
                mass += weight
                records += 1
                games.add(row.game_id)
                if row.y == error_label:
                    error += weight
                    error_games.add(row.game_id)
        lower = scores[split - 1] if split else ZERO
        next_score = scores[split] if split < size else None
        upper = ONE if next_score is None else min(ONE, next_score)
        lower_inclusive = positive or split == 0
        upper_inclusive = not positive or next_score is None or next_score > ONE
        if lower > upper or (lower == upper and not (lower_inclusive and upper_inclusive)):
            continue
        cuts.append(Cut(
            split, _exact_midpoint(lower, upper), lower, upper, lower_inclusive, upper_inclusive,
            mass, error, records, len(games), len(error_games),
        ))
    return list(reversed(cuts)) if positive else cuts


def _unit_decimal(value):
    try:
        number = Decimal(value)
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"须为 [0,1] 内的十进制数：{value}") from error
    if not number.is_finite() or not ZERO <= number <= ONE:
        raise ValueError(f"须为 [0,1] 内的有限数：{value}")
    return number


def maximum_coverage(negative, positive, *, neg_risk, pos_risk, total_units):
    """在两侧有正质量且 split_NEG ≤ split_POS 时查询全部并列最优划分。"""
    neg_budget, pos_budget = (Fraction(_unit_decimal(value)) for value in (neg_risk, pos_risk))

    def within(cut, budget):
        # 条件风险只在方向质量为正时有定义。
        return (cut.mass_units > 0
                and cut.error_units * budget.denominator <= cut.mass_units * budget.numerator)

    feasible_pos = [cut for cut in positive if within(cut, pos_budget)]
    pos_index, best_mass, matches = 0, -1, []
    for neg in negative:
        if not within(neg, neg_budget):
            continue
        while pos_index < len(feasible_pos) and feasible_pos[pos_index].split < neg.split:
            pos_index += 1
        if pos_index == len(feasible_pos):
            break
        # 对固定 NEG，合法 POS 中最小 split 保留最多明确质量。
        pos = feasible_pos[pos_index]
        covered = neg.mass_units + pos.mass_units
        if covered > best_mass:
            best_mass, matches = covered, []
        if covered == best_mass:
            matches.append({"neg_split": neg.split, "pos_split": pos.split,
                            "beta": str(neg.threshold), "alpha": str(pos.threshold)})
    return {
        "feasible": bool(matches),
        "coverage": _ratio(best_mass, total_units) if matches else None,
        "bnd_mass": _ratio(total_units - best_mass, total_units) if matches else None,
        "maximizers": matches,
    }


def _region_summary(selected, units, error_label, *, detailed=False):
    errors = [(row, weight) for row, weight in selected if row.y == error_label]
    region_units = sum(weight for _, weight in selected)
    error_units = sum(weight for _, weight in errors)
    summary = {
        "mass": _ratio(region_units, units), "records": len(selected),
        "games": len({row.game_id for row, _ in selected}),
        "error_mass": _ratio(error_units, units) if error_label is not None else None,
        "error_rate": _ratio(error_units, region_units) if error_label is not None else None,
        "error_games": len({row.game_id for row, _ in errors}) if error_label is not None else None,
    }
    if detailed and error_label is not None:
        summary.update(mass_units=region_units, error_units=error_units,
                       correct_mass=_ratio(region_units - error_units, units))
    return summary


def _rule_summary(entries, full_units, *, detailed=False):
    units = sum(weight for _, weight, _ in entries)
    positive = sum(weight for row, weight, _ in entries if row.y)
    regions = {}
    for region, error_label in ((Region.NEG, 1), (Region.BND, None), (Region.POS, 0)):
        selected = [(row, weight) for row, weight, decision in entries if decision is region]
        regions[region.value] = _region_summary(selected, units, error_label, detailed=detailed)
    return {
        **frozen.support([row for row, _, _ in entries]),
        "population_mass": _ratio(units, full_units), "game_equal_base_rate": _ratio(positive, units),
        "coverage": _ratio(sum(weight for _, weight, decision in entries if decision is not Region.BND), units),
        "regions": regions,
    }


def _candidate_summary(entries, full_units, *, detailed=False):
    groups = {
        "single_wolf": lambda row: row.alive_wolves == 1,
        "two_wolves": lambda row: row.alive_wolves == 2,
        "day_1": lambda row: row.day == 1,
        "day_2": lambda row: row.day == 2,
        "day_3_or_later": lambda row: row.day >= 3,
        "discussion": lambda row: row.phase == "discussion",
        "pk": lambda row: row.phase == "pk_discussion",
    }
    return {
        "overall": _rule_summary(entries, full_units, detailed=detailed),
        "subgroups": {
            name: _rule_summary([entry for entry in entries if contains(entry[0])], full_units,
                                detailed=detailed)
            for name, contains in groups.items()
        },
    }


def evaluate_candidate(rows, beta, alpha):
    beta, alpha = _unit_decimal(beta), _unit_decimal(alpha)
    if beta > alpha:
        raise ValueError("须满足 beta ≤ alpha")
    weights = game_weight_units(rows)
    entries = []
    for row, weight in zip(rows, weights, strict=True):
        score = Decimal.from_float(row.score)
        decision = Region.NEG if score < beta else Region.POS if score > alpha else Region.BND
        entries.append((row, weight, decision))
    return {"beta": str(beta), "alpha": str(alpha), **_candidate_summary(entries, sum(weights))}


def _negative_masks(rows, scores, pos, beta, gate):
    """分配互斥的 L 和 A。窗口和形状比较复现开发阶段的 float 语义。"""
    low = np.asarray([score < beta for score in scores], dtype=bool) & ~pos
    if gate is None:
        return low, np.zeros_like(low)
    feature, comparison, threshold = gate
    compare = lt if comparison == "<" else ge
    lower, upper = DISTRIBUTION_WINDOW
    eligible = np.asarray([
        row.alive_wolves == 2 and lower <= row.score < upper and compare(getattr(row, feature), threshold)
        for row in rows
    ], dtype=bool)
    return low, eligible & ~(pos | low)


def _negative_diagnostics(weighted_rows, mask, total_units):
    selected = [entry for entry, included in zip(weighted_rows, mask, strict=True) if included]
    result = _region_summary(selected, total_units, 1, detailed=True)
    empty_units = sum(weight for row, weight in selected if row.successful_empty)
    miss_units = sum(weight for row, weight in selected if not row.y and not row.successful_empty)
    return {
        **result, "successful_empty_mass": _ratio(empty_units, total_units),
        "nonempty_team_miss_mass": _ratio(miss_units, total_units),
        "team_hit_mass": result["error_mass"],
    }


def _check_negative_split(whole, parts):
    for field in ("mass_units", "error_units"):
        if whole[field] != sum(region[field] for region in parts):
            raise ValueError("NEG 质量或错误质量与子区域分解不一致")


def _bootstrap_metric(point, values):
    interval = report_study.interval(values)
    return {
        "point_estimate": point, **interval,
        "zero_denominator_replicates": interval["replicates"] - interval["valid_replicates"],
    }


def _budget_summary(region, budget, risk_metric):
    interval = risk_metric["percentile_95"]
    return {
        "development_budget": budget,
        "empirical_met": (region["mass_units"] > 0
                          and region["error_units"] <= Fraction(budget) * region["mass_units"]),
        "interval_upper_within_budget": None if interval is None else interval[1] <= float(budget),
    }


def _relative_negative(weighted_rows, total_units, low, additional, direct_mask, neg, direct_neg):
    mask = low | additional
    added_mask, removed_mask = mask & ~direct_mask, direct_mask & ~mask
    added = _negative_diagnostics(weighted_rows, added_mask, total_units)
    removed = _negative_diagnostics(weighted_rows, removed_mask, total_units)
    by_region = {
        "L": _negative_diagnostics(weighted_rows, added_mask & low, total_units),
        "A": _negative_diagnostics(weighted_rows, added_mask & additional, total_units),
    }
    _check_negative_split(added, by_region.values())
    for field in ("mass_units", "error_units"):
        if neg[field] - direct_neg[field] != added[field] - removed[field]:
            raise ValueError("相对 Direct 的 NEG 质量或错误质量与新增、撤回分解不一致")
    return {
        "added": added, "removed": removed, "added_by_region": by_region,
        "net_coverage_gain": _ratio(neg["mass_units"] - direct_neg["mass_units"], total_units),
    }


def evaluate_distribution(rows):
    """对同一 Fold 4 人口应用八个冻结规则，按整局配对重采样。"""
    weights = game_weight_units(rows)
    weighted_rows = list(zip(rows, weights, strict=True))
    total_units = sum(weights)
    scores = [Decimal.from_float(row.score) for row in rows]
    pos = np.asarray([score > DISTRIBUTION_ALPHA for score in scores], dtype=bool)
    labels = np.asarray([row.y for row in rows], dtype=bool)

    sizes = Counter(row.game_id for row in rows)
    games = sorted(sizes)
    game_index = {game: index for index, game in enumerate(games)}
    row_games = np.asarray([game_index[row.game_id] for row in rows])
    game_sizes = np.asarray([sizes[game] for game in games])
    game_weights = 1 / game_sizes
    draws = bootstrap_indices(games, frozen.BOOTSTRAP_REPLICATES, frozen.BOOTSTRAP_SEED)

    def counts(mask):
        return np.bincount(row_games[mask], minlength=len(games))

    def resample(numerator, denominator):
        return report_study.bootstrap_ratio(numerator, denominator, game_weights, draws)

    pos_counts = counts(pos)
    pos_risks = resample(counts(pos & ~labels), pos_counts)
    pos_selected = [entry for entry, included in zip(weighted_rows, pos, strict=True) if included]
    pos_summary = _region_summary(pos_selected, total_units, 0, detailed=True)
    pos_metric = _bootstrap_metric(pos_summary["error_rate"], pos_risks)
    pos_budget = _budget_summary(pos_summary, DISTRIBUTION_POS_BUDGET, pos_metric)

    operating_points = {}
    for neg_budget, betas in DISTRIBUTION_BETAS.items():
        rules, masks, replicates = {}, {}, {}
        for method, beta in betas.items():
            gate = DISTRIBUTION_GATES[method]
            low, additional = _negative_masks(rows, scores, pos, Decimal(beta), gate)
            negative = low | additional
            entries = []
            for (row, weight), is_pos, is_neg in zip(weighted_rows, pos, negative, strict=True):
                if is_pos:
                    decision = Region.POS
                elif is_neg:
                    decision = Region.NEG
                else:
                    decision = Region.BND
                entries.append((row, weight, decision))
            evaluated = _candidate_summary(entries, total_units, detailed=True)
            overall = evaluated["overall"]
            if (not np.array_equal(pos, [decision is Region.POS for _, _, decision in entries])
                    or overall["regions"]["POS"] != pos_summary):
                raise ValueError(f"{method} 的 POS 人口或统计与共同 POS 不一致")

            neg = overall["regions"]["NEG"]
            neg_counts, neg_errors = counts(negative), counts(negative & labels)
            metrics = {
                "coverage": resample(pos_counts + neg_counts, game_sizes),
                "neg_error_rate": resample(neg_errors, neg_counts),
                "neg_error_mass": resample(neg_errors, game_sizes),
            }
            bootstrap = {
                "coverage": _bootstrap_metric(overall["coverage"], metrics["coverage"]),
                "neg_error_rate": _bootstrap_metric(neg["error_rate"], metrics["neg_error_rate"]),
                "pos_error_rate": pos_metric,
                "neg_error_mass": _bootstrap_metric(neg["error_mass"], metrics["neg_error_mass"]),
            }
            rules[method] = {
                "beta": beta, "alpha": str(DISTRIBUTION_ALPHA),
                "gate": None if gate is None else dict(zip(("feature", "comparison", "threshold"), gate)),
                **evaluated, "bootstrap": bootstrap,
                "risk_budgets": {"NEG": _budget_summary(neg, neg_budget, bootstrap["neg_error_rate"]),
                                 "POS": pos_budget},
            }
            masks[method], replicates[method] = (low, additional), metrics

        direct_mask = masks["Direct"][0]
        direct_neg = rules["Direct"]["overall"]["regions"]["NEG"]
        for method, result in rules.items():
            if method == "Direct":
                continue
            low, additional = masks[method]
            neg = result["overall"]["regions"]["NEG"]
            subregions = {"L": _negative_diagnostics(weighted_rows, low, total_units),
                          "A": _negative_diagnostics(weighted_rows, additional, total_units)}
            _check_negative_split(neg, subregions.values())
            result["neg_subregions"] = subregions
            change = _relative_negative(weighted_rows, total_units, low, additional, direct_mask, neg, direct_neg)
            risk_diff = None
            if neg["error_rate"] is not None and direct_neg["error_rate"] is not None:
                risk_diff = neg["error_rate"] - direct_neg["error_rate"]
            points = {"coverage": change["net_coverage_gain"], "neg_error_rate": risk_diff,
                      "neg_error_mass": _ratio(neg["error_units"] - direct_neg["error_units"], total_units)}
            change["paired_differences"] = {
                name: _bootstrap_metric(point, replicates[method][name] - replicates["Direct"][name])
                for name, point in points.items()
            }
            result["relative_to_direct"] = change
        operating_points[neg_budget] = {"rules": rules}

    return {
        "weighting": "game_equal",
        "frozen_parameter_source": {"origin": "researcher_supplied_server_development_results",
                                    "folds": SELECTION_FOLDS},
        "primary_comparison": {"neg_budget": "0.15", "pos_budget": DISTRIBUTION_POS_BUDGET,
                               "reference": "Direct", "candidate": "TV-0.10"},
        "additional_neg_window": {"alive_wolves": 2, "lower_inclusive": DISTRIBUTION_WINDOW[0],
                                  "upper_exclusive": DISTRIBUTION_WINDOW[1], "comparison_type": "float"},
        "shared_pos": {
            "alpha": str(DISTRIBUTION_ALPHA), "matched_all_rules": True,
            "region": pos_summary, "risk_budget": pos_budget, "bootstrap": pos_metric,
            "paired_error_rate_difference": _bootstrap_metric(
                0.0 if pos_summary["mass_units"] else None, pos_risks - pos_risks,
            ),
        },
        "bootstrap": {"version": BOOTSTRAP_VERSION, "unit": "game", "seed": frozen.BOOTSTRAP_SEED,
                      "replicates": frozen.BOOTSTRAP_REPLICATES, "scope": "fixed_rules_exploratory_marginal",
                      "minimum_valid_fraction_for_interval": report_study.MIN_VALID_BOOTSTRAP},
        "operating_points": operating_points,
    }


def analyze_development(partitions, *, budgets=None):
    all_rows = [row for fold in DEVELOPMENT_FOLDS for row in partitions[fold]]
    reproduction = population_by_wolf_count(all_rows)
    _check_population(reproduction["overall"], HISTORICAL_POPULATION, "全 Development")
    rows = [row for fold in SELECTION_FOLDS for row in partitions[fold]]
    weights = game_weight_units(rows)
    blocks = score_blocks(rows, weights)
    total_units = sum(weights)
    negative = direction_cuts(blocks, Region.NEG)
    positive = direction_cuts(blocks, Region.POS)
    result = {
        "stage": "development_selection", "folds": SELECTION_FOLDS,
        "historical_population_reproduction": reproduction,
        "selection_population": population_by_wolf_count(rows),
        "score_diagnostics": score_diagnostics(blocks, total_units),
        "risk_coverage": {
            "total_weight_units": total_units,
            "joint_condition": "neg_split <= pos_split",
            "NEG": [cut.to_record(total_units) for cut in negative],
            "POS": [cut.to_record(total_units) for cut in positive],
        },
    }
    if budgets is not None:
        neg_risk, pos_risk = budgets
        result["budget_query"] = {
            "neg_risk": str(neg_risk), "pos_risk": str(pos_risk),
            **maximum_coverage(negative, positive, neg_risk=neg_risk, pos_risk=pos_risk, total_units=total_units),
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    develop = stages.add_parser("develop", help="核对全 Development 人口并在 Fold 0–3 做选择分析")
    develop.add_argument("--neg-risk", type=_unit_decimal, help="可选 NEG 风险预算，须与 --pos-risk 一并给定")
    develop.add_argument("--pos-risk", type=_unit_decimal, help="可选 POS 风险预算，须与 --neg-risk 一并给定")
    evaluate = stages.add_parser("evaluate", help="只用 Fold 4 评价给定的冻结阈值")
    evaluate.add_argument("--beta", type=_unit_decimal, required=True, help="NEG 切点，可直接复制输出中的十进制值")
    evaluate.add_argument("--alpha", type=_unit_decimal, required=True, help="POS 切点，可直接复制输出中的十进制值")
    stages.add_parser("evaluate-distribution", help="只用 Fold 4 评价八个冻结分布感知规则")
    args = parser.parse_args(argv)
    if args.stage == "develop":
        if (args.neg_risk is None) != (args.pos_risk is None):
            parser.error("须同时提供 NEG 与 POS 风险预算")
        budgets = None if args.neg_risk is None else (args.neg_risk, args.pos_risk)
        partitions, source = load_development(DEVELOPMENT_FOLDS)
        result = analyze_development(partitions, budgets=budgets)
    else:
        if args.stage == "evaluate" and args.beta > args.alpha:
            parser.error("须满足 beta ≤ alpha")
        partitions, source = load_development((EVALUATION_FOLD,))
        rows = partitions[EVALUATION_FOLD]
        audit = population_by_wolf_count(rows)
        _check_population(audit["overall"], EVALUATION_POPULATION, "Fold 4")
        if args.stage == "evaluate":
            stage = "exploratory_fold4_evaluation"
            evaluation = evaluate_candidate(rows, args.beta, args.alpha)
        else:
            stage = "exploratory_fold4_distribution_evaluation"
            evaluation = evaluate_distribution(rows)
        result = {"stage": stage, "folds": (EVALUATION_FOLD,), "population": audit, **evaluation}
    result["source"] = source
    result["boundary_policy"] = "DEFER_FIRST"
    result["threshold_encoding"] = "exact_decimal"
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
