"""用合成 PRE、报告和得分校验静态狼队事件研究。"""

from collections import Counter
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
from itertools import product
import json
import math
import random
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import analyze_observer_dtrs_team_event as study
from tests.observer_dtrs.conftest import observation_for, prefix_for
from werewolf.canonical_collection import BeliefObservationStatus
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.observer_dtrs.context import contexts_for_pre
from werewolf.observer_dtrs.decision import Region


WOLVES = PLAYER_IDS[:2]
ATTEMPT = "controlled-attempt"


def pre_fixture(*, game_id="controlled-game", dead=(), supports=None):
    pre = prefix_for(game_id=game_id, dead=dead)
    contexts = contexts_for_pre(pre, WOLVES)
    reports = [observation_for(ctx, (supports or {}).get(ctx.observer_id, ())) for ctx in contexts]
    values = [[0.0 if target == observer else 1 / 6 for target in PLAYER_IDS] for observer in PLAYER_IDS]
    values[2] = [0.25, 0.125, 0.0, 0.5, 0.125, 0.0, 0.0]
    q = {**{name: getattr(pre, name) for name in study.PRE_IDENTITY}, "q": values}
    return pre, contexts, {r.observation_id: r for r in reports}, q


def test_team_event_uses_successful_raw_support_and_audits_historical_categories():
    supports = dict(zip(PLAYER_IDS[2:], ((), (WOLVES[0],), (WOLVES[1],), WOLVES, ("player3",)), strict=True))
    pre, _, reports, q = pre_fixture(supports=supports)
    rows = list(study.pre_samples(pre, WOLVES, reports, q, attempt_id=ATTEMPT))

    assert [row.observer_id for row in rows] == list(PLAYER_IDS[2:])
    assert [row.y for row in rows] == [0, 1, 1, 1, 0]
    assert [row.speaker_hit for row in rows] == [False, True, False, True, False]
    assert rows[0].score == 0.375
    assert study.population(rows) == {
        "games": 1, "pres": 1, "rows": 5, "speaker_hits": 2, "teammate_only_hits": 1,
        "team_hits": 3, "team_misses": 2, "successful_empty": 1, "nonempty_team_misses": 1,
    }


def test_single_wolf_excludes_dead_teammate_without_renormalizing_q():
    pre, _, reports, q = pre_fixture(dead=(WOLVES[1], "player4"),
                                     supports={"player3": (WOLVES[1],), "player5": WOLVES})
    rows = list(study.pre_samples(pre, WOLVES, reports, q, attempt_id=ATTEMPT))
    assert [row.observer_id for row in rows] == ["player3", "player5", "player6", "player7"]
    assert all(row.alive_wolves == 1 and row.y == row.speaker_hit for row in rows)
    assert rows[0].score == 0.25 and rows[0].y == 0 and not rows[0].successful_empty
    assert rows[1].y == 1


@pytest.mark.parametrize("nonself, tv, entropy", (
    ((1 / 6,) * 6, 0, 1),
    ((0, 0, 1, 0, 0, 0), 5 / 6, 0),
    ((0.125, 0.25, 0.125, 0.25, 0.125, 0.125), 1 / 6, 0.5 * math.log(32) / math.log(6)),
))
def test_team_sample_uses_all_six_nonself_seats_for_row_shape(nonself, tv, entropy):
    pre, _, reports, q = pre_fixture(dead=("player4",))
    q["q"][2] = (*nonself[:2], 0.0, *nonself[2:])
    row = next(study.pre_samples(pre, WOLVES, reports, q, attempt_id=ATTEMPT))
    assert row.observer_id == "player3"
    assert row.score == sum(nonself[:2])
    assert row.uniform_tv == pytest.approx(tv)
    assert row.normalized_entropy == pytest.approx(entropy)


@pytest.mark.parametrize("problem", ("missing", "failed", "wrong_observer", "wrong_attempt"))
def test_invalid_report_fails_instead_of_becoming_negative(problem):
    pre, contexts, reports, q = pre_fixture()
    key = "report-player3"
    attempt = ATTEMPT
    if problem == "missing":
        del reports[key]
    elif problem == "failed":
        reports[key] = observation_for(contexts[0], status=BeliefObservationStatus.ERROR)
    elif problem == "wrong_observer":
        reports[key] = reports["report-player4"]
    else:
        attempt = "other-attempt"
    with pytest.raises(ValueError, match="原始报告|does not bind"):
        list(study.pre_samples(pre, WOLVES, reports, q, attempt_id=attempt))


@pytest.mark.parametrize("key", study.PRE_IDENTITY)
def test_q_must_match_the_authoritative_pre(key):
    pre, _, reports, q = pre_fixture()
    q[key] = "other"
    with pytest.raises(ValueError, match="Q 与 PRE 身份不符"):
        list(study.pre_samples(pre, WOLVES, reports, q, attempt_id=ATTEMPT))


@pytest.fixture
def mocked_development(monkeypatch):
    games, members, roles, predictions = {}, [], [], []
    folds = []
    for fold in study.DEVELOPMENT_FOLDS:
        game_id, bundle = f"game-{fold}", f"bundle-{fold}"
        pre, _, reports, q = pre_fixture(game_id=game_id)
        games[game_id] = SimpleNamespace(
            game_id=game_id, bundle_digest=bundle, belief_observations=tuple(reports.values()),
            authoritative_pre_prefixes=(pre,),
        )
        members.append({"game_id": game_id, "bundle_digest": bundle, "attempt_id": ATTEMPT})
        assignment = [(seat, "Werewolf" if seat in WOLVES else "Villager") for seat in PLAYER_IDS]
        roles.append(SimpleNamespace(game_id=game_id, bundle_digest=bundle, role_assignment=assignment))
        predictions.append(q)
        folds.append({"fold_index": fold, "held_out_games": [{"game_id": game_id, "bundle_digest": bundle}]})
    manifest = {
        "split_manifest_digest": study.frozen.SPLIT_DIGEST, "development_games": members,
        "development_fold_manifest_digest": "fold-digest",
        "development_game_set_digest": "game-set", "collection_identity_digest": "collection",
    }
    pub = SimpleNamespace(manifest_digest=study.frozen.PUBLICATION_DIGEST, manifest=manifest)
    q_manifest = {
        "publication_digest": pub.manifest_digest, "split_digest": manifest["split_manifest_digest"],
        "development_game_set_digest": "game-set", "collection_identity_digest": "collection",
        "row_and_column_seat_order": list(PLAYER_IDS),
        "file_table": {"predictions.jsonl": {"sha256": study.frozen.Q_SHA256}},
    }
    archive = SimpleNamespace(manifest_digest=study.frozen.Q_DIGEST, manifest=q_manifest)
    opened = []

    def open_envelope(path, **_):
        return {study.frozen.DEVELOPMENT: pub, study.frozen.Q_PATH: archive}[path]

    def read_file(artifact, name):
        if artifact is pub and name == "public/development_fold_manifest.json":
            return json.dumps({"folds": folds, "manifest_digest": "fold-digest"}).encode()
        assert artifact is archive and name == "predictions.jsonl"
        return "\n".join(json.dumps(q) for q in predictions).encode()

    def open_input(path, *, expected_publication_digest, expected_split_digest, game_ids):
        assert path == study.frozen.DEVELOPMENT
        assert (expected_publication_digest, expected_split_digest) == (
            pub.manifest_digest, manifest["split_manifest_digest"],
        )
        opened.extend(game_ids)
        return SimpleNamespace(public_view=SimpleNamespace(game_ids=game_ids, games=tuple(games[g] for g in game_ids)))

    monkeypatch.setattr(study, "open_artifact_envelope", open_envelope)
    monkeypatch.setattr(study, "read_artifact_file", read_file)
    monkeypatch.setattr(study, "open_development_input", open_input)
    monkeypatch.setattr(study, "open_role_sidecar", lambda _: SimpleNamespace(games=roles))
    monkeypatch.setattr(study, "EVALUATION_POPULATION", {"games": 1, "pres": 1, "rows": 5})
    monkeypatch.setattr(study, "HISTORICAL_POPULATION", {**study.HISTORICAL_POPULATION, "pres": len(games)})
    return SimpleNamespace(games=games, opened=opened, manifest=manifest, predictions=predictions)


def test_development_loader_uses_frozen_game_membership(mocked_development):
    parts, identity = study.load_development(study.DEVELOPMENT_FOLDS)
    assert mocked_development.opened == [f"game-{fold}" for fold in study.DEVELOPMENT_FOLDS]
    assert all(len(rows) == 5 and {row.game_id for row in rows} == {f"game-{fold}"} for fold, rows in parts.items())
    assert identity["fold_manifest_digest"] == "fold-digest"


@pytest.mark.parametrize("problem", ("duplicate_q", "duplicate_report", "wrong_bundle"))
def test_development_loader_rejects_invalid_frozen_population(mocked_development, problem):
    game = mocked_development.games["game-4"]
    if problem == "duplicate_q":
        mocked_development.predictions.append(mocked_development.predictions[-1])
    elif problem == "duplicate_report":
        game.belief_observations += game.belief_observations[:1]
    else:
        game.bundle_digest = "other-bundle"
    with pytest.raises(ValueError, match="重复|Bundle 身份"):
        study.load_development((study.EVALUATION_FOLD,))


def samples(scores, labels, games=None):
    games = range(len(scores)) if games is None else games
    return [
        study.TeamSample(str(game), f"pre-{i}", "player3", 1, "discussion", 2, float(score), y, bool(y), False,
                         uniform_tv=0.5, normalized_entropy=0.5)
        for i, (score, y, game) in enumerate(zip(scores, labels, games, strict=True))
    ]


def cut_tables(rows):
    weights = study.game_weight_units(rows)
    blocks = study.score_blocks(rows, weights)
    return study.direction_cuts(blocks, Region.NEG), study.direction_cuts(blocks, Region.POS), sum(weights)


def support_indices(rows, beta, alpha):
    scores = [Fraction.from_float(row.score) for row in rows]
    return tuple(i for i, value in enumerate(scores) if value < beta), tuple(
        i for i, value in enumerate(scores) if value > alpha
    )


def test_game_equal_auroc_weights_ties_and_preserves_class_distributions():
    rows = samples((0.75, 0.25, 0.75), (1, 0, 0), ("long", "long", "short"))
    weights = study.game_weight_units(rows)
    assert weights == [1, 1, 2]
    result = study.score_diagnostics(study.score_blocks(rows, weights), sum(weights))
    assert result["auroc"] == pytest.approx(2 / 3)
    assert result["game_equal_base_rate"] == 0.25
    score_bin = next(item for item in result["score_bins"] if item["lower"] == 0.7)
    assert score_bin["hit_rate"] == pytest.approx(1 / 3)
    assert score_bin["share_of_negative_mass"] == pytest.approx(2 / 3)
    assert score_bin["share_of_positive_mass"] == 1
    one_class = study.score_diagnostics(study.score_blocks(rows[:1], weights[:1]), 1)
    assert one_class["auroc"] is None and one_class["game_equal_base_rate"] == 1


def test_endpoints_and_ties_are_deferred_by_both_strict_boundaries():
    rows = samples((0, 0.5, 0.5, 1), (0, 0, 1, 1))
    negative, positive, total = cut_tables(rows)
    assert negative[0].to_record(total)["interval"] == {
        "lower": "0", "upper": "0", "lower_inclusive": True, "upper_inclusive": True,
    }
    assert negative[0].to_record(total)["error_rate"] is None
    assert [cut.records for cut in negative] == [0, 1, 3]
    assert [cut.records for cut in positive] == [3, 1, 0]
    assert positive[-1].threshold == 1
    assert positive[-1].to_record(total)["error_rate"] is None
    for beta, alpha, counts in ((0, 1, (0, 4, 0)), (0.5, 0.5, (1, 2, 1)), (0, 0, (0, 1, 3)), (1, 1, (3, 1, 0))):
        regions = study.evaluate_candidate(rows, beta, alpha)["overall"]["regions"]
        assert tuple(regions[key]["records"] for key in ("NEG", "BND", "POS")) == counts
        assert regions["BND"]["error_rate"] is None
    assert study.evaluate_candidate(rows, 0, 1)["overall"]["coverage"] == 0


@pytest.mark.parametrize("lower", (0.0, 0.5))
def test_exact_midpoint_preserves_the_split_between_adjacent_floats(lower):
    higher = math.nextafter(lower, 1)
    rows = samples((lower, higher), (0, 1))
    negative, positive, total = cut_tables(rows)
    result = study.maximum_coverage(negative, positive, neg_risk=0, pos_risk=0, total_units=total)
    assert result["feasible"] is True
    assert result["coverage"] == 1 and len(result["maximizers"]) == 1
    rule = result["maximizers"][0]
    assert Decimal.from_float(lower) < Decimal(rule["beta"]) == Decimal(rule["alpha"]) < Decimal.from_float(higher)
    evaluated = study.evaluate_candidate(rows, rule["beta"], rule["alpha"])["overall"]
    assert evaluated["coverage"] == 1
    assert evaluated["regions"]["NEG"]["error_rate"] == evaluated["regions"]["POS"]["error_rate"] == 0


def test_joint_query_handles_conflicting_directional_optima_with_equal_thresholds():
    rows = samples((0.125, 0.25, 0.375, 0.5, 0.625, 0.75), (0, 0, 0, 1, 1, 0))
    negative, positive, total = cut_tables(rows)
    best_neg = max(c.split for c in negative if 4 * c.error_units <= c.mass_units)
    best_pos = min(c.split for c in positive if 5 * c.error_units <= 2 * c.mass_units)
    assert (best_neg, best_pos) == (4, 3)
    result = study.maximum_coverage(negative, positive, neg_risk="0.25", pos_risk="0.4", total_units=total)
    assert result["coverage"] == 1 and len(result["maximizers"]) == 1
    rule = result["maximizers"][0]
    assert (rule["neg_split"], rule["pos_split"]) == (3, 3)
    assert Decimal(rule["beta"]) == Decimal(rule["alpha"]) == Decimal("0.4375")


def test_budget_scan_continues_past_a_cut_that_exceeds_the_budget():
    rows = samples((0.125, 0.25, 0.375, 0.5, 0.625), (0, 1, 0, 0, 1))
    negative, positive, total = cut_tables(rows)
    assert [cut.to_record(total)["error_rate"] for cut in negative] == [None, 0, 0.5, 1 / 3, 0.25, 0.4]
    result = study.maximum_coverage(negative, positive, neg_risk="0.25", pos_risk=0, total_units=total)
    assert result["coverage"] == 1
    assert [(rule["neg_split"], rule["pos_split"]) for rule in result["maximizers"]] == [(4, 4)]


@pytest.mark.parametrize(("labels", "budgets", "direction_feasibility"), (
    ((0, 0), ("0", "0"), (True, False)),
    ((1, 1), ("0", "0"), (False, True)),
    ((1, 0), ("0", "0"), (False, False)),
    ((1, 0, 0), ("0.5", "0.7"), (True, True)),
), ids=("only_neg", "only_pos", "neither", "overlapping_support"))
def test_bilateral_query_is_infeasible_without_two_compatible_supported_directions(
    labels, budgets, direction_feasibility,
):
    rows = samples(tuple(i / 8 for i in range(1, len(labels) + 1)), labels)
    negative, positive, total = cut_tables(rows)
    assert tuple(any(cut.mass_units and Fraction(cut.error_units, cut.mass_units) <= Fraction(budget)
                     for cut in cuts)
                 for cuts, budget in zip((negative, positive), budgets, strict=True)) == direction_feasibility
    result = study.maximum_coverage(
        negative, positive, neg_risk=budgets[0], pos_risk=budgets[1], total_units=total,
    )
    assert result == {"feasible": False, "coverage": None, "bnd_mass": None, "maximizers": []}


def test_bilateral_query_keeps_all_tied_supported_maximizers():
    rows = samples((0.125, 0.25, 0.375), (0, 0, 1))
    negative, positive, total = cut_tables(rows)
    result = study.maximum_coverage(negative, positive, neg_risk=1, pos_risk=1, total_units=total)
    assert result["feasible"] is True and result["coverage"] == 1
    assert [(rule["neg_split"], rule["pos_split"]) for rule in result["maximizers"]] == [(1, 1), (2, 2)]


def test_cuts_and_joint_query_match_independent_exhaustive_search():
    rng = random.Random(1729)
    pool = (0, 0.125, 0.25, 0.5, math.nextafter(0.5, 1), 0.75, 1)
    for _ in range(24):
        count = rng.randrange(2, 11)
        rows = samples(rng.choices(pool, k=count), rng.choices((0, 1), k=count),
                       rng.choices(("a", "b", "c"), k=count))
        sizes = Counter(row.game_id for row in rows)
        weights = [Fraction(1, len(sizes) * sizes[row.game_id]) for row in rows]
        endpoints = sorted({Fraction(0), Fraction(1), *(Fraction.from_float(row.score) for row in rows)})
        thresholds = endpoints + [(a + b) / 2 for a, b in zip(endpoints, endpoints[1:])]
        neg_sets = {support_indices(rows, threshold, Fraction(1))[0] for threshold in thresholds}
        pos_sets = {support_indices(rows, Fraction(0), threshold)[1] for threshold in thresholds}
        negative, positive, total = cut_tables(rows)

        for cuts, index, supports, error_label in ((negative, 0, neg_sets, 1), (positive, 1, pos_sets, 0)):
            observed = set()
            for cut in cuts:
                theta = Fraction(cut.threshold)
                bounds = (theta, Fraction(1)) if index == 0 else (Fraction(0), theta)
                side = support_indices(rows, *bounds)[index]
                errors = [i for i in side if rows[i].y == error_label]
                observed.add(side)
                assert Fraction(cut.mass_units, total) == sum(weights[i] for i in side)
                assert Fraction(cut.error_units, total) == sum(weights[i] for i in errors)
                assert (cut.records, cut.games, cut.error_games) == (
                    len(side), len({rows[i].game_id for i in side}), len({rows[i].game_id for i in errors}),
                )
            assert observed == supports

        for neg_budget, pos_budget in (("0", "0"), ("0.25", "0.4"), ("0.4", "0.25"), ("1", "1")):
            feasible = {}
            for beta, alpha in product(thresholds, repeat=2):
                if beta > alpha:
                    continue
                n, p = support_indices(rows, beta, alpha)
                n_mass, p_mass = sum(weights[i] for i in n), sum(weights[i] for i in p)
                if (n_mass > 0 and p_mass > 0
                        and sum(weights[i] for i in n if rows[i].y) <= Fraction(neg_budget) * n_mass
                        and sum(weights[i] for i in p if not rows[i].y) <= Fraction(pos_budget) * p_mass):
                    feasible[n, p] = n_mass + p_mass
            optimum = max(feasible.values(), default=None)
            result = study.maximum_coverage(negative, positive, neg_risk=neg_budget,
                                            pos_risk=pos_budget, total_units=total)
            found = set()
            for rule in result["maximizers"]:
                beta, alpha = Fraction(rule["beta"]), Fraction(rule["alpha"])
                n, p = support_indices(rows, beta, alpha)
                assert beta <= alpha and not set(n).intersection(p)
                found.add((n, p))
            assert found == {sides for sides, mass in feasible.items() if mass == optimum}
            assert result["feasible"] is (optimum is not None)
            assert result["coverage"] == (float(optimum) if optimum is not None else None)


def test_subgroup_rates_keep_the_complete_game_weights():
    rows = samples((0.125, 0.25, 0.875, 0.5, 0.5), (1, 0, 1, 0, 1), ("a", "a", "a", "a", "b"))
    rows = [replace(row, alive_wolves=1) if i in (0, 1, 4) else row for i, row in enumerate(rows)]
    result = study.evaluate_candidate(rows, "0.375", "0.75")
    one = result["subgroups"]["single_wolf"]
    assert one["population_mass"] == 0.75
    assert one["coverage"] == pytest.approx(1 / 3)
    assert one["regions"]["NEG"]["error_rate"] == 0.5
    assert one["regions"]["NEG"]["error_games"] == 1
    assert one["regions"]["POS"]["error_rate"] is None
    assert result["subgroups"]["pk"]["coverage"] is None
    assert result["overall"]["coverage"] == 0.375


def test_development_reproduction_precedes_search_and_fold4_scores_do_not_affect_it(monkeypatch):
    parts = {fold: samples((0.125,), (fold % 2,), (fold,)) for fold in study.DEVELOPMENT_FOLDS}
    expected = study.population([rows[0] for rows in parts.values()])
    monkeypatch.setattr(study, "HISTORICAL_POPULATION", expected)
    result = study.analyze_development(parts)
    assert "budget_query" not in result
    assert result["selection_population"]["overall"]["games"] == 4
    parts[study.EVALUATION_FOLD] = [replace(parts[study.EVALUATION_FOLD][0], score=0.875)]
    assert study.analyze_development(parts) == result
    monkeypatch.setattr(study, "HISTORICAL_POPULATION", {**expected, "teammate_only_hits": 1})
    with pytest.raises(ValueError, match=r'teammate_only_hits.*expected.*observed'):
        study.analyze_development(parts)


def test_develop_cli_queries_only_explicit_budgets(mocked_development, monkeypatch, capsys):
    monkeypatch.setattr(study, "HISTORICAL_POPULATION", {
        "games": 5, "pres": 5, "rows": 25, "speaker_hits": 0, "teammate_only_hits": 0,
        "team_hits": 0, "team_misses": 25, "successful_empty": 25, "nonempty_team_misses": 0,
    })
    assert study.main(["develop", "--neg-risk", "0", "--pos-risk", "0"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert mocked_development.opened == [f"game-{fold}" for fold in study.DEVELOPMENT_FOLDS]
    assert result["folds"] == [0, 1, 2, 3]
    assert result["historical_population_reproduction"]["overall"]["rows"] == 25
    assert result["selection_population"]["overall"]["rows"] == 20
    assert result["budget_query"] == {
        "neg_risk": "0", "pos_risk": "0", "feasible": False,
        "coverage": None, "bnd_mass": None, "maximizers": [],
    }


def test_evaluate_cli_applies_frozen_thresholds_without_loading_selection_or_searching(
    mocked_development, monkeypatch, capsys,
):
    def forbidden(*_, **__):
        pytest.fail("冻结候选评价不应重做阈值搜索")

    monkeypatch.setattr(study, "direction_cuts", forbidden)
    monkeypatch.setattr(study, "maximum_coverage", forbidden)
    assert study.main(["evaluate", "--beta", "0.375", "--alpha", "0.375"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert mocked_development.opened == ["game-4"]
    assert result["stage"] == "exploratory_fold4_evaluation" and result["folds"] == [4]
    assert (result["beta"], result["alpha"]) == ("0.375", "0.375")
    regions = result["overall"]["regions"]
    assert regions["BND"]["records"] == 1
    assert regions["POS"]["records"] == 0 and regions["POS"]["error_rate"] is None


@pytest.mark.parametrize("method, feature, cutoff, closer, farther, equality_passes", (
    ("TV-0.05", "uniform_tv", 0.05, 0, 1, False),
    ("TV-0.10", "uniform_tv", 0.10, 0, 1, False),
    ("Entropy-0.99", "normalized_entropy", 0.99, 1, 0, True),
))
def test_distribution_gate_uses_python_float_window_and_feature_boundaries(
    method, feature, cutoff, closer, farther, equality_passes,
):
    scores = (math.nextafter(0.3, 0.0), 0.3, math.nextafter(0.4, 0.0), 0.4, 0.33, 0.33, 0.33)
    passing = math.nextafter(cutoff, closer)
    shape = (passing, passing, passing, passing, cutoff, math.nextafter(cutoff, farther), passing)
    rows = [replace(row, **{feature: value}) for row, value in zip(samples(scores, (0,) * 7), shape, strict=True)]
    rows[-1] = replace(rows[-1], alive_wolves=1)
    decimals = [Decimal.from_float(row.score) for row in rows]
    pos = np.zeros(len(rows), dtype=bool)
    gate = study.DISTRIBUTION_GATES[method]
    low, additional = study._negative_masks(rows, decimals, pos, Decimal("0.1"), gate)
    assert not low.any()
    assert additional.tolist() == [False, True, True, False, equality_passes, False, False]

    hit_rows = [replace(row, y=1, speaker_hit=True, successful_empty=False) for row in rows]
    empty_rows = [replace(row, successful_empty=True) for row in rows]
    for changed_reports in (hit_rows, empty_rows):
        _, changed_additional = study._negative_masks(changed_reports, decimals, pos, Decimal("0.1"), gate)
        assert np.array_equal(changed_additional, additional)


def test_distribution_gives_pos_priority_and_keeps_strict_decimal_low_cut():
    rows = samples((0.3, 0.3125, 0.375, math.nextafter(0.375, 1.0)), (0, 0, 0, 0))
    rows = [replace(row, uniform_tv=0.07) for row in rows]
    scores = [Decimal.from_float(row.score) for row in rows]
    pos = np.asarray([score > Decimal("0.375") for score in scores])
    gate = study.DISTRIBUTION_GATES["TV-0.10"]
    low, additional = study._negative_masks(rows, scores, pos, Decimal("0.3125"), gate)
    assert pos.tolist() == [False, False, False, True]
    assert low.tolist() == [True, False, False, False]
    assert additional.tolist() == [False, True, True, False]

    # 0.3 的 float 小于精确 Decimal("0.3")，同时仍满足 float 窗口下界。
    low, additional = study._negative_masks(rows, scores, pos, Decimal("0.3"), gate)
    assert low.tolist() == [True, False, False, False]
    assert not additional[0]


@pytest.mark.parametrize("valid_pairs", (19, 18))
def test_distribution_decomposes_neg_and_resamples_the_same_whole_games(monkeypatch, valid_pairs):
    rows = samples((0.05, 0.10, 0.32, 0.33, 0.50, 0.25, 0.34, 0.36), (0, 1, 0, 1, 1, 0, 0, 0),
                   ("long",) * 6 + ("short",) * 2)
    rows = [replace(row, alive_wolves=1) if i in (0, 1, 5, 7) else row for i, row in enumerate(rows)]
    rows[2] = replace(rows[2], uniform_tv=0.04, normalized_entropy=0.995, successful_empty=True)
    rows[3] = replace(rows[3], uniform_tv=0.075)
    rows[6] = replace(rows[6], uniform_tv=0.07, day=2, phase="pk_discussion")
    calls = []

    def draws(games, replicates, seed):
        calls.append((games, replicates, seed))
        return np.asarray([[0, 0]] * 9 + [[0, 1]] * (valid_pairs - 9) + [[1, 1]] * (20 - valid_pairs))

    monkeypatch.setattr(study, "bootstrap_indices", draws)
    monkeypatch.setattr(study.frozen, "BOOTSTRAP_REPLICATES", 20)
    result = study.evaluate_distribution(rows)
    assert calls == [(["long", "short"], 20, 1729)]
    rules = result["operating_points"]["0.15"]["rules"]
    direct, tv = rules["Direct"], rules["TV-0.10"]
    assert result["shared_pos"]["region"]["correct_mass"] == pytest.approx(1 / 12)
    assert direct["risk_budgets"]["NEG"]["empirical_met"] is True
    assert tv["risk_budgets"]["NEG"]["empirical_met"] is False
    neg = tv["overall"]["regions"]["NEG"]
    assert (neg["mass"], neg["error_rate"], neg["error_mass"], neg["correct_mass"]) == pytest.approx(
        (7 / 12, 2 / 7, 1 / 6, 5 / 12),
    )
    assert (neg["records"], neg["games"], neg["error_games"]) == (5, 2, 1)
    low, additional = (tv["neg_subregions"][name] for name in ("L", "A"))
    assert low["mass"] == pytest.approx(1 / 6) and low["error_mass"] == pytest.approx(1 / 12)
    assert additional["mass"] == pytest.approx(5 / 12) and additional["error_rate"] == pytest.approx(0.2)
    assert (additional["successful_empty_mass"], additional["nonempty_team_miss_mass"],
            additional["team_hit_mass"]) == pytest.approx((1 / 12, 1 / 4, 1 / 12))
    for field in ("mass_units", "error_units"):
        assert neg[field] == low[field] + additional[field]

    single = tv["subgroups"]["single_wolf"]
    assert single["population_mass"] == 0.5
    assert single["coverage"] == pytest.approx(1 / 3)
    assert single["regions"]["NEG"]["error_rate"] == 0.5
    assert tv["subgroups"]["pk"]["population_mass"] == 0.25
    assert tv["subgroups"]["pk"]["coverage"] == 1

    change = tv["relative_to_direct"]
    assert change["added"]["mass"] == pytest.approx(0.5)
    assert change["removed"]["records"] == 0 and change["removed"]["error_rate"] is None
    assert change["added_by_region"]["L"]["error_mass"] == pytest.approx(1 / 12)
    assert change["added_by_region"]["A"] == additional
    assert change["net_coverage_gain"] == pytest.approx(0.5)
    paired = change["paired_differences"]
    assert paired["coverage"]["percentile_95"] == pytest.approx([0.5, 0.5])
    assert paired["neg_error_mass"]["point_estimate"] == pytest.approx(1 / 6)
    assert paired["neg_error_mass"]["valid_replicates"] == 20
    assert paired["neg_error_rate"]["point_estimate"] == pytest.approx(2 / 7)
    assert paired["neg_error_rate"]["valid_replicates"] == valid_pairs
    assert paired["neg_error_rate"]["zero_denominator_replicates"] == 20 - valid_pairs
    if valid_pairs == 19:
        assert paired["neg_error_rate"]["percentile_95"] == pytest.approx([2 / 7, 0.5])
    else:
        assert paired["neg_error_rate"]["percentile_95"] is None
    assert result["shared_pos"]["paired_error_rate_difference"]["point_estimate"] == 0
    for op in result["operating_points"].values():
        old = study.evaluate_candidate(rows, op["rules"]["Direct"]["beta"], str(study.DISTRIBUTION_ALPHA))
        for rule in op["rules"].values():
            assert rule["overall"]["regions"]["POS"] == result["shared_pos"]["region"]
        for key, region in old["overall"]["regions"].items():
            new = op["rules"]["Direct"]["overall"]["regions"][key]
            assert {name: new[name] for name in region} == region
        assert op["rules"]["Direct"]["overall"]["coverage"] == old["overall"]["coverage"]


def test_distribution_change_counts_withdrawals_in_coverage_and_error_mass(monkeypatch):
    monkeypatch.setattr(study.frozen, "BOOTSTRAP_REPLICATES", 20)
    monkeypatch.setattr(study, "DISTRIBUTION_BETAS", {"0.15": {"Direct": "0.2", "TV-0.10": "0.04"}})
    rows = samples((0.10, 0.33, 0.33, 0.5), (1, 0, 0, 1), ("g",) * 4)
    rows[1:3] = [replace(row, uniform_tv=0.04) for row in rows[1:3]]
    result = study.evaluate_distribution(rows)["operating_points"]["0.15"]["rules"]["TV-0.10"]
    change = result["relative_to_direct"]
    assert change["added"]["mass"] == 0.5 and change["added"]["error_rate"] == 0
    assert change["removed"]["mass"] == 0.25 and change["removed"]["error_rate"] == 1
    assert change["net_coverage_gain"] == 0.25
    assert change["paired_differences"]["neg_error_mass"]["point_estimate"] == -0.25


def test_budget_at_exact_empirical_boundary_can_have_an_interval_above_budget():
    region = {"mass_units": 20, "error_units": 3}
    metric = study._bootstrap_metric(0.15, np.asarray([0.1, 0.2]))
    assert study._budget_summary(region, "0.15", metric) == {
        "development_budget": "0.15", "empirical_met": True, "interval_upper_within_budget": False,
    }


def test_frozen_distribution_cli_uses_only_fold4_and_keeps_exact_parameters(
    mocked_development, monkeypatch, capsys,
):
    def forbidden(*_, **__):
        pytest.fail("冻结分布评价不应执行开发选择")

    for function in ("analyze_development", "direction_cuts", "maximum_coverage"):
        monkeypatch.setattr(study, function, forbidden)
    assert study.main(["evaluate-distribution"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert mocked_development.opened == ["game-4"]
    assert result["stage"] == "exploratory_fold4_distribution_evaluation" and result["folds"] == [4]
    assert result["weighting"] == "game_equal"
    assert result["primary_comparison"] == {
        "neg_budget": "0.15", "pos_budget": "0.10", "reference": "Direct", "candidate": "TV-0.10",
    }
    assert result["source"]["q_predictions_sha256"] == study.frozen.Q_SHA256
    shared_pos = result["shared_pos"]
    assert shared_pos["alpha"] == "0.387551836669445037841796875"
    assert shared_pos["matched_all_rules"] and shared_pos["region"]["error_rate"] is None
    assert shared_pos["risk_budget"]["empirical_met"] is False
    assert shared_pos["bootstrap"]["zero_denominator_replicates"] == 5000
    assert shared_pos["paired_error_rate_difference"]["percentile_95"] is None
    expected_betas = {
        "0.15": ("0.079652510583400726318359375", "0.124301455914974212646484375",
                 "0.1209847293794155120849609375", "0.12324838340282440185546875"),
        "0.20": ("0.1065414138138294219970703125", "0.17849795520305633544921875",
                 "0.183103973977267742156982421875", "0.18366961181163787841796875"),
    }
    assert set(result["operating_points"]) == set(expected_betas)
    for budget, op in result["operating_points"].items():
        rules = op["rules"]
        assert tuple(rules) == ("Direct", "TV-0.05", "TV-0.10", "Entropy-0.99")
        assert tuple(rule["beta"] for rule in rules.values()) == expected_betas[budget]
        assert all(rule["alpha"] == shared_pos["alpha"] for rule in rules.values())
        assert rules["Direct"]["risk_budgets"]["NEG"]["empirical_met"] is False
        assert rules["TV-0.10"]["risk_budgets"]["NEG"] == {
            "development_budget": budget, "empirical_met": True, "interval_upper_within_budget": True,
        }
        assert rules["TV-0.10"]["relative_to_direct"]["paired_differences"]["neg_error_rate"] == {
            "point_estimate": None, "percentile_95": None, "valid_replicates": 0, "replicates": 5000,
            "zero_denominator_replicates": 5000,
        }


@pytest.mark.parametrize("arguments", (
    ("evaluate", "--beta", "0.7", "--alpha", "0.3"),
    ("evaluate", "--beta", "NaN", "--alpha", "1"),
    ("develop", "--neg-risk", "0.3"),
))
def test_cli_rejects_invalid_operating_points_before_reading_frozen_data(arguments, monkeypatch):
    def forbidden(*_, **__):
        pytest.fail("参数非法时不应读冻结资产")

    monkeypatch.setattr(study, "load_development", forbidden)
    with pytest.raises(SystemExit) as error:
        study.main(arguments)
    assert error.value.code == 2
