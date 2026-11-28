"""合成数据上的双层一致性、精确搜索及产物联结测试。"""

from collections import Counter
from dataclasses import replace
from fractions import Fraction
from itertools import product
import json
import random
import sys

import numpy as np
import pytest

from scripts import analyze_observer_dtrs_report_state_risk as study
from tests.observer_dtrs.test_sequential import ATTEMPT, TEAM, pairs_for, trajectory
from tests.observer_dtrs.test_team_event import mocked_development  # noqa: F401
from werewolf.artifact_io import canonical_json_bytes, canonical_jsonl_bytes, publish_artifact
from werewolf.observer_dtrs import report_state_decision as decision
from werewolf.observer_dtrs import report_state_probability as probability
from werewolf.observer_dtrs.report_state_decision import BND, NEG, POS, Thresholds
from werewolf.observer_dtrs.probabilities import ProbabilityMatrix


@pytest.fixture
def base():
    game, _ = trajectory()
    return pairs_for(game)[0].start


def sample(context, state):
    f = (0.,) * probability.DUAL_COLUMN + (float(context.wolf_count == 2), 0., 0.)
    return study.probability_study.Sample(context, ATTEMPT, f, state,
                                          "error" if state is None else "success", state == 0, ())


def prediction(base, values, index, *, game_id=None):
    ps, pT, truth = values
    context = replace(base, game_id=game_id or base.game_id, boundary_id=f"pre-{index}", step_index=index)
    return study.Prediction(sample(context, truth), ps, pT)


def test_thresholds_are_inclusive_and_only_allow_the_six_ordered_pairs():
    rule = Thresholds.shared(.25, .75)
    probabilities = (0, .25, np.nextafter(.25, 1), .5, np.nextafter(.75, 0), .75, 1)
    actual = {"".join(d[0] for d in rule.regions(ps, pt))
              for ps, pt in product(probabilities, repeat=2) if ps <= pt}
    assert actual == set(decision.JOINT_REGIONS)
    assert rule.regions(.25, .25) == (NEG, NEG)
    assert rule.regions(.75, .75) == (POS, POS)
    assert rule.regions(np.nextafter(.25, 1), np.nextafter(.75, 0)) == (BND, BND)
    assert all(rule.regions(p, p)[0] == rule.regions(p, p)[1] for p in probabilities)


@pytest.mark.parametrize("parameters", ((.1, .8, .2, .8), (.2, .7, .1, .8),
                                         (.5, .9, .2, .9), (.2, .5, .2, .5)))
def test_illegal_four_thresholds_rejected(parameters):
    with pytest.raises(ValueError):
        Thresholds(*parameters)


def test_four_thresholds_preserve_order_and_do_not_apply_to_single_wolf(base):
    rule = Thresholds(.25, .875, .125, .625)
    grid = (0, .125, .25, .5, .625, .75, .875, 1)
    for ps, pT in product(grid, repeat=2):
        if ps <= pT:
            ds, dT = rule.regions(ps, pT)
            assert decision.REGIONS.index(ds) <= decision.REGIONS.index(dT)
    single = replace(base, alive_player_ids=tuple(s for s in base.alive_player_ids if s != TEAM[1]))
    population = study.Population([prediction(single, (.4, .4, 0), 0)])
    with pytest.raises(ValueError, match="双狼人口"):
        study.evaluate_rule(population, rule, study.BUDGETS["main"], subgroups=False)


def exhaustive(rows, direction, budget, shared):
    """直接按每一对阈值和每一行判定，不使用累计表或前缀最优。"""
    negate = direction == NEG
    grid = {0. if negate else 1.}
    grid.update(p for row in rows for p in (row.ps, row.pT) if (p < .5 if negate else p > .5))
    counts = Counter(row.game_id for row in rows)
    scored = []
    for s, t in product(grid, repeat=2):
        if t > s or (shared and s != t):
            continue
        masses, errors = [], []
        for theta, member, event_states in ((s, "ps", {2}), (t, "pT", {1, 2})):
            selected = [row for row in rows if (getattr(row, member) <= theta if negate else getattr(row, member) >= theta)]
            masses.append(sum(Fraction(1, counts[row.game_id]) for row in selected))
            errors.append(sum(Fraction(1, counts[row.game_id]) for row in selected
                              if (row.sample.state in event_states) == negate))
        if all(mass > 0 and error <= budget * mass for mass, error in zip(masses, errors, strict=True)):
            scored.append((-sum(masses), s if negate else -s, t if negate else -t, (s, t)))
    return min(scored)[-1] if scored else None


def test_prefix_search_matches_independent_small_exhaustive_search(base):
    rng = random.Random(559)
    grid = (0., .125, .25, np.nextafter(.5, 0).item(), .5, np.nextafter(.5, 1).item(), .75, .875, 1.)
    for _ in range(36):
        rows = [prediction(base, (*sorted(rng.choices(grid, k=2)), rng.randrange(3)), i, game_id=f"game-{i % 3}")
                for i in range(8)]
        cuts = study.direction_cuts(rows)
        for d, limit, shared in product((NEG, POS), (Fraction(1, 10), Fraction(1, 2)), (True, False)):
            actual = study.choose_direction(cuts[d], d, limit, shared=shared)
            assert actual["thresholds"] == exhaustive(rows, d, limit, shared)


def test_constrained_can_expand_coverage_on_the_same_double_population(base):
    values = ((.125, .25, 0), (.625, .75, 1), (.75, .875, 2), (.125, .625, 1))
    rows = [prediction(base, item, i) for i, item in enumerate(values)]
    rules = study.select_configurations(rows)["main"]
    shared = Thresholds(**rules["shared_double"]["thresholds"])
    constrained = Thresholds(**rules["constrained_double"]["thresholds"])
    assert shared.alpha_s == shared.alpha_T == .75
    assert constrained.alpha_s == .75 and constrained.alpha_T == .625
    pop = study.Population(rows)
    one = study.evaluate_rule(pop, shared, rules["budgets"], subgroups=False)["overall"]
    two = study.evaluate_rule(pop, constrained, rules["budgets"], subgroups=False)["overall"]
    assert one["all_four_budgets_met"] and two["all_four_budgets_met"]
    assert two["mean_coverage"] > one["mean_coverage"]


def test_infeasibility_distinguishes_zero_support_excess_error_and_order(base):
    def negative(values, shared):
        cuts = study.direction_cuts([prediction(base, item, i) for i, item in enumerate(values)])
        return study.choose_direction(cuts[NEG], NEG, Fraction(1, 10), shared=shared)

    assert negative(((.5, .5, 0),), True)["event_failures"]["speaker_hit"] == "no_positive_support"
    assert negative(((.125, .25, 2),), True)["event_failures"]["team_hit"] == "risk_budget_unmet"
    values = ((.125, .375, 0), (.25, .875, 2))
    assert negative(values, True)["reason"] == "no_common_feasible_threshold"
    ordered = negative(values, False)
    assert ordered["reason"] == "threshold_order_infeasible" and ordered["thresholds"] is None


def test_exact_budget_and_binary_ties_and_undefined_bootstrap_replicates(base):
    # 同一局中重复四条误判只占该局一次权重。
    rows = [prediction(base, (0, 0, 2), i, game_id="long") for i in range(4)]
    rows.append(prediction(base, (.125, .125, 0), 0, game_id="short"))
    tested = study.evaluate_rule(study.Population(rows), Thresholds.shared(.125, 1),
                                 {NEG: ".50", POS: ".10"}, subgroups=True)["overall"]
    neg = tested["events"]["speaker_hit"]["regions"][NEG]
    assert neg["risk"] == .5 and neg["empirical_budget_met"] is True
    assert (neg["records"], neg["games"], neg["game_equal_mass"]) == (5, 2, 1.)
    assert tested["events"]["team_hit"]["regions"][POS]["risk"] is None
    assert tested["all_four_budgets_met"] is False

    # Binary 的 0.5 平局仍为 NEG。只有一局进入 NP，空重采样保持未定义。
    rows = [prediction(base, (.5, .875, 1), 0, game_id="a"),
            prediction(base, (.75, .875, 2), 0, game_id="b")]
    draws = np.asarray([[0, 0], [1, 1], [0, 1]])
    tested = study.evaluate_rule(study.Population(rows), None, study.BUDGETS["main"],
                                 subgroups=False, draws=draws)["overall"]
    assert tested["joint_regions"]["NP"]["records"] == 1
    assert tested["np_e1_rate"]["value"] == 1 and tested["np_error_risk"]["value"] == 0
    ci = tested["np_error_risk"]["bootstrap"]
    assert ci["zero_denominator_replicates"] == 1 and ci["valid_replicates"] == 2
    assert ci["percentile_95"] is None


def test_empty_population_records_undefined_risk_and_coverage():
    tested = study.evaluate_rule(study.Population([]), Thresholds.shared(.1, .9), study.BUDGETS["main"],
                                 subgroups=True)["overall"]
    assert tested["mean_coverage"] is None
    assert tested["np_e1_rate"]["value"] is None
    assert tested["events"]["speaker_hit"]["regions"][NEG]["risk"] is None
    assert tested["all_four_budgets_met"] is False


def test_sequential_counts_failure_and_changed_team_without_skipping_opportunities():
    game, prefixes = trajectory()
    pairs = pairs_for(game)
    rows = []
    for pair in pairs:
        context = pair.start
        missing = context.boundary_id == prefixes[2].boundary_id and context.observer_id == "player3"
        value = .5 if context.phase == "discussion" and context.day == 1 else .875 if context.day == 1 else .125
        state = None if missing else 2 if context.day == 2 else 0
        rows.append(study.Prediction(sample(context, state), value, value))
    results = study.sequential_evaluation(rows, pairs, Thresholds.shared(.25, .75))
    population = results["population"]
    assert population["initial_report_unobserved_records"] == 1
    assert population["next_opportunity"]["records"] == 14
    assert population["next_successful_report"]["records"] == 13
    assert population["next_report_unobserved_statuses"] == {"error": 1}
    assert population["no_next"]["records"] == 10
    same = results["by_team_change"]["same_team"]["events"]["speaker_hit"]
    transition = same["transitions_given_next_prediction"][BND][POS]
    assert transition["value"] == 1 and transition["numerator"]["records"] == 10
    assert same["bnd_resolution_commitment_error"]["denominator"]["records"] == 9
    changed = results["by_team_change"]["changed_team"]["events"]["team_hit"]
    assert changed["transitions_given_next_prediction"][POS][NEG]["numerator"]["records"] == 4
    assert changed["next_commitment_error_by_direction"][NEG]["value"] == 1


def test_shared_loader_visits_verified_games_once(mocked_development):
    visited = []

    def visit(fold, game, wolves, *, publication_digest, attempt_id):
        visited.append((fold, game.game_id, wolves, publication_digest, attempt_id))

    parts, lineage = study.development.load_development((1, 4), visit_game=visit)
    assert list(parts) == [1, 4]
    assert visited == [(fold, f"game-{fold}", TEAM, lineage["publication_digest"], ATTEMPT) for fold in (1, 4)]


@pytest.fixture
def development_data(base):
    game, prefixes = trajectory()
    pairs = pairs_for(game)
    oof, partitions, selections = [], {}, {}
    for fold in study.development.SELECTION_FOLDS:
        training = [n for n in study.development.SELECTION_FOLDS if n != fold]
        method = probability.HIERARCHICAL if fold % 2 else probability.MULTINOMIAL
        value = .01 if fold % 2 else .001
        selections[str(fold)] = {"selection_folds": training, "selected_method": method, "best_lambda": {method: value}}
        rows = [prediction(base, item, i, game_id=f"selection-{fold}") for i, item in enumerate(
            ((.125, .25, 0), (.125, .875, 1), (.875, .875, 2), (.25, .5, None)))]
        partitions[fold] = [row.sample for row in rows]
        for row in rows:
            context = row.sample.context
            oof.append({"fold": fold, "method": method, "lambda": value, "attempt_id": ATTEMPT,
                        "game_id": context.game_id, "boundary_id": context.boundary_id, "observer_id": context.observer_id,
                        "prefix_digest": context.prefix_digest, "step_index": context.step_index,
                        "acting_wolf": context.acting_wolf, "alive_wolf_ids": context.alive_wolf_ids,
                        **probability.probability_record((1 - row.pT, row.pT - row.ps, row.ps))})

    n = len(probability.FEATURES)
    scaler = {"mean": [0.] * n, "std": [1.] * n}
    model = {"method": probability.HIERARCHICAL, "features": probability.FEATURES, "lambda": .01,
             "heads": {"speaker": {"intercept": [-5.], "coefficients": [[10.], *([[0.]] * (n - 1))],
                                   "standardization": scaler},
                       "teammate_given_no_speaker": {"intercept": [-5.], "coefficients": [[0.]] * n,
                                                      "standardization": scaler}}}
    partitions[4] = [sample(pair.start, 0) for pair in pairs]
    return partitions, oof, selections, model, game, prefixes, pairs


@pytest.mark.parametrize(("field", "value"), (("fold", 4), ("attempt_id", "another"),
                                              ("step_index", 900), ("method", "wrong"), ("lambda", 1)))
def test_oof_rejects_misbound_rows(development_data, field, value):
    partitions, records, selections, *_ = development_data
    modified = [{**records[0], field: value}, *records[1:]]
    with pytest.raises(ValueError, match="来源不匹配"):
        study.match_oof(partitions, modified, selections)


def test_match_requires_complete_oof_including_unobserved_rows(development_data):
    partitions, records, selections, *_ = development_data
    joined, used = study.match_oof(partitions, records, selections)
    assert len(joined) == 16 and len(study._supervised(joined)) == 12
    assert used[0]["method"] != used[1]["method"]
    with pytest.raises(ValueError, match="缺少"):
        study.match_oof(partitions, records[:-1], selections)
    with pytest.raises(ValueError, match="重复"):
        study.match_oof(partitions, [*records, records[0]], selections)


def test_fold4_cannot_reselect_thresholds(monkeypatch, development_data):
    partitions, records, selections, model, _, _, pairs = development_data
    monkeypatch.setattr(study, "BOOTSTRAP_REPLICATES", 10)
    rules, results = study.run_study(partitions, records, selections, model, pairs)
    changed = {**partitions, 4: [replace(row, state=2, successful_empty=False) for row in partitions[4]]}
    other_rules, other = study.run_study(changed, records, selections, model, pairs)
    assert other_rules == rules
    assert other["evaluation"]["configurations"]["main"]["shared"]["oof"] == (
        results["evaluation"]["configurations"]["main"]["shared"]["oof"])
    assert other["evaluation"]["configurations"]["main"]["shared"]["fold4"] != (
        results["evaluation"]["configurations"]["main"]["shared"]["fold4"])


def test_publish_and_read_back_from_frozen_probability_artifact(tmp_path, monkeypatch, development_data):
    partitions, records, selections, model, game, prefixes, _ = development_data
    lineage = {"publication_digest": "publication", "fold_manifest_digest": "folds", "q_manifest_digest": "frozen-q"}
    previous = tmp_path / "report-state-probability"
    probability_archive = publish_artifact(previous, manifest_fields={
        "artifact_type": probability.ARTIFACT_TYPE, "schema_version": probability.ARTIFACT_TYPE,
        "artifact_identity": previous.name, **lineage,
    }, files={"models.json": canonical_json_bytes({"selected_method": model["method"], "models": {model["method"]: model}}),
              "results.json": canonical_json_bytes({"outer_selection": selections,
                  "selection": {"selected_method": model["method"], "best_lambda": {model["method"]: model["lambda"]}}}),
              "oof_predictions.jsonl": canonical_jsonl_bytes(records)})
    destination = tmp_path / "report-state-risk"
    monkeypatch.setattr(study.probability_study, "DESTINATION", previous)
    monkeypatch.setattr(study, "DESTINATION", destination)
    monkeypatch.setattr(study, "BOOTSTRAP_REPLICATES", 10)
    monkeypatch.setattr(sys, "argv", ["report_state_risk"])

    def loader(folds, *, build_samples, visit_game):
        assert folds == study.development.DEVELOPMENT_FOLDS and build_samples is study.probability_study.pre_samples
        visit_game(4, game, TEAM, publication_digest=lineage["publication_digest"], attempt_id=ATTEMPT)
        return partitions, lineage

    monkeypatch.setattr(study.development, "load_development", loader)
    study.main()
    rules = json.loads((destination / "rules.json").read_text())
    results = json.loads((destination / "results.json").read_text())
    assert set(rules) == set(study.BUDGETS) and rules["main"]["shared"]["feasible"]
    assert results["oof_probability_selections"]["0"]["method"] == probability.MULTINOMIAL
    assert results["population"]["oof"]["unobserved_records"] == 4
    assert results["sequential_fold4_main_shared"]["population"]["no_next"]["records"] == 10
    with pytest.raises(ValueError, match="不匹配"):
        decision.load_shared_rule(destination, probability_manifest_digest="wrong-model")

    rule = decision.load_shared_rule(destination, probability_manifest_digest=probability_archive.manifest_digest)
    for pre, mass, direction in ((prefixes[0], .01, NEG), (prefixes[1], .5, BND),
                                 (prefixes[2], .95, POS), (prefixes[4], .01, NEG)):
        seats = study.development.frozen.PLAYER_IDS
        q = [[0 if observer == target else 1 / 6 if observer == pre.current_speaker else
              mass if target == pre.current_speaker else (1 - mass) / 5
              for target in seats] for observer in seats]
        matrix = ProbabilityMatrix(pre.game_id, pre.boundary_id, pre.prefix_digest, q)
        panel = probability.predict_pre(pre, TEAM, matrix, model)
        interpreted = decision.decide_pre(panel, rule)
        assert interpreted["step_index"] == pre.step_index
        assert interpreted["alive_wolf_ids"] == panel["alive_wolf_ids"]
        observer = interpreted["observers"]["player3"]
        assert observer["ds"] == observer["dT"] == direction
        assert observer["permitted_premises"] == ({} if direction == BND else dict.fromkeys(decision.EVENTS, direction == POS))
        assert observer["unresolved_premises"] == (list(decision.EVENTS) if direction == BND else [])
    with pytest.raises(FileExistsError):
        study.main()


def test_online_only_loads_a_feasible_shared_main_rule(tmp_path):
    destination = tmp_path / "report-state-risk"
    publish_artifact(destination, manifest_fields={
        "artifact_type": decision.ARTIFACT_TYPE, "schema_version": decision.ARTIFACT_TYPE,
        "artifact_identity": destination.name, "probability_manifest_digest": "model",
    }, files={"rules.json": canonical_json_bytes({"main": {"shared": {"feasible": False, "thresholds": None}}})})
    with pytest.raises(ValueError, match="没有可发布"):
        decision.load_shared_rule(destination, probability_manifest_digest="model")
