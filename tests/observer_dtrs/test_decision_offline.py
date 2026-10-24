"""Manually solvable, unequal-game-length examples; no formal data access."""
from fractions import Fraction as F

import pytest

from werewolf.observer_dtrs.decision import LossSpec, Region, TiePolicy
from werewolf.observer_dtrs.decision_offline import (
    ProbabilityInput, evaluate, evaluate_stability, realized_loss,
)


@pytest.fixture
def controlled():
    rows = (ProbabilityInput('a', 'game-A', F(9, 10)), ProbabilityInput('b1', 'game-B', F(1, 10)),
            ProbabilityInput('b2', 'game-B', F(1, 2)), ProbabilityInput('b3', 'game-B', F(9, 10)))
    targets = {'a': 0, 'b1': 1, 'b2': 1, 'b3': 1}
    spec = dict(loss=LossSpec(1, 1, F(1, 4), F(1, 4), TiePolicy.TERMINAL_POS_FIRST),
                game_sizes={'game-A': 1, 'game-B': 3}, weighting='game_equal',
                binary_tie_policy=TiePolicy.TERMINAL_NEG_FIRST)
    return rows, targets, spec


def test_manual_risk_masses_and_baselines(controlled):
    rows, targets, spec = controlled
    result = evaluate(rows, targets, **spec)
    assert result['realized_risk'] == pytest.approx(F(17, 24))
    assert result['predicted_minimum_risk'] == pytest.approx(F(1, 8))
    assert result['decided_coverage'] == pytest.approx(F(5, 6))
    assert result['false_positive_joint'] == .5
    assert result['false_negative_joint'] == pytest.approx(F(1, 6))
    assert result['selective_error_rate'] == .8
    assert result['regions']['POS']['proportion'] == pytest.approx(F(2, 3))
    assert result['regions']['BND']['proportion'] == pytest.approx(F(1, 6))
    assert result['fixed_action_risks'] == {'POS': .5, 'BND': .25, 'NEG': .5}
    assert result['forced_binary']['realized_risk'] == pytest.approx(F(5, 6))
    assert sum(v['loss_contribution'] for v in result['regions'].values()) == pytest.approx(result['realized_risk'])
    record = evaluate(rows, targets, **{**spec, 'weighting': 'record_equal'})
    assert record['realized_risk'] == F(9, 16)


def test_changed_supervision_never_changes_decisions(controlled):
    rows, targets, spec = controlled
    before = evaluate(rows, targets, **spec)
    after = evaluate(rows, {s: 1-y for s, y in targets.items()}, **spec)
    assert before['decisions'] == after['decisions']
    assert before['realized_risk'] != after['realized_risk']
    assert before['predicted_minimum_risk'] == after['predicted_minimum_risk']


def test_subgroups_keep_full_game_denominator(controlled):
    rows, targets, spec = controlled
    group = evaluate(rows, targets, **spec, subgroup_ids=('a', 'b1'))
    assert group['mass'] == pytest.approx(F(4, 3))
    assert group['regions']['POS']['proportion'] == .75
    assert group['realized_risk'] == 1
    empty = evaluate(rows, targets, **spec, subgroup_ids=())
    assert empty['rows'] == 0 and empty['realized_risk'] is None
    assert empty['selective_error_rate'] is None


def test_single_class_and_all_bnd_have_defined_na_metrics():
    rows = (ProbabilityInput('x', 'g', .5),)
    result = evaluate(rows, {'x': 1}, loss=LossSpec(1, 1, F(1, 4), F(1, 4), TiePolicy.DEFER_FIRST),
                      game_sizes={'g': 1}, weighting='game_equal', binary_tie_policy=TiePolicy.TERMINAL_NEG_FIRST)
    assert result['realized_risk'] == .25 and result['decided_coverage'] == 0
    assert result['POS_error_rate'] is None and result['NEG_error_rate'] is None
    assert result['selective_error_rate'] is None and result['false_positive_rate_given_Y0'] is None


def test_complete_population_and_label_contracts(controlled):
    rows, targets, spec = controlled
    for bad in ({**targets, 'a': True}, {**targets, 'a': 2}, {k:v for k,v in targets.items() if k != 'a'}):
        with pytest.raises(ValueError):
            evaluate(rows, bad, **spec)
    for bad_rows in (rows+rows[:1], rows[:-1], (object(),), ()):
        with pytest.raises(ValueError):
            evaluate(bad_rows, targets, **spec)
    with pytest.raises(ValueError):
        evaluate(rows, targets, **{**spec, 'game_sizes': {'game-A': True, 'game-B': 3}})
    with pytest.raises(ValueError):
        evaluate(rows, targets, **spec, subgroup_ids=('a', 'a'))
    with pytest.raises(ValueError):
        evaluate(rows, targets, **spec, subgroup_ids=('unknown',))


def test_realized_six_losses_are_in_the_explicit_offline_module(controlled):
    m = controlled[2]['loss']
    assert [realized_loss(a, y, m) for a in Region for y in (1, 0)] == [0, 1, F(1, 4), F(1, 4), 1, 0]
    with pytest.raises(ValueError):
        realized_loss(Region.POS, True, m)


def test_stability_is_label_free_and_game_weighted():
    rows = (ProbabilityInput('a', 'A', F(1, 4)),
            *(ProbabilityInput(f'b{i}', 'B', F(9, 10)) for i in range(3)))
    m = LossSpec(1, 1, F(1, 4), F(1, 4), TiePolicy.TERMINAL_NEG_FIRST)
    report = evaluate_stability(rows, loss=m, game_sizes={'A': 1, 'B': 3}, weighting='game_equal',
                               probability_radii=(0, F(1, 100)), cost_alternatives=(m.scaled(2),))
    assert report['probability_scenarios'][0]['unstable_quality_fraction'] == 0
    assert report['probability_scenarios'][1]['unstable_quality_fraction'] == .5
    assert report['cost_scenarios'][0]['flipped_quality_fraction'] == 0
    with pytest.raises(TypeError):
        evaluate_stability(rows, loss=m, game_sizes={'A': 1, 'B': 3}, weighting='game_equal',
                           probability_radii=(0,), cost_alternatives=(), targets={})
