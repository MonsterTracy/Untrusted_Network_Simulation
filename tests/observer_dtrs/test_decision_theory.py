"""Controlled math for the draft D3a protocol, independent of D2 artifacts."""
from fractions import Fraction as F

import pytest

from scripts.check_observer_dtrs_decision_theory import (
    ControlledLoss, check_theory, direct_decision, minimizing_actions, threshold_decision,
    interval_decisions, interval_regret_bound, regret, risks,
)


@pytest.fixture
def symmetric():
    return ControlledLoss(F(1, 2), F(1, 2), F(1, 8), F(1, 8))


def test_exact_crossings_and_boundary_ties(symmetric):
    assert symmetric.thresholds == (F(3, 4), F(1, 4), F(1, 2))
    assert minimizing_actions(F(1, 4), symmetric) == ('BND', 'NEG')
    assert minimizing_actions(F(3, 4), symmetric) == ('BND', 'POS')
    assert direct_decision(0, symmetric) == 'NEG'
    assert direct_decision(1, symmetric) == 'POS'
    assert direct_decision(F(1, 4), symmetric) == 'BND'


@pytest.mark.parametrize('cost,expected', [
    (F(1, 4), 'BND'), (F(1, 2), 'BND'), (F(3, 5), 'NEG')])
def test_ordering_alone_does_not_guarantee_bnd(cost, expected):
    loss = ControlledLoss(1, 1, cost, cost)
    assert direct_decision(F(1, 2), loss) == expected
    if cost > F(1, 2):
        assert all('BND' not in minimizing_actions(F(j, 100), loss) for j in range(101))
    if cost == F(1, 2):
        assert minimizing_actions(F(1, 2), loss) == ('BND', 'NEG', 'POS')


def test_zero_deferral_is_an_explicit_degeneracy():
    loss = ControlledLoss(1, 1, 0, 0)
    assert all(direct_decision(F(j, 100), loss) == 'BND' for j in range(101))


@pytest.mark.parametrize('p', [F(-1, 100), F(101, 100), float('nan'), float('inf'), True])
def test_invalid_probability_is_rejected(symmetric, p):
    with pytest.raises((ValueError, OverflowError)):
        risks(p, symmetric)


@pytest.mark.parametrize('values', [(0, 1, 0, 0), (1, 0, 0, 0), (1, 1, -1, 0),
                                         (1, 1, 1, 0), (1, 1, 0, 1), (True, 1, 0, 0)])
def test_invalid_or_unordered_costs_are_rejected(values):
    with pytest.raises(ValueError):
        ControlledLoss(*values)


def test_probability_stability_is_separate_from_bnd(symmetric):
    assert interval_decisions(F(1, 4), 0, symmetric) == ('BND',)
    assert interval_decisions(F(1, 4), F(1, 100), symmetric) == ('BND', 'NEG')
    assert interval_decisions(F(1, 2), F(1, 100), symmetric) == ('BND',)
    assert interval_decisions(F(99, 100), F(1, 100), symmetric) == ('POS',)
    assert interval_regret_bound(F(1, 2), F(1, 100), symmetric) == 0


def test_lower_expected_brier_can_have_larger_decision_regret(symmetric):
    eta, first, second = F(74, 100), F(76, 100), F(70, 100)
    assert (first - eta) ** 2 < (second - eta) ** 2
    assert regret(first, eta, symmetric) == F(1, 200)
    assert regret(second, eta, symmetric) == 0


def test_false_positive_asymmetry_moves_both_thresholds_up():
    equal = ControlledLoss(1, 1, F(1, 4), F(1, 4))
    costly_fp = ControlledLoss(4, 1, F(1, 4), 1)
    assert costly_fp.thresholds[0] > equal.thresholds[0]
    assert costly_fp.thresholds[1] > equal.thresholds[1]
    assert threshold_decision(F(7, 10), costly_fp) == direct_decision(F(7, 10), costly_fp)


def test_controlled_property_grid():
    report = check_theory()
    assert report['proposed_nondegenerate_scenarios'] == 45
    assert report['regret_bound_checks'] == 50 * 21 * 21
    assert report['formal_data_accessed'] is False
    assert report['cost_protocol_frozen'] is False
