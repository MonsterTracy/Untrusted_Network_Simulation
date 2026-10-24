"""Pure decision contracts against exact independent risk and threshold oracles."""
from dataclasses import replace
from fractions import Fraction as F
import inspect
import math

import pytest
import numpy as np

from werewolf.observer_dtrs.decision import (
    LossSpec, Region, TiePolicy, candidate_losses, conditional_risks, cost_stability,
    decide, forced_binary, probability_stability, scenario_regret, thresholds,
)


def loss(policy=TiePolicy.TERMINAL_POS_FIRST):
    return LossSpec(1, 1, F(1, 4), F(1, 4), policy)


def test_explicit_tie_policy_and_input_separation():
    with pytest.raises(TypeError):
        LossSpec(1, 1, F(1, 4), F(1, 4))
    with pytest.raises(ValueError, match='TiePolicy'):
        LossSpec(1, 1, F(1, 4), F(1, 4), 'default')
    assert set(inspect.signature(decide).parameters) == {'probability', 'loss'}
    for forbidden in ('y', 'supervision', 'support', 'Theta_AC', 'role_sidecar'):
        with pytest.raises(TypeError):
            decide(F(1, 2), loss(), **{forbidden: 1})


@pytest.mark.parametrize('policy', list(TiePolicy))
def test_d0_and_d3a_exact_boundary_differences(policy):
    m = loss(policy)
    alpha, beta = thresholds(m).alpha, thresholds(m).beta
    lower, upper = decide(beta, m), decide(alpha, m)
    assert lower.minimizing_actions == (Region.BND, Region.NEG)
    assert upper.minimizing_actions == (Region.POS, Region.BND)
    assert lower.risk_gap == upper.risk_gap == 0
    assert lower.region == (Region.BND if policy == TiePolicy.DEFER_FIRST else Region.NEG)
    assert upper.region == (Region.BND if policy == TiePolicy.DEFER_FIRST else Region.POS)
    assert decide(0, m).region == Region.NEG
    assert decide(1, m).region == Region.POS


def test_no_tolerance_band_or_threshold_snapping():
    m = loss()
    beta = F(1, 4)
    above = decide(beta+F(1, 10**60), m)
    assert float(above.probability) == float(beta)  # Display rounding cannot control decisions.
    assert above.region == Region.BND and above.risk_gap > 0
    assert decide(math.nextafter(.25, 0), m).region == Region.NEG
    assert decide(math.nextafter(.25, 1), m).region == Region.BND
    assert decide(math.nextafter(.75, 0), m).region == Region.BND
    assert decide(math.nextafter(.75, 1), m).region == Region.POS


@pytest.mark.parametrize('policy', list(TiePolicy))
def test_45_candidates_against_risk_oracle_and_scale_invariance(policy):
    candidates = candidate_losses(tie_policy=policy)
    assert len(candidates) == len({m.digest for m in candidates}) == 45
    for m in candidates:
        t = thresholds(m)
        assert t.regime == 'NONDEGENERATE' and 0 < t.beta < t.gamma < t.alpha < 1
        for p in {*[F(j, 20) for j in range(21)], t.alpha, t.beta, t.gamma}:
            oracle = {Region.POS: m.c_fp*(1-p), Region.BND: m.b1*p+m.b0*(1-p), Region.NEG: m.c_fn*p}
            expected = tuple(a for a in Region if oracle[a] == min(oracle.values()))
            d = decide(p, m)
            assert d.minimizing_actions == expected
            assert d.region == next(a for a in policy.order if a in expected)
            scaled = decide(p, m.scaled(F(7, 3)))
            assert scaled.region == d.region and scaled.minimizing_actions == d.minimizing_actions
            assert scaled.risk_gap == d.risk_gap*F(7, 3)
            assert scaled.thresholds == d.thresholds


@pytest.mark.parametrize('policy', list(TiePolicy))
def test_collapsed_and_dominated_bnd(policy):
    collapsed = LossSpec(1, 1, F(1, 2), F(1, 2), policy)
    d = decide(F(1, 2), collapsed)
    assert d.minimizing_actions == (Region.POS, Region.BND, Region.NEG)
    assert d.region == policy.order[0]
    assert d.thresholds.regime == 'COLLAPSED_BND' and d.threshold_distance is None
    dominated = LossSpec(1, 1, F(3, 5), F(3, 5), policy)
    for j in range(21):
        assert Region.BND not in decide(F(j, 20), dominated).minimizing_actions
    assert thresholds(dominated).regime == 'BND_DOMINATED'


@pytest.mark.parametrize('b1,b0,regime', [(0, 0, 'ZERO_DEFERRAL'),
    (F(1, 4), 0, 'NEG_INTERVAL_COLLAPSED'), (0, F(1, 4), 'POS_INTERVAL_COLLAPSED'),
    (1, 0, 'IDENTICAL_NEG_BND'), (0, 1, 'IDENTICAL_POS_BND')])
def test_weak_column_order_and_undefined_crossings(b1, b0, regime):
    m = LossSpec(1, 1, b1, b0, TiePolicy.DEFER_FIRST)
    assert thresholds(m).regime == regime
    if regime == 'IDENTICAL_NEG_BND':
        assert thresholds(m).beta is None
        assert decide(F(1, 4), m).minimizing_actions == (Region.BND, Region.NEG)
    if regime == 'IDENTICAL_POS_BND':
        assert thresholds(m).alpha is None
        assert decide(F(3, 4), m).minimizing_actions == (Region.POS, Region.BND)
    assert decide(F(1, 2), m).region == Region.BND


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), -.001, 1.001, True, '0.5', None])
def test_invalid_probabilities_rejected(value):
    with pytest.raises(ValueError):
        decide(value, loss())


@pytest.mark.parametrize('changes', [{'c_fp': 0}, {'c_fn': -1}, {'b0': -1}, {'b1': 2},
    {'b0': 2}, {'c_fp': float('nan')}, {'b1': float('inf')}, {'b0': True}, {'c_fn': '1'}])
def test_invalid_costs_rejected(changes):
    with pytest.raises(ValueError):
        replace(loss(), **changes)


def test_extreme_finite_costs_do_not_overflow_intermediate_sums():
    m = LossSpec(1e308, 1e308, 2.5e307, 2.5e307, TiePolicy.DEFER_FIRST)
    d = decide(.5, m)
    assert d.region == Region.BND and d.thresholds.gamma == F(1, 2)
    assert all(math.isfinite(v) for v in d.to_record()['risks'].values())
    assert d.to_record()['risks_exact']['BND'] == str(m.b1)


@pytest.mark.parametrize('numeric', [np.float32, np.float64, np.int64])
def test_native_and_numpy_real_scalars(numeric):
    m = LossSpec(numeric(1), numeric(1), F(1, 4), F(1, 4), TiePolicy.DEFER_FIRST)
    assert decide(numeric(0), m).region == Region.NEG
    assert decide(numeric(1), m).region == Region.POS


def test_probability_interval_and_cost_perturbation_are_diagnostics():
    m = loss()
    assert probability_stability(.25, m, radius=0)['stable_under_selected_tie_policy']
    crossing = probability_stability(.25, m, radius=F(1, 100))
    assert crossing['selected_regions'] == ['BND', 'NEG']
    assert not crossing['stable_under_selected_tie_policy']
    assert probability_stability(.5, m, radius=F(1, 100))['worst_case_regret'] == 0
    assert probability_stability(0, m, radius=.05)['interval_exact'][0] == '0'
    alternative = LossSpec(4, 1, F(1, 4), 1, m.tie_policy)
    d = cost_stability(.8, m, alternatives=(m.scaled(2), alternative))
    assert [r['region_flipped'] for r in d['alternatives']] == [False, True]
    with pytest.raises(ValueError):
        cost_stability(.8, m, alternatives=(replace(m, tie_policy=TiePolicy.DEFER_FIRST),))
    for bad in (-1, float('inf'), float('nan'), True):
        with pytest.raises(ValueError):
            probability_stability(.5, m, radius=bad)


def test_regret_bound_and_interval_endpoint_maximum():
    for m in candidate_losses(tie_policy=TiePolicy.DEFER_FIRST):
        for i in range(9):
            p = F(i, 8)
            for j in range(9):
                eta = F(j, 8)
                assert 0 <= scenario_regret(p, eta, m) <= (m.c_fp+m.c_fn)*abs(p-eta)
            radius = F(1, 10)
            low, high = max(F(0), p-radius), min(F(1), p+radius)
            maximum = max(scenario_regret(p, low, m), scenario_regret(p, high, m))
            for j in range(11):
                assert scenario_regret(p, low+(high-low)*F(j, 10), m) <= maximum
    m = loss()
    assert scenario_regret(F(76, 100), F(74, 100), m) == F(1, 100)
    assert scenario_regret(F(70, 100), F(74, 100), m) == 0


def test_binary_tie_is_explicit_and_independent():
    m = loss(TiePolicy.DEFER_FIRST)
    assert forced_binary(.5, m, tie_policy=TiePolicy.TERMINAL_NEG_FIRST) == Region.NEG
    assert forced_binary(.5, m, tie_policy=TiePolicy.TERMINAL_POS_FIRST) == Region.POS
    assert conditional_risks(.5, m).bnd < conditional_risks(.5, m).pos
