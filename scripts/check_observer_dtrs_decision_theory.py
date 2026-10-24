"""Exact, controlled D3a mathematics; no formal data or production classifier.

All scenarios below are illustrative, not frozen research loss parameters.
Only standard-library Fraction arithmetic is used; no fitting, targets or ToM I/O.
"""

import argparse
from dataclasses import dataclass
from fractions import Fraction as F
import json
from pathlib import Path


@dataclass(frozen=True)
class ControlledLoss:
    c_fp: F
    c_fn: F
    b1: F
    b0: F

    def __post_init__(self):
        for key in ('c_fp', 'c_fn', 'b1', 'b0'):
            value = getattr(self, key)
            if isinstance(value, bool):
                raise ValueError('boolean is not a controlled cost')
            object.__setattr__(self, key, F(value))
        if (self.c_fp <= 0 or self.c_fn <= 0
                or not 0 <= self.b0 < self.c_fp or not 0 <= self.b1 < self.c_fn):
            raise ValueError('controlled losses require positive error costs and column ordering')

    @property
    def thresholds(self):
        a, c, b1, b0 = self.c_fp, self.c_fn, self.b1, self.b0
        return ((a - b0) / (a - b0 + b1), b0 / (b0 + c - b1), a / (a + c))

    @property
    def relative_deferral(self):
        return self.b0 / self.c_fp + self.b1 / self.c_fn

    def scaled(self, factor):
        return ControlledLoss(*(v * factor for v in (self.c_fp, self.c_fn, self.b1, self.b0)))


def risks(probability, loss):
    if isinstance(probability, bool):
        raise ValueError('boolean is not a controlled probability')
    p = F(probability)
    if not 0 <= p <= 1:
        raise ValueError('controlled probability outside [0,1]')
    return {'POS': loss.c_fp * (1 - p), 'BND': loss.b1 * p + loss.b0 * (1 - p),
            'NEG': loss.c_fn * p}


def minimizing_actions(probability, loss):
    values = risks(probability, loss)
    lowest = min(values.values())
    return tuple(a for a in ('BND', 'NEG', 'POS') if values[a] == lowest)


def direct_decision(probability, loss):
    return minimizing_actions(probability, loss)[0]


def threshold_decision(probability, loss):
    """Independent closed-form shortcut, including dominated/collapsed BND."""
    risks(probability, loss)  # Check input; values do not control the shortcut.
    p = F(probability)
    alpha, beta, gamma = loss.thresholds
    if loss.relative_deferral < 1:
        return 'NEG' if p < beta else 'BND' if p <= alpha else 'POS'
    if loss.relative_deferral == 1 and p == gamma:
        return 'BND'
    return 'NEG' if p <= gamma else 'POS'


def regret(probability, true_probability, loss):
    values = risks(true_probability, loss)
    return values[direct_decision(probability, loss)] - min(values.values())


def probability_interval(probability, radius):
    if isinstance(probability, bool) or isinstance(radius, bool):
        raise ValueError('boolean is not a controlled interval input')
    p, e = F(probability), F(radius)
    if not 0 <= p <= 1 or e < 0:
        raise ValueError('invalid controlled interval')
    return max(F(0), p - e), min(F(1), p + e)


def interval_decisions(probability, radius, loss):
    low, high = probability_interval(probability, radius)
    points = sorted({low, high, *(v for v in loss.thresholds if low <= v <= high)})
    probes = [*points, *((a + b) / 2 for a, b in zip(points, points[1:]))]
    return tuple(sorted({direct_decision(p, loss) for p in probes}))


def interval_regret_bound(probability, radius, loss):
    """Exact scenario maximum for fixed decision; not a confidence interval."""
    return max(regret(probability, eta, loss) for eta in probability_interval(probability, radius))


def candidate_scenarios():
    for rho in (F(1, 4), F(1, 2), F(1), F(2), F(4)):
        a, c = rho / (1 + rho), 1 / (1 + rho)
        for kappa in (F(1, 5), F(1, 2), F(4, 5)):
            for h in (F(1, 4), F(1, 2), F(3, 4)):
                yield ControlledLoss(a, c, kappa * (1 - h) * c, kappa * h * a)


def check_theory():
    scenarios = tuple(candidate_scenarios())
    boundary_cases = (ControlledLoss(1, 1, F(1, 2), F(1, 2)),
                      ControlledLoss(1, 1, F(3, 5), F(3, 5)),
                      ControlledLoss(1, 1, 0, F(1, 4)),
                      ControlledLoss(1, 1, F(1, 4), 0), ControlledLoss(1, 1, 0, 0))
    classifications = regret_checks = interval_checks = 0
    for loss in (*scenarios, *boundary_cases):
        alpha, beta, gamma = loss.thresholds
        if loss in scenarios:
            assert 0 < beta < gamma < alpha < 1
        assert risks(alpha, loss)['POS'] == risks(alpha, loss)['BND']
        assert risks(beta, loss)['BND'] == risks(beta, loss)['NEG']
        assert risks(gamma, loss)['POS'] == risks(gamma, loss)['NEG']
        for p in sorted({*(F(j, 100) for j in range(101)), alpha, beta, gamma}):
            assert threshold_decision(p, loss) == direct_decision(p, loss)
            assert direct_decision(p, loss.scaled(F(7, 3))) == direct_decision(p, loss)
            classifications += 1
        for p in (F(j, 20) for j in range(21)):
            for eta in (F(j, 20) for j in range(21)):
                assert 0 <= regret(p, eta, loss) <= (loss.c_fp + loss.c_fn) * abs(p - eta)
                regret_checks += 1
            for epsilon in (F(0), F(1, 100), F(1, 40), F(1, 20)):
                low, high = probability_interval(p, epsilon)
                bound = interval_regret_bound(p, epsilon, loss)
                for j in range(11):
                    eta = low + (high - low) * F(j, 10)
                    assert regret(p, eta, loss) <= bound
                assert bound <= (loss.c_fp + loss.c_fn) * epsilon
                interval_checks += 1
    anchor = ControlledLoss(F(1, 2), F(1, 2), F(1, 8), F(1, 8))
    eta, p_good_brier, p_bad_brier = F(74, 100), F(76, 100), F(70, 100)
    assert (p_good_brier - eta) ** 2 < (p_bad_brier - eta) ** 2
    assert regret(p_good_brier, eta, anchor) > regret(p_bad_brier, eta, anchor)
    return {'status': 'CONTROLLED_THEORY_CHECKS_PASSED', 'formal_data_accessed': False,
        'formal_classification_or_fitting': False, 'cost_protocol_frozen': False,
        'arithmetic': 'exact_standard_library_Fraction',
        'proposed_nondegenerate_scenarios': len(scenarios), 'degenerate_controls': len(boundary_cases),
        'threshold_argmin_and_scale_checks': classifications, 'regret_bound_checks': regret_checks,
        'interval_scenario_checks': interval_checks,
        'illustrative_anchor': {'alpha': '3/4', 'beta': '1/4', 'gamma': '1/2',
            'p_at_beta_minimizers': minimizing_actions(F(1, 4), anchor),
            'p_at_alpha_minimizers': minimizing_actions(F(3, 4), anchor)},
        'brier_risk_counterexample': {'eta': str(eta), 'lower_brier_probability': str(p_good_brier),
            'higher_brier_probability': str(p_bad_brier),
            'lower_brier_regret': str(regret(p_good_brier, eta, anchor)),
            'higher_brier_regret': str(regret(p_bad_brier, eta, anchor))},
        'limitations': ['finite_controlled_grid_is_not_a_general_proof',
                        'scenario_intervals_are_not_statistical_confidence_intervals',
                        'no_formal_D3_results_or_cost_elicitation']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional new local controlled report; never overwrite')
    args = parser.parse_args(argv)
    result = check_theory()
    text = json.dumps(result, sort_keys=True, indent=2) + '\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(text)
    print(text, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
