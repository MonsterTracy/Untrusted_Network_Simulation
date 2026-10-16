"""Pure static cognitive decisions. No supervision, support, fitting or I/O.

Comparisons use exact rational values of the supplied real numbers. In particular,
a float retains its binary value; near ties are never snapped to a threshold.
"""
from dataclasses import dataclass, replace
from enum import Enum
from fractions import Fraction
import hashlib
import json
import math
from numbers import Rational, Real

VERSION = 'observer_dtrs_six_loss_decision_v1'


class Region(str, Enum):
    POS = 'POS'
    BND = 'BND'
    NEG = 'NEG'


class TiePolicy(str, Enum):
    # Both terminal-first orders preserve D0's nondegenerate boundary convention.
    TERMINAL_POS_FIRST = 'terminal_pos_then_neg_then_bnd'
    TERMINAL_NEG_FIRST = 'terminal_neg_then_pos_then_bnd'
    DEFER_FIRST = 'bnd_then_neg_then_pos'

    @property
    def order(self):
        return {
            TiePolicy.TERMINAL_POS_FIRST: (Region.POS, Region.NEG, Region.BND),
            TiePolicy.TERMINAL_NEG_FIRST: (Region.NEG, Region.POS, Region.BND),
            TiePolicy.DEFER_FIRST: (Region.BND, Region.NEG, Region.POS),
        }[self]


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} requires a finite real number, not bool/string')
    try:
        if not math.isfinite(float(value)):
            raise ValueError(f'{name} must be finite')
        if isinstance(value, Rational):
            return Fraction(int(value.numerator), int(value.denominator))
        return Fraction(*value.as_integer_ratio())
    except (TypeError, OverflowError, ZeroDivisionError) as error:
        raise ValueError(f'invalid {name}') from error


def _probability(value):
    value = _number(value, 'event probability')
    if not 0 <= value <= 1:
        raise ValueError('event probability outside [0,1]; no clipping')
    return value


def _policy(policy):
    if not isinstance(policy, TiePolicy):
        raise ValueError('an explicit TiePolicy is required')
    return policy


@dataclass(frozen=True)
class LossSpec:
    c_fp: Real
    c_fn: Real
    b1: Real
    b0: Real
    tie_policy: TiePolicy

    def __post_init__(self):
        _policy(self.tie_policy)
        for key in ('c_fp', 'c_fn', 'b1', 'b0'):
            object.__setattr__(self, key, _number(getattr(self, key), key))
        if (self.c_fp <= 0 or self.c_fn <= 0 or not 0 <= self.b0 <= self.c_fp
                or not 0 <= self.b1 <= self.c_fn):
            raise ValueError('require c_fp,c_fn>0 and weak column ordering 0<=b0<=c_fp, 0<=b1<=c_fn')

    def to_record(self):
        return {'schema_version': VERSION, 'tie_policy': self.tie_policy.value,
                'cost_units': 'as_supplied_no_automatic_normalization',
                'costs': {k: float(getattr(self, k)) for k in ('c_fp', 'c_fn', 'b1', 'b0')},
                'exact_costs': {k: str(getattr(self, k)) for k in ('c_fp', 'c_fn', 'b1', 'b0')}}

    @property
    def digest(self):
        raw = json.dumps(self.to_record(), sort_keys=True, separators=(',', ':'), allow_nan=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    def scaled(self, factor):
        factor = _number(factor, 'positive loss scale')
        if factor <= 0:
            raise ValueError('loss scale must be positive')
        return replace(self, **{k: getattr(self, k)*factor for k in ('c_fp', 'c_fn', 'b1', 'b0')})


@dataclass(frozen=True)
class ConditionalRisks:
    pos: Fraction
    bnd: Fraction
    neg: Fraction

    def items(self):
        return ((Region.POS, self.pos), (Region.BND, self.bnd), (Region.NEG, self.neg))

    def __getitem__(self, action):
        return dict(self.items())[action]


def conditional_risks(probability, loss):
    if not isinstance(loss, LossSpec):
        raise TypeError('validated LossSpec required')
    p = _probability(probability)
    return ConditionalRisks(loss.c_fp*(1-p), loss.b1*p + loss.b0*(1-p), loss.c_fn*p)


@dataclass(frozen=True)
class ThresholdInfo:
    alpha: Fraction | None
    beta: Fraction | None
    gamma: Fraction
    regime: str

    def to_record(self):
        values = {k: getattr(self, k) for k in ('alpha', 'beta', 'gamma')}
        return {'regime': self.regime, 'nondegenerate': self.regime == 'NONDEGENERATE',
                'values': {k: float(v) if v is not None else None for k, v in values.items()},
                'exact_values': {k: str(v) if v is not None else None for k, v in values.items()}}


def thresholds(loss):
    if not isinstance(loss, LossSpec):
        raise TypeError('validated LossSpec required')
    a, c, b1, b0 = loss.c_fp, loss.c_fn, loss.b1, loss.b0
    alpha = (a-b0)/(a-b0+b1) if a-b0+b1 else None
    beta = b0/(b0+c-b1) if b0+c-b1 else None
    if b0 == a and b1 == 0:
        regime = 'IDENTICAL_POS_BND'
    elif b0 == 0 and b1 == c:
        regime = 'IDENTICAL_NEG_BND'
    elif b0/a+b1/c > 1:
        regime = 'BND_DOMINATED'
    elif b0/a+b1/c == 1:
        regime = 'COLLAPSED_BND'
    elif b0 == b1 == 0:
        regime = 'ZERO_DEFERRAL'
    elif b0 == 0:
        regime = 'NEG_INTERVAL_COLLAPSED'
    elif b1 == 0:
        regime = 'POS_INTERVAL_COLLAPSED'
    else:
        regime = 'NONDEGENERATE'
    return ThresholdInfo(alpha, beta, a/(a+c), regime)


@dataclass(frozen=True)
class Decision:
    probability: Fraction
    loss_digest: str
    tie_policy: TiePolicy
    risks: ConditionalRisks
    minimizing_actions: tuple[Region, ...]
    region: Region
    risk_gap: Fraction
    thresholds: ThresholdInfo
    threshold_distance: Fraction | None

    def to_record(self):
        return {'schema_version': VERSION, 'probability': float(self.probability),
                'probability_exact': str(self.probability), 'loss_digest': self.loss_digest,
                'tie_policy': self.tie_policy.value,
                'risks': {a.value: float(v) for a, v in self.risks.items()},
                'risks_exact': {a.value: str(v) for a, v in self.risks.items()},
                'minimizing_actions': [a.value for a in self.minimizing_actions],
                'region': self.region.value, 'risk_gap': float(self.risk_gap),
                'risk_gap_exact': str(self.risk_gap), 'thresholds': self.thresholds.to_record(),
                'threshold_distance': float(self.threshold_distance) if self.threshold_distance is not None else None}


def decide(probability, loss):
    """Only event probability and explicit loss/tie specification enter selection."""
    p = _probability(probability)
    risks = conditional_risks(p, loss)
    minimum = min(v for _, v in risks.items())
    optimal = tuple(a for a, v in risks.items() if v == minimum)
    chosen = next(a for a in loss.tie_policy.order if a in optimal)
    th = thresholds(loss)
    distance = min(abs(p-th.alpha), abs(p-th.beta)) if th.regime == 'NONDEGENERATE' else None
    ordered = sorted(v for _, v in risks.items())
    return Decision(p, loss.digest, loss.tie_policy, risks, optimal, chosen,
                    ordered[1]-ordered[0], th, distance)


def forced_binary(probability, loss, *, tie_policy):
    """Binary baseline: its P/N exact tie order is explicit and independently recorded."""
    _policy(tie_policy)
    risks = conditional_risks(probability, loss)
    minimum = min(risks.pos, risks.neg)
    return next(a for a in tie_policy.order if a != Region.BND and risks[a] == minimum)


def scenario_regret(probability, true_probability, loss):
    """Hypothetical eta, never a per-record label estimate of true Bayes regret."""
    action = decide(probability, loss).region
    truth = conditional_risks(true_probability, loss)
    return truth[action] - min(v for _, v in truth.items())


def probability_stability(probability, loss, *, radius):
    """External perturbation scenario, not a statistical confidence interval or gate."""
    p, radius = _probability(probability), _number(radius, 'probability perturbation radius')
    if radius < 0:
        raise ValueError('probability radius must be nonnegative')
    low, high = max(Fraction(0), p-radius), min(Fraction(1), p+radius)
    th = thresholds(loss)
    points = sorted({low, high, *(v for v in (th.alpha, th.beta, th.gamma)
                                  if v is not None and low <= v <= high)})
    probes = [*points, *((a+b)/2 for a, b in zip(points, points[1:]))]
    choices = {decide(q, loss).region for q in probes}
    minimizers = {a for q in probes for a in decide(q, loss).minimizing_actions}
    return {'scope': 'external_probability_sensitivity_not_confidence_interval',
            'interval_exact': [str(low), str(high)],
            'selected_regions': [a.value for a in Region if a in choices],
            'optimal_actions_union': [a.value for a in Region if a in minimizers],
            'stable_under_selected_tie_policy': len(choices) == 1,
            'worst_case_regret': float(max(scenario_regret(p, q, loss) for q in (low, high))),
            'baseline': decide(p, loss).to_record()}


def cost_stability(probability, loss, *, alternatives):
    """Compare caller-declared matrices; no label access or best-cost selection."""
    baseline = decide(probability, loss)
    entries = []
    for other in alternatives:
        if not isinstance(other, LossSpec) or other.tie_policy != loss.tie_policy:
            raise ValueError('cost perturbations require validated losses with the same tie policy')
        result = decide(probability, other)
        entries.append({'loss': other.to_record(), 'decision': result.to_record(),
                        'region_flipped': result.region != baseline.region})
    return {'scope': 'declared_cost_sensitivity_not_cost_optimization',
            'baseline': baseline.to_record(), 'alternatives': entries}


def candidate_losses(*, tie_policy):
    """The 45 D3a normative candidates, all retained; none designated primary."""
    _policy(tie_policy)
    result = []
    for rho in map(Fraction, (Fraction(1, 4), Fraction(1, 2), 1, 2, 4)):
        a, c = rho/(1+rho), 1/(1+rho)
        for kappa in (Fraction(1, 5), Fraction(1, 2), Fraction(4, 5)):
            for h in (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)):
                result.append(LossSpec(a, c, kappa*(1-h)*c, kappa*h*a, tie_policy))
    return tuple(result)
