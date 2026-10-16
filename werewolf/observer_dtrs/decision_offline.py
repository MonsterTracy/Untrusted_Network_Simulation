"""Explicit supervised evaluation. Never imported by the pure decision module.

This is an in-memory evaluator, not a formal D2 artifact/provenance loader.
Callers must validate artifact identity before constructing these legal rows.
"""
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction

from werewolf.observer_dtrs.decision import (
    Region, LossSpec, _probability, decide, forced_binary,
    probability_stability, cost_stability,
)

VERSION = 'observer_dtrs_explicit_offline_decision_evaluation_v1'


@dataclass(frozen=True)
class ProbabilityInput:
    sample_id: str
    game_id: str
    probability: object

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in (self.sample_id, self.game_id)):
            raise ValueError('sample and game identities required')
        object.__setattr__(self, 'probability', _probability(self.probability))


def realized_loss(action, y, loss):
    if not isinstance(action, Region) or type(y) is not int or y not in (0, 1):
        raise ValueError('offline loss requires a Region and explicit binary integer label')
    if not isinstance(loss, LossSpec):
        raise TypeError('validated LossSpec required')
    return {Region.POS: (loss.c_fp, Fraction(0)), Region.BND: (loss.b0, loss.b1),
            Region.NEG: (Fraction(0), loss.c_fn)}[action][y]


def _population(rows, *, game_sizes, weighting):
    rows = tuple(rows)
    if not rows or any(not isinstance(r, ProbabilityInput) for r in rows):
        raise ValueError('explicit legal ProbabilityInput rows required')
    rows = tuple(sorted(rows, key=lambda r: (r.game_id, r.sample_id)))
    if (len({r.sample_id for r in rows}) != len(rows)
            or any(type(n) is not int or n <= 0 for n in game_sizes.values())
            or dict(Counter(r.game_id for r in rows)) != dict(game_sizes)):
        raise ValueError('require unique predictions and exact full eligible game population')
    if weighting not in ('game_equal', 'record_equal'):
        raise ValueError('unknown evaluation weighting')
    weights = {r.sample_id: Fraction(1, game_sizes[r.game_id]) if weighting == 'game_equal'
               else Fraction(1) for r in rows}
    return rows, weights


def _aggregate(rows, targets, actions, weights, loss):
    mass = sum((weights[r.sample_id] for r in rows), Fraction(0))
    def ratio(numerator, denominator=mass):
        return float(numerator/denominator) if denominator else None
    risk = sum((weights[r.sample_id]*realized_loss(actions[r.sample_id], targets[r.sample_id], loss)
                for r in rows), Fraction(0))
    fp = sum((weights[r.sample_id] for r in rows
              if actions[r.sample_id] == Region.POS and targets[r.sample_id] == 0), Fraction(0))
    fn = sum((weights[r.sample_id] for r in rows
              if actions[r.sample_id] == Region.NEG and targets[r.sample_id] == 1), Fraction(0))
    positive = sum((weights[r.sample_id]*targets[r.sample_id] for r in rows), Fraction(0))
    regions = {}
    for a in Region:
        chosen = [r for r in rows if actions[r.sample_id] == a]
        amass = sum((weights[r.sample_id] for r in chosen), Fraction(0))
        ay = sum((weights[r.sample_id]*targets[r.sample_id] for r in chosen), Fraction(0))
        acost = sum((weights[r.sample_id]*realized_loss(a, targets[r.sample_id], loss)
                     for r in chosen), Fraction(0))
        regions[a.value] = {'rows': len(chosen), 'games': len({r.game_id for r in chosen}),
                           'mass': float(amass), 'proportion': ratio(amass),
                           'event_rate': ratio(ay, amass), 'loss_contribution': ratio(acost)}
    decided = sum((weights[r.sample_id] for r in rows if actions[r.sample_id] != Region.BND), Fraction(0))
    pos_mass = sum((weights[r.sample_id] for r in rows if actions[r.sample_id] == Region.POS), Fraction(0))
    return {'rows': len(rows), 'games': len({r.game_id for r in rows}), 'mass': float(mass),
            'realized_risk': ratio(risk), 'regions': regions, 'decided_coverage': ratio(decided),
            'false_positive_joint': ratio(fp), 'false_negative_joint': ratio(fn),
            'false_positive_rate_given_Y0': ratio(fp, mass-positive),
            'false_negative_rate_given_Y1': ratio(fn, positive),
            'POS_error_rate': ratio(fp, pos_mass),
            'NEG_error_rate': regions['NEG']['event_rate'], 'selective_error_rate': ratio(fp+fn, decided)}


def evaluate(rows, targets, *, loss, game_sizes, weighting, binary_tie_policy, subgroup_ids=None):
    """Full-game weights are fixed before any optional audit subgroup selection."""
    rows, weights = _population(rows, game_sizes=game_sizes, weighting=weighting)
    if (set(targets) != {r.sample_id for r in rows}
            or any(type(v) is not int or v not in (0, 1) for v in targets.values())):
        raise ValueError('explicit targets must cover every legal sample exactly once')
    decisions = {r.sample_id: decide(r.probability, loss) for r in rows}
    binary = {r.sample_id: forced_binary(r.probability, loss, tie_policy=binary_tie_policy) for r in rows}
    if subgroup_ids is None:
        selected = rows
    else:
        ids = tuple(subgroup_ids)
        if len(ids) != len(set(ids)) or not set(ids) <= set(decisions):
            raise ValueError('audit subgroup must contain unique known sample IDs')
        selected = tuple(r for r in rows if r.sample_id in set(ids))
    actions = {sid: d.region for sid, d in decisions.items()}
    result = _aggregate(selected, targets, actions, weights, loss)
    total_mass = sum((weights[r.sample_id] for r in selected), Fraction(0))
    predicted = sum((weights[r.sample_id]*decisions[r.sample_id].risks[actions[r.sample_id]]
                     for r in selected), Fraction(0))
    result.update({'schema_version': VERSION, 'scope': 'explicit_offline_evaluation_not_a_verified_formal_artifact',
        'weighting': weighting, 'loss': loss.to_record(),
        'predicted_minimum_risk': float(predicted/total_mass) if total_mass else None,
        'fixed_action_risks': {a.value: _aggregate(selected, targets,
            {r.sample_id: a for r in selected}, weights, loss)['realized_risk'] for a in Region},
        'forced_binary': {'tie_policy': binary_tie_policy.value,
                          **_aggregate(selected, targets, binary, weights, loss)},
        'decisions': [{'sample_id': r.sample_id, 'game_id': r.game_id,
                       **decisions[r.sample_id].to_record()} for r in selected],
        'by_game': {g: _aggregate(tuple(r for r in selected if r.game_id == g), targets, actions, weights, loss)
                    for g in sorted(game_sizes)}})
    return result


def evaluate_stability(rows, *, loss, game_sizes, weighting, probability_radii, cost_alternatives):
    """Label-free stability, separate from realized loss and semantic BND."""
    rows, weights = _population(rows, game_sizes=game_sizes, weighting=weighting)
    total = sum(weights.values(), Fraction(0))
    radii, alternatives = tuple(probability_radii), tuple(cost_alternatives)
    probability_reports = []
    for radius in radii:
        entries = [(r, probability_stability(r.probability, loss, radius=radius)) for r in rows]
        unstable = sum((weights[r.sample_id] for r, d in entries
                        if not d['stable_under_selected_tie_policy']), Fraction(0))
        probability_reports.append({'radius': float(radius), 'unstable_quality_fraction': float(unstable/total),
                                    'rows': [{'sample_id': r.sample_id, **d} for r, d in entries]})
    cost_reports = []
    for alternative in alternatives:
        entries = [(r, cost_stability(r.probability, loss, alternatives=(alternative,))) for r in rows]
        flipped = sum((weights[r.sample_id] for r, d in entries if d['alternatives'][0]['region_flipped']), Fraction(0))
        cost_reports.append({'loss': alternative.to_record(), 'flipped_quality_fraction': float(flipped/total),
                             'rows': [{'sample_id': r.sample_id, **d} for r, d in entries]})
    return {'scope': 'label_free_sensitivity_not_confidence_or_cost_selection', 'weighting': weighting,
            'probability_scenarios': probability_reports, 'cost_scenarios': cost_reports}
