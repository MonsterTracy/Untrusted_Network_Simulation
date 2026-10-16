"""D3b controlled acceptance only; no formal artifact loader or server access."""
import argparse
from dataclasses import replace
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from werewolf.observer_dtrs.decision import LossSpec, TiePolicy, candidate_losses, decide
from werewolf.observer_dtrs.decision_offline import ProbabilityInput, evaluate, evaluate_stability


def controlled_acceptance():
    rows = (ProbabilityInput('a', 'controlled-A', F(9, 10)),
            ProbabilityInput('b1', 'controlled-B', F(1, 10)),
            ProbabilityInput('b2', 'controlled-B', F(1, 2)),
            ProbabilityInput('b3', 'controlled-B', F(9, 10)))
    targets = {'a': 0, 'b1': 1, 'b2': 1, 'b3': 1}
    losses = LossSpec(1, 1, F(1, 4), F(1, 4), TiePolicy.TERMINAL_POS_FIRST)
    sizes = {'controlled-A': 1, 'controlled-B': 3}
    common = dict(loss=losses, game_sizes=sizes, binary_tie_policy=TiePolicy.TERMINAL_NEG_FIRST)
    boundaries = {policy.value: {str(p): decide(p, replace(losses, tie_policy=policy)).to_record()
                               for p in (F(1, 4), F(1, 2), F(3, 4))} for policy in TiePolicy}
    evaluation = {w: evaluate(rows, targets, weighting=w, **common) for w in ('game_equal', 'record_equal')}
    root = Path(__file__).resolve().parents[1]
    files = ('werewolf/observer_dtrs/decision.py', 'werewolf/observer_dtrs/decision_offline.py',
             'tests/observer_dtrs/test_decision.py', 'tests/observer_dtrs/test_decision_offline.py',
             'scripts/check_observer_dtrs_decisions.py')
    return {'schema_version': 'observer_dtrs_d3b_controlled_acceptance_v1',
            'scope': 'artificial_two_games_four_probabilities_only', 'formal_data_read': False,
            'formal_D3_protocol_frozen': False, 'formal_D3_experiment_run': False,
            'candidate_count_per_explicit_policy': {p.value: len(candidate_losses(tie_policy=p)) for p in TiePolicy},
            'boundary_examples': boundaries, 'evaluation': evaluation,
            'stability': evaluate_stability(rows, loss=losses, game_sizes=sizes, weighting='game_equal',
                probability_radii=(0, F(1, 100), F(1, 20)),
                cost_alternatives=(losses.scaled(2), replace(losses, c_fp=4, b0=1))),
            'file_sha256': {f: hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New local controlled JSON; never overwrite')
    args = parser.parse_args(argv)
    result = controlled_acceptance()
    text = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(text)
    print(json.dumps({'scope': result['scope'], 'candidates_per_policy': 45,
        'game_equal_realized_risk': result['evaluation']['game_equal']['realized_risk'],
        'record_equal_realized_risk': result['evaluation']['record_equal']['realized_risk'],
        'formal_experiment_run': False}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
