"""Artificial-only Shadow acceptance and anchored inspection. No game/service entry."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection.pre import construct_authoritative_pre_prefix
from werewolf.canonical_collection.public_history import PLAYER_IDS, freeze_public_event_history
from werewolf.observer_dtrs.context import contexts_for_pre
from werewolf.observer_dtrs.d2_protocol import frozen_protocol
from werewolf.observer_dtrs.deployment_offline import publish_controlled_estimator
from werewolf.observer_dtrs.estimation import TrainingRow, fit_model
from werewolf.observer_dtrs.features import ObserverFeatures
from werewolf.observer_dtrs.online_context import ShadowRequest, digest
from werewolf.observer_dtrs.online_model import TomIdentity, read_deployment
from werewolf.observer_dtrs.probabilities import ProbabilityMatrix
from werewolf.observer_dtrs.shadow import QPrediction, ShadowEngine
from werewolf.structured_history import plan_structured_history


def artificial_identity():
    return TomIdentity('controlled_fixture', None, *[digest(['artificial', i]) for i in range(4)], None)


def artificial_requests():
    requests = []
    for phase in ('discussion', 'pk_discussion'):
        for dead in ((), ('player2', 'player7')):
            alive = tuple(p for p in PLAYER_IDS if p not in dead)
            events = []
            def add(kind, **fields):
                i = len(events)
                events.append({'event_id': f'artificial-event-{i}', 'event_index': i, 'event_type': kind, **fields})
            add('phase_change', day=0, phase='night')
            add('death_announcement', dead_players=list(dead))
            add('phase_change', day=1, phase='discussion')
            if phase == 'pk_discussion':
                half = len(alive)//2
                targets = [alive[0]]*half + [alive[1]]*half + [None]*(len(alive)%2)
                add('phase_change', day=1, phase='vote')
                add('vote_result', votes=[{'voter': v, 'target': t} for v, t in zip(alive, targets)])
                add('exile_result', exiled_players=[])
                add('phase_change', day=1, phase='pk_discussion')
            add('turn_start', speaker='player1')
            prefix = construct_authoritative_pre_prefix(game_id='artificial-shadow-game',
                boundary_id='artificial-pre', step_index=len(events)-1, report_trigger_id='artificial-trigger',
                current_speaker='player1', alive_observer_ids=alive,
                public_event_history=freeze_public_event_history(events), v1_annotations=(),
                belief_observation_ids_by_observer={p: 'artificial-report-id-'+p for p in alive})
            assert contexts_for_pre(prefix, ('player1', 'player2'))
            requests.append(ShadowRequest(prefix, ('player1', 'player2'), 'artificial-shadow-check', 'attempt-1', 0))
    return requests


class ArtificialProvider:
    identity = artificial_identity()

    def __init__(self, *, forbid_calls=False):
        self.calls = 0
        self.forbid_calls = forbid_calls

    def predict(self, prefix, *, deadline_monotonic):
        if self.forbid_calls:
            raise AssertionError('inspect must never call ToM')
        self.calls += 1
        matrix = [[0. if i == j else 1/6 for j in range(7)] for i in range(7)]
        matrix[2] = [.1, .2, 0., .1, .1, .2, .3]
        return QPrediction(ProbabilityMatrix(prefix.game_id, prefix.boundary_id, prefix.prefix_digest, matrix),
                           self.identity, plan_structured_history(prefix).plan_digest)


def _engine(root, manifest_digest, *, forbid_calls=False):
    model = read_deployment(root/'model', expected_manifest_digest=manifest_digest, expected_tom=artificial_identity())
    return ShadowEngine(tom_provider=ArtificialProvider(forbid_calls=forbid_calls), probability_provider=model,
        expected_deployment_digest=manifest_digest, journal_root=root/'journal')


def controlled(root):
    root = Path(root).resolve()
    if root.exists() or any((p/'manifest.json').exists() for p in root.parents):
        raise ValueError('controlled destination must be new and outside all immutable artifacts')
    protocol = frozen_protocol()
    rows = tuple(TrainingRow(f'artificial-{y}', f'artificial-game-{y}',
        ObserverFeatures(.1+.8*y, 0., .1+.8*y, None, 'discussion', 1), y) for y in (0, 1))
    source = {'scope': 'controlled_fixture', 'protocol_digest': protocol.digest, 'dataset_digest': None,
        'split_manifest_digest': None, 'tom_fold': None, 'outer_fold': None, 'inner_fold': None,
        'game_sizes': {r.game_id: 1 for r in rows}, 'origins': {'fixture': 'hand_checkable_only'},
        'implementation_digest': sha256_bytes(Path(__file__).read_bytes())}
    model = fit_model(rows, protocol=protocol, model_id='OBS-EQ1', bins=1, smoothing=1, source=source)
    a = publish_controlled_estimator(root/'model', model, tom=artificial_identity(),
        selection_rule_digest=digest({'rule': 'fixed_artificial_only_not_selected'}))
    engine = _engine(root, a.manifest_digest)
    requests = artificial_requests()
    results = [engine.infer(r) for r in requests]
    anchors = {r.digest: v.digest for r, v in zip(requests, results, strict=True)}
    inspection = engine.inspect(requests, expected_result_digests=anchors)
    # A fresh engine must reuse prior result bytes, without another provider call.
    resumed = _engine(root, a.manifest_digest)
    assert [resumed.infer(r).payload for r in requests] == [v.payload for v in results]
    assert resumed.tom_provider.calls == 0
    report = {'scope': 'controlled_shadow_not_formal_deployment', 'deployment_manifest_digest': a.manifest_digest,
        'engine_digest': digest(engine.identity), 'result_digests': anchors, 'inspection': inspection,
        'requests': len(requests), 'observer_rows': sum(len(v.record['rows']) for v in results),
        'cognitive_decisions': sum(len(row['decisions']) for v in results for row in v.record['rows']),
        'ToM_calls': engine.tom_provider.calls, 'restart_ToM_calls': resumed.tom_provider.calls,
        'real_weights_or_new_games': False}
    raw = canonical_json_bytes(report)
    with (root/'acceptance.json').open('xb') as stream:
        stream.write(raw)
    return {'summary_sha256': sha256_bytes(raw), **report}


def inspect_controlled(root, expected):
    root = Path(root).resolve()
    raw = (root/'acceptance.json').read_bytes()
    if sha256_bytes(raw) != expected:
        raise ValueError('external controlled summary anchor mismatch')
    report = json.loads(raw)
    if raw != canonical_json_bytes(report) or report['scope'] != 'controlled_shadow_not_formal_deployment':
        raise ValueError('invalid controlled acceptance record')
    engine = _engine(root, report['deployment_manifest_digest'], forbid_calls=True)
    if digest(engine.identity) != report['engine_digest']:
        raise ValueError('controlled engine identity mismatch')
    inspection = engine.inspect(artificial_requests(), expected_result_digests=report['result_digests'])
    if inspection != report['inspection']:
        raise ValueError('controlled inspection differs from sealed summary')
    return {'status': 'CONTROLLED_SHADOW_VERIFIED', 'summary_sha256': expected,
            'inspection': inspection, 'ToM_calls': engine.tom_provider.calls}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('controlled', 'inspect'))
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--expected-summary-digest')
    args = p.parse_args(argv)
    if args.command == 'controlled':
        if args.expected_summary_digest is not None:
            p.error('summary digest is for inspect only')
        result = controlled(args.destination)
    else:
        if args.expected_summary_digest is None:
            p.error('inspect requires an external summary SHA-256')
        result = inspect_controlled(args.destination, args.expected_summary_digest)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
