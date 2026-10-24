"""Artificial providers and PREs only. Never load real ToM or run a game."""
from copy import deepcopy
from dataclasses import replace
import inspect
import json
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tests.observer_dtrs.conftest import prefix_for
from tests.observer_dtrs.test_estimation import fit, row
from tests.observer_dtrs.test_sealed_d2 import rewrite
from werewolf.artifact_io import ArtifactValidationError
from werewolf.structured_history import plan_structured_history
from werewolf.observer_dtrs.d2_protocol import frozen_protocol
from werewolf.observer_dtrs.decision import decide
from werewolf.observer_dtrs.deployment_offline import publish_controlled_estimator
from werewolf.observer_dtrs.online_context import build_shadow_request
from werewolf.observer_dtrs.online_model import TomIdentity, read_deployment
from werewolf.observer_dtrs.probabilities import ProbabilityMatrix
from werewolf.observer_dtrs.shadow import ShadowEngine, ShadowQueue, QPrediction
from werewolf.observer_dtrs.shadow_artifacts import ShadowRecordedFailure


def tom_identity():
    return TomIdentity('controlled_fixture', None, '1'*64, '2'*64, '3'*64, '4'*64, None)


def observation(prefix, team=(1, 2)):
    runtime = 'speech_pk' if prefix.public_temporal_state.phase.value == 'pk_discussion' else 'speech'
    actor = int(prefix.current_speaker[-1])
    return {'identity': 'Werewolf', 'observer_id': actor, 'current_act_idx': actor,
        'phase': f'1_day_{runtime}', 'authoritative_public_state': {'day': 1, 'day_or_night': 'day',
            'phase': runtime, 'alive_players': [int(s[-1]) for s in prefix.alive_observer_ids]},
        'game_log': [SimpleNamespace(event='werewolf_team_info', content={'wolf_team': list(team)})]}


def request(*, phase='discussion', dead=(), actor='player1', team=(1, 2), attempt='attempt-1'):
    p = prefix_for(phase=phase, dead=dead, actor=actor)
    return build_shadow_request(p, observation(p, team), shadow_run_id='artificial-shadow', attempt_id=attempt)


class ArtificialTom:
    def __init__(self, identity=None, behavior=None):
        self.identity = identity or tom_identity()
        self.calls = 0
        self.behavior = behavior

    def predict(self, prefix, *, deadline_monotonic):
        self.calls += 1
        assert deadline_monotonic > 0
        if self.behavior:
            return self.behavior(prefix)
        values = [[0. if i == j else 1/6 for j in range(7)] for i in range(7)]
        values[2] = [.1, .2, 0., .1, .1, .2, .3]
        return QPrediction(ProbabilityMatrix(prefix.game_id, prefix.boundary_id, prefix.prefix_digest, values),
                           self.identity, plan_structured_history(prefix).plan_digest)


@pytest.fixture
def deployed(tmp_path):
    m = fit([row('one', 0, .1), row('two', 1, .9)], frozen_protocol())
    a = publish_controlled_estimator(tmp_path/'model', m, tom=tom_identity(), selection_rule_digest='5'*64)
    return read_deployment(a.path, expected_manifest_digest=a.manifest_digest, expected_tom=tom_identity())


def engine(tmp_path, deployed, **kw):
    tom = kw.pop('tom_provider', ArtificialTom())
    return ShadowEngine(tom_provider=tom, probability_provider=deployed,
        expected_deployment_digest=deployed.manifest_digest, journal_root=tmp_path/'journal', **kw)


@pytest.mark.parametrize('phase', ['discussion', 'pk_discussion'])
@pytest.mark.parametrize('dead', [(), ('player2', 'player7')])
def test_legal_population_features_and_all_costs(tmp_path, deployed, phase, dead):
    r = request(phase=phase, dead=dead)
    e = engine(tmp_path, deployed)
    result = e.infer(r).record
    assert len(result['rows']) == (4 if dead else 5)
    observer = next(x for x in result['rows'] if x['observer_id'] == 'player3')
    f = observer['features']
    assert f['s'] == .1 and f['m'] == (0 if dead else .2)
    assert f['r'] == pytest.approx(.1 if dead else .3)
    assert f['dW'] == (None if dead else .1) and f['phase'] == phase
    assert result['Q'][2][1] == .2 and result['Q'][2][6] == .3
    assert result['request']['step_index'] == r.prefix.step_index
    assert len(observer['decisions']) == 90
    for d in observer['decisions']:
        policy = next(p for p in e.protocol.policies if p.value == d['tie_policy'])
        expected = decide(observer['probability'], e.protocol.loss(int(d['scenario_id'][1:]), policy)).to_record()
        assert d == {'scenario_id': d['scenario_id'], **expected}
    assert e.inspect([r])['verified_results'] == 1 and e.tom_provider.calls == 1


def test_arbitrary_absolute_seats_and_dead_observers(tmp_path, deployed):
    r = request(actor='player5', team=(5, 3), dead=('player2', 'player7'))
    assert r.known_wolf_ids == ('player3', 'player5')
    e = engine(tmp_path, deployed)
    assert {x['observer_id'] for x in e.infer(r).record['rows']} == {'player1', 'player4', 'player6'}


@pytest.mark.parametrize('kind', ['identity', 'observer', 'actor', 'alive', 'phase', 'day', 'team', 'duplicate_team'])
def test_bad_observation_is_rejected_without_provider(kind):
    p = prefix_for();o = observation(p)
    if kind == 'identity': o['identity'] = 'Seer'
    elif kind == 'observer': o['observer_id'] = 2
    elif kind == 'actor': o['current_act_idx'] = 2
    elif kind == 'alive': o['authoritative_public_state']['alive_players'] = [1, 2, 3]
    elif kind == 'phase': o['authoritative_public_state']['phase'] = 'speech_pk'
    elif kind == 'day': o['authoritative_public_state']['day'] = True
    elif kind == 'team': o['game_log'][0].content['wolf_team'] = [1, 1]
    else: o['game_log'] *= 2
    with pytest.raises((ValueError, KeyError)):
        build_shadow_request(p, o, shadow_run_id='s', attempt_id='a')


def test_only_allowlisted_fields_are_read_and_no_state_or_rng_changes(tmp_path, deployed):
    class Trap:
        def __getattribute__(self, key):
            raise AssertionError('forbidden object touched: '+key)
    p = prefix_for();o = observation(p)
    untouched = deepcopy(o)
    for name in ('supervision', 'Y', 'role_sidecar', 'reports', 'env', 'agent'):
        o[name] = Trap()
    state = {'vote': [None]*7, 'memory': ['original'], 'events': ['original'], 'budget': 5}
    original = deepcopy(state);py = random.getstate();np_before = np.random.get_state()
    r = build_shadow_request(p, o, shadow_run_id='s', attempt_id='a')
    e = engine(tmp_path, deployed);first = e.infer(r).record
    for name in ('supervision', 'Y', 'role_sidecar', 'reports'):
        o[name] = {'changed': True}
    r2 = build_shadow_request(p, o, shadow_run_id='s', attempt_id='a')
    assert r.digest == r2.digest and e.infer(r2).record == first
    assert state == original and random.getstate() == py
    np_after = np.random.get_state()
    assert np_before[0] == np_after[0] and np.array_equal(np_before[1], np_after[1]) and np_before[2:] == np_after[2:]
    for k in untouched:
        assert o[k] == untouched[k]
    def keys(value):
        if isinstance(value, dict):
            return set(value).union(*(keys(v) for v in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(v) for v in value))
        return set()
    assert not keys(first) & {'Y', 'y', 'supervision', 'raw_suspicion_support', 'role_sidecar', 'action', 'training_stats'}
    assert set(inspect.signature(e.infer).parameters) == {'request'}


def test_duplicates_and_restart_do_not_repeat_tom(tmp_path, deployed):
    r = request();e = engine(tmp_path, deployed)
    first = e.infer(r)
    second = engine(tmp_path, deployed)
    assert second.infer(r).payload == first.payload and second.tom_provider.calls == 0
    assert second.inspect([r])['verified_results'] == 1
    before = second.journal.inventory()
    with pytest.raises(ValueError): second.inspect([replace(r, attempt_id='different')])
    assert second.journal.inventory() == before and second.tom_provider.calls == 0


@pytest.mark.parametrize('mode', ['nan', 'logq', 'wrong_pre', 'wrong_source', 'wrong_plan', 'timeout'])
def test_technical_failure_is_terminal_not_bnd(tmp_path, deployed, mode):
    def broken(prefix):
        if mode == 'timeout': raise TimeoutError('secret target text never logged')
        q = ArtificialTom().predict(prefix, deadline_monotonic=1)
        if mode in ('nan', 'logq'):
            values = [list(x) for x in q.matrix.values];values[0][1] = float('nan') if mode == 'nan' else -3.
            return replace(q, matrix=replace(q.matrix, values=values))
        if mode == 'wrong_pre': return replace(q, matrix=replace(q.matrix, boundary_id='other'))
        if mode == 'wrong_source': return replace(q, tom=replace(q.tom, weights_digest='f'*64))
        return replace(q, plan_digest='f'*64)
    e = engine(tmp_path, deployed, tom_provider=ArtificialTom(behavior=broken));r = request()
    with pytest.raises((ValueError, TimeoutError)): e.infer(r)
    with pytest.raises(ShadowRecordedFailure): e.infer(r)
    assert e.tom_provider.calls == 1 and e.inspect([r])['verified_failures'] == 1
    raw = (e.journal.root/'failures'/r.digest/'record.json').read_text()
    assert 'secret target' not in raw and json.loads(raw)['rows'] == []


def test_return_after_deadline_is_rejected(tmp_path, deployed):
    times = iter((1., 4.))
    e = engine(tmp_path, deployed, timeout_seconds=2, clock=lambda: next(times))
    with pytest.raises(TimeoutError): e.infer(request())


def test_interrupted_claim_inspection_is_readonly_then_recovery_stops(tmp_path, deployed):
    e = engine(tmp_path, deployed);r = request()
    with e.journal.lock(): assert e.journal.begin(r) is None
    assert e.inspect([r])['pending'] == 1 and not (e.journal.root/'failures').exists()
    with pytest.raises(ShadowRecordedFailure, match='INTERRUPTED'): e.infer(r)
    assert e.tom_provider.calls == 0


@pytest.mark.parametrize('mode', ['truncated', 'missing', 'rehash_probability', 'rehash_source'])
def test_damaged_log_stops_without_repair(tmp_path, deployed, mode):
    e = engine(tmp_path, deployed);r = request();e.infer(r)
    root = e.journal.root/'results'/r.digest
    if mode == 'truncated': (root/'record.json').write_bytes(b'broken')
    elif mode == 'missing': (root/'record.json').unlink()
    elif mode == 'rehash_probability': rewrite(root, transform=lambda v: v['rows'][0].__setitem__('probability', .123))
    else: rewrite(root, transform=lambda v: v['tom'].__setitem__('weights_digest', 'f'*64))
    before = {str(p): p.read_bytes() for p in e.journal.root.rglob('*') if p.is_file()}
    with pytest.raises((ValueError, ArtifactValidationError)): e.infer(r)
    assert e.tom_provider.calls == 1
    assert {str(p): p.read_bytes() for p in e.journal.root.rglob('*') if p.is_file()} == before


def test_log_publication_failure_does_not_return_success(tmp_path, deployed, monkeypatch):
    e = engine(tmp_path, deployed);r = request()
    def fail(*a, **kw): raise OSError('controlled disk failure')
    monkeypatch.setattr(e.journal, 'finish', fail)
    with pytest.raises(OSError): e.infer(r)
    assert not (e.journal.root/'results').exists()
    with pytest.raises(ShadowRecordedFailure): engine(tmp_path, deployed).infer(r)


def test_bounded_queue_never_calls_provider_on_submit(tmp_path, deployed):
    e = engine(tmp_path, deployed);q = ShadowQueue(1);r = request()
    assert q.submit(r)['status'] == 'QUEUED'
    assert q.submit(r)['status'] == 'DUPLICATE'
    assert q.submit(replace(r, attempt_id='different'))['status'] == 'SKIPPED_BACKPRESSURE'
    assert e.tom_provider.calls == 0
    assert q.work_one(e)['status'] == 'SUCCEEDED' and q.work_one(e) is None


def test_online_import_closure_has_no_private_or_game_modules():
    code = 'import sys,json; import werewolf.observer_dtrs.shadow; print(json.dumps(sorted(sys.modules)))'
    r = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, check=True)
    loaded = json.loads(r.stdout)
    banned = ('werewolf.observer_dtrs.deployment_offline', 'werewolf.observer_dtrs.dataset_offline',
        'werewolf.observer_dtrs.training_audit_offline', 'werewolf.observer_dtrs.labels',
        'werewolf.observer_dtrs.nested_validation', 'werewolf.observer_dtrs.sealed_d2_offline',
        'werewolf.development_publication', 'werewolf.envs', 'werewolf.agents', 'werewolf.phase2', 'run_random', 'torch')
    assert not any(m.startswith(banned) for m in loaded)


def test_external_result_anchor_detects_fully_rehashed_q_replacement(tmp_path, deployed):
    from werewolf.observer_dtrs.online_context import digest
    e = engine(tmp_path, deployed);r = request()
    first = e.infer(r)
    assert e.inspect([r], expected_result_digests={r.digest: first.digest})['external_result_anchors_verified']
    def rehash_all(record):
        q = record['Q'];q[2][0] = .15;q[2][1] = .15
        matrix = ProbabilityMatrix(r.prefix.game_id, r.prefix.boundary_id, r.prefix.prefix_digest, q)
        record.update(Q_digest=digest(q), rows=e._rows(r, matrix))
    rewrite(e.journal.root/'results'/r.digest, transform=rehash_all)
    # Self-consistency alone cannot authenticate a substituted, recomputed Q.
    assert not e.inspect([r])['external_result_anchors_verified']
    with pytest.raises(ValueError, match='external result anchors'):
        e.inspect([r], expected_result_digests={r.digest: first.digest})
    assert e.tom_provider.calls == 1


def test_source_or_contract_change_fails_before_any_new_prediction(tmp_path, deployed):
    e = engine(tmp_path, deployed);r = request();e.infer(r)
    e.tom_provider.identity = replace(tom_identity(), seal_digest='f'*64)
    with pytest.raises(ValueError, match='identity changed'):e.infer(r)
    assert e.tom_provider.calls == 1
    with pytest.raises((ValueError, ArtifactValidationError)):
        engine(tmp_path, deployed, timeout_seconds=20.)


def test_nested_input_artifact_or_foreign_directory_cannot_be_a_sink(tmp_path, deployed):
    with pytest.raises(ValueError, match='immutable artifact'):
        ShadowEngine(tom_provider=ArtificialTom(), probability_provider=deployed,
            expected_deployment_digest=deployed.manifest_digest, journal_root=tmp_path/'model'/'child')
    root = tmp_path/'foreign';root.mkdir();(root/'research.txt').write_text('preserve')
    with pytest.raises(ValueError, match='unrelated'):
        ShadowEngine(tom_provider=ArtificialTom(), probability_provider=deployed,
            expected_deployment_digest=deployed.manifest_digest, journal_root=root)
    assert (root/'research.txt').read_text() == 'preserve'


def test_capacity_failure_does_not_crop_pre_or_call_provider(tmp_path, deployed):
    tiny = replace(tom_identity(), max_seq_len=4)
    # Only source metadata changes in this explicitly synthetic fixture.
    m = fit([row('one', 0), row('two', 1)], frozen_protocol())
    a = publish_controlled_estimator(tmp_path/'tiny-model', m, tom=tiny, selection_rule_digest='6'*64)
    online = read_deployment(a.path, expected_manifest_digest=a.manifest_digest, expected_tom=tiny)
    e = engine(tmp_path, online, tom_provider=ArtificialTom(tiny))
    r = request(phase='pk_discussion')
    assert plan_structured_history(r.prefix).token_count > 4
    with pytest.raises(ValueError, match='capacity'):e.infer(r)
    assert e.tom_provider.calls == 0 and e.journal.inventory()['claims'] == 0


def test_rehashed_contract_and_forged_failure_do_not_pass_inspect(tmp_path, deployed):
    e = engine(tmp_path, deployed);r = request();e.infer(r)
    rewrite(e.journal.root/'contract', transform=lambda v: v.__setitem__('scope', 'fake'))
    with pytest.raises(ValueError, match='contract source'): e.inspect([r])
    with pytest.raises(ValueError, match='contract source'): e.infer(r)
    assert e.tom_provider.calls == 1


def test_inspect_refuses_concurrent_writer(tmp_path, deployed):
    e = engine(tmp_path, deployed);r = request();e.infer(r)
    with e.journal.lock():
        with pytest.raises(RuntimeError, match='writer'):e.inspect([r])


def test_artificial_cli_and_anchored_inspect(tmp_path):
    from scripts.check_observer_dtrs_shadow import controlled, inspect_controlled, main
    root = tmp_path/'cli-run'
    result = controlled(root)
    assert (result['requests'], result['observer_rows'], result['cognitive_decisions']) == (4, 18, 1620)
    assert result['ToM_calls'] == 4 and result['restart_ToM_calls'] == 0
    assert inspect_controlled(root, result['summary_sha256'])['ToM_calls'] == 0
    with pytest.raises(ValueError): inspect_controlled(root, 'f'*64)
    with pytest.raises(ValueError): controlled(root)
    with pytest.raises(SystemExit):main(['controlled','--destination',str(tmp_path/'bad'),'--run','real-D2'])
    assert not (tmp_path/'bad').exists()


def test_rehashed_failure_with_supervision_field_is_rejected(tmp_path, deployed):
    def fail(prefix):raise TimeoutError('controlled')
    e = engine(tmp_path, deployed, tom_provider=ArtificialTom(behavior=fail));r = request()
    with pytest.raises(TimeoutError):e.infer(r)
    rewrite(e.journal.root/'failures'/r.digest, transform=lambda v:v.__setitem__('Y', 1))
    with pytest.raises(ValueError):e.inspect([r])
    with pytest.raises(ValueError):e.infer(r)
