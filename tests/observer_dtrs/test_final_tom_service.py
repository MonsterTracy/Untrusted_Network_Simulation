"""Artificial capsule/PRE lifecycle and malformed IPC; no real models or games."""
from copy import deepcopy
from dataclasses import replace
import json
import random
import socket
import struct
import time

import numpy as np
import pytest
import torch

from tests.observer_dtrs.test_final_tom_capsule import artificial, load, contents
from tests.observer_dtrs.conftest import prefix_for
from werewolf.observer_dtrs.final_tom_client import ResidentOwner, ServiceTrust
from werewolf.observer_dtrs.final_tom_wire import (
    Channel, Limits, ProtocolError, PeerClosed, WireTimeout, ROOT, digest,
)


@pytest.fixture
def factory(artificial, tmp_path):
    owners = []
    model = load(artificial['capsule'])
    ident = model.identity
    trust = ServiceTrust(artificial['capsule'].manifest_digest, 'CONTROLLED_FIXTURE',
        ident['loader_source']['loader']['implementation_digest'], digest(ident['runtime']))
    def make(mode=None, limits=None, trust_override=None):
        owner = ResidentOwner(artificial['capsule'].path, trust=trust_override or trust,
            journal_root=tmp_path/f'owner-{len(owners)}', limits=limits or Limits(),
            cpu_threads=ident['runtime']['torch_threads'], interop_threads=ident['runtime']['torch_interop_threads'],
            _fixture_script=ROOT/'tests/observer_dtrs/resident_service_fixtures.py' if mode else None,
            _fixture_args=('--mode', mode, '--deny-root', str(artificial['root'])) if mode else ())
        owners.append(owner)
        return owner
    yield make, model
    for owner in owners:
        owner.close()
        if owner._process is not None and owner._process.poll() is None:
            owner._process.kill();owner._process.wait(timeout=3)


def wait_started(owner, ticket, timeout=3):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if any(e['kind'] == 'STARTED' and e['request_id'] == ticket.request_id for e in owner.journal.events): return
        if ticket.done.is_set(): pytest.fail(ticket.value.record)
        time.sleep(.005)
    pytest.fail('no STARTED within fixture deadline')


def test_resident_multiple_requests_exact_Q_source_and_rng(factory, artificial):
    make, original = factory;owner = make()
    snapshot = contents(artificial['capsule'].path)
    py, nps, tr = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    visible = {'prompt': 'fixed artificial prompt', 'votes': [1, 2], 'winner': None, 'events': []}
    before = deepcopy(visible)
    assert owner.start()
    pid, epoch, instance = owner.pid, owner.epoch, owner.session['instance']
    for ordinal, (phase, dead) in enumerate((('discussion', ()), ('pk_discussion', ()),
                                           ('discussion', ('player2',)), ('pk_discussion', ('player2', 'player7')))):
        p = prefix_for(phase=phase, dead=dead, game_id=f'controlled-{ordinal}')
        status, ticket = owner.submit(p, request_id=str(ordinal))
        assert status == 'SUBMITTED'
        result = ticket.result()
        assert result.status == 'SUCCEEDED', result.record
        assert result.matrix.values == original.predict(p).probabilities
        assert result.matrix.values[2][1] > 0 and result.matrix.values[2][6] > 0
        assert owner.pid == pid and owner.epoch == epoch and owner.session['instance'] == instance
        assert owner.submit(p, request_id=str(ordinal)) == ('DUPLICATE', ticket)
        assert owner.submit(p, request_id='alias-'+str(ordinal)) == ('DUPLICATE_PRE', ticket)
    assert visible == before and random.getstate() == py
    assert np.array_equal(np.random.get_state()[1], nps[1]) and torch.equal(torch.get_rng_state(), tr)
    assert np.random.get_state()[0] == nps[0] and np.random.get_state()[2:] == nps[2:]
    assert torch.get_num_threads() == threads and contents(artificial['capsule'].path) == snapshot
    assert owner.close() == 'CLOSED'
    audit = owner.inspect(expected_head=owner.journal.head)
    assert audit['submitted'] == audit['terminal'] == 4 and not audit['incomplete']
    assert len([e for e in audit['events'] if e['kind'] == 'READY']) == 1
    assert owner._process.poll() is not None


def test_service_does_not_import_training_and_denies_original_lifecycle(factory):
    make, _ = factory;owner = make('guarded')
    assert owner.start()
    _, t = owner.submit(prefix_for(), request_id='guarded')
    assert t.result().status == 'SUCCEEDED'


@pytest.mark.parametrize('mode,expected', [('cold_hang', 'COLD_START_TIMEOUT'), ('ready_hang', 'READY_TIMEOUT'),
                                          ('ready_source', 'PROTOCOL_ERROR')])
def test_startup_failure_kills_owned_process(factory, mode, expected):
    make, _ = factory
    limits = Limits()
    owner = make(mode, replace(limits, cold_start_s=.5 if mode == 'cold_hang' else limits.cold_start_s,
                               ready_s=.5 if mode == 'ready_hang' else limits.ready_s))
    assert not owner.start() and owner.state == 'UNHEALTHY'
    r = [e for e in owner.journal.events if e['kind'] == 'START_FAILED'][-1]['payload']
    assert r['status'] == expected and r['cleanup']['pid_exit_confirmed']
    assert owner._process.poll() is not None
    assert owner.submit(prefix_for(), request_id='no')[0] == 'SKIPPED_UNHEALTHY'


def test_runtime_trust_mismatch_is_not_silently_accepted(factory):
    make, model = factory
    ident = model.identity
    wrong = ServiceTrust(ident['capsule_digest'], ident['scope'],
                         ident['loader_source']['loader']['implementation_digest'], 'e'*64)
    owner = make(trust_override=wrong)
    assert not owner.start() and owner.state == 'UNHEALTHY' and owner._process.poll() is not None


@pytest.mark.parametrize('mode', ['wrong_token', 'wrong_epoch', 'wrong_instance', 'wrong_source', 'wrong_pre', 'bad_Q', 'partial', 'oversized'])
def test_bad_response_never_returns_Q_and_taints_instance(factory, mode):
    make, _ = factory;owner = make(mode, replace(Limits(), frame_s=.1))
    assert owner.start()
    _, ticket = owner.submit(prefix_for(), request_id='bad')
    r = ticket.result()
    assert r.status == 'PROTOCOL_ERROR', r.record
    assert r.record['prediction'] is None and owner.state == 'UNHEALTHY'
    assert owner.cleanup['pid_exit_confirmed'] and owner._process.poll() is not None
    with pytest.raises(ProtocolError): _ = r.matrix


def test_crash_and_at_most_once(factory):
    make, _ = factory;owner = make('crash');assert owner.start()
    p = prefix_for();_, t = owner.submit(p, request_id='crash')
    assert t.result().status == 'SERVICE_CRASH' and owner._process.poll() is not None
    assert owner.submit(p, request_id='crash') == ('DUPLICATE', t)
    assert sum(e['kind'] == 'DISPATCHED' for e in owner.journal.events) == 1


def test_confirmed_cancel_waits_for_thread_then_reuses_same_instance(factory):
    make, _ = factory;owner = make('delay');assert owner.start()
    pid = owner.pid;_, t = owner.submit(prefix_for(), request_id='cancel')
    wait_started(owner, t)
    r = owner.cancel(t)
    assert r.status == 'CLIENT_CANCELLED' and r.record['cancellation'] == 'CONFIRMED_IDLE'
    assert r.record['execution_stopped'] and owner.state == 'IDLE' and owner.pid == pid
    _, next_ticket = owner.submit(prefix_for(game_id='next'), request_id='next')
    assert next_ticket.result().status == 'SUCCEEDED'


def test_service_deadline_is_distinct_from_client_timeout(factory):
    make, _ = factory
    owner = make('service_timeout', replace(Limits(), service_s=.05, query_s=2.))
    assert owner.start();_, t = owner.submit(prefix_for(), request_id='timeout')
    r = t.result()
    assert r.status == 'SERVICE_TIMEOUT' and r.record['execution_stopped'] and owner.state == 'IDLE'


def test_client_timeout_late_success_discarded_then_idle(factory):
    make, _ = factory;owner = make('late_success');assert owner.start()
    _, t = owner.submit(prefix_for(), request_id='late', timeout_s=.1)
    r = t.result()
    assert r.status == 'CLIENT_TIMEOUT' and r.record['prediction'] is None
    assert r.record['cancellation'] == 'CONFIRMED_IDLE' and owner.state == 'IDLE'
    assert any(e['kind'] == 'LATE_SUCCESS_REJECTED' for e in owner.journal.events)


@pytest.mark.parametrize('mode', ['hang', 'ignore_term'])
def test_unconfirmed_cancel_kills_and_disables_old_service(factory, mode):
    make, _ = factory;owner = make(mode, replace(Limits(), cancel_s=.1, terminate_s=.1))
    assert owner.start();_, t = owner.submit(prefix_for(), request_id='stuck', timeout_s=.2)
    r = t.result()
    assert r.status == 'CANCEL_UNCONFIRMED' and r.record['trigger_status'] == 'CLIENT_TIMEOUT'
    assert owner.state == 'UNHEALTHY' and owner._process.poll() is not None and r.record['execution_stopped']
    if mode == 'ignore_term': assert owner.cleanup['kill_sent']
    assert owner.submit(prefix_for(game_id='new'), request_id='new')[0] == 'SKIPPED_UNHEALTHY'


def test_failed_cleanup_blocks_recovery_until_exit_verified(factory, monkeypatch):
    make, _ = factory;owner = make('hang', replace(Limits(), cancel_s=.05))
    assert owner.start();original = owner._stop_owned
    def cannot_stop():
        owner.cleanup = {'pid_exit_confirmed': False, 'terminate_sent': False, 'kill_sent': False}
        return False
    monkeypatch.setattr(owner, '_stop_owned', cannot_stop)
    _, t = owner.submit(prefix_for(), request_id='not-stopped', timeout_s=.1)
    r = t.result()
    assert r.status == 'CANCEL_UNCONFIRMED' and not r.record['execution_stopped']
    assert owner._process.poll() is None and owner.state == 'UNHEALTHY'
    with pytest.raises(ProtocolError, match='not confirmed'): owner.recover()
    assert owner.submit(prefix_for(game_id='new'), request_id='new')[0] == 'SKIPPED_UNHEALTHY'
    monkeypatch.setattr(owner, '_stop_owned', original);assert original()


def test_bounded_queue_and_pending_cancel_does_not_cancel_other_query(factory):
    make, _ = factory;owner = make('delay', replace(Limits(), max_pending=2));assert owner.start()
    _, a = owner.submit(prefix_for(game_id='a'), request_id='a');wait_started(owner, a)
    _, b = owner.submit(prefix_for(game_id='b'), request_id='b')
    assert owner.submit(prefix_for(game_id='c'), request_id='c')[0] == 'SKIPPED_BACKPRESSURE'
    assert owner.cancel(b).record['cancellation'] == 'NOT_DISPATCHED'
    assert a.result().status == 'SUCCEEDED' and b.result().status == 'CLIENT_CANCELLED'
    assert not any(e['kind'] == 'DISPATCHED' and e['request_id'] == 'b' for e in owner.journal.events)


def test_explicit_recovery_changes_epoch_and_does_not_replay(factory):
    make, _ = factory;owner = make('crash_on_game');assert owner.start()
    epoch, instance = owner.epoch, owner.session['instance']
    p = prefix_for();_, t = owner.submit(p, request_id='failed');assert t.result().status == 'SERVICE_CRASH'
    assert owner.recover() and owner.epoch != epoch and owner.session['instance'] != instance
    assert owner.submit(p, request_id='failed') == ('DUPLICATE', t)
    _, t2 = owner.submit(prefix_for(game_id='fresh'), request_id='fresh')
    assert t2.result().status == 'SUCCEEDED'


def test_close_timeout_is_bounded_and_process_exit_confirmed(factory):
    make, _ = factory;owner = make('close_hang', replace(Limits(), close_s=.1))
    assert owner.start();start = time.monotonic()
    assert owner.close() == 'CLOSE_TIMEOUT' and time.monotonic()-start < 2
    assert owner._process.poll() is not None


def test_write_timeout_does_not_retry_uncertain_request(factory):
    make, _ = factory;owner = make('write_hang', replace(Limits(), write_s=.05))
    assert owner.start();owner._channel.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    _, t = owner.submit(prefix_for(game_id='x'*400000), request_id='write')
    assert t.result().status == 'WRITE_TIMEOUT'
    assert owner.state == 'UNHEALTHY' and owner._process.poll() is not None


def test_audit_failure_stops_admission_and_does_not_return_unsealed_Q(factory, monkeypatch):
    make, _ = factory;owner = make();assert owner.start()
    original = owner.journal.append
    def unavailable(kind, **kw):
        if kind == 'OUTCOME': raise OSError('controlled disk failure')
        return original(kind, **kw)
    monkeypatch.setattr(owner.journal, 'append', unavailable)
    _, t = owner.submit(prefix_for(), request_id='audit')
    assert t.result().status == 'AUDIT_UNAVAILABLE'
    assert owner.audit_unavailable and owner.submit(prefix_for(game_id='next'), request_id='next')[0] == 'AUDIT_UNAVAILABLE'
    assert owner.inspect()['incomplete'] == ['audit']


def test_corrupt_journal_rejected_without_repair(factory):
    make, _ = factory;owner = make();assert owner.start()
    _, t = owner.submit(prefix_for(), request_id='audit');assert t.result().status == 'SUCCEEDED'
    owner.close();head = owner.journal.head
    assert not owner.inspect(expected_head=head)['incomplete']
    path = owner.journal.root/'events/00000000/record.json'
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError): owner.inspect(expected_head=head)
    assert path.read_bytes() == b'corrupt'


@pytest.mark.parametrize('case', ['oversized', 'partial', 'duplicate_json', 'cumulative_read', 'cumulative_write', 'peer_closed'])
def test_framing_bounds(case):
    a, b = socket.socketpair()
    limits = replace(Limits(), max_frame_bytes=64, max_request_bytes=32, max_read_bytes=64, max_write_bytes=64, frame_s=.03)
    c = Channel(a, limits)
    try:
        if case == 'oversized': b.sendall(struct.pack('!I', 65))
        elif case == 'partial': b.sendall(struct.pack('!I', 20)+b'{')
        elif case == 'duplicate_json':
            raw = b'{"x":1,"x":2}';b.sendall(struct.pack('!I', len(raw))+raw)
        elif case == 'cumulative_read':
            for _ in range(10):
                raw = b'{"x":1}';b.sendall(struct.pack('!I', len(raw))+raw)
                try: c.read(time.monotonic()+.1)
                except ProtocolError: return
            pytest.fail('no cumulative limit')
        elif case == 'cumulative_write':
            with pytest.raises(ProtocolError):
                for _ in range(10): c.write({'x': 1}, time.monotonic()+.1)
            return
        else:
            b.close()
            with pytest.raises(PeerClosed): c.read(time.monotonic()+.1)
            return
        if case == 'partial':
            with pytest.raises(WireTimeout): c.read(time.monotonic()+.1)
            time.sleep(.01)
        with pytest.raises(ProtocolError): c.read(time.monotonic()+.1)
    finally:
        c.close();b.close()


def test_cancel_inflight_on_close_cannot_leave_live_service(factory):
    make, _ = factory
    owner = make('hang', replace(Limits(), close_s=.2, cancel_s=.1, terminate_s=.1))
    assert owner.start();_, t = owner.submit(prefix_for(), request_id='closing')
    wait_started(owner, t)
    start = time.monotonic();owner.close()
    assert time.monotonic()-start < 3 and owner._process.poll() is not None
    assert t.done.is_set() and t.value.status != 'SUCCEEDED'
    assert not owner.inspect()['incomplete']


def test_queue_deadline_expires_without_second_execution(factory):
    make, _ = factory;owner = make('delay');assert owner.start()
    _, a = owner.submit(prefix_for(game_id='a'), request_id='a');wait_started(owner, a)
    _, b = owner.submit(prefix_for(game_id='b'), request_id='b', timeout_s=.02)
    assert a.result().status == 'SUCCEEDED'
    assert b.result().status == 'CLIENT_TIMEOUT'
    assert b.value.record['cancellation'] == 'NOT_DISPATCHED'
    assert not any(e['kind'] == 'DISPATCHED' and e['request_id'] == 'b' for e in owner.journal.events)
    assert not a.prefix_bytes and not b.prefix_bytes


def test_consumer_wait_timeout_triggers_verified_cancellation(factory):
    make, _ = factory;owner = make('delay');assert owner.start()
    _, t = owner.submit(prefix_for(), request_id='short-wait');wait_started(owner, t)
    r = t.result(timeout=.01)
    assert r.status == 'CLIENT_TIMEOUT' and r.record['execution_stopped']
    assert owner.state == 'IDLE'


def test_quota_and_request_id_conflict(factory):
    make, _ = factory;owner = make(limits=replace(Limits(), max_requests=1));assert owner.start()
    _, t = owner.submit(prefix_for(), request_id='one');assert t.result().status == 'SUCCEEDED'
    with pytest.raises(ProtocolError, match='another PRE'):
        owner.submit(prefix_for(game_id='changed'), request_id='one')
    assert owner.submit(prefix_for(game_id='two'), request_id='two')[0] == 'SKIPPED_QUOTA'
    exposed = owner.session;exposed['capsule']['capsule_digest'] = 'e'*64
    assert owner.session['capsule']['capsule_digest'] != 'e'*64


def test_mutated_owner_source_contract_is_rejected_before_spawn(factory):
    make, _ = factory;owner = make()
    owner.limits = replace(owner.limits, max_pending=3)
    assert not owner.start() and owner.state == 'UNHEALTHY' and owner._process is None


@pytest.mark.parametrize('field', ['loader_digest', 'capsule_digest'])
def test_capsule_loader_trust_mismatch(factory, field):
    make, model = factory;i = model.identity
    trust = ServiceTrust(i['capsule_digest'], i['scope'],
                          i['loader_source']['loader']['implementation_digest'], digest(i['runtime']))
    owner = make(trust_override=replace(trust, **{field: 'e'*64}))
    assert not owner.start() and owner.state == 'UNHEALTHY' and owner._process.poll() is not None


def test_audit_tail_corruption_disables_new_admission(factory):
    make, _ = factory;owner = make();assert owner.start()
    tail = owner.journal.root/'events'/f'{len(owner.journal.events)-1:08d}'/'record.json'
    tail.write_bytes(b'corrupt')
    status, ticket = owner.submit(prefix_for(), request_id='blocked')
    assert status == 'AUDIT_UNAVAILABLE' and ticket is None
    assert owner.audit_unavailable and owner.state == 'UNHEALTHY'


@pytest.mark.parametrize('change', ['pre', 'Q', 'source'])
def test_rehashed_audit_prediction_mismatch_is_rejected(factory, change):
    from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
    make, _ = factory;owner = make();assert owner.start()
    _, t = owner.submit(prefix_for(), request_id='audited');assert t.result().status == 'SUCCEEDED'
    owner.close()
    previous = None
    for node in sorted((owner.journal.root/'events').iterdir()):
        path = node/'record.json';r = json.loads(path.read_bytes())
        if r['kind'] == 'OUTCOME':
            if change == 'pre': r['payload']['prediction']['game_id'] = 'wrong-game'
            elif change == 'Q': r['payload']['prediction']['probabilities'][0][1] = .99
            else: r['payload']['session']['capsule']['capsule_digest'] = 'e'*64
        r['previous'] = previous;raw = canonical_json_bytes(r);path.write_bytes(raw)
        manifest = json.loads((node/'manifest.json').read_bytes())
        manifest['file_table']['record.json'] = {'sha256': sha256_bytes(raw), 'byte_size': len(raw)}
        manifest.pop('manifest_digest');manifest['manifest_digest'] = digest(manifest)
        (node/'manifest.json').write_bytes(canonical_json_bytes(manifest))
        previous = digest(r)
    with pytest.raises(ValueError): owner.inspect()


def test_owner_import_is_independent_of_torch_and_training(tmp_path):
    import os
    import subprocess
    import sys
    code = """
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.startswith(('torch','werewolf.tom.dataset','werewolf.tom.qwen3_final',
    'werewolf.development_publication','werewolf.observer_dtrs.final_tom_capsule',
    'werewolf.observer_dtrs.shadow','werewolf.envs','werewolf.agents')):
   raise AssertionError('owner imported forbidden module '+fullname)
sys.meta_path.insert(0,Block())
from werewolf.observer_dtrs.final_tom_client import ResidentOwner,ServiceTrust
print('PASS_OWNER_IMPORT_ISOLATION')
"""
    p = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True,
        env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1'}, timeout=10)
    assert p.returncode == 0, p.stdout+p.stderr


@pytest.mark.parametrize('dead', [(), ('player2',)])
def test_capsule_source_is_not_legacy_shadow_identity_and_controlled_p_bridge(factory, dead):
    from tests.observer_dtrs.test_shadow import request
    from werewolf.observer_dtrs.features import observer_features
    from werewolf.observer_dtrs.online_model import TomIdentity
    from werewolf.observer_dtrs.d3_protocol import load_protocol
    from werewolf.observer_dtrs.decision import decide
    make, _ = factory;owner = make();assert owner.start()
    r = request(dead=dead);_, t = owner.submit(r.prefix, request_id='bridge')
    out = t.result();assert out.status == 'SUCCEEDED'
    with pytest.raises((TypeError, ValueError)):
        TomIdentity(**owner.session)
    # Explicit artificial binding, not a real CAL estimator or legacy package conversion.
    class ArtificialProbability:
        def __init__(self, expected_capsule):
            self.expected_capsule = expected_capsule
        def predict(self, features, capsule_digest):
            if capsule_digest != self.expected_capsule: raise ValueError('fixture source mismatch')
            return .5
    model = ArtificialProbability(owner.trust.capsule_digest)
    protocol = load_protocol()
    for context in r.contexts:
        features = observer_features(context, out.matrix)
        p = model.predict(features, owner.session['capsule']['capsule_digest'])
        for i in range(45):
            for policy in protocol.policies:
                assert decide(p, protocol.loss(i, policy)).to_record()['probability_exact'] == '1/2'
    with pytest.raises(ValueError): model.predict(features, 'e'*64)
