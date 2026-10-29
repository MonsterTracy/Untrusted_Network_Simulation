"""Local artificial files/processes only. No env, game, model or randomization."""
from dataclasses import asdict
import json
import multiprocessing as mp
import os
from pathlib import Path
import threading

import pytest

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.observer_dtrs.action_evidence_ledger import digest, extend_ledger, validate_ledger
from werewolf.observer_dtrs.action_evidence_store import ActionEvidenceStore, StoreError, reconcile_readback
from werewolf.observer_dtrs import action_evidence_store as module
from tests.observer_dtrs.action_evidence_fixture import (
    LEDGER, FakeWorld, assign, committed, day, execution, opportunity, reservation, unsubmitted,
)


def namespace():
    op = opportunity()
    return dict(ledger_id=LEDGER, game_id=op['game_id'], game_attempt_id=op['game_attempt_id'])


def create(path):
    return ActionEvidenceStore.create(path/'journal', path/'custody', **namespace())


def reopen(path, anchor):
    return ActionEvidenceStore.reopen(path/'journal', path/'custody', expected_anchor=anchor, **namespace())


class Journal:
    """Reuse Phase-2D scenario helpers without changing their protected source."""
    def __init__(self, store):
        self.store = store

    @property
    def records(self):
        return self.store.records

    def append(self, operation, kind, payload):
        receipt = self.store.append(operation, kind, payload)
        assert receipt.execution_permission is False
        return next(r for r in self.records if r['record_digest'] == receipt.record_digest)


def setup_reserved(store):
    journal = Journal(store)
    op, a, expected = assign(journal)
    e = execution(journal, a, expected)
    r = reservation(journal, a, e)
    journal.append('unknown', 'COMMIT_WITNESS', dict(assignment_ref=a, status='COMMIT_UNKNOWN',
                   reservation_ref=r['record_digest'], evidence_ref=digest('artificial-unknown')))
    return journal, op, a, r


def read_all(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob('*') if p.is_file()}


def checkpoint(path):
    # TEST ORACLE ONLY: production must obtain this from independent trusted custody.
    return json.loads((path/'custody/anchors.jsonl').read_bytes().splitlines()[-1])['anchor_digest']


def test_roundtrip_complete_fixed_assignment_and_durable_receipt(tmp_path):
    with create(tmp_path) as store:
        j, op, a, reserved = setup_reserved(store)
        world = FakeWorld()
        result = world.commit(op, reserved['payload']['attempt_token'])
        j.append('readback', 'COMMIT_WITNESS', committed(a, reserved, result))
        j.append('day', 'DAY_WITNESS', day(a, op))
        saved = store.anchor_digest
        assert store.inspect().assignments[0].observed_no_exile is True
        assert not store.inspect().external_exactly_once_proven
        assert not store.inspect().external_evidence_truth_verified
    with reopen(tmp_path, saved) as store:
        assert store.inspect().assignments[0].commit_status == 'COMMITTED'
        assert len(store.records) == 8
        assert all(p.stat().st_mode & 0o777 == 0o600 for p in tmp_path.rglob('*') if p.is_file())


def test_unanchored_reopen_rejected_even_empty_and_pure_extend_not_a_permit(tmp_path):
    with create(tmp_path):
        pass
    before = read_all(tmp_path)
    with pytest.raises(StoreError, match='anchor'):
        reopen(tmp_path, None)
    records = extend_ledger((), ledger_id=LEDGER, operation_id='op', kind='OPPORTUNITY', payload=opportunity())
    assert len(records) == 1
    assert read_all(tmp_path) == before


def test_stale_head_and_valid_prefix_rollback_rejected(tmp_path):
    with create(tmp_path) as store:
        genesis = store.anchor_digest
        store.append('op', 'OPPORTUNITY', opportunity())
        latest = store.anchor_digest
    before = read_all(tmp_path)
    with pytest.raises(StoreError, match='stale'):
        reopen(tmp_path, genesis)
    assert read_all(tmp_path) == before
    (tmp_path/'journal/events.jsonl').write_bytes(b'')
    with pytest.raises(StoreError, match='suffix loss'):
        reopen(tmp_path, latest)
    assert (tmp_path/'journal/events.jsonl').read_bytes() == b''


def test_joint_rollback_detectable_only_if_independent_checkpoint_survives(tmp_path):
    with create(tmp_path) as store:
        old_anchor, old = store.anchor_digest, read_all(tmp_path)
        store.append('op', 'OPPORTUNITY', opportunity())
        latest = store.anchor_digest
    for name, raw in old.items():
        (tmp_path/name).write_bytes(raw)
    with pytest.raises(StoreError, match='stale'):
        reopen(tmp_path, latest)
    # Scope boundary demonstrated: rolling back the caller's checkpoint too is indistinguishable.
    with reopen(tmp_path, old_anchor) as store:
        assert store.inspect().opportunity_count == 0


@pytest.mark.parametrize('part', ['journal/events.jsonl', 'custody/anchors.jsonl'])
def test_partial_frame_never_truncated_or_repaired(tmp_path, part):
    with create(tmp_path) as store:
        store.append('op', 'OPPORTUNITY', opportunity())
        anchor = store.anchor_digest
    p = tmp_path/part
    p.write_bytes(p.read_bytes()[:-2])
    before = read_all(tmp_path)
    with pytest.raises(StoreError, match='partial frame'):
        reopen(tmp_path, anchor)
    assert before == read_all(tmp_path)


def test_valid_state_machine_fork_does_not_match_anchor(tmp_path):
    with create(tmp_path) as store:
        store.append('op', 'OPPORTUNITY', opportunity())
        anchor = store.anchor_digest
    other = opportunity()
    other['protocol_ref'] = digest('different-valid-protocol')
    fork = extend_ledger((), ledger_id=LEDGER, operation_id='op', kind='OPPORTUNITY', payload=other)
    validate_ledger(fork, ledger_id=LEDGER)  # A syntactically valid fork really is valid to Phase-2D.
    (tmp_path/'journal/events.jsonl').write_bytes(canonical_json_bytes(fork[0])+b'\n')
    before = read_all(tmp_path)
    with pytest.raises(StoreError, match='anchor/bytes/history'):
        reopen(tmp_path, anchor)
    assert before == read_all(tmp_path)


@pytest.mark.parametrize('field', ['game_id', 'game_attempt_id'])
def test_store_rejects_other_attempt_before_extending_even_distinct_boundary(tmp_path, field):
    with create(tmp_path) as store:
        op = opportunity(1)
        op[field] += '-another'
        before = read_all(tmp_path)
        with pytest.raises(StoreError, match='mixed execution'):
            store.append('mixed', 'OPPORTUNITY', op)
        assert before == read_all(tmp_path)


def test_same_attempt_multiple_pre_share_one_day_but_not_assignment(tmp_path):
    with create(tmp_path) as store:
        j = Journal(store)
        for i in (0, 1):
            op, a, _ = assign(j, i)
            j.append(f'day-{i}', 'DAY_WITNESS', day(a, op))
        view = store.inspect()
        assert len(view.assignments) == 2
        assert len(view.shared_observed_days) == 1


def test_duplicate_operations_retain_original_record_and_current_high_water(tmp_path):
    with create(tmp_path) as store:
        first = store.append('op', 'OPPORTUNITY', opportunity())
        store.append('op2', 'OPPORTUNITY', opportunity(1))
        before = read_all(tmp_path)
        again = store.append('op', 'OPPORTUNITY', opportunity())
        assert again.record_digest == first.record_digest
        assert again.sequence == 0 and again.anchor_digest == store.anchor_digest
        assert read_all(tmp_path) == before
        changed = opportunity()
        changed['capture_digest'] = digest('changed')
        with pytest.raises(ValueError, match='operation'):
            store.append('op', 'OPPORTUNITY', changed)


@pytest.mark.parametrize('fault', ['data_fsync', 'anchor_before_write', 'anchor_partial', 'ack_after_anchor'])
def test_storage_fault_poison_and_recovery_never_adopts_suffix(tmp_path, monkeypatch, fault):
    store = create(tmp_path)
    old = store.anchor_digest
    original = module._append_synced
    data_fd = store._fds[store.root/'events.jsonl']
    anchor_fd = store._fds[store.anchor_root/'anchors.jsonl']

    def faulty(fd, raw):
        if fd == data_fd and fault == 'data_fsync':
            module._write_all(fd, raw)
            raise OSError('injected data fsync failure')
        if fd == anchor_fd:
            if fault == 'anchor_before_write':
                raise OSError('injected anchor failure')
            if fault == 'anchor_partial':
                module._write_all(fd, raw[:len(raw)//2]); os.fsync(fd)
                raise OSError('injected half anchor')
            if fault == 'ack_after_anchor':
                original(fd, raw)
                raise OSError('injected ACK loss after durable anchor')
        original(fd, raw)

    monkeypatch.setattr(module, '_append_synced', faulty)
    with pytest.raises(OSError):
        store.append('op', 'OPPORTUNITY', opportunity())
    with pytest.raises(StoreError, match='unhealthy'):
        store.append('retry', 'OPPORTUNITY', opportunity())
    store.close()
    before = read_all(tmp_path)
    with pytest.raises(StoreError):
        reopen(tmp_path, old)
    assert before == read_all(tmp_path)
    if fault == 'ack_after_anchor':
        # Explicit test oracle reconciliation of already durable evidence, not action replay.
        with reopen(tmp_path, checkpoint(tmp_path)) as recovered:
            receipt = recovered.append('op', 'OPPORTUNITY', opportunity())
            assert not receipt.execution_permission
            assert recovered.inspect().opportunity_count == 1


def _child_hold(path, conn, anchor):
    try:
        store = create(Path(path)) if anchor is None else reopen(Path(path), anchor)
        store.append('owned-op', 'OPPORTUNITY', opportunity())
        conn.send(('owned', store.anchor_digest))
        conn.recv()  # Parent ALWAYS terminates/joins owned child in finally on test failure.
        store.close()
    except BaseException as error:
        conn.send(('error', type(error).__name__, str(error)))
    finally:
        conn.close()


def _child_crash(path, conn, anchor, stage):
    store = reopen(Path(path), anchor)
    original = module._append_synced
    data_fd = store._fds[store.root/'events.jsonl']
    anchor_fd = store._fds[store.anchor_root/'anchors.jsonl']

    def crash(fd, raw):
        if stage == 'partial_data' and fd == data_fd:
            module._write_all(fd, raw[:len(raw)//2]); os.fsync(fd)
            os._exit(41)
        original(fd, raw)
        if (stage == 'after_data' and fd == data_fd) or (stage == 'after_anchor' and fd == anchor_fd):
            os._exit(42)
    module._append_synced = crash
    if stage == 'before_data':
        os._exit(40)
    receipt = store.append('op', 'OPPORTUNITY', opportunity())
    conn.send(asdict(receipt))
    os._exit(43)


def _finish(process):
    process.join(5)
    if process.is_alive():
        process.terminate(); process.join(3)
    if process.is_alive():
        process.kill(); process.join(3)
    assert not process.is_alive()


def test_real_multiprocess_lock_contention_then_abrupt_owner_exit_and_restart(tmp_path):
    ctx = mp.get_context('spawn')
    parent, child = ctx.Pipe()
    p = ctx.Process(target=_child_hold, args=(str(tmp_path), child, None))
    p.start(); child.close()
    try:
        assert parent.poll(15)
        state = parent.recv()
        assert state[0] == 'owned', state
        before = read_all(tmp_path)
        with pytest.raises(StoreError, match='ownership busy'):
            reopen(tmp_path, state[1])
        assert read_all(tmp_path) == before
        p.kill(); _finish(p)
        with reopen(tmp_path, state[1]) as store:
            assert store.inspect().opportunity_count == 1
    finally:
        if p.is_alive(): p.kill()
        _finish(p); parent.close()


@pytest.mark.parametrize('stage', ['before_data', 'partial_data', 'after_data', 'after_anchor'])
def test_real_process_exit_at_storage_boundaries(tmp_path, stage):
    with create(tmp_path) as store:
        anchor = store.anchor_digest
    ctx = mp.get_context('spawn')
    parent, child = ctx.Pipe()
    p = ctx.Process(target=_child_crash, args=(str(tmp_path), child, anchor, stage))
    p.start(); child.close()
    try:
        _finish(p)
        assert p.exitcode in (40, 41, 42), p.exitcode
        before = read_all(tmp_path)
        if stage == 'before_data':
            with reopen(tmp_path, anchor) as store:
                assert not store.records
        else:
            with pytest.raises(StoreError):
                reopen(tmp_path, anchor)
            if stage == 'after_anchor':
                with reopen(tmp_path, checkpoint(tmp_path)) as store:
                    assert len(store.records) == 1
        assert before == read_all(tmp_path)
        (tmp_path/'process-evidence.json').write_bytes(canonical_json_bytes({
            'stage': stage, 'owned_child_exitcode': p.exitcode,
            'old_anchor': anchor, 'on_disk_test_oracle': checkpoint(tmp_path),
            'automatic_action_replay': False, 'files_unchanged_by_recovery': True}))
    finally:
        if p.is_alive(): p.kill()
        _finish(p); parent.close()


@pytest.mark.parametrize('alias', ['symlink_file', 'hardlink_file', 'symlink_root', 'extra_file', 'bad_mode'])
def test_unsafe_storage_rejected_without_repair(tmp_path, alias):
    with create(tmp_path) as store:
        anchor = store.anchor_digest
    p = tmp_path/'journal/events.jsonl'
    if alias in ('symlink_file', 'hardlink_file'):
        moved = tmp_path/'original'
        p.rename(moved)
        if alias == 'symlink_file': p.symlink_to(moved)
        else: os.link(moved, p)
    elif alias == 'symlink_root':
        (tmp_path/'journal').rename(tmp_path/'original')
        (tmp_path/'journal').symlink_to(tmp_path/'original')
    elif alias == 'extra_file': (tmp_path/'journal/extra').write_bytes(b'x')
    else: p.chmod(0o644)
    before = read_all(tmp_path)
    with pytest.raises((StoreError, OSError)):
        reopen(tmp_path, anchor)
    assert before == read_all(tmp_path)


def test_thread_and_fork_inherited_writer_cannot_append(tmp_path):
    with create(tmp_path) as store:
        errors = []
        t = threading.Thread(target=lambda: _try_wrong_thread(store, errors))
        t.start(); t.join(5)
        assert not t.is_alive() and errors == ['StoreError']
        pid = os.fork()
        if pid == 0:
            try:
                store.append('fork', 'OPPORTUNITY', opportunity())
            except StoreError:
                store.close(); os._exit(0)
            os._exit(1)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        store.append('parent', 'OPPORTUNITY', opportunity())


def _try_wrong_thread(store, errors):
    try:
        store.append('thread', 'OPPORTUNITY', opportunity())
    except BaseException as error:
        errors.append(type(error).__name__)


@pytest.mark.parametrize('applied', [False, True])
def test_external_readback_resolves_unknown_without_executor_or_second_commit(tmp_path, applied):
    with create(tmp_path) as store:
        j, op, a, r = setup_reserved(store)
        world = FakeWorld()
        external = world.commit(op, r['payload']['attempt_token']) if applied else None
        evidence = committed(a, r, external) if applied else unsubmitted(a, r, op)
        class Provider:
            calls = 0
            def readback(self, query):
                self.calls += 1
                assert json.loads(query.opportunity_json) == op
                assert query.attempt_token == r['payload']['attempt_token']
                return evidence
        provider = Provider()
        result = reconcile_readback(store, operation_id='rb', assignment_ref=a, provider=provider)
        assert not result.execution_permission and world.calls == int(applied)
        assert store.inspect().assignments[0].commit_status == evidence['status']
        reconcile_readback(store, operation_id='rb', assignment_ref=a, provider=provider)
        assert provider.calls == 1
        with pytest.raises(ValueError, match='at most one'):
            j.append('new-intent-not-idempotent-receipt', 'COMMIT_WITNESS',
                     {**r['payload'], 'attempt_token': 'different-second-attempt'})
        assert world.calls == int(applied)


@pytest.mark.parametrize('mode', ['unknown', 'exception', 'bad_attempt', 'bad_prefix', 'unstopped', 'wrong_token'])
def test_unresolved_or_bad_readback_never_acts_or_writes_success(tmp_path, mode):
    with create(tmp_path) as store:
        _, op, a, r = setup_reserved(store)
        before = read_all(tmp_path)
        payload = unsubmitted(a, r, op)
        if mode == 'bad_attempt': payload['external']['game_attempt_id'] = 'other'
        if mode == 'bad_prefix': payload['external']['prefix_digest'] = digest('other')
        if mode == 'unstopped': payload['external'].pop('stopped_executor_digest')
        if mode == 'wrong_token': payload['external']['attempt_token'] = 'other'
        class Provider:
            def readback(self, query):
                if mode == 'exception': raise TimeoutError('artificial external read timeout')
                return None if mode == 'unknown' else payload
        if mode == 'unknown':
            assert reconcile_readback(store, operation_id='rb', assignment_ref=a, provider=Provider()) is None
        else:
            with pytest.raises((ValueError, TimeoutError)):
                reconcile_readback(store, operation_id='rb', assignment_ref=a, provider=Provider())
        assert read_all(tmp_path) == before


def test_anchor_cannot_be_shared_with_distinct_journal(tmp_path):
    with create(tmp_path) as store:
        other = tmp_path/'other'; other.mkdir(mode=0o700)
        (other/'writer.lock').write_bytes(b''); (other/'writer.lock').chmod(0o600)
        with pytest.raises(StoreError, match='ownership busy'):
            ActionEvidenceStore.reopen(other, tmp_path/'custody', expected_anchor=store.anchor_digest, **namespace())


def test_swapped_lock_file_poisoned_before_append(tmp_path):
    with create(tmp_path) as store:
        p = tmp_path/'journal/writer.lock'
        p.rename(tmp_path/'old-lock')
        p.write_bytes(b''); p.chmod(0o600)
        before = read_all(tmp_path)
        with pytest.raises(StoreError, match='replaced'):
            store.append('op', 'OPPORTUNITY', opportunity())
        assert before == read_all(tmp_path)


def test_ack_is_returned_only_after_both_data_and_anchor_fsync(tmp_path, monkeypatch):
    with create(tmp_path) as store:
        events = []
        original = module._append_synced
        data = store._fds[store.root/'events.jsonl']
        anchor = store._fds[store.anchor_root/'anchors.jsonl']
        def tracked(fd, raw):
            original(fd, raw)
            events.append(('data' if fd == data else 'anchor', 'fsync_returned'))
        monkeypatch.setattr(module, '_append_synced', tracked)
        receipt = store.append('op', 'OPPORTUNITY', opportunity())
        assert events == [('data', 'fsync_returned'), ('anchor', 'fsync_returned')]
        assert receipt.anchor_digest == store.anchor_digest and not receipt.execution_permission


def test_missing_anchor_stops_without_writing(tmp_path):
    with create(tmp_path) as store:
        store.append('op', 'OPPORTUNITY', opportunity())
        anchor = store.anchor_digest
    (tmp_path/'custody/anchors.jsonl').write_bytes(b'')
    before = read_all(tmp_path)
    with pytest.raises(StoreError, match='missing high-water'):
        reopen(tmp_path, anchor)
    assert read_all(tmp_path) == before


def test_ledger_error_cannot_change_assignment_population_on_disk(tmp_path):
    with create(tmp_path) as store:
        j = Journal(store)
        op, a, _ = assign(j)
        prior = next(r['payload'] for r in store.records if r['record_digest'] == a)
        before = read_all(tmp_path)
        with pytest.raises(ValueError, match='duplicate assignment'):
            store.append('redraw', 'ASSIGNMENT', {**prior, 'assignment_id': 'other'})
        assert read_all(tmp_path) == before


def test_readback_ack_loss_restarts_evidence_only_with_explicit_latest_anchor(tmp_path, monkeypatch):
    store = create(tmp_path)
    j, op, a, r = setup_reserved(store)
    old = store.anchor_digest
    world = FakeWorld()
    result = world.commit(op, r['payload']['attempt_token'])
    evidence = committed(a, r, result, audit_gap=True)
    class Provider:
        calls = 0
        def readback(self, query):
            self.calls += 1
            return evidence
    provider = Provider()
    original = module._append_synced
    anchor_fd = store._fds[store.anchor_root/'anchors.jsonl']
    def ack_loss(fd, raw):
        original(fd, raw)
        if fd == anchor_fd: raise OSError('ACK lost after readback evidence fsync')
    monkeypatch.setattr(module, '_append_synced', ack_loss)
    with pytest.raises(OSError):
        reconcile_readback(store, operation_id='rb', assignment_ref=a, provider=provider)
    store.close()
    with pytest.raises(StoreError, match='stale'):
        reopen(tmp_path, old)
    with reopen(tmp_path, checkpoint(tmp_path)) as recovered:
        reconcile_readback(recovered, operation_id='rb', assignment_ref=a, provider=provider)
        assert world.calls == 1 and provider.calls == 1
        assert recovered.inspect().assignments[0].commit_status == 'COMMITTED'


def test_owner_io_and_readback_do_not_consume_python_or_numpy_rng(tmp_path):
    import random
    import numpy as np
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    with create(tmp_path) as store:
        _, op, a, r = setup_reserved(store)
        class Provider:
            def readback(self, query): return None
        reconcile_readback(store, operation_id='unknown-rb', assignment_ref=a, provider=Provider())
    assert random.getstate() == python_before
    after = np.random.get_state()
    assert numpy_before[0] == after[0] and np.array_equal(numpy_before[1], after[1])
    assert numpy_before[2:] == after[2:]


def test_distinct_attempts_same_game_boundary_have_independent_stores(tmp_path):
    first = tmp_path/'first'; second = tmp_path/'second'
    first.mkdir(); second.mkdir()
    op = opportunity()
    with create(first) as a:
        a.append('op', 'OPPORTUNITY', op)
        other = {**op, 'game_attempt_id': 'separate-explicit-attempt'}
        with ActionEvidenceStore.create(second/'journal', second/'custody',
                **{**namespace(), 'game_attempt_id': other['game_attempt_id']}) as b:
            b.append('op', 'OPPORTUNITY', other)
            assert a.records[0]['payload']['boundary_id'] == b.records[0]['payload']['boundary_id']
            assert a.anchor_digest != b.anchor_digest
            assert a.inspect().opportunity_count == b.inspect().opportunity_count == 1


def test_missing_day_is_not_empty_exile_after_anchored_restart(tmp_path):
    with create(tmp_path) as store:
        journal = Journal(store)
        op, a, _ = assign(journal)
        journal.append('missing-day', 'DAY_WITNESS', day(a, op, missing=True))
        anchor = store.anchor_digest
    with reopen(tmp_path, anchor) as store:
        view = store.inspect().assignments[0]
        assert view.day_status == 'MISSING' and view.observed_no_exile is None
        assert view.commit_status == 'NO_ATTEMPT'


def test_noncanonical_whitespace_frame_rejected_without_repair(tmp_path):
    with create(tmp_path) as store:
        store.append('op', 'OPPORTUNITY', opportunity())
        anchor = store.anchor_digest
    p = tmp_path/'journal/events.jsonl'
    p.write_bytes(b' ' + p.read_bytes())
    before = read_all(tmp_path)
    with pytest.raises(StoreError, match='noncanonical'):
        reopen(tmp_path, anchor)
    assert before == read_all(tmp_path)
