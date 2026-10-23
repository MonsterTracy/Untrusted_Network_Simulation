"""Artificial-only process, coverage, admission and durable recovery acceptance."""
from dataclasses import replace
import json
import os
import time

import pytest

from tests.observer_dtrs.test_shadow import deployed, observation, request
from tests.observer_dtrs.shadow_worker_fixtures import ProcessTom
from werewolf.observer_dtrs.online_context import digest
from werewolf.observer_dtrs.shadow_hook import ShadowPREHook
from werewolf.observer_dtrs.shadow_receipts import ReceiptLedger
from werewolf.observer_dtrs.shadow_worker import IsolatedShadowWorker, WorkerSpec


def worker(root, model, behavior='ok', **kwargs):
    timeout = kwargs.pop('timeout', 5.)
    marker = kwargs.pop('marker', None)
    return IsolatedShadowWorker(root, spec=WorkerSpec(ProcessTom(model.tom, behavior, marker), model, timeout), **kwargs)


@pytest.mark.parametrize('phase', ['discussion', 'pk_discussion'])
@pytest.mark.parametrize('dead', [(), ('player2', 'player7')])
def test_child_all_observers_exact_results_and_idempotency(tmp_path, deployed, phase, dead):
    r = request(phase=phase, dead=dead)
    w = worker(tmp_path/'worker', deployed, marker=str(tmp_path/'calls'))
    hook = ShadowPREHook(w, shadow_run_id=r.shadow_run_id, attempt_id=r.attempt_id)
    assert hook.capture(r.prefix, observation(r.prefix), canonical_attempt_id=hook.attempt_id) == 'SUBMITTED'
    assert hook.capture(r.prefix, observation(r.prefix), canonical_attempt_id=hook.attempt_id) == 'DUPLICATE'
    w.close()
    s = w.inspect([r])
    assert s['submitted'] == 1 and s['completed'] == 1 and s['pending'] == 0
    assert s['capture_statuses'] == {'SUBMITTED': 1, 'DUPLICATE': 1}
    assert s['outcome_statuses'] == {'SUCCEEDED': 1} and s['semantic_results_verified']
    pids = (tmp_path/'calls').read_text().splitlines()
    assert len(pids) == 1 and int(pids[0]) != os.getpid()
    results = w.engine.inspect([r])
    assert results['verified_results'] == 1
    reopened = worker(tmp_path/'worker', deployed, marker=str(tmp_path/'calls'))
    assert reopened.capture({}, request=r) == 'DUPLICATE'
    reopened.close()
    assert (tmp_path/'calls').read_text().splitlines() == pids
    assert reopened.inspect([r])['completed'] == 1


@pytest.mark.parametrize('behavior,status', [('hang', 'FAILED_TIMEOUT'), ('late', 'FAILED_TIMEOUT'),
                                             ('crash', 'FAILED_CRASH'), ('error', 'FAILED_INFERENCE')])
def test_hard_deadline_crash_late_results_and_next_request(tmp_path, deployed, behavior, status):
    # Deadline includes child spawn; marker proves the artificial provider started.
    root = tmp_path/'worker';marker = tmp_path/'calls'
    w = worker(root, deployed, behavior, timeout=1.5, marker=str(marker), max_pending=1)
    r = request()
    assert w.capture({}, request=r) == 'SUBMITTED'
    start = time.monotonic()
    w.close()
    assert time.monotonic()-start < 4.
    s = w.inspect([r])
    assert s['outcome_statuses'] == {status: 1}
    assert marker.exists() and len(marker.read_text().splitlines()) == 1
    pid = int(marker.read_text().strip())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert s['inference_inventory']['results'] == 0
    # Same contract/provider identity, next PRE can succeed after worker replacement.
    good = worker(root, deployed, timeout=1.5, marker=str(marker), max_pending=1)
    r2 = request(attempt='attempt-2')
    assert good.capture({}, request=r) == 'DUPLICATE'
    assert good.capture({}, request=r2) == 'SUBMITTED'
    good.close()
    out = good.inspect([r, r2])
    assert out['outcome_statuses'] == {status: 1, 'SUCCEEDED': 1}


def test_backpressure_never_waits_for_inference_and_all_captures_counted(tmp_path, deployed):
    w = worker(tmp_path/'worker', deployed, 'hang', timeout=1.5, max_pending=1)
    r, r2 = request(), request(attempt='attempt-2')
    assert w.capture({}, request=r) == 'SUBMITTED'
    start = time.monotonic()
    assert w.capture({}, request=r2) == 'SKIPPED_BACKPRESSURE'
    # Disk is local in this test; this is a controlled observation, not a hard I/O SLA.
    assert time.monotonic()-start < 1.
    hook = ShadowPREHook(w, shadow_run_id='s', attempt_id='a')
    o = observation(r.prefix);o['identity'] = 'Villager'
    assert hook.capture(r.prefix, o, canonical_attempt_id=hook.attempt_id) == 'SKIPPED_NONWOLF'
    o['identity'] = 'Werewolf';o['observer_id'] = 3
    assert hook.capture(r.prefix, o, canonical_attempt_id=hook.attempt_id) == 'CAPTURE_INVALID'
    hook.capture_error()
    w.close()
    s = w.inspect([r])
    assert s['capture_attempts'] == 5 and s['eligible_captures'] == 2
    assert s['unique_eligible_PREs'] == 2 and s['submitted'] == s['completed'] == 1
    assert s['capture_statuses'] == {'SUBMITTED': 1, 'SKIPPED_BACKPRESSURE': 1,
        'SKIPPED_NONWOLF': 1, 'CAPTURE_INVALID': 1, 'CAPTURE_ERROR': 1}


def test_durable_unstarted_admission_is_not_replayed(tmp_path, deployed):
    root = tmp_path/'worker';w = worker(root, deployed);w.close()
    r = request()
    # Artificial interruption between fsynced admission and in-memory enqueue.
    ledger = ReceiptLedger(root/'receipts', w.ledger.contract)
    ledger.append(kind='capture', status='SUBMITTED', pre={}, request=r.identity, request_digest=r.digest)
    ledger.close()
    recovered = worker(root, deployed, marker=str(tmp_path/'calls'))
    assert recovered.capture({}, request=r) == 'DUPLICATE'
    recovered.close()
    s = recovered.inspect([r])
    assert s['pending'] == 0 and s['outcome_statuses'] == {'INTERRUPTED_NO_RETRY': 1}
    assert not (tmp_path/'calls').exists()


def test_corruption_disables_capture_never_repairs_or_reclassifies(tmp_path, deployed):
    root = tmp_path/'worker';w = worker(root, deployed)
    hook = ShadowPREHook(w, shadow_run_id='s', attempt_id='a')
    r = request();o = observation(r.prefix);o['identity'] = 'Villager'
    assert hook.capture(r.prefix, o, canonical_attempt_id=hook.attempt_id) == 'SKIPPED_NONWOLF'
    p = root/'receipts/events/00000000/record.json'
    p.write_bytes(b'broken receipt')
    assert hook.capture(r.prefix, observation(r.prefix), canonical_attempt_id=hook.attempt_id) == 'AUDIT_UNAVAILABLE'
    assert w.audit_error
    w.close()
    assert p.read_bytes() == b'broken receipt'
    with pytest.raises(ValueError):
        w.inspect()
    with pytest.raises(ValueError):
        worker(root, deployed)


def test_result_corruption_and_external_receipt_anchor_detected(tmp_path, deployed):
    r = request();w = worker(tmp_path/'worker', deployed)
    w.capture({}, request=r);w.close()
    s = w.inspect([r])
    with pytest.raises(ValueError, match='head'):
        w.inspect([r], expected_head='0'*64)
    assert w.inspect([r], expected_head=s['head_digest'])['audit_complete']
    (w.root/'inference/results'/r.digest/'record.json').write_bytes(b'broken result')
    with pytest.raises(ValueError):
        w.inspect([r])


def test_single_writer_and_no_real_or_incompatible_provider(tmp_path, deployed):
    root = tmp_path/'worker';w = worker(root, deployed)
    try:
        with pytest.raises((ValueError, BlockingIOError)):
            worker(root, deployed)
    finally:
        w.close()
    with pytest.raises(ValueError, match='anchors'):
        IsolatedShadowWorker(tmp_path/'wrong', spec=WorkerSpec(ProcessTom(replace(deployed.tom,
            weights_digest='a'*64)), deployed))


def test_stale_cross_request_ipc_reply_cannot_be_accepted(tmp_path, deployed, monkeypatch):
    from tests.observer_dtrs.shadow_worker_fixtures import wrong_response
    from werewolf.observer_dtrs import shadow_worker
    monkeypatch.setattr(shadow_worker, '_child', wrong_response)
    w = worker(tmp_path/'worker', deployed);r = request()
    w.capture({}, request=r);w.close()
    s = w.inspect([r])
    assert s['outcome_statuses'] == {'FAILED_CRASH': 1}
    assert s['inference_inventory']['results'] == 0


def test_child_start_failure_is_durable(tmp_path, deployed):
    provider = ProcessTom(deployed.tom)
    provider.unpicklable = lambda: None
    w = IsolatedShadowWorker(tmp_path/'worker', spec=WorkerSpec(provider, deployed))
    r = request();w.capture({}, request=r);w.close()
    assert w.inspect([r])['outcome_statuses'] == {'FAILED_START': 1}


def test_owner_abort_never_waits_for_hanging_inference(tmp_path, deployed):
    w = worker(tmp_path/'worker', deployed, 'hang', timeout=20., max_pending=2)
    r, r2 = request(), request(attempt='attempt-2')
    w.capture({}, request=r);w.capture({}, request=r2)
    started = time.monotonic();w.close(timeout=0.)
    assert time.monotonic()-started < 2.
    s = w.inspect([r, r2])
    assert s['completed'] == 2 and s['pending'] == 0
    assert s['outcome_statuses'] == {'FAILED_STOPPED': 2}


def test_uncertain_outcome_write_never_reports_success(tmp_path, deployed, monkeypatch):
    w = worker(tmp_path/'worker', deployed)
    original = w.ledger.append
    def fail_outcome(**kw):
        if kw['kind'] == 'outcome':
            raise OSError('artificial fsync failure')
        return original(**kw)
    monkeypatch.setattr(w.ledger, 'append', fail_outcome)
    r = request();w.capture({}, request=r);w.close()
    s = w.inspect([r])
    assert w.audit_error and not s['audit_complete']
    assert s['completed'] == 0 and s['pending'] == 1
    assert s['inference_inventory']['results'] == 1  # unacknowledged candidate, NOT accepted success


def test_private_output_root_and_no_automatic_permission_changes(tmp_path, deployed):
    root = tmp_path/'shared';root.mkdir(mode=0o755)
    with pytest.raises(ValueError, match='private'):
        worker(root, deployed)
    assert root.stat().st_mode & 0o777 == 0o755
    w = worker(tmp_path/'private', deployed);w.close()
    assert w.root.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize('field', ['observer', 'phase', 'alive', 'identity'])
def test_invalid_nonwolf_view_is_not_silently_declared_ineligible(tmp_path, deployed, field):
    w = worker(tmp_path/'worker', deployed)
    r = request();o = observation(r.prefix);o['identity'] = 'Villager'
    if field == 'observer': o['observer_id'] = 3
    elif field == 'phase': o['authoritative_public_state']['phase'] = 'speech_pk'
    elif field == 'alive': o['authoritative_public_state']['alive_players'] = [1, 3, 4]
    else: o['identity'] = 'unknown'
    hook = ShadowPREHook(w, shadow_run_id='s', attempt_id='a')
    assert hook.capture(r.prefix, o, canonical_attempt_id=hook.attempt_id) == 'CAPTURE_INVALID'
    w.close()
    s = w.inspect()
    assert s['capture_statuses'] == {'CAPTURE_INVALID': 1} and s['eligibility_unknown'] == 1
    assert s['submitted'] == 0


def test_worker_cannot_create_directories_inside_existing_artifact(tmp_path, deployed):
    child = tmp_path/'model'/'forbidden-shadow'
    before = sorted(str(p.relative_to(tmp_path/'model')) for p in (tmp_path/'model').rglob('*'))
    with pytest.raises(ValueError, match='immutable artifact'):
        worker(child, deployed)
    assert not child.exists()
    assert sorted(str(p.relative_to(tmp_path/'model')) for p in (tmp_path/'model').rglob('*')) == before


def test_canonical_attempt_mismatch_is_capture_failure_without_inference(tmp_path, deployed):
    w = worker(tmp_path/'worker', deployed);r = request()
    hook = ShadowPREHook(w, shadow_run_id='s', attempt_id='configured')
    assert hook.capture(r.prefix, observation(r.prefix), canonical_attempt_id='actual') == 'CAPTURE_INVALID'
    w.close()
    s = w.inspect()
    assert s['capture_statuses'] == {'CAPTURE_INVALID': 1} and s['submitted'] == 0
    assert w.ledger.events[0]['pre']['attempt_id'] == 'actual'
    assert w.ledger.events[0]['pre']['configured_attempt_id'] == 'configured'
