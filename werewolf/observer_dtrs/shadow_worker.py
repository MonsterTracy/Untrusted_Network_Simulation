"""Bounded Shadow admission with a fresh, spawn-isolated, killable child per PRE.

    Only synthetic providers are qualified in D4c-1. No game/service integration.
"""
from dataclasses import dataclass
import math
import multiprocessing as mp
from pathlib import Path
from queue import Queue, Empty
from threading import BoundedSemaphore, Event, Lock, Thread
import time

from werewolf.artifact_io import read_artifact_file, sha256_bytes, verify_artifact
from werewolf.observer_dtrs.online_context import ShadowRequest, digest
from werewolf.observer_dtrs.online_model import FrozenEventEstimator
from werewolf.observer_dtrs.shadow import ShadowEngine, ShadowResult
from werewolf.observer_dtrs.shadow_artifacts import VERSION as JOURNAL_VERSION
from werewolf.observer_dtrs.shadow_receipts import ReceiptLedger


@dataclass(frozen=True)
class WorkerSpec:
    provider: object
    model: FrozenEventEstimator
    hard_timeout_seconds: float = 10.


def _engine(spec, root):
    return ShadowEngine(tom_provider=spec.provider, probability_provider=spec.model,
        expected_deployment_digest=spec.model.manifest_digest, journal_root=root,
        timeout_seconds=spec.hard_timeout_seconds)


def _implementation():
    root = Path(__file__).resolve().parents[2]
    names = ('run_random.py', 'werewolf/canonical_collection/runtime.py',
        *('werewolf/observer_dtrs/'+n+'.py' for n in (
            'online_context', 'online_model', 'shadow', 'shadow_artifacts',
            'shadow_receipts', 'shadow_worker', 'shadow_hook')))
    return {n: sha256_bytes((root/n).read_bytes()) for n in names}


def _child(spec, root, request, token, pipe):
    """No env, agent, observation, recorder, targets or training audit crosses IPC."""
    try:
        result = _engine(spec, root).infer(request)
        pipe.send({'status': 'SUCCEEDED', 'request_digest': request.digest,
                   'token': token, 'result_digest': result.digest})
    except Exception:
        pipe.send({'status': 'FAILED_INFERENCE', 'request_digest': request.digest,
                   'token': token, 'result_digest': None})
    finally:
        pipe.close()


class IsolatedShadowWorker:
    """submit never waits for queue space or inference; audit fsync is synchronous.

    The supervisor is a thread, but arbitrary inference runs only in a fresh child
    process. Capacity includes its active child. All lifecycle waits are explicit,
    outside the game loop. Disk failure disables further admission; no fallback.
    """

    def __init__(self, root, *, spec, max_pending=2):
        if (type(spec) is not WorkerSpec or type(spec.model) is not FrozenEventEstimator
                or spec.model.tom.family != 'controlled_fixture'
                or spec.model.record['scope'] != 'controlled_fixture'
                or type(max_pending) is not int or max_pending < 1
                or type(spec.hard_timeout_seconds) not in (int, float)
                or not math.isfinite(spec.hard_timeout_seconds) or spec.hard_timeout_seconds <= 0):
            raise ValueError('D4c-1 requires a controlled provider/model and bounded positive capacity/deadline')
        root = Path(root)
        if root.is_symlink():
            raise ValueError('worker root cannot be a symlink')
        self.root = root.resolve()
        if any((p/'manifest.json').exists() for p in (self.root, *self.root.parents)):
            raise ValueError('worker root cannot be inside an immutable artifact')
        if self.root.exists() and any(p.name not in {'inference', 'receipts'} or p.is_symlink()
                                      for p in self.root.iterdir()):
            raise ValueError('worker root contains unrelated files')
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.stat().st_mode & 0o077:
            raise ValueError('Shadow root must be owner-private; do not change existing permissions automatically')
        self.spec = spec
        self.engine = _engine(spec, self.root/'inference')  # validates anchors; never predicts
        self._lock = Lock()
        self._stop, self._closing = Event(), Event()
        self._slots = BoundedSemaphore(max_pending)
        self._queue = Queue(maxsize=max_pending)
        self.audit_error = False
        self.unrecorded_captures = 0
        self._closed = False
        self.ledger = ReceiptLedger(self.root/'receipts', {
            'scope': 'controlled_pre_hook_not_formal_deployment',
            'engine_digest': digest(self.engine.identity), 'max_pending': max_pending,
            'controlled_working_tree_sha256': _implementation(),
            'hard_timeout_seconds': spec.hard_timeout_seconds,
            'start_method': 'spawn', 'retry': 'at_most_once_no_automatic_replay'})
        # A previous admission is never replayed, including accepted-but-not-started.
        done = {r['request_digest'] for r in self.ledger.events if r['kind'] == 'outcome'}
        admitted = [r for r in self.ledger.events if r['status'] == 'SUBMITTED']
        self._seen = {r['request_digest'] for r in admitted}
        for r in admitted:
            if r['request_digest'] not in done:
                self.ledger.append(kind='outcome', status='INTERRUPTED_NO_RETRY',
                    request_digest=r['request_digest'], token=r['token'])
        self._thread = Thread(target=self._work, name='observer-shadow-supervisor', daemon=True)
        self._thread.start()

    def capture(self, pre, *, request=None, status=None):
        """Trusted capture sends only a legal request, or an explicit skip/error."""
        with self._lock:
            if self._closed:
                self.unrecorded_captures += 1
                return 'SKIPPED_DISABLED'
            if self.audit_error:
                self.unrecorded_captures += 1
                return 'AUDIT_UNAVAILABLE'
            slot = False
            try:
                if request is not None:
                    if type(request) is not ShadowRequest:
                        raise TypeError('only legal ShadowRequest admission')
                    if self._closing.is_set():
                        status = 'SKIPPED_DISABLED'
                    elif request.digest in self._seen:
                        status = 'DUPLICATE'
                    else:
                        slot = self._slots.acquire(blocking=False)
                        status = 'SUBMITTED' if slot else 'SKIPPED_BACKPRESSURE'
                r = self.ledger.append(kind='capture', status=status, pre=pre,
                    request=request.identity if request is not None else None,
                    request_digest=request.digest if request is not None else None)
                if status == 'SUBMITTED':
                    self._seen.add(request.digest)
                    self._queue.put_nowait((request, r['token']))
                return status
            except Exception:
                if slot:
                    self._slots.release()
                self.unrecorded_captures += 1
                self.audit_error = True
                self._stop.set()
                return 'AUDIT_UNAVAILABLE'

    def _outcome(self, request, token, status, result_digest=None):
        with self._lock:
            if self.audit_error:
                return
            try:
                self.ledger.append(kind='outcome', status=status, request_digest=request.digest,
                                   token=token, result_digest=result_digest)
            except Exception:
                self.audit_error = True
                self._stop.set()

    def _execute(self, request, token):
        ctx = mp.get_context('spawn')
        receive, send = ctx.Pipe(duplex=False)
        process = ctx.Process(target=_child, args=(self.spec, self.root/'inference', request, token, send),
                              name='observer-shadow-inference', daemon=True)
        status, result_hash = 'FAILED_START', None
        started = time.monotonic()
        try:
            process.start()
            send.close()
            while process.is_alive():
                if self._stop.is_set():
                    status = 'FAILED_STOPPED'
                    break
                if time.monotonic()-started >= self.spec.hard_timeout_seconds:
                    status = 'FAILED_TIMEOUT'
                    break
                process.join(timeout=.01)
            else:
                status = 'FAILED_CRASH'
                if time.monotonic()-started >= self.spec.hard_timeout_seconds:
                    status = 'FAILED_TIMEOUT'
                elif process.exitcode == 0 and receive.poll():
                    try:
                        reply = receive.recv()
                    except EOFError:
                        reply = None
                    if (type(reply) is dict and set(reply) == {'status', 'request_digest', 'token', 'result_digest'}
                            and reply['request_digest'] == request.digest and reply['token'] == token
                            and reply['status'] in ('SUCCEEDED', 'FAILED_INFERENCE')):
                        status = reply['status']
                        if status == 'SUCCEEDED':
                            a = verify_artifact(self.root/'inference'/'results'/request.digest,
                                expected_artifact_type='observer_dtrs_shadow_results',
                                expected_schema_version=JOURNAL_VERSION)
                            result = ShadowResult(read_artifact_file(a, 'record.json'))
                            if result.digest != reply['result_digest']:
                                raise ValueError('child result anchor mismatch')
                            result_hash = result.digest
        except Exception:
            status = 'FAILED_START' if process.pid is None else 'FAILED_INFERENCE'
            result_hash = None
        finally:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=.25)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1.)
                if process.is_alive():
                    # Never release capacity/start another child while termination is uncertain.
                    self.audit_error = True
                    self._stop.set()
                    raise RuntimeError('Shadow child could not be killed')
                process.join()
                process.close()
            receive.close()
            send.close()
        self._outcome(request, token, status, result_hash)

    def _work(self):
        while not self._closing.is_set() or not self._queue.empty():
            try:
                request, token = self._queue.get(timeout=.02)
            except Empty:
                continue
            try:
                if self._stop.is_set():
                    self._outcome(request, token, 'FAILED_STOPPED')
                else:
                    self._execute(request, token)
            except Exception:
                self.audit_error = True
                self._stop.set()
            finally:
                self._slots.release()
                self._queue.task_done()

    def inspect(self, requests=None, *, expected_head=None):
        with self._lock:
            summary = self.ledger.summary(expected_head=expected_head)
            if self.ledger.read() != self.ledger.events:
                raise ValueError('receipt history changed during ownership')
            # Owner calls this after draining. A running child may legitimately hold
            # the inference lock; never wait for it in capture/the game thread.
            inventory = self.engine.journal.inventory()
            successes = {r['request_digest']: r['result_digest'] for r in self.ledger.events
                         if r['kind'] == 'outcome' and r['status'] == 'SUCCEEDED'}
            expected = None if requests is None else {r.digest: r for r in requests}
            if expected is not None and (len(expected) != len(requests)
                    or set(expected) != self._seen):
                raise ValueError('expected admitted request population mismatch')
            for key, anchor in successes.items():
                a = verify_artifact(self.root/'inference'/'results'/key,
                                    expected_artifact_type='observer_dtrs_shadow_results',
                                    expected_schema_version=JOURNAL_VERSION)
                result = ShadowResult(read_artifact_file(a, 'record.json'))
                if (result.digest != anchor or result.record['engine_digest'] != digest(self.engine.identity)
                        or a.manifest['identity'] != {'engine_digest': digest(self.engine.identity),
                                                      'request_digest': key}):
                    raise ValueError('sealed worker result anchor mismatch')
                if expected is not None:
                    r = expected[key]
                    self.engine._validate_record(r, self.engine._binding(r), result.record)
            return {**summary, 'inference_inventory': inventory,
                    'semantic_results_verified': expected is not None,
                    'audit_complete': not self.audit_error and self.unrecorded_captures == 0,
                    'unrecorded_captures_this_process': self.unrecorded_captures,
                    'supervisor_alive': self._thread.is_alive()}

    def close(self, *, timeout=30.):
        """Explicit owner-side drain, NEVER called from the game loop/hook."""
        self._closing.set()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            self._stop.set()
            self._thread.join(timeout=2.)
        if self._thread.is_alive():
            raise RuntimeError('Shadow supervisor has not stopped; retain ownership')
        with self._lock:
            self._closed = True
            self.ledger.close()
