"""Exclusive local resident owner and bounded nonblocking admission.

No torch/loader/Agent imports in the owner. IPC uses no pickle, network listener,
shared service PID or automatic inference retry. Fsync is synchronous local I/O,
not a hard real-time promise; audit failure disables admission.
"""
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
from queue import Queue, Empty
import secrets
import subprocess
import sys
import socket
from threading import BoundedSemaphore, Event, Lock, RLock, Thread
import time

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.observer_dtrs.final_tom_service_audit import ServiceJournal, inspect_journal
from werewolf.observer_dtrs.final_tom_wire import (
    Channel, Limits, ROOT, digest, envelope, frame, hex_id, require, service_source,
    ProtocolError, PeerClosed, WireTimeout,
)
from werewolf.observer_dtrs.probabilities import ProbabilityMatrix
from werewolf.tom.final_capacity import derived_capacity, validate_final_pre


@dataclass(frozen=True)
class ServiceTrust:
    capsule_digest: str
    capsule_scope: str
    loader_digest: str
    runtime_digest: str

    def __post_init__(self):
        for v in (self.capsule_digest, self.loader_digest, self.runtime_digest): hex_id(v)
        require(self.capsule_scope in ('CONTROLLED_FIXTURE', 'PINNED_FINAL_INFERENCE_UNQUALIFIED'))


@dataclass(frozen=True)
class ResidentResult:
    payload: bytes

    @property
    def record(self):
        return json.loads(self.payload)

    @property
    def status(self):
        return self.record['status']

    @property
    def matrix(self):
        r = self.record
        require(r['status'] == 'SUCCEEDED', 'technical failure has no Q')
        p = r['prediction']
        return ProbabilityMatrix(p['game_id'], p['boundary_id'], p['prefix_digest'], p['probabilities'])


@dataclass
class Ticket:
    request_id: str
    epoch: str
    token: str
    pre: dict
    prefix_bytes: bytes
    deadline: float
    owner: object
    done: Event = field(default_factory=Event)
    cancel_requested: Event = field(default_factory=Event)
    reason: str = 'CLIENT_CANCELLED'
    value: ResidentResult | None = None

    def result(self, timeout=None):
        if timeout is None:
            timeout = max(0., self.deadline-time.monotonic())
        require(type(timeout) in (int, float) and math.isfinite(timeout) and timeout >= 0, 'bounded wait required')
        if not self.done.wait(timeout):
            # A short consumer deadline is a cancellation, never a detached live query.
            self.owner.cancel(self, reason='CLIENT_TIMEOUT')
        if self.value is None:
            raise TimeoutError('CANCEL_UNCONFIRMED; instance disabled, no Q')
        return self.value


class ResidentOwner:
    def __init__(self, capsule_path, *, trust, journal_root, limits=Limits(), cpu_threads=1, interop_threads=1,
                 _fixture_script=None, _fixture_args=()):
        require(type(trust) is ServiceTrust and type(limits) is Limits)
        require(all(type(v) is int and 1 <= v <= 64 for v in (cpu_threads, interop_threads)))
        require(_fixture_script is None or trust.capsule_scope == 'CONTROLLED_FIXTURE', 'fault fixture is not a formal entry')
        self.capsule_path = str(Path(capsule_path).absolute())
        self.trust, self.limits = trust, limits
        self._entry = Path(_fixture_script or ROOT/'scripts/serve_final_tom_capsule.py').resolve(strict=True)
        self._entry_args = tuple(_fixture_args)
        self._transport_scope = 'CONTROLLED_FAULT_FIXTURE' if _fixture_script else 'CAPSULE_CPU_QUALIFICATION_ONLY'
        self.cpu_threads, self.interop_threads = cpu_threads, interop_threads
        self.state, self.epoch, self._session_bytes = 'NEW', secrets.token_hex(32), None
        self._tainted = False
        self._boot = secrets.token_hex(32)
        self._process = self._channel = None
        self._lock, self._process_lock = RLock(), Lock()
        self._closing = Event()
        self._queue = Queue(maxsize=limits.max_pending)
        self._slots = BoundedSemaphore(limits.max_pending)
        self._tickets, self._pre_keys = {}, {}
        self._pending, self._current = 0, None
        self.audit_unavailable, self.unrecorded_events = False, 0
        self._thread = None
        self.cleanup = {'pid_exit_confirmed': True, 'terminate_sent': False, 'kill_sent': False}
        self.close_status = None
        self.journal = ServiceJournal(journal_root, {
            'version': 'observer_dtrs_capsule_service_owner_v1', 'trust': asdict(trust), 'limits': asdict(limits),
            'concurrency': 1, 'cpu_threads': cpu_threads, 'interop_threads': interop_threads,
            'service_source': service_source(), 'owner_sha256': sha256_bytes(Path(__file__).read_bytes()),
            'audit_source_sha256': sha256_bytes((ROOT/'werewolf/observer_dtrs/final_tom_service_audit.py').read_bytes()),
            'entry_sha256': sha256_bytes(self._entry.read_bytes()), 'transport_scope': self._transport_scope,
            'retry': 'at_most_once_no_automatic_retry', 'formal_qualification': 'NOT_ESTABLISHED',
        }, max_events=limits.max_requests*8+64)

    @property
    def pid(self):
        return self._process.pid if self._process is not None else None

    @property
    def session(self):
        return json.loads(self._session_bytes) if self._session_bytes is not None else None

    def _log(self, kind, ticket=None, **payload):
        try:
            return self.journal.append(kind, epoch=ticket.epoch if ticket else self.epoch,
                request_id=ticket.request_id if ticket else None, token=ticket.token if ticket else None, payload=payload)
        except Exception:
            self.audit_unavailable = True;self.unrecorded_events += 1;self.state = 'UNHEALTHY'
            raise

    def _stop_owned(self):
        """Only the Popen object created by this owner; no external PID parameter."""
        with self._process_lock:
            p = self._process
            info = {'pid_exit_confirmed': p is None or p.poll() is not None,
                    'terminate_sent': False, 'kill_sent': False}
            if p is not None and p.poll() is None:
                try:
                    p.terminate();info['terminate_sent'] = True
                    p.wait(timeout=self.limits.terminate_s)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        p.kill();info['kill_sent'] = True
                        p.wait(timeout=self.limits.kill_s)
                    except (OSError, subprocess.TimeoutExpired):
                        pass
                info['pid_exit_confirmed'] = p.poll() is not None
            self.cleanup = info
            return info['pid_exit_confirmed']

    def _fatal(self, status):
        self.state = 'UNHEALTHY';self._tainted = True
        self._stop_owned()
        return {'status': status, 'prediction': None, 'execution_stopped': self.cleanup['pid_exit_confirmed'],
                'state': 'UNHEALTHY', 'cleanup': dict(self.cleanup)}

    def start(self):
        with self._lock:
            require(self.state == 'NEW' and not self._closing.is_set(), 'explicit start/recover only')
            self.state = 'STARTING'
        try:
            contract = self.journal.contract
            require(asdict(self.trust) == contract['trust'] and asdict(self.limits) == contract['limits']
                    and self.cpu_threads == contract['cpu_threads'] and self.interop_threads == contract['interop_threads']
                    and body_source(self._entry) == contract['entry_sha256']
                    and self._transport_scope == contract['transport_scope']
                    and service_source() == contract['service_source'], 'owner contract changed')
            self._log('EPOCH_START', status='STARTING')
            parent, child = socket.socketpair()
            env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME') and not k.startswith('GIT_')}
            env.update(PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
                       OMP_NUM_THREADS=str(self.cpu_threads), MKL_NUM_THREADS=str(self.cpu_threads))
            cold_deadline = time.monotonic()+self.limits.cold_start_s
            try:
                self._process = subprocess.Popen([sys.executable, '-B', str(self._entry), *self._entry_args,
                    '--fd', str(child.fileno())], cwd=ROOT, env=env, pass_fds=(child.fileno(),),
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True)
            except Exception:
                parent.close()
                raise
            finally:
                child.close()
            self._channel = Channel(parent, self.limits)
            self._channel.write(frame('START', self.epoch, None, self._boot, None, {
                'capsule_path': self.capsule_path, 'trust': asdict(self.trust), 'limits': asdict(self.limits),
                'cpu_threads': self.cpu_threads, 'interop_threads': self.interop_threads,
                'entry_sha256': contract['entry_sha256'], 'transport_scope': self._transport_scope}),
                min(cold_deadline, time.monotonic()+self.limits.write_s))
            message = self._channel.read(cold_deadline)
            instance = message.get('instance');hex_id(instance)
            kind, body = envelope(message, epoch=self.epoch, instance=instance, token=self._boot, request_id=None)
            require(kind == 'BOOTED' and set(body) == {'pid', 'source', 'entry_sha256', 'transport_scope'}
                    and body['pid'] == self.pid and body['source'] == contract['service_source']
                    and body['entry_sha256'] == contract['entry_sha256']
                    and body['transport_scope'] == self._transport_scope, 'bootstrap source mismatch')
            if time.monotonic() > cold_deadline: raise WireTimeout()
            self._startup_phase = 'ready'
            ready_deadline = time.monotonic()+self.limits.ready_s
            message = self._channel.read(ready_deadline)
            kind, body = envelope(message, epoch=self.epoch, instance=instance, token=self._boot, request_id=None)
            require(kind == 'READY' and set(body) == {'session', 'state', 'pid', 'model_load_count'}
                    and body['state'] == 'IDLE' and body['pid'] == self.pid and body['model_load_count'] == 1)
            session = body['session']
            require(set(session) == {'schema_version', 'epoch', 'instance', 'capsule', 'service_source',
                                    'entry_sha256', 'transport_scope', 'qualification'})
            identity = session['capsule']
            require(session['schema_version'] == 'observer_dtrs_capsule_service_source_v1'
                    and session['epoch'] == self.epoch and session['instance'] == instance
                    and session['service_source'] == contract['service_source']
                    and session['entry_sha256'] == contract['entry_sha256']
                    and session['transport_scope'] == self._transport_scope
                    and session['qualification'] == 'NOT_ESTABLISHED'
                    and identity['family'] == 'final_tom_capsule'
                    and identity['capsule_digest'] == self.trust.capsule_digest
                    and identity['scope'] == self.trust.capsule_scope
                    and identity['loader_source']['loader']['implementation_digest'] == self.trust.loader_digest
                    and digest(identity['runtime']) == self.trust.runtime_digest, 'ready identity mismatch')
            if time.monotonic() > ready_deadline: raise WireTimeout()
            require(not self._closing.is_set(), 'owner closed during startup')
            self._session_bytes = canonical_json_bytes(session)
            self._log('READY', status='IDLE', session=self.session, pid=self.pid, model_load_count=1)
            self.state = 'IDLE'
            if self._thread is None:
                self._thread = Thread(target=self._work, daemon=True, name='capsule-owner-supervisor')
                self._thread.start()
            return True
        except WireTimeout:
            status = 'READY_TIMEOUT' if getattr(self, '_startup_phase', None) == 'ready' else 'COLD_START_TIMEOUT'
        except (PeerClosed, BrokenPipeError, ConnectionResetError):
            status = 'SERVICE_CRASH'
        except Exception:
            status = 'PROTOCOL_ERROR'
        result = self._fatal(status)
        try: self._log('START_FAILED', **result)
        except Exception: pass
        return False

    def recover(self):
        with self._lock:
            require(self.state == 'UNHEALTHY' and not self.audit_unavailable and self._pending == 0
                    and self._current is None and not self._closing.is_set(), 'recovery not safe')
            require(self._process is None or self._process.poll() is not None, 'old execution not confirmed stopped')
            self.inspect(expected_head=self.journal.head)  # full chain before opening a new epoch; never repair
            if self._channel is not None: self._channel.close()
            self.epoch, self._boot, self._session_bytes = secrets.token_hex(32), secrets.token_hex(32), None
            self.state = 'NEW';self._startup_phase = None;self._tainted = False
        return self.start()

    def submit(self, prefix, *, request_id, timeout_s=None):
        # Captures immutable public input; never waits for model or queue space.
        plan = validate_final_pre(prefix, derived_capacity(1024))
        raw = canonical_json_bytes(prefix.to_record())
        require(len(raw) <= self.limits.max_request_bytes, 'PRE request byte limit; no truncation')
        require(type(request_id) is str and 0 < len(request_id) <= 128)
        duration = self.limits.query_s if timeout_s is None else timeout_s
        require(type(duration) in (int, float) and math.isfinite(duration) and 0 < duration <= self.limits.query_s)
        pre = {'game_id': prefix.game_id, 'boundary_id': prefix.boundary_id, 'prefix_digest': prefix.prefix_digest,
               'plan_digest': plan.plan_digest, 'token_count': plan.token_count}
        key = (prefix.game_id, prefix.boundary_id, prefix.prefix_digest)
        with self._lock:
            if request_id in self._tickets:
                old = self._tickets[request_id]
                require(old.pre == pre, 'request ID reused for another PRE')
                self._log('ADMISSION_REJECTED', status='DUPLICATE', request_id=request_id)
                return 'DUPLICATE', old
            if key in self._pre_keys:
                self._log('ADMISSION_REJECTED', status='DUPLICATE_PRE', request_id=request_id)
                return 'DUPLICATE_PRE', self._tickets[self._pre_keys[key]]
            status = None
            if self.audit_unavailable: return 'AUDIT_UNAVAILABLE', None
            if self._tainted or self.state not in ('IDLE', 'RUNNING') or self._closing.is_set(): status = 'SKIPPED_UNHEALTHY'
            elif len(self._tickets) >= self.limits.max_requests: status = 'SKIPPED_QUOTA'
            elif not self._slots.acquire(blocking=False): status = 'SKIPPED_BACKPRESSURE'
            if status:
                self._log('ADMISSION_REJECTED', status=status, request_id=request_id, pre=pre)
                return status, None
            ticket = Ticket(request_id, self.epoch, secrets.token_hex(32), pre, raw,
                            time.monotonic()+duration, self)
            try:
                self._log('ADMISSION', ticket, status='SUBMITTED', pre=pre, request_bytes=len(raw))
            except Exception:
                self._slots.release();return 'AUDIT_UNAVAILABLE', None
            self._tickets[request_id] = ticket;self._pre_keys[key] = request_id;self._pending += 1
            self._queue.put_nowait(ticket)
            return 'SUBMITTED', ticket

    def _message(self, kind, ticket, payload):
        return frame(kind, ticket.epoch, self.session['instance'], ticket.token, ticket.request_id, payload)

    def _response(self, ticket, message):
        kind, body = envelope(message, epoch=ticket.epoch, instance=self.session['instance'],
                              token=ticket.token, request_id=ticket.request_id)
        require(type(body) is dict and body.get('session') == self.session, 'response source mismatch')
        return kind, body

    def _cancel_active(self, ticket, trigger):
        self.state = 'CANCELING'
        deadline = time.monotonic()+self.limits.cancel_s
        try:
            self._channel.write(self._message('CANCEL', ticket, {}), min(deadline, time.monotonic()+self.limits.write_s))
            while True:
                kind, body = self._response(ticket, self._channel.read(deadline))
                if kind == 'STARTED': continue
                require(kind == 'RESULT' and set(body) == {'status', 'prediction', 'session', 'execution_stopped', 'state'})
                if body['status'] == 'SUCCEEDED':
                    self._log('LATE_SUCCESS_REJECTED', ticket, status='NOT_ACCEPTED')
                    continue
                require(body['status'] in ('CANCELLED', 'SERVICE_TIMEOUT', 'INFERENCE_ERROR')
                        and body['prediction'] is None and body['execution_stopped'] is True and body['state'] == 'IDLE')
                require(not self._tainted, 'instance already disabled')
                if time.monotonic() > deadline: raise WireTimeout()
                self.state = 'IDLE'
                return {'status': trigger, 'prediction': None, 'execution_stopped': True, 'state': 'IDLE',
                        'cancellation': 'CONFIRMED_IDLE', 'service_status': body['status']}
        except (WireTimeout, PeerClosed, OSError):
            return {**self._fatal('CANCEL_UNCONFIRMED'), 'trigger_status': trigger}
        except Exception:
            return {**self._fatal('PROTOCOL_ERROR'), 'trigger_status': trigger}

    def _execute(self, ticket):
        if ticket.cancel_requested.is_set() or self._closing.is_set():
            return {'status': ticket.reason, 'prediction': None, 'execution_stopped': True, 'state': self.state,
                    'cancellation': 'NOT_DISPATCHED'}
        if time.monotonic() >= ticket.deadline:
            return {'status': 'CLIENT_TIMEOUT', 'prediction': None, 'execution_stopped': True,
                    'state': self.state, 'cancellation': 'NOT_DISPATCHED'}
        if self._tainted or self.state != 'IDLE':
            return {'status': 'SKIPPED_UNHEALTHY', 'prediction': None, 'execution_stopped': True, 'state': self.state}
        try:
            self.state = 'RUNNING'
            self._log('DISPATCHED', ticket, status='WRITE_STARTED', pre=ticket.pre)
            self._channel.write(self._message('QUERY', ticket, {'pre': json.loads(ticket.prefix_bytes),
                'pre_identity': ticket.pre, 'deadline_monotonic': ticket.deadline,
                'session_digest': digest(self.session)}), min(ticket.deadline, time.monotonic()+self.limits.write_s))
            started = False
            while True:
                if ticket.cancel_requested.is_set() or self._closing.is_set():
                    return self._cancel_active(ticket, ticket.reason)
                if time.monotonic() >= ticket.deadline:
                    return self._cancel_active(ticket, 'CLIENT_TIMEOUT')
                try:
                    message = self._channel.read(min(ticket.deadline, time.monotonic()+.01))
                except WireTimeout:
                    continue
                kind, body = self._response(ticket, message)
                if kind == 'STARTED':
                    require(not started and set(body) == {'state', 'session'} and body['state'] == 'RUNNING')
                    started = True;self._log('STARTED', ticket, status='RUNNING')
                    continue
                require(kind == 'RESULT' and started and set(body) == {
                    'status', 'prediction', 'session', 'execution_stopped', 'state'}
                    and body['execution_stopped'] is True and body['state'] == 'IDLE')
                if time.monotonic() >= ticket.deadline:
                    return self._cancel_active(ticket, 'CLIENT_TIMEOUT')
                if body['status'] == 'SUCCEEDED':
                    p = body['prediction']
                    require(set(p) == set(ticket.pre) | {'probabilities', 'capsule_identity'}
                            and {n: p[n] for n in ticket.pre} == ticket.pre
                            and p['capsule_identity'] == self.session['capsule'], 'prediction PRE/source mismatch')
                    ProbabilityMatrix(p['game_id'], p['boundary_id'], p['prefix_digest'], p['probabilities'])
                else:
                    require(body['status'] in ('SERVICE_TIMEOUT', 'INFERENCE_ERROR') and body['prediction'] is None)
                require(not self._tainted, 'instance already disabled')
                if time.monotonic() >= ticket.deadline:
                    return self._cancel_active(ticket, 'CLIENT_TIMEOUT')
                self.state = 'IDLE'
                return body
        except WireTimeout:
            return self._fatal('WRITE_TIMEOUT')  # a partial write is uncertain; no retry
        except (PeerClosed, BrokenPipeError, ConnectionResetError):
            return self._fatal('SERVICE_CRASH')
        except Exception:
            return self._fatal('AUDIT_UNAVAILABLE' if self.audit_unavailable else 'PROTOCOL_ERROR')

    def _work(self):
        while not self._closing.is_set() or not self._queue.empty():
            try: ticket = self._queue.get(timeout=.02)
            except Empty: continue
            with self._lock:
                if ticket.done.is_set():
                    self._pending -= 1;self._slots.release();self._queue.task_done()
                    continue
                self._current = ticket
            result = self._execute(ticket)
            if ticket.cancel_requested.is_set() and result['status'] == 'SUCCEEDED':
                result = {'status': ticket.reason, 'prediction': None, 'execution_stopped': True,
                          'state': self.state, 'cancellation': 'ALREADY_STOPPED_RESPONSE_REJECTED'}
            try:
                self._log('OUTCOME', ticket, **result, pre=ticket.pre)
            except Exception:
                result = self._fatal('AUDIT_UNAVAILABLE')
            ticket.value = ResidentResult(canonical_json_bytes(result))
            ticket.prefix_bytes = b''
            with self._lock:
                self._pending -= 1;self._current = None;self._slots.release()
            ticket.done.set();self._queue.task_done()

    def cancel(self, ticket, *, reason='CLIENT_CANCELLED'):
        require(self._tickets.get(ticket.request_id) is ticket and reason in ('CLIENT_CANCELLED', 'CLIENT_TIMEOUT', 'OWNER_CLOSED'))
        if ticket.done.is_set(): return ticket.value
        with self._lock:
            ticket.reason = reason;ticket.cancel_requested.set()
            if self._current is not ticket:
                result = {'status': reason, 'prediction': None, 'execution_stopped': True,
                          'state': self.state, 'cancellation': 'NOT_DISPATCHED'}
                try:
                    self._log('OUTCOME', ticket, **result, pre=ticket.pre)
                except Exception:
                    result = self._fatal('AUDIT_UNAVAILABLE')
                ticket.value = ResidentResult(canonical_json_bytes(result));ticket.done.set()
                ticket.prefix_bytes = b''
                # Keep its capacity slot until the supervisor removes its queue item.
                return ticket.value
            self.state = 'CANCELING' if self.state == 'RUNNING' else self.state
        budget = self.limits.cancel_s+self.limits.terminate_s+self.limits.kill_s+.1
        if not ticket.done.wait(budget):
            self._fatal('CANCEL_UNCONFIRMED')  # no reuse even when supervisor/audit is stuck
            return None
        return ticket.value

    def close(self):
        if self.state == 'CLOSED': return self.close_status
        deadline = time.monotonic()+self.limits.close_s
        self._closing.set()
        for ticket in list(self._tickets.values()):
            if not ticket.done.is_set(): ticket.reason = 'OWNER_CLOSED';ticket.cancel_requested.set()
        if self._thread is not None: self._thread.join(timeout=max(0., deadline-time.monotonic()))
        orderly = self._thread is None or not self._thread.is_alive()
        status = 'CLOSED'
        if orderly and self.session is not None and self.state == 'IDLE' and self._process.poll() is None:
            try:
                self._channel.write(frame('CLOSE', self.epoch, self.session['instance'], self._boot, None, {}),
                                    min(deadline, time.monotonic()+self.limits.write_s))
                kind, body = envelope(self._channel.read(deadline), epoch=self.epoch, instance=self.session['instance'],
                                      token=self._boot, request_id=None)
                require(kind == 'CLOSED' and body == {'state': 'CLOSED', 'session': self.session})
                if time.monotonic() > deadline: raise WireTimeout()
                self._process.wait(timeout=max(.001, deadline-time.monotonic()))
            except Exception:
                status = 'CLOSE_TIMEOUT'
        elif self._process is not None and self._process.poll() is None:
            status = 'CLOSE_TIMEOUT'
        stopped = self._stop_owned()
        if self._channel is not None: self._channel.close()
        if not orderly and self._thread is not None:
            self._thread.join(timeout=self.limits.terminate_s+self.limits.kill_s)
            orderly = not self._thread.is_alive()
        if not stopped or not orderly: status = 'CLEANUP_UNCONFIRMED'
        try: self._log('CLOSE', status=status, cleanup=dict(self.cleanup), supervisor_stopped=orderly)
        except Exception: status = 'AUDIT_UNAVAILABLE'
        if orderly:
            try: self.inspect(expected_head=self.journal.head)
            except Exception:
                self.audit_unavailable = True;status = 'AUDIT_UNAVAILABLE'
        self.close_status = status
        self.state = 'CLOSED' if stopped and orderly else 'UNHEALTHY'
        if orderly and stopped: self.journal.close()
        return status

    def inspect(self, *, expected_head=None):
        return inspect_journal(self.journal.root, expected_head=expected_head)


def body_source(path):
    return sha256_bytes(path.read_bytes())
