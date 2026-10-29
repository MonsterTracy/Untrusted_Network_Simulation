"""Spawn-safe local lease: only a socket/trust bytes cross, never the live owner.

Qualification adapter, not a TomIdentity conversion or CAL/Shadow authorization.
The parent must keep the broker alive until a child has exited. Closing a lease
or losing the child cancels its active request through the original owner.
"""
from dataclasses import dataclass, field
import json
import math
import secrets
import socket
from threading import Event, Lock, Thread
import time

from werewolf.artifact_io import canonical_json_bytes
from werewolf.observer_dtrs.final_tom_client import ResidentOwner, ResidentResult
from werewolf.observer_dtrs.final_tom_wire import Channel, ProtocolError, WireTimeout, digest, require

VERSION = 'observer_dtrs_qual_broker_v1'


def _message(kind, lease, request_id, payload):
    return {'version': VERSION, 'kind': kind, 'lease': lease,
            'request_id': request_id, 'payload': payload}


def _validate(message, lease):
    require(type(message) is dict and set(message) == {'version','kind','lease','request_id','payload'}
            and message['version'] == VERSION and message['lease'] == lease, 'broker lease mismatch')
    return message['kind'], message['request_id'], message['payload']


@dataclass
class SpawnLease:
    """Transfer once with multiprocessing spawn; single client thread per lease.

    Pass to a child, then call broker.release_parent_copy(lease) in the parent.
    A lease is epoch-specific; owner recovery always needs a fresh lease.
    """
    sock: socket.socket
    binding_bytes: bytes
    limits: object
    _channel: object = field(default=None,init=False,repr=False)

    @property
    def binding(self):
        return json.loads(self.binding_bytes)

    def predict(self, prefix, *, request_id, timeout_s):
        require(type(timeout_s) in (int,float) and math.isfinite(timeout_s)
                and 0 < timeout_s <= self.limits.query_s, 'bounded broker query required')
        deadline = time.monotonic()+timeout_s
        if self._channel is None:
            self._channel = Channel(self.sock,self.limits)
        channel = self._channel
        binding = self.binding
        try:
            channel.write(_message('QUERY', binding, request_id,
                {'pre': prefix.to_record(), 'deadline_monotonic': deadline}),
                min(deadline, time.monotonic()+self.limits.write_s))
            kind, rid, body = _validate(channel.read(deadline), binding)
            require(kind == 'RESULT' and rid == request_id and type(body) is dict
                    and set(body) == {'admission','result'}, 'broker response identity')
            require(time.monotonic() < deadline, 'late broker response')
            if body['result'] is not None:
                result = ResidentResult(canonical_json_bytes(body['result']))
                if result.status == 'SUCCEEDED':
                    p = result.record['prediction']
                    require(p['game_id'] == prefix.game_id and p['boundary_id'] == prefix.boundary_id
                            and p['prefix_digest'] == prefix.prefix_digest
                            and digest(result.record['session']) == binding['session_digest'],
                            'broker result PRE/source mismatch')
                    result.matrix  # original probability validator; no conversion/renormalization
                return body['admission'], result
            return body['admission'], None
        except Exception:
            # A lost/expired lease cannot be reused. EOF makes the parent cancel;
            # no claim of confirmed stop is made by the client itself.
            self.close()
            raise

    def close(self):
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()


class ResidentBroker:
    """No listener/network/pickle message protocol; explicitly owned socketpairs."""

    def __init__(self, owner, *, max_clients=1):
        require(type(owner) is ResidentOwner and owner.state == 'IDLE'
                and type(max_clients) is int and 1 <= max_clients <= owner.limits.max_pending,
                'started exclusive owner and bounded broker required')
        self.owner, self.max_clients = owner, max_clients
        self._lock, self._closing = Lock(), Event()
        self._sessions = []

    def lease(self):
        with self._lock:
            active = sum(t.is_alive() for _,t in self._sessions)
            require(not self._closing.is_set() and active < self.max_clients
                    and len(self._sessions) < self.owner.limits.max_requests
                    and self.owner.state in ('IDLE','RUNNING'), 'broker unavailable/backpressure')
            parent, child = socket.socketpair()
            binding = {'schema_version': VERSION, 'token': secrets.token_hex(32),
                       'epoch': self.owner.epoch, 'session_digest': digest(self.owner.session)}
            thread = Thread(target=self._serve, args=(parent, binding), daemon=True,
                            name='capsule-qualification-broker')
            self._sessions.append((parent,thread));thread.start()
            return SpawnLease(child, canonical_json_bytes(binding), self.owner.limits)

    @staticmethod
    def release_parent_copy(lease):
        # Do not shutdown: the child owns the duplicated FD after process.start().
        lease.sock.close()

    def _serve(self, sock, binding):
        channel, active = Channel(sock, self.owner.limits), None
        requests = 0
        try:
            while not self._closing.is_set():
                require(binding['epoch'] == self.owner.epoch
                        and binding['session_digest'] == digest(self.owner.session), 'stale broker epoch')
                try:
                    message = channel.read(time.monotonic()+.02)
                except WireTimeout:
                    continue
                kind, rid, payload = _validate(message, binding)
                require(kind == 'QUERY' and type(rid) is str and 0 < len(rid) <= 128
                        and type(payload) is dict and set(payload) == {'pre','deadline_monotonic'},
                        'broker query schema')
                requests += 1
                require(requests <= self.owner.limits.max_requests, 'broker request quota')
                deadline = payload['deadline_monotonic']
                require(type(deadline) in (int,float) and math.isfinite(deadline)
                        and 0 < deadline-time.monotonic() <= self.owner.limits.query_s,
                        'expired/invalid broker deadline')
                from werewolf.canonical_collection.game_bundle import _prefix_from_record
                prefix = _prefix_from_record(payload['pre'])
                require(prefix.to_record() == payload['pre'], 'broker PRE allowlist/roundtrip')
                # Keep the public 128-character ID contract and the Owner's
                # separate 128-character bound, including the lease namespace.
                admission, active = self.owner.submit(prefix, request_id=binding['token'][:16]+':'+digest(rid),
                    timeout_s=deadline-time.monotonic())
                if active is not None and admission == 'SUBMITTED':
                    while not active.done.wait(.01):
                        if self._closing.is_set() or time.monotonic() >= deadline:
                            self.owner.cancel(active, reason='CLIENT_TIMEOUT');break
                        # A closed/crashed child cannot leave its request running.
                        try:
                            extra = channel.read(time.monotonic()+.001)
                        except WireTimeout:
                            continue
                        raise ProtocolError('no concurrent query on a lease')
                # Duplicate IDs never replay a cached success in a new client call.
                result = active.value.record if admission == 'SUBMITTED' and active.done.is_set() else None
                if result is not None and result['status'] == 'SUCCEEDED':
                    require(time.monotonic() < deadline, 'expired broker success')
                channel.write(_message('RESULT', binding, rid, {'admission': admission, 'result': result}),
                    min(deadline, time.monotonic()+self.owner.limits.write_s))
                active = None
        except (ValueError, OSError, TimeoutError, KeyError, TypeError):
            pass  # original owner journal is the durable request/error evidence
        finally:
            if active is not None and not active.done.is_set():
                self.owner.cancel(active, reason='CLIENT_CANCELLED')
            channel.close()

    def close(self):
        self._closing.set()
        budget = self.owner.limits.cancel_s+self.owner.limits.terminate_s+self.owner.limits.kill_s+.5
        deadline = time.monotonic()+budget
        for sock, thread in self._sessions:
            thread.join(timeout=max(0., deadline-time.monotonic()))
        stopped = all(not thread.is_alive() for _,thread in self._sessions)
        require(stopped, 'broker stop unconfirmed; owner must not be reused')
        return True
