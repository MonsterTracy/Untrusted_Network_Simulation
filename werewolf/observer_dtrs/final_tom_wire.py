"""Local, bounded, canonical JSON IPC; no model, training or game imports."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import select
import socket
import struct
import time

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes

VERSION = 'observer_dtrs_resident_tom_ipc_v1'
ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return sha256_bytes(canonical_json_bytes(value))


def require(test, reason='PROTOCOL_ERROR'):
    if not test:
        raise ProtocolError(reason)


class ProtocolError(ValueError):
    pass


class WireTimeout(TimeoutError):
    pass


class PeerClosed(ConnectionError):
    pass


@dataclass(frozen=True)
class Limits:
    cold_start_s: float = 10.
    ready_s: float = 15.
    write_s: float = 1.
    query_s: float = 10.
    service_s: float = 8.
    cancel_s: float = 1.
    close_s: float = 2.
    terminate_s: float = .5
    kill_s: float = 1.
    frame_s: float = 1.
    max_frame_bytes: int = 4*1024*1024
    max_request_bytes: int = 2*1024*1024
    max_read_bytes: int = 64*1024*1024
    max_write_bytes: int = 64*1024*1024
    max_pending: int = 2
    max_requests: int = 1024

    def __post_init__(self):
        for n, v in asdict(self).items():
            if n.endswith('_s'):
                require(type(v) in (int, float) and math.isfinite(v) and 0 < v <= 3600, 'invalid deadline')
            else:
                require(type(v) is int and 0 < v <= 2**30, 'invalid bound')
        require(self.max_request_bytes <= self.max_frame_bytes
                and self.max_read_bytes >= self.max_frame_bytes
                and self.max_write_bytes >= self.max_frame_bytes, 'inconsistent byte bounds')


def hex_id(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None, 'invalid identity/token')


def frame(kind, epoch, instance, token, request_id, payload):
    return {'version': VERSION, 'kind': kind, 'epoch': epoch, 'instance': instance,
            'token': token, 'request_id': request_id, 'payload': payload}


def envelope(message, *, epoch, instance, token, request_id):
    require(type(message) is dict and set(message) == {
        'version', 'kind', 'epoch', 'instance', 'token', 'request_id', 'payload'})
    require(message['version'] == VERSION and message['epoch'] == epoch
            and message['instance'] == instance and message['token'] == token
            and message['request_id'] == request_id, 'response identity/token mismatch')
    return message['kind'], message['payload']


def service_source():
    names = ('werewolf/observer_dtrs/final_tom_wire.py', 'werewolf/observer_dtrs/final_tom_service.py',
             'scripts/serve_final_tom_capsule.py')
    hashes = {n: sha256_bytes((ROOT/n).read_bytes()) for n in names}
    return {'sha256': hashes, 'digest': digest(hashes)}


class Channel:
    """One reader/writer thread. Partial frames survive polling but have an age cap."""
    def __init__(self, sock, limits):
        self.sock, self.limits = sock, limits
        sock.setblocking(False)
        self.buffer = bytearray()
        self.received = self.sent = 0
        self.partial_since = None

    def _wait(self, write, deadline):
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise WireTimeout()
        r, w, _ = select.select([] if write else [self.sock], [self.sock] if write else [], [], remaining)
        if not (w if write else r):
            raise WireTimeout()

    def write(self, message, deadline):
        payload = canonical_json_bytes(message)
        require(0 < len(payload) <= self.limits.max_frame_bytes, 'frame too large')
        data = struct.pack('!I', len(payload))+payload
        require(self.sent+len(data) <= self.limits.max_write_bytes, 'cumulative write budget')
        offset = 0
        while offset < len(data):
            self._wait(True, deadline)
            try:
                n = self.sock.send(memoryview(data)[offset:])
            except BlockingIOError:
                continue
            if n == 0:
                raise PeerClosed()
            offset += n
            self.sent += n
        if time.monotonic() > deadline:
            raise WireTimeout()

    def read(self, deadline):
        while True:
            if self.partial_since is not None and time.monotonic()-self.partial_since > self.limits.frame_s:
                raise ProtocolError('partial frame expired')
            if len(self.buffer) >= 4:
                size = struct.unpack('!I', self.buffer[:4])[0]
                require(0 < size <= self.limits.max_frame_bytes, 'frame too large')
                if len(self.buffer) >= size+4:
                    raw = bytes(self.buffer[4:size+4]);del self.buffer[:size+4]
                    self.partial_since = time.monotonic() if self.buffer else None
                    try:
                        record = json.loads(raw)
                        require(canonical_json_bytes(record) == raw, 'noncanonical/duplicate JSON')
                    except (UnicodeError, ValueError, TypeError) as error:
                        raise ProtocolError('invalid frame JSON') from error
                    return record
            actual_deadline = deadline
            if self.partial_since is not None:
                actual_deadline = min(deadline, self.partial_since+self.limits.frame_s)
            self._wait(False, actual_deadline)
            # Never buffer more than one frame + the fixed prefix.
            wanted = 4-len(self.buffer) if len(self.buffer) < 4 else size+4-len(self.buffer)
            try:
                data = self.sock.recv(min(65536, wanted))
            except BlockingIOError:
                continue
            if not data:
                raise PeerClosed()
            self.received += len(data)
            require(self.received <= self.limits.max_read_bytes, 'cumulative read budget')
            if not self.buffer:
                self.partial_since = time.monotonic()
            self.buffer.extend(data)

    def close(self):
        self.sock.close()
