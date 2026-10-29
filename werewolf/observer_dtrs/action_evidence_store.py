"""Research-only durable custody for the Phase-2D validator; never a game driver.

One store is pinned to ONE game execution attempt. flock excludes cooperating
local writers, not hostile users, remote writers or external game executors.
A caller-custodied high-water digest is REQUIRED on reopen. A colocally rolled
back journal AND anchor AND caller checkpoint cannot be detected here.
"""
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import stat
import threading

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.observer_dtrs.action_evidence_ledger import digest, extend_ledger, validate_ledger

VERSION = 'research_action_evidence_store_v1'
MAX_BYTES = 16 * 1024 * 1024


class StoreError(ValueError):
    pass


def _require(ok, message):
    if not ok:
        raise StoreError(message)


def _text(value):
    _require(type(value) is str and bool(value.strip()), 'nonempty identity required')


def _hash(value):
    _require(type(value) is str and len(value) == 64 and
             all(c in '0123456789abcdef' for c in value), 'trusted anchor digest required')


def _write_all(fd, raw):
    while raw:
        n = os.write(fd, raw)
        if n <= 0:
            raise OSError('short write without progress')
        raw = raw[n:]


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _open_file(path, *, create=False):
    flags = os.O_RDWR | os.O_NOFOLLOW | (os.O_CREAT | os.O_EXCL if create else 0)
    fd = os.open(path, flags, 0o600)
    try:
        s = os.fstat(fd)
        _require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and s.st_uid == os.getuid()
                 and stat.S_IMODE(s.st_mode) == 0o600, 'unsafe file ownership/type/mode')
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read(fd):
    size = os.fstat(fd).st_size
    _require(size <= MAX_BYTES, 'research journal size bound exceeded')
    raw = os.pread(fd, size + 1, 0)
    _require(len(raw) == size, 'concurrent size change')
    return raw


def _frames(raw):
    _require(not raw or raw.endswith(b'\n'), 'partial frame; no automatic repair')
    try:
        rows = tuple(json.loads(line) for line in raw.splitlines())
        _require(b''.join(canonical_json_bytes(r) + b'\n' for r in rows) == raw,
                 'noncanonical journal')
        return rows
    except (ValueError, TypeError) as error:
        raise StoreError('invalid/noncanonical journal; no automatic repair') from error


def _append_synced(fd, raw):
    _require(os.fstat(fd).st_size + len(raw) <= MAX_BYTES, 'research journal size bound exceeded')
    os.lseek(fd, 0, os.SEEK_END)
    _write_all(fd, raw)
    os.fsync(fd)


@dataclass(frozen=True)
class DurableReceipt:
    operation_id: str
    record_digest: str
    sequence: int
    anchor_digest: str
    execution_permission: bool = False


class ActionEvidenceStore:
    """Single process/thread owner; explicit create versus anchored reopen.

    append -> data fsync -> independent-path anchor append/fsync -> receipt.
    Interrupted writes are NEVER truncated, adopted, repaired or replayed.
    The separate anchor path must have independent custody for rollback claims;
    mere path separation on one host does not establish that trust.
    """

    @classmethod
    def create(cls, root, anchor_root, *, ledger_id, game_id, game_attempt_id):
        for value in (ledger_id, game_id, game_attempt_id):
            _text(value)
        return cls(root, anchor_root, create=True, expected_anchor=None,
                   namespace={'ledger_id': ledger_id, 'game_id': game_id,
                              'game_attempt_id': game_attempt_id})

    @classmethod
    def reopen(cls, root, anchor_root, *, ledger_id, game_id, game_attempt_id, expected_anchor):
        _hash(expected_anchor)  # None is not an unanchored recovery option, even for empty history.
        for value in (ledger_id, game_id, game_attempt_id):
            _text(value)
        return cls(root, anchor_root, create=False, expected_anchor=expected_anchor,
                   namespace={'ledger_id': ledger_id, 'game_id': game_id,
                              'game_attempt_id': game_attempt_id})

    def __init__(self, root, anchor_root, *, create, expected_anchor, namespace):
        self._fds = {}
        self._pid, self._thread = os.getpid(), threading.get_ident()
        self._healthy = True
        root, anchor_root = Path(root).absolute(), Path(anchor_root).absolute()
        _require(not root.is_symlink() and not anchor_root.is_symlink(), 'symlink root rejected')
        self.root, self.anchor_root = root.resolve(), anchor_root.resolve()
        _require(self.root != self.anchor_root and self.root not in self.anchor_root.parents
                 and self.anchor_root not in self.root.parents, 'separate custody directories required')
        self._namespace = dict(namespace)
        self._contract = {'version': VERSION, 'scope': 'RESEARCH_ONLY_NO_EXECUTION_PERMISSION',
                          **namespace, 'journal_root': str(self.root), 'anchor_root': str(self.anchor_root)}
        try:
            for folder in (self.root, self.anchor_root):
                if create:
                    folder.mkdir(mode=0o700)  # Exclusive, no mkdir(parents), no cleanup on failure.
                    _sync_dir(folder.parent)
                s = folder.lstat()
                _require(stat.S_ISDIR(s.st_mode) and s.st_uid == os.getuid()
                         and stat.S_IMODE(s.st_mode) == 0o700, 'unsafe directory ownership/type/mode')
            self._dirs = {p: (p.stat().st_dev, p.stat().st_ino) for p in (self.root, self.anchor_root)}
            # Both locks live for the entire owner lifetime. Never unlink lock files.
            for folder in (self.root, self.anchor_root):
                fd = _open_file(folder / 'writer.lock', create=create)
                self._fds[folder / 'writer.lock'] = fd
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise StoreError('writer ownership busy') from error
            for folder, names in ((self.root, ('contract.json', 'events.jsonl')),
                                  (self.anchor_root, ('anchors.jsonl',))):
                for name in names:
                    p = folder / name
                    self._fds[p] = _open_file(p, create=create)
            if create:
                _append_synced(self._fds[self.root / 'contract.json'], canonical_json_bytes(self._contract))
                genesis = self._anchor_record((), b'', previous=None, generation=0)
                _append_synced(self._fds[self.anchor_root / 'anchors.jsonl'], canonical_json_bytes(genesis) + b'\n')
                _sync_dir(self.root)
                _sync_dir(self.anchor_root)
                expected_anchor = genesis['anchor_digest']
            self._records, self._anchor, _ = self._validate(expected_anchor)
        except BaseException:
            self.close()
            raise

    def _owner(self):
        _require(self._healthy and self._fds, 'writer closed or unhealthy; stop')
        _require(os.getpid() == self._pid and threading.get_ident() == self._thread,
                 'only original owner process/thread may use writer')
        for path, (dev, ino) in self._dirs.items():
            s = path.lstat()
            _require(stat.S_ISDIR(s.st_mode) and (s.st_dev, s.st_ino) == (dev, ino)
                     and stat.S_IMODE(s.st_mode) == 0o700, 'custody directory replaced')
        for path, fd in self._fds.items():
            a, b = path.lstat(), os.fstat(fd)
            _require(stat.S_ISREG(a.st_mode) and a.st_nlink == 1 and
                     stat.S_IMODE(a.st_mode) == 0o600 and (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino),
                     'custody file replaced/aliased')

    def _anchor_record(self, records, raw, *, previous, generation):
        r = {'version': VERSION, 'contract_digest': digest(self._contract),
             'generation': generation, 'previous': previous,
             'record_count': len(records), 'head': records[-1]['record_digest'] if records else None,
             'byte_count': len(raw), 'bytes_digest': sha256_bytes(raw)}
        return {**r, 'anchor_digest': digest(r)}

    def _validate(self, expected_anchor):
        self._owner()
        _require(set(p.name for p in self.root.iterdir()) == {'writer.lock', 'contract.json', 'events.jsonl'}
                 and set(p.name for p in self.anchor_root.iterdir()) == {'writer.lock', 'anchors.jsonl'},
                 'unexpected custody files')
        _require(_read(self._fds[self.root / 'contract.json']) == canonical_json_bytes(self._contract),
                 'attempt/contract identity mismatch')
        raw = _read(self._fds[self.root / 'events.jsonl'])
        records = _frames(raw)
        validate_ledger(records, ledger_id=self._namespace['ledger_id'])
        for r in records:
            if r['kind'] == 'OPPORTUNITY':
                _require(all(r['payload'][k] == self._namespace[k] for k in ('game_id', 'game_attempt_id')),
                         'mixed execution attempt rejected')
        anchors = _frames(_read(self._fds[self.anchor_root / 'anchors.jsonl']))
        _require(bool(anchors), 'missing high-water anchor')
        # Every committed prefix is tied to its actual canonical bytes, not merely a head string.
        prefix_raw, previous = b'', None
        for i, a in enumerate(anchors):
            _require(i <= len(records), 'suffix loss relative to high-water anchor')
            if i:
                prefix_raw += canonical_json_bytes(records[i-1]) + b'\n'
            expected = self._anchor_record(records[:i], prefix_raw, previous=previous, generation=i)
            _require(canonical_json_bytes(a) == canonical_json_bytes(expected),
                     'anchor/bytes/history mismatch; no automatic reconciliation')
            previous = a['anchor_digest']
        _require(len(anchors) == len(records) + 1, 'unanchored journal suffix; stop')
        _require(previous == expected_anchor, 'stale/untrusted caller high-water anchor; stop')
        return records, anchors[-1], raw

    @property
    def anchor_digest(self):
        self._owner()
        return self._anchor['anchor_digest']

    @property
    def records(self):
        self._owner()
        return tuple(json.loads(canonical_json_bytes(r)) for r in self._records)

    def inspect(self):
        records, _, _ = self._validate(self.anchor_digest)
        return validate_ledger(records, ledger_id=self._namespace['ledger_id'])

    def append(self, operation_id, kind, payload):
        self._owner()
        try:
            records, anchor, raw = self._validate(self._anchor['anchor_digest'])
            if kind == 'OPPORTUNITY':
                _require(all(payload.get(k) == self._namespace[k] for k in ('game_id', 'game_attempt_id')),
                         'mixed execution attempt rejected')
            extended = extend_ledger(records, ledger_id=self._namespace['ledger_id'],
                                     operation_id=operation_id, kind=kind, payload=payload)
            if len(extended) == len(records):
                original = next(r for r in records if r['operation_id'] == operation_id)
                return DurableReceipt(operation_id, original['record_digest'], original['sequence'],
                                      anchor['anchor_digest'])
            r = extended[-1]
            line = canonical_json_bytes(r) + b'\n'
            next_anchor = self._anchor_record(extended, raw + line, previous=anchor['anchor_digest'],
                                              generation=anchor['generation'] + 1)
            _append_synced(self._fds[self.root / 'events.jsonl'], line)
            _append_synced(self._fds[self.anchor_root / 'anchors.jsonl'], canonical_json_bytes(next_anchor) + b'\n')
            self._records, self._anchor = extended, next_anchor
            return DurableReceipt(operation_id, r['record_digest'], r['sequence'], next_anchor['anchor_digest'])
        except BaseException:
            # Includes fsync/anchor/ACK uncertainty; never use this instance again.
            self._healthy = False
            raise

    def close(self):
        # close, never LOCK_UN: an inherited child's close must not unlock parent's owner.
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
        self._healthy = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@dataclass(frozen=True)
class ReadbackQuery:
    assignment_ref: str
    reservation_ref: str
    attempt_token: str
    opportunity_json: bytes


def reconcile_readback(store, *, operation_id, assignment_ref, provider):
    """Read-only external provider; NO executor/callback/fallback is accepted.

    None means unresolved, with no fabricated evidence. Provider truth, executor
    cessation and authority are outside this module. Even NOT_COMMITTED issues
    no permission. The Phase-2D state machine checks the complete PRE binding.
    """
    store.inspect()
    records = store.records
    assignment = next((r for r in records if r['record_digest'] == assignment_ref
                       and r['kind'] == 'ASSIGNMENT'), None)
    _require(assignment is not None, 'unknown assignment')
    op = next(r['payload'] for r in records if r['record_digest'] == assignment['payload']['opportunity_ref'])
    reservation = next((r for r in records if r['kind'] == 'COMMIT_WITNESS' and
                        r['payload'].get('assignment_ref') == assignment_ref and
                        r['payload']['status'] == 'ATTEMPT_RESERVED'), None)
    _require(reservation is not None, 'no anchored reservation for external readback')
    view = store.inspect()
    current = next(a for a in view.assignments if a.assignment_ref == assignment_ref)
    if current.commit_status in ('COMMITTED', 'NOT_COMMITTED'):
        # Identical operation may be re-read after ACK loss, but do not call provider again.
        prior = next((r for r in records if r['operation_id'] == operation_id), None)
        _require(prior is not None and prior['kind'] == 'COMMIT_WITNESS' and
                 prior['payload']['assignment_ref'] == assignment_ref and
                 prior['payload']['status'] == current.commit_status,
                 'already resolved; no new readback decision')
        return store.append(operation_id, prior['kind'], prior['payload'])
    query = ReadbackQuery(assignment_ref, reservation['record_digest'],
                         reservation['payload']['attempt_token'], canonical_json_bytes(op))
    result = provider.readback(query)
    if result is None:
        return None
    _require(type(result) is dict and result.get('status') in ('COMMITTED', 'NOT_COMMITTED')
             and result.get('assignment_ref') == assignment_ref
             and result.get('reservation_ref') == reservation['record_digest'],
             'unresolved/mismatched external readback; stop')
    return store.append(operation_id, 'COMMIT_WITNESS', result)
