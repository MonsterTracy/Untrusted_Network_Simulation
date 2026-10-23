"""Private, single-writer durable capture/outcome ledger. Never a gameplay sink."""
from collections import Counter
import fcntl
import json
from pathlib import Path

from werewolf.artifact_io import canonical_json_bytes, publish_artifact, read_artifact_file, verify_artifact
from werewolf.observer_dtrs.online_context import digest

VERSION = 'observer_dtrs_shadow_receipts_v1'
CAPTURE_STATUSES = {'SUBMITTED', 'DUPLICATE', 'SKIPPED_BACKPRESSURE', 'SKIPPED_NONWOLF',
                    'CAPTURE_INVALID', 'CAPTURE_ERROR', 'SKIPPED_DISABLED'}
OUTCOME_STATUSES = {'SUCCEEDED', 'FAILED_TIMEOUT', 'FAILED_CRASH', 'FAILED_INFERENCE',
                    'FAILED_START', 'FAILED_STOPPED', 'INTERRUPTED_NO_RETRY'}


def _read(path, kind, identity):
    a = verify_artifact(path, expected_artifact_type='observer_dtrs_shadow_receipts_'+kind,
                        expected_schema_version=VERSION)
    if a.manifest['identity'] != identity or set(a.manifest['file_table']) != {'record.json'}:
        raise ValueError('Shadow receipt identity/inventory mismatch')
    raw = read_artifact_file(a, 'record.json')
    value = json.loads(raw)
    if raw != canonical_json_bytes(value):
        raise ValueError('noncanonical Shadow receipt')
    return value


class ReceiptLedger:
    """Immutable artifact units, hash chain, exclusive lifetime writer lock.

    publish_artifact fsyncs files and directories before returning. No repair,
    truncation or overwrite on corrupt/uncertain storage. The owner serializes calls.
    """

    def __init__(self, root, contract):
        root = Path(root)
        if root.is_symlink():
            raise ValueError('receipt root cannot be a symlink')
        self.root = root.resolve()
        if any((p/'manifest.json').exists() for p in (self.root, *self.root.parents)):
            raise ValueError('receipt root cannot be inside an immutable artifact')
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = (self.root/'writer.lock').open('a')
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.contract = json.loads(canonical_json_bytes(contract))
            self.identity = {'contract_digest': digest(self.contract)}
            if (self.root/'contract').exists():
                if _read(self.root/'contract', 'contract', self.identity) != self.contract:
                    raise ValueError('different Shadow receipt contract')
            elif set(p.name for p in self.root.iterdir()) == {'writer.lock'}:
                self._publish('contract', '', self.contract, self.identity)
            else:
                raise ValueError('missing receipt contract; no automatic repair')
            self.events = self.read()
        except BaseException:
            self.close()
            raise

    def _publish(self, kind, key, value, identity):
        path = self.root/kind/key if key else self.root/kind
        return publish_artifact(path, manifest_fields={
            'artifact_type': 'observer_dtrs_shadow_receipts_'+kind,
            'schema_version': VERSION, 'identity': identity},
            files={'record.json': canonical_json_bytes(value)})

    def read(self, *, expected_head=None):
        if any(p.is_symlink() or p.name not in {'contract', 'events', 'writer.lock'}
               for p in self.root.iterdir()):
            raise ValueError('unexpected receipt entry')
        if _read(self.root/'contract', 'contract', self.identity) != self.contract:
            raise ValueError('receipt contract changed')
        paths = sorted((self.root/'events').iterdir()) if (self.root/'events').exists() else []
        events, previous, admitted, completed = [], None, {}, set()
        for i, path in enumerate(paths):
            if path.name != f'{i:08d}':
                raise ValueError('missing/unexpected receipt sequence')
            r = _read(path, 'events', {**self.identity, 'sequence': i})
            if (set(r) != {'sequence', 'previous', 'kind', 'status', 'pre', 'request',
                           'request_digest', 'token', 'result_digest'}
                    or r['sequence'] != i or r['previous'] != previous):
                raise ValueError('invalid receipt chain')
            key = r['request_digest']
            if r['kind'] == 'capture':
                if r['status'] not in CAPTURE_STATUSES or r['result_digest'] is not None:
                    raise ValueError('invalid capture receipt')
                if r['request'] is not None:
                    if digest(r['request']) != key:
                        raise ValueError('receipt request digest mismatch')
                elif key is not None:
                    raise ValueError('receipt missing request')
                if r['status'] in {'SUBMITTED', 'DUPLICATE', 'SKIPPED_BACKPRESSURE'} and key is None:
                    raise ValueError('eligible receipt missing legal request')
                if r['status'] == 'SUBMITTED':
                    if key in admitted or r['token'] != digest([self.identity, key, i]):
                        raise ValueError('duplicate/invalid admission token')
                    admitted[key] = r['token']
                elif r['token'] is not None:
                    raise ValueError('nonadmission token')
                if r['status'] == 'DUPLICATE' and key not in admitted:
                    raise ValueError('duplicate without admission')
            elif r['kind'] == 'outcome':
                if (r['status'] not in OUTCOME_STATUSES or key not in admitted or key in completed
                        or r['token'] != admitted[key] or r['request'] is not None or r['pre'] is not None
                        or ((r['status'] == 'SUCCEEDED') != (r['result_digest'] is not None))):
                    raise ValueError('invalid/duplicate terminal outcome')
                completed.add(key)
            else:
                raise ValueError('invalid receipt kind')
            previous = digest(r)
            events.append(r)
        if expected_head is not None and previous != expected_head:
            raise ValueError('external receipt head mismatch')
        return events

    def append(self, *, kind, status, pre=None, request=None, request_digest=None,
               token=None, result_digest=None):
        # Validate the complete short per-game ledger before extending it.
        if self.read() != self.events:
            raise ValueError('receipt history changed')
        i = len(self.events)
        if status == 'SUBMITTED':
            token = digest([self.identity, request_digest, i])
        r = {'sequence': i, 'previous': digest(self.events[-1]) if self.events else None,
             'kind': kind, 'status': status, 'pre': pre, 'request': request,
             'request_digest': request_digest, 'token': token, 'result_digest': result_digest}
        r = json.loads(canonical_json_bytes(r))
        self._publish('events', f'{i:08d}', r, {**self.identity, 'sequence': i})
        self.events.append(r)
        return r

    def summary(self, *, expected_head=None):
        events = self.read(expected_head=expected_head)
        captures = [r for r in events if r['kind'] == 'capture']
        outcomes = [r for r in events if r['kind'] == 'outcome']
        c, o = Counter(r['status'] for r in captures), Counter(r['status'] for r in outcomes)
        eligible = [r for r in captures if r['request'] is not None]
        return {'capture_attempts': len(captures), 'eligible_captures': len(eligible),
                'unique_eligible_PREs': len({r['request_digest'] for r in eligible}),
                'eligibility_unknown': c['CAPTURE_INVALID']+c['CAPTURE_ERROR'],
                'capture_statuses': dict(c), 'outcome_statuses': dict(o),
                'submitted': c['SUBMITTED'], 'completed': len(outcomes),
                'pending': c['SUBMITTED']-len(outcomes),
                'head_digest': digest(events[-1]) if events else None}

    def close(self):
        if not self._lock.closed:
            self._lock.close()
