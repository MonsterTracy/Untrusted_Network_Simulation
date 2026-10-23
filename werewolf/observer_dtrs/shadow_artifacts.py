"""Immutable private Shadow journal, exact identities, at-most-once call recovery."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path

from werewolf.artifact_io import canonical_json_bytes, publish_artifact, read_artifact_file, verify_artifact
from werewolf.observer_dtrs.online_context import digest

VERSION = 'observer_dtrs_shadow_journal_v1'


class ShadowRecordedFailure(RuntimeError):
    """A terminal technical failure, never a cognitive BND."""


def _read(path, kind, identity):
    a = verify_artifact(path, expected_artifact_type='observer_dtrs_shadow_'+kind,
                        expected_schema_version=VERSION)
    if a.manifest['identity'] != identity or set(a.manifest['file_table']) != {'record.json'}:
        raise ValueError('Shadow journal identity/inventory mismatch')
    raw = read_artifact_file(a, 'record.json')
    record = json.loads(raw)
    if raw != canonical_json_bytes(record):
        raise ValueError('noncanonical Shadow journal record')
    if kind == 'failures' and (set(record) != {'status', 'code', 'stage', 'request_digest', 'engine_digest', 'rows'}
            or record['status'] != 'FAILED' or record['rows'] != []
            or record['code'] not in ('INFERENCE_TIMEOUT', 'INFERENCE_FAILED', 'INTERRUPTED_NO_RESULT')
            or record['stage'] not in ('ToM', 'composition', 'recovery')
            or record['request_digest'] != identity['request_digest']
            or record['engine_digest'] != identity['engine_digest']):
        raise ValueError('invalid Shadow technical failure record')
    return record


class ShadowJournal:
    """One private new run directory, never a canonical writer or public sink."""

    def __init__(self, root, *, engine_identity):
        if Path(root).is_symlink():
            raise ValueError('Shadow root must not be a symlink')
        self.root = Path(root).resolve()
        self._engine = canonical_json_bytes(engine_identity)
        if any((p/'manifest.json').exists() for p in (self.root, *self.root.parents)):
            raise ValueError('Shadow output cannot be inside an immutable artifact')
        if self.root.exists():
            allowed = {'contract', 'claims', 'results', 'failures', 'writer.lock'}
            if any(p.name not in allowed or p.is_symlink() for p in self.root.iterdir()):
                raise ValueError('Shadow directory contains unrelated files; stop without cleanup')
        self.root.mkdir(parents=True, exist_ok=True)
        with self.lock():
            if (self.root/'contract').exists():
                if _read(self.root/'contract', 'contract', {'engine_digest': self.engine_digest}) != self.engine:
                    raise ValueError('different Shadow contract')
            else:
                # Existing records without a contract are not a fresh run.
                if any((self.root/n).exists() for n in ('claims', 'results', 'failures')):
                    raise ValueError('missing Shadow contract; no automatic repair')
                self.publish('contract', '', self.engine, {'engine_digest': self.engine_digest})

    @property
    def engine(self):
        return json.loads(self._engine)

    @property
    def engine_digest(self):
        return digest(self.engine)

    @contextmanager
    def lock(self):
        with (self.root/'writer.lock').open('a') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError('Shadow writer already active') from error
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def publish(self, kind, key, record, identity):
        path = self.root/kind/key if key else self.root/kind
        return publish_artifact(path, manifest_fields={'artifact_type': 'observer_dtrs_shadow_'+kind,
            'schema_version': VERSION, 'identity': identity}, files={'record.json': canonical_json_bytes(record)})

    def begin(self, request):
        """Called while holding lock. Existing claims NEVER trigger another ToM call."""
        identity = {'engine_digest': self.engine_digest, 'request_digest': request.digest}
        key = request.digest
        self._verify_contract()
        if (self.root/'claims'/key).exists():
            if canonical_json_bytes(_read(self.root/'claims'/key, 'claims', identity)) != canonical_json_bytes(request.identity):
                raise ValueError('Shadow request binding mismatch')
            success = (self.root/'results'/key).exists()
            failure = (self.root/'failures'/key).exists()
            if success and failure:
                raise ValueError('conflicting Shadow terminal records')
            if success:
                return _read(self.root/'results'/key, 'results', identity)
            if failure:
                r = _read(self.root/'failures'/key, 'failures', identity)
                raise ShadowRecordedFailure(r['code'])
            self.fail(request, 'INTERRUPTED_NO_RESULT', 'recovery')
            raise ShadowRecordedFailure('INTERRUPTED_NO_RESULT')
        if any((self.root/n/key).exists() for n in ('results', 'failures')):
            raise ValueError('Shadow result without claim')
        self.publish('claims', key, request.identity, identity)
        return None

    def finish(self, request, result):
        self.publish('results', request.digest, result,
                     {'engine_digest': self.engine_digest, 'request_digest': request.digest})

    def fail(self, request, code, stage):
        # Do not persist raw exception messages/provider payloads: they may contain secrets.
        self.publish('failures', request.digest,
            {'status': 'FAILED', 'code': code, 'stage': stage, 'request_digest': request.digest,
             'engine_digest': self.engine_digest, 'rows': []},
            {'engine_digest': self.engine_digest, 'request_digest': request.digest})

    def _verify_contract(self):
        record = _read(self.root/'contract', 'contract', {'engine_digest': self.engine_digest})
        if canonical_json_bytes(record) != self._engine:
            raise ValueError('Shadow contract source changed')

    def _read_terminal(self, request):
        """Caller holds lock. Never create claims or convert pending calls."""
        identity = {'engine_digest': self.engine_digest, 'request_digest': request.digest}
        if canonical_json_bytes(_read(self.root/'claims'/request.digest, 'claims', identity)) != canonical_json_bytes(request.identity):
            raise ValueError('Shadow claim identity mismatch')
        present = [n for n in ('results', 'failures') if (self.root/n/request.digest).exists()]
        if len(present) > 1:
            raise ValueError('conflicting Shadow terminal records')
        return (present[0], _read(self.root/present[0]/request.digest, present[0], identity)) if present else ('pending', None)

    def inventory(self):
        """Read/validate every committed unit, including missing claims/terminal conflicts."""
        with self.lock():
            return self._inventory()

    def _inventory(self):
        """Caller holds lock so a full semantic inspect sees one stable inventory."""
        self._verify_contract()
        allowed = {'contract', 'claims', 'results', 'failures', 'writer.lock'}
        if any(p.name not in allowed or p.is_symlink() for p in self.root.iterdir()):
            raise ValueError('unexpected Shadow journal entry')
        keys = {n: {p.name for p in (self.root/n).iterdir()} if (self.root/n).exists() else set()
                for n in ('claims', 'results', 'failures')}
        if (keys['results'] & keys['failures'] or
                not (keys['results'] | keys['failures']) <= keys['claims']):
            raise ValueError('invalid Shadow journal terminal inventory')
        for kind, ids in keys.items():
            for key in ids:
                identity = {'engine_digest': self.engine_digest, 'request_digest': key}
                r = _read(self.root/kind/key, kind, identity)
                if kind == 'claims' and digest(r) != key:
                    raise ValueError('Shadow claim digest mismatch')
        return {**{k: len(v) for k, v in keys.items()}, 'request_digests': sorted(keys['claims']),
                'pending': len(keys['claims']-keys['results']-keys['failures'])}
