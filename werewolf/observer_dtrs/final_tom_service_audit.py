"""New-run-only, durable owner journal. No automatic repair or request replay."""
import fcntl
from collections import Counter
import json
import os
from pathlib import Path
from threading import Lock

from werewolf.artifact_io import canonical_json_bytes, publish_artifact, read_artifact_file, verify_artifact
from werewolf.observer_dtrs.final_tom_wire import VERSION, Limits, digest, hex_id, require
from werewolf.observer_dtrs.probabilities import ProbabilityMatrix


def _read(path, kind):
    a = verify_artifact(path, expected_artifact_type='observer_dtrs_resident_'+kind, expected_schema_version=VERSION)
    require(set(a.manifest) == {'artifact_type', 'schema_version', 'file_table', 'manifest_digest'}
            and set(a.manifest['file_table']) == {'record.json'}, 'audit inventory/schema')
    raw = read_artifact_file(a, 'record.json');r = json.loads(raw)
    require(canonical_json_bytes(r) == raw, 'audit canonical bytes')
    return r


def inspect_journal(root, *, expected_head=None):
    root = Path(root)
    require(not root.is_symlink() and {p.name for p in root.iterdir()} <= {'contract', 'events', 'writer.lock'})
    contract = _read(root/'contract', 'contract')
    Limits(**contract['limits'])
    events, previous, admitted, dispatched, done = [], None, {}, set(), set()
    pres, sessions, epochs, started = {}, {}, set(), set()
    directory = root/'events'
    require(not directory.is_symlink(), 'audit events alias')
    paths = sorted(directory.iterdir()) if directory.exists() else []
    for i, path in enumerate(paths):
        require(path.name == f'{i:08d}', 'missing audit sequence')
        r = _read(path, 'event')
        require(set(r) == {'sequence', 'previous', 'contract_digest', 'kind', 'request_id', 'epoch', 'token', 'payload'}
                and r['sequence'] == i and r['previous'] == previous
                and r['contract_digest'] == digest(contract), 'audit chain identity')
        rid = r['request_id']
        require(r['kind'] in {'EPOCH_START', 'READY', 'START_FAILED', 'ADMISSION', 'ADMISSION_REJECTED',
                             'DISPATCHED', 'STARTED', 'OUTCOME', 'LATE_SUCCESS_REJECTED', 'CLOSE'}, 'audit event kind')
        hex_id(r['epoch'])
        if r['kind'] == 'EPOCH_START':
            require(r['epoch'] not in epochs, 'epoch reused')
            epochs.add(r['epoch'])
        if r['kind'] == 'READY':
            s = r['payload']['session'];c = s['capsule'];t = contract['trust']
            require(r['epoch'] in epochs and r['epoch'] not in sessions and s['epoch'] == r['epoch']
                    and s['schema_version'] == 'observer_dtrs_capsule_service_source_v1'
                    and s['service_source'] == contract['service_source']
                    and s['entry_sha256'] == contract['entry_sha256']
                    and s['transport_scope'] == contract['transport_scope']
                    and c['capsule_digest'] == t['capsule_digest'] and c['scope'] == t['capsule_scope']
                    and c['loader_source']['loader']['implementation_digest'] == t['loader_digest']
                    and digest(c['runtime']) == t['runtime_digest'], 'audit ready source mismatch')
            hex_id(s['instance']);sessions[r['epoch']] = s
        if r['kind'] == 'ADMISSION' and r['payload']['status'] == 'SUBMITTED':
            require(rid not in admitted and r['epoch'] in sessions, 'duplicate/unready durable admission')
            hex_id(r['token'])
            admitted[rid] = (r['epoch'], r['token'])
            pres[rid] = r['payload']['pre']
        if r['kind'] in ('DISPATCHED', 'STARTED', 'OUTCOME'):
            require(rid in admitted and admitted[rid] == (r['epoch'], r['token']), 'audit request binding')
        if r['kind'] == 'DISPATCHED':
            require(rid not in dispatched and rid not in done, 'request redispatched')
            dispatched.add(rid)
        if r['kind'] == 'STARTED':
            require(rid in dispatched and rid not in done and rid not in started, 'started without unique dispatch')
            started.add(rid)
        if r['kind'] == 'OUTCOME':
            require(rid not in done, 'terminal evidence overwritten')
            body = r['payload']
            require(body['pre'] == pres[rid] and type(body['execution_stopped']) is bool, 'outcome PRE binding')
            if body['status'] == 'SUCCEEDED':
                require(rid in started and body['session'] == sessions[r['epoch']]
                        and body['execution_stopped'] is True and body['state'] == 'IDLE', 'unconfirmed success')
                p = body['prediction']
                require(set(p) == set(pres[rid]) | {'probabilities', 'capsule_identity'}
                        and {n: p[n] for n in pres[rid]} == pres[rid]
                        and p['capsule_identity'] == sessions[r['epoch']]['capsule'], 'audit prediction identity')
                ProbabilityMatrix(p['game_id'], p['boundary_id'], p['prefix_digest'], p['probabilities'])
            else:
                require(body['status'] in {'CLIENT_TIMEOUT', 'CLIENT_CANCELLED', 'CANCEL_UNCONFIRMED',
                    'OWNER_CLOSED', 'SKIPPED_UNHEALTHY', 'SERVICE_TIMEOUT', 'WRITE_TIMEOUT',
                    'SERVICE_CRASH', 'PROTOCOL_ERROR', 'INFERENCE_ERROR', 'AUDIT_UNAVAILABLE'}
                    and body['prediction'] is None, 'technical failure has no prediction')
            require(body['execution_stopped'] or body['state'] != 'IDLE', 'idle with unconfirmed execution')
            done.add(rid)
        previous = digest(r);events.append(r)
    if expected_head is not None:
        require(previous == expected_head, 'external audit head mismatch')
    outcomes = [r for r in events if r['kind'] == 'OUTCOME']
    return {'contract': contract, 'events': events, 'head': previous,
            'submitted': len(admitted), 'dispatched': len(dispatched), 'started': len(started),
            'terminal': len(done), 'incomplete': sorted(set(admitted)-done),
            'outcome_counts': dict(Counter(r['payload']['status'] for r in outcomes)),
            'admission_rejections': dict(Counter(r['payload']['status'] for r in events
                                                 if r['kind'] == 'ADMISSION_REJECTED')),
            'execution_unconfirmed': [r['request_id'] for r in outcomes if not r['payload']['execution_stopped']]}


class ServiceJournal:
    def __init__(self, root, contract, max_events):
        p = Path(root).absolute()
        require('..' not in p.parts and p.parent.resolve(strict=True) == p.parent and not p.exists()
                and not p.is_symlink(), 'new external owner journal required')
        for ancestor in p.parents:
            require(not any((ancestor/n).exists() or (ancestor/n).is_symlink() for n in
                ('.git', 'manifest.json', 'qwen3_final_manifest.json', 'phase_codebook.manifest.json')),
                'journal inside source/artifact')
        p.mkdir(mode=0o700)
        self.root, self.contract, self.max_events = p, contract, max_events
        fd = os.open(p/'writer.lock', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._file = os.fdopen(fd, 'w')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self._mutex, self.events, self.head = Lock(), [], None
        publish_artifact(p/'contract', manifest_fields={'artifact_type': 'observer_dtrs_resident_contract',
            'schema_version': VERSION}, files={'record.json': canonical_json_bytes(contract)})

    def append(self, kind, *, epoch, request_id=None, token=None, payload=None):
        with self._mutex:
            require(len(self.events) < self.max_events, 'audit capacity exhausted')
            require(_read(self.root/'contract', 'contract') == self.contract, 'owner contract changed on disk')
            if self.events:
                require(_read(self.root/'events'/f'{len(self.events)-1:08d}', 'event') == self.events[-1],
                        'audit tail changed on disk')
            r = {'sequence': len(self.events), 'previous': self.head, 'contract_digest': digest(self.contract),
                 'kind': kind, 'request_id': request_id, 'epoch': epoch, 'token': token, 'payload': payload}
            publish_artifact(self.root/'events'/f'{len(self.events):08d}', manifest_fields={
                'artifact_type': 'observer_dtrs_resident_event', 'schema_version': VERSION},
                files={'record.json': canonical_json_bytes(r)})
            self.events.append(r);self.head = digest(r)
            return r

    def close(self):
        if not self._file.closed:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
            self._file.close()
