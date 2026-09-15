"""Shared JSONL wire contract; no model, env, Actor or planner imports."""
from dataclasses import asdict, dataclass
import json
import re
from werewolf.artifact_io import canonical_json_bytes, sha256_bytes

REMOTE_TOM_PROTOCOL_VERSION = "remote_tom_plan_v1"
PLANNER_BASELINE_COMMIT = "8886af9505036b55662d54ae9a108b09121e8c71"

class RemoteToMError(RuntimeError):
    pass

@dataclass(frozen=True)
class WorkerIdentity:
    planner_baseline_commit: str
    adapter_digest: str
    runtime_digest: str
    source_digest: str
    experiment_digest: str
    seal_digest: str
    checkpoint_digest: str
    condition: str

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == 'condition':
                if value not in ('implicit', 'explicit_day_phase'):
                    raise ValueError('unsupported temporal condition')
            elif type(value) is not str or re.fullmatch(
                    '[0-9a-f]{40}' if name == 'planner_baseline_commit' else '[0-9a-f]{64}', value) is None:
                raise ValueError(f'complete explicit worker identity required: {name}')


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RemoteToMError('duplicate JSON field')
        result[key] = value
    return result


def _invalid_constant(value):
    raise RemoteToMError('non-JSON numeric constant')


def decode_json_line(line):
    if type(line) is not str or not line.endswith('\n') or '\n' in line[:-1] or '\r' in line:
        raise RemoteToMError('expected exactly one newline-terminated JSON line')
    try:
        value = json.loads(line, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    except (ValueError, TypeError) as error:
        raise RemoteToMError('malformed JSON response') from error
    if type(value) is not dict:
        raise RemoteToMError('expected JSON object')
    return value


def encode_json_line(record):
    return canonical_json_bytes(record).decode('utf-8') + '\n'



def bind_request(record):
    record = dict(record)
    record['request_id'] = 'rtom-' + sha256_bytes(canonical_json_bytes(record))
    record['request_digest'] = sha256_bytes(canonical_json_bytes(record))
    return record


def validate_request(record, identity):
    fields = {'protocol_version', 'request_id', 'request_digest', 'game_id', 'boundary_id',
        'parent_digest', 'parent_pre', 'speaker', 'day', 'phase', 'candidates',
        'candidate_order_version', 'public_queue_rule_version', 'alive_wolves', 'condition',
        'expected_worker_identity'}
    if type(record) is not dict or set(record) != fields:
        raise RemoteToMError('request schema mismatch')
    body = {k: v for k, v in record.items() if k not in ('request_id', 'request_digest')}
    if bind_request(body) != record:
        raise RemoteToMError('request binding mismatch')
    if record['protocol_version'] != REMOTE_TOM_PROTOCOL_VERSION:
        raise RemoteToMError('protocol version mismatch')
    if (record['expected_worker_identity'] != asdict(identity)
            or identity.planner_baseline_commit != PLANNER_BASELINE_COMMIT
            or record['condition'] != identity.condition):
        raise RemoteToMError('worker identity mismatch')
    if type(record['day']) is not int or record['day'] < 0:
        raise RemoteToMError('invalid day')
    if record['phase'] not in ('discussion', 'pk_discussion'):
        raise RemoteToMError('invalid speech phase')
    if type(record['candidates']) is not list or not record['candidates']:
        raise RemoteToMError('invalid candidates')
    for plan in record['candidates']:
        if type(plan) is not dict or set(plan) != {'action', 'target'}:
            raise RemoteToMError('invalid candidate schema')
    wolves = record['alive_wolves']
    seats = tuple(f'player{i}' for i in range(1, 8))
    if (type(wolves) is not list or not wolves or any(p not in seats for p in wolves)
            or wolves != [p for p in seats if p in wolves]):
        raise RemoteToMError('noncanonical alive wolves')
    return record


def validate_response(response, request):
    status = response.get('status')
    if status == 'error':
        required = {'protocol_version', 'status', 'error_code', 'error_message'}
        if not required <= set(response) or set(response) - required - {'request_id', 'worker_identity'}:
            raise RemoteToMError('error response schema mismatch')
        if response['protocol_version'] != REMOTE_TOM_PROTOCOL_VERSION:
            raise RemoteToMError('error protocol mismatch')
        if 'request_id' in response and response['request_id'] != request['request_id']:
            raise RemoteToMError('error request mismatch')
        if 'worker_identity' in response and response['worker_identity'] != request['expected_worker_identity']:
            raise RemoteToMError('error identity mismatch')
        if any(type(response[k]) is not str or not response[k].strip() for k in ('error_code', 'error_message')):
            raise RemoteToMError('malformed worker error')
        raise RemoteToMError('worker reported execution failure')
    required = {'protocol_version', 'request_id', 'request_digest', 'status', 'worker_identity',
                'selected_candidate_index', 'selected_plan'}
    if status != 'ok' or set(response) != required:
        raise RemoteToMError('response schema mismatch')
    for key in ('protocol_version', 'request_id', 'request_digest'):
        if response[key] != request[key]:
            raise RemoteToMError(f'response binding mismatch: {key}')
    if response['worker_identity'] != request['expected_worker_identity']:
        raise RemoteToMError('worker identity mismatch')
    index = response['selected_candidate_index']
    if type(index) is not int or not 0 <= index < len(request['candidates']):
        raise RemoteToMError('selected index out of range')
    if response['selected_plan'] != request['candidates'][index]:
        raise RemoteToMError('selected plan/index mismatch')
    return index
