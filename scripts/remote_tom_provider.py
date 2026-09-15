"""Gameplay-side JSONL protocol client. No worker/model/evaluator imports."""
from dataclasses import asdict, dataclass
import json
import re
from typing import Protocol

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection.pre import validate_authoritative_pre_prefix
from werewolf.canonical_collection.public_history import PLAYER_IDS
from scripts.speech_versions import CANDIDATE_ORDER_VERSION, QUEUE_RULE_VERSION

REMOTE_TOM_PROTOCOL_VERSION = "remote_tom_plan_v1"


class RemoteToMError(RuntimeError):
    pass


class JsonLineTransport(Protocol):
    def request(self, line: str) -> str:
        """One JSON line in/out. Implementations MUST bound waits and raise on
        timeout, process exit/unavailability or framing error; never retry or
        reconnect. stdout is protocol-only, stderr separate. No worker here.
        """
        ...


@dataclass(frozen=True)
class WorkerIdentity:
    planner_baseline_commit: str
    worker_commit: str
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
                    '[0-9a-f]{40}' if name in ('planner_baseline_commit', 'worker_commit') else '[0-9a-f]{64}', value) is None:
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


def build_request(opportunity, *, alive_wolves, worker_identity):
    """Encode an already gated Phase 4A opportunity without new PRE semantics."""
    if not isinstance(worker_identity, WorkerIdentity):
        raise TypeError('expected complete WorkerIdentity')
    parent = validate_authoritative_pre_prefix(opportunity.parent)
    if opportunity.speaker != parent.current_speaker:
        raise ValueError('public opportunity speaker mismatch')
    if (type(alive_wolves) is not tuple or not alive_wolves
            or len(set(alive_wolves)) != len(alive_wolves)
            or any(p not in parent.alive_observer_ids for p in alive_wolves)
            or parent.current_speaker not in alive_wolves
            or set(alive_wolves) == set(parent.alive_observer_ids)):
        raise ValueError('invalid alive wolf objective population')
    if type(opportunity.candidates) is not tuple or not opportunity.candidates:
        raise ValueError('expected nonempty canonical candidate tuple')
    candidates = [p.public_payload() for p in opportunity.candidates]
    if len({canonical_json_bytes(p) for p in candidates}) != len(candidates):
        raise ValueError('duplicate candidate')
    record = {
        'protocol_version': REMOTE_TOM_PROTOCOL_VERSION,
        'game_id': parent.game_id, 'boundary_id': parent.boundary_id,
        'parent_digest': parent.prefix_digest, 'parent_pre': parent.to_record(),
        'speaker': opportunity.speaker, 'day': parent.public_temporal_state.day,
        'phase': parent.public_temporal_state.phase.value,
        'candidates': candidates, 'candidate_order_version': CANDIDATE_ORDER_VERSION,
        'public_queue_rule_version': QUEUE_RULE_VERSION,
        'alive_wolves': [p for p in PLAYER_IDS if p in alive_wolves],
        'condition': worker_identity.condition, 'expected_worker_identity': asdict(worker_identity),
    }
    # Both bindings are planner-private: small wolf-seat sets are enumerable.
    # Never use them as public/Actor/agent-visible trace correlation IDs.
    record['request_id'] = 'rtom-' + sha256_bytes(canonical_json_bytes(record))
    record['request_digest'] = sha256_bytes(canonical_json_bytes(record))
    return record


def validate_response(response, request):
    status = response.get('status')
    shared = {'protocol_version', 'request_id', 'request_digest', 'status', 'worker_identity'}
    required = shared | ({'selected_index', 'selected_plan'} if status == 'ok'
                         else {'error_code', 'error_message'})
    if status not in ('ok', 'error') or set(response) != required:
        raise RemoteToMError('response schema mismatch')
    for key in ('protocol_version', 'request_id', 'request_digest'):
        if response[key] != request[key]:
            raise RemoteToMError(f'response binding mismatch: {key}')
    if response['worker_identity'] != request['expected_worker_identity']:
        raise RemoteToMError('worker identity mismatch')
    if status == 'error':
        if any(type(response[k]) is not str or not response[k].strip() for k in ('error_code', 'error_message')):
            raise RemoteToMError('malformed worker error')
        # Do not reflect worker messages/private diagnostics into public logs.
        raise RemoteToMError('worker reported execution failure')
    index = response['selected_index']
    if type(index) is not int or not 0 <= index < len(request['candidates']):
        raise RemoteToMError('selected index out of range')
    if response['selected_plan'] != request['candidates'][index]:
        raise RemoteToMError('selected plan/index mismatch')
    return index


class RemoteToMPlanProvider:
    """Private objective supplier receives only the public opportunity.

    alive_wolves_for(opportunity) is an explicit private configuration seam,
    never a role table or env/Actor/recorder passed to the provider. It must
    supply the lawful current alive wolf IDs. Caller owns transport lifecycle.
    """
    def __init__(self, *, transport: JsonLineTransport, worker_identity: WorkerIdentity, alive_wolves_for):
        if not isinstance(worker_identity, WorkerIdentity):
            raise TypeError('expected complete WorkerIdentity')
        if not callable(alive_wolves_for) or not callable(getattr(transport, 'request', None)):
            raise TypeError('transport and objective-state supplier are required')
        self.transport = transport
        self.worker_identity = worker_identity
        self.alive_wolves_for = alive_wolves_for

    def __call__(self, opportunity):
        request = build_request(opportunity, alive_wolves=self.alive_wolves_for(opportunity),
                                worker_identity=self.worker_identity)
        try:
            response = decode_json_line(self.transport.request(encode_json_line(request)))
            index = validate_response(response, request)
        except RemoteToMError:
            raise
        except Exception as error:
            raise RemoteToMError('remote transport failed') from error
        return opportunity.candidates[index]
