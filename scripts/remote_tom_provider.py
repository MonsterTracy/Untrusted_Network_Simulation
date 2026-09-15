"""Gameplay-side provider; all shared wire semantics live in the protocol."""
from dataclasses import asdict
from typing import Protocol
from werewolf.artifact_io import canonical_json_bytes
from werewolf.canonical_collection.pre import validate_authoritative_pre_prefix
from werewolf.canonical_collection.public_history import PLAYER_IDS
from scripts.speech_versions import CANDIDATE_ORDER_VERSION, QUEUE_RULE_VERSION
from scripts.remote_tom_protocol import (
    WorkerIdentity, RemoteToMError, REMOTE_TOM_PROTOCOL_VERSION,
    decode_json_line, encode_json_line, bind_request, validate_response,
)

class JsonLineTransport(Protocol):
    def request(self, line: str) -> str:
        """One JSON line in/out. Implementations MUST bound waits and raise on
        timeout, process exit/unavailability or framing error; never retry or
        reconnect. stdout is protocol-only, stderr separate. No worker here.
        """
        ...


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
    return bind_request(record)


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
