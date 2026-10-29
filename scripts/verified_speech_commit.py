"""Phase 3B orchestration at an already collected PRE; no planner imports."""
from scripts.speech_realization import PublicRealizationContext, realize_parse_verify
from werewolf.agents.prompt_template_v0 import build_public_claim_catalog
from werewolf.canonical_collection.public_history import freeze_public_event_history
from werewolf.speech.verified_commit import bind_verified_speech


from dataclasses import dataclass
import json
import os
import threading

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection.frozen_json import FrozenJSONValue, freeze_json_object


class PreCommitAuditError(ValueError):
    """Audit not confirmed. Never means native submission was absent."""


def _audit_require(ok, message):
    if not ok:
        raise PreCommitAuditError(message)


def _digest(value):
    return sha256_bytes(canonical_json_bytes(value))


def capture_precommit_binding(pre, *, attempt_id, handoff_digest, ledger_id,
                              assignment_ref, owner_epoch, payload):
    """Immutable public identity only; no authority/role/report is inferred."""
    for value in (attempt_id, ledger_id, owner_epoch):
        _audit_require(type(value) is str and bool(value.strip()), 'nonempty audit identity required')
    for value in (assignment_ref, handoff_digest):
        _audit_require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
                       'canonical audit digest required')
    return freeze_json_object({
        'scope': 'CONTROLLED_CPU_AUDIT_NOT_REAL_EXECUTION_AUTHORITY',
        'ledger_id': ledger_id, 'assignment_ref': assignment_ref, 'owner_epoch': owner_epoch,
        'game_id': pre.game_id, 'game_attempt_id': attempt_id, 'boundary_id': pre.boundary_id,
        'prefix_digest': pre.prefix_digest, 'capture_digest': _digest(pre.to_record()),
        'public_history_digest': pre.public_event_digest, 'handoff_digest': handoff_digest,
        'actor': pre.current_speaker, 'day': pre.public_temporal_state.day,
        'phase': pre.public_temporal_state.phase.value, 'step_index': pre.step_index,
        'payload_digest': _digest(payload)}, 'precommit binding')


@dataclass(frozen=True)
class PreCommitEvidence:
    binding_json: bytes
    envelope_snapshot: bytes
    envelope_digest: str

    def to_record(self):
        return {'binding': json.loads(self.binding_json),
                'envelope': json.loads(self.envelope_snapshot), 'envelope_digest': self.envelope_digest}

    @property
    def digest(self):
        return _digest(self.to_record())


@dataclass(frozen=True)
class AuditAcknowledgement:
    """Writer acknowledgement + exact Phase-2D chain. NOT an execution permit."""
    receipt: object  # Runtime checked against the existing DurableReceipt, lazily.
    records_json: bytes


class PreCommitAudit:
    """One-shot local audit invocation, not a game/assignment permission owner.

    Caller retains responsibility for one authorized executor per assignment.
    Recreating this object or fabricating a receipt does not establish ownership
    or actual persistence. No CLI/game hook enables this controlled interface.
    """
    def __init__(self, binding, observe):
        if type(binding) is not FrozenJSONValue or not callable(observe):
            raise TypeError('immutable binding and explicit audit callable required')
        self._binding, self._observe = binding, observe
        self._pid, self._thread = os.getpid(), threading.get_ident()
        self._claimed, self._called = False, False

    @property
    def binding(self):
        return self._binding

    def _owner(self):
        _audit_require(os.getpid() == self._pid and threading.get_ident() == self._thread,
                       'audit object belongs to another process/thread')

    def _claim(self, actual):
        self._owner()
        _audit_require(not self._claimed, 'audit object already consumed; no retry/fallback')
        self._claimed = True  # Failure is terminal for THIS invocation too.
        _audit_require(actual.canonical_bytes == self.binding.canonical_bytes, 'precommit binding mismatch')

    def _confirm(self, evidence):
        self._owner()
        _audit_require(self._claimed and not self._called, 'audit callback cannot repeat')
        self._called = True
        # Only immutable bytes/scalars cross this call. No env/recorder/Agent.
        original = (evidence.binding_json, evidence.envelope_snapshot, evidence.envelope_digest)
        ack = self._observe(evidence)
        _audit_require(original == (evidence.binding_json, evidence.envelope_snapshot, evidence.envelope_digest),
                       'immutable audit evidence was changed')
        if type(ack) is not AuditAcknowledgement or type(ack.records_json) is not bytes:
            raise PreCommitAuditError('intent persistence acknowledgement required')
        # Keep the default path independent of experimental store imports.
        from werewolf.observer_dtrs.action_evidence_store import DurableReceipt
        from werewolf.observer_dtrs.action_evidence_ledger import validate_ledger
        receipt = ack.receipt
        _audit_require(type(receipt) is DurableReceipt and receipt.execution_permission is False,
                       'original record-only DurableReceipt required, never a permit')
        rows = json.loads(ack.records_json)
        _audit_require(type(rows) is list and bool(rows) and canonical_json_bytes(rows) == ack.records_json,
                       'exact canonical ledger snapshot required')
        b = json.loads(evidence.binding_json)
        validate_ledger(rows, ledger_id=b['ledger_id'], expected_head=receipt.record_digest)
        last = rows[-1]
        _audit_require(last['kind'] == 'COMMIT_WITNESS' and last['payload']['status'] == 'ATTEMPT_RESERVED'
                       and receipt.operation_id == last['operation_id']
                       and type(receipt.sequence) is int and receipt.sequence == last['sequence']
                       and type(receipt.anchor_digest) is str and len(receipt.anchor_digest) == 64
                       and all(c in '0123456789abcdef' for c in receipt.anchor_digest),
                       'acknowledgement does not identify this reserved intent')
        p = last['payload']
        _audit_require(p['assignment_ref'] == b['assignment_ref'] and p['evidence_ref'] == evidence.digest,
                       'intent/evidence digest mismatch')
        by_ref = {r['record_digest']: r for r in rows}
        assigned = by_ref[b['assignment_ref']]
        op = by_ref[assigned['payload']['opportunity_ref']]['payload']
        _audit_require(assigned['kind'] == 'ASSIGNMENT' and all(op[k] == b[k] and type(op[k]) is type(b[k])
                       for k in ('game_id','game_attempt_id','boundary_id','prefix_digest','capture_digest',
                                 'public_history_digest','actor','day','phase','step_index')),
                       'assignment/opportunity differs from live PRE')
        candidate = next(c for c in op['candidates'] if _digest(c) == assigned['payload']['candidate_id'])
        envelope = json.loads(evidence.envelope_snapshot)
        _audit_require(_digest(candidate['plan']) == b['payload_digest']
                       and candidate['expected_v1'] == envelope['expected'], 'assigned plan differs from envelope')
        execution = by_ref[p['execution_ref']]
        _audit_require(execution['kind'] == 'EXECUTION' and execution['payload']['status'] == 'VERIFIED'
                       and execution['payload']['assignment_ref'] == b['assignment_ref']
                       and execution['payload']['speech_digest'] == sha256_bytes(envelope['speech'].encode('utf-8'))
                       and execution['payload']['actions'] == [envelope['expected']],
                       'VERIFIED execution differs from original envelope')


def _live_audit_binding(payload, env, recorder, audit):
    pending = recorder._pending
    _audit_require(pending is not None and pending.handoff is not None, 'pending PRE was consumed or lost')
    pre = recorder.public_pre_for_handoff(pending.handoff)
    b = audit.binding.to_value()
    _audit_require(recorder._env is env and env.phase in ('speech', 'speech_pk')
                   and pre.current_speaker == f'player{env.current_act_idx + 1}'
                   and pre.public_temporal_state.day == env.day
                   and pre.public_temporal_state.phase.value == ('discussion' if env.phase == 'speech' else 'pk_discussion')
                   and freeze_public_event_history(env.public_events).digest == pre.public_event_digest,
                   'live PRE changed during audit')
    return capture_precommit_binding(pre, attempt_id=recorder.claim.attempt_id,
        handoff_digest=pending.handoff.handoff_digest, ledger_id=b['ledger_id'],
        assignment_ref=b['assignment_ref'], owner_epoch=b['owner_epoch'], payload=payload)


def realize_and_commit(payload, *, env, actor, recorder, precommit_audit=None):
    """Reuse original realization/parsing, then the original native commit.

    before_agent_act must already have collected the real PRE/handoff. This
    function never creates fake belief observations or changes their population.
    On semantic failure that existing pending slot remains unchanged.
    Optional audit is CONTROLLED CPU ONLY, default off. It confirms records,
    never supplies authorization. All exceptions propagate; no retry/fallback.
    """
    pending = recorder._pending
    if (recorder._env is not env or pending is None or pending.handoff is None
            or pending.prefix is None or pending.raw_action is not None
            or env.phase not in ('speech', 'speech_pk')):
        raise ValueError("constrained realization requires an existing pending speech PRE")
    history_digest = freeze_public_event_history(env.public_events).digest
    speaker = f"player{env.current_act_idx + 1}"
    if (pending.actor_id != speaker or pending.event_count_before != len(env.public_events)
            or pending.prefix.public_event_history.digest != history_digest):
        raise ValueError("pending PRE does not match current opportunity")
    if precommit_audit is not None:
        if type(precommit_audit) is not PreCommitAudit:
            raise TypeError('explicit one-shot PreCommitAudit required')
        audit_binding = _live_audit_binding(payload, env, recorder, precommit_audit)
        precommit_audit._claim(audit_binding)
        # Candidate eligibility is a PRE precondition, not a language outcome.
        from scripts.speech_planning import generate_candidates
        _audit_require(type(payload) is dict and payload in tuple(
            plan.public_payload() for plan in generate_candidates(pending.prefix)),
            'audited plan is not a legal candidate at the current PRE')
    context = PublicRealizationContext(speaker, env.day,
        'discussion' if env.phase == 'speech' else 'pk_discussion',
        build_public_claim_catalog(env.get_observation()))
    realized = realize_parse_verify(payload, context, actor=actor, perceiver=env.speech_perceiver,
        actor_context=recorder.call_audit.action_context(acting_player_id=env.current_act_idx + 1,
            boundary_id=pending.handoff.boundary_id, is_public_speech=True),
        perception_context=recorder.call_audit.speech_perception_context(
            event_id=f"event-{len(env.public_events):06d}",
            boundary_id=pending.handoff.boundary_id, speaker_id=env.current_act_idx + 1))
    envelope = bind_verified_speech(speech=realized.speech, expected=realized.expected,
        perception=realized.perception, day=context.day, phase=env.phase,
        public_history_digest=history_digest, perceiver=env.speech_perceiver)
    if precommit_audit is not None:
        envelope.validated_perception()
        evidence = PreCommitEvidence(audit_binding.canonical_bytes, envelope.snapshot, envelope.digest)
        precommit_audit._confirm(evidence)
        _audit_require(_live_audit_binding(payload, env, recorder, precommit_audit).canonical_bytes
                       == audit_binding.canonical_bytes, 'live binding changed after audit')
        envelope.validated_perception()
    result = recorder.commit_verified_speech(env, envelope)
    if precommit_audit is not None:
        _audit_require(isinstance(result, tuple) and len(result) == 4,
                       'native commit result is unconfirmed; stop without retry or fallback')
    return result
