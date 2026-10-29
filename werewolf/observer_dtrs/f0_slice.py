"""Controlled single-attempt F0 slice; no production execution qualification.

Uses the original runtime, recorder, realization and commit. This module does
not supply global ownership, external readback or independent anchor custody.
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import threading

from scripts.speech_planning import SpeechPlan, PlanningAction, generate_candidates, CANDIDATE_ORDER_VERSION
from scripts.speech_realization import expected_action
from scripts.verified_speech_commit import (
    PreCommitAudit, AuditAcknowledgement, capture_precommit_binding, realize_and_commit,
)
from werewolf.artifact_io import canonical_json_bytes, publish_artifact, sha256_bytes
from werewolf.canonical_collection.attempt_ledger import (
    construct_attempt_claim, initialize_attempt_ledger, publish_ledger_record, validate_attempt_ledger,
)
from werewolf.canonical_collection.frozen_json import FrozenJSONValue, freeze_json_object
from werewolf.canonical_collection.public_history import freeze_public_event_history
from werewolf.observer_dtrs.action_evidence_ledger import digest
from werewolf.observer_dtrs.action_evidence_store import ActionEvidenceStore, MAX_BYTES

SCOPE = 'CONTROLLED_CPU_ONLY_NOT_REAL_F0_AUTHORIZATION'
CONTRACT_DIRECTORY = 'f0-slice-contract'


def _require(ok, message):
    if not ok:
        raise ValueError(message)


@dataclass(frozen=True)
class F0SliceConfig:
    study_id: str
    qualification_id: str
    plan: SpeechPlan
    eligible_phases: tuple[str, ...] = ('discussion', 'pk_discussion')
    scope: str = SCOPE

    def __post_init__(self):
        _require(all(type(s) is str and s.strip() for s in (self.study_id, self.qualification_id)),
                 'explicit study and qualification identities required')
        _require(type(self.plan) is SpeechPlan and self.plan.action in (
            PlanningAction.SELF_DEFEND, PlanningAction.NO_COMMITMENT), 'one fixed targetless F0 plan required')
        _require(type(self.eligible_phases) is tuple and bool(self.eligible_phases)
                 and len(set(self.eligible_phases)) == len(self.eligible_phases)
                 and set(self.eligible_phases) <= {'discussion', 'pk_discussion'}, 'invalid declared F0 phase population')
        _require(self.scope == SCOPE, 'real F0 blocked: G3/G4/G5 and source/service approval required')

    def to_record(self):
        return {'study_id': self.study_id, 'qualification_id': self.qualification_id,
                'fixed_plan': self.plan.public_payload(), 'eligible_phases': list(self.eligible_phases),
                'scope': self.scope, 'ToM_D2_D3': 'NOT_USED'}


@dataclass(frozen=True)
class F0SliceProduct:
    """A partial execution result, NEVER CanonicalGameProduct or a winner."""
    evidence: FrozenJSONValue
    artifact_path: str | None = None
    manifest_digest: str | None = None


class F0SliceSealError(RuntimeError):
    """Preserve captured evidence; do not re-execute after a publication failure."""
    def __init__(self, product):
        super().__init__('F0 slice persistence unconfirmed; stop without retry or fallback')
        self.product = product


class F0SliceControl:
    """One cooperating process/thread, single use. Not a native capability fence."""
    def __init__(self, config):
        _require(type(config) is F0SliceConfig, 'explicit controlled F0 config required')
        self._config = config
        self._pid, self._thread = os.getpid(), threading.get_ident()
        self._used = self._bound = self._assigned = self._closed = False
        self._store = self._packet = self._target = None
        self._ack_returned = False
        self._running = False
        self._native_return = 'NOT_REACHED'
        self._assignment_ack = self._intent_ack = None

    @property
    def config(self):
        return self._config

    def require_bound(self, recorder):
        self._owner()
        _require(self._bound and not self._closed and recorder is self._recorder,
                 'unbound/repeated F0 runtime')

    def begin_runtime(self, recorder):
        self.require_bound(recorder)
        _require(not self._running, 'unbound/repeated F0 runtime')
        self._running = True

    def _owner(self):
        _require(os.getpid() == self._pid and threading.get_ident() == self._thread,
                 'F0 control belongs to another cooperating process/thread')

    def _start(self):
        self._owner()
        _require(not self._used, 'F0 control already consumed; no retry/resume')
        self._used = True

    def bind(self, recorder, destination):
        self._owner()
        _require(self._used and not self._bound and not self._closed, 'F0 attempt cannot rebind')
        _require(not recorder.prefixes and not recorder.submitted_gameplay_actions
                 and recorder._env is None, 'F0 must bind before the original Recorder start')
        self._bound = True
        self._recorder = recorder
        self._store = ActionEvidenceStore.create(Path(destination)/'action-journal',
            Path(destination)/'controlled-anchor', ledger_id=self.config.qualification_id,
            game_id=recorder.game_id, game_attempt_id=recorder.claim.attempt_id)

    def handle_pre(self, *, env, recorder, call_audit, observation, handoff):
        self._owner()
        _require(self._bound and not self._closed and recorder is self._recorder,
                 'wrong/closed F0 runtime')
        _require(not self._assigned, 'fixed assignment already consumed')
        pre = recorder.public_pre_for_handoff(handoff)
        _require(handoff.observer_id == f"player{observation['current_act_idx']}", 'actor/handoff mismatch')
        if (observation.get('identity') != 'Werewolf'
                or observation['current_act_idx'] not in observation['authoritative_public_state']['alive_players']
                or pre.public_temporal_state.phase.value not in self.config.eligible_phases):
            return None  # ONLY unassigned non-target opportunities use the original Agent.
        self._assigned = True  # Any failure below is terminal, never another opportunity.
        self._target = (pre, handoff, len(recorder.submitted_gameplay_actions), len(env.public_events))
        payload = self.config.plan.public_payload()
        candidates = [{'plan': p.public_payload(),
                       'expected_v1': expected_action(p.public_payload(), pre.current_speaker).to_record()}
                      for p in generate_candidates(pre)]
        selected = next(c for c in candidates if c['plan'] == payload)
        op = {'opportunity_id': pre.boundary_id, 'protocol_ref': digest(self.config.to_record()),
              'game_id': pre.game_id, 'game_attempt_id': recorder.claim.attempt_id,
              'boundary_id': pre.boundary_id, 'prefix_digest': pre.prefix_digest,
              'step_index': pre.step_index, 'actor': pre.current_speaker,
              'day': pre.public_temporal_state.day, 'phase': pre.public_temporal_state.phase.value,
              'public_history_digest': pre.public_event_digest, 'capture_digest': digest(pre.to_record()),
              'candidate_order_version': CANDIDATE_ORDER_VERSION, 'candidates': candidates}
        op_ref = self._store.append('opportunity', 'OPPORTUNITY', op).record_digest
        self._assignment_ack = self._store.append('assignment', 'ASSIGNMENT', {
            'assignment_id': self.config.qualification_id, 'opportunity_ref': op_ref,
            'candidate_id': digest(selected), 'support': [digest(selected)], 'declared_probabilities': None,
            'treatment_ref': digest(payload), 'assignment_source': 'INJECTED_FIXED_NOT_RANDOMIZED'})
        self._assignment = self._assignment_ack.record_digest
        self._started = self._store.append('started', 'EXECUTION', {'assignment_ref': self._assignment,
            'status': 'STARTED', 'execution_id': self.config.qualification_id,
            'implementation_ref': sha256_bytes(Path(realize_and_commit.__code__.co_filename).read_bytes())}).record_digest
        binding = capture_precommit_binding(pre, attempt_id=recorder.claim.attempt_id,
            handoff_digest=handoff.handoff_digest, ledger_id=self.config.qualification_id,
            assignment_ref=self._assignment, owner_epoch=f'cooperating-pid-{self._pid}-thread-{self._thread}',
            payload=payload)
        return realize_and_commit(payload, env=env, actor=self._agents[observation['current_act_idx']-1],
            recorder=recorder, precommit_audit=PreCommitAudit(binding, self._observe))

    def _observe(self, packet):
        self._owner()
        _require(self._packet is None, 'F0 intent cannot repeat')
        self._packet = packet
        envelope = json.loads(packet.envelope_snapshot)
        execution = self._store.append('verified', 'EXECUTION', {'assignment_ref': self._assignment,
            'status': 'VERIFIED', 'started_ref': self._started, 'evidence_ref': packet.digest,
            'speech_digest': sha256_bytes(envelope['speech'].encode()), 'actions': [envelope['expected']]}).record_digest
        receipt = self._store.append('intent', 'COMMIT_WITNESS', {'assignment_ref': self._assignment,
            'status': 'ATTEMPT_RESERVED', 'execution_ref': execution,
            'attempt_token': self.config.qualification_id, 'evidence_ref': packet.digest})
        self._intent_ack = receipt
        ack = AuditAcknowledgement(receipt, canonical_json_bytes(self._store.records))
        self._ack_returned = True
        return ack

    def boundary_result(self, *, env, recorder, native_result):
        self._owner()
        _require(self._assigned and not self._closed and recorder is self._recorder, 'invalid F0 stop')
        self._native_return = 'RETURNED_4_TUPLE'
        return self.capture(env=env, recorder=recorder, error=None)

    def capture(self, *, env, recorder, error):
        """Read original evidence only. Unknown attribution/cessation stay unknown."""
        self._owner()
        self._closed = True
        history = freeze_public_event_history(env.public_events)
        actions = [a.to_record() for a in recorder.submitted_gameplay_actions]
        target, native, event, annotation = None, None, None, None
        matches, events, annotations, conflicts = [], [], [], []
        if self._target:
            pre, handoff, before_actions, before_events = self._target
            target = {'pre': pre.to_record(), 'handoff': handoff.to_record(),
                      'before_action_count': before_actions, 'before_event_count': before_events}
            records = history.to_records()
            if records[:before_events] != pre.public_event_history.to_records():
                conflicts.append('PRE_PUBLIC_HISTORY_CHANGED')
            matches = [a.to_record() for a in recorder.submitted_gameplay_actions
                       if a.boundary_id == pre.boundary_id]
            # Event/V1 observation is independent of Recorder publication. The env
            # may have published before a Recorder update/return failed.
            events = [e for e in records[before_events:] if e['event_type'] == 'public_speech']
            event_ids = {e['event_id'] for e in events}
            annotations = [a.to_record() for a in env.speech_annotations if a.event_id in event_ids]
            if len(matches) > 1 or len(events) > 1 or len(annotations) > 1:
                conflicts.append('MULTIPLE_TARGET_NATIVE_EVIDENCE')
            if len(matches) == 1:
                native = matches[0]
                if (native['speaker_pre_belief_handoff'] != handoff.to_record()
                        or native['actor_id'] != pre.current_speaker or native['step_index'] != pre.step_index
                        or native['action_type'] != 'public_speech'):
                    conflicts.append('NATIVE_PRE_IDENTITY_MISMATCH')
            if len(events) == 1:
                event = events[0]
                if (event['speaker'] != pre.current_speaker or event['day'] != pre.public_temporal_state.day
                        or event['phase'] != pre.public_temporal_state.phase.value):
                    conflicts.append('PUBLIC_PRE_IDENTITY_MISMATCH')
            if len(annotations) == 1:
                annotation = annotations[0]
                if annotation['actions'] != [expected_action(self.config.plan.public_payload(), pre.current_speaker).to_record()]:
                    conflicts.append('V1_NOT_EXACT_FIXED_PLAN')
            if native is not None and event is not None:
                if (native['resulting_public_event_ids'] != [event['event_id']]
                        or native['action_payload'] != event['raw_text']):
                    conflicts.append('NATIVE_EVENT_MISMATCH')
        complete_local = native is not None and event is not None and annotation is not None
        status = ('STOPPED_CONFLICT' if conflicts else
                  'BOUNDARY_RETURNED' if error is None and complete_local else
                  'STOPPED_FAILURE' if error is not None else
                  'STOPPED_UNCONFIRMED' if self._assigned else 'NO_ELIGIBLE_PRE')
        # Snapshot EXISTING custody bytes, not a second mutable journal or a
        # recovered/acknowledged state. Partial writes/ACK loss remain unknown.
        custody, custody_errors = {}, {}
        if self._store is not None:
            for name, path in (('contract', self._store.root/'contract.json'),
                               ('journal', self._store.root/'events.jsonl'),
                               ('anchor', self._store.anchor_root/'anchors.jsonl')):
                try:
                    with path.open('rb') as handle:
                        raw = handle.read(MAX_BYTES + 1)
                    _require(len(raw) <= MAX_BYTES, 'custody snapshot size limit')
                    custody[name] = {'bytes_hex': raw.hex(), 'sha256': sha256_bytes(raw), 'byte_size': len(raw)}
                except (OSError, ValueError) as snapshot_error:
                    custody_errors[name] = type(snapshot_error).__name__
        result = {'schema_version': 'controlled_f0_slice_v1', 'artifact_kind': 'PARTIAL_NOT_CANONICAL_GAME',
            **self.config.to_record(), 'slice_status': status, 'game_id': recorder.game_id,
            'claim': recorder.claim.to_record(), 'target': target,
            'assignment_ref': getattr(self, '_assignment', None),
            'target_locked': self._assigned,
            'assignment_persistence': ('ACKNOWLEDGED_LOCAL' if self._assignment_ack is not None
                                       else 'UNKNOWN_AFTER_TARGET_LOCK' if self._assigned else 'NOT_ASSIGNED'),
            'custody_bytes': custody, 'custody_snapshot_errors': custody_errors,
            'assignment_ack': None if self._assignment_ack is None else self._assignment_ack.__dict__,
            'intent_ack': None if self._intent_ack is None else self._intent_ack.__dict__,
            'audit_evidence': None if self._packet is None else self._packet.to_record(),
            'native_return': self._native_return if error is None else ('UNKNOWN_AFTER_INTENT' if self._ack_returned else 'NOT_REACHED'),
            'native_occurrence': 'CONFLICT' if conflicts else 'OBSERVED_LOCAL' if native is not None or event is not None else 'UNKNOWN',
            'native_action_status': 'CONFLICT' if len(matches) > 1 else 'OBSERVED_LOCAL' if native is not None else 'UNKNOWN',
            'speech_event_status': 'CONFLICT' if len(events) > 1 else 'OBSERVED_LOCAL' if event is not None else 'UNKNOWN',
            'v1_status': 'CONFLICT' if len(annotations) > 1 else 'OBSERVED_LOCAL' if annotation is not None else 'UNKNOWN',
            'local_evidence_conflicts': conflicts,
            'target_native_actions': matches, 'target_public_speech_events': events,
            'target_v1_annotations': annotations,
            'assignment_native_attribution': 'UNKNOWN_NATIVE_SCHEMA_HAS_NO_ASSIGNMENT_TOKEN',
            'external_attempt_readback': 'UNKNOWN', 'independent_high_water': 'NOT_QUALIFIED',
            'control_stop': 'CURRENT_CALL_STACK_RETURNING', 'all_native_capable_executors_stopped': 'UNKNOWN',
            'owner': {'pid': self._pid, 'thread_id': self._thread, 'scope': 'COOPERATING_ONLY_NOT_FENCED'},
            'native_action': native, 'public_speech_event': event, 'v1_annotation': annotation,
            'public_event_history': history.to_records(), 'prefixes': [v.to_record() for v in recorder.prefixes],
            'submitted_gameplay_actions': actions, 'v1_annotations': [a.to_record() for a in env.speech_annotations],
            'backend_call_ids': [r.call_id for r in recorder.call_audit.records],
            'call_budget': recorder.call_audit.summary().to_record(),
            'error_category': None if error is None else type(error).__name__,
            'underlying_error_category': None if error is None else getattr(error, 'error_category', type(error).__name__),
            'plan_source_revision': recorder.plan.source_revision,
            'control_source_sha256': sha256_bytes(Path(__file__).read_bytes()),
            'source_qualification': 'NOT_FORMALLY_FIXED', 'winner': None, 'execution_permission': False}
        return F0SliceProduct(freeze_json_object(result, 'F0 restricted partial evidence'))


def run_f0_attempt(*, factory, plan, destination, attempt_id, timestamp_utc):
    """Exactly one NEW controlled attempt, no collect(), recovery, loop or F1.

    Real services are NOT qualified by this entry; the only accepted config scope
    is controlled CPU. A claim intentionally stays open: no fabricated full-game
    terminal is published. Ordinary collect refuses this directory and factory.
    """
    control = factory.f0_control
    _require(type(control) is F0SliceControl, 'explicit F0-only factory required')
    control._start()
    destination = Path(destination)
    destination.mkdir(mode=0o700)  # Exclusive new namespace; no resume/overwrite.
    publish_artifact(destination/CONTRACT_DIRECTORY, manifest_fields={
        'artifact_type': 'controlled_f0_contract', 'schema_version': 'controlled_f0_contract_v1'},
        files={'contract.json': canonical_json_bytes(control.config.to_record())})
    ledger = initialize_attempt_ledger(destination, plan)
    _require(not validate_attempt_ledger(ledger, plan).claims, 'F0 requires an unclaimed native attempt')
    claim = construct_attempt_claim(plan, ordinal=0, attempt_id=attempt_id, claim_timestamp_utc=timestamp_utc)
    publish_ledger_record(ledger, plan, claim)
    runtime = None
    try:
        runtime = factory(plan=plan, claim=claim)
        _require(runtime.recorder.claim == claim and runtime.recorder.plan == plan,
                 'native plan/claim mismatch before lifecycle')
        control._agents = runtime.agents
        control.bind(runtime.recorder, destination)
        product = runtime.run_f0_slice()
    finally:
        if control._store is not None:
            control._store.close()
    try:
        artifact = publish_artifact(destination/'slice', manifest_fields={
            'artifact_type': 'controlled_f0_slice', 'schema_version': 'controlled_f0_slice_v1',
            'claim_digest': claim.record_digest, 'scope': SCOPE},
            files={'restricted-slice.json': product.evidence.canonical_bytes})
    except Exception as error:
        raise F0SliceSealError(product) from error
    return F0SliceProduct(product.evidence, str(artifact.path), artifact.manifest_digest)
