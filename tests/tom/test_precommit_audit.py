"""Controlled partial-game transitions; scripted existing Actor/Parser only.

No true model/service, random assignment, full game or experiment approval.
"""
from copy import deepcopy
from dataclasses import replace, FrozenInstanceError, fields
import json
import random
import threading
from unittest.mock import Mock

import numpy as np
import pytest

from scripts.speech_planning import generate_candidates, SpeechPlan, PlanningAction as A
from scripts.speech_realization import expected_action
from scripts import verified_speech_commit as module
from scripts.verified_speech_commit import (
    realize_and_commit, capture_precommit_binding, PreCommitAudit,
    PreCommitAuditError, PreCommitEvidence, AuditAcknowledgement,
)
from tests.tom.test_verified_speech_commit import prepare, pending, snapshot, actor_for, response_for
from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection.frozen_json import freeze_json_object
from werewolf.observer_dtrs.action_evidence_store import ActionEvidenceStore
from werewolf.observer_dtrs.action_evidence_ledger import digest


class AuditOwnerFixture:
    """TEST record custodian, not a game committer or production permission issuer."""
    def __init__(self, path, pre, recorder, payload):
        self.store = ActionEvidenceStore.create(path/'journal',path/'anchor',ledger_id='controlled-precommit',
            game_id=pre.game_id,game_attempt_id=recorder.claim.attempt_id)
        self.calls = 0
        self.evidence = None
        candidates=[{'plan':p.public_payload(),'expected_v1':expected_action(p.public_payload(),pre.current_speaker).to_record()}
                    for p in generate_candidates(pre)]
        op={'opportunity_id':'op','protocol_ref':digest('CONTROLLED_NOT_APPROVED_FOR_GAMEPLAY'),
            'game_id':pre.game_id,'game_attempt_id':recorder.claim.attempt_id,'boundary_id':pre.boundary_id,
            'prefix_digest':pre.prefix_digest,'step_index':pre.step_index,'actor':pre.current_speaker,
            'day':pre.public_temporal_state.day,'phase':pre.public_temporal_state.phase.value,
            'public_history_digest':pre.public_event_digest,'capture_digest':digest(pre.to_record()),
            'candidate_order_version':'original-generator-fixture','candidates':candidates}
        op_ref=self.store.append('op','OPPORTUNITY',op).record_digest
        candidate=next(c for c in candidates if c['plan']==payload)
        self.assignment_ref=self.store.append('assignment','ASSIGNMENT',{'assignment_id':'assignment',
            'opportunity_ref':op_ref,'candidate_id':digest(candidate),'support':[digest(candidate)],
            'declared_probabilities':None,'treatment_ref':digest('FIXED_INJECTED_TEST_PLAN'),
            'assignment_source':'INJECTED_FIXED_NOT_RANDOMIZED'}).record_digest
        self.started_ref=self.store.append('start','EXECUTION',{'assignment_ref':self.assignment_ref,
            'status':'STARTED','execution_id':'controlled','implementation_ref':digest('scripted-original-Actor-Parser')}).record_digest
        self.binding=capture_precommit_binding(pre,attempt_id=recorder.claim.attempt_id,
            handoff_digest=recorder._pending.handoff.handoff_digest,ledger_id='controlled-precommit',
            assignment_ref=self.assignment_ref,owner_epoch='CONTROLLED_EPOCH_NOT_A_REAL_PERMISSION',payload=payload)

    def observe(self, evidence):
        self.calls += 1;self.evidence=evidence
        # The external owner must prevent a second reservation across recreated
        # objects. Do NOT interpret an old idempotent receipt as new permission.
        current=self.store.inspect().assignments[0]
        if current.commit_status!='NO_ATTEMPT':raise PreCommitAuditError('owner already reserved; stop')
        envelope=json.loads(evidence.envelope_snapshot)
        execution_ref=self.store.append('verified','EXECUTION',{'assignment_ref':self.assignment_ref,
            'status':'VERIFIED','started_ref':self.started_ref,'evidence_ref':evidence.digest,
            'speech_digest':sha256_bytes(envelope['speech'].encode('utf-8')),'actions':[envelope['expected']]}).record_digest
        receipt=self.store.append('reserved','COMMIT_WITNESS',{'assignment_ref':self.assignment_ref,
            'status':'ATTEMPT_RESERVED','execution_ref':execution_ref,'attempt_token':'CONTROLLED_RECORD_TOKEN_ONLY',
            'evidence_ref':evidence.digest})
        assert receipt.execution_permission is False
        return AuditAcknowledgement(receipt,canonical_json_bytes(self.store.records))

    def close(self):self.store.close()


def setup(path, phase='speech', action=A.NO_COMMITMENT):
    env,recorder,backend=prepare(phase)
    handoff=pending(env,recorder)
    pre=recorder.public_pre_for_handoff(handoff)
    # Targetless plans have identical eligibility in discussion and PK.
    payload=SpeechPlan(action).public_payload()
    expected=expected_action(payload,pre.current_speaker)
    actor,_=actor_for(expected);backend.perception_response=response_for(expected)
    owner=AuditOwnerFixture(path,pre,recorder,payload)
    return env,recorder,backend,actor,payload,owner


# Exact pre-edit orchestration is a test oracle only, not a production fallback.
# It stays independent of the live Git HEAD and temporary acceptance directories.
ORIGINAL_ORACLE_SOURCE = '"""Phase 3B orchestration at an already collected PRE; no planner imports."""\nfrom scripts.speech_realization import PublicRealizationContext, realize_parse_verify\nfrom werewolf.agents.prompt_template_v0 import build_public_claim_catalog\nfrom werewolf.canonical_collection.public_history import freeze_public_event_history\nfrom werewolf.speech.verified_commit import bind_verified_speech\n\n\ndef realize_and_commit(payload, *, env, actor, recorder):\n    """Realize once, perceive once, verify then transactionally commit.\n\n    before_agent_act must already have collected the real PRE/handoff. This\n    function never creates fake belief observations or changes their population.\n    On semantic failure that existing pending slot remains unchanged.\n    """\n    pending = recorder._pending\n    if (recorder._env is not env or pending is None or pending.handoff is None\n            or pending.prefix is None or pending.raw_action is not None\n            or env.phase not in (\'speech\', \'speech_pk\')):\n        raise ValueError("constrained realization requires an existing pending speech PRE")\n    history_digest = freeze_public_event_history(env.public_events).digest\n    speaker = f"player{env.current_act_idx + 1}"\n    if (pending.actor_id != speaker or pending.event_count_before != len(env.public_events)\n            or pending.prefix.public_event_history.digest != history_digest):\n        raise ValueError("pending PRE does not match current opportunity")\n    context = PublicRealizationContext(speaker, env.day,\n        \'discussion\' if env.phase == \'speech\' else \'pk_discussion\',\n        build_public_claim_catalog(env.get_observation()))\n    realized = realize_parse_verify(payload, context, actor=actor, perceiver=env.speech_perceiver,\n        actor_context=recorder.call_audit.action_context(acting_player_id=env.current_act_idx + 1,\n            boundary_id=pending.handoff.boundary_id, is_public_speech=True),\n        perception_context=recorder.call_audit.speech_perception_context(\n            event_id=f"event-{len(env.public_events):06d}",\n            boundary_id=pending.handoff.boundary_id, speaker_id=env.current_act_idx + 1))\n    envelope = bind_verified_speech(speech=realized.speech, expected=realized.expected,\n        perception=realized.perception, day=context.day, phase=env.phase,\n        public_history_digest=history_digest, perceiver=env.speech_perceiver)\n    return recorder.commit_verified_speech(env, envelope)\n'


def original_oracle():
    namespace = {}
    exec(compile(ORIGINAL_ORACLE_SOURCE, '<pre-edit-test-oracle>', 'exec'), namespace)
    return namespace['realize_and_commit']


def observation(env,recorder,backend,actor):
    return {'native':snapshot(env,recorder),'actor_requests':deepcopy(actor.backend.calls),
            'parser_requests':deepcopy(getattr(backend,'calls',[])),
            'call_records':deepcopy(recorder.call_audit.records),'call_budget':recorder.call_audit.summary()}


@pytest.mark.parametrize('phase',['speech','speech_pk'])
@pytest.mark.parametrize('action',[A.NO_COMMITMENT,A.SELF_DEFEND])
def test_default_off_matches_original_oracle_and_audit_on_adds_no_game_effect(tmp_path,phase,action):
    outcomes=[];rng=[]
    for i,fn in enumerate((original_oracle(),realize_and_commit,realize_and_commit)):
        path=tmp_path/str(i);path.mkdir()
        env,rec,backend,actor,payload,owner=setup(path,phase,action)
        try:
            actor_before=len(actor.backend.calls)
            kwargs={} if i<2 else {'precommit_audit':PreCommitAudit(owner.binding,owner.observe)}
            p_before=random.getstate();n_before=np.random.get_state()
            fn(payload,env=env,actor=actor,recorder=rec,**kwargs)
            assert random.getstate()==p_before
            n_after=np.random.get_state();assert n_before[0]==n_after[0] and np.array_equal(n_before[1],n_after[1]) and n_before[2:]==n_after[2:]
            assert len(actor.backend.calls)-actor_before==1
            assert owner.calls==(1 if i==2 else 0)
            outcomes.append(observation(env,rec,backend,actor));rng.append(env._rng.getstate())
        finally:owner.close()
    assert outcomes[0]==outcomes[1]==outcomes[2]
    assert rng[0]==rng[1]==rng[2]


@pytest.mark.parametrize('mode',['raise','none','false','invalid_snapshot','wrong_receipt','permit_true','wrong_evidence'])
def test_observer_failure_or_unconfirmed_ack_stops_before_native_commit(tmp_path,mode):
    env,rec,backend,actor,payload,owner=setup(tmp_path)
    before=snapshot(env,rec);native=Mock(wraps=rec.commit_verified_speech);rec.commit_verified_speech=native
    def callback(packet):
        if mode=='raise':raise OSError('audit fsync unconfirmed')
        if mode=='none':return None
        if mode=='false':return False
        ack=owner.observe(packet)
        if mode=='invalid_snapshot':return replace(ack,records_json=b'[]')
        if mode=='wrong_receipt':return replace(ack,receipt=replace(ack.receipt,record_digest='f'*64))
        if mode=='permit_true':return replace(ack,receipt=replace(ack.receipt,execution_permission=True))
        rows=json.loads(ack.records_json);rows[-1]['payload']['evidence_ref']='f'*64
        return replace(ack,records_json=canonical_json_bytes(rows))
    audit=PreCommitAudit(owner.binding,callback)
    try:
        with pytest.raises((ValueError,OSError)):
            realize_and_commit(payload,env=env,actor=actor,recorder=rec,precommit_audit=audit)
        native.assert_not_called();assert snapshot(env,rec)==before
        # Reusing the audit fails BEFORE a second Actor call, not as a fallback.
        calls=len(actor.backend.calls)
        with pytest.raises(PreCommitAuditError,match='consumed'):
            realize_and_commit(payload,env=env,actor=actor,recorder=rec,precommit_audit=audit)
        assert len(actor.backend.calls)==calls
    finally:owner.close()


@pytest.mark.parametrize('key,value',[('game_attempt_id','other'),('prefix_digest','f'*64),('payload_digest','f'*64),('handoff_digest','f'*64)])
def test_bad_binding_fails_before_actor_parser_and_callback(tmp_path,key,value):
    env,rec,backend,actor,payload,owner=setup(tmp_path);b=owner.binding.to_value();b[key]=value
    cb=Mock(side_effect=AssertionError('callback must not run'))
    audit=PreCommitAudit(freeze_json_object(b,'bad-binding'),cb)
    before=snapshot(env,rec)
    try:
        with pytest.raises(PreCommitAuditError,match='binding mismatch'):
            realize_and_commit(payload,env=env,actor=actor,recorder=rec,precommit_audit=audit)
        cb.assert_not_called();assert actor.backend.calls==[] and snapshot(env,rec)==before
    finally:owner.close()


def test_callback_receives_only_immutable_evidence_and_cannot_change_plan(tmp_path):
    env,rec,backend,actor,payload,owner=setup(tmp_path);observed=[]
    def callback(packet):
        assert {x.name for x in fields(packet)}=={'binding_json','envelope_snapshot','envelope_digest'}
        with pytest.raises(FrozenInstanceError):packet.binding_json=b'changed'
        assert sha256_bytes(packet.envelope_snapshot)==packet.envelope_digest
        observed.append(packet.to_record())
        ack=owner.observe(packet)
        payload['action']='SELF_DEFEND'  # Malicious closure, not an input handle.
        return ack
    native=Mock(wraps=rec.commit_verified_speech);rec.commit_verified_speech=native
    try:
        with pytest.raises(PreCommitAuditError,match='changed after audit'):
            realize_and_commit(payload,env=env,actor=actor,recorder=rec,precommit_audit=PreCommitAudit(owner.binding,callback))
        native.assert_not_called();assert len(observed)==1
        assert 'suspicion_support' not in json.dumps(observed)
    finally:owner.close()


def test_callback_invoked_once_and_new_object_cannot_reserve_same_store_assignment_again(tmp_path):
    env,rec,backend,actor,payload,owner=setup(tmp_path)
    # Stop after confirmed audit, before a native commit. A recreated observer
    # is NOT a new authorization; the existing store owner rejects it.
    native=Mock(side_effect=TimeoutError('commit outcome unknown'));rec.commit_verified_speech=native
    try:
        with pytest.raises(TimeoutError):realize_and_commit(payload,env=env,actor=actor,recorder=rec,precommit_audit=PreCommitAudit(owner.binding,owner.observe))
        assert native.call_count==1 and owner.calls==1
        with pytest.raises(PreCommitAuditError,match='already reserved'):
            realize_and_commit(payload,env=env,actor=actor,recorder=rec,precommit_audit=PreCommitAudit(owner.binding,owner.observe))
        assert native.call_count==1 and owner.calls==2
        assert owner.store.inspect().assignments[0].commit_intents==1
    finally:owner.close()


def test_store_fsync_failure_cannot_be_turned_into_a_successful_ack(tmp_path,monkeypatch):
    env,rec,backend,actor,payload,owner=setup(tmp_path)
    from werewolf.observer_dtrs import action_evidence_store as store_module
    monkeypatch.setattr(store_module,'_append_synced',Mock(side_effect=OSError('injected fsync failure')))
    native=Mock(wraps=rec.commit_verified_speech);rec.commit_verified_speech=native
    before=snapshot(env,rec)
    try:
        with pytest.raises(OSError,match='fsync failure'):
            realize_and_commit(payload,env=env,actor=actor,recorder=rec,precommit_audit=PreCommitAudit(owner.binding,owner.observe))
        native.assert_not_called();assert snapshot(env,rec)==before
    finally:owner.close()


def test_commit_success_then_research_log_failure_does_not_repeat_native_submission(tmp_path,monkeypatch):
    env,rec,backend,actor,payload,owner=setup(tmp_path);audit=PreCommitAudit(owner.binding,owner.observe)
    native=Mock(wraps=rec.commit_verified_speech);rec.commit_verified_speech=native
    try:
        realize_and_commit(payload,env=env,actor=actor,recorder=rec,precommit_audit=audit)
        reserved_records=owner.store.records
        from werewolf.observer_dtrs import action_evidence_store as store_module
        disk_write=Mock(side_effect=OSError('post-commit research persistence failed'))
        monkeypatch.setattr(store_module,'_append_synced',disk_write)
        with pytest.raises(OSError,match='post-commit'):
            owner.store.append('post-commit-unknown','COMMIT_WITNESS',{
                'assignment_ref':owner.assignment_ref,'status':'COMMIT_UNKNOWN',
                'reservation_ref':owner.store.records[-1]['record_digest'],'evidence_ref':digest('post-commit-unconfirmed')})
        assert len(rec.submitted_gameplay_actions)==1 and native.call_count==1
        calls=len(actor.backend.calls)
        with pytest.raises(ValueError):realize_and_commit(payload,env=env,actor=actor,recorder=rec,precommit_audit=audit)
        assert native.call_count==1 and len(actor.backend.calls)==calls
        # Phase-2G can observe native occurrence; this observer does not manufacture
        # a COMMITTED witness or change the original research state itself.
        assert not owner.store._healthy
        from werewolf.observer_dtrs.action_evidence_ledger import validate_ledger
        disk_write.assert_called_once()
        disk_records=[json.loads(line) for line in (owner.store.root/'events.jsonl').read_bytes().splitlines()]
        assert disk_records==list(reserved_records)
        assert validate_ledger(disk_records,ledger_id='controlled-precommit').assignments[0].commit_status=='AWAITING_EXTERNAL_READBACK'
    finally:owner.close()


def test_late_text_without_env_cannot_commit_and_never_calls_observer(tmp_path):
    env,rec,backend,actor,payload,owner=setup(tmp_path);before=snapshot(env,rec)
    release=threading.Event();text=[]
    worker=threading.Thread(target=lambda:(release.wait(3),text.append('late public text only')))
    actor._generate_public_speech=Mock(side_effect=TimeoutError('text request timed out'))
    audit=PreCommitAudit(owner.binding,owner.observe);native=Mock(wraps=rec.commit_verified_speech);rec.commit_verified_speech=native
    worker.start()
    try:
        with pytest.raises(TimeoutError):realize_and_commit(payload,env=env,actor=actor,recorder=rec,precommit_audit=audit)
        release.set();worker.join(3)
        assert not worker.is_alive() and text and snapshot(env,rec)==before
        native.assert_not_called();assert owner.calls==0
    finally:release.set();worker.join(3);owner.close()


def test_late_executor_with_commit_capability_is_unknown_not_stopped_by_observer(tmp_path):
    env,rec,backend,actor,payload,owner=setup(tmp_path);native=rec.commit_verified_speech
    release=threading.Event();threads=[];errors=[]
    def uncertain(env_arg,envelope):
        def late():
            if not release.wait(3):return
            try:native(env_arg,envelope)
            except BaseException as e:errors.append(repr(e))
        t=threading.Thread(target=late);threads.append(t);t.start()
        raise TimeoutError('native executor not confirmed stopped')
    rec.commit_verified_speech=Mock(side_effect=uncertain);audit=PreCommitAudit(owner.binding,owner.observe)
    try:
        with pytest.raises(TimeoutError):realize_and_commit(payload,env=env,actor=actor,recorder=rec,precommit_audit=audit)
        before_calls=len(actor.backend.calls)
        with pytest.raises(PreCommitAuditError,match='consumed'):
            realize_and_commit(payload,env=env,actor=actor,recorder=rec,precommit_audit=audit)
        assert len(actor.backend.calls)==before_calls and rec.commit_verified_speech.call_count==1
        # Lack of an action *now* is NOT proof of NOT_COMMITTED: late executor
        # still holds the original native call. The hook does not cancel it.
        assert rec.submitted_gameplay_actions==[]
        release.set()
        for t in threads:t.join(3)
        assert not any(t.is_alive() for t in threads) and not errors
        assert len(rec.submitted_gameplay_actions)==1
        assert owner.store.inspect().assignments[0].commit_status=='AWAITING_EXTERNAL_READBACK'
    finally:
        release.set()
        for t in threads:t.join(3)
        owner.close()


def test_wrong_thread_cannot_reuse_owner_audit(tmp_path):
    env,rec,backend,actor,payload,owner=setup(tmp_path);audit=PreCommitAudit(owner.binding,owner.observe);errors=[]
    def call():
        try:realize_and_commit(payload,env=env,actor=actor,recorder=rec,precommit_audit=audit)
        except BaseException as e:errors.append(e)
    t=threading.Thread(target=call);t.start();t.join(3)
    try:
        assert not t.is_alive() and len(errors)==1 and isinstance(errors[0],PreCommitAuditError)
        assert actor.backend.calls==[] and owner.calls==0
    finally:owner.close()


def test_wrong_assignment_is_checked_against_ack_not_invented_from_live_game(tmp_path):
    env,rec,backend,actor,payload,owner=setup(tmp_path)
    b=owner.binding.to_value();b['assignment_ref']='f'*64
    native=Mock(wraps=rec.commit_verified_speech);rec.commit_verified_speech=native
    try:
        with pytest.raises(PreCommitAuditError,match='intent/evidence'):
            realize_and_commit(payload,env=env,actor=actor,recorder=rec,
                precommit_audit=PreCommitAudit(freeze_json_object(b,'wrong-assignment'),owner.observe))
        assert owner.calls==1 and len(actor.backend.calls)==1
        native.assert_not_called()
    finally:owner.close()


@pytest.mark.parametrize('failure',['actor_retry','parser_retry','actor_exhausted','parser_exhausted'])
def test_original_bounded_retry_requests_and_rng_unchanged(failure,tmp_path):
    from tests.tom.test_speech_realization import ActorBackend,ParserBackend
    from werewolf.agents.gpt_agent import GAMEPLAY_GENERATION_MAX_ATTEMPTS
    from werewolf.agents.llm_agent import GameplayGenerationExhausted
    from scripts.speech_realization import SemanticVerificationError
    from werewolf.speech.speech_perceiver import SPEECH_PARSER_GENERATION_MAX_ATTEMPTS
    results=[]
    for i,fn in enumerate((original_oracle(),realize_and_commit,realize_and_commit)):
        path=tmp_path/str(i);path.mkdir();env,rec,backend,actor,payload,owner=setup(path)
        good_actor=actor.backend.speech
        class ScriptedActor(ActorBackend):
            def chat_with_metadata(self,**kwargs):
                self.calls.append(deepcopy(kwargs))
                bad=failure.startswith('actor') and ('exhausted' in failure or len(self.calls)==1)
                return ('' if bad else good_actor),{'finish_reason':'stop'}
        class ScriptedParser(ParserBackend):
            def chat(self,**kwargs):
                self.calls.append(deepcopy(kwargs))
                bad=failure.startswith('parser') and ('exhausted' in failure or len(self.calls)==1)
                return 'invalid parser output' if bad else self.response
        actor.backend=ScriptedActor(good_actor)
        parser=ScriptedParser(response_for(expected_action(payload,rec._pending.actor_id)))
        env.speech_perceiver.backend=parser
        before=snapshot(env,rec);py=random.getstate();np_before=np.random.get_state()
        try:
            kwargs={} if i<2 else {'precommit_audit':PreCommitAudit(owner.binding,owner.observe)}
            if 'exhausted' in failure:
                with pytest.raises(GameplayGenerationExhausted if failure=='actor_exhausted' else SemanticVerificationError):
                    fn(payload,env=env,actor=actor,recorder=rec,**kwargs)
                assert snapshot(env,rec)==before and owner.calls==0
            else:
                fn(payload,env=env,actor=actor,recorder=rec,**kwargs)
                assert owner.calls==(1 if i==2 else 0)
            expected_actor=GAMEPLAY_GENERATION_MAX_ATTEMPTS if failure=='actor_exhausted' else (2 if failure=='actor_retry' else 1)
            expected_parser=0 if failure=='actor_exhausted' else (SPEECH_PARSER_GENERATION_MAX_ATTEMPTS if failure=='parser_exhausted' else (2 if failure=='parser_retry' else 1))
            assert len(actor.backend.calls)==expected_actor and len(parser.calls)==expected_parser
            assert random.getstate()==py
            after=np.random.get_state();assert np_before[0]==after[0] and np.array_equal(np_before[1],after[1]) and np_before[2:]==after[2:]
            results.append((snapshot(env,rec),actor.backend.calls,parser.calls))
        finally:owner.close()
    assert results[0]==results[1]==results[2]


def test_env_published_but_recorder_update_failed_is_unknown_and_never_retried(tmp_path):
    env,rec,backend,actor,payload,owner=setup(tmp_path);audit=PreCommitAudit(owner.binding,owner.observe)
    publish=env._publish_verified_speech
    def publish_then_fail(staged):
        publish(staged)
        raise OSError('after env publish before recorder update')
    env._publish_verified_speech=Mock(side_effect=publish_then_fail)
    native=Mock(wraps=rec.commit_verified_speech);rec.commit_verified_speech=native
    before_events=len(env.public_events)
    try:
        with pytest.raises(OSError,match='after env publish'):
            realize_and_commit(payload,env=env,actor=actor,recorder=rec,precommit_audit=audit)
        assert len(env.public_events)>before_events and not rec.submitted_gameplay_actions
        assert rec._pending is not None and native.call_count==1
        assert owner.store.inspect().assignments[0].commit_status=='AWAITING_EXTERNAL_READBACK'
        calls=len(actor.backend.calls)
        with pytest.raises(ValueError):realize_and_commit(payload,env=env,actor=actor,recorder=rec,precommit_audit=audit)
        assert native.call_count==1 and len(actor.backend.calls)==calls
    finally:owner.close()


@pytest.mark.parametrize('assigned',[False,True])
def test_existing_run_random_none_means_preassignment_baseline_not_assigned_failure(assigned):
    # Only exercise the real hook control flow with non-game stubs. Stop before
    # any env.step or real Agent call; there is no completed game or random draw.
    from contextlib import nullcontext
    from types import SimpleNamespace
    from run_random import eval as run_eval
    handoff=SimpleNamespace(boundary_id='controlled-boundary')
    env=SimpleNamespace(phase='speech',public_events=[],reset=Mock(return_value={'current_act_idx':1}),
                        step=Mock(side_effect=AssertionError('no game execution')))
    rec=SimpleNamespace(start=Mock(),before_agent_act=Mock(return_value=handoff),
        failure_from_exception=lambda e:RuntimeError('controlled stop'),after_env_step=Mock())
    actor=SimpleNamespace(reset=Mock(),act_with_pre_speech_belief=Mock(side_effect=ValueError('baseline sentinel')))
    pilot=SimpleNamespace(handle_pre=Mock(side_effect=PreCommitAuditError('assigned F0 stop')) if assigned else Mock(return_value=None),
                          after_step=Mock())
    audit=SimpleNamespace(action_context=lambda **kw:nullcontext())
    with pytest.raises(RuntimeError) as stopped:
        run_eval(env,[actor],None,canonical_recorder=rec,call_audit=audit,online_pilot=pilot)
    assert isinstance(stopped.value.__cause__,PreCommitAuditError if assigned else ValueError)
    assert actor.act_with_pre_speech_belief.call_count==(0 if assigned else 1)
    rec.before_agent_act.assert_called_once();pilot.handle_pre.assert_called_once()
    env.step.assert_not_called();pilot.after_step.assert_not_called();rec.after_env_step.assert_not_called()


def test_mutating_frozen_packet_via_object_bypass_still_rejects_before_native(tmp_path):
    env,rec,backend,actor,payload,owner=setup(tmp_path)
    def callback(packet):
        ack=owner.observe(packet)
        object.__setattr__(packet,'envelope_digest','f'*64)
        return ack
    native=Mock(wraps=rec.commit_verified_speech);rec.commit_verified_speech=native
    try:
        with pytest.raises(PreCommitAuditError,match='evidence was changed'):
            realize_and_commit(payload,env=env,actor=actor,recorder=rec,precommit_audit=PreCommitAudit(owner.binding,callback))
        native.assert_not_called()
    finally:owner.close()
