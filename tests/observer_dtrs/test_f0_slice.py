"""Original production start/night/PRE lifecycle, artificial CPU backends only."""
from copy import deepcopy
import json
import random
import re
import threading
from unittest.mock import Mock

import numpy as np
import pytest

from scripts.speech_planning import SpeechPlan, PlanningAction as A
from tests.canonical_collection.test_game_bundle import _plan, _claim
from tests.canonical_collection.test_runtime_game_evidence import ROLES, _Agent
from tests.runtime.test_runtime_config import new_config
from werewolf.agents.gpt_agent import GPTAgent
from werewolf.artifact_io import verify_artifact
from werewolf.canonical_collection import production_runtime as production
from werewolf.canonical_collection.attempt_ledger import validate_attempt_ledger
from werewolf.canonical_collection.collector import collect, CanonicalGameProduct
from werewolf.canonical_collection.runtime import CanonicalGameRecorder
from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
from werewolf.observer_dtrs import f0_slice as f0
from werewolf.speech.speech_perceiver import SpeechPerceiver


_ORIGINAL_FACTORY_CALL = production.Classic7RuntimeFactory.__call__


class ArtificialBackend:
    def __init__(self, action=A.NO_COMMITMENT, failure=None):
        self.action, self.failure = action, failure
        self.requests = []
        self.before_actor = lambda: None

    def chat(self, *, messages, **kwargs):
        prompt = messages[-1]['content']
        self.requests.append(('chat', prompt))
        if 'suspected_werewolves' in prompt:
            required = prompt.split('MUST INCLUDE: ', 1)[1].splitlines()[0]
            return json.dumps({'suspected_werewolves': [] if required == '<none>' else required.split(', ')})
        tail = prompt.rsplit('待解析的玩家发言：', 1)[1]
        seat = re.search(r'player([1-7]):', tail).group(1)
        if 'F0_TEXT' not in tail:
            return f'player{seat} | no_commitment | NONE'
        if self.failure == 'parser': return 'INVALID_FULL_RESPONSE'
        if self.failure == 'multi':
            return f'player{seat} | no_commitment | NONE\nplayer{seat} | point_as_non_werewolf | player{seat}'
        if self.action == A.SELF_DEFEND: return f'player{seat} | point_as_non_werewolf | player{seat}'
        return f'player{seat} | no_commitment | NONE'

    def chat_with_metadata(self, **kwargs):
        self.before_actor()
        self.requests.append(('actor', deepcopy(kwargs)))
        if self.failure == 'actor': raise TimeoutError('artificial text-only timeout')
        return 'F0_TEXT 我暂时说明自己的立场。', {'finish_reason': 'stop'}


class ScriptedAgent(GPTAgent):
    """Only baseline actions scripted; target uses INHERITED original Actor."""
    report_suspected_werewolves_readonly = _Agent.report_suspected_werewolves_readonly

    def __init__(self, backend, seat, trace):
        super().__init__(backend=backend, model_name='fixture-model-v1')
        self.seat, self.trace = seat, trace
        self.backend_id = 'deterministic-fixture-backend-v1'

    def act(self, observation):
        self.trace.append(('baseline', self.seat, observation['phase']))
        if 'skill_witch' in observation['phase']:
            return next(a for a in observation['valid_action'] if 'pass' in a[0])
        if 'skill_wolf' in observation['phase']:
            return next(a for a in observation['valid_action'] if a[1] != -1)
        if 'vote' in observation['phase']:
            return ('vote', (2,3,1,1,2,3,0)[self.seat-1])
        return observation['valid_action'][0]

    def act_with_pre_speech_belief(self, observation, *, pre_speech_belief):
        self.trace.append(('baseline_speech', self.seat, pre_speech_belief.boundary_id))
        return ('speech_pk' if 'speech_pk' in observation['phase'] else 'speech', f'BASELINE player{self.seat}')


def assembly(monkeypatch, *, action=A.NO_COMMITMENT, failure=None, control=True, phase='discussion'):
    backend = ArtificialBackend(action, failure)
    trace, runtimes = [], []
    def build(config, *, log_save_path, random_seed, backends):
        assert log_save_path is None
        wrapped = backends['deepseek']
        env = WerewolfTextEnvV0(speech_perceiver=SpeechPerceiver(backend=wrapped,
            model_name='fixture-model-v1'), random_seed=random_seed, log_save_path=None)
        agents = [ScriptedAgent(wrapped, seat, trace) for seat in range(1,8)]
        return env, agents, list(ROLES), []
    monkeypatch.setattr(production, 'build_runtime', build)
    config = f0.F0SliceConfig('artificial-f0-study', 'artificial-fixed-qualification', SpeechPlan(action), (phase,))
    ctrl = f0.F0SliceControl(config) if control else None
    factory = production.Classic7RuntimeFactory(runtime_config=new_config(), backends={'deepseek':backend},
        configured_call_limit=2000, f0_control=ctrl)
    original = _ORIGINAL_FACTORY_CALL
    def capture(self, **kwargs):
        runtime = original(self, **kwargs);runtimes.append(runtime);return runtime
    monkeypatch.setattr(factory.__class__, '__call__', capture)
    plan = _plan(ordered_seed_pool=(101,202), target_canonical_success_count=1)
    return factory, ctrl, backend, trace, runtimes, plan


def execute(tmp_path, factory, plan):
    return f0.run_f0_attempt(factory=factory, plan=plan, destination=tmp_path/'qualification',
        attempt_id='artificial-fresh-attempt', timestamp_utc='2026-10-29T00:00:00Z')


@pytest.mark.parametrize('phase', ['discussion','pk_discussion'])
@pytest.mark.parametrize('action', [A.SELF_DEFEND,A.NO_COMMITMENT])
def test_original_start_full_prelude_one_target_normal_slice(tmp_path,monkeypatch,phase,action):
    factory,ctrl,backend,trace,runtimes,plan = assembly(monkeypatch,action=action,phase=phase)
    starts=[];old_start=CanonicalGameRecorder.start
    def start(rec,env,**kwargs):
        assert env.day==0 and env.phase.startswith('skill_')
        assert not rec.prefixes and not rec.submitted_gameplay_actions
        starts.append(tuple(env.public_events));return old_start(rec,env,**kwargs)
    monkeypatch.setattr(CanonicalGameRecorder,'start',start)
    def before_actor():
        assert [r['kind'] for r in ctrl._store.records]==['OPPORTUNITY','ASSIGNMENT','EXECUTION']
        assert ctrl._store.inspect().assignments[0].commit_status=='NO_ATTEMPT'
    backend.before_actor=before_actor
    original=CanonicalGameRecorder.commit_verified_speech;counts=[]
    def native(rec,*args,**kwargs):
        assert ctrl._store.inspect().assignments[0].commit_status=='AWAITING_EXTERNAL_READBACK'
        assert ctrl._intent_ack.execution_permission is False
        counts.append(len(rec.submitted_gameplay_actions))
        return original(rec,*args,**kwargs)
    monkeypatch.setattr(CanonicalGameRecorder,'commit_verified_speech',native)
    product=execute(tmp_path,factory,plan);v=product.evidence.to_value();runtime=runtimes[0]
    assert type(product) is f0.F0SliceProduct and not isinstance(product,CanonicalGameProduct)
    assert v['slice_status']=='BOUNDARY_RETURNED' and v['winner'] is None
    assert len(starts)==len(runtimes)==len(counts)==1
    assert counts[0]>0 and any(a['action_type']!='public_speech' for a in v['submitted_gameplay_actions'][:-1])
    assert len(v['prefixes'])>1  # At least one genuine preceding non-target speech.
    assert v['target']['before_action_count']==len(v['submitted_gameplay_actions'])-1
    target=v['target']['pre']; assert target['boundary_id']==v['native_action']['boundary_id']
    assert v['native_occurrence']=='OBSERVED_LOCAL' and v['assignment_native_attribution'].startswith('UNKNOWN')
    assert v['all_native_capable_executors_stopped']=='UNKNOWN' and not v['execution_permission']
    assert len(runtime.recorder.observations)==sum(len(pre.alive_observer_ids) for pre in runtime.recorder.prefixes)
    assert runtime.recorder._pending is None and runtime.recorder._winner is None
    assert len([r for r in backend.requests if r[0]=='actor'])==1
    assert len(v['public_event_history'])==len(runtime.env.public_events)
    with pytest.raises(RuntimeError,match='not complete'):runtime.recorder.complete_evidence()
    ledger=validate_attempt_ledger(tmp_path/'qualification'/'attempt_ledger',plan)
    assert len(ledger.claims)==1 and ledger.terminals==() and ledger.canonical_success_count==0
    assert not (tmp_path/'qualification'/'games').exists()
    artifact=verify_artifact(product.artifact_path,expected_artifact_type='controlled_f0_slice',
                             expected_schema_version='controlled_f0_slice_v1')
    assert artifact.manifest_digest==product.manifest_digest
    # Re-entry cannot reset env, recollect reports, submit or claim another game.
    before=(len(trace),len(backend.requests),len(runtime.env.public_events))
    with pytest.raises(ValueError,match='consumed'):execute(tmp_path,factory,plan)
    with pytest.raises(ValueError,match='repeated'):runtime.run_f0_slice()
    assert (len(trace),len(backend.requests),len(runtime.env.public_events))==before
    evidence=tmp_path/'case-evidence.json'
    evidence.write_text(json.dumps({'phase':phase,'action':action.value,'prelude_actions':counts[0],
       'prefixes':len(v['prefixes']),'observations':len(runtime.recorder.observations),
       'target':target['boundary_id'],'manifest_digest':product.manifest_digest,
       'status':v['slice_status'],'real_F0':'NOT_RUN'}))


@pytest.mark.parametrize('failure',['actor','parser','multi','ack','native_none','native_timeout','native_after_publish'])
def test_assigned_failure_stops_one_attempt_without_fallback(tmp_path,monkeypatch,failure):
    factory,ctrl,backend,trace,runtimes,plan=assembly(monkeypatch,failure=failure)
    calls=[];old=CanonicalGameRecorder.commit_verified_speech
    def native(rec,*args,**kwargs):
        calls.append(1)
        if failure=='native_none':return None
        if failure=='native_timeout':raise TimeoutError('unconfirmed native')
        result=old(rec,*args,**kwargs)
        if failure=='native_after_publish':raise OSError('post-publish return lost')
        return result
    monkeypatch.setattr(CanonicalGameRecorder,'commit_verified_speech',native)
    if failure=='ack':
        monkeypatch.setattr(ctrl,'_observe',Mock(side_effect=OSError('intent ACK unconfirmed')))
    product=execute(tmp_path,factory,plan);v=product.evidence.to_value();runtime=runtimes[0]
    assert v['slice_status']=='STOPPED_FAILURE'
    assert len(runtimes)==1 and len(calls)==(1 if failure.startswith('native') else 0)
    assert v['assignment_ref'] and len(runtime.recorder.prefixes)==len(v['prefixes'])
    assert v['native_return'] == ('UNKNOWN_AFTER_INTENT' if failure.startswith('native') else 'NOT_REACHED')
    if failure=='native_after_publish': assert v['native_occurrence']=='OBSERVED_LOCAL'
    else: assert v['native_occurrence']=='UNKNOWN'
    assert not any(t[0]=='baseline_speech' and t[2]==v['target']['pre']['boundary_id'] for t in trace)
    assert len(validate_attempt_ledger(tmp_path/'qualification'/'attempt_ledger',plan).claims)==1
    with pytest.raises(ValueError,match='consumed'):execute(tmp_path,factory,plan)
    assert len(calls)==(1 if failure.startswith('native') else 0)


def test_seal_failure_retains_product_and_never_reexecutes(tmp_path,monkeypatch):
    factory,ctrl,backend,trace,runtimes,plan=assembly(monkeypatch)
    original=f0.publish_artifact
    def publish(dest,**kwargs):
        if str(dest).endswith('/slice'):raise OSError('slice log fsync failure')
        return original(dest,**kwargs)
    monkeypatch.setattr(f0,'publish_artifact',publish)
    with pytest.raises(f0.F0SliceSealError) as error:execute(tmp_path,factory,plan)
    assert error.value.product.evidence.to_value()['native_occurrence']=='OBSERVED_LOCAL'
    n=len(runtimes[0].recorder.submitted_gameplay_actions)
    with pytest.raises(ValueError,match='consumed'):execute(tmp_path,factory,plan)
    assert len(runtimes[0].recorder.submitted_gameplay_actions)==n


def test_ordinary_collect_refuses_f0_before_claim_or_loop(tmp_path,monkeypatch):
    factory,ctrl,backend,trace,runtimes,plan=assembly(monkeypatch)
    with pytest.raises(TypeError,match='single-attempt'):
        collect(plan=plan,runtime_factory=factory,destination=tmp_path/'wrong')
    assert not (tmp_path/'wrong').exists() and not runtimes
    execute(tmp_path,factory,plan)
    ordinary,_,_,_,_,_=assembly(monkeypatch,control=False)
    with pytest.raises(TypeError,match='single-attempt'):
        collect(plan=plan,runtime_factory=ordinary,destination=tmp_path/'qualification')


@pytest.mark.parametrize('scope',['REAL_F0','APPROVED',''])
def test_strict_contract_not_waived(scope):
    with pytest.raises(ValueError,match='blocked'):
        f0.F0SliceConfig('study','qualification',SpeechPlan(A.NO_COMMITMENT),scope=scope)


def test_targeted_plan_and_control_identity_cannot_rebind(tmp_path,monkeypatch):
    with pytest.raises(ValueError): f0.F0SliceConfig('s','q',SpeechPlan(A.CLEAR,'player3'))
    factory,ctrl,backend,trace,runtimes,plan=assembly(monkeypatch)
    with pytest.raises(AttributeError):ctrl.config=f0.F0SliceConfig('s','q',SpeechPlan(A.SELF_DEFEND))
    runtime=factory(plan=plan,claim=_claim(plan))
    with pytest.raises(ValueError,match='unbound'):runtime.run_f0_slice()
    with pytest.raises(TypeError,match='cannot enter'):runtime.run()
    assert runtime.recorder._env is None and not backend.requests


def test_late_native_capability_remains_unknown_after_slice_seal(tmp_path,monkeypatch):
    factory,ctrl,backend,trace,runtimes,plan=assembly(monkeypatch)
    original=CanonicalGameRecorder.commit_verified_speech
    release=threading.Event();threads=[];errors=[]
    def uncertain(rec,*args,**kwargs):
        def late():
            if release.wait(3):
                try: original(rec,*args,**kwargs)
                except BaseException as error:errors.append(type(error).__name__)
        thread=threading.Thread(target=late);threads.append(thread);thread.start()
        raise TimeoutError('native capability holder not stopped')
    monkeypatch.setattr(CanonicalGameRecorder,'commit_verified_speech',uncertain)
    try:
        product=execute(tmp_path,factory,plan);v=product.evidence.to_value()
        assert v['native_occurrence']=='UNKNOWN' and v['native_return']=='UNKNOWN_AFTER_INTENT'
        assert v['all_native_capable_executors_stopped']=='UNKNOWN'
        count=len(runtimes[0].recorder.submitted_gameplay_actions)
        release.set()
        for thread in threads: thread.join(3)
        assert not errors and not any(t.is_alive() for t in threads)
        assert len(runtimes[0].recorder.submitted_gameplay_actions)==count+1
        assert product.evidence.to_value()==v  # Point-in-time evidence, no invented final resolution.
    finally:
        release.set()
        for thread in threads:thread.join(3)


def test_same_output_namespace_cannot_be_reopened_as_fresh_attempt(tmp_path,monkeypatch):
    factory,ctrl,_,_,_,plan=assembly(monkeypatch);execute(tmp_path,factory,plan)
    another,_,backend,_,runtimes,_=assembly(monkeypatch)
    with pytest.raises(FileExistsError):execute(tmp_path,another,plan)
    assert not runtimes and not backend.requests
    assert len(validate_attempt_ledger(tmp_path/'qualification'/'attempt_ledger',plan).claims)==1


def test_wrong_thread_cannot_start_owner(tmp_path,monkeypatch):
    factory,ctrl,backend,trace,runtimes,plan=assembly(monkeypatch);errors=[]
    def wrong():
        try: execute(tmp_path,factory,plan)
        except ValueError as error:errors.append(str(error))
    thread=threading.Thread(target=wrong);thread.start();thread.join(3)
    assert len(errors)==1 and 'another cooperating' in errors[0]
    assert not runtimes and not (tmp_path/'qualification').exists()


def test_returned_tuple_without_native_evidence_is_not_success(tmp_path,monkeypatch):
    factory,ctrl,_,_,_,plan=assembly(monkeypatch)
    monkeypatch.setattr(CanonicalGameRecorder,'commit_verified_speech',Mock(return_value=({},0,False,{})))
    product=execute(tmp_path,factory,plan);v=product.evidence.to_value()
    assert v['slice_status']=='STOPPED_UNCONFIRMED' and v['native_occurrence']=='UNKNOWN'
    assert v['native_action'] is None and v['native_return']=='RETURNED_4_TUPLE'


def test_lost_ACK_keeps_original_reservation_no_retry(tmp_path,monkeypatch):
    factory,ctrl,backend,trace,runtimes,plan=assembly(monkeypatch)
    observe=ctrl._observe
    def lost(packet):
        observe(packet)
        raise OSError('ACK lost after synced reservation')
    monkeypatch.setattr(ctrl,'_observe',lost)
    native=Mock(wraps=CanonicalGameRecorder.commit_verified_speech)
    monkeypatch.setattr(CanonicalGameRecorder,'commit_verified_speech',native)
    product=execute(tmp_path,factory,plan);v=product.evidence.to_value()
    native.assert_not_called()
    records=[json.loads(line) for line in (tmp_path/'qualification/action-journal/events.jsonl').read_bytes().splitlines()]
    assert records[-1]['payload']['status']=='ATTEMPT_RESERVED'
    assert v['native_return']=='UNKNOWN_AFTER_INTENT' and v['native_occurrence']=='UNKNOWN'


def test_env_published_but_recorder_update_failed_preserves_public_observation(tmp_path,monkeypatch):
    factory,ctrl,_,_,runtimes,plan=assembly(monkeypatch)
    original=WerewolfTextEnvV0._publish_verified_speech
    def fail_after_env(env,staged):
        original(env,staged)
        raise OSError('env published before recorder update')
    monkeypatch.setattr(WerewolfTextEnvV0,'_publish_verified_speech',fail_after_env)
    product=execute(tmp_path,factory,plan);v=product.evidence.to_value()
    assert v['slice_status']=='STOPPED_FAILURE'
    assert v['native_action'] is None and v['native_action_status']=='UNKNOWN'
    assert v['speech_event_status']==v['v1_status']=='OBSERVED_LOCAL'
    assert v['native_occurrence']=='OBSERVED_LOCAL' and v['native_return']=='UNKNOWN_AFTER_INTENT'
    assert v['assignment_native_attribution'].startswith('UNKNOWN')
    assert runtimes[0].recorder._pending is not None


def test_conflicting_native_records_are_sealed_as_conflict_not_repaired(tmp_path,monkeypatch):
    factory,ctrl,_,_,_,plan=assembly(monkeypatch)
    original=CanonicalGameRecorder.commit_verified_speech
    def duplicate(rec,*args,**kwargs):
        result=original(rec,*args,**kwargs)
        rec.submitted_gameplay_actions.append(rec.submitted_gameplay_actions[-1])
        return result
    monkeypatch.setattr(CanonicalGameRecorder,'commit_verified_speech',duplicate)
    product=execute(tmp_path,factory,plan);v=product.evidence.to_value()
    assert v['slice_status']=='STOPPED_CONFLICT' and v['native_occurrence']=='CONFLICT'
    assert len(v['target_native_actions'])==2 and v['native_action'] is None
    assert 'MULTIPLE_TARGET_NATIVE_EVIDENCE' in v['local_evidence_conflicts']


def test_assignment_bytes_present_ACK_lost_is_unknown_not_unassigned(tmp_path,monkeypatch):
    from werewolf.observer_dtrs.action_evidence_store import ActionEvidenceStore
    factory,ctrl,backend,_,_,plan=assembly(monkeypatch)
    append=ActionEvidenceStore.append
    def lose_ack(store,operation,kind,payload):
        receipt=append(store,operation,kind,payload)
        if kind=='ASSIGNMENT': raise OSError('assignment ACK lost')
        return receipt
    monkeypatch.setattr(ActionEvidenceStore,'append',lose_ack)
    native=Mock();monkeypatch.setattr(CanonicalGameRecorder,'commit_verified_speech',native)
    product=execute(tmp_path,factory,plan);v=product.evidence.to_value()
    assert v['target_locked'] and v['assignment_ref'] is None
    assert v['assignment_persistence']=='UNKNOWN_AFTER_TARGET_LOCK'
    journal=bytes.fromhex(v['custody_bytes']['journal']['bytes_hex'])
    assert json.loads(journal.splitlines()[-1])['kind']=='ASSIGNMENT'
    assert not [r for r in backend.requests if r[0]=='actor']
    native.assert_not_called()


@pytest.mark.parametrize('field,value',[('actor_id','player7'),('step_index',999999)])
def test_wrong_native_PRE_identity_is_conflict(tmp_path,monkeypatch,field,value):
    from dataclasses import replace
    factory,ctrl,_,_,_,plan=assembly(monkeypatch)
    original=CanonicalGameRecorder.commit_verified_speech
    def corrupt(rec,*args,**kwargs):
        result=original(rec,*args,**kwargs)
        rec.submitted_gameplay_actions[-1]=replace(rec.submitted_gameplay_actions[-1],**{field:value})
        return result
    monkeypatch.setattr(CanonicalGameRecorder,'commit_verified_speech',corrupt)
    v=execute(tmp_path,factory,plan).evidence.to_value()
    assert v['slice_status']=='STOPPED_CONFLICT'
    assert 'NATIVE_PRE_IDENTITY_MISMATCH' in v['local_evidence_conflicts']


def test_factory_cannot_substitute_old_claim_before_runtime(tmp_path,monkeypatch):
    from dataclasses import replace
    factory,ctrl,backend,trace,runtimes,plan=assembly(monkeypatch)
    call=production.Classic7RuntimeFactory.__call__
    def wrong(self,**kwargs):
        runtime=call(self,**kwargs)
        runtime.recorder.claim=replace(runtime.recorder.claim,attempt_id='old-attempt')
        return runtime
    monkeypatch.setattr(production.Classic7RuntimeFactory,'__call__',wrong)
    with pytest.raises(ValueError,match='claim mismatch'):execute(tmp_path,factory,plan)
    assert not trace and not backend.requests and not runtimes[0].recorder.prefixes
