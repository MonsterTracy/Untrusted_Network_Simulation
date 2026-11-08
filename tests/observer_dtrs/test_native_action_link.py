"""Artificial native values, never an Actor/Parser/game/replay execution.

_fixture constructs canonical data by hand. Wrapper objects below SIMULATE the
original reader's return; they are not a real sealed archive qualification.
"""
from dataclasses import replace
import builtins
import json
import os
from pathlib import Path
import random

import pytest

from tests.canonical_collection.test_game_bundle import _fixture
from werewolf.artifact_io import VerifiedArtifact, canonical_json_bytes, sha256_bytes
from werewolf.artifact_io.canonical import _build_manifest
from werewolf.canonical_collection.game_bundle import (
    VerifiedCanonicalGameBundle, _bundle_files, _manifest_fields, _replay_result, _history_from_records,
)
from werewolf.canonical_collection.attempt_ledger import (
    construct_attempt_claim, construct_attempt_terminal, TerminalOutcome,
)
from werewolf.canonical_collection.public_history import freeze_public_event_history
from werewolf.canonical_collection.speech import (
    construct_v1_speech_annotation, V1SpeechAction, V1PerceptionAttempt, V1AnnotationStatus,
    V1_SPEECH_PROMPT_VERSION, V1_SPEECH_PARSER_VERSION,
)
from werewolf.canonical_collection.pre import construct_authoritative_pre_prefix, construct_speaker_pre_belief_handoff
from werewolf.canonical_collection.trajectory_evidence import (
    construct_submitted_gameplay_action, construct_private_replay_evidence,
)
from werewolf.observer_dtrs.action_evidence_ledger import digest, extend_ledger
from werewolf.observer_dtrs.native_action_link import link_native_evidence, evaluate_day_wolf_exile, Status
from werewolf.observer_dtrs.online_context import ShadowRequest, build_shadow_request

P = tuple(f'player{i}' for i in range(1,8))
LEDGER = 'artificial-native-link-fixture'


@pytest.fixture(autouse=True)
def reject_all_runtime_execution(monkeypatch):
    # Applies BEFORE even the manual fixture constructors run.
    def deny(*args,**kwargs):raise AssertionError('Actor/Parser/game/replay/repair execution prohibited')
    from werewolf.agents.gpt_agent import GPTAgent
    from werewolf.speech.speech_perceiver import SpeechPerceiver
    from werewolf.canonical_collection import game_bundle, attempt_ledger
    monkeypatch.setattr(GPTAgent,'_generate_public_speech',deny)
    monkeypatch.setattr(SpeechPerceiver,'parse_with_audit',deny)
    monkeypatch.setattr(game_bundle,'validate_deterministic_replay',deny)
    monkeypatch.setattr(game_bundle,'validate_canonical_game_bundle',deny)
    monkeypatch.setattr(game_bundle,'publish_canonical_game_bundle',deny)
    monkeypatch.setattr(attempt_ledger,'recover_attempt_ledger',deny)


def annotation(history, status, actions):
    attempt = V1PerceptionAttempt(attempt_index=1, call_id='HANDWRITTEN-NO-PARSER-CALL',
        backend_id='ARTIFICIAL', model_id='ARTIFICIAL', prompt_version=V1_SPEECH_PROMPT_VERSION, parser_version=V1_SPEECH_PARSER_VERSION,
        status=status, raw_response='HANDWRITTEN FIXTURE NOT PARSER OUTPUT',
        error_category='ARTIFICIAL_FAILURE' if status is V1AnnotationStatus.ERROR else None,
        error_message='no parser called' if status is V1AnnotationStatus.ERROR else None)
    return construct_v1_speech_annotation(history,status=status,actions=actions,attempts=(attempt,))


def wrap(plan, claim, evidence):
    replay = _replay_result(evidence.public_event_stream.digest, 'ARTIFICIAL-REPLAY-RECORD-NOT-EXECUTED')
    fields = _manifest_fields(plan=plan,claim=claim,evidence=evidence,replay_result=replay)
    manifest, _ = _build_manifest(fields,_bundle_files(evidence),'manifest.json')
    artifact = VerifiedArtifact(Path('/ARTIFICIAL-NOT-AN-OPENED-ARCHIVE'), 'manifest.json', manifest, manifest['manifest_digest'])
    return VerifiedCanonicalGameBundle(artifact=artifact,game_id=evidence.game_id,
        public_event_stream=evidence.public_event_stream,authoritative_pre_prefixes=evidence.authoritative_pre_prefixes,
        belief_observations=evidence.belief_observations,speech_annotations_v1=evidence.speech_annotations_v1,
        call_budget_summary=evidence.call_budget_summary,parser_summary=evidence.parser_summary,
        submitted_gameplay_actions=evidence.submitted_gameplay_actions,backend_call_evidence=evidence.backend_call_evidence,
        private_replay_evidence=evidence.private_replay_evidence,replay_result=replay)


def records_for(plan, claim, evidence):
    pre = evidence.authoritative_pre_prefixes[0]
    # One fixed injected plan; no draw, choice algorithm, runtime intent or fallback.
    candidate={'plan':{'action':'ACCUSE_WOLF','target':'player3'},
               'expected_v1':['player1','point_as_werewolf','player3']}
    op={'opportunity_id':'op','protocol_ref':digest('UNFROZEN-CONTROLLED-NATIVE-LINK'),
        'game_id':pre.game_id,'game_attempt_id':claim.attempt_id,'boundary_id':pre.boundary_id,
        'prefix_digest':pre.prefix_digest,'step_index':pre.step_index,'actor':pre.current_speaker,
        'day':pre.public_temporal_state.day,'phase':pre.public_temporal_state.phase.value,
        'public_history_digest':pre.public_event_digest,'capture_digest':digest(pre.to_record()),
        'candidate_order_version':'fixed-injected-test-support-not-full-sampling-qualification','candidates':[candidate]}
    rows=extend_ledger((),ledger_id=LEDGER,operation_id='capture',kind='OPPORTUNITY',payload=op)
    assigned={'assignment_id':'assignment','opportunity_ref':rows[-1]['record_digest'],
        'candidate_id':digest(candidate),'support':[digest(candidate)],'declared_probabilities':None,
        'treatment_ref':digest('ARTIFICIAL-NOT-A-TRIAL'),'assignment_source':'INJECTED_FIXED_NOT_RANDOMIZED'}
    rows=extend_ledger(rows,ledger_id=LEDGER,operation_id='assignment',kind='ASSIGNMENT',payload=assigned)
    return rows


def fixture(day='resolved', semantic='exact', terminal=True):
    plan,claim,evidence,_unused_executor = _fixture()  # Pure constructors only; NEVER call executor.execute.
    raw=[{'event_id':'event-0','event_index':0,'event_type':'phase_change','day':0,'phase':'night'},
         {'event_id':'event-1','event_index':1,'event_type':'death_announcement','dead_players':[]},
         {'event_id':'event-2','event_index':2,'event_type':'phase_change','day':1,'phase':'discussion'},
         {'event_id':'event-3','event_index':3,'event_type':'turn_start','speaker':'player1'},
         {'event_id':'event-4','event_index':4,'event_type':'public_speech','speaker':'player1','raw_text':'player3 is suspicious'}]
    speech_history=freeze_public_event_history(raw)
    if semantic!='exact':
        status=V1AnnotationStatus.OK
        acts=(V1SpeechAction('player1','point_as_werewolf','player3'),)
        if semantic=='multi':acts += (V1SpeechAction('player1','support','player2'),)
        elif semantic=='mismatch':acts=(V1SpeechAction('player1','point_as_non_werewolf','player3'),)
        elif semantic=='empty':status=V1AnnotationStatus.NO_ACTION;acts=()
        elif semantic=='error':status=V1AnnotationStatus.ERROR;acts=()
        evidence=replace(evidence,speech_annotations_v1=(annotation(speech_history,status,acts),))
    def event(kind,**payload):
        raw.append(dict(event_id=f'event-{len(raw)}',event_index=len(raw),event_type=kind,**payload))
    def ballot(tie=False):
        targets=('player2','player3','player1','player1','player2','player3',None) if tie else ('player3','player3','player2','player3','player3','player3','player3')
        event('vote_result',votes=[{'voter':p,'target':t} for p,t in zip(P,targets)])
    if day!='missing':
        event('phase_change',day=1,phase='vote');ballot(day in ('pk_open','pk_resolved','pk_empty'))
        if day in ('pk_open','pk_resolved','pk_empty'):
            event('exile_result',exiled_players=[]);event('phase_change',day=1,phase='pk_discussion')
            if day!='pk_open':
                event('phase_change',day=1,phase='pk_vote');ballot(day=='pk_empty')
                event('exile_result',exiled_players=[] if day=='pk_empty' else ['player3'])
        else:
            event('exile_result',exiled_players=['player2'] if day=='conflict' else ['player3'])
    history=freeze_public_event_history(raw)
    private=evidence.private_replay_evidence
    private=construct_private_replay_evidence(game_id=evidence.game_id,seed=claim.seed,
        role_assignment=dict(private.role_assignment),runtime_configuration=private.runtime_configuration,
        initial_runtime_state=private.initial_runtime_state,
        replay_inputs={'winner':'Werewolf','fixture':'ARTIFICIAL_NOT_GAME_EXECUTED'},
        expected_public_event_digest=history.digest)
    evidence=replace(evidence,public_event_stream=history,private_replay_evidence=private)
    rows=records_for(plan,claim,evidence);bundle=wrap(plan,claim,evidence)
    closed=construct_attempt_terminal(plan,claim,outcome=TerminalOutcome.CANONICAL_SUCCESS,
        terminal_timestamp_utc='2026-09-03T02:02:03Z',canonical_game_bundle_id=bundle.game_id,
        canonical_game_bundle_digest=bundle.manifest_digest) if terminal else None
    return dict(records=rows,ledger_id=LEDGER,assignment_ref=rows[-1]['record_digest'],plan=plan,claim=claim,
                bundle=bundle,expected_bundle_digest=bundle.manifest_digest,attempt_terminal=closed),evidence


def run(args):return link_native_evidence(**args)


def rewrap(args,evidence):
    a=dict(args);a['bundle']=wrap(a['plan'],a['claim'],evidence);a['expected_bundle_digest']=a['bundle'].manifest_digest
    if a['attempt_terminal'] is not None:
        a['attempt_terminal']=construct_attempt_terminal(a['plan'],a['claim'],outcome=TerminalOutcome.CANONICAL_SUCCESS,
            terminal_timestamp_utc='2026-09-03T02:02:03Z',canonical_game_bundle_id=a['bundle'].game_id,
            canonical_game_bundle_digest=a['bundle'].manifest_digest)
    return a


def research_commit(args,status='COMMIT_UNKNOWN',wrong_event=False):
    rows=args['records'];a=args['assignment_ref']
    def add(name,kind,p):
        nonlocal rows
        rows=extend_ledger(rows,ledger_id=LEDGER,operation_id=name,kind=kind,payload=p)
        return rows[-1]['record_digest']
    started=add('start','EXECUTION',dict(assignment_ref=a,status='STARTED',execution_id='manual',implementation_ref=digest('manual')))
    verified=add('verified','EXECUTION',dict(assignment_ref=a,status='VERIFIED',started_ref=started,
        evidence_ref=digest('manual-verified'),speech_digest=sha256_bytes(args['bundle'].submitted_gameplay_actions[0].action_payload.to_value().encode('utf-8')),
        actions=[['player1','point_as_werewolf','player3']]))
    reserved=add('reserve','COMMIT_WITNESS',dict(assignment_ref=a,status='ATTEMPT_RESERVED',
        execution_ref=verified,attempt_token='RESEARCH-TOKEN-NEVER-IN-NATIVE-SCHEMA',evidence_ref=digest('manual-intent')))
    if status=='COMMIT_UNKNOWN':
        add('unknown','COMMIT_WITNESS',dict(assignment_ref=a,status=status,reservation_ref=reserved,evidence_ref=digest('unknown')))
    else:
        op=rows[0]['payload'];bundle=args['bundle'];action=bundle.submitted_gameplay_actions[0]
        x={k:op[k] for k in ('game_id','game_attempt_id','boundary_id','prefix_digest','step_index','actor','day','phase')}
        x.update(attempt_token='RESEARCH-TOKEN-NEVER-IN-NATIVE-SCHEMA',readback_digest=digest('manual-readback'))
        if status=='COMMITTED':
            x.update(action_id=action.action_id,event_id='wrong' if wrong_event else 'event-4',
                     annotation_digest=bundle.speech_annotations_v1[0].annotation_digest,
                     post_history_digest=digest('manual-post'))
        else:x.update(public_history_digest=op['public_history_digest'],stopped_executor_digest=digest('manual-stopped'))
        add('claim','COMMIT_WITNESS',dict(assignment_ref=a,status=status,reservation_ref=reserved,external=x,had_persistence_failure=False))
    return {**args,'records':rows}


def test_native_occurrence_and_day_are_observed_but_assignment_token_is_unknown():
    a,_=fixture();r=run(a)
    assert r.submission.status is Status.OBSERVED
    assert r.assignment_link.status is Status.UNKNOWN
    assert r.assignment_link.reason=='NATIVE_SCHEMA_HAS_NO_ASSIGNMENT_TOKEN'
    assert r.semantics.status is Status.OBSERVED and json.loads(r.semantics.evidence_json)['classification']=='EXACT_SINGLE'
    assert r.day.status is Status.OBSERVED
    assert r.completion.status is Status.UNKNOWN
    assert r.completion.reason=='CANONICAL_SUCCESS_NOT_FULL_GAME_TERMINAL_PROOF'
    assert json.loads(r.completion.evidence_json)['canonical_attempt_closed']
    raw=r.to_record();assert not raw['submission_permission'] and not raw['F1_permission'] and not raw['external_exactly_once_proven']


@pytest.mark.parametrize('semantic,category',[('multi','MULTI_ACTION'),('empty','EMPTY_ACTIONS'),('mismatch','SINGLE_MISMATCH'),('error','PARSE_ERROR')])
def test_all_semantics_are_separate_from_native_submission(semantic,category):
    a,_=fixture(semantic=semantic);r=run(a)
    assert r.submission.status is Status.OBSERVED and r.semantics.status is Status.OBSERVED
    assert json.loads(r.semantics.evidence_json)['classification']==category
    assert not json.loads(r.semantics.evidence_json)['exact_single_compliance']
    assert r.assignment_link.status is Status.UNKNOWN and r.day.status is Status.OBSERVED


@pytest.mark.parametrize('day,status',[('missing',Status.UNKNOWN),('pk_open',Status.UNKNOWN),('pk_resolved',Status.OBSERVED),('pk_empty',Status.OBSERVED),('conflict',Status.CONFLICT)])
def test_native_day_resolver_distinguishes_pk_frontier_and_final_empty(day,status):
    a,_=fixture(day=day);r=run(a)
    assert r.day.status is status
    assert r.submission.status is Status.OBSERVED
    if day=='pk_empty':assert json.loads(r.day.evidence_json)['observed_no_exile'] is True
    if day=='pk_open':assert 'no_exile' not in r.day.evidence_json.decode()


@pytest.mark.parametrize('case',['no_bundle','no_anchor','no_claim','no_plan'])
def test_missing_provenance_is_unknown_and_not_a_default_match(case):
    a,_=fixture()
    a[{'no_bundle':'bundle','no_anchor':'expected_bundle_digest','no_claim':'claim','no_plan':'plan'}[case]]=None
    r=run(a)
    assert r.submission.status is Status.UNKNOWN and r.assignment_link.status is Status.UNKNOWN


@pytest.mark.parametrize('part',['claim','bundle_manifest','plan'])
def test_cross_attempt_or_source_conflicts(part):
    a,e=fixture()
    if part=='claim':a['claim']=construct_attempt_claim(a['plan'],ordinal=0,attempt_id='other',claim_timestamp_utc='2026-09-03T01:02:03Z')
    elif part=='plan':a['plan']=replace(a['plan'],source_revision='a'*40)
    else:
        m={**a['bundle'].manifest,'attempt_id':'other'}
        m['manifest_digest']=digest({k:v for k,v in m.items() if k!='manifest_digest'})
        artifact=replace(a['bundle'].artifact,manifest=m,manifest_digest=m['manifest_digest'])
        a['bundle']=replace(a['bundle'],artifact=artifact);a['expected_bundle_digest']=m['manifest_digest']
    r=run(a);assert r.submission.status is Status.CONFLICT
    assert r.day.status is Status.CONFLICT and r.completion.status is Status.CONFLICT


@pytest.mark.parametrize('field,value',[('prefix_digest','a'*64),('step_index',4),('actor','player2'),('phase','pk_discussion'),('public_history_digest','b'*64),('capture_digest','c'*64)])
def test_wrong_pre_research_identity_cannot_be_joined_by_seat_or_phase(field,value):
    a,_=fixture();p=dict(a['records'][0]['payload']);p[field]=value
    if field=='actor':
        p['candidates']=[{**c,'expected_v1':[value,*c['expected_v1'][1:]]} for c in p['candidates']]
    rows=extend_ledger((),ledger_id=LEDGER,operation_id='capture',kind='OPPORTUNITY',payload=p)
    assigned={**a['records'][1]['payload'],'opportunity_ref':rows[0]['record_digest'],
              'candidate_id':digest(p['candidates'][0]),'support':[digest(p['candidates'][0])]}
    rows=extend_ledger(rows,ledger_id=LEDGER,operation_id='assignment',kind='ASSIGNMENT',payload=assigned)
    r=run({**a,'records':rows,'assignment_ref':rows[-1]['record_digest']})
    assert r.submission.status is Status.CONFLICT and r.assignment_link.status is Status.CONFLICT
    assert r.day.status is Status.OBSERVED


@pytest.mark.parametrize('case',['missing','duplicate','event_missing','wrong_text','wrong_handoff','wrong_actor','wrong_step'])
def test_missing_and_conflicting_action_witnesses_are_not_repaired(case):
    a,e=fixture();act=e.submitted_gameplay_actions[0]
    if case=='missing':e=replace(e,submitted_gameplay_actions=())
    elif case=='duplicate':e=replace(e,submitted_gameplay_actions=(act,act))
    else:
        kwargs=dict(action_id=act.action_id,step_index=act.step_index,actor_id=act.actor_id,action_type=act.action_type,
            action_payload=act.action_payload,resulting_public_event_ids=act.resulting_public_event_ids,
            boundary_id=act.boundary_id,speaker_pre_belief_handoff=act.speaker_pre_belief_handoff,fallback_used=False)
        if case=='event_missing':kwargs['resulting_public_event_ids']=('absent',)
        if case=='wrong_text':kwargs['action_payload']='different public speech'
        if case=='wrong_actor':kwargs['actor_id']='player2'
        if case=='wrong_step':kwargs['step_index']=9
        if case=='wrong_handoff':kwargs['speaker_pre_belief_handoff']=replace(act.speaker_pre_belief_handoff,observation_id='wrong')
        e=replace(e,submitted_gameplay_actions=(construct_submitted_gameplay_action(**kwargs),))
    r=run(rewrap(a,e))
    assert r.submission.status is (Status.UNKNOWN if case=='missing' else Status.CONFLICT)
    assert r.day.status is Status.OBSERVED


@pytest.mark.parametrize('case',['missing','duplicate','bad_digest'])
def test_annotation_availability_and_corruption_do_not_erase_native_submission(case):
    a,e=fixture();ann=e.speech_annotations_v1[0]
    anns=() if case=='missing' else (ann,ann) if case=='duplicate' else (replace(ann,annotation_digest='f'*64),)
    r=run(rewrap(a,replace(e,speech_annotations_v1=anns)))
    assert r.submission.status is Status.OBSERVED
    assert r.semantics.status is (Status.UNKNOWN if case=='missing' else Status.CONFLICT)


@pytest.mark.parametrize('case',['tamper','stale_anchor','payload_descriptor'])
def test_digest_or_descriptor_conflicts_stop_all_source_claims(case):
    a,_=fixture();bundle=a['bundle'];m=json.loads(canonical_json_bytes(bundle.manifest))
    if case=='tamper':m['source_revision']='a'*40
    elif case=='stale_anchor':a['expected_bundle_digest']='f'*64
    else:
        m['file_table']['private/submitted_gameplay_actions.jsonl']['sha256']='f'*64
        m['manifest_digest']=digest({k:v for k,v in m.items() if k!='manifest_digest'})
        a['expected_bundle_digest']=m['manifest_digest']
    a['bundle']=replace(bundle,artifact=replace(bundle.artifact,manifest=m,manifest_digest=m['manifest_digest']))
    r=run(a);assert all(getattr(r,k).status is Status.CONFLICT for k in ('submission','assignment_link','semantics','day','completion'))


@pytest.mark.parametrize('claim_status',['COMMIT_UNKNOWN','COMMITTED','NOT_COMMITTED'])
def test_research_commit_state_and_token_never_manufacture_native_attribution(claim_status):
    a,_=fixture();r=run(research_commit(a,claim_status))
    assert r.submission.status is Status.OBSERVED
    assert r.assignment_link.status is (Status.CONFLICT if claim_status=='NOT_COMMITTED' else Status.UNKNOWN)
    assert r.day.status is Status.OBSERVED


def test_wrong_research_committed_event_only_conflicts_attribution_not_observed_native_event():
    a,_=fixture();r=run(research_commit(a,'COMMITTED',wrong_event=True))
    assert r.submission.status is Status.OBSERVED and r.assignment_link.status is Status.CONFLICT
    assert r.semantics.status is Status.OBSERVED


@pytest.mark.parametrize('case',['missing','wrong_attempt','wrong_bundle','bad_replay'])
def test_completed_attempt_is_not_a_full_game_winner_certificate(case):
    a,_=fixture()
    if case=='missing':a['attempt_terminal']=None
    elif case=='wrong_attempt':a['attempt_terminal']=replace(a['attempt_terminal'],attempt_id='other')
    elif case=='wrong_bundle':a['attempt_terminal']=replace(a['attempt_terminal'],canonical_game_bundle_digest='f'*64)
    else:a['bundle']=replace(a['bundle'],replay_result=replace(a['bundle'].replay_result,public_event_digest='f'*64))
    r=run(a)
    assert r.completion.status is (Status.UNKNOWN if case=='missing' else Status.CONFLICT)
    assert r.submission.status is Status.OBSERVED


def test_no_io_no_model_parser_actor_replay_or_restricted_payload_access(monkeypatch):
    a,_=fixture()
    class Forbidden:
        def __getattribute__(self,name):raise AssertionError('restricted native payload read:'+name)
    a['bundle']=replace(a['bundle'],private_replay_evidence=Forbidden(),belief_observations=Forbidden(),backend_call_evidence=Forbidden())
    def deny(*args,**kwargs):raise AssertionError('forbidden I/O or runtime execution')
    monkeypatch.setattr(builtins,'open',deny);monkeypatch.setattr(os,'open',deny)
    monkeypatch.setattr(Path,'read_bytes',deny);monkeypatch.setattr(Path,'write_bytes',deny)
    from werewolf.agents.gpt_agent import GPTAgent
    from werewolf.speech.speech_perceiver import SpeechPerceiver
    monkeypatch.setattr(GPTAgent,'_generate_public_speech',deny)
    monkeypatch.setattr(SpeechPerceiver,'parse_with_audit',deny)
    import werewolf.canonical_collection.attempt_ledger as attempts
    monkeypatch.setattr(attempts,'recover_attempt_ledger',deny)
    state=random.getstate()
    first=run(a).to_record();second=run(a).to_record()
    assert first==second and random.getstate()==state
    assert first['submission']['status']=='OBSERVED'
    assert first['completion']['status']=='UNKNOWN'
    assert not any(x in json.dumps(first) for x in ['role_assignment','suspicion_support','private_prompt','actual_role'])


def test_inputs_are_unchanged_outputs_have_independent_immutable_evidence():
    a,_=fixture();before=canonical_json_bytes(a['records']);manifest=canonical_json_bytes(a['bundle'].manifest)
    r=run(a);out=r.to_record();out['sources']['game_id']='tampered'
    assert canonical_json_bytes(a['records'])==before and canonical_json_bytes(a['bundle'].manifest)==manifest
    assert run(a).to_record()==r.to_record() and r.to_record()['sources']['game_id']!='tampered'


def test_verified_execution_text_contradiction_is_independent_of_native_submission():
    a,_=fixture();a=research_commit(a)
    rebuilt=();refs={}
    for row in a['records']:
        p=dict(row['payload'])
        for k in ('opportunity_ref','assignment_ref','started_ref','execution_ref','reservation_ref'):
            if p.get(k) in refs:p[k]=refs[p[k]]
        if row['kind']=='EXECUTION' and p['status']=='VERIFIED':p['speech_digest']=digest('different text')
        rebuilt=extend_ledger(rebuilt,ledger_id=LEDGER,operation_id=row['operation_id'],kind=row['kind'],payload=p)
        refs[row['record_digest']]=rebuilt[-1]['record_digest']
    r=run({**a,'records':rebuilt,'assignment_ref':refs[a['assignment_ref']]})
    assert r.submission.status is Status.OBSERVED and r.assignment_link.status is Status.CONFLICT
    assert r.assignment_link.reason=='RESEARCH_VERIFIED_EXECUTION_NATIVE_CONFLICT'


def test_research_single_verified_action_does_not_match_native_multi_action():
    a,_=fixture(semantic='multi');r=run(research_commit(a))
    assert r.submission.status is Status.OBSERVED and r.assignment_link.status is Status.CONFLICT
    assert json.loads(r.semantics.evidence_json)['classification']=='MULTI_ACTION'


def test_duplicate_boundary_does_not_collapse_two_native_pre_records():
    a,e=fixture();r=run(rewrap(a,replace(e,authoritative_pre_prefixes=e.authoritative_pre_prefixes*2)))
    assert r.submission.status is Status.CONFLICT and r.day.status is Status.OBSERVED


def test_injected_native_completion_or_token_flag_does_not_upgrade_missing_proof():
    a,_=fixture();m={**a['bundle'].manifest,'full_game_verified':True,'attempt_token':'invented'}
    m['manifest_digest']=digest({k:v for k,v in m.items() if k!='manifest_digest'})
    a['bundle']=replace(a['bundle'],artifact=replace(a['bundle'].artifact,manifest=m,manifest_digest=m['manifest_digest']))
    a['expected_bundle_digest']=m['manifest_digest']
    r=run(a)
    # Even an invented metadata flag is never a native terminal certificate.
    # Terminal still binds the old bundle, so this is additionally a conflict.
    assert r.assignment_link.status is Status.CONFLICT
    assert r.completion.status is Status.CONFLICT
    assert r.submission.reason=='NATIVE_MANIFEST_SCHEMA_CONFLICT'


def test_fake_handoff_digest_is_rejected_even_with_matching_seats():
    a,e=fixture();act=e.submitted_gameplay_actions[0]
    h=replace(act.speaker_pre_belief_handoff,handoff_digest='f'*64)
    act=replace(act,speaker_pre_belief_handoff=h)
    r=run(rewrap(a,replace(e,submitted_gameplay_actions=(act,))))
    assert r.submission.status is Status.CONFLICT


def test_research_record_digest_tamper_rejected_without_source_fallback():
    a,_=fixture();rows=list(a['records']);rows[0]={**rows[0],'record_digest':'f'*64}
    r=run({**a,'records':tuple(rows)})
    assert all(getattr(r,k).status is Status.CONFLICT for k in ('submission','assignment_link','day','completion'))


def test_unknown_completion_contains_input_refs_without_reading_a_winner():
    a,_=fixture();r=run(a);v=r.to_record()
    assert v['sources']['attempt_terminal_input_digest']==digest(a['attempt_terminal'].to_record())
    assert v['sources']['expected_bundle_digest']==a['bundle'].manifest_digest
    assert len(v['sources']['native_payloads'])==4
    assert 'winner' not in v['completion']['evidence']
    assert not v['sources']['restricted_belief_or_replay_payloads_read']


@pytest.mark.parametrize('case',['illegal_plan','plan_v1_mismatch'])
def test_bad_declared_plan_does_not_erase_independently_observed_native_submission(case):
    a,_=fixture();p=dict(a['records'][0]['payload'])
    c={'plan':{'action':'SUPPORT','target':'player3'},'expected_v1':['player1','support','player3']} if case=='illegal_plan' else {
        'plan':{'action':'ACCUSE_WOLF','target':'player3'},'expected_v1':['player1','point_as_non_werewolf','player3']}
    p['candidates']=[c]
    rows=extend_ledger((),ledger_id=LEDGER,operation_id='capture',kind='OPPORTUNITY',payload=p)
    assigned={**a['records'][1]['payload'],'opportunity_ref':rows[-1]['record_digest'],
              'candidate_id':digest(c),'support':[digest(c)]}
    rows=extend_ledger(rows,ledger_id=LEDGER,operation_id='assignment',kind='ASSIGNMENT',payload=assigned)
    r=run({**a,'records':rows,'assignment_ref':rows[-1]['record_digest']})
    assert r.submission.status is Status.OBSERVED
    assert r.assignment_link.status is Status.CONFLICT and r.semantics.status is Status.CONFLICT
    assert r.day.status is Status.OBSERVED


def outcome_fixture(kind='normal_exile', *, exiled='player3', semantic='exact'):
    """Artificial public facts and actor view; NO real observation custody."""
    from tests.observer_dtrs.test_public_behavior_research import _complete_public_day_history
    from tests.observer_dtrs.test_shadow import observation
    a, e = fixture(day='missing', semantic=semantic)
    raw = e.public_event_stream.to_records()
    suffix = _complete_public_day_history(kind).to_records()
    start = next(i for i, r in enumerate(suffix) if r['event_type'] == 'phase_change' and r['phase'] == 'vote')
    for record in suffix[start:]:
        record = dict(record)
        if kind == 'normal_exile':
            if record['event_type'] == 'vote_result':
                other = next(p for p in P if p != exiled)
                record['votes'] = [{'voter': p, 'target': other if p == exiled else exiled} for p in P]
            elif record['event_type'] == 'exile_result':
                record['exiled_players'] = [exiled]
        record.update(event_id=f'event-{len(raw)}', event_index=len(raw))
        raw.append(record)
    e = replace(e, public_event_stream=_history_from_records(raw))
    a = rewrap(a, e)
    pre = e.authoritative_pre_prefixes[0]
    view = observation(pre)
    request = build_shadow_request(pre, view, shadow_run_id='artificial-shadow', attempt_id=a['claim'].attempt_id)
    return {**a, 'request': request, 'acting_observation': view}, e


def evaluate(args):
    return evaluate_day_wolf_exile(**args).to_record()


@pytest.mark.parametrize('exiled', ['player1', 'player2', 'player3'])
def test_wolf_actor_teammate_or_nonwolf_exile_does_not_certify_artificial_team(exiled):
    a, _ = outcome_fixture(exiled=exiled)
    out = evaluate(a)
    evidence = out['evidence']
    assert out['status'] == 'UNKNOWN' and evidence['E'] is None
    assert evidence['public_day']['status'] == 'OBSERVED'
    assert evidence['public_day']['evidence']['exiled_seats'] == [exiled]
    team = evidence['PRE_wolf_team']
    assert team['status'] == 'UNKNOWN' and team['reason'] == 'ACTOR_OBSERVATION_ORIGIN_UNVERIFIED'
    assert team['evidence']['content_binding_checked']
    # Even when one is dead after exile, the claimed set is frozen at PRE.
    assert team['evidence']['claimed_PRE_alive_wolf_ids'] == ['player1', 'player2']
    assert not team['evidence']['upstream_origin_verified']
    assert evidence['native_link']['assignment_link']['status'] == 'UNKNOWN'
    assert evidence['native_link']['completion']['status'] == 'UNKNOWN'


@pytest.mark.parametrize('kind', ['normal_abstain', 'pk_tie', 'pk_abstain'])
def test_complete_final_empty_exile_is_a_fact_but_not_qualified_E_zero(kind):
    a, _ = outcome_fixture(kind)
    out = evaluate(a)['evidence']
    assert out['public_day']['status'] == 'OBSERVED'
    assert out['public_day']['evidence']['observed_no_exile']
    assert out['public_day']['evidence']['exiled_seats'] == []
    assert out['E'] is None


def test_normal_tie_without_completed_pk_cannot_be_zero():
    a, _ = outcome_fixture('pk_incomplete')
    out = evaluate(a)
    assert out['status'] == 'UNKNOWN' and out['evidence']['E'] is None
    assert out['evidence']['public_day']['status'] == 'UNKNOWN'
    assert out['evidence']['native_link']['day']['status'] == 'UNKNOWN'


@pytest.mark.parametrize('kind', ['normal_missing_voter', 'pk_illegal_target', 'self_vote', 'pk_missing_speaker'])
def test_stronger_public_rejection_never_falls_back_to_observed_native_day(kind):
    a, e = outcome_fixture('pk_exile' if kind == 'pk_missing_speaker' else
                           'normal_exile' if kind == 'self_vote' else kind)
    if kind in ('self_vote', 'pk_missing_speaker'):
        raw = e.public_event_stream.to_records()
        if kind == 'self_vote':
            vote = next(r for r in raw if r['event_type'] == 'vote_result')
            vote['votes'][0]['target'] = 'player1'
        else:
            raw = [r for r in raw if not (r['event_type'] in ('turn_start', 'public_speech') and r['phase'] == 'pk_discussion'
                                         and r['speaker'] == 'player2')]
            for i, r in enumerate(raw):
                r.update(event_id=f'event-{i}', event_index=i)
        a = rewrap(a, replace(e, public_event_stream=_history_from_records(raw)))
    out = evaluate(a)
    assert out['evidence']['native_link']['day']['status'] == 'OBSERVED'
    assert out['status'] == 'CONFLICT' and out['evidence']['E'] is None
    assert out['evidence']['public_day']['reason'] == 'PUBLIC_DAY_WITNESSES_DISAGREE'


@pytest.mark.parametrize('field,value', [('event_id', 'wrong'), ('exiled_seats', ['player1']),
    ('day', 2), ('final_phase', 'pk_vote'), ('day_events_digest', 'f'*64)])
def test_contradictory_day_identity_and_source_are_conflicts(monkeypatch, field, value):
    import werewolf.observer_dtrs.native_action_link as module
    a, _ = outcome_fixture()
    original = module._day
    def contradictory(bundle, op):
        old = original(bundle, op)
        return replace(old, evidence_json=canonical_json_bytes({**json.loads(old.evidence_json), field: value}))
    monkeypatch.setattr(module, '_day', contradictory)
    out = evaluate(a)
    assert out['status'] == 'CONFLICT' and out['evidence']['E'] is None
    assert out['evidence']['public_day']['status'] == 'CONFLICT'
    assert out['evidence']['public_day']['reason'] == 'PUBLIC_DAY_WITNESS_IDENTITY_CONFLICT'


@pytest.mark.parametrize('field,value', [('game_id', 'other'), ('boundary_id', 'other'),
    ('prefix_digest', 'f'*64), ('step_index', 7), ('current_speaker', 'player2')])
def test_wrong_request_pre_identity_cannot_support_outcome(field, value):
    a, _ = outcome_fixture()
    # Corrupt a copied request without invoking any runtime or original reader.
    pre = replace(a['request'].prefix, **{field: value})
    object.__setattr__(a['request'], 'prefix', pre)
    out = evaluate(a)
    assert out['status'] == 'CONFLICT' and out['evidence']['E'] is None
    assert out['evidence']['public_day']['status'] == 'OBSERVED'


def test_wrong_request_attempt_is_separate_from_observable_day():
    a, _ = outcome_fixture()
    a['request'] = replace(a['request'], attempt_id='other')
    out = evaluate(a)
    assert out['status'] == 'CONFLICT' and out['evidence']['E'] is None
    assert out['evidence']['PRE_wolf_team']['reason'] == 'OUTCOME_PRE_ATTEMPT_CONFLICT'
    assert out['evidence']['public_day']['status'] == 'OBSERVED'


@pytest.mark.parametrize('kind', ['missing_view', 'seat_tuple', 'context', 'constructed_request', 'claimed_trust'])
def test_declared_wolves_types_and_self_asserted_custody_never_qualify_E(kind):
    a, _ = outcome_fixture()
    if kind == 'missing_view':
        a['acting_observation'] = None
    elif kind == 'seat_tuple':
        a['request'] = ('player1', 'player2')
    elif kind == 'context':
        a['request'] = a['request'].contexts[0]
    elif kind == 'constructed_request':
        r = a['request']
        a['request'] = ShadowRequest(r.prefix, r.known_wolf_ids, r.shadow_run_id, r.attempt_id)
    else:
        a['acting_observation'].update(trusted=True, source_digest='f'*64,
                                       original_execution_verified=True)
    out = evaluate(a)
    assert out['status'] == 'UNKNOWN' and out['evidence']['E'] is None
    assert not out['evidence']['PRE_wolf_team']['evidence']['upstream_origin_verified']


@pytest.mark.parametrize('kind', ['actor', 'identity', 'alive', 'team'])
def test_actor_view_content_must_match_native_pre_before_origin_remains_unknown(kind):
    a, _ = outcome_fixture()
    view = a['acting_observation']
    if kind == 'actor':view['observer_id'] = 2
    elif kind == 'identity':view['identity'] = 'Villager'
    elif kind == 'alive':view['authoritative_public_state']['alive_players'] = [1, 2, 3]
    else:view['game_log'][0].content['wolf_team'] = [1, 3]
    out = evaluate(a)
    assert out['status'] == 'CONFLICT' and out['evidence']['E'] is None
    assert out['evidence']['public_day']['status'] == 'OBSERVED'


@pytest.mark.parametrize('semantic', ['multi', 'mismatch', 'empty', 'error', 'missing_submission'])
def test_observable_day_never_manufactures_submission_or_semantic_success(semantic):
    a, e = outcome_fixture(semantic='exact' if semantic == 'missing_submission' else semantic)
    if semantic == 'missing_submission':a = rewrap(a, replace(e, submitted_gameplay_actions=()))
    expected = run({k: v for k, v in a.items() if k not in ('request', 'acting_observation')}).to_record()
    out = evaluate(a)['evidence']
    assert out['public_day']['status'] == 'OBSERVED' and out['E'] is None
    assert out['native_link'] == expected
    assert out['native_link']['assignment_link']['status'] == 'UNKNOWN'
    assert not out['randomized_assignment_inferred'] and not out['continuation_verified']
    if semantic == 'missing_submission':assert out['native_link']['submission']['status'] == 'UNKNOWN'
    else:assert not out['native_link']['semantics']['evidence']['exact_single_compliance']


def outcome_records(a, op):
    rows = extend_ledger((), ledger_id=LEDGER, operation_id='capture', kind='OPPORTUNITY', payload=op)
    candidate = digest(op['candidates'][0])
    assigned = {**a['records'][1]['payload'], 'opportunity_ref': rows[-1]['record_digest'],
                'candidate_id': candidate, 'support': [candidate]}
    return extend_ledger(rows, ledger_id=LEDGER, operation_id='assignment', kind='ASSIGNMENT', payload=assigned)


@pytest.mark.parametrize('field,value', [('game_id', 'other'), ('game_attempt_id', 'other'),
    ('boundary_id', 'other'), ('prefix_digest', 'f'*64), ('actor', 'player2')])
def test_wrong_native_opportunity_cannot_borrow_a_day_outcome(field, value):
    a, _ = outcome_fixture()
    op = {**a['records'][0]['payload'], field: value}
    if field == 'actor':
        op['candidates'] = [{**op['candidates'][0], 'expected_v1': ['player2', 'point_as_werewolf', 'player3']}]
    a['records'] = outcome_records(a, op)
    a['assignment_ref'] = a['records'][-1]['record_digest']
    out = evaluate(a)
    assert out['status'] == 'CONFLICT' and out['evidence']['E'] is None


def test_complete_public_witness_cannot_override_unknown_native_day(monkeypatch):
    import werewolf.observer_dtrs.native_action_link as module
    a, _ = outcome_fixture()
    monkeypatch.setattr(module, '_day', lambda *args: module._finding(Status.UNKNOWN, 'ARTIFICIAL_MISSING'))
    out = evaluate(a)
    assert out['status'] == 'CONFLICT' and out['evidence']['E'] is None
    assert out['evidence']['public_day']['reason'] == 'PUBLIC_DAY_WITNESSES_DISAGREE'


def test_two_pre_references_share_one_day_fact_without_inventing_trial_units():
    from tests.observer_dtrs.test_shadow import observation
    a, e = outcome_fixture()
    raw = e.public_event_stream.to_records()
    raw[5:5] = [dict(event_type='turn_start', day=1, phase='discussion', speaker='player2'),
                dict(event_type='public_speech', day=1, phase='discussion', speaker='player2', raw_text='Artificial reply.')]
    for i, r in enumerate(raw):r.update(event_id=f'event-{i}', event_index=i)
    history = _history_from_records(raw)
    pre = construct_authoritative_pre_prefix(game_id=e.game_id, boundary_id='boundary-005', step_index=5,
        report_trigger_id='pre-public-speech-005', current_speaker='player2', alive_observer_ids=P,
        public_event_history=_history_from_records(raw[:6]), v1_annotations=e.speech_annotations_v1,
        belief_observation_ids_by_observer={p: 'second-observation-'+p for p in P})
    a = rewrap(a, replace(e, public_event_stream=history,
                         authoritative_pre_prefixes=(*e.authoritative_pre_prefixes, pre)))
    op = {**a['records'][0]['payload'], 'boundary_id': pre.boundary_id, 'prefix_digest': pre.prefix_digest,
          'step_index': pre.step_index, 'actor': pre.current_speaker,
          'public_history_digest': pre.public_event_digest, 'capture_digest': digest(pre.to_record()),
          'candidates': [{'plan': {'action': 'ACCUSE_WOLF', 'target': 'player3'},
                          'expected_v1': ['player2', 'point_as_werewolf', 'player3']}]}
    rows = outcome_records(a, op)
    view = observation(pre)
    second = {**a, 'records': rows, 'assignment_ref': rows[-1]['record_digest'], 'acting_observation': view,
              'request': build_shadow_request(pre, view, shadow_run_id='artificial-shadow', attempt_id=a['claim'].attempt_id)}
    first, second = evaluate(a)['evidence'], evaluate(second)['evidence']
    assert first['public_day']['status'] == second['public_day']['status'] == 'OBSERVED'
    assert first['public_day']['evidence']['shared_day_fact_digest'] == second['public_day']['evidence']['shared_day_fact_digest']
    assert first['E'] is second['E'] is None
    assert not first['randomized_assignment_inferred'] and not second['randomized_assignment_inferred']


def test_outcome_is_pure_and_does_not_access_private_sources_or_upgrade_native_findings(monkeypatch):
    from copy import deepcopy
    a, _ = outcome_fixture()
    class Forbidden:
        def __getattribute__(self, name):raise AssertionError('private source accessed: '+name)
    a['bundle'] = replace(a['bundle'], private_replay_evidence=Forbidden(),
        belief_observations=Forbidden(), backend_call_evidence=Forbidden())
    before_records = canonical_json_bytes(a['records'])
    before_manifest = canonical_json_bytes(a['bundle'].manifest)
    before_view = deepcopy(a['acting_observation'])
    def deny(*args, **kwargs):raise AssertionError('I/O prohibited')
    monkeypatch.setattr(builtins, 'open', deny);monkeypatch.setattr(os, 'open', deny)
    monkeypatch.setattr(Path, 'read_bytes', deny);monkeypatch.setattr(Path, 'write_bytes', deny)
    state = random.getstate()
    first = evaluate(a)
    assert first == evaluate(a) and random.getstate() == state
    assert canonical_json_bytes(a['records']) == before_records
    assert canonical_json_bytes(a['bundle'].manifest) == before_manifest
    assert a['acting_observation'] == before_view
    assert first['evidence']['public_day']['status'] == 'OBSERVED'
    assert first['evidence']['E'] is None
    assert not first['evidence']['native_link']['submission_permission']
    assert not first['evidence']['native_link']['external_custody_verified_by_linker']


def test_failed_assignment_execution_does_not_erase_day_or_fabricate_submission():
    a, e = outcome_fixture()
    a = rewrap(a, replace(e, submitted_gameplay_actions=()))
    started = extend_ledger(a['records'], ledger_id=LEDGER, operation_id='start', kind='EXECUTION',
        payload=dict(assignment_ref=a['assignment_ref'], status='STARTED', execution_id='manual',
                     implementation_ref=digest('ARTIFICIAL-NO-EXECUTION')))
    a['records'] = extend_ledger(started, ledger_id=LEDGER, operation_id='failed', kind='EXECUTION',
        payload=dict(assignment_ref=a['assignment_ref'], status='FAILED', started_ref=started[-1]['record_digest'],
                     evidence_ref=digest('ARTIFICIAL-FAILURE'), speech_digest=None, actions=None, error_code='ACTOR_FAILURE'))
    out = evaluate(a)['evidence']
    assert out['public_day']['status'] == 'OBSERVED' and out['E'] is None
    assert out['native_link']['submission']['status'] == 'UNKNOWN'
    assert out['native_link']['semantics']['status'] == 'UNKNOWN'
    assert out['native_link']['assignment_link']['status'] == 'UNKNOWN'
    assert out['native_link']['sources']['research_input_digest'] == digest(a['records'])


@pytest.mark.parametrize('case', ['missing_prefix', 'missing_bundle', 'missing_anchor'])
def test_missing_source_is_unknown_without_turning_absence_into_conflict_or_E(case):
    a, e = outcome_fixture()
    if case == 'missing_prefix':a = rewrap(a, replace(e, authoritative_pre_prefixes=()))
    elif case == 'missing_bundle':a['bundle'] = None
    else:a['expected_bundle_digest'] = None
    out = evaluate(a)
    assert out['status'] == 'UNKNOWN' and out['evidence']['E'] is None
    if case == 'missing_prefix':
        assert out['evidence']['public_day']['status'] == 'OBSERVED'
        assert out['evidence']['PRE_wolf_team']['reason'] == 'OUTCOME_NATIVE_PRE_MISSING'
    else:assert out['evidence']['public_day']['status'] == 'UNKNOWN'
