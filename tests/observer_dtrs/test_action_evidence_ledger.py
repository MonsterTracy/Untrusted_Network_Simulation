"""Pure state contracts and artificial journal faults; no game/env/services."""
from copy import deepcopy
from dataclasses import asdict
import json
import random
from unittest.mock import patch

import numpy as np
import pytest

from tests.observer_dtrs.action_evidence_fixture import (
    LEDGER,FixtureJournal,FixtureFault,FakeWorld,opportunity,assign,execution,reservation,
    committed,unsubmitted,day,view,
)
from werewolf.artifact_io import canonical_json_bytes
from werewolf.observer_dtrs.action_evidence_ledger import digest,extend_ledger,validate_ledger,LedgerError


def ready(tmp_path):
    j=FixtureJournal(tmp_path/'journal.jsonl');op,a,expected=assign(j);e=execution(j,a,expected)
    return j,op,a,e


def test_complete_chain_has_one_intent_and_observed_empty_exile_not_reward(tmp_path):
    j,op,a,e=ready(tmp_path); world=FakeWorld();r=reservation(j,a,e)
    assert world.calls==0
    receipt=world.commit(op,r['payload']['attempt_token'])
    j.append('commit','COMMIT_WITNESS',committed(a,r,receipt))
    j.append('day','DAY_WITNESS',day(a,op))
    result=validate_ledger(j.records,ledger_id=LEDGER,expected_head=j.head)
    assert len(result.assignments)==1 and result.assignments[0].commit_intents==1
    assert result.assignments[0].commit_status=='COMMITTED' and result.assignments[0].observed_no_exile is True
    assert len(result.shared_observed_days)==1 and result.external_exactly_once_proven is False
    assert result.external_evidence_truth_verified is False and world.calls==1


@pytest.mark.parametrize('failure',('ACTOR_FAILURE','PARSER_FAILURE','SEMANTIC_MISMATCH'))
def test_execution_failure_keeps_assignment_and_cannot_reserve(tmp_path,failure):
    j=FixtureJournal(tmp_path/'journal.jsonl');op,a,expected=assign(j)
    e=execution(j,a,expected,failure=failure)
    with pytest.raises(LedgerError):reservation(j,a,e)
    j.append('missing','DAY_WITNESS',day(a,op,missing=True))
    v=validate_ledger(j.records,ledger_id=LEDGER)
    assert len(v.assignments)==1 and v.assignments[0].commit_intents==0
    assert v.assignments[0].day_status=='MISSING' and v.assignments[0].observed_no_exile is None
    if failure=='SEMANTIC_MISMATCH':assert len(e['payload']['actions'])==2


def test_assignment_survives_interruption_before_execution(tmp_path):
    j=FixtureJournal(tmp_path/'journal.jsonl');op,a,_=assign(j);h=j.head
    recovered=FixtureJournal.recover(j.path,expected_head=h)
    recovered.append('missing','DAY_WITNESS',day(a,op,missing=True))
    assert view(recovered)['assignments'][0]['execution_status']=='NOT_STARTED'


def test_duplicate_assignment_is_not_redrawn(tmp_path):
    j=FixtureJournal(tmp_path/'journal.jsonl');assign(j)
    p=deepcopy(j.records[-1]['payload']);p['assignment_id']='another-id'
    with pytest.raises(LedgerError,match='duplicate assignment'):j.append('redraw','ASSIGNMENT',p)
    assert len(validate_ledger(j.records,ledger_id=LEDGER).assignments)==1


def test_identical_operation_is_idempotent_but_changed_content_conflicts(tmp_path):
    j=FixtureJournal(tmp_path/'journal.jsonl');assign(j)
    last=j.records[-1];raw=j.path.read_bytes();h=j.head
    j.append(last['operation_id'],last['kind'],last['payload'])
    assert j.path.read_bytes()==raw and j.head==h
    p=deepcopy(last['payload']);p['assignment_id']='changed'
    with pytest.raises(LedgerError,match='operation identity'):j.append(last['operation_id'],last['kind'],p)


def test_same_physical_pre_cannot_hide_behind_new_opportunity_id(tmp_path):
    j=FixtureJournal(tmp_path/'journal.jsonl');assign(j)
    p=opportunity();p['opportunity_id']='new-name';p['prefix_digest']=digest('other-prefix')
    with pytest.raises(LedgerError,match='duplicate PRE'):j.append('duplicate-pre','OPPORTUNITY',p)


@pytest.mark.parametrize('did_apply',(False,True))
def test_commit_unknown_blocks_retry_until_external_readback_even_after_day(tmp_path,did_apply):
    j,op,a,e=ready(tmp_path);r=reservation(j,a,e);world=FakeWorld()
    external=world.commit(op,r['payload']['attempt_token']) if did_apply else None
    j.append('unknown','COMMIT_WITNESS',{'assignment_ref':a,'status':'COMMIT_UNKNOWN',
             'reservation_ref':r['record_digest'],'evidence_ref':digest('artificial-unknown')})
    j.append('missing','DAY_WITNESS',day(a,op,missing=True))
    recovered=FixtureJournal.recover(j.path,expected_head=j.head)
    assert view(recovered)['assignments'][0]['commit_status']=='COMMIT_UNKNOWN'
    p={**r['payload'],'attempt_token':'new-token'}
    with pytest.raises(LedgerError,match='at most one'):recovered.append('retry','COMMIT_WITNESS',p)
    recovered.append('readback','COMMIT_WITNESS',committed(a,r,external) if did_apply else unsubmitted(a,r,op))
    assert view(recovered)['assignments'][0]['commit_status']==('COMMITTED' if did_apply else 'NOT_COMMITTED')
    assert world.calls==int(did_apply)
    with pytest.raises(LedgerError):recovered.append('still-no-retry','COMMIT_WITNESS',p)


def test_commit_succeeded_but_journal_write_failed_recovers_evidence_not_action(tmp_path):
    j,op,a,e=ready(tmp_path);r=reservation(j,a,e);anchor=j.head;world=FakeWorld()
    external=world.commit(op,r['payload']['attempt_token'])
    payload=committed(a,r,external,audit_gap=True)
    with pytest.raises(FixtureFault):j.append('readback','COMMIT_WITNESS',payload,fault='BEFORE_WRITE')
    recovered=FixtureJournal.recover(j.path,expected_head=anchor)
    assert view(recovered)['assignments'][0]['commit_status']=='AWAITING_EXTERNAL_READBACK'
    recovered.append('readback','COMMIT_WITNESS',payload)
    raw=recovered.path.read_bytes()
    for _ in range(3):
        recovered=FixtureJournal.recover(recovered.path,expected_head=recovered.head)
        recovered.append('readback','COMMIT_WITNESS',payload)
    assert world.calls==1 and recovered.path.read_bytes()==raw
    assert recovered.records[-1]['payload']['had_persistence_failure'] is True


def test_ack_lost_does_not_silently_adopt_unknown_head_or_repeat_action(tmp_path):
    j,op,a,e=ready(tmp_path);r=reservation(j,a,e);old_head=j.head;world=FakeWorld()
    payload=committed(a,r,world.commit(op,r['payload']['attempt_token']))
    with pytest.raises(FixtureFault) as fault:j.append('commit','COMMIT_WITNESS',payload,fault='ACK_LOST')
    with pytest.raises(LedgerError,match='anchor'):FixtureJournal.recover(j.path,expected_head=old_head)
    with pytest.raises(ValueError,match='unanchored'):FixtureJournal.recover(j.path,expected_head=None)
    # Explicit artificial fault oracle; not an implementation of real head reconciliation.
    recovered=FixtureJournal.recover(j.path,expected_head=fault.value.oracle_head)
    raw=recovered.path.read_bytes();recovered.append('commit','COMMIT_WITNESS',payload)
    assert recovered.path.read_bytes()==raw and world.calls==1


def test_partial_frame_stops_and_never_repairs_or_truncates(tmp_path):
    j,op,a,e=ready(tmp_path);old=j.head
    p={'assignment_ref':a,'status':'ATTEMPT_RESERVED','execution_ref':e['record_digest'],
       'attempt_token':'fixture-partial','evidence_ref':digest('fixture')}
    with pytest.raises(FixtureFault):j.append('partial','COMMIT_WITNESS',p,fault='PARTIAL_WRITE')
    raw=j.path.read_bytes()
    for _ in range(2):
        with pytest.raises(ValueError,match='partial'):FixtureJournal.recover(j.path,expected_head=old)
    assert j.path.read_bytes()==raw


@pytest.mark.parametrize('field',('game_id','game_attempt_id','boundary_id','prefix_digest','step_index','actor','day','phase','attempt_token'))
def test_external_readback_must_bind_exact_opportunity_and_attempt(tmp_path,field):
    j,op,a,e=ready(tmp_path);r=reservation(j,a,e);external=FakeWorld().commit(op,r['payload']['attempt_token'])
    external[field]='wrong'
    with pytest.raises(LedgerError,match='identity/token'):j.append('wrong-readback','COMMIT_WITNESS',committed(a,r,external))


def test_precommit_unsubmitted_witness_cannot_implicitly_authorize_a_commit(tmp_path):
    j,op,a,e=ready(tmp_path)
    j.append('precommit-failure','COMMIT_WITNESS',unsubmitted(a,None,op))
    with pytest.raises(LedgerError):reservation(j,a,e)
    assert view(j)['assignments'][0]['commit_intents']==0


def test_shared_day_counts_two_assignments_but_one_public_day(tmp_path):
    j=FixtureJournal(tmp_path/'journal.jsonl');op,a,_=assign(j);op2,b,_=assign(j,1)
    j.append('day-a','DAY_WITNESS',day(a,op));j.append('day-b','DAY_WITNESS',day(b,op2))
    v=validate_ledger(j.records,ledger_id=LEDGER)
    assert len(v.assignments)==2 and len(v.shared_observed_days)==1
    assert all(x.observed_no_exile is True for x in v.assignments)


def test_shared_day_conflict_rejected_and_missing_can_be_recovered_with_history(tmp_path):
    j=FixtureJournal(tmp_path/'journal.jsonl');op,a,_=assign(j);op2,b,_=assign(j,1)
    missing=j.append('missing','DAY_WITNESS',day(a,op,missing=True))
    j.append('observed','DAY_WITNESS',day(a,op,supersedes=missing['record_digest']))
    with pytest.raises(LedgerError,match='conflicting'):j.append('conflict','DAY_WITNESS',day(b,op2,exiles=('player7',)))
    j.append('same-public-day','DAY_WITNESS',day(b,op2))
    assert len(validate_ledger(j.records,ledger_id=LEDGER).assignments)==2
    assert any(r['payload'].get('status')=='MISSING' for r in j.records)
    with pytest.raises(LedgerError):j.append('downgrade','DAY_WITNESS',day(a,op,missing=True))


def test_missing_day_cannot_be_encoded_as_empty_exile(tmp_path):
    j=FixtureJournal(tmp_path/'journal.jsonl');op,a,_=assign(j)
    p=day(a,op,missing=True);p['witness']['exiled_seats']=[]
    with pytest.raises(LedgerError,match='missing outcome'):j.append('bad-missing','DAY_WITNESS',p)


def test_valid_prefix_requires_external_anchor_to_detect_suffix_loss(tmp_path):
    j,op,a,e=ready(tmp_path);r=reservation(j,a,e)
    prefix=j.records[:-1]
    assert validate_ledger(prefix,ledger_id=LEDGER).external_head_checked is False
    with pytest.raises(LedgerError,match='anchor'):validate_ledger(prefix,ledger_id=LEDGER,expected_head=j.head)


def test_two_valid_forks_do_not_prove_one_external_commit(tmp_path):
    j,op,a,e=ready(tmp_path)
    p={'assignment_ref':a,'status':'ATTEMPT_RESERVED','execution_ref':e['record_digest'],
       'attempt_token':'branch-A','evidence_ref':digest('fixture')}
    branch_a=extend_ledger(j.records,ledger_id=LEDGER,operation_id='forkA',kind='COMMIT_WITNESS',payload=p)
    branch_b=extend_ledger(j.records,ledger_id=LEDGER,operation_id='forkB',kind='COMMIT_WITNESS',payload={**p,'attempt_token':'branch-B'})
    assert all(validate_ledger(x,ledger_id=LEDGER).external_exactly_once_proven is False for x in (branch_a,branch_b))
    world=FakeWorld();world.commit(op,'branch-A');world.commit(op,'branch-B')
    assert world.calls==2  # Demonstrates missing external fencing, not success of exactly-once.


@pytest.mark.parametrize('mutation',('digest','previous','sequence','duplicate','reward_field'))
def test_tampered_history_or_extra_research_fields_rejected(tmp_path,mutation):
    j,op,a,e=ready(tmp_path);records=deepcopy(list(j.records))
    if mutation=='duplicate':records.append(records[-1])
    elif mutation=='reward_field':
        records[-1]['payload']['reward']=1
        records[-1]['record_digest']=digest({k:v for k,v in records[-1].items() if k!='record_digest'})
    else:records[-1][{'digest':'record_digest','previous':'previous','sequence':'sequence'}[mutation]]=None
    with pytest.raises(LedgerError):validate_ledger(records,ledger_id=LEDGER)


@pytest.mark.parametrize('weights',(['0'],['1/2'],[True],['NaN']))
def test_declared_fixture_weights_do_not_become_live_randomization(tmp_path,weights):
    j=FixtureJournal(tmp_path/'journal.jsonl');op,a,_=assign(j)
    # Rebuild just before assignment; this does not discard or alter the journal.
    records=j.records[:-1];p=deepcopy(j.records[-1]['payload']);p['declared_probabilities']=weights
    with pytest.raises(LedgerError):extend_ledger(records,ledger_id=LEDGER,operation_id='bad-weights',kind='ASSIGNMENT',payload=p)


def test_pure_validation_and_append_do_not_read_write_mutate_or_consume_rng(tmp_path):
    j,op,a,e=ready(tmp_path);before=canonical_json_bytes(j.records);rng=random.getstate();nrng=np.random.get_state()
    with patch('builtins.open',side_effect=AssertionError('pure interface must not open files')):
        result=validate_ledger(j.records,ledger_id=LEDGER,expected_head=j.head)
        new=extend_ledger(j.records,ledger_id=LEDGER,operation_id='missing',kind='DAY_WITNESS',payload=day(a,op,missing=True))
    assert canonical_json_bytes(j.records)==before and len(new)==len(j.records)+1
    assert result.external_exactly_once_proven is False and random.getstate()==rng
    after=np.random.get_state();assert after[0]==nrng[0] and np.array_equal(after[1],nrng[1]) and after[2:]==nrng[2:]


def test_started_execution_interruption_never_restarts_or_authorizes_commit(tmp_path):
    j=FixtureJournal(tmp_path/'journal.jsonl');op,a,expected=assign(j)
    payload={'assignment_ref':a,'status':'STARTED','execution_id':'unresolved','implementation_ref':digest('fixture')}
    started=j.append('start','EXECUTION',payload);recovered=FixtureJournal.recover(j.path,expected_head=j.head)
    with pytest.raises(LedgerError):recovered.append('new-start','EXECUTION',payload)
    p={'assignment_ref':a,'status':'ATTEMPT_RESERVED','execution_ref':started['record_digest'],
       'attempt_token':'unverified','evidence_ref':digest('fixture')}
    with pytest.raises(LedgerError):recovered.append('premature-commit','COMMIT_WITNESS',p)


def test_execution_predecessor_cannot_be_taken_from_another_assignment(tmp_path):
    j=FixtureJournal(tmp_path/'journal.jsonl');op,a,expected=assign(j);execution(j,a,expected)
    op2,b,expected2=assign(j,1);other=execution(j,b,expected2)
    with pytest.raises(LedgerError):reservation(j,a,other)


@pytest.mark.parametrize('field,value',(('game_id','other-game'),('day',9),('day',True)))
def test_day_witness_cannot_cross_game_or_day(tmp_path,field,value):
    j=FixtureJournal(tmp_path/'journal.jsonl');op,a,_=assign(j);p=day(a,op)
    p['witness'][field]=value
    with pytest.raises(LedgerError,match='day witness identity'):j.append('wrong-day','DAY_WITNESS',p)


def test_fixed_fixture_scope_does_not_admit_live_randomization(tmp_path):
    j=FixtureJournal(tmp_path/'journal.jsonl');assign(j);records=j.records[:-1]
    p=deepcopy(j.records[-1]['payload']);p['assignment_source']='REAL_RANDOMIZATION'
    with pytest.raises(LedgerError,match='live randomization'):
        extend_ledger(records,ledger_id=LEDGER,operation_id='real',kind='ASSIGNMENT',payload=p)


def test_retry_of_old_operation_returns_original_receipt_not_current_head(tmp_path):
    j,op,a,e=ready(tmp_path)
    old=next(r for r in j.records if r['kind']=='ASSIGNMENT');raw=j.path.read_bytes()
    receipt=j.append(old['operation_id'],old['kind'],old['payload'])
    assert receipt==old and receipt['record_digest']!=j.head and j.path.read_bytes()==raw


def test_failed_execution_still_accepts_an_independent_day_witness(tmp_path):
    j=FixtureJournal(tmp_path/'journal.jsonl');op,a,expected=assign(j)
    execution(j,a,expected,failure='ACTOR_FAILURE')
    j.append('independent-day','DAY_WITNESS',day(a,op,exiles=('player7',)))
    state=validate_ledger(j.records,ledger_id=LEDGER)
    assert len(state.assignments)==1 and state.assignments[0].execution_status=='FAILED'
    assert state.assignments[0].day_status=='OBSERVED' and state.assignments[0].commit_intents==0
