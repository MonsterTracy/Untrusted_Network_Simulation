"""Artificial single-writer journal + unfenced fake world. TESTS ONLY.

Not a production store/owner: recovery anchors are injected by the test harness;
there is no interprocess lock, real env/recorder, Actor, randomizer or fallback.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path

from scripts.speech_planning import generate_candidates
from scripts.speech_realization import expected_action
from tests.tom.test_counterfactual_consumer import PublicWitness, P
from werewolf.artifact_io import canonical_json_bytes
from werewolf.observer_dtrs.action_evidence_ledger import digest, extend_ledger, validate_ledger

LEDGER = 'artificial-phase2d-fixture-not-a-study'


class FixtureFault(OSError):
    def __init__(self, stage, oracle_head=None):
        super().__init__(stage)
        self.oracle_head = oracle_head  # Artificial fault oracle, NOT a real recovery protocol.


class FixtureJournal:
    def __init__(self, path):
        self.path = Path(path)
        fd = os.open(self.path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        os.fsync(fd); os.close(fd)
        fd = os.open(self.path.parent,os.O_RDONLY); os.fsync(fd); os.close(fd)
        self.records = ()

    @property
    def head(self):
        return self.records[-1]['record_digest'] if self.records else None

    @classmethod
    def recover(cls, path, *, expected_head):
        raw = Path(path).read_bytes()
        if raw and (not raw.endswith(b'\n') or expected_head is None):
            raise ValueError('partial/unanchored recovery: stop without repair')
        records = tuple(json.loads(line) for line in raw.splitlines())
        if b''.join(canonical_json_bytes(r)+b'\n' for r in records) != raw:
            raise ValueError('noncanonical journal: stop without repair')
        validate_ledger(records,ledger_id=LEDGER,expected_head=expected_head)
        obj = cls.__new__(cls); obj.path = Path(path); obj.records = records
        return obj

    def append(self, operation, kind, payload, *, fault=None):
        raw = b''.join(canonical_json_bytes(r)+b'\n' for r in self.records)
        if self.path.read_bytes() != raw:
            raise ValueError('fixture writer view changed')
        records = extend_ledger(self.records,ledger_id=LEDGER,operation_id=operation,kind=kind,payload=payload)
        if len(records) == len(self.records):
            return next(r for r in records if r['operation_id'] == operation)
        new = records[-1]; line = canonical_json_bytes(new)+b'\n'
        if fault == 'BEFORE_WRITE': raise FixtureFault(fault)
        with self.path.open('ab') as f:
            f.write(line[:len(line)//2] if fault == 'PARTIAL_WRITE' else line)
            f.flush(); os.fsync(f.fileno())
        if fault:
            raise FixtureFault(fault,new['record_digest'] if fault == 'ACK_LOST' else None)
        self.records = records
        return new


def opportunity(index=0):
    w = PublicWitness()
    if index:
        w.speech(P[0],'no_commitment',None,'人工fixture，不是受控比较方案。')
        w.event('turn_start',speaker=P[1])
    pre = w.pre()
    return {'opportunity_id':f'op-{index}','protocol_ref':digest('UNFROZEN-CONTROLLED-FIXTURE'),
        'game_id':pre.game_id,'game_attempt_id':'artificial-attempt','boundary_id':pre.boundary_id,
        'prefix_digest':pre.prefix_digest,'step_index':pre.step_index,'actor':pre.current_speaker,
        'day':pre.public_temporal_state.day,'phase':pre.public_temporal_state.phase.value,
        'public_history_digest':pre.public_event_digest,'capture_digest':digest(pre.to_record()),
        'candidate_order_version':'original-generate_candidates-fixture',
        'candidates':[{'plan':p.public_payload(),'expected_v1':expected_action(p.public_payload(),pre.current_speaker).to_record()}
                      for p in generate_candidates(pre)]}


def assign(journal, index=0):
    op = opportunity(index)
    record = journal.append(f'op-{index}','OPPORTUNITY',op)
    # Fixed injected fixture, not an action choice, fallback or random draw.
    candidate = next(c for c in op['candidates'] if c['plan']['action']=='NO_COMMITMENT')
    assignment = {'assignment_id':f'assignment-{index}','opportunity_ref':record['record_digest'],
        'candidate_id':digest(candidate),'support':[digest(candidate)],'declared_probabilities':None,
        'treatment_ref':digest('UNFROZEN-ARTIFICIAL-TREATMENT'),'assignment_source':'INJECTED_FIXED_NOT_RANDOMIZED'}
    a = journal.append(f'assign-{index}','ASSIGNMENT',assignment)
    return op,a['record_digest'],candidate['expected_v1']


def execution(journal, assignment_ref, expected, *, failure=None):
    s = journal.append('start-'+assignment_ref,'EXECUTION',{'assignment_ref':assignment_ref,'status':'STARTED',
        'execution_id':'fixture-'+assignment_ref,'implementation_ref':digest('artificial-no-Actor-Parser')})
    p = {'assignment_ref':assignment_ref,'started_ref':s['record_digest'],
         'status':'FAILED' if failure else 'VERIFIED','evidence_ref':digest('artificial-execution-evidence'),
         'speech_digest':digest('artificial-public-speech') if not failure or failure=='SEMANTIC_MISMATCH' else None,
         'actions':([expected,[expected[0],'support','player3']] if failure=='SEMANTIC_MISMATCH' else None) if failure else [expected]}
    if failure: p['error_code'] = failure
    return journal.append('execution-'+assignment_ref,'EXECUTION',p)


def reservation(journal, assignment_ref, execution_record):
    return journal.append('intent-'+assignment_ref,'COMMIT_WITNESS',{'assignment_ref':assignment_ref,
        'status':'ATTEMPT_RESERVED','execution_ref':execution_record['record_digest'],
        'attempt_token':'fixture-token-'+assignment_ref,'evidence_ref':digest('artificial-intent')})


class FakeWorld:
    """Intentionally NOT fenced: direct duplicate calls really increment twice."""
    def __init__(self): self.calls = 0; self.last = None

    def commit(self, op, token):
        self.calls += 1
        self.last = {**{k:op[k] for k in ('game_id','game_attempt_id','boundary_id','prefix_digest','step_index','actor','day','phase')},
            'attempt_token':token,'action_id':f'artificial-action-{self.calls}','event_id':f'artificial-event-{self.calls}',
            'annotation_digest':digest('artificial-annotation'),'post_history_digest':digest(['artificial-post',self.calls]),
            'readback_digest':digest(['artificial-readback',self.calls])}
        return dict(self.last)


def committed(assignment_ref, reserved, external, *, audit_gap=False):
    return {'assignment_ref':assignment_ref,'status':'COMMITTED','reservation_ref':reserved['record_digest'],
            'external':external,'had_persistence_failure':audit_gap}


def unsubmitted(assignment_ref, reserved, op):
    return {'assignment_ref':assignment_ref,'status':'NOT_COMMITTED',
        'reservation_ref':reserved['record_digest'] if reserved else None,'had_persistence_failure':False,
        'external':{**{k:op[k] for k in ('game_id','game_attempt_id','boundary_id','prefix_digest','step_index','actor','day','phase')},
            'attempt_token':reserved['payload']['attempt_token'] if reserved else None,
            'public_history_digest':op['public_history_digest'],'stopped_executor_digest':digest('artificial-stopped-executor'),
            'readback_digest':digest('artificial-unsubmitted-readback')}}


def day(assignment_ref, op, *, missing=False, supersedes=None, exiles=()):
    return {'assignment_ref':assignment_ref,'status':'MISSING' if missing else 'OBSERVED','supersedes_ref':supersedes,
        'witness':{'game_id':op['game_id'],'day':op['day'],
            'resolution_ref':None if missing else digest(['artificial-public-day',op['game_id'],op['day']]),
            'exiled_seats':None if missing else list(exiles),
            'missing_reason_ref':digest('artificial-technical-missing') if missing else None}}


def view(journal):
    return asdict(validate_ledger(journal.records,ledger_id=LEDGER,expected_head=journal.head))
