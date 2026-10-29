"""Pure Phase-2D evidence validation; fixed injected assignments only.

No journal I/O, RNG, Actor, Parser, env, recorder or commit is called here.
Records attest supplied evidence: hashes cannot establish its external truth,
PRE/candidate admission, durability, inter-writer exclusion or exactly-once play.
"""
from dataclasses import dataclass
from fractions import Fraction
import json
import re

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.canonical_collection.speech import V1SpeechAction

VERSION = 'action_evidence_controlled_validator_v1'
SCOPE = 'FIXED_INJECTION_ONLY_NOT_EXECUTABLE_G2'


class LedgerError(ValueError):
    pass


def digest(value):
    return sha256_bytes(canonical_json_bytes(value))


def _require(ok, message):
    if not ok:
        raise LedgerError(message)


def _fields(value, names):
    _require(type(value) is dict and set(value) == set(names.split()), 'record fields mismatch')


def _text(value):
    _require(type(value) is str and bool(value.strip()), 'nonempty identity required')


def _hash(value):
    _require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None,
             'invalid evidence digest')


def _actions(values, actor):
    _require(type(values) is list, 'full V1 action list required')
    for v in values:
        _require(type(v) is list and len(v) == 3, 'invalid V1 triplet')
        a = V1SpeechAction(*v)
        _require(a.subject == actor, 'V1 actor mismatch')


@dataclass(frozen=True)
class AssignmentView:
    assignment_id: str
    assignment_ref: str
    opportunity_id: str
    game_id: str
    day: int
    execution_status: str
    commit_status: str
    commit_intents: int
    day_status: str
    observed_no_exile: bool | None


@dataclass(frozen=True)
class LedgerView:
    head: str | None
    opportunity_count: int
    assignments: tuple[AssignmentView, ...]
    shared_observed_days: tuple[tuple[str, int, str], ...]
    external_head_checked: bool
    external_exactly_once_proven: bool = False
    external_evidence_truth_verified: bool = False


def _replay(records, ledger_id, expected_head):
    _text(ledger_id)
    previous, operations, opportunities, physical, assigned, ids, tokens, shared = None, set(), {}, set(), {}, set(), set(), {}
    for sequence, r in enumerate(records):
        _fields(r, 'version scope ledger_id operation_id sequence previous kind payload record_digest')
        _require(r['version'] == VERSION and r['scope'] == SCOPE and r['ledger_id'] == ledger_id,
                 'ledger/version/scope mismatch')
        _text(r['operation_id'])
        _require(r['operation_id'] not in operations, 'duplicate operation in stored chain')
        _require(type(r['sequence']) is int and r['sequence'] == sequence and r['previous'] == previous,
                 'broken sequence/previous digest')
        _require(digest({k:v for k,v in r.items() if k != 'record_digest'}) == r['record_digest'],
                 'record digest mismatch')
        operations.add(r['operation_id']); previous = r['record_digest']
        p, kind = r['payload'], r['kind']
        if kind == 'OPPORTUNITY':
            _fields(p, 'opportunity_id protocol_ref game_id game_attempt_id boundary_id prefix_digest step_index actor day phase public_history_digest capture_digest candidate_order_version candidates')
            for k in ('opportunity_id','game_id','game_attempt_id','boundary_id','candidate_order_version'):
                _text(p[k])
            for k in ('protocol_ref','prefix_digest','public_history_digest','capture_digest'):
                _hash(p[k])
            _require(type(p['step_index']) is int and p['step_index'] >= 0 and type(p['day']) is int
                     and p['day'] >= 0 and p['actor'] in PLAYER_IDS
                     and p['phase'] in ('discussion','pk_discussion'), 'invalid opportunity identity')
            key = (p['game_id'],p['boundary_id'])
            _require(key not in physical and p['opportunity_id'] not in ids, 'duplicate PRE/opportunity')
            physical.add(key); ids.add(p['opportunity_id'])
            _require(type(p['candidates']) is list and bool(p['candidates']), 'candidate capture required')
            candidates = {}
            for c in p['candidates']:
                _fields(c, 'plan expected_v1')
                _fields(c['plan'], 'action target'); _text(c['plan']['action'])
                _require(c['plan']['target'] is None or c['plan']['target'] in PLAYER_IDS,
                         'invalid captured target')
                _actions([c['expected_v1']], p['actor'])
                cid = digest(c)
                _require(cid not in candidates, 'duplicate captured candidate')
                candidates[cid] = c
            opportunities[previous] = (p,candidates)
            continue
        if kind == 'ASSIGNMENT':
            _fields(p, 'assignment_id opportunity_ref candidate_id support declared_probabilities treatment_ref assignment_source')
            _text(p['assignment_id']); _hash(p['treatment_ref'])
            _require(p['assignment_source'] == 'INJECTED_FIXED_NOT_RANDOMIZED', 'no live randomization admission')
            _require(p['opportunity_ref'] in opportunities, 'unknown opportunity predecessor')
            _require(p['assignment_id'] not in ids and not any(a['op_ref'] == p['opportunity_ref'] for a in assigned.values()),
                     'duplicate assignment; no redraw')
            op,candidates = opportunities[p['opportunity_ref']]
            support = p['support']
            _require(type(support) is list and len(set(support)) == len(support) and bool(support)
                     and set(support) <= set(candidates) and p['candidate_id'] in support,
                     'assignment outside declared captured support')
            probabilities = p['declared_probabilities']
            if probabilities is not None:
                _require(type(probabilities) is list and len(probabilities) == len(support)
                         and all(type(v) is str for v in probabilities), 'exact fixture probability declarations required')
                weights = [Fraction(v) for v in probabilities]
                _require(all(v > 0 for v in weights) and sum(weights) == 1, 'invalid declared probability vector')
            ids.add(p['assignment_id'])
            assigned[previous] = {'id':p['assignment_id'],'op':op,'op_ref':p['opportunity_ref'],
                'expected':candidates[p['candidate_id']]['expected_v1'], 'exec':'NOT_STARTED',
                'start_ref':None,'verified_ref':None,'commit':'NO_ATTEMPT','reservation':None,'token':None,
                'day':'PENDING','day_ref':None,'no_exile':None}
            continue
        _require(kind in ('EXECUTION','COMMIT_WITNESS','DAY_WITNESS') and type(p) is dict,
                 'unsupported event kind')
        _require(p.get('assignment_ref') in assigned, 'unknown assignment predecessor')
        a = assigned[p['assignment_ref']]; op = a['op']; status = p.get('status')
        if kind == 'EXECUTION':
            if status == 'STARTED':
                _fields(p, 'assignment_ref status execution_id implementation_ref')
                _text(p['execution_id']); _hash(p['implementation_ref'])
                _require(a['exec'] == 'NOT_STARTED' and a['commit'] == 'NO_ATTEMPT' and a['day'] == 'PENDING',
                         'execution cannot restart or follow a day witness')
                a['exec'],a['start_ref'] = 'STARTED',previous
            else:
                common = 'assignment_ref status started_ref evidence_ref speech_digest actions'
                _fields(p, common + (' error_code' if status == 'FAILED' else ''))
                _require(status in ('VERIFIED','FAILED') and a['exec'] == 'STARTED'
                         and p['started_ref'] == a['start_ref'], 'invalid execution completion')
                _hash(p['evidence_ref'])
                if p['speech_digest'] is not None: _hash(p['speech_digest'])
                if p['actions'] is not None: _actions(p['actions'],op['actor'])
                if status == 'VERIFIED':
                    _require(p['speech_digest'] is not None and p['actions'] == [a['expected']],
                             'verification requires the full exact singleton')
                    a['verified_ref'] = previous
                else:
                    _require(p['error_code'] in ('ACTOR_FAILURE','PARSER_FAILURE','SEMANTIC_MISMATCH','EXECUTION_ABORTED'),
                             'unknown execution failure')
                    if p['error_code'] == 'SEMANTIC_MISMATCH':
                        _require(p['actions'] is not None and p['actions'] != [a['expected']],
                                 'semantic mismatch must retain the complete actual actions')
                a['exec'] = status
        elif kind == 'COMMIT_WITNESS':
            if status == 'ATTEMPT_RESERVED':
                _fields(p, 'assignment_ref status execution_ref attempt_token evidence_ref')
                _text(p['attempt_token']); _hash(p['evidence_ref'])
                _require(a['exec'] == 'VERIFIED' and p['execution_ref'] == a['verified_ref']
                         and a['commit'] == 'NO_ATTEMPT' and a['day'] == 'PENDING'
                         and p['attempt_token'] not in tokens, 'at most one commit intent; no retry/fallback')
                tokens.add(p['attempt_token'])
                a['reservation'],a['token'],a['commit'] = previous,p['attempt_token'],'AWAITING_EXTERNAL_READBACK'
            elif status == 'COMMIT_UNKNOWN':
                _fields(p, 'assignment_ref status reservation_ref evidence_ref'); _hash(p['evidence_ref'])
                _require(a['commit'] == 'AWAITING_EXTERNAL_READBACK' and p['reservation_ref'] == a['reservation'],
                         'invalid unknown-commit evidence')
                a['commit'] = 'COMMIT_UNKNOWN'
            else:
                _fields(p, 'assignment_ref status reservation_ref external had_persistence_failure')
                _require(status in ('COMMITTED','NOT_COMMITTED') and type(p['had_persistence_failure']) is bool,
                         'invalid external readback status')
                allowed = ('AWAITING_EXTERNAL_READBACK','COMMIT_UNKNOWN')
                _require(a['commit'] in allowed or (status == 'NOT_COMMITTED' and a['commit'] == 'NO_ATTEMPT'
                         and a['exec'] != 'NOT_STARTED'), 'resolved commit cannot change or repeat')
                _require(p['reservation_ref'] == a['reservation'], 'wrong reservation predecessor')
                e = p['external']
                common = 'game_id game_attempt_id boundary_id prefix_digest step_index actor day phase attempt_token readback_digest'
                _fields(e, common + (' action_id event_id annotation_digest post_history_digest'
                                    if status == 'COMMITTED' else ' public_history_digest stopped_executor_digest'))
                identity = ('game_id','game_attempt_id','boundary_id','prefix_digest','step_index','actor','day','phase')
                _require(all(e[k] == op[k] and type(e[k]) is type(op[k]) for k in identity)
                         and e['attempt_token'] == a['token'], 'external identity/token mismatch')
                _hash(e['readback_digest'])
                if status == 'COMMITTED':
                    for k in ('action_id','event_id'): _text(e[k])
                    for k in ('annotation_digest','post_history_digest'): _hash(e[k])
                    _require(e['post_history_digest'] != op['public_history_digest'], 'commit has no new public history')
                else:
                    _hash(e['stopped_executor_digest'])
                    _require(e['public_history_digest'] == op['public_history_digest'], 'unsubmitted readback history mismatch')
                a['commit'] = status
        else:
            _fields(p, 'assignment_ref status witness supersedes_ref')
            _require(status in ('OBSERVED','MISSING') and
                     (a['day'] == 'PENDING' or (a['day'] == 'MISSING' and status == 'OBSERVED'))
                     and p['supersedes_ref'] == a['day_ref'], 'invalid/repeated day witness transition')
            w = p['witness']
            _fields(w, 'game_id day resolution_ref exiled_seats missing_reason_ref')
            _require(type(w['day']) is int and (w['game_id'],w['day']) == (op['game_id'],op['day']),
                     'day witness identity mismatch')
            if status == 'MISSING':
                _require(w['resolution_ref'] is None and w['exiled_seats'] is None, 'missing outcome is not no-exile')
                _hash(w['missing_reason_ref']); a['no_exile'] = None
            else:
                _hash(w['resolution_ref'])
                seats = w['exiled_seats']
                _require(w['missing_reason_ref'] is None and type(seats) is list and len(seats) <= 1
                         and all(s in PLAYER_IDS for s in seats), 'invalid public exile witness')
                key = (op['game_id'],op['day']); wd = digest(w)
                _require(key not in shared or shared[key] == wd, 'conflicting shared public day witness')
                shared[key] = wd; a['no_exile'] = not seats
            a['day'],a['day_ref'] = status,previous
    if expected_head is not None:
        _hash(expected_head); _require(previous == expected_head, 'external head anchor mismatch')
    views = tuple(AssignmentView(a['id'],ref,a['op']['opportunity_id'],a['op']['game_id'],a['op']['day'],
                   a['exec'],a['commit'],int(a['reservation'] is not None),a['day'],a['no_exile'])
                  for ref,a in assigned.items())
    return LedgerView(previous,len(opportunities),views,tuple((g,d,h) for (g,d),h in sorted(shared.items())),
                      expected_head is not None)


def validate_ledger(records, *, ledger_id, expected_head=None):
    """Validate supplied records only; expected_head can detect prefix truncation.

    Without an external anchor a valid prefix/fork remains valid. Returned state
    is evidence bookkeeping, never external environment state or a call permit.
    """
    try:
        clean = json.loads(canonical_json_bytes(records))
        _require(type(clean) is list, 'ordered record sequence required')
        return _replay(clean,ledger_id,expected_head)
    except LedgerError:
        raise
    except (ValueError,TypeError,KeyError,IndexError,ZeroDivisionError) as error:
        raise LedgerError(f'invalid evidence ledger: {error}') from error


def extend_ledger(records, *, ledger_id, operation_id, kind, payload):
    """Pure prospective append. Identical operation retry is a no-op.

    The caller owns durability and serialization. No action may be triggered by
    constructing this return value; it has not been persisted by this function.
    """
    validate_ledger(records,ledger_id=ledger_id)
    clean = json.loads(canonical_json_bytes(records)); p = json.loads(canonical_json_bytes(payload))
    _text(operation_id)
    for old in clean:
        if old['operation_id'] == operation_id:
            _require(old['kind'] == kind and old['payload'] == p, 'operation identity reused with different content')
            return tuple(clean)
    r = {'version':VERSION,'scope':SCOPE,'ledger_id':ledger_id,'operation_id':operation_id,
         'sequence':len(clean),'previous':clean[-1]['record_digest'] if clean else None,
         'kind':kind,'payload':p}
    r['record_digest'] = digest(r)
    clean.append(r)
    validate_ledger(clean,ledger_id=ledger_id)
    return tuple(clean)
