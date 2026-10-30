"""Conditional historical public-behavior diagnostics; no game or policy execution.

PRE-only features and future public targets deliberately have different APIs.
All upstream D1/D2 authentication remains with the original offline readers.
"""
from collections import Counter
import math

import numpy as np

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection.pre import validate_authoritative_pre_prefix
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.canonical_collection.game_bundle import _history_from_records, _annotation_from_record
from werewolf.phase2_offline import _resolved_exiles, _vote_winner, Phase2DataError
from werewolf.canonical_collection.speech import validate_v1_annotation_binding
from werewolf.observer_dtrs.baselines import objective, sigmoid, SOLVER

VERSION='historical_public_accusation_v1'
# Provenance of the historical public-only helper, now maintained below in-repo.
PUBLIC_AUDIT_ORIGIN_SHA256='50996ec5b66277fdd46565dbbbfc5c2933097838ef5e6faff2b312b70b695b61'
COUNTS=('day','step','pk','alive_count','event_count','day_speeches','phase_speeches',
        'observer_day_speeches','observer_phase_speeches','observer_total_speeches',
        'last_observer_event_gap','no_last_observer','latest_V1_missing','latest_V1_failed',
        'latest_V1_actions','observer_day_accuse_actor','observer_day_clear_actor',
        'latest_accuse_actor','latest_clear_actor','others_day_accuse_actor',
        'others_day_clear_actor','observer_day_accuse_other','latest_actor_claim_conflict')
PUBLIC_NAMES=COUNTS+tuple('actor_'+p for p in PLAYER_IDS)+tuple('observer_'+p for p in PLAYER_IDS)
LEGAL_NAMES=('legal_alive_wolf_count',)+tuple('legal_teammate_'+p for p in PLAYER_IDS)


def public_features(pre, observer_id):
    """No future/history/target argument is accepted. Never inspect private Q/Y."""
    validate_authoritative_pre_prefix(pre)
    if observer_id not in pre.alive_observer_ids or observer_id==pre.current_speaker:
        raise ValueError('living non-actor observer required')
    events=pre.public_event_history.events
    speeches=[e for e in events if e.event_type=='public_speech']
    day=[e for e in speeches if e.temporal_state.day==pre.public_temporal_state.day]
    phase=[e for e in day if e.temporal_state.phase==pre.public_temporal_state.phase]
    own=[e for e in speeches if e.speaker==observer_id]
    own_day=[e for e in day if e.speaker==observer_id]
    own_phase=[e for e in phase if e.speaker==observer_id]
    annotations={a.event_id:a for a in pre.v1_annotations}
    last=own[-1] if own else None
    a=annotations.get(last.event_id) if last else None
    def count(events,kind,target):
        return sum(sum(x.action==kind and x.object==target for x in v.actions)
                   for e in events if (v:=annotations.get(e.event_id)) and v.is_successful)
    hit=count(own[-1:],'point_as_werewolf',pre.current_speaker)
    clear=count(own[-1:],'point_as_non_werewolf',pre.current_speaker)
    other=[e for e in day if e.speaker!=observer_id]
    other_targets=sum(sum(x.action=='point_as_werewolf' and x.object!=pre.current_speaker for x in v.actions)
                      for e in own_day if (v:=annotations.get(e.event_id)) and v.is_successful)
    values=(pre.public_temporal_state.day,pre.step_index,int(pre.public_temporal_state.phase.value=='pk_discussion'),
        len(pre.alive_observer_ids),len(events),len(day),len(phase),len(own_day),len(own_phase),len(own),
        len(events)-last.event_index if last else 0,int(last is None),int(a is None),int(bool(a and not a.is_successful)),
        len(a.actions) if a and a.is_successful else 0,count(own_day,'point_as_werewolf',pre.current_speaker),
        count(own_day,'point_as_non_werewolf',pre.current_speaker),hit,clear,
        count(other,'point_as_werewolf',pre.current_speaker),count(other,'point_as_non_werewolf',pre.current_speaker),
        other_targets,int(hit>0 and clear>0),*(int(pre.current_speaker==p) for p in PLAYER_IDS),
        *(int(observer_id==p) for p in PLAYER_IDS))
    assert len(values)==len(PUBLIC_NAMES)
    return tuple(float(x) for x in values)


def legal_features(context):
    """Acting wolf's legal team context; explicitly NOT purely public information."""
    teammate=next(p for p in context.known_wolf_ids if p!=context.acting_wolf)
    return (float(context.wolf_count),*(float(p==teammate) for p in PLAYER_IDS))


def observe_after_speech(pre, observer_id, history, annotations, day_witness):
    """Future-only offline label. Missing opportunity/ambiguous V1 never means 0.

    The caller authenticates independent public files and the original day witness.
    This function verifies their exact PRE/target/cutoff binding, not provenance by itself.
    """
    validate_authoritative_pre_prefix(pre)
    if observer_id not in pre.alive_observer_ids or observer_id==pre.current_speaker:
        raise ValueError('observer identity mismatch')
    old=pre.public_event_history.to_records()
    if history.to_records()[:len(old)]!=old or any(annotations.get(a.event_id)!=a for a in pre.v1_annotations):
        raise ValueError('future public stream does not bind frozen PRE')
    base={'observer_id':observer_id,'Z':None,'status':None,'future_speech_ids':[],
          'positive_event_ids':[],'clear_event_ids':[],'conflict_event_ids':[],
          'cutoff_event_id':None,'target_speech_event_id':None}
    if len(history.events)<=len(old):return {**base,'status':'TARGET_SPEECH_MISSING'}
    target=history.events[len(old)]
    if (target.event_type!='public_speech' or target.speaker!=pre.current_speaker
            or target.temporal_state!=pre.public_temporal_state):
        raise ValueError('wrong immediate target speech')
    base['target_speech_event_id']=target.event_id
    if (day_witness['game_id']!=pre.game_id or day_witness['day']!=pre.public_temporal_state.day):
        raise ValueError('wrong game/day witness')
    if day_witness['day_status']!='COMPLETE':return {**base,'status':'DAY_INCOMPLETE'}
    stop=day_witness['final_exile_event_index']
    if type(stop) is not int or not target.event_index<stop<len(history.events):raise ValueError('invalid day cutoff')
    cutoff=history.events[stop]
    if (cutoff.event_id!=day_witness['final_exile_event_id'] or cutoff.event_type!='exile_result'
            or cutoff.temporal_state.day!=pre.public_temporal_state.day):raise ValueError('cutoff identity mismatch')
    base['cutoff_event_id']=cutoff.event_id
    future=[e for e in history.events if target.event_index<e.event_index<=stop and e.event_type=='public_speech'
            and e.speaker==observer_id and e.temporal_state.day==pre.public_temporal_state.day]
    base['future_speech_ids']=[e.event_id for e in future]
    if not future:return {**base,'status':'NO_FUTURE_SPEECH'}
    failures=[]
    for e in future:
        a=annotations.get(e.event_id)
        if a is None:failures.append('V1_MISSING');continue
        validate_v1_annotation_binding(a,history)
        if a.event_id!=e.event_id or a.event_index!=e.event_index:raise ValueError('future annotation identity mismatch')
        if not a.is_successful:failures.append('V1_FAILED');continue
        hit=any(x.action=='point_as_werewolf' and x.object==pre.current_speaker for x in a.actions)
        clear=any(x.action=='point_as_non_werewolf' and x.object==pre.current_speaker for x in a.actions)
        if hit:base['positive_event_ids'].append(e.event_id)
        if clear:base['clear_event_ids'].append(e.event_id)
        if hit and clear:base['conflict_event_ids'].append(e.event_id);failures.append('V1_CONFLICT')
    if failures:return {**base,'status':sorted(set(failures))[0],'all_failure_reasons':sorted(set(failures))}
    return {**base,'status':'OBSERVED','Z':int(bool(base['positive_event_ids'])),
            'across_time_accuse_and_clear':bool(base['positive_event_ids'] and base['clear_event_ids'])}


def game_weights(game_ids):
    sizes=Counter(game_ids)
    if not sizes:raise ValueError('empty training population')
    return np.asarray([1/(len(sizes)*sizes[g]) for g in game_ids],dtype=np.float64)


def fit_target(X, y, game_ids):
    """Same fixed new-Z L2 map for all representations; no tuning or fallback.

    Reuses the frozen baseline's generic objective and stable sigmoid, but is NOT
    a D2 probability model: its target, covariates and output identity are new.
    """
    X=np.asarray(X,dtype=np.float64);y=np.asarray(y)
    if X.ndim!=2 or y.shape!=(len(X),) or len(game_ids)!=len(X) or not np.isfinite(X).all():
        raise ValueError('invalid training arrays')
    if any(type(v) not in (int,np.int64,np.int32,np.uint8) or int(v) not in (0,1) for v in y):
        raise ValueError('binary target required')
    if len(set(map(int,y)))!=2:raise ValueError('UNSUPPORTED_SINGLE_CLASS_TRAINING')
    weights=game_weights(game_ids);mean=weights@X
    variance=weights@((X-mean)**2);std=np.sqrt(variance);std[np.ptp(X,axis=0)==0]=1.
    # Zero-variance columns are a numerical preprocessing case, not a new probability threshold.
    design=np.column_stack([np.ones(len(X)),(X-mean)/std]);beta=np.zeros(design.shape[1])
    tau=1/len(X)
    for iteration in range(SOLVER['max_iterations']+1):
        loss,gradient,hessian=objective(beta,design,y,weights,tau)
        norm=float(np.max(np.abs(gradient)))
        if norm<=SOLVER['gradient_tolerance']:break
        if iteration==SOLVER['max_iterations']:raise RuntimeError('new-Z Logistic nonconvergence')
        direction=np.linalg.solve(hessian,gradient);descent=float(gradient@direction);step=1.
        for _ in range(SOLVER['max_backtracks']):
            candidate=beta-step*direction;new_loss,new_gradient,_=objective(candidate,design,y,weights,tau)
            tolerance=32*np.finfo(float).eps*max(1.,abs(loss))
            roundoff=(step*descent<=tolerance and new_loss<=loss+tolerance and float(np.max(np.abs(new_gradient)))<norm)
            if new_loss<=loss-SOLVER['armijo']*step*descent or roundoff:beta=candidate;break
            step/=2
        else:raise RuntimeError('new-Z Logistic line search failed')
    if not np.isfinite(beta).all():raise RuntimeError('invalid new-Z coefficients')
    return {'version':VERSION,'mean':mean.tolist(),'std':std.tolist(),'beta':beta.tolist(),
            'train_rows':len(X),'train_games':len(set(game_ids)),'positive':int(y.sum()),
            'tau':tau,'solver':SOLVER,'iterations':iteration,'gradient_inf':norm,'objective':loss}


def target_logits(model,X):
    X=np.asarray(X,dtype=np.float64)
    if X.ndim!=2 or X.shape[1]!=len(model['mean']) or not np.isfinite(X).all():raise ValueError('invalid prediction features')
    design=np.column_stack([np.ones(len(X)),(X-np.asarray(model['mean']))/np.asarray(model['std'])])
    z=design@np.asarray(model['beta'])
    if not np.isfinite(z).all():raise ValueError('invalid prediction logits')
    return z


def row_scores(logits,y):
    z=np.asarray(logits,dtype=np.float64);y=np.asarray(y)
    if z.shape!=y.shape or not np.isfinite(z).all() or not np.isin(y,[0,1]).all():raise ValueError('invalid score arrays')
    p=sigmoid(z)
    # Stable exact logit expression: no clipping or altered D3 regions.
    return p,(p-y)**2,np.logaddexp(0,z)-y*z


def paired_game_cv(blocks,game_ids):
    """Pre-existing source-fold x outer-block IDs, no label-based split."""
    blocks=np.asarray(blocks)
    if blocks.ndim!=2 or blocks.shape[1]!=2 or not np.issubdtype(blocks.dtype,np.integer) or not np.isin(blocks,range(5)).all():
        raise ValueError('invalid source/outer block IDs')
    if len(game_ids)!=len(blocks) or any(len({tuple(b) for b,g in zip(blocks,game_ids) if g==name})!=1 for name in set(game_ids)):
        raise ValueError('same game crosses source/outer blocks')
    seen=np.zeros(len(blocks),int)
    for source in range(5):
        for outer in range(5):
            train=np.flatnonzero((blocks[:,0]==source)&(blocks[:,1]!=outer))
            test=np.flatnonzero((blocks[:,0]==source)&(blocks[:,1]==outer))
            if not len(train) or not len(test):raise ValueError('UNSUPPORTED_EMPTY_SOURCE_BLOCK')
            seen[test]+=1;yield (source,outer),train,test
    if not np.all(seen==1):raise ValueError('incomplete OOF coverage')


def parse_public_game(event_records, annotation_records):
    history = _history_from_records(event_records)
    if history.to_records() != event_records:
        raise ValueError('public event records do not roundtrip canonically')
    annotations = {}
    for record in annotation_records:
        a = _annotation_from_record(record, history)
        validate_v1_annotation_binding(a, history)
        if a.event_id in annotations:
            raise ValueError('duplicate V1 annotation identity')
        annotations[a.event_id] = a
    indices = [a.event_index for a in annotations.values()]
    if indices != sorted(indices):
        raise ValueError('V1 annotation order is not canonical')
    speech_ids = {e.event_id for e in history.events if e.event_type == 'public_speech'}
    if not set(annotations) <= speech_ids:
        raise ValueError('V1 refers to a non-speech event')
    return history, annotations


def public_day_witness(history, day):
    """Public ballot/exile completeness, without roles, outcomes Y or winner.

    The original _resolved_exiles owns normal/PK resolution. Additional checks
    use public alive seats and the runtime's complete ballot/PK-target contract.
    A final exile event ends the vote process even when no next night is logged.
    This does not prove a game's terminal status or perform native replay.
    """
    events = [e for e in history.events if e.temporal_state.day == day]
    votes = [e for e in events if e.event_type == 'vote_result']
    exiles = [e for e in events if e.event_type == 'exile_result']
    base = {'day': day, 'scope': 'PUBLIC_DAY_RESOLUTION_NOT_WOLF_LABEL_OR_GAME_WINNER',
        'vote_records': [e.to_record() for e in votes],
        'exile_records': [e.to_record() for e in exiles],
        'PK_phase_event_ids': [e.event_id for e in events
            if e.event_type == 'phase_change' and e.temporal_state.phase.value in ('pk_discussion', 'pk_vote')],
        'public_speech_event_ids': [e.event_id for e in events if e.event_type == 'public_speech'],
        'game_terminal_verified': False, 'wolf_exile_target_computed': False}
    if not votes and not exiles:
        return {**base, 'day_status': 'UNOBSERVABLE', 'reason': 'NO_PUBLIC_DAY_RESOLUTION'}
    try:
        final = _resolved_exiles(events).get(day)
        if final is None:
            raise ValueError('no resolved final exile event')
        alive = set(PLAYER_IDS)
        for e in history.events:
            if e.event_index > final.event_index:
                break  # Later nights/days do not enter this day's witness.
            if e.event_type == 'vote_result' and e.temporal_state.day == day:
                if {v.voter for v in e.votes} != alive:
                    raise ValueError('ballot does not cover every publicly alive voter')
                if any(v.target is not None and (v.target not in alive or v.target == v.voter)
                       for v in e.votes):
                    raise ValueError('public ballot contains a dead/self target')
                if e.temporal_state.phase.value == 'pk_vote':
                    ordinary = [v for v in votes if v.temporal_state.phase.value == 'vote']
                    tied = _vote_winner(ordinary[0])
                    if any(v.target is not None and v.target not in tied for v in e.votes):
                        raise ValueError('PK ballot target is outside the ordinary tied set')
            if e.event_type in ('death_announcement', 'exile_result'):
                if not set(e.affected_players) <= alive:
                    raise ValueError('repeated or invalid public elimination')
                alive.difference_update(e.affected_players)
        normal = [v for v in votes if v.temporal_state.phase.value == 'vote']
        tied = _vote_winner(normal[0])
        pk = len(tied) > 1
        pk_changes = [e for e in events if e.event_type == 'phase_change'
                      and e.temporal_state.phase.value in ('pk_discussion', 'pk_vote')]
        if [e.temporal_state.phase.value for e in pk_changes] != (['pk_discussion', 'pk_vote'] if pk else []):
            raise ValueError('PK phase evidence disagrees with the complete ballot resolution')
        if pk:
            speakers = [e.speaker for e in events if e.event_type == 'public_speech'
                        and e.temporal_state.phase.value == 'pk_discussion']
            if len(speakers) != len(tied) or set(speakers) != set(tied):
                raise ValueError('PK discussion is missing/repeating a tied speaker')
        return {**base, 'day_status': 'COMPLETE', 'reason': None,
            'resolution': 'PK' if pk else 'NORMAL',
            'final_exile_event_id': final.event_id, 'final_exile_event_index': final.event_index,
            'final_exile_phase': final.temporal_state.phase.value,
            'exiled_seats': list(final.affected_players), 'no_exile': not final.affected_players,
            'final_exile_record_sha256': sha256_bytes(canonical_json_bytes(final.to_record()))}
    except (Phase2DataError, ValueError, IndexError) as error:
        return {**base, 'day_status': 'UNOBSERVABLE', 'reason': 'INCOMPLETE_OR_INCONSISTENT_DAY_RESOLUTION',
                'error_type': type(error).__name__, 'error': str(error)}
