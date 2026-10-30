"""Synthetic public-event and target-map contracts; no historical Z or models.

Uses reviewed public witness constructors, never an environment/game loop.
All future labels are separate from PRE feature construction.
"""
from copy import deepcopy
from dataclasses import replace
import inspect
import random

import numpy as np
import pytest

from werewolf.observer_dtrs import public_behavior_research as study
from tests.tom.test_counterfactual_consumer import PublicWitness, P
from tests.observer_dtrs.test_speech_evidence import legal_population_pre
from werewolf.artifact_io import canonical_json_bytes
from werewolf.canonical_collection import (
    construct_v1_speech_annotation, freeze_public_event_history,
    V1AnnotationStatus as S, V1PerceptionAttempt, V1SpeechAction,
    V1_SPEECH_PROMPT_VERSION, V1_SPEECH_PARSER_VERSION,
)
from werewolf.observer_dtrs.context import contexts_for_pre


def add_annotation(w, speaker, actions=(), *, status=None):
    """Construct an explicitly artificial V1, without an Actor/Parser call."""
    w.event('public_speech', speaker=speaker, raw_text='Controlled public fixture.')
    status=status or (S.OK if actions else S.NO_ACTION)
    attempt=V1PerceptionAttempt(1,'test-call-'+str(len(w.events)),'fixture','fixture',
        V1_SPEECH_PROMPT_VERSION,V1_SPEECH_PARSER_VERSION,status,
        None if status is S.ERROR else 'Controlled parser fixture.',
        'fixture_error' if status is S.ERROR else None,
        'controlled failure' if status is S.ERROR else None)
    a=construct_v1_speech_annotation(freeze_public_event_history(w.events),
        status=status,actions=tuple(actions),attempts=(attempt,))
    w.annotations.append(a)
    return a


def future_fixture(*, phase='discussion', batches=(('hit',),), observer=P[1], later_day=False):
    """One fixed target speech, optional same-day observer turns, final cutoff."""
    w=PublicWitness(phase);pre=w.pre()
    target=add_annotation(w,P[0],(V1SpeechAction(P[0],'no_commitment',None),))
    for batch in batches:
        w.event('turn_start',speaker=observer)
        actions=[]
        for kind in batch:
            if kind in ('hit','clear'):
                actions.append(V1SpeechAction(observer,
                    'point_as_werewolf' if kind=='hit' else 'point_as_non_werewolf',P[0]))
            elif kind=='other':actions.append(V1SpeechAction(observer,'point_as_werewolf',P[4]))
            elif kind not in ('failed','empty'):raise AssertionError(kind)
        add_annotation(w,observer,actions,status=S.ERROR if 'failed' in batch else None)
    w.event('phase_change',day=1,phase='pk_vote' if phase=='pk_discussion' else 'vote')
    # The label helper consumes only a bound final public cutoff, not roles/outcome.
    w.event('vote_result',votes=[{'voter':p,'target':None} for p in P])
    w.event('exile_result',exiled_players=[])
    stop=len(w.events)-1
    cutoff={'game_id':pre.game_id,'day':1,'day_status':'COMPLETE','final_exile_event_index':stop,
            'final_exile_event_id':w.events[stop]['event_id']}
    if later_day:
        w.event('phase_change',day=1,phase='night');w.event('death_announcement',dead_players=[])
        w.event('phase_change',day=2,phase='discussion');w.event('turn_start',speaker=observer)
        add_annotation(w,observer,(V1SpeechAction(observer,'point_as_werewolf',P[0]),))
    h=freeze_public_event_history(w.events)
    return pre,h,{a.event_id:a for a in w.annotations},cutoff,target


@pytest.mark.parametrize('phase',['discussion','pk_discussion'])
@pytest.mark.parametrize('batches,z',[
    ((('hit',),),1),((('clear',),),0),((('other',),),0),((('empty',),),0),
    ((('hit',),('clear',)),1),((('clear',),('hit',)),1),
])
def test_current_actor_explicit_public_target_in_both_phases(phase,batches,z):
    pre,h,a,cut,target=future_fixture(phase=phase,batches=batches)
    before=canonical_json_bytes(pre.to_record())
    out=study.observe_after_speech(pre,P[1],h,a,cut)
    assert out['status']=='OBSERVED' and out['Z']==z
    assert out['target_speech_event_id']==target.event_id
    assert target.event_id not in out['future_speech_ids']
    assert out['cutoff_event_id']==cut['final_exile_event_id']
    assert canonical_json_bytes(pre.to_record())==before


@pytest.mark.parametrize('batches,status',[
    ((),'NO_FUTURE_SPEECH'),((('failed',),),'V1_FAILED'),
    ((('hit','clear'),),'V1_CONFLICT'),
])
def test_missing_opportunity_failed_v1_and_same_turn_conflict_never_zero(batches,status):
    pre,h,a,cut,_=future_fixture(batches=batches)
    out=study.observe_after_speech(pre,P[1],h,a,cut)
    assert out['status']==status and out['Z'] is None


def test_partial_positive_does_not_hide_other_unparseable_future_turn():
    pre,h,a,cut,_=future_fixture(batches=(('hit',),('failed',)))
    out=study.observe_after_speech(pre,P[1],h,a,cut)
    assert out['status']=='V1_FAILED' and out['Z'] is None
    assert len(out['positive_event_ids'])==1


def test_cross_time_reversal_is_not_same_annotation_conflict():
    pre,h,a,cut,_=future_fixture(batches=(('hit',),('clear',)))
    out=study.observe_after_speech(pre,P[1],h,a,cut)
    assert out['Z']==1 and out['across_time_accuse_and_clear'] is True
    assert out['conflict_event_ids']==[]


def test_missing_v1_does_not_become_negative_or_positive():
    pre,h,a,cut,_=future_fixture()
    future=next(e for e in h.events if e.event_type=='public_speech' and e.speaker==P[1])
    del a[future.event_id]
    out=study.observe_after_speech(pre,P[1],h,a,cut)
    assert out['status']=='V1_MISSING' and out['Z'] is None


def test_truncated_stream_and_incomplete_day_keep_unknown():
    pre=PublicWitness().pre()
    a={x.event_id:x for x in pre.v1_annotations}
    out=study.observe_after_speech(pre,P[1],pre.public_event_history,a,{'game_id':pre.game_id,'day':1,'day_status':'UNOBSERVABLE'})
    assert out['status']=='TARGET_SPEECH_MISSING' and out['Z'] is None
    pre,h,a,cut,_=future_fixture()
    out=study.observe_after_speech(pre,P[1],h,a,{**cut,'day_status':'UNOBSERVABLE'})
    assert out['status']=='DAY_INCOMPLETE' and out['Z'] is None


@pytest.mark.parametrize('field,value',[
    ('game_id','different-game'),('day',2),('final_exile_event_index',0),('final_exile_event_index',True),
    ('final_exile_event_index',100000),('final_exile_event_id','wrong-event'),
])
def test_day_cutoff_identity_is_checked(field,value):
    pre,h,a,cut,_=future_fixture()
    with pytest.raises(ValueError):study.observe_after_speech(pre,P[1],h,a,{**cut,field:value})


def test_future_annotation_cross_event_identity_is_rejected():
    pre,h,a,cut,_=future_fixture()
    future=next(e for e in h.events if e.event_type=='public_speech' and e.speaker==P[1])
    a[future.event_id]=replace(a[future.event_id],event_index=future.event_index+1)
    with pytest.raises(ValueError):study.observe_after_speech(pre,P[1],h,a,cut)


def test_wrong_pre_annotation_and_wrong_target_speaker_rejected():
    pre,h,a,cut,target=future_fixture()
    past=pre.v1_annotations[0];del a[past.event_id]
    with pytest.raises(ValueError,match='bind'):study.observe_after_speech(pre,P[1],h,a,cut)
    pre,h,a,cut,target=future_fixture()
    target_event=h.events[len(pre.public_event_history.events)]
    mutated=replace(target_event,speaker=P[2])
    h=replace(h,events=tuple(mutated if e.event_id==target.event_id else e for e in h.events))
    with pytest.raises(ValueError,match='target speech'):study.observe_after_speech(pre,P[1],h,a,cut)


def test_later_day_and_past_accusations_cannot_be_future_labels():
    pre,h,a,cut,_=future_fixture(batches=(),later_day=True)
    out=study.observe_after_speech(pre,P[1],h,a,cut)
    assert out['status']=='NO_FUTURE_SPEECH' and out['Z'] is None
    pre,h,a,cut,_=future_fixture(batches=(('empty',),),later_day=True)
    out=study.observe_after_speech(pre,P[1],h,a,cut)
    assert out['status']=='OBSERVED' and out['Z']==0
    assert not out['positive_event_ids']


@pytest.mark.parametrize('bad',[P[0],'player8',None])
def test_actor_or_nonexistent_observer_is_rejected(bad):
    pre,h,a,cut,_=future_fixture()
    with pytest.raises(ValueError):study.public_features(pre,bad)
    with pytest.raises(ValueError):study.observe_after_speech(pre,bad,h,a,cut)


def test_features_have_only_frozen_pre_input_and_future_counterfactual_isolation():
    pre,h,a,cut,_=future_fixture(batches=(('hit',),))
    before=canonical_json_bytes(pre.to_record())
    features=study.public_features(pre,P[1])
    assert len(features)==len(study.PUBLIC_NAMES) and np.isfinite(features).all()
    assert tuple(inspect.signature(study.public_features).parameters)==('pre','observer_id')
    with pytest.raises(TypeError):study.public_features(pre,P[1],future=h)
    result=study.observe_after_speech(pre,P[1],h,a,cut)
    assert result['Z']==1
    a={k:v for k,v in a.items() if k not in result['future_speech_ids']}
    assert study.observe_after_speech(pre,P[1],h,a,cut)['Z'] is None
    assert study.public_features(pre,P[1])==features
    assert canonical_json_bytes(pre.to_record())==before


@pytest.mark.parametrize('dead',[(),(P[1],),(P[5],)])
def test_legal_team_context_separate_from_public_features(dead):
    pre=legal_population_pre('discussion',dead)
    for c in contexts_for_pre(pre,P[:2]):
        v=study.legal_features(c)
        assert len(v)==len(study.LEGAL_NAMES)==8
        assert v[0]==c.wolf_count==(1 if P[1] in dead else 2)
        assert sum(v[1:])==1 and v[2]==1
        assert not set(study.LEGAL_NAMES)&set(study.PUBLIC_NAMES)


def toy_arrays():
    X=np.array([[0.,2.],[1.,3.],[2.,2.],[0.,2.],[1.,3.],[2.,2.],[1.,2.],[0.,4.]])
    y=np.array([0,1,1,1,0,0,1,0],dtype=int)
    games=['g0','g0','g1','g1','g1','g2','g3','g3']
    return X,y,games


def test_training_scaler_is_game_equal_and_fixed_without_test_information():
    X,y,games=toy_arrays();model=study.fit_target(X,y,games)
    weights=study.game_weights(games)
    np.testing.assert_allclose(weights@X,model['mean'],rtol=0,atol=1e-15)
    for game in set(games):assert weights[np.array(games)==game].sum()==pytest.approx(.25)
    assert model['train_rows']==len(X) and model['train_games']==4
    assert model['positive']==int(y.sum()) and model['tau']==1/len(X)
    frozen=deepcopy(model)
    p1=study.target_logits(model,np.array([[5.,8.],[-3.,2.]]))
    study.target_logits(model,np.array([[1e6,1e8],[-1e6,-1e8]]))
    assert model==frozen and np.isfinite(p1).all()


def test_zero_variance_columns_have_deterministic_scale_without_fallback_model():
    X,y,games=toy_arrays();X[:,1]=3
    m=study.fit_target(X,y,games)
    assert m['std'][1]==1 and m['mean'][1]==pytest.approx(3)
    assert np.isfinite(study.target_logits(m,X)).all()


@pytest.mark.parametrize('y',[[0]*8,[1]*8])
def test_single_class_training_explicitly_stops(y):
    X,_,games=toy_arrays()
    with pytest.raises(ValueError,match='SINGLE_CLASS'):study.fit_target(X,y,games)


@pytest.mark.parametrize('bad',['nan','infinity','target_shape','target_float','target_bool','target_outside','games_length'])
def test_invalid_training_arrays_and_targets_rejected(bad):
    X,y,games=toy_arrays()
    if bad=='nan':X[0,0]=np.nan
    elif bad=='infinity':X[0,0]=np.inf
    elif bad=='target_shape':y=y[:-1]
    elif bad=='target_float':y=y.astype(float)
    elif bad=='target_bool':y=y.astype(bool)
    elif bad=='target_outside':y[0]=2
    else:games=games[:-1]
    with pytest.raises(ValueError):study.fit_target(X,y,games)


def test_fixed_source_outer_split_is_complete_and_game_disjoint():
    blocks=np.array([(s,o) for s in range(5) for o in range(5) for _ in range(3)],dtype=int)
    games=np.array([f'g{s}-{o}' for s,o in blocks]);seen=np.zeros(len(blocks),int)
    splits=list(study.paired_game_cv(blocks,games))
    assert len(splits)==25
    for (source,outer),train,test in splits:
        assert not set(train)&set(test) and not set(games[train])&set(games[test])
        assert np.all(blocks[train,0]==source) and np.all(blocks[train,1]!=outer)
        assert np.all(blocks[test]==(source,outer));seen[test]+=1
    assert np.all(seen==1)


@pytest.mark.parametrize('blocks',[
    np.zeros((5,2),int),np.array([[5,0],[0,0]]),np.array([[0.,0.],[0.,1.]]),
    np.zeros((25,3),int),np.zeros((2,),int),
])
def test_invalid_or_unsupported_block_population_stops(blocks):
    with pytest.raises(ValueError):list(study.paired_game_cv(blocks,['g'+str(i) for i in range(len(blocks))]))


def test_same_game_cannot_cross_source_or_outer_block_even_if_all_blocks_exist():
    blocks=np.array([(s,o) for s in range(5) for o in range(5)],dtype=int)
    games=['g'+str(i) for i in range(25)]
    games[1]=games[0]
    with pytest.raises(ValueError,match='same game'):list(study.paired_game_cv(blocks,games))
    with pytest.raises(ValueError):list(study.paired_game_cv(blocks,games[:-1]))


def test_test_labels_cannot_change_training_scaler_fit_or_predictions():
    X,y,games=toy_arrays()
    train=np.array([0,1,2,3,4,5]);test=np.array([6,7])
    first=study.fit_target(X[train],y[train],[games[i] for i in train])
    alternative=y.copy();alternative[test]=1-alternative[test]
    second=study.fit_target(X[train],alternative[train],[games[i] for i in train])
    assert first==second
    np.testing.assert_array_equal(study.target_logits(first,X[test]),study.target_logits(second,X[test]))
    # Labels change evaluation only; held-out single-class labels remain scoreable.
    p,b,l=study.row_scores(study.target_logits(first,X[test]),alternative[test])
    assert len(p)==len(test) and np.isfinite(b).all() and np.isfinite(l).all()


def test_extreme_logit_scores_are_stable_without_probability_clipping():
    p,b,l=study.row_scores(np.array([-1000.,0.,1000.]),np.array([1,0,0]))
    np.testing.assert_array_equal(p,np.array([0.,.5,1.]))
    np.testing.assert_allclose(b,[1.,.25,1.],rtol=0,atol=0)
    np.testing.assert_allclose(l,[1000.,np.log(2),1000.],rtol=0,atol=0)
    p,b,l=study.row_scores(np.array([-1000.,1000.]),np.array([0,1]))
    np.testing.assert_array_equal(l,[0.,0.])


@pytest.mark.parametrize('logits,y',[
    ([np.nan],[0]),([np.inf],[1]),([0.,1.],[0]),([0.],[2]),
])
def test_invalid_prediction_scores_rejected(logits,y):
    with pytest.raises(ValueError):study.row_scores(logits,y)


def test_training_and_feature_calls_do_not_consume_python_or_numpy_rng():
    py=random.getstate();nr=np.random.get_state()
    X,y,games=toy_arrays();m=study.fit_target(X,y,games)
    study.row_scores(study.target_logits(m,X),y)
    study.public_features(PublicWitness().pre(),P[1])
    assert random.getstate()==py
    end=np.random.get_state()
    assert end[0]==nr[0] and end[2:]==nr[2:]
    np.testing.assert_array_equal(end[1],nr[1])


def test_public_behavior_cli_requires_explicit_portable_input_paths():
    from pathlib import Path
    from scripts import analyze_observer_dtrs_public_behavior as cli
    expected={'dataset':'inputs/dataset','d2_run':'inputs/d2','public_pre_file':'inputs/public-pres.jsonl',
              'public_archive':'inputs/public-streams.zip','plan':'inputs/analysis-plan.md','output':'outputs/new-run'}
    options=[part for name,value in expected.items() for part in ('--'+name.replace('_','-'),value)]
    args=cli.argument_parser().parse_args(options)
    for name,value in expected.items():assert getattr(args,name)==Path(value)
    assert '/Users/' not in cli.argument_parser().format_help()
    for option in options[::2]:
        omitted=options.copy();i=omitted.index(option);del omitted[i:i+2]
        with pytest.raises(SystemExit) as error:cli.argument_parser().parse_args(omitted)
        assert error.value.code==2


def test_public_behavior_cli_help_loads_without_external_result_python(monkeypatch,capsys):
    import runpy
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[2];script=root/'scripts/analyze_observer_dtrs_public_behavior.py'
    original_open=Path.open
    def guarded_open(path,*args,**kwargs):
        if 'Untrusted_Network_Simulation-results' in path.parts:raise AssertionError('no historical files during CLI help')
        return original_open(path,*args,**kwargs)
    monkeypatch.chdir(root.parent)
    monkeypatch.setattr(sys,'path',list(sys.path));monkeypatch.setattr(sys,'argv',[str(script),'--help'])
    monkeypatch.setitem(sys.modules,'public_speech_day_audit',None);monkeypatch.setattr(Path,'open',guarded_open)
    with pytest.raises(SystemExit) as error:runpy.run_path(str(script),run_name='__main__')
    assert error.value.code==0 and '--public-archive' in capsys.readouterr().out


def test_public_cli_uses_versioned_repository_public_validators():
    from scripts import analyze_observer_dtrs_public_behavior as cli
    assert cli.parse_public_game is study.parse_public_game and cli.public_day_witness is study.public_day_witness
    assert study.parse_public_game.__module__==study.__name__ and study.public_day_witness.__module__==study.__name__
    assert study.PUBLIC_AUDIT_ORIGIN_SHA256=='50996ec5b66277fdd46565dbbbfc5c2933097838ef5e6faff2b312b70b695b61'


def test_repository_public_parser_roundtrips_without_historical_python(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules,'public_speech_day_audit',None)
    _,history,annotations,_,_=future_fixture()
    actual_history,actual_annotations=study.parse_public_game(history.to_records(),[a.to_record() for a in annotations.values()])
    assert actual_history.to_records()==history.to_records() and actual_annotations==annotations


@pytest.mark.parametrize('kind',['duplicate','corrupt_digest'])
def test_repository_public_parser_rejects_corrupt_or_duplicate_annotations(kind):
    _,history,annotations,_,_=future_fixture();records=[a.to_record() for a in annotations.values()]
    if kind=='duplicate':records.append(records[-1])
    else:records[0]={**records[0],'annotation_digest':'0'*64}
    with pytest.raises(ValueError):study.parse_public_game(history.to_records(),records)


def _complete_public_day_history(kind):
    """Pure in-memory public ballot fixture; no environment execution."""
    events=[]
    def add(event_type,**fields):events.append({'event_id':f'e{len(events)}','event_index':len(events),'event_type':event_type,**fields})
    add('phase_change',day=0,phase='night');add('death_announcement',dead_players=[])
    add('phase_change',day=1,phase='discussion')
    for player in P:
        add('turn_start',speaker=player);add('public_speech',speaker=player,raw_text='Controlled public speech.')
    add('phase_change',day=1,phase='vote')
    if kind.startswith('pk'):
        targets=(P[1],P[2],P[0],P[0],P[1],P[2],None)
        add('vote_result',votes=[{'voter':player,'target':target} for player,target in zip(P,targets)])
        add('exile_result',exiled_players=[])
        if kind=='pk_incomplete':return freeze_public_event_history(events)
        add('phase_change',day=1,phase='pk_discussion')
        for player in P[:3]:
            add('turn_start',speaker=player);add('public_speech',speaker=player,raw_text='Controlled PK speech.')
        add('phase_change',day=1,phase='pk_vote')
        if kind=='pk_tie':targets,exile=(P[1],P[0],P[1],P[0],None,None,None),[]
        elif kind=='pk_abstain':targets,exile=(None,)*7,[]
        elif kind=='pk_illegal_target':targets,exile=(P[3],P[3],P[3],P[0],P[3],P[3],P[3]),[P[3]]
        else:targets,exile=(P[1],P[0],P[0],P[0],P[0],P[0],P[0]),[P[0]]
    else:
        targets=(None,)*7 if kind=='normal_abstain' else tuple(P[3] if player==P[2] else P[2] for player in P)
        exile=[] if kind=='normal_abstain' else [P[2]]
    ballot=[{'voter':player,'target':target} for player,target in zip(P,targets)]
    if kind=='normal_missing_voter':ballot=ballot[:-1]
    add('vote_result',votes=ballot);add('exile_result',exiled_players=exile)
    return freeze_public_event_history(events)


@pytest.mark.parametrize('kind,resolution,no_exile',[
    ('normal_exile','NORMAL',False),('normal_abstain','NORMAL',True),('pk_exile','PK',False),
    ('pk_tie','PK',True),('pk_abstain','PK',True)])
def test_repository_day_witness_preserves_complete_normal_pk_and_empty_exile(kind,resolution,no_exile):
    history=_complete_public_day_history(kind);result=study.public_day_witness(history,1)
    assert result['day_status']=='COMPLETE' and result['resolution']==resolution
    assert result['no_exile'] is no_exile and result['final_exile_event_id']==history.events[-1].event_id
    assert result['game_terminal_verified'] is False and result['wolf_exile_target_computed'] is False


@pytest.mark.parametrize('kind',['pk_incomplete','normal_missing_voter','pk_illegal_target'])
def test_repository_day_witness_keeps_incomplete_or_illegal_resolution_unknown(kind):
    result=study.public_day_witness(_complete_public_day_history(kind),1)
    assert result['day_status']=='UNOBSERVABLE' and 'no_exile' not in result


def test_repository_day_witness_without_ballot_is_not_zero_outcome():
    result=study.public_day_witness(PublicWitness().pre().public_event_history,1)
    assert result['day_status']=='UNOBSERVABLE' and 'exiled_seats' not in result
