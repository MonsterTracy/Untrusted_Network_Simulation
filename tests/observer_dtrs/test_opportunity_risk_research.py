"""Artificial CPU contracts for the new O/T targets; no real effects are read."""
from copy import deepcopy
import inspect
import random

import numpy as np
import pytest

from werewolf.observer_dtrs import opportunity_risk_research as study
from tests.observer_dtrs.test_public_behavior_research import future_fixture,toy_arrays
from tests.tom.test_counterfactual_consumer import PublicWitness,P
from werewolf.observer_dtrs.public_behavior_research import (
    observe_after_speech,public_features,fit_target,target_logits,paired_game_cv,game_weights,
)
from werewolf.artifact_io import canonical_json_bytes


@pytest.mark.parametrize('phase',['discussion','pk_discussion'])
def test_no_speech_is_zero_only_for_new_complete_window_event(phase):
    pre,h,a,cut,_=future_fixture(phase=phase,batches=())
    old=observe_after_speech(pre,P[1],h,a,cut);frozen=deepcopy(old)
    out=study.full_window_targets(old)
    assert out['O']==0 and out['T']==0 and out['Z'] is None
    assert out['status']=='NO_FUTURE_SPEECH' and old==frozen
    assert out['new_status']=='COMPLETE_NO_OPPORTUNITY_EVENT_ZERO'
    assert out['target_speech_event_id'] and out['cutoff_event_id']


@pytest.mark.parametrize('batches,z',[
    ((('hit',),),1),((('clear',),),0),((('empty',),),0),
    ((('other',),),0),((('hit',),('clear',)),1),
])
def test_opportunity_and_conditional_event_are_distinct(batches,z):
    pre,h,a,cut,_=future_fixture(batches=batches)
    old=observe_after_speech(pre,P[1],h,a,cut)
    out=study.observe_complete_window(pre,P[1],h,a,cut)
    assert out['O']==1 and out['T']==z==out['Z']
    assert all(out[k]==v for k,v in old.items())


@pytest.mark.parametrize('batches,status',[
    ((('failed',),),'V1_FAILED'),((('hit','clear'),),'V1_CONFLICT'),
    ((('hit',),('failed',)),'V1_FAILED'),
])
def test_speech_exists_even_when_v1_prevents_event_label(batches,status):
    pre,h,a,cut,_=future_fixture(batches=batches)
    out=study.observe_complete_window(pre,P[1],h,a,cut)
    assert out['O']==1 and out['T'] is None and out['Z'] is None
    assert out['status']==status and out['new_status']=='OPPORTUNITY_OBSERVED_EVENT_UNKNOWN'


def test_missing_v1_keeps_known_opportunity_but_unknown_event():
    pre,h,a,cut,_=future_fixture()
    event=next(e for e in h.events if e.event_type=='public_speech' and e.speaker==P[1])
    del a[event.event_id]
    out=study.observe_complete_window(pre,P[1],h,a,cut)
    assert out['O']==1 and out['T'] is None and out['status']=='V1_MISSING'


def test_incomplete_or_truncated_window_is_not_a_structural_zero():
    pre,h,a,cut,_=future_fixture(batches=())
    out=study.observe_complete_window(pre,P[1],h,a,{**cut,'day_status':'UNOBSERVABLE'})
    assert out['O'] is None and out['T'] is None and out['Z'] is None
    pre=PublicWitness().pre()
    out=study.observe_complete_window(pre,P[1],pre.public_event_history,
        {x.event_id:x for x in pre.v1_annotations},{'game_id':pre.game_id,'day':1,'day_status':'UNOBSERVABLE'})
    assert out['O'] is None and out['T'] is None and out['status']=='TARGET_SPEECH_MISSING'


@pytest.mark.parametrize('field,value',[
    ('game_id','wrong-game'),('day',2),('final_exile_event_id','wrong-event'),
    ('final_exile_event_index',0),('final_exile_event_index',100000),
])
def test_ineligible_source_or_cutoff_never_generates_zero(field,value):
    pre,h,a,cut,_=future_fixture(batches=())
    with pytest.raises(ValueError):study.observe_complete_window(pre,P[1],h,a,{**cut,field:value})


@pytest.mark.parametrize('kind',[
    'unknown_status','missing_target_witness','missing_cutoff_witness','unexpected_future',
    'duplicate_future','zero_with_old_Z','observed_without_future','observed_nonbinary_Z',
    'ambiguous_with_Z','incomplete_with_Z',
])
def test_invalid_conditional_evidence_does_not_silently_zero_fill(kind):
    pre,h,a,cut,_=future_fixture(batches=())
    old=observe_after_speech(pre,P[1],h,a,cut)
    if kind=='unknown_status':old['status']='UNTRUSTED_SOURCE'
    elif kind=='missing_target_witness':old['target_speech_event_id']=None
    elif kind=='missing_cutoff_witness':old['cutoff_event_id']=None
    elif kind=='unexpected_future':old['future_speech_ids']=['unexpected']
    elif kind=='duplicate_future':old['future_speech_ids']=['dup','dup']
    elif kind=='zero_with_old_Z':old['Z']=0
    elif kind=='observed_without_future':old.update(status='OBSERVED',Z=0)
    elif kind=='observed_nonbinary_Z':old.update(status='OBSERVED',Z=True,future_speech_ids=['event'])
    elif kind=='ambiguous_with_Z':old.update(status='V1_FAILED',Z=0,future_speech_ids=['event'])
    elif kind=='incomplete_with_Z':old.update(status='DAY_INCOMPLETE',Z=0)
    with pytest.raises(ValueError):study.full_window_targets(old)


def test_future_outcomes_do_not_enter_pre_feature_or_rewrite_history():
    pre,h,a,cut,_=future_fixture(batches=(('hit',),))
    before=canonical_json_bytes(pre.to_record());x=public_features(pre,P[1])
    out=study.observe_complete_window(pre,P[1],h,a,cut)
    assert out['T']==1 and out['O']==1
    a={k:v for k,v in a.items() if k not in out['future_speech_ids']}
    ambiguous=study.observe_complete_window(pre,P[1],h,a,cut)
    assert ambiguous['O']==1 and ambiguous['T'] is None
    assert public_features(pre,P[1])==x and canonical_json_bytes(pre.to_record())==before
    assert tuple(inspect.signature(public_features).parameters)==('pre','observer_id')


def inherited_fixture():
    # Three full-reference games have sizes 2/4/6; observed counts are 2/1/3.
    X=np.column_stack([np.arange(12,dtype=float),np.full(12,3.)])
    games=['a']*2+['b']*4+['c']*6
    o=np.array([1,1,1,0,0,0,1,1,1,0,0,0],dtype=int)
    t=np.array([0,1,0,0,0,0,1,0,1,0,0,0],dtype=int)
    return X,o,t,games


def test_conditional_weights_inherit_full_game_population_not_old_reweighting():
    X,o,t,games=inherited_fixture();keep=o==1
    weights=game_weights(games)[keep];weights/=weights.sum()
    m=study.fit_conditional(X,o,t,games)
    np.testing.assert_allclose(m['mean'],weights@X[keep],rtol=0,atol=1e-14)
    old_weights=game_weights(np.asarray(games)[keep])
    assert abs((old_weights@X[keep])[0]-m['mean'][0])>.1
    variance=weights@((X[keep]-np.asarray(m['mean']))**2)
    assert m['std'][0]==pytest.approx(np.sqrt(variance[0])) and m['std'][1]==1
    assert m['full_reference_rows']==12 and m['full_reference_games']==3
    assert m['train_rows']==6 and m['train_games']==3 and m['positive']==3
    assert m['tau']==1/6 and m['weight_rule'].startswith('inherit_1_over_full_reference')


def test_all_opportunity_case_reuses_original_objective_and_predictions():
    X,y,games=toy_arrays();o=np.ones(len(y),dtype=int)
    original=fit_target(X,y,games);new=study.fit_conditional(X,o,y,games)
    np.testing.assert_allclose(new['mean'],original['mean'],rtol=0,atol=1e-14)
    np.testing.assert_allclose(new['std'],original['std'],rtol=0,atol=1e-14)
    np.testing.assert_allclose(new['beta'],original['beta'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(target_logits(new,X),target_logits(original,X),rtol=0,atol=1e-12)


def test_unobserved_covariate_rows_do_not_affect_conditional_scale_but_game_sizes_do():
    X,o,t,games=inherited_fixture();m=study.fit_conditional(X,o,t,games)
    change=X.copy();change[o==0]=[1e8,-1e8]
    same=study.fit_conditional(change,o,t,games)
    assert same==m


@pytest.mark.parametrize('kind',['no_opportunity','all_negative','all_positive'])
def test_conditional_class_support_stops_without_constant_fallback(kind):
    X,o,t,games=inherited_fixture()
    if kind=='no_opportunity':o[:]=0;t[:]=0
    elif kind=='all_negative':t[:]=0
    else:t=o.copy()
    with pytest.raises(ValueError,match='CONDITIONAL_CLASS_SUPPORT'):study.fit_conditional(X,o,t,games)


@pytest.mark.parametrize('kind',['nan_X','bad_shape','bad_O','bad_T','T_exceeds_O','bool_O','float_T','unknown_T','short_games'])
def test_invalid_or_unknown_training_targets_stop(kind):
    X,o,t,games=inherited_fixture()
    if kind=='nan_X':X[0,0]=np.nan
    elif kind=='bad_shape':X=X[:-1]
    elif kind=='bad_O':o[0]=2
    elif kind=='bad_T':t[0]=2
    elif kind=='T_exceeds_O':t[3]=1
    elif kind=='bool_O':o=o.astype(bool)
    elif kind=='float_T':t=t.astype(float)
    elif kind=='unknown_T':t=t.astype(object);t[3]=None
    elif kind=='short_games':games=games[:-1]
    with pytest.raises(ValueError):study.fit_conditional(X,o,t,games)


def test_held_out_labels_cannot_change_fit_or_prediction():
    X,o,t,games=inherited_fixture();train=np.arange(9);test=np.arange(9,12)
    m=study.fit_conditional(X[train],o[train],t[train],np.array(games)[train])
    other_o=o.copy();other_t=t.copy();other_o[test]=1;other_t[test]=[0,1,1]
    same=study.fit_conditional(X[train],other_o[train],other_t[train],np.array(games)[train])
    assert same==m
    np.testing.assert_array_equal(target_logits(m,X[test]),target_logits(same,X[test]))


def test_every_pre_gets_product_without_actual_future_opportunity_gate():
    assert tuple(inspect.signature(study.product_scores).parameters)==('opportunity_logits','conditional_logits','target')
    p,b,ll=study.product_scores(np.array([0.,0.]),np.array([0.,0.]),np.array([0,1]))
    np.testing.assert_array_equal(p,[.25,.25])
    np.testing.assert_array_equal(b,[.0625,.5625])
    np.testing.assert_allclose(ll,[-np.log(.75),-np.log(.25)],rtol=0,atol=1e-15)
    with pytest.raises(TypeError):study.product_scores([0.],[0.],[0],opportunity=[0])


def test_probability_is_independent_of_scoring_targets_and_not_zeroed_for_negatives():
    a=np.array([-2.,0.,2.]);b=np.array([1.,0.,-1.])
    p0,_,_=study.product_scores(a,b,np.zeros(3,int))
    p1,_,_=study.product_scores(a,b,np.ones(3,int))
    np.testing.assert_array_equal(p0,p1)
    assert np.all((p0>0)&(p0<1))
    np.testing.assert_array_equal(p0,study.sigmoid(a)*study.sigmoid(b))


@pytest.mark.parametrize('a,b',[(-1000.,-1000.),(-1000.,1000.),(1000.,-1000.),(1000.,1000.)])
@pytest.mark.parametrize('target',[0,1])
def test_extreme_product_probability_and_marginal_log_loss_remain_stable(a,b,target):
    p,bs,ll=study.product_scores([a],[b],[target])
    assert np.isfinite(p).all() and np.isfinite(bs).all() and np.isfinite(ll).all()
    assert 0<=p[0]<=1 and bs[0]==(p[0]-target)**2
    if a==b==1000. and target==0:assert ll[0]==pytest.approx(1000.-np.log(2),abs=1e-12)
    if a==b==-1000. and target==1:assert ll[0]==2000.
    if target==1 and a+b==0:assert ll[0]==1000.


@pytest.mark.parametrize('a,b,y',[([np.nan],[0.],[0]),([0.],[np.inf],[1]),([0.,1.],[0.],[0]),([0.],[0.],[2]),([[0.]],[[0.]],[[0]])])
def test_invalid_product_arrays_are_rejected(a,b,y):
    with pytest.raises(ValueError):study.product_scores(a,b,y)


def test_source_outer_game_splits_keep_all_rows_and_no_game_overlap():
    blocks=np.array([(s,o) for s in range(5) for o in range(5) for _ in range(2)],int)
    games=np.array([f'g{s}-{o}' for s,o in blocks]);seen=np.zeros(len(games),int)
    for (source,outer),train,test in paired_game_cv(blocks,games):
        assert not set(games[train])&set(games[test]) and np.all(blocks[train,0]==source)
        seen[test]+=1
    np.testing.assert_array_equal(seen,np.ones(len(games),int))
    games[2]=games[0]
    with pytest.raises(ValueError,match='same game'):list(paired_game_cv(blocks,games))


def test_new_pure_math_and_targets_do_not_consume_rng():
    py=random.getstate();nr=np.random.get_state()
    X,o,t,games=inherited_fixture();m=study.fit_conditional(X,o,t,games)
    study.product_scores(target_logits(m,X),target_logits(m,X),t)
    pre,h,a,cut,_=future_fixture(batches=())
    study.observe_complete_window(pre,P[1],h,a,cut)
    assert random.getstate()==py
    now=np.random.get_state();assert now[0]==nr[0] and now[2:]==nr[2:]
    np.testing.assert_array_equal(now[1],nr[1])


def test_opportunity_cli_requires_explicit_portable_input_paths():
    from pathlib import Path
    from scripts import analyze_observer_dtrs_opportunity_risk as cli
    expected={'prior_run':'inputs/prior-run','prior_archive':'inputs/prior-run.zip','public_pre_file':'inputs/public-pres.jsonl',
              'public_archive':'inputs/public-streams.zip','d2_run':'inputs/d2','plan':'inputs/analysis-plan.md','output':'outputs/new-run'}
    options=[part for name,value in expected.items() for part in ('--'+name.replace('_','-'),value)]
    args=cli.argument_parser().parse_args(options)
    for name,value in expected.items():assert getattr(args,name)==Path(value)
    assert '/Users/' not in cli.argument_parser().format_help()
    for option in options[::2]:
        omitted=options.copy();i=omitted.index(option);del omitted[i:i+2]
        with pytest.raises(SystemExit) as error:cli.argument_parser().parse_args(omitted)
        assert error.value.code==2


def test_opportunity_cli_help_loads_without_external_result_python(monkeypatch,capsys):
    import runpy
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[2];script=root/'scripts/analyze_observer_dtrs_opportunity_risk.py'
    original_open=Path.open
    def guarded_open(path,*args,**kwargs):
        if 'Untrusted_Network_Simulation-results' in path.parts:raise AssertionError('no historical files during CLI help')
        return original_open(path,*args,**kwargs)
    monkeypatch.chdir(root.parent)
    monkeypatch.setattr(sys,'path',list(sys.path));monkeypatch.setattr(sys,'argv',[str(script),'--help'])
    monkeypatch.setitem(sys.modules,'public_speech_day_audit',None);monkeypatch.setattr(Path,'open',guarded_open)
    with pytest.raises(SystemExit) as error:runpy.run_path(str(script),run_name='__main__')
    assert error.value.code==0 and '--prior-archive' in capsys.readouterr().out


def test_opportunity_archive_contexts_open_the_supplied_paths_without_files():
    import ast
    import inspect
    from copy import deepcopy
    from pathlib import Path
    from types import SimpleNamespace
    from scripts import analyze_observer_dtrs_opportunity_risk as cli
    opened=[]
    class MemoryArchive:
        def __enter__(self):return self
        def __exit__(self,*args):return False
    def open_archive(path):
        opened.append(path)
        return MemoryArchive()
    tree=ast.parse(inspect.getsource(cli.main))
    contexts=[item for node in ast.walk(tree) if isinstance(node,ast.With) for item in node.items
              if any(isinstance(x,ast.Name) and x.id=='zipfile' for x in ast.walk(item.context_expr))]
    namespace={'zipfile':SimpleNamespace(ZipFile=open_archive),
               'archive':Path('relocated/prior.zip'),'public_archive':Path('relocated/public.zip')}
    for item in contexts:
        block=ast.Module(body=[ast.With(items=[deepcopy(item)],body=[ast.Pass()])],type_ignores=[])
        exec(compile(ast.fix_missing_locations(block),'<CLI archive context smoke>','exec'),namespace)
    assert opened==[Path('relocated/prior.zip'),Path('relocated/public.zip')]
