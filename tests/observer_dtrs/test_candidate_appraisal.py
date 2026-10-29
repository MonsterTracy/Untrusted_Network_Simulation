"""Artificial cognitive risk/independent preference tests; no real model/data."""
from dataclasses import replace
from fractions import Fraction as F
import json,random
from pathlib import Path
import numpy as np
import pytest

from scripts.speech_planning import PlanningAction as A, SpeechPlan, generate_candidates
from tests.observer_dtrs.test_speech_evidence import legal_population_pre, panel
from tests.observer_dtrs.test_speech_runtime import witness, P
from werewolf.observer_dtrs.context import contexts_for_pre
from werewolf.observer_dtrs.contextual_policy import contextual_loss
from werewolf.observer_dtrs.decision import Region, TiePolicy, LossSpec, decide, candidate_losses, thresholds
from werewolf.observer_dtrs.speech_evidence import process_cognition
from werewolf.observer_dtrs.candidate_appraisal import (
    BasisKind as B, GateMode as G, PublicReference, CandidateBasis,
    bind_bases, artificial_action_values, appraise_candidates, judgment_risk,
)


def setup(p=F(1,2),*,loss=None,kind=B.ASSUME_HIT):
    pre=witness().pre();candidates=generate_candidates(pre)
    evidence=process_cognition(pre,panel=panel(pre,p,loss=loss))
    observer=evidence.selected.observer_id
    bases=tuple(CandidateBasis(f'b{j:02}',plan,kind,None if kind is B.PUBLIC_ONLY else observer)
                for j,plan in enumerate(candidates))
    catalog=bind_bases(pre,candidates=candidates,bases=bases)
    values=artificial_action_values(pre,candidates=candidates,
        scores=tuple(F(len(candidates)-j) for j in range(len(candidates))),fixture_id='not-real-effect')
    return pre,evidence,catalog,values


def review(pre,evidence,catalog,values,mode=G.THREE_WAY):
    return appraise_candidates(pre,evidence,catalog,mode=mode,values=values)


@pytest.mark.parametrize('tie',list(TiePolicy))
def test_original_context_costs_non_degeneracy_thresholds_and_scale(tie):
    expected={('discussion',2):(1,F(1,4),F(3,4)),('discussion',1):(2,F(1,7),F(3,5)),
              ('pk_discussion',2):(2,F(1,4),F(3,7)),('pk_discussion',1):(4,F(1,7),F(3,11))}
    for (phase,nW),(miss,beta,alpha) in expected.items():
        pre=legal_population_pre(phase,() if nW==2 else (P[1],))
        loss=contextual_loss(contexts_for_pre(pre,P[:2])[0],tie_policy=tie)
        th=thresholds(loss)
        assert loss.c_fp==1 and loss.c_fn==miss and th.beta==beta and th.alpha==alpha
        assert 0<th.beta<th.gamma<th.alpha<1 and th.regime=='NONDEGENERATE'
        assert loss.b0/loss.c_fp+loss.b1/loss.c_fn<1
        for p in (F(0),beta,(beta+alpha)/2,alpha,F(1)):
            for judgment in Region:
                a=judgment_risk(p,loss,judgment);b=judgment_risk(p,loss.scaled(F(7,3)),judgment)
                assert F(a['excess_exact'])>=0
                assert F(b['excess_exact'])==F(a['excess_exact'])*F(7,3)
                assert b['normalized_excess_exact']==a['normalized_excess_exact']
                assert a['decision']['region']==b['decision']['region']


@pytest.mark.parametrize('p',[None,True,'0.5',-1,F(11,10),float('nan'),float('inf')])
def test_bad_probabilities_rejected_by_original_d3(p):
    loss=LossSpec(1,1,F(1,4),F(1,4),TiePolicy.TERMINAL_NEG_FIRST)
    with pytest.raises((TypeError,ValueError)):judgment_risk(p,loss,Region.POS)


@pytest.mark.parametrize('costs',[(0,1,0,0),(1,-1,0,0),(1,1,-1,0),(1,1,0,2),(1,1,float('nan'),0)])
def test_original_invalid_cost_contract_unchanged(costs):
    with pytest.raises(ValueError):LossSpec(*costs,TiePolicy.TERMINAL_NEG_FIRST)


def test_exact_hand_calculated_excess_and_normalization():
    loss=LossSpec(1,1,F(1,4),F(1,4),TiePolicy.TERMINAL_NEG_FIRST)
    at_half={r:judgment_risk(F(1,2),loss,r) for r in Region}
    assert [at_half[r]['excess_exact'] for r in Region]==['1/4','0','1/4']
    assert at_half[Region.POS]['normalized_excess_exact']=='1/8'
    for b in (thresholds(loss).beta,thresholds(loss).alpha):
        zero=[r for r in Region if F(judgment_risk(b,loss,r)['excess_exact'])==0]
        assert len(zero)==2
    with pytest.raises(TypeError):judgment_risk(.5,loss,'POS')


def test_same_plan_different_prebound_bases_admitted_independently():
    pre,evidence,catalog,values=setup(F(1))
    plan=catalog.candidates[0];observer=evidence.selected.observer_id
    bases=(CandidateBasis('hit',plan,B.ASSUME_HIT,observer),CandidateBasis('miss',plan,B.ASSUME_MISS,observer),
           CandidateBasis('both',plan,B.KEEP_BOTH,observer))
    result=review(pre,evidence,bind_bases(pre,candidates=catalog.candidates,bases=bases),values)
    assert result.selected_plan==plan and result.selected_basis_id=='hit'
    r=result.audit['candidates'][0]['basis_reviews']
    assert {v['basis']['basis_id']:v['admitted'] for v in r}=={'both':False,'hit':True,'miss':False}
    # The same legal action can instead have a public-only basis; no action-level ban.
    public=CandidateBasis('public',plan,B.PUBLIC_ONLY)
    changed=bind_bases(pre,candidates=catalog.candidates,bases=(public,))
    assert review(pre,evidence,changed,values).selected_plan==plan
    assert review(pre,process_cognition(pre,panel=panel(pre,F(0))),changed,values).selected_plan==plan


def test_three_regions_change_bases_without_hard_coding_actions():
    pre,_,catalog,values=setup()
    observer=P[6];plans=(SpeechPlan(A.OPPOSE,observer),SpeechPlan(A.SELF_DEFEND),SpeechPlan(A.NO_COMMITMENT))
    bases=tuple(CandidateBasis(name,plan,kind,observer) for name,plan,kind in zip(
        ('positive','boundary','negative'),plans,(B.ASSUME_HIT,B.KEEP_BOTH,B.ASSUME_MISS)))
    bound=bind_bases(pre,candidates=catalog.candidates,bases=bases)
    for p,plan in zip((F(1),F(1,2),F(0)),plans):
        evidence=process_cognition(pre,panel=panel(pre,p))
        assert review(pre,evidence,bound,values).selected_plan==plan


@pytest.mark.parametrize('mode',list(G))
def test_public_only_candidates_are_cognition_independent(mode):
    pre,evidence,catalog,values=setup(F(1),kind=B.PUBLIC_ONLY)
    first=review(pre,evidence,catalog,values,mode)
    other=process_cognition(pre,panel=panel(pre,F(0)))
    second=review(pre,other,catalog,values,mode)
    assert first.selected_plan==second.selected_plan==catalog.candidates[0]
    assert first.audit['assessable_candidate_count']==len(catalog.candidates)


def test_gate_shrinks_assessable_set_and_no_implicit_reward_or_default_action():
    pre,evidence,catalog,values=setup(F(0),kind=B.ASSUME_HIT)
    observed=review(pre,evidence,catalog,values,G.RECORD_ONLY)
    gated=review(pre,evidence,catalog,values,G.THREE_WAY)
    assert observed.selected_plan and observed.audit['assessable_candidate_count']==len(catalog.candidates)
    assert gated.status=='NO_ADMISSIBLE_BASIS' and gated.selected_plan is None
    assert gated.audit['legal_candidate_count']==len(catalog.candidates)
    public=bind_bases(pre,candidates=catalog.candidates,bases=(CandidateBasis('explicit',catalog.candidates[0],B.PUBLIC_ONLY),))
    unavailable=review(pre,evidence,public,None)
    assert unavailable.status=='ACTION_VALUE_UNAVAILABLE' and unavailable.selected_plan is None
    assert unavailable.audit['value_source'] is None
    empty=bind_bases(pre,candidates=catalog.candidates,bases=())
    assert review(pre,evidence,empty,values).status=='NO_ADMISSIBLE_BASIS'


def test_exact_tie_distinguishes_zero_excess_from_selected_tie_policy():
    loss=LossSpec(1,1,F(1,4),F(1,4),TiePolicy.TERMINAL_NEG_FIRST)
    pre,evidence,catalog,values=setup(F(1,4),loss=loss,kind=B.KEEP_BOTH)
    result=review(pre,evidence,catalog,values)
    assert result.status=='NO_ADMISSIBLE_BASIS'
    basis=result.audit['candidates'][0]['basis_reviews'][0]
    assert basis['cognitive_risk']['excess_exact']=='0'
    assert basis['reason']=='ZERO_EXCESS_ALTERNATIVE_REJECTED_BY_EXPLICIT_TIE'
    defer=replace(loss,tie_policy=TiePolicy.DEFER_FIRST)
    evidence=process_cognition(pre,panel=panel(pre,F(1,4),loss=defer))
    assert review(pre,evidence,catalog,values).selected_plan is not None


def test_binary_ablation_never_silently_uses_bnd_and_retains_full_risk_reference():
    pre,evidence,catalog,values=setup(F(1,2),kind=B.KEEP_BOTH)
    assert review(pre,evidence,catalog,values,G.THREE_WAY).selected_plan is not None
    result=review(pre,evidence,catalog,values,G.FORCED_BINARY)
    assert result.status=='NO_ADMISSIBLE_BASIS'
    assert all(b['reason']=='BINARY_MECHANISM_EXCLUDES_DEFER_BASIS' for r in result.audit['candidates'] for b in r['basis_reviews'])
    observer=evidence.selected.observer_id
    miss=bind_bases(pre,candidates=catalog.candidates,bases=(CandidateBasis('negative',catalog.candidates[0],B.ASSUME_MISS,observer),))
    r=review(pre,evidence,miss,values,G.FORCED_BINARY)
    assert r.selected_plan==catalog.candidates[0]
    assert F(r.audit['candidates'][0]['basis_reviews'][0]['cognitive_risk']['excess_exact'])>0


@pytest.mark.parametrize('tie',list(TiePolicy))
def test_all_45_cost_scenarios_and_exact_boundaries_match_continuous_argmin(tie):
    pre,_,catalog,values=setup()
    observer=P[6];plan=catalog.candidates[0]
    bound=bind_bases(pre,candidates=catalog.candidates,bases=tuple(
        CandidateBasis(kind.name,plan,kind,observer) for kind in (B.ASSUME_HIT,B.ASSUME_MISS,B.KEEP_BOTH)))
    for loss in candidate_losses(tie_policy=tie):
        th=thresholds(loss)
        for p in (F(0),th.beta,(th.beta+th.alpha)/2,th.alpha,F(1)):
            evidence=process_cognition(pre,panel=panel(pre,p,loss=loss))
            a=review(pre,evidence,bound,values,G.THREE_WAY)
            b=review(pre,evidence,bound,values,G.CONTINUOUS_EQUIVALENCE)
            assert a.selected_plan==b.selected_plan and a.selected_basis_id==b.selected_basis_id
            assert a.audit['candidates']==b.audit['candidates']


@pytest.mark.parametrize('phase',['discussion','pk_discussion'])
@pytest.mark.parametrize('dead',[(),(P[1],),(P[5],)])
def test_legal_phase_single_double_wolf_and_dead_observer_mapping(phase,dead):
    pre=legal_population_pre(phase,dead);candidates=generate_candidates(pre)
    evidence=process_cognition(pre,panel=panel(pre))
    bases=tuple(CandidateBasis(f'x{j}',p,B.PUBLIC_ONLY) for j,p in enumerate(candidates))
    catalog=bind_bases(pre,candidates=candidates,bases=bases)
    values=artificial_action_values(pre,candidates=candidates,scores=tuple(F(0) for _ in candidates),fixture_id='ties')
    assert review(pre,evidence,catalog,values).selected_plan==candidates[0]


def test_public_refs_stale_pre_duplicate_reason_and_candidate_tamper_rejected():
    pre,evidence,catalog,values=setup();ann=pre.v1_annotations[-1]
    ref=PublicReference(ann.event_id,ann.annotation_digest)
    basis=CandidateBasis('public',catalog.candidates[0],B.PUBLIC_ONLY,public_references=(ref,))
    bound=bind_bases(pre,candidates=catalog.candidates,bases=(basis,))
    assert review(pre,evidence,bound,values).selected_plan is not None
    for b in (replace(basis,public_references=(PublicReference(ann.event_id,'0'*64),)),
              replace(basis,public_references=(PublicReference('future-event','0'*64),))):
        with pytest.raises(ValueError,match='public reference'):bind_bases(pre,candidates=catalog.candidates,bases=(b,))
    with pytest.raises(ValueError,match='duplicate'):bind_bases(pre,candidates=catalog.candidates,bases=(basis,basis))
    with pytest.raises(ValueError,match='candidate'):bind_bases(pre,candidates=catalog.candidates[::-1],bases=(basis,))
    with pytest.raises(ValueError,match='another PRE'):review(pre,evidence,replace(bound,prefix_digest='0'*64),values)
    with pytest.raises(ValueError,match='value PRE'):review(pre,evidence,bound,replace(values,prefix_digest='0'*64))
    with pytest.raises(TypeError):review(pre,evidence,bound,object())
    with pytest.raises(TypeError):review(pre,evidence,bound,values,mode='three_way')


@pytest.mark.parametrize('score',[True,float('nan'),float('inf'),'1'])
def test_malformed_artificial_value_source_not_used_as_risk_or_fallback(score):
    pre,_,catalog,_=setup()
    with pytest.raises(ValueError):artificial_action_values(pre,candidates=catalog.candidates,
        scores=tuple(score for _ in catalog.candidates),fixture_id='bad')


def test_multiple_reasons_do_not_boost_action_value_and_permutation_is_reproducible():
    pre,evidence,catalog,values=setup(F(1),kind=B.PUBLIC_ONLY)
    extra=CandidateBasis('extra',catalog.candidates[-1],B.PUBLIC_ONLY)
    a=bind_bases(pre,candidates=catalog.candidates,bases=catalog.bases+(extra,))
    b=bind_bases(pre,candidates=catalog.candidates,bases=tuple(reversed(a.bases)))
    assert a.digest==b.digest
    assert review(pre,evidence,a,values).selected_plan==review(pre,evidence,b,values).selected_plan==catalog.candidates[0]


def test_missing_cognitive_row_and_illegal_wolf_observer_never_become_bnd():
    pre,evidence,catalog,values=setup()
    selected=next(r.cognition for r in evidence.rows if r.cognition.context.observer_id==evidence.selected.observer_id)
    subset=process_cognition(pre,panel=(selected,))
    other=next(x for x in evidence.eligible_observers if x!=selected.context.observer_id)
    for observer,reason in ((other,'COGNITIVE_OBSERVER_ROW_UNAVAILABLE_NO_FALLBACK'),(P[1],'BASIS_OBSERVER_OUTSIDE_LIVING_NONWOLF_DOMAIN')):
        bound=bind_bases(pre,candidates=catalog.candidates,bases=(CandidateBasis('x',catalog.candidates[0],B.ASSUME_HIT,observer),))
        result=review(pre,subset,bound,values)
        assert result.status=='NO_ADMISSIBLE_BASIS'
        assert result.audit['candidates'][0]['basis_reviews'][0]['reason']==reason


def test_labels_cannot_enter_and_pure_calls_preserve_rng_files_and_candidate_space(monkeypatch):
    pre,evidence,catalog,values=setup(kind=B.PUBLIC_ONLY)
    a=random.getstate();b=np.random.get_state();before=catalog.digest
    def no_io(*args,**kwargs):raise AssertionError('pure appraisal must not access files')
    outputs=[]
    with monkeypatch.context() as m:
        m.setattr('builtins.open',no_io);m.setattr(Path,'open',no_io)
        for hidden in ({'Y':0,'report':()}, {'Y':1,'report':('player1',)}):
            outputs.append(review(pre,evidence,catalog,values).audit)
            with pytest.raises(TypeError):appraise_candidates(pre,evidence,catalog,mode=G.THREE_WAY,values=values,supervision=hidden)
    assert outputs[0]==outputs[1] and catalog.digest==before
    assert generate_candidates(pre)==catalog.candidates and random.getstate()==a
    c=np.random.get_state();assert b[0]==c[0] and np.array_equal(b[1],c[1]) and b[2:]==c[2:]


def test_action_selection_and_admissibility_invariant_under_positive_cost_scaling():
    loss=LossSpec(1,2,F(1,2),F(1,4),TiePolicy.TERMINAL_NEG_FIRST)
    pre,evidence,catalog,values=setup(F(3,5),loss=loss,kind=B.ASSUME_HIT)
    for mode in G:
        a=review(pre,evidence,catalog,values,mode)
        scaled=process_cognition(pre,panel=panel(pre,F(3,5),loss=loss.scaled(F(13,7))))
        b=review(pre,scaled,catalog,values,mode)
        assert a.selected_plan==b.selected_plan and a.status==b.status
        assert [r['assessable'] for r in a.audit['candidates']]==[r['assessable'] for r in b.audit['candidates']]


def test_independent_action_preferences_change_selection_without_changing_risk():
    pre,evidence,catalog,values=setup(F(1),kind=B.ASSUME_HIT)
    reverse=artificial_action_values(pre,candidates=catalog.candidates,
        scores=tuple(reversed(values.scores)),fixture_id='reversed-artificial-preference')
    a=review(pre,evidence,catalog,values);b=review(pre,evidence,catalog,reverse)
    assert a.selected_plan==catalog.candidates[0] and b.selected_plan==catalog.candidates[-1]
    assert [[x['cognitive_risk'] for x in row['basis_reviews']] for row in a.audit['candidates']]==[
        [x['cognitive_risk'] for x in row['basis_reviews']] for row in b.audit['candidates']]


@pytest.mark.parametrize('loss',[LossSpec(1,1,1,1,TiePolicy.TERMINAL_NEG_FIRST),
    LossSpec(1,1,F(1,2),F(1,2),TiePolicy.DEFER_FIRST),
    LossSpec(1,1,0,0,TiePolicy.DEFER_FIRST)])
def test_degenerate_costs_keep_original_argmin_no_threshold_shortcut(loss):
    pre,_,catalog,values=setup()
    observer=P[6]
    bound=bind_bases(pre,candidates=catalog.candidates,bases=tuple(
        CandidateBasis(k.name,catalog.candidates[0],k,observer) for k in (B.ASSUME_HIT,B.ASSUME_MISS,B.KEEP_BOTH)))
    for p in (F(0),F(1,2),F(1)):
        evidence=process_cognition(pre,panel=panel(pre,p,loss=loss))
        a=review(pre,evidence,bound,values,G.THREE_WAY)
        b=review(pre,evidence,bound,values,G.CONTINUOUS_EQUIVALENCE)
        assert a.selected_basis_id==b.selected_basis_id


def test_predeclaration_catalog_does_not_depend_on_probability_or_final_selection():
    from scripts.compare_observer_dtrs_appraisal import catalogs_for_pre
    pre,evidence,catalog,values=setup()
    from werewolf.observer_dtrs.speech_evidence import select_observer
    selected=select_observer(pre,P[:2])
    a=catalogs_for_pre(pre,catalog.candidates,selected)
    other=process_cognition(pre,panel=panel(pre,F(1)))
    review(pre,other,a['cyclic_cognitive'],values)
    b=catalogs_for_pre(pre,catalog.candidates,selected)
    assert {n:c.digest for n,c in a.items()}=={n:c.digest for n,c in b.items()}


def test_complete_controlled_comparison_has_budget_pairing_and_equivalence():
    from scripts.compare_observer_dtrs_appraisal import compare_scene,scenarios,synthetic_model,summarize,CATALOGS,MODES,COSTS
    protocol,model=synthetic_model()
    rows=[r for scene in scenarios() for r in compare_scene(scene,protocol,model)]
    result=summarize(rows)
    assert result['rows']==30*len(CATALOGS)*len(MODES)*len(COSTS)==1200
    for c in result['contrasts']:
        if c['left'][1]==G.THREE_WAY.value and c['right'][1]==G.CONTINUOUS_EQUIVALENCE.value:
            assert c['selected_plan_changes']==c['selected_basis_changes']==c['assessable_count_changes']==0
        if c['catalog'] in ('public_only','same_action_multiple_bases') and 'S1a' not in (c['left'][1],c['right'][1]):
            assert c['selected_plan_changes']==0
        if c['catalog']=='cyclic_cognitive' and 'S1a' not in (c['left'][1],c['right'][1]):
            assert c['selected_action_type_changes']==0
            assert c['target_only_changes']==c['selected_plan_changes']
    with pytest.raises(ValueError,match='cell'):summarize(rows[:-1])
    with pytest.raises(ValueError,match='cell'):summarize(rows+rows[:1])
    tamper=[dict(r) for r in rows];tamper[0]['pairing']={**tamper[0]['pairing'],'q_digest':'wrong'}
    with pytest.raises(ValueError,match='unpaired'):summarize(tamper)


def test_comparison_really_binds_before_one_Q_and_one_D2_call(monkeypatch):
    from scripts import compare_observer_dtrs_appraisal as c
    protocol,model=c.synthetic_model();events=[];counts={'Q':0,'D2':0,'appraisal':0}
    original_source=c.source;original_estimate=c.estimate_event;original_appraise=c.appraise_candidates
    class SourceSpy:
        def __init__(self,wrapped):self.wrapped=wrapped
        def probabilities(self,pre):
            events.append('Q');counts['Q']+=1
            return self.wrapped.probabilities(pre)
    def source_spy(*args,**kwargs):return SourceSpy(original_source(*args,**kwargs))
    def estimate_spy(*args,**kwargs):
        events.append('D2');counts['D2']+=1
        return original_estimate(*args,**kwargs)
    def appraise_spy(*args,**kwargs):
        events.append('appraisal');counts['appraisal']+=1
        return original_appraise(*args,**kwargs)
    monkeypatch.setattr(c,'source',source_spy);monkeypatch.setattr(c,'estimate_event',estimate_spy)
    monkeypatch.setattr(c,'appraise_candidates',appraise_spy)
    rows=c.compare_scene(c.scenarios()[0],protocol,model,capture_bindings=lambda _:events.append('bound'))
    assert events[:3]==['bound','Q','D2']
    assert counts=={'Q':1,'D2':1,'appraisal':32} and len(rows)==40
    assert len({r['pairing']['action_value_digest'] for r in rows})==1
