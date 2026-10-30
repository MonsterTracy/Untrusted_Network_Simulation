"""Controlled CPU contracts only; none of these objects are formal artifacts.

Uses already reviewed synthetic public PRE/Q helpers. No targets, real models,
environment execution, action values or historical private payloads are opened.
"""
from copy import deepcopy
from dataclasses import asdict, fields, replace
from fractions import Fraction
import builtins
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest

from werewolf.observer_dtrs import cognition_gate_research as study
from scripts.speech_planning import PlanningAction, SpeechPlan, generate_candidates
from tests.observer_dtrs.test_speech_evidence import legal_population_pre
from tests.observer_dtrs.test_speech_runtime import source, witness
from tests.tom.test_counterfactual_consumer import P
from werewolf.artifact_io import canonical_json_bytes
from werewolf.canonical_collection import (
    V1AnnotationStatus, V1SpeechAction, construct_v1_speech_annotation,
    freeze_public_event_history,
)
from werewolf.observer_dtrs.candidate_appraisal import (
    BasisKind, CandidateBasis, GateMode, PublicReference, appraise_candidates,
    bind_bases,
)
from werewolf.observer_dtrs.context import contexts_for_pre
from werewolf.observer_dtrs.dataset import LegalDatasetRow, ObserverDatasetView, sample_id
from werewolf.observer_dtrs.decision import Region, TiePolicy, thresholds
from werewolf.observer_dtrs.decision_offline import ProbabilityInput
from werewolf.observer_dtrs.features import observer_features
from werewolf.observer_dtrs.probabilities import ProbabilityRow
from werewolf.observer_dtrs.sealed_d2_offline import PredictionBatch, SampleOrigin
from werewolf.observer_dtrs.speech_evidence import ObserverCognition, process_cognition

POLICIES=(TiePolicy.TERMINAL_NEG_FIRST,TiePolicy.DEFER_FIRST)
DATASET='a'*64
MODEL='d'*64
BLOCK='b'*64


def controlled_inputs(pre=None, p=.5):
    pre=witness().pre() if pre is None else pre
    q=source(pre).matrix
    legal=tuple(LegalDatasetRow(sample_id(c),c,q.row(c.observer_id),observer_features(c,q))
                for c in contexts_for_pre(pre,P[:2]))
    view=ObserverDatasetView(SimpleNamespace(manifest_digest=DATASET,manifest={
        'games':[{'game_id':pre.game_id,'fold_index':0}]}),(pre.game_id,),legal)
    predictions=tuple(ProbabilityInput(r.sample_id,r.context.game_id,p) for r in legal)
    origins=tuple(SampleOrigin(r.sample_id,r.context.game_id,r.context.phase,
        r.context.wolf_count,0,1,BLOCK) for r in legal)
    return pre,view,batch_for(predictions,origins)


def batch_for(rows,origins,*,run_change=None,identity_change=None):
    rows,origins=tuple(rows),tuple(origins)
    pd=study.digest(sorted([r.sample_id,r.game_id,float(r.probability)] for r in rows))
    od=study.digest(sorted((asdict(o) for o in origins),key=lambda o:o['sample_id']))
    run={'scope':'CONTROLLED_UNIT_FIXTURE_NOT_FORMAL_DATA',
         'dataset_digest':DATASET,'prediction_digests':{'game_equal/OBS-EQ1':pd},
         'sample_origins_digest':od,'outer_references':{'0-1':{'game_equal':{'predictions':MODEL}}},
         'block_digests':{'0-1':BLOCK}}
    if run_change:run_change(run)
    identity={'run':run,'run_identity_digest':study.digest(run),'model_id':'OBS-EQ1',
        'weighting':'game_equal','predictions_digest':pd,'origins_digest':od,
        'outer_prediction_digest_by_block':{'0-1':MODEL}}
    if identity_change:identity_change(identity)
    return PredictionBatch(rows,origins,canonical_json_bytes(identity))


def historical(pre=None,p=.5):
    pre,view,batch=controlled_inputs(pre,p)
    assert batch.identity['run']['scope']=='CONTROLLED_UNIT_FIXTURE_NOT_FORMAL_DATA'
    rows=study.bind_inputs(view,batch)[(pre.game_id,pre.boundary_id)]
    return pre,rows


@pytest.mark.parametrize('phase',['discussion','pk_discussion'])
@pytest.mark.parametrize('dead',[(),(P[1],),(P[5],)])
def test_complete_domain_and_dead_target_probability_mass(phase,dead):
    pre,rows=historical(legal_population_pre(phase,dead))
    selection=study.validate_pre_inputs(pre,rows)
    assert selection.observer_id in {r.legal.context.observer_id for r in rows}
    assert tuple(r.legal.context for r in rows)==contexts_for_pre(pre,P[:2])
    assert tuple(r.legal.context.observer_id for r in rows)==tuple(
        x for x in pre.alive_observer_ids if x not in P[:2])
    assert all(r.legal.features.phase==phase for r in rows)
    if P[1] in dead:
        for r in rows:
            assert r.legal.probability.values[1]==.2
            assert r.legal.features.s==.2 and r.legal.features.m==0
            assert r.legal.features.r==.2 and r.legal.features.dW is None
    else:assert all(r.legal.features.nW==2 for r in rows)


@pytest.mark.parametrize('field,value',[
    ('model_id','OBS-LR-r'),('weighting','record_equal'),
    ('run_identity_digest','0'*64),('predictions_digest','0'*64),
    ('origins_digest','0'*64),('outer_prediction_digest_by_block',{'0-1':'0'*64}),
])
def test_model_weight_and_digest_identity_rejected(field,value):
    _,view,batch=controlled_inputs()
    changed=batch_for(batch.rows,batch.origins,identity_change=lambda x:x.update({field:value}))
    with pytest.raises(ValueError):study.bind_inputs(view,changed)


@pytest.mark.parametrize('field,value',[
    ('dataset_digest','f'*64),('prediction_digests',{'game_equal/OBS-EQ1':'f'*64}),
    ('sample_origins_digest','f'*64),('block_digests',{'0-1':'f'*64}),
])
def test_rehashed_but_wrong_run_source_is_rejected(field,value):
    _,view,batch=controlled_inputs()
    changed=batch_for(batch.rows,batch.origins,run_change=lambda x:x.update({field:value}))
    with pytest.raises(ValueError):study.bind_inputs(view,changed)


@pytest.mark.parametrize('kind',['missing_p','duplicate_p','missing_origin','duplicate_origin','missing_legal','duplicate_legal'])
def test_sample_id_join_rejects_missing_and_duplicate_records(kind):
    _,view,b=controlled_inputs();rows,origins=b.rows,b.origins
    if kind=='missing_p':rows=rows[:-1]
    elif kind=='duplicate_p':rows=rows+rows[:1]
    elif kind=='missing_origin':origins=origins[:-1]
    elif kind=='duplicate_origin':origins=origins+origins[:1]
    elif kind=='missing_legal':view=replace(view,rows=view.rows[:-1])
    else:view=replace(view,rows=view.rows+view.rows[:1])
    with pytest.raises(ValueError):study.bind_inputs(view,batch_for(rows,origins))


@pytest.mark.parametrize('kind',['wrong_sample','wrong_game','wrong_phase','wrong_nW','wrong_tom_fold','wrong_outer_fold','wrong_block'])
def test_rehashed_row_origin_cannot_cross_game_phase_or_fold(kind):
    _,view,b=controlled_inputs();origins=list(b.origins);predictions=list(b.rows)
    if kind=='wrong_sample':predictions[0]=replace(predictions[0],sample_id='foreign')
    elif kind=='wrong_game':predictions[0]=replace(predictions[0],game_id='foreign')
    else:
        changes={'wrong_phase':{'phase':'pk_discussion'},'wrong_nW':{'nW':1},
            'wrong_tom_fold':{'tom_fold':2},'wrong_outer_fold':{'outer_fold':2},
            'wrong_block':{'block_digest':'f'*64}}
        origins[0]=replace(origins[0],**changes[kind])
    with pytest.raises((ValueError,KeyError)):
        study.bind_inputs(view,batch_for(predictions,origins))


def test_alignment_is_by_sample_id_not_payload_row_order():
    pre,view,b=controlled_inputs()
    expected=study.bind_inputs(view,b)
    shuffled=batch_for(tuple(reversed(b.rows)),tuple(reversed(b.origins)))
    assert study.bind_inputs(view,shuffled)==expected
    rows=expected[(pre.game_id,pre.boundary_id)]
    assert all(r.prediction_reference==MODEL for r in rows)


@pytest.mark.parametrize('kind',['sample','Q_boundary','Q_prefix','features','dataset_fold'])
def test_legal_q_context_and_population_source_mismatch(kind):
    _,view,b=controlled_inputs();row=view.rows[0]
    if kind=='sample':row=replace(row,sample_id='f'*64)
    elif kind=='Q_boundary':row=replace(row,probability=replace(row.probability,boundary_id='other-pre'))
    elif kind=='Q_prefix':row=replace(row,probability=replace(row.probability,prefix_digest='0'*64))
    elif kind=='features':row=replace(row,features=replace(row.features,r=.123))
    else:
        view=replace(view,artifact=SimpleNamespace(manifest_digest=DATASET,manifest={
            'games':[{'game_id':row.context.game_id,'fold_index':4}]}))
    if kind!='dataset_fold':view=replace(view,rows=(row,*view.rows[1:]))
    with pytest.raises(ValueError):study.bind_inputs(view,b)


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-0.1,1.1,True])
def test_historical_observer_rejects_invalid_event_probability(bad):
    _,rows=historical()
    with pytest.raises((ValueError,TypeError)):replace(rows[0],probability=bad)


@pytest.mark.parametrize('raw',[True,False,'0.5'])
def test_rehashed_batch_must_validate_original_probability_scalar_before_float(raw):
    _,view,batch=controlled_inputs()
    # Corrupted typed fixture: a reader must not turn bool/text into legal p.
    bad=replace(batch.rows[0])
    object.__setattr__(bad,'probability',raw)
    forged=batch_for((bad,*batch.rows[1:]),batch.origins)
    with pytest.raises((ValueError,TypeError)):study.bind_inputs(view,forged)


@pytest.mark.parametrize('kind',['missing','duplicate','reorder','model','actor','phase','raw_row'])
def test_actual_pre_requires_complete_and_same_source_observer_population(kind):
    pre,rows=historical()
    if kind=='missing':rows=rows[:-1]
    elif kind=='duplicate':rows=rows+rows[:1]
    elif kind=='reorder':rows=rows[::-1]
    elif kind=='model':rows=(replace(rows[0],prediction_reference='e'*64),*rows[1:])
    elif kind=='raw_row':rows=({'Y':1},*rows[1:])
    else:
        context=replace(rows[0].legal.context,**({'acting_wolf':P[1]} if kind=='actor' else {'phase':'pk_discussion'}))
        legal=replace(rows[0].legal,context=context,sample_id=sample_id(context),
                      features=observer_features(context,rows[0].legal.probability))
        origin=replace(rows[0].origin,sample_id=legal.sample_id,phase=context.phase)
        rows=(study.HistoricalObserver(legal,rows[0].probability,origin,MODEL),*rows[1:])
    with pytest.raises((ValueError,TypeError)):study.validate_pre_inputs(pre,rows)


def test_public_rule_uses_actual_self_accusation_only_and_never_fills_no_commitment():
    pre,rows=historical();catalog,facts=study.public_reply_catalog(pre,rows)
    assert catalog.candidates==generate_candidates(pre)
    assert len(catalog.bases)==1
    basis=catalog.bases[0]
    assert basis.kind is BasisKind.PUBLIC_ONLY and basis.observer_id is None
    assert basis.plan==SpeechPlan(PlanningAction.SELF_DEFEND)
    assert basis.public_references==(PublicReference(facts[-1]['event_id'],facts[-1]['annotation_digest']),)
    assert facts[-1]['status']=='VERIFIED_PUBLIC_REPLY_CUE'
    pre,rows=historical(witness(target=P[1]).pre())
    catalog,facts=study.public_reply_catalog(pre,rows)
    assert not catalog.bases
    assert all(f['status']!='VERIFIED_PUBLIC_REPLY_CUE' for f in facts)
    assert catalog.candidates==generate_candidates(pre)


def test_conflicting_successful_v1_does_not_become_public_reply_certificate():
    w=witness();w.events.pop();old=w.annotations[-1]
    actions=old.actions+(V1SpeechAction(P[6],'point_as_non_werewolf',P[0]),)
    attempt=replace(old.attempts[0],raw_response=f'{P[6]} | point_as_werewolf | {P[0]}\n{P[6]} | point_as_non_werewolf | {P[0]}')
    w.annotations[-1]=construct_v1_speech_annotation(freeze_public_event_history(w.events),
        status=V1AnnotationStatus.OK,actions=actions,attempts=(attempt,))
    w.event('turn_start',speaker=P[0]);pre,rows=historical(w.pre())
    catalog,facts=study.public_reply_catalog(pre,rows)
    assert not catalog.bases and facts[-1]['status']=='CONFLICTING_V1_SELF_CLAIMS'


@pytest.mark.parametrize('scenario',range(45))
@pytest.mark.parametrize('policy',POLICIES)
def test_registered_costs_all_arms_keep_full_candidates_and_no_invented_values(scenario,policy):
    pre,rows=historical();output=study.audit_pre(pre,rows,scenario=scenario,policy=policy)
    for scope,arms in output.items():
        assert set(arms)=={a.value for a in GateMode}
        for audit in arms.values():
            assert audit['status']=='ACTION_VALUE_UNAVAILABLE'
            assert audit['selected_plan'] is None and audit['selected_basis_id'] is None
            assert audit['value_source'] is None and audit['value_source_digest'] is None
            assert audit['legal_candidate_count']==len(generate_candidates(pre))
            assert audit['assessable_candidate_count']==1
            assert len(audit['candidates'])==len(generate_candidates(pre))
            assert all(c['game_rule_legal'] and c['artificial_action_score_exact'] is None for c in audit['candidates'])
        assert arms[GateMode.THREE_WAY.value]['candidates']==arms[GateMode.CONTINUOUS_EQUIVALENCE.value]['candidates']
    assert output['full'][GateMode.THREE_WAY.value]['candidates']==output['selected'][GateMode.THREE_WAY.value]['candidates']


def test_no_qualified_basis_returns_explicit_no_admissible_status_for_every_legal_candidate():
    pre,rows=historical(legal_population_pre('discussion',()))
    result=study.audit_pre(pre,rows,scenario=22,policy=POLICIES[0])
    for arms in result.values():
        for audit in arms.values():
            assert audit['status']=='NO_ADMISSIBLE_BASIS'
            assert audit['selected_plan'] is None and audit['assessable_candidate_count']==0
            assert len(audit['candidates'])==len(generate_candidates(pre))
            assert all(c['exclusion']=='NO_DECLARED_BASIS' for c in audit['candidates'])


def test_rehashed_valid_reference_cannot_be_repurposed_as_unregistered_candidate_reason():
    pre,rows=historical();catalog,_=study.public_reply_catalog(pre,rows)
    wrong=replace(catalog.bases[0],plan=SpeechPlan(PlanningAction.NO_COMMITMENT))
    different=bind_bases(pre,candidates=catalog.candidates,bases=(wrong,))
    with pytest.raises(ValueError,match='catalog/source/semantics'):
        study.audit_pre(pre,rows,scenario=22,policy=POLICIES[0],catalog=different)
    bad=replace(catalog.bases[0],public_references=(PublicReference(catalog.bases[0].public_references[0].event_id,'0'*64),))
    with pytest.raises(ValueError,match='public reference'):
        bind_bases(pre,candidates=catalog.candidates,bases=(bad,))


def test_cognitive_basis_differences_remain_explicit_controlled_hypotheses_not_real_catalog():
    pre,rows=historical();loss=study.registered_loss(22,POLICIES[0]);th=thresholds(loss)
    p=(th.alpha+th.beta)/2;decision=study.profile(p,22,POLICIES[0])[0]
    assert decision.region is Region.BND
    panel=tuple(ObserverCognition(r.legal.context,r.legal.features,MODEL,'CONTROLLED',loss,decision) for r in rows)
    g=process_cognition(pre,panel=panel);observer=g.selected.observer_id
    bases=(CandidateBasis('controlled.hit',SpeechPlan(PlanningAction.SELF_DEFEND),BasisKind.ASSUME_HIT,observer),
           CandidateBasis('controlled.miss',SpeechPlan(PlanningAction.OPPOSE,observer),BasisKind.ASSUME_MISS,observer),
           CandidateBasis('controlled.defer',SpeechPlan(PlanningAction.NO_COMMITMENT),BasisKind.KEEP_BOTH,observer))
    catalog=bind_bases(pre,candidates=generate_candidates(pre),bases=bases)
    results={m:appraise_candidates(pre,g,catalog,mode=m,values=None) for m in GateMode}
    assert results[GateMode.RECORD_ONLY].audit['assessable_candidate_count']==3
    assert results[GateMode.THREE_WAY].audit['assessable_candidate_count']==1
    assert results[GateMode.THREE_WAY].audit['candidates']==results[GateMode.CONTINUOUS_EQUIVALENCE].audit['candidates']
    binary=results[GateMode.FORCED_BINARY].audit
    assert binary['assessable_candidate_count']==1
    defer=next(c for c in binary['candidates'] if c['plan']==SpeechPlan(PlanningAction.NO_COMMITMENT).public_payload())
    assert not defer['assessable'] and defer['basis_reviews'][0]['reason']=='BINARY_MECHANISM_EXCLUDES_DEFER_BASIS'
    assert all(r.selected_plan is None and r.audit['value_source'] is None for r in results.values())
    # The real-data diagnostic catalog never imports this arbitrary fixture map.
    assert all(b.kind is BasisKind.PUBLIC_ONLY for b in study.public_reply_catalog(pre,rows)[0].bases)


@pytest.mark.parametrize('scenario',[-1,45,False,True,'22',None])
def test_bad_cost_scenario_cannot_use_a_warm_registered_cache(scenario):
    for i in (0,1):study.registered_loss(i,POLICIES[0])
    with pytest.raises((ValueError,TypeError)):study.registered_loss(scenario,POLICIES[0])


@pytest.mark.parametrize('policy',[TiePolicy.TERMINAL_POS_FIRST,'unregistered',None])
def test_unregistered_tie_policy_is_rejected(policy):
    with pytest.raises((ValueError,TypeError)):study.registered_loss(22,policy)


@pytest.mark.parametrize('probability',[float('nan'),float('inf'),-.01,1.01,False,True])
def test_profile_rejects_invalid_probability_even_with_warm_endpoint_cache(probability):
    study.profile(0,22,POLICIES[0]);study.profile(1,22,POLICIES[0])
    with pytest.raises((ValueError,TypeError)):study.profile(probability,22,POLICIES[0])


def test_supervision_is_not_accepted_or_observed_and_calls_are_pure(monkeypatch):
    pre,view,batch=controlled_inputs();rows=study.bind_inputs(view,batch)[(pre.game_id,pre.boundary_id)]
    study.registered_loss(22,POLICIES[0]);study.profile(.5,22,POLICIES[0])
    before_py,before_np=random.getstate(),np.random.get_state()
    outputs=[]
    def forbidden(*a,**k):raise AssertionError('unexpected filesystem read/write')
    with monkeypatch.context() as m:
        m.setattr(builtins,'open',forbidden);m.setattr(Path,'open',forbidden)
        m.setattr(Path,'read_text',forbidden);m.setattr(Path,'read_bytes',forbidden)
        for private in ({'Y':0,'successful_empty':False},{'Y':1,'successful_empty':True}):
            # External truth is deliberately not passed into any diagnostic.
            outputs.append(study.audit_pre(pre,rows,scenario=22,policy=POLICIES[0]))
            with pytest.raises(TypeError):study.audit_pre(pre,rows,scenario=22,policy=POLICIES[0],supervision=private)
            with pytest.raises(TypeError):study.bind_inputs({'Y':private['Y']},batch)
            assert study.bind_inputs(view,batch)[(pre.game_id,pre.boundary_id)]==rows
    assert outputs[0]==outputs[1] and random.getstate()==before_py
    after_np=np.random.get_state()
    assert before_np[0]==after_np[0] and np.array_equal(before_np[1],after_np[1]) and before_np[2:]==after_np[2:]
    assert not {'Y','targets','supervision','sidecar','successful_empty'} & {f.name for f in fields(study.HistoricalObserver)}
    with pytest.raises(TypeError):study.HistoricalObserver(rows[0].legal,.5,rows[0].origin,MODEL,Y=1)
    with pytest.raises(TypeError):study.bind_inputs(view,{'rows':batch.rows,'Y':1})


def test_cognition_cli_requires_explicit_portable_input_paths():
    from scripts import analyze_observer_dtrs_cognition_gates as cli
    options=['--dataset','inputs/dataset','--d2-run','inputs/d2',
             '--public-pre-file','inputs/public-pres.jsonl','--output','outputs/new-run']
    args=cli.argument_parser().parse_args(options)
    assert args.dataset==Path('inputs/dataset') and args.d2_run==Path('inputs/d2')
    assert args.public_pre_file==Path('inputs/public-pres.jsonl') and args.output==Path('outputs/new-run')
    assert '/Users/' not in cli.argument_parser().format_help()
    for option in options[::2]:
        omitted=options.copy();i=omitted.index(option);del omitted[i:i+2]
        with pytest.raises(SystemExit) as error:cli.argument_parser().parse_args(omitted)
        assert error.value.code==2


def test_cognition_cli_help_loads_outside_checkout_without_result_python(monkeypatch,capsys):
    import runpy
    import sys
    root=Path(__file__).resolve().parents[2];script=root/'scripts/analyze_observer_dtrs_cognition_gates.py'
    original_open=Path.open
    def guarded_open(path,*args,**kwargs):
        if 'Untrusted_Network_Simulation-results' in path.parts:raise AssertionError('no historical files during CLI help')
        return original_open(path,*args,**kwargs)
    monkeypatch.chdir(root.parent)
    monkeypatch.setattr(sys,'path',list(sys.path));monkeypatch.setattr(sys,'argv',[str(script),'--help'])
    monkeypatch.setitem(sys.modules,'public_speech_day_audit',None);monkeypatch.setattr(Path,'open',guarded_open)
    with pytest.raises(SystemExit) as error:runpy.run_path(str(script),run_name='__main__')
    assert error.value.code==0 and '--public-pre-file' in capsys.readouterr().out
