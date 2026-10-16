"""Artificial engineering acceptance only. No formal D3 or real labels loaded."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pytest

from werewolf.artifact_io import ArtifactValidationError, canonical_json_bytes
from werewolf.observer_dtrs import d3_runner as runner
from werewolf.observer_dtrs.d3_protocol import D3Protocol, load_protocol, cost_edges
from werewolf.observer_dtrs.d3_inputs_offline import controlled_inputs
from werewolf.observer_dtrs.decision import LossSpec, TiePolicy, Region, decide
from werewolf.observer_dtrs.decision_offline import evaluate
from werewolf.observer_dtrs.validation_artifacts import digest
from tests.observer_dtrs.test_sealed_d2 import rewrite


@pytest.fixture(scope='module')
def inputs():
    return controlled_inputs()


@pytest.fixture(scope='module')
def full_run(tmp_path_factory,inputs):
    root=tmp_path_factory.mktemp('d3-complete')
    result=runner.execute_controlled(root,inputs=inputs)
    return root,result


def _protocol_with_status(status):
    record=load_protocol().record
    record['status']=status
    return D3Protocol(canonical_json_bytes(record))


@pytest.mark.parametrize('status',['READY_FOR_REVIEW_NOT_FROZEN','FROZEN_FOR_EXECUTION'])
def test_registered_costs_policies_and_adjacency(status):
    p=_protocol_with_status(status)
    assert p.record['status']==status
    assert len(cost_edges())==96 and len(set(cost_edges()))==96
    for s in range(45):
        neg=p.loss(s,TiePolicy.TERMINAL_NEG_FIRST)
        assert neg.c_fp+neg.c_fn==1
        assert neg.b0/neg.c_fp+neg.b1/neg.c_fn<1
        for q in (0,neg.c_fp/(neg.c_fp+neg.c_fn),1):
            pos=replace(neg,tie_policy=TiePolicy.TERMINAL_POS_FIRST)
            assert decide(q,neg).region==decide(q,pos).region
    if status=='FROZEN_FOR_EXECUTION':
        p.require_frozen(p.digest)
    else:
        with pytest.raises(ValueError,match='frozen'):
            p.require_frozen(p.digest)
    r=p.record;r['loss_scenarios'][0]['exact_costs']['b0']='1/3'
    with pytest.raises(ValueError,match='45 costs'):
        D3Protocol(canonical_json_bytes(r))


@pytest.mark.parametrize('model',['OBS-EQ0','OBS-EQ1','OBS-LR-r','OBS-EF'])
@pytest.mark.parametrize('w',['game_equal','record_equal'])
def test_manual_cases_match_existing_evaluator(inputs,model,w):
    p=load_protocol();policy=p.policies[0];s=22
    actual=runner._case(inputs,p,scenario=s,policy=policy,model=model,weighting=w)
    ref=evaluate(inputs.rows(model,w),{sid:r['y'] for sid,r in inputs.open_targets().items()},
        loss=p.loss(s,policy),game_sizes=inputs.identity['game_sizes'],weighting=w,
        binary_tie_policy=TiePolicy.TERMINAL_NEG_FIRST)
    point=actual['groups']['overall']
    for key in ('realized_risk','decided_coverage','POS_error_rate','NEG_error_rate','selective_error_rate',
                'false_positive_joint','false_negative_joint','predicted_minimum_risk'):
        assert point[key]==pytest.approx(ref[key]) if ref[key] is not None else point[key] is None
    for a in Region:
        for key in ('mass','proportion','event_rate','loss_contribution'):
            assert point['regions'][a.value][key]==pytest.approx(ref['regions'][a.value][key])
        assert point['fixed_action_risks'][a.value]==pytest.approx(ref['fixed_action_risks'][a.value])
    assert point['forced_binary_risk']==pytest.approx(ref['forced_binary']['realized_risk'])
    assert set(actual['by_game'])==set(inputs.identity['game_sizes'])
    assert sum(g['loss_sum'] for g in actual['by_game'].values())/sum(g['mass'] for g in actual['by_game'].values())==pytest.approx(point['realized_risk'])
    for r in actual['decisions']:
        assert 'y' not in r and 'successful_empty' not in r and 'support' not in r


def _changed(inputs,field,value):
    identity=inputs.identity;identity[field+'_digest']=digest(value)
    return replace(inputs,**{('_targets_payload' if field=='targets' else '_support_payload' if field=='support' else 'legal_payload'):canonical_json_bytes(value)},
                   identity_payload=canonical_json_bytes(identity)).validate()


def test_supervision_and_support_counterfactual_isolation(inputs):
    p=load_protocol();kw=dict(scenario=22,policy=p.policies[0],model='OBS-EQ1',weighting='game_equal')
    a=runner._case(inputs,p,**kw)
    targets=inputs.open_targets()
    sid=next(iter(targets))
    targets[sid].update(y=1-targets[sid]['y'],successful_empty=False)
    b=runner._case(_changed(inputs,'targets',targets),p,**kw)
    assert a['decisions']==b['decisions']
    assert a['groups']['overall']['realized_risk']!=b['groups']['overall']['realized_risk']
    support=inputs.open_support()
    for key,values in support.items():
        if not key.endswith('OBS-LR-r'):
            for r in values.values():r.update(games=1000,kish=500.)
    c=runner._case(_changed(inputs,'support',support),p,**kw)
    assert a['decisions']==c['decisions'] and a['groups']['overall']==c['groups']['overall']


def test_subgroup_weights_and_zero_classes(inputs):
    p=load_protocol();w='game_equal';s=22
    result=runner._case(inputs,p,scenario=s,policy=p.policies[0],model='OBS-EQ1',weighting=w)
    ids=[sid for sid,r in inputs.legal.items() if r['phase']=='pk_discussion' and r['nW']==1]
    ref=evaluate(inputs.rows('OBS-EQ1',w),{sid:r['y'] for sid,r in inputs.open_targets().items()},
        loss=p.loss(s,p.policies[0]),game_sizes=inputs.identity['game_sizes'],weighting=w,
        binary_tie_policy=TiePolicy.TERMINAL_NEG_FIRST,subgroup_ids=ids)
    assert result['groups']['phase_nW/pk_discussion/1']['mass']==pytest.approx(ref['mass'])
    assert result['groups']['phase_nW/pk_discussion/1']['realized_risk']==pytest.approx(ref['realized_risk'])
    assert result['groups']['support/NA']['realized_risk'] is None
    targets={sid:{'y':1,'successful_empty':False} for sid in inputs.legal}
    one=runner._case(_changed(inputs,'targets',targets),p,scenario=s,policy=p.policies[0],model='OBS-EQ1',weighting=w)
    assert one['groups']['overall']['false_positive_rate_given_Y0'] is None
    assert len(one['by_game'])==len(inputs.identity['game_sizes'])


def test_bootstrap_exact_pairing_determinism_and_ratios(inputs):
    order=inputs.identity['game_order'];draws=np.frombuffer(inputs.bootstrap_payload,dtype='<i8').reshape(2000,-1)
    a={g:{'loss_sum':i/20,'mass':1+(i%3)} for i,g in enumerate(order)}
    b={g:{'loss_sum':a[g]['loss_sum']+.1*a[g]['mass'],'mass':a[g]['mass']} for g in order}
    result=runner.bootstrap_risks(a,inputs=inputs,other=b)
    assert result['interval']==pytest.approx([.1,.1])
    matrix=np.array([[a[g]['loss_sum'],a[g]['mass']] for g in order]);v=matrix[draws].sum(axis=1)
    assert runner.bootstrap_risks(a,inputs=inputs)['interval']==pytest.approx(np.quantile(v[:,0]/v[:,1],[.025,.975],method='linear'))
    assert runner.bootstrap_risks(a,inputs=inputs)==runner.bootstrap_risks(a,inputs=inputs)
    bad=deepcopy(b);bad[order[0]]['mass']+=1
    with pytest.raises(ValueError,match='denominators'):runner.bootstrap_risks(a,inputs=inputs,other=bad)
    zero={g:{'loss_sum':0.,'mass':0.} for g in order}
    assert runner.bootstrap_risks(zero,inputs=inputs)['interval'] is None


@pytest.mark.parametrize('kind',['missing','duplicate','model','weight','stratum','bootstrap','target','support'])
def test_rehashed_input_corruption_rejected(inputs,kind):
    legal=inputs.legal;sid=next(iter(legal))
    if kind=='missing':
        legal.pop(sid)
    elif kind=='duplicate':
        legal['duplicate']=legal[sid]
    elif kind=='model':
        legal[sid]['p']['game_equal'].pop('OBS-EF')
    elif kind=='weight':
        legal[sid]['p']['old_weight']=legal[sid]['p'].pop('game_equal')
    elif kind=='stratum':
        legal[sid]['tom_fold']=4
    elif kind=='bootstrap':
        raw=bytearray(inputs.bootstrap_payload);raw[:8]=int(3).to_bytes(8,'little')
        identity=inputs.identity;identity['bootstrap_sha256']=__import__('hashlib').sha256(raw).hexdigest()
        with pytest.raises(ValueError):replace(inputs,bootstrap_payload=bytes(raw),identity_payload=canonical_json_bytes(identity)).validate()
        return
    elif kind=='target':
        t=inputs.open_targets();t[sid]['y']=True
        with pytest.raises(ValueError):_changed(inputs,'targets',t)
        return
    else:
        s=inputs.open_support();s['game_equal/OBS-LR-r'][sid]['games']=10
        with pytest.raises(ValueError):_changed(inputs,'support',s)
        return
    with pytest.raises(ValueError):_changed(inputs,'legal',legal)


def test_full_controlled_grid_is_complete_and_auditable(full_run):
    root,result=full_run
    assert result['scope']=='controlled_fixture' and len(result['cases'])==720
    assert len(result['comparisons'])==180 and len(result['cost_edges'])==192
    assert runner.inspect_result(root)['status']=='D3_ALL_CASES_COMPLETE'
    assert len(list((root/'cases').glob('*/*/*/*/manifest.json')))==720
    assert (root/'bootstrap/indices.int64le').exists()
    case=json.loads((root/'cases/TERMINAL_NEG_FIRST/C22/game_equal/OBS-EQ1/record.json').read_bytes())
    other=json.loads((root/'cases/DEFER_FIRST/C22/game_equal/OBS-EQ1/record.json').read_bytes())
    assert case['loss']['exact_costs']==other['loss']['exact_costs']
    assert case['loss_digest']!=other['loss_digest']
    assert case['groups']['overall']['probability_stability'] is not None
    assert other['groups']['overall']['probability_stability'] is None


def test_interrupt_resume_and_idempotence(tmp_path,inputs):
    root=tmp_path/'resume';count=0
    def stop(stage,path):
        nonlocal count
        count+=1
        if count==2:raise KeyboardInterrupt('controlled interruption')
    with pytest.raises(KeyboardInterrupt):runner.execute_controlled(root,inputs=inputs,hook=stop)
    assert not (root/'complete').exists() and (root/'failures/000000/manifest.json').exists()
    first=root/'cases/TERMINAL_NEG_FIRST/C00/game_equal/OBS-EQ0/manifest.json';raw=first.read_bytes()
    result=runner.execute_controlled(root,inputs=inputs)
    assert first.read_bytes()==raw and len(result['cases'])==720
    # A second replay reaches the same complete seal; no additional completion is fabricated.
    assert runner.execute_controlled(root,inputs=inputs)['manifest_digest']==result['manifest_digest']


@pytest.mark.parametrize('mode',['missing','truncated','rehash','wrong_identity','renamed_complete'])
def test_completed_corruption_stops_without_repair(full_run,tmp_path,inputs,mode):
    src,_=full_run;root=tmp_path/'bad';shutil.copytree(src,root)
    case=root/'cases/TERMINAL_NEG_FIRST/C00/game_equal/OBS-EQ0'
    if mode=='missing':(case/'record.json').unlink()
    elif mode=='truncated':(case/'record.json').write_bytes(b'broken')
    elif mode=='rehash':rewrite(case,transform=lambda r:r['groups']['overall'].__setitem__('realized_risk',.123))
    elif mode=='wrong_identity':rewrite(case,transform_identity=lambda i:i.__setitem__('model','OBS-EF'))
    else:
        rewrite(root/'complete',transform=lambda r:r['cases'].__setitem__('invalid',r['cases'].pop(next(iter(r['cases'])))))
    before={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file() and 'failures' not in p.parts}
    with pytest.raises((ValueError,ArtifactValidationError)):
        if mode=='renamed_complete':runner.inspect_result(root)
        else:runner.execute_controlled(root,inputs=inputs)
    assert all((root/p).read_bytes()==v for p,v in before.items())
    if mode=='missing':assert not (case/'record.json').exists()


def test_formal_guard_precedes_any_real_input_access_or_output(tmp_path,monkeypatch):
    seen=[]
    def guard(revision):
        seen.append(revision);raise ValueError('uncommitted production rejected')
    def forbidden(*a,**kw):raise AssertionError('guard must stop before input reading')
    monkeypatch.setattr(runner,'checked_source',guard);monkeypatch.setattr(runner,'prepare_formal',forbidden)
    with pytest.raises(ValueError,match='uncommitted'):
        runner.execute_formal(tmp_path/'no-output',run_path=tmp_path/'run',dataset_path=tmp_path/'data',
            protocol=load_protocol(),expected_protocol_digest=load_protocol().digest,expected_revision='a'*40,artifact_root=tmp_path)
    assert seen==['a'*40] and not (tmp_path/'no-output').exists()


def test_unfrozen_protocol_cannot_proceed_even_after_source_guard(tmp_path,monkeypatch):
    p=_protocol_with_status('READY_FOR_REVIEW_NOT_FROZEN')
    monkeypatch.setattr(runner,'checked_source',lambda revision:{'controlled_test_only':True})
    def forbidden(*args,**kwargs):
        raise AssertionError('unfrozen protocol must stop before real input access')
    monkeypatch.setattr(runner,'prepare_formal',forbidden)
    with pytest.raises(ValueError,match='frozen'):
        runner.execute_formal(tmp_path/'no-output',run_path=tmp_path/'run',dataset_path=tmp_path/'data',
            protocol=p,expected_protocol_digest=p.digest,expected_revision='a'*40,artifact_root=tmp_path)
    assert not (tmp_path/'no-output').exists()


def test_controlled_mode_refuses_formal_claims_and_cli_formal_paths(tmp_path,inputs):
    identity=inputs.identity;identity['scope']='formal_D3_posthoc_development'
    with pytest.raises(ValueError,match='formal material'):
        runner.execute_controlled(tmp_path/'blocked',inputs=replace(inputs,identity_payload=canonical_json_bytes(identity)))
    from scripts.run_observer_dtrs_decisions import main
    with pytest.raises(SystemExit):main(['controlled','--destination',str(tmp_path/'bad'),'--run','formal-run'])
    assert not (tmp_path/'blocked').exists() and not (tmp_path/'bad').exists()


def test_exact_policy_ties_and_degenerate_diagnostic_scope(full_run):
    root,result=full_run
    tie=json.loads((root/'tie_sensitivity/C22/game_equal/OBS-EQ1/record.json').read_bytes())
    assert tie['changed_rows']>0
    diag=json.loads((root/'diagnostics/record.json').read_bytes())
    assert diag['scope']=='label_free_degenerate_diagnostic_not_formal_cost_grid'
    assert {c['loss']['tie_policy'] for c in diag['cases']}=={p.value for p in TiePolicy}
    assert len(result['tie_sensitivity'])==360
    c=json.loads((root/'cases/TERMINAL_NEG_FIRST/C22/game_equal/OBS-EQ1/record.json').read_bytes())
    point=c['groups']['overall'];costs=c['loss']['costs']
    assert 0<=point['realized_risk']<=max(costs.values())
    assert sum(g['loss_contribution'] for g in point['regions'].values())==pytest.approx(point['realized_risk'])
    assert sum(g['rows'] for g in point['regions'].values())==c['rows']


def test_cost_edge_transitions_are_complete_not_optimized(full_run):
    root,result=full_run
    for w in ('game_equal','record_equal'):
        edge=json.loads((root/f'cost_edges/{w}/00-01/record.json').read_bytes())
        assert len(edge['transition_mass_fraction'])==9
        assert sum(edge['transition_mass_fraction'].values())==pytest.approx(1)
        assert 0<=edge['flipped_mass_fraction']<=1
    assert len(result['cost_edges'])==192


def test_output_cannot_enter_existing_artifacts_or_other_runs(tmp_path,inputs):
    from werewolf.observer_dtrs.validation_artifacts import publish_unit
    a=publish_unit(tmp_path/'artifact','unrelated',{}, {})
    with pytest.raises(ValueError,match='immutable artifact'):
        runner.execute_controlled(a.path/'child',inputs=inputs)
    root=tmp_path/'other-run'
    publish_unit(root/'contract','validation_contract',{}, {})
    with pytest.raises(ValueError,match='different experiment'):
        runner.execute_controlled(root,inputs=inputs)
    assert not (root/'runner.lock').exists()


def test_committed_source_would_still_need_separate_protocol_review(inputs):
    p=_protocol_with_status('READY_FOR_REVIEW_NOT_FROZEN')
    # Controlled simulation of the reviewed status, not a formal execution or source-guard bypass.
    r=p.record;r['status']='FROZEN_FOR_EXECUTION'
    frozen=D3Protocol(canonical_json_bytes(r))
    frozen.require_frozen(frozen.digest)
    with pytest.raises(ValueError,match='frozen'):frozen.require_frozen(p.digest)
    with pytest.raises(ValueError):p.loss(True,p.policies[0])


def test_case_region_records_do_not_change_with_probability_row_order(inputs):
    p=load_protocol();kw=dict(scenario=22,policy=p.policies[0],model='OBS-EQ1',weighting='game_equal')
    value=inputs.legal
    # Canonical JSON reorders keys; semantic ordering does not depend on source rows.
    reverse={key:value[key] for key in reversed(value)}
    other=_changed(inputs,'legal',reverse)
    assert runner._case(inputs,p,**kw)==runner._case(other,p,**kw)


def test_direct_secondary_bootstrap_not_subtracted_intervals(full_run,inputs):
    root,_=full_run
    base=root/'cases/TERMINAL_NEG_FIRST/C22/game_equal'
    eq1=json.loads((base/'OBS-EQ1/record.json').read_bytes())
    lr=json.loads((base/'OBS-LR-r/record.json').read_bytes())
    ef=json.loads((base/'OBS-EF/record.json').read_bytes())
    comparisons=json.loads((root/'comparisons/TERMINAL_NEG_FIRST/C22/game_equal/record.json').read_bytes())
    assert set(comparisons)=={'OBS-EQ1_minus_OBS-EQ0','OBS-LR-r_minus_OBS-EQ1','OBS-EF_minus_OBS-EQ1',
                             'OBS-LR-r_minus_OBS-EQ0','OBS-EF_minus_OBS-EQ0'}
    for name,value in [('OBS-LR-r',lr),('OBS-EF',ef)]:
        r=comparisons[name+'_minus_OBS-EQ1']
        assert r['role']=='key_secondary'
        assert r['point']==value['groups']['overall']['realized_risk']-eq1['groups']['overall']['realized_risk']
        assert r['interval']==runner.bootstrap_risks(eq1['by_game'],inputs=inputs,other=value['by_game'])
    assert '96_preset' in load_protocol().record['amendments_from_D3a']['cost_sensitivity']


def test_prepared_case_and_bootstrap_cache_exact_equivalence(inputs):
    p=load_protocol();prepared=runner._Prepared(inputs)
    kw=dict(scenario=22,policy=p.policies[0],model='OBS-EQ1',weighting='game_equal')
    a=runner._case(inputs,p,**kw);b=runner._case(inputs,p,_prepared=prepared,**kw)
    assert canonical_json_bytes(a)==canonical_json_bytes(b)
    first=runner.bootstrap_risks(a['by_game'],inputs=inputs)
    assert runner.bootstrap_risks(a['by_game'],inputs=inputs,_prepared=prepared)==first
    assert prepared.bootstrap_cache
    # Individual row records remain independent objects after the pure-string cache optimization.
    duplicate=[]
    for row in b['decisions']:
        for old in duplicate:
            if old['risks_exact']==row['risks_exact']:
                assert old['risks_exact'] is not row['risks_exact']
                assert old['minimizing_actions'] is not row['minimizing_actions']
        duplicate.append(row)


def _reanchor_inspection(root,section,key,*,mutate=lambda r:None,identity=lambda i:None):
    path=root/section/key if key else root/section
    h=rewrite(path,transform=mutate,transform_identity=identity)
    return rewrite(root/'complete',transform=lambda r: r[section].__setitem__(key,h) if key else r.__setitem__(section,h))


@pytest.mark.parametrize('section,key,mutate,identity',[
 ('cases','TERMINAL_NEG_FIRST/C00/game_equal/OBS-EQ0',lambda r:None,lambda i:i.__setitem__('model','OBS-EF')),
 ('cases','TERMINAL_NEG_FIRST/C00/game_equal/OBS-EQ0',lambda r:r.__setitem__('scenario',1),lambda i:None),
 ('source','',lambda r:r.__setitem__('fake_source',True),lambda i:None),
 ('restricted_support','',lambda r:r.__setitem__('legal_digest','f'*64),lambda i:None),
 ('bootstrap','',lambda r:r['game_order'].reverse(),lambda i:None),
 ('comparisons','TERMINAL_NEG_FIRST/C00/game_equal',lambda r:r.pop('OBS-LR-r_minus_OBS-EQ1'),lambda i:None),
 ('comparisons','TERMINAL_NEG_FIRST/C00/game_equal',lambda r:r['OBS-LR-r_minus_OBS-EQ1']['interval'].__setitem__('draws_digest','f'*64),lambda i:None),
 ('tie_sensitivity','C00/game_equal/OBS-EQ0',lambda r:None,lambda i:i['cases'].reverse()),
 ('cost_edges','game_equal/00-01',lambda r:r.__setitem__('axis','rho'),lambda i:None),
])
def test_inspect_rejects_rehashed_complete_identity_and_link_corruption(full_run,tmp_path,section,key,mutate,identity):
    source,_=full_run;root=tmp_path/'corrupt';shutil.copytree(source,root)
    _reanchor_inspection(root,section,key,mutate=mutate,identity=identity)
    with pytest.raises((ValueError,ArtifactValidationError)):runner.inspect_result(root)


def test_inspect_external_anchor_and_source_scope(full_run):
    root,result=full_run
    assert runner.inspect_result(root,expected_manifest_digest=result['manifest_digest'])['external_complete_anchor_verified']
    with pytest.raises((ValueError,ArtifactValidationError)):
        runner.inspect_result(root,expected_manifest_digest='f'*64)


def test_checkpoint_rejects_identity_before_expensive_computation(tmp_path):
    from werewolf.observer_dtrs.validation_artifacts import publish_unit
    path=tmp_path/'case';publish_unit(path,'D3_case',{'id':'old'},{'value':1})
    def forbidden():raise AssertionError('identity must be verified before replay')
    with pytest.raises((ValueError,ArtifactValidationError)):
        runner._checkpoint(path,'D3_case',{'id':'different'},forbidden)


@pytest.mark.parametrize('state',['clean','untracked','dirty','blob_mismatch','head_changed'])
def test_d3_extra_source_guard_controlled_states(monkeypatch,state):
    import subprocess
    import werewolf.observer_dtrs.d3_protocol as protocol_module
    extras=('scripts/run_observer_dtrs_decisions.py','configs/observer_dtrs/d3-execution-v1.json')
    monkeypatch.setattr(protocol_module,'source_snapshot',lambda **kw:{'scope':'controlled_guard_unit_test','head':'a'*40,'files':{'werewolf/example.py':'controlled text\n'}})
    def output(args,**kw):
        if args[:2]==['git','ls-files']:
            return '\n'.join(extras if state!='untracked' else extras[:1])
        if args[:3]==['git','diff','HEAD']:
            return 'controlled dirty diff' if state=='dirty' else ''
        if args[:2]==['git','cat-file']:
            captured={'werewolf/example.py':'controlled text\n',**{p:Path(p).read_text() for p in extras}}
            result=b''
            for i,name in enumerate(sorted(captured)):
                value=captured[name].encode()
                if state=='blob_mismatch' and i==0:value+=b'controlled mismatch'
                result+=b'a'*40+b' blob '+str(len(value)).encode()+b'\n'+value+b'\n'
            return result
        if args[:3]==['git','rev-parse','HEAD']:
            return ('b' if state=='head_changed' else 'a')*40+'\n'
        raise AssertionError(args)
    monkeypatch.setattr(subprocess,'check_output',output)
    if state=='clean':
        assert set(protocol_module.checked_source('a'*40)['D3_execution_files'])==set(extras)
    else:
        with pytest.raises(ValueError):protocol_module.checked_source('a'*40)


def test_exact_endpoint_summary_matches_original_probe_enumeration():
    from werewolf.observer_dtrs.decision import candidate_losses,probability_stability,thresholds
    from werewolf.observer_dtrs.d3_stability import interval_summary_engine
    rng=np.random.Generator(np.random.PCG64(20261016))
    for policy in TiePolicy:
        matrices=[*candidate_losses(tie_policy=policy),
                  LossSpec(1,1,0,0,policy),LossSpec(1,1,1,0,policy),LossSpec(1,1,0,1,policy),
                  LossSpec(1,1,Fraction(1,2),Fraction(1,2),policy),LossSpec(1,1,Fraction(3,4),Fraction(3,4),policy)]
        for loss in matrices:
            engine=interval_summary_engine(loss);th=thresholds(loss)
            points=[0,1,Fraction(1,2),Fraction(1,3),float(rng.random())]
            for q in (th.alpha,th.beta,th.gamma):
                if q is not None and 0<=q<=1:
                    points.extend([q,float(q)])
                    points.extend(x for x in (math.nextafter(float(q),0.),math.nextafter(float(q),1.)) if 0<=x<=1)
            for p in points:
                d=decide(p,loss)
                for radius in (0,Fraction(1,100),Fraction(1,40),Fraction(1,20)):
                    expected=probability_stability(p,loss,radius=radius)
                    assert engine(d,radius)=={k:expected[k] for k in ('stable_under_selected_tie_policy','worst_case_regret')}
    loss=load_protocol().loss(0,TiePolicy.TERMINAL_NEG_FIRST);engine=interval_summary_engine(loss)
    with pytest.raises(ValueError):engine(decide(.5,replace(loss,b1=loss.b1/2)),.01)


def test_pressure_fixture_is_explicitly_synthetic_and_bounded():
    from scripts.benchmark_observer_dtrs_decisions import artificial_inputs
    data=artificial_inputs(25,50,'unique')
    assert data.identity['scope']=='controlled_fixture'
    assert len(data.identity['game_sizes'])==25 and data.identity['samples']==50
    assert 'artificial_pressure_only' in data.identity['research_scope']
    for args in ((1501,22685,'unique'),(1500,22686,'unique'),(25,24,'pooled'),(25,50,'formal')):
        with pytest.raises(ValueError):artificial_inputs(*args)
