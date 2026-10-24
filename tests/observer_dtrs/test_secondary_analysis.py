"""Controlled mathematical tests, no formal predictions or performance claims."""
from copy import deepcopy
from dataclasses import replace
import math

import numpy as np
import pytest

from tests.observer_dtrs.test_estimation import features, row, fit
from werewolf.observer_dtrs.d2_protocol import frozen_protocol
from werewolf.observer_dtrs.baselines import design, objective, fit_logistic, empirical_model, predict_many, bin_audit
from werewolf.observer_dtrs.statistical_analysis import analyze, bootstrap_plan, MODELS, WEIGHTINGS


@pytest.fixture
def protocol():
    return frozen_protocol()


def test_logistic_gradient_hessian_and_frozen_six_column_design():
    f=[features(.2,'discussion',1),features(.7,'discussion',2),features(.1,'pk_discussion',1),features(.9,'pk_discussion',2)]
    X=np.array([design(v,.5,.2) for v in f])
    assert X.shape==(4,6)
    assert X[0].tolist()==[1,0,0,0,-1.4999999999999998,0]
    beta=np.array([.2,-.1,.3,.4,.5,-.2]);y=np.array([0,1,0,1]);w=np.array([.1,.2,.3,.4])
    loss,grad,hess=objective(beta,X,y,w,.1)
    delta=1e-5
    for j in range(6):
        step=np.zeros(6);step[j]=delta
        plus=objective(beta+step,X,y,w,.1);minus=objective(beta-step,X,y,w,.1)
        assert grad[j]==pytest.approx((plus[0]-minus[0])/(2*delta),abs=1e-9)
        assert hess[:,j]==pytest.approx((plus[1]-minus[1])/(2*delta),abs=1e-9)
    assert np.linalg.eigvalsh(hess).min()>0


def test_logistic_constant_preprocessing_single_class_and_input_order(protocol):
    rows=[row(f'g{j}',j%2,.3) for j in range(20)]
    parent=fit(rows,protocol)
    kwargs=dict(protocol=protocol,tau=.01,weighting='game_equal',source=parent.record['source'])
    model=fit_logistic(rows,**kwargs)
    assert model['std_r']==0
    assert model==fit_logistic(reversed(rows),**kwargs)
    assert model['convergence']['gradient_inf']<=1e-9
    assert predict_many(model,[features(.99)])[0]['probability']==pytest.approx(.5)
    positive=fit_logistic([replace(r,y=1) for r in rows],**kwargs)
    assert positive['constant']==pytest.approx(20.5/21)
    assert positive['coefficients'] is None
    with pytest.raises(ValueError):
        fit_logistic(rows,**{**kwargs,'tau':.02})


def test_empirical_is_exact_unsmoothed_selected_eq1_structure_and_merge_trace(protocol):
    rows=[row(f'g{j:02}',int(j>=10),(.1,.4,.8)[j//10]) for j in range(30)]
    eq=fit(rows,protocol)
    ef=empirical_model(eq)
    assert ef['parent_EQ1_digest']==eq.digest
    assert [v['probability'] for v in predict_many(ef,[features(.1),features(.4),features(.9)])]==[0,1,1]
    assert ef['state_nodes']['discussion/2']['edges']==eq.record['state_nodes']['discussion/2']['edges']
    trace=bin_audit(fit([row(f'x{j}',j%2,(j+1)/100) for j in range(40)],protocol,bins=5),
                     [row(f'x{j}',j%2,(j+1)/100) for j in range(40)],protocol)
    assert trace['discussion/2']['final_edges']==[.16]
    assert len(trace['discussion/2']['merges'])==3


def analysis_fixture():
    blocks={};rows=[]
    for k in range(5):
        for j in range(5):
            game=f'g-{k}-{j}';blocks[(k,j)]=[game]
            for h in range(1+(j%2)):
                y=int((k+j+h)%3!=0)
                rows.append({'sample_id':f'{game}-{h}','game_id':game,'phase':'discussion','nW':1,
                    'y':y,'successful_empty':False,
                    'predictions':{w:{m:.5 for m in MODELS} for w in WEIGHTINGS}})
    return rows,blocks


def test_bootstrap_cluster_plan_paired_scores_calibration_and_determinism(protocol):
    rows,blocks=analysis_fixture()
    order,indices=bootstrap_plan(blocks,protocol=protocol,formal=False)
    assert indices.shape==(2000,25)
    assert np.array_equal(indices,np.tile(np.arange(25),(2000,1)))
    summary,raw=analyze(rows,blocks=blocks,protocol=protocol,scope='controlled_fixture')
    again,other=analyze(list(reversed(rows)),blocks=dict(reversed(list(blocks.items()))),protocol=protocol,scope='controlled_fixture')
    # Floating summation order is fixed by runner sample order; these exact .5 rows coincide.
    assert summary==again and raw==other
    for w in WEIGHTINGS:
        delta=summary['comparisons'][w]['OBS-EQ1_minus_OBS-EQ0']
        assert delta['point']['brier']==0 and delta['intervals']['brier']==[0,0]
        result=summary['models'][w]['OBS-EQ0']
        assert result['groups']['overall']['brier']==.25
        assert result['groups']['overall']['log_loss']==pytest.approx(math.log(2))
        assert result['calibration'][5]['games']==25
        assert result['calibration'][5]['intervals'] is not None
        assert result['calibration'][0]['observed_rate'] is None
    with pytest.raises(ValueError):
        bootstrap_plan(blocks,protocol=protocol,formal=True)
    with pytest.raises(ValueError):
        analyze(rows+rows[:1],blocks=blocks,protocol=protocol,scope='controlled_fixture')


def test_bootstrap_draws_never_cross_source_outer_blocks(protocol):
    blocks={(k,j):[f'{k}-{j}-{h}' for h in range(3)] for k in range(5) for j in range(5)}
    order,indices=bootstrap_plan(blocks,protocol=protocol,formal=False)
    for b in range(25):
        assert indices[:,b*3:(b+1)*3].min()>=b*3
        assert indices[:,b*3:(b+1)*3].max()<(b+1)*3
    _,same=bootstrap_plan(blocks,protocol=protocol,formal=False)
    assert np.array_equal(indices,same)


def test_calibration_endpoints_game_weights_and_empty_report_audit(protocol):
    rows,blocks=analysis_fixture()
    for j,r in enumerate(rows):
        p=0. if j%2 else 1.
        r['predictions']={w:{m:p for m in MODELS} for w in WEIGHTINGS}
        r['successful_empty']=bool(r['y'] and j%2==0)
    summary,_=analyze(rows,blocks=blocks,protocol=protocol,scope='controlled_fixture')
    result=summary['models']['game_equal']['OBS-EQ0']
    assert sum(b['rows'] for b in result['calibration'])==len(rows)
    assert sum(b['rows'] for b in result['calibration'][1:9])==0
    assert result['clipped_rows']==len(rows)
    assert result['groups']['successful_empty']['positive']==result['groups']['successful_empty']['rows']
    assert sum(b['mass'] for b in result['calibration'])==pytest.approx(25)
