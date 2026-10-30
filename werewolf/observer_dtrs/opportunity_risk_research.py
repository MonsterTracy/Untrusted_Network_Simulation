"""Separate complete-window O/T targets and a same-population hurdle predictor.

No production policy, event execution, D1 supervision, or future-feature API.
The native public reader and original conditional outcome helper authenticate
time boundaries; manually fabricated evidence is not formal provenance.
"""
from collections import Counter
import numpy as np

from werewolf.observer_dtrs.public_behavior_research import observe_after_speech,game_weights
from werewolf.observer_dtrs.baselines import sigmoid,objective,SOLVER

VERSION='complete_PRE_opportunity_risk_v1'


def full_window_targets(conditional):
    """Convert validated old conditional evidence; never rewrite old Z.

    Caller must authenticate the source and exact PRE; nonempty witness strings
    alone cannot certify a complete window or authorize real inference.
    """
    old=dict(conditional)
    status=old['status'];future=old['future_speech_ids']
    if type(future) is not list or len(set(future))!=len(future):raise ValueError('invalid future event identity list')
    incomplete=('TARGET_SPEECH_MISSING','DAY_INCOMPLETE')
    if status in incomplete:
        if old['Z'] is not None:raise ValueError('incomplete window cannot have a target')
        return {**old,'O':None,'T':None,'new_status':'WINDOW_UNKNOWN'}
    if status not in ('OBSERVED','NO_FUTURE_SPEECH','V1_MISSING','V1_FAILED','V1_CONFLICT'):
        raise ValueError('unsupported conditional evidence status')
    if not old.get('target_speech_event_id') or not old.get('cutoff_event_id'):
        raise ValueError('complete target and cutoff witnesses required; no zero fill')
    if status=='NO_FUTURE_SPEECH':
        if future or old['Z'] is not None:raise ValueError('no-opportunity target inconsistency')
        return {**old,'O':0,'T':0,'new_status':'COMPLETE_NO_OPPORTUNITY_EVENT_ZERO'}
    if not future:raise ValueError('observed opportunity requires a speech event')
    if status=='OBSERVED':
        if type(old['Z']) is not int or old['Z'] not in (0,1):raise ValueError('binary conditional target required')
        return {**old,'O':1,'T':old['Z'],'new_status':'COMPLETE_OBSERVED_OPPORTUNITY'}
    if old['Z'] is not None:raise ValueError('ambiguous conditional target cannot be observed')
    return {**old,'O':1,'T':None,'new_status':'OPPORTUNITY_OBSERVED_EVENT_UNKNOWN'}


def observe_complete_window(pre,observer_id,history,annotations,day_witness):
    return full_window_targets(observe_after_speech(pre,observer_id,history,annotations,day_witness))


def fit_conditional(X,opportunity,target,game_ids):
    """Condition on O=1 while inheriting full-reference game weights.

    Original public_behavior_research.fit_target remains unchanged. The solver
    and objective are reused; only this new conditional weighting differs.
    """
    X=np.asarray(X,dtype=np.float64);o=np.asarray(opportunity);t=np.asarray(target)
    if X.ndim!=2 or o.shape!=(len(X),) or t.shape!=(len(X),) or len(game_ids)!=len(X) or not np.isfinite(X).all():
        raise ValueError('invalid full-reference arrays')
    if (not np.issubdtype(o.dtype,np.integer) or not np.issubdtype(t.dtype,np.integer)
            or not np.isin(o,[0,1]).all() or not np.isin(t,[0,1]).all() or np.any(t>o)):
        raise ValueError('binary coherent O/T required')
    keep=o==1
    if len(set(t[keep]))!=2:raise ValueError('UNSUPPORTED_CONDITIONAL_CLASS_SUPPORT')
    weights=game_weights(game_ids)[keep];weights/=weights.sum();xc=X[keep];y=t[keep]
    mean=weights@xc;std=np.sqrt(weights@((xc-mean)**2));std[np.ptp(xc,axis=0)==0]=1.
    design=np.column_stack([np.ones(len(xc)),(xc-mean)/std]);beta=np.zeros(design.shape[1]);tau=1/len(xc)
    for iteration in range(SOLVER['max_iterations']+1):
        loss,gradient,hessian=objective(beta,design,y,weights,tau);norm=float(np.max(np.abs(gradient)))
        if norm<=SOLVER['gradient_tolerance']:break
        if iteration==SOLVER['max_iterations']:raise RuntimeError('weighted conditional nonconvergence')
        direction=np.linalg.solve(hessian,gradient);descent=float(gradient@direction);step=1.
        for _ in range(SOLVER['max_backtracks']):
            candidate=beta-step*direction;new_loss,new_gradient,_=objective(candidate,design,y,weights,tau)
            tolerance=32*np.finfo(float).eps*max(1.,abs(loss))
            roundoff=(step*descent<=tolerance and new_loss<=loss+tolerance and float(np.max(np.abs(new_gradient)))<norm)
            if new_loss<=loss-SOLVER['armijo']*step*descent or roundoff:beta=candidate;break
            step/=2
        else:raise RuntimeError('weighted conditional line search failed')
    if not np.isfinite(beta).all():raise RuntimeError('invalid weighted conditional fit')
    return {'version':VERSION,'mean':mean.tolist(),'std':std.tolist(),'beta':beta.tolist(),
        'full_reference_rows':len(X),'full_reference_games':len(set(game_ids)),
        'train_rows':len(xc),'train_games':len({g for g,k in zip(game_ids,keep) if k}),'positive':int(y.sum()),
        'weight_rule':'inherit_1_over_full_reference_game_size_then_global_normalize',
        'tau':tau,'solver':SOLVER,'iterations':iteration,'gradient_inf':norm,'objective':loss}


def product_scores(opportunity_logits,conditional_logits,target):
    """Score marginal T=o*z at EVERY test PRE. No future O argument is accepted."""
    a=np.asarray(opportunity_logits,dtype=np.float64);b=np.asarray(conditional_logits,dtype=np.float64);y=np.asarray(target)
    if (a.shape!=b.shape or a.shape!=y.shape or a.ndim!=1 or not np.isfinite(a).all()
            or not np.isfinite(b).all() or not np.isin(y,[0,1]).all()):raise ValueError('invalid two-stage arrays')
    log_o=-np.logaddexp(0,-a);log_no_o=-np.logaddexp(0,a)
    log_c=-np.logaddexp(0,-b);log_no_c=-np.logaddexp(0,b)
    log_t=log_o+log_c;log_no_t=np.logaddexp(log_no_o,log_o+log_no_c)
    probability=sigmoid(a)*sigmoid(b)
    if not np.isfinite(probability).all() or not ((probability>=0)&(probability<=1)).all():raise ValueError('invalid product probability')
    loss=-y*log_t-(1-y)*log_no_t
    if not np.isfinite(loss).all():raise ValueError('product score is not representable in float64')
    return probability,(probability-y)**2,loss
