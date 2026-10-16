"""Exact endpoint reduction of the existing interval-stability SUMMARY only.

For affine action risks and a fixed total tie order, each selected-action set is
an intersection of open/closed half-lines, hence convex. Identical selections
at both interval endpoints imply identical selection everywhere between them.
For a fixed action, regret is the maximum of affine risk differences, hence
convex; its interval maximum occurs at an endpoint. No threshold approximation,
probability clipping, supervision, support, or alternative decision rule is used.
"""
from fractions import Fraction

from werewolf.observer_dtrs.decision import Decision,LossSpec,Region,_number


def interval_summary_engine(loss):
    if not isinstance(loss,LossSpec):raise TypeError('validated LossSpec required')
    expected_digest=loss.digest
    a,c,b1,b0=loss.c_fp,loss.c_fn,loss.b1,loss.b0
    def risks(q):
        return {Region.POS:a*(1-q),Region.BND:b1*q+b0*(1-q),Region.NEG:c*q}
    def choose(values):
        minimum=min(values.values())
        return next(action for action in loss.tie_policy.order if values[action]==minimum)
    def summary(decision,radius):
        if (not isinstance(decision,Decision) or decision.loss_digest!=expected_digest
                or decision.tie_policy!=loss.tie_policy):raise ValueError('decision/loss binding mismatch')
        radius=_number(radius,'probability perturbation radius')
        if radius<0:raise ValueError('probability radius must be nonnegative')
        if radius==0:
            return {'stable_under_selected_tie_policy':True,'worst_case_regret':0.}
        p=decision.probability
        low,high=max(Fraction(0),p-radius),min(Fraction(1),p+radius)
        left,right=risks(low),risks(high)
        regret=max(left[decision.region]-min(left.values()),right[decision.region]-min(right.values()))
        return {'stable_under_selected_tie_policy':choose(left)==choose(right),'worst_case_regret':float(regret)}
    return summary
