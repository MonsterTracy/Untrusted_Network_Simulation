"""Synthetic contract checks; no gameplay replay or real LLM use."""

from dataclasses import replace

import pytest

from scripts.phase2_integration_seam import prepare_phase2_intervention_speech
from tests.phase2.test_decision_opportunity import opportunity
from werewolf.phase2_actions import Action, InformationRequestV1, push_plan
from werewolf.phase2_intervention import (
    CheckpointUnavailable, capture_intervention_checkpoint, plan_paired_branches,
    run_intervention_branch,
)
from werewolf.phase2_language import Phase2SpeechSemanticV1, PublicLanguageContextV1
from werewolf.phase2_offline import (
    ReferenceTables, ResolvedOutcome, ValueEstimate, ValueTable,
)
from werewolf.phase2_outcome import OutcomeError, extract_phase2_day_outcome
from werewolf.phase2_probe_value import ProbeValueUnavailable, evaluate_probe_value
from werewolf.phase2_risk import (
    Phase2TerminalLossTableV1, RiskError, classic_static_threshold_reduction,
    compare_available_action_risks,
    load_production_terminal_loss_table, terminal_action_risks,
    terminal_risks_for_opportunity,
)
from werewolf.phase2_treatment import build_phase2_treatment
from werewolf.phase2_verified_speech import verified_phase2_speech


def values():
    cells = {(2, 5): .4, (2, 4): .6, (1, 5): .2}
    estimates = {state: ValueEstimate(value, "empirical", None, 1, 1, 1, 1, value)
                 for state, value in cells.items()}
    return ReferenceTables({"g": 0}, {}, ValueTable(None, 1, estimates))


@pytest.mark.parametrize("exiled,category,state,loss", [
    ("player2", ResolvedOutcome.TARGET_J_EXILED, (2, 4), .4),
    ("player3", ResolvedOutcome.OTHER_NONWOLF_EXILED, (2, 4), .4),
    ("player1", ResolvedOutcome.ACTING_WOLF_EXILED, (1, 5), .8),
    ("player5", ResolvedOutcome.TEAMMATE_WOLF_EXILED, (1, 5), .8),
    (None, ResolvedOutcome.NO_EXILE, (2, 5), .6),
])
def test_outcome_uses_frozen_taxonomy_transition_and_reference(exiled, category, state, loss):
    result = extract_phase2_day_outcome(opportunity(), exiled, values())
    assert result.category is category
    assert result.s_plus == state
    assert result.l_ref == pytest.approx(loss)
    assert result.reference.outcome is category


def test_outcome_rejects_exile_outside_pre_alive():
    with pytest.raises(OutcomeError):
        extract_phase2_day_outcome(opportunity(), "player8", values())


def test_terminal_risk_exact_edges_and_missing_losses():
    opp = opportunity()
    table = Phase2TerminalLossTableV1(opp.digest(), "synthetic", .2, .8, .6, .3)
    assert terminal_action_risks(.25, table, opportunity_digest=opp.digest(),
                                 redirect_legal=True) == {
        Action.PUSH: pytest.approx(.65), Action.REDIRECT: pytest.approx(.375)}
    assert terminal_action_risks(0, table, opportunity_digest=opp.digest(),
                                 redirect_legal=True)[Action.PUSH] == .8
    assert terminal_action_risks(1, table, opportunity_digest=opp.digest(),
                                 redirect_legal=True)[Action.PUSH] == .2
    assert terminal_risks_for_opportunity(opp, table)[Action.PUSH] == pytest.approx(.56)
    with pytest.raises(RiskError):
        terminal_action_risks(1.01, table, opportunity_digest=opp.digest(), redirect_legal=True)
    with pytest.raises(RiskError):
        terminal_action_risks(.5, replace(table, lambda_n_plus=None, lambda_n_minus=None),
                              opportunity_digest=opp.digest(), redirect_legal=True)
    with pytest.raises(RiskError):
        load_production_terminal_loss_table()


def test_nonproduction_comparison_requires_all_and_only_legal_risks():
    actions = (Action.PUSH, Action.REDIRECT, Action.PROBE)
    result = compare_available_action_risks(actions, {
        Action.PUSH: .3, Action.REDIRECT: .3, Action.PROBE: .5})
    assert result.selected is Action.PUSH and result.tied_minimum == (Action.PUSH, Action.REDIRECT)
    assert not result.production_router
    with pytest.raises(RiskError):
        compare_available_action_risks(actions, {Action.PUSH: .3, Action.REDIRECT: .4})
    with pytest.raises(RiskError):
        compare_available_action_risks((Action.PUSH,), {Action.PUSH: .3, Action.PROBE: .2})


def test_classic_threshold_is_algebraic_audit_only():
    result = classic_static_threshold_reduction(
        p_plus=0, p_minus=4, b_plus=1, b_minus=1, n_plus=4, n_minus=0)
    assert result.alpha == pytest.approx(.75)
    assert result.beta == pytest.approx(.25)
    assert result.gamma == pytest.approx(.5)
    assert result.boundary_has_strict_region and not result.production_probe_router


class SyntheticProbe:
    evidence_artifact_digest = ""

    def observation_distribution(self, opportunity, request):
        return {"answer": .25, "silence": .75}

    def transition_state(self, opportunity, request, observation):
        return (opportunity.digest(), observation)

    def continuation_value(self, state):
        return {"answer": .1, "silence": .5}[state[1]]


def test_probe_exact_sequential_expectation_and_production_fail_closed():
    opp = opportunity()
    request = InformationRequestV1(opp.candidate_j, opp.candidate_j)
    result = evaluate_probe_value(opp, request, SyntheticProbe(), kappa=.2,
                                  allow_synthetic=True)
    assert result.risk == pytest.approx(.2 + .25 * .1 + .75 * .5)
    assert not result.production_ready
    with pytest.raises(ProbeValueUnavailable):
        evaluate_probe_value(opp, request, SyntheticProbe(), kappa=.2)
    with pytest.raises(ProbeValueUnavailable):
        evaluate_probe_value(opp, request, None, kappa=.2, allow_synthetic=True)


def test_paired_plans_share_checkpoint_seed_but_cannot_run():
    opp = opportunity()
    treatments = tuple(build_phase2_treatment(
        opp, action, assignment_source="paired_branch", assignment_probability=1.0,
        randomization_key="pilot-x") for action in opp.legal_actions)
    branches = plan_paired_branches("full-state-checkpoint-required", treatments,
                                    downstream_seed=123)
    assert len({branch.checkpoint_identity for branch in branches}) == 1
    assert len({branch.downstream_seed for branch in branches}) == 1
    assert len({branch.branch_identity for branch in branches}) == len(branches)
    assert all(not branch.executable for branch in branches)
    with pytest.raises(CheckpointUnavailable):
        capture_intervention_checkpoint(opp)
    with pytest.raises(CheckpointUnavailable):
        run_intervention_branch(branches[0])


class Actor:
    def __init__(self, texts):
        self.texts = iter(texts)

    def realize(self, plan, public_context, *, failure_reason=None):
        return next(self.texts)


class Perceiver:
    def __init__(self, targets):
        self.targets = iter(targets)

    def perceive(self, text, public_context):
        target = next(self.targets)
        return Phase2SpeechSemanticV1(public_context.speaker, public_context.phase,
                                      (target,), (), (target,), ())


def test_verified_speech_seam_is_commit_ready_only_after_verifier_success():
    opp = opportunity()
    c = opp.legal_context
    public = PublicLanguageContextV1(c.phase, c.acting_wolf, c.alive,
                                     c.competition, c.public_history_digest)
    plan = push_plan(c, opp.candidate_j)
    success = verified_phase2_speech(plan, c, public,
                                     actor=Actor(["本轮投player2。"]),
                                     perceiver=Perceiver(["player2"]))
    assert success.success and success.public_text == "本轮投player2。"
    assert success.attempt_count == 1 and success.failure_reason is None
    failure = verified_phase2_speech(plan, c, public,
                                     actor=Actor(["投player3。", "还是投player3。"]),
                                     perceiver=Perceiver(["player3", "player3"]))
    assert not failure.success and failure.public_text is None
    assert failure.attempt_count == 2
    assert failure.failure_reason == "PUSH_COMMITMENT_MISMATCH"


def test_opt_in_integration_prepares_speech_without_gameplay_or_vote():
    opp = opportunity()
    c = opp.legal_context
    public = PublicLanguageContextV1(c.phase, c.acting_wolf, c.alive,
                                     c.competition, c.public_history_digest)
    treatment = build_phase2_treatment(opp, Action.PUSH,
        assignment_source="paired_branch", assignment_probability=1,
        randomization_key="pilot-x")
    result = prepare_phase2_intervention_speech(opp, treatment, public,
        actor=Actor(["本轮投player2。"]), perceiver=Perceiver(["player2"]))
    assert result.verified.success and result.verified.public_text == "本轮投player2。"
    assert not result.gameplay_committed and not result.actual_vote_executed
