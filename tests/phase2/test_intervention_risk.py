"""Synthetic contract checks; no gameplay replay or real LLM use."""

import pytest

from scripts.phase2_integration_seam import prepare_phase2_intervention_speech
from tests.phase2.test_decision_opportunity import opportunity
from werewolf.phase2_actions import Action, push_plan
from werewolf.phase2_intervention import (
    CheckpointUnavailable, capture_intervention_checkpoint, plan_paired_branches,
    run_intervention_branch,
)
from werewolf.phase2_language import Phase2SpeechSemanticV1, PublicLanguageContextV1
from werewolf.phase2_offline import (
    ReferenceTables, ResolvedOutcome, ValueEstimate, ValueTable,
)
from werewolf.phase2_outcome import OutcomeError, extract_phase2_day_outcome
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
