"""Runtime-safe opportunity and controlled-treatment contracts."""

from dataclasses import replace

import pytest

from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.phase2_actions import Action, ActionContextV1
from werewolf.phase2_decision_opportunity import (
    DecisionOpportunityError, build_phase2_decision_opportunity,
)
from werewolf.phase2_mapper_runtime import RuntimeInference
from werewolf.phase2_offline import R2Input
from werewolf.phase2_treatment import TreatmentError, build_phase2_treatment


def context(*, phase="speech", actor="player1", pk_targets=("player2", "player5")):
    alive = PLAYER_IDS
    competition = alive if phase == "speech" else tuple(
        p for p in PLAYER_IDS if p in (actor, *pk_targets))
    start = competition.index(actor)
    return ActionContextV1("g", "b", "prefix", "public", phase, actor, alive,
                           frozenset(("player1", "player5")), competition,
                           competition[start:] + competition[:start])


def q_matrix():
    return [[0.0 if i == j else 1 / 6 for j in range(7)] for i in range(7)]


class Mapper:
    artifact_digest = "a" * 64

    def infer(self, q, *, candidate_j, phase, competition, alive,
              known_wolves, acting_wolf, public_speaker_queue, **kwargs):
        p = .8 if candidate_j == "player3" else .4
        observers = tuple(player for player in alive
                          if player not in known_wolves and player != candidate_j)
        remaining = len(public_speaker_queue) - public_speaker_queue.index(acting_wolf) - 1
        z = R2Input(.1, .0, .0, .5, phase, len(observers), len(competition), remaining)
        return RuntimeInference(p, 0.0, z, observers, competition, (), ())


def opportunity(c=None, j="player2"):
    return build_phase2_decision_opportunity(c or context(), j, q_matrix(), Mapper(),
                                             q_source_digest="b" * 64)


def test_ordinary_opportunity_contains_only_current_legal_evidence():
    opp = opportunity()
    assert opp.legal_actions == (Action.PUSH, Action.REDIRECT, Action.PROBE)
    assert opp.redirect_target == "player3"
    assert opp.continuation_actor == "player5"
    assert opp.s_pre == (2, 5)
    assert opp.remaining_speakers == 6
    assert len(opp.evidence_panel) == 5
    record = opp.to_record()
    assert set(record) == {"schema_version", "identity", "public_legal", "s_pre",
                           "evidence", "legal_actions", "redirect_target", "probe"}
    assert not any(term in str(record).lower() for term in (
        "future_vote", "future_exile", "theta_ac", "report_validity"))
    assert opp.digest() == opportunity().digest()


def test_pk_single_candidate_has_no_redirect_and_no_probe_without_later_wolf():
    pk = context(phase="speech_pk", actor="player5", pk_targets=("player2", "player5"))
    opp = opportunity(pk)
    assert opp.legal_actions == (Action.PUSH,)
    assert opp.redirect_target is None and opp.observation_window is None
    with pytest.raises(DecisionOpportunityError, match="no legal candidate"):
        opportunity(pk, "player3")


def test_pk_probe_window_and_state_validation():
    pk = context(phase="speech_pk", pk_targets=("player2", "player5"))
    opp = opportunity(pk)
    assert opp.legal_actions == (Action.PUSH, Action.PROBE)
    assert opp.observation_window.expected_speakers == ("player2",)
    with pytest.raises(DecisionOpportunityError):
        replace(opp, remaining_speakers=0)
    with pytest.raises(DecisionOpportunityError):
        replace(opp, redirect_target="player2")


def test_no_candidate_pk_and_untrusted_q_lineage_fail_closed():
    all_wolf_pk = context(phase="speech_pk", pk_targets=("player1", "player5"))
    assert all_wolf_pk.legal_targets == ()
    with pytest.raises(DecisionOpportunityError, match="no legal candidate"):
        opportunity(all_wolf_pk, "player2")
    with pytest.raises(DecisionOpportunityError, match="Q source provenance"):
        build_phase2_decision_opportunity(context(), "player2", q_matrix(), Mapper(),
                                           q_source_digest="unverified")


def test_mapper_audit_cannot_claim_wrong_observer_panel():
    class WrongPanel(Mapper):
        def infer(self, q, **kwargs):
            result = super().infer(q, **kwargs)
            return replace(result, observer_ids=("player7",))
    with pytest.raises(DecisionOpportunityError, match="audited R2 context"):
        build_phase2_decision_opportunity(context(), "player2", q_matrix(), WrongPanel(),
                                           q_source_digest="b" * 64)


@pytest.mark.parametrize("action", list(Action))
def test_treatment_freezes_each_legal_action_and_assignment(action):
    opp = opportunity()
    treatment = build_phase2_treatment(opp, action, assignment_source="paired_branch",
                                       assignment_probability=1.0, randomization_key="pilot-1")
    assert treatment.plan.candidate_j == opp.candidate_j
    assert treatment.plan.action is action
    assert treatment.treatment_id == build_phase2_treatment(
        opp, action, assignment_source="paired_branch", assignment_probability=1.0,
        randomization_key="pilot-1").treatment_id
    assert treatment.to_record()["observational_vote_intent_is_treatment"] is False


def test_treatment_rejects_unavailable_action_and_observational_source():
    pk = context(phase="speech_pk", actor="player5", pk_targets=("player2", "player5"))
    opp = opportunity(pk)
    with pytest.raises(TreatmentError, match="unavailable"):
        build_phase2_treatment(opp, Action.REDIRECT, assignment_source="paired_branch",
                               assignment_probability=1, randomization_key="x")
    with pytest.raises(TreatmentError, match="controlled assignment"):
        build_phase2_treatment(opp, Action.PUSH, assignment_source="observational_gameplay",
                               assignment_probability=1, randomization_key="x")
    with pytest.raises(TreatmentError):
        build_phase2_treatment(opp, Action.PUSH, assignment_source="paired_branch",
                               assignment_probability=0, randomization_key="x")
