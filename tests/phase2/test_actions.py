"""Synthetic, outcome-free Phase-2 Action Contract V1 checks."""

from dataclasses import replace

import pytest

from scripts.analyze_phase2_action_availability import analyze_contexts
from werewolf.artifact_io import canonical_json_bytes
from werewolf.phase2_action_audit import audit_structured_plan
from werewolf.phase2_actions import (
    Action, ActionContextV1, ActionContractError, InformationRequestV1,
    NO_STANCE, Phase2SemanticPlanV1, probe_continuation, probe_plan,
    push_plan, redirect_plan, select_redirect_target, verify_plan,
)


def context(*, phase="speech", queue=None, competition=None, actor="player1",
            alive=None, boundary="b1"):
    alive = alive or tuple(f"player{i}" for i in range(1, 8))
    competition = competition or (alive if phase == "speech" else
                                  ("player1", "player2", "player5"))
    queue = queue or (alive if phase == "speech" else competition)
    return ActionContextV1("game", boundary, "prefix", "public", phase, actor,
                           alive, frozenset(("player1", "player5")),
                           competition, queue)


def test_push_legal_and_illegal_candidate():
    current = context()
    plan = push_plan(current, "player2")
    assert verify_plan(current, plan).valid
    assert (plan.commitment_target, plan.vote_intent) == ("player2", "player2")
    with pytest.raises(ActionContractError, match="ILLEGAL_CANDIDATE"):
        push_plan(current, "player5")
    assert verify_plan(current, replace(plan, vote_intent="player3")).invalid_reason == "PUSH_SEMANTICS_MISMATCH"


def test_redirect_selector_max_and_canonical_tie_break():
    current = context()
    scores = {j: .1 for j in current.legal_targets}
    scores.update(player3=.8, player4=.8)
    assert select_redirect_target(current, "player2", scores) == "player3"
    assert select_redirect_target(current, "player2", {k: v for k, v in scores.items()
                                                        if k != "player2"}) == "player3"
    plan = redirect_plan(current, "player2", scores)
    assert (plan.redirect_target, plan.rejected_target, plan.vote_intent) == (
        "player3", "player2", "player3")
    assert verify_plan(current, plan).valid
    assert verify_plan(current, replace(plan, redirect_target="player2")).invalid_reason == "ILLEGAL_REDIRECT_TARGET"
    with pytest.raises(ActionContractError, match="cover all alternatives"):
        redirect_plan(current, "player2", {"player3": .8})


def test_redirect_unavailable_without_alternative_and_pk_universe():
    pk = context(phase="speech_pk")
    assert pk.legal_targets == ("player2",)
    assert select_redirect_target(pk, "player2", {}) is None
    with pytest.raises(ActionContractError, match="NO_REDIRECT_ALTERNATIVE"):
        redirect_plan(pk, "player2", {})
    assert push_plan(pk, "player2").candidate_j == "player2"
    with pytest.raises(ActionContractError, match="ILLEGAL_CANDIDATE"):
        push_plan(pk, "player3")


def test_probe_direct_queue_and_pk():
    ordinary = context()
    plan = probe_plan(ordinary, "player3")
    assert plan.continuation_actor == "player5"
    assert plan.expected_observation_window.expected_speakers == (
        "player2", "player3", "player4")
    assert plan.information_request == InformationRequestV1("player3", "player3")
    assert plan.vote_intent == NO_STANCE
    assert verify_plan(ordinary, plan).valid
    pk = context(phase="speech_pk")
    assert probe_plan(pk, "player2").continuation_actor == "player5"


def test_probe_no_later_wolf_or_j_after_teammate():
    late_wolf = context(actor="player5")
    assert probe_continuation(late_wolf, "player6") is None
    with pytest.raises(ActionContractError, match="NO_PROBE_CONTINUATION"):
        probe_plan(late_wolf, "player6")
    teammate_before_j = context()
    assert teammate_before_j.public_speaker_queue.index("player5") < teammate_before_j.public_speaker_queue.index("player6")
    assert probe_continuation(teammate_before_j, "player6") is None
    assert probe_continuation(context(), "player6") is None


def test_reject_noncyclic_public_queue():
    with pytest.raises(ActionContractError, match="cyclic"):
        context(queue=("player1", "player5", "player2", "player3",
                       "player4", "player6", "player7"))


def test_no_stance_is_not_probe_and_structured_verifier_rejects_drift():
    current = context()
    good = probe_plan(current, "player2")
    assert verify_plan(current, replace(good, information_request=None)).invalid_reason == "INVALID_INFORMATION_REQUEST"
    assert verify_plan(current, replace(good, information_request=InformationRequestV1(
        "player2", "player3"))).invalid_reason == "INVALID_INFORMATION_REQUEST"
    assert verify_plan(current, replace(good, vote_intent="player2")).invalid_reason == "PROBE_SEMANTICS_MISMATCH"
    assert verify_plan(current, replace(good, commitment_target="player2")).invalid_reason == "PROBE_SEMANTICS_MISMATCH"
    no_request = Phase2SemanticPlanV1(Action.PROBE, "player2", None, None, None,
                                      None, NO_STANCE, current.phase, current.acting_wolf,
                                      current.legal_targets, *probe_continuation(current, "player2"))
    assert not verify_plan(current, no_request).valid


def test_audit_is_deterministic_and_does_not_claim_language_execution():
    current = context()
    first = audit_structured_plan(current, probe_plan(current, "player2"))
    second = audit_structured_plan(current, probe_plan(current, "player2"))
    assert first.execution_valid and first.invalid_reason is None
    assert first.generated_text_digest is None and first.request_executed is None
    assert first.observation_event_ids is None and first.information_gain is None
    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.canonical_bytes() == canonical_json_bytes(first.to_record())


def test_synthetic_action_sets_and_no_future_dependency():
    first = context(boundary="b1")
    second = context(phase="speech_pk", boundary="b2")
    third = context(actor="player5", boundary="b3")
    result = analyze_contexts((first, second, third))
    assert result["population"]["candidate_pre"] == 3
    assert result["population"]["candidate_rows"] == 11
    assert result["population"]["push_available"] == 11
    assert result["population"]["redirect_available"] == 10
    assert result["population"]["probe_available"] == 4
    assert result["population"]["probe_eligible_pre"] == 2
    assert result["population"]["action_sets"] == {
        "{P,B}": 1, "{P,B,N}": 3, "{P,N}": 7}
    assert result["probe_continuation_relation"] == {"teammate": 4}
    assert result["future_outcome_used"] is False
    assert not hasattr(first, "resolved_outcome")
    with pytest.raises(ValueError, match="duplicate"):
        analyze_contexts((first, first))
