"""Synthetic online randomization and three-table audit; no model calls."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.phase2.test_decision_opportunity import Mapper, context, opportunity, q_matrix
from tests.phase2.test_intervention_risk import values
from werewolf.phase2_actions import Action
from werewolf.phase2_decision_opportunity import build_phase2_decision_opportunity
from werewolf.phase2_online_plan import (
    FROZEN, OnlinePilotGameStateV1, OnlinePilotPlanError,
    Phase2OnlineTerminalPilotPlanV1, assign_terminal_action, select_candidate,
)
from werewolf.phase2_online_records import (
    OnlinePilotRecordError, Phase2OnlineInterventionRecordV1,
    record_online_day_outcome, record_online_execution,
)
from werewolf.phase2_outcome import (
    extract_phase2_day_outcome, reference_tables_digest,
)
from werewolf.phase2_pilot_dataset import (
    PilotDatasetError, analyze_phase2_online_support_record,
    build_phase2_consequence_dataset,
)


def _assignment(opp=None, seed=17):
    opp = opp or opportunity()
    plan = Phase2OnlineTerminalPilotPlanV1("study", seed)
    selection = select_candidate(plan, opp.legal_context)
    if selection.candidate_j != opp.candidate_j:
        opp = build_phase2_decision_opportunity(
            opp.legal_context, selection.candidate_j, q_matrix(), Mapper(),
            q_source_digest=opp.q_source_digest)
    return plan, assign_terminal_action(plan, selection, opp)


def test_frozen_uniform_candidate_rule_and_deterministic_half_half_assignment():
    opp = opportunity()
    plan, assignment = _assignment(opp)
    assert plan.selection_status == FROZEN
    assert assignment == assign_terminal_action(
        plan, select_candidate(plan, opp.legal_context), assignment.opportunity)
    assert assignment.selection.candidate_j in assignment.selection.candidate_pool
    assert assignment.selection.candidate_selection_probability == 1 / len(
        assignment.selection.candidate_pool)
    assert {select_candidate(Phase2OnlineTerminalPilotPlanV1("study", seed),
                             opp.legal_context).candidate_j for seed in range(40)} == set(
        opp.legal_context.legal_targets)
    assert assignment.treatment.action in (Action.PUSH, Action.REDIRECT)
    assert assignment.treatment.assignment_probability == .5
    assert assignment.assignment_key == assignment.treatment.randomization_key
    assert assignment.to_record()["legal_action_set"][:2] == ["PUSH", "REDIRECT"]
    assert {_assignment(opp, seed)[1].treatment.action for seed in range(40)} == {
        Action.PUSH, Action.REDIRECT}
    assert select_candidate(plan, context(phase="speech_pk", actor="player5",
                                          pk_targets=("player2", "player5"))) is None
    with pytest.raises(OnlinePilotPlanError):
        Phase2OnlineTerminalPilotPlanV1("study", 1, push_probability=1.0)
    with pytest.raises(OnlinePilotPlanError):
        replace(plan, selection_status="PROVISIONAL_NEEDS_REVIEW")


def test_one_assignment_per_game_including_language_failure():
    plan, first_assignment = _assignment(opportunity(), 17)
    opp = first_assignment.opportunity
    selection = first_assignment.selection
    state = OnlinePilotGameStateV1()
    first = state.try_assign(plan, selection, opp)
    assert first is not None
    assert state.try_assign(plan, selection, opp) is None
    assert len(state.assignments) == 1
    assert state.eligible_opportunities == 1


def test_reference_table_content_digest_is_stable():
    assert reference_tables_digest(values()) == reference_tables_digest(values())
    assert len(reference_tables_digest(values())) == 64


def _completed_record(opp, seed, *, success):
    _, assignment = _assignment(opp, seed)
    initial = Phase2OnlineInterventionRecordV1(assignment)
    if success:
        link = SimpleNamespace(canonical_event_id="event-000004",
                               canonical_event_digest="a" * 64,
                               public_text_digest="b" * 64)
        row = record_online_execution(initial, language_audit_digest="c" * 64,
                                      attempt_count=1, backend_calls=(),
                                      canonical_link=link)
    else:
        row = record_online_execution(initial, language_audit_digest="d" * 64,
                                      attempt_count=2, backend_calls=(),
                                      failure_reason="REDIRECT_REQUEST_CONFLICT")
        return row
    outcome = extract_phase2_day_outcome(assignment.opportunity, "player2", values())
    return record_online_day_outcome(row, outcome,
                                     reference_artifact_digest="e" * 64)


def test_failed_language_keeps_assignment_but_not_consequence():
    first = _completed_record(opportunity(), 1, success=False)
    with pytest.raises(OnlinePilotRecordError, match="successful execution"):
        record_online_day_outcome(
            first, extract_phase2_day_outcome(first.assignment.opportunity,
                                              "player2", values()),
            reference_artifact_digest="e" * 64)
    other_context = replace(context(), game_id="g2", boundary_id="b2")
    second_opp = build_phase2_decision_opportunity(
        other_context, "player2", q_matrix(), Mapper(), q_source_digest="b" * 64)
    second = _completed_record(second_opp, 1, success=True)
    dataset = build_phase2_consequence_dataset(
        source_commit="abc", online_pilot_plan_digest=first.assignment.selection.plan_digest,
        online_records=(first, second),
        games_seen=2, eligible_opportunities=2, allow_synthetic=True)
    assert len(dataset.assignments) == len(dataset.executions) == 2
    assert len(dataset.consequences) == 1
    assert dataset.executions[0]["success"] is False
    assert dataset.assignments[0]["assigned_action"] in ("PUSH", "REDIRECT")
    report = analyze_phase2_online_support_record(dataset.to_record())
    assert report["games_seen"] == 2 and report["games_assigned"] == 2
    assert sum(report["execution_success_by_action"].values()) == 1
    assert report["fitted_lambda"] is None
    with pytest.raises(PilotDatasetError):
        build_phase2_consequence_dataset(
            source_commit="abc",
            online_pilot_plan_digest=first.assignment.selection.plan_digest,
            online_records=(first, first), games_seen=2,
            eligible_opportunities=2, allow_synthetic=True)
    with pytest.raises(OnlinePilotRecordError):
        replace(first, execution_success=True)
