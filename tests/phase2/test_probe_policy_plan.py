"""Public selection, whole-regime assignment, and ledger round-trip contracts."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json

import pytest

from werewolf.phase2_actions import Action, ActionContextV1, ObservationWindowV1
from werewolf.phase2_decision_opportunity import CandidateEvidenceV1, Phase2DecisionOpportunityV1
from werewolf.phase2_online_plan import (
    PROBE_SELECTION_RULE, SELECTION_RULE, OnlinePilotGameStateV1, OnlinePilotPlanError,
    OnlineTerminalAssignmentV1, Phase2OnlineProbePilotPlanV1,
    Phase2OnlineTerminalPilotPlanV1, ProbeStrategy,
    assign_probe_strategy, probe_assignment_from_record, probe_candidate_pool,
    probe_opportunity_from_record, select_probe_candidate,
    select_candidate,
)
from werewolf.phase2_treatment import build_phase2_treatment


def public_context(*, phase="speech", singleton=False, game_id="game-1"):
    alive = ("player1", "player2", "player3", "player4", "player5", "player6", "player7")
    if singleton:
        alive = ("player1", "player2", "player5", "player6", "player7")
    competition = alive
    if phase == "speech_pk":
        competition = (("player1", "player2", "player5", "player6") if singleton else
                       ("player1", "player2", "player3", "player5", "player6"))
    return ActionContextV1(
        game_id, "pre-1", "a" * 64, "b" * 64, phase, "player1", alive,
        frozenset(("player1", "player5")), competition, competition)


@pytest.mark.parametrize(("phase", "expected"), [
    ("speech", ("player2", "player3", "player4")),
    ("speech_pk", ("player2", "player3")),
])
def test_uniform_selection_uses_probe_and_redirect_joint_legal_pool(phase, expected):
    plan = Phase2OnlineProbePilotPlanV1("probe-study", 11, 4, 10)
    context = public_context(phase=phase)

    assert probe_candidate_pool(context) == expected
    selection = select_probe_candidate(plan, context)
    assert selection.candidate_pool == expected
    assert selection.candidate_j in expected
    assert selection.candidate_selection_probability == 1 / len(expected)
    assert selection.rule == PROBE_SELECTION_RULE
    assert select_probe_candidate(plan, context) == selection


def public_opportunity(*, phase="speech", probabilities=None, game_id="game-1"):
    context = public_context(phase=phase, singleton=True, game_id=game_id)
    probabilities = probabilities or {"player2": 0.99, "player6": 0.88, "player7": 0.12}
    panel = tuple(CandidateEvidenceV1(j, probabilities[j], "f" * 64)
                  for j in context.legal_targets)
    return Phase2DecisionOpportunityV1(
        context, "player2", (2, 3), 0, 4 if phase == "speech" else 3,
        "c" * 64, "d" * 64, "e" * 64, panel,
        (Action.PUSH, Action.REDIRECT, Action.PROBE), "player6",
        "CURRENT_SUSPICION_BASIS", "player5",
        ObservationWindowV1("pre-1", "player5", phase, ("player2",)))


@pytest.mark.parametrize("phase", ["speech", "speech_pk"])
def test_whole_strategy_assignment_is_reproducible_and_balanced_by_contract(phase):
    opportunity = public_opportunity(phase=phase)
    observed = set()
    for seed in range(16):
        plan = Phase2OnlineProbePilotPlanV1("probe-study", seed, 4, 10)
        selection = select_probe_candidate(plan, opportunity.legal_context)
        assignment = assign_probe_strategy(plan, selection, opportunity)

        assert assign_probe_strategy(plan, selection, opportunity) == assignment
        assert assignment.treatment.assignment_probability == 0.5
        assert assignment.assignment_key != selection.candidate_selection_key
        assert assignment.treatment.randomization_key == assignment.assignment_key
        assert assignment.legal_action_set == (Action.PUSH, Action.REDIRECT, Action.PROBE)
        if assignment.strategy is ProbeStrategy.IMMEDIATE_REDIRECT:
            assert assignment.treatment.action is Action.REDIRECT
            assert assignment.treatment.plan.redirect_target == "player6"
        else:
            assert assignment.strategy is ProbeStrategy.PROBE_THEN_REDIRECT
            assert assignment.treatment.action is Action.PROBE
            assert assignment.treatment.plan.continuation_actor == "player5"
            assert assignment.treatment.plan.expected_observation_window.expected_speakers == ("player2",)
        observed.add(assignment.strategy)
    assert observed == {ProbeStrategy.IMMEDIATE_REDIRECT, ProbeStrategy.PROBE_THEN_REDIRECT}


@pytest.mark.parametrize("phase", ["speech", "speech_pk"])
def test_json_assignment_round_trip_reconstructs_both_whole_strategies(phase):
    opportunity = public_opportunity(phase=phase)
    for seed in range(16):
        plan = Phase2OnlineProbePilotPlanV1("probe-study", seed, 4, 10)
        selection = select_probe_candidate(plan, opportunity.legal_context)
        assignment = assign_probe_strategy(plan, selection, opportunity)
        record = json.loads(json.dumps(assignment.to_record()))

        restored = probe_assignment_from_record(record, plan)
        assert restored == assignment
        assert restored.digest() == assignment.digest()


@pytest.mark.parametrize(("path", "value"), [
    (("assignment_seed",), 99),
    (("assignment_key",), "0" * 64),
    (("assigned_strategy",), "UNKNOWN"),
    (("assigned_action",), "PUSH"),
    (("assignment_probability",), 1.0),
    (("continuation_schedule",), {"candidate_j": "player7"}),
    (("selection", "candidate_pool"), ["player2", "player6"]),
    (("selection", "candidate_j"), "player6"),
    (("selection", "candidate_selection_key"), "0" * 64),
    (("opportunity", "public_legal", "J"), ["player6", "player2", "player7"]),
    (("opportunity", "probe", "continuation_actor"), "player6"),
    (("opportunity", "evidence", "q_source_kind"), "OTHER"),
    (("treatment", "randomization_key"), "0" * 64),
    (("unexpected_field",), True),
])
def test_assignment_loader_rejects_tampering(path, value):
    plan = Phase2OnlineProbePilotPlanV1("probe-study", 11, 4, 10)
    opportunity = public_opportunity()
    selection = select_probe_candidate(plan, opportunity.legal_context)
    assignment = assign_probe_strategy(plan, selection, opportunity)
    record = deepcopy(assignment.to_record())
    parent = record
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value

    with pytest.raises(OnlinePilotPlanError):
        probe_assignment_from_record(record, plan)


def test_shared_game_state_persists_one_whole_probe_assignment_per_game():
    plan = Phase2OnlineProbePilotPlanV1("probe-study", 11, 4, 10)
    opportunity = public_opportunity()
    selection = select_probe_candidate(plan, opportunity.legal_context)
    state = OnlinePilotGameStateV1()
    persisted = []

    assignment = state.try_assign(plan, selection, opportunity, persist=persisted.append)
    assert assignment == assign_probe_strategy(plan, selection, opportunity)
    assert persisted == [assignment]
    assert state.try_assign(plan, selection, opportunity, persist=persisted.append) is None
    assert persisted == [assignment]


def test_failed_assignment_persistence_does_not_register_runtime_assignment():
    plan = Phase2OnlineProbePilotPlanV1("probe-study", 11, 4, 10)
    opportunity = public_opportunity()
    selection = select_probe_candidate(plan, opportunity.legal_context)
    state = OnlinePilotGameStateV1()

    def unavailable_ledger(_assignment):
        raise OSError("durable write unavailable")

    with pytest.raises(OSError, match="durable write unavailable"):
        state.try_assign(plan, selection, opportunity, persist=unavailable_ledger)
    assert state.assignments == {}


@pytest.mark.parametrize("phase", ["speech", "speech_pk"])
def test_singleton_probe_pool_is_legal_without_weakening_terminal_selection(phase):
    plan = Phase2OnlineProbePilotPlanV1("probe-study", 11, 4, 10)
    context = public_context(phase=phase, singleton=True)
    selection = select_probe_candidate(plan, context)
    assert selection.candidate_pool == ("player2",)
    assert selection.candidate_selection_probability == 1.0
    with pytest.raises(OnlinePilotPlanError):
        replace(selection, rule=SELECTION_RULE)

    terminal = Phase2OnlineTerminalPilotPlanV1("terminal-study", 11)
    assert select_candidate(terminal, context).candidate_pool == (
        ("player2", "player6", "player7") if phase == "speech" else ("player2", "player6"))
    assert terminal.target_assignment_count == 120


def test_joint_pool_rejects_missing_redirect_alternative_or_later_wolf():
    plan = Phase2OnlineProbePilotPlanV1("probe-study", 11, 4, 10)
    only_one_target = replace(public_context(phase="speech_pk"),
                              competition=("player1", "player2", "player5"),
                              public_speaker_queue=("player1", "player2", "player5"))
    last_wolf = replace(public_context(), acting_wolf="player5")
    for context in (only_one_target, last_wolf):
        assert probe_candidate_pool(context) == ()
        assert select_probe_candidate(plan, context) is None


def test_candidate_selection_and_strategy_draw_do_not_depend_on_score_panel():
    plan = Phase2OnlineProbePilotPlanV1("probe-study", 11, 4, 10)
    before = public_opportunity()
    after = public_opportunity(probabilities={"player2": 0.0, "player6": 1.0, "player7": 0.0})
    selection = select_probe_candidate(plan, before.legal_context)
    assert select_probe_candidate(plan, after.legal_context) == selection
    first = assign_probe_strategy(plan, selection, before)
    second = assign_probe_strategy(plan, selection, after)
    assert first.strategy is second.strategy
    assert first.assignment_key == second.assignment_key
    assert first.opportunity.digest() != second.opportunity.digest()


@pytest.mark.parametrize(("field", "value"), [
    ("assignment_seed", True), ("assignment_seed", -1),
    ("target_assignment_count", 0), ("target_assignment_count", True),
    ("max_games_attempted", None), ("max_games_attempted", 3),
    ("campaign_purpose", "formal"), ("candidate_selection_rule", SELECTION_RULE),
    ("selection_status", "DRAFT"),
])
def test_probe_plan_requires_explicit_valid_frozen_budgets(field, value):
    values = dict(pilot_id="probe-study", assignment_seed=11,
                  target_assignment_count=4, max_games_attempted=10)
    values[field] = value
    with pytest.raises(OnlinePilotPlanError):
        Phase2OnlineProbePilotPlanV1(**values)


def test_probe_plan_has_no_formal_sample_defaults_and_is_immutable():
    with pytest.raises(TypeError):
        Phase2OnlineProbePilotPlanV1("probe-study", 11)
    plan = Phase2OnlineProbePilotPlanV1("probe-study", 11, 17, 23, campaign_purpose="pilot")
    assert plan.to_record()["target_assignment_count"] == 17
    assert plan.to_record()["max_assignments_per_game"] == 1
    assert plan.to_record()["immediate_redirect_probability"] == 0.5
    assert plan.to_record()["probe_then_redirect_probability"] == 0.5
    with pytest.raises(FrozenInstanceError):
        plan.assignment_seed = 12


def test_record_cannot_be_replayed_under_a_different_frozen_plan():
    plan = Phase2OnlineProbePilotPlanV1("probe-study", 11, 4, 10)
    opportunity = public_opportunity()
    selection = select_probe_candidate(plan, opportunity.legal_context)
    record = assign_probe_strategy(plan, selection, opportunity).to_record()
    with pytest.raises(OnlinePilotPlanError):
        probe_assignment_from_record(record, replace(plan, pilot_id="another-study"))


def test_shared_state_stops_after_explicit_probe_assignment_target():
    plan = Phase2OnlineProbePilotPlanV1("probe-study", 11, 1, 2)
    state = OnlinePilotGameStateV1()
    first = public_opportunity(game_id="game-1")
    second = public_opportunity(game_id="game-2")
    assert state.try_assign(plan, select_probe_candidate(plan, first.legal_context), first) is not None
    assert state.try_assign(plan, select_probe_candidate(plan, second.legal_context), second) is None


def test_strict_opportunity_loader_supports_original_j_at_later_teammate_pre():
    initial = public_opportunity()
    context = replace(initial.legal_context, acting_wolf="player5", boundary_id="pre-3",
                      prefix_digest="1" * 64, public_history_digest="2" * 64)
    later = replace(initial, legal_context=context, current_position=2, remaining_speakers=2,
                    legal_actions=(Action.PUSH, Action.REDIRECT), probe_request_type=None,
                    continuation_actor=None, observation_window=None)
    assert probe_opportunity_from_record(later.to_record()) == later
    altered = deepcopy(later.to_record())
    altered["redirect_target"] = "player7"
    with pytest.raises(OnlinePilotPlanError):
        probe_opportunity_from_record(altered)


def test_probe_selection_cannot_be_labeled_as_a_terminal_assignment():
    plan = Phase2OnlineProbePilotPlanV1("probe-study", 11, 4, 10)
    context = replace(public_context(phase="speech_pk"),
                      competition=("player1", "player2", "player3", "player5"),
                      public_speaker_queue=("player1", "player2", "player3", "player5"))
    selection = select_probe_candidate(plan, context)
    assert selection.candidate_pool == context.legal_targets
    j = selection.candidate_j
    opportunity = Phase2DecisionOpportunityV1(
        context, j, (2, 5), 0, 3, "c" * 64, "d" * 64, "e" * 64,
        (CandidateEvidenceV1("player2", 0.9, "f" * 64),
         CandidateEvidenceV1("player3", 0.2, "f" * 64)),
        (Action.PUSH, Action.REDIRECT, Action.PROBE), "player3" if j == "player2" else "player2",
        "CURRENT_SUSPICION_BASIS", "player5",
        ObservationWindowV1("pre-1", "player5", "speech_pk", ("player2", "player3")))
    treatment = build_phase2_treatment(
        opportunity, Action.REDIRECT, assignment_source="randomized_pilot",
        assignment_probability=0.5, randomization_key="c" * 64)
    with pytest.raises(OnlinePilotPlanError):
        OnlineTerminalAssignmentV1(selection, opportunity, treatment, 11,
                                   "c" * 64, opportunity.legal_actions)
