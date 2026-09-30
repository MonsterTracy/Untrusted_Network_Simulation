"""Durable randomized assignment and restart behavior without model calls."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.phase2.test_decision_opportunity import Mapper, opportunity, q_matrix
from werewolf.phase2_decision_opportunity import build_phase2_decision_opportunity
from werewolf.phase2_online_ledger import (
    OnlinePilotAssignmentLedgerV1, OnlinePilotLedgerError,
)
from werewolf.phase2_online_plan import (
    FORMAL_PILOT, QUALIFICATION, OnlinePilotGameStateV1,
    Phase2OnlineTerminalPilotPlanV1, assign_terminal_action, select_candidate,
)
from werewolf.phase2_online_records import (
    Phase2OnlineInterventionRecordV1, record_online_execution,
)
from werewolf.phase2_pilot_dataset import (
    PilotDatasetError, analyze_phase2_online_support_record,
    build_phase2_online_dataset_from_ledger, require_online_dataset_access,
)
from werewolf.artifact_io import canonical_json_bytes, sha256_bytes


def _ledger(tmp_path, *, purpose=QUALIFICATION, cap=None):
    plan = Phase2OnlineTerminalPilotPlanV1(
        "study", 17, campaign_purpose=purpose, max_games_attempted=cap)
    return OnlinePilotAssignmentLedgerV1(
        tmp_path / "assignment-ledger.jsonl", plan=plan, source_commit="a" * 40)


def _game_assignment(plan, game_id):
    original = opportunity()
    context = replace(original.legal_context, game_id=game_id,
                      boundary_id=f"{game_id}-pre")
    selection = select_candidate(plan, context)
    chosen = build_phase2_decision_opportunity(
        context, selection.candidate_j, q_matrix(), Mapper(),
        q_source_digest=original.q_source_digest)
    return assign_terminal_action(plan, selection, chosen)


def test_write_ahead_exactly_once_and_interrupted_resume(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.start_game("g1")
    assignment = _game_assignment(ledger.plan, "g1")
    assert ledger.persist_assignment(assignment) is True
    assert ledger.persist_assignment(assignment) is False
    assert sum(row["kind"] == "ASSIGNMENT" for row in ledger.snapshot()["events"]) == 1
    with pytest.raises(OnlinePilotLedgerError, match="conflicting"):
        ledger._append("ASSIGNMENT", {"game_id": "g1", "assignment_id": "wrong"})
    resumed = OnlinePilotAssignmentLedgerV1(
        ledger.path, plan=ledger.plan, source_commit="a" * 40)
    assert resumed.mark_interrupted_on_resume() == ("g1",)
    assert resumed.mark_interrupted_on_resume() == ()
    assert resumed.snapshot()["games"]["g1"]["ASSIGNMENT"]["assignment_id"] == assignment.digest()
    with pytest.raises(OnlinePilotLedgerError, match="cannot replay"):
        resumed.start_game("g1")
    assert resumed.status() == "RUNNING"


def test_persistence_failure_prevents_in_memory_assignment_and_treatment(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.start_game("g1")
    assignment = _game_assignment(ledger.plan, "g1")
    state = OnlinePilotGameStateV1()
    def fail(_assignment):
        raise OSError("simulated fsync failure")
    with pytest.raises(OSError, match="fsync"):
        state.try_assign(ledger.plan, assignment.selection,
                         assignment.opportunity, persist=fail)
    assert not state.assignments
    assert "ASSIGNMENT" not in ledger.snapshot()["games"]["g1"]


def test_qualification_target_and_safety_cap_are_assignment_based(tmp_path):
    ledger = _ledger(tmp_path, cap=10)
    for index in range(10):
        game_id = f"g{index}"
        ledger.start_game(game_id)
        ledger.persist_assignment(_game_assignment(ledger.plan, game_id))
    assert ledger.plan.target_assignment_count == 10
    assert ledger.status() == "READY_TO_SEAL"
    with pytest.raises(OnlinePilotLedgerError, match="max_games_attempted"):
        ledger.start_game("g11")
    assert ledger.snapshot()["assignment_count"] == 10
    incomplete = _ledger(tmp_path / "other", cap=10)
    for index in range(10):
        incomplete.start_game(f"g{index}")
    assert incomplete.status() == "INCOMPLETE"
    assert incomplete.snapshot()["assignment_count"] == 0


def test_formal_target_is_exactly_120_assignments_even_without_execution():
    plan = Phase2OnlineTerminalPilotPlanV1("formal", 17, campaign_purpose=FORMAL_PILOT)
    state = OnlinePilotGameStateV1()
    for index in range(120):
        assignment = _game_assignment(plan, f"g{index}")
        assert state.try_assign(plan, assignment.selection, assignment.opportunity) is not None
    extra = _game_assignment(plan, "g120")
    assert state.try_assign(plan, extra.selection, extra.opportunity) is None
    assert len(state.assignments) == 120
    assert plan.target_assignment_count == 120


def test_ledger_recovers_assignment_without_execution_and_excludes_qualification(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.start_game("g1")
    ledger.persist_assignment(_game_assignment(ledger.plan, "g1"))
    ledger.mark_interrupted_on_resume()
    dataset = build_phase2_online_dataset_from_ledger(ledger, allow_synthetic=True)
    assert len(dataset.assignments) == 1
    assert dataset.executions == dataset.consequences == ()
    assert dataset.manifest["missing_execution_count"] == 1
    assert dataset.manifest["estimator_eligible"] is False
    support = analyze_phase2_online_support_record(dataset.to_record())
    assert support["execution_missing"] == support["consequence_missing"] == 1
    assert support["candidate_pool_sizes"]


def test_estimation_access_rejects_qualification_mixed_and_synthetic(tmp_path):
    qualification = _ledger(tmp_path / "qualification")
    qualification.start_game("q0")
    qualification.persist_assignment(_game_assignment(qualification.plan, "q0"))
    qualification.mark_interrupted_on_resume()
    q_data = build_phase2_online_dataset_from_ledger(
        qualification, execution_verifier=lambda *_: True)
    with pytest.raises(PilotDatasetError, match="qualification"):
        require_online_dataset_access((q_data,))
    audit = require_online_dataset_access((q_data,), audit_only=True)
    assert audit.datasets == (q_data,) and audit.estimator_eligible is False

    formal = _ledger(tmp_path / "formal", purpose=FORMAL_PILOT)
    for index in range(120):
        game_id = f"f{index}"
        formal.start_game(game_id)
        formal.persist_assignment(_game_assignment(formal.plan, game_id))
    formal.mark_interrupted_on_resume()
    p_data = build_phase2_online_dataset_from_ledger(
        formal, execution_verifier=lambda *_: True)
    assert require_online_dataset_access((p_data,)).estimator_eligible is True
    assert require_online_dataset_access((p_data,), audit_only=True).estimator_eligible is False
    with pytest.raises(PilotDatasetError, match="one formal"):
        require_online_dataset_access((q_data, p_data))
    synthetic = build_phase2_online_dataset_from_ledger(formal, allow_synthetic=True)
    with pytest.raises(PilotDatasetError, match="qualification or incomplete"):
        require_online_dataset_access((synthetic,))


def test_resume_preserves_executed_assignment_with_missing_consequence(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.start_game("g1")
    assignment = _game_assignment(ledger.plan, "g1")
    ledger.persist_assignment(assignment)
    link = SimpleNamespace(canonical_event_id="event-000004",
                           canonical_event_digest="a" * 64,
                           public_text_digest="b" * 64)
    executed = record_online_execution(
        Phase2OnlineInterventionRecordV1(assignment),
        language_audit_digest="c" * 64, attempt_count=1,
        backend_calls=(), canonical_link=link)
    ledger.persist_execution(executed)
    resumed = OnlinePilotAssignmentLedgerV1(
        ledger.path, plan=ledger.plan, source_commit="a" * 40)
    assert resumed.mark_interrupted_on_resume() == ("g1",)
    stage = resumed.snapshot()["games"]["g1"]
    assert stage["INTERRUPTED"]["reason"] == "EXECUTION_WITHOUT_CONSEQUENCE"
    assert stage["ASSIGNMENT"]["assignment_id"] == assignment.digest()
    with pytest.raises(OnlinePilotLedgerError, match="cannot replay"):
        resumed.start_game("g1")


def test_ledger_rejects_tampering_and_wrong_plan(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.start_game("g1")
    with pytest.raises(OnlinePilotLedgerError, match="another campaign"):
        OnlinePilotAssignmentLedgerV1(
            ledger.path, plan=replace(ledger.plan, assignment_seed=18),
            source_commit="a" * 40)
    with ledger.path.open("ab") as stream:
        stream.write(b'{"invalid":true}\n')
    with pytest.raises(OnlinePilotLedgerError, match="hash chain"):
        ledger.snapshot()


def test_recovery_recomputes_draws_even_if_hash_chain_is_rewritten(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.start_game("g1")
    ledger.persist_assignment(_game_assignment(ledger.plan, "g1"))
    rows = list(ledger.snapshot()["events"])
    selected = rows[-1]["payload"]["assignment"]["selection"]
    selected["candidate_j"] = next(seat for seat in selected["candidate_pool"]
                                   if seat != selected["candidate_j"])
    rows[-1]["payload"]["assignment_id"] = sha256_bytes(canonical_json_bytes(
        rows[-1]["payload"]["assignment"]))
    rows[-1]["digest"] = sha256_bytes(canonical_json_bytes({
        key: value for key, value in rows[-1].items() if key != "digest"}))
    ledger.path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
    with pytest.raises(OnlinePilotLedgerError, match="randomization/provenance"):
        ledger.snapshot()
