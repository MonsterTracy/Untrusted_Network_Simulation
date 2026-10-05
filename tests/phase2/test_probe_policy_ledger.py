"""Public ledger contract tests with literal, independently declared stage traces."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json

import pytest

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.phase2_actions import Action, ActionContextV1, ObservationWindowV1
from werewolf.phase2_decision_opportunity import CandidateEvidenceV1, Phase2DecisionOpportunityV1
from werewolf.phase2_online_ledger import (
    OnlinePilotAssignmentLedgerV1, OnlinePilotLedgerError, PROBE_LEDGER_VERSION,
)
from werewolf.phase2_online_plan import (
    Phase2OnlineProbePilotPlanV1, ProbeStrategy, assign_probe_strategy,
    select_probe_candidate,
)


SOURCE_COMMIT = "9" * 40
EMPTY_CALL_DIGEST = hashlib.sha256(b"[]").hexdigest()


def public_opportunity():
    alive = ("player1", "player2", "player5", "player6", "player7")
    context = ActionContextV1(
        "game-1", "pre-1", "a" * 64, "b" * 64, "speech", "player1",
        alive, frozenset(("player1", "player5")), alive, alive,
    )
    panel = (
        CandidateEvidenceV1("player2", .99, "f" * 64),
        CandidateEvidenceV1("player6", .88, "f" * 64),
        CandidateEvidenceV1("player7", .12, "f" * 64),
    )
    return Phase2DecisionOpportunityV1(
        context, "player2", (2, 3), 0, 4, "c" * 64, "d" * 64, "e" * 64,
        panel, (Action.PUSH, Action.REDIRECT, Action.PROBE), "player6",
        "CURRENT_SUSPICION_BASIS", "player5",
        ObservationWindowV1("pre-1", "player5", "speech", ("player2",)),
    )


def frozen_assignment(strategy=ProbeStrategy.PROBE_THEN_REDIRECT, *, count=1, cap=2):
    opportunity = public_opportunity()
    for seed in range(100):
        plan = Phase2OnlineProbePilotPlanV1("probe-ledger", seed, count, cap)
        selection = select_probe_candidate(plan, opportunity.legal_context)
        assignment = assign_probe_strategy(plan, selection, opportunity)
        if assignment.strategy == strategy:
            return plan, assignment
    raise AssertionError("literal fixture needs a seed for each strategy")


@dataclass
class LiteralRecord:
    """Exercise the public writer without using the production lifecycle builder."""

    assignment: object
    raw: dict

    @property
    def game_id(self):
        return self.assignment.opportunity.legal_context.game_id

    @property
    def backend_calls(self):
        return ()

    @property
    def day_outcome(self):
        return self.raw["day_consequence"]

    @property
    def final_game_result(self):
        return self.raw["offline_audit"]["final_game_result"]

    @property
    def theta_audit_label(self):
        return None

    def to_record(self):
        return deepcopy(self.raw)


def literal_snapshots(assignment, *, invalid=True):
    raw = {
        "schema_version": "phase2_online_probe_record_v1",
        "assignment": assignment.to_record(),
        "lifecycle": ["ASSIGNED"],
        "stages": {"T1": None, "T3": None},
        "continuation": {"opportunity": None, "treatment": None},
        "execution": {
            "success": None, "failure_reason": None, "language_attempt_count": None,
            "language_audit_digest": None, "backend_call_audit_digest": EMPTY_CALL_DIGEST,
            "canonical_commit_success": None, "canonical_event_id": None,
            "canonical_event_digest": None, "generated_text_digest": None,
        },
        "observations": [], "structural_failure_reason": None,
        "day_consequence": None,
        "offline_audit": {"theta_ac": None, "final_game_result": None},
    }
    probe = assignment.strategy == ProbeStrategy.PROBE_THEN_REDIRECT
    assert invalid or not probe  # successful two-stage traces live in runner tests
    snapshots = [deepcopy(raw)]
    raw["lifecycle"].append("T1_ATTEMPTED")
    snapshots.append(deepcopy(raw))
    raw["lifecycle"].append("T1_LANGUAGE_INVALID" if invalid else "T1_COMMITTED")
    raw["stages"]["T1"] = {
        "opportunity": assignment.opportunity.to_record(),
        "treatment": assignment.treatment.to_record(),
        "language_audit_digest": "1" * 64, "attempt_count": 2,
        "backend_calls": [], "success": not invalid,
        "failure_reason": "LANGUAGE_INVALID" if invalid else None,
        "canonical_event_id": None if invalid else "speech-1",
        "canonical_event_digest": None if invalid else "2" * 64,
        "generated_text_digest": None if invalid else "3" * 64,
    }
    snapshots.append(deepcopy(raw))
    if probe:
        raw["lifecycle"].append("T3_CANCELLED")
        snapshots.append(deepcopy(raw))
    raw["lifecycle"].append("EXECUTION_RECORDED")
    raw["execution"].update({
        "success": not invalid,
        "failure_reason": "LANGUAGE_INVALID" if invalid else None,
        "language_attempt_count": 2, "language_audit_digest": "1" * 64,
        "canonical_commit_success": not invalid,
        "canonical_event_id": None if invalid else "speech-1",
        "canonical_event_digest": None if invalid else "2" * 64,
        "generated_text_digest": None if invalid else "3" * 64,
    })
    snapshots.append(deepcopy(raw))
    raw["lifecycle"].append("DAY_CONSEQUENCE_RECORDED")
    raw["day_consequence"] = {
        "exiled_player": "player2", "Y": "j_exiled", "s_plus": [2, 2],
        "v_ref": .6, "l_ref": .4, "reference_artifact_digest": "4" * 64,
    }
    snapshots.append(deepcopy(raw))
    raw["lifecycle"].append("GAME_RESULT_RECORDED")
    raw["offline_audit"]["final_game_result"] = "Werewolf"
    snapshots.append(deepcopy(raw))
    return snapshots


def started_ledger(tmp_path, strategy=ProbeStrategy.PROBE_THEN_REDIRECT, *, count=1, cap=2):
    plan, assignment = frozen_assignment(strategy, count=count, cap=cap)
    ledger = OnlinePilotAssignmentLedgerV1(
        tmp_path / "events.jsonl", plan=plan, source_commit=SOURCE_COMMIT,
    )
    ledger.start_game("game-1")
    ledger.persist_assignment(assignment)
    return ledger, assignment


def persist_through_execution(ledger, assignment, snapshots):
    for raw in snapshots:
        if raw["lifecycle"][-1] in ("DAY_CONSEQUENCE_RECORDED", "GAME_RESULT_RECORDED"):
            break
        ledger.persist_strategy_stage(LiteralRecord(assignment, raw))
    completed = next(raw for raw in snapshots if raw["lifecycle"][-1] == "EXECUTION_RECORDED")
    ledger.persist_execution(LiteralRecord(assignment, completed))


def call_payload(ledger, assignment, *, sequence=1, treatment_id=None):
    treatment = assignment.treatment.to_record()
    return {
        "game_id": "game-1", "assignment_id": assignment.digest(),
        "call": {
            "sequence": sequence, "pilot_id": ledger.plan.pilot_id, "game_id": "game-1",
            "assignment_id": assignment.digest(),
            "treatment_id": treatment_id or treatment["treatment_id"],
            "opportunity_digest": treatment["opportunity_digest"],
        },
    }


def test_probe_header_single_candidate_and_recovery(tmp_path):
    ledger, assignment = started_ledger(tmp_path)
    snapshot = ledger.snapshot()
    assert snapshot["events"][0]["payload"]["schema_version"] == PROBE_LEDGER_VERSION
    assert assignment.selection.candidate_pool == ("player2",)
    assert snapshot["assignment_count"] == 1
    recovered = OnlinePilotAssignmentLedgerV1(
        ledger.path, plan=ledger.plan, source_commit=SOURCE_COMMIT,
    )
    assert recovered.snapshot() == snapshot
    assert recovered.status() == "READY_TO_SEAL"
    assert not recovered.sealable()


def test_recovery_recomputes_assignment_after_hash_chain_rewrite(tmp_path):
    ledger, _ = started_ledger(tmp_path)
    rows = [json.loads(line) for line in ledger.path.read_text().splitlines()]
    rows[-1]["payload"]["assignment"]["assignment_seed"] += 1
    previous = None
    for row in rows:
        row["previous_digest"] = previous
        row["digest"] = sha256_bytes(canonical_json_bytes({
            key: value for key, value in row.items() if key != "digest"
        }))
        previous = row["digest"]
    ledger.path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
    with pytest.raises(OnlinePilotLedgerError, match="malformed"):
        OnlinePilotAssignmentLedgerV1(ledger.path, plan=ledger.plan, source_commit=SOURCE_COMMIT)


def test_literal_lifecycle_is_required_idempotent_and_immutable(tmp_path):
    ledger, assignment = started_ledger(tmp_path)
    snapshots = literal_snapshots(assignment)
    with pytest.raises(OnlinePilotLedgerError, match="initial"):
        ledger.persist_strategy_stage(LiteralRecord(assignment, snapshots[1]))
    record = LiteralRecord(assignment, snapshots[0])
    assert ledger.persist_strategy_stage(record)
    before = ledger.path.read_bytes()
    assert not ledger.persist_strategy_stage(record)
    assert ledger.path.read_bytes() == before
    changed = deepcopy(snapshots[0])
    changed["observations"] = [{
        "event_type": "public_speech", "event_id": "speech-j", "speaker": "player2",
        "text": "My current suspicion is player6.",
    }]
    with pytest.raises(OnlinePilotLedgerError, match="conflicting"):
        ledger.persist_strategy_stage(LiteralRecord(assignment, changed))
    assert ledger.path.read_bytes() == before
    with pytest.raises(OnlinePilotLedgerError, match="exactly one"):
        ledger.persist_strategy_stage(LiteralRecord(assignment, snapshots[2]))


def test_backend_calls_need_persisted_stage_plan_and_continuous_sequence(tmp_path):
    ledger, assignment = started_ledger(tmp_path)
    call = call_payload(ledger, assignment)
    with pytest.raises(OnlinePilotLedgerError, match="writeahead"):
        ledger._append("BACKEND_CALL", call)
    snapshots = literal_snapshots(assignment)
    ledger.persist_strategy_stage(LiteralRecord(assignment, snapshots[0]))
    with pytest.raises(OnlinePilotLedgerError, match="outside prepared"):
        ledger._append("BACKEND_CALL", call)
    ledger.persist_strategy_stage(LiteralRecord(assignment, snapshots[1]))
    with pytest.raises(OnlinePilotLedgerError, match="plan binding"):
        ledger._append("BACKEND_CALL", call_payload(ledger, assignment, treatment_id="unplanned-T3"))
    with pytest.raises(OnlinePilotLedgerError, match="sequence"):
        ledger._append("BACKEND_CALL", call_payload(ledger, assignment, sequence=2))
    assert ledger._append("BACKEND_CALL", call)
    assert not ledger._append("BACKEND_CALL", call)
    assert ledger._append("BACKEND_CALL", call_payload(ledger, assignment, sequence=2))
    assert [row["call"]["sequence"] for row in ledger.snapshot()["games"]["game-1"]["BACKEND_CALL"]] == [1, 2]


def test_execution_cannot_precede_final_lifecycle_or_drop_recorded_calls(tmp_path):
    ledger, assignment = started_ledger(tmp_path)
    snapshots = literal_snapshots(assignment)
    ledger.persist_strategy_stage(LiteralRecord(assignment, snapshots[0]))
    with pytest.raises(OnlinePilotLedgerError, match="lifecycle/call"):
        ledger.persist_execution(LiteralRecord(assignment, snapshots[0]))
    ledger.persist_strategy_stage(LiteralRecord(assignment, snapshots[1]))
    ledger._append("BACKEND_CALL", call_payload(ledger, assignment))
    for raw in snapshots[2:-2]:
        ledger.persist_strategy_stage(LiteralRecord(assignment, raw))
    with pytest.raises(OnlinePilotLedgerError, match="lifecycle/call"):
        ledger.persist_execution(LiteralRecord(assignment, snapshots[-3]))


@pytest.mark.parametrize("strategy", list(ProbeStrategy))
def test_failed_language_requires_actual_day_and_keeps_final_game_result_secondary(tmp_path, strategy):
    ledger, assignment = started_ledger(tmp_path, strategy)
    snapshots = literal_snapshots(assignment)
    persist_through_execution(ledger, assignment, snapshots)
    stage = ledger.snapshot()["games"]["game-1"]
    assert stage["EXECUTION"]["execution"]["success"] is False
    expected = ["ASSIGNED", "T1_ATTEMPTED", "T1_LANGUAGE_INVALID"]
    if strategy == ProbeStrategy.PROBE_THEN_REDIRECT:
        expected.append("T3_CANCELLED")
    expected.append("EXECUTION_RECORDED")
    assert stage["STRATEGY_STAGE"]["record"]["lifecycle"] == expected
    assert stage["STRATEGY_STAGE_HISTORY"][-1] == stage["STRATEGY_STAGE"]
    assert ledger.status() == "READY_TO_SEAL"
    assert not ledger.sealable()
    day = LiteralRecord(assignment, snapshots[-2])
    ledger.persist_strategy_stage(day)
    ledger.persist_consequence(day)
    assert ledger.sealable()
    final = LiteralRecord(assignment, snapshots[-1])
    ledger.persist_strategy_stage(final)
    with pytest.raises(OnlinePilotLedgerError, match="final game result"):
        ledger.persist_game_result("game-1", "Villager")
    ledger.persist_game_result("game-1", "Werewolf")
    assert ledger.sealable()
    assert ledger.mark_interrupted_on_resume() == ()


@pytest.mark.parametrize("with_day", [False, True])
def test_resume_distinguishes_missing_primary_outcome_from_secondary_tail(tmp_path, with_day):
    ledger, assignment = started_ledger(tmp_path)
    snapshots = literal_snapshots(assignment)
    persist_through_execution(ledger, assignment, snapshots)
    if with_day:
        day = LiteralRecord(assignment, snapshots[-2])
        ledger.persist_strategy_stage(day)
        ledger.persist_consequence(day)
    assert ledger.mark_interrupted_on_resume() == ("game-1",)
    assert ledger.mark_interrupted_on_resume() == ()
    marker = "POST_ENDPOINT_TAIL_INTERRUPTED" if with_day else "INTERRUPTED"
    reason = ledger.snapshot()["games"]["game-1"][marker]["reason"]
    assert reason == ("CONSEQUENCE_WITHOUT_GAME_RESULT" if with_day else "EXECUTION_WITHOUT_CONSEQUENCE")
    assert ledger.status() == ("READY_TO_SEAL" if with_day else "INCOMPLETE")
    assert ledger.sealable() is with_day
    with pytest.raises(OnlinePilotLedgerError, match="replay"):
        ledger.start_game("game-1")
    with pytest.raises(OnlinePilotLedgerError, match="closed post-endpoint|interrupted"):
        ledger._append("BACKEND_CALL", call_payload(ledger, assignment))


def test_cap_does_not_autosize_incomplete_campaign(tmp_path):
    ledger, _ = started_ledger(tmp_path, count=2, cap=2)
    ledger.start_game("game-2")
    assert ledger.status() == "INCOMPLETE"
    assert ledger.plan.target_assignment_count == 2
    assert not ledger.sealable()
    with pytest.raises(OnlinePilotLedgerError, match="max_games_attempted"):
        ledger.start_game("game-3")


def test_later_snapshot_cannot_rewrite_committed_stage(tmp_path):
    ledger, assignment = started_ledger(tmp_path, ProbeStrategy.IMMEDIATE_REDIRECT)
    snapshots = literal_snapshots(assignment, invalid=False)
    for raw in snapshots[:3]:
        ledger.persist_strategy_stage(LiteralRecord(assignment, raw))
    rewritten = deepcopy(snapshots[3])
    rewritten["stages"]["T1"]["canonical_event_id"] = "replacement"
    rewritten["execution"]["canonical_event_id"] = "replacement"
    with pytest.raises(OnlinePilotLedgerError, match="evidence changed"):
        ledger.persist_strategy_stage(LiteralRecord(assignment, rewritten))


def test_structural_failure_closes_backend_access_and_cannot_seal(tmp_path):
    ledger, assignment = started_ledger(tmp_path)
    initial = literal_snapshots(assignment)[0]
    ledger.persist_strategy_stage(LiteralRecord(assignment, initial))
    failed = deepcopy(initial)
    failed["lifecycle"].append("STRUCTURAL_FAILURE")
    failed["structural_failure_reason"] = "CANONICAL_COMMIT_UNCERTAIN"
    ledger.persist_strategy_stage(LiteralRecord(assignment, failed))
    with pytest.raises(OnlinePilotLedgerError, match="closed probe"):
        ledger._append("BACKEND_CALL", call_payload(ledger, assignment))
    assert not ledger.sealable()
    assert ledger.mark_interrupted_on_resume() == ("game-1",)
    assert ledger.status() == "INCOMPLETE"


def test_consequence_binds_original_assignment_and_actual_day_data(tmp_path):
    ledger, assignment = started_ledger(tmp_path)
    snapshots = literal_snapshots(assignment)
    persist_through_execution(ledger, assignment, snapshots)
    day = LiteralRecord(assignment, snapshots[-2])
    ledger.persist_strategy_stage(day)
    payload = {
        "game_id": "game-1", "assignment_id": assignment.digest(),
        "day_consequence": deepcopy(day.raw["day_consequence"]),
        "final_game_result": None, "theta_audit_label": None,
    }
    wrong_assignment = deepcopy(payload)
    wrong_assignment["assignment_id"] = "0" * 64
    with pytest.raises(OnlinePilotLedgerError, match="consequence differs"):
        ledger._append("CONSEQUENCE", wrong_assignment)
    wrong_day = deepcopy(payload)
    wrong_day["day_consequence"]["l_ref"] = 0
    with pytest.raises(OnlinePilotLedgerError, match="consequence differs"):
        ledger._append("CONSEQUENCE", wrong_day)
    assert ledger._append("CONSEQUENCE", payload)
