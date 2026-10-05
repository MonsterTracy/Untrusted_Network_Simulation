"""Crash-prefix recovery on the production runner's real persistent journal.

No simulator checkpoint exists. Assigned games cannot resume an unfinished
stage; completed journals can resume sealing without another controlled act.
"""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.phase2.test_probe_policy_runner import (
    canonical_pre, fixture, next_pre, observe, pre, runtime,
)
from werewolf.phase2_online_ledger import (
    OnlinePilotAssignmentLedgerV1, OnlinePilotLedgerError,
)
from werewolf.phase2_online_plan import Phase2OnlineProbePilotPlanV1, ProbeStrategy


SOURCE_COMMIT = "a" * 40


def completed_probe_trace(runtime, monkeypatch, tmp_path, *, target_assignment_count=1):
    """Produce a real journal once, then use its intact crash prefixes."""
    import tests.phase2.test_probe_policy_runner as fixtures

    # Freeze the larger target before the journal's START and randomization.
    with monkeypatch.context() as plan_patch:
        plan_patch.setattr(fixtures, "Phase2OnlineProbePilotPlanV1", lambda pilot, seed, target, cap:
            Phase2OnlineProbePilotPlanV1(pilot, seed, target_assignment_count, cap))
        f = fixture(runtime, monkeypatch, tmp_path, ProbeStrategy.PROBE_THEN_REDIRECT)
    assert pre(f) is not None
    observe(f)
    next_pre(f)
    assert pre(f) is not None
    f.env.phase = "night"
    f.env.public_events.append({
        "event_type": "exile_result", "exiled_players": ["player2"],
    })
    f.pilot.after_step(env=f.env, done=False, info={})
    f.pilot.after_game(env=f.env, winner="Werewolf")
    assert f.ledger.sealable() is (target_assignment_count == 1)
    return f


def crash_prefix(f, tmp_path, kind, lifecycle=None):
    lines = f.ledger.path.read_bytes().splitlines(keepends=True)
    for number, line in enumerate(lines):
        row = json.loads(line)
        if row["kind"] == kind and (
                lifecycle is None
                or row["payload"]["record"]["lifecycle"][-1] == lifecycle):
            break
    else:
        raise AssertionError(f"trace does not contain {kind}/{lifecycle}")
    path = tmp_path / "crash-prefix.jsonl"
    original = b"".join(lines[:number + 1])
    path.write_bytes(original)
    return OnlinePilotAssignmentLedgerV1(
        path, plan=f.pilot.plan, source_commit=SOURCE_COMMIT,
    ), original


def restarted_runner(runtime, f, ledger):
    original = f.pilot
    return runtime.OnlineTerminalPilotRunnerV1(
        plan=original.plan, predictor=original.predictor, mapper=original.mapper,
        backend=original.backend, model_name=original.model_name,
        reference_tables=original.reference_tables,
        reference_artifact_digest=original.reference_artifact_digest,
        ledger=ledger,
    )


@pytest.mark.parametrize("kind,lifecycle,reason", [
    ("ASSIGNMENT", None, "ASSIGNMENT_WITHOUT_EXECUTION"),
    ("STRATEGY_STAGE", "T1_COMMITTED", "ASSIGNMENT_WITHOUT_EXECUTION"),
    ("STRATEGY_STAGE", "T3_SCHEDULED", "ASSIGNMENT_WITHOUT_EXECUTION"),
    ("STRATEGY_STAGE", "T3_PREPARED", "ASSIGNMENT_WITHOUT_EXECUTION"),
    ("STRATEGY_STAGE", "T3_COMMITTED", "ASSIGNMENT_WITHOUT_EXECUTION"),
    ("EXECUTION", None, "EXECUTION_WITHOUT_CONSEQUENCE"),
    ("STRATEGY_STAGE", "DAY_CONSEQUENCE_RECORDED", "EXECUTION_WITHOUT_CONSEQUENCE"),
])
def test_assigned_crash_prefix_is_retained_and_never_replayed(
        runtime, monkeypatch, tmp_path, kind, lifecycle, reason):
    f = completed_probe_trace(runtime, monkeypatch, tmp_path / "complete",
                              target_assignment_count=2)
    ledger, original = crash_prefix(f, tmp_path, kind, lifecycle)
    before = ledger.snapshot()
    assignment = before["games"]["game-1"]["ASSIGNMENT"]
    calls = (tuple(f.q_calls), tuple(f.executions))

    def forbidden_selection(*args, **kwargs):
        raise AssertionError("resume must not draw a new candidate or strategy")

    monkeypatch.setattr(runtime, "select_probe_candidate", forbidden_selection)
    resumed = restarted_runner(runtime, f, ledger)
    after = ledger.snapshot()

    assert ledger.path.read_bytes().startswith(original)
    assert len(after["events"]) == len(before["events"]) + 1
    assert after["events"][-1]["kind"] == "INTERRUPTED"
    assert after["games"]["game-1"]["INTERRUPTED"]["reason"] == reason
    assert after["assignment_count"] == before["assignment_count"] == 1
    assert after["games_attempted"] == before["games_attempted"] == 1
    assert after["games"]["game-1"]["ASSIGNMENT"] == assignment
    assert resumed.state.assignments == {"game-1": assignment}
    assert resumed.state.games_seen == {"game-1"}
    for stage, payload in before["games"]["game-1"].items():
        assert after["games"]["game-1"][stage] == payload
    assert assignment["assignment"]["opportunity"]["identity"]["boundary_id"] == "pre-1"
    assert assignment["assignment"]["opportunity"]["identity"]["candidate_j"] == "player2"
    assert ledger.status() == "INCOMPLETE"
    assert not ledger.sealable()

    # Recovery is idempotent and cannot discard the failed assignment by
    # advancing the campaign or turn the old T3 into a fresh opportunity.
    frozen_bytes = ledger.path.read_bytes()
    assert ledger.mark_interrupted_on_resume() == ()
    with pytest.raises(OnlinePilotLedgerError, match="replay"):
        ledger.start_game("game-1")
    for game_id in ("game-1", "replacement-game"):
        with pytest.raises(ValueError, match="target or safety cap"):
            resumed.start_game(game_id)
    with pytest.raises(ValueError, match="game/recorder identity"):
        resumed.handle_pre(env=f.env, recorder=f.recorder,
                           call_audit=f.call_audit, observation=f.observation,
                           handoff=f.handoff)
    assert ledger.path.read_bytes() == frozen_bytes
    assert (tuple(f.q_calls), tuple(f.executions)) == calls


@pytest.mark.parametrize("target_assignment_count", [1, 2])
def test_endpoint_tail_crash_preserves_primary_row_and_never_replays_old_game(
        runtime, monkeypatch, tmp_path, target_assignment_count):
    f = completed_probe_trace(runtime, monkeypatch, tmp_path / "complete",
                              target_assignment_count=target_assignment_count)
    ledger, original = crash_prefix(f, tmp_path, "CONSEQUENCE")
    before = ledger.snapshot()
    old_stage = before["games"]["game-1"]
    assert old_stage["STRATEGY_STAGE"]["record"]["lifecycle"][-1] == "DAY_CONSEQUENCE_RECORDED"
    assert "GAME_RESULT" not in old_stage
    calls = (tuple(f.q_calls), tuple(f.executions))

    def forbidden_selection(*args, **kwargs):
        raise AssertionError("resume must not draw the old candidate or strategy again")

    with monkeypatch.context() as recovery_patch:
        recovery_patch.setattr(runtime, "select_probe_candidate", forbidden_selection)
        resumed = restarted_runner(runtime, f, ledger)
        with pytest.raises(OnlinePilotLedgerError, match="replay"):
            ledger.start_game("game-1")
        with pytest.raises(ValueError):
            resumed.start_game("game-1")
        with pytest.raises(ValueError, match="game/recorder identity"):
            resumed.handle_pre(env=f.env, recorder=f.recorder,
                               call_audit=f.call_audit, observation=f.observation,
                               handoff=f.handoff)

    after = ledger.snapshot()
    assert ledger.path.read_bytes().startswith(original)
    assert len(after["events"]) == len(before["events"]) + 1
    assert after["events"][-1]["kind"] == "POST_ENDPOINT_TAIL_INTERRUPTED"
    assert after["games"]["game-1"]["POST_ENDPOINT_TAIL_INTERRUPTED"] == {
        "game_id": "game-1", "assignment_id": old_stage["ASSIGNMENT"]["assignment_id"],
        "reason": "CONSEQUENCE_WITHOUT_GAME_RESULT"}
    assert "INTERRUPTED" not in after["games"]["game-1"]
    for stage, payload in old_stage.items():
        assert after["games"]["game-1"][stage] == payload
    assert after["assignment_count"] == after["games_attempted"] == 1
    assert resumed.state.assignments == {"game-1": old_stage["ASSIGNMENT"]}
    assert resumed.state.games_seen == {"game-1"}
    assert resumed.records == {}
    assert (tuple(f.q_calls), tuple(f.executions)) == calls
    frozen_bytes = ledger.path.read_bytes()
    assert ledger.mark_interrupted_on_resume() == ()
    assert ledger.path.read_bytes() == frozen_bytes
    with pytest.raises(OnlinePilotLedgerError):
        ledger.persist_game_result("game-1", "Werewolf")
    with pytest.raises(OnlinePilotLedgerError):
        ledger.persist_strategy_stage(f.pilot.records["game-1"])
    assert ledger.path.read_bytes() == frozen_bytes
    recovered = OnlinePilotAssignmentLedgerV1(
        ledger.path, plan=f.pilot.plan, source_commit=SOURCE_COMMIT)
    assert recovered.snapshot() == after

    if target_assignment_count == 1:
        assert recovered.status() == "READY_TO_SEAL" and recovered.sealable()
        with pytest.raises(ValueError, match="target or safety cap"):
            resumed.start_game("game-2")
        assert ledger.path.read_bytes() == frozen_bytes
        return

    assert recovered.status() == "RUNNING" and not recovered.sealable()
    resumed.start_game("game-2")
    prefix = canonical_pre(game_id="game-2")
    f.recorder.game_id = "game-2"
    f.recorder._pending.prefix = prefix
    f.handoff.boundary_id, f.handoff.prefix_digest, f.handoff.observer_id = (
        prefix.boundary_id, prefix.prefix_digest, prefix.current_speaker)
    f.observation["current_act_idx"] = 1
    f.env.phase = "speech"
    f.env.public_events = []
    monkeypatch.setattr(runtime, "build_phase2_decision_opportunity",
        lambda context, candidate, *args, **kwargs: replace(f.initial, legal_context=context))
    monkeypatch.setattr(runtime, "prepare_phase2_intervention_speech",
        lambda *args, **kwargs: SimpleNamespace(verified=SimpleNamespace(
            success=True, attempt_count=1, failure_reason=None,
            language_audit=SimpleNamespace(structured_execution_valid=True,
                                          canonical_bytes=lambda: b"new-game-audit"))))
    f.pilot = resumed
    assert pre(f) is not None
    final = ledger.snapshot()
    assert final["assignment_count"] == final["games_attempted"] == 2
    assert final["games"]["game-1"] == after["games"]["game-1"]
    new_assignment = final["games"]["game-2"]["ASSIGNMENT"]
    assert new_assignment["game_id"] == "game-2"
    assert new_assignment["assignment_id"] != old_stage["ASSIGNMENT"]["assignment_id"]
    assert new_assignment["assignment"]["assignment_key"] != old_stage["ASSIGNMENT"]["assignment"]["assignment_key"]
    assert ledger.path.read_bytes().startswith(frozen_bytes)


def test_completed_game_before_publication_can_resume_sealing_only(
        runtime, monkeypatch, tmp_path):
    f = completed_probe_trace(runtime, monkeypatch, tmp_path / "complete")
    ledger, original = crash_prefix(f, tmp_path, "GAME_RESULT")
    before = ledger.snapshot()
    resumed = restarted_runner(runtime, f, ledger)
    assert ledger.mark_interrupted_on_resume() == ()
    assert ledger.snapshot() == before
    assert ledger.path.read_bytes() == original
    assert ledger.status() == "READY_TO_SEAL" and ledger.sealable()
    stage = before["games"]["game-1"]
    assert {"ASSIGNMENT", "EXECUTION", "CONSEQUENCE", "GAME_RESULT"} <= stage.keys()
    assert stage["STRATEGY_STAGE"]["record"]["lifecycle"][-1] == "GAME_RESULT_RECORDED"
    assert resumed.state.assignments == {"game-1": stage["ASSIGNMENT"]}
    for game_id in ("game-1", "extra-game"):
        with pytest.raises(ValueError, match="target or safety cap"):
            resumed.start_game(game_id)
    assert ledger.path.read_bytes() == original


def test_zero_assignment_finished_game_resumes_with_new_game_only(
        runtime, monkeypatch, tmp_path):
    f = fixture(runtime, monkeypatch, tmp_path, ProbeStrategy.PROBE_THEN_REDIRECT)
    f.observation["identity"] = "Villager"
    assert pre(f) is None
    f.pilot.after_game(env=f.env, winner="Werewolf")
    before = f.ledger.snapshot()
    assert before["assignment_count"] == 0
    assert f.executions == [] and f.q_calls == []
    recovered = OnlinePilotAssignmentLedgerV1(
        f.ledger.path, plan=f.pilot.plan, source_commit=SOURCE_COMMIT,
    )
    resumed = restarted_runner(runtime, f, recovered)
    assert recovered.mark_interrupted_on_resume() == ()
    assert recovered.status() == "RUNNING" and not recovered.sealable()
    assert recovered.snapshot() == before
    with pytest.raises(OnlinePilotLedgerError, match="replay"):
        resumed.start_game("game-1")
    resumed.start_game("game-2")
    assert recovered.snapshot()["games_attempted"] == 2
    assert recovered.snapshot()["assignment_count"] == 0
    assert "INTERRUPTED" not in recovered.snapshot()["games"]["game-1"]
    assert resumed.state.assignments == {}
