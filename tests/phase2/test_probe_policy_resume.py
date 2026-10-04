"""Crash-prefix recovery on the production runner's real persistent journal.

No simulator checkpoint exists. Assigned games cannot resume an unfinished
stage; completed journals can resume sealing without another controlled act.
"""

import json

import pytest

from tests.phase2.test_probe_policy_runner import (
    fixture, next_pre, observe, pre, runtime,
)
from werewolf.phase2_online_ledger import (
    OnlinePilotAssignmentLedgerV1, OnlinePilotLedgerError,
)
from werewolf.phase2_online_plan import ProbeStrategy


SOURCE_COMMIT = "a" * 40


def completed_probe_trace(runtime, monkeypatch, tmp_path):
    """Produce a real journal once, then use its intact crash prefixes."""
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
    assert f.ledger.sealable()
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
    ("CONSEQUENCE", None, "CONSEQUENCE_WITHOUT_GAME_RESULT"),
])
def test_assigned_crash_prefix_is_retained_and_never_replayed(
        runtime, monkeypatch, tmp_path, kind, lifecycle, reason):
    f = completed_probe_trace(runtime, monkeypatch, tmp_path / "complete")
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
