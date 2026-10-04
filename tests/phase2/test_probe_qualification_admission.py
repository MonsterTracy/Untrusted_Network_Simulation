"""Admission checks against actual temporary artifacts and durable journals.

All experiments here are fictional deterministic fixtures. The production
runner, ledger, dataset builder, artifact publisher and admission validator
are exercised; these tests do not certify real canonical gameplay evidence.
Only optional server import assembly and clean-Git source lookup are isolated.
"""

import ast
from dataclasses import replace
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from scripts import phase2_online_campaign as campaign
from tests.canonical_collection.test_attempt_ledger import _plan
from tests.phase2.test_probe_policy_runner import fixture, next_pre, observe, pre, runtime
from werewolf.artifact_io import (
    canonical_json_bytes, canonical_jsonl_bytes, publish_artifact, sha256_bytes,
)
from werewolf.phase2_online_ledger import OnlinePilotAssignmentLedgerV1
from werewolf.phase2_online_plan import (
    PROBE_PLAN_VERSION, Phase2OnlineProbePilotPlanV1, Phase2OnlineTerminalPilotPlanV1,
    ProbeStrategy, assign_probe_strategy, select_probe_candidate,
)
from werewolf.phase2_online_preflight import SOURCE_FILES
from werewolf.phase2_pilot_dataset import (
    analyze_phase2_online_support_record, build_phase2_online_dataset_from_ledger,
)


SOURCE_COMMIT = "a" * 40
NAME = "paper-phase2-online-probe-qualification-v1"
PROTOCOL_DIGEST = sha256_bytes(b"immutable fixture protocol")


@pytest.fixture
def admission():
    scope = dict(campaign.__dict__)
    server = ast.parse(Path("werewolf/phase2_online_server.py").read_text())
    definitions = [node for node in server.body if isinstance(node, ast.FunctionDef)
                   and node.name in ("_read_json", "load_online_plan")]
    scope.update(re=re, PROBE_PLAN_VERSION=PROBE_PLAN_VERSION,
                 Phase2OnlineProbePilotPlanV1=Phase2OnlineProbePilotPlanV1,
                 Phase2OnlineTerminalPilotPlanV1=Phase2OnlineTerminalPilotPlanV1)
    exec(compile(ast.Module(body=definitions, type_ignores=[]),
                 "werewolf/phase2_online_server.py", "exec"), scope)
    source = ast.parse(Path("scripts/phase2_online_campaign.py").read_text())
    definition = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                      and node.name == "probe_qualification_inputs")
    definition.body = [node for node in definition.body if not (
        isinstance(node, ast.ImportFrom) and node.module == "werewolf.phase2_online_server")]
    exec(compile(ast.Module(body=[definition], type_ignores=[]),
                 "scripts/phase2_online_campaign.py", "exec"), scope)
    return scope["probe_qualification_inputs"]


def qualification_journal(runtime, monkeypatch, tmp_path, *, invalid_probe=False,
                          exiled_player="player2", winner="Werewolf"):
    f = fixture(runtime, monkeypatch, tmp_path / "runner-fixture",
                ProbeStrategy.PROBE_THEN_REDIRECT)
    initial = f.initial
    second = replace(initial, legal_context=replace(initial.legal_context, game_id="game-2"))
    for seed in range(100):
        plan = Phase2OnlineProbePilotPlanV1(NAME, seed, 2, 4)
        strategies = [assign_probe_strategy(plan, select_probe_candidate(plan, op.legal_context),
                                           op).strategy for op in (initial, second)]
        if strategies == [ProbeStrategy.PROBE_THEN_REDIRECT, ProbeStrategy.IMMEDIATE_REDIRECT]:
            break
    else:
        raise AssertionError("no deterministic seed for the two-arm fixture")
    work = tmp_path / "work"
    ledger = OnlinePilotAssignmentLedgerV1(work / "assignment-ledger.jsonl",
                                           plan=plan, source_commit=SOURCE_COMMIT)
    original = f.pilot
    f.ledger = ledger
    f.pilot = runtime.OnlineTerminalPilotRunnerV1(
        plan=plan, predictor=original.predictor, mapper=original.mapper,
        backend=original.backend, model_name=original.model_name,
        reference_tables=original.reference_tables,
        reference_artifact_digest=original.reference_artifact_digest, ledger=ledger,
    )
    original_opportunity = runtime.build_phase2_decision_opportunity

    def opportunity(context, *args, **kwargs):
        result = original_opportunity(context, *args, **kwargs)
        return replace(result, legal_context=context)

    def prepare(op, treatment, public, *, actor, perceiver):
        game_id = op.legal_context.game_id
        latest = ledger.snapshot()["games"][game_id]["STRATEGY_STAGE"]["record"]
        assert latest["lifecycle"][-1] in ("T1_ATTEMPTED", "T3_PREPARED")
        invalid = invalid_probe and game_id == "game-1" and treatment.action.value == "PROBE"
        return SimpleNamespace(verified=SimpleNamespace(
            success=not invalid, attempt_count=2 if invalid else 1,
            failure_reason="LANGUAGE_INVALID" if invalid else None,
            language_audit=SimpleNamespace(structured_execution_valid=True,
                                          canonical_bytes=lambda: b"fixture-audit")))

    monkeypatch.setattr(runtime, "build_phase2_decision_opportunity", opportunity)
    monkeypatch.setattr(runtime, "prepare_phase2_intervention_speech", prepare)
    for game_id, strategy in zip(("game-1", "game-2"), strategies):
        f.env.public_events = []
        f.env.phase = "speech"
        f.recorder.game_id = game_id
        prefix = f.recorder._pending.prefix
        prefix.game_id, prefix.current_speaker = game_id, "player1"
        prefix.boundary_id, prefix.prefix_digest, prefix.phase = "pre-1", "a" * 64, "speech"
        f.handoff.boundary_id, f.handoff.prefix_digest, f.handoff.observer_id = "pre-1", "a" * 64, "player1"
        f.observation["current_act_idx"] = 1
        f.pilot.start_game(game_id)
        first = pre(f)
        if strategy is ProbeStrategy.PROBE_THEN_REDIRECT and not invalid_probe:
            assert first is not None
            observe(f)
            next_pre(f)
            assert pre(f) is not None
        elif strategy is ProbeStrategy.IMMEDIATE_REDIRECT:
            assert first is not None
        else:
            assert first is None
        f.env.phase = "night"
        f.env.public_events.append({"event_type": "exile_result", "exiled_players": [exiled_player]})
        f.pilot.after_step(env=f.env, done=False, info={})
        f.pilot.after_game(env=f.env, winner=winner)
    assert ledger.sealable()
    return f, work


def qualification_evidence(runtime, monkeypatch, tmp_path, *, mutation=None,
                           invalid_probe=False, exiled_player="player2", winner="Werewolf"):
    f, work = qualification_journal(runtime, monkeypatch, tmp_path,
        invalid_probe=invalid_probe, exiled_player=exiled_player, winner=winner)
    # allow_synthetic explicitly labels this local table fixture. Production
    # publish_online_pilot forbids it; generic publication here tests admission
    # integrity, not the separate canonical-execution certification guard.
    dataset = build_phase2_online_dataset_from_ledger(f.ledger, allow_synthetic=True)
    support = analyze_phase2_online_support_record(dataset.to_record())
    destination = tmp_path / NAME
    game_plan = _plan(
        collection_id=f.pilot.plan.pilot_id, source_revision=SOURCE_COMMIT,
        ordered_seed_pool=(901, 902, 903, 904), target_canonical_success_count=2,
        environment_provenance={"probe_policy_protocol_sha256": PROTOCOL_DIGEST},
    )
    bound = {
        "source_commit": SOURCE_COMMIT, "pilot_plan": f.pilot.plan.to_record(),
        "canonical_game_plan": game_plan.to_record(),
        "paths": {"work_directory": str(work), "destination": str(destination)},
    }
    bound["inputs_digest"] = campaign.digest(bound)
    run_inputs = work / "run_inputs.json"
    run_inputs.write_bytes(canonical_json_bytes(bound))
    source = {"commit": SOURCE_COMMIT, "branch": "twd/mainline",
              "tracked_worktree_clean": True, "staged_tracked_changes": False,
              "source_sha256": {name: sha256_bytes(Path(name).read_bytes()) for name in SOURCE_FILES}}
    fields = {
        "artifact_type": "phase2_online_probe_pilot",
        "schema_version": "phase2_online_probe_pilot_v1", "study_name": NAME,
        "campaign_purpose": "qualification", "completion_status": "COMPLETE",
        "pilot_plan_digest": f.pilot.plan.digest(), "source_provenance": source,
        "ledger_digest": dataset.manifest["ledger_digest"],
        "server_run_provenance": {"inputs_digest": bound["inputs_digest"],
                                   "game_plan_digest": game_plan.plan_digest},
    }
    files = {
        "pilot_plan.json": canonical_json_bytes(f.pilot.plan.to_record()),
        "assignments.jsonl": canonical_jsonl_bytes(dataset.assignments),
        "executions.jsonl": canonical_jsonl_bytes(dataset.executions),
        "consequences.jsonl": canonical_jsonl_bytes(dataset.consequences),
        "backend_calls.jsonl": canonical_jsonl_bytes(dataset.backend_calls),
        "strategy_stages.jsonl": canonical_jsonl_bytes(dataset.strategy_stages),
        "metrics/support.json": canonical_json_bytes(support),
    }
    if mutation in ("execution_columns", "consequence_winner", "assignment_row", "strategy_history"):
        name = {"execution_columns": "executions.jsonl", "consequence_winner": "consequences.jsonl",
                "assignment_row": "assignments.jsonl", "strategy_history": "strategy_stages.jsonl"}[mutation]
        rows = [json.loads(line) for line in files[name].splitlines()]
        if mutation == "execution_columns":
            rows[0]["assignment_id"] = rows[0].pop("assignment_digest")
        elif mutation == "consequence_winner":
            rows[0]["final_game_result"] = "Villager" if winner == "Werewolf" else "Werewolf"
        elif mutation == "assignment_row":
            rows[0]["assignment_seed"] += 1
        else:
            rows.pop(0)
        files[name] = canonical_jsonl_bytes(rows)
    elif mutation == "ledger_digest":
        fields["ledger_digest"] = "0" * 64
    elif mutation == "support_incomplete":
        support["itt_outcome_complete"] = False
        files["metrics/support.json"] = canonical_json_bytes(support)
    artifact = publish_artifact(destination, manifest_fields=fields, files=files)
    profile = {"probe_protocol_digest": PROTOCOL_DIGEST, "probe_qualification": {
        "artifact": str(destination), "run_inputs": str(run_inputs),
        "manifest_digest": artifact.manifest_digest, "inputs_digest": bound["inputs_digest"],
    }}
    import werewolf.phase2_online_preflight as preflight
    monkeypatch.setattr(preflight, "freeze_online_source_provenance", lambda repo: source)
    return SimpleNamespace(f=f, profile=profile, bound=bound, artifact=artifact,
                           source=source, run_inputs=run_inputs)


@pytest.mark.parametrize("exiled_player,winner", [("player2", "Werewolf"), ("player1", "Villager")])
def test_complete_two_strategy_artifact_is_admitted_with_exact_builder_columns(
        admission, runtime, monkeypatch, tmp_path, exiled_player, winner):
    evidence = qualification_evidence(runtime, monkeypatch, tmp_path,
        exiled_player=exiled_player, winner=winner)
    bound, artifact = admission(evidence.profile)
    assert bound == evidence.bound
    assert artifact.manifest_digest == evidence.artifact.manifest_digest
    assert evidence.f.ledger.snapshot()["assignment_count"] == 2
    assert evidence.f.ledger.sealable()
    assert json.loads((artifact.path / "metrics/support.json").read_bytes())["estimator_eligible"] is False


@pytest.mark.parametrize("mutation", [
    "execution_columns", "consequence_winner", "assignment_row", "strategy_history",
    "ledger_digest", "support_incomplete",
])
def test_valid_outer_artifact_digest_cannot_hide_ledger_or_table_mismatch(
        admission, runtime, monkeypatch, tmp_path, mutation):
    evidence = qualification_evidence(runtime, monkeypatch, tmp_path, mutation=mutation)
    with pytest.raises(ValueError, match="differs from ledger|ledger/population"):
        admission(evidence.profile)


@pytest.mark.parametrize("mutation", [
    "artifact_bytes", "manifest_pin", "inputs_pin", "source_bytes", "protocol_pin",
    "incomplete_ledger", "ledger_bytes",
])
def test_admission_fails_closed_on_artifact_inputs_source_protocol_and_journal_drift(
        admission, runtime, monkeypatch, tmp_path, mutation):
    evidence = qualification_evidence(runtime, monkeypatch, tmp_path)
    if mutation == "artifact_bytes":
        (evidence.artifact.path / "executions.jsonl").write_bytes(b"{}\n")
    elif mutation == "manifest_pin":
        evidence.profile["probe_qualification"]["manifest_digest"] = "0" * 64
    elif mutation == "inputs_pin":
        evidence.profile["probe_qualification"]["inputs_digest"] = "0" * 64
    elif mutation == "source_bytes":
        evidence.source["source_sha256"]["werewolf/phase2_online_runner.py"] = "0" * 64
    elif mutation == "protocol_pin":
        evidence.profile["probe_protocol_digest"] = "0" * 64
    elif mutation == "incomplete_ledger":
        lines = evidence.f.ledger.path.read_bytes().splitlines(keepends=True)
        assert json.loads(lines[-1])["kind"] == "GAME_RESULT"
        evidence.f.ledger.path.write_bytes(b"".join(lines[:-1]))
    else:
        with evidence.f.ledger.path.open("ab") as file:
            file.write(b"{")
    with pytest.raises(ValueError):
        admission(evidence.profile)


def test_complete_language_invalid_probe_does_not_satisfy_t3_qualification_gate(
        admission, runtime, monkeypatch, tmp_path):
    evidence = qualification_evidence(runtime, monkeypatch, tmp_path, invalid_probe=True)
    assert evidence.f.ledger.sealable()
    snapshot = evidence.f.ledger.snapshot()["games"]["game-1"]
    assert "T3_CANCELLED" in snapshot["STRATEGY_STAGE"]["record"]["lifecycle"]
    assert snapshot["EXECUTION"]["execution"]["success"] is False
    assert "CONSEQUENCE" in snapshot and "GAME_RESULT" in snapshot
    with pytest.raises(ValueError, match="mechanism gate failed"):
        admission(evidence.profile)

