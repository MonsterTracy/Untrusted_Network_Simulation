"""Real canonical PRE/call/history validation with synthetic immutable evidence.

No server assembly, language dispatch, simulator step, or package shims.
"""
import ast
from copy import deepcopy
import json
from pathlib import Path
import re

import pytest

from tests.canonical_collection.test_pre_prefix import _prefix, _events
from tests.phase2.test_decision_opportunity import Mapper, q_matrix
from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection.game_bundle import _backend_call_from_record, _prefix_from_record
from werewolf.canonical_collection.public_history import freeze_public_event_history
from werewolf.canonical_collection.trajectory_evidence import (
    BackendCallPurpose, BackendCallStatus, construct_backend_call_evidence,
)
from werewolf.phase2_actions import Action, context_from_pre
from werewolf.phase2_decision_opportunity import build_phase2_decision_opportunity
from werewolf.phase2_online_plan import (
    PROBE_PLAN_VERSION, Phase2OnlineProbePilotPlanV1, Phase2OnlineTerminalPilotPlanV1,
    probe_opportunity_from_record,
)
from werewolf.phase2_treatment import build_phase2_treatment
from werewolf.phase2_offline import classify_outcome


@pytest.fixture
def server():
    source = ast.parse(Path("werewolf/phase2_online_server.py").read_text())
    nodes = [node for node in source.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in ("_read_json", "load_online_plan", "ServerRuntimeFactory")]
    scope = dict(json=json, Path=Path, re=re, canonical_json_bytes=canonical_json_bytes,
                 sha256_bytes=sha256_bytes, classify_outcome=classify_outcome,
                 _prefix_from_record=_prefix_from_record,
                 _backend_call_from_record=_backend_call_from_record,
                 freeze_public_event_history=freeze_public_event_history,
                 probe_opportunity_from_record=probe_opportunity_from_record,
                 PROBE_PLAN_VERSION=PROBE_PLAN_VERSION,
                 Phase2OnlineProbePilotPlanV1=Phase2OnlineProbePilotPlanV1,
                 Phase2OnlineTerminalPilotPlanV1=Phase2OnlineTerminalPilotPlanV1)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "phase2_online_server.py", "exec"), scope)
    return scope


@pytest.mark.parametrize("purpose", ["qualification", "pilot"])
def test_same_server_loader_accepts_explicit_probe_plan_without_old_fixed_n(server, tmp_path, purpose):
    plan = Phase2OnlineProbePilotPlanV1("local-fixture", 11, 3, 9, campaign_purpose=purpose)
    path = tmp_path / "plan.json"
    path.write_bytes(canonical_json_bytes(plan.to_record()))
    assert server["load_online_plan"](path, purpose) == plan
    with pytest.raises(ValueError):
        server["load_online_plan"](path, "pilot" if purpose == "qualification" else "qualification")
    raw = plan.to_record()
    raw["target_assignment_count"] = None
    path.write_bytes(canonical_json_bytes(raw))
    with pytest.raises(ValueError):
        server["load_online_plan"](path, purpose)


def canonical_stage(success=True, *, treatment_override=None, assignment_id="a" * 64):
    prefix = _prefix()
    context = context_from_pre(prefix, frozenset(("player2", "player5")))
    op = build_phase2_decision_opportunity(context, "player3", q_matrix(), Mapper(),
                                          q_source_digest="b" * 64)
    treatment = build_phase2_treatment(op, Action.PROBE, assignment_source="randomized_pilot",
                                      assignment_probability=.5, randomization_key="fixture")
    treatment = treatment_override or treatment
    events = _events()
    text = ("player3，你当前主要怀疑谁，依据哪些公开信息？" if treatment.action is Action.PROBE
            else f"本轮不以player3为主要放逐目标，本轮投{treatment.plan.redirect_target}。")
    raw = {"event_id": "probe-committed", "event_index": len(events),
           "event_type": "public_speech", "speaker": context.acting_wolf, "raw_text": text}
    if success:
        events.append(raw)
    calls, evidence = [], []
    for sequence, role in enumerate(("realization", "perception"), 1):
        operation = f"phase2-{treatment.treatment_id[:16]}-{sequence:03d}"
        kwargs = {"messages": [{"role": "user", "content": "public-only fixture"}], "model": "fixture"}
        response = (text if role == "realization" else "{}", {"finish_reason": "stop"})
        actual = construct_backend_call_evidence(call_id=operation + "-attempt-01",
            operation_id=operation, purpose=BackendCallPurpose.RUNTIME,
            status=BackendCallStatus.SUCCESS, boundary_id=context.boundary_id,
            observer_id=context.acting_wolf, attempt_index=1,
            backend_identity="fixture", model_identity="fixture", parser_identity="fixture",
            prompt_identity="fixture", retry_policy_identity="fixture", call_budget_identity="fixture",
            private_payload={"method": "chat_with_metadata", "kwargs": kwargs, "response": response})
        evidence.append(actual.to_record())
        calls.append({"sequence": sequence, "role": role, "attempt_index": 1,
             "pilot_id": "fixture", "assignment_id": assignment_id, "game_id": context.game_id,
             "boundary_id": context.boundary_id, "prefix_digest": context.prefix_digest,
             "opportunity_digest": op.digest(), "treatment_id": treatment.treatment_id,
             "backend_identity": "fixture", "model_identity": "fixture",
             "request_digest": sha256_bytes(canonical_json_bytes(kwargs)),
             "response_digest": sha256_bytes(canonical_json_bytes(response)),
             "canonical_call_id": actual.call_id, "error_category": None,
             "perception_public_only": role == "perception"})
    stage = {"opportunity": op.to_record(), "treatment": treatment.to_record(),
             "language_audit_digest": "f" * 64, "attempt_count": 1,
             "backend_calls": calls, "success": success,
             "failure_reason": None if success else "LANGUAGE_INVALID",
             "canonical_event_id": raw["event_id"] if success else None,
             "canonical_event_digest": sha256_bytes(canonical_json_bytes(raw)) if success else None,
             "generated_text_digest": sha256_bytes(text.encode()) if success else None}
    canonical = {"authoritative_pre_prefixes": [prefix.to_record()],
                 "public_events": events, "backend_calls": evidence}
    return prefix, stage, canonical


@pytest.mark.parametrize("success", [True, False])
def test_server_verifier_uses_actual_authoritative_pre_and_dispatch_evidence(server, success):
    prefix, stage, canonical = canonical_stage(success)
    verified_prefix, history = server["ServerRuntimeFactory"]._verify_probe_stage(
        prefix.game_id, "a" * 64, stage, canonical, invalid_at_pre=not success)
    assert verified_prefix == prefix
    assert len(history.events) == len(prefix.public_event_history.events) + int(success)


@pytest.mark.parametrize("mutation", ["missing_calls", "wrong_actor", "wrong_prefix", "request_digest",
                                     "private_perception", "wrong_backend", "wrong_text"])
def test_server_canonical_stage_verifier_fails_closed_on_contradictory_proof(server, mutation):
    prefix, stage, canonical = canonical_stage()
    if mutation == "missing_calls":
        stage["backend_calls"] = []
    elif mutation == "wrong_actor":
        stage["opportunity"]["identity"]["acting_wolf"] = "player5"
    elif mutation == "wrong_prefix":
        stage["backend_calls"][0]["prefix_digest"] = "0" * 64
    elif mutation == "request_digest":
        stage["backend_calls"][0]["request_digest"] = "0" * 64
    elif mutation == "private_perception":
        stage["backend_calls"][1]["perception_public_only"] = False
    elif mutation == "wrong_backend":
        stage["backend_calls"][1]["backend_identity"] = "another"
    else:
        stage["generated_text_digest"] = "0" * 64
    with pytest.raises(ValueError):
        server["ServerRuntimeFactory"]._verify_probe_stage(prefix.game_id, "a" * 64, stage, canonical)


def test_invalid_stage_must_not_have_appended_public_text(server):
    prefix, stage, canonical = canonical_stage()
    stage["success"] = False
    with pytest.raises(ValueError, match="new public event"):
        server["ServerRuntimeFactory"]._verify_probe_stage(
            prefix.game_id, "a" * 64, stage, canonical, invalid_at_pre=True)


@pytest.mark.parametrize(("strategy_name", "success"), [
    ("IMMEDIATE_REDIRECT", True), ("IMMEDIATE_REDIRECT", False), ("PROBE_THEN_REDIRECT", False),
])
def test_full_server_ledger_proof_retains_t0_consequence_for_invalid_strategy(server, tmp_path, strategy_name, success):
    from dataclasses import replace
    from types import SimpleNamespace
    from tests.phase2.test_outcome_and_execution import values
    from werewolf.phase2_backend_audit import Phase2BackendCallV1
    from werewolf.phase2_offline import ValueEstimate
    from werewolf.phase2_online_ledger import OnlinePilotAssignmentLedgerV1
    from werewolf.phase2_online_plan import ProbeStrategy, assign_probe_strategy, select_probe_candidate
    from werewolf.phase2_online_records import Phase2OnlineProbeRecordV1, Phase2StrategyStageExecutionV1
    from werewolf.phase2_outcome import extract_phase2_day_outcome, reference_tables_digest

    prefix, template, _ = canonical_stage()
    op = probe_opportunity_from_record(template["opportunity"])
    for seed in range(100):
        plan = Phase2OnlineProbePilotPlanV1("canonical-local-fixture", seed, 1, 4)
        selection = select_probe_candidate(plan, op.legal_context)
        if selection.candidate_j == op.candidate_j:
            assignment = assign_probe_strategy(plan, selection, op)
            if assignment.strategy.value == strategy_name:
                break
    else:
        raise AssertionError("fixture strategy seed absent")
    _, stage, canonical = canonical_stage(success, treatment_override=assignment.treatment,
                                         assignment_id=assignment.digest())
    for raw in stage["backend_calls"]:
        raw["pilot_id"] = plan.pilot_id
    calls = tuple(Phase2BackendCallV1(**raw) for raw in stage["backend_calls"])
    link = SimpleNamespace(canonical_event_id=stage["canonical_event_id"],
              canonical_event_digest=stage["canonical_event_digest"],
              public_text_digest=stage["generated_text_digest"]) if success else None
    execution = Phase2StrategyStageExecutionV1(op, assignment.treatment, "f" * 64, 1, calls,
                                              link, None if success else "LANGUAGE_INVALID")
    ledger = OnlinePilotAssignmentLedgerV1(tmp_path / "ledger.jsonl", plan=plan, source_commit="a" * 40)
    ledger.start_game(prefix.game_id)
    ledger.persist_assignment(assignment)
    proofs = {}
    record = Phase2OnlineProbeRecordV1(assignment)
    ledger.persist_strategy_stage(record)
    def append(event, evidence, **changes):
        nonlocal record
        record = replace(record, lifecycle=record.lifecycle + (event,), **changes)
        ledger.persist_strategy_stage(record)
        proofs[f"strategy-{len(record.lifecycle):02d}-{event.lower()}"] = {
            "phase2_record": record.to_record(), "canonical_runtime": deepcopy(evidence)}
    before = {"public_events": _events(), "authoritative_pre_prefixes": [prefix.to_record()],
              "backend_calls": []}
    append("T1_ATTEMPTED", before)
    for call in calls:
        ledger.persist_backend_call(call)
    append("T1_COMMITTED" if success else "T1_LANGUAGE_INVALID", canonical, t1_execution=execution)
    if assignment.strategy is ProbeStrategy.PROBE_THEN_REDIRECT:
        append("T3_CANCELLED", canonical)
    append("EXECUTION_RECORDED", canonical)
    ledger.persist_execution(record)
    proofs["execution"] = {"phase2_record": record.to_record(), "canonical_runtime": deepcopy(canonical)}
    reference = values()
    reference.v_ref_pub_full.cells[(2, 3)] = ValueEstimate(.55, "empirical", None, 1, 1, 1, 1, .55)
    def event(kind, **fields):
        events = canonical["public_events"]
        events.append({"event_id": f"terminal-{len(events)}", "event_index": len(events),
                       "event_type": kind, **fields})
    if not success:
        event("public_speech", speaker=prefix.current_speaker, raw_text="baseline恢复后的有效公开发言。")
    event("phase_change", day=1, phase="vote")
    event("vote_result", votes=[{"voter": f"player{i}", "target": "player3"} for i in range(1, 7)])
    event("exile_result", exiled_players=["player3"])
    event("phase_change", day=1, phase="night")
    outcome = extract_phase2_day_outcome(op, "player3", reference)
    append("DAY_CONSEQUENCE_RECORDED", canonical, day_outcome=outcome,
           reference_artifact_digest=reference_tables_digest(reference))
    ledger.persist_consequence(record)
    proofs["consequence"] = {"phase2_record": record.to_record(), "canonical_runtime": deepcopy(canonical)}
    factory = object.__new__(server["ServerRuntimeFactory"])
    factory.reference = reference
    factory.args = SimpleNamespace(reference_tables_digest=reference_tables_digest(reference))
    factory._stage_evidence = lambda game_id, name: proofs[name]
    stages = ledger.snapshot()["games"][prefix.game_id]
    assert factory.verify_execution(prefix.game_id, stages) is True
    assert stages["CONSEQUENCE"]["day_consequence"]["l_ref"] == pytest.approx(.45)
    assert stages["EXECUTION"]["execution"]["success"] is success
    assert ledger.sealable()  # verified day endpoint is complete; whole-game audit is secondary
    proofs["strategy-02-t1_attempted"]["canonical_runtime"]["backend_calls"] = deepcopy(canonical["backend_calls"])
    with pytest.raises(ValueError, match="precedes lifecycle writeahead"):
        factory.verify_execution(prefix.game_id, stages)
