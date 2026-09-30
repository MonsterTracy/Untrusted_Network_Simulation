"""Scripted online Pilot-T on a genuine wolf speech PRE; no live LLM."""

from copy import deepcopy
from contextlib import nullcontext
from dataclasses import replace
import json
import re
from types import SimpleNamespace

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("openai")
pytest.importorskip("torch")

from tests.canonical_collection.test_final_capacity import _night
from tests.canonical_collection.test_runtime_game_evidence import _runtime, ROLES
from tests.phase2.test_decision_opportunity import Mapper, q_matrix
from tests.phase2.test_intervention_risk import values
from werewolf.phase2_actions import Action, context_from_pre
from werewolf.phase2_decision_opportunity import build_phase2_decision_opportunity
from werewolf.phase2_online_plan import (
    Phase2OnlinePilotPlanV1, assign_terminal_action, select_candidate,
)
from scripts.run_phase2_online_intervention_pilot import (
    OnlineTerminalPilotRunnerV1, OnlinePilotRuntimeBundleV1,
    run_online_game, run_online_campaign, publish_online_pilot,
)
from scripts.phase2_intervention_preflight import PilotPreflightV1
from tests.phase2.test_online_ledger import _game_assignment
from werewolf.phase2_pilot_dataset import build_phase2_online_dataset_from_ledger
from run_random import eval as run_game
from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
from werewolf.phase2_outcome import reference_tables_digest
from werewolf.phase2_online_ledger import OnlinePilotAssignmentLedgerV1


class _StopAfterFirstDayEnv(WerewolfTextEnvV0):
    def step(self, action):
        before = len(self.public_events)
        observation, reward, done, info = super().step(action)
        if any(event["event_type"] == "exile_result"
               for event in self.public_events[before:]) and self.phase != "speech_pk":
            return observation, reward, True, {"Werewolf": -1}
        return observation, reward, done, info


def _speech_pre():
    _, _, env, agents, audit, recorder, raw_backend = _runtime()
    env.reset(roles=ROLES)
    env._rng.seed(6)
    _night(env, 0)
    assert env.current_act_idx == 1 and env.phase == "speech"
    recorder.start(env, roles=ROLES)
    observation = env.get_observation()
    handoff = recorder.before_agent_act(
        env, step_idx=0, acting_player_id=2,
        delivered_observation=observation, speech_kind="speech")
    prefix = recorder._pending.prefix
    context = context_from_pre(prefix, frozenset(("player1", "player2")))
    opp = build_phase2_decision_opportunity(
        context, "player3", q_matrix(), Mapper(), q_source_digest="b" * 64)
    return env, agents, audit, recorder, raw_backend, observation, handoff, opp


def _seed_for(opp, action):
    for seed in range(100):
        plan = Phase2OnlinePilotPlanV1("synthetic", seed)
        selection = select_candidate(plan, opp.legal_context)
        if (selection.candidate_j == opp.candidate_j
                and assign_terminal_action(plan, selection, opp).treatment.action is action):
            return seed
    raise AssertionError("no deterministic seed produced requested arm")


@pytest.mark.parametrize("action", (Action.PUSH, Action.REDIRECT))
def test_online_runner_commits_exactly_one_verified_speech(action, tmp_path):
    env, _, audit, recorder, raw, observation, handoff, opp = _speech_pre()
    j, k = opp.candidate_j, opp.redirect_target
    raw.supports_json_schema = True
    if action is Action.PUSH:
        text = f"本轮放逐票投给{j}。"
        payload = {"commitment_targets": [j], "rejected_targets": [],
                   "vote_intent_targets": [j], "information_requests": [],
                   "abstain_intent": False, "private_fact_claim": False}
    else:
        text = f"本轮不以{j}为主要放逐目标，本轮放逐票投给{k}。"
        payload = {"commitment_targets": [k], "rejected_targets": [j],
                   "vote_intent_targets": [k], "information_requests": [],
                   "abstain_intent": False, "private_fact_claim": False}
    plan = Phase2OnlinePilotPlanV1("synthetic", _seed_for(opp, action))
    ledger = OnlinePilotAssignmentLedgerV1(
        tmp_path / "ledger.jsonl", plan=plan, source_commit="a" * 40)
    def chat_with_metadata(**kwargs):
        stages = ledger.snapshot()["games"][recorder.game_id]
        assert "ASSIGNMENT" in stages  # fsynced before the first backend call
        return (json.dumps(payload) if "response_format" in kwargs else text), {
            "finish_reason": "stop"}
    raw.chat_with_metadata = chat_with_metadata
    predictor = SimpleNamespace(predict=lambda prefix: q_matrix(), seal_digest="b" * 64)
    pilot = OnlineTerminalPilotRunnerV1(
        plan=plan, predictor=predictor, mapper=Mapper(),
        backend=env.speech_perceiver.backend, model_name="fixture-model-v1",
        reference_tables=values(), reference_artifact_digest=reference_tables_digest(values()),
        ledger=ledger)
    pilot.start_game(recorder.game_id)
    votes_before = deepcopy(env.vote_target)
    event_count = len(env.public_events)
    result = pilot.handle_pre(env=env, recorder=recorder, call_audit=audit,
                              observation=observation, handoff=handoff)
    assert result is not None
    assert pilot.records[recorder.game_id].execution_success is True
    assert pilot.records[recorder.game_id].assignment.treatment.action is action
    assert [row.role for row in pilot.records[recorder.game_id].backend_calls] == [
        "realization", "perception"]
    assert all(call.pilot_id == plan.pilot_id and
               call.assignment_id == pilot.records[recorder.game_id].assignment.digest()
               for call in pilot.records[recorder.game_id].backend_calls)
    assert env.public_events[event_count]["raw_text"] == text
    assert env.vote_target == votes_before
    assert pilot.handle_pre(env=env, recorder=recorder, call_audit=audit,
                            observation=observation, handoff=handoff) is None
    assert len(pilot.state.assignments) == 1
    assert ledger.snapshot()["assignment_count"] == 1
    persisted = ledger.snapshot()["games"][recorder.game_id]
    assert "EXECUTION" in persisted
    assert [entry["call"]["role"] for entry in persisted["BACKEND_CALL"]] == [
        "realization", "perception"]


def test_assignment_ledger_failure_prevents_any_phase2_backend_call(tmp_path,
                                                                    monkeypatch):
    env, _, audit, recorder, raw, observation, handoff, opp = _speech_pre()
    plan = Phase2OnlinePilotPlanV1("synthetic", _seed_for(opp, Action.PUSH))
    ledger = OnlinePilotAssignmentLedgerV1(
        tmp_path / "ledger.jsonl", plan=plan, source_commit="a" * 40)
    pilot = OnlineTerminalPilotRunnerV1(
        plan=plan, predictor=SimpleNamespace(predict=lambda prefix: q_matrix(),
                                             seal_digest="b" * 64),
        mapper=Mapper(), backend=env.speech_perceiver.backend,
        model_name="fixture-model-v1", reference_tables=values(),
        reference_artifact_digest=reference_tables_digest(values()), ledger=ledger)
    pilot.start_game(recorder.game_id)
    before = len(raw.requests)
    monkeypatch.setattr(ledger, "persist_assignment",
                        lambda assignment: (_ for _ in ()).throw(OSError("fsync failed")))
    with pytest.raises(OSError, match="fsync failed"):
        pilot.handle_pre(env=env, recorder=recorder, call_audit=audit,
                         observation=observation, handoff=handoff)
    assert len(raw.requests) == before
    assert ledger.snapshot()["assignment_count"] == 0
    assert not pilot.state.assignments


def test_uncertain_post_commit_error_is_interrupted_not_false_failure(tmp_path,
                                                                    monkeypatch):
    env, _, audit, recorder, raw, observation, handoff, opp = _speech_pre()
    raw.supports_json_schema = True
    j = opp.candidate_j
    payload = {"commitment_targets": [j], "rejected_targets": [],
               "vote_intent_targets": [j], "information_requests": [],
               "abstain_intent": False, "private_fact_claim": False}
    raw.chat_with_metadata = lambda **kwargs: (
        json.dumps(payload) if "response_format" in kwargs else f"本轮放逐票投给{j}。",
        {"finish_reason": "stop"})
    plan = Phase2OnlinePilotPlanV1("synthetic", _seed_for(opp, Action.PUSH))
    ledger = OnlinePilotAssignmentLedgerV1(
        tmp_path / "ledger.jsonl", plan=plan, source_commit="a" * 40)
    pilot = OnlineTerminalPilotRunnerV1(
        plan=plan, predictor=SimpleNamespace(predict=lambda prefix: q_matrix(),
                                             seal_digest="b" * 64),
        mapper=Mapper(), backend=env.speech_perceiver.backend,
        model_name="fixture-model-v1", reference_tables=values(),
        reference_artifact_digest=reference_tables_digest(values()), ledger=ledger)
    pilot.start_game(recorder.game_id)
    import scripts.run_phase2_online_intervention_pilot as runner_module
    def uncertain_commit(**kwargs):
        env.public_events.append({"event_type": "public_speech", "raw_text": "partial"})
        raise RuntimeError("after public event")
    monkeypatch.setattr(runner_module, "commit_phase2_verified_speech", uncertain_commit)
    with pytest.raises(RuntimeError, match="after public event"):
        pilot.handle_pre(env=env, recorder=recorder, call_audit=audit,
                         observation=observation, handoff=handoff)
    stage = ledger.snapshot()["games"][recorder.game_id]
    assert "ASSIGNMENT" in stage and "EXECUTION" not in stage
    assert pilot.records[recorder.game_id].execution_success is None
    assert ledger.mark_interrupted_on_resume() == (recorder.game_id,)


def test_online_runner_records_invalid_language_without_reassignment(tmp_path):
    env, _, audit, recorder, raw, observation, handoff, opp = _speech_pre()
    raw.supports_json_schema = True
    wrong = "本轮放逐票投给player7。"
    payload = {"commitment_targets": ["player7"], "rejected_targets": [],
               "vote_intent_targets": ["player7"], "information_requests": [],
               "abstain_intent": False, "private_fact_claim": False}
    raw.chat_with_metadata = lambda **kwargs: (
        json.dumps(payload) if "response_format" in kwargs else wrong,
        {"finish_reason": "stop"})
    predictor = SimpleNamespace(predict=lambda prefix: q_matrix(), seal_digest="b" * 64)
    plan = Phase2OnlinePilotPlanV1("synthetic", _seed_for(opp, Action.PUSH))
    ledger = OnlinePilotAssignmentLedgerV1(
        tmp_path / "ledger.jsonl", plan=plan, source_commit="a" * 40)
    pilot = OnlineTerminalPilotRunnerV1(
        plan=plan,
        predictor=predictor, mapper=Mapper(), backend=env.speech_perceiver.backend,
        model_name="fixture-model-v1", reference_tables=values(),
        reference_artifact_digest=reference_tables_digest(values()), ledger=ledger)
    pilot.start_game(recorder.game_id)
    before = deepcopy(env.public_events)
    assert pilot.handle_pre(env=env, recorder=recorder, call_audit=audit,
                            observation=observation, handoff=handoff) is None
    row = pilot.records[recorder.game_id]
    assert row.execution_success is False and row.canonical_event_digest is None
    assert len(row.backend_calls) == 4  # actor/perceiver then one repair
    assert env.public_events == before
    assert pilot.handle_pre(env=env, recorder=recorder, call_audit=audit,
                            observation=observation, handoff=handoff) is None
    assert len(pilot.state.assignments) == 1
    # The baseline game still reaches its real day resolution. Failed language
    # remains an assignment/execution row, never a treatment consequence.
    for _ in range(30):
        current = env.get_observation()
        if env.phase in ("speech", "speech_pk"):
            action = (env.phase, "继续按原策略讨论。")
            context = audit.speech_perception_context(
                event_id=f"event-{len(env.public_events):06d}",
                boundary_id="synthetic-followup", speaker_id=env.current_act_idx + 1)
        else:
            action = next((item for item in current["valid_action"]
                           if item[1] != -1), current["valid_action"][0])
            context = nullcontext()
        with context:
            _, _, done, info = env.step(action)
        pilot.after_step(env=env, done=done, info=info)
        if any(event["event_type"] == "exile_result" for event in env.public_events):
            break
    assert any(event["event_type"] == "exile_result" for event in env.public_events)
    assert pilot.records[recorder.game_id].day_outcome is None
    assert "CONSEQUENCE" not in ledger.snapshot()["games"][recorder.game_id]


def test_explicit_runner_refuses_missing_online_preflight():
    env, agents, audit, recorder, _, _, _, _ = _speech_pre()
    pilot = OnlineTerminalPilotRunnerV1(
        plan=Phase2OnlinePilotPlanV1("synthetic", 1),
        predictor=SimpleNamespace(predict=lambda prefix: q_matrix(),
                                  seal_digest="b" * 64),
        mapper=Mapper(), backend=env.speech_perceiver.backend,
        model_name="fixture-model-v1", reference_tables=values(),
        reference_artifact_digest=reference_tables_digest(values()))
    with pytest.raises(ValueError, match="preflight"):
        run_online_game(env, agents, ROLES, recorder=recorder,
                        call_audit=audit, pilot=pilot,
                        preflight=SimpleNamespace(online_randomized_pilot_ready=False,
                                                  online_plan_digest=pilot.plan.digest(),
                                                  runtime_token=None))
    assert pilot._active_game_id is None


def test_qualification_artifact_requires_ten_sealed_assignments(tmp_path):
    plan = Phase2OnlinePilotPlanV1("qualification", 17,
                                   campaign_purpose="qualification")
    ledger = OnlinePilotAssignmentLedgerV1(
        tmp_path / "running.jsonl", plan=plan, source_commit="a" * 40)
    backend, predictor, mapper, reference = object(), SimpleNamespace(
        seal_digest="b" * 64), Mapper(), values()
    pilot = SimpleNamespace(plan=plan, ledger=ledger, backend=backend,
                            predictor=predictor, mapper=mapper,
                            reference_tables=reference,
                            reference_artifact_digest=reference_tables_digest(reference))
    destination = tmp_path / "paper-phase2-online-terminal-qualification-v1"
    preflight = PilotPreflightV1(
        source_head="a" * 40,
        source_provenance={"commit": "a" * 40, "branch": "twd/mainline",
                           "tracked_worktree_clean": True,
                           "staged_tracked_changes": False, "source_sha256": {}},
        smoke_v3_manifest_digest="c" * 64, online_plan_digest=plan.digest(),
        destination=str(destination),
        runtime_token=(0, 0, 0, id(backend), id(predictor), id(mapper),
                       id(reference), id(ledger)),
        tracked_clean=True, frozen_artifacts_verified=True,
        final_mapper_lineage_verified=True, frozen_q_runtime_available=True,
        smoke_v3_gate_passed=True, online_plan_frozen=True,
        treatment_randomization_valid=True, outcome_reference_available=True,
        destination_absent=True, checkpoint_executable=False,
        canonical_commit_bound=True, phase2_backend_audit_bound=True,
        assignment_ledger_bound=True, branch_executor_executable=False,
        blockers=(), source_commit_pin="a" * 40,
        reference_tables_digest_pin=reference_tables_digest(reference))
    for index in range(9):
        game_id = f"g{index}"
        ledger.start_game(game_id)
        ledger.persist_assignment(_game_assignment(plan, game_id))
    ledger.mark_interrupted_on_resume()
    incomplete = build_phase2_online_dataset_from_ledger(
        ledger, execution_verifier=lambda *_: True)
    with pytest.raises(ValueError, match="real execution proof"):
        publish_online_pilot(destination, pilot=pilot, dataset=incomplete,
                             preflight=preflight)
    ledger.start_game("g9")
    ledger.persist_assignment(_game_assignment(plan, "g9"))
    assert not ledger.sealable()
    ledger.mark_interrupted_on_resume()
    complete = build_phase2_online_dataset_from_ledger(
        ledger, execution_verifier=lambda *_: True)
    with pytest.raises(ValueError, match="real execution proof"):
        publish_online_pilot(destination, pilot=pilot, dataset=complete,
                             preflight=replace(preflight, source_commit_pin="0" * 40))
    with pytest.raises(ValueError, match="real execution proof"):
        publish_online_pilot(destination, pilot=pilot, dataset=complete,
                             preflight=replace(preflight,
                                               reference_tables_digest_pin="0" * 64))
    digest = publish_online_pilot(destination, pilot=pilot, dataset=complete,
                                  preflight=preflight)
    assert len(digest) == 64 and destination.is_dir()
    assert complete.manifest["estimator_eligible"] is False
    with pytest.raises(FileExistsError):
        publish_online_pilot(destination, pilot=pilot, dataset=complete,
                             preflight=preflight)


def test_campaign_resume_skips_prior_games_and_stops_at_assignment_target(tmp_path,
                                                                            monkeypatch):
    plan = Phase2OnlinePilotPlanV1("qualification", 17,
                                   campaign_purpose="qualification")
    ledger = OnlinePilotAssignmentLedgerV1(
        tmp_path / "running.jsonl", plan=plan, source_commit="a" * 40)
    ledger.start_game("g0")
    ledger.persist_assignment(_game_assignment(plan, "g0"))
    seen = []
    def factory(game_id):
        seen.append(game_id)
        return OnlinePilotRuntimeBundleV1(
            None, (), (), SimpleNamespace(game_id=game_id), None,
            SimpleNamespace(ledger=ledger, plan=plan), None)
    def execute(env, agents, roles, *, recorder, call_audit, pilot, preflight):
        ledger.start_game(recorder.game_id)
        ledger.persist_assignment(_game_assignment(plan, recorder.game_id))
    import scripts.run_phase2_online_intervention_pilot as runner_module
    monkeypatch.setattr(runner_module, "run_online_game", execute)
    result = run_online_campaign(tuple(f"g{i}" for i in range(12)), factory,
                                 ledger=ledger)
    assert result["assignment_count"] == 10
    assert result["status"] == "READY_TO_SEAL"
    assert seen == [f"g{i}" for i in range(1, 10)]


def test_online_day_resolution_uses_actual_vote_event_and_reference_table(tmp_path):
    env, _, audit, recorder, raw, observation, handoff, opp = _speech_pre()
    raw.supports_json_schema = True
    text = "本轮放逐票投给player3。"
    semantic = {"commitment_targets": ["player3"], "rejected_targets": [],
                "vote_intent_targets": ["player3"], "information_requests": [],
                "abstain_intent": False, "private_fact_claim": False}
    raw.chat_with_metadata = lambda **kwargs: (
        json.dumps(semantic) if "response_format" in kwargs else text,
        {"finish_reason": "stop"})
    predictor = SimpleNamespace(predict=lambda prefix: q_matrix(), seal_digest="b" * 64)
    plan = Phase2OnlinePilotPlanV1("synthetic", _seed_for(opp, Action.PUSH))
    ledger = OnlinePilotAssignmentLedgerV1(
        tmp_path / "ledger.jsonl", plan=plan, source_commit="a" * 40)
    pilot = OnlineTerminalPilotRunnerV1(
        plan=plan,
        predictor=predictor, mapper=Mapper(), backend=env.speech_perceiver.backend,
        model_name="fixture-model-v1", reference_tables=values(),
        reference_artifact_digest=reference_tables_digest(values()), ledger=ledger)
    pilot.start_game(recorder.game_id)
    result = pilot.handle_pre(env=env, recorder=recorder, call_audit=audit,
                              observation=observation, handoff=handoff)
    pilot.after_step(env=env, done=result[2], info=result[3])
    for _ in range(30):
        if pilot.records[recorder.game_id].day_outcome is not None:
            break
        current = env.get_observation()
        if env.phase in ("speech", "speech_pk"):
            action = (env.phase, "继续按原策略讨论。")
            context = audit.speech_perception_context(
                event_id=f"event-{len(env.public_events):06d}",
                boundary_id="synthetic-followup", speaker_id=env.current_act_idx + 1)
        else:
            action = next((item for item in current["valid_action"]
                           if item[1] != -1), current["valid_action"][0])
            context = nullcontext()
        with context:
            _, _, done, info = env.step(action)
        pilot.after_step(env=env, done=done, info=info)
    row = pilot.records[recorder.game_id]
    assert row.day_outcome is not None
    assert row.day_outcome.category.value in {
        "target_j_exiled", "other_nonwolf_exiled", "acting_wolf_exiled",
        "teammate_wolf_exiled", "no_exile"}
    assert row.day_outcome.l_ref == 1 - row.day_outcome.v_ref
    assert any(event["event_type"] == "vote_result" for event in env.public_events)
    assert "CONSEQUENCE" in ledger.snapshot()["games"][recorder.game_id]
    dataset = build_phase2_online_dataset_from_ledger(ledger, allow_synthetic=True)
    assert len(dataset.assignments) == len(dataset.executions) == len(dataset.consequences) == 1
    assert dataset.consequences[0]["day_consequence"]["l_ref"] == row.day_outcome.l_ref


def test_full_canonical_game_accepts_phase2_runtime_call_audit():
    _, _, env, agents, audit, recorder, raw = _runtime()
    # The shared fixture normally ends after its first speech. Continue the
    # genuine engine through the first day resolution, then terminate the
    # synthetic fixture so this test has a strict bound.
    env.__class__ = _StopAfterFirstDayEnv
    raw.supports_json_schema = True

    def model_response(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        if "response_format" not in kwargs:
            direct = re.search(r"唯一目标(player[1-7])", prompt)
            if direct:
                return f"本轮放逐票投给{direct.group(1)}。", {"finish_reason": "stop"}
            j, k = re.search(r"拒绝目标(player[1-7])和唯一转向目标(player[1-7])",
                             prompt).groups()
            return f"本轮不以{j}为主要放逐目标，本轮放逐票投给{k}。", {
                "finish_reason": "stop"}
        text = prompt.split("待解析的本轮公开发言：\n", 1)[1].split(
            "\ncommitment_targets：", 1)[0]
        votes = re.findall(r"放逐票投给(player[1-7])", text)
        rejected = re.findall(r"不以(player[1-7])为主要放逐目标", text)
        payload = {"commitment_targets": votes[-1:], "rejected_targets": rejected,
                   "vote_intent_targets": votes[-1:], "information_requests": [],
                   "abstain_intent": False, "private_fact_claim": False}
        return json.dumps(payload), {"finish_reason": "stop"}

    raw.chat_with_metadata = model_response
    predictor = SimpleNamespace(predict=lambda prefix: q_matrix(), seal_digest="b" * 64)
    pilot = OnlineTerminalPilotRunnerV1(
        plan=Phase2OnlinePilotPlanV1("synthetic", 17), predictor=predictor,
        mapper=Mapper(), backend=env.speech_perceiver.backend,
        model_name="fixture-model-v1", reference_tables=values(),
        reference_artifact_digest=reference_tables_digest(values()))
    pilot.start_game(recorder.game_id)
    run_game(env, agents, ROLES, canonical_recorder=recorder,
             call_audit=audit, online_pilot=pilot)
    evidence = recorder.complete_evidence()
    row = pilot.records[recorder.game_id]
    assert row.execution_success and row.day_outcome is not None, (
        row.execution_failure_reason, row.language_attempt_count,
        [(call.role, call.error_category) for call in row.backend_calls])
    assert row.final_game_result in ("Werewolf", "Villager")
    assert len(pilot.state.assignments) == 1
    assert len([call for call in audit.records
                if call.operation_id.startswith("phase2-")]) == 2
    assert evidence.game_id == recorder.game_id
