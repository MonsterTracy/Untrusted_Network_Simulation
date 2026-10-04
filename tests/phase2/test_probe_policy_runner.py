"""Deterministic public-hook fixtures; no environment rollout or backend calls.

The runtime assembly imports optional server dependencies absent locally.
Compile the actual runner and production eval definitions unchanged, excluding
only run_random's module-level assembly import. No substitute gameplay loop.
"""
import ast
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.phase2.test_probe_policy_plan import public_opportunity
from tests.phase2.test_intervention_risk import values
from tests.canonical_collection.test_pre_prefix import _attempt
from werewolf.canonical_collection.pre import (
    AuthoritativePREPrefix, construct_authoritative_pre_prefix, validate_authoritative_pre_prefix,
)
from werewolf.canonical_collection.public_history import freeze_public_event_history
from werewolf.canonical_collection.speech import V1AnnotationStatus, construct_v1_speech_annotation
from werewolf.phase2_actions import Action, context_from_pre
from werewolf.phase2_decision_opportunity import CandidateEvidenceV1, Phase2DecisionOpportunityV1
from werewolf.phase2_online_plan import (
    Phase2OnlineProbePilotPlanV1, ProbeStrategy, assign_probe_strategy, select_probe_candidate,
)
from werewolf.phase2_online_ledger import OnlinePilotAssignmentLedgerV1
from werewolf.phase2_online_records import validate_probe_lifecycle_record
from werewolf.phase2_outcome import reference_tables_digest
from werewolf.phase2_offline import ValueEstimate


@pytest.fixture
def runtime(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    source = ast.parse((root / "werewolf/phase2_online_runner.py").read_text())
    source.body = [node for node in source.body if not
                   (isinstance(node, ast.ImportFrom) and node.module == "run_random")]
    scope = {"__name__": "probe_runner_test_source"}
    # Dataclass resolves module globals via sys.modules.
    import sys
    from types import ModuleType
    module = ModuleType(scope["__name__"])
    module.__file__ = str(root / "werewolf/phase2_online_runner.py")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    production = ast.parse((root / "run_random.py").read_text())
    definitions = [node for node in production.body if isinstance(node, ast.FunctionDef)
                   and node.name in ("eval", "_act_at_boundary")]
    game_scope = {"nullcontext": nullcontext,
                  "PUBLIC_SPEECH_RUNTIME_PHASES": frozenset(("speech", "speech_pk"))}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "run_random.py", "exec"), game_scope)
    module.__dict__["run_canonical_game"] = game_scope["eval"]
    exec(compile(source, "werewolf/phase2_online_runner.py", "exec"), module.__dict__)
    return module


def canonical_pre(*, phase="speech", actor="player1", game_id="game-1", boundary="pre-1"):
    """Construct a real, validated PRE with canonical public phase and turn order."""
    events, annotations = [], []
    def event(kind, **fields):
        events.append(dict(event_id=f"event-{len(events)}", event_index=len(events),
                           event_type=kind, **fields))
    alive = ("player1", "player2", "player5", "player6", "player7")
    event("phase_change", day=0, phase="night")
    event("death_announcement", dead_players=["player3", "player4"])
    event("phase_change", day=1, phase="discussion")
    members = alive
    if phase == "speech_pk":
        event("phase_change", day=1, phase="vote")
        event("vote_result", votes=[dict(voter=voter, target=target) for voter, target in
              zip(alive, ("player2", "player5", "player6", "player1", None))])
        event("exile_result", exiled_players=[])
        event("phase_change", day=1, phase="pk_discussion")
        members = alive[:-1]
    for speaker in members[:members.index(actor) + 1]:
        event("turn_start", speaker=speaker)
        if speaker != actor:
            event("public_speech", speaker=speaker, raw_text="我没有新的信息。")
            annotations.append(construct_v1_speech_annotation(
                freeze_public_event_history(events), status=V1AnnotationStatus.NO_ACTION,
                actions=(), attempts=(replace(_attempt(), status=V1AnnotationStatus.NO_ACTION,
                                              raw_response="None"),)))
    return construct_authoritative_pre_prefix(
        game_id=game_id, boundary_id=boundary, step_index=len(events) - 1,
        report_trigger_id=f"trigger-{boundary}", current_speaker=actor,
        alive_observer_ids=alive, public_event_history=freeze_public_event_history(events),
        v1_annotations=annotations,
        belief_observation_ids_by_observer={p: f"{boundary}-{p}" for p in alive})


def fixture(runtime, monkeypatch, tmp_path, strategy, phase="speech", invalid_stage=None):
    prefix = canonical_pre(phase=phase)
    initial = replace(public_opportunity(phase=phase), legal_context=context_from_pre(
        prefix, frozenset(("player1", "player5"))))
    for seed in range(100):
        plan = Phase2OnlineProbePilotPlanV1("probe-local-fixture", seed, 1, 3)
        selection = select_probe_candidate(plan, initial.legal_context)
        if assign_probe_strategy(plan, selection, initial).strategy is strategy:
            break
    ledger = OnlinePilotAssignmentLedgerV1(tmp_path / "ledger.jsonl", plan=plan,
                                           source_commit="a" * 40)
    recorder = SimpleNamespace(game_id="game-1", _pending=SimpleNamespace(prefix=prefix))
    handoff = SimpleNamespace(boundary_id=prefix.boundary_id, prefix_digest=prefix.prefix_digest,
                              observer_id=prefix.current_speaker)
    observation = {"identity": "Werewolf", "current_act_idx": 1,
                   "game_log": [SimpleNamespace(event="werewolf_team_info",
                        content={"wolf_team": [1, 5]})]}
    env = SimpleNamespace(public_events=[], phase=phase, vote_target=[None] * 7)
    calls = []
    predictor = SimpleNamespace(seal_digest="c" * 64, predict=lambda pre: calls.append(pre.boundary_id))
    mapper = SimpleNamespace(artifact_digest="e" * 64, infer=lambda *a, **kw: None)
    def opportunity(context, candidate, q, mapper, *, q_source_digest):
        assert candidate == "player2"
        if context.acting_wolf == "player1":
            return initial
        # Full T3 legal panel changes the alternative winner from 6 to 7.
        panel = tuple(CandidateEvidenceV1(j, p, "f" * 64) for j, p in
                      (("player2", .1), ("player6", .2), ("player7", .9)))
        if phase == "speech_pk":
            panel = tuple(CandidateEvidenceV1(j, p, "f" * 64) for j, p in
                          (("player2", .1), ("player6", .9)))
        return Phase2DecisionOpportunityV1(context, candidate, (2, 3), 2,
                  len(context.public_speaker_queue) - 3, "c" * 64, "d" * 64, "e" * 64,
                  panel, (Action.PUSH, Action.REDIRECT),
                  "player7" if phase == "speech" else "player6", None, None, None)
    monkeypatch.setattr(runtime, "build_phase2_decision_opportunity", opportunity)
    monkeypatch.setattr(runtime, "public_language_context_from_pre", lambda *a: None)
    executions = []
    def prepare(op, treatment, public, *, actor, perceiver):
        stage = "T1" if op.legal_context.acting_wolf == "player1" else "T3"
        stages = ledger.snapshot()["games"]["game-1"]
        assert "ASSIGNMENT" in stages
        latest = stages["STRATEGY_STAGE"]["record"]
        assert latest["lifecycle"][-1] == ("T1_ATTEMPTED" if stage == "T1" else "T3_PREPARED")
        executions.append(treatment)
        return SimpleNamespace(verified=SimpleNamespace(success=stage != invalid_stage,
            attempt_count=2 if stage == invalid_stage else 1,
            failure_reason="LANGUAGE_INVALID" if stage == invalid_stage else None,
            language_audit=SimpleNamespace(structured_execution_valid=True, canonical_bytes=lambda: b"fixture-audit")))
    monkeypatch.setattr(runtime, "prepare_phase2_intervention_speech", prepare)
    monkeypatch.setattr(runtime, "assess_phase2_checkpoint_closure", lambda *a, **kw: None)
    def commit(**kwargs):
        op = kwargs["opportunity"]
        event = {"event_type": "public_speech", "event_id": f"e{len(env.public_events)}",
                 "speaker": op.legal_context.acting_wolf, "raw_text": "verified fixture text"}
        env.public_events.append(event)
        return (observation, 0, False, {}), SimpleNamespace(
            canonical_event_id=event["event_id"], canonical_event_digest="c" * 64,
            public_text_digest="d" * 64)
    monkeypatch.setattr(runtime, "commit_phase2_verified_speech", commit)
    reference = values()
    reference.v_ref_pub_full.cells[(1, 3)] = ValueEstimate(.2, "empirical", None, 1, 1, 1, 1, .2)
    pilot = runtime.OnlineTerminalPilotRunnerV1(
        plan=plan, predictor=predictor, mapper=mapper,
        backend=SimpleNamespace(supports_json_schema=True, chat_with_metadata=lambda **kw: (_ for _ in ()).throw(AssertionError())),
        model_name="fixture", reference_tables=reference,
        reference_artifact_digest=reference_tables_digest(reference), ledger=ledger)
    audit = SimpleNamespace(context=lambda **kw: nullcontext())
    pilot.start_game("game-1")
    return SimpleNamespace(pilot=pilot, env=env, recorder=recorder, handoff=handoff,
                           observation=observation, call_audit=audit, ledger=ledger,
                           executions=executions, q_calls=calls, initial=initial)


def pre(f):
    return f.pilot.handle_pre(env=f.env, recorder=f.recorder, call_audit=f.call_audit,
                             observation=f.observation, handoff=f.handoff)


def next_pre(f, actor="player5", *, phase=None, boundary="pre-3"):
    prefix = canonical_pre(phase=phase or f.initial.legal_context.phase, actor=actor,
                           game_id=f.recorder.game_id, boundary=boundary)
    f.recorder._pending.prefix = prefix
    f.handoff.boundary_id, f.handoff.prefix_digest, f.handoff.observer_id = (
        prefix.boundary_id, prefix.prefix_digest, prefix.current_speaker)
    f.observation["current_act_idx"] = int(actor.removeprefix("player"))


def observe(f, text="我没有新的信息。"):
    f.pilot.after_step(env=f.env, done=False, info={})
    f.env.public_events.append({"event_id": "answer-1", "event_type": "public_speech",
                               "speaker": "player2", "raw_text": text})
    f.pilot.after_step(env=f.env, done=False, info={})


@pytest.mark.parametrize("phase", ["speech", "speech_pk"])
def test_control_immediate_redirect_writeahead_and_no_second_assignment(runtime, monkeypatch, tmp_path, phase):
    f = fixture(runtime, monkeypatch, tmp_path, ProbeStrategy.IMMEDIATE_REDIRECT, phase)
    assert pre(f) is not None
    assert f.pilot.records["game-1"].lifecycle == (
        "ASSIGNED", "T1_ATTEMPTED", "T1_COMMITTED", "EXECUTION_RECORDED")
    assert f.executions[0].action is Action.REDIRECT
    assert pre(f) is None
    assert f.ledger.snapshot()["assignment_count"] == 1
    assert f.env.vote_target == [None] * 7


@pytest.mark.parametrize("phase", ["speech", "speech_pk"])
@pytest.mark.parametrize("text", ["我没有新的信息。", "我怀疑player6，因为公开投票。"])
def test_probe_success_real_canonical_pre_exact_continuation(runtime, monkeypatch, tmp_path, phase, text):
    f = fixture(runtime, monkeypatch, tmp_path, ProbeStrategy.PROBE_THEN_REDIRECT, phase)
    prefix = f.recorder._pending.prefix
    assert type(prefix) is AuthoritativePREPrefix
    assert validate_authoritative_pre_prefix(prefix) is prefix
    assert not hasattr(prefix, "phase")
    assert prefix.public_temporal_state.phase.value == (
        "discussion" if phase == "speech" else "pk_discussion")
    assert pre(f) is not None
    assert f.pilot.records["game-1"].lifecycle[-1] == "T3_SCHEDULED"
    f.pilot.after_step(env=f.env, done=False, info={})
    next_pre(f, "player2", boundary="pre-2")
    assert pre(f) is None
    f.env.public_events.append({"event_id": "answer-1", "event_type": "public_speech",
                               "speaker": "player2", "raw_text": text})
    f.pilot.after_step(env=f.env, done=False, info={})
    next_pre(f)
    assert type(f.recorder._pending.prefix) is AuthoritativePREPrefix
    validate_authoritative_pre_prefix(f.recorder._pending.prefix)
    assert pre(f) is not None
    record = f.pilot.records["game-1"]
    assert record.lifecycle == ("ASSIGNED", "T1_ATTEMPTED", "T1_COMMITTED", "T3_SCHEDULED",
          "T3_REACHED", "T3_PREPARED", "T3_COMMITTED", "EXECUTION_RECORDED")
    assert [t.action for t in f.executions] == [Action.PROBE, Action.REDIRECT]
    assert {t.candidate_j for t in f.executions} == {"player2"}
    assert f.executions[1].plan.acting_wolf == "player5"
    assert f.executions[1].plan.redirect_target == ("player7" if phase == "speech" else "player6")
    assert f.executions[1].assignment_source == "strategy_continuation"
    assert f.executions[1].assignment_probability == 1
    assert f.ledger.snapshot()["assignment_count"] == 1
    assert f.q_calls == ["pre-1", "pre-3"]
    validate_probe_lifecycle_record(record.to_record())


@pytest.mark.parametrize(("initial_phase", "continuation_phase"), [
    ("speech", "speech_pk"), ("speech_pk", "speech"),
])
def test_real_canonical_pre_phase_mismatch_preserves_assignment_and_stops_before_q(
        runtime, monkeypatch, tmp_path, initial_phase, continuation_phase):
    f = fixture(runtime, monkeypatch, tmp_path, ProbeStrategy.PROBE_THEN_REDIRECT, initial_phase)
    pre(f)
    observe(f)
    next_pre(f, phase=continuation_phase)
    assert type(f.recorder._pending.prefix) is AuthoritativePREPrefix
    with pytest.raises(ValueError, match="Probe continuation phase/boundary mismatch"):
        pre(f)
    assert f.q_calls == ["pre-1"]
    assert len(f.executions) == 1
    assert f.pilot.records["game-1"].lifecycle[-1] == "STRUCTURAL_FAILURE"
    assert f.ledger.snapshot()["assignment_count"] == 1
    assert not f.ledger.sealable()


@pytest.mark.parametrize("strategy", list(ProbeStrategy))
def test_t1_invalid_cancels_probe_and_all_itt_day_consequence_retained(runtime, monkeypatch, tmp_path, strategy):
    f = fixture(runtime, monkeypatch, tmp_path, strategy, invalid_stage="T1")
    assert pre(f) is None
    assert f.env.public_events == []
    record = f.pilot.records["game-1"]
    if strategy is ProbeStrategy.PROBE_THEN_REDIRECT:
        assert "T3_CANCELLED" in record.lifecycle
    next_pre(f)
    assert pre(f) is None
    assert len(f.executions) == 1
    f.env.phase = "night"
    f.env.public_events.append({"event_type": "exile_result", "exiled_players": ["player2"]})
    f.pilot.after_step(env=f.env, done=False, info={})
    f.pilot.after_game(env=f.env, winner="Werewolf")
    record = f.pilot.records["game-1"]
    assert record.execution_success is False and record.day_outcome is not None
    assert record.day_outcome.reference.s_pre == (2, 3)
    stages = f.ledger.snapshot()["games"]["game-1"]
    assert {"ASSIGNMENT", "EXECUTION", "CONSEQUENCE", "GAME_RESULT"} <= stages.keys()
    assert f.ledger.sealable()


@pytest.mark.parametrize("mismatch", ["phase", "boundary", "missing_window", "actor", "Q", "commit"])
def test_structural_mismatch_stops_without_fallback(runtime, monkeypatch, tmp_path, mismatch):
    f = fixture(runtime, monkeypatch, tmp_path, ProbeStrategy.PROBE_THEN_REDIRECT)
    pre(f)
    if mismatch != "missing_window":
        observe(f)
    next_pre(f, "player6" if mismatch == "actor" else "player5",
             phase="speech_pk" if mismatch == "phase" else None,
             boundary="pre-1" if mismatch == "boundary" else "pre-3")
    if mismatch == "Q":
        f.pilot.predictor.predict = lambda pre: (_ for _ in ()).throw(ValueError("missing Q"))
    if mismatch == "commit":
        monkeypatch.setattr(runtime, "commit_phase2_verified_speech",
                            lambda **kw: (_ for _ in ()).throw(ValueError("commit uncertain")))
    with pytest.raises(ValueError):
        pre(f)
    assert f.pilot.records["game-1"].lifecycle[-1] == "STRUCTURAL_FAILURE"
    assert "EXECUTION" not in f.ledger.snapshot()["games"]["game-1"]
    assert not f.ledger.sealable()
    assert f.ledger.snapshot()["assignment_count"] == 1


def test_real_production_callable_signature_accepts_same_hook(runtime):
    runtime.require_online_gameplay_hook()
    import inspect
    assert inspect.signature(runtime.run_canonical_game).parameters["online_pilot"].kind is inspect.Parameter.KEYWORD_ONLY


def test_probe_path_has_no_terminal_risk_lambda_or_router_import(runtime):
    source = Path("werewolf/phase2_online_runner.py").read_text()
    assert not any(name in source for name in ("phase2_terminal_estimator", "phase2_risk", "phase2_probe_value", "ThreeWayRouter"))


def test_t3_language_invalid_stays_itt_without_further_control(runtime, monkeypatch, tmp_path):
    f = fixture(runtime, monkeypatch, tmp_path, ProbeStrategy.PROBE_THEN_REDIRECT, invalid_stage="T3")
    pre(f)
    observe(f)
    next_pre(f)
    before = len(f.env.public_events)
    assert pre(f) is None
    assert len(f.env.public_events) == before
    assert f.pilot.records["game-1"].lifecycle[-2:] == ("T3_LANGUAGE_INVALID", "EXECUTION_RECORDED")
    f.env.phase = "night"
    f.env.public_events.append({"event_type": "exile_result", "exiled_players": ["player1"]})
    f.pilot.after_step(env=f.env, done=False, info={})
    record = f.pilot.records["game-1"]
    # Category uses T0 actor, not T3 actor; the original acting wolf was exiled.
    assert record.day_outcome.category.value == "acting_wolf_exiled"
    assert record.execution_success is False
    assert "CONSEQUENCE" in f.ledger.snapshot()["games"]["game-1"]


def test_missing_continuation_halts_on_phase_advance(runtime, monkeypatch, tmp_path):
    f = fixture(runtime, monkeypatch, tmp_path, ProbeStrategy.PROBE_THEN_REDIRECT)
    pre(f)
    f.env.phase = "vote"
    with pytest.raises(ValueError, match="unavailable"):
        f.pilot.after_step(env=f.env, done=False, info={})
    assert f.pilot.records["game-1"].lifecycle[-1] == "STRUCTURAL_FAILURE"


def test_zero_assignment_game_and_intermediate_vote_tie(runtime, monkeypatch, tmp_path):
    f = fixture(runtime, monkeypatch, tmp_path, ProbeStrategy.IMMEDIATE_REDIRECT)
    f.observation["identity"] = "Villager"
    assert pre(f) is None
    f.pilot.after_game(env=f.env, winner="Werewolf")
    assert f.ledger.snapshot()["assignment_count"] == 0
    f = fixture(runtime, monkeypatch, tmp_path / "another", ProbeStrategy.IMMEDIATE_REDIRECT)
    pre(f)
    f.env.phase = "speech_pk"
    f.env.public_events.append({"event_type": "exile_result", "exiled_players": []})
    f.pilot.after_step(env=f.env, done=False, info={})
    assert f.pilot.records["game-1"].day_outcome is None


def test_resume_marks_scheduled_strategy_interrupted_without_replay(runtime, monkeypatch, tmp_path):
    f = fixture(runtime, monkeypatch, tmp_path, ProbeStrategy.PROBE_THEN_REDIRECT)
    pre(f)
    first = f.pilot
    resumed = runtime.OnlineTerminalPilotRunnerV1(plan=first.plan, predictor=first.predictor,
        mapper=first.mapper, backend=first.backend, model_name=first.model_name,
        reference_tables=first.reference_tables, reference_artifact_digest=first.reference_artifact_digest,
        ledger=f.ledger)
    assert "INTERRUPTED" in f.ledger.snapshot()["games"]["game-1"]
    assert not f.ledger.sealable()
    with pytest.raises(ValueError):
        resumed.start_game("game-1")


def test_actual_eval_body_installs_pre_hook_and_preserves_baseline_vote(runtime):
    seen = []
    class Agent:
        def reset(self):
            pass
        def act_with_pre_speech_belief(self, observation, *, pre_speech_belief):
            return {"type": "speech", "content": "baseline speech"}
        def act(self, observation):
            return {"type": "vote", "content": "player3"}
    class Env:
        phase = "speech"
        public_events = []
        def reset(self, *, roles):
            return {"current_act_idx": 1}
        def step(self, action):
            seen.append((self.phase, action))
            was_vote = self.phase == "vote"
            self.phase = "vote"
            return {"current_act_idx": 1}, 0, was_vote, {"Werewolf": 1}
    env = Env()
    recorder = SimpleNamespace(start=lambda *a, **kw: None,
        before_agent_act=lambda *a, **kw: SimpleNamespace(boundary_id="pre-1") if env.phase == "speech" else None,
        after_agent_act=lambda *a: None, after_env_step=lambda *a, **kw: None,
        finish=lambda *a, **kw: None)
    audit = SimpleNamespace(action_context=lambda **kw: nullcontext(),
                            speech_perception_context=lambda **kw: nullcontext())
    hooks = []
    pilot = SimpleNamespace(handle_pre=lambda **kw: hooks.append("PRE"),
            after_step=lambda **kw: hooks.append("STEP"), after_game=lambda **kw: hooks.append("END"))
    assert runtime.run_canonical_game(env, (Agent(),), ("Werewolf",),
        canonical_recorder=recorder, call_audit=audit, online_pilot=pilot) == "Werewolf win"
    assert hooks == ["PRE", "STEP", "STEP", "END"]
    assert seen == [("speech", {"type": "speech", "content": "baseline speech"}),
                    ("vote", {"type": "vote", "content": "player3"})]


def test_t3_backend_sequences_continue_original_assignment(runtime):
    from tests.phase2.test_online_backend_audit import FakeCanonicalAudit, FakeAuditedBackend
    from werewolf.phase2_backend_audit import Phase2BackendCallAuditV1
    from werewolf.phase2_treatment import build_phase2_treatment
    op = public_opportunity()
    treatment = build_phase2_treatment(op, Action.REDIRECT, assignment_source="strategy_continuation",
                                      assignment_probability=1, randomization_key="parent-bound-fixture")
    canonical = FakeCanonicalAudit()
    audit = Phase2BackendCallAuditV1(op, treatment, canonical, pilot_id="probe-fixture",
                                    assignment_id="a" * 64, sequence_offset=4)
    backend = FakeAuditedBackend(canonical, ["本轮投player6。"])
    audit.actor_call(backend=backend, model="fixture", prompt="public actor requirements",
                    failure_reason=None, temperature=0, max_tokens=128)
    assert audit.records[0].sequence == 5
    assert audit.records[0].assignment_id == "a" * 64
    assert audit.records[0].attempt_index == 1


def test_initial_q_failure_blocks_campaign_instead_of_sampling_another_game(runtime, monkeypatch, tmp_path):
    f = fixture(runtime, monkeypatch, tmp_path, ProbeStrategy.PROBE_THEN_REDIRECT)
    f.pilot.predictor.predict = lambda pre: (_ for _ in ()).throw(ValueError("initial Q missing"))
    with pytest.raises(ValueError, match="initial Q missing"):
        pre(f)
    snapshot = f.ledger.snapshot()
    assert snapshot["assignment_count"] == 0
    failure = snapshot["games"]["game-1"]["PREPARATION_FAILURE"]
    assert failure["candidate_selection"]["candidate_j"] == "player2"
    assert not f.executions
    assert f.ledger.status() == "INCOMPLETE" and not f.ledger.sealable()
    recovered = OnlinePilotAssignmentLedgerV1(f.ledger.path, plan=f.pilot.plan, source_commit="a" * 40)
    assert recovered.status() == "INCOMPLETE"
    f.pilot._active_game_id = None
    with pytest.raises(ValueError, match="target or safety cap"):
        f.pilot.start_game("another-game")
