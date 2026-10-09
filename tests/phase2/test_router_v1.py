"""Router policies use the real canonical PRE and the existing Probe executor.

No rollout or LLM: only language result/commit fixtures replace external I/O.
"""
from dataclasses import replace
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tests.phase2.test_probe_policy_runner import (
    runtime, fixture, canonical_pre, pre, next_pre, observe,
)
from tests.phase2.test_probe_policy_plan import public_context, public_opportunity
from werewolf.canonical_collection.pre import AuthoritativePREPrefix, construct_authoritative_pre_prefix
from werewolf.canonical_collection.public_history import freeze_public_event_history
from werewolf.canonical_collection.speech import V1AnnotationStatus, construct_v1_speech_annotation
from tests.canonical_collection.test_pre_prefix import _attempt
from werewolf.phase2_actions import Action, context_from_pre
from werewolf.phase2_online_plan import (
    Phase2RouterPolicyV1, ProbeStrategy, OnlinePilotPlanError, assign_router_policy,
    select_probe_candidate, probe_candidate_pool, router_assignment_from_record,
    probe_assignment_from_record, Phase2OnlineProbePilotPlanV1,
)
from werewolf.phase2_online_records import validate_probe_lifecycle_record, Phase2OnlineProbeRecordV1
import werewolf.phase2_online_plan as plans


def policy(strategy=ProbeStrategy.PROBE_THEN_REDIRECT, **changes):
    return replace(Phase2RouterPolicyV1(strategy, "local-shared-sampling", 71, "a" * 40), **changes)


def later_day_pre(prefix):
    """Extend the earlier real prefix causally through day 1 to day 2."""
    events = prefix.public_event_history.to_records()
    for row in events:
        if row["event_type"] != "phase_change":
            row.pop("day")
            row.pop("phase")
    annotations = list(prefix.v1_annotations)
    def event(kind, **fields):
        events.append(dict(event_id=f"event-{len(events)}", event_index=len(events),
                           event_type=kind, **fields))
    for speaker in ("player5", "player6", "player7"):
        if speaker != "player5":
            event("turn_start", speaker=speaker)
        event("public_speech", speaker=speaker, raw_text="我没有新的信息。")
        annotations.append(construct_v1_speech_annotation(
            freeze_public_event_history(events), status=V1AnnotationStatus.NO_ACTION,
            actions=(), attempts=(replace(_attempt(), status=V1AnnotationStatus.NO_ACTION,
                                          raw_response="None"),)))
    event("phase_change", day=1, phase="vote")
    event("vote_result", votes=[dict(voter=p, target=None) for p in prefix.alive_observer_ids])
    event("exile_result", exiled_players=[])
    event("phase_change", day=1, phase="night")
    event("death_announcement", dead_players=[])
    event("phase_change", day=2, phase="discussion")
    event("turn_start", speaker="player1")
    return construct_authoritative_pre_prefix(
        game_id=prefix.game_id, boundary_id="pre-1", step_index=len(events) - 1,
        report_trigger_id="trigger-day2", current_speaker="player1",
        alive_observer_ids=prefix.alive_observer_ids,
        public_event_history=freeze_public_event_history(events), v1_annotations=annotations,
        belief_observation_ids_by_observer={p: f"day2-{p}" for p in prefix.alive_observer_ids})


def router_fixture(runtime, monkeypatch, tmp_path, strategy, phase="speech", invalid_stage=None):
    f = fixture(runtime, monkeypatch, tmp_path, strategy, phase)
    writes = []
    def sink(kind, evidence):
        writes.append((kind, evidence.to_record() if hasattr(evidence, "to_record") else evidence))
    old = f.pilot
    f.pilot = runtime.OnlineTerminalPilotRunnerV1(
        plan=policy(strategy), predictor=old.predictor, mapper=old.mapper,
        backend=old.backend, model_name=old.model_name, reference_tables=old.reference_tables,
        reference_artifact_digest=old.reference_artifact_digest, record_evidence=sink)
    f.pilot.start_game("game-1")
    f.writes = writes
    def prepare(op, treatment, public, *, actor, perceiver):
        stage = "T1" if op.legal_context.acting_wolf == "player1" else "T3"
        # The production path must persist assignment/stage before external I/O.
        assert any(kind == "assignment" for kind, _ in writes)
        expected = "T1_ATTEMPTED" if stage == "T1" else "T3_PREPARED"
        assert writes[-1][1]["lifecycle"][-1] == expected
        f.executions.append(treatment)
        return SimpleNamespace(verified=SimpleNamespace(
            success=stage != invalid_stage, attempt_count=2 if stage == invalid_stage else 1,
            failure_reason="LANGUAGE_INVALID" if stage == invalid_stage else None,
            language_audit=SimpleNamespace(structured_execution_valid=True,
                                           canonical_bytes=lambda: b"router-fixture-audit")))
    monkeypatch.setattr(runtime, "prepare_phase2_intervention_speech", prepare)
    def forbidden(*args, **kwargs):
        raise AssertionError("deployment must not randomize strategy or terminal action")
    monkeypatch.setattr(plans, "assign_probe_strategy", forbidden)
    monkeypatch.setattr(plans, "assign_terminal_action", forbidden)
    return f


@pytest.mark.parametrize("phase", ["speech", "speech_pk"])
def test_shared_candidate_semantics_identity_and_reproducibility(phase):
    context = public_context(phase=phase)
    research, control = policy(), policy(ProbeStrategy.IMMEDIATE_REDIRECT)
    assert research.digest() != control.digest()
    assert research.candidate_sampling_digest() == control.candidate_sampling_digest()
    a, b = select_probe_candidate(research, context), select_probe_candidate(control, context)
    assert a == b == select_probe_candidate(research, context)
    assert a.candidate_pool == probe_candidate_pool(context)
    assert a.candidate_selection_probability == 1 / len(a.candidate_pool)
    # Equal seeds alone do not share the key. PRE/game/plan identity also matters.
    for other in (replace(context, game_id="other-game"),
                  replace(context, boundary_id="other-boundary"),
                  replace(context, prefix_digest="9" * 64)):
        assert select_probe_candidate(research, other).candidate_selection_key != a.candidate_selection_key
    assert select_probe_candidate(replace(research, candidate_sampling_id="different"), context) != a


@pytest.mark.parametrize("strategy", list(ProbeStrategy))
@pytest.mark.parametrize("phase", ["speech", "speech_pk"])
def test_deterministic_assignment_audit_reproduces_and_cannot_load_as_pilot(strategy, phase):
    p = policy(strategy)
    op = public_opportunity(phase=phase)
    selected = select_probe_candidate(p, op.legal_context)
    assigned = assign_router_policy(p, selected, op)
    assert assigned.treatment.assignment_source == "deterministic_policy"
    assert assigned.treatment.assignment_probability == 1
    assert assigned.treatment.action is (Action.PROBE if strategy is ProbeStrategy.PROBE_THEN_REDIRECT else Action.REDIRECT)
    raw = assigned.to_record()
    assert raw["policy"]["policy_id"] == p.policy_id
    assert raw["policy"]["candidate_sampling"]["source_revision"] == "a" * 40
    assert "assignment_seed" not in raw
    assert router_assignment_from_record(raw) == assigned
    raw["policy_digest"] = "0" * 64
    with pytest.raises(OnlinePilotPlanError):
        router_assignment_from_record(raw)
    with pytest.raises(OnlinePilotPlanError):
        probe_assignment_from_record(assigned.to_record(), Phase2OnlineProbePilotPlanV1("pilot", 71, 20, 80))


@pytest.mark.parametrize("field", ["schema_version", "policy_digest"])
def test_execution_validator_rejects_policy_provenance_tampering(field):
    p, op = policy(), public_opportunity()
    assignment = assign_router_policy(p, select_probe_candidate(p, op.legal_context), op)
    raw = Phase2OnlineProbeRecordV1(assignment).to_record()
    raw["assignment"][field] = "phase2_online_probe_assignment_v1"
    with pytest.raises(OnlinePilotPlanError):
        validate_probe_lifecycle_record(raw)


@pytest.mark.parametrize("strategy", list(ProbeStrategy))
def test_terminal_only_does_not_consume_later_probe_opportunity(runtime, monkeypatch, tmp_path, strategy):
    f = router_fixture(runtime, monkeypatch, tmp_path, strategy)
    # Last alive wolf at the end of day 1's queue: J >= 2, but B empty.
    next_pre(f, "player5", boundary="earlier-terminal-only")
    context = context_from_pre(f.recorder._pending.prefix, frozenset(("player1", "player5")))
    assert len(context.legal_targets) >= 2 and not probe_candidate_pool(context)
    assert pre(f) is None
    assert not f.pilot.state.assignments and not f.q_calls and not f.executions
    # A later day's queue starts at the other wolf, preserving a real PRE.
    prefix = later_day_pre(f.recorder._pending.prefix)
    f.recorder._pending.prefix = prefix
    f.handoff.boundary_id, f.handoff.prefix_digest, f.handoff.observer_id = (
        prefix.boundary_id, prefix.prefix_digest, prefix.current_speaker)
    f.observation["current_act_idx"] = 1
    f.initial = replace(f.initial, legal_context=context_from_pre(
        prefix, frozenset(("player1", "player5"))))
    monkeypatch.setattr(runtime, "build_phase2_decision_opportunity", lambda *a, **kw: f.initial)
    assert prefix.public_temporal_state.day == 2
    assert type(f.recorder._pending.prefix) is AuthoritativePREPrefix
    assert pre(f) is not None
    assert len(f.pilot.state.assignments) == 1
    if strategy is ProbeStrategy.IMMEDIATE_REDIRECT:
        assert pre(f) is None
    assert sum(kind == "assignment" for kind, _ in f.writes) == 1


@pytest.mark.parametrize("phase", ["speech", "speech_pk"])
def test_research_reuses_t3_fixed_candidate_actor_fresh_panel(runtime, monkeypatch, tmp_path, phase):
    f = router_fixture(runtime, monkeypatch, tmp_path, ProbeStrategy.PROBE_THEN_REDIRECT, phase)
    assert pre(f) is not None
    original = f.pilot.records["game-1"].assignment
    observe(f)
    next_pre(f)
    assert pre(f) is not None
    record = f.pilot.records["game-1"]
    assert f.q_calls == ["pre-1", "pre-3"]
    assert record.t3_opportunity.candidate_j == original.opportunity.candidate_j
    assert record.t3_opportunity.legal_context.acting_wolf == original.opportunity.continuation_actor
    assert record.t3_treatment.plan.redirect_target == ("player7" if phase == "speech" else "player6")
    assert record.t3_treatment.assignment_source == "strategy_continuation"
    assert len(f.pilot.state.assignments) == 1 and pre(f) is None
    validate_probe_lifecycle_record(record.to_record())


@pytest.mark.parametrize("strategy", list(ProbeStrategy))
def test_invalid_language_is_a_recorded_failure_not_not_applicable(runtime, monkeypatch, tmp_path, strategy):
    f = router_fixture(runtime, monkeypatch, tmp_path, strategy, invalid_stage="T1")
    assert pre(f) is None  # Canonical hook resumes baseline for language invalid.
    record = f.pilot.records["game-1"]
    assert record.execution_success is False
    if strategy is ProbeStrategy.PROBE_THEN_REDIRECT:
        assert "T3_CANCELLED" in record.lifecycle
    next_pre(f)
    assert pre(f) is None and len(f.executions) == 1
    assert len(f.pilot.state.assignments) == 1
    f.env.public_events.append({"event_type": "exile_result", "exiled_players": ["player6"]})
    f.pilot.after_step(env=f.env, done=False, info={})
    assert f.pilot.records["game-1"].day_outcome is not None


def test_structural_failure_preserved_and_raised(runtime, monkeypatch, tmp_path):
    f = router_fixture(runtime, monkeypatch, tmp_path, ProbeStrategy.PROBE_THEN_REDIRECT)
    pre(f)
    observe(f)
    next_pre(f, phase="speech_pk")
    with pytest.raises(ValueError, match="phase/boundary mismatch"):
        pre(f)
    assert f.pilot.records["game-1"].lifecycle[-1] == "STRUCTURAL_FAILURE"
    assert f.writes[-1][1]["structural_failure_reason"]
    with pytest.raises(ValueError, match="structurally failed"):
        pre(f)


def test_t3_invalid_preserves_original_assignment_and_no_further_action(runtime, monkeypatch, tmp_path):
    f = router_fixture(runtime, monkeypatch, tmp_path, ProbeStrategy.PROBE_THEN_REDIRECT,
                       invalid_stage="T3")
    pre(f)
    assigned = f.pilot.state.assignments["game-1"]
    observe(f)
    next_pre(f)
    assert pre(f) is None
    record = f.pilot.records["game-1"]
    assert "T3_LANGUAGE_INVALID" in record.lifecycle and record.execution_success is False
    assert f.pilot.state.assignments["game-1"] is assigned
    assert pre(f) is None and len(f.executions) == 2


def test_no_opportunity_no_phase2_calls_or_assignment(runtime, monkeypatch, tmp_path):
    f = router_fixture(runtime, monkeypatch, tmp_path, ProbeStrategy.IMMEDIATE_REDIRECT)
    next_pre(f, "player5")
    assert pre(f) is None
    f.pilot.after_step(env=f.env, done=False, info={})
    f.pilot.after_game(env=f.env, winner="Villager")
    assert f.q_calls == [] and f.executions == [] and f.pilot.records == {}
    assert [kind for kind, _ in f.writes] == ["game-start"]
    with pytest.raises(ValueError, match="replay"):
        f.pilot.start_game("game-1")


def test_evidence_sink_required_and_failure_blocks_language(runtime, monkeypatch, tmp_path):
    f = router_fixture(runtime, monkeypatch, tmp_path, ProbeStrategy.PROBE_THEN_REDIRECT)
    with pytest.raises(ValueError):
        runtime.OnlineTerminalPilotRunnerV1(
            plan=f.pilot.plan, predictor=f.pilot.predictor, mapper=f.pilot.mapper,
            backend=f.pilot.backend, model_name=f.pilot.model_name,
            reference_tables=f.pilot.reference_tables,
            reference_artifact_digest=f.pilot.reference_artifact_digest)
    def unavailable(kind, evidence):
        if kind == "assignment":
            raise OSError("evidence write failed")
        f.writes.append((kind, evidence))
    f.pilot.record_evidence = unavailable
    with pytest.raises(OSError, match="write failed"):
        pre(f)
    assert not f.pilot.state.assignments and not f.executions
    assert f.writes[-1][0] == "preparation-failure"


def test_randomized_campaign_entry_does_not_admit_router(runtime, monkeypatch, tmp_path):
    f = router_fixture(runtime, monkeypatch, tmp_path, ProbeStrategy.IMMEDIATE_REDIRECT)
    runtime.require_online_gameplay_hook()
    with pytest.raises(TypeError, match="not randomized pilot preflight"):
        runtime.run_online_game(f.env, (), (), recorder=f.recorder,
                               call_audit=f.call_audit, pilot=f.pilot, preflight=None)


@pytest.mark.parametrize("eligible", [False, True])
def test_actual_production_eval_installs_router_hook_and_baseline_vote(runtime, monkeypatch, tmp_path, eligible):
    f = router_fixture(runtime, monkeypatch, tmp_path, ProbeStrategy.IMMEDIATE_REDIRECT)
    if not eligible:
        next_pre(f, "player5")
    seen = []
    class Agent:
        def reset(self):
            pass
        def act_with_pre_speech_belief(self, observation, *, pre_speech_belief):
            return {"type": "speech", "content": "baseline speech"}
        def act(self, observation):
            return {"type": "vote", "content": "player6"}
    def step(action):
        seen.append((f.env.phase, action))
        was_vote = f.env.phase == "vote"
        f.env.phase = "vote"
        if was_vote:
            f.env.public_events.append({"event_type": "exile_result", "exiled_players": ["player6"]})
        return f.observation, 0, was_vote, {"Werewolf": 1}
    f.env.reset = lambda **kw: f.observation
    f.env.step = step
    f.recorder.start = lambda *a, **kw: None
    f.recorder.before_agent_act = lambda *a, **kw: f.handoff if f.env.phase == "speech" else None
    f.recorder.after_agent_act = lambda *a, **kw: None
    f.recorder.after_env_step = lambda *a, **kw: None
    f.recorder.finish = lambda *a, **kw: None
    f.call_audit.action_context = lambda **kw: nullcontext()
    f.call_audit.speech_perception_context = lambda **kw: nullcontext()
    commit = runtime.commit_phase2_verified_speech
    def commit_then_vote(**kwargs):
        result = commit(**kwargs)
        f.env.phase = "vote"
        return result
    monkeypatch.setattr(runtime, "commit_phase2_verified_speech", commit_then_vote)
    assert runtime.run_canonical_game(f.env, tuple(Agent() for _ in range(7)), (),
        canonical_recorder=f.recorder, call_audit=f.call_audit, online_pilot=f.pilot) == "Werewolf win"
    assert seen[-1] == ("vote", {"type": "vote", "content": "player6"})
    if eligible:
        assert len(seen) == 1
        assert f.pilot.records["game-1"].lifecycle[-1] == "GAME_RESULT_RECORDED"
    else:
        assert seen[0] == ("speech", {"type": "speech", "content": "baseline speech"})
        assert not f.pilot.state.assignments and not f.q_calls


def test_backend_sidecar_persisted_with_policy_and_original_assignment(runtime, monkeypatch, tmp_path):
    from tests.phase2.test_online_backend_audit import FakeCanonicalAudit, FakeAuditedBackend
    f = router_fixture(runtime, monkeypatch, tmp_path, ProbeStrategy.PROBE_THEN_REDIRECT)
    f.call_audit = FakeCanonicalAudit()
    f.pilot.backend = FakeAuditedBackend(f.call_audit, ["T1 scripted text", "T3 scripted text"])
    prepare = runtime.prepare_phase2_intervention_speech
    def with_backend_call(op, treatment, public, *, actor, perceiver):
        result = prepare(op, treatment, public, actor=actor, perceiver=perceiver)
        actor.call_audit.actor_call(backend=actor.backend, model="fixture", prompt="scripted requirements",
                                   failure_reason=None, temperature=0, max_tokens=128)
        return result
    monkeypatch.setattr(runtime, "prepare_phase2_intervention_speech", with_backend_call)
    pre(f)
    observe(f)
    next_pre(f)
    pre(f)
    assigned = f.pilot.state.assignments["game-1"]
    calls = [raw for kind, raw in f.writes if kind == "backend-call"]
    assert len(calls) == 2
    assert [raw["sequence"] for raw in calls] == [1, 2]
    assert all(raw["pilot_id"] == assigned.policy.policy_id and
               raw["assignment_id"] == assigned.digest() for raw in calls)
