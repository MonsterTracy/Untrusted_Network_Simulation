"""Actual server resume input guards without optional model/runtime assembly.

Compile the production server entry points unchanged. Qualification evidence,
frozen artifact loading and worker startup are isolated; input equality and
the real durable Probe ledger constructor remain the production implementations.
"""

import ast
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from scripts.qwen3_gameplay_predictor import FIT_DIGEST, SEAL_DIGEST
from tests.phase2.test_outcome_and_execution import values
from tests.phase2.test_probe_plan_provenance import (
    SOURCE_COMMIT, probe_inputs, server_input_scope, shared_admission_fixture,
)
from tests.phase2.test_probe_endpoint_preflight import assess_bound_ledger
from tests.phase2.test_probe_policy_runner import (
    canonical_pre, fixture, next_pre, observe, pre, runtime,
)
from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection.attempt_ledger import construct_collection_plan
from werewolf.phase2_online_ledger import OnlinePilotAssignmentLedgerV1, OnlinePilotLedgerError
from werewolf.phase2_online_plan import ProbeStrategy
from werewolf.phase2_outcome import reference_tables_digest
from werewolf.runtime_config import normalize_runtime_config


ROOT = Path(__file__).resolve().parents[2]


def literal_constants(path, names):
    """Read production identity constants without importing optional servers."""
    result = {}
    for node in ast.parse((ROOT / path).read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    result[target.id] = ast.literal_eval(node.value)
    assert set(result) == set(names)
    return result


@pytest.fixture
def resumed_inputs(server_input_scope, tmp_path, monkeypatch):
    scope = server_input_scope
    repo, plan, _, profile = probe_inputs(tmp_path)
    from scripts.phase2_online_campaign import PROBE_PROFILES

    args = SimpleNamespace(
        repo=repo, plan=tmp_path / "pilot-plan.json", game_plan=tmp_path / "game-plan.json",
        destination=tmp_path / scope["PROBE_QUALIFICATION_NAME"],
        work_directory=tmp_path / "work", q_python=sys.executable,
        campaign_purpose=plan.campaign_purpose, source_commit=SOURCE_COMMIT,
        mapper_manifest_digest="b" * 64, smoke_v3_manifest_digest="d" * 64,
        reference_tables_digest=reference_tables_digest(values()),
        resume=False, preflight_only=False,
    )
    for key in ("runtime_config", "deployment_config", "publication", "evaluation_root",
                "mapper", "smoke_v3", "q_checkout", "q_fit"):
        setattr(args, key, tmp_path / key)
    args.runtime_config.write_bytes((ROOT / "configs/runtime/local-qwen35-9b.yaml").read_bytes())
    args.deployment_config.write_bytes((ROOT / "configs/deployment/qwen35-9b.yaml").read_bytes())
    config = normalize_runtime_config(yaml.safe_load(args.runtime_config.read_bytes()))
    deployment = yaml.safe_load(args.deployment_config.read_bytes())
    for key in ("plan", "game_plan", "work_directory", "destination"):
        profile[key] = str(getattr(args, key))

    # Preserve the production canonical-plan identity comparison as well as the
    # full server loader. Only collect_games' optional module imports are omitted.
    collection = ModuleType("scripts.collect_games")
    constant_sources = {
        "werewolf/agents/prompt_template_v0.py": ("STRICT_CLASSIC7_GAMEPLAY_PROMPT_PROFILE",),
        "werewolf/agents/gpt_agent.py": ("GAMEPLAY_GENERATION_MAX_ATTEMPTS",),
        "werewolf/speech/private_belief_perceiver.py": ("LABEL_GENERATION_MAX_ATTEMPTS",),
        "werewolf/speech/speech_perceiver.py": ("SPEECH_PARSER_GENERATION_MAX_ATTEMPTS",),
        "werewolf/canonical_collection/speech.py": (
            "V1_SPEECH_PARSER_VERSION", "V1_SPEECH_PROMPT_VERSION", "V1_ANNOTATION_SCHEMA_VERSION"),
        "werewolf/canonical_collection/public_history.py": ("PUBLIC_EVENT_SCHEMA_VERSION",),
        "werewolf/canonical_collection/pre.py": ("AUTHORITATIVE_PRE_PREFIX_SCHEMA_VERSION",),
        "werewolf/canonical_collection/trajectory_evidence.py": ("BELIEF_OBSERVATION_SCHEMA_VERSION",),
        "werewolf/canonical_collection/game_bundle.py": ("CANONICAL_GAME_BUNDLE_SCHEMA_VERSION",),
    }
    for path, names in constant_sources.items():
        collection.__dict__.update(literal_constants(path, names))
    definitions = [node for node in ast.parse((ROOT / "scripts/collect_games.py").read_text()).body
                   if isinstance(node, ast.FunctionDef) and node.name == "plan_fields"]
    exec(compile(ast.Module(body=definitions, type_ignores=[]),
                 "scripts/collect_games.py", "exec"), collection.__dict__)
    monkeypatch.setitem(sys.modules, collection.__name__, collection)

    provenance = {
        "served_model_name": deployment["served-model-name"],
        "runtime_config_sha256": sha256_bytes(canonical_json_bytes(config)),
        "serve_config_sha256": sha256_bytes(args.deployment_config.read_bytes()),
        "probe_policy_protocol_sha256": profile["probe_protocol_digest"],
        "probe_campaign_profile_digest": sha256_bytes(canonical_json_bytes(profile)),
    }
    game_plan = construct_collection_plan(ordered_seed_pool=(11, 12, 13, 14),
        **collection.plan_fields({"collection_id": plan.pilot_id, "target_games": 2,
            "seed_pool_size": 4, "call_limit": 10000}, SOURCE_COMMIT, provenance))
    args.plan.write_bytes(canonical_json_bytes(plan.to_record()))
    args.game_plan.write_bytes(canonical_json_bytes(game_plan.to_record()))
    profile_path = repo / "configs/phase2" / PROBE_PROFILES["qualification"].name
    profile_path.parent.mkdir(parents=True)
    profile_path.write_bytes(canonical_json_bytes(profile))
    shared_admission_fixture(monkeypatch, args, profile, calls=[])

    tracked = repo / "tracked.py"
    tracked.write_text("pass\n")
    smoke = ModuleType("scripts.run_phase2_language_smoke")
    smoke.__dict__.update(literal_constants("scripts/run_phase2_language_smoke.py", ("VERSION",)))
    smoke.SOURCE_FILES = ("tracked.py",)
    monkeypatch.setitem(sys.modules, smoke.__name__, smoke)
    smoke_source = {"commit": SOURCE_COMMIT, "tracked_worktree_clean": True,
        "staged_tracked_changes": False,
        "source_sha256": {"tracked.py": sha256_bytes(tracked.read_bytes())}}
    definitions = [node for node in ast.parse((ROOT / "werewolf/phase2_online_server.py").read_text()).body
                   if isinstance(node, ast.FunctionDef) and node.name == "_verify_smoke_source"]
    scope.update(__file__=str(repo / "werewolf/phase2_online_server.py"), yaml=yaml,
        canonical_json_bytes=canonical_json_bytes, sha256_bytes=sha256_bytes,
        INPUT_VERSION=literal_constants("werewolf/phase2_online_server.py", ("INPUT_VERSION",))["INPUT_VERSION"],
        FIT_DIGEST=FIT_DIGEST, SEAL_DIGEST=SEAL_DIGEST,
        normalize_runtime_config=normalize_runtime_config,
        freeze_online_source_provenance=lambda path: {"commit": SOURCE_COMMIT},
        load_runtime_mapper=lambda path, expected_manifest_digest: SimpleNamespace(
            artifact_digest=expected_manifest_digest),
        build_development_layer=lambda *args: SimpleNamespace(values=values()),
        reference_tables_digest=reference_tables_digest,
        _smoke_v3_gate=lambda *args: True,
        verify_artifact=lambda *unused, **kwargs: SimpleNamespace(
            manifest_digest=args.smoke_v3_manifest_digest,
            manifest={"source": smoke_source}),
    )
    exec(compile(ast.Module(body=definitions, type_ignores=[]),
                 "werewolf/phase2_online_server.py", "exec"), scope)

    # Produce run_inputs with the complete production loader, never a handcrafted
    # or permissive replacement for the record compared by its resume guard.
    *_, bound = scope["load_server_inputs"](args)
    args.work_directory.mkdir()
    (args.work_directory / "run_inputs.json").write_bytes(canonical_json_bytes(bound))
    ledger = OnlinePilotAssignmentLedgerV1(args.work_directory / "assignment-ledger.jsonl",
        plan=plan, source_commit=SOURCE_COMMIT)
    game_ids = tuple(f"{game_plan.collection_id}-game-{ordinal:06d}-seed-{seed}"
                     for ordinal, seed in enumerate(game_plan.ordered_seed_pool))
    args.resume = True
    startup = []
    def forbidden(*args, **kwargs):
        startup.append("runtime startup")
        raise AssertionError("input drift must be rejected before worker/backend startup")
    scope["Qwen3GameplayPredictorClient"] = forbidden
    scope["load_named_backends"] = forbidden
    return SimpleNamespace(scope=scope, args=args, plan=plan, game_plan=game_plan,
        bound=bound, ledger=ledger, startup=startup, game_ids=game_ids)


@pytest.mark.parametrize("mutation", ["input_pin", "source", "work_path", "plan"])
def test_actual_resume_input_drift_fails_before_startup_without_mutating_ledger(resumed_inputs, mutation):
    f = resumed_inputs
    persisted = deepcopy(f.bound)
    if mutation == "input_pin":
        persisted["q_python"] = str(f.args.repo / "different-python")
    elif mutation == "source":
        persisted["source_commit"] = "b" * 40
    elif mutation == "work_path":
        persisted["paths"]["work_directory"] = str(f.args.repo / "different-work")
    else:
        persisted["pilot_plan"]["assignment_seed"] += 1
    persisted["inputs_digest"] = sha256_bytes(canonical_json_bytes({
        key: value for key, value in persisted.items() if key != "inputs_digest"}))
    inputs_path = f.args.work_directory / "run_inputs.json"
    inputs_path.write_bytes(canonical_json_bytes(persisted))
    ledger_bytes, inputs_bytes = f.ledger.path.read_bytes(), inputs_path.read_bytes()

    with pytest.raises(ValueError, match="exact existing campaign inputs and ledger"):
        f.scope["execute_server_campaign"](f.args)
    assert f.startup == []
    assert f.ledger.path.read_bytes() == ledger_bytes
    assert inputs_path.read_bytes() == inputs_bytes
    assert not f.args.destination.exists()


def test_actual_resume_accepts_exact_persisted_inputs_without_mutating_ledger(resumed_inputs):
    f = resumed_inputs
    ledger_bytes = f.ledger.path.read_bytes()
    loaded_plan, loaded_game, *_, bound = f.scope["load_server_inputs"](f.args)
    assert loaded_plan == f.plan and loaded_game == f.game_plan
    assert bound == f.bound
    recovered = OnlinePilotAssignmentLedgerV1(f.ledger.path,
        plan=loaded_plan, source_commit=bound["source_commit"])
    assert recovered.snapshot() == f.ledger.snapshot()
    assert f.startup == [] and recovered.path.read_bytes() == ledger_bytes


def test_actual_server_rejects_tampered_journal_copy_before_runtime_construction(resumed_inputs):
    f = resumed_inputs
    f.ledger.start_game(f.game_ids[0])
    original = f.ledger.path.read_bytes()
    tampered = original.replace(f.game_ids[0].encode(), b"changed-game-identity")
    assert tampered != original
    f.ledger.path.write_bytes(tampered)
    startup = []
    def worker(**kwargs):
        startup.append("worker handshake")
        return nullcontext(None)
    def backends(*args, **kwargs):
        startup.append("backend assembly")
        return {}
    f.scope.update(Qwen3GameplayPredictorClient=worker, load_named_backends=backends,
        OnlinePilotAssignmentLedgerV1=OnlinePilotAssignmentLedgerV1, tempfile=tempfile,
        ServerRuntimeFactory=lambda **kwargs: pytest.fail("corrupt ledger reached runtime factory"))

    with pytest.raises(OnlinePilotLedgerError, match="hash chain"):
        f.scope["execute_server_campaign"](f.args)
    assert startup == ["worker handshake", "backend assembly"]
    assert f.ledger.path.read_bytes() == tampered
    assert not f.args.destination.exists()


@pytest.mark.parametrize("preflight_only", [True, False])
def test_actual_server_endpoint_resume_previews_then_starts_next_planned_seed(
        resumed_inputs, runtime, monkeypatch, tmp_path, preflight_only):
    """Production input/resume/campaign bodies and Probe writes, no LLM assembly.

    Optional Q/backend startup and simulator construction use local capabilities.
    The real preflight assesses the journal; the real campaign chooses its next
    game. A scripted run_online_game stops immediately after the actual runner
    durably starts that game, before any new candidate draw or backend call.
    """
    server = resumed_inputs
    trace = fixture(runtime, monkeypatch, tmp_path / "capabilities",
                    ProbeStrategy.PROBE_THEN_REDIRECT)
    original_builder = runtime.build_phase2_decision_opportunity
    monkeypatch.setattr(runtime, "build_phase2_decision_opportunity",
        lambda context, candidate, *args, **kwargs: replace(
            original_builder(context, candidate, *args, **kwargs), legal_context=context))
    monkeypatch.setattr(runtime, "prepare_phase2_intervention_speech",
        lambda *args, **kwargs: SimpleNamespace(verified=SimpleNamespace(
            success=True, attempt_count=1, failure_reason=None,
            language_audit=SimpleNamespace(structured_execution_valid=True,
                                          canonical_bytes=lambda: b"planned-endpoint-audit"))))
    old_pilot = trace.pilot
    reference = values()
    trace.pilot = runtime.OnlineTerminalPilotRunnerV1(
        plan=server.plan, predictor=old_pilot.predictor, mapper=old_pilot.mapper,
        backend=old_pilot.backend, model_name=old_pilot.model_name,
        reference_tables=reference, reference_artifact_digest=reference_tables_digest(reference),
        ledger=server.ledger)
    trace.ledger = server.ledger
    old_game, next_game = server.game_ids[:2]
    prefix = canonical_pre(game_id=old_game)
    trace.recorder.game_id, trace.recorder._pending.prefix = old_game, prefix
    trace.handoff.boundary_id, trace.handoff.prefix_digest, trace.handoff.observer_id = (
        prefix.boundary_id, prefix.prefix_digest, prefix.current_speaker)
    trace.pilot.start_game(old_game)
    assert pre(trace) is not None
    if trace.pilot.records[old_game].assignment.strategy is ProbeStrategy.PROBE_THEN_REDIRECT:
        observe(trace)
        next_pre(trace)
        assert pre(trace) is not None
    trace.env.phase = "night"
    trace.env.public_events.append({"event_type": "exile_result", "exiled_players": ["player2"]})
    trace.pilot.after_step(env=trace.env, done=False, info={})
    before = server.ledger.snapshot()
    old_stage = before["games"][old_game]
    assert old_stage["STRATEGY_STAGE"]["record"]["lifecycle"][-1] == "DAY_CONSEQUENCE_RECORDED"
    assert "CONSEQUENCE" in old_stage and "GAME_RESULT" not in old_stage
    original = server.ledger.path.read_bytes()
    model_activity = (tuple(trace.q_calls), tuple(trace.executions))
    calls, started = [], []

    def no_draw(*args, **kwargs):
        raise AssertionError("recovery and next GAME_STARTED must not redraw an assignment")
    monkeypatch.setattr(runtime, "select_probe_candidate", no_draw)
    monkeypatch.setattr(runtime.OnlinePilotGameStateV1, "try_assign", no_draw)

    class Factory:
        def __init__(self, *, args, plan, game_plan, config, call_limit, mapper,
                     reference, predictor, backends, ledger, work_directory):
            self.ledger, self.work = ledger, work_directory
            self.game_ids = tuple(f"{game_plan.collection_id}-game-{ordinal:06d}-seed-{seed}"
                                 for ordinal, seed in enumerate(game_plan.ordered_seed_pool))
            assert self.game_ids == server.game_ids

        def __call__(self, game_id, *, persist_claim=True, publication_only=False):
            assert game_id == next_game and not publication_only
            temporary = self.work != server.args.work_directory
            stage = self.ledger.snapshot()["games"][old_game]
            assert "POST_ENDPOINT_TAIL_INTERRUPTED" in stage
            assert stage["ASSIGNMENT"] == old_stage["ASSIGNMENT"]
            if temporary:
                assert server.ledger.path.read_bytes() == original
            pilot = runtime.OnlineTerminalPilotRunnerV1(
                plan=server.plan, predictor=trace.pilot.predictor, mapper=trace.pilot.mapper,
                backend=trace.pilot.backend, model_name=trace.pilot.model_name,
                reference_tables=reference, reference_artifact_digest=reference_tables_digest(reference),
                ledger=self.ledger)
            report = assess_bound_ledger(runtime, monkeypatch, tmp_path, trace, self.ledger,
                expected_ledger_path=self.work / "assignment-ledger.jsonl")
            assert report.assignment_ledger_bound and report.ready and report.blockers == ()
            calls.append((temporary, game_id, persist_claim, report))
            return runtime.OnlinePilotRuntimeBundleV1(
                SimpleNamespace(), (), (), SimpleNamespace(game_id=game_id),
                SimpleNamespace(), pilot, report)

    class StopAfterNextSeed(RuntimeError):
        pass

    def scripted_game(env, agents, roles, *, recorder, call_audit, pilot, preflight):
        assert preflight.assignment_ledger_bound and preflight.ready
        pilot.start_game(recorder.game_id)
        started.append(recorder.game_id)
        raise StopAfterNextSeed("planned next seed durably started")

    monkeypatch.setattr(runtime, "run_online_game", scripted_game)
    server.scope.update(Qwen3GameplayPredictorClient=lambda **kwargs: nullcontext(None),
        load_named_backends=lambda *args, **kwargs: {},
        OnlinePilotAssignmentLedgerV1=OnlinePilotAssignmentLedgerV1, tempfile=tempfile,
        ServerRuntimeFactory=Factory, run_online_campaign=runtime.run_online_campaign)
    server.args.preflight_only = preflight_only
    if preflight_only:
        assert server.scope["execute_server_campaign"](server.args) == 0
        assert [(temporary, game, persist) for temporary, game, persist, _ in calls] == [
            (True, next_game, False)]
        assert started == []
        assert server.ledger.path.read_bytes() == original
        assert server.ledger.snapshot() == before
    else:
        with pytest.raises(StopAfterNextSeed, match="durably started"):
            server.scope["execute_server_campaign"](server.args)
        assert [(temporary, game, persist) for temporary, game, persist, _ in calls] == [
            (True, next_game, False), (False, next_game, False), (False, next_game, True)]
        assert started == [next_game]
        after = server.ledger.snapshot()
        assert server.ledger.path.read_bytes().startswith(original)
        assert [row["kind"] for row in after["events"][len(before["events"]):]] == [
            "POST_ENDPOINT_TAIL_INTERRUPTED", "GAME_STARTED"]
        assert after["games_attempted"] == 2 and after["assignment_count"] == 1
        for stage, payload in old_stage.items():
            assert after["games"][old_game][stage] == payload
        assert "GAME_RESULT" not in after["games"][old_game]
        assert set(after["games"][next_game]) == {"GAME_STARTED"}
        assert next_game.endswith(f"-seed-{server.game_plan.ordered_seed_pool[1]}")
    assert (tuple(trace.q_calls), tuple(trace.executions)) == model_activity
    assert not server.args.destination.exists()
