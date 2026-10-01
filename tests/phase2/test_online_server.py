"""Real server assembly with scripted dependencies, never live model calls."""

from copy import deepcopy
from dataclasses import replace
import inspect
from pathlib import Path
import json
import re
import subprocess
from types import SimpleNamespace

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("openai")
pytest.importorskip("torch")

import yaml

from scripts.run_phase2_online_intervention_pilot import build_parser
from scripts.collect_games import plan_fields
import scripts.run_phase2_language_smoke as smoke
from tests.phase2.test_decision_opportunity import Mapper, q_matrix
from tests.phase2.test_intervention_risk import values
from tests.phase2.test_online_ledger import _game_assignment
from werewolf.artifact_io import canonical_json_bytes, canonical_jsonl_bytes, publish_artifact, sha256_bytes
from werewolf.canonical_collection.attempt_ledger import construct_collection_plan
from werewolf.phase2_mapper_final import FINAL_VERSION
from werewolf.phase2_online_plan import Phase2OnlineTerminalPilotPlanV1
from werewolf.phase2_online_records import Phase2OnlineInterventionRecordV1, record_online_execution
from werewolf.phase2_outcome import reference_tables_digest
import werewolf.phase2_mapper_runtime as mapper_module
import werewolf.phase2_online_preflight as preflight
import werewolf.phase2_online_runner as runner
import werewolf.phase2_online_server as server
from run_random import eval as production_eval


ROOT = Path(__file__).resolve().parents[2]


def _git(path, *args):
    return subprocess.check_output(("git", "-C", str(path), *args), text=True).strip()


def _setup(tmp_path, monkeypatch, purpose="qualification"):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "tracked.py").write_text("pass\n")
    for name in smoke.SOURCE_FILES:
        destination = repo / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    head = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(preflight, "SOURCE_FILES", ("tracked.py",))
    monkeypatch.setattr(server, "_require_package_source", lambda _: None)
    config_path = ROOT / "configs/runtime/local-qwen35-9b.yaml"
    deployment_path = ROOT / "configs/deployment/qwen35-9b.yaml"
    config = server.normalize_runtime_config(yaml.safe_load(config_path.read_bytes()))
    count = 10 if purpose == "qualification" else 120
    plan = Phase2OnlineTerminalPilotPlanV1(
        f"test-{purpose}", 17, campaign_purpose=purpose, max_games_attempted=count + 2)
    fields = plan_fields({"collection_id": plan.pilot_id, "target_games": 1,
                          "seed_pool_size": count + 2, "call_limit": 10000}, head,
        {"served_model_name": "qwen35-9b",
         "runtime_config_sha256": sha256_bytes(canonical_json_bytes(config)),
         "serve_config_sha256": sha256_bytes(deployment_path.read_bytes())})
    game_plan = construct_collection_plan(ordered_seed_pool=tuple(range(101, 103 + count)), **fields)
    plan_path, game_path = tmp_path / "pilot-plan.json", tmp_path / "game-plan.json"
    plan_path.write_bytes(canonical_json_bytes(plan.to_record()))
    game_path.write_bytes(canonical_json_bytes(game_plan.to_record()))
    mapper_artifact = publish_artifact(tmp_path / "mapper", manifest_fields={
        "artifact_type": "phase2_full_development_oof_mapper", "schema_version": FINAL_VERSION},
        files={"fixture.json": b"{}"})
    mapper = Mapper()
    mapper.artifact_digest = mapper_artifact.manifest_digest
    mapper.model_digest = "f" * 64
    def load_mapper(path, *, expected_manifest_digest):
        if expected_manifest_digest != mapper.artifact_digest:
            raise ValueError("mapper mismatch")
        return mapper
    monkeypatch.setattr(server, "load_runtime_mapper", load_mapper)
    monkeypatch.setattr(mapper_module, "load_runtime_mapper", load_mapper)
    monkeypatch.setattr(server, "build_development_layer", lambda *_: SimpleNamespace(values=values()))
    cases = [{"game_id": f"g{i}", "boundary_id": f"b{i}"} for i in range(58)]
    selection_digest = sha256_bytes(canonical_json_bytes(cases))
    monkeypatch.setattr(smoke, "EXPECTED_CASE_SELECTION_DIGEST", selection_digest)
    from werewolf.phase2_actions import CONTRACT_VERSION
    from werewolf.phase2_language import LANGUAGE_VERSION
    smoke_fields = {"artifact_type": "phase2_language_execution_smoke", "schema_version": smoke.VERSION,
        "study_name": smoke.NAME, "action_contract_version": CONTRACT_VERSION,
        "language_semantic_version": LANGUAGE_VERSION, "final_mapper_manifest_digest": mapper.artifact_digest,
        "case_selection": {"digest": selection_digest, "count": 58},
        "source": {"commit": "1" * 40, "tracked_worktree_clean": True,
                   "staged_tracked_changes": False,
                   "source_sha256": {name: sha256_bytes((ROOT / name).read_bytes())
                                     for name in smoke.SOURCE_FILES}}}
    def make_smoke(path, passed):
        return publish_artifact(path, manifest_fields=smoke_fields, files={
            "selected_cases.json": canonical_json_bytes(cases),
            "case_executions.jsonl": canonical_jsonl_bytes([{}] * 58),
            "metrics.json": canonical_json_bytes({"overall": {"case_count": 58},
                "gate": {"passed": passed, "checks": {"valid": passed}}})})
    smoke_artifact = make_smoke(tmp_path / "smoke", True)
    counters = {"q_start": 0, "q_predict": 0, "language": 0, "closed": 0}
    class Predictor:
        fit_digest = server.FIT_DIGEST
        seal_digest = server.SEAL_DIGEST
        def __init__(self, **kwargs):
            counters["q_start"] += 1
        def __enter__(self):
            return self
        def __exit__(self, *_):
            counters["closed"] += 1
        def predict(self, *_):
            counters["q_predict"] += 1
            raise AssertionError("no model inference in a CLI/preflight test")
    monkeypatch.setattr(server, "Qwen3GameplayPredictorClient", Predictor)
    monkeypatch.setattr(preflight, "_frozen_q_runtime_ready", lambda p: type(p) is Predictor)
    def no_language(**kwargs):
        counters["language"] += 1
        raise AssertionError("no live or scripted language dispatch in CLI assembly tests")
    raw = SimpleNamespace(chat=no_language, chat_with_metadata=no_language,
                          supports_json_schema=True, client=SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(server, "load_named_backends", lambda *_args, **_kwargs: {"local_qwen": raw})
    args = build_parser().parse_args([
        "--campaign-purpose", purpose, "--plan", str(plan_path), "--game-plan", str(game_path),
        "--runtime-config", str(config_path), "--deployment-config", str(deployment_path),
        "--publication", str(tmp_path / "publication"), "--evaluation-root", str(tmp_path / "evaluation"),
        "--mapper", str(mapper_artifact.path), "--mapper-manifest-digest", mapper.artifact_digest,
        "--reference-tables-digest", reference_tables_digest(values()),
        "--smoke-v3", str(smoke_artifact.path), "--smoke-v3-manifest-digest", smoke_artifact.manifest_digest,
        "--q-checkout", str(tmp_path / "q-checkout"), "--q-fit", str(tmp_path / "q-fit"),
        "--source-commit", head, "--repo", str(repo), "--work-directory", str(tmp_path / "work"),
        "--destination", str(tmp_path / (runner.QUALIFICATION_NAME if count == 10 else runner.ARTIFACT_NAME))])
    return args, plan, counters, make_smoke


def test_full_preflight_only_has_no_game_assignment_inference_or_campaign_files(tmp_path, monkeypatch, capsys):
    args, _, counters, _ = _setup(tmp_path, monkeypatch)
    args.preflight_only = True
    assert server.execute_server_campaign(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["preflight"]["ready"] is True
    assert output["preflight"]["checkpoint_executable"] is False
    assert not args.work_directory.exists() and not args.destination.exists()
    assert counters == {"q_start": 1, "q_predict": 0, "language": 0, "closed": 1}


def test_failed_full_runtime_preflight_prevents_campaign_start(tmp_path, monkeypatch):
    args, _, counters, _ = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(preflight, "_frozen_q_runtime_ready", lambda _: False)
    assert server.execute_server_campaign(args) == 1
    assert not args.work_directory.exists() and not args.destination.exists()
    assert counters["language"] == counters["q_predict"] == 0


def test_missing_production_gameplay_hook_fails_before_campaign_files(tmp_path, monkeypatch, capsys):
    args, _, counters, _ = _setup(tmp_path, monkeypatch)
    # Exact production signature at d6c8bce, rather than a **kwargs mock.
    def old_eval(env, agent_list, roles_, *, canonical_recorder, call_audit,
                 planning_mode="original", plan_provider=None, speech_treatment=None,
                 ablation_trace=None, wolf_speech_tom=None):
        raise AssertionError("preflight must not execute gameplay")
    monkeypatch.setattr(runner, "run_canonical_game", old_eval)
    assert server.execute_server_campaign(args) == 1
    output = json.loads(capsys.readouterr().out)
    assert "PRODUCTION_GAMEPLAY_HOOK_UNAVAILABLE" in output["preflight"]["blockers"]
    assert not args.work_directory.exists() and not args.destination.exists()
    assert counters["language"] == counters["q_predict"] == 0


@pytest.mark.parametrize("fault", ("source", "dirty", "staged", "smoke", "gate", "mapper", "reference", "smoke_source"))
def test_wrong_frozen_input_fails_before_worker_or_language(tmp_path, monkeypatch, fault):
    args, _, counters, make_smoke = _setup(tmp_path, monkeypatch)
    if fault == "source":
        args.source_commit = "0" * 40
    elif fault in ("dirty", "staged"):
        (args.repo / "tracked.py").write_text("dirty\n")
        if fault == "staged":
            _git(args.repo, "add", "tracked.py")
    elif fault == "smoke":
        args.smoke_v3_manifest_digest = "0" * 64
    elif fault == "gate":
        artifact = make_smoke(tmp_path / "failed-smoke", False)
        args.smoke_v3, args.smoke_v3_manifest_digest = artifact.path, artifact.manifest_digest
    elif fault == "mapper":
        args.mapper_manifest_digest = "0" * 64
    elif fault == "smoke_source":
        path = args.repo / "werewolf/phase2_language.py"
        path.write_bytes(path.read_bytes() + b"\n# changed source\n")
        _git(args.repo, "add", "werewolf/phase2_language.py")
        _git(args.repo, "commit", "-qm", "changed language bytes")
        args.source_commit = _git(args.repo, "rev-parse", "HEAD")
        game = server._read_json(args.game_plan)
        from werewolf.canonical_collection.attempt_ledger import collection_plan_from_record
        prior = collection_plan_from_record(game)
        fields = {key: value for key, value in game.items() if key not in (
            "schema_version", "plan_digest", "runtime_provenance_digest", "ordered_seed_pool")}
        fields["environment_provenance"] = dict(prior.environment_provenance)
        fields["source_revision"] = args.source_commit
        changed = construct_collection_plan(ordered_seed_pool=prior.ordered_seed_pool, **fields)
        args.game_plan.write_bytes(canonical_json_bytes(changed.to_record()))
    else:
        args.reference_tables_digest = "0" * 64
    with pytest.raises(ValueError):
        server.execute_server_campaign(args)
    assert counters["q_start"] == counters["language"] == 0
    assert not args.work_directory.exists() and not args.destination.exists()


def test_installed_package_must_match_repo(tmp_path):
    server._require_package_source(ROOT)
    with pytest.raises(ValueError, match="originate"):
        server._require_package_source(tmp_path)


def test_unrelated_untracked_file_does_not_block_full_preflight(tmp_path, monkeypatch):
    args, _, counters, _ = _setup(tmp_path, monkeypatch)
    research = args.repo / "untracked-research.md"
    research.write_text("existing research\n")
    args.preflight_only = True
    assert server.execute_server_campaign(args) == 0
    assert research.read_text() == "existing research\n"
    assert counters["language"] == counters["q_predict"] == 0


def test_fixed_plan_count_and_purpose_cannot_be_overridden(tmp_path, monkeypatch):
    args, plan, _, _ = _setup(tmp_path, monkeypatch)
    assert server.load_online_plan(args.plan, "qualification").target_assignment_count == 10
    with pytest.raises(ValueError):
        server.load_online_plan(args.plan, "pilot")
    raw = plan.to_record()
    raw["target_assignment_count"] = 120
    args.plan.write_bytes(canonical_json_bytes(raw))
    with pytest.raises(ValueError):
        server.load_online_plan(args.plan, "qualification")


def test_destination_and_implicit_restart_fail_closed(tmp_path, monkeypatch):
    args, _, counters, _ = _setup(tmp_path, monkeypatch)
    args.destination.mkdir()
    with pytest.raises(FileExistsError):
        server.execute_server_campaign(args)
    args.destination.rmdir()
    args.work_directory.mkdir()
    with pytest.raises(FileExistsError, match="resume"):
        server.execute_server_campaign(args)
    args.resume = True
    with pytest.raises(ValueError, match="resume"):
        server.execute_server_campaign(args)
    assert counters["q_start"] == counters["language"] == 0


def _script_games(monkeypatch, *, stop_after=None):
    started, publications = [], []
    def game(env, agents, roles, *, recorder, call_audit, pilot, preflight):
        assert preflight.ready and not preflight.publication_only
        pilot.start_game(recorder.game_id)
        assigned = _game_assignment(pilot.plan, recorder.game_id)
        assigned = pilot.state.try_assign(pilot.plan, assigned.selection, assigned.opportunity,
                                           persist=pilot.ledger.persist_assignment)
        record = record_online_execution(Phase2OnlineInterventionRecordV1(assigned),
            language_audit_digest="c" * 64, attempt_count=0, backend_calls=(), failure_reason="SCRIPTED_FAILURE")
        pilot.ledger.persist_execution(record)
        started.append(recorder.game_id)
        if stop_after is not None and len(started) == stop_after:
            raise RuntimeError("scripted interruption")
    monkeypatch.setattr(runner, "run_online_game", game)
    # Orchestrator-only fixtures use no canonical gameplay; the real verifier
    # is separately tested against genuine PRE/call/event evidence.
    monkeypatch.setattr(server.ServerRuntimeFactory, "verify_execution", lambda *_: True)
    def publish(destination, **kwargs):
        publications.append((destination, kwargs["dataset"], kwargs["preflight"]))
        return "f" * 64
    monkeypatch.setattr(server, "publish_online_pilot", publish)
    return started, publications


@pytest.mark.parametrize("purpose,count", (("qualification", 10), ("pilot", 120)))
def test_cli_runs_only_requested_fixed_campaign(tmp_path, monkeypatch, purpose, count):
    args, _, counters, _ = _setup(tmp_path, monkeypatch, purpose)
    started, published = _script_games(monkeypatch)
    assert server.execute_server_campaign(args) == 0
    assert len(started) == count and len(published) == 1
    dataset = published[0][1]
    assert dataset.manifest["campaign_purpose"] == purpose
    assert dataset.manifest["assignment_count"] == dataset.manifest["target_assignment_count"] == count
    assert counters["language"] == counters["q_predict"] == 0


def test_resume_reuses_assignments_and_does_not_replay_old_games(tmp_path, monkeypatch):
    args, _, _, _ = _setup(tmp_path, monkeypatch)
    started, published = _script_games(monkeypatch, stop_after=3)
    with pytest.raises(RuntimeError, match="interruption"):
        server.execute_server_campaign(args)
    before = (args.work_directory / "assignment-ledger.jsonl").read_bytes()
    old_ids = tuple(started)
    assert not published
    args.resume = True
    more, published = _script_games(monkeypatch)
    assert server.execute_server_campaign(args) == 0
    assert len(more) == 7 and not set(old_ids) & set(more)
    assert (args.work_directory / "assignment-ledger.jsonl").read_bytes().startswith(before)
    assert published[0][1].manifest["assignment_count"] == 10


def test_completed_ledger_resume_only_seals_without_another_game(tmp_path, monkeypatch):
    args, _, _, _ = _setup(tmp_path, monkeypatch)
    started, _ = _script_games(monkeypatch)
    monkeypatch.setattr(server, "publish_online_pilot", lambda *_args, **_kwargs:
                        (_ for _ in ()).throw(OSError("publish interrupted")))
    with pytest.raises(OSError):
        server.execute_server_campaign(args)
    assert len(started) == 10
    args.resume = True
    more, published = _script_games(monkeypatch)
    assert server.execute_server_campaign(args) == 0
    assert more == [] and published[0][2].publication_only is True


def test_resume_rejects_changed_binding_and_leaves_ledger_unchanged(tmp_path, monkeypatch):
    args, _, _, _ = _setup(tmp_path, monkeypatch)
    _script_games(monkeypatch, stop_after=1)
    with pytest.raises(RuntimeError):
        server.execute_server_campaign(args)
    ledger_path = args.work_directory / "assignment-ledger.jsonl"
    before = ledger_path.read_bytes()
    raw = server._read_json(args.work_directory / "run_inputs.json")
    raw["q_python"] = "different"
    (args.work_directory / "run_inputs.json").write_bytes(canonical_json_bytes(raw))
    args.resume = True
    with pytest.raises(ValueError, match="resume"):
        server.execute_server_campaign(args)
    assert ledger_path.read_bytes() == before


@pytest.mark.parametrize("mode", ("success", "language_failure", "zero_assignment"))
def test_server_factory_executes_real_production_loop(tmp_path, monkeypatch, mode):
    from tests.canonical_collection.test_runtime_game_evidence import (
        _Agent, _Backend, _OneSpeechEnvironment,
    )
    from tests.phase2.test_online_runner_integration import _StopAfterFirstDayEnv

    args, _, _, _ = _setup(tmp_path, monkeypatch)
    plan, game_plan, config, cap, mapper, reference, _ = server.load_server_inputs(args)
    predictor = server.Qwen3GameplayPredictorClient()
    q_predictions = []
    def predict(prefix):
        q_predictions.append(prefix.prefix_digest)
        return q_matrix()
    predictor.predict = predict
    backends = server.load_named_backends(config)
    raw = backends["local_qwen"]
    raw.chat = _Backend().chat
    ledger = server.OnlinePilotAssignmentLedgerV1(
        args.work_directory / "assignment-ledger.jsonl", plan=plan,
        source_commit=args.source_commit)
    factory = server.ServerRuntimeFactory(
        args=args, plan=plan, game_plan=game_plan, config=config, call_limit=cap,
        mapper=mapper, reference=reference, predictor=predictor, backends=backends,
        ledger=ledger, work_directory=args.work_directory)
    bundle = factory(factory.game_ids[0])
    assert bundle.preflight.ready
    assert runner.run_canonical_game is production_eval
    inspect.signature(production_eval).bind(
        bundle.env, bundle.agents, bundle.roles, canonical_recorder=bundle.recorder,
        call_audit=bundle.call_audit, online_pilot=bundle.pilot)

    # Only external agents/backend responses and the fixture's stopping bound
    # are scripted. Factory, eval, PRE, assignment, commit and voting are real.
    baseline_votes, language_calls = [], []
    class BaselineAgent(_Agent):
        def act(self, observation):
            action = super().act(observation)
            if "vote" in observation["phase"]:
                baseline_votes.append((self.seat, action))
            return action
    agents = tuple(BaselineAgent(bundle.pilot.backend, seat) for seat in range(1, 8))
    bundle.recorder.belief_collector.agents = agents
    bundle = replace(bundle, agents=agents)
    bundle.env.__class__ = (_OneSpeechEnvironment if mode == "zero_assignment"
                           else _StopAfterFirstDayEnv)
    if mode == "zero_assignment":
        # Select a bounded fixture whose only public PRE belongs to a non-wolf.
        for seed in range(20):
            trial = _OneSpeechEnvironment(
                speech_perceiver=bundle.env.speech_perceiver, random_seed=seed,
                log_save_path=None)
            observation = trial.reset(roles=bundle.roles)
            while trial.phase not in ("speech", "speech_pk"):
                observation, _, _, _ = trial.step(
                    agents[observation["current_act_idx"] - 1].act(observation))
            if trial.roles[trial.current_act_idx] != "Werewolf":
                bundle.env._rng.seed(seed)
                break
        else:
            pytest.fail("no non-wolf first speaker in the deterministic fixture")
        baseline_votes.clear()

    def language_response(**kwargs):
        # Reopen the durable journal at every dispatch, rather than checking
        # only the runner's in-memory assignment state.
        persisted = server.OnlinePilotAssignmentLedgerV1(
            ledger.path, plan=plan, source_commit=args.source_commit)
        assert persisted.snapshot()["assignment_count"] == 1
        stages = persisted.snapshot()["games"][bundle.recorder.game_id]
        assert "ASSIGNMENT" in stages
        language_calls.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        if "response_format" not in kwargs:
            if mode == "language_failure":
                return "本轮弃票。", {"finish_reason": "stop"}
            direct = re.search(r"唯一目标(player[1-7])", prompt)
            if direct:
                return f"本轮放逐票投给{direct.group(1)}。", {"finish_reason": "stop"}
            j, k = re.search(r"拒绝目标(player[1-7])和唯一转向目标(player[1-7])", prompt).groups()
            return f"本轮不以{j}为主要放逐目标，本轮放逐票投给{k}。", {"finish_reason": "stop"}
        text = prompt.split("待解析的本轮公开发言：\n", 1)[1].split("\ncommitment_targets：", 1)[0]
        votes = re.findall(r"放逐票投给(player[1-7])", text)
        rejected = re.findall(r"不以(player[1-7])为主要放逐目标", text)
        return json.dumps({"commitment_targets": votes[-1:], "rejected_targets": rejected,
            "vote_intent_targets": votes[-1:], "information_requests": [],
            "abstain_intent": "弃票" in text, "private_fact_claim": False}), {"finish_reason": "stop"}
    raw.chat_with_metadata = language_response
    result = runner.run_online_game(
        bundle.env, bundle.agents, bundle.roles, recorder=bundle.recorder,
        call_audit=bundle.call_audit, pilot=bundle.pilot, preflight=bundle.preflight)
    assert result in ("Werewolf win", "Villager win")
    evidence = bundle.recorder.complete_evidence()
    snapshot = ledger.snapshot()
    assert snapshot["assignment_count"] == int(mode != "zero_assignment")
    assert len(q_predictions) == int(mode != "zero_assignment")
    assert evidence.game_id == bundle.recorder.game_id
    if mode == "zero_assignment":
        assert not language_calls and not bundle.pilot.state.assignments
    else:
        stages = snapshot["games"][bundle.recorder.game_id]
        assert stages["EXECUTION"]["execution"]["success"] is (mode == "success")
        assert len(bundle.pilot.state.assignments) == 1
        assert factory.verify_execution(bundle.recorder.game_id, stages)
        assert bool(stages.get("CONSEQUENCE")) is (mode == "success")
        actual_votes = [event["votes"] for event in bundle.env.public_events
                        if event["event_type"] == "vote_result"]
        assert actual_votes and baseline_votes
        for votes in actual_votes:
            for vote in votes:
                seat = int(vote["voter"].removeprefix("player"))
                expected = next(action[1] for actor, action in baseline_votes if actor == seat)
                assert vote["target"] == (None if expected == 0 else f"player{expected}")
