"""Admission regression: real loader/module, scripted transport, no GPU/network."""
from copy import deepcopy
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
from types import ModuleType, SimpleNamespace

import pytest

from scripts import phase2_gameplay_campaign as cli
from werewolf import phase2_gameplay_server as server
from werewolf.phase2_gameplay import digest, bind_campaign_profile, validate_profile, VERSION, qualification_diagnostics, freeze_source
from werewolf.artifact_io import publish_artifact, canonical_json_bytes
from tests.phase2.test_gameplay_campaign import plan, GameplayLedger
from tests.phase2.test_probe_policy_runner import runtime
from tests.runtime.test_runtime_config import new_config
from werewolf.runtime_config import normalize_runtime_config


@pytest.fixture
def real_loader(monkeypatch):
    """Execute the actual factory module; replace only optional SDK transport."""
    root = Path(__file__).resolve().parents[2]
    package = ModuleType("werewolf.backends")
    package.__path__ = [str(root / "werewolf/backends")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    calls = []
    class Backend:
        def __init__(self, *, api_key, base_url, default_model, supports_json_schema, max_retries):
            calls.append((base_url, default_model, max_retries))
            self.client = self
        def close(self):
            pass
        def chat_with_metadata(self, **kwargs):
            raise AssertionError("network/generation forbidden")
    transport = ModuleType("werewolf.backends.openai_compatible")
    transport.OpenAICompatibleBackend = Backend
    monkeypatch.setitem(sys.modules, transport.__name__, transport)
    dotenv = ModuleType("dotenv")
    dotenv.load_dotenv = lambda **kw: pytest.fail("dotenv must not be read")
    monkeypatch.setitem(sys.modules, dotenv.__name__, dotenv)
    spec = importlib.util.spec_from_file_location("werewolf.backends.factory", root / "werewolf/backends/factory.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    assert Path(module.load_named_backends.__code__.co_filename) == root / "werewolf/backends/factory.py"
    return module, calls


def assembly(monkeypatch, runtime):
    """Script native assembly only; the adapter and production signature are real."""
    effects = []
    def forbidden(*args, **kwargs):
        pytest.fail("static/check mode must not run gameplay")
    class Client:
        def __init__(self, **kwargs):
            effects.append("worker")
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    for name, fields in {
        "scripts.qwen3_gameplay_predictor": {"Qwen3GameplayPredictorClient": Client},
        "werewolf.canonical_collection.production_runtime": {"Classic7RuntimeFactory": forbidden,
            "classic7_replay_executor": lambda config: None, "run_game": runtime.run_canonical_game},
        "werewolf.phase2_online_runner": {"OnlineTerminalPilotRunnerV1": forbidden},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(fields)
        monkeypatch.setitem(sys.modules, name, module)
    return effects


def test_runtime_check_uses_actual_loader_and_keeps_transport_zero(tmp_path, monkeypatch, runtime, real_loader):
    loader, calls = real_loader
    effects = assembly(monkeypatch, runtime)
    config = new_config()
    config["backends"]["deepseek"].update(base_url="http://127.0.0.1:8000/v1", api_key_env=None,
                                          default_model=config["parser"]["model"])
    config = normalize_runtime_config(config)
    original = deepcopy(config)
    p = plan(paths={"plan": str(tmp_path / "plan"), "work": str(tmp_path / "work"), "destination": str(tmp_path / "pub")})
    monkeypatch.setattr("werewolf.phase2_gameplay.load_campaign_profile", lambda *a: p["campaign_profile"])
    p["runtime_inputs"]["q_python"] = "/fixture/python"
    p["plan_digest"] = digest({k: v for k, v in p.items() if k != "plan_digest"})
    monkeypatch.setattr(server, "server_inputs", lambda *a: ({"q_checkout": tmp_path / "q", "q_fit": tmp_path / "fit"}, config, 10, None, None))
    monkeypatch.setattr(server, "verify_runtime_service", lambda *a: effects.append("service"))
    monkeypatch.setattr(server, "freeze_source", lambda *a: p["source"])
    result = server.execute_server(p, repo=Path(__file__).resolve().parents[2], work=tmp_path / "work",
        destination=tmp_path / "pub", runtime_check_only=True, allow_gpu=True)
    assert result["status"] == "RUNTIME_CHECK_PASSED" and result["runtime_check_passed"] is True
    assert result["gameplay_run"] is False and effects == ["service", "worker"]
    assert calls == [("http://127.0.0.1:8000/v1", config["parser"]["model"], 0)]
    assert config == original and not (tmp_path / "work").exists()
    with pytest.raises(ValueError, match="unsupported runtime fields"):
        loader.load_named_backends(config["backends"], env_file=None, max_retries=0)
    config["unknown"] = True
    with pytest.raises(ValueError, match="unsupported runtime fields"):
        loader.load_named_backends(config, env_file=None, max_retries=0)


@pytest.mark.parametrize("command", ["run", "resume", "runtime-check", "prepare", "status", "validate"])
def test_resume_check_rejected_before_any_input_or_side_effect(tmp_path, monkeypatch, capsys, command):
    ledger = tmp_path / "game-ledger.jsonl"
    ledger.write_bytes(b"unchanged sentinel\n")
    def forbidden(*a, **kw):
        pytest.fail("invalid CLI must fail before input/worker/gameplay")
    monkeypatch.setattr(cli, "open_plan", forbidden)
    monkeypatch.setattr(cli, "prepare", forbidden)
    monkeypatch.setattr(cli, "execute_server", forbidden)
    with pytest.raises(SystemExit) as error:
        cli.main([command, "--plan", str(tmp_path / "plan"), "--plan-manifest-digest", "a" * 64,
                  "--source-commit", "a" * 40, "--exclusions", str(tmp_path / "exclusions"),
                  "--work-directory", str(tmp_path / "work"), "--destination", str(tmp_path / "pub"), "--resume-check"])
    assert error.value.code == 2 and ledger.read_bytes() == b"unchanged sentinel\n"
    assert "--resume-check is only allowed with preflight" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["run", "resume", "runtime-check"])
def test_gpu_modes_require_explicit_permission_before_inputs(monkeypatch, capsys, tmp_path, command):
    monkeypatch.setattr(cli, "open_plan", lambda *a: pytest.fail("GPU mode needs explicit opt-in first"))
    with pytest.raises(SystemExit) as error:
        cli.main([command, "--plan", str(tmp_path / "plan"), "--plan-manifest-digest", "a" * 64])
    assert error.value.code == 2
    assert "requires --allow-gpu" in capsys.readouterr().err


def test_static_cli_rejects_gpu_request_before_inputs(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "open_plan", lambda *a: pytest.fail("static cannot enable GPU"))
    with pytest.raises(SystemExit) as error:
        cli.main(["preflight", "--allow-gpu", "--plan", str(tmp_path / "plan"), "--plan-manifest-digest", "a" * 64])
    assert error.value.code == 2 and "--allow-gpu is only allowed" in capsys.readouterr().err


def test_server_gpu_permission_is_checked_before_assembly(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "server_inputs", lambda *a: pytest.fail("GPU permission needed before assembly"))
    with pytest.raises(ValueError, match="explicit GPU permission"):
        server.execute_server(plan(), repo=tmp_path, work=tmp_path / "work", destination=tmp_path / "pub")
    assert not (tmp_path / "work").exists()


def frozen_profile(tmp_path, monkeypatch):
    """Synthetic operator paths: never interpreted as verified server paths."""
    root = tmp_path / "repo"
    profile_path = root / "configs/phase2/router-gameplay-qualification-v1.json"
    p = plan(20, paths={"plan": str(tmp_path / "plan"), "work": str(tmp_path / "work"), "destination": str(tmp_path / "pub")},
             gate="full-games-and-both-mechanisms-v1")
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(json.dumps(p["campaign_profile"]))
    monkeypatch.setattr(cli, "ROOT", root)
    monkeypatch.setattr(cli, "PROFILES", {"qualification": profile_path})
    return root, profile_path, p


@pytest.mark.parametrize("missing", ["paths", "excluded_game_plans"])
def test_unverified_operator_bindings_block_prepare_before_source_or_runtime(tmp_path, monkeypatch, missing):
    root, path, p = frozen_profile(tmp_path, monkeypatch)
    profile = deepcopy(p["campaign_profile"])
    profile[missing] = {key: None for key in profile["paths"]} if missing == "paths" else None
    path.write_text(json.dumps(profile))
    monkeypatch.setattr(cli, "freeze_source", lambda *a: pytest.fail("unverified profile must reject before native imports"))
    with pytest.raises(ValueError, match="unverified/unfrozen"):
        cli.prepare(SimpleNamespace(profile=None, purpose="qualification"))
    assert not Path(p["campaign_profile"]["paths"]["plan"]).exists()


@pytest.mark.parametrize("field", ["plan", "work", "destination"])
def test_alternate_campaign_directory_cannot_restart_a_failed_campaign(tmp_path, monkeypatch, field):
    root, path, p = frozen_profile(tmp_path, monkeypatch)
    paths = p["campaign_profile"]["paths"]
    ledger = GameplayLedger(paths["work"], p)
    ledger.initialize()
    ledger.start(p["allocations"][0])
    before = ledger.path.read_bytes()
    args = SimpleNamespace(profile=None, purpose="qualification", plan=Path(paths["plan"]),
                           work_directory=Path(paths["work"]), destination=Path(paths["destination"]))
    setattr(args, {"plan": "plan", "work": "work_directory", "destination": "destination"}[field], tmp_path / "alternate")
    monkeypatch.setattr(cli, "freeze_source", lambda *a: pytest.fail("alternate paths must reject before source/runtime"))
    with pytest.raises(ValueError, match="cannot change"):
        cli.prepare(args)
    with pytest.raises(ValueError, match="cannot change"):
        bind_campaign_profile(p, root, work=tmp_path / "alternate", destination=paths["destination"])
    assert ledger.path.read_bytes() == before and not (tmp_path / "alternate").exists()


def test_foreign_profile_cannot_override_canonical_campaign_identity(tmp_path, monkeypatch):
    root, path, p = frozen_profile(tmp_path, monkeypatch)
    clone = tmp_path / "cloned-profile.json"
    clone.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="canonical source-frozen profile"):
        cli.prepare(SimpleNamespace(profile=clone, purpose="qualification"))
    assert not (tmp_path / "plan").exists()


def test_clean_old_source_cannot_admit_untracked_gameplay_implementation(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    git("init", "-q", "-b", "twd/mainline")
    (repo / "run_random.py").write_text("# synthetic old source\n")
    git("add", "run_random.py")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "old fixture")
    old = git("rev-parse", "HEAD")
    for name in ("werewolf/phase2_gameplay.py", "werewolf/phase2_gameplay_server.py", "scripts/phase2_gameplay_campaign.py",
                 "docs/research/phase2-router-v1-contract.md", "configs/phase2/router-gameplay-qualification-v1.json",
                 "configs/phase2/router-gameplay-formal-v1.json"):
        file = repo / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("untracked implementation\n")
    assert git("status", "--porcelain", "--untracked-files=no") == ""
    with pytest.raises(ValueError, match="must be committed"):
        freeze_source(repo)
    assert git("rev-parse", "HEAD") == old


def prepared_artifact(path, p, *, work=None):
    return publish_artifact(path, manifest_fields={"artifact_type": "phase2_router_gameplay_plan", "schema_version": VERSION,
        "purpose": p["purpose"], "source": p["source"], "plan_digest": p["plan_digest"]}, files={
        "run_plan.json": canonical_json_bytes(p), "game_plan.json": canonical_json_bytes(p["game_plan"]),
        "operator_paths.json": canonical_json_bytes({"work": work or p["campaign_profile"]["paths"]["work"],
            "destination": p["campaign_profile"]["paths"]["destination"]})})


def test_plan_location_and_rehashed_operator_paths_are_source_bound(tmp_path, monkeypatch):
    root, path, p = frozen_profile(tmp_path, monkeypatch)
    first = prepared_artifact(tmp_path / "plan", p)
    loaded, paths = cli.open_plan(first.path, first.manifest_digest)
    assert loaded == p and paths["work"] == str(tmp_path / "work")
    moved = prepared_artifact(tmp_path / "moved-plan", p)
    with pytest.raises(ValueError, match="cannot change"):
        cli.open_plan(moved.path, moved.manifest_digest)
    first_path = p["campaign_profile"]["paths"]["plan"]
    # Direct server admission also enforces the source-frozen tuple, independently of CLI.
    with pytest.raises(ValueError, match="cannot change"):
        bind_campaign_profile(p, root, work=tmp_path / "elsewhere", destination=paths["destination"], prepared_path=first_path)
    tampered = prepared_artifact(tmp_path / "tampered-plan", p, work=str(tmp_path / "elsewhere"))
    with pytest.raises(ValueError, match="cannot change"):
        cli.open_plan(tampered.path, tampered.manifest_digest)
    profile = json.loads(path.read_bytes())
    profile["paths"]["work"] = str(tmp_path / "elsewhere")
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="source-frozen campaign profile"):
        cli.open_plan(first.path, first.manifest_digest)


def test_rehashed_plan_at_correct_location_cannot_replace_work_directory(tmp_path, monkeypatch):
    root, path, p = frozen_profile(tmp_path, monkeypatch)
    artifact = prepared_artifact(tmp_path / "plan", p, work=str(tmp_path / "alternate-work"))
    with pytest.raises(ValueError, match="cannot change"):
        cli.open_plan(artifact.path, artifact.manifest_digest)
    assert not (tmp_path / "alternate-work").exists()


def test_qualification_budget_allocation_and_gate_ignore_outcome_direction():
    profile = cli.load_campaign_profile(cli.ROOT, "qualification")
    assert profile["randomized_games"] == 20 and profile["planning_status"] == "FROZEN"
    assert profile["qualification_gate"] == "full-games-and-both-mechanisms-v1"
    p = plan(20, gate=profile["qualification_gate"])
    assert {arm: sum(a["assigned_policy"] == arm for a in p["allocations"])
            for arm in ("IMMEDIATE_REDIRECT", "PROBE_THEN_REDIRECT")} == {"IMMEDIATE_REDIRECT": 10, "PROBE_THEN_REDIRECT": 10}
    outcomes = [dict(a, winner="Werewolf", router_process={"t1_committed": True,
                "t3_committed": a["assigned_policy"] == "PROBE_THEN_REDIRECT"}) for a in p["allocations"]]
    assert all(qualification_diagnostics(outcomes).values())
    assert qualification_diagnostics(outcomes) == qualification_diagnostics([dict(r, winner="Villager") for r in outcomes])
    for r in outcomes:
        r["router_process"]["t3_committed"] = False
    assert qualification_diagnostics(outcomes)["probe_then_redirect_committed"] is False
    for r in outcomes:
        r["router_process"]["t1_committed"] = False
    assert qualification_diagnostics(outcomes)["immediate_redirect_committed"] is False
    with pytest.raises(ValueError, match="unfrozen"):
        validate_profile(cli.load_campaign_profile(cli.ROOT, "formal"), ready=True)


def test_qualification_profile_binds_six_operator_verified_production_pins():
    """Authority is the operator's server decoder result, not synthetic JSON."""
    profile = validate_profile(cli.load_campaign_profile(cli.ROOT, "qualification"), ready=True)
    prefix = "/data/yuxiao/Untrusted_Network_Simulation/"
    assert profile["paths"] == {
        "plan": prefix + "plans/phase2-router-gameplay-qualification-v1-plan",
        "work": prefix + "paper-studies/phase2-router-gameplay-qualification-v1-work",
        "destination": prefix + "paper-studies/paper-phase2-router-gameplay-qualification-v1",
    }
    expected = {
        "paper-phase2-online-terminal-qualification-v1-canonical-game-plan.json": "a6b575bc72bfab802be3019e740c4f98c9d5bb9254ca47a2d6c1571ea3be9661",
        "phase2-online-terminal-pilot-v1-game-plan.json": "add3fe53054d9b89b6c6570b585bce13070f010dcf2e28a40a4671fcd713c64b",
        "phase2-online-probe-qualification-v1-game-plan.json": "4719b85665febcb47479a594f13c00940a6636061ebe9410c4b68391be4bda6e",
        "phase2-online-probe-qualification-v2-game-plan.json": "ef8e2d0d239f3e0d98ef84ed44c51d3cbceb80b4ddacee5b52fff6a7ac0bd151",
        "phase2-online-probe-qualification-v3-game-plan.json": "97ecfabf670d05d799fb39819ca4e78275416d004cdb936eb455ec18f40a1aca",
        "phase2-online-probe-pilot-v1-game-plan.json": "7f69c01059a350fdc0bef9f21febe70e36b67901f3e3e6200a9a05f6c3c3aa57",
    }
    assert profile["excluded_game_plans"] == [
        {"path": prefix + "plans/" + name, "plan_digest": sha} for name, sha in expected.items()]
    # Structural readiness does not read, reconstruct or certify those remote pools.
    assert profile["qualification"] is None


def test_new_qualification_pool_uses_unmodified_production_seed_function():
    # Isolate the pure production definition from this Mac's optional CLI SDK imports.
    path = cli.ROOT / "scripts/collect_games.py"
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if
             isinstance(node, ast.FunctionDef) and node.name == "derive_seed_pool" or
             isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CALIBRATION_SEEDS" for t in node.targets)]
    scope = {"hashlib": hashlib}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    derive_seed_pool, calibration = scope["derive_seed_pool"], scope["CALIBRATION_SEEDS"]
    profile = cli.load_campaign_profile(cli.ROOT, "qualification")
    seeds = derive_seed_pool(profile["campaign_id"], profile["randomized_games"])
    assert Path(derive_seed_pool.__code__.co_filename) == cli.ROOT / "scripts/collect_games.py"
    assert len(seeds) == len(set(seeds)) == 20
    assert seeds == derive_seed_pool(profile["campaign_id"], 20)
    assert not set(seeds).intersection(calibration)
    # Historical 1320-seed non-overlap is deliberately not asserted without the real files.


def test_pinned_profile_bytes_are_part_of_real_git_source_provenance(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    git("init", "-q", "-b", "twd/mainline")
    required = ("werewolf/phase2_gameplay.py", "werewolf/phase2_gameplay_server.py", "scripts/phase2_gameplay_campaign.py",
                "docs/research/phase2-router-v1-contract.md", "configs/phase2/router-gameplay-qualification-v1.json",
                "configs/phase2/router-gameplay-formal-v1.json")
    for name in required:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((cli.ROOT / name).read_bytes())
    git("add", *required)
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture source")
    frozen = freeze_source(repo)
    assert set(required) <= frozen["source_sha256"].keys()
    for name in required:
        assert frozen["source_sha256"][name] == hashlib.sha256((repo / name).read_bytes()).hexdigest()
    profile_path = repo / "configs/phase2/router-gameplay-qualification-v1.json"
    p = json.loads(profile_path.read_bytes())
    p["excluded_game_plans"][0]["plan_digest"] = "b" * 64
    profile_path.write_text(json.dumps(p))
    with pytest.raises(ValueError, match="clean"):
        freeze_source(repo)
    git("add", "configs/phase2/router-gameplay-qualification-v1.json")
    with pytest.raises(ValueError, match="clean"):
        freeze_source(repo)
    assert git("rev-parse", "HEAD") == frozen["commit"]


@pytest.mark.parametrize("mutation", [None, "dirty_source", "source_revision", "fit_file", "seal_binding"])
def test_static_q_uses_real_manifest_chain_without_loading_worker_or_model(tmp_path, monkeypatch, mutation):
    from scripts import qwen3_gameplay_predictor as q
    checkout = tmp_path / "q-checkout"
    checkout.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(checkout), *args], text=True).strip()
    git("init", "-b", "fixture")
    (checkout / "source.txt").write_text("synthetic Q source\n")
    git("add", "source.txt")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "fixture")
    fit = publish_artifact(tmp_path / "data/artifacts/fit", manifest_name="qwen3_final_manifest.json",
        manifest_fields={"artifact_type": "qwen3_all_development_final", "schema_version": "classic7_qwen3_all_development_final_v1"},
        files={"fit.json": b"{}"})
    binding = {"fit_digest": fit.manifest_digest}
    runs = fit.path.parent.parent / "runs" / fit.manifest_digest
    terminal = publish_artifact(runs / "terminal", manifest_fields={"artifact_type": "qwen3_final_terminal",
        "schema_version": "classic7_qwen3_final_terminal_v1", "binding": binding, "model_digest": "a" * 64}, files={"checkpoint.json": b"{}"})
    seal = publish_artifact(runs / "seal", manifest_fields={"artifact_type": "qwen3_final_seal",
        "schema_version": "classic7_qwen3_final_seal_v1", "binding": binding if mutation != "seal_binding" else {"fit_digest": "b" * 64},
        "terminal_digest": terminal.manifest_digest}, files={"seal.json": b"{}"})
    for key, value in {"SOURCE_REVISION": git("rev-parse", "HEAD"), "FIT_DIGEST": fit.manifest_digest,
                       "TERMINAL_DIGEST": terminal.manifest_digest, "SEAL_DIGEST": seal.manifest_digest, "MODEL_DIGEST": "a" * 64}.items():
        monkeypatch.setattr(q, key, value)
    monkeypatch.setattr(q, "Qwen3GameplayPredictorClient", lambda **kw: pytest.fail("static cannot start worker"))
    monkeypatch.setattr(q, "_serve", lambda *a: pytest.fail("static cannot restore/migrate a model"))
    monkeypatch.setitem(sys.modules, "werewolf.tom.qwen3_final", None)
    monkeypatch.setenv("GIT_DIR", "/invalid/environment-must-be-ignored")
    if mutation == "dirty_source":
        (checkout / "untracked.txt").write_text("dirty")
    elif mutation == "source_revision":
        monkeypatch.setattr(q, "SOURCE_REVISION", "b" * 40)
    elif mutation == "fit_file":
        (fit.path / "fit.json").write_text("tamper")
    args = ({"q_checkout": checkout, "q_fit": fit.path}, {"q_fit_digest": fit.manifest_digest, "q_seal_digest": seal.manifest_digest})
    if mutation is None:
        server.verify_q_static(*args)
    else:
        with pytest.raises(ValueError):
            server.verify_q_static(*args)
