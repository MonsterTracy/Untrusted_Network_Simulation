"""Operator orchestration with real frozen plans/ledger and full server preflight.

All campaign paths and Git commits are temporary test fixtures. No live LLM.
"""

from dataclasses import replace
import json
from pathlib import Path

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("openai")
pytest.importorskip("torch")

from scripts import phase2_online_campaign as campaign
from tests.phase2.test_online_server import _setup, _git
from tests.phase2.test_online_ledger import _game_assignment
from werewolf.artifact_io import canonical_json_bytes, publish_artifact
from werewolf.canonical_collection.attempt_ledger import collection_plan_from_record
from werewolf.phase2_online_ledger import OnlinePilotAssignmentLedgerV1
from werewolf.phase2_online_plan import Phase2OnlineTerminalPilotPlanV1
from werewolf.phase2_online_runner import ARTIFACT_NAME, QUALIFICATION_NAME, ARTIFACT_VERSION
from werewolf import phase2_online_server as server


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    args, old_plan, counters, _ = _setup(tmp_path, monkeypatch, purpose="pilot")
    plan = replace(old_plan, pilot_id=ARTIFACT_NAME)
    profile = {"schema_version": campaign.PROFILE_VERSION, "campaign_purpose": "pilot",
        "pilot_id": ARTIFACT_NAME, "assignment_seed": plan.assignment_seed,
        "max_games_attempted": plan.max_games_attempted, "target_assignment_count": 120,
        "estimator_eligible": True, **{k: str(getattr(args, k)) for k in
            ("plan", "game_plan", "work_directory", "destination")}}
    paths = {k: str(getattr(args, k)) for k in ("repo", "runtime_config", "deployment_config",
        "publication", "evaluation_root", "mapper", "smoke_v3", "q_checkout", "q_fit")}
    paths.update(work_directory=str(tmp_path / "qualification-work"),
                 destination=str(tmp_path / QUALIFICATION_NAME))
    bound = {"paths": paths, "source_commit": args.source_commit,
        "canonical_game_plan": campaign.read_json(args.game_plan),
        **{k: getattr(args, k) for k in ("mapper_manifest_digest", "reference_tables_digest",
                                       "smoke_v3_manifest_digest", "q_python")}}
    bound["canonical_game_plan"]["ordered_seed_pool"] = list(range(801, 841))
    # qualification_inputs is an external artifact seam, tested separately below.
    monkeypatch.setattr(campaign, "qualification_inputs", lambda: (pins, bound, None))
    monkeypatch.setattr(campaign, "REPO", args.repo)
    monkeypatch.setattr(campaign, "PROFILE", args.repo / "configs/phase2/profile.json")
    monkeypatch.setattr(campaign, "PINS", args.repo / "configs/phase2/pins.json")
    operator = args.repo / "scripts/phase2_online_campaign.py"
    operator.parent.mkdir(parents=True, exist_ok=True)
    operator.write_bytes(Path(campaign.__file__).read_bytes())
    monkeypatch.setattr(campaign, "__file__", str(operator))
    write(campaign.PROFILE, profile)
    write(campaign.PINS, {"test_only": True})
    _git(args.repo, "add", ".")
    _git(args.repo, "commit", "-qm", "operator profile")
    head = _git(args.repo, "rev-parse", "HEAD")
    game = campaign.canonical_plan(plan, head, bound)
    # Reconstruct instead of editing a frozen record/digest.
    from scripts.collect_games import plan_fields
    from werewolf.canonical_collection.attempt_ledger import construct_collection_plan
    environment = dict(game.environment_provenance)
    extra = {k: v for k, v in environment.items() if k not in (
        "configured_call_limit", "gameplay_prompt_profile", "seed_pool_size", "target_success_count")}
    development = construct_collection_plan(ordered_seed_pool=tuple(range(10001, 12251)),
        **plan_fields({"collection_id": "development", "target_games": 1500,
            "seed_pool_size": 2250, "call_limit": 10000}, head, extra))
    dev_path = tmp_path / "development-plan.json"
    write(dev_path, development.to_record())
    publication = {"collection_plan_digest": development.plan_digest}
    publication["manifest_digest"] = campaign.digest(publication)
    write(Path(paths["publication"]) / "manifest.json", publication)
    pins = {"development_game_plan": str(dev_path), "development_game_plan_digest": development.plan_digest,
            "publication_manifest_digest": publication["manifest_digest"]}
    write(args.plan, plan.to_record())
    write(args.game_plan, game.to_record())
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    return profile, bound, pins, counters, args, head


def absent_plans(profile):
    Path(profile["plan"]).unlink()
    Path(profile["game_plan"]).unlink()


@pytest.mark.parametrize("staged", (False, True))
def test_prepare_refuses_dirty_tracked_or_index(setup, staged):
    profile, _, _, counters, args, _ = setup
    absent_plans(profile)
    (args.repo / "tracked.py").write_text("changed\n")
    if staged:
        _git(args.repo, "add", "tracked.py")
        (args.repo / "tracked.py").write_text("pass\n")
    with pytest.raises(ValueError, match="clean"):
        campaign.prepare(profile)
    assert not Path(profile["plan"]).exists() and counters["q_start"] == 0


def test_prepare_exclusive_clean_head_and_untracked_allowed(setup, capsys):
    profile, _, _, _, args, head = setup
    absent_plans(profile)
    (args.repo / "unrelated.md").write_text("leave alone\n")
    assert campaign.main(["prepare", "formal"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["source_commit"] == head
    assert result["seed_overlap_counts"] == {"qualification": 0, "development": 0, "calibration": 0}
    game = collection_plan_from_record(campaign.read_json(profile["game_plan"]))
    from scripts.collect_games import derive_seed_pool
    assert game.source_revision == head
    assert game.ordered_seed_pool == derive_seed_pool(ARTIFACT_NAME, 122)
    before = Path(profile["plan"]).read_bytes()
    assert campaign.main(["prepare", "formal"]) == 1
    assert Path(profile["plan"]).read_bytes() == before


@pytest.mark.parametrize("field", ("assignment_seed", "max_games_attempted"))
def test_missing_preregistration_fails_without_plan(setup, field):
    profile, *_ = setup
    absent_plans(profile)
    profile[field] = None
    with pytest.raises(ValueError, match=field):
        campaign.prepare(profile)
    assert not Path(profile["plan"]).exists()


@pytest.mark.parametrize("value", (None, ":4096:8", ":16:8", ""))
def test_cublas_fail_closed(setup, monkeypatch, value):
    _, _, _, counters, _, _ = setup
    if value is not None:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", value)
    if value in (None, ":4096:8"):
        campaign.frozen_environment()
        import os
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    else:
        assert campaign.main(["preflight", "formal"]) == 1
    assert counters["q_start"] == 0


@pytest.mark.parametrize("pool", ("qualification", "development", "calibration"))
def test_seed_overlap_fails_no_reroll(setup, monkeypatch, pool):
    profile, bound, pins, _, _, _ = setup
    absent_plans(profile)
    if pool == "qualification":
        seed = bound["canonical_game_plan"]["ordered_seed_pool"][0]
    elif pool == "development":
        seed = campaign.read_json(pins["development_game_plan"])["ordered_seed_pool"][-1]
    else:
        seed = 900000001
    import scripts.collect_games as collection
    monkeypatch.setattr(collection, "derive_seed_pool", lambda *_: tuple(range(1, 122)) + (seed,))
    with pytest.raises(ValueError, match="overlap"):
        campaign.prepare(profile)
    assert not Path(profile["plan"]).exists()


def test_full_preflight_no_game_assignment_inference_language_or_durable_work(setup, capsys):
    profile, _, _, counters, _, _ = setup
    assert campaign.main(["preflight", "formal"]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rows[1]["preflight"]["ready"] is True
    assert rows[1]["preflight"]["online_randomized_pilot_ready"] is True
    assert counters == {"q_start": 1, "q_predict": 0, "language": 0, "closed": 1}
    assert not Path(profile["work_directory"]).exists()
    assert not Path(profile["destination"]).exists()


@pytest.mark.parametrize("fault", ("missing_plan", "missing_game_plan", "destination", "work", "head", "pins"))
def test_run_fails_before_q_on_invalid_frozen_inputs(setup, fault):
    profile, bound, _, counters, args, _ = setup
    if fault.startswith("missing_"):
        Path(profile[fault.removeprefix("missing_")]).unlink()
    elif fault in ("destination", "work"):
        Path(profile["destination" if fault == "destination" else "work_directory"]).mkdir()
    elif fault == "head":
        _git(args.repo, "commit", "--allow-empty", "-qm", "different source")
    else:
        bound["canonical_game_plan"]["environment_provenance"]["model_manifest_sha256"] = "f" * 64
    assert campaign.main(["run", "formal"]) == 1
    assert counters["q_start"] == counters["language"] == 0


def test_resume_uses_exact_bound_inputs_without_rerandomization(setup):
    profile, _, _, counters, _, head = setup
    args = campaign.server_args(profile, "resume")
    args.resume = False
    bound = server.load_server_inputs(args)[-1]
    work = Path(profile["work_directory"])
    work.mkdir()
    write(work / "run_inputs.json", bound)
    ledger = OnlinePilotAssignmentLedgerV1(work / "assignment-ledger.jsonl",
        plan=campaign.pilot_plan(profile), source_commit=head)
    ledger.start_game("g1")
    assignment = _game_assignment(ledger.plan, "g1")
    ledger.persist_assignment(assignment)
    before = ledger.path.read_bytes()
    # Real loader accepts an exact match; full preflight remains delegated.
    assert server.load_server_inputs(campaign.server_args(profile, "resume"))[-1] == bound
    bound["reference_tables_digest"] = "f" * 64
    write(work / "run_inputs.json", bound)
    assert campaign.main(["resume", "formal"]) == 1
    assert ledger.path.read_bytes() == before
    assert counters["q_start"] == 0


def test_status_readonly_not_started_and_partial_assignment(setup, monkeypatch):
    profile, _, _, _, _, head = setup
    result = campaign.status(profile)
    assert result["status"] == "NOT_STARTED" and result["remaining_target"] == 120
    args = campaign.server_args(profile, "run")
    bound = server.load_server_inputs(args)[-1]
    work = Path(profile["work_directory"])
    write(work / "run_inputs.json", bound)
    ledger = OnlinePilotAssignmentLedgerV1(work / "assignment-ledger.jsonl",
        plan=campaign.pilot_plan(profile), source_commit=head)
    ledger.start_game("g1")
    assigned = _game_assignment(ledger.plan, "g1")
    ledger.persist_assignment(assigned)
    before = {p: p.read_bytes() for p in work.rglob("*") if p.is_file()}
    monkeypatch.setattr(OnlinePilotAssignmentLedgerV1, "_append", lambda *_: pytest.fail("status must not write"))
    result = campaign.status(profile)
    assert result["assignments"] == 1 and result["executions"] == 0
    assert result["remaining_target"] == 119
    assert result["PUSH"] + result["REDIRECT"] == 1
    assert before == {p: p.read_bytes() for p in work.rglob("*") if p.is_file()}
    ledger.path.unlink()
    with pytest.raises(FileNotFoundError):
        campaign.status(profile)
    assert not ledger.path.exists()


@pytest.mark.parametrize("command", ("prepare", "preflight", "run", "resume"))
def test_qualification_only_readonly_status(setup, command, monkeypatch):
    monkeypatch.setattr(campaign, "qualification_inputs", lambda: pytest.fail("no qualification execution"))
    assert campaign.main([command, "qualification"]) == 1


def test_formal_paths_cannot_mutate_qualification(setup):
    profile, bound, *_ = setup
    profile["work_directory"] = str(Path(bound["paths"]["work_directory"]) / "new")
    with pytest.raises(ValueError, match="qualification"):
        campaign.protect_qualification(profile, bound)


def test_verified_qualification_pins_and_digest_tamper(tmp_path, monkeypatch):
    """Real artifact publication/verification; explicitly synthetic test evidence."""
    plan = Phase2OnlineTerminalPilotPlanV1(QUALIFICATION_NAME, 17,
        campaign_purpose="qualification", max_games_attempted=40)
    from scripts.collect_games import plan_fields
    from werewolf.canonical_collection.attempt_ledger import construct_collection_plan
    game = construct_collection_plan(ordered_seed_pool=tuple(range(1, 41)),
        **plan_fields({"collection_id": QUALIFICATION_NAME, "target_games": 40,
            "seed_pool_size": 40, "call_limit": 1000}, "a" * 40, {"served_model_name": "fixture"}))
    work, destination = tmp_path / "work", tmp_path / QUALIFICATION_NAME
    bound = {"pilot_plan": plan.to_record(), "canonical_game_plan": game.to_record(),
        "source_commit": "a" * 40, "paths": {"destination": str(destination), "work_directory": str(work)},
        "mapper_manifest_digest": "b" * 64, "reference_tables_digest": "c" * 64,
        "smoke_v3_manifest_digest": "d" * 64, "q_seal_digest": "e" * 64}
    bound["inputs_digest"] = campaign.digest(bound)
    write(work / "run_inputs.json", bound)
    ledger = OnlinePilotAssignmentLedgerV1(work / "assignment-ledger.jsonl", plan=plan, source_commit="a" * 40)
    for index in range(10):
        ledger.start_game(f"g{index}")
        ledger.persist_assignment(_game_assignment(plan, f"g{index}"))
    ledger.mark_interrupted_on_resume()
    terminal = ledger.snapshot()["events"][-1]["digest"]
    artifact = publish_artifact(destination, manifest_fields={
        "artifact_type": "phase2_online_terminal_pilot", "schema_version": ARTIFACT_VERSION,
        "study_name": QUALIFICATION_NAME, "campaign_purpose": "qualification",
        "completion_status": "COMPLETE", "pilot_plan_digest": plan.digest(),
        "ledger_digest": terminal,
        "source_provenance": {"commit": "a" * 40},
        "server_run_provenance": {"inputs_digest": bound["inputs_digest"], "game_plan_digest": game.plan_digest},
        "mapper_artifact_digest": "b" * 64, "reference_artifact_digest": "c" * 64,
        "smoke_v3_manifest_digest": "d" * 64, "q_source_digest": "e" * 64}, files={
        "pilot_plan.json": canonical_json_bytes(plan.to_record()),
        "metrics/support.json": canonical_json_bytes({"estimator_eligible": False, "games_assigned": 10})})
    pins = {"schema_version": "phase2_online_campaign_runtime_pins_v1",
        "qualification_artifact": str(destination), "qualification_manifest_digest": artifact.manifest_digest,
        "qualification_run_inputs": str(work / "run_inputs.json"), "qualification_inputs_digest": bound["inputs_digest"]}
    monkeypatch.setattr(campaign, "PINS", tmp_path / "pins.json")
    write(campaign.PINS, pins)
    assert campaign.qualification_inputs()[1] == bound
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "unchanged-for-readonly-status")
    monkeypatch.setattr(OnlinePilotAssignmentLedgerV1, "_append", lambda *_: pytest.fail("readonly"))
    result = campaign.status(None, qualification=True)
    assert result["status"] == "COMPLETE" and result["assignments"] == 10
    assert result["estimator_eligible"] is False and result["ledger_terminal_digest"] == terminal
    assert campaign.main(["status", "qualification"]) == 0
    import os
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == "unchanged-for-readonly-status"
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    bound["mapper_manifest_digest"] = "f" * 64
    write(work / "run_inputs.json", bound)
    with pytest.raises(ValueError, match="binding"):
        campaign.qualification_inputs()
