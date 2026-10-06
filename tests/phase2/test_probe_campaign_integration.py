"""Operator seams with real Git, frozen JSON plans, and the durable Probe ledger.

The real collect_games and server helper bodies are compiled without importing
optional simulator/model assembly. No gameplay, model dispatch, or package shim
is executed. Synthetic evidence is confined to temporary test directories.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from scripts import phase2_online_campaign as campaign
from werewolf.artifact_io import canonical_json_bytes
from werewolf.artifact_io.canonical import ensure_durable_directory
from werewolf.canonical_collection import attempt_ledger as canonical
from werewolf.canonical_collection import game_bundle, pre, public_history, speech, trajectory_evidence
from werewolf.phase2_online_ledger import OnlinePilotAssignmentLedgerV1
from werewolf.phase2_online_plan import PROBE_PLAN_VERSION, Phase2OnlineProbePilotPlanV1, Phase2OnlineTerminalPilotPlanV1
from werewolf.phase2_online_preflight import SOURCE_FILES
from tests.phase2.test_probe_policy_ledger import frozen_assignment, LiteralRecord, literal_snapshots

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SEEDS = (196812520389446858, 8584631348015271354, 7279773504223456818,
                  2711710838872032642, 6642236894969768391)


def git(repo, *arguments):
    return subprocess.check_output(("git", "-C", str(repo), *arguments), text=True).strip()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def definitions(path, names, **scope):
    """Compile exact production bodies; do not substitute permissive callables."""
    module = ModuleType(path.stem)
    module.__dict__.update(scope)
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if
             isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
             or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                 and target.id in names for target in node.targets)]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    compiled = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(compiled, str(path), "exec"), module.__dict__)
    return module


def constant(path, name):
    return definitions(ROOT / path, {name}).__dict__[name]


@pytest.fixture
def assembly(monkeypatch):
    collection = definitions(ROOT / "scripts/collect_games.py", {
        "CALIBRATION_SEEDS", "derive_seed_pool", "plan_fields", "load_plan", "publish_plan", "read_json"},
        hashlib=hashlib, json=json, Path=Path,
        canonical_json_bytes=canonical_json_bytes,
        ensure_durable_directory=ensure_durable_directory,
        _publish_bytes_noreplace=canonical._publish_bytes_noreplace,
        collection_plan_from_record=canonical.collection_plan_from_record,
        validate_collection_plan=canonical.validate_collection_plan,
        STRICT_CLASSIC7_GAMEPLAY_PROMPT_PROFILE=constant("werewolf/agents/prompt_template_v0.py", "STRICT_CLASSIC7_GAMEPLAY_PROMPT_PROFILE"),
        GAMEPLAY_GENERATION_MAX_ATTEMPTS=constant("werewolf/agents/gpt_agent.py", "GAMEPLAY_GENERATION_MAX_ATTEMPTS"),
        LABEL_GENERATION_MAX_ATTEMPTS=constant("werewolf/speech/private_belief_perceiver.py", "LABEL_GENERATION_MAX_ATTEMPTS"),
        SPEECH_PARSER_GENERATION_MAX_ATTEMPTS=constant("werewolf/speech/speech_perceiver.py", "SPEECH_PARSER_GENERATION_MAX_ATTEMPTS"),
        PUBLIC_EVENT_SCHEMA_VERSION=public_history.PUBLIC_EVENT_SCHEMA_VERSION,
        AUTHORITATIVE_PRE_PREFIX_SCHEMA_VERSION=pre.AUTHORITATIVE_PRE_PREFIX_SCHEMA_VERSION,
        BELIEF_OBSERVATION_SCHEMA_VERSION=trajectory_evidence.BELIEF_OBSERVATION_SCHEMA_VERSION,
        V1_ANNOTATION_SCHEMA_VERSION=speech.V1_ANNOTATION_SCHEMA_VERSION,
        V1_SPEECH_PARSER_VERSION=speech.V1_SPEECH_PARSER_VERSION,
        V1_SPEECH_PROMPT_VERSION=speech.V1_SPEECH_PROMPT_VERSION,
        CANONICAL_GAME_BUNDLE_SCHEMA_VERSION=game_bundle.CANONICAL_GAME_BUNDLE_SCHEMA_VERSION)
    server = definitions(ROOT / "werewolf/phase2_online_server.py", {
        "_read_json", "_publish_json", "load_online_plan"},
        json=json, re=re, Path=Path, ensure_durable_directory=ensure_durable_directory,
        _publish_bytes_noreplace=canonical._publish_bytes_noreplace,
        canonical_json_bytes=canonical_json_bytes, PROBE_PLAN_VERSION=PROBE_PLAN_VERSION,
        Phase2OnlineProbePilotPlanV1=Phase2OnlineProbePilotPlanV1,
        Phase2OnlineTerminalPilotPlanV1=Phase2OnlineTerminalPilotPlanV1)
    runner = definitions(ROOT / "werewolf/phase2_online_runner.py", {
        "ARTIFACT_NAME", "QUALIFICATION_NAME", "ARTIFACT_VERSION"})
    for name, module in (("scripts.collect_games", collection),
                         ("werewolf.phase2_online_server", server),
                         ("werewolf.phase2_online_runner", runner)):
        monkeypatch.setitem(sys.modules, name, module)
    return collection, server


@pytest.fixture
def operator(tmp_path, monkeypatch, assembly):
    collection, server = assembly
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "user.email", "test@example.com")
    for name in (*SOURCE_FILES, "configs/phase2/online-terminal-pilot-formal-v1.json",
                 "configs/phase2/online-terminal-runtime-pins-v1.json",
                 "configs/phase2/online-probe-qualification-v1.json",
                 "configs/phase2/online-probe-qualification-v2.json",
                 "configs/phase2/online-probe-qualification-v3.json",
                 "configs/phase2/online-probe-formal-v1.json",
                 "docs/research/phase2-probe-policy-protocol-v1.md"):
        destination = repo / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    monkeypatch.setattr(campaign, "REPO", repo)
    monkeypatch.setattr(campaign, "__file__", str(repo / "scripts/phase2_online_campaign.py"))
    monkeypatch.setattr(campaign, "PROFILE", repo / "configs/phase2/online-terminal-pilot-formal-v1.json")
    monkeypatch.setattr(campaign, "PINS", repo / "configs/phase2/online-terminal-runtime-pins-v1.json")
    monkeypatch.setattr(campaign, "PROBE_PROFILES", {
        "qualification": repo / "configs/phase2/online-probe-qualification-v3.json",
        "formal": repo / "configs/phase2/online-probe-formal-v1.json"})
    profile = campaign.read_profile(policy="probe", qualification=True)
    profile.update(planning_status="FROZEN", assignment_seed=frozen_assignment()[0].assignment_seed,
                   target_assignment_count=2, max_games_attempted=5)
    paths = {name: str(tmp_path / name) for name in ("runtime_config", "deployment_config",
             "publication", "evaluation_root", "mapper", "smoke_v3", "q_checkout", "q_fit")}
    paths.update(repo=str(repo), work_directory=str(tmp_path / "terminal-qualification-work"),
                 destination=str(tmp_path / "terminal-qualification"))
    for name in ("plan", "game_plan", "work_directory"):
        profile[name] = str(tmp_path / ("probe-" + name))
    profile["destination"] = str(tmp_path / campaign.PROBE_NAMES["qualification"])
    terminal_profile = campaign.read_profile()
    terminal_profile.update({name: str(tmp_path / ("terminal-" + name)) for name in
                             ("plan", "game_plan", "work_directory")})
    terminal_profile["destination"] = str(tmp_path / terminal_profile["pilot_id"])
    write(campaign.PROFILE, terminal_profile)
    provenance = {"served_model_name": "synthetic-test-only", "runtime_config_sha256": "1" * 64,
                  "serve_config_sha256": "2" * 64, "model_manifest_sha256": "3" * 64}
    terminal = canonical.construct_collection_plan(ordered_seed_pool=tuple(range(101, 106)),
        **collection.plan_fields({"collection_id": terminal_profile["pilot_id"], "target_games": 5,
                                 "seed_pool_size": 5, "call_limit": 10000}, "a" * 40, provenance))
    write(terminal_profile["game_plan"], terminal.to_record())
    profile["excluded_game_plans"] = [{"path": terminal_profile["game_plan"], "plan_digest": terminal.plan_digest}]
    old_profile = campaign.read_json(repo / "configs/phase2/online-probe-qualification-v1.json")
    for name in ("plan", "game_plan", "work_directory", "destination"):
        old_profile[name] = str(tmp_path / ("failed-probe-v1-" + name))
    write(repo / "configs/phase2/online-probe-qualification-v1.json", old_profile)
    old_probe = canonical.construct_collection_plan(ordered_seed_pool=tuple(range(201, 281)),
        **collection.plan_fields({"collection_id": old_profile["pilot_id"], "target_games": 80,
                                 "seed_pool_size": 80, "call_limit": 10000}, "a" * 40, provenance))
    write(old_profile["game_plan"], old_probe.to_record())
    profile["excluded_game_plans"].append({"path": old_profile["game_plan"],
                                         "plan_digest": old_probe.plan_digest})
    failed_v2 = campaign.read_json(repo / "configs/phase2/online-probe-qualification-v2.json")
    for name in ("plan", "game_plan", "work_directory", "destination"):
        failed_v2[name] = str(tmp_path / ("failed-probe-v2-" + name))
    write(repo / "configs/phase2/online-probe-qualification-v2.json", failed_v2)
    failed_v2_game = canonical.construct_collection_plan(ordered_seed_pool=tuple(range(301, 381)),
        **collection.plan_fields({"collection_id": failed_v2["pilot_id"], "target_games": 80,
                                 "seed_pool_size": 80, "call_limit": 10000}, "a" * 40, provenance))
    write(failed_v2["game_plan"], failed_v2_game.to_record())
    profile["excluded_game_plans"].append({"path": failed_v2["game_plan"],
                                         "plan_digest": failed_v2_game.plan_digest})
    write(campaign.PROBE_PROFILES["qualification"], profile)
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "synthetic operator source")
    head = git(repo, "rev-parse", "HEAD")
    qualified = canonical.construct_collection_plan(ordered_seed_pool=tuple(range(801, 841)),
        **collection.plan_fields({"collection_id": "terminal-qualification", "target_games": 40,
                                 "seed_pool_size": 40, "call_limit": 10000}, "b" * 40, provenance))
    bound = {"paths": paths, "source_commit": "b" * 40, "canonical_game_plan": qualified.to_record(),
             "mapper_manifest_digest": "4" * 64, "reference_tables_digest": "5" * 64,
             "smoke_v3_manifest_digest": "6" * 64, "q_python": "/synthetic/python"}
    development = canonical.construct_collection_plan(ordered_seed_pool=tuple(range(10001, 12251)),
        **collection.plan_fields({"collection_id": "development", "target_games": 1500,
                                 "seed_pool_size": 2250, "call_limit": 10000}, "c" * 40, provenance))
    dev_path = tmp_path / "development.json"
    write(dev_path, development.to_record())
    publication = {"collection_plan_digest": development.plan_digest}
    publication["manifest_digest"] = campaign.digest(publication)
    write(Path(paths["publication"]) / "manifest.json", publication)
    pins = {"development_game_plan": str(dev_path), "development_game_plan_digest": development.plan_digest,
            "publication_manifest_digest": publication["manifest_digest"]}
    # Existing terminal qualification artifact admission is an external seam;
    # these tests exercise actual Probe pin/exclusion and operator contracts.
    monkeypatch.setattr(campaign, "qualification_inputs", lambda: (pins, bound, None))
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    return SimpleNamespace(profile=profile, repo=repo, head=head, bound=bound, pins=pins,
                           terminal=terminal, collection=collection, server=server)


def test_formal_partial_freeze_is_readable_but_cannot_prepare(assembly, tmp_path):
    profile = campaign.read_profile(policy="probe", qualification=False)
    # A partial profile still fails closed after the committed profile is frozen.
    profile.update(planning_status="UNFROZEN", max_games_attempted=None)
    assert profile["planning_status"] == "UNFROZEN"
    assert profile["target_assignment_count"] == 200
    assert profile["assignment_seed"] == 6449283966833280907
    assert profile["max_games_attempted"] is None
    with pytest.raises(ValueError, match="not frozen"):
        campaign.prepare(profile, source_commit="a" * 40)


def test_prepare_binds_clean_source_protocol_profile_and_fixed_seed_pool_exclusively(operator):
    op = operator
    (op.repo / "unrelated-untracked.md").write_text("must not block")
    result = campaign.prepare(op.profile, source_commit=op.head)
    game = canonical.collection_plan_from_record(campaign.read_json(op.profile["game_plan"]))
    assert result["source_commit"] == game.source_revision == op.head
    assert game.ordered_seed_pool == EXPECTED_SEEDS
    environment = dict(game.environment_provenance)
    assert environment["probe_policy_protocol_sha256"] == op.profile["probe_protocol_digest"]
    assert environment["probe_campaign_profile_digest"] == campaign.digest(op.profile)
    assert result["seed_overlap_counts"] == {"qualification": 0, "development": 0,
        "calibration": 0, "paper-phase2-online-terminal-pilot-v1": 0,
        "paper-phase2-online-probe-qualification-v1": 0,
        "paper-phase2-online-probe-qualification-v2": 0}
    before = Path(op.profile["plan"]).read_bytes(), Path(op.profile["game_plan"]).read_bytes()
    with pytest.raises(FileExistsError):
        campaign.prepare(op.profile, source_commit=op.head)
    assert before == (Path(op.profile["plan"]).read_bytes(), Path(op.profile["game_plan"]).read_bytes())


@pytest.mark.parametrize("fault", ["source_mismatch", "tracked_dirty", "index_dirty"])
def test_prepare_blocks_source_fault_before_runtime_or_publication(operator, fault):
    op = operator
    source = "f" * 40 if fault == "source_mismatch" else op.head
    if fault != "source_mismatch":
        path = op.repo / "scripts/phase2_online_campaign.py"
        path.write_text(path.read_text() + "\n# dirty\n")
        if fault == "index_dirty":
            git(op.repo, "add", str(path))
            assert git(op.repo, "diff", "--name-only") == ""
            assert git(op.repo, "diff", "--cached", "--name-only")
    with pytest.raises(ValueError, match="clean|source-commit"):
        campaign.prepare(op.profile, source_commit=source)
    assert not Path(op.profile["plan"]).exists()
    assert not Path(op.profile["work_directory"]).exists()


@pytest.fixture
def formal_operator(operator, monkeypatch):
    op = operator
    profile = campaign.read_profile(policy="probe")
    for key in ("plan", "game_plan", "work_directory"):
        profile[key] = str(op.repo.parent / ("formal-" + key))
    profile["destination"] = str(op.repo.parent / profile["pilot_id"])
    profile["excluded_game_plans"] = op.profile["excluded_game_plans"]
    runtime_pins = {key: str(index) * 64 for index, key in enumerate((
        "q_fit_digest", "q_seal_digest", "runtime_config_sha256", "deployment_config_sha256"), 1)}
    op.bound.update(runtime_pins)
    qualification_profile = campaign.read_profile(policy="probe", qualification=True)
    qualification_profile.update(target_assignment_count=20, max_games_attempted=80)
    qualified = campaign.canonical_plan(
        campaign.pilot_plan(qualification_profile), op.head, op.bound, profile=qualification_profile)
    assert len(qualified.ordered_seed_pool) == 80
    probe_bound = {**op.bound, "source_commit": op.head,
                   "pilot_plan": campaign.pilot_plan(qualification_profile).to_record(),
                   "canonical_game_plan": qualified.to_record(),
                   "paths": {key: op.profile[key] for key in (
                       "plan", "game_plan", "work_directory", "destination")}}
    profile["probe_qualification"]["artifact"] = probe_bound["paths"]["destination"]
    profile["probe_qualification"]["run_inputs"] = str(
        Path(probe_bound["paths"]["work_directory"]) / "run_inputs.json")
    # Durable artifact admission is tested separately with the real journal.
    # Here its verified output enters the production prepare/exclusion path.
    monkeypatch.setattr(campaign, "probe_qualification_inputs", lambda _: (probe_bound, None))
    write(campaign.PROBE_PROFILES["formal"], profile)
    git(op.repo, "add", "configs/phase2/online-probe-formal-v1.json")
    git(op.repo, "commit", "-qm", "synthetic frozen formal profile")
    return SimpleNamespace(op=op, profile=profile, qualified=qualified,
                           head=git(op.repo, "rev-parse", "HEAD"))


def test_frozen_formal_prepare_uses_existing_v3_admission_and_excludes_full_pool(formal_operator):
    f = formal_operator
    result = campaign.prepare(f.profile, source_commit=f.head)
    game = canonical.collection_plan_from_record(campaign.read_json(f.profile["game_plan"]))
    assert result["status"] == "FROZEN" and result["seed_pool_size"] == 800
    assert game.source_revision == f.head
    assert game.ordered_seed_pool == f.op.collection.derive_seed_pool(f.profile["pilot_id"], 800)
    assert f.op.server.load_online_plan(f.profile["plan"], "pilot").target_assignment_count == 200
    _, _, _, excluded = campaign.probe_inputs(f.profile)
    v3 = [old for old in excluded if old.collection_id == campaign.PROBE_NAMES["qualification"]]
    assert v3 == [f.qualified] and len(v3[0].ordered_seed_pool) == 80
    assert len(excluded) == 4  # Terminal formal, V1, V2, and V3 exactly once.
    assert all(set(game.ordered_seed_pool).isdisjoint(old.ordered_seed_pool) for old in excluded)
    assert all(count == 0 for count in result["seed_overlap_counts"].values())
    assert result["seed_overlap_counts"][campaign.PROBE_NAMES["qualification"]] == 0


def test_formal_prepare_rejects_even_unexecuted_last_v3_planned_seed(formal_operator, monkeypatch):
    f = formal_operator
    last = f.qualified.ordered_seed_pool[-1]
    original = f.op.collection.derive_seed_pool
    monkeypatch.setattr(f.op.collection, "derive_seed_pool", lambda identity, size:
                        (last, *original(identity, size)[1:]))
    with pytest.raises(ValueError, match="seed overlap; no skip/reroll"):
        campaign.prepare(f.profile, source_commit=f.head)
    assert not Path(f.profile["plan"]).exists()
    assert not Path(f.profile["game_plan"]).exists()


@pytest.mark.parametrize("pool", ["qualification", "development", "calibration", "terminal", "probe_v1_planned"])
def test_overlap_is_rejected_without_reroll_or_plan_publication(operator, monkeypatch, pool):
    op = operator
    seed = {"qualification": 801, "development": 10001,
            "calibration": 900000001, "terminal": 101, "probe_v1_planned": 280}[pool]
    monkeypatch.setattr(op.collection, "derive_seed_pool", lambda identity, size:
                        (seed, *EXPECTED_SEEDS[1:]))
    with pytest.raises(ValueError, match="overlap"):
        campaign.prepare(op.profile, source_commit=op.head)
    assert not Path(op.profile["plan"]).exists()


@pytest.mark.parametrize("version", ("v1", "v2"))
@pytest.mark.parametrize("fault", ["missing", "unbound", "wrong_digest", "reuse_path"])
def test_v3_prepare_requires_bound_prior_exclusions_and_preserves_evidence(operator, version, fault):
    op = operator
    old_profile = campaign.read_json(op.repo / f"configs/phase2/online-probe-qualification-{version}.json")
    old_bytes = Path(old_profile["game_plan"]).read_bytes()
    index = next(i for i, item in enumerate(op.profile["excluded_game_plans"])
                 if item["path"] == old_profile["game_plan"])
    if fault == "missing":
        op.profile["excluded_game_plans"].pop(index)
    elif fault == "unbound":
        op.profile["excluded_game_plans"][index]["plan_digest"] = None
    elif fault == "wrong_digest":
        op.profile["excluded_game_plans"][index]["plan_digest"] = "0" * 64
    else:
        op.profile["work_directory"] = old_profile["work_directory"]
    with pytest.raises(ValueError, match="V[12] seed pool|digest is not frozen|digest differs|overlap qualification evidence"):
        campaign.prepare(op.profile, source_commit=op.head)
    assert Path(old_profile["game_plan"]).read_bytes() == old_bytes
    assert not Path(op.profile["plan"]).exists()


@pytest.mark.parametrize("command", ["preflight", "run", "resume"])
def test_existing_server_parser_receives_frozen_probe_identity_and_only_requested_mode(operator, command):
    op = operator
    campaign.prepare(op.profile, source_commit=op.head)
    args = campaign.server_args(op.profile, command)
    assert args.campaign_purpose == "qualification"
    assert args.source_commit == op.head
    assert args.plan == Path(op.profile["plan"])
    assert args.game_plan == Path(op.profile["game_plan"])
    assert args.resume is (command == "resume")
    assert args.preflight_only is (command == "preflight")
    assert op.server.load_online_plan(args.plan, args.campaign_purpose) == campaign.pilot_plan(op.profile)
    assert args.mapper_manifest_digest == op.bound["mapper_manifest_digest"]
    assert args.reference_tables_digest == op.bound["reference_tables_digest"]


@pytest.mark.parametrize("fault", ["source_mismatch", "tracked_dirty", "protocol", "excluded_plan"])
def test_cli_run_rejects_invalid_bindings_before_server_execution(operator, monkeypatch, fault):
    op = operator
    campaign.prepare(op.profile, source_commit=op.head)
    monkeypatch.setattr(op.server, "execute_server_campaign", lambda args: pytest.fail("must fail before runtime"), raising=False)
    if fault == "source_mismatch":
        git(op.repo, "commit", "--allow-empty", "-qm", "source mismatch")
    elif fault == "tracked_dirty":
        (op.repo / "scripts/phase2_online_campaign.py").write_text("dirty\n")
    elif fault == "protocol":
        path = op.repo / "docs/research/phase2-probe-policy-protocol-v1.md"
        path.write_text("changed protocol\n")
        git(op.repo, "add", ".")
        git(op.repo, "commit", "-qm", "changed protocol")
    else:
        raw = campaign.read_json(op.profile["excluded_game_plans"][0]["path"])
        raw["source_revision"] = "f" * 40
        write(op.profile["excluded_game_plans"][0]["path"], raw)
    assert campaign.main(["run", "qualification", "--policy", "probe"]) == 1
    assert not Path(op.profile["work_directory"]).exists()


def test_cli_delegates_preflight_and_resume_to_same_server_entry(operator, monkeypatch, capsys):
    op = operator
    assert campaign.main(["prepare", "qualification", "--policy", "probe", "--source-commit", op.head]) == 0
    capsys.readouterr()
    calls = []
    def execute(args):
        calls.append(args)
        # A fake dispatch cannot claim a completed/published qualification.
        return 7 if args.resume else 0
    monkeypatch.setattr(op.server, "execute_server_campaign", execute, raising=False)
    assert campaign.main(["preflight", "qualification", "--policy", "probe"]) == 0
    assert campaign.main(["resume", "qualification", "--policy", "probe"]) == 7
    assert [args.preflight_only for args in calls] == [True, False]
    assert [args.resume for args in calls] == [False, True]
    assert all(args.campaign_purpose == "qualification" for args in calls)
    assert not Path(op.profile["work_directory"]).exists()


def test_status_reads_real_partial_probe_ledger_without_writes_or_rescheduling(operator, monkeypatch):
    op = operator
    campaign.prepare(op.profile, source_commit=op.head)
    assert campaign.status(op.profile)["status"] == "NOT_STARTED"
    work = Path(op.profile["work_directory"])
    bound = {"source_commit": op.head, "pilot_plan": campaign.pilot_plan(op.profile).to_record(),
             "paths": {"work_directory": str(work), "destination": op.profile["destination"]}}
    bound["inputs_digest"] = campaign.digest(bound)
    write(work / "run_inputs.json", bound)
    ledger = OnlinePilotAssignmentLedgerV1(work / "assignment-ledger.jsonl",
        plan=campaign.pilot_plan(op.profile), source_commit=op.head)
    _, assignment = frozen_assignment(count=2, cap=5)
    # The authoritative constructor verifies this assignment against this plan;
    # use the same frozen randomization seed but the operator campaign identity.
    from werewolf.phase2_online_plan import assign_probe_strategy, select_probe_candidate
    assignment = assign_probe_strategy(ledger.plan,
        select_probe_candidate(ledger.plan, assignment.opportunity.legal_context), assignment.opportunity)
    ledger.start_game(assignment.opportunity.legal_context.game_id)
    ledger.persist_assignment(assignment)
    ledger.persist_strategy_stage(LiteralRecord(assignment, literal_snapshots(assignment)[0]))
    before = {path: path.read_bytes() for path in work.rglob("*") if path.is_file()}
    monkeypatch.setattr(OnlinePilotAssignmentLedgerV1, "_append", lambda *args, **kwargs: pytest.fail("status is read-only"))
    result = campaign.status(op.profile)
    assert result["assignments"] == 1
    assert result["executions"] == result["consequences"] == result["backend_calls"] == 0
    assert result["unfinished_assigned_games"] == ["game-1"]
    assert result["resume_behavior"] == "FAIL_CLOSED_ASSIGNED_GAME"
    assert result["IMMEDIATE_REDIRECT"] + result["PROBE_THEN_REDIRECT"] == 1
    assert result["qualification_gate"]["passed"] is False
    assert before == {path: path.read_bytes() for path in work.rglob("*") if path.is_file()}


@pytest.mark.parametrize("existing", ["plan", "game_plan", "work_directory", "destination"])
def test_prepare_never_overwrites_or_reuses_existing_campaign_paths(operator, existing):
    op = operator
    path = Path(op.profile[existing])
    if existing in ("plan", "game_plan"):
        path.write_bytes(b"existing evidence\n")
    else:
        path.mkdir()
        (path / "evidence").write_bytes(b"preserve\n")
    before = path.read_bytes() if path.is_file() else (path / "evidence").read_bytes()
    with pytest.raises(FileExistsError):
        campaign.prepare(op.profile, source_commit=op.head)
    assert before == (path.read_bytes() if path.is_file() else (path / "evidence").read_bytes())
    if existing != "plan":
        assert not Path(op.profile["plan"]).exists()


def test_formal_prepare_requires_frozen_probe_qualification_admission(operator):
    op = operator
    formal = campaign.read_profile(policy="probe", qualification=False)
    formal.update(planning_status="FROZEN", assignment_seed=19,
                  target_assignment_count=3, max_games_attempted=6,
                  excluded_game_plans=op.profile["excluded_game_plans"])
    for name in ("plan", "game_plan", "work_directory"):
        formal[name] = str(op.repo.parent / ("formal-probe-" + name))
    formal["destination"] = str(op.repo.parent / campaign.PROBE_NAMES["pilot"])
    formal["probe_qualification"].update(manifest_digest=None, inputs_digest=None)
    with pytest.raises(ValueError, match="Probe qualification digests are not frozen"):
        campaign.prepare(formal, source_commit=op.head)
    assert not Path(formal["plan"]).exists()
    assert not Path(formal["work_directory"]).exists()


def test_explicit_source_commit_required_and_unfrozen_status_remains_read_only(operator, capsys):
    op = operator
    assert campaign.main(["prepare", "qualification", "--policy", "probe"]) == 1
    assert not Path(op.profile["plan"]).exists()
    unfrozen = campaign.read_profile(policy="probe", qualification=False)
    result = campaign.status(unfrozen)
    assert result["status"] == "NOT_STARTED"
    assert result["remaining_target"] == 200
