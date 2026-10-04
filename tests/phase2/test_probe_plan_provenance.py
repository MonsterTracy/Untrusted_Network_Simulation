"""Pure Probe protocol/profile binding and actual server input-order tests."""

import ast
from contextlib import ExitStack, nullcontext
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from types import SimpleNamespace

import pytest

from tests.canonical_collection.test_attempt_ledger import _plan
from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection.attempt_ledger import collection_plan_from_record
from werewolf.phase2_online_plan import (
    PROBE_PLAN_VERSION, Phase2OnlineProbePilotPlanV1, Phase2OnlineTerminalPilotPlanV1,
)
from werewolf.phase2_online_preflight import verify_probe_plan_provenance
from werewolf.phase2_online_ledger import OnlinePilotAssignmentLedgerV1


SOURCE_COMMIT = "a" * 40
PROTOCOL_PATH = "docs/research/phase2-probe-policy-protocol-v1.md"


def probe_inputs(tmp_path, *, purpose="qualification"):
    repo = tmp_path / "repo"
    protocol = repo / PROTOCOL_PATH
    protocol.parent.mkdir(parents=True)
    protocol.write_text("Frozen local Probe policy fixture.\n")
    plan = Phase2OnlineProbePilotPlanV1(
        "local-probe-fixture", 712, 2, 4, campaign_purpose=purpose,
    )
    profile = {
        "schema_version": "phase2_online_probe_campaign_profile_v1",
        "estimator_eligible": False,
        "planning_status": "FROZEN",
        "probe_protocol_digest": sha256_bytes(protocol.read_bytes()),
        "pilot_id": plan.pilot_id, "campaign_purpose": plan.campaign_purpose,
        "assignment_seed": plan.assignment_seed,
        "target_assignment_count": plan.target_assignment_count,
        "max_games_attempted": plan.max_games_attempted,
    }
    game_plan = _plan(
        collection_id=plan.pilot_id, source_revision=SOURCE_COMMIT,
        ordered_seed_pool=(11, 12, 13, 14), target_canonical_success_count=2,
        environment_provenance={
            "probe_policy_protocol_sha256": profile["probe_protocol_digest"],
            "probe_campaign_profile_digest": sha256_bytes(canonical_json_bytes(profile)),
        },
    )
    return repo, plan, game_plan, profile


@pytest.mark.parametrize("purpose", ["qualification", "pilot"])
def test_matching_real_probe_plan_protocol_and_profile_are_accepted(tmp_path, purpose):
    repo, plan, game_plan, profile = probe_inputs(tmp_path, purpose=purpose)
    assert verify_probe_plan_provenance(repo, plan, game_plan, profile) is None


@pytest.mark.parametrize("mutation", [
    "changed_protocol", "changed_profile", "missing_protocol_pin",
    "missing_profile_pin", "unfrozen", "assignment_seed", "target_assignment_count",
    "max_games_attempted", "pilot_id", "campaign_purpose", "profile_protocol_pin",
    "missing_profile", "missing_protocol_file", "wrong_schema", "estimator_eligible",
])
def test_probe_provenance_rejects_drift_and_unfrozen_registration(tmp_path, mutation):
    repo, plan, game_plan, profile = probe_inputs(tmp_path)
    provenance = dict(game_plan.environment_provenance)
    if mutation == "changed_protocol":
        (repo / PROTOCOL_PATH).write_text("Altered policy after freezing.\n")
    elif mutation == "changed_profile":
        profile["note"] = "A newly edited profile cannot keep the old digest."
    elif mutation in ("missing_protocol_pin", "missing_profile_pin"):
        del provenance["probe_policy_protocol_sha256" if mutation == "missing_protocol_pin"
                       else "probe_campaign_profile_digest"]
    elif mutation == "missing_profile":
        profile = None
    elif mutation == "missing_protocol_file":
        (repo / PROTOCOL_PATH).unlink()
    else:
        if mutation == "unfrozen":
            profile["planning_status"] = "UNFROZEN"
        elif mutation == "profile_protocol_pin":
            profile["probe_protocol_digest"] = "0" * 64
        elif mutation in ("assignment_seed", "target_assignment_count", "max_games_attempted"):
            profile[mutation] += 1
        elif mutation == "pilot_id":
            profile[mutation] = "different-campaign"
        elif mutation == "wrong_schema":
            profile["schema_version"] = "phase2_online_terminal_campaign_profile_v1"
        elif mutation == "estimator_eligible":
            profile[mutation] = True
        else:
            profile[mutation] = "pilot"
        # A valid outer profile hash cannot hide disagreement with the plan.
        provenance["probe_campaign_profile_digest"] = sha256_bytes(canonical_json_bytes(profile))
    game_plan = _plan(
        collection_id=plan.pilot_id, source_revision=SOURCE_COMMIT,
        ordered_seed_pool=(11, 12, 13, 14), target_canonical_success_count=2,
        environment_provenance=provenance,
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_probe_plan_provenance(repo, plan, game_plan, profile)


@pytest.fixture
def server_input_scope():
    # Execute the production definitions unchanged, excluding only optional
    # server import assembly. All tests terminate before runtime construction.
    source = ast.parse(Path("werewolf/phase2_online_server.py").read_text())
    definitions = [node for node in source.body if isinstance(node, ast.FunctionDef)
                   and node.name in ("_read_json", "load_online_plan", "_require_package_source",
                                     "load_server_inputs", "execute_server_campaign")]
    scope = dict(
        os=os, Path=Path, shutil=shutil, re=re, json=json, ExitStack=ExitStack,
        PROBE_PLAN_VERSION=PROBE_PLAN_VERSION,
        Phase2OnlineProbePilotPlanV1=Phase2OnlineProbePilotPlanV1,
        Phase2OnlineTerminalPilotPlanV1=Phase2OnlineTerminalPilotPlanV1,
        collection_plan_from_record=collection_plan_from_record,
        verify_probe_plan_provenance=verify_probe_plan_provenance,
    )
    # Use the actual production constants, never a second test-only identity.
    runner = ast.parse(Path("werewolf/phase2_online_runner.py").read_text())
    for node in runner.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in (
                        "PROBE_QUALIFICATION_NAME", "PROBE_ARTIFACT_NAME"):
                    scope[target.id] = ast.literal_eval(node.value)
    exec(compile(ast.Module(body=definitions, type_ignores=[]),
                 "werewolf/phase2_online_server.py", "exec"), scope)
    return scope


def server_args(tmp_path, repo, plan, game_plan, profile):
    plan_path, game_plan_path = tmp_path / "pilot-plan.json", tmp_path / "game-plan.json"
    purpose = "qualification" if plan.campaign_purpose == "qualification" else "formal"
    args = SimpleNamespace(
        repo=repo, destination=tmp_path / ("paper-phase2-online-probe-qualification-v2"
            if purpose == "qualification" else "paper-phase2-online-probe-pilot-v1"),
        work_directory=tmp_path / "work", q_python=sys.executable,
        plan=plan_path, campaign_purpose=plan.campaign_purpose, source_commit=SOURCE_COMMIT,
        game_plan=game_plan_path,
        runtime_config=tmp_path / "unread-runtime-config.yaml",
        mapper_manifest_digest="b" * 64, reference_tables_digest="c" * 64,
        smoke_v3_manifest_digest="d" * 64,
    )
    for key in ("deployment_config", "publication", "evaluation_root", "mapper",
                "smoke_v3", "q_checkout", "q_fit"):
        setattr(args, key, tmp_path / key)
    for key in ("plan", "game_plan", "work_directory", "destination"):
        profile[key] = str(getattr(args, key))
    provenance = dict(game_plan.environment_provenance)
    provenance["probe_campaign_profile_digest"] = sha256_bytes(canonical_json_bytes(profile))
    game_plan = _plan(
        collection_id=plan.pilot_id, source_revision=SOURCE_COMMIT,
        ordered_seed_pool=(11, 12, 13, 14), target_canonical_success_count=2,
        environment_provenance=provenance,
    )
    from scripts.phase2_online_campaign import PROBE_PROFILES
    profile_path = repo / "configs/phase2" / PROBE_PROFILES[purpose].name
    profile_path.parent.mkdir(parents=True)
    profile_path.write_bytes(canonical_json_bytes(profile))
    plan_path.write_bytes(canonical_json_bytes(plan.to_record()))
    game_plan_path.write_bytes(canonical_json_bytes(game_plan.to_record()))
    return args


def shared_admission_fixture(monkeypatch, args, profile, *, calls, rejection=None):
    """Isolate qualified external evidence, retaining the actual server guard."""
    from scripts import phase2_online_campaign as campaign
    qualified = {key: getattr(args, key) for key in (
        "mapper_manifest_digest", "reference_tables_digest", "smoke_v3_manifest_digest")}
    qualified["q_python"] = str(Path(args.q_python).resolve())
    qualified["paths"] = {key: str(getattr(args, key)) for key in (
        "runtime_config", "deployment_config", "publication", "evaluation_root",
        "mapper", "smoke_v3", "q_checkout", "q_fit")}
    game = collection_plan_from_record(json.loads(args.game_plan.read_bytes()))

    def probe_inputs(current_profile):
        calls.append("qualification admission")
        assert current_profile == profile
        if rejection == "qualification":
            raise ValueError("qualification admission rejected")
        return {}, qualified, None, (game,) if rejection == "overlap" else ()

    def canonical_plan(current_plan, head, bound, *, profile):
        calls.append("canonical plan binding")
        assert current_plan.to_record() == json.loads(args.plan.read_bytes())
        assert head == SOURCE_COMMIT and bound == qualified
        return game

    monkeypatch.setattr(campaign, "probe_inputs", probe_inputs)
    monkeypatch.setattr(campaign, "canonical_plan", canonical_plan)
    # Preserve checked_seed_overlaps itself. Its independent frozen-data
    # lookup is isolated, while excluded-plan intersections stay real.
    monkeypatch.setattr(campaign, "seed_overlaps", lambda seeds, pins, bound: {"development": 0})
    return qualified


@pytest.mark.parametrize("mutation", ["protocol", "profile", "source"])
def test_actual_server_entry_rejects_changed_pins_before_worker_or_backend(
        server_input_scope, tmp_path, mutation):
    repo, plan, game_plan, profile = probe_inputs(tmp_path)
    args = server_args(tmp_path, repo, plan, game_plan, profile)
    scope = server_input_scope
    scope["__file__"] = str(repo / "werewolf/phase2_online_server.py")
    scope["freeze_online_source_provenance"] = lambda path: {
        "commit": "b" * 40 if mutation == "source" else SOURCE_COMMIT,
    }
    calls = []

    def guard(*args):
        calls.append("protocol/profile guard")
        return verify_probe_plan_provenance(*args)

    def forbidden(*args, **kwargs):
        calls.append("forbidden runtime/model work")
        raise AssertionError("invalid provenance must fail before worker/backend startup")

    scope["verify_probe_plan_provenance"] = guard
    scope["Qwen3GameplayPredictorClient"] = forbidden
    scope["load_named_backends"] = forbidden
    scope["normalize_runtime_config"] = forbidden
    scope["load_runtime_mapper"] = forbidden
    if mutation == "protocol":
        (repo / PROTOCOL_PATH).write_text("Changed after plans were frozen.\n")
    elif mutation == "profile":
        profile["assignment_seed"] += 1
        (repo / "configs/phase2/online-probe-qualification-v2.json").write_bytes(
            canonical_json_bytes(profile))
    message = "source/index" if mutation == "source" else "protocol/profile"
    with pytest.raises(ValueError, match=message):
        scope["execute_server_campaign"](args)
    assert calls == ([] if mutation == "source" else ["protocol/profile guard"])
    assert not args.work_directory.exists() and not args.destination.exists()


def test_actual_input_loader_reaches_config_only_after_valid_probe_guard(
        server_input_scope, tmp_path, monkeypatch):
    repo, plan, game_plan, profile = probe_inputs(tmp_path)
    args = server_args(tmp_path, repo, plan, game_plan, profile)
    scope = server_input_scope
    scope["__file__"] = str(repo / "werewolf/phase2_online_server.py")
    scope["freeze_online_source_provenance"] = lambda path: {"commit": SOURCE_COMMIT}
    calls = []

    def guard(*args):
        calls.append("verified")
        return verify_probe_plan_provenance(*args)

    scope["verify_probe_plan_provenance"] = guard
    shared_admission_fixture(monkeypatch, args, profile, calls=calls)
    scope["normalize_runtime_config"] = lambda value: value
    scope["yaml"] = SimpleNamespace(safe_load=lambda value: value)
    # The next production input is intentionally absent. This proves the
    # valid binding is accepted without assembling any model or runtime.
    with pytest.raises(FileNotFoundError):
        scope["load_server_inputs"](args)
    assert calls == ["verified", "qualification admission", "canonical plan binding"]


@pytest.mark.parametrize("purpose", ["qualification", "pilot"])
@pytest.mark.parametrize("mutation", [
    "qualification", "overlap", "runtime_config", "deployment_config", "publication",
    "evaluation_root", "mapper", "smoke_v3", "q_checkout", "q_fit",
    "mapper_manifest_digest", "reference_tables_digest", "smoke_v3_manifest_digest", "q_python",
])
def test_raw_server_cli_cannot_bypass_shared_admission_or_qualified_pins(
        server_input_scope, tmp_path, monkeypatch, purpose, mutation):
    repo, plan, game_plan, profile = probe_inputs(tmp_path, purpose=purpose)
    args = server_args(tmp_path, repo, plan, game_plan, profile)
    scope = server_input_scope
    scope["__file__"] = str(repo / "werewolf/phase2_online_server.py")
    scope["freeze_online_source_provenance"] = lambda path: {"commit": SOURCE_COMMIT}
    calls = []
    qualified = shared_admission_fixture(monkeypatch, args, profile, calls=calls,
                                         rejection=mutation)

    def forbidden(*args, **kwargs):
        raise AssertionError("raw CLI rejection must precede Q/backend/config work")

    scope["normalize_runtime_config"] = forbidden
    scope["Qwen3GameplayPredictorClient"] = forbidden
    scope["load_named_backends"] = forbidden
    if mutation in qualified["paths"]:
        qualified["paths"][mutation] = str(tmp_path / "changed-qualified-path")
        message = "runtime path differs"
    elif mutation in qualified:
        qualified[mutation] = "/different/frozen/python" if mutation == "q_python" else "0" * 64
        message = "runtime pin differs"
    else:
        message = "qualification admission rejected" if mutation == "qualification" else "seed overlap"
    with pytest.raises(ValueError, match=message):
        scope["execute_server_campaign"](args)
    assert calls[0] == "qualification admission"
    assert not args.work_directory.exists() and not args.destination.exists()


@pytest.mark.parametrize("purpose", ["qualification", "pilot"])
@pytest.mark.parametrize("path_key", ["plan", "game_plan", "work_directory", "destination"])
def test_raw_server_cli_paths_must_equal_frozen_profile(
        server_input_scope, tmp_path, monkeypatch, purpose, path_key):
    repo, plan, game_plan, profile = probe_inputs(tmp_path, purpose=purpose)
    args = server_args(tmp_path, repo, plan, game_plan, profile)
    profile[path_key] = str(tmp_path / "different-frozen-profile-path")
    provenance = dict(game_plan.environment_provenance)
    provenance["probe_campaign_profile_digest"] = sha256_bytes(canonical_json_bytes(profile))
    game_plan = _plan(
        collection_id=plan.pilot_id, source_revision=SOURCE_COMMIT,
        ordered_seed_pool=(11, 12, 13, 14), target_canonical_success_count=2,
        environment_provenance=provenance,
    )
    args.game_plan.write_bytes(canonical_json_bytes(game_plan.to_record()))
    purpose_name = "qualification" if purpose == "qualification" else "formal"
    from scripts.phase2_online_campaign import PROBE_PROFILES
    (repo / "configs/phase2" / PROBE_PROFILES[purpose_name].name).write_bytes(canonical_json_bytes(profile))
    calls = []
    shared_admission_fixture(monkeypatch, args, profile, calls=calls)
    scope = server_input_scope
    scope["__file__"] = str(repo / "werewolf/phase2_online_server.py")
    scope["freeze_online_source_provenance"] = lambda path: {"commit": SOURCE_COMMIT}
    with pytest.raises(ValueError, match="server paths differ from frozen profile"):
        scope["execute_server_campaign"](args)
    assert calls == ["qualification admission", "canonical plan binding"]
    assert not args.work_directory.exists() and not args.destination.exists()


def test_old_v1_resume_is_rejected_before_worker_or_durable_write(server_input_scope, tmp_path):
    repo, plan, game_plan, profile = probe_inputs(tmp_path)
    args = server_args(tmp_path, repo, plan, game_plan, profile)
    args.resume = True
    args.destination = tmp_path / "paper-phase2-online-probe-qualification-v1"
    args.work_directory.mkdir()
    sentinel = args.work_directory / "assignment-ledger.jsonl"
    sentinel.write_bytes(b"immutable failed V1 evidence\n")
    server_input_scope["__file__"] = str(repo / "werewolf/phase2_online_server.py")
    server_input_scope["Qwen3GameplayPredictorClient"] = lambda **kw: pytest.fail("must reject before worker")
    with pytest.raises(ValueError, match="distinct fixed artifact names"):
        server_input_scope["execute_server_campaign"](args)
    assert sentinel.read_bytes() == b"immutable failed V1 evidence\n"
    assert not args.destination.exists()


def test_actual_server_preflight_only_certifies_runtime_without_game_or_durable_work(
        server_input_scope, tmp_path):
    scope = server_input_scope
    repo, plan, game_plan, profile = probe_inputs(tmp_path)
    args = server_args(tmp_path, repo, plan, game_plan, profile)
    args.resume, args.preflight_only = False, True
    seen = []
    scope["load_server_inputs"] = lambda args: (plan, game_plan, {}, 10000, None, None, {})
    scope["tempfile"] = tempfile
    scope["OnlinePilotAssignmentLedgerV1"] = OnlinePilotAssignmentLedgerV1
    scope["Qwen3GameplayPredictorClient"] = lambda **kw: nullcontext("fixture-worker-handshake")
    scope["load_named_backends"] = lambda *a, **kw: {}
    class Factory:
        def __init__(self, **kwargs):
            assert kwargs["ledger"].plan == plan
            assert not kwargs["work_directory"].is_relative_to(args.work_directory)
            self.game_ids = ("fixture-unclaimed-game",)
        def __call__(self, game_id, **kwargs):
            seen.append((game_id, kwargs))
            return SimpleNamespace(preflight=SimpleNamespace(ready=True, to_record=lambda: {"ready": True}))
    scope["ServerRuntimeFactory"] = Factory
    scope["run_online_campaign"] = lambda *a, **kw: pytest.fail("preflight must not run game")
    scope["publish_online_pilot"] = lambda *a, **kw: pytest.fail("preflight must not publish")
    assert scope["execute_server_campaign"](args) == 0
    assert seen == [("fixture-unclaimed-game", {"persist_claim": False, "publication_only": False})]
    assert not args.work_directory.exists()
    assert not args.destination.exists()
