"""Thin operator CLI for terminal and Probe campaigns on the same server runtime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
PROFILE = REPO / "configs/phase2/online-terminal-pilot-formal-v1.json"
PINS = REPO / "configs/phase2/online-terminal-runtime-pins-v1.json"
PROFILE_VERSION = "phase2_online_campaign_profile_v1"
PROBE_PROFILE_VERSION = "phase2_online_probe_campaign_profile_v1"
PROBE_PROFILES = {"qualification": REPO / "configs/phase2/online-probe-qualification-v2.json",
                  "formal": REPO / "configs/phase2/online-probe-formal-v1.json"}
PROBE_NAMES = {"qualification": "paper-phase2-online-probe-qualification-v2",
               "pilot": "paper-phase2-online-probe-pilot-v1"}
CUBLAS = ":4096:8"


def frozen_environment():
    value = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if value is not None and value != CUBLAS:
        raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS


def read_json(path):
    from werewolf.artifact_io.canonical import _reject_duplicate_keys, _reject_json_constant
    return json.loads(Path(path).read_bytes(), object_pairs_hook=_reject_duplicate_keys,
                      parse_constant=_reject_json_constant)


def digest(value):
    from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
    return sha256_bytes(canonical_json_bytes(value))


def read_profile(*, policy="terminal", qualification=False):
    probe = policy == "probe"
    profile = read_json(PROBE_PROFILES["qualification" if qualification else "formal"]
                        if probe else PROFILE)
    fields = {"schema_version", "campaign_purpose", "pilot_id", "assignment_seed",
              "max_games_attempted", "target_assignment_count", "estimator_eligible",
              "plan", "game_plan", "work_directory", "destination"}
    if probe:
        fields |= {"planning_status", "probe_protocol_digest", "excluded_game_plans"}
        purpose = "qualification" if qualification else "pilot"
        if purpose == "pilot":
            fields.add("probe_qualification")
        if (set(profile) != fields or profile["schema_version"] != PROBE_PROFILE_VERSION
                or profile["campaign_purpose"] != purpose
                or profile["pilot_id"] != PROBE_NAMES[purpose]
                or profile["estimator_eligible"] is not False
                or profile["planning_status"] not in ("UNFROZEN", "FROZEN")
                or not isinstance(profile["probe_protocol_digest"], str)
                or len(profile["probe_protocol_digest"]) != 64
                or any(c not in "0123456789abcdef" for c in profile["probe_protocol_digest"])
                or not isinstance(profile["excluded_game_plans"], list)
                or not profile["excluded_game_plans"]):
            raise ValueError("invalid Probe campaign profile")
        for item in profile["excluded_game_plans"]:
            if (set(item) != {"path", "plan_digest"}
                    or not Path(item["path"]).is_absolute()
                    or item["plan_digest"] is not None and (
                        not isinstance(item["plan_digest"], str) or len(item["plan_digest"]) != 64
                        or any(c not in "0123456789abcdef" for c in item["plan_digest"]))):
                raise ValueError("excluded canonical plans require absolute paths and digests")
        for key in ("assignment_seed", "target_assignment_count", "max_games_attempted"):
            value = profile[key]
            if value is not None and (type(value) is not int or value < (0 if key == "assignment_seed" else 1)):
                raise ValueError(f"invalid Probe profile {key}")
        if purpose == "pilot":
            pins = profile["probe_qualification"]
            if (not isinstance(pins, dict) or set(pins) != {
                    "artifact", "run_inputs", "manifest_digest", "inputs_digest"}
                    or not all(Path(pins[k]).is_absolute() for k in ("artifact", "run_inputs"))):
                raise ValueError("invalid Probe qualification pointers")
        expected_name = PROBE_NAMES[purpose]
    else:
        from werewolf.phase2_online_plan import TARGET_ASSIGNMENTS
        from werewolf.phase2_online_runner import ARTIFACT_NAME
        if (set(profile) != fields or profile["schema_version"] != PROFILE_VERSION
            or profile["campaign_purpose"] != "pilot"
            or profile["pilot_id"] != ARTIFACT_NAME
            or type(profile["target_assignment_count"]) is not int
            or profile["target_assignment_count"] != TARGET_ASSIGNMENTS["pilot"]
            or profile["estimator_eligible"] is not True):
            raise ValueError("invalid formal campaign profile")
        expected_name = ARTIFACT_NAME
    paths = []
    for key in ("plan", "game_plan", "work_directory", "destination"):
        path = Path(profile[key])
        if not path.is_absolute():
            raise ValueError(f"profile {key} must be absolute")
        paths.append(path.resolve())
    if any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
           for i, a in enumerate(paths) for b in paths[i + 1:]):
        raise ValueError("plan/work/destination paths must be disjoint")
    if Path(profile["destination"]).name != expected_name:
        raise ValueError("formal destination must use the fixed pilot artifact name")
    return profile


def pilot_plan(profile):
    from werewolf.phase2_online_plan import Phase2OnlineTerminalPilotPlanV1, Phase2OnlineProbePilotPlanV1
    if profile["schema_version"] == PROBE_PROFILE_VERSION:
        if profile["planning_status"] != "FROZEN" or any(profile[k] is None for k in (
                "assignment_seed", "target_assignment_count", "max_games_attempted")):
            raise ValueError("Probe target/seed/cap are not frozen; prepare is disabled")
        return Phase2OnlineProbePilotPlanV1(profile["pilot_id"], profile["assignment_seed"],
            profile["target_assignment_count"], profile["max_games_attempted"],
            campaign_purpose=profile["campaign_purpose"])
    if profile["assignment_seed"] is None:
        raise ValueError("formal assignment_seed is not frozen; provide it before prepare")
    if profile["max_games_attempted"] is None:
        raise ValueError("formal max_games_attempted is not frozen; provide the safety cap")
    return Phase2OnlineTerminalPilotPlanV1(profile["pilot_id"], profile["assignment_seed"],
        campaign_purpose=profile["campaign_purpose"],
        max_games_attempted=profile["max_games_attempted"])


def qualification_inputs():
    """Pins are pointers to verified completed evidence, not copied runtime artifacts."""
    from werewolf.artifact_io import read_artifact_file, verify_artifact
    from werewolf.canonical_collection.attempt_ledger import collection_plan_from_record
    from werewolf.phase2_online_plan import Phase2OnlineTerminalPilotPlanV1
    from werewolf.phase2_online_runner import ARTIFACT_VERSION, QUALIFICATION_NAME
    pins = read_json(PINS)
    if pins["schema_version"] != "phase2_online_campaign_runtime_pins_v1":
        raise ValueError("invalid runtime pins schema")
    artifact = verify_artifact(Path(pins["qualification_artifact"]),
        expected_artifact_type="phase2_online_terminal_pilot", expected_schema_version=ARTIFACT_VERSION)
    manifest = artifact.manifest
    if (artifact.manifest_digest != pins["qualification_manifest_digest"]
            or manifest["study_name"] != QUALIFICATION_NAME
            or manifest["campaign_purpose"] != "qualification"
            or manifest["completion_status"] != "COMPLETE"):
        raise ValueError("completed qualification provenance differs")
    bound = read_json(pins["qualification_run_inputs"])
    if (bound["inputs_digest"] != pins["qualification_inputs_digest"]
            or bound["inputs_digest"] != digest({k: v for k, v in bound.items() if k != "inputs_digest"})
            or manifest["server_run_provenance"]["inputs_digest"] != bound["inputs_digest"]
            or manifest["source_provenance"]["commit"] != bound["source_commit"]):
        raise ValueError("qualification run_inputs binding differs")
    raw = bound["pilot_plan"]
    plan = Phase2OnlineTerminalPilotPlanV1(raw["pilot_id"], raw["assignment_seed"],
        campaign_purpose=raw["campaign_purpose"], max_games_attempted=raw["max_games_attempted"])
    support = json.loads(read_artifact_file(artifact, "metrics/support.json"))
    game = collection_plan_from_record(bound["canonical_game_plan"])
    if (plan.campaign_purpose != "qualification" or raw != plan.to_record()
            or json.loads(read_artifact_file(artifact, "pilot_plan.json")) != raw
            or manifest["pilot_plan_digest"] != plan.digest()
            or game.collection_id != plan.pilot_id or game.source_revision != bound["source_commit"]
            or manifest["server_run_provenance"]["game_plan_digest"] != game.plan_digest
            or support["estimator_eligible"] is not False
            or support["games_assigned"] != plan.target_assignment_count
            or Path(bound["paths"]["destination"]).resolve() != artifact.path.resolve()
            or Path(pins["qualification_run_inputs"]).resolve() !=
               Path(bound["paths"]["work_directory"]).resolve() / "run_inputs.json"):
        raise ValueError("qualification plan/population lineage differs")
    for field, manifest_field in (("mapper_manifest_digest", "mapper_artifact_digest"),
                                  ("reference_tables_digest", "reference_artifact_digest"),
                                  ("smoke_v3_manifest_digest", "smoke_v3_manifest_digest"),
                                  ("q_seal_digest", "q_source_digest")):
        if bound[field] != manifest[manifest_field]:
            raise ValueError(f"qualification {field} lineage differs")
    return pins, bound, artifact


def probe_qualification_gate(records, *, complete):
    """Mechanism coverage on already validated, latest durable lifecycle records."""
    checks = {"all_assignments_complete": complete,
        "immediate_redirect_committed": any(
            row["assignment"]["assigned_strategy"] == "IMMEDIATE_REDIRECT"
            and "T1_COMMITTED" in row["lifecycle"] for row in records),
        "probe_then_redirect_committed": any(
            row["assignment"]["assigned_strategy"] == "PROBE_THEN_REDIRECT"
            and "T3_COMMITTED" in row["lifecycle"] for row in records)}
    return {"passed": all(checks.values()), "checks": checks}


def probe_qualification_inputs(profile):
    """Formal admission requires immutable, completed Probe qualification evidence."""
    from werewolf.artifact_io import read_artifact_file, verify_artifact
    from werewolf.phase2_online_ledger import OnlinePilotAssignmentLedgerV1
    from werewolf.phase2_online_server import load_online_plan
    from werewolf.phase2_online_preflight import freeze_online_source_provenance
    pins = profile["probe_qualification"]
    if any(pins[k] is None for k in ("manifest_digest", "inputs_digest")):
        raise ValueError("Probe qualification digests are not frozen")
    artifact = verify_artifact(Path(pins["artifact"]),
        expected_artifact_type="phase2_online_probe_pilot",
        expected_schema_version="phase2_online_probe_pilot_v1")
    manifest = artifact.manifest
    bound = read_json(pins["run_inputs"])
    if (artifact.manifest_digest != pins["manifest_digest"]
            or manifest["study_name"] != PROBE_NAMES["qualification"]
            or manifest["campaign_purpose"] != "qualification"
            or manifest["completion_status"] != "COMPLETE"
            or bound["inputs_digest"] != pins["inputs_digest"]
            or bound["inputs_digest"] != digest({k: v for k, v in bound.items() if k != "inputs_digest"})
            or manifest["server_run_provenance"]["inputs_digest"] != bound["inputs_digest"]
            or manifest["source_provenance"]["commit"] != bound["source_commit"]
            or Path(pins["run_inputs"]).resolve() !=
               Path(bound["paths"]["work_directory"]).resolve() / "run_inputs.json"
            or Path(bound["paths"]["destination"]).resolve() != artifact.path.resolve()):
        raise ValueError("Probe qualification provenance differs")
    source = freeze_online_source_provenance(REPO)
    if source is None or source["source_sha256"] != manifest["source_provenance"]["source_sha256"]:
        raise ValueError("Probe runtime source bytes differ from qualification")
    plan = load_online_plan(artifact.path / "pilot_plan.json", "qualification")
    if (bound["pilot_plan"] != plan.to_record() or manifest["pilot_plan_digest"] != plan.digest()
            or bound["canonical_game_plan"]["source_revision"] != bound["source_commit"]
            or bound["canonical_game_plan"]["collection_id"] != plan.pilot_id
            or manifest["server_run_provenance"]["game_plan_digest"] !=
               bound["canonical_game_plan"]["plan_digest"]
            or bound["canonical_game_plan"]["environment_provenance"].get(
                "probe_policy_protocol_sha256") != profile["probe_protocol_digest"]):
        raise ValueError("Probe qualification plan/protocol differs")
    ledger = object.__new__(OnlinePilotAssignmentLedgerV1)
    ledger.path = Path(bound["paths"]["work_directory"]) / "assignment-ledger.jsonl"
    ledger.plan, ledger.source_commit = plan, bound["source_commit"]
    snapshot = ledger.snapshot()
    stages = list(snapshot["games"].values())
    if not ledger.sealable():
        raise ValueError("Probe qualification is not complete/sealable")
    records = [s["STRATEGY_STAGE"]["record"] for s in stages if "ASSIGNMENT" in s]
    support = json.loads(read_artifact_file(artifact, "metrics/support.json"))
    if (manifest["ledger_digest"] != snapshot["events"][-1]["digest"]
            or json.loads(read_artifact_file(artifact, "pilot_plan.json")) != bound["pilot_plan"]
            or support["estimator_eligible"] is not False
            or support["games_assigned"] != plan.target_assignment_count
            or support["itt_outcome_complete"] is not True):
        raise ValueError("Probe qualification ledger/population differs")
    # Published tables must be exactly the authoritative durable evidence.
    expected = {
        "assignments.jsonl": [s["ASSIGNMENT"]["assignment"] for s in stages if "ASSIGNMENT" in s],
        "executions.jsonl": [dict(game_id=s["EXECUTION"]["game_id"],
            assignment_digest=s["EXECUTION"]["assignment_id"], **s["EXECUTION"]["execution"],
            theta_audit_label=None) for s in stages if "EXECUTION" in s],
        "consequences.jsonl": [dict(game_id=s["CONSEQUENCE"]["game_id"],
            assignment_digest=s["CONSEQUENCE"]["assignment_id"],
            day_consequence=s["CONSEQUENCE"]["day_consequence"], theta_audit_label=None,
            final_game_result=s["GAME_RESULT"]["winner"]) for s in stages if "CONSEQUENCE" in s],
        "backend_calls.jsonl": [call["call"] for s in stages for call in s.get("BACKEND_CALL", ())],
        "strategy_stages.jsonl": [r for s in stages for r in s.get("STRATEGY_STAGE_HISTORY", ())],
    }
    for name, rows in expected.items():
        if [json.loads(line) for line in read_artifact_file(artifact, name).splitlines()] != rows:
            raise ValueError(f"Probe qualification {name} differs from ledger")
    gate = probe_qualification_gate(records, complete=ledger.sealable())
    if not gate["passed"]:
        raise ValueError(f"Probe qualification mechanism gate failed: {gate['checks']}")
    return bound, artifact


def clean_source(*, probe=False):
    from werewolf.phase2_online_preflight import _git, freeze_online_source_provenance
    source = freeze_online_source_provenance(REPO)
    if source is None:
        raise ValueError("tracked source and staged index must be clean")
    # Untracked research is allowed; the new operator code/config must be committed.
    paths = [Path(__file__), PROFILE, PINS]
    if probe:
        paths.extend((*PROBE_PROFILES.values(), REPO / "docs/research/phase2-probe-policy-protocol-v1.md"))
    for path in paths:
        name = str(path.resolve().relative_to(REPO.resolve()))
        if _git(REPO, "ls-files", "--error-unmatch", name).returncode:
            raise ValueError(f"operator source/config must be committed before freezing: {name}")
    return source


def seed_overlaps(seeds, pins, bound):
    from scripts.collect_games import CALIBRATION_SEEDS
    from werewolf.canonical_collection.attempt_ledger import collection_plan_from_record
    development = collection_plan_from_record(read_json(pins["development_game_plan"]))
    publication = read_json(Path(bound["paths"]["publication"]) / "manifest.json")
    if (development.plan_digest != pins["development_game_plan_digest"]
            or publication["manifest_digest"] != pins["publication_manifest_digest"]
            or digest({k: v for k, v in publication.items() if k != "manifest_digest"}) !=
               publication["manifest_digest"]
            or publication["collection_plan_digest"] != development.plan_digest
            or development.target_canonical_success_count != 1500):
        raise ValueError("development seed pool/publication provenance differs")
    selected = set(seeds)
    overlaps = {"qualification": len(selected.intersection(bound["canonical_game_plan"]["ordered_seed_pool"])),
                "development": len(selected.intersection(development.ordered_seed_pool)),
                "calibration": len(selected.intersection(CALIBRATION_SEEDS))}
    if any(overlaps.values()):
        raise ValueError(f"seed overlap; no skip/reroll: {overlaps}")
    return overlaps


def canonical_plan(plan, head, bound, *, profile=None):
    from scripts.collect_games import derive_seed_pool, plan_fields
    from werewolf.canonical_collection.attempt_ledger import construct_collection_plan, validate_collection_plan
    provenance = bound["canonical_game_plan"]["environment_provenance"]
    extra = {k: v for k, v in provenance.items() if k not in (
        "configured_call_limit", "gameplay_prompt_profile", "seed_pool_size", "target_success_count")}
    if profile is not None:
        extra.update(probe_policy_protocol_sha256=profile["probe_protocol_digest"],
                     probe_campaign_profile_digest=digest(profile))
    return validate_collection_plan(construct_collection_plan(
        ordered_seed_pool=derive_seed_pool(plan.pilot_id, plan.max_games_attempted),
        **plan_fields({"collection_id": plan.pilot_id, "target_games": plan.max_games_attempted,
                      "seed_pool_size": plan.max_games_attempted,
                      "call_limit": int(provenance["configured_call_limit"])}, head, extra)))


def protect_qualification(profile, bound):
    for key in ("plan", "game_plan", "work_directory", "destination"):
        output = Path(profile[key]).resolve()
        for evidence in (bound["paths"][k] for k in (
                "plan", "game_plan", "work_directory", "destination") if k in bound["paths"]):
            protected = Path(evidence).resolve()
            if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
                raise ValueError("formal paths must not overlap qualification evidence")


def probe_inputs(profile):
    """Reuse qualified model/runtime pins; add Probe-specific evidence separation."""
    from werewolf.artifact_io import sha256_bytes
    from werewolf.canonical_collection.attempt_ledger import collection_plan_from_record
    if sha256_bytes((REPO / "docs/research/phase2-probe-policy-protocol-v1.md").read_bytes()) != profile["probe_protocol_digest"]:
        raise ValueError("Probe protocol digest differs")
    pins, bound, artifact = qualification_inputs()
    protect_qualification(profile, bound)
    terminal_profile = read_profile()
    protect_qualification(profile, {"paths": terminal_profile})
    other = read_profile(policy="probe", qualification=profile["campaign_purpose"] != "qualification")
    protect_qualification(profile, {"paths": other})
    failed_probe = read_json(REPO / "configs/phase2/online-probe-qualification-v1.json")
    protect_qualification(profile, {"paths": failed_probe})
    excluded = []
    for item in profile["excluded_game_plans"]:
        if item["plan_digest"] is None:
            raise ValueError("excluded game-plan digest is not frozen")
        game = collection_plan_from_record(read_json(item["path"]))
        if game.plan_digest != item["plan_digest"]:
            raise ValueError("excluded game-plan digest differs")
        excluded.append(game)
    if not any(g.collection_id == terminal_profile["pilot_id"] for g in excluded):
        raise ValueError("Probe must exclude the formal Terminal seed pool")
    if not any(g.collection_id == "paper-phase2-online-probe-qualification-v1" for g in excluded):
        raise ValueError("Probe must exclude the failed Qualification V1 seed pool")
    if profile["campaign_purpose"] == "pilot":
        probe_bound, _ = probe_qualification_inputs(profile)
        protect_qualification(profile, probe_bound)
        for key in ("mapper_manifest_digest", "reference_tables_digest", "smoke_v3_manifest_digest",
                    "q_fit_digest", "q_seal_digest", "runtime_config_sha256", "deployment_config_sha256"):
            if probe_bound[key] != bound[key]:
                raise ValueError(f"Probe qualification runtime pin differs: {key}")
        excluded.append(collection_plan_from_record(probe_bound["canonical_game_plan"]))
    return pins, bound, artifact, excluded


def checked_seed_overlaps(game, pins, bound, excluded=()):
    counts = seed_overlaps(game.ordered_seed_pool, pins, bound)
    for old in excluded:
        counts[old.collection_id] = len(set(game.ordered_seed_pool).intersection(old.ordered_seed_pool))
    if any(counts.values()):
        raise ValueError(f"seed overlap; no skip/reroll: {counts}")
    return counts


def prepare(profile, *, source_commit=None):
    plan = pilot_plan(profile)
    probe = profile["schema_version"] == PROBE_PROFILE_VERSION
    source = clean_source(probe=True) if probe else clean_source()
    if probe and source_commit != source["commit"]:
        raise ValueError("Probe prepare requires explicit --source-commit equal to clean HEAD")
    for key in ("plan", "game_plan", "work_directory", "destination"):
        if os.path.lexists(profile[key]):
            raise FileExistsError(f"prepare refuses existing {key}; no overwrite or implicit resume")
    if probe:
        pins, bound, _, excluded = probe_inputs(profile)
    else:
        pins, bound, _ = qualification_inputs()
        excluded = ()
    protect_qualification(profile, bound)
    if Path(bound["paths"]["repo"]).resolve() != REPO.resolve():
        raise ValueError("prepare must run in the verified server checkout")
    game = canonical_plan(plan, source["commit"], bound, profile=profile) if probe else canonical_plan(plan, source["commit"], bound)
    overlaps = checked_seed_overlaps(game, pins, bound, excluded)
    if (clean_source(probe=True) if probe else clean_source()) != source:
        raise ValueError("source changed during prepare")
    from scripts.collect_games import publish_plan
    from werewolf.phase2_online_server import _publish_json
    # Both are exclusive publications. A partial failure is retained, never silently regenerated.
    _publish_json(Path(profile["plan"]), plan.to_record())
    publish_plan(Path(profile["game_plan"]), game)
    return {"status": "FROZEN", "source_commit": source["commit"],
            "pilot_plan_digest": plan.digest(), "game_plan_digest": game.plan_digest,
            "assignment_seed": plan.assignment_seed, "seed_pool_size": len(game.ordered_seed_pool),
            "seed_overlap_counts": overlaps}


def server_args(profile, command):
    plan = pilot_plan(profile)
    from scripts.run_phase2_online_intervention_pilot import build_parser
    from werewolf.canonical_collection.attempt_ledger import collection_plan_from_record
    from werewolf.phase2_online_server import load_online_plan
    probe = profile["schema_version"] == PROBE_PROFILE_VERSION
    source = clean_source(probe=True) if probe else clean_source()
    frozen = load_online_plan(Path(profile["plan"]), profile["campaign_purpose"])
    game = collection_plan_from_record(read_json(profile["game_plan"]))
    if frozen != plan or game.source_revision != source["commit"]:
        raise ValueError("frozen plans differ from current profile/source")
    if probe:
        pins, bound, _, excluded = probe_inputs(profile)
    else:
        pins, bound, _ = qualification_inputs()
        excluded = ()
    protect_qualification(profile, bound)
    expected = (canonical_plan(plan, source["commit"], bound, profile=profile) if probe
                else canonical_plan(plan, source["commit"], bound))
    if game != expected:
        raise ValueError("frozen canonical plan differs from qualified runtime pins/production derivation")
    checked_seed_overlaps(game, pins, bound, excluded)
    values = {**{k: bound["paths"][k] for k in ("runtime_config", "deployment_config", "publication",
                "evaluation_root", "mapper", "smoke_v3", "q_checkout", "q_fit")},
              **{k: bound[k] for k in ("mapper_manifest_digest", "reference_tables_digest",
                  "smoke_v3_manifest_digest", "q_python")},
              **{k: profile[k] for k in ("plan", "game_plan", "work_directory", "destination")},
              "repo": str(REPO), "campaign_purpose": profile["campaign_purpose"], "source_commit": game.source_revision}
    argv = [part for k, v in values.items() for part in ("--" + k.replace("_", "-"), str(v))]
    if command == "preflight":
        argv.append("--preflight-only")
    if command == "resume":
        argv.append("--resume")
    return build_parser().parse_args(argv)


def status(profile, *, qualification=False):
    """Use the existing ledger validator without its write-on-missing constructor."""
    from werewolf.phase2_online_ledger import OnlinePilotAssignmentLedgerV1
    from werewolf.artifact_io import verify_artifact
    probe = profile is not None and profile["schema_version"] == PROBE_PROFILE_VERSION
    if not probe:
        from werewolf.phase2_online_runner import ARTIFACT_VERSION
    else:
        ARTIFACT_VERSION = "phase2_online_probe_pilot_v1"
    artifact_type = "phase2_online_probe_pilot" if probe else "phase2_online_terminal_pilot"
    artifact = None
    if qualification and not probe:
        _, bound, artifact = qualification_inputs()
        profile = {**bound["pilot_plan"], **bound["paths"], "estimator_eligible": False}
        plan_path = artifact.path / "pilot_plan.json"
        source = bound["source_commit"]
    else:
        plan_path = Path(profile["plan"])
        source = (read_json(profile["game_plan"])["source_revision"]
                  if Path(profile["game_plan"]).is_file() else None)
    work, destination = Path(profile["work_directory"]), Path(profile["destination"])
    arms = ("IMMEDIATE_REDIRECT", "PROBE_THEN_REDIRECT") if probe else ("PUSH", "REDIRECT")
    result = {"status": "NOT_STARTED", "games_seen": 0, "assignments": 0,
              **{arm: 0 for arm in arms}, "executions": 0, "execution_failures": 0,
              "consequences": 0, "backend_calls": 0,
              "remaining_target": profile["target_assignment_count"],
              "estimator_eligible": profile["estimator_eligible"],
              "work_directory": str(work), "destination": str(destination),
              "source_commit": source, "ledger_terminal_digest": None}
    if not work.exists() and not destination.exists():
        return result
    from werewolf.phase2_online_server import load_online_plan
    bound = read_json(work / "run_inputs.json")
    plan = load_online_plan(plan_path, profile["campaign_purpose"])
    if (bound["source_commit"] != source or bound["pilot_plan"] != plan.to_record()
            or bound["inputs_digest"] != digest({k: v for k, v in bound.items() if k != "inputs_digest"})
            or Path(bound["paths"]["work_directory"]).resolve() != work.resolve()
            or Path(bound["paths"]["destination"]).resolve() != destination.resolve()):
        raise ValueError("status work evidence belongs to different inputs")
    ledger = object.__new__(OnlinePilotAssignmentLedgerV1)
    ledger.path, ledger.plan, ledger.source_commit = work / "assignment-ledger.jsonl", plan, source
    snap = ledger.snapshot()  # O_RDONLY + shared lock, fails if missing; never creates START.
    stages = list(snap["games"].values())
    result.update(status=ledger.status(), games_seen=snap["games_attempted"],
        assignments=snap["assignment_count"], ledger_terminal_digest=snap["events"][-1]["digest"],
        executions=sum("EXECUTION" in s for s in stages),
        execution_failures=sum("EXECUTION" in s and not s["EXECUTION"]["execution"]["success"] for s in stages),
        consequences=sum("CONSEQUENCE" in s for s in stages),
        backend_calls=sum(len(s.get("BACKEND_CALL", [])) for s in stages),
        remaining_target=max(0, plan.target_assignment_count - snap["assignment_count"]))
    for action in arms:
        result[action] = sum("ASSIGNMENT" in s and
            s["ASSIGNMENT"]["assignment"]["assigned_strategy" if probe else "assigned_action"] == action for s in stages)
    if probe:
        records = [s["STRATEGY_STAGE"]["record"] for s in stages if "STRATEGY_STAGE" in s]
        unfinished = [game for game, s in snap["games"].items() if "ASSIGNMENT" in s
                      and not {"EXECUTION", "CONSEQUENCE", "GAME_RESULT"}.issubset(s)]
        result.update(unfinished_assigned_games=unfinished,
            interrupted_games=sum("INTERRUPTED" in s for s in stages),
            preparation_failures=sum("PREPARATION_FAILURE" in s for s in stages),
            lifecycle_counts={event: sum(event in r["lifecycle"] for r in records) for event in (
                "T1_COMMITTED", "T1_LANGUAGE_INVALID", "T3_SCHEDULED", "T3_REACHED",
                "T3_PREPARED", "T3_COMMITTED", "T3_LANGUAGE_INVALID", "T3_CANCELLED", "STRUCTURAL_FAILURE")},
            resume_behavior="FAIL_CLOSED_ASSIGNED_GAME" if unfinished else "NO_ASSIGNED_GAME_REPLAY")
        if plan.campaign_purpose == "qualification":
            result["qualification_gate"] = probe_qualification_gate(records, complete=ledger.sealable())
    if destination.exists():
        artifact = artifact or verify_artifact(destination,
            expected_artifact_type=artifact_type, expected_schema_version=ARTIFACT_VERSION)
        if (artifact.manifest["campaign_purpose"] != plan.campaign_purpose
                or artifact.manifest["completion_status"] != "COMPLETE"
                or artifact.manifest["pilot_plan_digest"] != plan.digest()
                or artifact.manifest["ledger_digest"] != result["ledger_terminal_digest"]
                or artifact.manifest["server_run_provenance"]["inputs_digest"] != bound["inputs_digest"]):
            raise ValueError("published artifact differs from work evidence")
        result["status"] = "COMPLETE"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "run", "resume", "status"))
    parser.add_argument("campaign", choices=("formal", "qualification"))
    parser.add_argument("--policy", choices=("terminal", "probe"), default="terminal")
    parser.add_argument("--source-commit", help="explicit clean execution HEAD for Probe prepare")
    args = parser.parse_args(argv)
    try:
        if args.source_commit is not None and (args.policy != "probe" or args.command != "prepare"):
            raise ValueError("--source-commit is only for Probe prepare; execution uses the frozen game plan")
        if args.campaign == "qualification" and args.policy == "terminal":
            if args.command != "status":
                raise ValueError("qualification is completed and read-only; only status is allowed")
            output = status(None, qualification=True)
        else:
            if args.command != "status":
                frozen_environment()  # Before importing/starting any Q worker.
            profile = read_profile(policy="probe", qualification=args.campaign == "qualification") if args.policy == "probe" else read_profile()
            if args.command == "status":
                output = status(profile)
            else:
                if args.command == "prepare":
                    output = prepare(profile, source_commit=args.source_commit) if args.policy == "probe" else prepare(profile)
                else:
                    server = server_args(profile, args.command)
                    from werewolf.phase2_online_server import execute_server_campaign
                    print(json.dumps({"source_commit": server.source_commit,
                        "pilot_plan_digest": digest(read_json(server.plan)),
                        "game_plan_digest": read_json(server.game_plan)["plan_digest"],
                        "destination_absent": not os.path.lexists(server.destination),
                        "reference_tables_digest": server.reference_tables_digest}, sort_keys=True), flush=True)
                    code = execute_server_campaign(server)
                    if (code == 0 and args.policy == "probe" and args.campaign == "qualification"
                            and args.command != "preflight"):
                        output = status(profile)
                        print(json.dumps(output, sort_keys=True), flush=True)
                        return 0 if output["qualification_gate"]["passed"] else 1
                    return code
        print(json.dumps(output, sort_keys=True), flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"Campaign failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
