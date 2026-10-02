"""Thin operator CLI for frozen Phase-2 campaigns; qualification is read-only."""

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
CUBLAS = ":4096:8"


def frozen_environment():
    value = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if value is not None and value != CUBLAS:
        raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS


def read_json(path):
    from werewolf.phase2_online_server import _read_json
    return _read_json(path)


def digest(value):
    from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
    return sha256_bytes(canonical_json_bytes(value))


def read_profile():
    from werewolf.phase2_online_plan import TARGET_ASSIGNMENTS
    from werewolf.phase2_online_runner import ARTIFACT_NAME
    profile = read_json(PROFILE)
    fields = {"schema_version", "campaign_purpose", "pilot_id", "assignment_seed",
              "max_games_attempted", "target_assignment_count", "estimator_eligible",
              "plan", "game_plan", "work_directory", "destination"}
    if (set(profile) != fields or profile["schema_version"] != PROFILE_VERSION
            or profile["campaign_purpose"] != "pilot"
            or profile["pilot_id"] != ARTIFACT_NAME
            or type(profile["target_assignment_count"]) is not int
            or profile["target_assignment_count"] != TARGET_ASSIGNMENTS["pilot"]
            or profile["estimator_eligible"] is not True):
        raise ValueError("invalid formal campaign profile")
    paths = []
    for key in ("plan", "game_plan", "work_directory", "destination"):
        path = Path(profile[key])
        if not path.is_absolute():
            raise ValueError(f"profile {key} must be absolute")
        paths.append(path.resolve())
    if any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
           for i, a in enumerate(paths) for b in paths[i + 1:]):
        raise ValueError("plan/work/destination paths must be disjoint")
    if Path(profile["destination"]).name != ARTIFACT_NAME:
        raise ValueError("formal destination must use the fixed pilot artifact name")
    return profile


def pilot_plan(profile):
    from werewolf.phase2_online_plan import Phase2OnlineTerminalPilotPlanV1
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


def clean_source():
    from werewolf.phase2_online_preflight import _git, freeze_online_source_provenance
    source = freeze_online_source_provenance(REPO)
    if source is None:
        raise ValueError("tracked source and staged index must be clean")
    # Untracked research is allowed; the new operator code/config must be committed.
    for path in (Path(__file__), PROFILE, PINS):
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


def canonical_plan(plan, head, bound):
    from scripts.collect_games import derive_seed_pool, plan_fields
    from werewolf.canonical_collection.attempt_ledger import construct_collection_plan, validate_collection_plan
    provenance = bound["canonical_game_plan"]["environment_provenance"]
    extra = {k: v for k, v in provenance.items() if k not in (
        "configured_call_limit", "gameplay_prompt_profile", "seed_pool_size", "target_success_count")}
    return validate_collection_plan(construct_collection_plan(
        ordered_seed_pool=derive_seed_pool(plan.pilot_id, plan.max_games_attempted),
        **plan_fields({"collection_id": plan.pilot_id, "target_games": plan.max_games_attempted,
                      "seed_pool_size": plan.max_games_attempted,
                      "call_limit": int(provenance["configured_call_limit"])}, head, extra)))


def protect_qualification(profile, bound):
    for key in ("plan", "game_plan", "work_directory", "destination"):
        output = Path(profile[key]).resolve()
        for evidence in (bound["paths"]["work_directory"], bound["paths"]["destination"]):
            protected = Path(evidence).resolve()
            if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
                raise ValueError("formal paths must not overlap qualification evidence")


def prepare(profile):
    from scripts.collect_games import publish_plan
    from werewolf.phase2_online_server import _publish_json
    source = clean_source()
    plan = pilot_plan(profile)
    for key in ("plan", "game_plan", "work_directory", "destination"):
        if os.path.lexists(profile[key]):
            raise FileExistsError(f"prepare refuses existing {key}; no overwrite or implicit resume")
    pins, bound, _ = qualification_inputs()
    protect_qualification(profile, bound)
    if Path(bound["paths"]["repo"]).resolve() != REPO.resolve():
        raise ValueError("prepare must run in the verified server checkout")
    game = canonical_plan(plan, source["commit"], bound)
    overlaps = seed_overlaps(game.ordered_seed_pool, pins, bound)
    if clean_source() != source:
        raise ValueError("source changed during prepare")
    # Both are exclusive publications. A partial failure is retained, never silently regenerated.
    _publish_json(Path(profile["plan"]), plan.to_record())
    publish_plan(Path(profile["game_plan"]), game)
    return {"status": "FROZEN", "source_commit": source["commit"],
            "pilot_plan_digest": plan.digest(), "game_plan_digest": game.plan_digest,
            "assignment_seed": plan.assignment_seed, "seed_pool_size": len(game.ordered_seed_pool),
            "seed_overlap_counts": overlaps}


def server_args(profile, command):
    from scripts.run_phase2_online_intervention_pilot import build_parser
    from werewolf.canonical_collection.attempt_ledger import collection_plan_from_record
    from werewolf.phase2_online_server import load_online_plan
    source = clean_source()
    plan = pilot_plan(profile)
    frozen = load_online_plan(Path(profile["plan"]), "pilot")
    game = collection_plan_from_record(read_json(profile["game_plan"]))
    if frozen != plan or game.source_revision != source["commit"]:
        raise ValueError("frozen plans differ from current profile/source")
    pins, bound, _ = qualification_inputs()
    protect_qualification(profile, bound)
    if game != canonical_plan(plan, source["commit"], bound):
        raise ValueError("frozen canonical plan differs from qualified runtime pins/production derivation")
    seed_overlaps(game.ordered_seed_pool, pins, bound)
    values = {**{k: bound["paths"][k] for k in ("runtime_config", "deployment_config", "publication",
                "evaluation_root", "mapper", "smoke_v3", "q_checkout", "q_fit")},
              **{k: bound[k] for k in ("mapper_manifest_digest", "reference_tables_digest",
                  "smoke_v3_manifest_digest", "q_python")},
              **{k: profile[k] for k in ("plan", "game_plan", "work_directory", "destination")},
              "repo": str(REPO), "campaign_purpose": "pilot", "source_commit": game.source_revision}
    argv = [part for k, v in values.items() for part in ("--" + k.replace("_", "-"), str(v))]
    if command == "preflight":
        argv.append("--preflight-only")
    if command == "resume":
        argv.append("--resume")
    return build_parser().parse_args(argv)


def status(profile, *, qualification=False):
    """Use the existing ledger validator without its write-on-missing constructor."""
    from werewolf.phase2_online_ledger import OnlinePilotAssignmentLedgerV1
    from werewolf.phase2_online_server import load_online_plan
    from werewolf.artifact_io import verify_artifact
    from werewolf.phase2_online_runner import ARTIFACT_VERSION
    artifact = None
    if qualification:
        _, bound, artifact = qualification_inputs()
        profile = {**bound["pilot_plan"], **bound["paths"], "estimator_eligible": False}
        plan_path = artifact.path / "pilot_plan.json"
        source = bound["source_commit"]
    else:
        plan_path = Path(profile["plan"])
        source = (read_json(profile["game_plan"])["source_revision"]
                  if Path(profile["game_plan"]).is_file() else None)
    work, destination = Path(profile["work_directory"]), Path(profile["destination"])
    result = {"status": "NOT_STARTED", "games_seen": 0, "assignments": 0,
              "PUSH": 0, "REDIRECT": 0, "executions": 0, "execution_failures": 0,
              "consequences": 0, "backend_calls": 0,
              "remaining_target": profile["target_assignment_count"],
              "estimator_eligible": profile["estimator_eligible"],
              "work_directory": str(work), "destination": str(destination),
              "source_commit": source, "ledger_terminal_digest": None}
    if not work.exists() and not destination.exists():
        return result
    bound = read_json(work / "run_inputs.json")
    plan = load_online_plan(plan_path, "qualification" if qualification else "pilot")
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
    for action in ("PUSH", "REDIRECT"):
        result[action] = sum("ASSIGNMENT" in s and
            s["ASSIGNMENT"]["assignment"]["assigned_action"] == action for s in stages)
    if destination.exists():
        artifact = artifact or verify_artifact(destination,
            expected_artifact_type="phase2_online_terminal_pilot", expected_schema_version=ARTIFACT_VERSION)
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
    args = parser.parse_args(argv)
    try:
        if args.campaign == "qualification":
            if args.command != "status":
                raise ValueError("qualification is completed and read-only; only status is allowed")
            output = status(None, qualification=True)
        else:
            if args.command != "status":
                frozen_environment()  # Before importing/starting any Q worker.
            profile = read_profile()
            if args.command == "status":
                output = status(profile)
            else:
                if args.command == "prepare":
                    output = prepare(profile)
                else:
                    from werewolf.phase2_online_server import execute_server_campaign
                    server = server_args(profile, args.command)
                    print(json.dumps({"source_commit": server.source_commit,
                        "pilot_plan_digest": digest(read_json(server.plan)),
                        "game_plan_digest": read_json(server.game_plan)["plan_digest"],
                        "destination_absent": not os.path.lexists(server.destination),
                        "reference_tables_digest": server.reference_tables_digest}, sort_keys=True), flush=True)
                    return execute_server_campaign(server)
        print(json.dumps(output, sort_keys=True), flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"Campaign failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
