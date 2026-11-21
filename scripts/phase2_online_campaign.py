"""Thin operator CLI for terminal and Probe campaigns on the same server runtime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


CUBLAS = ":4096:8"


def frozen_environment():
    value = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if value is not None and value != CUBLAS:
        raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS


def clean_source(*, probe=False):
    from werewolf.interventions.online.plan import PROFILE, PROBE_PROFILES, REPO
    from werewolf.interventions.online.preflight import PINS, _git, freeze_online_source_provenance
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


def prepare(profile, *, source_commit=None):
    from werewolf.interventions.online.plan import (
        PROBE_PROFILE_VERSION, REPO, canonical_plan, checked_seed_overlaps, pilot_plan,
    )
    from werewolf.interventions.online.preflight import probe_inputs, protect_qualification, qualification_inputs

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
    from werewolf.interventions.online.server import _publish_json
    # Both are exclusive publications. A partial failure is retained, never silently regenerated.
    _publish_json(Path(profile["plan"]), plan.to_record())
    publish_plan(Path(profile["game_plan"]), game)
    return {"status": "FROZEN", "source_commit": source["commit"],
            "pilot_plan_digest": plan.digest(), "game_plan_digest": game.plan_digest,
            "assignment_seed": plan.assignment_seed, "seed_pool_size": len(game.ordered_seed_pool),
            "seed_overlap_counts": overlaps}


def server_args(profile, command):
    from werewolf.interventions.online.plan import (
        PROBE_PROFILE_VERSION, REPO, canonical_plan, checked_seed_overlaps,
        load_online_plan, pilot_plan, read_json,
    )

    plan = pilot_plan(profile)
    from scripts.run_phase2_online_intervention_pilot import build_parser
    from werewolf.canonical_collection.attempt_ledger import collection_plan_from_record
    from werewolf.interventions.online.preflight import probe_inputs, protect_qualification, qualification_inputs
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
    from werewolf.interventions.online.ledger import OnlinePilotAssignmentLedgerV1
    from werewolf.artifact_io import verify_artifact
    from werewolf.interventions.online.plan import PROBE_PROFILE_VERSION, digest, load_online_plan, read_json
    from werewolf.interventions.online.preflight import probe_qualification_gate, qualification_inputs

    probe = profile is not None and profile["schema_version"] == PROBE_PROFILE_VERSION
    if not probe:
        from werewolf.interventions.online.runner import ARTIFACT_VERSION
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
                      and not {"EXECUTION", "CONSEQUENCE"}.issubset(s)]
        result.update(unfinished_assigned_games=unfinished,
            interrupted_games=sum("INTERRUPTED" in s for s in stages),
            post_endpoint_tail_interruptions=sum("POST_ENDPOINT_TAIL_INTERRUPTED" in s for s in stages),
            whole_game_results=sum("GAME_RESULT" in s for s in stages),
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
            from werewolf.interventions.online.plan import digest, read_json, read_profile

            profile = read_profile(policy="probe", qualification=args.campaign == "qualification") if args.policy == "probe" else read_profile()
            if args.command == "status":
                output = status(profile)
            else:
                if args.command == "prepare":
                    output = prepare(profile, source_commit=args.source_commit) if args.policy == "probe" else prepare(profile)
                else:
                    server = server_args(profile, args.command)
                    from werewolf.interventions.online.server import execute_server_campaign
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
