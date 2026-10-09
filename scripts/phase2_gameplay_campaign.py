"""Prepare/preflight/run/resume/validate independent whole-game Router studies."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from werewolf.artifact_io import canonical_json_bytes, publish_artifact, read_artifact_file, verify_artifact
from werewolf.artifact_io.canonical import _load_canonical_json, _reject_duplicate_keys, _reject_json_constant
from werewolf.canonical_collection.attempt_ledger import collection_plan_from_record, construct_collection_plan
from werewolf.phase2_gameplay import (
    NAMES, VERSION, GameplayLedger, digest, freeze_source, make_plan, safe_path,
    validate_plan, validate_publication, validate_profile, load_campaign_profile, bind_campaign_profile,
)
from werewolf.phase2_gameplay_server import execute_server, verify_router


ROOT = Path(__file__).resolve().parents[1]
PROFILES = {p: ROOT / f"configs/phase2/router-gameplay-{p}-v1.json" for p in NAMES}


def read_json(path):
    return json.loads(safe_path(path).read_bytes(), object_pairs_hook=_reject_duplicate_keys,
                      parse_constant=_reject_json_constant)


def check_paths(*paths):
    paths = [safe_path(p) for p in paths]
    if any(a == b or a.is_relative_to(b) or b.is_relative_to(a) for i, a in enumerate(paths) for b in paths[i + 1:]):
        raise ValueError("plan, work, publication and input paths must be disjoint")
    return paths


def prepare(args):
    if args.profile is not None and safe_path(args.profile) != safe_path(PROFILES[args.purpose]):
        raise ValueError("prepare requires the canonical source-frozen profile; no alternate campaign paths")
    profile = validate_profile(load_campaign_profile(ROOT, args.purpose), ready=True)
    n = profile["randomized_games"]
    outputs = check_paths(args.plan, args.work_directory, args.destination)
    if outputs != [safe_path(profile["paths"][key]) for key in ("plan", "work", "destination")]:
        raise ValueError("campaign cannot change its frozen plan/work/destination paths")
    if any(os.path.lexists(path) for path in outputs):
        raise FileExistsError("prepare requires new plan/work/publication paths; failed campaigns cannot restart")
    exclusion_pins = read_json(args.exclusions)
    if exclusion_pins != profile["excluded_game_plans"]:
        raise ValueError("exclusions differ from the source-frozen historical pointers")
    source = freeze_source(ROOT)
    if args.source_commit != source["commit"]:
        raise ValueError("prepare source differs from the explicit clean commit")
    from scripts.collect_games import derive_seed_pool, plan_fields
    from scripts.phase2_online_campaign import qualification_inputs, seed_overlaps
    pins, bound, qualified = qualification_inputs()
    excluded = []
    for pin in exclusion_pins:
        if set(pin) != {"path", "plan_digest"}:
            raise ValueError("exclusions require exact canonical plan pointers/digests")
        game = collection_plan_from_record(read_json(pin["path"]))
        if game.plan_digest != pin["plan_digest"]:
            raise ValueError("historical excluded plan mismatch")
        excluded.append(game)
    required = {"paper-phase2-online-terminal-pilot-v1", "paper-phase2-online-probe-pilot-v1",
                *[f"paper-phase2-online-probe-qualification-v{i}" for i in (1, 2, 3)]}
    if not required <= {g.collection_id for g in excluded}:
        raise ValueError("exclude Terminal formal and Probe V1/V2/V3/formal full planned pools")
    seeds = derive_seed_pool(profile["campaign_id"], n)
    seed_overlaps(seeds, pins, bound)
    # The already frozen ToM ablation prederived 100 candidate seeds are excluded.
    tom_seeds = derive_seed_pool("paper-tom-gameplay-ablation-v1", 100)
    if set(seeds) & set(tom_seeds):
        raise ValueError("ToM gameplay seed overlap; no reroll")
    development = collection_plan_from_record(read_json(pins["development_game_plan"]))
    previous = collection_plan_from_record(bound["canonical_game_plan"])
    excluded.extend((development, previous))
    admission = profile["qualification"]
    if args.purpose == "formal":
        if not isinstance(admission, dict) or set(admission) != {"path", "manifest_digest"}:
            raise ValueError("formal requires an explicitly reviewed gameplay qualification")
        from werewolf.canonical_collection.production_runtime import classic7_replay_executor
        import yaml
        artifact = verify_artifact(safe_path(admission["path"]), expected_artifact_type="phase2_router_gameplay", expected_schema_version=VERSION)
        qplan = validate_plan(_load_canonical_json(read_artifact_file(artifact, "run_plan.json")))
        replay = classic7_replay_executor(yaml.safe_load(Path(qplan["runtime_inputs"]["paths"]["runtime_config"]).read_bytes()))
        artifact = validate_publication(admission["path"], admission["manifest_digest"], replay, verify_router)
        if artifact.manifest["purpose"] != "qualification" or artifact.manifest["qualification_gate_passed"] is not True:
            raise ValueError("qualification lacks reviewed mechanism gate")
        for name, sha in qplan["source"]["source_sha256"].items():
            if name.endswith(".py") or name == "docs/research/phase2-router-v1-contract.md":
                if source["source_sha256"].get(name) != sha:
                    raise ValueError("runtime source differs from qualification")
        excluded.append(collection_plan_from_record(qplan["game_plan"]))
    provenance = dict(previous.environment_provenance)
    extra = {k: v for k, v in provenance.items() if k not in (
        "configured_call_limit", "gameplay_prompt_profile", "seed_pool_size", "target_success_count")}
    game = construct_collection_plan(ordered_seed_pool=seeds,
        **plan_fields({"collection_id": profile["campaign_id"], "target_games": n, "seed_pool_size": n,
                       "call_limit": int(provenance["configured_call_limit"])}, source["commit"], extra))
    runtime = {k: bound[k] for k in ("paths", "q_python", "q_fit_digest", "q_seal_digest",
        "mapper_manifest_digest", "reference_tables_digest", "smoke_v3_manifest_digest",
        "runtime_config_sha256", "deployment_config_sha256")}
    runtime["additional_seed_exclusions"] = {"tom_gameplay_seed_rule": "paper-tom-gameplay-ablation-v1:100",
                                            "seeds": list(tom_seeds)}
    runtime["qualified_runtime_manifest_digest"] = qualified.manifest_digest
    plan = make_plan(purpose=args.purpose, game_plan=game, source=source, runtime_inputs=runtime,
        exclusions=[g.to_record() for g in excluded], qualification=admission,
        qualification_gate=profile["qualification_gate"], campaign_profile=profile)
    protected = [safe_path(bound["paths"][k]) for k in (
        "publication", "evaluation_root", "mapper", "smoke_v3", "q_checkout", "q_fit", "work_directory", "destination")]
    protected.extend(safe_path(p["path"]) for p in exclusion_pins)
    for name in ("online-terminal-pilot-formal-v1.json", "online-probe-qualification-v1.json",
                 "online-probe-qualification-v2.json", "online-probe-qualification-v3.json", "online-probe-formal-v1.json"):
        historical = read_json(ROOT / "configs/phase2" / name)
        protected.extend(safe_path(historical[k]) for k in ("plan", "game_plan", "work_directory", "destination"))
    if any(a == b or a.is_relative_to(b) or b.is_relative_to(a) for a in outputs for b in protected):
        raise ValueError("gameplay outputs must not overlap historical evidence or model inputs")
    if freeze_source(ROOT) != source:
        raise ValueError("source changed during prepare")
    artifact = publish_artifact(safe_path(args.plan), manifest_fields={"artifact_type": "phase2_router_gameplay_plan",
        "schema_version": VERSION, "purpose": args.purpose, "source": source, "plan_digest": plan["plan_digest"]},
        files={"run_plan.json": canonical_json_bytes(plan), "game_plan.json": canonical_json_bytes(game.to_record()),
               "operator_paths.json": canonical_json_bytes({"work": str(safe_path(args.work_directory)),
                                                         "destination": str(safe_path(args.destination))})})
    return {"plan_manifest_digest": artifact.manifest_digest, "plan_digest": plan["plan_digest"],
            "allocation_counts": {arm: sum(a["assigned_policy"] == arm for a in plan["allocations"])
                                  for arm in ("IMMEDIATE_REDIRECT", "PROBE_THEN_REDIRECT")}}


def open_plan(path, expected_digest):
    artifact = verify_artifact(safe_path(path), expected_artifact_type="phase2_router_gameplay_plan", expected_schema_version=VERSION)
    if artifact.manifest_digest != expected_digest:
        raise ValueError("prepared plan manifest mismatch")
    plan = validate_plan(_load_canonical_json(read_artifact_file(artifact, "run_plan.json")))
    if (artifact.manifest["source"] != plan["source"] or artifact.manifest["purpose"] != plan["purpose"] or
            artifact.manifest["plan_digest"] != plan["plan_digest"] or
            _load_canonical_json(read_artifact_file(artifact, "game_plan.json")) != plan["game_plan"]):
        raise ValueError("prepared canonical plan binding mismatch")
    paths = _load_canonical_json(read_artifact_file(artifact, "operator_paths.json"))
    if set(paths) != {"work", "destination"}:
        raise ValueError("prepared operator path schema mismatch")
    bind_campaign_profile(plan, ROOT, work=paths["work"], destination=paths["destination"], prepared_path=path)
    check_paths(path, paths["work"], paths["destination"])
    return plan, paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "runtime-check", "run", "resume", "status", "validate"))
    parser.add_argument("--purpose", choices=tuple(NAMES), default="qualification")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-manifest-digest")
    parser.add_argument("--source-commit")
    parser.add_argument("--exclusions", type=Path)
    parser.add_argument("--work-directory", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--publication-manifest-digest")
    parser.add_argument("--resume-check", action="store_true", help="read-only resume preflight")
    parser.add_argument("--allow-gpu", action="store_true", help="explicit opt-in for runtime-check/run/resume")
    args = parser.parse_args(argv)
    if args.resume_check and args.command != "preflight":
        parser.error("--resume-check is only allowed with preflight")
    if args.command in ("runtime-check", "run", "resume") and not args.allow_gpu:
        parser.error(f"{args.command} requires --allow-gpu")
    if args.allow_gpu and args.command not in ("runtime-check", "run", "resume"):
        parser.error("--allow-gpu is only allowed with runtime-check/run/resume")
    if args.command == "prepare":
        if any(v is None for v in (args.source_commit, args.exclusions, args.work_directory, args.destination)):
            parser.error("prepare requires source, exclusions, work-directory and destination")
        result = prepare(args)
    else:
        if args.plan_manifest_digest is None:
            parser.error("expected prepared plan manifest digest is required")
        plan, paths = open_plan(args.plan, args.plan_manifest_digest)
        if plan["purpose"] != args.purpose:
            raise ValueError("qualification/formal cannot be interchanged")
        if args.command == "status":
            result = GameplayLedger(paths["work"], plan).status()
        elif args.command == "validate":
            if args.publication_manifest_digest is None:
                parser.error("validate requires publication-manifest-digest")
            from werewolf.canonical_collection.production_runtime import classic7_replay_executor
            import yaml
            replay = classic7_replay_executor(yaml.safe_load(Path(plan["runtime_inputs"]["paths"]["runtime_config"]).read_bytes()))
            artifact = validate_publication(paths["destination"], args.publication_manifest_digest, replay, verify_router)
            result = {"valid": True, "manifest_digest": artifact.manifest_digest}
        else:
            from scripts.phase2_online_campaign import frozen_environment
            frozen_environment()
            result = execute_server(plan, repo=ROOT, work=paths["work"], destination=paths["destination"],
                                    preflight_only=args.command == "preflight",
                                    runtime_check_only=args.command == "runtime-check", allow_gpu=args.allow_gpu,
                                    resume=args.command == "resume" or args.resume_check)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
