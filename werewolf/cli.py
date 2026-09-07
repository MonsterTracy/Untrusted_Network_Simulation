"""The five production operations; no alternate scientific lineages."""

import argparse
import json
from pathlib import Path

import yaml

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes


def build_parser():
    parser = argparse.ArgumentParser(prog="classic7-tom")
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--plan", type=Path, required=True)
    collect.add_argument("--runtime-config", type=Path, required=True)
    collect.add_argument("--call-limit", type=int, required=True)
    collect.add_argument("--destination", type=Path, required=True)
    publication = commands.add_parser("publish-development")
    publication.add_argument("--collection", type=Path, required=True)
    publication.add_argument("--runtime-config", type=Path, required=True)
    publication.add_argument("--destination", type=Path, required=True)
    experiment = commands.add_parser("prepare-experiment")
    experiment.add_argument("--publication", type=Path, required=True)
    experiment.add_argument("--protocol", type=Path, required=True)
    experiment.add_argument("--destination", type=Path, required=True)
    oof = commands.add_parser("run-development-oof")
    oof.add_argument("--experiment", type=Path, required=True)
    validate = commands.add_parser("validate-artifact")
    validate.add_argument("path", type=Path)
    validate.add_argument("--runtime-config", type=Path)
    return parser


def _runtime(path):
    from werewolf.runtime_config import normalize_runtime_config
    if path is None:
        raise ValueError("runtime configuration is required for deterministic collection replay")
    return normalize_runtime_config(yaml.safe_load(path.read_text(encoding="utf-8")))


def validate_artifact(path, runtime_config=None):
    from werewolf.development_publication import open_publication, open_role_sidecar, open_verified_collection
    from werewolf.canonical_collection.production_runtime import classic7_replay_executor
    from werewolf.tom.experiment import open_experiment, validate_experiment_preflight
    from werewolf.tom.reporting import validate_existing_evaluation
    from werewolf.tom.state import validate_model_state
    path = Path(path)
    if (path / "collection_plan.json").exists():
        return open_verified_collection(path, replay_executor=classic7_replay_executor(_runtime(runtime_config))).plan.plan_digest
    if (path / "experiment_manifest.json").exists():
        experiment = open_experiment(path)
        validate_experiment_preflight(experiment)
        if (experiment.runs_path / "checkpoint_set_manifest.json").exists():
            validate_existing_evaluation(experiment)
        return experiment.digest
    if (path / "phase_codebook.manifest.json").exists():
        from werewolf.tom.temporal import TemporalCodeProvider
        manifest = json.loads((path / "phase_codebook.manifest.json").read_bytes())
        return TemporalCodeProvider.explicit(path, manifest["parent_digest"]).artifact_digest
    manifest = json.loads((path / "manifest.json").read_bytes())
    kind = manifest.get("artifact_type")
    if kind == "development_publication":
        from werewolf.artifact_io import open_artifact_envelope
        from werewolf.development_publication import DEVELOPMENT_PUBLICATION_SCHEMA_VERSION
        open_artifact_envelope(path, expected_artifact_type=kind,
            expected_schema_version=DEVELOPMENT_PUBLICATION_SCHEMA_VERSION)
        publication = open_publication(path)
        open_role_sidecar(publication)
        return publication.manifest_digest
    if kind == "canonical_model_state":
        return validate_model_state(path).artifact.manifest_digest
    if kind == "canonical_game_bundle":
        collection = open_verified_collection(path.parent.parent,
            replay_executor=classic7_replay_executor(_runtime(runtime_config)))
        games = [game for game in collection.games if game.game_id == manifest["game_id"]]
        if len(games) != 1 or games[0].path != path:
            raise ValueError("bundle is not a ledger-bound collection member")
        return games[0].manifest_digest
    raise ValueError(f"unsupported standalone artifact type: {kind!r}; validate its owning collection/experiment")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "collect":
        from werewolf.backends import load_named_backends
        from werewolf.canonical_collection.attempt_ledger import collection_plan_from_record
        from werewolf.canonical_collection.collector import collect
        from werewolf.canonical_collection.production_runtime import Classic7RuntimeFactory
        config = _runtime(args.runtime_config)
        plan = collection_plan_from_record(json.loads(args.plan.read_bytes()))
        declared = dict(plan.environment_provenance)
        if (declared.get("runtime_config_sha256") != sha256_bytes(canonical_json_bytes(config))
            or declared.get("configured_call_limit") != str(args.call_limit)):
            raise ValueError("Collection Plan must bind normalized runtime config and call limit")
        result = collect(plan=plan, destination=args.destination, runtime_factory=Classic7RuntimeFactory(
            runtime_config=config, backends=load_named_backends(config, max_retries=0), configured_call_limit=args.call_limit))
        identity = plan.plan_digest
    elif args.command == "publish-development":
        from werewolf.development_publication import open_verified_collection, publish_development
        from werewolf.canonical_collection.production_runtime import classic7_replay_executor
        collection = open_verified_collection(args.collection, replay_executor=classic7_replay_executor(_runtime(args.runtime_config)))
        identity = publish_development(collection, args.destination, publication_id=args.destination.name).public.manifest_digest
    elif args.command == "prepare-experiment":
        from werewolf.development_publication import open_publication
        from werewolf.tom.experiment import ExperimentConfig, prepare_experiment
        config = ExperimentConfig(**json.loads(args.protocol.read_bytes()))
        identity = prepare_experiment(open_publication(args.publication), config, args.destination).digest
    elif args.command == "run-development-oof":
        from werewolf.tom.experiment import open_experiment
        from werewolf.tom.reporting import run_development_oof
        identity = run_development_oof(open_experiment(args.experiment))["record_digest"]
    else:
        identity = validate_artifact(args.path, args.runtime_config)
    print(identity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
