"""Scientific lifecycle commands and an isolated synthetic engineering check."""

import argparse
import json
import os
from pathlib import Path

import yaml

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes


def build_parser():
    parser = argparse.ArgumentParser(prog="uns")
    parser.add_argument("--storage-profile", type=Path, default=os.environ.get("UNS_STORAGE_PROFILE"),
        help="explicit deployment JSON path (or UNS_STORAGE_PROFILE)")
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--plan", type=Path, required=True)
    collect.add_argument("--runtime-config", type=Path, required=True)
    collect.add_argument("--call-limit", type=int, required=True)
    collect.add_argument("--destination", type=Path, required=True)
    collect.add_argument("--resume", action="store_true", help="explicitly continue the exact collection ledger")
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
    oof.add_argument("--resume", action="store_true", help="explicitly continue the exact existing run under its recovery policy")
    validate = commands.add_parser("validate-artifact")
    validate.add_argument("path", type=Path)
    validate.add_argument("--runtime-config", type=Path)
    validate.add_argument("--experiment", type=Path, help="owning sealed experiment for final publication validation")
    capacity = commands.add_parser("capacity-check", help="synthetic engineering check, not a scientific run")
    capacity.add_argument("--pre-count-per-game", type=int, required=True)
    capacity.add_argument("--max-seq-len", type=int, required=True)
    capacity.add_argument("--game-batch-size", type=int, required=True)
    capacity.add_argument("--device", choices=("cpu", "cuda"), required=True)
    final = commands.add_parser("prepare-final-experiment")
    final.add_argument("--publication", type=Path, required=True)
    final.add_argument("--protocol", type=Path, required=True)
    final.add_argument("--destination", type=Path, required=True)
    fit = commands.add_parser("run-final-fit")
    fit.add_argument("--experiment", type=Path, required=True)
    fit.add_argument("--resume", action="store_true")
    seal = commands.add_parser("seal-final-models")
    seal.add_argument("--experiment", type=Path, required=True)
    final_publication = commands.add_parser("publish-final-evaluation")
    final_publication.add_argument("--experiment", type=Path, required=True)
    final_publication.add_argument("--collection", type=Path, required=True)
    final_publication.add_argument("--runtime-config", type=Path, required=True)
    final_publication.add_argument("--destination", type=Path, required=True)
    final_evaluation = commands.add_parser("run-final-evaluation")
    final_evaluation.add_argument("--experiment", type=Path, required=True)
    final_evaluation.add_argument("--publication", type=Path, required=True)
    final_evaluation.add_argument("--resume", action="store_true")
    return parser


def _storage_root(profile):
    if profile is None:
        raise ValueError("storage profile is required; no default artifact location")
    value = json.loads(Path(profile).read_bytes())
    if not isinstance(value, dict) or set(value) != {"artifact_root"} or not isinstance(value["artifact_root"], str):
        raise ValueError("storage profile requires exactly artifact_root")
    root = Path(value["artifact_root"])
    if not root.is_absolute() or ".." in root.parts or not root.is_dir():
        raise ValueError("artifact root must be an existing absolute directory")
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("artifact root cannot traverse symbolic links")
    return root.resolve(strict=True)


def _artifact_path(root, value, kind=None):
    path = Path(value)
    if ".." in path.parts or str(path) == ".":
        raise ValueError("invalid artifact path")
    if not path.is_absolute():
        if len(path.parts) == 1 and kind is not None:
            path = Path(kind) / path
            if kind == "experiments":
                # Keep immutable inputs and mutable runs in one experiment namespace.
                path = path / "experiments" / "experiment"
        path = root / path
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("artifact path cannot traverse symbolic links")
    path = path.resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("artifact path must be inside artifact root")
    return path


def _require_resume(path, resume):
    started = path.exists() and any(path.iterdir())
    if started and not resume:
        raise ValueError("existing run requires explicit --resume")
    if resume and not started:
        raise ValueError("--resume requires existing run records")


def _experiment_path(root, value):
    path = _artifact_path(root, value, "experiments")
    parts = path.relative_to(root).parts
    if len(parts) != 4 or parts[0] != "experiments" or parts[2:] != ("experiments", "experiment"):
        raise ValueError("experiment must use experiments/ID/experiments/experiment layout")
    _artifact_path(root, path.parent.parent / "runs")
    return path


def _runtime(path):
    from werewolf.runtime_config import normalize_runtime_config
    if path is None:
        raise ValueError("runtime configuration is required for deterministic collection replay")
    return normalize_runtime_config(yaml.safe_load(path.read_text(encoding="utf-8")))


def validate_artifact(path, runtime_config=None, final_experiment_path=None):
    from werewolf.development_publication import open_publication, open_role_sidecar, open_verified_collection
    from werewolf.canonical_collection.production_runtime import classic7_replay_executor
    from werewolf.tom.experiment import open_experiment, validate_experiment_preflight
    from werewolf.tom.reporting import validate_existing_evaluation
    from werewolf.tom.state import validate_model_state
    path = Path(path)
    if final_experiment_path is not None:
        from werewolf.tom.final_experiment import open_final_experiment
        from werewolf.tom.final_training import verify_final_seal
        from werewolf.final_publication import open_final_publication
        from werewolf.tom.population import select_final_primary_population
        experiment = open_final_experiment(final_experiment_path)
        seal = verify_final_seal(experiment)
        publication = open_final_publication(path, model_seal_digest=seal["record_digest"])
        select_final_primary_population(publication)
        return publication.manifest_digest
    if (path / "collection_plan.json").exists() and (path / "manifest.json").exists():
        raise ValueError("final publication validation requires --experiment with a model seal")
    if (path / "final_experiment_manifest.json").exists():
        from werewolf.tom.final_experiment import open_final_experiment, final_training_inputs
        from werewolf.tom.final_training import verify_final_seal
        from werewolf.tom.final_evaluation import validate_final_evaluation
        experiment = open_final_experiment(path)
        final_training_inputs(experiment)
        if (experiment.runs_path / "final_model_seal.json").exists():
            verify_final_seal(experiment)
        if (experiment.runs_path / "final_evaluation_consumption.json").exists():
            validate_final_evaluation(experiment)
        return experiment.digest
    if (path / "collection_plan.json").exists() and not (path / "manifest.json").exists():
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
    if args.command == "capacity-check":
        from werewolf.tom.capacity_check import NOTICE, check_training_capacity
        print(NOTICE, flush=True)
        result = check_training_capacity(pre_count_per_game=args.pre_count_per_game,
            max_seq_len=args.max_seq_len, game_batch_size=args.game_batch_size, device=args.device)
        print(json.dumps(result, sort_keys=True))
        return 0
    root = _storage_root(args.storage_profile)
    if args.command in {"prepare-final-experiment", "run-final-fit", "seal-final-models", "publish-final-evaluation", "run-final-evaluation"}:
        return _final_operation(args, root)
    if args.command == "collect":
        args.destination = _artifact_path(root, args.destination, "canonical")
        _require_resume(args.destination, args.resume)
    elif args.command == "publish-development":
        args.collection = _artifact_path(root, args.collection, "canonical")
        args.destination = _artifact_path(root, args.destination, "publications")
    elif args.command == "prepare-experiment":
        args.publication = _artifact_path(root, args.publication, "publications")
        args.destination = _experiment_path(root, args.destination)
    elif args.command == "run-development-oof":
        args.experiment = _experiment_path(root, args.experiment)
    else:
        args.path = _artifact_path(root, args.path)
        if args.experiment is not None:
            args.experiment = _experiment_path(root, args.experiment)
        if (args.path / "final_experiment_manifest.json").is_file():
            _experiment_path(root, args.path)
            manifest = json.loads((args.path / "final_experiment_manifest.json").read_bytes())
            _artifact_path(root, manifest["publication_path"])
            consumption = args.path.parent.parent / "runs" / manifest["manifest_digest"] / "final_evaluation_consumption.json"
            if consumption.exists():
                _artifact_path(root, json.loads(consumption.read_bytes())["publication_path"])
        if (args.path / "experiment_manifest.json").is_file():
            _experiment_path(root, args.path)
            manifest = json.loads((args.path / "experiment_manifest.json").read_bytes())
            _artifact_path(root, manifest["publication_path"])
            _artifact_path(root, args.path.parent.parent / "runs")
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
        experiment = open_experiment(args.experiment)
        _artifact_path(root, experiment.manifest["publication_path"])
        runs = _artifact_path(root, experiment.runs_path)
        _require_resume(runs, args.resume)
        identity = run_development_oof(experiment)["record_digest"]
    else:
        identity = (validate_artifact(args.path, args.runtime_config, args.experiment) if args.experiment is not None
                    else validate_artifact(args.path, args.runtime_config))
    print(identity)
    return 0


def _final_operation(args, root):
    from werewolf.tom.final_experiment import open_final_experiment, prepare_final_experiment
    from werewolf.tom.final_training import train_final_condition, seal_final_models, verify_final_seal
    from werewolf.tom.experiment import ExperimentConfig, TEMPORAL_CONDITIONS
    if args.command == "prepare-final-experiment":
        from werewolf.development_publication import open_publication
        config = ExperimentConfig(**json.loads(args.protocol.read_bytes()))
        publication = open_publication(_artifact_path(root, args.publication, "publications"))
        identity = prepare_final_experiment(publication, config, _experiment_path(root, args.destination)).digest
    else:
        experiment = open_final_experiment(_experiment_path(root, args.experiment))
        _artifact_path(root, experiment.manifest["publication_path"])
        _artifact_path(root, experiment.runs_path)
        if args.command == "run-final-fit":
            _require_resume(experiment.runs_path, args.resume)
            for condition in TEMPORAL_CONDITIONS:
                train_final_condition(experiment, condition, resume=args.resume)
            identity = experiment.digest
        elif args.command == "seal-final-models":
            identity = seal_final_models(experiment)["record_digest"]
        elif args.command == "publish-final-evaluation":
            verify_final_seal(experiment)  # before runtime/replay or collection access
            from werewolf.development_publication import open_verified_collection
            from werewolf.canonical_collection.production_runtime import classic7_replay_executor
            from werewolf.tom.final_evaluation import publish_final_evaluation
            collection = open_verified_collection(_artifact_path(root, args.collection, "canonical"),
                replay_executor=classic7_replay_executor(_runtime(args.runtime_config)))
            destination = _artifact_path(root, args.destination, "publications")
            identity = publish_final_evaluation(experiment, collection, destination, publication_id=destination.name).manifest_digest
        else:
            from werewolf.tom.final_evaluation import run_final_evaluation
            identity = run_final_evaluation(experiment, _artifact_path(root, args.publication, "publications"), resume=args.resume)["record_digest"]
    print(identity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
