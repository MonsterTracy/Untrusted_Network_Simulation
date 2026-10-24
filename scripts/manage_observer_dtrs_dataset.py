"""Publish, verify or reconstruct an explicitly offline Observer-DTRS dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from werewolf.observer_dtrs.dataset_offline import (
    audit_dataset_sources, publish_dataset, verify_dataset,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser("publish")
    publish.add_argument("--destination", type=Path, required=True)
    publish.add_argument("--dataset-id", required=True)
    publish.add_argument("--base-revision", required=True)
    publish.add_argument("--storage-profile", type=Path,
                         help="When supplied, destination must be inside configured artifact_root")
    verify = commands.add_parser("verify")
    replay = commands.add_parser("audit-sources")
    for command in (publish, replay):
        command.add_argument("--publication", type=Path, required=True)
        command.add_argument("--oof-root", type=Path, required=True)
        command.add_argument("--expected-oof-seal-digest", required=True)
    for command in (verify, replay):
        command.add_argument("--dataset", type=Path, required=True)
        command.add_argument("--expected-dataset-digest", required=True)
    args = parser.parse_args(argv)
    if args.command == "publish":
        if args.storage_profile:
            root = Path(json.loads(args.storage_profile.read_text())["artifact_root"]).resolve()
            if not args.destination.resolve().is_relative_to(root):
                parser.error("dataset destination is outside configured artifact_root")
        destination = args.destination.resolve()
        for source in (args.publication.resolve(), args.oof_root.resolve()):
            if destination.is_relative_to(source) or source.is_relative_to(destination):
                parser.error("dataset destination must be separate from upstream artifacts")
        artifact = publish_dataset(destination, dataset_id=args.dataset_id,
            publication_path=args.publication, evaluation_path=args.oof_root,
            expected_seal_digest=args.expected_oof_seal_digest, base_revision=args.base_revision)
        result = {"path": str(artifact.path), "manifest_digest": artifact.manifest_digest,
                  "coverage": artifact.manifest["coverage"]}
    elif args.command == "verify":
        result = verify_dataset(args.dataset, expected_digest=args.expected_dataset_digest)
    else:
        result = audit_dataset_sources(args.dataset, publication_path=args.publication,
            evaluation_path=args.oof_root, expected_seal_digest=args.expected_oof_seal_digest,
            expected_digest=args.expected_dataset_digest)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
