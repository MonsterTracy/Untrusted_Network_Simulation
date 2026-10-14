"""Publish/verify the frozen D2 protocol and label-independent splits; no fitting."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from werewolf.observer_dtrs.d2_protocol import frozen_protocol, publish_protocol, read_protocol
from werewolf.observer_dtrs.splits import publish_splits, read_splits, source_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("publish", "verify"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--protocol-destination", type=Path, required=True)
    parser.add_argument("--split-destination", type=Path, required=True)
    parser.add_argument("--expected-protocol-manifest-digest")
    parser.add_argument("--expected-split-manifest-digest")
    parser.add_argument("--storage-profile", type=Path)
    args = parser.parse_args(argv)
    protocol = frozen_protocol()
    if args.command == "publish":
        destinations = (args.protocol_destination.resolve(), args.split_destination.resolve())
        sources = (args.dataset.resolve(),)
        if args.storage_profile:
            root = Path(json.loads(args.storage_profile.read_text())["artifact_root"]).resolve()
            if any(not d.is_relative_to(root) for d in destinations):
                parser.error("D2 destinations must be inside configured artifact_root")
        if (any(d.is_relative_to(s) or s.is_relative_to(d) for d in destinations for s in sources)
                or destinations[0].is_relative_to(destinations[1]) or destinations[1].is_relative_to(destinations[0])):
            parser.error("D2 destinations must be separate from each other and source data")
        # Validate the source anchor before any publication mutation.
        source_manifest(args.dataset, protocol)
        pa = publish_protocol(args.protocol_destination, protocol)
        sa = publish_splits(args.split_destination, args.dataset, protocol)
        pd, sd = pa.manifest_digest, sa.manifest_digest
    else:
        pd, sd = args.expected_protocol_manifest_digest, args.expected_split_manifest_digest
        if not pd or not sd:
            parser.error("verification requires both manifest trust anchors")
    read_protocol(args.protocol_destination, expected_manifest_digest=pd)
    split = read_splits(args.split_destination, dataset_manifest=source_manifest(args.dataset, protocol),
                        protocol=protocol, expected_manifest_digest=sd)
    print(json.dumps({"protocol_path": str(args.protocol_destination), "protocol_digest": protocol.digest,
        "protocol_manifest_digest": pd, "split_path": str(args.split_destination),
        "split_manifest_digest": sd, "outer_assignment_digest": split["outer_assignment_digest"],
        "coverage": split["coverage"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
