"""Controlled acceptance or explicitly committed formal nested validation."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from werewolf.observer_dtrs.d2_protocol import frozen_protocol, read_protocol
from werewolf.observer_dtrs.validation_data import FormalData, controlled_fixture
from werewolf.observer_dtrs.nested_validation import execute, verify_result
from werewolf.observer_dtrs.validation_artifacts import VERSION
from werewolf.artifact_io import verify_artifact, read_artifact_file


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('controlled','run','inspect'))
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--protocol-artifact',type=Path)
    parser.add_argument('--expected-protocol-manifest-digest')
    parser.add_argument('--split',type=Path)
    parser.add_argument('--expected-split-manifest-digest')
    parser.add_argument('--source-revision')
    parser.add_argument('--storage-profile',type=Path)
    parser.add_argument('--block',nargs=2,type=int,metavar=('TOM_FOLD','OUTER_FOLD'))
    args=parser.parse_args(argv)
    if args.command=='inspect':
        print(json.dumps(verify_result(args.destination),sort_keys=True))
        return 0
    protocol=frozen_protocol()
    if args.command=='controlled':
        data=controlled_fixture(protocol)
    else:
        required=(args.dataset,args.protocol_artifact,args.expected_protocol_manifest_digest,args.split,
                  args.expected_split_manifest_digest,args.source_revision,args.storage_profile)
        if any(v is None for v in required):
            parser.error('formal run requires dataset/protocol/split trust anchors, committed source revision and storage profile')
        read_protocol(args.protocol_artifact,expected_manifest_digest=args.expected_protocol_manifest_digest)
        root=Path(json.loads(args.storage_profile.read_text())['artifact_root']).resolve()
        destination=args.destination.resolve()
        protected=(args.dataset.resolve(),args.protocol_artifact.resolve(),args.split.resolve())
        if (not destination.is_relative_to(root) or any(destination.is_relative_to(p) or p.is_relative_to(destination) for p in protected)):
            parser.error('new run destination must be separate from sources and inside artifact_root')
        data=FormalData(args.dataset,split_path=args.split,split_digest=args.expected_split_manifest_digest,protocol=protocol)
    block=tuple(args.block) if args.block else None
    if block is not None and any(v not in range(5) for v in block):
        parser.error('block IDs must be 0..4')
    print(json.dumps(execute(args.destination,data=data,expected_revision=args.source_revision,block=block),sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
