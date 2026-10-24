"""D3 controlled acceptance, seal inspection, or separately frozen formal execution."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from werewolf.observer_dtrs.d3_protocol import load_protocol
from werewolf.observer_dtrs.d3_runner import execute_controlled, execute_formal, inspect_result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('controlled','inspect','run'))
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--protocol',type=Path)
    p.add_argument('--expected-protocol-digest')
    p.add_argument('--source-revision')
    p.add_argument('--expected-manifest-digest',help='Optional trusted D3 complete SHA-256 for inspect only')
    p.add_argument('--run',type=Path)
    p.add_argument('--dataset',type=Path)
    p.add_argument('--storage-profile',type=Path)
    args=p.parse_args(argv)
    if args.command!='inspect' and args.expected_manifest_digest is not None:
        p.error('expected-manifest-digest is for inspect only')
    if args.command=='inspect':
        result=inspect_result(args.destination,expected_manifest_digest=args.expected_manifest_digest)
    elif args.command=='controlled':
        if any(v is not None for v in (args.run,args.dataset,args.storage_profile,args.source_revision,args.expected_protocol_digest)):
            p.error('controlled command accepts no formal inputs, source claims or storage profile')
        result=execute_controlled(args.destination,protocol=load_protocol() if args.protocol is None else load_protocol(args.protocol))
    else:
        if any(v is None for v in (args.run,args.dataset,args.storage_profile,args.source_revision,args.expected_protocol_digest)):
            p.error('formal execution requires run/dataset/storage-profile/source-revision/expected-protocol-digest')
        protocol=load_protocol() if args.protocol is None else load_protocol(args.protocol)
        profile=json.loads(args.storage_profile.read_bytes())
        result=execute_formal(args.destination,run_path=args.run,dataset_path=args.dataset,protocol=protocol,
            expected_protocol_digest=args.expected_protocol_digest,expected_revision=args.source_revision,artifact_root=profile['artifact_root'])
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
