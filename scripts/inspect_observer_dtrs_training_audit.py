"""Read-only D1c/outer-model/support audit; no decision or fitting entry point."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from werewolf.observer_dtrs.sealed_d2_offline import SourceAnchors, read_formal_d2
from werewolf.observer_dtrs.training_audit_offline import crosscheck_d1c, verify_outer_artifacts


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--expected-complete-digest', required=True)
    p.add_argument('--expected-source-revision', required=True)
    p.add_argument('--expected-dataset-digest', required=True)
    p.add_argument('--expected-split-digest', required=True)
    p.add_argument('--output', type=Path, help='New local audit report, never overwrite or write inside source artifacts')
    args = p.parse_args(argv)
    if args.output and any(args.output.resolve().is_relative_to(root.resolve()) for root in (args.run, args.dataset)):
        p.error('audit output must be outside both immutable input directories')
    view = read_formal_d2(args.run, anchors=SourceAnchors(args.expected_complete_digest,
        args.expected_source_revision, args.expected_dataset_digest, args.expected_split_digest))
    link = crosscheck_d1c(view, args.dataset, expected_dataset_digest=args.expected_dataset_digest)
    outer = verify_outer_artifacts(args.run, view, link)
    report = {'D1c': {k:v for k,v in link.identity.items() if k not in ('dataset_origins','game_records')},
        'outer_audit': outer.report, 'source_revision': args.expected_source_revision,
        'local_run_path': str(args.run.resolve()), 'local_dataset_path': str(args.dataset.resolve()),
        'server_access': False, 'formal_D3_protocol_frozen': False}
    text = json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(text)
    print(text, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
