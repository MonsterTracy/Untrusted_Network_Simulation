"""Read-only minimal D2 source/coverage audit; never chooses losses or regions."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from werewolf.observer_dtrs.sealed_d2_offline import SourceAnchors, read_formal_d2
from werewolf.observer_dtrs.statistical_analysis import MODELS, WEIGHTINGS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--expected-complete-digest', required=True)
    parser.add_argument('--expected-source-revision', required=True)
    parser.add_argument('--expected-dataset-digest', required=True)
    parser.add_argument('--expected-split-digest', required=True)
    parser.add_argument('--output', type=Path, help='Optional new local audit JSON, never overwrite')
    args = parser.parse_args(argv)
    view = read_formal_d2(args.run, anchors=SourceAnchors(args.expected_complete_digest,
        args.expected_source_revision, args.expected_dataset_digest, args.expected_split_digest))
    identity = view.identity
    report = {k: v for k, v in identity.items()
              if k not in ('game_sizes', 'outer_references', 'block_digests')}
    report.update({'local_run_path': str(args.run.resolve()), 'games': len(identity['game_sizes']),
        'prediction_rows_by_weighting_model': {
            w: {m: len(view.batch(m, w).rows) for m in MODELS} for w in WEIGHTINGS},
        'd3_regions_or_risks_computed': False, 'loss_protocol_selected_or_frozen': False,
        'server_access': False})
    text = json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(text)
    print(text, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
