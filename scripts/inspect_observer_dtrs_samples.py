"""Inspect offline samples and optional sealed OOF features; no inference or writes."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from werewolf.observer_dtrs.labels import LABEL_VERSION
from werewolf.observer_dtrs.offline import SAMPLE_VERSION, build_observer_samples
from werewolf.observer_dtrs.features import FEATURE_VERSION, observer_features
from werewolf.observer_dtrs.oof import load_qwen3_oof


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--game-id", action="append", help="Explicit whole-game partition; repeatable")
    parser.add_argument("--oof-root", type=Path, help="Historical sealed Qwen3 evaluation root")
    parser.add_argument("--expected-oof-seal-digest", help="Explicit trusted evaluation seal SHA-256")
    args = parser.parse_args(argv)
    if (args.oof_root is None) != (args.expected_oof_seal_digest is None):
        parser.error("--oof-root and --expected-oof-seal-digest must be supplied together")
    samples = build_observer_samples(args.publication, game_ids=args.game_id)
    report = {
        "schema_version": SAMPLE_VERSION, "label_version": LABEL_VERSION,
        "scope": "restricted_offline_supervision",
        "sample_count": len(samples),
        "games_with_samples": len({s.context.game_id for s in samples}),
        "wolf_pre_count": len({(s.context.game_id, s.context.boundary_id) for s in samples}),
        "phase_sample_counts": dict(Counter(s.context.phase for s in samples)),
        "positive_count": sum(s.supervision.event_label_y for s in samples),
    }
    if args.oof_root is not None:
        predictions = load_qwen3_oof(args.publication, args.oof_root,
            (s.context for s in samples), expected_seal_digest=args.expected_oof_seal_digest)
        features = [observer_features(p.context, p.probability) for p in predictions]
        report["oof"] = {
            "feature_version": FEATURE_VERSION, "feature_count": len(features),
            "seal_digest": args.expected_oof_seal_digest,
            "fold_sample_counts": dict(Counter(p.source.fold_index for p in predictions)),
            "nW_sample_counts": dict(Counter(f.nW for f in features)),
        }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
