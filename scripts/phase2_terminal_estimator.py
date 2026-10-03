"""Independent sealed-input Terminal Estimator V1 analysis; never runs gameplay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from werewolf.phase2_terminal_estimator_io import analyze


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal", type=Path, required=True, help="frozen Formal Pilot-T artifact")
    parser.add_argument("--itt", type=Path, required=True, help="frozen 120-row ITT artifact")
    parser.add_argument("--destination", type=Path, required=True, help="new independent estimator artifact directory")
    args = parser.parse_args(argv)
    try:
        summary = analyze(args.formal, args.itt, args.destination)
        print(json.dumps(summary, sort_keys=True, allow_nan=False))
        return 0 if summary["status"] == "SANITY_PASSED_FOR_REVIEW" else 2
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"Terminal estimator failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
