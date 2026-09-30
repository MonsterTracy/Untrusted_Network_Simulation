"""Read a controlled pilot dataset JSON and print descriptive support only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from werewolf.phase2_pilot_dataset import (
    ONLINE_VERSION, analyze_phase2_consequence_support_record,
    analyze_phase2_online_support_record,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="serialized Phase2ConsequenceDatasetV1 JSON")
    args = parser.parse_args(argv)
    payload = json.loads(args.dataset.read_bytes())
    report = (analyze_phase2_online_support_record(payload)
              if payload.get("manifest", {}).get("schema_version") == ONLINE_VERSION
              else analyze_phase2_consequence_support_record(payload))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
