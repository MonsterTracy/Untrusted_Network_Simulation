"""Execute one frozen Online Pilot-T qualification or formal campaign.

Help uses only the standard library. Install this checkout with pip install -e
'.[mapper,tom]' before execution; no sys.path mutation is used by this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-purpose", choices=("qualification", "pilot"), required=True,
                        help="fixed 10 or 120 assignments; never chains campaigns")
    parser.add_argument("--plan", type=Path, required=True, help="frozen Pilot-T V1 plan JSON")
    parser.add_argument("--game-plan", type=Path, required=True,
                        help="frozen canonical CollectionPlan JSON: ordered seeds and runtime provenance")
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--deployment-config", type=Path, required=True)
    parser.add_argument("--publication", type=Path, required=True,
                        help="frozen 1500-game development publication for reference tables")
    parser.add_argument("--evaluation-root", type=Path, required=True,
                        help="sealed OOF evaluation used by the existing offline builder")
    parser.add_argument("--mapper", type=Path, required=True)
    parser.add_argument("--mapper-manifest-digest", required=True)
    parser.add_argument("--reference-tables-digest", required=True,
                        help="independently preregistered reference table content SHA256")
    parser.add_argument("--smoke-v3", type=Path, required=True)
    parser.add_argument("--smoke-v3-manifest-digest", required=True)
    parser.add_argument("--q-checkout", type=Path, required=True,
                        help="clean pinned final-Q worker checkout")
    parser.add_argument("--q-fit", type=Path, required=True)
    parser.add_argument("--q-python", default=sys.executable, help="worker Python executable")
    parser.add_argument("--source-commit", required=True, help="preregistered Pilot-T source HEAD")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--work-directory", type=Path, required=True,
                        help="separate durable ledger and restricted canonical evidence directory")
    parser.add_argument("--destination", type=Path, required=True,
                        help="absent fixed qualification/pilot artifact destination")
    parser.add_argument("--resume", action="store_true", help="reuse the exact existing campaign inputs/ledger")
    parser.add_argument("--preflight-only", action="store_true",
                        help="assemble runtime and Q handshake using a temporary ledger; no game or publication")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from werewolf.phase2_online_server import execute_server_campaign
    try:
        return execute_server_campaign(args)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"Pilot-T failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


def __getattr__(name):
    """Preserve library imports while production implementation lives in werewolf."""
    if name.startswith("__"):
        raise AttributeError(name)
    from werewolf import phase2_online_runner
    return getattr(phase2_online_runner, name)


if __name__ == "__main__":
    raise SystemExit(main())
