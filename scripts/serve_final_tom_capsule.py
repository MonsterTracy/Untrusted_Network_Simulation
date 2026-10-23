"""Owner-launched local IPC process. Not a network listener or game entry."""
import argparse
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from werewolf.observer_dtrs.final_tom_service import serve


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fd', required=True, type=int)
    args = p.parse_args()
    try:
        return serve(args.fd)
    except Exception:
        return 2  # no payload/exception leakage; process exit also stops inference


if __name__ == '__main__':
    os._exit(main())  # never wait indefinitely for a failed inference thread at shutdown
