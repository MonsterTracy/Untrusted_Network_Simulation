"""Export only audited final-ToM inference bytes; never run a model forward."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from werewolf.observer_dtrs.final_tom_capsule_build import compile_capsule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', required=True, type=Path)
    parser.add_argument('--audit', required=True, type=Path, help='Previously accepted chain-files evidence')
    parser.add_argument('--audit-sha256', required=True, help='Externally saved SHA-256 of exact evidence file bytes')
    parser.add_argument('--output', required=True, type=Path, help='New external capsule directory; existing parent')
    args = parser.parse_args()
    try:
        artifact = compile_capsule(args.fit, args.output, audit_path=args.audit,
                                   expected_audit_raw_sha256=args.audit_sha256)
    except (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError):
        print(json.dumps({'status': 'FAILED', 'reason': 'invalid_source_or_output_no_fallback'}))
        return 2
    print(json.dumps({'status': 'PUBLISHED_NOT_QUALIFIED', 'manifest_digest': artifact.manifest_digest,
                      'scope': artifact.manifest['scope']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
