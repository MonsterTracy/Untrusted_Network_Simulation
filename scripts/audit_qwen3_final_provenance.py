"""Collect read-only final-ToM evidence; no predictor/model execution entry."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from werewolf.artifact_io import sha256_bytes
from werewolf.observer_dtrs.final_tom_audit import (
    ROOT, audit_final_tom, checkout_identity, digest, write_evidence,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', required=True, type=Path, help='Original absolute final fit directory')
    parser.add_argument('--predictor-checkout', required=True, type=Path, help='Existing clean pinned f199 checkout')
    parser.add_argument('--mode', choices=('metadata', 'chain-files'), default='metadata')
    parser.add_argument('--output', required=True, type=Path, help='New evidence file outside audited trees; parent must exist')
    args = parser.parse_args()
    report = audit_final_tom(args.fit, args.predictor_checkout, chain_files=args.mode == 'chain-files')
    # Audit tool identity is deliberately separate from the frozen fit source.
    try:
        tool = checkout_identity(ROOT)
    except (OSError, ValueError, subprocess.SubprocessError):
        tool = {'status': 'UNAVAILABLE'}
    paths = ('scripts/audit_qwen3_final_provenance.py', 'werewolf/observer_dtrs/final_tom_audit.py',
             'scripts/qwen3_gameplay_predictor.py')
    report['audit_tool'] = {**tool, 'file_sha256': {p: sha256_bytes((ROOT / p).read_bytes()) for p in paths}}
    report.pop('evidence_digest')
    report['evidence_digest'] = digest(report)
    try:
        raw_sha = write_evidence(args.output, report, fit_path=args.fit, predictor_checkout=args.predictor_checkout)
    except (OSError, ValueError):
        print(json.dumps({'status': 'OUTPUT_REFUSED', 'reason': 'new_external_output_required'}))
        return 3
    print(json.dumps({'status': report['status'], 'evidence_digest': report['evidence_digest'],
                      'evidence_file_raw_sha256': raw_sha}))
    return 0 if report['status'] == 'PASS_WITH_EXPLICIT_SCOPE_LIMITS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
