"""Explicit CPU QUAL preparation/verification; no training, collection or approval."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    snap=sub.add_parser('snapshot');snap.add_argument('--output',type=Path,required=True)
    check=sub.add_parser('verify-source')
    check.add_argument('--snapshot',type=Path,required=True);check.add_argument('--snapshot-digest',required=True)
    run=sub.add_parser('cpu')
    for flag in ('capsule','snapshot','pres','reference-checkout','audit','output'):
        run.add_argument('--'+flag,type=Path,required=True)
    for flag in ('capsule-digest','snapshot-digest','pres-sha256','audit-sha256'):
        run.add_argument('--'+flag,required=True)
    run.add_argument('--cpu-threads',type=int,default=1)
    args=p.parse_args(argv)
    try:
        from werewolf.observer_dtrs.qual_sources import publish_snapshot,verify_snapshot
        if args.command == 'snapshot':
            a=publish_snapshot(args.output)
            result={'status':'QUAL_PREPARATION_ONLY','source_snapshot_manifest':a.manifest_digest,
                    'inventory_digest':a.manifest['inventory_digest']}
        elif args.command == 'verify-source':
            r=verify_snapshot(args.snapshot,args.snapshot_digest)
            result={'status':'QUAL_PREPARATION_ONLY','source_bytes':'PASS',
                    'base_head':r['base_head'],'inventory_digest':r['inventory_digest']}
        else:
            if not 1 <= args.cpu_threads <= 64:
                raise ValueError('bounded CPU threads required')
            # This command is an independent offline process, never a game callback.
            import torch
            torch.set_num_threads(args.cpu_threads);torch.set_num_interop_threads(1)
            from werewolf.observer_dtrs.final_tom_qualification import qualify_cpu
            a=qualify_cpu(capsule_path=args.capsule,capsule_digest=args.capsule_digest,
                source_path=args.snapshot,source_digest=args.snapshot_digest,
                pre_path=args.pres,pre_sha256=args.pres_sha256,
                audit_path=args.audit,audit_raw_sha256=args.audit_sha256,
                reference_checkout=args.reference_checkout,destination=args.output)
            result={'status':'REAL_MODEL_LOCAL_VALIDATED','manifest_digest':a.manifest_digest,
                    'authorization':'NOT_FORMAL_DEPLOYMENT_NOT_CAL_TEST'}
    except Exception:
        print(json.dumps({'status':'QUAL_BLOCKED','reason':'INPUT_SOURCE_NUMERICAL_OR_LIFECYCLE_CHECK_FAILED',
                          'automatic_repair':False}))
        return 2
    print(json.dumps(result,sort_keys=True));return 0


if __name__ == '__main__':
    raise SystemExit(main())
