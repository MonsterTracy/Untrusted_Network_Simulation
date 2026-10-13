"""Manual server snapshot: stdlib only, metadata checks, no fsync or inference.

Prints JSON to stdout. Does not import the project, read gameplay payloads,
rehash model weights, validate/recover/publish artifacts, or connect anywhere.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess

def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _json(path):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("snapshot rejects symlink paths")
    def unique(pairs):
        value = {}
        for k, v in pairs:
            if k in value:
                raise ValueError("duplicate JSON field")
            value[k] = v
        return value
    def reject_constant(value):
        raise ValueError("nonfinite JSON")
    return json.loads(path.read_bytes(), object_pairs_hook=unique, parse_constant=reject_constant)


def _seeds(campaign):
    return [int.from_bytes(hashlib.sha256(
        f"classic7-canonical-collection-v1:{campaign['collection_id']}:{i}".encode()).digest()[:8], "big")
        & ((1 << 63) - 1) for i in range(campaign["seed_pool_size"])]


def _ledger_summary(root, plan):
    claims, terminals = {}, {}
    for path in sorted((root / "attempt_ledger").iterdir()):
        if path.name == ".staging":
            continue
        record = _json(path)
        digest = record["record_digest"]
        if (hashlib.sha256(_canonical({k: v for k, v in record.items() if k != "record_digest"})).hexdigest() != digest
                or path.read_bytes() != _canonical(record)):
            raise ValueError("ledger record canonical bytes/digest mismatch")
        ordinal, seed, kind = record["ordinal"], record["seed"], record["record_type"]
        if (type(ordinal) is not int or type(seed) is not int or not 0 <= ordinal < len(plan["ordered_seed_pool"])
                or seed != plan["ordered_seed_pool"][ordinal] or kind not in {"claim", "terminal"}
                or path.name != f"{ordinal}-{seed}.{kind}.json"
                or record["plan_digest"] != plan["plan_digest"] or record["collection_id"] != plan["collection_id"]
                or record["runtime_provenance_digest"] != plan["runtime_provenance_digest"]):
            raise ValueError("ledger metadata identity mismatch")
        target = claims if kind == "claim" else terminals
        if ordinal in target:
            raise ValueError("duplicate ledger ordinal")
        target[ordinal] = record
    if sorted(claims) != list(range(len(claims))) or set(terminals) - set(claims):
        raise ValueError("ledger claim/terminal coverage mismatch")
    for ordinal, record in terminals.items():
        if (record["attempt_id"] != claims[ordinal]["attempt_id"]
                or record["outcome"] not in {"canonical_success", "canonical_failure", "interrupted_failure"}):
            raise ValueError("ledger terminal metadata mismatch")
    opened = sorted(set(claims) - set(terminals))
    if len(opened) > 1 or opened and opened != [len(claims) - 1]:
        raise ValueError("ledger has non-tail open claims")
    outcomes = Counter(r["outcome"] for r in terminals.values())
    parse = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
    durations = [(parse(terminals[i]["terminal_timestamp_utc"]) - parse(claims[i]["claim_timestamp_utc"])).total_seconds() for i in sorted(terminals)]
    gaps = [(parse(claims[i]["claim_timestamp_utc"]) - parse(terminals[i-1]["terminal_timestamp_utc"])).total_seconds() for i in range(1, len(claims)) if i-1 in terminals]
    result = {"claims": len(claims), "terminals": len(terminals), "outcomes": dict(outcomes), "open_claims": len(opened),
              "target_reached_in_metadata": outcomes["canonical_success"] == plan["target_canonical_success_count"],
              "attempt_mean_seconds": statistics.mean(durations) if durations else None,
              "gap_mean_seconds": statistics.mean(gaps) if gaps else None,
              "last_100_gap_mean_seconds": statistics.mean(gaps[-100:]) if gaps else None}
    if claims and terminals:
        result.update(start_utc=claims[0]["claim_timestamp_utc"], end_utc=terminals[max(terminals)]["terminal_timestamp_utc"])
        result["wall_hours"] = (parse(result["end_utc"]) - parse(result["start_utc"])).total_seconds()/3600
    return result


def _campaign(value):
    fields = {"collection_id", "target_games", "seed_pool_size", "call_limit"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("campaign requires exactly four collection parameters")
    identity = value["collection_id"]
    if (not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", identity)
            or ".." in identity):
        raise ValueError("unsafe Collection ID")
    if any(type(value[k]) is not int or value[k] <= 0 for k in fields - {"collection_id"}):
        raise ValueError("campaign budgets must be positive integers")
    if value["seed_pool_size"] < value["target_games"]:
        raise ValueError("seed pool cannot be smaller than the success target")
    return dict(value)


def _pool(values):
    if not isinstance(values, list) or any(type(v) is not int or v < 0 for v in values):
        raise ValueError("invalid seed-pool metadata")
    if len(set(values)) != len(values):
        raise ValueError("duplicate seed-pool metadata")
    return values


def _seed_inventory(history_roots, campaigns, excluded_seeds=()):
    paths = set()
    for root in history_roots:
        root = Path(root)
        if root.is_symlink() or not root.is_dir():
            raise ValueError("historical root must be an existing directory, not a symlink")
        paths.update(root.rglob("collection_plan.json"))
    existing, slots, plans = set(), 0, 0
    for path in sorted(paths):
        record = _json(path)
        values = _pool(record["ordered_seed_pool"])
        existing.update(values); slots += len(values); plans += 1
    pools = {}
    for value in campaigns:
        campaign = _campaign(value)
        identity = campaign["collection_id"]
        if identity in pools:
            raise ValueError("duplicate candidate Collection ID")
        pools[identity] = _seeds(campaign)
    pairwise = [{"left": left, "right": right, "overlap": len(set(pools[left]) & set(pools[right]))}
                for i, left in enumerate(sorted(pools)) for right in sorted(pools)[i+1:]]
    return {"historical_roots": [str(p) for p in history_roots], "metadata_only": True,
            "historical_plan_count": plans, "historical_seed_slots": slots, "historical_unique_seeds": len(existing),
            "pairwise_candidate_overlap": pairwise,
            "pools": {name: {"size": len(values), "unique": len(set(values)),
                "sha256": hashlib.sha256(_canonical(values)).hexdigest(), "historical_overlap": len(set(values) & existing),
                "excluded_overlap": len(set(values) & set(excluded_seeds))} for name, values in pools.items()}}


def _command(argv):
    # Explicit local binaries only; no shell, SSH, HTTP, activation or GPU init.
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    try:
        p = subprocess.run(argv, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        return {"exit_code": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()[:1000]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": type(error).__name__}


def _versions(environment):
    wanted = {"openai", "numpy", "gymnasium", "pyyaml", "pydantic", "vllm", "torch", "transformers", "flashinfer-python", "flashinfer-jit-cache"}
    result = {}
    for path in sorted(environment.glob("lib/python*/site-packages/*.dist-info/METADATA")):
        name = version = None
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("Name: "): name = line[6:].strip().lower().replace('_', '-')
                if line.startswith("Version: "): version = line[9:].strip()
                if name and version: break
        if name in wanted:
            if name in result: raise ValueError("duplicate installed distribution metadata")
            result[name] = version
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--collection", type=Path, help="optional reference collection directory")
    parser.add_argument("--plan-digest", help="trusted reference Plan SHA-256; required with --collection")
    parser.add_argument("--campaign", type=Path, action="append", default=[], help="candidate four-field JSON; repeatable")
    parser.add_argument("--history-root", type=Path, action="append", default=[], help="explicit historical Plan search root; repeatable")
    parser.add_argument("--excluded-seeds", type=Path, help="explicit JSON array of forbidden/calibration seeds")
    parser.add_argument("--artifact-root", type=Path, action="append", default=[], help="additional candidate output root; repeatable")
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--serve-config", type=Path)
    parser.add_argument("--model-revision-file", type=Path)
    parser.add_argument("--environment", type=Path, action="append", default=[])
    parser.add_argument("--service-port", type=int)
    args = parser.parse_args(argv)
    if (args.collection is None) != (args.plan_digest is None):
        parser.error("provide both --collection and --plan-digest")
    if args.plan_digest is not None and not re.fullmatch(r"[0-9a-f]{64}", args.plan_digest):
        parser.error("trusted Plan digest must be lowercase SHA-256")
    if args.campaign and not args.history_root:
        parser.error("candidate seed checks require at least one explicit --history-root")
    if args.service_port is not None and not 1 <= args.service_port <= 65535:
        parser.error("service port must be in 1..65535")
    report = {"snapshot_utc": datetime.now(timezone.utc).isoformat(), "metadata_only": True,
              "deep_bundle_replay_performed": False, "model_weights_rehashed": False,
              "git": {name: _command(['git', '-c', 'core.fsmonitor=false', '-c', 'core.untrackedCache=false', '-C', str(args.repo), *command]) for name, command in
                      [('head', ['rev-parse', 'HEAD']), ('branch', ['branch', '--show-current']), ('status', ['status', '--porcelain', '--untracked-files=all'])]},
              "gpu": _command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu', '--format=csv']),
              "gpu_processes": _command(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv']),
              "memory": _command(['free', '-h']), "disk": _command(['df', '-h', str(args.data_root)]),
              "inodes": _command(['df', '-i', str(args.data_root)]), "errors": []}
    if args.service_port is not None:
        report['service_listener'] = _command(['ss', '-ltnp', f'( sport = :{args.service_port} )'])
    if args.collection is not None:
        try:
            root = args.collection
            plan = _json(root / 'collection_plan.json')
            if (root / 'collection_plan.json').read_bytes() != _canonical(plan):
                raise ValueError('reference Plan bytes are not canonical')
            digest = hashlib.sha256(_canonical({k: v for k, v in plan.items() if k != 'plan_digest'})).hexdigest()
            if digest != plan['plan_digest'] or digest != args.plan_digest:
                raise ValueError('reference Plan differs from trusted digest')
            keys = ('hf_revision', 'model_manifest_sha256', 'runtime_config_sha256', 'serve_config_sha256',
                    'client_python', 'client_openai', 'serving_python', 'serving_vllm', 'serving_torch', 'serving_torch_cuda')
            report['reference_plan'] = {k: plan[k] for k in ('collection_id', 'source_revision', 'plan_digest', 'runtime_provenance_digest', 'target_canonical_success_count', 'call_budget_identity', 'runtime_identity', 'model_identity', 'parser_identity', 'prompt_identity', 'retry_policy_identity', 'bundle_schema_version')}
            report['reference_plan']['seed_pool_size'] = len(plan['ordered_seed_pool'])
            report['reference_plan']['recorded_provenance'] = {k: plan['environment_provenance'].get(k) for k in keys}
            report['reference_ledger_metadata'] = _ledger_summary(root, plan)
            failures = Counter()
            for path in (root/'attempts').glob('*/failure_evidence.json'):
                record = _json(path); failures.update([(record['stage'], record['error_category'])])
            report['failure_types'] = [{'stage': stage, 'category': category, 'count': count} for (stage, category), count in sorted(failures.items())]
            calls = [_json(p)['used_calls'] for p in (root/'games').glob('*/audit/call_budget_summary.json')]
            report['successful_call_summaries'] = {'count': len(calls), 'total': sum(calls), 'mean': statistics.mean(calls) if calls else None, 'max': max(calls) if calls else None}
        except (OSError, ValueError, KeyError, TypeError) as error:
            report['errors'].append({'section': 'collection_metadata', 'type': type(error).__name__})
    try:
        campaigns = [_campaign(_json(path)) for path in args.campaign]
        excluded = _pool(_json(args.excluded_seeds)) if args.excluded_seeds else []
        report['seed_inventory'] = _seed_inventory(args.history_root, campaigns, excluded)
        roots = [args.data_root, *args.artifact_root]
        report['candidate_path_exists'] = {c['collection_id']: {str(root): (root/'canonical'/c['collection_id']).exists() for root in roots} for c in campaigns}
    except (OSError, ValueError, KeyError, TypeError) as error:
        report['errors'].append({'section': 'seed_inventory', 'type': type(error).__name__})
    for name, path in [('runtime_config', args.runtime_config), ('serve_config', args.serve_config)]:
        if path is None: continue
        try: report[name+'_raw_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error: report['errors'].append({'section': name, 'type': type(error).__name__})
    if args.model_revision_file is not None:
        try: report['current_hf_revision'] = args.model_revision_file.read_text().strip()
        except OSError as error: report['errors'].append({'section': 'model_revision', 'type': type(error).__name__})
    report['environments'] = {}
    for environment in args.environment:
        try:
            report['environments'][str(environment)] = {'python': _command([str(environment/'bin/python'), '-I', '-B', '-S', '-c', 'import sys; print(sys.version)']), 'distribution_metadata': _versions(environment)}
        except (OSError, ValueError) as error:
            report['errors'].append({'section': 'environment_metadata', 'type': type(error).__name__})
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 2 if report['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
