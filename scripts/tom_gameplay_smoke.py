"""Run one paired production-config Wolf-NoToM / Wolf+ToM gameplay smoke."""

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from scripts import collect_games as operator
from scripts.qwen3_gameplay_predictor import Qwen3GameplayPredictorClient
from scripts.tom_gameplay_ablation import run_ablation
from werewolf import cli
from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.backends import load_named_backends
from werewolf.canonical_collection.attempt_ledger import (
    construct_attempt_claim,
    construct_collection_plan,
    initialize_attempt_ledger,
    publish_ledger_record,
    validate_attempt_ledger,
    validate_collection_plan,
)
from werewolf.canonical_collection.production_runtime import Classic7RuntimeFactory
from werewolf.canonical_collection.public_history import PublicPhase

DEFAULT_PAIR_ID = "tom-gameplay-smoke-v1"


def timestamp():
    return datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")


def request_has_tom_header(record):
    payload = record.private_payload.to_value()
    request = {"args": payload["args"], "kwargs": payload["kwargs"]}
    return "PRIVATE ToM INFORMATION" in json.dumps(
        request, ensure_ascii=False
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head")
    parser.add_argument(
        "--predictor-checkout", default=os.environ.get("PREDICTOR_CHECKOUT")
    )
    parser.add_argument("--fit-path", default=os.environ.get("QWEN3_FIT_PATH"))
    parser.add_argument("--pair-id", default=DEFAULT_PAIR_ID)
    args = parser.parse_args(argv)
    if not args.predictor_checkout:
        parser.error("--predictor-checkout is required (or PREDICTOR_CHECKOUT)")
    if not args.fit_path:
        parser.error("--fit-path is required (or QWEN3_FIT_PATH)")
    return args


def main(argv=None):
    args = parse_args(argv)
    checkout = Path(args.predictor_checkout).resolve()
    fit = Path(args.fit_path).resolve()
    pair_id = args.pair_id
    print("PREFLIGHT", flush=True)
    head = operator.clean_head()
    if args.expected_head is not None and head != args.expected_head:
        raise ValueError(
            f"clean gameplay HEAD mismatch: expected {args.expected_head}, found {head}"
        )
    provenance, base_url = operator.inspect_inputs()
    operator.live_preflight(base_url, provenance["served_model_name"])
    assert operator.clean_head() == head

    config = cli._runtime(operator.RUNTIME)
    assert sha256_bytes(canonical_json_bytes(config)) == (
        provenance["runtime_config_sha256"]
    )
    storage_root = cli._storage_root(operator.STORAGE)
    formal_campaign = operator.validate_campaign(
        operator.read_json(
            storage_root / "operator/collection_campaign.json"
        )
    )
    call_limit = formal_campaign["call_limit"]
    seed = operator.derive_seed_pool(pair_id, 1)[0]

    backends = load_named_backends(config, max_retries=0)
    factory = Classic7RuntimeFactory(
        runtime_config=config,
        backends=backends,
        configured_call_limit=call_limit,
    )
    print(
        f"PREFLIGHT PASS seed={seed} call_limit={call_limit}",
        flush=True,
    )

    reference_roles = None
    with tempfile.TemporaryDirectory(
        prefix="tom-gameplay-paired-smoke-", dir="/tmp"
    ) as scratch:
        for slug, arm in (
            ("notom", "Wolf-NoToM"),
            ("tom", "Wolf+ToM"),
        ):
            print(f"START {arm}", flush=True)

            campaign = operator.validate_campaign({
                "collection_id": f"{pair_id}-{slug}",
                "target_games": 1,
                "seed_pool_size": 1,
                "call_limit": call_limit,
            })
            smoke_provenance = {
                **provenance,
                "seed_rule_identity": (
                    f"paired-smoke-v1:derive_seed_pool:{pair_id}:ordinal0"
                ),
                "gameplay_smoke_arm": arm,
            }
            plan = validate_collection_plan(
                construct_collection_plan(
                    ordered_seed_pool=(seed,),
                    **operator.plan_fields(
                        campaign, head, smoke_provenance
                    ),
                )
            )

            collection_dir = Path(scratch) / slug
            collection_dir.mkdir()
            ledger_dir = initialize_attempt_ledger(
                collection_dir, plan
            )
            claim = construct_attempt_claim(
                plan,
                ordinal=0,
                attempt_id=f"smoke-{slug}-000",
                claim_timestamp_utc=timestamp(),
            )
            publish_ledger_record(ledger_dir, plan, claim)
            assert validate_attempt_ledger(
                ledger_dir, plan
            ).open_claim == claim

            runtime = factory(plan=plan, claim=claim)
            assert runtime.env.random_seed == seed
            roles = tuple(runtime.roles)
            if reference_roles is None:
                reference_roles = roles
            assert roles == reference_roles, (
                "paired initial role assignments differ"
            )

            treatment_audit = []
            predictor_requests = None

            if arm == "Wolf-NoToM":
                result = run_ablation(
                    runtime.env,
                    runtime.agents,
                    runtime.roles,
                    recorder=runtime.recorder,
                    call_audit=runtime.call_audit,
                    arm=arm,
                    predictor=None,
                    treatment_audit=treatment_audit,
                )
            else:
                # Predictor is first created after NoToM has fully finished.
                with Qwen3GameplayPredictorClient(
                    checkout=checkout,
                    fit_path=fit,
                    python_executable=sys.executable,
                ) as predictor:
                    print(
                        "SEALED TOM PREDICTOR READY",
                        flush=True,
                    )
                    result = run_ablation(
                        runtime.env,
                        runtime.agents,
                        runtime.roles,
                        recorder=runtime.recorder,
                        call_audit=runtime.call_audit,
                        arm=arm,
                        predictor=predictor,
                        treatment_audit=treatment_audit,
                    )
                    predictor_requests = predictor._next_request
                    assert predictor_requests == len(treatment_audit)

            evidence = runtime.recorder.complete_evidence()
            assert evidence.call_budget_summary.used_calls == len(
                runtime.call_audit.records
            )

            if arm == "Wolf-NoToM":
                assert treatment_audit == []
                assert all(
                    not request_has_tom_header(call)
                    for call in runtime.call_audit.records
                ), "NoToM backend request contains ToM information"
            else:
                assert treatment_audit, (
                    "+ToM never reached a wolf speech boundary"
                )
                prefixes = {
                    prefix.boundary_id: prefix
                    for prefix in evidence.authoritative_pre_prefixes
                }
                treated_boundaries = {
                    entry["boundary_id"]
                    for entry in treatment_audit
                }

                for entry in treatment_audit:
                    prefix = prefixes[entry["boundary_id"]]
                    speaker = prefix.current_speaker
                    assert entry["game_id"] == evidence.game_id
                    assert entry["prefix_digest"] == prefix.prefix_digest
                    assert prefix.public_temporal_state.phase in {
                        PublicPhase.DISCUSSION,
                        PublicPhase.PK_DISCUSSION,
                    }
                    assert roles[
                        int(speaker.removeprefix("player")) - 1
                    ] == "Werewolf"
                    assert any(
                        action.boundary_id == entry["boundary_id"]
                        and action.actor_id == speaker
                        and action.action_type == "public_speech"
                        for action in evidence.submitted_gameplay_actions
                    )

                seen_tom_boundaries = set()
                for call in runtime.call_audit.records:
                    if not request_has_tom_header(call):
                        continue
                    assert call.boundary_id in treated_boundaries, (
                        "ToM header reached an untreated backend boundary"
                    )
                    prefix = prefixes[call.boundary_id]
                    assert call.observer_id == prefix.current_speaker, (
                        "ToM header reached a different speaker"
                    )
                    assert roles[
                        int(call.observer_id.removeprefix("player")) - 1
                    ] == "Werewolf"
                    assert prefix.public_temporal_state.phase in {
                        PublicPhase.DISCUSSION,
                        PublicPhase.PK_DISCUSSION,
                    }
                    seen_tom_boundaries.add(call.boundary_id)

                assert seen_tom_boundaries == treated_boundaries, (
                    "a treatment prediction did not reach a backend request"
                )

            print(f"END {arm}", flush=True)
            print(json.dumps({
                "arm": arm,
                "game_id": evidence.game_id,
                "seed": seed,
                "roles_equal": roles == reference_roles,
                "result": result,
                "backend_calls": (
                    evidence.call_budget_summary.used_calls
                ),
                "speech_boundaries": len(
                    evidence.authoritative_pre_prefixes
                ),
                "treatment_count": len(treatment_audit),
                "predictor_requests": predictor_requests,
            }, sort_keys=True), flush=True)

    print("PAIRED_GAMEPLAY_SMOKE_PASS", flush=True)


if __name__ == "__main__":
    main()
