"""Run one paired production-config Wolf-NoToM / Wolf+ToM gameplay smoke."""

import argparse
import json
import os
import sys
import tempfile
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

from scripts import collect_games as operator
from scripts.analyze_tom_gameplay_ablation import _coverage_counts
from scripts.qwen3_gameplay_predictor import Qwen3GameplayPredictorClient
from scripts.run_werewolf_baseline import _read_plan
from scripts.tom_gameplay_ablation import Epoch35GameplayPredictor, run_ablation
from scripts.tom_gameplay_collection import BASELINE_PLAN_SHA256, NOTOM, TOM, _validate_treatment
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
from werewolf.game_runtime import count_game_votes, summarize_game_votes

DEFAULT_PAIR_ID = "tom-gameplay-smoke-v1"


def timestamp():
    return datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head")
    parser.add_argument("--predictor-checkout")
    parser.add_argument("--fit-path")
    parser.add_argument("--checkpoint", type=Path, help="Epoch 35 Checkpoint（ToM 使用 CPU）")
    parser.add_argument("--temporal", type=Path, help="Epoch 35 绑定的 Temporal")
    parser.add_argument("--baseline-plan", type=Path, help="用于排除正式 210 场景的冻结 Plan")
    parser.add_argument("--pair-id", default=DEFAULT_PAIR_ID)
    args = parser.parse_args(argv)
    epoch35_inputs = (args.checkpoint, args.temporal, args.baseline_plan)
    if any(epoch35_inputs):
        if not all(epoch35_inputs) or args.predictor_checkout is not None or args.fit_path is not None:
            parser.error("use --checkpoint, --temporal and --baseline-plan together, without historical predictor flags")
        return args
    if args.predictor_checkout is None:
        args.predictor_checkout = os.environ.get("PREDICTOR_CHECKOUT")
    if args.fit_path is None:
        args.fit_path = os.environ.get("QWEN3_FIT_PATH")
    if not args.predictor_checkout:
        parser.error("--predictor-checkout is required (or PREDICTOR_CHECKOUT)")
    if not args.fit_path:
        parser.error("--fit-path is required (or QWEN3_FIT_PATH)")
    return args


def _epoch35_metrics(evidence, arm, treatments, attempts, *, request_count, predictor):
    """复用正式分析的请求匹配，核对预测、载荷、回执和已提交发言。"""
    prefixes = evidence.authoritative_pre_prefixes
    roles = dict(evidence.private_replay_evidence.role_assignment)
    coverage = _coverage_counts(
        prefixes, roles,
        {"treatments": treatments, "treatment_count": len(treatments),
         "predictor_requests": request_count, "prediction_attempts": attempts},
        evidence.backend_call_evidence, [action.to_record() for action in evidence.submitted_gameplay_actions],
        game_id=evidence.game_id, arm=arm, reported_treatments=len(treatments), predictor=predictor,
    )
    expected = coverage["E"] if arm == TOM else 0
    integration = coverage["tom_integration"]
    if (any(integration[key] != expected for key in ("A", "P", "S"))
            or coverage["U"] != expected or coverage["response_and_submitted_action_decisions"] != expected
            or any(not item["expected_by_assignment"] for item in coverage["anomalies"])):
        raise ValueError(f"{arm} Smoke lacks consistent prediction, injection or response evidence: {coverage}")
    boundary_ids = {entry["boundary_id"] for entry in treatments}
    return {
        "coverage": coverage,
        "treatments_by_phase": {
            phase.value: sum(pre.boundary_id in boundary_ids and pre.public_temporal_state.phase is phase
                             for pre in prefixes)
            for phase in (PublicPhase.DISCUSSION, PublicPhase.PK_DISCUSSION)
        },
        "votes": summarize_game_votes([count_game_votes(evidence.public_event_stream, roles)]),
    }


def main(argv=None):
    args = parse_args(argv)
    epoch35 = args.checkpoint is not None
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
    epoch35_predictor = None
    if epoch35:
        baseline = _read_plan(cli._artifact_path(storage_root, args.baseline_plan))
        if baseline["plan_sha256"] != BASELINE_PLAN_SHA256:
            raise ValueError("Smoke requires the frozen Baseline Plan")
        if any(scenario["seed"] == seed for scenario in baseline["scenarios"]):
            raise ValueError("Smoke seed overlaps the formal Baseline Plan")
        epoch35_predictor = Epoch35GameplayPredictor(
            cli._artifact_path(storage_root, args.checkpoint),
            cli._artifact_path(storage_root, args.temporal), "cpu",
        )

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
        prefix="tom-gameplay-paired-smoke-", dir=Path("/tmp").resolve()
    ) as scratch:
        for slug, arm in (
            ("notom", NOTOM),
            ("tom", TOM),
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

            runtime = factory(plan=plan, claim=claim, record_requests=epoch35)
            if runtime.env.random_seed != seed:
                raise ValueError("Smoke runtime seed differs from the pair seed")
            roles = tuple(runtime.roles)
            if reference_roles is None:
                reference_roles = roles
            elif roles != reference_roles:
                raise ValueError("paired initial role assignments differ")

            treatment_audit = []
            prediction_attempts = [] if epoch35 else None
            predictor_requests = None

            if arm == NOTOM:
                predictor_context = nullcontext()
            elif epoch35:
                predictor_context = nullcontext(epoch35_predictor)
            else:
                predictor_context = Qwen3GameplayPredictorClient(
                    checkout=Path(args.predictor_checkout).resolve(),
                    fit_path=Path(args.fit_path).resolve(), python_executable=sys.executable,
                )
            with predictor_context as predictor:
                if predictor is not None:
                    ready = "EPOCH35 TOM PREDICTOR READY" if epoch35 else "SEALED TOM PREDICTOR READY"
                    print(ready, flush=True)
                result = run_ablation(
                    runtime.env,
                    runtime.agents,
                    runtime.roles,
                    recorder=runtime.recorder,
                    call_audit=runtime.call_audit,
                    arm=arm,
                    predictor=predictor,
                    treatment_audit=treatment_audit,
                    prediction_attempts=prediction_attempts,
                )
                if predictor is not None:
                    predictor_requests = len(prediction_attempts) if epoch35 else predictor._next_request

            evidence = runtime.recorder.complete_evidence()
            assert evidence.call_budget_summary.used_calls == len(
                runtime.call_audit.records
            )

            if arm == TOM and not treatment_audit:
                raise ValueError("+ToM never reached a wolf speech boundary")
            summary = {
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
            }
            if epoch35:
                winner = evidence.private_replay_evidence.replay_inputs.to_value().get("winner")
                if winner not in {"Werewolf", "Villager"} or result != f"{winner} win":
                    raise ValueError("Smoke result differs from the native terminal winner")
                predictor_identity = epoch35_predictor.identity if arm == TOM else None
                summary.update(_epoch35_metrics(
                    evidence, arm, treatment_audit, prediction_attempts,
                    request_count=predictor_requests, predictor=predictor_identity,
                ))
                summary.update(winner=winner, source_revision=head, baseline_plan_sha256=baseline["plan_sha256"],
                               public_event_digest=evidence.public_event_stream.digest,
                               predictor=predictor_identity, prediction_attempts=prediction_attempts,
                               treatments=treatment_audit)
            else:
                _validate_treatment(evidence, arm, treatment_audit, predictor_requests)
            print(f"END {arm}", flush=True)
            print(json.dumps(summary, sort_keys=True), flush=True)

    print("PAIRED_GAMEPLAY_SMOKE_PASS", flush=True)


if __name__ == "__main__":
    main()
