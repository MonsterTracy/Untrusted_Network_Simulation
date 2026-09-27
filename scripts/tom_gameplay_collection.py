"""Collect the frozen, paired Wolf-NoToM / Wolf+ToM gameplay experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from scripts import collect_games as operator
from scripts.qwen3_gameplay_predictor import (
    FIT_DIGEST,
    SEAL_DIGEST,
    SOURCE_REVISION,
    Qwen3GameplayPredictorClient,
)
from scripts.tom_gameplay_ablation import run_ablation
from werewolf import cli
from werewolf.artifact_io import (
    canonical_json_bytes,
    publish_artifact,
    read_artifact_file,
    sha256_bytes,
    verify_artifact,
)
from werewolf.backends import load_named_backends
from werewolf.canonical_collection.attempt_ledger import (
    CollectionSeedPoolExhausted,
    TerminalOutcome,
    construct_collection_plan,
    load_collection_plan,
    validate_attempt_ledger,
    validate_collection_plan,
)
from werewolf.canonical_collection.collector import CanonicalGameProduct, collect
from werewolf.canonical_collection.failure_evidence import (
    CanonicalFailureStage,
    validate_canonical_failure_evidence,
)
from werewolf.canonical_collection.game_bundle import validate_canonical_game_bundle
from werewolf.canonical_collection.production_runtime import (
    Classic7RuntimeFactory,
    classic7_replay_executor,
)
from werewolf.canonical_collection.public_history import PublicPhase


EXPERIMENT_ID = "paper-tom-gameplay-ablation-v1"
TARGET_PAIRS = 40
CANDIDATE_COUNT = 100
NOTOM = "Wolf-NoToM"
TOM = "Wolf+ToM"
ARMS = (NOTOM, TOM)
TOM_HEADER = "PRIVATE ToM INFORMATION"
_SCHEMA = "tom_gameplay_collection_v1"


def _order(ordinal: int) -> tuple[str, str]:
    return (NOTOM, TOM) if ordinal % 2 == 0 else (TOM, NOTOM)


def _slug(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError("invalid ablation arm")
    return "notom" if arm == NOTOM else "tom"


def _digest(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _publish_record(path: Path, kind: str, record: dict[str, Any]):
    """Publish an immutable, canonical, self-digested private record."""
    payload = canonical_json_bytes(record)
    artifact = publish_artifact(
        path,
        manifest_fields={
            "artifact_type": kind,
            "schema_version": _SCHEMA,
            "record_digest": sha256_bytes(payload),
        },
        files={"record.json": payload},
    )
    loaded, verified = _read_record(path, kind)
    if loaded != record or artifact.manifest_digest != verified.manifest_digest:
        raise ValueError(f"{kind} changed during publication")
    return verified.manifest_digest


def _read_record(path: Path, kind: str):
    artifact = verify_artifact(
        path, expected_artifact_type=kind, expected_schema_version=_SCHEMA
    )
    payload = read_artifact_file(artifact, "record.json")
    record = json.loads(payload)
    if (
        type(record) is not dict
        or canonical_json_bytes(record) != payload
        or set(artifact.manifest) != {
            "artifact_type", "schema_version", "record_digest", "file_table",
            "manifest_digest",
        }
        or artifact.manifest["record_digest"] != sha256_bytes(payload)
    ):
        raise ValueError(f"{kind} record is not canonical")
    return record, artifact


def _experiment_record(head: str, provenance: dict[str, str], call_limit: int):
    seeds = operator.derive_seed_pool(EXPERIMENT_ID, CANDIDATE_COUNT)
    if len(seeds) != CANDIDATE_COUNT:
        raise ValueError("candidate pool size mismatch")
    return {
        "experiment_id": EXPERIMENT_ID,
        "source_revision": head,
        "ordered_seed_pool": list(seeds),
        "candidate_seed_pool_digest": _digest(list(seeds)),
        "target_valid_pairs": TARGET_PAIRS,
        "candidate_count": CANDIDATE_COUNT,
        "arms": list(ARMS),
        "arm_order": "candidate-ordinal-even-notom-first-odd-tom-first-v1",
        "pair_failure_policy": "terminal-first-arm-fail-fast-v1",
        "runtime_provenance": provenance,
        "configured_call_limit": call_limit,
        "predictor_source_revision": SOURCE_REVISION,
        "predictor_fit_digest": FIT_DIGEST,
        "predictor_seal_digest": SEAL_DIGEST,
    }


def _arm_path(root: Path, ordinal: int, arm: str) -> Path:
    return root / "candidates" / f"{ordinal:03d}" / _slug(arm)


def _arm_plan(record: dict[str, Any], ordinal: int, arm: str):
    seed = record["ordered_seed_pool"][ordinal]
    identity = f"{EXPERIMENT_ID}-{ordinal:03d}-{_slug(arm)}"
    campaign = operator.validate_campaign({
        "collection_id": identity,
        "target_games": 1,
        "seed_pool_size": 1,
        "call_limit": record["configured_call_limit"],
    })
    provenance = {
        **record["runtime_provenance"],
        "seed_rule_identity": f"{operator.SEED_RULE}:{EXPERIMENT_ID}",
        "gameplay_ablation_experiment_id": EXPERIMENT_ID,
        "gameplay_ablation_candidate_ordinal": str(ordinal),
        "gameplay_ablation_candidate_pool_digest": record["candidate_seed_pool_digest"],
        "gameplay_ablation_arm": arm,
        "gameplay_ablation_fit_digest": FIT_DIGEST if arm == TOM else "none",
        "gameplay_ablation_seal_digest": SEAL_DIGEST if arm == TOM else "none",
    }
    fields = operator.plan_fields(campaign, record["source_revision"], provenance)
    fields["agent_identity"] = (
        "classic7-gpt-pre-belief-handoff-qwen3-tom-v1" if arm == TOM
        else "classic7-gpt-pre-belief-handoff-v1"
    )
    return validate_collection_plan(
        construct_collection_plan(ordered_seed_pool=(seed,), **fields)
    )


def _has_tom_header(call) -> bool:
    payload = call.private_payload.to_value()
    request = {"args": payload.get("args", []), "kwargs": payload.get("kwargs", {})}
    return TOM_HEADER in json.dumps(request, ensure_ascii=False)


def _validate_treatment(evidence, arm: str, entries: list[dict], request_count):
    prefixes = {prefix.boundary_id: prefix
                for prefix in evidence.authoritative_pre_prefixes}
    roles = dict(evidence.private_replay_evidence.role_assignment)
    if arm == NOTOM:
        if entries or request_count is not None:
            raise ValueError("NoToM received treatment or predictor requests")
        if any(_has_tom_header(call) for call in evidence.backend_call_evidence):
            raise ValueError("NoToM backend request contains ToM information")
        return
    if type(request_count) is not int or request_count != len(entries):
        raise ValueError("+ToM predictor request count differs from treatment audit")
    treated = set()
    required = {
        "game_id", "boundary_id", "prefix_digest", "observer_ids",
        "payload_digest", "fit_digest", "seal_digest",
    }
    for entry in entries:
        if type(entry) is not dict or set(entry) != required:
            raise ValueError("invalid treatment audit entry")
        boundary = entry["boundary_id"]
        if boundary in treated or boundary not in prefixes:
            raise ValueError("duplicate or unknown treatment boundary")
        treated.add(boundary)
        prefix = prefixes[boundary]
        speaker = prefix.current_speaker
        if (
            entry["game_id"] != evidence.game_id
            or entry["prefix_digest"] != prefix.prefix_digest
            or entry["fit_digest"] != FIT_DIGEST
            or entry["seal_digest"] != SEAL_DIGEST
            or roles[speaker] != "Werewolf"
            or prefix.public_temporal_state.phase not in {
                PublicPhase.DISCUSSION, PublicPhase.PK_DISCUSSION
            }
            or not any(
                action.boundary_id == boundary
                and action.actor_id == speaker
                and action.action_type == "public_speech"
                for action in evidence.submitted_gameplay_actions
            )
        ):
            raise ValueError("treatment audit does not match wolf public speech")
    seen = set()
    for call in evidence.backend_call_evidence:
        if not _has_tom_header(call):
            continue
        if call.boundary_id not in treated:
            raise ValueError("ToM request reached an untreated boundary")
        prefix = prefixes[call.boundary_id]
        if (
            call.observer_id != prefix.current_speaker
            or roles[call.observer_id] != "Werewolf"
            or prefix.public_temporal_state.phase not in {
                PublicPhase.DISCUSSION, PublicPhase.PK_DISCUSSION
            }
        ):
            raise ValueError("ToM request reached a non-wolf speech context")
        seen.add(call.boundary_id)
    if seen != treated:
        raise ValueError("treatment prediction did not reach a backend request")


def _audit_record(*, record, ordinal, seed, arm, plan, claim, evidence,
                  entries, request_count):
    return {
        "experiment_id": EXPERIMENT_ID,
        "candidate_ordinal": ordinal,
        "seed": seed,
        "arm": arm,
        "plan_digest": plan.plan_digest,
        "claim_digest": claim.record_digest,
        "game_id": evidence.game_id,
        "treatment_count": len(entries),
        "predictor_requests": request_count,
        "treatments": entries,
        "fit_digest": FIT_DIGEST if arm == TOM else None,
        "seal_digest": SEAL_DIGEST if arm == TOM else None,
        "candidate_seed_pool_digest": record["candidate_seed_pool_digest"],
    }


class _AblationRuntime:
    def __init__(self, *, runtime, plan, claim, arm, ordinal, record,
                 collection_path, predictor_checkout, fit_path):
        self.runtime = runtime
        self.plan = plan
        self.claim = claim
        self.arm = arm
        self.ordinal = ordinal
        self.record = record
        self.collection_path = collection_path
        self.predictor_checkout = predictor_checkout
        self.fit_path = fit_path

    def run(self) -> CanonicalGameProduct:
        runtime = self.runtime
        treatments: list[dict] = []
        requests = None
        try:
            if self.arm == NOTOM:
                result = run_ablation(
                    runtime.env, runtime.agents, runtime.roles,
                    recorder=runtime.recorder, call_audit=runtime.call_audit,
                    arm=NOTOM, predictor=None, treatment_audit=treatments,
                )
            else:
                with Qwen3GameplayPredictorClient(
                    checkout=self.predictor_checkout,
                    fit_path=self.fit_path,
                    python_executable=sys.executable,
                ) as predictor:
                    result = run_ablation(
                        runtime.env, runtime.agents, runtime.roles,
                        recorder=runtime.recorder, call_audit=runtime.call_audit,
                        arm=TOM, predictor=predictor, treatment_audit=treatments,
                    )
                    requests = predictor._next_request
            evidence = runtime.recorder.complete_evidence()
            winner = evidence.private_replay_evidence.replay_inputs.to_value().get("winner")
            if winner not in {"Werewolf", "Villager"} or result != f"{winner} win":
                raise ValueError("game result differs from canonical private replay winner")
            _validate_treatment(evidence, self.arm, treatments, requests)
            _publish_record(
                self.collection_path / "private_treatment_audit",
                "tom_gameplay_treatment_audit",
                _audit_record(
                    record=self.record, ordinal=self.ordinal, seed=self.claim.seed,
                    arm=self.arm, plan=self.plan, claim=self.claim, evidence=evidence,
                    entries=treatments, request_count=requests,
                ),
            )
        except Exception as error:
            raise runtime.recorder.failure_from_exception(
                error, default_stage=CanonicalFailureStage.RUNTIME
            ) from error
        return CanonicalGameProduct(
            evidence=evidence, replay_executor=runtime.replay_executor
        )


class _AblationRuntimeFactory:
    def __init__(self, *, base_factory, arm, ordinal, record, collection_path,
                 predictor_checkout, fit_path):
        self.base_factory = base_factory
        self.arm = arm
        self.ordinal = ordinal
        self.record = record
        self.collection_path = collection_path
        self.predictor_checkout = predictor_checkout
        self.fit_path = fit_path

    def __call__(self, *, plan, claim):
        return _AblationRuntime(
            runtime=self.base_factory(plan=plan, claim=claim),
            plan=plan, claim=claim, arm=self.arm, ordinal=self.ordinal,
            record=self.record, collection_path=self.collection_path,
            predictor_checkout=self.predictor_checkout, fit_path=self.fit_path,
        )


def _arm_status(*, root, record, ordinal, arm, replay_executor):
    path = _arm_path(root, ordinal, arm)
    if not path.exists():
        return None
    plan = _arm_plan(record, ordinal, arm)
    bound = load_collection_plan(path)
    if bound.plan_digest != plan.plan_digest:
        raise ValueError("arm CollectionPlan differs from frozen experiment")
    state = validate_attempt_ledger(path / "attempt_ledger", plan)
    if state.open_claim is not None:
        return {"status": "open"}
    if not state.claims:
        return None
    if len(state.claims) != 1 or len(state.terminals) != 1:
        raise ValueError("single-seed arm has an invalid attempt count")
    claim, terminal = state.claims[0], state.terminals[0]
    common = {
        "arm": arm,
        "collection_id": plan.collection_id,
        "plan_digest": plan.plan_digest,
        "claim_digest": claim.record_digest,
        "terminal_digest": terminal.record_digest,
        "terminal_outcome": terminal.outcome.value,
        "game_id": None,
        "bundle_digest": None,
        "winner": None,
        "roles": None,
        "backend_calls": None,
        "speech_boundaries": None,
        "treatment_count": None,
        "predictor_fit_digest": FIT_DIGEST if arm == TOM else None,
        "predictor_seal_digest": SEAL_DIGEST if arm == TOM else None,
        "source_revision": plan.source_revision,
        "runtime_provenance_digest": plan.runtime_provenance_digest,
        "treatment_audit_digest": None,
        "failure_stage": None,
        "failure_type": None,
    }
    if terminal.outcome is TerminalOutcome.CANONICAL_SUCCESS:
        bundle = validate_canonical_game_bundle(
            path / "games" / terminal.canonical_game_bundle_id,
            plan=plan, claim=claim, replay_executor=replay_executor,
        )
        if bundle.manifest_digest != terminal.canonical_game_bundle_digest:
            raise ValueError("arm Bundle differs from success terminal")
        audit, artifact = _read_record(
            path / "private_treatment_audit", "tom_gameplay_treatment_audit"
        )
        required = {
            "experiment_id", "candidate_ordinal", "seed", "arm",
            "plan_digest", "claim_digest", "game_id", "treatment_count",
            "predictor_requests", "treatments", "fit_digest", "seal_digest",
            "candidate_seed_pool_digest",
        }
        if (
            set(audit) != required
            or audit["experiment_id"] != EXPERIMENT_ID
            or audit["candidate_ordinal"] != ordinal
            or audit["seed"] != claim.seed
            or audit["arm"] != arm
            or audit["plan_digest"] != plan.plan_digest
            or audit["claim_digest"] != claim.record_digest
            or audit["game_id"] != bundle.game_id
            or audit["candidate_seed_pool_digest"] != record["candidate_seed_pool_digest"]
            or audit["fit_digest"] != (FIT_DIGEST if arm == TOM else None)
            or audit["seal_digest"] != (SEAL_DIGEST if arm == TOM else None)
            or type(audit["treatment_count"]) is not int
            or audit["treatment_count"] != len(audit["treatments"])
        ):
            raise ValueError("private treatment audit identity mismatch")
        _validate_treatment(
            bundle, arm, audit["treatments"], audit["predictor_requests"]
        )
        winner = bundle.private_replay_evidence.replay_inputs.to_value().get("winner")
        if winner not in {"Werewolf", "Villager"}:
            raise ValueError("invalid recorded game winner")
        common.update({
            "status": "success",
            "game_id": bundle.game_id,
            "bundle_digest": bundle.manifest_digest,
            "winner": winner,
            "roles": dict(bundle.private_replay_evidence.role_assignment),
            "backend_calls": bundle.call_budget_summary.used_calls,
            "speech_boundaries": len(bundle.authoritative_pre_prefixes),
            "treatment_count": audit["treatment_count"],
            "treatment_audit_digest": artifact.manifest_digest,
        })
    elif terminal.outcome is TerminalOutcome.CANONICAL_FAILURE:
        failure = validate_canonical_failure_evidence(
            path / "attempts" / claim.attempt_id / "failure_evidence.json",
            plan=plan, claim=claim,
        )
        if failure.file_sha256 != terminal.failure_evidence_digest:
            raise ValueError("failure evidence differs from terminal")
        common.update({
            "status": "failure",
            "game_id": failure.evidence.game_id,
            "failure_stage": failure.evidence.stage.value,
            "failure_type": failure.evidence.error_category,
        })
    elif terminal.outcome is TerminalOutcome.INTERRUPTED_FAILURE:
        common.update({
            "status": "failure",
            "failure_stage": "interrupted",
            "failure_type": "interrupted_failure",
        })
    else:
        raise ValueError("unknown arm terminal outcome")
    return common


def _execute_arm(*, root, record, ordinal, arm, base_factory,
                 replay_executor, predictor_checkout, fit_path):
    status = _arm_status(
        root=root, record=record, ordinal=ordinal, arm=arm,
        replay_executor=replay_executor,
    )
    if status is not None and status["status"] != "open":
        return status
    path = _arm_path(root, ordinal, arm)
    plan = _arm_plan(record, ordinal, arm)
    factory = _AblationRuntimeFactory(
        base_factory=base_factory, arm=arm, ordinal=ordinal, record=record,
        collection_path=path, predictor_checkout=predictor_checkout,
        fit_path=fit_path,
    )
    try:
        collect(plan=plan, runtime_factory=factory, destination=path)
    except CollectionSeedPoolExhausted:
        # One failed singleton Plan has no further claim slot. The terminal,
        # not this exception, determines the arm result.
        pass
    status = _arm_status(
        root=root, record=record, ordinal=ordinal, arm=arm,
        replay_executor=replay_executor,
    )
    if status is None or status["status"] == "open":
        raise ValueError("collector returned without a terminal arm outcome")
    return status


def _pair_record(record, ordinal, order, statuses):
    first = statuses[order[0]]
    second = statuses[order[1]]
    if first is None or first["status"] not in {"success", "failure"}:
        raise ValueError("first arm lacks a terminal outcome")
    if first["status"] == "failure" and second is not None:
        raise ValueError("pair violates fail-fast terminal order")
    if first["status"] == "success" and (
        second is None or second["status"] not in {"success", "failure"}
    ):
        raise ValueError("successful first arm requires a terminal second arm")
    roles_equal = (
        first["roles"] == second["roles"]
        if first["status"] == "success" and second["status"] == "success"
        else None
    )
    valid = (
        first["status"] == "success"
        and second is not None
        and second["status"] == "success"
        and roles_equal is True
    )
    failure_arm = None
    failure_stage = None
    failure_type = None
    if not valid:
        if first["status"] == "failure":
            failed = first
        elif second["status"] == "failure":
            failed = second
        else:
            failed = None
        if failed is not None:
            failure_arm = failed["arm"]
            failure_stage = failed["failure_stage"]
            failure_type = failed["failure_type"]
        else:
            failure_stage = "pair_validation"
            failure_type = "initial_roles_mismatch"
    return {
        "experiment_id": EXPERIMENT_ID,
        "candidate_ordinal": ordinal,
        "seed": record["ordered_seed_pool"][ordinal],
        "candidate_seed_pool_digest": record["candidate_seed_pool_digest"],
        "arm_execution_order": list(order),
        "arms": {arm: statuses[arm] if statuses[arm] is not None else {
            "arm": arm, "status": "not_run"
        } for arm in ARMS},
        "roles_equal": roles_equal,
        "status": "completed_valid" if valid else "invalid",
        "failure_arm": failure_arm,
        "failure_stage": failure_stage,
        "failure_type": failure_type,
    }


def _pair_path(root: Path, ordinal: int) -> Path:
    return root / "pairs" / f"{ordinal:03d}"


def collect_pairs(*, root, record, base_factory, replay_executor,
                  predictor_checkout, fit_path):
    completed = []
    invalid = []
    pair_digests = []
    seeds = record["ordered_seed_pool"]
    for ordinal, seed in enumerate(seeds):
        if len(completed) == TARGET_PAIRS:
            break
        order = _order(ordinal)
        pair_path = _pair_path(root, ordinal)
        if pair_path.exists():
            statuses = {
                arm: _arm_status(
                    root=root, record=record, ordinal=ordinal, arm=arm,
                    replay_executor=replay_executor,
                ) for arm in ARMS
            }
            if any(item is not None and item["status"] == "open"
                   for item in statuses.values()):
                raise ValueError("published pair record has an open arm claim")
            expected = _pair_record(record, ordinal, order, statuses)
            existing, artifact = _read_record(pair_path, "tom_gameplay_pair")
            if existing != expected:
                raise ValueError("published pair record differs from arm artifacts")
            digest = artifact.manifest_digest
        else:
            statuses = {arm: None for arm in ARMS}
            first = order[0]
            statuses[first] = _execute_arm(
                root=root, record=record, ordinal=ordinal, arm=first,
                base_factory=base_factory, replay_executor=replay_executor,
                predictor_checkout=predictor_checkout, fit_path=fit_path,
            )
            if statuses[first]["status"] == "success":
                second = order[1]
                statuses[second] = _execute_arm(
                    root=root, record=record, ordinal=ordinal, arm=second,
                    base_factory=base_factory, replay_executor=replay_executor,
                    predictor_checkout=predictor_checkout, fit_path=fit_path,
                )
            elif _arm_path(root, ordinal, order[1]).exists():
                raise ValueError("second arm exists after failed first arm")
            pair = _pair_record(record, ordinal, order, statuses)
            digest = _publish_record(pair_path, "tom_gameplay_pair", pair)
            expected = pair
        pair_digests.append({"candidate_ordinal": ordinal, "seed": seed,
                             "pair_digest": digest})
        if expected["status"] == "completed_valid":
            completed.append({"candidate_ordinal": ordinal, "seed": seed})
        else:
            invalid.append({
                "candidate_ordinal": ordinal, "seed": seed,
                "failure_arm": expected["failure_arm"],
                "failure_stage": expected["failure_stage"],
                "failure_type": expected["failure_type"],
                "pair_digest": digest,
            })
    if len(completed) != TARGET_PAIRS:
        raise CollectionSeedPoolExhausted(
            "100 frozen candidate seeds exhausted before 40 valid pairs"
        )
    reasons: dict[str, int] = {}
    for item in invalid:
        key = f"{item['failure_stage']}:{item['failure_type']}"
        reasons[key] = reasons.get(key, 0) + 1
    final_record = {
        "experiment_id": EXPERIMENT_ID,
        "source_revision": record["source_revision"],
        "runtime_provenance": record["runtime_provenance"],
        "experiment_plan_digest": _read_record(
            root / "experiment_plan", "tom_gameplay_experiment_plan"
        )[1].manifest_digest,
        "candidate_seed_pool_digest": record["candidate_seed_pool_digest"],
        "attempted_candidate_count": len(pair_digests),
        "completed_valid_count": len(completed),
        "included_pairs": completed,
        "invalid_pairs": invalid,
        "failure_reason_counts": reasons,
        "pair_record_digests": pair_digests,
    }
    return _publish_record(root / "final", "tom_gameplay_final", final_record)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictor-checkout", type=Path, required=True)
    parser.add_argument("--fit-path", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    head = operator.clean_head()
    provenance, base_url = operator.inspect_inputs()
    operator.live_preflight(base_url, provenance["served_model_name"])
    config = cli._runtime(operator.RUNTIME)
    if _digest(config) != provenance["runtime_config_sha256"]:
        raise ValueError("runtime configuration changed during preflight")
    storage = cli._storage_root(operator.STORAGE)
    campaign = operator.validate_campaign(operator.read_json(
        storage / "operator/collection_campaign.json"
    ))
    call_limit = campaign["call_limit"]
    if operator.clean_head() != head:
        raise ValueError("gameplay source changed during preflight")
    expected = _experiment_record(head, provenance, call_limit)
    root = cli._artifact_path(
        storage, Path("gameplay_ablations") / EXPERIMENT_ID
    )
    plan_path = root / "experiment_plan"
    if args.resume:
        if not plan_path.is_dir():
            raise ValueError("--resume requires an existing frozen experiment plan")
        record, _ = _read_record(plan_path, "tom_gameplay_experiment_plan")
        if record != expected:
            raise ValueError("frozen experiment plan, seed pool, or provenance mismatch")
    else:
        if root.exists():
            raise ValueError("existing gameplay ablation root requires --resume")
        _publish_record(plan_path, "tom_gameplay_experiment_plan", expected)
        record = expected
    backends = load_named_backends(config, max_retries=0)
    base_factory = Classic7RuntimeFactory(
        runtime_config=config, backends=backends,
        configured_call_limit=call_limit,
    )
    digest = collect_pairs(
        root=root, record=record, base_factory=base_factory,
        replay_executor=classic7_replay_executor(config),
        predictor_checkout=args.predictor_checkout.resolve(),
        fit_path=args.fit_path.resolve(),
    )
    print(f"PAIRED_GAMEPLAY_COLLECTION_COMPLETE {digest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
