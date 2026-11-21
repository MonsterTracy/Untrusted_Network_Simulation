"""Read-only analysis of the completed formal paired ToM gameplay collection."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

from scripts import collect_games as operator
from scripts import tom_gameplay_collection as collection
from werewolf import cli
from werewolf.artifact_io import (
    canonical_json_bytes, open_artifact_envelope, read_artifact_file, sha256_bytes,
)
from werewolf.canonical_collection.game_bundle import (
    CANONICAL_GAME_BUNDLE_SCHEMA_VERSION, _backend_call_from_record,
    _prefix_from_record, _private_replay_from_record,
)


EXPECTED_FINAL_DIGEST = "067e1f18ccb9dc684fe720d740ee7b3206d227a7f3ab2d49a3d3ffc747252a18"
EXPECTED_ATTEMPTED = 42
EXPECTED_INVALID = {
    14: (collection.NOTOM, "GameplayGenerationExhausted"),
    15: (collection.TOM, "GameplayGenerationExhausted"),
}
BOOTSTRAP_SEED = 20260928
BOOTSTRAP_REPLICATES = 100000
_HEX = set("0123456789abcdef")


def _digest(value: object) -> bool:
    return type(value) is str and len(value) == 64 and set(value) <= _HEX


def _int(value: object, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _plan(root: Path):
    record, artifact = collection._read_record(
        root / "experiment_plan", "tom_gameplay_experiment_plan"
    )
    if record != collection._experiment_record(
        record["source_revision"], record["runtime_provenance"],
        record["configured_call_limit"],
    ):
        raise ValueError("experiment plan differs from frozen protocol or seed pool")
    return record, artifact.manifest_digest


def _final(root: Path, plan: dict, plan_digest: str, expected_digest: str):
    record, artifact = collection._read_record(root / "final", "tom_gameplay_final")
    if artifact.manifest_digest != expected_digest:
        raise ValueError("formal final collection digest mismatch")
    required = {
        "experiment_id", "source_revision", "runtime_provenance",
        "experiment_plan_digest", "candidate_seed_pool_digest",
        "attempted_candidate_count", "completed_valid_count", "included_pairs",
        "invalid_pairs", "failure_reason_counts", "pair_record_digests",
    }
    if (
        set(record) != required
        or record["experiment_id"] != collection.EXPERIMENT_ID
        or record["source_revision"] != plan["source_revision"]
        or record["runtime_provenance"] != plan["runtime_provenance"]
        or record["experiment_plan_digest"] != plan_digest
        or record["candidate_seed_pool_digest"] != plan["candidate_seed_pool_digest"]
        or record["attempted_candidate_count"] != EXPECTED_ATTEMPTED
        or record["completed_valid_count"] != collection.TARGET_PAIRS
        or len(record["pair_record_digests"]) != EXPECTED_ATTEMPTED
        or len(record["included_pairs"]) != collection.TARGET_PAIRS
        or len(record["invalid_pairs"]) != len(EXPECTED_INVALID)
    ):
        raise ValueError("formal final collection is incomplete or inconsistent")
    return record, artifact.manifest_digest


def _check_arm(plan: dict, ordinal: int, seed: int, arm: str, status: dict,
               *, check_treatments=True):
    if type(status) is not dict or status.get("arm") != arm:
        raise ValueError("pair arm identity mismatch")
    if status.get("status") == "not_run":
        if status != {"arm": arm, "status": "not_run"}:
            raise ValueError("invalid not-run arm record")
        return
    arm_plan = collection._arm_plan(plan, ordinal, arm)
    if (
        status.get("collection_id") != arm_plan.collection_id
        or status.get("plan_digest") != arm_plan.plan_digest
        or status.get("source_revision") != plan["source_revision"]
        or status.get("runtime_provenance_digest") != arm_plan.runtime_provenance_digest
        or status.get("predictor_fit_digest") != (
            collection.FIT_DIGEST if arm == collection.TOM else None
        )
        or status.get("predictor_seal_digest") != (
            collection.SEAL_DIGEST if arm == collection.TOM else None
        )
        or not _digest(status.get("claim_digest"))
        or not _digest(status.get("terminal_digest"))
    ):
        raise ValueError("pair arm provenance or terminal identity mismatch")
    if status.get("status") == "success":
        expected_game_id = (
            f"{arm_plan.collection_id}-game-000000-seed-{seed}"
        )
        if (
            status.get("terminal_outcome") != "canonical_success"
            or status.get("game_id") != expected_game_id
            or status.get("winner") not in {"Werewolf", "Villager"}
            or not _digest(status.get("bundle_digest"))
            or not _digest(status.get("treatment_audit_digest"))
            or type(status.get("roles")) is not dict
            or not _int(status.get("backend_calls"))
            or not _int(status.get("speech_boundaries"))
            or not _int(status.get("treatment_count"))
            or (check_treatments and arm == collection.NOTOM
                and status["treatment_count"] != 0)
            or status.get("failure_stage") is not None
            or status.get("failure_type") is not None
        ):
            raise ValueError("invalid canonical-success arm record")
    elif status.get("status") == "failure":
        if (
            status.get("terminal_outcome") not in {
                "canonical_failure", "interrupted_failure"
            }
            or type(status.get("failure_stage")) is not str
            or type(status.get("failure_type")) is not str
            or status.get("winner") is not None
            or status.get("bundle_digest") is not None
        ):
            raise ValueError("invalid technical-failure arm record")
    else:
        raise ValueError("unknown pair arm status")


def _pairs(root: Path, plan: dict, final: dict, *, check_treatments=True):
    included = []
    valid_pairs = []
    invalid = []
    seen_ordinals = set()
    seen_seeds = set()
    reasons = Counter()
    for ordinal, reference in enumerate(final["pair_record_digests"]):
        seed = plan["ordered_seed_pool"][ordinal]
        if (
            type(reference) is not dict
            or set(reference) != {"candidate_ordinal", "seed", "pair_digest"}
            or reference["candidate_ordinal"] != ordinal
            or reference["seed"] != seed
            or not _digest(reference["pair_digest"])
            or ordinal in seen_ordinals
            or seed in seen_seeds
        ):
            raise ValueError("duplicate or reordered candidate/seed reference")
        seen_ordinals.add(ordinal)
        seen_seeds.add(seed)
        pair, artifact = collection._read_record(
            collection._pair_path(root, ordinal), "tom_gameplay_pair"
        )
        if artifact.manifest_digest != reference["pair_digest"]:
            raise ValueError("pair artifact digest differs from final manifest")
        if (
            pair.get("experiment_id") != collection.EXPERIMENT_ID
            or pair.get("candidate_ordinal") != ordinal
            or pair.get("seed") != seed
            or pair.get("candidate_seed_pool_digest") != plan["candidate_seed_pool_digest"]
            or pair.get("arm_execution_order") != list(collection._order(ordinal))
            or type(pair.get("arms")) is not dict
            or set(pair["arms"]) != set(collection.ARMS)
        ):
            raise ValueError("pair identity, seed, or arm order mismatch")
        for arm in collection.ARMS:
            _check_arm(plan, ordinal, seed, arm, pair["arms"][arm],
                       check_treatments=check_treatments)
        expected = collection._pair_record(
            plan, ordinal, collection._order(ordinal), {
                arm: None if pair["arms"][arm]["status"] == "not_run"
                else pair["arms"][arm] for arm in collection.ARMS
            }
        )
        if pair != expected:
            raise ValueError("pair status or paired-role validation mismatch")
        if pair["status"] == "completed_valid":
            if (
                pair["roles_equal"] is not True
                or any(pair["arms"][arm]["status"] != "success"
                       for arm in collection.ARMS)
            ):
                raise ValueError("included pair lacks two valid arms")
            included.append({"candidate_ordinal": ordinal, "seed": seed})
            valid_pairs.append(pair)
        else:
            item = {
                "candidate_ordinal": ordinal, "seed": seed,
                "failure_arm": pair["failure_arm"],
                "failure_stage": pair["failure_stage"],
                "failure_type": pair["failure_type"],
                "pair_digest": artifact.manifest_digest,
            }
            invalid.append(item)
            reasons[f"{item['failure_stage']}:{item['failure_type']}"] += 1
    if (
        included != final["included_pairs"]
        or invalid != final["invalid_pairs"]
        or dict(reasons) != final["failure_reason_counts"]
        or len(included) != collection.TARGET_PAIRS
        or {item["candidate_ordinal"]: (
            item["failure_arm"], item["failure_type"]
        ) for item in invalid} != EXPECTED_INVALID
    ):
        raise ValueError("final inclusion or formal failure diagnostics mismatch")
    return valid_pairs, invalid


def _exact_mcnemar(n01: int, n10: int) -> float:
    discordant = n01 + n10
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(n01, n10) + 1))
    return min(1.0, 2.0 * tail / (1 << discordant))


def _percentile(sorted_values: list[float], fraction: float) -> float:
    index = (len(sorted_values) - 1) * fraction
    low = math.floor(index)
    high = math.ceil(index)
    return sorted_values[low] + (index - low) * (
        sorted_values[high] - sorted_values[low]
    )


def _paired_bootstrap(effects: list[int], *, seed: int = BOOTSTRAP_SEED,
                      replicates: int = BOOTSTRAP_REPLICATES) -> list[float]:
    if not effects or any(effect not in {-1, 0, 1} for effect in effects):
        raise ValueError("bootstrap requires binary paired effects")
    if not _int(replicates, minimum=1):
        raise ValueError("bootstrap replicate count must be positive")
    generator = random.Random(seed)
    size = len(effects)
    estimates = sorted(
        sum(effects[generator.randrange(size)] for _ in range(size)) / size
        for _ in range(replicates)
    )
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def analyze_collection(root: Path, *, expected_final_digest: str = EXPECTED_FINAL_DIGEST):
    root = Path(root)
    plan, plan_digest = _plan(root)
    final, final_digest = _final(root, plan, plan_digest, expected_final_digest)
    valid_pairs, invalid = _pairs(root, plan, final)
    outcomes = Counter()
    effects = []
    treatment_counts = []
    calls = {arm: [] for arm in collection.ARMS}
    speeches = {arm: [] for arm in collection.ARMS}
    for pair in valid_pairs:
        arms = pair["arms"]
        notom_win = arms[collection.NOTOM]["winner"] == "Werewolf"
        tom_win = arms[collection.TOM]["winner"] == "Werewolf"
        outcomes[int(notom_win), int(tom_win)] += 1
        effects.append(int(tom_win) - int(notom_win))
        treatment_counts.append(arms[collection.TOM]["treatment_count"])
        for arm in collection.ARMS:
            calls[arm].append(arms[arm]["backend_calls"])
            speeches[arm].append(arms[arm]["speech_boundaries"])
    n = len(valid_pairs)
    n00, n01, n10, n11 = (outcomes[a, b] for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)))
    if n00 + n01 + n10 + n11 != n:
        raise ValueError("paired outcome counts do not sum to N")
    wins = {collection.NOTOM: n10 + n11, collection.TOM: n01 + n11}
    delta = (wins[collection.TOM] - wins[collection.NOTOM]) / n
    if delta != (n01 - n10) / n or delta != sum(effects) / n:
        raise ValueError("paired effect disagrees with discordant counts")
    arm_results = {}
    for arm in collection.ARMS:
        arm_results[arm] = {
            "wins": wins[arm], "losses": n - wins[arm], "win_rate": wins[arm] / n,
            "backend_calls": {
                "mean": statistics.mean(calls[arm]),
                "median": statistics.median(calls[arm]),
            },
            "speech_boundaries": {
                "mean": statistics.mean(speeches[arm]),
                "median": statistics.median(speeches[arm]),
            },
        }
    return {
        "experiment_id": collection.EXPERIMENT_ID,
        "source_collection_digest": final_digest,
        "N": n,
        "attempted_candidates": final["attempted_candidate_count"],
        "invalid_candidate_count": len(invalid),
        "invalid_candidates": [
            {key: item[key] for key in (
                "candidate_ordinal", "seed", "failure_arm", "failure_stage",
                "failure_type",
            )} for item in invalid
        ],
        "arms": arm_results,
        "paired": {
            "n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "discordant_count": n01 + n10,
            "discordant_rate": (n01 + n10) / n,
            "delta_hat": delta,
            "bootstrap_95_ci": _paired_bootstrap(effects),
            "exact_mcnemar_p": _exact_mcnemar(n01, n10),
        },
        "treatment": {
            "zero_treatment_games": sum(count == 0 for count in treatment_counts),
            "positive_treatment_games": sum(count > 0 for count in treatment_counts),
            "total_treatments": sum(treatment_counts),
            "mean_treatments": statistics.mean(treatment_counts),
        },
        "analysis": {
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
    }


UNKNOWN = "UNKNOWN"
_COVERAGE_FILES = (
    "public/authoritative_pre_prefixes.jsonl",
    "private/backend_call_evidence.jsonl",
    "private/submitted_gameplay_actions.jsonl",
    "private/private_replay_evidence.json",
)
_COVERAGE_ANOMALIES = (
    "eligible_without_treatment", "treatment_without_matching_request",
    "treatment_payload_mismatch", "treatment_outside_eligible", "duplicate_treatment",
)


def _request_tom_payload(call):
    """只解析认知模板的私有区段，避免把公开发言中的引用当作注入。"""
    request = call.private_payload.to_value()
    if "args" not in request or "kwargs" not in request:
        raise ValueError("request args/kwargs unavailable")
    messages = request["kwargs"].get("messages")
    if messages is None and request["args"]:
        messages = request["args"][0]
    if not isinstance(messages, list):
        raise ValueError("request messages unavailable")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate ToM payload key")
            result[key] = value
        return result

    decoder = json.JSONDecoder(object_pairs_hook=unique_keys)
    found = []
    marker = "\n\n" + collection.TOM_HEADER + "\n"
    for message in messages:
        text = message.get("content")
        if not isinstance(text, str):
            raise ValueError("non-text request message")
        end = text.rfind("\n\nDISCUSSION ACTION SEMANTICS\n")
        belief = text.rfind("\n\nPRE-SPEECH PRIVATE BELIEF (IMMUTABLE INPUT)\n", 0, end)
        if end < 0 or belief < 0:
            if marker in text:
                raise ValueError("ToM block outside recognizable day cognition")
            continue
        start = text.find(marker, belief, end)
        if start < 0:
            continue
        block = text[start + len(marker):end]
        json_start = block.find("{")
        if json_start < 0:
            raise ValueError("ToM payload missing")
        payload, consumed = decoder.raw_decode(block, json_start)
        if not isinstance(payload, dict) or block[consumed:].strip():
            raise ValueError("ToM payload is not one complete JSON object")
        if (payload.get("target_ids") != [f"player{seat}" for seat in range(1, 8)]
                or not isinstance(payload.get("rows"), list)
                or any(not isinstance(row, dict) or "observer_id" not in row
                       for row in payload["rows"])):
            raise ValueError("ToM payload row/target structure unavailable")
        # 与 hook 的 canonical_json_bytes 完全相同，不只比较标题或局部字段。
        found.append((payload, sha256_bytes(canonical_json_bytes(payload))))
    if len(found) > 1:
        raise ValueError("multiple ToM payload blocks in one request")
    return found[0] if found else None


def _coverage_counts(prefixes, roles, audit, calls, actions, *, game_id, arm,
                     reported_treatments):
    """先独立构造机会，再匹配 treatment 与已调用后端的完整输入。"""
    by_boundary = {prefix.boundary_id: prefix for prefix in prefixes}
    if (len(by_boundary) != len(prefixes)
            or any(prefix.game_id != game_id for prefix in prefixes)):
        raise ValueError("duplicate PRE boundary or game identity mismatch")
    eligible = {
        prefix.boundary_id for prefix in prefixes
        if prefix.public_temporal_state.phase.value in {"discussion", "pk_discussion"}
        and prefix.current_speaker in prefix.alive_observer_ids
        and roles[prefix.current_speaker] == "Werewolf"
    }
    anomalies = []

    def note(kind, boundary=None, **details):
        prefix = by_boundary.get(boundary)
        anomalies.append({
            "kind": kind, "game_id": game_id, "boundary_id": boundary,
            "current_speaker": prefix.current_speaker if prefix else UNKNOWN,
            "expected_by_assignment": (
                arm == collection.NOTOM and kind == "eligible_without_treatment"
            ),
            **details,
        })

    entries = audit.get("treatments")
    entries_known = isinstance(entries, list)
    if not entries_known:
        note("treatments_unavailable")
        entries = []
    treated = set()
    expected_payloads = {}
    for index, entry in enumerate(entries):
        boundary = entry.get("boundary_id")
        if boundary in treated:
            note("duplicate_treatment", boundary, audit_index=index)
        treated.add(boundary)
        if boundary not in eligible:
            note("treatment_outside_eligible", boundary, audit_index=index)
        prefix = by_boundary.get(boundary)
        if (prefix is None or entry.get("game_id") != game_id
                or entry.get("prefix_digest") != prefix.prefix_digest
                or entry.get("fit_digest") != collection.FIT_DIGEST
                or entry.get("seal_digest") != collection.SEAL_DIGEST
                or not _digest(entry.get("payload_digest"))):
            note("treatment_identity_mismatch", boundary, audit_index=index)
            continue
        expected_payloads.setdefault(boundary, []).append((index, entry))
    if entries_known:
        for boundary in sorted(eligible - treated):
            note("eligible_without_treatment", boundary)
        if audit.get("treatment_count") != len(entries) or reported_treatments != len(entries):
            note("treatment_count_mismatch", audit_count=audit.get("treatment_count"),
                 pair_count=reported_treatments, counted=len(entries))

    requests = audit.get("predictor_requests", UNKNOWN)
    if requests is None and arm == collection.NOTOM:
        requests = 0  # 原 NoToM 明确不构造预测器；缺失字段不按 0 处理。
    if not _int(requests):
        requests = UNKNOWN
        note("predictor_requests_unavailable")
    elif entries_known and requests != len(entries):
        note("predictor_count_mismatch", requests=requests, treatments=len(entries))

    matched = set()
    matched_entries = set()
    returned = set()
    uncertain = set()
    payload_requests = 0
    request_unknown = False
    call_ids = set()
    for call in calls:
        if call.call_id in call_ids:
            raise ValueError("duplicate backend call_id")
        call_ids.add(call.call_id)
        boundary = call.boundary_id
        try:
            extracted = _request_tom_payload(call)
        except (ValueError, TypeError, KeyError) as error:
            request_unknown = True
            uncertain.add(boundary)
            note("request_payload_unverifiable", boundary,
                 call_id=call.call_id, reason=str(error))
            continue
        if extracted is None:
            continue
        payload_requests += 1
        payload, digest = extracted
        prefix = by_boundary.get(boundary)
        if (prefix is None or boundary not in eligible
                or call.observer_id != prefix.current_speaker
                or call.purpose.value != "speech_generation"):
            note("request_outside_eligible", boundary, call_id=call.call_id)
            continue
        candidates = expected_payloads.get(boundary, [])
        if not candidates:
            note("request_without_treatment", boundary, call_id=call.call_id)
            continue
        observer_ids = [row.get("observer_id") for row in payload.get("rows", [])]
        matching_entries = {index for index, entry in candidates
                            if entry["payload_digest"] == digest
                            and entry.get("observer_ids") == observer_ids}
        if not matching_entries:
            note("treatment_payload_mismatch", boundary, call_id=call.call_id,
                 request_payload_digest=digest,
                 treatment_payload_digests=[entry["payload_digest"] for _, entry in candidates])
            continue
        matched.add(boundary)
        matched_entries.update(matching_entries)
        if call.status.value == "success" and call.private_payload.to_value().get("response") is not None:
            returned.add(boundary)
    if entries_known:
        for index, entry in enumerate(entries):
            boundary = entry.get("boundary_id")
            if index not in matched_entries and boundary not in uncertain:
                note("treatment_without_matching_request", boundary, audit_index=index)
    if arm == collection.NOTOM and (entries or payload_requests or requests not in (0, UNKNOWN)):
        note("unexpected_tom_in_notom")

    submitted = {action["boundary_id"] for action in actions
                 if action.get("action_type") == "public_speech"
                 and action.get("boundary_id") in by_boundary
                 and action.get("actor_id") == by_boundary[action["boundary_id"]].current_speaker}
    # 无法解析且可能匹配的请求，不能悄悄当成“未注入”；已证明的匹配仍单列。
    unknown_u = not entries_known or bool((uncertain & treated & eligible) - matched)
    injected = UNKNOWN if unknown_u else len(matched)
    return {
        "E": len(eligible), "T": len(entries) if entries_known else UNKNOWN,
        "C": requests, "H": UNKNOWN if request_unknown else payload_requests,
        "U": injected, "U_verified": len(matched), "H_verified": payload_requests,
        "coverage": (UNKNOWN if injected == UNKNOWN else
                     injected / len(eligible) if eligible else "NOT_APPLICABLE"),
        "response_and_submitted_action_decisions": UNKNOWN if unknown_u else len(returned & submitted),
        "anomalies": anomalies,
    }


def _coverage_game(root, plan, pair, arm):
    status = pair["arms"][arm]
    base = collection._arm_path(root, pair["candidate_ordinal"], arm)
    bundle = open_artifact_envelope(
        base / "games" / status["game_id"],
        expected_artifact_type="canonical_game_bundle",
        expected_schema_version=CANONICAL_GAME_BUNDLE_SCHEMA_VERSION,
    )
    expected = {
        "game_id": status["game_id"], "artifact_identity": status["game_id"],
        "source_revision": plan["source_revision"], "seed": pair["seed"],
        "collection_id": status["collection_id"],
        "collection_plan_digest": status["plan_digest"],
        "claim_record_digest": status["claim_digest"],
        "runtime_provenance_digest": status["runtime_provenance_digest"],
        "canonical_eligibility": True,
    }
    if (bundle.manifest_digest != status["bundle_digest"]
            or any(bundle.manifest.get(key) != value for key, value in expected.items())):
        raise ValueError("coverage Bundle identity differs from frozen pair")
    audit, artifact = collection._read_record(
        base / "private_treatment_audit", "tom_gameplay_treatment_audit"
    )
    audit_expected = {
        "experiment_id": collection.EXPERIMENT_ID,
        "candidate_ordinal": pair["candidate_ordinal"], "seed": pair["seed"],
        "arm": arm, "plan_digest": status["plan_digest"],
        "claim_digest": status["claim_digest"], "game_id": status["game_id"],
        "candidate_seed_pool_digest": plan["candidate_seed_pool_digest"],
        "fit_digest": collection.FIT_DIGEST if arm == collection.TOM else None,
        "seal_digest": collection.SEAL_DIGEST if arm == collection.TOM else None,
    }
    if (artifact.manifest_digest != status["treatment_audit_digest"]
            or any(audit.get(key) != value for key, value in audit_expected.items())):
        raise ValueError("coverage treatment audit identity differs from frozen pair")
    # 只校验并读取统计所需文件；不启动完整游戏重放、预测器或模型验证。
    records = {}
    for name in _COVERAGE_FILES:
        data = read_artifact_file(bundle, name)
        records[name] = ([json.loads(line) for line in data.splitlines()]
                         if name.endswith(".jsonl") else json.loads(data))
    private = _private_replay_from_record(records["private/private_replay_evidence.json"])
    roles = dict(private.role_assignment)
    if private.game_id != status["game_id"] or private.seed != pair["seed"] or roles != status["roles"]:
        raise ValueError("coverage frozen role assignment differs from pair")
    prefixes = [_prefix_from_record(value)
                for value in records["public/authoritative_pre_prefixes.jsonl"]]
    calls = [_backend_call_from_record(value)
             for value in records["private/backend_call_evidence.jsonl"]]
    arm_plan = collection._arm_plan(plan, pair["candidate_ordinal"], arm)
    for call in calls:
        for key in ("backend_identity", "model_identity", "parser_identity", "prompt_identity",
                    "retry_policy_identity", "call_budget_identity"):
            if getattr(call, key) != getattr(arm_plan, key):
                raise ValueError(f"coverage backend {key} differs from frozen plan")
    return {
        "candidate_ordinal": pair["candidate_ordinal"], "seed": pair["seed"],
        "arm": arm, "game_id": status["game_id"],
        "source_revision": plan["source_revision"],
        "bundle_digest": bundle.manifest_digest,
        "treatment_audit_digest": artifact.manifest_digest,
        **_coverage_counts(
            prefixes, roles, audit, calls, records["private/submitted_gameplay_actions.jsonl"],
            game_id=status["game_id"], arm=arm,
            reported_treatments=status["treatment_count"],
        ),
    }


def analyze_coverage(root: Path, *, expected_final_digest=EXPECTED_FINAL_DIGEST):
    """仅对 Final 纳入的 80 局做离线统计；异常不改变人口或历史胜率。"""
    root = Path(root).resolve(strict=True)
    plan, plan_digest = _plan(root)
    final, final_digest = _final(root, plan, plan_digest, expected_final_digest)
    pairs, _ = _pairs(root, plan, final, check_treatments=False)
    missing = []
    for pair in pairs:
        for arm in collection.ARMS:
            base = collection._arm_path(root, pair["candidate_ordinal"], arm)
            game = base / "games" / pair["arms"][arm]["game_id"]
            required = [base / "private_treatment_audit" / name for name in ("manifest.json", "record.json")]
            required += [game / name for name in ("manifest.json", *_COVERAGE_FILES)]
            missing.extend(str(path) for path in required if not path.is_file())
    if missing:
        raise ValueError("coverage UNKNOWN: required Artifact files missing:\n" + "\n".join(missing))
    pair_digests = {item["candidate_ordinal"]: item["pair_digest"]
                    for item in final["pair_record_digests"]}
    games = [{"pair_digest": pair_digests[pair["candidate_ordinal"]],
              **_coverage_game(root, plan, pair, arm)}
             for pair in pairs for arm in collection.ARMS]
    if len({row["game_id"] for row in games}) != 80:
        raise ValueError("coverage requires 80 distinct included games")
    arms = {}
    for arm in collection.ARMS:
        rows = [row for row in games if row["arm"] == arm]
        totals = {}
        for field in ("E", "T", "C", "H", "U"):
            values = [row[field] for row in rows]
            totals[field] = UNKNOWN if UNKNOWN in values else sum(values)
        totals["games"] = len(rows)
        totals["mean_per_game"] = {
            field: UNKNOWN if totals[field] == UNKNOWN else totals[field] / len(rows)
            for field in ("E", "T", "C", "H", "U")
        }
        totals["coverage"] = (UNKNOWN if totals["U"] == UNKNOWN else
                              totals["U"] / totals["E"] if totals["E"] else "NOT_APPLICABLE")
        totals["games_with_injection"] = (
            UNKNOWN if totals["U"] == UNKNOWN else sum(row["U"] > 0 for row in rows)
        )
        returned = [row["response_and_submitted_action_decisions"] for row in rows]
        totals["response_and_submitted_action_decisions"] = UNKNOWN if UNKNOWN in returned else sum(returned)
        anomalies = [entry for row in rows for entry in row["anomalies"]]
        counts = Counter(entry["kind"] for entry in anomalies)
        totals["anomaly_counts"] = {**dict.fromkeys(_COVERAGE_ANOMALIES, 0), **counts}
        totals["unexpected_anomaly_count"] = sum(not entry["expected_by_assignment"] for entry in anomalies)
        arms[arm] = totals
    tom = arms[collection.TOM]
    return {
        "summary": {
            "experiment_id": collection.EXPERIMENT_ID, "source_collection_digest": final_digest,
            "collection_source_revision": plan["source_revision"], "N": len(pairs),
            "analysis_script_sha256": sha256_bytes(Path(__file__).read_bytes()),
            "definitions": {
                "decision_key": "(game_id, boundary_id, current_speaker)",
                "E": "不同存活狼人普通/PK发言PRE；两组相同结构条件，独立于treatment",
                "T": "原始treatments列表长度，重复条目保留并报告",
                "C": "predictor_requests客户端请求计数，不是全部游戏LLM调用数",
                "H": "认知请求中可完整解析的ToM payload数，包含认知重试及不匹配payload",
                "U": "合格决策点中完整canonical payload摘要及观察者行身份匹配的后端提交，按决策点去重",
                "coverage": "各组sum(U)/sum(E)，E=0为NOT_APPLICABLE；UNKNOWN不按0处理",
                "response_and_submitted_action_decisions": "匹配请求成功返回且同边界有已提交发言；不证明逐响应采用或模型内在利用",
                "expected_by_assignment": "NoToM的eligible_without_treatment是预期结果，仍逐点记录",
                "verification": "Final/pair/audit摘要链及选取的Bundle文件SHA；不重新执行游戏或完整Bundle replay",
            },
            "arms": arms,
            "historical_166_treatments_confirmed": tom["T"] == 166 if tom["T"] != UNKNOWN else UNKNOWN,
            "actual_injection_equals_166": tom["U"] == 166 if tom["U"] != UNKNOWN else UNKNOWN,
        },
        "games": games,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--coverage", action="store_true", help="离线核查接入覆盖率，不重算胜率Bootstrap")
    parser.add_argument("--game-details-output", type=Path, help="用户指定的游戏级CSV输出路径")
    args = parser.parse_args(argv)
    if args.coverage and (args.collection_root is None or args.output_dir is None or args.game_details_output is None):
        parser.error("--coverage requires --collection-root, --output-dir and --game-details-output")
    if args.game_details_output is not None and not args.coverage:
        parser.error("--game-details-output requires --coverage")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.collection_root is None:
        storage = cli._storage_root(operator.STORAGE)
        root = cli._artifact_path(
            storage, Path("gameplay_ablations") / collection.EXPERIMENT_ID
        )
    else:
        root = args.collection_root
    root = root.resolve(strict=True)
    output = (args.output_dir or root.parent / f"{collection.EXPERIMENT_ID}-analysis").resolve()
    if output == root or root in output.parents:
        raise ValueError("analysis output directory must be outside formal collection")
    details_path = args.game_details_output.resolve() if args.coverage else None
    if details_path is not None and (details_path == root or root in details_path.parents
                                     or details_path == output / "analysis.json"):
        raise ValueError("game details must be outside formal collection and distinct from summary")
    if args.coverage:
        coverage = analyze_coverage(root, expected_final_digest=EXPECTED_FINAL_DIGEST)
        result = coverage["summary"]
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(coverage["games"][0]), lineterminator="\n")
        writer.writeheader()
        for row in coverage["games"]:
            writer.writerow({**row, "anomalies": json.dumps(row["anomalies"], ensure_ascii=False, sort_keys=True)})
        details_payload = stream.getvalue().encode("utf-8")
    else:
        result = analyze_collection(root, expected_final_digest=EXPECTED_FINAL_DIGEST)
    payload = (json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    destinations = [(output / "analysis.json", payload)]
    if details_path is not None:
        destinations.append((details_path, details_payload))
    # 先检查两个输出，发现冲突时不先写出另一份结果；任何已有文件均不覆盖。
    for destination, content in destinations:
        if destination.exists() and destination.read_bytes() != content:
            raise ValueError(f"existing {destination.name} differs from reproducible result")
    for destination, content in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with destination.open("xb") as handle:
                handle.write(content)
        except FileExistsError:
            if destination.read_bytes() != content:
                raise ValueError(f"existing {destination.name} differs from reproducible result") from None
    sys.stdout.buffer.write(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
