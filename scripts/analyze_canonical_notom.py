"""从已发布的 Development 对局离线统计 Canonical-NoToM。"""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
from datetime import datetime
import json
from math import comb
from pathlib import Path
from random import Random
import sys

from scripts.analyze_tom_gameplay_ablation import _auxiliary_envelope, _game_cost
from werewolf.artifact_io import (
    ArtifactValidationError, canonical_json_bytes, open_artifact_envelope, read_artifact_file, sha256_bytes,
)
from werewolf.artifact_io.canonical import _load_canonical_json
from werewolf.canonical_collection.attempt_ledger import (
    TerminalOutcome, _decode_claim_record, _FINAL_RECORD_NAME, _load_canonical_record,
    _record_path, _state_from_records, _terminal_from_record, load_collection_plan,
)
from werewolf.canonical_collection.game_bundle import (
    _backend_call_from_record, _call_budget_from_record, _decode_jsonl, _private_replay_from_record,
)
from werewolf.canonical_collection.production_runtime import build_classic7_replay_environment
from werewolf.canonical_collection.public_history import PLAYER_IDS, freeze_public_event_history
from werewolf.canonical_collection.runtime import replay_classic7_actions
from werewolf.development_publication import _collection_identity, open_role_sidecar
from werewolf.development_publication_v2 import PUBLICATION_TYPE, PUBLICATION_VERSION
from werewolf.game_runtime import (
    UNKNOWN, _require_legal_gameplay_action, count_game_votes, gameplay_ratio, summarize_game_votes,
)
from werewolf.interventions.gameplay import safe_path, write_once
from werewolf.tom.development_input import open_development_input


SAMPLE_SEED = 20261128
SAMPLE_SIZE = 210
_REPLAY_FILE = "private/private_replay_evidence.json"
_CALL_FILE = "private/backend_call_evidence.jsonl"
_BUDGET_FILE = "audit/call_budget_summary.json"
_EVIDENCE_ERRORS = (OSError, ValueError, KeyError, TypeError)


def _read_ledger(collection, plan):
    """核对原 Claim/Terminal 身份。这里不读取失败载荷或任何 Test Bundle。"""
    directory = safe_path(collection / "attempt_ledger")
    claims, terminals = [], []
    for path in directory.iterdir():
        if path.name == ".staging":
            continue
        match = _FINAL_RECORD_NAME.fullmatch(path.name)
        if match is None or path.is_symlink() or not path.is_file():
            raise ValueError(f"invalid Attempt Ledger entry: {path.name}")
        value = _load_canonical_record(path)
        if match.group("record_type") == "claim":
            record = _decode_claim_record(value)
            claims.append(record)
        else:
            record = _terminal_from_record(value, plan)
            terminals.append(record)
        if path != _record_path(directory, record):
            raise ValueError("Attempt Ledger filename differs from its record")
    return _state_from_records(plan, claims, terminals)


def _native_outcome(bundle, member, roles):
    """用原 Seed 和已提交动作回放，核对存档赢家及公开轨迹摘要。"""
    private = _private_replay_from_record(_load_canonical_json(read_artifact_file(bundle, _REPLAY_FILE)))
    if (private.game_id != member["game_id"] or private.seed != member["seed"]
            or dict(private.role_assignment) != roles
            or private.initial_runtime_state.to_value() != {"day": 0, "phase": "night"}):
        raise ValueError("replay input differs from the original Development game")
    config = private.runtime_configuration.to_value()
    if sha256_bytes(canonical_json_bytes(config)) != bundle.manifest["runtime_configuration_digest"]:
        raise ValueError("replay runtime configuration differs from the original Bundle")
    profiles = config["agent_config"]["all_candidates"]
    if any(profile["agent_type"] != "gpt" or "decision_method" in profile["model_params"] for profile in profiles):
        raise ValueError("archived agent config is outside the original Canonical GPTAgent path")
    replay = private.replay_inputs.to_value()
    env = build_classic7_replay_environment(private.seed, config["env_config"])
    events, winner = replay_classic7_actions(
        env, [roles[seat] for seat in PLAYER_IDS], replay["submitted_actions"],
        require_full_game=True, validate_action=_require_legal_gameplay_action,
    )
    history = freeze_public_event_history(events)
    if winner != replay["winner"] or history.digest != private.expected_public_event_digest:
        raise ValueError("replay disagrees with the archived native outcome or public history")

    wolves = {seat for seat, role in roles.items() if role == "Werewolf"}
    night = {seat for event in history.events if event.event_type == "death_announcement"
             for seat in event.affected_players if seat in wolves}
    poisoned = {PLAYER_IDS[seat] for seat in env.witch_poison_target.values()}
    deaths = {
        "exile": sum(len(wolves.intersection(event.affected_players))
                     for event in history.events if event.event_type == "exile_result"),
        "witch_poison": len(night & poisoned),
        "unattributed_night": len(night - poisoned),
    }
    return winner, history, deaths, config


def _archived_cost(bundle, files, claim, terminal):
    """保留已审计调用数。Token 与服务耗时只按实际请求回执计算。"""
    budget = None
    calls = None
    missing = []
    if _BUDGET_FILE in files:
        try:
            budget = _call_budget_from_record(_load_canonical_json(read_artifact_file(bundle, _BUDGET_FILE)))
        except _EVIDENCE_ERRORS:
            missing.append("invalid_call_budget")
    else:
        missing.append("missing_call_budget")
    cost = _game_cost(None)
    if _CALL_FILE in files:
        try:
            raw = read_artifact_file(bundle, _CALL_FILE)
            # 无传输回执且已知派发数时，免解码私有 Prompt 和响应。文件字节仍核对冻结摘要。
            if budget is None or b'"request_record":' in raw:
                calls = [_backend_call_from_record(record) for record in _decode_jsonl(raw, _CALL_FILE)]
                if budget is not None and tuple(call.call_digest for call in calls) != budget.opaque_call_digests:
                    raise ArtifactValidationError("call records disagree with the call budget")
                cost = _game_cost(calls)
            if cost["model_requests"] == UNKNOWN:
                missing.append("incomplete_transport_receipts")
        except _EVIDENCE_ERRORS:
            calls = None
            missing.append("invalid_backend_receipts")
    else:
        missing.append("missing_backend_receipts")
    if budget is not None:
        cost["model_requests"] = budget.used_calls
    elif calls is not None:
        cost["model_requests"] = len(calls)
    start = datetime.fromisoformat(claim.claim_timestamp_utc.replace("Z", "+00:00"))
    end = datetime.fromisoformat(terminal.terminal_timestamp_utc.replace("Z", "+00:00"))
    elapsed = (end - start).total_seconds()
    cost["attempt_wall_seconds"] = elapsed if elapsed >= 0 else UNKNOWN
    return cost, missing


def _population_summary(games):
    total = len(games)
    wins = sum(game["winner"] == "Werewolf" for game in games)
    votes = summarize_game_votes([game["votes"] for game in games])
    votes["WER"] = votes["wolf_exile_rate"]
    deaths = {name: sum(game["deaths"][name] for game in games)
              for name in ("exile", "witch_poison", "unattributed_night")}
    costs = {}
    for name in (*_game_cost(None), "attempt_wall_seconds"):
        known = [game["cost"][name] for game in games if game["cost"][name] != UNKNOWN]
        costs[name] = {"sum": sum(known) if len(known) == total else UNKNOWN,
                       "observed_sum": sum(known), "games_available": len(known), "games": total}
    return {"games": total, "wolf_wins": wins, "villager_wins": total - wins,
            "WWR": gameplay_ratio(wins, total), "votes": votes, "wolf_deaths": deaths, "cost": costs,
            "missing_metric_evidence": dict(Counter(reason for game in games for reason in game["missing"]))}


def _wwr_interval(wins, sample_size, population_size):
    """反演超几何双尾检验。区间估计冻结有限人口的狼人胜率。"""
    possible = range(wins, population_size - sample_size + wins + 1)
    denominator = comb(population_size, sample_size)

    def tail(total_wins, start, stop):
        return sum(comb(total_wins, x) * comb(population_size - total_wins, sample_size - x)
                   for x in range(start, stop))

    # 每侧显著性 1/40，用整数运算保留尾部边界。
    lower = bisect_left(possible, denominator, key=lambda k: 40 * tail(k, wins, sample_size + 1))
    upper = bisect_right(possible, -denominator, key=lambda k: -40 * tail(k, 0, wins + 1)) - 1
    counts = [possible[lower], possible[upper]]
    return {"confidence": 0.95, "method": "exact_hypergeometric", "target": "finite_valid_development_population",
            "population_games": population_size, "wolf_win_count_bounds": counts,
            "bounds": [count / population_size for count in counts]}


def analyze(collection_path, publication_path, *, publication_digest, split_digest, sample_output, sample_size=SAMPLE_SIZE):
    collection_path, publication_path, sample_output = map(safe_path, (collection_path, publication_path, sample_output))
    if any(sample_output.is_relative_to(path) for path in (collection_path.parent, publication_path.parent)):
        raise ValueError("sample output must be outside the Canonical and Publication archive directories")
    envelope = open_artifact_envelope(publication_path, expected_artifact_type=PUBLICATION_TYPE,
                                      expected_schema_version=PUBLICATION_VERSION)
    if envelope.manifest_digest != publication_digest:
        raise ArtifactValidationError("Development Publication differs from the trusted digest")
    members = envelope.manifest["development_games"]
    # 现有 partition reader 核对完整成员、Fold 与 Sidecar；对局核验逐个读取原 Bundle。
    publication = open_development_input(publication_path, expected_publication_digest=publication_digest,
                                         expected_split_digest=split_digest, game_ids=(members[0]["game_id"],))
    roles = {game.game_id: dict(game.role_assignment) for game in open_role_sidecar(publication).games}
    plan = load_collection_plan(collection_path)
    if (plan.collection_id != publication.public_view.collection_id
            or plan.plan_digest != publication.public_view.collection_plan_digest
            or plan.source_revision != publication.manifest["source_revision"]):
        raise ArtifactValidationError("Collection Plan differs from the Development source")
    ledger = _read_ledger(collection_path, plan)
    if (not ledger.target_reached or ledger.open_claim is not None
            or _collection_identity(plan, ledger) != publication.public_view.collection_identity_digest):
        raise ArtifactValidationError("Attempt Ledger differs from the closed Development source")

    folds = publication.public_view.fold_manifest.folds
    fold_by_game = {game_id: fold.fold_index for fold in folds for game_id in fold.game_ids}
    terminals = {item.ordinal: item for item in ledger.terminals}
    eligible, excluded = {}, Counter()
    configurations = {}
    for member in members:
        game_id, ordinal = member["game_id"], member["ordinal"]
        claim, terminal = ledger.claims[ordinal], terminals[ordinal]
        expected = {"game_id": game_id, "seed": claim.seed, "ordinal": ordinal, "attempt_id": claim.attempt_id,
                    "claim_record_digest": claim.record_digest, "collection_id": plan.collection_id,
                    "collection_plan_digest": plan.plan_digest, "runtime_provenance_digest": plan.runtime_provenance_digest,
                    "source_revision": plan.source_revision, "canonical_eligibility": True}
        if (terminal.outcome is not TerminalOutcome.CANONICAL_SUCCESS or terminal.canonical_game_bundle_id != game_id
                or terminal.canonical_game_bundle_digest != member["bundle_digest"] or claim.seed != member["seed"]):
            raise ArtifactValidationError("Development member differs from the completed Attempt Ledger")
        try:
            bundle, files = _auxiliary_envelope(safe_path(collection_path / "games" / game_id),
                digest=member["bundle_digest"], artifact_type="canonical_game_bundle", schema_version=plan.bundle_schema_version)
        except _EVIDENCE_ERRORS:
            excluded["invalid_bundle_manifest_or_inventory"] += 1
            continue
        if bundle is None:
            excluded["missing_bundle_manifest"] += 1
            continue
        if any(bundle.manifest.get(key) != value for key, value in expected.items()):
            excluded["bundle_identity_mismatch"] += 1
            continue
        if _REPLAY_FILE not in files:
            excluded["missing_native_replay_evidence"] += 1
            continue
        try:
            winner, history, deaths, config = _native_outcome(bundle, member, roles[game_id])
        except (*_EVIDENCE_ERRORS, AssertionError, IndexError):
            excluded["native_replay_failed"] += 1
            continue

        cost, missing = _archived_cost(bundle, files, claim, terminal)
        try:
            votes = count_game_votes(history, roles[game_id])
        except _EVIDENCE_ERRORS:
            votes = UNKNOWN
            missing.append("invalid_vote_evidence")
        key = canonical_json_bytes(config)
        if key not in configurations:
            configurations[key] = {"agents": config["agent_config"]["all_candidates"], "parser": config["parser"],
                                   "env_config": config["env_config"], "games": 0}
        configurations[key]["games"] += 1
        eligible[game_id] = {"winner": winner, "votes": votes, "deaths": deaths, "cost": cost, "missing": missing,
                             "identity": {"game_id": game_id, "seed": claim.seed, "roles": roles[game_id],
                                          "source_revision": plan.source_revision, "bundle_digest": bundle.manifest_digest,
                                          "collection_id": plan.collection_id, "fold": fold_by_game[game_id]}}

    outcomes = Counter(terminal.outcome.value for terminal in ledger.terminals)
    provenance = dict(plan.environment_provenance)
    report = {
        "method": "Canonical-NoToM", "scope": "development_only",
        "source": {"collection_id": plan.collection_id, "source_revision": plan.source_revision,
                   "collection_plan_digest": plan.plan_digest, "publication_digest": publication.manifest_digest,
                   "split_digest": split_digest, "fold_manifest_digest": publication.public_view.fold_manifest.manifest_digest,
                   "fold_sizes": {fold.fold_index: len(fold.game_ids) for fold in folds},
                   "identities": {name: getattr(plan, name) for name in ("runtime_identity", "agent_identity", "model_identity",
                                  "backend_identity", "prompt_identity", "parser_identity", "retry_policy_identity")},
                   "deployment": {name: provenance.get(name, UNKNOWN) for name in ("hf_revision", "model_manifest_sha256",
                                  "runtime_config_sha256", "serve_config_sha256", "served_model_name", "model_path")},
                   "archived_runtime_configurations": list(configurations.values())},
        "coverage": {"claimed_attempts": len(ledger.claims), "terminal_outcomes": dict(outcomes),
                     "unknown_open_attempts": len(ledger.claims) - len(ledger.terminals),
                     "sealed_test_games": publication.split_header["test_count"], "published_development_games": len(members),
                     "excluded_development_games": dict(excluded)},
        "all_valid_games": _population_summary(list(eligible.values())),
    }
    if len(eligible) < sample_size:
        report["sample"] = {"status": "insufficient_valid_games", "required": sample_size, "available": len(eligible)}
        return report

    population_ids = sorted(eligible)
    sampled_ids = Random(SAMPLE_SEED).sample(population_ids, sample_size)
    sample = [eligible[game_id] for game_id in sampled_ids]
    sample_record = {"method": report["method"], "sampling_seed": SAMPLE_SEED, "size": sample_size,
                     "population_games": len(eligible), "publication_digest": publication_digest,
                     "population_ids_sha256": sha256_bytes(canonical_json_bytes(population_ids)),
                     "collection_plan_digest": plan.plan_digest, "split_digest": split_digest,
                     "games": [game["identity"] for game in sample]}
    write_once(sample_output, sample_record)
    sample_report = _population_summary(sample)
    report["sample"] = {**sample_report, "sampling_seed": SAMPLE_SEED, "sample_list": str(sample_output),
                        "WWR_95_CI": _wwr_interval(sample_report["wolf_wins"], sample_size, len(eligible))}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--publication-digest", required=True)
    parser.add_argument("--split-digest", required=True)
    parser.add_argument("--sample-output", type=Path, required=True, help="独立于冻结归档的新抽样名单路径")
    args = parser.parse_args(argv)
    try:
        report = analyze(args.collection, args.publication, publication_digest=args.publication_digest,
                         split_digest=args.split_digest, sample_output=args.sample_output)
    except _EVIDENCE_ERRORS as error:
        print(f"Canonical-NoToM: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    return int(report["sample"].get("status") == "insufficient_valid_games")


if __name__ == "__main__":
    sys.exit(main())
