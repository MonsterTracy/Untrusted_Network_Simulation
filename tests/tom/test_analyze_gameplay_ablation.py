"""Focused read-only analysis tests against the formal artifact schema."""

import hashlib
import csv
import json
import shutil
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts import analyze_tom_gameplay_ablation as analysis
from scripts import tom_gameplay_collection as collection
from tests.canonical_collection.test_game_bundle import _fixture
from tests.tom.test_gameplay_ablation import _matrix, _wolf_observation
from werewolf.agents.prompt_template_v0 import (
    build_day_cognition_prompt, build_public_claim_catalog, freeze_discussion_candidates,
)
from werewolf.artifact_io import (
    ArtifactValidationError, canonical_json_bytes, canonical_jsonl_bytes,
    publish_artifact, sha256_bytes,
)
from werewolf.canonical_collection.attempt_ledger import construct_attempt_claim
from werewolf.canonical_collection.game_bundle import _bundle_files
from werewolf.canonical_collection.trajectory_evidence import (
    BackendCallPurpose, BackendCallStatus, construct_backend_call_evidence,
)
from scripts.tom_gameplay_ablation import tom_payload


def _arm(plan, ordinal, arm, *, winner="Villager", treatment_count=0,
         failure=False):
    seed = plan["ordered_seed_pool"][ordinal]
    arm_plan = collection._arm_plan(plan, ordinal, arm)
    result = {
        "arm": arm,
        "status": "failure" if failure else "success",
        "collection_id": arm_plan.collection_id,
        "plan_digest": arm_plan.plan_digest,
        "claim_digest": "a" * 64,
        "terminal_digest": "b" * 64,
        "terminal_outcome": "canonical_failure" if failure else "canonical_success",
        "game_id": f"{arm_plan.collection_id}-game-000000-seed-{seed}",
        "bundle_digest": None if failure else "c" * 64,
        "winner": None if failure else winner,
        "roles": None if failure else {"player1": "Werewolf", "player2": "Werewolf"},
        "backend_calls": None if failure else 12,
        "speech_boundaries": None if failure else 4,
        "treatment_count": None if failure else treatment_count,
        "predictor_fit_digest": collection.FIT_DIGEST if arm == collection.TOM else None,
        "predictor_seal_digest": collection.SEAL_DIGEST if arm == collection.TOM else None,
        "source_revision": arm_plan.source_revision,
        "runtime_provenance_digest": arm_plan.runtime_provenance_digest,
        "treatment_audit_digest": None if failure else "d" * 64,
        "failure_stage": "runtime" if failure else None,
        "failure_type": "GameplayGenerationExhausted" if failure else None,
    }
    return result


def _formal_collection(root, *, coverage=False):
    plan = collection._experiment_record(
        "f" * 40,
        {"served_model_name": "fixture-model", "runtime_config_sha256": "e" * 64},
        32,
    )
    plan_digest = collection._publish_record(
        root / "experiment_plan", "tom_gameplay_experiment_plan", plan
    )
    included = []
    invalid = []
    references = []
    valid_index = 0
    for ordinal in range(42):
        seed = plan["ordered_seed_pool"][ordinal]
        order = collection._order(ordinal)
        if ordinal in (14, 15):
            statuses = {
                order[0]: _arm(plan, ordinal, order[0], failure=True),
                order[1]: None,
            }
        else:
            category = valid_index % 4
            notom_win = category in (2, 3)
            tom_win = category in (1, 3)
            statuses = {
                collection.NOTOM: _arm(
                    plan, ordinal, collection.NOTOM,
                    winner="Werewolf" if notom_win else "Villager",
                ),
                collection.TOM: _arm(
                    plan, ordinal, collection.TOM,
                    winner="Werewolf" if tom_win else "Villager",
                    treatment_count=0 if valid_index == 0 else 2,
                ),
            }
            valid_index += 1
            if coverage:
                for arm in collection.ARMS:
                    _coverage_artifacts(root, plan, ordinal, arm, statuses[arm])
        pair = collection._pair_record(plan, ordinal, order, statuses)
        pair_digest = collection._publish_record(
            collection._pair_path(root, ordinal), "tom_gameplay_pair", pair
        )
        references.append({
            "candidate_ordinal": ordinal, "seed": seed, "pair_digest": pair_digest,
        })
        if pair["status"] == "completed_valid":
            included.append({"candidate_ordinal": ordinal, "seed": seed})
        else:
            invalid.append({
                "candidate_ordinal": ordinal, "seed": seed,
                "failure_arm": pair["failure_arm"],
                "failure_stage": pair["failure_stage"],
                "failure_type": pair["failure_type"],
                "pair_digest": pair_digest,
            })
    final = {
        "experiment_id": collection.EXPERIMENT_ID,
        "source_revision": plan["source_revision"],
        "runtime_provenance": plan["runtime_provenance"],
        "experiment_plan_digest": plan_digest,
        "candidate_seed_pool_digest": plan["candidate_seed_pool_digest"],
        "attempted_candidate_count": 42,
        "completed_valid_count": 40,
        "included_pairs": included,
        "invalid_pairs": invalid,
        "failure_reason_counts": {"runtime:GameplayGenerationExhausted": 2},
        "pair_record_digests": references,
    }
    digest = collection._publish_record(root / "final", "tom_gameplay_final", final)
    return root, digest


@pytest.fixture(scope="module")
def formal(tmp_path_factory):
    return _formal_collection(tmp_path_factory.mktemp("formal-gameplay-analysis") / "collection")


@pytest.fixture(scope="module")
def result(formal):
    root, digest = formal
    return analysis.analyze_collection(root, expected_final_digest=digest)


def _copy_formal(formal, tmp_path):
    root, _ = formal
    destination = tmp_path / "collection"
    shutil.copytree(root, destination)
    return destination


def _replace(root, path, kind, record):
    shutil.rmtree(path)
    return collection._publish_record(path, kind, record)


def _tree_hashes(root):
    return {
        path.relative_to(root): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file()
    }


def test_primary_uses_only_complete_paired_seeds_and_zero_treatment(result):
    assert result["experiment_id"] == collection.EXPERIMENT_ID
    assert result["N"] == 40
    assert result["attempted_candidates"] == 42
    assert result["invalid_candidate_count"] == 2
    assert [(item["candidate_ordinal"], item["failure_arm"], item["failure_type"])
            for item in result["invalid_candidates"]] == [
        (14, collection.NOTOM, "GameplayGenerationExhausted"),
        (15, collection.TOM, "GameplayGenerationExhausted"),
    ]
    paired = result["paired"]
    assert (paired["n00"], paired["n01"], paired["n10"], paired["n11"]) == (10, 10, 10, 10)
    assert sum(paired[key] for key in ("n00", "n01", "n10", "n11")) == result["N"]
    assert paired["delta_hat"] == (paired["n01"] - paired["n10"]) / result["N"]
    assert result["treatment"]["zero_treatment_games"] == 1
    assert result["treatment"]["positive_treatment_games"] == 39
    assert result["arms"][collection.NOTOM]["wins"] == 20
    assert result["arms"][collection.TOM]["wins"] == 20


def test_exact_mcnemar_small_samples():
    assert analysis._exact_mcnemar(0, 0) == 1.0
    assert analysis._exact_mcnemar(0, 4) == 0.125
    assert analysis._exact_mcnemar(1, 3) == 0.625
    assert analysis._exact_mcnemar(3, 1) == 0.625


def test_paired_bootstrap_is_seeded_and_resamples_whole_effects(result):
    effects = [-1, 0, 1, 1]
    first = analysis._paired_bootstrap(effects, seed=20260928, replicates=1000)
    assert first == analysis._paired_bootstrap(effects, seed=20260928, replicates=1000)
    assert first[0] <= sum(effects) / len(effects) <= first[1]
    assert result["analysis"] == {
        "bootstrap_seed": 20260928, "bootstrap_replicates": 100000,
    }
    assert len(result["paired"]["bootstrap_95_ci"]) == 2


def test_analysis_is_read_only_and_output_schema_is_json(formal, result):
    root, digest = formal
    before = _tree_hashes(root)
    again = analysis.analyze_collection(root, expected_final_digest=digest)
    assert again == result
    assert _tree_hashes(root) == before
    round_trip = json.loads(json.dumps(result, allow_nan=False))
    assert round_trip == result
    assert set(result) == {
        "experiment_id", "source_collection_digest", "N",
        "attempted_candidates", "invalid_candidate_count", "invalid_candidates",
        "arms", "paired", "treatment", "analysis",
    }
    assert result["source_collection_digest"] == digest


def test_final_digest_and_completion_fail_closed(formal, tmp_path):
    root, digest = formal
    with pytest.raises(ValueError, match="digest mismatch"):
        analysis.analyze_collection(root, expected_final_digest="0" * 64)
    clone = _copy_formal(formal, tmp_path)
    final, _ = collection._read_record(clone / "final", "tom_gameplay_final")
    final["completed_valid_count"] = 39
    changed = _replace(clone, clone / "final", "tom_gameplay_final", final)
    with pytest.raises(ValueError, match="incomplete"):
        analysis.analyze_collection(clone, expected_final_digest=changed)
    assert digest != changed


def test_seed_mismatch_and_duplicate_references_fail_closed(formal, tmp_path):
    clone = _copy_formal(formal, tmp_path)
    pair_path = collection._pair_path(clone, 0)
    pair, _ = collection._read_record(pair_path, "tom_gameplay_pair")
    pair["seed"] += 1
    pair_digest = _replace(clone, pair_path, "tom_gameplay_pair", pair)
    final, _ = collection._read_record(clone / "final", "tom_gameplay_final")
    final["pair_record_digests"][0]["pair_digest"] = pair_digest
    final_digest = _replace(clone, clone / "final", "tom_gameplay_final", final)
    with pytest.raises(ValueError, match="seed"):
        analysis.analyze_collection(clone, expected_final_digest=final_digest)

    clone = _copy_formal(formal, tmp_path / "second")
    final, _ = collection._read_record(clone / "final", "tom_gameplay_final")
    final["pair_record_digests"][1]["seed"] = final["pair_record_digests"][0]["seed"]
    final_digest = _replace(clone, clone / "final", "tom_gameplay_final", final)
    with pytest.raises(ValueError, match="duplicate or reordered"):
        analysis.analyze_collection(clone, expected_final_digest=final_digest)

    clone = _copy_formal(formal, tmp_path / "third")
    final, _ = collection._read_record(clone / "final", "tom_gameplay_final")
    final["pair_record_digests"][1]["candidate_ordinal"] = 0
    final_digest = _replace(clone, clone / "final", "tom_gameplay_final", final)
    with pytest.raises(ValueError, match="duplicate or reordered"):
        analysis.analyze_collection(clone, expected_final_digest=final_digest)


def test_roles_and_arm_identity_fail_closed(formal, tmp_path):
    clone = _copy_formal(formal, tmp_path)
    pair_path = collection._pair_path(clone, 0)
    pair, _ = collection._read_record(pair_path, "tom_gameplay_pair")
    pair["arms"][collection.TOM]["roles"]["player1"] = "Villager"
    pair_digest = _replace(clone, pair_path, "tom_gameplay_pair", pair)
    final, _ = collection._read_record(clone / "final", "tom_gameplay_final")
    final["pair_record_digests"][0]["pair_digest"] = pair_digest
    final_digest = _replace(clone, clone / "final", "tom_gameplay_final", final)
    with pytest.raises(ValueError, match="paired-role"):
        analysis.analyze_collection(clone, expected_final_digest=final_digest)

    clone = _copy_formal(formal, tmp_path / "other-seed")
    pair_path = collection._pair_path(clone, 0)
    pair, _ = collection._read_record(pair_path, "tom_gameplay_pair")
    pair["arms"][collection.TOM]["game_id"] = pair["arms"][collection.TOM][
        "game_id"
    ].replace(str(pair["seed"]), str(pair["seed"] + 1))
    pair_digest = _replace(clone, pair_path, "tom_gameplay_pair", pair)
    final, _ = collection._read_record(clone / "final", "tom_gameplay_final")
    final["pair_record_digests"][0]["pair_digest"] = pair_digest
    final_digest = _replace(clone, clone / "final", "tom_gameplay_final", final)
    with pytest.raises(ValueError, match="canonical-success arm"):
        analysis.analyze_collection(clone, expected_final_digest=final_digest)


def test_main_writes_only_sibling_analysis_dir(formal, tmp_path, monkeypatch, capsys):
    root, digest = formal
    output = tmp_path / "independent-analysis"
    before = _tree_hashes(root)
    monkeypatch.setattr(analysis, "EXPECTED_FINAL_DIGEST", digest)
    assert analysis.main([
        "--collection-root", str(root), "--output-dir", str(output),
    ]) == 0
    payload = json.loads((output / "analysis.json").read_text())
    assert payload["N"] == 40
    assert json.loads(capsys.readouterr().out) == payload
    assert _tree_hashes(root) == before
    assert analysis.main([
        "--collection-root", str(root), "--output-dir", str(output),
    ]) == 0
    assert _tree_hashes(root) == before
    with pytest.raises(ValueError, match="outside formal collection"):
        analysis.main([
            "--collection-root", str(root),
            "--output-dir", str(root / "analysis"),
        ])


def _coverage_case(plan=None, claim=None, *, game_id="game-000"):
    """复用真实 PRE 和认知模板，所有调用均为内存中的封存格式样例。"""
    plan, claim, evidence, _ = _fixture(plan, claim, game_id=game_id)
    prefix = evidence.authoritative_pre_prefixes[0]
    observation = _wolf_observation()
    payload = tom_payload(prefix, observation, _matrix())
    prompt_args = {
        "candidate_snapshot": freeze_discussion_candidates(observation),
        "claim_catalog": build_public_claim_catalog(observation),
        "pre_speech_belief": evidence.submitted_gameplay_actions[0].speaker_pre_belief_handoff,
    }

    def call(index=1, *, value=payload, status="success", positional=False):
        prompt = build_day_cognition_prompt(
            observation, **prompt_args,
            tom_context=None if value is None else canonical_json_bytes(value).decode(),
        )
        messages = [{"role": "user", "content": prompt}]
        return construct_backend_call_evidence(
            call_id=f"cognition-{index}", operation_id=f"dispatch-{index}",
            purpose=BackendCallPurpose.SPEECH_GENERATION,
            status=BackendCallStatus(status), boundary_id=prefix.boundary_id,
            observer_id=prefix.current_speaker, attempt_index=1,
            **{key: getattr(plan, key) for key in (
                "backend_identity", "model_identity", "parser_identity",
                "prompt_identity", "retry_policy_identity", "call_budget_identity",
            )},
            private_payload={
                "method": "chat_with_metadata",
                "args": [messages] if positional else [],
                "kwargs": {} if positional else {"messages": messages},
                "response": ["{}", {"finish_reason": "stop"}] if status == "success" else None,
            },
        )

    entry = {
        "game_id": game_id, "boundary_id": prefix.boundary_id,
        "prefix_digest": prefix.prefix_digest,
        "observer_ids": [row["observer_id"] for row in payload["rows"]],
        "payload_digest": sha256_bytes(canonical_json_bytes(payload)),
        "fit_digest": collection.FIT_DIGEST, "seal_digest": collection.SEAL_DIGEST,
    }
    return SimpleNamespace(
        plan=plan, claim=claim, evidence=evidence, prefix=prefix, payload=payload,
        roles=dict(evidence.private_replay_evidence.role_assignment),
        audit={"treatments": [entry], "treatment_count": 1, "predictor_requests": 1},
        call=call,
    )


def _coverage_artifacts(root, record, ordinal, arm, status):
    """构造 80 局 Artifact 读取闭环；不生成游戏或调用模型。"""
    plan = collection._arm_plan(record, ordinal, arm)
    claim = construct_attempt_claim(
        plan, ordinal=0, attempt_id=f"coverage-claim-{ordinal}",
        claim_timestamp_utc="2026-09-27T00:00:00Z",
    )
    case = _coverage_case(plan, claim, game_id=status["game_id"])
    treated = arm == collection.TOM and status["treatment_count"] > 0
    calls = [case.call(1, value=case.payload if treated else None)]
    if treated:
        calls = [case.call(1, status="error"), case.call(2)]
    entries = case.audit["treatments"] if treated else []
    base = collection._arm_path(root, ordinal, arm)
    files = _bundle_files(case.evidence)
    files["private/backend_call_evidence.jsonl"] = canonical_jsonl_bytes(call.to_record() for call in calls)
    bundle = publish_artifact(
        base / "games" / status["game_id"],
        manifest_fields={
            "artifact_type": "canonical_game_bundle",
            "schema_version": analysis.CANONICAL_GAME_BUNDLE_SCHEMA_VERSION,
            "artifact_identity": status["game_id"], "game_id": status["game_id"],
            "source_revision": record["source_revision"], "seed": claim.seed,
            "collection_id": plan.collection_id, "collection_plan_digest": plan.plan_digest,
            "claim_record_digest": claim.record_digest,
            "runtime_provenance_digest": plan.runtime_provenance_digest,
            "canonical_eligibility": True,
        },
        files=files,
    )
    audit = collection._audit_record(
        record=record, ordinal=ordinal, seed=claim.seed, arm=arm, plan=plan,
        claim=claim, evidence=case.evidence, entries=entries,
        request_count=(len(entries) if arm == collection.TOM else None),
    )
    audit_digest = collection._publish_record(
        base / "private_treatment_audit", "tom_gameplay_treatment_audit", audit,
    )
    status.update(
        claim_digest=claim.record_digest, bundle_digest=bundle.manifest_digest,
        treatment_audit_digest=audit_digest, roles=case.roles,
        treatment_count=len(entries), speech_boundaries=1, backend_calls=len(calls),
    )


def _count_case(case, *, prefixes=None, roles=None, audit=None, calls=None,
                actions=None, arm=collection.TOM):
    return analysis._coverage_counts(
        [case.prefix] if prefixes is None else prefixes,
        case.roles if roles is None else roles,
        case.audit if audit is None else audit,
        [case.call()] if calls is None else calls,
        [action.to_record() for action in case.evidence.submitted_gameplay_actions]
        if actions is None else actions,
        game_id=case.prefix.game_id, arm=arm, reported_treatments=1,
    )


def test_coverage_matches_full_payload_and_deduplicates_cognition_retries():
    case = _coverage_case()
    result = _count_case(case, calls=[case.call(1, status="error"), case.call(2, positional=True)])
    assert {key: result[key] for key in ("E", "T", "C", "H", "U")} == {
        "E": 1, "T": 1, "C": 1, "H": 2, "U": 1,
    }
    assert result["coverage"] == 1.0
    assert result["response_and_submitted_action_decisions"] == 1
    assert result["anomalies"] == []
    # 注入是后端提交，不要求模型已正常返回，更不等于证明认知采用。
    failed = _count_case(case, calls=[case.call(status="error")], actions=[])
    assert failed["U"] == 1
    assert failed["response_and_submitted_action_decisions"] == 0


def test_coverage_opportunities_are_independent_of_treatment_and_arm():
    case = _coverage_case()
    prefixes = [case.prefix]
    for boundary, phase, speaker, alive in (
        ("wolf-pk", "pk_discussion", "player2", ("player1", "player2", "player3")),
        ("villager-speech", "discussion", "player3", ("player1", "player2", "player3")),
        ("dead-wolf", "discussion", "player2", ("player1", "player3")),
        ("wolf-vote", "vote", "player1", ("player1", "player2", "player3")),
    ):
        prefixes.append(SimpleNamespace(
            game_id=case.prefix.game_id, boundary_id=boundary,
            current_speaker=speaker, alive_observer_ids=alive,
            public_temporal_state=SimpleNamespace(phase=SimpleNamespace(value=phase)),
        ))
    empty = {"treatments": [], "treatment_count": 0, "predictor_requests": None}
    result = _count_case(case, prefixes=prefixes, audit=empty,
                         calls=[case.call(value=None)], arm=collection.NOTOM)
    assert result["E"] == 2
    assert result["U"] == result["T"] == result["C"] == result["H"] == 0
    assert result["coverage"] == 0.0
    missing = [entry for entry in result["anomalies"] if entry["kind"] == "eligible_without_treatment"]
    assert {entry["boundary_id"] for entry in missing} == {case.prefix.boundary_id, "wolf-pk"}
    assert all(entry["expected_by_assignment"] for entry in missing)


@pytest.mark.parametrize("change,kind", (
    ("missing_request", "treatment_without_matching_request"),
    ("wrong_payload", "treatment_payload_mismatch"),
    ("wrong_role", "treatment_outside_eligible"),
    ("duplicate", "duplicate_treatment"),
    ("wrong_prefix", "treatment_identity_mismatch"),
    ("wrong_observer_rows", "treatment_payload_mismatch"),
))
def test_coverage_reports_anomalies_without_removing_games(change, kind):
    case = _coverage_case()
    audit = deepcopy(case.audit)
    calls = [case.call()]
    roles = dict(case.roles)
    if change == "missing_request":
        calls = [case.call(value=None)]
    elif change == "wrong_payload":
        changed = deepcopy(case.payload)
        changed["rows"][0]["probabilities"][0] += 0.01
        calls = [case.call(value=changed)]
    elif change == "wrong_role":
        roles["player1"] = "Villager"
    elif change == "duplicate":
        audit["treatments"].append(deepcopy(audit["treatments"][0]))
        audit["treatment_count"] = audit["predictor_requests"] = 2
    elif change == "wrong_prefix":
        audit["treatments"][0]["prefix_digest"] = "0" * 64
    elif change == "wrong_observer_rows":
        audit["treatments"][0]["observer_ids"] = ["player7"]
    result = _count_case(case, audit=audit, calls=calls, roles=roles)
    assert kind in {entry["kind"] for entry in result["anomalies"]}
    assert all(entry["game_id"] == case.prefix.game_id for entry in result["anomalies"])
    assert result["T"] == (2 if change == "duplicate" else 1)
    assert result["U"] == (1 if change == "duplicate" else 0)


def test_coverage_duplicate_with_different_payload_keeps_orphan_entry():
    case = _coverage_case()
    audit = deepcopy(case.audit)
    audit["treatments"].append({**audit["treatments"][0], "payload_digest": "0" * 64})
    result = _count_case(case, audit=audit)
    assert result["U"] == 1
    assert any(entry["kind"] == "treatment_without_matching_request" and entry["audit_index"] == 1
               for entry in result["anomalies"])


def test_coverage_missing_predictor_counter_is_unknown_not_inferred_from_treatments():
    case = _coverage_case()
    audit = deepcopy(case.audit)
    del audit["predictor_requests"]
    result = _count_case(case, audit=audit)
    assert result["C"] == "UNKNOWN"
    assert result["U"] == result["T"] == 1


def test_coverage_unparseable_payload_is_unknown_and_not_a_zero_injection():
    case = _coverage_case()
    record = case.call().to_record()
    prompt = record["private_payload"]["kwargs"]["messages"][0]["content"]
    record["private_payload"]["kwargs"]["messages"][0]["content"] = prompt.replace(
        canonical_json_bytes(case.payload).decode(), "{broken-json}",
    )
    # 通过原构造器重新生成 digest，表示日志原本就封存了不能解析的请求。
    call = construct_backend_call_evidence(
        **{key: getattr(case.call(), key) for key in (
            "call_id", "operation_id", "purpose", "status", "boundary_id", "observer_id",
            "attempt_index", "backend_identity", "model_identity", "parser_identity",
            "prompt_identity", "retry_policy_identity", "call_budget_identity",
        )}, private_payload=record["private_payload"],
    )
    result = _count_case(case, calls=[call])
    assert result["H"] == result["U"] == "UNKNOWN"
    assert result["U_verified"] == 0
    assert "request_payload_unverifiable" in {entry["kind"] for entry in result["anomalies"]}


def test_coverage_does_not_treat_quoted_public_header_or_response_as_injection():
    case = _coverage_case()
    record = case.call(value=None).to_record()
    request = record["private_payload"]
    text = request["kwargs"]["messages"][0]["content"]
    request["kwargs"]["messages"][0]["content"] = (
        "公开发言引用\n\nPRIVATE ToM INFORMATION\n{}\n\n" + text
    )
    request["response"] = "PRIVATE ToM INFORMATION " + canonical_json_bytes(case.payload).decode()
    call = SimpleNamespace(private_payload=SimpleNamespace(to_value=lambda: request))
    assert analysis._request_tom_payload(call) is None


@pytest.mark.parametrize("replacement", (
    "", "{}", '{"target_ids":[],"target_ids":[],"rows":[]}',
))
def test_coverage_title_alone_or_invalid_json_never_proves_payload(replacement):
    case = _coverage_case()
    request = case.call().private_payload.to_value()
    message = request["kwargs"]["messages"][0]
    message["content"] = message["content"].replace(canonical_json_bytes(case.payload).decode(), replacement)
    call = SimpleNamespace(private_payload=SimpleNamespace(to_value=lambda: request))
    with pytest.raises(ValueError):
        analysis._request_tom_payload(call)


def test_coverage_notom_contamination_is_reported_not_silently_zeroed():
    case = _coverage_case()
    result = _count_case(case, arm=collection.NOTOM)
    assert result["E"] == result["T"] == result["H"] == result["U"] == 1
    assert "unexpected_tom_in_notom" in {entry["kind"] for entry in result["anomalies"]}


def test_coverage_zero_opportunities_are_not_zero_percent():
    case = _coverage_case()
    roles = {**case.roles, "player1": "Villager"}
    result = _count_case(case, roles=roles,
                         audit={"treatments": [], "treatment_count": 0, "predictor_requests": None},
                         calls=[case.call(value=None)], arm=collection.NOTOM)
    assert result["E"] == result["U"] == 0
    assert result["coverage"] == "NOT_APPLICABLE"


@pytest.fixture(scope="module")
def coverage_formal(tmp_path_factory):
    return _formal_collection(tmp_path_factory.mktemp("coverage-analysis") / "collection", coverage=True)


def test_coverage_reads_only_included_artifacts_and_never_runs_models(coverage_formal, monkeypatch):
    root, digest = coverage_formal
    before = _tree_hashes(root)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("offline coverage must not execute games, models or win-rate bootstrap")

    monkeypatch.setattr(collection, "Qwen3GameplayPredictorClient", forbidden)
    monkeypatch.setattr(collection, "run_ablation", forbidden)
    monkeypatch.setattr(collection, "classic7_replay_executor", forbidden)
    monkeypatch.setattr(analysis, "_paired_bootstrap", forbidden)
    result = analysis.analyze_coverage(root, expected_final_digest=digest)
    summary, games = result["summary"], result["games"]
    assert summary["N"] == 40
    assert len(games) == len({row["game_id"] for row in games}) == 80
    assert not {14, 15} & {row["candidate_ordinal"] for row in games}
    no, tom = (summary["arms"][arm] for arm in collection.ARMS)
    assert {field: no[field] for field in ("E", "T", "C", "H", "U")} == {
        "E": 40, "T": 0, "C": 0, "H": 0, "U": 0,
    }
    assert {field: tom[field] for field in ("E", "T", "C", "H", "U")} == {
        "E": 40, "T": 39, "C": 39, "H": 78, "U": 39,
    }
    assert tom["coverage"] == tom["mean_per_game"]["U"] == 39 / 40
    assert tom["games_with_injection"] == 39
    assert no["unexpected_anomaly_count"] == 0
    assert no["anomaly_counts"]["eligible_without_treatment"] == 40
    assert tom["anomaly_counts"]["eligible_without_treatment"] == 1
    assert summary["historical_166_treatments_confirmed"] is False
    assert summary["actual_injection_equals_166"] is False
    assert _tree_hashes(root) == before


def test_coverage_preflights_missing_files_and_rejects_corruption(coverage_formal, tmp_path):
    root = _copy_formal(coverage_formal, tmp_path)
    missing = next(root.glob("candidates/*/tom/games/*/private/backend_call_evidence.jsonl"))
    missing.unlink()
    with pytest.raises(ValueError, match="coverage UNKNOWN: required Artifact files missing"):
        analysis.analyze_coverage(root, expected_final_digest=coverage_formal[1])
    clone = _copy_formal(coverage_formal, tmp_path / "corrupt")
    path = next(clone.glob("candidates/*/tom/games/*/private/backend_call_evidence.jsonl"))
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ArtifactValidationError, match="size mismatch"):
        analysis.analyze_coverage(clone, expected_final_digest=coverage_formal[1])


def test_coverage_pools_opportunities_and_propagates_unknown_without_dropping_games(
    coverage_formal, monkeypatch,
):
    original = analysis._coverage_game

    def varied(*args):
        row = original(*args)
        if row["arm"] == collection.NOTOM:
            row["E"] = 3
        elif row["candidate_ordinal"] == 0:
            row["C"] = "UNKNOWN"
        else:
            row["E"] = 2
        return row

    monkeypatch.setattr(analysis, "_coverage_game", varied)
    result = analysis.analyze_coverage(coverage_formal[0], expected_final_digest=coverage_formal[1])
    no, tom = (result["summary"]["arms"][arm] for arm in collection.ARMS)
    assert len(result["games"]) == 80
    assert no["E"] == 120
    assert tom["E"] == 79 and tom["U"] == 39
    assert tom["coverage"] == 39 / 79
    assert tom["C"] == tom["mean_per_game"]["C"] == "UNKNOWN"


def test_coverage_cli_separates_outputs_and_never_overwrites_history(
    coverage_formal, tmp_path, monkeypatch, capsys,
):
    root, digest = coverage_formal
    output = tmp_path / "chosen-summary"
    details = tmp_path / "chosen-details.csv"
    before = _tree_hashes(root)
    monkeypatch.setattr(analysis, "EXPECTED_FINAL_DIGEST", digest)
    arguments = ["--coverage", "--collection-root", str(root),
                 "--output-dir", str(output), "--game-details-output", str(details)]
    assert analysis.main(arguments) == 0
    summary = json.loads((output / "analysis.json").read_bytes())
    assert json.loads(capsys.readouterr().out) == summary
    with details.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 80
    assert len(json.loads(rows[0]["anomalies"])) == 1
    assert analysis.main(arguments) == 0
    capsys.readouterr()
    assert _tree_hashes(root) == before
    details.write_text("existing analysis, must remain unchanged")
    (output / "analysis.json").unlink()
    with pytest.raises(ValueError, match="differs from reproducible result"):
        analysis.main(arguments)
    assert not (output / "analysis.json").exists()
    assert details.read_text() == "existing analysis, must remain unchanged"
    with pytest.raises(ValueError, match="outside formal collection"):
        analysis.main([*arguments[:-1], str(root / "details.csv")])
    assert _tree_hashes(root) == before


def test_coverage_requires_explicit_input_and_output_paths():
    with pytest.raises(SystemExit):
        analysis.parse_args(["--coverage"])
    with pytest.raises(SystemExit):
        analysis.parse_args(["--game-details-output", "chosen.csv"])
