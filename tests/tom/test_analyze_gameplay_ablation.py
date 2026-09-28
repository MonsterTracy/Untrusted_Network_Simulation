"""Focused read-only analysis tests against the formal artifact schema."""

import hashlib
import json
import shutil

import pytest

from scripts import analyze_tom_gameplay_ablation as analysis
from scripts import tom_gameplay_collection as collection


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


def _formal_collection(root):
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
