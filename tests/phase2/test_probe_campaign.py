"""Probe operator boundaries; all evidence and paths are temporary fixtures."""

import hashlib
from pathlib import Path
from statistics import NormalDist

import pytest

from scripts import phase2_online_campaign as campaign


@pytest.mark.parametrize(("qualification", "seed", "target", "cap", "status"), (
    (True, 766378239883399222, 20, 80, "FROZEN"),
    (False, 6449283966833280907, 200, None, "UNFROZEN"),
))
def test_probe_profiles_freeze_identity_seed_and_preregistered_budgets(
        qualification, seed, target, cap, status):
    profile = campaign.read_profile(policy="probe", qualification=qualification)
    payload = (profile["pilot_id"] + ":assignment").encode("utf-8")
    derived = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)
    assert profile["assignment_seed"] == derived == seed
    assert profile["target_assignment_count"] == target
    assert profile["max_games_attempted"] == cap
    assert profile["planning_status"] == status
    assert profile["estimator_eligible"] is False


def test_qualification_plan_preserves_equal_randomization_and_one_assignment_per_game():
    profile = campaign.read_profile(policy="probe", qualification=True)
    plan = campaign.pilot_plan(profile).to_record()
    assert plan["target_assignment_count"] == 20
    assert plan["max_games_attempted"] == 80
    assert plan["immediate_redirect_probability"] == 0.5
    assert plan["probe_then_redirect_probability"] == 0.5
    assert plan["max_assignments_per_game"] == 1


def test_v2_identity_preserves_v1_and_requires_its_entire_planned_pool():
    profile = campaign.read_profile(policy="probe", qualification=True)
    v1 = campaign.read_json(campaign.REPO / "configs/phase2/online-probe-qualification-v1.json")
    formal = campaign.read_profile(policy="probe", qualification=False)
    assert profile["pilot_id"] == "paper-phase2-online-probe-qualification-v2"
    assert v1["pilot_id"] == "paper-phase2-online-probe-qualification-v1"
    assert v1["assignment_seed"] == 4325493448326950820
    assert profile["probe_protocol_digest"] == v1["probe_protocol_digest"] == formal["probe_protocol_digest"]
    for field in ("plan", "game_plan", "work_directory", "destination"):
        assert profile[field] != v1[field]
        assert "qualification-v2" in profile[field]
    assert v1["game_plan"] in {item["path"] for item in profile["excluded_game_plans"]}
    assert v1["game_plan"] in {item["path"] for item in formal["excluded_game_plans"]}
    for current in (profile, formal):
        exclusion = next(item for item in current["excluded_game_plans"] if item["path"] == v1["game_plan"])
        assert exclusion["plan_digest"] == "4719b85665febcb47479a594f13c00940a6636061ebe9410c4b68391be4bda6e"
    assert formal["probe_qualification"]["artifact"] == profile["destination"]
    assert formal["probe_qualification"]["run_inputs"] == str(Path(profile["work_directory"]) / "run_inputs.json")


def test_formal_precision_target_uses_frozen_terminal_max_variance():
    profile = campaign.read_profile(policy="probe", qualification=False)
    variance, half_width = 0.030782462625115643, 0.05
    planned_n = 4 * NormalDist().inv_cdf(0.975) ** 2 * variance / half_width ** 2
    assert planned_n == pytest.approx(189.19930011830033)
    assert profile["target_assignment_count"] == 200 >= planned_n


def test_formal_missing_cap_blocks_prepare_even_if_marked_frozen(tmp_path):
    profile = campaign.read_profile(policy="probe", qualification=False)
    for key in ("plan", "game_plan", "work_directory", "destination"):
        profile[key] = str(tmp_path / key)
    with pytest.raises(ValueError, match="not frozen"):
        campaign.prepare(profile, source_commit="a" * 40)
    profile["planning_status"] = "FROZEN"
    with pytest.raises(ValueError, match="not frozen"):
        campaign.prepare(profile, source_commit="a" * 40)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("qualification", (True, False))
def test_probe_status_unstarted_is_readonly_without_server_imports(qualification, tmp_path, monkeypatch):
    profile = campaign.read_profile(policy="probe", qualification=qualification)
    for key in ("plan", "game_plan", "work_directory", "destination"):
        profile[key] = str(tmp_path / key)
    before = list(tmp_path.iterdir())
    result = campaign.status(profile)
    assert result["status"] == "NOT_STARTED"
    assert result["remaining_target"] == (20 if qualification else 200)
    assert result["IMMEDIATE_REDIRECT"] == result["PROBE_THEN_REDIRECT"] == 0
    assert result["estimator_eligible"] is False
    assert list(tmp_path.iterdir()) == before


def test_operator_reader_rejects_duplicate_keys_and_nonfinite(tmp_path):
    path = tmp_path / "profile.json"
    for text in ('{"assignment_seed":1,"assignment_seed":2}', '{"seed":NaN}'):
        path.write_text(text)
        with pytest.raises(ValueError):
            campaign.read_json(path)


def test_probe_qualification_gate_uses_mechanisms_not_outcome():
    records = [
        {"assignment": {"assigned_strategy": "IMMEDIATE_REDIRECT"},
         "lifecycle": ["T1_COMMITTED", "GAME_RESULT_RECORDED"],
         "day_consequence": {"l_ref": 1}},
        {"assignment": {"assigned_strategy": "PROBE_THEN_REDIRECT"},
         "lifecycle": ["T1_COMMITTED", "T3_SCHEDULED", "T3_REACHED", "T3_COMMITTED",
                       "GAME_RESULT_RECORDED"], "day_consequence": {"l_ref": 0}},
    ]
    gate = campaign.probe_qualification_gate(records, complete=True)
    assert gate["passed"] is True
    for record in records:
        record["day_consequence"]["l_ref"] = .5
    assert campaign.probe_qualification_gate(records, complete=True) == gate
    assert not campaign.probe_qualification_gate(records, complete=False)["passed"]
    records[1]["lifecycle"] = ["T1_LANGUAGE_INVALID", "T3_CANCELLED", "GAME_RESULT_RECORDED"]
    assert not campaign.probe_qualification_gate(records, complete=True)["passed"]
