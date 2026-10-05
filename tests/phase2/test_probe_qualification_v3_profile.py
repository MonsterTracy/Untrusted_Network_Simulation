"""Frozen V3 registration and full-pool guards, without server/model assembly."""

import hashlib
from pathlib import Path

import pytest

from scripts import phase2_online_campaign as campaign
from tests.canonical_collection.test_attempt_ledger import _plan
from tests.phase2.test_probe_campaign_integration import assembly
from werewolf.artifact_io import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[2]
EXCLUSION_DIGESTS = {
    "phase2-online-terminal-pilot-v1-game-plan.json":
        "add3fe53054d9b89b6c6570b585bce13070f010dcf2e28a40a4671fcd713c64b",
    "phase2-online-probe-qualification-v1-game-plan.json":
        "4719b85665febcb47479a594f13c00940a6636061ebe9410c4b68391be4bda6e",
    "phase2-online-probe-qualification-v2-game-plan.json":
        "ef8e2d0d239f3e0d98ef84ed44c51d3cbceb80b4ddacee5b52fff6a7ac0bd151",
}


def test_v3_freezes_new_identity_seed_budget_paths_and_prior_plan_digests():
    profile = campaign.read_profile(policy="probe", qualification=True)
    v2 = campaign.read_json(ROOT / "configs/phase2/online-probe-qualification-v2.json")
    assert profile["pilot_id"] == "paper-phase2-online-probe-qualification-v3"
    payload = (profile["pilot_id"] + ":assignment").encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)
    assert profile["assignment_seed"] == seed == 6288282358363007265
    assert profile["target_assignment_count"] == 20
    assert profile["max_games_attempted"] == 80
    assert profile["planning_status"] == "FROZEN"
    assert profile["estimator_eligible"] is False
    assert profile["probe_protocol_digest"] == v2["probe_protocol_digest"]
    for field in ("plan", "game_plan", "work_directory", "destination"):
        assert profile[field] == v2[field].replace("qualification-v2", "qualification-v3")
    assert {Path(item["path"]).name: item["plan_digest"]
            for item in profile["excluded_game_plans"]} == EXCLUSION_DIGESTS
    plan = campaign.pilot_plan(profile).to_record()
    assert plan["immediate_redirect_probability"] == plan["probe_then_redirect_probability"] == 0.5
    assert plan["max_assignments_per_game"] == 1


@pytest.mark.parametrize(("version", "expected_digest"), (
    ("v1", "bdb9c239291752931fa37617fd72076569cd39122287b2862f2bf8f6f4d955c7"),
    ("v2", "77db9c5eb54602ac4faa0d6a410b1adf7a162879a89e33da5f187516b532ee45"),
))
def test_historical_qualification_profiles_remain_byte_identical(version, expected_digest):
    path = ROOT / f"configs/phase2/online-probe-qualification-{version}.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_digest


def test_v3_is_disjoint_from_all_prior_planned_pools_using_production_derivation(assembly):
    collection, _ = assembly
    pools = {}
    for version in ("v1", "v2", "v3"):
        profile = campaign.read_json(ROOT / f"configs/phase2/online-probe-qualification-{version}.json")
        pools[version] = set(collection.derive_seed_pool(
            profile["pilot_id"], profile["max_games_attempted"]))
        assert len(pools[version]) == 80
    assert pools["v3"].isdisjoint(pools["v1"])
    assert pools["v3"].isdisjoint(pools["v2"])
    assert pools["v1"].isdisjoint(pools["v2"])
    terminal = campaign.read_json(ROOT / "configs/phase2/online-terminal-pilot-formal-v1.json")
    terminal_pool = set(collection.derive_seed_pool(terminal["pilot_id"], terminal["max_games_attempted"]))
    assert len(terminal_pool) == 240
    # Static frozen-convention check only; no actual server plan is available
    # here, so the excluded plan's digest must still be verified at admission.
    assert pools["v3"].isdisjoint(terminal_pool)


def test_formal_only_points_to_v3_and_stays_unfrozen_before_any_prepare_work(tmp_path, monkeypatch):
    formal = campaign.read_profile(policy="probe")
    v3 = campaign.read_profile(policy="probe", qualification=True)
    assert formal["target_assignment_count"] == 200
    assert formal["assignment_seed"] == 6449283966833280907
    assert formal["max_games_attempted"] is None
    assert formal["planning_status"] == "UNFROZEN"
    assert formal["probe_qualification"] == {
        "artifact": v3["destination"],
        "run_inputs": str(Path(v3["work_directory"]) / "run_inputs.json"),
        "manifest_digest": None, "inputs_digest": None,
    }
    assert {Path(item["path"]).name: item["plan_digest"]
            for item in formal["excluded_game_plans"]} == EXCLUSION_DIGESTS
    for field in ("plan", "game_plan", "work_directory", "destination"):
        formal[field] = str(tmp_path / field)
    monkeypatch.setattr(campaign, "clean_source", lambda **kwargs: pytest.fail("prepare must stop before source/runtime"))
    for status in ("UNFROZEN", "FROZEN"):
        formal["planning_status"] = status
        with pytest.raises(ValueError, match="not frozen"):
            campaign.prepare(formal, source_commit="a" * 40)
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def loaded_exclusions(tmp_path, monkeypatch, assembly):
    """Synthetic complete canonical pools; real remote Terminal plan is unavailable."""
    profile = campaign.read_profile(policy="probe", qualification=True)
    repo = tmp_path / "repo"
    protocol = repo / "docs/research/phase2-probe-policy-protocol-v1.md"
    protocol.parent.mkdir(parents=True)
    protocol.write_bytes((ROOT / "docs/research/phase2-probe-policy-protocol-v1.md").read_bytes())
    for version in ("v1", "v2"):
        relative = f"configs/phase2/online-probe-qualification-{version}.json"
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    identities = ("paper-phase2-online-terminal-pilot-v1",
                  "paper-phase2-online-probe-qualification-v1",
                  "paper-phase2-online-probe-qualification-v2")
    profile["excluded_game_plans"] = []
    for index, identity in enumerate(identities, 1):
        old = _plan(collection_id=identity, ordered_seed_pool=tuple(range(index * 1000 + 1, index * 1000 + 81)),
                    target_canonical_success_count=20)
        path = tmp_path / f"excluded-{index}.json"
        path.write_bytes(canonical_json_bytes(old.to_record()))
        profile["excluded_game_plans"].append({"path": str(path), "plan_digest": old.plan_digest})
    bound = {"paths": {field: str(tmp_path / ("qualified-" + field))
                       for field in ("plan", "game_plan", "work_directory", "destination")}}
    monkeypatch.setattr(campaign, "REPO", repo)
    monkeypatch.setattr(campaign, "qualification_inputs", lambda: ({}, bound, None))
    # Isolate existing Terminal/development/calibration evidence admission only;
    # excluded canonical parsing, digest verification, and full-pool overlap stay real.
    monkeypatch.setattr(campaign, "seed_overlaps", lambda *args: {
        "qualification": 0, "development": 0, "calibration": 0})
    excluded = campaign.probe_inputs(profile)[-1]
    assert tuple(game.collection_id for game in excluded) == identities
    assert all(len(game.ordered_seed_pool) == 80 for game in excluded)
    return excluded


@pytest.mark.parametrize("pool_index", (0, 1, 2))
def test_loaded_full_terminal_v1_v2_pools_reject_their_last_planned_seed(loaded_exclusions, pool_index):
    old = loaded_exclusions[pool_index]
    game = _plan(collection_id="paper-phase2-online-probe-qualification-v3",
                 ordered_seed_pool=(old.ordered_seed_pool[-1], *range(5001, 5080)),
                 target_canonical_success_count=20)
    with pytest.raises(ValueError, match="seed overlap; no skip/reroll"):
        campaign.checked_seed_overlaps(game, {}, {}, loaded_exclusions)
