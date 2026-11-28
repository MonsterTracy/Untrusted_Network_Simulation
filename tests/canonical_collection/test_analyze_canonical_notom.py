"""验证 Development 限定、抽样、原生回放及缺失指标口径。"""

import json
from pathlib import Path
from random import Random
from types import SimpleNamespace

import pytest

from scripts import analyze_canonical_notom as analysis
from tests.canonical_collection.test_canonical_release import native_split_collection
from tests.canonical_collection.test_game_bundle import _claim, _plan
from tests.canonical_collection.test_terminal_audit import _NativeAgent, _native_game
from tests.runtime.test_runtime_config import new_config
from werewolf.artifact_io import ArtifactValidationError, canonical_json_bytes, canonical_jsonl_bytes, publish_artifact
from werewolf.backends import OpenAICompatibleBackend
from werewolf.canonical_collection.call_audit import AuditedBackend, CanonicalCallAudit
from werewolf.canonical_release import publish_canonical_release
from werewolf.game_runtime import count_game_votes, summarize_game_votes
from werewolf.runtime_config import normalize_runtime_config


@pytest.fixture
def released_games(native_split_collection):
    root, collection, protocol, config = native_split_collection
    release = publish_canonical_release(collection, protocol, root / "releases/release", runtime_config=config,
                                       artifact_root=root)
    publication = root / "publications/release-development"
    manifest = json.loads((publication / "manifest.json").read_bytes())
    members = manifest["development_games"]
    options = {"publication_digest": release.manifest["development_publication_digest"],
               "split_digest": release.manifest["split_manifest_digest"],
               "sample_output": root / "analysis/canonical-notom-sample.json"}
    return root, collection, publication, members, options


def test_statistics_and_sample_use_only_released_development(released_games, monkeypatch):
    root, collection, publication, members, options = released_games
    dev_ids = {game["game_id"] for game in members}
    sealed = [game.path for game in collection.games if game.game_id not in dev_ids]
    sealed.extend((root / "splits/release", root / "terminal_audits/release"))
    expected = [game for game in collection.games if game.game_id in dev_ids]
    before = {game.path / "manifest.json": (game.path / "manifest.json").read_bytes() for game in expected}
    read_bytes = Path.read_bytes

    def read_development_only(path):
        assert not any(path.is_relative_to(directory) for directory in sealed)
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_development_only)
    report = analysis.analyze(collection.path, publication, sample_size=2, **options)
    assert report["coverage"] == {"claimed_attempts": 6, "terminal_outcomes": {"canonical_success": 6},
                                   "unknown_open_attempts": 0, "sealed_test_games": 1,
                                   "published_development_games": 5, "excluded_development_games": {}}
    assert report["source"]["fold_sizes"] == dict.fromkeys(range(5), 1)
    overall = report["all_valid_games"]
    assert (overall["games"], overall["wolf_wins"], overall["villager_wins"]) == (5, 0, 5)
    vote_counts = summarize_game_votes([count_game_votes(game.public_event_stream,
        dict(game.private_replay_evidence.role_assignment)) for game in expected])
    assert overall["votes"] == {**vote_counts, "WER": vote_counts["wolf_exile_rate"]}
    assert overall["cost"]["model_requests"]["sum"] == sum(game.call_budget_summary.used_calls for game in expected)
    assert overall["cost"]["total_tokens"]["sum"] == "UNKNOWN"
    assert overall["cost"]["attempt_wall_seconds"]["games_available"] == 5

    sample = json.loads(options["sample_output"].read_bytes())
    assert sample["sampling_seed"] == 20261128
    assert [game["game_id"] for game in sample["games"]] == Random(20261128).sample(sorted(dev_ids), 2)
    expected_by_id = {game.game_id: game for game in expected}
    for record in sample["games"]:
        game = expected_by_id[record["game_id"]]
        assert record["seed"] == game.private_replay_evidence.seed
        assert record["roles"] == dict(game.private_replay_evidence.role_assignment)
        assert record["bundle_digest"] == game.manifest_digest and record["source_revision"] == collection.plan.source_revision
        assert "winner" not in record
    assert report["sample"]["games"] == 2 and report["sample"]["WWR_95_CI"]["population_games"] == 5
    assert all(path.read_bytes() == original for path, original in before.items())


def test_missing_metrics_keep_valid_games_and_missing_replay_does_not(released_games):
    _, collection, publication, members, options = released_games
    metric_bundle, replay_bundle = [collection.path / "games" / row["game_id"] for row in members[:2]]
    (metric_bundle / analysis._BUDGET_FILE).unlink()
    (metric_bundle / analysis._CALL_FILE).unlink()
    (replay_bundle / analysis._REPLAY_FILE).unlink()
    report = analysis.analyze(collection.path, publication, sample_size=4, **options)
    assert report["coverage"]["excluded_development_games"] == {"missing_native_replay_evidence": 1}
    overall = report["all_valid_games"]
    assert overall["WWR"]["denominator"] == overall["votes"]["games_with_vote_evidence"] == 4
    assert overall["cost"]["model_requests"]["sum"] == "UNKNOWN"
    assert overall["cost"]["model_requests"]["games_available"] == 3
    assert overall["missing_metric_evidence"] == {"missing_call_budget": 1, "missing_backend_receipts": 1,
                                                   "incomplete_transport_receipts": 3}
    sampled = {game["game_id"] for game in json.loads(options["sample_output"].read_bytes())["games"]}
    assert metric_bundle.name in sampled and replay_bundle.name not in sampled


def test_output_remains_outside_archives_and_digest_is_required(released_games):
    _, collection, publication, _, options = released_games
    with pytest.raises(ValueError, match="outside"):
        analysis.analyze(collection.path, publication, sample_size=2,
                         **{**options, "sample_output": collection.path / "sample.json"})
    with pytest.raises(ArtifactValidationError, match="trusted digest"):
        analysis.analyze(collection.path, publication, sample_size=2,
                         **{**options, "publication_digest": "0" * 64})
    assert not options["sample_output"].exists()


def test_wwr_interval_respects_sampling_without_replacement():
    assert analysis._wwr_interval(0, 1, 100)["wolf_win_count_bounds"] == [0, 97]
    assert analysis._wwr_interval(1, 1, 100)["wolf_win_count_bounds"] == [3, 100]
    assert analysis._wwr_interval(37, 100, 100)["bounds"] == [0.37, 0.37]


def test_archived_token_receipts_and_claim_to_terminal_time(tmp_path):
    def completion(**_kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="fixture"), finish_reason="stop")],
                               usage={"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13})

    audit = CanonicalCallAudit(plan=_plan(), configured_call_limit=3, record_requests=True)
    client = SimpleNamespace(max_retries=0, chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
    backend = AuditedBackend(OpenAICompatibleBackend(client=client), audit)
    with audit.action_context(acting_player_id=1, boundary_id=None, is_public_speech=False):
        assert backend.chat(messages=[], model="fixture") == "fixture"
    files = {analysis._BUDGET_FILE: canonical_json_bytes(audit.summary().to_record()),
             analysis._CALL_FILE: canonical_jsonl_bytes(call.to_record() for call in audit.records)}
    bundle = publish_artifact(tmp_path / "cost", files=files,
                              manifest_fields={"artifact_type": "fixture", "schema_version": "fixture"})
    claim = SimpleNamespace(claim_timestamp_utc="2026-10-12T00:00:00Z")
    terminal = SimpleNamespace(terminal_timestamp_utc="2026-10-12T00:00:07Z")
    cost, missing = analysis._archived_cost(bundle, files, claim, terminal)
    assert missing == [] and cost["model_requests"] == 1
    assert [cost[key] for key in ("input_tokens", "output_tokens", "total_tokens")] == [11, 2, 13]
    assert cost["attempt_wall_seconds"] == 7 and cost["backend_elapsed_seconds"] >= 0
    assert cost["game_elapsed_seconds"] == "UNKNOWN"


def test_wolf_poison_counts_use_actual_native_witch_action(tmp_path):
    class PoisonWolfAgent(_NativeAgent):
        def act(self, observation):
            if "skill_witch" in observation["phase"] and ("witch_poison", 1) in observation["valid_action"]:
                return "witch_poison", 1
            return super().act(observation)

    plan = _plan()
    claim = _claim(plan)
    bundle = _native_game(tmp_path, PoisonWolfAgent, plan=plan, claim=claim,
                          config=normalize_runtime_config(new_config()), game_id="poisoned-wolf")
    roles = dict(bundle.private_replay_evidence.role_assignment)
    assert roles["player1"] == "Werewolf"
    _, _, deaths, _ = analysis._native_outcome(bundle.artifact, {"game_id": bundle.game_id, "seed": claim.seed}, roles)
    assert deaths["witch_poison"] == 1 and deaths["unattributed_night"] == 0
