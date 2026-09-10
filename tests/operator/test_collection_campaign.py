"""Operator tests with deterministic runtime fixtures; no model, GPU, or network."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from scripts import collect_games as operator
from werewolf.artifact_io import canonical_json_bytes, sha256_bytes


CAMPAIGN = {"collection_id": "development-qwen35-9b-300-v1",
            "target_games": 300, "seed_pool_size": 450, "call_limit": 1000}


def test_campaign_accepts_explicit_values():
    assert operator.validate_campaign(CAMPAIGN) == CAMPAIGN


@pytest.mark.parametrize("field,value", [
    ("target_games", 0), ("target_games", -1), ("target_games", True),
    ("seed_pool_size", True), ("seed_pool_size", 299), ("seed_pool_size", 450.0),
    ("call_limit", 0), ("call_limit", -1), ("call_limit", False),
    ("collection_id", ""), ("collection_id", "../escape"),
    ("collection_id", "/absolute"), ("collection_id", "a/b"),
    ("collection_id", "a..b"), ("collection_id", "a\\b"),
])
def test_campaign_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        operator.validate_campaign({**CAMPAIGN, field: value})


def test_campaign_rejects_unknown_missing_and_duplicate_fields(tmp_path):
    with pytest.raises(ValueError):
        operator.validate_campaign({**CAMPAIGN, "force": True})
    with pytest.raises(ValueError):
        operator.validate_campaign({})
    path = tmp_path / "duplicate.json"
    path.write_text('{"call_limit":1000,"call_limit":2000}')
    with pytest.raises(ValueError, match="duplicate"):
        operator.read_json(path)


def test_seed_pool_is_content_independent_ordered_and_unique():
    pool = operator.derive_seed_pool(CAMPAIGN["collection_id"], 450)
    assert pool == operator.derive_seed_pool(CAMPAIGN["collection_id"], 450)
    assert pool[:300] == operator.derive_seed_pool(CAMPAIGN["collection_id"], 300)
    assert pool != operator.derive_seed_pool("different-campaign", 450)
    assert len(set(pool)) == 450
    assert all(0 <= seed < 2**63 and seed not in operator.CALIBRATION_SEEDS for seed in pool)


@pytest.mark.parametrize("seed,size", [(900000001, 1), (900000015, 1), (42, 2)])
def test_seed_collision_and_calibration_are_fail_closed(monkeypatch, seed, size):
    monkeypatch.setattr(operator.hashlib, "sha256",
                        lambda _: SimpleNamespace(digest=lambda: seed.to_bytes(8, "big") + bytes(24)))
    with pytest.raises(ValueError, match="no skip or re-roll"):
        operator.derive_seed_pool("campaign", size)


@pytest.fixture
def campaign_run(tmp_path, monkeypatch):
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(json.dumps(CAMPAIGN))
    provenance = {"runtime_config_sha256": "a" * 64, "serve_config_sha256": "b" * 64,
                  "model_manifest_sha256": "c" * 64, "hf_revision": "d" * 40,
                  "served_model_name": "qwen35-9b", "seed_rule_identity": operator.SEED_RULE}
    monkeypatch.setattr(operator.cli, "_storage_root", lambda _: tmp_path)
    monkeypatch.setattr(operator, "clean_head", lambda: "e" * 40)
    monkeypatch.setattr(operator, "inspect_inputs", lambda: (dict(provenance), "http://127.0.0.1:8000/v1"))
    health = Mock()
    monkeypatch.setattr(operator, "live_preflight", health)
    collector = Mock(return_value=0)
    monkeypatch.setattr(operator.cli, "main", collector)
    return SimpleNamespace(root=tmp_path, path=campaign_path, provenance=provenance,
                           health=health, collector=collector,
                           plan=tmp_path / "plans" / (CAMPAIGN["collection_id"] + ".json"),
                           destination=tmp_path / "canonical" / CAMPAIGN["collection_id"])


def begin(fixture):
    operator.run_campaign(fixture.path)
    fixture.destination.mkdir(parents=True)
    (fixture.destination / "existing-ledger-evidence").write_text("keep")


def test_new_campaign_freezes_once_reloads_and_delegates(campaign_run, monkeypatch, capsys):
    construct = Mock(wraps=operator.construct_collection_plan)
    reload_plan = Mock(wraps=operator.load_plan)
    monkeypatch.setattr(operator, "construct_collection_plan", construct)
    monkeypatch.setattr(operator, "load_plan", reload_plan)
    assert operator.run_campaign(campaign_run.path) == 0
    construct.assert_called_once()
    reload_plan.assert_called_once_with(campaign_run.plan)
    plan = reload_plan(campaign_run.plan)
    assert plan.source_revision == "e" * 40
    assert plan.target_canonical_success_count == 300
    assert len(plan.ordered_seed_pool) == 450
    assert dict(plan.environment_provenance)["configured_call_limit"] == "1000"
    assert plan.call_budget_identity == "classic7-backend-dispatch-cap-1000-v1"
    campaign_run.collector.assert_called_once_with([
        "--storage-profile", str(operator.STORAGE), "collect", "--plan", str(campaign_run.plan),
        "--runtime-config", str(operator.RUNTIME), "--call-limit", "1000",
        "--destination", CAMPAIGN["collection_id"]])
    assert "COLLECTION PLAN FROZEN" in capsys.readouterr().out


def test_resume_reuses_plan_without_seed_regeneration(campaign_run, monkeypatch, capsys):
    begin(campaign_run)
    before = campaign_run.plan.read_bytes()
    monkeypatch.setattr(operator, "derive_seed_pool", Mock(side_effect=AssertionError("regenerated seeds")))
    monkeypatch.setattr(operator, "construct_collection_plan", Mock(side_effect=AssertionError("new Plan")))
    operator.run_campaign(campaign_run.path, resume=True)
    assert campaign_run.plan.read_bytes() == before
    assert campaign_run.collector.call_args.args[0][-1] == "--resume"
    assert (campaign_run.destination / "existing-ledger-evidence").read_text() == "keep"
    assert "USING EXISTING FROZEN COLLECTION PLAN" in capsys.readouterr().out


@pytest.mark.parametrize("field,value", [("target_games", 301), ("seed_pool_size", 451), ("call_limit", 999)])
def test_resume_rejects_changed_campaign(campaign_run, field, value):
    begin(campaign_run)
    campaign_run.collector.reset_mock()
    campaign_run.path.write_text(json.dumps({**CAMPAIGN, field: value}))
    with pytest.raises(ValueError, match="mismatch"):
        operator.run_campaign(campaign_run.path, resume=True)
    campaign_run.collector.assert_not_called()


@pytest.mark.parametrize("field", ["runtime_config_sha256", "model_manifest_sha256",
                                  "serve_config_sha256", "hf_revision"])
def test_resume_rejects_changed_provenance(campaign_run, field):
    begin(campaign_run)
    campaign_run.provenance[field] = "changed"
    campaign_run.collector.reset_mock()
    with pytest.raises(ValueError, match="mismatch"):
        operator.run_campaign(campaign_run.path, resume=True)
    campaign_run.collector.assert_not_called()


def test_resume_rejects_changed_head(campaign_run, monkeypatch):
    begin(campaign_run)
    monkeypatch.setattr(operator, "clean_head", lambda: "f" * 40)
    with pytest.raises(ValueError, match="source_revision"):
        operator.run_campaign(campaign_run.path, resume=True)


def test_dirty_source_and_failed_health_do_not_publish(campaign_run, monkeypatch):
    monkeypatch.setattr(operator, "clean_head", Mock(side_effect=ValueError("dirty checkout")))
    with pytest.raises(ValueError, match="dirty"):
        operator.run_campaign(campaign_run.path)
    assert not campaign_run.plan.exists()
    monkeypatch.setattr(operator, "clean_head", lambda: "e" * 40)
    campaign_run.health.side_effect = RuntimeError("unhealthy")
    with pytest.raises(RuntimeError, match="unhealthy"):
        operator.run_campaign(campaign_run.path)
    assert not campaign_run.plan.exists()
    campaign_run.collector.assert_not_called()


def test_source_change_during_preflight_fails(campaign_run, monkeypatch):
    monkeypatch.setattr(operator, "clean_head", Mock(side_effect=["e" * 40, "f" * 40]))
    with pytest.raises(ValueError, match="changed during"):
        operator.run_campaign(campaign_run.path)
    assert not campaign_run.plan.exists()


@pytest.mark.parametrize("plan_exists,destination_exists,resume", [
    (False, False, True), (False, True, False), (False, True, True),
    (True, False, False), (True, False, True), (True, True, False),
])
def test_plan_destination_state_machine(campaign_run, plan_exists, destination_exists, resume):
    if plan_exists:
        operator.run_campaign(campaign_run.path)
    if destination_exists:
        campaign_run.destination.mkdir(parents=True)
    campaign_run.collector.reset_mock()
    with pytest.raises(ValueError):
        operator.run_campaign(campaign_run.path, resume=resume)
    campaign_run.collector.assert_not_called()


def test_plan_publication_cannot_overwrite(campaign_run):
    operator.run_campaign(campaign_run.path)
    before = campaign_run.plan.read_bytes()
    plan = operator.load_plan(campaign_run.plan)
    with pytest.raises(FileExistsError):
        operator.publish_plan(campaign_run.plan, plan)
    assert campaign_run.plan.read_bytes() == before


@pytest.mark.parametrize("response_ids,unhealthy", [(["wrong-model"], False), (["qwen35-9b"], True)])
def test_live_preflight_rejects_health_or_model(monkeypatch, response_ids, unhealthy):
    client = Mock()
    health = Mock()
    if unhealthy:
        health.raise_for_status.side_effect = RuntimeError("health error")
    models = Mock()
    models.json.return_value = {"data": [{"id": item} for item in response_ids]}
    client.get.side_effect = [health, models]
    context = MagicMock()
    context.__enter__.return_value = client
    factory = Mock(return_value=context)
    monkeypatch.setattr(operator.openai, "DefaultHttpx2Client", factory)
    with pytest.raises((ValueError, RuntimeError)):
        operator.live_preflight("http://127.0.0.1:8000/v1", "qwen35-9b")
    factory.assert_called_once_with(trust_env=False, follow_redirects=False)
    assert client.get.call_args_list[0].args == ("http://127.0.0.1:8000/health",)


def test_model_manifest_sorted_complete_and_symlink_rejected(tmp_path):
    (tmp_path / "HF_REVISION").write_text("a" * 40)
    (tmp_path / "z").write_bytes(b"weights")
    (tmp_path / "a").mkdir()
    (tmp_path / "a/tokenizer.json").write_bytes(b"{}")
    records = [{"path": name, "size_bytes": len(data), "sha256": sha256_bytes(data)}
               for name, data in [("HF_REVISION", b"a" * 40), ("a/tokenizer.json", b"{}"), ("z", b"weights")]]
    assert operator.model_manifest(tmp_path) == (sha256_bytes(canonical_json_bytes(records)), "a" * 40)
    (tmp_path / "link").symlink_to(tmp_path / "z")
    with pytest.raises(ValueError, match="symlink"):
        operator.model_manifest(tmp_path)


def test_clean_head_rejects_dirty_tree(monkeypatch):
    monkeypatch.setattr(operator.subprocess, "check_output", Mock(return_value=" M changed.py\n"))
    with pytest.raises(ValueError, match="clean Git"):
        operator.clean_head()


def test_clean_head_reads_actual_revision(monkeypatch):
    command = Mock(side_effect=["", "a" * 40 + "\n"])
    monkeypatch.setattr(operator.subprocess, "check_output", command)
    assert operator.clean_head() == "a" * 40
    assert command.call_args.args[0][-2:] == ["rev-parse", "HEAD"]


def test_plan_reload_rejects_modified_digest(campaign_run):
    operator.run_campaign(campaign_run.path)
    value = operator.read_json(campaign_run.plan)
    value["plan_digest"] = "0" * 64
    campaign_run.plan.write_bytes(canonical_json_bytes(value))
    with pytest.raises(ValueError):
        operator.load_plan(campaign_run.plan)


def test_model_manifest_rejects_missing_revision(tmp_path):
    (tmp_path / "weights").write_bytes(b"weights")
    with pytest.raises(FileNotFoundError):
        operator.model_manifest(tmp_path)


def test_inspect_inputs_uses_current_config_bytes(monkeypatch):
    monkeypatch.setattr(operator, "model_manifest", lambda _: ("a" * 64, "b" * 40))
    monkeypatch.setattr(operator, "software_versions", lambda: {"client_openai": "test-version"})
    provenance, base_url = operator.inspect_inputs()
    runtime = operator.normalize_runtime_config(operator.yaml.safe_load(operator.RUNTIME.read_bytes()))
    assert provenance["runtime_config_sha256"] == sha256_bytes(canonical_json_bytes(runtime))
    assert provenance["serve_config_sha256"] == sha256_bytes(operator.DEPLOYMENT.read_bytes())
    assert provenance["model_manifest_sha256"] == "a" * 64
    assert provenance["hf_revision"] == "b" * 40
    assert base_url == "http://127.0.0.1:8000/v1"


def test_live_preflight_success_uses_only_explicit_endpoints(monkeypatch):
    client = Mock()
    models = Mock()
    models.json.return_value = {"data": [{"id": "qwen35-9b"}]}
    client.get.side_effect = [Mock(), models]
    context = MagicMock()
    context.__enter__.return_value = client
    monkeypatch.setattr(operator.openai, "DefaultHttpx2Client", Mock(return_value=context))
    operator.live_preflight("http://127.0.0.1:8000/v1", "qwen35-9b")
    assert [call.args[0] for call in client.get.call_args_list] == [
        "http://127.0.0.1:8000/health", "http://127.0.0.1:8000/v1/models"]


@pytest.fixture
def production_identity(monkeypatch):
    monkeypatch.setattr(operator, "model_manifest", lambda _: ("a" * 64, "b" * 40))
    monkeypatch.setattr(operator, "software_versions", lambda: {"client_openai": "test-version"})
    provenance, _ = operator.inspect_inputs()
    config = operator.normalize_runtime_config(operator.yaml.safe_load(operator.RUNTIME.read_bytes()))
    fields = operator.plan_fields({**CAMPAIGN, "target_games": 1, "seed_pool_size": 1},
                                  "e" * 40, provenance)
    return operator.construct_collection_plan(ordered_seed_pool=(101,), **fields), config


def test_plan_matches_production_identity(production_identity):
    plan, config = production_identity
    assert plan.model_identity == config["parser"]["model"]
    assert plan.prompt_identity == operator.V1_SPEECH_PROMPT_VERSION
    assert plan.parser_identity == operator.V1_SPEECH_PARSER_VERSION
    bound = dict(plan.environment_provenance)
    assert bound["hf_revision"] == "b" * 40
    assert bound["model_manifest_sha256"] == "a" * 64
    assert bound["served_model_name"] == plan.model_identity
    assert bound["gameplay_prompt_profile"] == operator.STRICT_CLASSIC7_GAMEPLAY_PROMPT_PROFILE


def test_production_runtime_identity_wiring(production_identity):
    from tests.canonical_collection.test_game_bundle import _claim
    from werewolf.canonical_collection.production_runtime import Classic7RuntimeFactory

    plan, config = production_identity
    backend = Mock()
    runtime = Classic7RuntimeFactory(
        runtime_config=config, backends={config["parser"]["backend"]: backend},
        configured_call_limit=1000,
    )(plan=plan, claim=_claim(plan))
    perceiver = runtime.env.speech_perceiver
    assert perceiver.model_name == plan.model_identity == config["parser"]["model"]
    assert perceiver.backend.canonical_backend_identity == plan.backend_identity
    backend.chat.assert_not_called()
    backend.chat_with_metadata.assert_not_called()


@pytest.mark.parametrize("invalid_identity", [None, "model_identity", "prompt_identity"])
def test_finite_bundle_identity_contract(production_identity, tmp_path, monkeypatch, invalid_identity):
    from dataclasses import replace
    from tests.canonical_collection import test_game_bundle as fixtures
    from werewolf.canonical_collection import publish_canonical_game_bundle

    plan, _ = production_identity
    if invalid_identity is not None:
        annotation = fixtures._annotation
        old_value = (f"qwen3.5-9b:{'b' * 40}:{'a' * 64}" if invalid_identity == "model_identity"
                     else f"{operator.STRICT_CLASSIC7_GAMEPLAY_PROMPT_PROFILE}:{operator.V1_SPEECH_PROMPT_VERSION}")
        # Only the V1 annotation receives the wrong identity; calls/parent retain the real Plan.
        monkeypatch.setattr(fixtures, "_annotation", lambda history, parent:
                            annotation(history, replace(parent, **{invalid_identity: old_value})))
    if invalid_identity == "prompt_identity":
        # The V1 dataclass rejects this value before a Bundle can even be constructed.
        with pytest.raises(ValueError, match="unsupported V1 prompt_version"):
            fixtures._fixture(plan)
        return
    _, claim, evidence, replay = fixtures._fixture(plan)
    destination = tmp_path / evidence.game_id
    if invalid_identity == "model_identity":
        with pytest.raises(ValueError, match="V1 attempt model_identity does not match Collection Plan"):
            publish_canonical_game_bundle(destination, plan=plan, claim=claim,
                                         evidence=evidence, replay_executor=replay)
        assert not destination.exists()
    else:
        bundle = publish_canonical_game_bundle(destination, plan=plan, claim=claim,
                                              evidence=evidence, replay_executor=replay)
        assert bundle.manifest_digest
