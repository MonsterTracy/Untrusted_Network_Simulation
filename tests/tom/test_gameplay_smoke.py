"""CLI boundary for the standalone paired gameplay smoke."""

import json
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import tom_gameplay_ablation as ablation, tom_gameplay_smoke as smoke
from scripts.qwen3_gameplay_predictor import FIT_DIGEST, SEAL_DIGEST
from scripts.run_werewolf_baseline import _scenarios
from tests.canonical_collection.test_game_bundle import _public_events
from tests.canonical_collection.test_public_history import _event, _phase
from tests.tom.test_analyze_gameplay_ablation import _coverage_case
from tests.tom.test_gameplay_ablation import _matrix, _wolf_observation
from werewolf.agents.gpt_agent import GPTAgent
from werewolf.backends import OpenAICompatibleBackend
from werewolf.canonical_collection.frozen_json import freeze_json_value
from werewolf.canonical_collection.public_history import PLAYER_IDS, freeze_public_event_history


def test_cli_paths_override_environment_and_pair_id(monkeypatch):
    monkeypatch.setenv("PREDICTOR_CHECKOUT", "/env/checkout")
    monkeypatch.setenv("QWEN3_FIT_PATH", "/env/fit")
    args = smoke.parse_args([
        "--predictor-checkout", "/cli/checkout",
        "--fit-path", "/cli/fit",
        "--pair-id", "smoke-pair",
        "--expected-head", "expected-head",
    ])
    assert (args.predictor_checkout, args.fit_path, args.pair_id, args.expected_head) == (
        "/cli/checkout", "/cli/fit", "smoke-pair", "expected-head"
    )


def test_cli_uses_environment_defaults(monkeypatch):
    monkeypatch.setenv("PREDICTOR_CHECKOUT", "/env/checkout")
    monkeypatch.setenv("QWEN3_FIT_PATH", "/env/fit")
    args = smoke.parse_args([])
    assert (args.predictor_checkout, args.fit_path, args.pair_id, args.expected_head) == (
        "/env/checkout", "/env/fit", "tom-gameplay-smoke-v1", None
    )


@pytest.mark.parametrize("argv", [[], ["--predictor-checkout", "/tmp/checkout"]])
def test_cli_requires_predictor_paths_without_environment(monkeypatch, argv):
    monkeypatch.delenv("PREDICTOR_CHECKOUT", raising=False)
    monkeypatch.delenv("QWEN3_FIT_PATH", raising=False)
    with pytest.raises(SystemExit, match="2"):
        smoke.parse_args(argv)


def test_expected_head_mismatch_fails_before_server_preflight(monkeypatch):
    clean_head = Mock(return_value="actual-head")
    inspect_inputs = Mock(side_effect=AssertionError("server preflight must not start"))
    monkeypatch.setattr(smoke.operator, "clean_head", clean_head)
    monkeypatch.setattr(smoke.operator, "inspect_inputs", inspect_inputs)
    with pytest.raises(ValueError, match="clean gameplay HEAD mismatch"):
        smoke.main([
            "--predictor-checkout", "/tmp/checkout",
            "--fit-path", "/tmp/fit",
            "--expected-head", "different-head",
        ])
    clean_head.assert_called_once_with()
    inspect_inputs.assert_not_called()


@pytest.mark.parametrize("expected_head", [None, "actual-head"])
def test_current_clean_head_reaches_preflight(monkeypatch, expected_head):
    clean_head = Mock(return_value="actual-head")
    inspect_inputs = Mock(side_effect=RuntimeError("preflight reached"))
    monkeypatch.setattr(smoke.operator, "clean_head", clean_head)
    monkeypatch.setattr(smoke.operator, "inspect_inputs", inspect_inputs)
    argv = ["--predictor-checkout", "/tmp/checkout", "--fit-path", "/tmp/fit"]
    if expected_head is not None:
        argv.extend(["--expected-head", expected_head])
    with pytest.raises(RuntimeError, match="preflight reached"):
        smoke.main(argv)
    clean_head.assert_called_once_with()
    inspect_inputs.assert_called_once_with()


@pytest.fixture
def epoch35_args(monkeypatch, tmp_path):
    baseline = {"default_profile": "local_qwen", "scenarios": _scenarios("local_qwen")}
    baseline["plan_sha256"] = smoke.sha256_bytes(smoke.canonical_json_bytes(baseline))
    plan = tmp_path / "scenarios.json"
    plan.write_bytes(smoke.canonical_json_bytes(baseline))
    monkeypatch.setattr(smoke, "BASELINE_PLAN_SHA256", baseline["plan_sha256"])
    config = smoke.cli._runtime(smoke.operator.RUNTIME)
    provenance = {"served_model_name": "qwen35-9b",
                  "runtime_config_sha256": smoke.sha256_bytes(smoke.canonical_json_bytes(config))}
    monkeypatch.setattr(smoke.operator, "clean_head", lambda: "a" * 40)
    monkeypatch.setattr(smoke.operator, "inspect_inputs", lambda: (provenance, "http://fixture/v1"))
    monkeypatch.setattr(smoke.operator, "live_preflight", Mock())
    monkeypatch.setattr(smoke.cli, "_storage_root", lambda _profile: tmp_path)
    monkeypatch.setattr(smoke.operator, "read_json", lambda _path: {
        "collection_id": "fixture", "target_games": 1, "seed_pool_size": 1, "call_limit": 32,
    })
    return ["--checkpoint", str(tmp_path / "checkpoint"), "--temporal", str(tmp_path / "temporal"),
            "--baseline-plan", str(plan)]


def test_epoch35_args_ignore_historical_environment_and_reject_mixed_modes(monkeypatch, epoch35_args):
    monkeypatch.setenv("PREDICTOR_CHECKOUT", "/env/checkout")
    monkeypatch.setenv("QWEN3_FIT_PATH", "/env/fit")
    args = smoke.parse_args(epoch35_args)
    assert (args.predictor_checkout, args.fit_path) == (None, None)
    assert args.temporal.name == "temporal"
    for argv in (epoch35_args[:-2], [*epoch35_args, "--fit-path", "/other/fit"]):
        with pytest.raises(SystemExit, match="2"):
            smoke.parse_args(argv)


@pytest.mark.parametrize("fault", ("wrong_plan", "overlap"))
def test_epoch35_requires_seed_outside_the_frozen_plan(monkeypatch, epoch35_args, fault):
    load = Mock()
    monkeypatch.setattr(smoke, "Epoch35GameplayPredictor", load)
    if fault == "wrong_plan":
        monkeypatch.setattr(smoke, "BASELINE_PLAN_SHA256", "b" * 64)
    else:
        seed = _scenarios("local_qwen")[0]["seed"]
        monkeypatch.setattr(smoke.operator, "derive_seed_pool", lambda *_args: (seed,))
    with pytest.raises(ValueError, match="frozen Baseline Plan|seed overlaps"):
        smoke.main(epoch35_args)
    load.assert_not_called()


@pytest.mark.parametrize("mode", ("epoch35", "historical", "missing_receipt"))
def test_paired_smoke_runs_shared_hook_and_checks_receipt(monkeypatch, tmp_path, capsys, epoch35_args, mode):
    predictor = SimpleNamespace(
        predict=Mock(return_value=_matrix()), identity={"checkpoint_digest": "c" * 64},
        fit_digest=FIT_DIGEST, seal_digest=SEAL_DIGEST, _next_request=1,
    )
    load = Mock(return_value=predictor)
    historical = Mock(return_value=nullcontext(predictor))
    monkeypatch.setattr(smoke, "Epoch35GameplayPredictor", load)
    monkeypatch.setattr(smoke, "Qwen3GameplayPredictorClient", historical)
    # 该合成种子的座位恰好匹配共享证据夹具。
    monkeypatch.setattr(smoke.operator, "derive_seed_pool", lambda *_args: (244,))
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{}"), finish_reason="stop")])
    create = Mock(return_value=response)
    client = SimpleNamespace(max_retries=0, chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    backend = OpenAICompatibleBackend(client=client)
    if mode == "missing_receipt":
        backend = SimpleNamespace(chat_with_metadata=Mock(return_value=("{}", {})))
    monkeypatch.setattr(smoke, "load_named_backends", Mock(return_value={"local_qwen": backend}))
    assignments = []

    def run_game(env, agents, roles, *, canonical_recorder, call_audit, wolf_speech_tom=None):
        case = _coverage_case(canonical_recorder.plan, canonical_recorder.claim, game_id=canonical_recorder.game_id)
        assignments.append((env.random_seed, tuple(roles)))
        assert list(roles) == [case.roles[seat] for seat in PLAYER_IDS]
        assert all(isinstance(agent, GPTAgent) and agent.decision_method is None for agent in agents)
        assert call_audit.record_requests == (mode != "historical")
        payload = json.loads(wolf_speech_tom(case.prefix, _wolf_observation())) if wolf_speech_tom else None
        request = case.call(value=payload).private_payload.to_value()["kwargs"]
        with call_audit.action_context(acting_player_id=1, boundary_id=case.prefix.boundary_id, is_public_speech=True):
            agents[0].backend.chat_with_metadata(model="qwen35-9b", **request)
        original = case.evidence
        votes = [{"voter": seat, "target": "player1" if seat == "player3" else None} for seat in PLAYER_IDS]
        history = freeze_public_event_history([
            *_public_events(), _phase(5, 1, "vote"), _event(6, "vote_result", votes=votes),
            _event(7, "exile_result", exiled_players=["player1"]),
        ])
        replay = replace(original.private_replay_evidence, replay_inputs=freeze_json_value({"winner": "Villager"}),
                         expected_public_event_digest=history.digest)
        evidence = replace(original, public_event_stream=history, backend_call_evidence=call_audit.records,
                           call_budget_summary=call_audit.summary(), private_replay_evidence=replay)
        canonical_recorder.complete_evidence = lambda: evidence
        return "Villager win"

    monkeypatch.setattr(ablation, "run_game", run_game)
    if mode == "historical":
        smoke.main(["--predictor-checkout", "/fixture/checkout", "--fit-path", "/fixture/fit"])
        historical.assert_called_once()
        load.assert_not_called()
    elif mode == "missing_receipt":
        with pytest.raises(ValueError, match="lacks consistent prediction, injection or response evidence"):
            smoke.main(epoch35_args)
    else:
        smoke.main(epoch35_args)
        assert [value.name for value in load.call_args.args[:2]] == ["checkpoint", "temporal"]
        assert load.call_count == 1 and load.call_args.args[2] == "cpu"
        historical.assert_not_called()
    assert len(assignments) == 2 and assignments[0] == assignments[1]
    predictor.predict.assert_called_once()
    assert not (tmp_path / "gameplay_ablations").exists()
    output = capsys.readouterr().out
    if mode == "missing_receipt":
        assert "PAIRED_GAMEPLAY_SMOKE_PASS" not in output
        return
    assert "PAIRED_GAMEPLAY_SMOKE_PASS" in output and create.call_count == 2
    left, right = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
    assert (left["treatment_count"], right["treatment_count"]) == (0, 1)
    if mode == "epoch35":
        assert left["predictor"] is None and right["predictor"] == predictor.identity
        assert right["coverage"]["tom_integration"]["S"] == 1
        assert right["coverage"]["response_and_submitted_action_decisions"] == 1
        assert right["treatments_by_phase"] == {"discussion": 1, "pk_discussion": 0}
        assert right["votes"]["D1WER"]["value"] == 1
