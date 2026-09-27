"""CLI boundary for the standalone paired gameplay smoke."""

from unittest.mock import Mock

import pytest

from scripts import tom_gameplay_smoke as smoke


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
