"""CLI help and argument contract need neither model dependencies nor credentials."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_phase2_online_intervention_pilot import build_parser, main


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("mode", ("module", "direct"))
def test_help_exits_zero_without_dependencies_or_files(tmp_path, mode):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = str(ROOT) if mode == "module" else ""
    target = (["-m", "scripts.run_phase2_online_intervention_pilot"] if mode == "module" else
              [str(ROOT / "scripts/run_phase2_online_intervention_pilot.py")])
    before = sorted(tmp_path.rglob("*"))
    result = subprocess.run([sys.executable, "-S", *target, "--help"],
                            cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "--campaign-purpose {qualification,pilot}" in result.stdout
    assert "--preflight-only" in result.stdout
    assert "--resume" in result.stdout
    assert sorted(tmp_path.rglob("*")) == before


def test_parser_has_no_research_policy_or_count_overrides():
    options = {flag for action in build_parser()._actions for flag in action.option_strings}
    for forbidden in ("--target-assignment-count", "--sample-count", "--push-probability",
                      "--candidate-policy", "--lambda", "--probe-policy", "--threshold"):
        assert forbidden not in options
        with pytest.raises(SystemExit) as error:
            main([forbidden, "1"])
        assert error.value.code == 2
