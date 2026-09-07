import pytest

from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0


def test_game_runtime_rejects_guard_variant_before_gameplay():
    with pytest.raises(ValueError):
        WerewolfTextEnvV0(n_guard=1, n_witch=0, log_save_path=None)
