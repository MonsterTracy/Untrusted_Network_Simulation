import unittest
import random
import subprocess
import sys
from unittest.mock import patch

from werewolf.game_runtime import build_runtime


ROLES = [
    "Werewolf",
    "Villager",
    "Seer",
    "Witch",
    "Villager",
    "Villager",
    "Werewolf",
]


class RecordingBackend:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def chat(
        self,
        messages,
        model=None,
        temperature=0.7,
        max_tokens=None,
        response_format=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "messages": messages,
                "model": model,
            }
        )
        return "[]"


def runtime_config():
    return {
        "backends": {
            "parser-api": {
                "type": "openai_compatible",
            },
            "agent-api-a": {
                "type": "openai_compatible",
            },
            "agent-api-b": {
                "type": "openai_compatible",
            },
        },
        "parser": {
            "backend": "parser-api",
            "model": "parser-model",
        },
        "env_config": {
            "n_player": 7,
            "n_role": 4,
            "n_werewolf": 2,
            "n_seer": 1,
            "n_guard": 0,
            "n_witch": 1,
            "n_hunter": 0,
            "n_villager": 3,
        },
        "agent_config": {
            "must_include": ["profile-b"],
            "all_candidates": [
                {
                    "profile_name": "profile-a",
                    "agent_type": "gpt",
                    "backend": "agent-api-a",
                    "model": "agent-model-a",
                    "model_params": {"temperature": 0.2},
                    "sample_ratio": 1.0,
                },
                {
                    "profile_name": "profile-b",
                    "agent_type": "gpt",
                    "backend": "agent-api-b",
                    "model": "agent-model-b",
                    "model_params": {"temperature": 0.7},
                    "sample_ratio": 1.0,
                },
            ],
        },
    }


class RunRandomProfileTest(unittest.TestCase):
    def setUp(self):
        self.parser_backend = RecordingBackend("parser")
        self.agent_backend_a = RecordingBackend("agent-a")
        self.agent_backend_b = RecordingBackend("agent-b")
        self.backends = {
            "parser-api": self.parser_backend,
            "agent-api-a": self.agent_backend_a,
            "agent-api-b": self.agent_backend_b,
        }

    def test_root_source_module_exports_the_same_runtime(self):
        """保留的根Source位置只导出同一实现，不拥有第二套执行逻辑。"""
        import run_random
        from werewolf import game_runtime

        for name in ('eval', 'build_runtime', 'assign_agents', 'PUBLIC_SPEECH_RUNTIME_PHASES'):
            self.assertIs(getattr(run_random, name), getattr(game_runtime, name))
        self.assertEqual(game_runtime.eval.__module__, 'werewolf.game_runtime')
        self.assertFalse(hasattr(run_random, 'main_cli'))

    def test_internal_callers_load_package_runtime_without_root_module(self):
        """生产采集与Phase-2直接使用正式Runtime，不依赖根文件反向装配。"""
        code = '''
import sys
from werewolf import game_runtime
from werewolf.canonical_collection import production_runtime
from werewolf.interventions.online import runner as phase2_online_runner
assert production_runtime.build_runtime is game_runtime.build_runtime
assert production_runtime.run_game is game_runtime.eval
assert phase2_online_runner.run_canonical_game is game_runtime.eval
assert 'run_random' not in sys.modules
'''
        subprocess.run([sys.executable, '-B', '-c', code], check=True)

    def test_source_inventories_cover_the_runtime_implementation(self):
        """显式Source清单必须绑定迁移后的实现，不能只记录根转发文件。"""
        from pathlib import Path
        from werewolf import game_runtime
        from werewolf.artifact_io import sha256_bytes
        from werewolf.observer_dtrs.shadow_worker import _implementation
        from werewolf.interventions.online.preflight import SOURCE_FILES

        name = 'werewolf/game_runtime.py'
        self.assertIn(name, SOURCE_FILES)
        self.assertEqual(_implementation()[name], sha256_bytes(Path(game_runtime.__file__).read_bytes()))

    def test_seed_preserves_roles_profiles_and_next_random_draw(self):
        """固定迁移前观察到的角色、Profile和后续RNG状态，防止改变随机调用顺序。"""
        expected = (
            (0, ['Villager', 'Villager', 'Werewolf', 'Werewolf', 'Seer', 'Villager', 'Witch'],
             ['profile-b', 'profile-b', 'profile-a', 'profile-a', 'profile-b', 'profile-b', 'profile-b'],
             2577854692418029171),
            (3, ['Werewolf', 'Villager', 'Villager', 'Witch', 'Seer', 'Villager', 'Werewolf'],
             ['profile-b', 'profile-a', 'profile-a', 'profile-a', 'profile-a', 'profile-a', 'profile-b'],
             9979206796010795124),
            (17, ['Seer', 'Werewolf', 'Werewolf', 'Witch', 'Villager', 'Villager', 'Villager'],
             ['profile-b', 'profile-a', 'profile-a', 'profile-b', 'profile-b', 'profile-b', 'profile-b'],
             15858755842042771698),
        )
        state = random.getstate()
        try:
            for seed, expected_roles, expected_profiles, expected_draw in expected:
                with self.subTest(seed=seed):
                    _, _, roles, profiles = build_runtime(
                        runtime_config(), None, random_seed=seed, backends=self.backends)
                    self.assertEqual(roles, expected_roles)
                    self.assertEqual(profiles, expected_profiles)
                    self.assertEqual(random.getrandbits(64), expected_draw)
        finally:
            random.setstate(state)

    def test_new_schema_wires_parser_and_profile_backends(self):
        env, agents, roles, profile_names = build_runtime(
            runtime_config(),
            log_save_path=None,
            roles=ROLES,
            random_seed=3,
            backends=self.backends,
        )

        self.assertEqual(roles, ROLES)
        self.assertIs(
            env.speech_perceiver.backend,
            self.parser_backend,
        )
        self.assertEqual(
            env.speech_perceiver.model_name,
            "parser-model",
        )
        self.assertEqual(profile_names[0], "profile-b")
        self.assertIn("profile-a", profile_names)
        for agent, profile_name in zip(agents, profile_names):
            if profile_name == "profile-a":
                self.assertIs(agent.backend, self.agent_backend_a)
                self.assertEqual(agent.model_name, "agent-model-a")
                self.assertEqual(agent.temperature, 0.2)
            else:
                self.assertIs(agent.backend, self.agent_backend_b)
                self.assertEqual(agent.model_name, "agent-model-b")
                self.assertEqual(agent.temperature, 0.7)

    @patch("werewolf.game_runtime.load_named_backends")
    def test_default_runtime_path_loads_named_backends(
        self,
        load_named_backends,
    ):
        load_named_backends.return_value = self.backends

        build_runtime(
            runtime_config(),
            log_save_path=None,
            roles=ROLES,
            random_seed=3,
        )

        load_named_backends.assert_called_once()
        normalized = load_named_backends.call_args.args[0]
        self.assertIn("backends", normalized)
        self.assertEqual(
            normalized["parser"]["backend"],
            "parser-api",
        )

    def test_empty_eligible_profiles_raise_value_error(self):
        config = runtime_config()
        config["agent_config"]["must_include"] = []
        config["agent_config"]["all_candidates"] = [
            config["agent_config"]["all_candidates"][0]
        ]

        with self.assertRaises(ValueError) as raised:
            build_runtime(
                config,
                log_save_path=None,
                roles=ROLES,
                random_seed=3,
                backends=self.backends,
            )

        self.assertIn("eligible", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
