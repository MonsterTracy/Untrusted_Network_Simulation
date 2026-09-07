from copy import deepcopy
import unittest

from werewolf.runtime_config import normalize_runtime_config


def new_config():
    return {
        "backends": {
            "deepseek": {
                "type": "openai_compatible",
                "base_url": "https://api.deepseek.com",
                "api_key_env": "DEEPSEEK_API_KEY",
            },
        },
        "parser": {
            "backend": "deepseek",
            "model": "deepseek-v4-flash",
            "model_params": {"temperature": 0.0},
        },
        "agent_config": {
            "must_include": ["deepseek-flash-a"],
            "all_candidates": [
                {
                    "profile_name": "deepseek-flash-a",
                    "agent_type": "gpt",
                    "backend": "deepseek",
                    "model": "deepseek-v4-flash",
                    "model_params": {"temperature": 1.0},
                    "sample_ratio": 0.5,
                },
                {
                    "profile_name": "deepseek-flash-b",
                    "agent_type": "gpt",
                    "backend": "deepseek",
                    "model": "deepseek-v4-flash",
                    "model_params": {"temperature": 0.7},
                    "sample_ratio": 0.5,
                },
            ],
        },
        "env_config": {"n_player": 7},
    }



class RuntimeConfigNormalizationTest(unittest.TestCase):
    def test_new_schema_normalizes_successfully(self):
        normalized = normalize_runtime_config(new_config())

        self.assertEqual(
            normalized["backends"]["deepseek"],
            {
                "type": "openai_compatible",
                "base_url": "https://api.deepseek.com",
                "api_key_env": "DEEPSEEK_API_KEY",
                "default_model": None,
                "supports_json_schema": False,
            },
        )
        self.assertEqual(
            normalized["parser"],
            {
                "backend": "deepseek",
                "model": "deepseek-v4-flash",
                "model_params": {"temperature": 0.0},
            },
        )
        self.assertEqual(
            normalized["agent_config"]["must_include"],
            ["deepseek-flash-a"],
        )

    def test_profiles_can_share_backend_and_model_when_names_differ(self):
        candidates = normalize_runtime_config(new_config())[
            "agent_config"
        ]["all_candidates"]

        self.assertEqual(
            [candidate["profile_name"] for candidate in candidates],
            ["deepseek-flash-a", "deepseek-flash-b"],
        )
        self.assertEqual(
            {candidate["backend"] for candidate in candidates},
            {"deepseek"},
        )
        self.assertEqual(
            {candidate["model"] for candidate in candidates},
            {"deepseek-v4-flash"},
        )


    def test_backend_and_backends_together_raise(self):
        config = new_config()
        config["backend"] = {"type": "openai_compatible"}

        with self.assertRaises(ValueError):
            normalize_runtime_config(config)

    def test_backend_json_schema_capability_must_be_boolean(self):
        config = new_config()
        config["backends"]["deepseek"][
            "supports_json_schema"
        ] = True
        normalized = normalize_runtime_config(config)
        self.assertTrue(
            normalized["backends"]["deepseek"][
                "supports_json_schema"
            ]
        )

        config["backends"]["deepseek"][
            "supports_json_schema"
        ] = "true"
        with self.assertRaisesRegex(
            ValueError,
            "supports_json_schema",
        ):
            normalize_runtime_config(config)

    def test_gameplay_max_tokens_is_optional_and_strictly_positive(self):
        normalized = normalize_runtime_config(new_config())
        self.assertNotIn(
            "gameplay_max_tokens",
            normalized["agent_config"]["all_candidates"][0][
                "model_params"
            ],
        )

        config = new_config()
        config["agent_config"]["all_candidates"][0][
            "model_params"
        ]["gameplay_max_tokens"] = 512
        normalized = normalize_runtime_config(config)
        self.assertEqual(
            normalized["agent_config"]["all_candidates"][0][
                "model_params"
            ]["gameplay_max_tokens"],
            512,
        )

        for invalid in (
            True,
            False,
            0,
            -1,
            1.5,
            "512",
            None,
        ):
            with self.subTest(invalid=invalid):
                config = new_config()
                config["agent_config"]["all_candidates"][0][
                    "model_params"
                ]["gameplay_max_tokens"] = invalid
                with self.assertRaisesRegex(
                    ValueError,
                    "gameplay_max_tokens",
                ):
                    normalize_runtime_config(config)

    def test_separate_belief_reporter_config_is_rejected(self):
        config = new_config()
        config["belief"] = {}

        with self.assertRaisesRegex(
            ValueError,
            "unsupported runtime fields",
        ):
            normalize_runtime_config(config)


    def test_missing_parser_backend_raises(self):
        config = new_config()
        config["parser"]["backend"] = "missing"

        with self.assertRaises(ValueError):
            normalize_runtime_config(config)

    def test_missing_parser_model_raises(self):
        config = new_config()
        config["parser"].pop("model")

        with self.assertRaises(ValueError):
            normalize_runtime_config(config)

    def test_duplicate_profile_name_raises(self):
        config = new_config()
        config["agent_config"]["all_candidates"][1][
            "profile_name"
        ] = "deepseek-flash-a"

        with self.assertRaises(ValueError):
            normalize_runtime_config(config)

    def test_missing_profile_backend_raises(self):
        config = new_config()
        config["agent_config"]["all_candidates"][0][
            "backend"
        ] = "missing"

        with self.assertRaises(ValueError):
            normalize_runtime_config(config)

    def test_missing_profile_model_raises(self):
        config = new_config()
        config["agent_config"]["all_candidates"][0].pop("model")

        with self.assertRaises(ValueError):
            normalize_runtime_config(config)

    def test_must_include_unknown_profile_raises(self):
        config = new_config()
        config["agent_config"]["must_include"] = ["missing-profile"]

        with self.assertRaises(ValueError):
            normalize_runtime_config(config)






if __name__ == "__main__":
    unittest.main()
