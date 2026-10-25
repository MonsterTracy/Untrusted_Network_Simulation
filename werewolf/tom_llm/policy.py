"""Frozen v1 policy; edited defaults cannot silently become a new study."""

from copy import deepcopy
from pathlib import Path

from werewolf.tom_llm.common import digest, require, strict_json
from werewolf.tom_llm.prompt import PROMPT_VERSION, TEMPLATE_VERSION, asset_bytes
from werewolf.artifact_io import sha256_bytes

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/paper/tom-llm-reasoner-v1.json"
BASE_COMMIT = "d8397ce56c65d6d46b24a6b53f0b71431c04267b"
POLICY = {
    "schema_version": "classic7_llm_reasoner_policy_v1",
    "model": "Qwen/Qwen3-8B", "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
    "dtype": "bfloat16", "quantization": None, "mode": "zero_shot_nonthinking",
    "chat_template_version": TEMPLATE_VERSION, "prompt_version": PROMPT_VERSION,
    "decoding": {"temperature": 0, "top_p": 1, "top_k": -1, "min_p": 0,
                 "n": 1, "max_tokens": 512, "seed": 20261014, "enable_thinking": False},
    "normalization": {"sum_tolerance": .001, "epsilon": 1e-6, "fallback": "uniform_non_self"},
    "semantic_attempts": 1, "transport_attempts": 2, "sdk_max_retries": 0,
    "max_structured_tokens": 1024, "context_length": 8192,
    "dispatch_population": "public_alive", "evaluation_population": "non_wolf_alive",
    "reference_temporal_condition": "explicit_day_phase", "canonical_shift": 0,
    "subset": {"version": "tom_llm_subset_v1", "seed": 20261014, "games_per_fold": 30},
    "bootstrap": {"replicates": 10000, "confidence": .95, "unit": "game",
                  "interval_method": "percentile_linear", "seed": 2196221715671680944},
    "limitations": ["pretraining_and_posttraining", "explicit_rules_and_action_lexicon",
                    "qwen_family_related_to_generator", "method_comparison_not_architecture_ablation"],
}

# Filled from the reviewed immutable assets during implementation; not computed from a new config.
ASSET_DIGESTS = {'chat_template.jinja': '9e35b8b74492fa12cfcc68184f6992e900048bd61b36a7ec438737d8628e2f1f', 'system.txt': 'beb5d6e2f8511092a71d721283bda9b47a15f381d90984bd9dafef7e5c5c06cd'}


def validate_policy(config):
    require(type(config) is dict and digest(config) == digest(POLICY), "configuration differs from frozen v1 policy")
    require(ASSET_DIGESTS and all(sha256_bytes(asset_bytes(name)) == expected
                                for name, expected in ASSET_DIGESTS.items()), "frozen prompt/template asset drift")
    return deepcopy(config)


def load_policy(path=CONFIG_PATH):
    return validate_policy(strict_json(Path(path).read_bytes()))


def policy_binding():
    config = load_policy()
    return {"config": config, "config_digest": digest(config), "asset_digests": dict(ASSET_DIGESTS)}
