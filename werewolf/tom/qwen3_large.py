"""独立的 Qwen3-ToM-Large 图；不注册历史对照或训练生命周期。"""

from copy import deepcopy

import torch
from torch.nn import functional as F
from transformers import Qwen3Config

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.tom.backbone_model import _ObserverConditionedBackbone, _Qwen3Stack
from werewolf.tom.dataset import ExperimentCapacity
from werewolf.tom.protocol import digest_fields


MODEL_VERSION = "classic7_qwen3_tom_large_v1"
TEMPORAL_EXTENSION = "fixed_right_zero_pad_256_to_512_v1"
QWEN3_LARGE_CONFIG = {
    "hidden_size": 512, "num_hidden_layers": 8,
    "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 64,
    "intermediate_size": 1536, "max_position_embeddings": 1024,
    "hidden_act": "silu", "rms_norm_eps": 1e-6,
    "attention_bias": False, "attention_dropout": .1, "initializer_range": .02,
    "use_cache": False, "sliding_window": None, "layer_types": ["full_attention"] * 8,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
}


def graph_identity():
    """图身份显式包含时间零扩展与完整模型参数预算。"""
    config = deepcopy(QWEN3_LARGE_CONFIG)
    graph = {
        "architecture": "qwen3_tom_large", "model_version": MODEL_VERSION,
        "resolved_config": config,
        "resolved_config_digest": sha256_bytes(canonical_json_bytes(config)),
        "shell_graph_version": "classic7_observer_shell_512_v1",
        "dtype": "float32", "attention_backend": "eager", "continuous_input_scale": 1,
        "temporal_extension": TEMPORAL_EXTENSION,
        "output_contract": "non_self_suspicion_simplex",
        "sequence_parameters": 25175552, "shell_parameters": 1089543,
        "total_parameters": 26265095,
    }
    return {**graph, "graph_digest": sha256_bytes(canonical_json_bytes(graph))}


class Qwen3ToMLarge(_ObserverConditionedBackbone):
    """完整公开 PRE 输入，七观察者读出；初始化种子与划分种子独立。"""

    def __init__(self, temporal, *, initialization_seed):
        super().__init__()
        if type(initialization_seed) is not int or not 0 <= initialization_seed < 2**63:
            raise ValueError("invalid Qwen3-ToM-Large initialization seed")
        if torch.get_default_dtype() != torch.float32:
            raise ValueError("Qwen3-ToM-Large requires FP32 construction")
        self.graph = graph_identity()
        self.capacity = ExperimentCapacity(1024)
        self.temporal = temporal
        with torch.random.fork_rng(devices=[]):
            seed = int.from_bytes(digest_fields(MODEL_VERSION, initialization_seed, "shell"), "big") % 2**63
            torch.random.default_generator.manual_seed(seed)
            self._initialize_shell(512)
        with torch.random.fork_rng(devices=[]):
            seed = int.from_bytes(digest_fields(MODEL_VERSION, initialization_seed, "backbone"), "big") % 2**63
            torch.random.default_generator.manual_seed(seed)
            config = Qwen3Config(**self.graph["resolved_config"])
            config._attn_implementation = "eager"
            self.transformer = _Qwen3Stack(config)
        sequence = sum(parameter.numel() for parameter in self.transformer.parameters())
        total = sum(parameter.numel() for parameter in self.parameters())
        if sequence != self.graph["sequence_parameters"] or total != self.graph["total_parameters"]:
            raise ValueError("Qwen3-ToM-Large parameter count/config mismatch")

    def _temporal_code(self, day_ids, phase_ids):
        code = self.temporal.code(day_ids, phase_ids)
        if code.shape != (*day_ids.shape, 256):
            raise ValueError("Qwen3-ToM-Large requires the original 256-wide temporal code")
        return F.pad(code, (0, 256))
