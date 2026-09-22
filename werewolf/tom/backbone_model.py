"""Frozen four-backbone study graph; native HF blocks without vocabulary parameters."""

from copy import deepcopy
import torch
from torch import nn
from transformers import GPT2Config, Qwen2Config, Qwen3Config, Gemma3TextConfig
from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2PreTrainedModel, GPT2Block
from transformers.models.qwen2.modeling_qwen2 import (Qwen2Model, Qwen2PreTrainedModel,
    Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding)
from transformers.models.qwen3.modeling_qwen3 import (Qwen3Model, Qwen3PreTrainedModel,
    Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding)
from transformers.models.gemma3.modeling_gemma3 import (Gemma3TextModel, Gemma3PreTrainedModel,
    Gemma3DecoderLayer, Gemma3RMSNorm, Gemma3RotaryEmbedding)

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection.speech import V1_ACTIONS
from werewolf.structured_history import STRUCTURED_TOKEN_TYPES
from werewolf.tom.dataset import ExperimentCapacity
from werewolf.tom.model import relative_player_indices
from werewolf.tom.protocol import digest_fields

SHELL_VERSION = "classic7_observer_shell_256_v1"
BACKBONE_VERSION = "classic7_native_continuous_backbone_v1"
ARCHITECTURES = ("gpt2", "qwen2", "qwen3", "gemma3_text")
SEQUENCE_COUNTS = dict(zip(ARCHITECTURES, (2896384, 3150080, 3148288, 3250816)))
SHELL_COUNT = 282631
BACKBONE_CONFIGS = {
    "gpt2": {"n_embd": 256, "n_layer": 4, "n_head": 8, "n_inner": 768,
        "n_positions": 1024, "activation_function": "gelu_new", "layer_norm_epsilon": 1e-5,
        "resid_pdrop": .1, "embd_pdrop": .1, "attn_pdrop": .1,
        "initializer_range": .02, "scale_attn_weights": True,
        "scale_attn_by_inverse_layer_idx": False, "reorder_and_upcast_attn": False,
        "add_cross_attention": False, "use_cache": False},
    "qwen2": {"hidden_size": 256, "num_hidden_layers": 4, "num_attention_heads": 8,
        "num_key_value_heads": 4, "head_dim": 32, "intermediate_size": 768,
        "max_position_embeddings": 1024, "hidden_act": "silu", "rms_norm_eps": 1e-6,
        "attention_dropout": .1, "initializer_range": .02, "use_cache": False,
        "use_sliding_window": False, "sliding_window": None,
        "layer_types": ["full_attention"] * 4,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0}},
    "qwen3": {"hidden_size": 256, "num_hidden_layers": 4, "num_attention_heads": 8,
        "num_key_value_heads": 4, "head_dim": 32, "intermediate_size": 768,
        "max_position_embeddings": 1024, "hidden_act": "silu", "rms_norm_eps": 1e-6,
        "attention_bias": False, "attention_dropout": .1, "initializer_range": .02,
        "use_cache": False, "sliding_window": None, "layer_types": ["full_attention"] * 4,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0}},
    "gemma3_text": {"hidden_size": 256, "num_hidden_layers": 6, "num_attention_heads": 8,
        "num_key_value_heads": 4, "head_dim": 32, "intermediate_size": 448,
        "max_position_embeddings": 1024, "hidden_activation": "gelu_pytorch_tanh",
        "rms_norm_eps": 1e-6, "attention_bias": False, "attention_dropout": .1,
        "initializer_range": .02, "use_bidirectional_attention": False, "use_cache": False,
        "query_pre_attn_scalar": 32, "sliding_window": 256,
        "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
        "attn_logit_softcapping": None, "final_logit_softcapping": None,
        "rope_parameters": {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 1000000.0}}},
}


# Only constructors differ from HF: omit token embeddings, retain native forward/init.
# The study owns continuous inputs only; no tokenizer, LM head, or vision model exists.
class _GPT2Stack(GPT2Model):
    def __init__(self, config):
        GPT2PreTrainedModel.__init__(self, config)
        self.embed_dim = config.hidden_size
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.gradient_checkpointing = False
        self._attn_implementation = config._attn_implementation
        self.post_init()


class _Qwen2Stack(Qwen2Model):
    def __init__(self, config):
        Qwen2PreTrainedModel.__init__(self, config)
        self.layers = nn.ModuleList([Qwen2DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = False
        self.post_init()


class _Qwen3Stack(Qwen3Model):
    def __init__(self, config):
        Qwen3PreTrainedModel.__init__(self, config)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = False
        self.post_init()


class _Gemma3Stack(Gemma3TextModel):
    def __init__(self, config):
        Gemma3PreTrainedModel.__init__(self, config)
        self.layers = nn.ModuleList([Gemma3DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.post_init()


def build_backbone(architecture):
    if architecture not in ARCHITECTURES:
        raise ValueError("unknown backbone architecture")
    values = deepcopy(BACKBONE_CONFIGS[architecture])
    if architecture == "gpt2":
        config = GPT2Config(**values)
        stack = _GPT2Stack
    elif architecture == "qwen2":
        config = Qwen2Config(**values)
        stack = _Qwen2Stack
    elif architecture == "qwen3":
        config = Qwen3Config(**values)
        stack = _Qwen3Stack
    else:
        config = Gemma3TextConfig(**values)
        stack = _Gemma3Stack
    config._attn_implementation = "eager"
    if config.hidden_size != 256 or config.max_position_embeddings != 1024:
        raise ValueError("frozen backbone width/capacity mismatch")
    return stack(config)


def initialization_seeds(study_seed, fold, architecture):
    if (type(study_seed) is not int or not 0 <= study_seed < 2**63
            or type(fold) is not int or not 0 <= fold < 5 or architecture not in ARCHITECTURES):
        raise ValueError("invalid backbone initialization identity")
    return {
        "shell": int.from_bytes(digest_fields("backbone_study_shell_v1", study_seed, fold), "big") % 2**63,
        "backbone": int.from_bytes(digest_fields("backbone_study_native_v1", study_seed, architecture, fold), "big") % 2**63,
    }


def graph_identity(architecture):
    if architecture not in ARCHITECTURES:
        raise ValueError("unknown backbone architecture")
    config = deepcopy(BACKBONE_CONFIGS[architecture])
    graph = {"architecture": architecture, "resolved_config": config,
        "resolved_config_digest": sha256_bytes(canonical_json_bytes(config)),
        "shell_graph_version": SHELL_VERSION, "backbone_graph_version": BACKBONE_VERSION,
        "dtype": "float32", "attention_backend": "eager", "continuous_input_scale": 1,
        "sequence_parameters": SEQUENCE_COUNTS[architecture], "shell_parameters": SHELL_COUNT,
        "total_parameters": SEQUENCE_COUNTS[architecture] + SHELL_COUNT}
    return {**graph, "graph_digest": sha256_bytes(canonical_json_bytes(graph))}


class BackboneToM(nn.Module):
    def __init__(self, architecture, temporal, *, study_seed, fold):
        super().__init__()
        if torch.get_default_dtype() != torch.float32:
            raise ValueError("backbone study requires FP32 construction")
        self.graph = graph_identity(architecture)
        self.capacity = ExperimentCapacity(1024)
        self.temporal = temporal
        seeds = initialization_seeds(study_seed, fold, architecture)
        # Both construction domains restore the caller's RNG; no constructor drift
        # can consume the independent training stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seeds["shell"])
            self.source_embedding = nn.Embedding(8, 256, padding_idx=0)
            self.target_embedding = nn.Embedding(8, 256, padding_idx=0)
            self.action_embedding = nn.Embedding(len(V1_ACTIONS) + 1, 256, padding_idx=0)
            self.event_embedding = nn.Embedding(len(STRUCTURED_TOKEN_TYPES) + 1, 256, padding_idx=0)
            self.observer_embedding = nn.Embedding(7, 256)
            self.source_relative_embedding = nn.Embedding(8, 256, padding_idx=7)
            self.target_relative_embedding = nn.Embedding(8, 256, padding_idx=7)
            self.relation_projection = nn.Linear(2, 256, bias=False)
            self.query_attention = nn.MultiheadAttention(256, 8, batch_first=True)
            self.query_norm = nn.LayerNorm(256)
            self.output = nn.Linear(256, 7)
            for embedding in (self.source_embedding, self.target_embedding, self.action_embedding,
                              self.event_embedding, self.observer_embedding,
                              self.source_relative_embedding, self.target_relative_embedding):
                nn.init.normal_(embedding.weight, std=.02)
                if embedding.padding_idx is not None:
                    with torch.no_grad():
                        embedding.weight[embedding.padding_idx].zero_()
            nn.init.normal_(self.relation_projection.weight, std=.02)
            nn.init.normal_(self.output.weight, std=.02)
            nn.init.zeros_(self.output.bias)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seeds["backbone"])
            self.transformer = build_backbone(architecture)
        sequence = sum(p.numel() for p in self.transformer.parameters())
        total = sum(p.numel() for p in self.parameters())
        if (sequence != self.graph["sequence_parameters"] or total != self.graph["total_parameters"]
                or not .9 * 3150080 <= sequence <= 1.1 * 3150080):
            raise ValueError("frozen backbone parameter count/budget mismatch")

    def forward(self, event_ids, source_ids, action_ids, target_ids, day_ids, phase_ids, attention_mask):
        if self.temporal is None:
            raise ValueError("model forward requires verified temporal artifacts")
        shape = event_ids.shape
        if len(shape) != 2 or not 1 <= shape[1] <= self.capacity.max_seq_len:
            raise ValueError("public sequence shape/capacity mismatch")
        if attention_mask.dtype != torch.bool or attention_mask.shape != shape:
            raise ValueError("attention mask must be matching boolean tensor")
        if day_ids.shape != shape or phase_ids.shape != shape:
            raise ValueError("public temporal shape mismatch")
        lengths = attention_mask.sum(-1)
        expected_mask = torch.arange(shape[1], device=event_ids.device)[None] < lengths[:, None]
        if torch.any(lengths == 0) or not torch.equal(expected_mask, attention_mask):
            raise ValueError("public history must be nonempty and right padded")
        for ids, size in ((event_ids, len(STRUCTURED_TOKEN_TYPES)), (source_ids, 7),
                          (target_ids, 7), (action_ids, len(V1_ACTIONS))):
            if ids.dtype != torch.long or ids.shape != shape or torch.any(ids < 0) or torch.any(ids > size):
                raise ValueError("invalid public token IDs")
        if torch.any(event_ids[attention_mask] == 0):
            raise ValueError("real tokens cannot have padding event IDs")
        base = (self.source_embedding(source_ids) + self.target_embedding(target_ids)
                + self.action_embedding(action_ids) + self.event_embedding(event_ids))
        base = base + self.temporal.code(day_ids, phase_ids).to(base.dtype)
        hidden = self.transformer(inputs_embeds=base, attention_mask=attention_mask.long(), use_cache=False).last_hidden_state
        sources = relative_player_indices(source_ids)
        targets = relative_player_indices(target_ids)
        flags = torch.stack((sources == 0, targets == 0), -1).to(hidden.dtype)
        relative = (hidden[:, None] + self.source_relative_embedding(sources)
                    + self.target_relative_embedding(targets) + self.relation_projection(flags))
        batch, length = shape
        queries = self.observer_embedding(torch.arange(7, device=event_ids.device))[None].expand(batch, -1, -1)
        keys = relative.reshape(batch * 7, length, 256)
        context, _ = self.query_attention(queries.reshape(batch * 7, 1, 256), keys, keys,
            key_padding_mask=(~attention_mask)[:, None].expand(-1, 7, -1).reshape(batch * 7, length), need_weights=False)
        logits = self.output(self.query_norm(queries + context.reshape(batch, 7, 256)))
        diagonal = torch.eye(7, dtype=torch.bool, device=logits.device)
        if not torch.isfinite(logits[:, ~diagonal]).all():
            raise ValueError("non-finite non-self logits")
        logp = logits.masked_fill(diagonal, -torch.inf).log_softmax(-1)
        if not torch.isfinite(logp[:, ~diagonal]).all() or not torch.allclose(logp.exp().sum(-1), torch.ones_like(logp[..., 0]), atol=1e-6):
            raise ValueError("invalid Non-Self Suspicion Simplex output")
        return logp
