"""Seven fixed identifiers: three available architectures, four explicitly blocked."""

from dataclasses import dataclass

import torch
from torch import nn
from transformers import GPT2Config, GPT2Model, Qwen2Config, Qwen2Model, Qwen3Config, Qwen3Model


@dataclass(frozen=True)
class BackboneConfig:
    backbone: str
    hidden_size: int = 256
    num_layers: int = 4
    intermediate_size: int = 768
    num_heads: int = 8
    num_key_value_heads: int = 4

    def __post_init__(self):
        if self.backbone == "mamba2":
            raise ValueError("mamba2 BLOCKED: mamba-ssm/causal-conv1d CUDA kernels unavailable; no fallback")
        if self.backbone in ("gated_deltanet", "gated_deltanet2", "qwen35_hybrid"):
            raise ValueError(f"{self.backbone} BLOCKED: no validated dependency/backend path; no fallback")
        if self.backbone not in ("gpt2", "qwen2", "qwen3"):
            raise ValueError("unknown backbone")
        if type(self.hidden_size) is not int or self.hidden_size != 256:
            raise ValueError("architecture study requires hidden_size == 256")
        if any(type(v) is not int or v <= 0 for v in
               (self.num_layers, self.intermediate_size, self.num_heads, self.num_key_value_heads)):
            raise ValueError("backbone dimensions must be positive integers")
        if 256 % self.num_heads or (256 // self.num_heads) % 2:
            raise ValueError("head dimension must be integral and even")
        if self.backbone == "gpt2":
            if self.num_key_value_heads != self.num_heads:
                raise ValueError("GPT2 uses multi-head attention: KV heads must equal heads")
        elif self.num_heads % self.num_key_value_heads:
            raise ValueError("attention heads must be divisible by KV heads")


class SequenceBackbone(nn.Module):
    """[B,L,256] + nonempty right-padding bool mask -> [B,L,256]."""

    def __init__(self, config: BackboneConfig, max_seq_len: int):
        super().__init__()
        if type(max_seq_len) is not int or max_seq_len <= 0:
            raise ValueError("max_seq_len must be a positive integer")
        self.max_seq_len = max_seq_len
        if config.backbone == "gpt2":
            hf = GPT2Config(vocab_size=1, n_positions=max_seq_len, n_embd=256,
                n_layer=config.num_layers, n_head=config.num_heads, n_inner=config.intermediate_size,
                activation_function="gelu_new", resid_pdrop=.1, embd_pdrop=.1,
                attn_pdrop=.1, layer_norm_epsilon=1e-5, use_cache=False,
                bos_token_id=None, eos_token_id=None, pad_token_id=None)
            hf._attn_implementation = "eager"
            self.model = GPT2Model(hf)
            self.model.wte = None
        else:
            kwargs = dict(vocab_size=1, hidden_size=256, intermediate_size=config.intermediate_size,
                num_hidden_layers=config.num_layers, num_attention_heads=config.num_heads,
                num_key_value_heads=config.num_key_value_heads, hidden_act="silu",
                max_position_embeddings=max_seq_len, attention_dropout=.1, rms_norm_eps=1e-6,
                use_cache=False, bos_token_id=0, eos_token_id=0, pad_token_id=0)
            if config.backbone == "qwen2":
                hf = Qwen2Config(**kwargs)
                model_type = Qwen2Model
            else:
                # Qwen3 defaults to head_dim=128, independent of hidden size.
                hf = Qwen3Config(**kwargs, head_dim=256 // config.num_heads)
                model_type = Qwen3Model
            hf._attn_implementation = "eager"
            self.model = model_type(hf)
            self.model.embed_tokens = None

    def forward(self, hidden_states, attention_mask):
        if (hidden_states.ndim != 3 or hidden_states.shape[-1] != 256
                or not 1 <= hidden_states.shape[1] <= self.max_seq_len):
            raise ValueError("backbone input must be [B,L,256] within capacity")
        if attention_mask.dtype != torch.bool or attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError("backbone mask must be matching boolean tensor")
        lengths = attention_mask.sum(-1)
        expected = torch.arange(hidden_states.shape[1], device=attention_mask.device)[None] < lengths[:, None]
        if torch.any(lengths == 0) or not torch.equal(attention_mask, expected):
            raise ValueError("backbone history must be nonempty and right padded")
        return self.model(inputs_embeds=hidden_states, attention_mask=attention_mask.long(),
                          use_cache=False).last_hidden_state
