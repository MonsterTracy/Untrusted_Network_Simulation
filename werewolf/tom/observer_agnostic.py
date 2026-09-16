"""Shared absolute-seat logits; observer identity appears only in self exclusion."""

import torch
from torch import nn
from transformers import Qwen2Config, Qwen2Model

from werewolf.canonical_collection.speech import V1_ACTIONS
from werewolf.structured_history import STRUCTURED_TOKEN_TYPES
from werewolf.tom.dataset import ExperimentCapacity
from werewolf.tom.temporal import TemporalCodeProvider

MODEL_GRAPH_VERSION = "classic7_observer_agnostic_public_history_v1"
MODEL_GRAPH = {"version": MODEL_GRAPH_VERSION, "hidden_size": 256, "layers": 4, "heads": 8,
               "key_value_heads": 4, "intermediate_size": 768, "attention_dropout": .1,
               "readout": "single_shared_query", "seat_space": "absolute",
               "self_exclusion": "post_readout_masked_log_softmax",
               "output_contract": "non_self_suspicion_simplex"}


class ObserverAgnosticToM(nn.Module):
    def __init__(self, capacity: ExperimentCapacity, temporal: TemporalCodeProvider | None):
        super().__init__()
        self.capacity = capacity
        self.temporal = temporal  # immutable external bytes, not a buffer/parameter
        self.source_embedding = nn.Embedding(8, 256, padding_idx=0)
        self.target_embedding = nn.Embedding(8, 256, padding_idx=0)
        self.action_embedding = nn.Embedding(len(V1_ACTIONS) + 1, 256, padding_idx=0)
        self.event_embedding = nn.Embedding(len(STRUCTURED_TOKEN_TYPES) + 1, 256, padding_idx=0)
        config = Qwen2Config(vocab_size=1, hidden_size=MODEL_GRAPH["hidden_size"], intermediate_size=MODEL_GRAPH["intermediate_size"],
            num_hidden_layers=MODEL_GRAPH["layers"], num_attention_heads=MODEL_GRAPH["heads"], num_key_value_heads=MODEL_GRAPH["key_value_heads"],
            hidden_act="silu", max_position_embeddings=capacity.max_seq_len,
            attention_dropout=MODEL_GRAPH["attention_dropout"], rms_norm_eps=1e-6, use_cache=False,
            bos_token_id=0, eos_token_id=0, pad_token_id=0)
        config._attn_implementation = "eager"
        self.transformer = Qwen2Model(config)
        self.transformer.embed_tokens = None  # input is exclusively structured embeddings
        self.shared_query = nn.Parameter(torch.empty(1, 256))
        self.query_attention = nn.MultiheadAttention(256, 8, batch_first=True)
        self.query_norm = nn.LayerNorm(256)
        self.output = nn.Linear(256, 7)
        for embedding in (self.source_embedding, self.target_embedding, self.action_embedding,
                          self.event_embedding):
            nn.init.normal_(embedding.weight, std=.02)
            if embedding.padding_idx is not None:
                with torch.no_grad():
                    embedding.weight[embedding.padding_idx].zero_()
        nn.init.normal_(self.shared_query, std=.02)
        nn.init.normal_(self.output.weight, std=.02)
        nn.init.zeros_(self.output.bias)

    def shared_logits(self, event_ids, source_ids, action_ids, target_ids, day_ids, phase_ids, attention_mask):
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
        query = self.shared_query[None].expand(shape[0], -1, -1)
        context, _ = self.query_attention(query, hidden, hidden,
            key_padding_mask=~attention_mask, need_weights=False)
        logits = self.output(self.query_norm(query + context)).squeeze(1)
        if not torch.isfinite(logits).all():
            raise ValueError("non-finite shared logits")
        return logits

    def forward(self, event_ids, source_ids, action_ids, target_ids, day_ids, phase_ids, attention_mask):
        logits = self.shared_logits(event_ids, source_ids, action_ids, target_ids,
                                    day_ids, phase_ids, attention_mask)
        return non_self_log_probabilities(logits)


def non_self_log_probabilities(logits):
    """Parameter-free absolute-seat projection; never remap seats by observer."""
    if logits.ndim != 2 or logits.shape[-1] != 7 or not torch.isfinite(logits).all():
        raise ValueError("shared logits must be finite [batch, seven absolute seats]")
    diagonal = torch.eye(7, dtype=torch.bool, device=logits.device)
    logp = logits[:, None, :].expand(-1, 7, -1).masked_fill(diagonal, -torch.inf).log_softmax(-1)
    if not torch.isfinite(logp[:, ~diagonal]).all() or not torch.allclose(
            logp.exp().sum(-1), torch.ones_like(logp[..., 0]), atol=1e-6):
        raise ValueError("invalid Non-Self Suspicion Simplex output")
    return logp
