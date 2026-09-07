"""One random-init Qwen2 graph with public observer-relative readout."""

import torch
from torch import nn
from transformers import Qwen2Config, Qwen2Model

from werewolf.canonical_collection.speech import V1_ACTIONS
from werewolf.structured_history import STRUCTURED_TOKEN_TYPES
from werewolf.tom.dataset import ExperimentCapacity
from werewolf.tom.temporal import TemporalCodeProvider

MODEL_GRAPH_VERSION = "classic7_public_observer_qwen2_v1"
MODEL_GRAPH = {"version": MODEL_GRAPH_VERSION, "hidden_size": 256, "layers": 4, "heads": 8,
               "key_value_heads": 4, "intermediate_size": 768, "attention_dropout": .1,
               "output_contract": "non_self_suspicion_simplex"}


def relative_player_indices(player_ids):
    observers = torch.arange(7, device=player_ids.device).view(1, 7, 1)
    return torch.where(player_ids[:, None] == 0, 7, (player_ids[:, None] - 1 - observers) % 7)


class ObserverConditionedToM(nn.Module):
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
