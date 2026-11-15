"""仅使用合成公开 PRE 验证 Large 图，不执行优化器或正式训练。"""

import inspect

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from tests.development_publication.test_development_publication import _publication
from werewolf.tom.dataset import CanonicalToMDataset, ExperimentCapacity, PublicTensors
from werewolf.tom.population import select_primary_population
from werewolf.tom.qwen3_large import Qwen3ToMLarge, graph_identity
from werewolf.tom.scoring import game_balanced_cross_entropy
from werewolf.tom.temporal import TemporalCodeProvider, publish_temporal


@pytest.fixture(scope="module")
def example(tmp_path_factory):
    root = tmp_path_factory.mktemp("qwen3-large-synthetic")
    _, handles = _publication(root)
    view = handles.public.public_view
    data = CanonicalToMDataset(view, view.game_ids, ExperimentCapacity(1024))
    sample = data[0]
    parent = "a" * 64
    publish_temporal(root / "temporal", view.max_observed_day, parent)
    temporal = TemporalCodeProvider.explicit(root / "temporal", parent)
    primary = select_primary_population(handles.public)
    return temporal, sample, primary.mask(sample.game_id, sample.boundary_id)


@pytest.fixture(scope="module")
def model(example):
    return Qwen3ToMLarge(example[0], initialization_seed=17)


def public_inputs(sample):
    # 只去掉右侧 Padding，完整保留合成样本的所有公开 PRE token。
    return {name: value[:sample.token_count].unsqueeze(0)
            for name, value in sample.public.kwargs().items()}


def test_large_native_graph_and_measured_parameters(model):
    graph = model.graph
    assert graph == graph_identity()
    assert len(model.transformer.layers) == 8
    assert all(isinstance(layer, Qwen3DecoderLayer) for layer in model.transformer.layers)
    config = model.transformer.config
    for name, value in graph["resolved_config"].items():
        assert getattr(config, name) == value
    assert config.hidden_size == 512 and config.intermediate_size == 1536
    assert (config.num_attention_heads, config.num_key_value_heads, config.head_dim) == (8, 4, 64)
    assert config.layer_types == ["full_attention"] * 8
    assert config.use_cache is False and config.attention_dropout == .1
    assert config._attn_implementation == "eager"
    sequence = sum(parameter.numel() for parameter in model.transformer.parameters())
    total = sum(parameter.numel() for parameter in model.parameters())
    assert sequence == graph["sequence_parameters"] == 25175552
    assert total - sequence == graph["shell_parameters"] == 1089543
    assert total == graph["total_parameters"] == 26265095
    assert all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters())
    assert set(model.state_dict()) == set(dict(model.named_parameters()))
    assert not any(part in name for name in model.state_dict()
                   for part in ("embed_tokens", "lm_head", "wte", "vision"))
    assert not any(isinstance(module, torch.nn.Embedding)
                   for module in model.transformer.modules())


def test_large_temporal_zero_extension_preserves_original_code(model, example):
    temporal, sample, _ = example
    original_day, original_phase = temporal.day.tobytes(), temporal.phase.tobytes()
    public = public_inputs(sample)
    expected = temporal.code(public["day_ids"], public["phase_ids"])
    extended = model._temporal_code(public["day_ids"], public["phase_ids"])
    assert extended.shape == (*public["day_ids"].shape, 512)
    assert torch.equal(extended[..., :256], expected)
    assert torch.count_nonzero(extended[..., 256:]) == 0
    assert not extended.requires_grad
    assert temporal.day.tobytes() == original_day and temporal.phase.tobytes() == original_phase
    assert model.graph["temporal_extension"] == "fixed_right_zero_pad_256_to_512_v1"


def test_large_public_only_forward_backward_and_non_self_simplex(model, example):
    _, sample, primary = example
    model.train()
    model.zero_grad(set_to_none=True)
    public = public_inputs(sample)
    assert set(inspect.signature(model.forward).parameters) == set(PublicTensors.__dataclass_fields__)
    output = model(**public)
    assert output.shape == (1, 7, 7)
    diagonal = torch.eye(7, dtype=torch.bool)
    assert torch.isfinite(output[:, ~diagonal]).all()
    assert torch.isneginf(output[:, diagonal]).all()
    probability = output.exp()
    assert torch.isfinite(probability).all()
    assert torch.count_nonzero(probability[:, diagonal]) == 0
    assert torch.allclose(probability.sum(-1), torch.ones(1, 7), atol=1e-6)
    mask = (primary & sample.label_observed).unsqueeze(0)
    loss = game_balanced_cross_entropy([(output, sample.q.unsqueeze(0), mask)])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(torch.count_nonzero(p.grad) for p in model.transformer.parameters())
    model.zero_grad(set_to_none=True)
    with pytest.raises(TypeError):
        model(**public, role_assignment=["Werewolf"] * 7)


def test_large_rejects_capacity_and_invalid_masks_before_decoder(model, example, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid public input reached decoder")
    monkeypatch.setattr(model.transformer, "forward", unexpected)
    public = public_inputs(example[1])
    too_long = {name: value[:, :1].expand(1, 1025) for name, value in public.items()}
    with pytest.raises(ValueError, match="capacity"):
        model(**too_long)
    with pytest.raises(ValueError, match="boolean"):
        model(**{**public, "attention_mask": public["attention_mask"].long()})
    invalid = public["attention_mask"].clone()
    invalid[:, 0] = False
    with pytest.raises(ValueError, match="right padded"):
        model(**{**public, "attention_mask": invalid})


def test_large_native_causality_and_padding(model):
    model.eval()
    torch.manual_seed(71)
    values = torch.randn(2, 9, 512)
    mask = torch.arange(9)[None] < torch.tensor([9, 5])[:, None]
    changed = values.clone()
    changed[:, 5:] += 100
    with torch.inference_mode():
        original = model.transformer(inputs_embeds=values, attention_mask=mask, use_cache=False).last_hidden_state
        altered = model.transformer(inputs_embeds=changed, attention_mask=mask, use_cache=False).last_hidden_state
    assert original.shape == (2, 9, 512)
    assert torch.equal(original[:, :5], altered[:, :5])
    assert torch.isfinite(original).all()


def test_large_initialization_preserves_rng_and_is_repeatable(model):
    before = torch.get_rng_state().clone()
    repeated = Qwen3ToMLarge(None, initialization_seed=17)
    assert torch.equal(before, torch.get_rng_state())
    assert all(torch.equal(value, repeated.state_dict()[name])
               for name, value in model.state_dict().items())
    with pytest.raises(ValueError, match="temporal"):
        repeated(**{name: torch.ones(1, 1, dtype=torch.bool if name == "attention_mask" else torch.long)
                    for name in PublicTensors.__dataclass_fields__})


@pytest.mark.parametrize("seed", [True, -1, 2**63, 1.5])
def test_large_requires_explicit_valid_initialization_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        Qwen3ToMLarge(None, initialization_seed=seed)
