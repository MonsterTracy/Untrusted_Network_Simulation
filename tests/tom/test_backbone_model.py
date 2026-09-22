"""Synthetic graph contracts; no publication results or formal training."""
from copy import deepcopy

import pytest
import torch

from werewolf.tom import backbone_model as m
from werewolf.tom.dataset import ExperimentCapacity
from werewolf.tom.model import ObserverConditionedToM
from werewolf.tom.temporal import TemporalCodeProvider, publish_temporal
from werewolf.tom.state import model_tensor_values, restore_model_tensors
from werewolf.artifact_io import publish_tensor_state, verify_tensor_state


@pytest.fixture
def temporal(tmp_path):
    publish_temporal(tmp_path / 'temporal', 3, 'a' * 64)
    return tmp_path / 'temporal'


def public(length):
    ids = torch.ones(2, length, dtype=torch.long)
    mask = torch.arange(length)[None] < torch.tensor([length, max(1, length - 2)])[:, None]
    return dict(event_ids=ids, source_ids=ids, action_ids=ids, target_ids=ids,
                day_ids=ids, phase_ids=ids, attention_mask=mask)


@pytest.mark.parametrize('architecture,sequence,total', [
    ('gpt2', 2896384, 3179015), ('qwen2', 3150080, 3432711),
    ('qwen3', 3148288, 3430919), ('gemma3_text', 3250816, 3533447)])
def test_exact_native_graph_and_parameter_ownership(architecture, sequence, total):
    model = m.BackboneToM(architecture, None, study_seed=17, fold=0)
    stack = model.transformer
    assert sum(p.numel() for p in stack.parameters()) == sequence
    assert sum(p.numel() for p in model.parameters()) == total
    assert total - sequence == 282631
    assert .9 * 3150080 <= sequence <= 1.1 * 3150080
    assert set(model.state_dict()) == set(dict(model.named_parameters()))
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert not any(any(part in n for part in ('wte', 'embed_tokens', 'lm_head', 'vision')) for n in model.state_dict())
    embeddings = [n for n, module in stack.named_modules() if isinstance(module, torch.nn.Embedding)]
    assert embeddings == (['wpe'] if architecture == 'gpt2' else [])
    assert stack.config._attn_implementation == 'eager'
    assert stack.config.use_cache is False
    assert stack.config.hidden_size == 256
    assert stack.config.max_position_embeddings == 1024
    for key, value in m.BACKBONE_CONFIGS[architecture].items():
        assert getattr(stack.config, key) == value
    optimizer = torch.optim.AdamW(model.parameters())
    assert {id(p) for g in optimizer.param_groups for p in g['params']} == {id(p) for p in model.parameters()}


@pytest.mark.parametrize('architecture', m.ARCHITECTURES)
def test_causal_padding_and_deterministic_forward(architecture):
    torch.manual_seed(90)
    stack = m.build_backbone(architecture).eval()
    x = torch.randn(2, 9, 256)
    mask = torch.arange(9)[None] < torch.tensor([9, 5])[:, None]
    with torch.no_grad():
        a = stack(inputs_embeds=x, attention_mask=mask, use_cache=False).last_hidden_state
        b = stack(inputs_embeds=x, attention_mask=mask, use_cache=False).last_hidden_state
        changed = x.clone()
        changed[:, 5:] = torch.randn_like(changed[:, 5:]) * 100
        c = stack(inputs_embeds=changed, attention_mask=mask, use_cache=False).last_hidden_state
    assert a.shape == (2, 9, 256)
    assert torch.equal(a, b)
    assert torch.equal(a[:, :5], c[:, :5])
    assert torch.isfinite(a).all()


def test_gemma_native_window_pattern_and_unscaled_inputs():
    stack = m.build_backbone('gemma3_text').eval()
    assert stack.config.layer_types == ['sliding_attention'] * 5 + ['full_attention']
    assert stack.config.sliding_window == 256
    assert [layer.self_attn.sliding_window for layer in stack.layers] == [256] * 5 + [None]
    x = torch.randn(1, 258, 256)
    observed = []
    masks = []
    def capture(module, args, kwargs):
        observed.append(args[0].detach().clone())
        masks.append(kwargs['attention_mask'].detach().clone())
    hooks = [layer.register_forward_pre_hook(capture, with_kwargs=True) for layer in stack.layers]
    with torch.no_grad():
        result = stack(inputs_embeds=x, attention_mask=torch.ones(1, 258, dtype=torch.bool), use_cache=False)
    for hook in hooks:
        hook.remove()
    assert torch.equal(observed[0], x)  # not x * sqrt(256)
    assert result.last_hidden_state.shape == x.shape
    for mask in masks[:5]:
        assert mask[0, 0, 257, 1] < -1e10
        assert mask[0, 0, 257, 2] == 0
        assert mask[0, 0, 0, 1] < -1e10
    assert masks[5][0, 0, 257, 0] == 0


def test_fold_shell_pairing_rng_isolation_and_native_initialization(monkeypatch):
    torch.manual_seed(771)
    before = torch.get_rng_state().clone()
    models = [m.BackboneToM(a, None, study_seed=77, fold=2) for a in m.ARCHITECTURES]
    assert torch.equal(before, torch.get_rng_state())
    shell = {k: v for k, v in models[0].state_dict().items() if not k.startswith('transformer.')}
    for model in models[1:]:
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in shell.items())
    other = m.BackboneToM('qwen2', None, study_seed=77, fold=3)
    assert not torch.equal(shell['observer_embedding.weight'], other.observer_embedding.weight)
    build = m.build_backbone
    def consume_more(architecture):
        torch.rand(1000)
        return build(architecture)
    monkeypatch.setattr(m, 'build_backbone', consume_more)
    perturbed = m.BackboneToM('qwen2', None, study_seed=77, fold=2)
    assert all(torch.equal(v, perturbed.state_dict()[k]) for k, v in shell.items())
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(models[3].transformer.norm.weight, torch.zeros(256))
    assert torch.equal(models[1].transformer.norm.weight, torch.ones(256))
    # Native initialization receives only the backbone seed, independently of shell work.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(m.initialization_seeds(77, 2, 'qwen2')['backbone'])
        native = build('qwen2')
    assert all(torch.equal(v, native.state_dict()[k]) for k, v in models[1].transformer.state_dict().items())


@pytest.mark.parametrize('explicit', [False, True])
@pytest.mark.parametrize('length', [3, 17, 1024])
def test_qwen2_strict_parity_and_tensor_roundtrip(temporal, tmp_path, explicit, length):
    provider = TemporalCodeProvider(temporal, 'a' * 64, explicit)
    old = ObserverConditionedToM(ExperimentCapacity(1024), provider).eval()
    model = m.BackboneToM('qwen2', provider, study_seed=13, fold=1).eval()
    model.load_state_dict(old.state_dict(), strict=True)
    x = torch.randn(2, length, 256)
    inputs = public(length)
    with torch.no_grad():
        assert torch.equal(old.transformer(inputs_embeds=x, attention_mask=inputs['attention_mask'], use_cache=False).last_hidden_state,
                           model.transformer(inputs_embeds=x, attention_mask=inputs['attention_mask'], use_cache=False).last_hidden_state)
        expected = old(**inputs)
        assert torch.equal(expected, model(**inputs))
    publish_tensor_state(tmp_path / 'state', manifest_fields={
        'artifact_type': 'synthetic_parity', 'schema_version': 'v1'}, tensors=model_tensor_values(model))
    state = verify_tensor_state(tmp_path / 'state', expected_artifact_type='synthetic_parity', expected_schema_version='v1')
    restored = m.BackboneToM('qwen2', provider, study_seed=99, fold=4).eval()
    restore_model_tensors(restored, state.tensors)
    with torch.no_grad():
        assert torch.equal(expected, restored(**inputs))


def test_unknown_architecture_width_and_non_fp32_fail_closed(monkeypatch):
    with pytest.raises(ValueError, match='unknown'):
        m.BackboneToM('mamba2', None, study_seed=1, fold=0)
    config = deepcopy(m.BACKBONE_CONFIGS)
    config['qwen3']['hidden_size'] = 128
    monkeypatch.setattr(m, 'BACKBONE_CONFIGS', config)
    with pytest.raises(ValueError, match='width'):
        m.build_backbone('qwen3')
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(ValueError, match='FP32'):
            m.BackboneToM('qwen2', None, study_seed=1, fold=0)
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize('architecture', m.ARCHITECTURES)
def test_complete_shell_forward_backward_and_rejected_masks(architecture, temporal):
    from werewolf.tom.scoring import game_balanced_cross_entropy
    provider = TemporalCodeProvider.explicit(temporal, 'a' * 64)
    model = m.BackboneToM(architecture, provider, study_seed=31, fold=0)
    inputs = public(7)
    output = model(**inputs)
    assert output.shape == (2, 7, 7)
    assert torch.equal(output.exp().diagonal(dim1=-2, dim2=-1), torch.zeros(2, 7))
    q = torch.zeros_like(output)
    for observer in range(7):
        q[:, observer, (observer + 1) % 7] = 1
    loss = game_balanced_cross_entropy([(output, q, torch.ones(2, 7, dtype=torch.bool))])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(torch.count_nonzero(p.grad) for p in model.transformer.parameters())
    invalid = inputs['attention_mask'].clone()
    invalid[:, 0] = False
    with pytest.raises(ValueError, match='right padded'):
        model(**{**inputs, 'attention_mask': invalid})
    with pytest.raises(ValueError, match='boolean'):
        model(**{**inputs, 'attention_mask': inputs['attention_mask'].long()})
