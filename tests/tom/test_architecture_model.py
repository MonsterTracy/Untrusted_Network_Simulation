"""Offline, synthetic architecture checks; no formal study or trained state."""
import inspect
import socket

import pytest
import torch
from transformers import PreTrainedModel

from werewolf.tom.architecture_model import ArchitectureToM
from werewolf.tom.sequence_backbones import BackboneConfig
from werewolf.tom.dataset import ExperimentCapacity, CanonicalToMDataset
from werewolf.tom.model import ObserverConditionedToM
from werewolf.tom.temporal import publish_temporal, TemporalCodeProvider

FAMILIES = ('gpt2', 'qwen2', 'qwen3')


def config(family, **kwargs):
    return BackboneConfig(family, num_key_value_heads=8 if family == 'gpt2' else 4, **kwargs)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def prohibited(*a, **kw):
        pytest.fail('network/pretrained loading forbidden')
    monkeypatch.setattr(socket.socket, 'connect', prohibited)
    monkeypatch.setattr(PreTrainedModel, 'from_pretrained', prohibited)
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def temporal(tmp_path):
    publish_temporal(tmp_path/'temporal', 2, 'a'*64)
    return {c: getattr(TemporalCodeProvider, c)(tmp_path/'temporal', 'a'*64)
            for c in ('implicit', 'explicit')}


def public():
    return dict(event_ids=torch.ones(2, 6, dtype=torch.long),
        source_ids=torch.tensor([[1,2,3,4,5,6],[1,2,3,0,0,0]]),
        target_ids=torch.tensor([[2,3,4,5,6,7],[3,4,5,0,0,0]]),
        action_ids=torch.ones(2, 6, dtype=torch.long),
        day_ids=torch.ones(2, 6, dtype=torch.long),
        phase_ids=torch.ones(2, 6, dtype=torch.long),
        attention_mask=torch.tensor([[True]*6,[True]*3+[False]*3]))


@pytest.mark.parametrize('family', FAMILIES)
def test_interface_causal_padding_and_initialization(family):
    cfg = config(family)
    rng = torch.get_rng_state().clone()
    a = ArchitectureToM(ExperimentCapacity(64), None, cfg, seed=19).eval()
    assert torch.equal(torch.get_rng_state(), rng)
    b = ArchitectureToM(ExperimentCapacity(64), None, cfg, seed=19).eval()
    assert set(a.state_dict()) == set(b.state_dict())
    assert all(torch.equal(v, b.state_dict()[k]) for k,v in a.state_dict().items())
    c = ArchitectureToM(ExperimentCapacity(64), None, cfg, seed=20)
    assert any(not torch.equal(v, c.state_dict()[k]) for k,v in a.state_dict().items())
    x = torch.randn(2,6,256); mask = public()['attention_mask']
    with torch.inference_mode():
        y = a.backbone(x,mask)
        assert y.shape == x.shape and torch.isfinite(y).all()
        future=x.clone(); future[:,3:]+=11
        assert torch.equal(y[:,:3],a.backbone(future,mask)[:,:3])
        padding=x.clone(); padding[1,3:]-=7
        assert torch.equal(y[1,:3],a.backbone(padding,mask)[1,:3])
    for invalid in (torch.zeros_like(mask), ~mask):
        with pytest.raises(ValueError, match='right padded'):
            a.backbone(x,invalid)
    with pytest.raises(ValueError, match='boolean'):
        a.backbone(x,mask.long())
    assert not any('embed_tokens' in k or '.wte.' in k or 'lm_head' in k for k in a.state_dict())


@pytest.mark.parametrize('family,expected_backbone', [
    ('gpt2',2896384),('qwen2',3150080),('qwen3',3148288)])
def test_parameter_counts_at_1024(family,expected_backbone):
    model=ArchitectureToM(ExperimentCapacity(1024),None,config(family),seed=19)
    total=sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone=sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    assert backbone == expected_backbone
    assert total-backbone == 282631
    print(f'{family}: backbone={backbone}, shared={total-backbone}, total={total}, max_seq_len=1024')
    if family=='gpt2':
        from transformers.models.gpt2.modeling_gpt2 import GPT2Block
        decoder=model.backbone.model
        assert decoder.wpe.weight.shape==(1024,256)
        assert decoder.drop.p==.1
        assert all(isinstance(block,GPT2Block) for block in decoder.h)
        assert all(block.attn.attn_dropout.p==block.attn.resid_dropout.p==block.mlp.dropout.p==.1
                   for block in decoder.h)


@pytest.mark.parametrize('condition', ('implicit','explicit'))
def test_qwen2_bitwise_backbone_and_complete_shell_parity(temporal, condition):
    original = ObserverConditionedToM(ExperimentCapacity(64),temporal[condition]).eval()
    study = ArchitectureToM(ExperimentCapacity(64),temporal[condition],config('qwen2'),seed=91).eval()
    mapped={('backbone.model.'+k[len('transformer.'):] if k.startswith('transformer.') else k):v
            for k,v in original.state_dict().items()}
    study.load_state_dict(mapped,strict=True)
    data=public()
    base=(original.source_embedding(data['source_ids'])+original.target_embedding(data['target_ids'])
          +original.action_embedding(data['action_ids'])+original.event_embedding(data['event_ids']))
    base=base+temporal[condition].code(data['day_ids'],data['phase_ids']).to(base.dtype)
    with torch.inference_mode():
        ref=original.transformer(inputs_embeds=base,attention_mask=data['attention_mask'].long(),use_cache=False).last_hidden_state
        assert torch.equal(ref,study.backbone(base,data['attention_mask']))
        assert torch.equal(original(**data),study(**data))


@pytest.mark.parametrize('condition', ('implicit','explicit'))
def test_outer_initialization_and_frozen_shell_for_every_backbone(temporal,condition):
    models=[ArchitectureToM(ExperimentCapacity(64),temporal[condition],config(f),seed=17).eval() for f in FAMILIES]
    outer={k:v for k,v in models[0].state_dict().items() if not k.startswith('backbone.')}
    for model in models:
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in outer.items())
        assert set(inspect.signature(model.forward).parameters)==set(public())
        # Independent oracle: the untouched Full shell, with only its sequence
        # computation delegated to the candidate. No readout assertions are weakened.
        oracle=ObserverConditionedToM(ExperimentCapacity(64),temporal[condition]).eval()
        oracle.load_state_dict({**oracle.state_dict(),**outer},strict=True)
        def sequence(*,inputs_embeds,attention_mask,use_cache):
            from types import SimpleNamespace
            return SimpleNamespace(last_hidden_state=model.backbone(inputs_embeds,attention_mask.bool()))
        oracle.transformer.forward=sequence
        data=public()
        with torch.inference_mode():
            output=model(**data)
            assert torch.equal(output,oracle(**data))
            assert output.shape==(2,7,7)
            assert torch.all(output.exp().diagonal(dim1=-2,dim2=-1)==0)
            assert torch.allclose(output.exp().sum(-1),torch.ones(2,7))
            assert not torch.equal(output[:,0],output[:,1])
            changed={k:v.clone() for k,v in data.items()}
            changed['source_ids'][1,3:]=7
            changed['target_ids'][1,3:]=6
            assert torch.equal(output[1],model(**changed)[1])


@pytest.mark.parametrize('family', FAMILIES)
def test_explicit_depth_and_ffn_configuration(family):
    model=ArchitectureToM(ExperimentCapacity(16),None,config(family,num_layers=2,intermediate_size=512,num_heads=8),seed=1)
    hf=model.backbone.model.config
    assert hf.num_hidden_layers==2 and hf.hidden_size==256
    assert (hf.n_inner if family=='gpt2' else hf.intermediate_size)==512


@pytest.mark.parametrize('family', FAMILIES)
def test_legal_alternative_heads(family):
    cfg=BackboneConfig(family,num_layers=1,num_heads=4,
                       num_key_value_heads=4 if family=='gpt2' else 2)
    model=ArchitectureToM(ExperimentCapacity(8),None,cfg,seed=3).eval()
    assert model.backbone(torch.randn(1,3,256),torch.ones(1,3,dtype=torch.bool)).shape==(1,3,256)


@pytest.mark.parametrize('options', [dict(num_heads=3),dict(num_layers=0),
                                    dict(num_heads=8,num_key_value_heads=3)])
def test_invalid_architecture_dimensions(options):
    with pytest.raises(ValueError):
        BackboneConfig('qwen3',**options)


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('width', (128,512))
def test_width_fails_closed(family,width):
    with pytest.raises(ValueError,match='hidden_size == 256'):
        config(family,hidden_size=width)


@pytest.mark.parametrize('family', ('mamba2','gated_deltanet','gated_deltanet2','qwen35_hybrid','unknown'))
def test_unavailable_or_unknown_no_fallback(family):
    with pytest.raises(ValueError,match='BLOCKED' if family!='unknown' else 'unknown'):
        BackboneConfig(family)


@pytest.mark.parametrize('family', FAMILIES)
def test_existing_fit_steps_with_real_synthetic_dataset(tmp_path,family):
    from tests.development_publication.test_development_publication import _publication
    from werewolf.tom.population import select_primary_population
    from werewolf.tom.training import fit_steps
    _,handles=_publication(tmp_path)
    view=handles.public.public_view
    capacity=ExperimentCapacity(view.max_structured_token_count)
    data=CanonicalToMDataset(view,view.game_ids[:1],capacity)
    sample=data[0]
    masks=select_primary_population(handles.public).rows
    publish_temporal(tmp_path/'temporal',view.max_observed_day,'b'*64)
    provider=TemporalCodeProvider.implicit(tmp_path/'temporal','b'*64)
    model=ArchitectureToM(capacity,provider,config(family,num_layers=1),seed=37)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    before=model.output.weight.detach().clone()
    logs=[]
    schedule={'optimizer_steps':1,'batches':[[{'game_id':sample.game_id,'shift':1}]]}
    assert fit_steps(model,optimizer,{sample.game_id:[sample]},masks,schedule,
        device='cpu',learning_rate=.001,start=0,logs=logs,gate=lambda:None,on_step=lambda *a:None)==1
    assert len(logs)==1 and torch.isfinite(torch.tensor(logs[0]['loss']))
    assert not torch.equal(before,model.output.weight)
