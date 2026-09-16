"""Synthetic public tensors and initial states only; no scientific publication access."""
import inspect
from itertools import combinations

import pytest
import torch

from werewolf.tom.dataset import ExperimentCapacity, PublicTensors
from werewolf.tom.model import ObserverConditionedToM, MODEL_GRAPH as FULL_GRAPH
from werewolf.tom.observer_agnostic import ObserverAgnosticToM, non_self_log_probabilities
from werewolf.tom.observer_agnostic_state import (
    SHARED_KEYS, FULL_ONLY_KEYS, publish_initial_state, load_initial_state, query_initialization,
)
from werewolf.tom.protocol import digest_fields
from werewolf.tom.state import publish_model_state
from werewolf.tom.temporal import publish_temporal, TemporalCodeProvider


@pytest.fixture(scope="module")
def public():
    return PublicTensors(event_ids=torch.tensor([[1,2,1]]), source_ids=torch.tensor([[1,3,7]]),
        action_ids=torch.tensor([[0,1,0]]), target_ids=torch.tensor([[0,5,2]]),
        day_ids=torch.tensor([[0,1,1]]), phase_ids=torch.tensor([[0,1,1]]),
        attention_mask=torch.ones(1,3,dtype=torch.bool))


@pytest.fixture(scope="module")
def temporal(tmp_path_factory):
    path = tmp_path_factory.mktemp("agnostic-temporal")/'codes'
    publish_temporal(path,1,'a'*64)
    return {c: constructor(path,'a'*64) for c,constructor in (
        ('implicit',TemporalCodeProvider.implicit), ('explicit_day_phase',TemporalCodeProvider.explicit))}


@pytest.fixture(scope="module")
def model(temporal):
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(321)
        return ObserverAgnosticToM(ExperimentCapacity(3), temporal['implicit']).eval()


def test_no_observer_input_test(model, public):
    assert set(inspect.signature(model.shared_logits).parameters) == set(public.kwargs())
    with torch.no_grad():
        reference = model.shared_logits(**public.kwargs())
        for requested_observer in range(7):
            z = model.shared_logits(**public.kwargs())
            assert torch.equal(reference,z)
            assert torch.equal(model(**public.kwargs())[:,requested_observer],
                               non_self_log_probabilities(reference)[:,requested_observer])
    with pytest.raises(TypeError): model.shared_logits(**public.kwargs(), observer_id=0)


def test_common_target_odds_invariance(model, public):
    with torch.no_grad(): p = model(**public.kwargs()).exp()[0].double()
    for i1,i2 in combinations(range(7),2):
        for j,k in combinations([j for j in range(7) if j not in (i1,i2)],2):
            torch.testing.assert_close(p[i1,j]/p[i1,k],p[i2,j]/p[i2,k],rtol=1e-6,atol=1e-7)


def test_diagonal_and_simplex(model, public):
    with torch.no_grad(): logp = model(**public.kwargs())
    diagonal=torch.eye(7,dtype=torch.bool)
    assert logp.shape == (1,7,7)
    assert torch.isneginf(logp[:,diagonal]).all()
    assert torch.isfinite(logp[:,~diagonal]).all()
    assert torch.equal(logp.exp()[:,diagonal],torch.zeros(1,7))
    torch.testing.assert_close(logp.exp().sum(-1),torch.ones(1,7),rtol=1e-6,atol=1e-7)


def test_absolute_seat_semantics():
    z=torch.tensor([[.1,.2,.4,.8,1.6,3.2,6.4]],dtype=torch.float64)
    p=non_self_log_probabilities(z).exp()
    for i in range(7):
        expected=z[0].exp(); expected[i]=0; expected/=expected.sum()
        torch.testing.assert_close(p[0,i],expected,rtol=1e-12,atol=1e-14)
        assert p[0,i].argmax().item() == (5 if i==6 else 6)


def test_forbidden_modules_and_parameter_counts(model):
    forbidden=('observer_embedding','source_relative_embedding','target_relative_embedding','relation_projection')
    assert not any(word in name for name in (*model.state_dict(), *dict(model.named_modules())) for word in forbidden)
    with torch.random.fork_rng(devices=[]): full=ObserverConditionedToM(ExperimentCapacity(3),None)
    assert set(full.state_dict()) == set(SHARED_KEYS)|FULL_ONLY_KEYS
    assert set(model.state_dict()) == set(SHARED_KEYS)|{'shared_query'}
    assert sum(p.numel() for p in full.parameters()) == 3432711
    assert sum(p.numel() for p in model.parameters()) == 3426567


def test_public_only_boundary(model,public):
    with torch.no_grad(): before=model(**public.kwargs())
    for key in ('q','role_truth','primary_eligibility','private_evidence'):
        for value in (None,torch.ones(7,7),{'player1':'Werewolf'}):
            with pytest.raises(TypeError): model(**public.kwargs(),**{key:value})
            with torch.no_grad(): assert torch.equal(before,model(**public.kwargs()))


@pytest.mark.parametrize('change', ['capacity','left_padding','empty','temporal','source'])
def test_public_validation_same_fail_closed_contract(model,public,change):
    values={k:v.clone() for k,v in public.kwargs().items()}
    if change=='capacity': values={k:torch.cat((v,v),-1) for k,v in values.items()}
    elif change=='left_padding': values['attention_mask'][0,0]=False
    elif change=='empty': values['attention_mask'].zero_()
    elif change=='temporal': values['day_ids'][0,0]=2
    else: values['source_ids'][0,0]=8
    with pytest.raises(ValueError): model(**values)


@pytest.fixture
def initial(tmp_path):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        full=ObserverConditionedToM(ExperimentCapacity(3),None)
    provenance={'purpose':'initial','protocol_digest':'a'*64,'fold':0,
        'initialization_seed':int.from_bytes(digest_fields('paired_initial_v1',101,0),'big')%2**63,
        'model_graph':FULL_GRAPH}
    parent=publish_model_state(tmp_path/'full',full,provenance)
    baseline=publish_initial_state(parent.artifact.path,parent.artifact.manifest_digest,tmp_path/'baseline',
                                   initialization_seed=101,fold=0)
    return full,parent,baseline


def test_shared_initial_parameters_and_baseline_temporal_pairing(initial,temporal):
    full,parent,state=initial
    models=[]
    for condition in ('implicit','explicit_day_phase'):
        model=ObserverAgnosticToM(ExperimentCapacity(3),temporal[condition])
        loaded=load_initial_state(state.artifact.path,model,state.artifact.manifest_digest,
                                 full_initial_path=parent.artifact.path)
        assert loaded.artifact.manifest_digest == state.artifact.manifest_digest
        for key in SHARED_KEYS: assert torch.equal(full.state_dict()[key],model.state_dict()[key])
        models.append(model)
    assert all(torch.equal(v,models[1].state_dict()[k]) for k,v in models[0].state_dict().items())
    p=state.artifact.manifest['provenance']
    assert p['parent_full_initial_state_digest']==parent.artifact.manifest_digest
    assert len(p['parameter_mapping'])==len(SHARED_KEYS)
    assert set(p['shared_tensor_digests'])==set(SHARED_KEYS)


def test_query_initialization():
    rng=torch.get_rng_state().clone()
    seed,a=query_initialization(101,0)
    again,b=query_initialization(101,0)
    other,c=query_initialization(101,1)
    assert seed==again and seed!=other and torch.equal(a,b) and not torch.equal(a,c)
    assert torch.equal(rng,torch.get_rng_state())
    expected=torch.empty(1,256).normal_(0,.02,generator=torch.Generator().manual_seed(seed))
    assert torch.equal(a,expected)
    for initial_seed,fold in ((True,0),(-1,0),(101,True),(101,5)):
        with pytest.raises(ValueError): query_initialization(initial_seed,fold)


def test_shared_initialization_is_independent_of_ambient_rng(initial,tmp_path):
    full,parent,first=initial
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(876)
        torch.randn(111)
        second=publish_initial_state(parent.artifact.path,parent.artifact.manifest_digest,tmp_path/'second',
                                     initialization_seed=101,fold=0)
    assert first.artifact.manifest_digest==second.artifact.manifest_digest


def test_baseline_uses_existing_ce_training_interface(public,temporal):
    from werewolf.tom.scoring import game_balanced_cross_entropy
    with torch.random.fork_rng(devices=[]):
        model=ObserverAgnosticToM(ExperimentCapacity(3),temporal['implicit'])
        q=torch.zeros(1,7,7)
        for observer in range(7): q[0,observer,(observer+1)%7]=1
        logp=model(**public.kwargs())
        loss=game_balanced_cross_entropy([(logp,q,torch.ones(1,7,dtype=torch.bool))])
        assert torch.isfinite(loss)
        loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_reject_trained_parent_or_wrong_binding(initial,tmp_path):
    full,parent,state=initial
    p=dict(parent.artifact.manifest['provenance']); p['purpose']='terminal'
    trained=publish_model_state(tmp_path/'trained',full,p)
    with pytest.raises(ValueError,match='untrained'):
        publish_initial_state(trained.artifact.path,trained.artifact.manifest_digest,tmp_path/'bad',initialization_seed=101,fold=0)
    with pytest.raises(ValueError,match='digest'):
        publish_initial_state(parent.artifact.path,'b'*64,tmp_path/'bad',initialization_seed=101,fold=0)
    with pytest.raises(ValueError,match='untrained'):
        publish_initial_state(parent.artifact.path,parent.artifact.manifest_digest,tmp_path/'bad',initialization_seed=101,fold=1)
    assert not (tmp_path/'bad').exists()


def test_rehashed_baseline_tensor_corruption_rejected(initial,tmp_path):
    from werewolf.artifact_io import publish_tensor_state,TensorValue
    full,parent,state=initial
    tensors={k:TensorValue(v.array.copy(),True) for k,v in state.tensors.items()}
    tensors['shared_query'].array[0,0]+=1
    corrupt=publish_tensor_state(tmp_path/'corrupt',manifest_fields={k:v for k,v in state.artifact.manifest.items()
        if k not in ('manifest_digest','file_table','tensor_container')},tensors=tensors)
    model=ObserverAgnosticToM(ExperimentCapacity(3),None)
    with pytest.raises(ValueError,match='tensor binding'):
        load_initial_state(corrupt.artifact.path,model,corrupt.artifact.manifest_digest,full_initial_path=parent.artifact.path)
