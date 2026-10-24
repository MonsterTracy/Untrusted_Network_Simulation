"""Artificial full-graph weights only; no real final weights, data or games."""
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from werewolf.artifact_io import canonical_json_bytes, publish_artifact, sha256_bytes
from werewolf.artifact_io.tensor_state import TensorValue, encode_tensor_container
from werewolf.artifact_io.tensor_state import decode_tensor_container
from werewolf.observer_dtrs import final_tom_audit as audit
from werewolf.observer_dtrs import final_tom_capsule as capsule
from werewolf.observer_dtrs.final_tom_capsule_build import compile_capsule
from werewolf.observer_dtrs.final_tom_graph import InferenceGraph, graph_identity
from werewolf.tom.backbone_model import BackboneToM, graph_identity as original_graph_identity
from werewolf.tom.dataset import ExperimentCapacity, tensorize_public_pre
from werewolf.tom.state import model_tensor_values
from werewolf.tom.temporal import TemporalCodeProvider, publish_temporal
from tests.observer_dtrs.conftest import prefix_for
from tests.observer_dtrs.test_final_tom_audit import git


def contents(root):
    return {str(p.relative_to(root)): (sha256_bytes(p.read_bytes()), p.stat().st_mode)
            for p in root.rglob('*') if p.is_file()}


@pytest.fixture(scope='module')
def artificial(tmp_path_factory):
    root = tmp_path_factory.mktemp('artificial-final-capsule')
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    checkout = root/'predictor'
    (checkout/'werewolf').mkdir(parents=True)
    (checkout/'werewolf/core.py').write_text('# artificial fixture only\n')
    (checkout/'run_random.py').write_text('# artificial fixture only\n')
    git(checkout, 'init', '-q')
    git(checkout, 'add', '--', 'werewolf/core.py', 'run_random.py')
    git(checkout, 'commit', '-qm', 'artificial fixture')
    source = {k: v for k, v in audit.checkout_identity(checkout).items()
              if k in ('source_revision', 'implementation_digest')}
    pub = publish_artifact(root/'parents/publication', manifest_fields={
        'artifact_type': 'development_publication', 'schema_version': 'classic7_development_publication_v1',
        'publication_id': 'artificial', 'role_sidecar_digest': 'b'*64}, files={'roles.json': b'RESTRICTED_ROLE_SENTINEL'})
    initial = {'architecture': 'qwen3', 'fold': 0, 'temporal_condition': 'explicit_day_phase', 'state_digest': 'a'*64}
    study = publish_artifact(root/'parents/study', manifest_fields={
        'artifact_type': 'backbone_tom_study', 'schema_version': 'classic7_backbone_study_v1',
        'publication_path': str(pub.path), 'initializations': [initial],
        'protocol_inputs': {'publication_digest': pub.manifest_digest}}, files={})
    fit_path = root/'lifecycle/fits/artificial'
    config = json.loads((capsule.ROOT/'configs/paper/tom-backbone-study-v1.json').read_bytes())['reference_protocol']
    p = {'schema_version': audit.FIT_VERSION, 'architecture': 'qwen3', 'config': config,
        'temporal_condition': 'explicit_day_phase', 'population': 'non_wolf_alive', 'partition': 'all_development',
        'graph': graph_identity(), 'capacity': capsule.public_contract()['capacity'],
        'rng_version': 'classic7_qwen3_final_training_rng_v1', 'publication_digest': pub.manifest_digest,
        'study_digest': study.manifest_digest, 'initial_state_digest': initial['state_digest'],
        'source': source, 'runtime': {'implementation_digest': source['implementation_digest'],
            'source_revision': config['source_revision'], 'python': 'fixture'}, 'destination_path': str(fit_path)}
    protocol = capsule.digest(p)
    temporal = publish_temporal(root/'original-temporal', 256, protocol)
    provider = TemporalCodeProvider.explicit(temporal.path, protocol)
    model = BackboneToM('qwen3', provider, study_seed=718, fold=0).eval()
    container, payload = encode_tensor_container(model_tensor_values(model), payload_path='model.bin')
    files = {'primary.json': b'RESTRICTED_PRIMARY_SENTINEL', 'schedule.json': b'RESTRICTED_SCHEDULE_SENTINEL'}
    files.update({'temporal/'+q.name: q.read_bytes() for q in temporal.path.iterdir()})
    fit = publish_artifact(fit_path, manifest_name='qwen3_final_manifest.json', manifest_fields={
        'artifact_type': 'qwen3_all_development_final', 'schema_version': audit.FIT_VERSION,
        'protocol_inputs': p, 'protocol_digest': protocol, 'study_path': str(study.path),
        'publication_path': str(pub.path), 'publication_id': 'artificial',
        'game_ids': [f'artificial-game-{i:04d}' for i in range(1500)], 'primary_sidecar_digest': 'b'*64,
        'initialization': initial, 'temporal_artifact_digest': temporal.manifest_digest,
        'schedule_digest': sha256_bytes(files['schedule.json']), 'optimizer_steps': 31500}, files=files)
    tiny_model, tiny_payload = encode_tensor_container({'tiny': TensorValue(np.array([.5], dtype='<f4'), True)}, payload_path='model.bin')
    opt, opt_bytes = encode_tensor_container({'optimizer_sentinel': TensorValue(np.array([0], dtype='<f4'), False)}, payload_path='optimizer.bin')
    rng, rng_bytes = encode_tensor_container({'rng_sentinel': TensorValue(np.array([0], dtype='<i4'), False)}, payload_path='rng.bin')
    runs = fit.path.parent.parent/'runs'/fit.manifest_digest
    previous, ancestry = None, []
    for step in audit.STEPS:
        mc, mb = (container, payload) if step == 31500 else (tiny_model, tiny_payload)
        a = publish_artifact(runs/'recovery'/str(step), manifest_fields={
            'artifact_type': 'qwen3_final_recovery', 'schema_version': 'classic7_qwen3_final_recovery_v1',
            'binding': audit._binding(fit), 'optimizer_step': step, 'schedule_position': step,
            'previous_checkpoint_digest': previous, 'containers': {'model': mc, 'optimizer': opt, 'rng': rng},
            'optimizer_scalars': {}, 'rng_scalars': {}, 'loss_digest': sha256_bytes(b'RESTRICTED_LOSS_SENTINEL')},
            files={'model.bin': mb, 'optimizer.bin': opt_bytes, 'rng.bin': rng_bytes, 'loss.jsonl': b'RESTRICTED_LOSS_SENTINEL'})
        previous = a.manifest_digest
        ancestry.append({'step': step, 'digest': previous})
    terminal = publish_artifact(runs/'terminal', manifest_fields={
        'artifact_type': 'qwen3_final_terminal', 'schema_version': 'classic7_qwen3_final_terminal_v1',
        'binding': audit._binding(fit), 'terminal_step': 31500, 'last_recovery_digest': previous,
        'model_digest': capsule.digest(container), 'checkpoint_ancestry': ancestry}, files={})
    seal = publish_artifact(runs/'seal', manifest_fields={
        'artifact_type': 'qwen3_final_seal', 'schema_version': 'classic7_qwen3_final_seal_v1',
        'binding': audit._binding(fit), 'terminal_digest': terminal.manifest_digest, 'terminal_step': 31500}, files={})
    pins = audit._Pins(source['source_revision'], fit.manifest_digest, terminal.manifest_digest,
                       seal.manifest_digest, capsule.digest(container))
    report = audit.audit_final_tom(fit.path, checkout, chain_files=True, _pins=pins)
    assert report['status'] == 'PASS_WITH_EXPLICIT_SCOPE_LIMITS', report
    evidence = root/'chain-evidence.json'
    evidence.write_bytes(canonical_json_bytes(report)+b'\n')
    audit_sha = sha256_bytes(evidence.read_bytes())
    output = compile_capsule(fit.path, root/'capsule', audit_path=evidence, expected_audit_raw_sha256=audit_sha, _pins=pins)
    yield {'root': root, 'fit': fit, 'runs': runs, 'model': model, 'capsule': output,
           'audit': evidence, 'audit_sha': audit_sha, 'pins': pins, 'report': report}
    torch.set_num_threads(threads)


def load(artifact):
    return capsule.CapsulePredictor(artifact.path, expected_manifest_digest=artifact.manifest_digest,
                                    expected_scope='CONTROLLED_FIXTURE')


def compile_to(fixture, output, **kw):
    return compile_capsule(fixture['fit'].path, output, audit_path=fixture['audit'],
        expected_audit_raw_sha256=fixture['audit_sha'], _pins=fixture['pins'], **kw)


def republish(a, output, *, spec_change=None, manifest_change=None, file_change=None):
    fields = {k: deepcopy(v) for k, v in a.manifest.items() if k not in ('manifest_digest', 'file_table')}
    files = {n: (a.path/n).read_bytes() for n in a.manifest['file_table']}
    if spec_change:
        spec = json.loads(files['inference.json']);spec_change(spec)
        files['inference.json'] = canonical_json_bytes(spec)
        fields['inference_digest'] = capsule.digest(spec)
    if file_change: file_change(files)
    if manifest_change: manifest_change(fields)
    return publish_artifact(output, manifest_fields=fields, files=files)


@pytest.mark.parametrize('phase', ['discussion', 'pk_discussion'])
@pytest.mark.parametrize('dead', [(), ('player2', 'player7')])
def test_original_tensor_temporal_and_numeric_equivalence(artificial, phase, dead):
    a = artificial['capsule']; before = contents(a.path)
    py = random.getstate();npstate = np.random.get_state();rng = torch.get_rng_state().clone()
    deployed = load(a)
    assert torch.equal(torch.get_rng_state(), rng) and random.getstate() == py
    assert np.array_equal(npstate[1], np.random.get_state()[1])
    assert graph_identity() == original_graph_identity('qwen3')
    assert all(torch.equal(v, deployed._model.state_dict()[n]) for n, v in artificial['model'].state_dict().items())
    for n in ('day', 'phase'):
        assert np.array_equal(getattr(deployed._model.temporal, n), getattr(artificial['model'].temporal, n))
    p = prefix_for(phase=phase, dead=dead, actor='player5')
    plan, tensors = tensorize_public_pre(p, ExperimentCapacity(1024))
    new_plan, arrays = capsule.encode_public_pre(p)
    assert new_plan == plan
    assert all(torch.equal(arrays[n][0], t) for n, t in tensors.kwargs().items())
    with torch.inference_mode():
        original_log = artificial['model'](**{n: t[None] for n, t in tensors.kwargs().items()})[0]
        new_log = deployed._model(**arrays)[0]
    assert torch.equal(original_log, new_log)
    prediction = deployed.predict(p)
    assert prediction.probabilities == tuple(tuple(row) for row in original_log.exp().tolist())
    q = np.asarray(prediction.probabilities)
    assert np.array_equal(q.diagonal(), np.zeros(7)) and np.allclose(q.sum(1), 1)
    assert q[2, 1] > 0 and q[2, 6] > 0  # death does not remove/renormalize target columns
    assert prediction.plan_digest == plan.plan_digest and prediction.game_id == p.game_id
    assert prediction.identity['origin']['source_revision'] == artificial['pins'].revision
    assert prediction.identity['original_f199_execution'] is False
    assert prediction.identity['qualification'] == 'NOT_ESTABLISHED'
    assert contents(a.path) == before and torch.equal(torch.get_rng_state(), rng)


def test_forward_implementation_is_reviewed_exact_projection():
    old = ast.parse((capsule.ROOT/'werewolf/tom/backbone_model.py').read_text())
    new = ast.parse((capsule.ROOT/'werewolf/observer_dtrs/final_tom_graph.py').read_text())
    def method(tree, name):
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        return ast.dump(next(n for n in cls.body if getattr(n, 'name', '') == 'forward'), include_attributes=False)
    assert method(old, 'BackboneToM') == method(new, 'InferenceGraph')


def test_compile_whitelist_and_restricted_payloads_never_open(artificial, tmp_path, monkeypatch):
    before = contents(artificial['root'])
    old = Path.open
    def guarded(path, *args, **kw):
        if path.name in ('primary.json', 'schedule.json', 'optimizer.bin', 'rng.bin', 'loss.jsonl', 'roles.json'):
            raise AssertionError('restricted payload opened')
        return old(path, *args, **kw)
    with monkeypatch.context() as m:
        m.setattr(Path, 'open', guarded)
        a = compile_to(artificial, tmp_path/'export')
        deployed = load(a)
        deployed.predict(prefix_for())
    assert contents(artificial['root']) == before
    assert set(a.manifest['file_table']) == capsule.FILES
    spec = json.loads((a.path/'inference.json').read_bytes())
    assert len(spec['model_container']['tensors']) == len(artificial['model'].state_dict())
    text = canonical_json_bytes(a.manifest)+canonical_json_bytes(spec)
    for forbidden in (b'SENTINEL', b'artificial-game-', b'primary.json', b'optimizer_scalars', b'role_assignment'):
        assert forbidden not in text
    assert a.manifest['origin']['model_bin_raw_sha256'] != a.manifest['origin']['model_container_descriptor_sha256']


def test_independent_fresh_process_loader_no_training_or_shadow_imports(artificial, tmp_path):
    a = artificial['capsule']
    prefix = tmp_path/'public-pre.json';prefix.write_bytes(canonical_json_bytes(prefix_for().to_record()))
    code = '''
import importlib.abc,sys,json
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  blocked=('werewolf.development_publication','werewolf.observer_dtrs.final_tom_audit',
   'werewolf.observer_dtrs.final_tom_capsule_build','werewolf.observer_dtrs.shadow',
   'werewolf.tom.dataset','werewolf.tom.backbone_model','werewolf.tom.model',
   'werewolf.tom.qwen3_final','werewolf.tom.experiment','werewolf.tom.training',
   'werewolf.observer_dtrs.dataset_offline','werewolf.observer_dtrs.offline',
   'werewolf.observer_dtrs.training_audit','werewolf.phase2_mapper','werewolf.envs','werewolf.agents')
  if fullname.startswith(blocked): raise AssertionError('forbidden import '+fullname)
sys.meta_path.insert(0,Block())
from werewolf.observer_dtrs.final_tom_capsule import CapsulePredictor,loader_source
from werewolf.canonical_collection.game_bundle import _prefix_from_record
from pathlib import Path
model=CapsulePredictor(sys.argv[1],expected_manifest_digest=sys.argv[2],expected_scope='CONTROLLED_FIXTURE')
p=_prefix_from_record(json.loads(Path(sys.argv[3]).read_bytes()))
model.predict(p)
modules=sorted(n for n in sys.modules if n=='werewolf' or n.startswith('werewolf.'))
paths={n.replace('.','/')+('/__init__.py' if hasattr(sys.modules[n],'__path__') else '.py') for n in modules}
assert paths==set(loader_source()['file_sha256']), (paths,set(loader_source()['file_sha256']))
print(json.dumps({'status':'PASS_INDEPENDENT_PUBLIC_ONLY','runtime_modules':len(modules)}))
'''
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1'}
    result = subprocess.run([sys.executable, '-B', '-c', code, str(a.path), a.manifest_digest, str(prefix)],
                            capture_output=True, text=True, env=env, timeout=60)
    assert result.returncode == 0, result.stdout+result.stderr
    assert 'PASS_INDEPENDENT_PUBLIC_ONLY' in result.stdout


@pytest.mark.parametrize('case', ['raw', 'missing', 'extra', 'manifest_pin', 'scope', 'source', 'graph',
                                 'contract', 'container', 'origin', 'temporal_parent', 'temporal_bytes'])
def test_corrupt_or_rehashed_capsules_are_rejected(artificial, tmp_path, case):
    a = artificial['capsule']
    if case in ('raw', 'missing', 'extra'):
        shutil.copytree(a.path, tmp_path/'bad');out = tmp_path/'bad'
        if case == 'raw': (out/'model.bin').write_bytes(b'bad')
        elif case == 'missing': (out/'model.bin').unlink()
        else: (out/'targets.json').write_bytes(b'not allowed')
        with pytest.raises(ValueError):
            capsule.CapsulePredictor(out, expected_manifest_digest=a.manifest_digest, expected_scope='CONTROLLED_FIXTURE')
        return
    spec_changes = {
        'source': lambda s: s['loader_source'].update(implementation_digest='e'*64),
        'graph': lambda s: s['graph']['resolved_config'].update(hidden_size=512),
        'contract': lambda s: s['public_contract'].update(dead_target_mass='drop'),
        'container': lambda s: s['model_container']['tensors'].pop(),
        'temporal_parent': lambda s: s.update(temporal_parent_protocol='e'*64)}
    b = republish(a, tmp_path/'bad', spec_change=spec_changes.get(case),
        manifest_change=(lambda m: m['origin'].update(publication_digest='invalid')) if case == 'origin' else None,
        file_change=(lambda f: f.update({'temporal/canonical_day_code_table.bin': b'bad table'})) if case == 'temporal_bytes' else None)
    with pytest.raises(ValueError):
        capsule.CapsulePredictor(b.path, expected_manifest_digest='e'*64 if case == 'manifest_pin' else b.manifest_digest,
                                expected_scope='PINNED_FINAL_INFERENCE_UNQUALIFIED' if case == 'scope' else 'CONTROLLED_FIXTURE')


@pytest.mark.parametrize('case', ['audit_sha', 'audit_fail', 'metadata_only', 'scope', 'source', 'raw_sha'])
def test_compiler_fails_without_matching_external_audit(artificial, tmp_path, case):
    r = deepcopy(artificial['report'])
    if case == 'audit_fail': r['status'] = 'FAILED'
    elif case == 'metadata_only': r['mode'] = 'metadata'
    elif case == 'scope': r['scope'] = 'PINNED_FINAL_PROVENANCE_ONLY'
    elif case == 'source': r['checks']['source']['implementation_digest'] = 'e'*64
    elif case == 'raw_sha': r['checks']['recovery']['model_bin_raw_sha256'] = 'e'*64
    r.pop('evidence_digest');r['evidence_digest'] = capsule.digest(r)
    evidence = tmp_path/'evidence.json';evidence.write_bytes(canonical_json_bytes(r))
    with pytest.raises(ValueError):
        compile_capsule(artificial['fit'].path, tmp_path/'export', audit_path=evidence,
            expected_audit_raw_sha256='e'*64 if case == 'audit_sha' else sha256_bytes(evidence.read_bytes()),
            _pins=artificial['pins'])
    assert not (tmp_path/'export').exists()


@pytest.mark.parametrize('target', ['fit', 'runs', 'temporal', 'alias', 'dotdot', 'checkout', 'existing'])
def test_output_protection(artificial, tmp_path, target):
    before = contents(artificial['root'])
    if target == 'fit': out = artificial['fit'].path/'export'
    elif target == 'runs': out = artificial['runs']/'export'
    elif target == 'temporal': out = artificial['fit'].path/'temporal/export'
    elif target == 'alias':
        (tmp_path/'alias').symlink_to(artificial['fit'].path, target_is_directory=True);out = tmp_path/'alias/export'
    elif target == 'dotdot':
        (tmp_path/'child').mkdir();out = tmp_path/'child/../export'
    elif target == 'checkout': out = artificial['root']/'predictor/export'
    else: out = artificial['capsule'].path
    with pytest.raises(ValueError): compile_to(artificial, out)
    assert contents(artificial['root']) == before


def test_public_cli_rejects_fixture_origin_without_output(artificial, tmp_path):
    before = contents(artificial['root'])
    result = subprocess.run([sys.executable, '-B', 'scripts/build_final_tom_capsule.py',
        '--fit', str(artificial['fit'].path), '--audit', str(artificial['audit']),
        '--audit-sha256', artificial['audit_sha'], '--output', str(tmp_path/'capsule')],
        capture_output=True, text=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, timeout=60)
    assert result.returncode == 2 and json.loads(result.stdout)['status'] == 'FAILED'
    assert not (tmp_path/'capsule').exists() and contents(artificial['root']) == before


def test_new_loader_cannot_open_original_training_lifecycle(artificial, tmp_path, monkeypatch):
    isolated = republish(artificial['capsule'], tmp_path/'independent')
    original_open = Path.open
    def guarded(path, *a, **k):
        assert not path.is_relative_to(artificial['root']), 'original lifecycle accessed'
        return original_open(path, *a, **k)
    with monkeypatch.context() as m:
        m.setattr(Path, 'open', guarded)
        load(isolated).predict(prefix_for())


def test_capacity_and_invalid_pre_fail_without_forward(artificial, monkeypatch):
    m = load(artificial['capsule'])
    monkeypatch.setattr(m._model, 'forward', lambda **kw: pytest.fail('forward after invalid PRE'))
    with pytest.raises((TypeError, ValueError)):
        m.predict({'raw_report': 'not a frozen public PRE'})
    # Validator is shared; exercise long/day-overflow plans without pretending a real game.
    from types import SimpleNamespace
    import werewolf.tom.final_capacity as capacity
    for plan in (SimpleNamespace(token_count=1025, tokens=()),
                 SimpleNamespace(token_count=4, tokens=(SimpleNamespace(day=257),))):
        with monkeypatch.context() as patch:
            patch.setattr(capacity, 'plan_structured_history', lambda _: plan)
            with pytest.raises(ValueError, match='capacity'):
                m.predict(prefix_for())


def test_same_legal_input_ignores_external_supervision_and_d2_identity(artificial, tmp_path, monkeypatch):
    m = load(artificial['capsule']);p = prefix_for()
    targets = tmp_path/'restricted-supervision.json'
    results = []
    original_open = Path.open
    def guarded(path, *a, **k):
        assert path != targets, 'online reader touched supervision'
        return original_open(path, *a, **k)
    for y, support in ((1, ['player1']), (0, ['player7'])):
        targets.write_bytes(canonical_json_bytes({'Y': y, 'suspicion_report': support}))
        with monkeypatch.context() as patch:
            patch.setattr(Path, 'open', guarded)
            results.append(m.predict(p))
    assert results[0] == results[1]
    assert 'supervision' not in m.identity and 'support' not in m.identity
    from werewolf.observer_dtrs.online_model import TomIdentity
    with pytest.raises((TypeError, ValueError)):
        TomIdentity(**m.identity)  # no silent conversion to OOF/f199 or fixture deployment identity


@pytest.mark.parametrize('case', ['missing', 'extra', 'shape', 'dtype', 'nonfinite', 'not_trainable'])
def test_semantic_tensor_restore_rejects_even_rehashed_container(artificial, tmp_path, case):
    a = artificial['capsule'];spec = json.loads((a.path/'inference.json').read_bytes())
    values = decode_tensor_container(spec['model_container'], (a.path/'model.bin').read_bytes(), payload_path='model.bin')
    tensors = {n: TensorValue(v.array.copy(), v.trainable) for n, v in values.items()}
    name = 'output.bias'
    if case == 'missing': tensors.pop(name)
    elif case == 'extra': tensors['extra'] = TensorValue(np.zeros(1, dtype='<f4'), True)
    elif case == 'shape': tensors[name] = TensorValue(np.zeros(8, dtype='<f4'), True)
    elif case == 'dtype': tensors[name] = TensorValue(np.zeros(7, dtype='<f8'), False)
    elif case == 'not_trainable': tensors[name] = TensorValue(tensors[name].array, False)
    else: tensors[name].array[0] = np.nan
    container, raw = encode_tensor_container(tensors, payload_path='model.bin')
    b = republish(a, tmp_path/'bad', spec_change=lambda s: s.update(model_container=container),
        manifest_change=lambda m: m['origin'].update(model_container_descriptor_sha256=capsule.digest(container),
                                                     model_bin_raw_sha256=sha256_bytes(raw)),
        file_change=lambda f: f.update({'model.bin': raw}))
    with pytest.raises(ValueError, match='model (state|tensor)'):
        load(b)


def test_valid_but_insufficient_temporal_capacity_is_rejected(artificial, tmp_path):
    a = artificial['capsule'];parent = a.manifest['origin']['protocol_digest']
    temporal = publish_temporal(tmp_path/'short-time', 255, parent)
    b = republish(a, tmp_path/'bad', spec_change=lambda s: s.update(temporal_digest=temporal.manifest_digest),
        manifest_change=lambda m: m['origin'].update(temporal_manifest_digest=temporal.manifest_digest),
        file_change=lambda f: f.update({'temporal/'+n: (temporal.path/n).read_bytes() for n in capsule.TEMPORAL_FILES}))
    with pytest.raises(ValueError, match='capacity'):
        load(b)


def test_nonfinite_output_does_not_become_uniform_fallback(artificial, monkeypatch):
    m = load(artificial['capsule'])
    monkeypatch.setattr(m._model, 'forward', lambda **kw: torch.full((1, 7, 7), float('nan')))
    with pytest.raises(ValueError, match='no fallback'):
        m.predict(prefix_for())


def test_runtime_is_observed_separately_and_not_claimed_original(artificial):
    m = load(artificial['capsule']);i = m.identity
    assert set(i) == {'family', 'scope', 'capsule_digest', 'origin', 'loader_source',
                      'runtime', 'qualification', 'original_f199_execution'}
    assert i['loader_source']['formal_execution_guard'] == 'NOT_ESTABLISHED'
    assert i['runtime']['device'] == 'cpu' and i['runtime']['dtype'] == 'float32'
    assert i['origin']['saved_runtime_digest'] == artificial['report']['checks']['source']['runtime']['saved_runtime_digest']
    assert not any(p.requires_grad for p in m._model.parameters()) and not m._model.training


def test_actual_canonical_capacity_boundary_and_overflow(artificial, monkeypatch):
    from tests.canonical_collection.test_final_capacity import sparse_pre
    m = load(artificial['capsule'])
    p = sparse_pre(256)
    result = m.predict(p)
    assert result.token_count == 1024
    monkeypatch.setattr(m._model, 'forward', lambda **kw: pytest.fail('overcapacity forward'))
    with pytest.raises(ValueError, match='sequence capacity'):
        m.predict(sparse_pre(257))


def test_loader_refuses_capsule_alias_and_non_fp32_graph(artificial, tmp_path):
    (tmp_path/'alias').symlink_to(artificial['capsule'].path, target_is_directory=True)
    with pytest.raises(ValueError, match='aliases'):
        capsule.CapsulePredictor(tmp_path/'alias', expected_manifest_digest=artificial['capsule'].manifest_digest,
                                 expected_scope='CONTROLLED_FIXTURE')
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(ValueError, match='FP32'):
            load(artificial['capsule'])
    finally:
        torch.set_default_dtype(dtype)


def test_source_inventory_refuses_wrong_import_origin(tmp_path, monkeypatch):
    substitute = tmp_path/'substitute.py';substitute.write_text('# not an authorized runtime source\n')
    monkeypatch.setattr(sys.modules['werewolf.tom.state'], '__file__', str(substitute))
    with pytest.raises(ValueError, match='module origin'):
        capsule.loader_source()


def test_public_speech_actions_targets_and_temporal_encoding_match(artificial):
    from tests.canonical_collection.test_structured_token_plan import _prefix
    p = _prefix()
    plan, old = tensorize_public_pre(p, ExperimentCapacity(1024))
    new_plan, arrays = capsule.encode_public_pre(p)
    assert plan == new_plan and {'speech_action', 'vote', 'exiled_player'} <= {t.token_type for t in plan.tokens}
    assert all(torch.equal(arrays[n][0], t) for n, t in old.kwargs().items())
    with torch.inference_mode():
        expected = artificial['model'](**{n: t[None] for n, t in old.kwargs().items()})[0].exp()
    assert load(artificial['capsule']).predict(p).probabilities == tuple(tuple(row) for row in expected.tolist())
