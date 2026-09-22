"""Synthetic publications and source attestation; no real study or OOF reads."""
from copy import deepcopy
import json

import pytest

from tests.development_publication.test_development_publication import _publication
from tests.tom.test_paper_study_execution import rewritten
from werewolf.artifact_io import canonical_json_bytes, publish_artifact, verify_artifact
from werewolf.tom import backbone_study as m
from werewolf.tom.backbone_model import ARCHITECTURES, BackboneToM, graph_identity
from werewolf.tom.experiment import ExperimentConfig, runtime_provenance


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp('backbone-study')
    _, handles = _publication(root)
    design = m._frozen_design()
    design['publication_id'] = handles.public.public_view.publication_id
    design['game_count'] = len(handles.public.public_view.game_ids)
    design['reference_protocol']['device'] = 'cpu'
    config = ExperimentConfig(**design['reference_protocol'])
    source = {'source_revision': 'synthetic-execution-revision',
              'implementation_digest': runtime_provenance(config)['implementation_digest']}
    config_path = root / 'backbone.json'
    config_path.write_bytes(canonical_json_bytes(design))
    with pytest.MonkeyPatch.context() as patch:
        # Replace only external publication size/name, execution device and Git identity.
        # Real publication verification, all graph/state validators and source comparisons run.
        patch.setattr(m, '_frozen_design', lambda: deepcopy(design))
        patch.setattr(m, 'attest_source', lambda: dict(source))
        study = m.prepare_study(config_path, handles.public.path, root / 'study')
        yield root, study, handles, config_path, design, source


def test_frozen_config_exact_design_and_reference_protocol():
    design = json.loads(m.CONFIG_PATH.read_bytes())
    reference = json.loads((m.CONFIG_PATH.parents[1] / 'formal/development-experiment-v1/protocol.json').read_bytes())
    assert design['reference_protocol'] == reference
    assert design['publication_id'] == 'paper-development-qwen35-9b-1500-v1'
    assert design['game_count'] == 1500
    assert list(design['architectures']) == list(ARCHITECTURES)
    assert design['temporal_conditions'] == ['explicit_day_phase']
    assert design['fold_count'] == 5 and design['terminal_lineages'] == 20
    for architecture in ARCHITECTURES:
        assert design['architectures'][architecture] == graph_identity(architecture)['resolved_config']
    assert design['initialization']['training_rng'] == 'reset_from_protocol_rng_seed_after_model_construction'


def test_prepare_open_twenty_paired_initials(prepared):
    root, artifact, handles, _, design, source = prepared
    opened = m.open_study(artifact.path)
    assert opened.manifest_digest == artifact.manifest_digest
    manifest = opened.manifest
    assert manifest['protocol_inputs']['source'] == source
    assert source['source_revision'] != design['reference_protocol']['source_revision']
    assert manifest['protocol_inputs']['publication_digest'] == handles.public.manifest_digest
    assert manifest['protocol_inputs']['fold_manifest_digest'] == handles.public.public_view.fold_manifest.manifest_digest
    assert manifest['protocol_inputs']['primary_identity']['population_identity'] == 'non_wolf_alive'
    assert len(manifest['initializations']) == 20
    assert [(r['architecture'], r['temporal_condition'], r['fold']) for r in manifest['initializations']] == [
        (a, 'explicit_day_phase', f) for a in ARCHITECTURES for f in range(5)]
    assert not (root / 'runs').exists()
    for fold in range(5):
        rows = [r for r in manifest['initializations'] if r['fold'] == fold]
        assert len({r['shell_initialization_digest'] for r in rows}) == 1
        assert len({r['backbone_initialization_digest'] for r in rows}) == 4
        assert len({r['combined_initialization_digest'] for r in rows}) == 4
    assert len({r['shell_initialization_digest'] for r in manifest['initializations']}) == 5


@pytest.mark.parametrize('change', [
    lambda m: m['protocol_inputs']['source'].update(source_revision='other-head'),
    lambda m: m['protocol_inputs']['source'].update(implementation_digest='0'*64),
    lambda m: m['runtime'].update(implementation_digest='0'*64),
    lambda m: m['protocol_inputs'].update(publication_digest='0'*64),
    lambda m: m['protocol_inputs']['design']['reference_protocol'].update(rotation_cycles=4),
    lambda m: m['protocol_inputs']['graphs'][0]['resolved_config'].update(n_inner=1024),
    lambda m: m['folds'][0].update(optimizer_steps=1),
    lambda m: m['initializations'].pop(),
    lambda m: m['initializations'][0].update(temporal_condition='implicit'),
    lambda m: m.update(temporal_artifact_digest='0'*64),
])
def test_open_rejects_rehashed_semantic_corruption(prepared, change):
    _, artifact, *_ = prepared
    with rewritten(artifact.path / 'manifest.json', change, 'manifest_digest'):
        with pytest.raises(ValueError):
            m.open_study(artifact.path)


def test_initial_state_rejects_wrong_graph_fold_study_and_checkpoint(prepared, tmp_path):
    _, artifact, _, _, design, _ = prepared
    initial = artifact.path / 'initial/qwen2/0'
    row = next(r for r in artifact.manifest['initializations'] if r['architecture'] == 'qwen2' and r['fold'] == 0)
    seed = design['reference_protocol']['initialization_seed']
    model = BackboneToM('qwen2', None, study_seed=seed, fold=0)
    kwargs = dict(study_digest=artifact.manifest['study_digest'], study_seed=seed,
                  fold=0, expected_digest=row['state_digest'])
    state = m.load_initial_state(initial, model, **kwargs)
    for wrong in (dict(study_digest='0'*64), dict(fold=1), dict(expected_digest='0'*64), dict(study_seed=seed+1)):
        with pytest.raises(ValueError, match='binding'):
            m.load_initial_state(initial, model, **{**kwargs, **wrong})
    for architecture in ('gpt2', 'qwen3', 'gemma3_text'):
        other = BackboneToM(architecture, None, study_seed=seed, fold=0)
        with pytest.raises(ValueError, match='binding'):
            m.load_initial_state(initial, other, **kwargs)
    # A shape-compatible Qwen2 graph with altered config identity is also rejected.
    model.graph = deepcopy(model.graph)
    model.graph['resolved_config']['attention_dropout'] = 0
    with pytest.raises(ValueError, match='binding'):
        m.load_initial_state(initial, model, **kwargs)
    fields = {k: v for k, v in state.artifact.manifest.items() if k not in ('manifest_digest', 'file_table')}
    fields['artifact_type'] = 'training_checkpoint'
    fake = publish_artifact(tmp_path / 'checkpoint', manifest_fields=fields,
        files={'tensors.bin': (initial / 'tensors.bin').read_bytes()})
    with pytest.raises(ValueError):
        m.load_initial_state(fake.path, model, **{**kwargs, 'expected_digest': fake.manifest_digest})
    from werewolf.tom.state import validate_model_state
    with pytest.raises(ValueError):
        validate_model_state(initial)  # New initial artifacts are never old Full artifacts.


def test_prepare_failures_do_not_publish(prepared, monkeypatch, tmp_path):
    _, _, handles, config_path, design, source = prepared
    bad = deepcopy(design)
    bad['temporal_conditions'] = ['implicit', 'explicit_day_phase']
    bad_path = tmp_path / 'bad.json'
    bad_path.write_bytes(canonical_json_bytes(bad))
    with pytest.raises(ValueError, match='frozen design'):
        m.prepare_study(bad_path, handles.public.path, tmp_path / 'rejected-config')
    assert not (tmp_path / 'rejected-config').exists()
    with monkeypatch.context() as patch:
        patch.setattr(m, 'attest_source', lambda: {**source, 'implementation_digest': '0'*64})
        with pytest.raises(ValueError, match='implementation'):
            m.prepare_study(config_path, handles.public.path, tmp_path / 'rejected-source')
        assert not (tmp_path / 'rejected-source').exists()
    with monkeypatch.context() as patch:
        def fail_construction(*args, **kwargs):
            raise ValueError('synthetic initialization failure')
        patch.setattr(m, 'BackboneToM', fail_construction)
        with pytest.raises(ValueError, match='initialization failure'):
            m.prepare_study(config_path, handles.public.path, tmp_path / 'rejected-initial')
        assert not (tmp_path / 'rejected-initial').exists()


def test_cli_prepare_and_open_own_real_artifacts(prepared, capsys):
    from werewolf.cli import main
    root, _, handles, config_path, _, _ = prepared
    profile = root / 'storage.json'
    profile.write_text(json.dumps({'artifact_root': str(root)}))
    base = ['--storage-profile', str(profile)]
    assert main(base + ['prepare-backbone-tom-study', '--config', str(config_path),
        '--publication', str(handles.public.path), '--destination', 'cli-study']) == 0
    prepared_digest = capsys.readouterr().out.strip()
    assert main(base + ['open-backbone-tom-study', '--study', 'cli-study']) == 0
    assert capsys.readouterr().out.strip() == prepared_digest
    assert verify_artifact(root / 'cli-study', expected_artifact_type='backbone_tom_study',
                           expected_schema_version=m.STUDY_VERSION).manifest_digest == prepared_digest
