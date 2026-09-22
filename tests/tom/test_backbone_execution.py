"""Tiny verified synthetic publications only; execution attestation is synthetic."""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil

import pytest
import torch

from tests.development_publication.test_development_publication import _publication
from tests.tom.test_paper_study_execution import rewritten
from werewolf.artifact_io import canonical_json_bytes, verify_artifact
from werewolf.tom import backbone_execution as e, backbone_study as p
from werewolf.tom.experiment import ExperimentConfig, runtime_provenance
from werewolf.tom.protocol import training_schedule


def _engineering_process(root, expected, a, fold, interrupt, connection):
    """Fresh process, real worker; substitute only synthetic fixture external facts."""
    metadata = json.loads((Path(root) / 'contract/manifest.json').read_bytes())
    snapshot = json.loads((Path(metadata['prepared_path']) / 'manifest.json').read_bytes())
    p._frozen_design = lambda: deepcopy(snapshot['protocol_inputs']['design'])
    e.attest_source = lambda: dict(metadata['execution_source'])
    e._process_entry(root, expected, a, fold, interrupt, connection)


@pytest.fixture(scope='module')
def fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp('backbone-execution')
    _, handles = _publication(root / 'synthetic-publication')
    design = p._frozen_design()
    design['publication_id'] = handles.public.public_view.publication_id
    design['game_count'] = len(handles.public.public_view.game_ids)
    design['reference_protocol']['device'] = 'cuda' if os.environ.get('UNS_BACKBONE_ENGINEERING_CUDA') == '1' else 'cpu'
    config = ExperimentConfig(**design['reference_protocol'])
    digest = runtime_provenance(config)['implementation_digest']
    source = {'source_revision': 'a'*40, 'implementation_digest': digest}
    config_path = root / 'design.json'
    config_path.write_bytes(canonical_json_bytes(design))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(p, '_frozen_design', lambda: deepcopy(design))
        patch.setattr(p, 'attest_source', lambda: dict(source))
        study = p.prepare_study(config_path, handles.public.path, root / 'paper-studies/prepared')
        # Keep the prepared bytes exactly fixed; execution uses another source revision.
        patch.setattr(e, 'attest_source', lambda: {**source, 'source_revision': 'b'*40})
        patch.setattr(e, '_process_entry', _engineering_process)
        execution = e.create_execution(study.path, study.manifest_digest, engineering=True)
        yield root, study, execution, design


@pytest.fixture(scope='module')
def completed(fixture):
    root, study, execution, design = fixture
    before = {str(f.relative_to(study.path)): f.read_bytes() for f in study.path.rglob('*') if f.is_file()}
    results = {}
    for a in e.ARCHITECTURES:
        with pytest.raises(InterruptedError):
            e.run_lineage(execution.root, a, 0, interrupt_after=1)
        assert not (execution.root / 'runs' / a / '0/terminal').exists()
        results[a] = e.run_lineage(execution.root, a, 0)
        e.validate_terminal(execution, a, 0)
    assert before == {str(f.relative_to(study.path)): f.read_bytes() for f in study.path.rglob('*') if f.is_file()}
    print('ENGINEERING ONLY peak memory:', json.dumps(results, sort_keys=True))
    return fixture


def test_twenty_lineages_and_formal_budget():
    assert len(e.LINEAGES) == len(set(e.LINEAGES)) == 20
    assert {t for _, _, t in e.LINEAGES} == {'explicit_day_phase'}
    design = p._frozen_design()
    c = design['reference_protocol']
    assert c['recovery_cadence'] == 1000
    assert training_schedule(c['schedule_seed'], 0, [f'g{i:04}' for i in range(1200)],
                             c['rotation_cycles'], c['game_batch_size'])['optimizer_steps'] == 25200


def test_snapshot_two_stage_binding_and_old_opener_unchanged(fixture, monkeypatch):
    _, study, execution, _ = fixture
    assert e.open_execution(execution.root).artifact.manifest_digest == execution.artifact.manifest_digest
    identity = e.snapshot_identity(study)
    for k in identity:
        with pytest.raises(ValueError, match='identity'):
            e.validate_frozen_snapshot(study.path, {**identity, k: '0' * len(identity[k])})
    with monkeypatch.context() as patch:
        patch.setattr(p, 'attest_source', e.attest_source)
        with pytest.raises(ValueError, match='source/config'):
            p.open_study(study.path)
    for k, value in [('source_revision', 'c'*40), ('implementation_digest', '0'*64)]:
        with monkeypatch.context() as patch:
            source = e.attest_source()
            patch.setattr(e, 'attest_source', lambda: {**source, k: value})
            with pytest.raises(ValueError, match='source|runtime'):
                e.open_execution(execution.root)
    with pytest.raises(ValueError, match='25200'):
        e.create_execution(study.path, study.manifest_digest)
    assert not (study.path.parent / 'executions').exists()


def test_unknown_lineage_rejected(fixture):
    execution = fixture[2]
    for a, f, t in [('unknown', 0, 'explicit_day_phase'), ('gpt2', 5, 'explicit_day_phase'),
                     ('gpt2', False, 'explicit_day_phase'), ('gpt2', 0, 'implicit')]:
        with pytest.raises(ValueError, match='unknown'):
            e._lineage(execution, a, f, t)


def test_training_rng_reset_follows_all_construction(fixture, monkeypatch):
    execution = fixture[2]
    model, optimizer = e._model_optimizer(execution, 'qwen2', 0)
    expected = e.training._rng_capture()
    constructor = e.BackboneToM
    def noisy(*args, **kwargs):
        model = constructor(*args, **kwargs)
        torch.rand(731)
        e.np.random.rand(10)
        e.random.random()
        return model
    monkeypatch.setattr(e, 'BackboneToM', noisy)
    other, _ = e._model_optimizer(execution, 'qwen2', 0)
    actual = e.training._rng_capture()
    assert e._same(actual[1], expected[1])
    assert actual[0].keys() == expected[0].keys()
    for k in actual[0]:
        assert actual[0][k].array.tobytes() == expected[0][k].array.tobytes()
    assert all(torch.equal(v, other.state_dict()[k]) for k, v in model.state_dict().items())


@pytest.mark.parametrize('change', [
    lambda m: m['binding'].update(architecture='gpt2'),
    lambda m: m['binding'].update(fold=1),
    lambda m: m['binding'].update(temporal_condition='implicit'),
    lambda m: m['binding']['graph'].update(graph_digest='0'*64),
    lambda m: m['binding']['graph'].update(resolved_config_digest='0'*64),
    lambda m: m['binding']['initialization'].update(combined_initialization_digest='0'*64),
    lambda m: m['binding']['preparation'].update(study_digest='0'*64),
    lambda m: m['binding']['preparation'].update(manifest_digest='0'*64),
    lambda m: m['binding'].update(schedule_digest='0'*64),
    lambda m: m.update(optimizer_step=2),
    lambda m: m.update(schedule_position=2),
    lambda m: m['binding']['execution_source'].update(source_revision='0'*40),
    lambda m: m['optimizer_scalars']['param_groups'][0].update(lr=.9),
])
def test_checkpoint_mismatch_rejected(completed, change):
    execution = completed[2]
    path = execution.root / 'runs/qwen2/0/recovery/1/manifest.json'
    model, optimizer = e._model_optimizer(execution, 'qwen2', 0)
    with rewritten(path, change, 'manifest_digest'):
        with pytest.raises(ValueError):
            e._read_checkpoint(execution, 'qwen2', 0, 1, None, [], model, optimizer)


def test_four_architectures_process_restart_and_exact_recovery(completed):
    root, study, execution, _ = completed
    # Identical preparation bytes in a separate engineering execution, uninterrupted.
    clone = root / 'comparison/prepared'
    shutil.copytree(study.path, clone)
    uninterrupted = e.create_execution(clone, study.manifest_digest, engineering=True)
    for a in e.ARCHITECTURES:
        e.run_lineage(uninterrupted.root, a, 0)
        recovered = execution.root / 'runs' / a / '0/recovery/2'
        reference = uninterrupted.root / 'runs' / a / '0/recovery/2'
        for name in ('model.bin', 'optimizer.bin', 'rng.bin', 'loss.jsonl'):
            assert (recovered / name).read_bytes() == (reference / name).read_bytes()
        left = json.loads((recovered / 'manifest.json').read_bytes())
        right = json.loads((reference / 'manifest.json').read_bytes())
        for field in ('containers', 'optimizer_scalars', 'rng_scalars', 'loss_digest'):
            assert e._same(left[field], right[field])


def test_terminal_validation_duplicate_and_missing_seal(completed):
    execution = completed[2]
    with pytest.raises(ValueError, match='missing'):
        e.seal_execution(execution.root)
    assert not (execution.root / 'seal').exists()
    with pytest.raises(ValueError, match='duplicate'):
        e.run_lineage(execution.root, 'gpt2', 0)
    terminal = execution.root / 'runs/qwen2/0/terminal/manifest.json'
    with rewritten(terminal, lambda m: m.update(actual_optimizer_steps=1), 'manifest_digest'):
        with pytest.raises(ValueError, match='terminal'):
            e.validate_terminal(execution, 'qwen2', 0)
    duplicate = terminal.parent.parent / 'duplicate-terminal'
    duplicate.mkdir()
    try:
        with pytest.raises(ValueError, match='duplicate'):
            e.validate_terminal(execution, 'qwen2', 0)
    finally:
        duplicate.rmdir()


def test_all_twenty_validate_before_engineering_seal(completed):
    execution = completed[2]
    for a in e.ARCHITECTURES:
        for fold in range(1, 5):
            e.run_lineage(execution.root, a, fold)
    seal = e.seal_execution(execution.root)
    assert seal.manifest['artifact_type'] == 'backbone_engineering_seal'
    assert len(seal.manifest['terminals']) == 20
    assert seal.manifest['execution_source'] == execution.manifest['execution_source']
    assert e.seal_execution(execution.root).manifest_digest == seal.manifest_digest
    with pytest.raises(ValueError, match='sealed'):
        e.run_lineage(execution.root, 'gpt2', 0)
    with pytest.raises(ValueError):
        verify_artifact(seal.path, expected_artifact_type='backbone_execution_seal', expected_schema_version=e.SEAL_VERSION)
