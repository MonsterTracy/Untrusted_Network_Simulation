"""Small synthetic tensors/manifests only; no real ToM or restoration."""

import ast
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from werewolf.artifact_io import canonical_json_bytes, publish_artifact, sha256_bytes
from werewolf.artifact_io.tensor_state import TensorValue, encode_tensor_container
from werewolf.observer_dtrs import final_tom_audit as audit
from werewolf.tom.final_capacity import derived_capacity
from werewolf.tom.temporal import publish_temporal


def git(root, *args):
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_AUTHOR_NAME='Fixture', GIT_AUTHOR_EMAIL='fixture@example.invalid',
               GIT_COMMITTER_NAME='Fixture', GIT_COMMITTER_EMAIL='fixture@example.invalid')
    return subprocess.check_output(['git', '-C', str(root), '-c', 'core.hooksPath=/dev/null', *args],
                                   env=env, stderr=subprocess.PIPE).decode().strip()


def rewrite(path, change):
    m = json.loads(path.read_bytes())
    change(m)
    m.pop('manifest_digest')
    m['manifest_digest'] = audit.digest(m)
    path.write_bytes(canonical_json_bytes(m))


def build(root, *, temporal_change=None, runtime_impl=None, config_runtime_revision=None):
    checkout = root / 'predictor'
    (checkout / 'werewolf').mkdir(parents=True)
    (checkout / 'werewolf/core.py').write_text('# artificial implementation only\n')
    (checkout / 'run_random.py').write_text('# artificial control source only\n')
    git(checkout, 'init', '-q')
    git(checkout, 'add', '--', 'werewolf/core.py', 'run_random.py')
    git(checkout, 'commit', '-qm', 'controlled source')
    source = audit.checkout_identity(checkout)
    saved_source = {k: source[k] for k in ('source_revision', 'implementation_digest')}
    design = json.loads((audit.ROOT / 'configs/paper/tom-backbone-study-v1.json').read_bytes())
    config = design['reference_protocol']
    runtime = {'implementation_digest': runtime_impl or source['implementation_digest'],
               'source_revision': config_runtime_revision or config['source_revision'],
               'python': 'fixture-version', 'platform': 'fixture-platform'}
    initial = {'architecture': 'qwen3', 'fold': 0, 'temporal_condition': 'explicit_day_phase',
               'state_digest': 'a' * 64}
    # Only envelope-level links, intentionally not real publication/Primary data.
    pub = publish_artifact(root / 'parents/publication', manifest_fields={
        'artifact_type': 'development_publication', 'schema_version': 'classic7_development_publication_v1',
        'publication_id': 'artificial-publication', 'role_sidecar_digest': 'b' * 64},
        files={'restricted/role_sidecar.json': b'RESTRICTED_SENTINEL_NEVER_PARSE_OR_EMIT'})
    study = publish_artifact(root / 'parents/study', manifest_fields={
        'artifact_type': 'backbone_tom_study', 'schema_version': 'classic7_backbone_study_v1',
        'publication_path': str(pub.path), 'initializations': [initial],
        'protocol_inputs': {'publication_digest': pub.manifest_digest}}, files={})
    fit_path = root / 'lifecycle/fits/artificial'
    p = {'schema_version': audit.FIT_VERSION, 'architecture': 'qwen3',
        'temporal_condition': 'explicit_day_phase', 'population': 'non_wolf_alive', 'partition': 'all_development',
        'config': config, 'graph': {'graph_digest': 'c' * 64}, 'capacity': derived_capacity(1024),
        'rng_version': 'classic7_qwen3_final_training_rng_v1', 'publication_digest': pub.manifest_digest,
        'study_digest': study.manifest_digest, 'initial_state_digest': initial['state_digest'],
        'source': saved_source, 'runtime': runtime, 'destination_path': str(fit_path)}
    protocol_digest = audit.digest(p)
    temp = publish_temporal(root / 'artificial-temporal', 256, protocol_digest)
    if temporal_change:
        temporal_change(temp.path)
    tm = json.loads((temp.path / temp.manifest_name).read_bytes())
    files = {'schedule.json': canonical_json_bytes({'artificial': 'no schedule reconstruction claimed'}),
             'primary.json': b'RESTRICTED_PRIMARY_SENTINEL_ONLY_HASHED'}
    files.update({'temporal/' + q.name: q.read_bytes() for q in temp.path.iterdir()})
    fit = publish_artifact(fit_path, manifest_name='qwen3_final_manifest.json', manifest_fields={
        'artifact_type': 'qwen3_all_development_final', 'schema_version': audit.FIT_VERSION,
        'protocol_inputs': p, 'protocol_digest': protocol_digest, 'study_path': str(study.path),
        'publication_path': str(pub.path), 'publication_id': 'artificial-publication',
        'game_ids': [f'game-{i:04d}' for i in range(1500)], 'primary_sidecar_digest': 'b' * 64,
        'initialization': initial, 'temporal_artifact_digest': tm['manifest_digest'],
        'schedule_digest': sha256_bytes(files['schedule.json']), 'optimizer_steps': 31500}, files=files)
    binding = audit._binding(fit)
    runs = fit.path.parent.parent / 'runs' / fit.manifest_digest
    containers, payloads = {}, {}
    for name in ('model', 'optimizer', 'rng'):
        containers[name], payloads[name + '.bin'] = encode_tensor_container(
            {'artificial_tensor': TensorValue(np.array([.25, .75], dtype='<f4'), name == 'model')},
            payload_path=name + '.bin')
    payloads['loss.jsonl'] = b'{"artificial":true}\n'  # No semantic loss/schedule check claimed.
    previous, ancestry = None, []
    for step in audit.STEPS:
        a = publish_artifact(runs / 'recovery' / str(step), manifest_fields={
            'artifact_type': 'qwen3_final_recovery', 'schema_version': 'classic7_qwen3_final_recovery_v1',
            'binding': binding, 'optimizer_step': step, 'schedule_position': step,
            'previous_checkpoint_digest': previous, 'containers': containers,
            'optimizer_scalars': {'restricted': 'OPTIMIZER_SENTINEL'}, 'rng_scalars': {},
            'loss_digest': sha256_bytes(payloads['loss.jsonl'])}, files=payloads)
        previous = a.manifest_digest
        ancestry.append({'step': step, 'digest': previous})
    terminal = publish_artifact(runs / 'terminal', manifest_fields={
        'artifact_type': 'qwen3_final_terminal', 'schema_version': 'classic7_qwen3_final_terminal_v1',
        'binding': binding, 'terminal_step': 31500, 'last_recovery_digest': previous,
        'model_digest': audit.digest(containers['model']), 'checkpoint_ancestry': ancestry}, files={})
    seal = publish_artifact(runs / 'seal', manifest_fields={
        'artifact_type': 'qwen3_final_seal', 'schema_version': 'classic7_qwen3_final_seal_v1',
        'binding': binding, 'terminal_digest': terminal.manifest_digest, 'terminal_step': 31500}, files={})
    pins = audit._Pins(source['source_revision'], fit.manifest_digest, terminal.manifest_digest,
                       seal.manifest_digest, audit.digest(containers['model']))
    return fit.path, checkout, runs, pins


def run(fixture, *, chain=False, pins=None):
    fit, checkout, _, defaults = fixture
    return audit.audit_final_tom(fit, checkout, chain_files=chain, _pins=pins or defaults)


def snapshot(root):
    return {str(p.relative_to(root)): (sha256_bytes(p.read_bytes()), p.stat().st_mode,
                                      p.stat().st_size, p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('chain', [False, True])
def test_controlled_success_readonly_redacted_and_explicit_scope(tmp_path, chain):
    fixture = build(tmp_path)
    before = snapshot(tmp_path)
    r = run(fixture, chain=chain)
    assert r['status'] == 'PASS_WITH_EXPLICIT_SCOPE_LIMITS', r
    assert r['scope'] == 'CONTROLLED_FIXTURE'
    assert snapshot(tmp_path) == before
    c = r['checks']['recovery']
    assert c['checkpoint_count'] == 32
    raw = (fixture[2] / 'recovery/31500/model.bin').read_bytes()
    assert c['model_bin_raw_sha256'] == hashlib.sha256(raw).hexdigest()
    assert c['model_bin_raw_sha256'] != c['model_container_descriptor_sha256']
    assert c['all_recovery_payloads'] == ('PASS_BYTES_AND_CONTAINER_CODECS' if chain else 'NOT_RUN')
    assert r['full_semantic_recovery'] == r['inference_qualification'] == 'NOT_RUN'
    assert r['checks']['research_links']['restricted_population_reconstruction'] == 'NOT_RUN'
    assert r['checks']['source']['runtime']['runtime_qualification'] == 'NOT_ESTABLISHED'
    text = json.dumps(r)
    for forbidden in ('SENTINEL', 'game-0000', str(tmp_path), 'artificial_tensor', 'optimizer_scalars'):
        assert forbidden not in text


@pytest.mark.parametrize('field', ['fit', 'terminal', 'seal', 'model_descriptor', 'revision'])
def test_production_pins_fail_closed_and_cannot_confuse_raw_and_descriptor(tmp_path, field):
    f = build(tmp_path)
    wrong = sha256_bytes((f[2] / 'recovery/31500/model.bin').read_bytes()) if field == 'model_descriptor' else 'e' * (40 if field == 'revision' else 64)
    r = run(f, pins=replace(f[3], **{field: wrong}))
    assert r['status'] == 'FAILED'


@pytest.mark.parametrize('case', ['binding', 'previous', 'step', 'container', 'missing', 'extra', 'model_bytes'])
def test_corrupt_or_rehashed_recovery_fails(tmp_path, case):
    f = build(tmp_path)
    node = f[2] / 'recovery/1000'
    manifest = node / 'manifest.json'
    changes = {
        'binding': lambda m: m['binding'].update(fit_digest='d' * 64),
        'previous': lambda m: m.update(previous_checkpoint_digest='d' * 64),
        'step': lambda m: m.update(schedule_position=2000),
        'container': lambda m: m['containers']['model'].update(payload_sha256='d' * 64),
    }
    if case in changes:
        rewrite(manifest, changes[case])
    elif case == 'missing':
        (node / 'rng.bin').unlink()
    elif case == 'extra':
        (node / 'unexpected.bin').write_bytes(b'not allowed')
    else:
        (f[2] / 'recovery/31500/model.bin').write_bytes(b'corrupt!')
    assert run(f)['status'] == 'FAILED'


def test_metadata_does_not_claim_old_payload_integrity(tmp_path):
    f = build(tmp_path)
    (f[2] / 'recovery/1000/optimizer.bin').write_bytes(b'corrupt!')
    assert run(f)['status'] == 'PASS_WITH_EXPLICIT_SCOPE_LIMITS'
    assert run(f, chain=True)['status'] == 'FAILED'


@pytest.mark.parametrize('case', ['duplicate', 'order', 'last', 'step'])
def test_reanchored_terminal_ancestry_still_requires_original_binding(tmp_path, case):
    f = build(tmp_path)
    path = f[2] / 'terminal/manifest.json'
    def change(m):
        if case == 'duplicate':
            m['checkpoint_ancestry'][1] = m['checkpoint_ancestry'][0]
        elif case == 'order':
            m['checkpoint_ancestry'].reverse()
        elif case == 'last':
            m['last_recovery_digest'] = 'd' * 64
        else:
            m['terminal_step'] = 31000
    rewrite(path, change)
    terminal = json.loads(path.read_bytes())['manifest_digest']
    path = f[2] / 'seal/manifest.json'
    rewrite(path, lambda m: m.update(terminal_digest=terminal))
    seal = json.loads(path.read_bytes())['manifest_digest']
    # This reanchoring exists solely in the visible CONTROLLED_FIXTURE API.
    r = run(f, pins=replace(f[3], terminal=terminal, seal=seal))
    assert r['failure']['code'] == 'terminal_seal_binding'


@pytest.mark.parametrize('case', ['shape', 'parent', 'day_parent', 'bytes', 'day_nan'])
def test_temporal_corruption_with_rehashed_fit_is_rejected(tmp_path, case):
    def change(root):
        if case in ('shape', 'parent'):
            rewrite(root / 'phase_codebook.manifest.json',
                    lambda m: m.update(**({'day_shape': [256, 128]} if case == 'shape' else {'parent_digest': 'd' * 64})))
        elif case == 'day_parent':
            path = root / 'canonical_day_code_table.manifest.json'
            value = json.loads(path.read_bytes());value['parent_digest'] = 'd' * 64
            path.write_bytes(canonical_json_bytes(value))
            def update(m):
                m['file_table'][path.name].update(sha256=sha256_bytes(path.read_bytes()))
            rewrite(root / 'phase_codebook.manifest.json', update)
        else:
            path = root / 'canonical_day_code_table.bin'
            path.write_bytes(np.full((257, 128), np.nan if case == 'day_nan' else 0., dtype='<f4').tobytes())
            if case == 'day_nan':
                raw_sha = sha256_bytes(path.read_bytes())
                dm = root / 'canonical_day_code_table.manifest.json'
                value = json.loads(dm.read_bytes());value['sha256'] = raw_sha
                dm.write_bytes(canonical_json_bytes(value))
                def update(m):
                    m['day_digest'] = raw_sha
                    m['file_table'][path.name]['sha256'] = raw_sha
                    m['file_table'][dm.name].update(sha256=sha256_bytes(dm.read_bytes()))
                rewrite(root / 'phase_codebook.manifest.json', update)
    f = build(tmp_path, temporal_change=change)
    assert run(f)['status'] == 'FAILED'


@pytest.mark.parametrize('case', ['source_dirty', 'assumed_unchanged', 'runtime_impl', 'config_revision'])
def test_source_and_runtime_bindings_not_interchangeable(tmp_path, case):
    f = build(tmp_path, runtime_impl='d' * 64 if case == 'runtime_impl' else None,
              config_runtime_revision='d' * 40 if case == 'config_revision' else None)
    if case in ('source_dirty', 'assumed_unchanged'):
        if case == 'assumed_unchanged':
            git(f[1], 'update-index', '--assume-unchanged', 'werewolf/core.py')
        (f[1] / 'werewolf/core.py').write_text('# changed bytes\n')
    assert run(f)['failure']['stage'] == 'source'


def test_symlink_ancestors_are_rejected(tmp_path):
    f = build(tmp_path)
    link = tmp_path / 'alias'
    link.symlink_to(f[0].parent, target_is_directory=True)
    r = audit.audit_final_tom(link / f[0].name, f[1], _pins=f[3])
    assert r['failure']['code'] == 'artifact_symlink_ancestor'


def test_evidence_output_exclusive_outside_protected_artifacts(tmp_path):
    f = build(tmp_path)
    r = run(f)
    path = tmp_path / 'evidence.json'
    raw_sha = audit.write_evidence(path, r, fit_path=f[0], predictor_checkout=f[1])
    assert raw_sha == sha256_bytes(path.read_bytes())
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_bytes()) == r
    # A shared storage ancestor is not itself an artifact: allow an independent
    # evidence directory next to fits/runs, even for shallow original layouts.
    external = tmp_path / 'lifecycle/source-audits'
    external.mkdir()
    audit.write_evidence(external / 'evidence.json', r, fit_path=f[0], predictor_checkout=f[1])
    with pytest.raises(FileExistsError):
        audit.write_evidence(path, r, fit_path=f[0], predictor_checkout=f[1])
    for bad in (f[0] / 'evidence.json', f[2] / 'evidence.json', f[1] / 'evidence.json',
                tmp_path / 'parents/publication/evidence.json'):
        with pytest.raises(ValueError):
            audit.write_evidence(bad, r, fit_path=f[0], predictor_checkout=f[1])
        assert not bad.exists()


def test_cli_missing_evidence_fails_safely_and_does_not_load_tom(tmp_path):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    cmd = [sys.executable, '-B', str(audit.ROOT / 'scripts/audit_qwen3_final_provenance.py'),
           '--fit', str(tmp_path / 'lifecycle/fits/DO_NOT_LEAK'),
           '--predictor-checkout', str(tmp_path / 'DO_NOT_LEAK'), '--output', str(tmp_path / 'evidence.json')]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert p.returncode == 2, p.stderr
    r = json.loads((tmp_path / 'evidence.json').read_bytes())
    assert r['status'] == 'FAILED'
    assert 'DO_NOT_LEAK' not in json.dumps(r)
    assert r['full_semantic_recovery'] == 'NOT_RUN'
    assert json.loads(p.stdout)['evidence_file_raw_sha256'] == sha256_bytes((tmp_path / 'evidence.json').read_bytes())


@pytest.mark.parametrize('target,style', [
    (target, style) for target in ('predictor', 'fit', 'runs', 'recovery', 'temporal', 'publication', 'other_source')
    for style in ('direct', 'input_symlink', 'input_dotdot', 'output_alias', 'output_dotdot')
])
def test_cli_rejected_paths_preserve_all_inputs(tmp_path, target, style):
    """Public CLI, production pins, failed audit: no output inside any input."""
    f = build(tmp_path / 'artifacts')
    other = tmp_path / 'other-source'
    other.mkdir()
    (other / 'run_random.py').write_text('# controlled source\n')
    git(other, 'init', '-q');git(other, 'add', '--', 'run_random.py');git(other, 'commit', '-qm', 'fixture')
    roots = {'predictor': f[1], 'fit': f[0], 'runs': f[2], 'recovery': f[2] / 'recovery',
             'temporal': f[0] / 'temporal', 'publication': tmp_path / 'artifacts/parents/publication',
             'other_source': other}
    fit_path, checkout = f[0], f[1]
    if style == 'input_symlink':
        fit_path, checkout = tmp_path / 'fit-alias', tmp_path / 'predictor-alias'
        fit_path.symlink_to(f[0], target_is_directory=True)
        checkout.symlink_to(f[1], target_is_directory=True)
    elif style == 'input_dotdot':
        fit_path, checkout = f[0] / '..' / f[0].name, f[1] / '..' / f[1].name
    output = roots[target] / 'forbidden-evidence.json'
    if style == 'output_alias':
        alias = tmp_path / 'output-alias'
        alias.symlink_to(roots[target], target_is_directory=True)
        output = alias / output.name
    elif style == 'output_dotdot':
        output = roots[target] / '..' / roots[target].name / output.name
    before = snapshot(tmp_path)
    entries = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*'))
    identities = [audit.checkout_identity(p) for p in (f[1], other)]
    assert all(i['clean'] for i in identities)
    cmd = [sys.executable, '-B', str(audit.ROOT / 'scripts/audit_qwen3_final_provenance.py'),
           '--fit', str(fit_path), '--predictor-checkout', str(checkout), '--output', str(output)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', PYTHONPATH=''))
    # Check the P1 symptom, not just the return code. These include Git metadata.
    assert snapshot(tmp_path) == before, 'protected inputs changed after failed CLI audit'
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*')) == entries
    assert [audit.checkout_identity(p) for p in (f[1], other)] == identities
    assert p.returncode == 3, p.stderr
    assert json.loads(p.stdout)['status'] == 'OUTPUT_REFUSED'


@pytest.mark.parametrize('kind', ['success', 'bad_manifest', 'source_mismatch'])
@pytest.mark.parametrize('mode', ['metadata', 'chain-files'])
def test_controlled_cli_external_success_failure_and_overwrite(tmp_path, monkeypatch, capsys, kind, mode):
    """Run the unmodified CLI with explicitly synthetic pins at the audit seam.

    Production CLI has no pin override. Real public subprocess rejection cases
    above use unchanged production pins, including the original P1 failure path.
    """
    import runpy
    f = build(tmp_path / 'artifacts')
    if kind == 'bad_manifest':
        path = f[0] / 'qwen3_final_manifest.json'
        path.write_bytes(path.read_bytes() + b'\n')
    elif kind == 'source_mismatch':
        (f[1] / 'werewolf/core.py').write_text('# intentional source mismatch\n')
    original = audit.audit_final_tom
    def controlled(fit_path, checkout, *, chain_files=False):
        return original(fit_path, checkout, chain_files=chain_files, _pins=f[3])
    monkeypatch.setattr(audit, 'audit_final_tom', controlled)
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    output = tmp_path / 'external-evidence.json'
    monkeypatch.setattr(sys, 'argv', ['audit_qwen3_final_provenance.py', '--fit', str(f[0]),
        '--predictor-checkout', str(f[1]), '--mode', mode, '--output', str(output)])
    before = snapshot(tmp_path / 'artifacts')
    identity = audit.checkout_identity(f[1])
    def invoke():
        with pytest.raises(SystemExit) as result:
            runpy.run_path(str(audit.ROOT / 'scripts/audit_qwen3_final_provenance.py'), run_name='__main__')
        return result.value.code, json.loads(capsys.readouterr().out)
    code, stdout = invoke()
    assert code == (0 if kind == 'success' else 2)
    r = json.loads(output.read_bytes())
    assert r['scope'] == 'CONTROLLED_FIXTURE'
    assert r['status'] == ('PASS_WITH_EXPLICIT_SCOPE_LIMITS' if kind == 'success' else 'FAILED')
    if kind != 'success':
        assert r['failure']['stage'] == ('fit' if kind == 'bad_manifest' else 'source')
    assert stdout['evidence_file_raw_sha256'] == sha256_bytes(output.read_bytes())
    assert r['evidence_digest'] == audit.digest({k: v for k, v in r.items() if k != 'evidence_digest'})
    assert output.stat().st_mode & 0o777 == 0o600
    first = output.read_bytes()
    assert invoke()[0] == 3
    assert output.read_bytes() == first
    assert snapshot(tmp_path / 'artifacts') == before
    assert audit.checkout_identity(f[1]) == identity


@pytest.mark.parametrize('kind', ['dangling_input', 'input_loop', 'dangling_output', 'output_loop', 'output_file_alias'])
def test_unresolvable_or_existing_aliases_fail_closed(tmp_path, kind):
    f = build(tmp_path / 'artifacts')
    fit, checkout, output = f[0], f[1], tmp_path / 'new.json'
    if kind in ('dangling_input', 'input_loop'):
        checkout = tmp_path / 'bad-checkout'
        checkout.symlink_to(tmp_path / 'missing' if kind == 'dangling_input' else checkout,
                            target_is_directory=True)
    elif kind in ('dangling_output', 'output_loop'):
        parent = tmp_path / 'bad-parent'
        parent.symlink_to(tmp_path / 'missing' if kind == 'dangling_output' else parent,
                          target_is_directory=True)
        output = parent / 'new.json'
    else:
        output.symlink_to(f[2] / 'recovery/31500/model.bin')
    before = snapshot(tmp_path / 'artifacts')
    with pytest.raises((OSError, ValueError)):
        audit.write_evidence(output, {'status': 'FAILED'}, fit_path=fit, predictor_checkout=checkout)
    assert snapshot(tmp_path / 'artifacts') == before


def test_success_and_failure_reports_cannot_write_audit_source_alias(tmp_path, monkeypatch):
    f = build(tmp_path / 'artifacts')
    fake_source = tmp_path / 'audit-source'
    fake_source.mkdir()
    alias = tmp_path / 'audit-source-alias'
    alias.symlink_to(fake_source, target_is_directory=True)
    monkeypatch.setattr(audit, 'ROOT', alias)
    for status in ('FAILED', 'PASS_WITH_EXPLICIT_SCOPE_LIMITS'):
        with pytest.raises(ValueError):
            audit.write_evidence(fake_source / 'forbidden.json', {'status': status},
                                 fit_path=f[0], predictor_checkout=f[1])
    assert list(fake_source.iterdir()) == []


def test_writer_fsyncs_file_and_parent_after_exclusive_creation(tmp_path, monkeypatch):
    import stat
    f = build(tmp_path / 'artifacts')
    original = os.fsync
    synced = []
    def fsync(fd):
        synced.append('directory' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file')
        return original(fd)
    monkeypatch.setattr(os, 'fsync', fsync)
    audit.write_evidence(tmp_path / 'external.json', {'status': 'FAILED'}, fit_path=f[0], predictor_checkout=f[1])
    assert synced == ['file', 'directory']


def test_existing_directory_identity_blocks_case_alias(tmp_path):
    f = build(tmp_path / 'artifacts')
    # POSIX resolve() need not canonicalize case on a case-insensitive volume.
    alias = f[2].with_name(f[2].name.upper())
    if not alias.is_dir() or not alias.samefile(f[2]):
        pytest.skip('filesystem does not expose this case alias')
    before = snapshot(tmp_path)
    with pytest.raises(ValueError):
        audit.write_evidence(alias / 'forbidden.json', {'status': 'FAILED'},
                             fit_path=f[0], predictor_checkout=f[1])
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize('style', ['symlink', 'dotdot'])
def test_cli_failed_alias_inputs_can_write_only_external_evidence(tmp_path, style):
    f = build(tmp_path / 'artifacts')
    fit, checkout = f[0], f[1]
    if style == 'symlink':
        fit, checkout = tmp_path / 'fit-alias', tmp_path / 'checkout-alias'
        fit.symlink_to(f[0], target_is_directory=True);checkout.symlink_to(f[1], target_is_directory=True)
    else:
        fit, checkout = f[0] / '..' / f[0].name, f[1] / '..' / f[1].name
    output = tmp_path / 'external.json'
    before = snapshot(tmp_path / 'artifacts')
    identity = audit.checkout_identity(f[1])
    cmd = [sys.executable, '-B', str(audit.ROOT / 'scripts/audit_qwen3_final_provenance.py'),
           '--fit', str(fit), '--predictor-checkout', str(checkout), '--output', str(output)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', PYTHONPATH=''))
    assert p.returncode == 2, p.stderr
    assert json.loads(output.read_bytes())['status'] == 'FAILED'
    assert json.loads(p.stdout)['evidence_file_raw_sha256'] == sha256_bytes(output.read_bytes())
    assert snapshot(tmp_path / 'artifacts') == before
    assert audit.checkout_identity(f[1]) == identity


@pytest.mark.parametrize('kind', ['file', 'directory', 'hardlink', 'dangling_symlink'])
def test_existing_output_objects_are_never_replaced(tmp_path, kind):
    f = build(tmp_path / 'artifacts')
    output = tmp_path / 'existing'
    if kind == 'file':
        output.write_bytes(b'existing evidence is immutable')
    elif kind == 'directory':
        output.mkdir()
    elif kind == 'hardlink':
        output.hardlink_to(f[2] / 'recovery/31500/model.bin')
    else:
        output.symlink_to(tmp_path / 'absent')
    before = snapshot(tmp_path)
    entries = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*'))
    with pytest.raises(FileExistsError):
        audit.write_evidence(output, {'status': 'FAILED'}, fit_path=f[0], predictor_checkout=f[1])
    assert snapshot(tmp_path) == before
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*')) == entries


def test_audit_pins_match_client_without_importing_or_instantiating_client():
    tree = ast.parse((audit.ROOT / 'scripts/qwen3_gameplay_predictor.py').read_text())
    constants = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                 if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                 and isinstance(n.value, ast.Constant)}
    pins = audit._Pins()
    assert (pins.revision, pins.fit, pins.terminal, pins.seal, pins.model_descriptor) == tuple(
        constants[k] for k in ('SOURCE_REVISION', 'FIT_DIGEST', 'TERMINAL_DIGEST', 'SEAL_DIGEST', 'MODEL_DIGEST'))
    code = "from werewolf.observer_dtrs.final_tom_audit import audit_final_tom; import sys; assert 'werewolf.tom.qwen3_final' not in sys.modules; assert 'run_random' not in sys.modules; assert 'scripts.qwen3_gameplay_predictor' not in sys.modules"
    subprocess.run([sys.executable, '-B', '-c', code], check=True, cwd=audit.ROOT)


def test_reader_source_equality_and_drift_are_explicit(tmp_path):
    paths = ('werewolf/artifact_io/__init__.py', 'werewolf/artifact_io/canonical.py',
             'werewolf/artifact_io/tensor_state.py', 'werewolf/tom/temporal.py',
             'werewolf/tom/final_capacity.py', 'werewolf/canonical_collection/pre.py',
             'werewolf/canonical_collection/public_history.py', 'werewolf/structured_history.py',
             'configs/paper/tom-backbone-study-v1.json')
    for p in paths:
        target = tmp_path / p
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((audit.ROOT / p).read_bytes())
    assert audit._reader_sources(tmp_path)['status'] == 'PASS_BYTES_MATCH_PINNED_CHECKOUT'
    (tmp_path / paths[0]).write_bytes(b'# drift\n')
    with pytest.raises(ValueError, match='audit_reader_source_mismatch'):
        audit._reader_sources(tmp_path)


def test_manifest_change_during_read_stops_audit(tmp_path, monkeypatch):
    f = build(tmp_path)
    original = audit._chain
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        rewrite(f[2] / 'terminal/manifest.json', lambda m: m.update(terminal_step=1))
        return result
    monkeypatch.setattr(audit, '_chain', changed)
    r = run(f)
    assert r['failure']['stage'] == 'concurrent_manifest_change'
    assert r['failure']['code'] == 'manifest_changed_during_audit'


def test_full_existing_recovery_verifier_is_never_called(tmp_path, monkeypatch):
    # Guard imports, rather than constructing/importing a real model just to mock it.
    import builtins
    original = builtins.__import__
    def restricted(name, *args, **kwargs):
        assert name not in ('werewolf.tom.qwen3_final', 'werewolf.tom.backbone_model', 'run_random')
        return original(name, *args, **kwargs)
    f = build(tmp_path)
    monkeypatch.setattr(builtins, '__import__', restricted)
    assert run(f, chain=True)['full_semantic_recovery'] == 'NOT_RUN'
