"""Tiny synthetic publications only; never opens real paper results."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

import pytest

from tests.development_publication.test_development_publication import _publication
from tests.tom.test_experiment import experiment_config
from werewolf.artifact_io import canonical_json_bytes, sha256_bytes, publish_artifact, verify_artifact
from werewolf.tom import paper_study_execution as m, training
from werewolf.tom.experiment import runtime_provenance
from werewolf.tom.paper_study import CONTRACT, STUDY_VERSION


def synthetic_contract(path):
    # Only Git/source attestation is replaced; all experiment/checkpoint validation is real.
    return verify_artifact(path, expected_artifact_type='paper_tom_study_contract',
                           expected_schema_version=STUDY_VERSION)


def prepared(tmp_path, cadence=7):
    _, handles = _publication(tmp_path)
    config = replace(experiment_config(handles.public.public_view.max_structured_token_count),
                     bootstrap_replicates=10000, recovery_cadence=cadence)
    contract = deepcopy(CONTRACT)
    contract['publication_id'] = handles.public.public_view.publication_id
    source = runtime_provenance(config)
    artifact = publish_artifact(tmp_path/'contract', manifest_fields={
        'artifact_type':'paper_tom_study_contract', 'schema_version':STUDY_VERSION,
        'contract':contract, 'source':{k:source[k] for k in ('source_revision','implementation_digest')}}, files={})
    return m.prepare_study(artifact.path, handles.public, config, tmp_path/'study')


@pytest.fixture(autouse=True)
def source_attestation(monkeypatch):
    monkeypatch.setattr(m, 'open_contract', synthetic_contract)


def test_scientific_revision_is_separate_from_execution_attestation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from werewolf.tom import paper_study as contracts

    _, handles = _publication(tmp_path)
    config = replace(experiment_config(handles.public.public_view.max_structured_token_count),
                     bootstrap_replicates=10000, source_revision='old-scientific-revision')
    design = deepcopy(CONTRACT)
    design['publication_id'] = handles.public.public_view.publication_id
    monkeypatch.setattr(contracts, 'CONTRACT', design)
    # Only the external Git attestation is synthetic. Do not replace open_contract:
    # its real HEAD/digest checks must run in both lifecycle entry points.
    source = {'source_revision': 'current-execution-revision',
              'implementation_digest': runtime_provenance(config)['implementation_digest']}
    monkeypatch.setattr(contracts, 'attest_source', lambda: dict(source))
    monkeypatch.setattr(m, 'open_contract', contracts.open_contract)
    design_path = tmp_path/'design.json'
    design_path.write_bytes(canonical_json_bytes(design))
    protocol_path = tmp_path/'protocol.json'
    protocol_bytes = canonical_json_bytes(asdict(config))
    protocol_path.write_bytes(protocol_bytes)
    contract = contracts.prepare_contract(design_path, tmp_path/'contract')
    study = m.prepare_study(contract.path, handles.public, config, tmp_path/'study')
    reopened = m.open_study(study.artifact.path)
    assert reopened.artifact.manifest_digest == study.artifact.manifest_digest
    assert reopened.full.config.source_revision == 'old-scientific-revision'
    assert reopened.full.manifest['runtime']['source_revision'] == 'old-scientific-revision'
    assert contract.manifest['source']['source_revision'] == 'current-execution-revision'
    assert reopened.full.manifest['runtime']['implementation_digest'] == source['implementation_digest']

    for field, wrong, message in (
        ('source_revision', 'different-execution-revision', 'current HEAD'),
        ('implementation_digest', '0'*64, 'implementation digest')):
        with monkeypatch.context() as patch:
            patch.setattr(contracts, 'attest_source', lambda: {**source, field: wrong})
            with pytest.raises(ValueError, match=message):
                m.open_study(study.artifact.path)
            with pytest.raises(ValueError, match=message):
                m.prepare_study(contract.path, handles.public, config, tmp_path/'rejected')
            assert not (tmp_path/'rejected').exists()

    def corrupt_runtime(value):
        value['runtime']['implementation_digest'] = '0'*64
    with rewritten(study.full.path/'experiment_manifest.json', corrupt_runtime, 'manifest_digest'):
        with pytest.raises(ValueError):
            m.open_study(study.artifact.path)
    with monkeypatch.context() as patch:
        runtime = {**study.full.manifest['runtime'], 'implementation_digest': '0'*64}
        patch.setattr(m, 'prepare_experiment', lambda *a: SimpleNamespace(manifest={'runtime': runtime}))
        with pytest.raises(ValueError, match='implementation binding mismatch'):
            m.prepare_study(contract.path, handles.public, config, tmp_path/'bad-runtime')
        assert not (tmp_path/'bad-runtime/execution').exists()

    with pytest.raises(ValueError, match='publication/config binding'):
        m.prepare_study(contract.path, handles.public, replace(config, bootstrap_replicates=9999),
                        tmp_path/'bad-control')
    def corrupt_publication(value):
        value['contract']['publication_id'] = 'different-publication'
    with monkeypatch.context() as patch:
        patch.setattr(contracts, 'CONTRACT', {**design, 'publication_id': 'different-publication'})
        with rewritten(contract.path/'manifest.json', corrupt_publication, 'manifest_digest'):
            with pytest.raises(ValueError, match='publication/config binding'):
                m.prepare_study(contract.path, handles.public, config, tmp_path/'bad-publication')
            with pytest.raises(ValueError, match='parent/source/control'):
                m.open_study(study.artifact.path)
    assert protocol_path.read_bytes() == protocol_bytes
    assert config.source_revision == 'old-scientific-revision'


@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(m, 'open_contract', synthetic_contract)
        study = prepared(tmp_path_factory.mktemp('paper-tiny'))
        for lineage in m.EXPECTED_LINEAGES:
            m._train_lineage(study, *lineage)
        return study


@contextmanager
def rewritten(path, change, digest_key):
    original = path.read_bytes()
    value = json.loads(original)
    del value[digest_key]
    change(value)
    value[digest_key] = sha256_bytes(canonical_json_bytes(value))
    path.write_bytes(canonical_json_bytes(value))
    try:
        yield
    finally:
        path.write_bytes(original)


def test_exact_twenty_and_shared_controls(completed):
    assert m.EXPECTED_LINEAGES == tuple((f,c,i) for f in m.FAMILIES
        for c in ('implicit','explicit_day_phase') for i in range(5))
    assert len(set(m.EXPECTED_LINEAGES)) == 20
    a, f = completed.agnostic, completed.full
    assert a.artifact.manifest['full_experiment_digest'] == f.digest
    assert 'shared_controls' not in a.artifact.manifest
    assert all('/initial_state/' in path for path in a.artifact.manifest['file_table'])
    assert a.manifest['protocol_digest'] != f.manifest['protocol_digest']
    assert a.config == f.config
    for i in range(5):
        assert a.schedule(i) == f.schedule(i)
        assert a.fold(i)['paired_initial_state_digest'] != f.fold(i)['paired_initial_state_digest']


@pytest.mark.parametrize('change', ['folds','games','schedule','rotations','optimizer','max_seq_len','temporal','population'])
def test_shared_control_mismatch_rejected(completed, change):
    def corrupt(value):
        if change == 'folds': value['folds'].reverse()
        elif change == 'games': value['game_ids'].reverse()
        elif change == 'schedule': value['folds'][0]['schedule_digest'] = '0'*64
        elif change == 'rotations': value['protocol_inputs']['config']['rotation_cycles'] += 1
        elif change == 'optimizer': value['protocol_inputs']['config']['learning_rate'] *= 2
        elif change == 'max_seq_len': value['protocol_inputs']['config']['max_seq_len'] += 1
        elif change == 'temporal': value['temporal_artifact_digest'] = '0'*64
        else: value['primary_sidecar_digest'] = '0'*64
    # There is exactly one control owner. Even a rehashed replacement of that
    # owner is rejected by the immutable execution/agnostic parent bindings.
    with rewritten(completed.full.path/'experiment_manifest.json', corrupt, 'manifest_digest'):
        with pytest.raises(ValueError, match='mismatch|five folds'):
            m.open_study(completed.artifact.path)


def test_missing_twentieth_and_preseal_barrier(completed):
    with pytest.raises(ValueError, match='all-twenty'):
        m.evaluation_eligibility(completed.artifact.path)
    _, root = m._lineage(completed, *m.EXPECTED_LINEAGES[-1])
    path = root/'terminal_checkpoint'; hidden = root/'.missing_terminal'
    path.rename(hidden)
    try:
        with pytest.raises((ValueError, FileNotFoundError)):
            m.seal_study(completed.artifact.path)
        assert not completed.seal_path.exists()
    finally:
        hidden.rename(path)


def test_extra_lineage_rejected(completed):
    path = completed.agnostic.runs_path/'implicit'/'5'; path.mkdir()
    try:
        with pytest.raises(ValueError, match='unexpected study fold'):
            m.seal_study(completed.artifact.path)
    finally:
        path.rmdir()


@pytest.mark.parametrize('field', ['model_family','fold','temporal_condition','study_digest'])
def test_wrong_lineage_binding(completed, field):
    _, path = m._lineage(completed, *m.EXPECTED_LINEAGES[-1])
    def change(v): v[field] = 0 if field == 'fold' else 'wrong'
    with rewritten(path/'study_lineage.json', change, 'record_digest'):
        with pytest.raises(ValueError, match='lineage parent'):
            m.seal_study(completed.artifact.path)


@pytest.mark.parametrize('field', ['purpose','experiment_digest','fold','temporal_condition',
                                  'schedule_digest','paired_initial_state_digest'])
def test_wrong_terminal_provenance(completed, field):
    _, path = m._lineage(completed, *m.EXPECTED_LINEAGES[-1])
    def change(v): v['provenance'][field] = 0 if field == 'fold' else 'wrong'
    with rewritten(path/'terminal_checkpoint/manifest.json', change, 'manifest_digest'):
        with pytest.raises(ValueError, match='terminal checkpoint provenance'):
            m.seal_study(completed.artifact.path)


def test_recovery_is_not_terminal(completed):
    _, path = m._lineage(completed, *m.EXPECTED_LINEAGES[-1])
    target = path/'terminal_checkpoint'; backup = path/'.terminal_backup'
    target.rename(backup)
    shutil.copytree(path/'recovery/7', target)
    try:
        with pytest.raises(ValueError): m.seal_study(completed.artifact.path)
    finally:
        shutil.rmtree(target); backup.rename(target)


def test_corrupt_terminal_rejected(completed):
    _, path = m._lineage(completed, *m.EXPECTED_LINEAGES[-1])
    path = path/'terminal_checkpoint/tensors.bin'; original = path.read_bytes()
    path.write_bytes(b'corrupt')
    try:
        with pytest.raises(ValueError): m.seal_study(completed.artifact.path)
    finally:
        path.write_bytes(original)


def test_complete_seal_immutable_training_closed_and_evaluation_opens(completed, monkeypatch):
    seal = m.seal_study(completed.artifact.path)
    before = completed.seal_path.read_bytes()
    try:
        assert len(seal['checkpoints']) == 20
        read = Path.read_bytes
        def no_held_out(path):
            assert not any(part.startswith('held_out_') for part in path.parts)
            assert not ('public' in path.parts and 'games' in path.parts)
            return read(path)
        monkeypatch.setattr(Path, 'read_bytes', no_held_out)
        assert m.evaluation_eligibility(completed.artifact.path) == seal
        for family in m.FAMILIES:
            with pytest.raises(ValueError, match='permanently closes'):
                m._train_lineage(completed, family, 'implicit', 0)
        with pytest.raises(ValueError, match='permanently closes'): m.run_study(completed.artifact.path)
        with pytest.raises(ValueError, match='permanently closes'): m.seal_study(completed.artifact.path)
        assert completed.seal_path.read_bytes() == before
        for bad in (seal['checkpoints'][:-1], seal['checkpoints']+[seal['checkpoints'][0]],
                    seal['checkpoints'][:-1]+[seal['checkpoints'][0]]):
            with rewritten(completed.seal_path, lambda v: v.update(checkpoints=bad), 'record_digest'):
                with pytest.raises(ValueError, match='coverage/identity'):
                    m.evaluation_eligibility(completed.artifact.path)
    finally:
        completed.seal_path.unlink()


def test_completed_lineages_are_not_retrained(completed, monkeypatch):
    monkeypatch.setattr(training, 'fit_steps', lambda *a, **k: pytest.fail('terminal retrained'))
    monkeypatch.setattr(m, '_process_entry', lambda *a: pytest.fail('terminal spawned'))
    for lineage in m.EXPECTED_LINEAGES:
        m._train_lineage(completed, *lineage)
    try:
        seal = m.run_study(completed.artifact.path)
        assert len(seal['checkpoints']) == 20
    finally:
        completed.seal_path.unlink(missing_ok=True)


def test_baseline_uses_shared_training_and_recovers_exactly(tmp_path, monkeypatch):
    study = prepared(tmp_path, cadence=2)
    calls = []
    fit, ce = training.fit_steps, training.game_balanced_cross_entropy
    def fit_spy(*args, **kwargs):
        calls.append(args[4])
        return fit(*args, **kwargs)
    losses = []
    def ce_spy(values):
        losses.append(len(values))
        return ce(values)
    monkeypatch.setattr(training, 'fit_steps', fit_spy)
    monkeypatch.setattr(training, 'game_balanced_cross_entropy', ce_spy)
    lineage = (m.FAMILIES[1], 'implicit', 0)
    expected = m._train_lineage(study, *lineage)
    assert calls == [study.full.schedule(0)] and len(losses) == 7
    assert all(n == study.full.config.game_batch_size for n in losses)
    _, path = m._lineage(study, *lineage)
    expected_digest = expected.manifest_digest
    expected_log = (path/'training_log.jsonl').read_bytes()
    # Synthetic interruption: retain exactly the durable step-2 recovery prefix.
    shutil.rmtree(path/'terminal_checkpoint')
    for name in ('training_log.jsonl','run_provenance.json'): (path/name).unlink()
    for point in ('4','6','7'): shutil.rmtree(path/'recovery'/point)
    actual = m._train_lineage(study, *lineage)
    assert actual.manifest_digest == expected_digest
    assert (path/'training_log.jsonl').read_bytes() == expected_log
    assert len(losses) == 7+5  # no restarted/repeated first two optimizer steps


def test_cli_dispatch_is_prepare_and_train_only(tmp_path, completed, monkeypatch, capsys):
    from werewolf.cli import main, build_parser
    from werewolf import development_publication
    root = tmp_path/'storage'; root.mkdir()
    profile = tmp_path/'profile.json'
    profile.write_text(json.dumps({'artifact_root':str(root)}))
    protocol = tmp_path/'protocol.json'
    protocol.write_bytes(canonical_json_bytes(asdict(completed.full.config)))
    calls = []
    publication = object()
    monkeypatch.setattr(development_publication, 'open_publication', lambda p: publication)
    def prepare(contract, actual_publication, config, destination):
        assert actual_publication is publication and config == completed.full.config
        calls.append((contract, destination))
        return completed
    monkeypatch.setattr(m, 'prepare_study', prepare)
    assert main(['--storage-profile',str(profile),'prepare-paper-tom-study',
        '--study-contract','contract','--publication','publication',
        '--protocol',str(protocol),'--destination','paper']) == 0
    assert calls == [(root/'contract',root/'paper')]
    monkeypatch.setattr(m, 'run_study', lambda path: {'record_digest':'a'*64})
    assert main(['--storage-profile',str(profile),'run-paper-tom-study','--study','paper/execution']) == 0
    assert capsys.readouterr().out.splitlines() == [completed.artifact.manifest_digest,'a'*64]
    for command in ('prepare-paper-tom-study','run-paper-tom-study'):
        options = {option for action in build_parser()._subparsers._group_actions[0].choices[command]._actions
                   for option in action.option_strings}
        assert not options & {'--model','--fold','--condition','--evaluate','--fallback'}


def test_duplicate_checkpoint_directory_rejected(completed):
    _, lineage = m._lineage(completed, *m.EXPECTED_LINEAGES[-1])
    extra = lineage/'duplicate_terminal'; extra.mkdir()
    try:
        with pytest.raises(ValueError, match='unexpected study lineage'):
            m.seal_study(completed.artifact.path)
    finally:
        extra.rmdir()


def _synthetic_process_entry(path, digest, connection, crash):
    import os
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(m, 'open_contract', synthetic_contract)
        if crash:
            original = training._publish_recovery
            def durable_then_crash(target, *args):
                result = original(target, *args)
                if target.name == '2':
                    os._exit(73)
                return result
            patch.setattr(training, '_publish_recovery', durable_then_crash)
        m._process_entry(path, digest, m.FAMILIES[1], 'implicit', 0, connection)


def test_baseline_fresh_process_crash_and_failure_policy(tmp_path):
    import multiprocessing
    study = prepared(tmp_path, cadence=2)
    lineage = (m.FAMILIES[1], 'implicit', 0)
    expected = m._train_lineage(study, *lineage).manifest_digest
    shutil.rmtree(study.agnostic.runs_path)
    context = multiprocessing.get_context('spawn')
    def worker(crash):
        reader, writer = context.Pipe(duplex=False)
        process = context.Process(target=_synthetic_process_entry,
            args=(str(study.artifact.path),study.artifact.manifest_digest,writer,crash))
        process.start(); writer.close()
        process.join(45)
        if process.is_alive():
            process.kill(); process.join()
            pytest.fail('synthetic worker did not finish')
        try:
            if crash:
                assert process.exitcode == 73
                with pytest.raises(EOFError): reader.recv()
                return None
            assert process.exitcode == 0
            return reader.recv()
        finally:
            reader.close()
    worker(True)
    _, path = m._lineage(study, *lineage)
    assert not (path/'failure.json').exists()
    assert sorted(p.name for p in (path/'recovery').iterdir()) == ['2']
    assert worker(False) == (True, expected)
    # An invalid durable recovery is a permanent error, not a request to restart.
    shutil.rmtree(path/'terminal_checkpoint')
    (path/'recovery/2/rng_state.bin').write_bytes(b'corrupt')
    success, error = worker(False)
    assert not success and 'mismatch' in error
    assert (path/'failure.json').is_file()
    with pytest.raises(ValueError, match='permanently failed'):
        m._train_lineage(study, *lineage)


def _source_bound_process_entry(*args):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(m, 'open_contract', synthetic_contract)
        m._process_entry(*args)


def test_run_dispatches_only_unfinished_lineage_then_seals(completed, monkeypatch):
    lineage = m.EXPECTED_LINEAGES[-1]
    expected = m._verify_terminal(completed, *lineage).manifest_digest
    _, path = m._lineage(completed, *lineage)
    terminal = path/'terminal_checkpoint'; backup = path/'.terminal_backup'
    terminal.rename(backup)
    # The child still runs the real production worker; only synthetic source
    # attestation is installed in that fresh process.
    monkeypatch.setattr(m, '_process_entry', _source_bound_process_entry)
    try:
        seal = m.run_study(completed.artifact.path)
        assert seal['checkpoints'][-1]['checkpoint_digest'] == expected
        assert len(seal['checkpoints']) == 20
        assert m.evaluation_eligibility(completed.artifact.path) == seal
    finally:
        completed.seal_path.unlink(missing_ok=True)
        if terminal.exists(): shutil.rmtree(terminal)
        backup.rename(terminal)
