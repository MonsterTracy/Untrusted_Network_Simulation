"""Read-only final-ToM evidence collection; never a model/recovery execution gate.

Only the original verifier can attest byte-exact optimizer/RNG restoration. This
module deliberately does not import qwen3_final, construct a model, or restore
state. Even chain-files mode is NOT that semantic verification.
"""

from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess

from werewolf.artifact_io import (
    canonical_json_bytes, open_artifact_envelope, read_artifact_file, sha256_bytes,
)
from werewolf.artifact_io.tensor_state import decode_tensor_container
from werewolf.tom.final_capacity import derived_capacity


VERSION = 'observer_dtrs_final_tom_readonly_audit_v1'
FIT_VERSION = 'classic7_qwen3_all_development_final_v1'
STEPS = tuple(range(1000, 31001, 1000)) + (31500,)
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class _Pins:
    revision: str = 'f199a9e6c64168a412de91ff8ca04d0c795c57e2'
    fit: str = '87c2485acf48443034c59430d79f098a13151677852988bf691599ca2d2f8248'
    terminal: str = 'b977ec24d6292ad329e664c931a1aae39dc4738b4b33685b19d92fd4fb1921af'
    seal: str = 'fcbb81ee43050142ebdf01c0aee576156221d725598ff5ee2c02055049f72718'
    model_descriptor: str = 'fdf858b5e21a1951da9ed2c2b9e988f8a1f1edcf99d2bb84f54faf055dc7e3e9'


def digest(value):
    return sha256_bytes(canonical_json_bytes(value))


class AuditMismatch(ValueError):
    """Safe, constant error codes; never copy an artifact value to the report."""


def require(condition, code):
    if not condition:
        raise AuditMismatch(code)


def _git(root, *args):
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    env['GIT_OPTIONAL_LOCKS'] = '0'
    return subprocess.check_output(['git', '-C', str(root), *args], env=env,
                                   stderr=subprocess.PIPE, timeout=30)


def checkout_identity(root):
    """Observe the checkout, without refreshing its index or emitting filenames."""
    root = Path(root).absolute()
    require(root.resolve() == root, 'checkout_symlink')
    require(Path(_git(root, 'rev-parse', '--show-toplevel').decode().strip()) == root,
            'checkout_root')
    head = _git(root, 'rev-parse', 'HEAD').decode().strip()
    status = _git(root, 'status', '--porcelain=v1', '--untracked-files=all',
                  '--ignore-submodules=none')
    paths = sorted((root / 'werewolf').rglob('*.py')) + [root / 'run_random.py']
    require(all(p.is_file() and not p.is_symlink() for p in paths), 'source_files')
    hashes = {p.relative_to(root).as_posix(): sha256_bytes(p.read_bytes()) for p in paths}
    entries = _git(root, 'ls-tree', '-rz', 'HEAD', '--', 'werewolf', 'run_random.py').split(b'\0')
    blobs = {}
    for entry in entries:
        if entry:
            info, name = entry.split(b'\t', 1)
            name = name.decode()
            if name.endswith('.py'):
                blobs[name] = info.split()[2].decode()
    # The pinned historical repository uses SHA-1 Git objects. This compares
    # bytes (including assumed-unchanged files), without one Git process/file.
    observed_blobs = {}
    for name in hashes:
        data = (root / name).read_bytes()
        observed_blobs[name] = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    blobs_match = blobs == observed_blobs
    return {'source_revision': head, 'clean': not bool(status),
            'implementation_digest': digest(hashes), 'head_blobs_match': blobs_match}


def _envelope(path, kind, version, manifest='manifest.json'):
    path = Path(path).absolute()
    require(path.resolve() == path, 'artifact_symlink_ancestor')
    return open_artifact_envelope(path, expected_artifact_type=kind,
                                  expected_schema_version=version, manifest_name=manifest)


def _payloads(envelope):
    # Verify one payload at a time; no Primary/report/optimizer content is returned.
    for name in envelope.manifest['file_table']:
        read_artifact_file(envelope, name)


def _fit(path, pins):
    fit = _envelope(path, 'qwen3_all_development_final', FIT_VERSION, 'qwen3_final_manifest.json')
    m = fit.manifest
    require(fit.manifest_digest == pins.fit, 'fit_digest')
    expected_fields = {'artifact_type', 'schema_version', 'protocol_inputs', 'protocol_digest',
                       'study_path', 'publication_path', 'publication_id', 'game_ids',
                       'primary_sidecar_digest', 'initialization', 'temporal_artifact_digest',
                       'schedule_digest', 'optimizer_steps', 'file_table', 'manifest_digest'}
    require(set(m) == expected_fields, 'fit_fields')
    p = m['protocol_inputs']
    design = json.loads((ROOT / 'configs/paper/tom-backbone-study-v1.json').read_bytes())
    require(p['config'] == design['reference_protocol'] and p['architecture'] == 'qwen3'
            and p['schema_version'] == FIT_VERSION and p['temporal_condition'] == 'explicit_day_phase'
            and p['partition'] == 'all_development' and p['population'] == 'non_wolf_alive'
            and p['rng_version'] == 'classic7_qwen3_final_training_rng_v1'
            and p['capacity'] == derived_capacity(1024) and m['optimizer_steps'] == 31500
            and p['destination_path'] == str(fit.path) and digest(p) == m['protocol_digest'],
            'fit_protocol')
    require(len(m['game_ids']) == 1500 and sorted(set(m['game_ids'])) == m['game_ids'], 'fit_games')
    expected_files = {'schedule.json', 'primary.json'} | {f'temporal/{n}' for n in (
        'phase_codebook.manifest.json', 'phase_codebook.bin',
        'canonical_day_code_table.manifest.json', 'canonical_day_code_table.bin')}
    require(set(m['file_table']) == expected_files, 'fit_files')
    _payloads(fit)  # Primary is hashed, not parsed, returned, or independently reconstructed.
    require(m['file_table']['schedule.json']['sha256'] == m['schedule_digest'], 'schedule_digest')
    return fit


def _research_links(fit):
    m, p = fit.manifest, fit.manifest['protocol_inputs']
    study = _envelope(m['study_path'], 'backbone_tom_study', 'classic7_backbone_study_v1')
    pub = _envelope(m['publication_path'], 'development_publication', 'classic7_development_publication_v1')
    s, v = study.manifest, pub.manifest
    require(study.manifest_digest == p['study_digest'] and pub.manifest_digest == p['publication_digest'],
            'research_parent_digest')
    rows = [r for r in s['initializations'] if r['architecture'] == 'qwen3' and r['fold'] == 0
            and r['temporal_condition'] == 'explicit_day_phase']
    require(rows == [m['initialization']] and p['initial_state_digest'] == rows[0]['state_digest'],
            'initialization_binding')
    require(s['publication_path'] == m['publication_path']
            and s['protocol_inputs']['publication_digest'] == p['publication_digest']
            and v['publication_id'] == m['publication_id']
            and v['role_sidecar_digest'] == m['primary_sidecar_digest'], 'research_identity')
    return {'status': 'PASS_ENVELOPES_ONLY', 'study_digest': study.manifest_digest,
            'publication_digest': pub.manifest_digest, 'initial_state_digest': p['initial_state_digest'],
            'primary_sidecar_digest': m['primary_sidecar_digest'],
            'restricted_population_reconstruction': 'NOT_RUN',
            'study_publication_payload_integrity': 'NOT_RUN'}


def _binding(fit):
    m, p = fit.manifest, fit.manifest['protocol_inputs']
    return {'fit_digest': fit.manifest_digest, 'protocol_digest': m['protocol_digest'],
            'architecture': 'qwen3', 'temporal_condition': 'explicit_day_phase',
            'partition': 'all_development', 'initial_state_digest': p['initial_state_digest'],
            'schedule_digest': m['schedule_digest'], 'source': p['source'], 'runtime': p['runtime']}


def _recovery(fit, runs, step, previous):
    a = _envelope(runs / 'recovery' / str(step), 'qwen3_final_recovery', 'classic7_qwen3_final_recovery_v1')
    m = a.manifest
    expected = {'artifact_type', 'schema_version', 'binding', 'optimizer_step', 'schedule_position',
                'previous_checkpoint_digest', 'containers', 'optimizer_scalars', 'rng_scalars',
                'loss_digest', 'file_table', 'manifest_digest'}
    require(set(m) == expected and m['binding'] == _binding(fit)
            and type(m['optimizer_step']) is int and m['optimizer_step'] == step
            and type(m['schedule_position']) is int and m['schedule_position'] == step
            and m['previous_checkpoint_digest'] == previous
            and set(m['containers']) == {'model', 'optimizer', 'rng'}
            and set(m['file_table']) == {'model.bin', 'optimizer.bin', 'rng.bin', 'loss.jsonl'},
            'recovery_binding')
    for name, c in m['containers'].items():
        f = m['file_table'][name + '.bin']
        require(c['payload_path'] == name + '.bin' and c['payload_sha256'] == f['sha256']
                and c['payload_byte_length'] == f['byte_size'], 'container_file_binding')
    require(m['loss_digest'] == m['file_table']['loss.jsonl']['sha256'], 'loss_file_binding')
    return a


def _chain(fit, pins, *, chain_files):
    runs = fit.path.parent.parent / 'runs' / fit.manifest_digest
    require({p.name for p in runs.iterdir()} == {'recovery', 'terminal', 'seal'}, 'run_inventory')
    require({p.name for p in (runs / 'recovery').iterdir()} == {str(s) for s in STEPS}, 'recovery_inventory')
    terminal = _envelope(runs / 'terminal', 'qwen3_final_terminal', 'classic7_qwen3_final_terminal_v1')
    seal = _envelope(runs / 'seal', 'qwen3_final_seal', 'classic7_qwen3_final_seal_v1')
    require(terminal.manifest_digest == pins.terminal and seal.manifest_digest == pins.seal, 'terminal_seal_digest')
    previous, ancestry, all_artifacts = None, [], [terminal, seal]
    last = None
    for step in STEPS:
        a = _recovery(fit, runs, step, previous)
        all_artifacts.append(a)
        if chain_files:
            _payloads(a)
            for name, container in a.manifest['containers'].items():
                decode_tensor_container(container, read_artifact_file(a, name + '.bin'), payload_path=name + '.bin')
        previous = a.manifest_digest
        ancestry.append({'step': step, 'digest': previous})
        last = a
    model = last.manifest['containers']['model']
    descriptor_digest = digest(model)
    require(descriptor_digest == pins.model_descriptor, 'model_descriptor_digest')
    expected_terminal = {'artifact_type': 'qwen3_final_terminal',
        'schema_version': 'classic7_qwen3_final_terminal_v1', 'binding': _binding(fit),
        'terminal_step': 31500, 'last_recovery_digest': previous, 'model_digest': descriptor_digest,
        'checkpoint_ancestry': ancestry, 'file_table': {}, 'manifest_digest': pins.terminal}
    expected_seal = {'artifact_type': 'qwen3_final_seal', 'schema_version': 'classic7_qwen3_final_seal_v1',
        'binding': _binding(fit), 'terminal_digest': pins.terminal, 'terminal_step': 31500,
        'file_table': {}, 'manifest_digest': pins.seal}
    require(terminal.manifest == expected_terminal and seal.manifest == expected_seal, 'terminal_seal_binding')
    raw = read_artifact_file(last, 'model.bin')
    tensors = decode_tensor_container(model, raw, payload_path='model.bin')
    evidence = {'status': 'PASS_MANIFEST_ANCESTRY_AND_LAST_MODEL_BYTES',
        'checkpoint_count': len(ancestry), 'ancestry_digest': digest(ancestry),
        'terminal_digest': pins.terminal, 'seal_digest': pins.seal, 'last_recovery_digest': previous,
        'model_container_descriptor_sha256': descriptor_digest,
        'model_bin_raw_sha256': sha256_bytes(raw), 'model_bin_bytes': len(raw), 'model_tensor_count': len(tensors),
        'all_recovery_payloads': 'PASS_BYTES_AND_CONTAINER_CODECS' if chain_files else 'NOT_RUN',
        'loss_schedule_semantics': 'NOT_RUN', 'optimizer_rng_restore_and_recapture': 'NOT_RUN',
        'full_original_recovery_revalidation': 'NOT_RUN_REQUIRES_MODEL_AND_STATE_RESTORE'}
    return evidence, all_artifacts


def _temporal(fit):
    # Reuse the historical validator. Importing this table provider constructs
    # no ToM, CUDA context, optimizer or predictor; trigonometry is not recomputed.
    from werewolf.tom.temporal import TemporalCodeProvider
    a = _envelope(fit.path / 'temporal', 'deterministic_temporal_code', 'classic7_temporal_code_v1',
                  'phase_codebook.manifest.json')
    require(a.manifest_digest == fit.manifest['temporal_artifact_digest'], 'temporal_digest')
    provider = TemporalCodeProvider.explicit(a.path, fit.manifest['protocol_digest'])
    require(list(provider.day.shape) == [257, 128], 'temporal_capacity')
    m = a.manifest
    return {'status': 'PASS_EXISTING_TEMPORAL_READER', 'manifest_digest': a.manifest_digest,
            'phase_manifest_raw_sha256': sha256_bytes((a.path / a.manifest_name).read_bytes()),
            'day_manifest_raw_sha256': m['file_table']['canonical_day_code_table.manifest.json']['sha256'],
            'phase_bin_raw_sha256': m['phase_digest'], 'day_bin_raw_sha256': m['day_digest'],
            'phase_shape': m['phase_shape'], 'day_shape': m['day_shape'],
            'parent_protocol_digest': m['parent_digest'], 'capacity': fit.manifest['protocol_inputs']['capacity']}, a


def _runtime(saved):
    observed = {'python': platform.python_version(), 'platform': platform.platform()}
    for name in ('numpy', 'torch', 'transformers'):
        try:
            observed[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            observed[name] = None
    matches = {k: observed[k] == saved.get(k) for k in observed}
    return {'saved_runtime_digest': digest(saved), 'observed_package_versions': observed,
            'matches_saved_observed_fields': matches,
            'observed_fields_status': 'MATCH_ONLY_LISTED_FIELDS' if all(matches.values()) else 'MISMATCH_OR_PACKAGE_MISSING',
            'saved_config_source_revision': saved.get('source_revision'),
            'gpu_determinism_and_full_runtime_equality': 'NOT_RUN',
            'runtime_qualification': 'NOT_ESTABLISHED'}


def _reader_sources(checkout):
    # The new caller/tool and historical predictor are separate identities.
    # Reused readers must nevertheless have their audited historical semantics.
    paths = ('werewolf/artifact_io/__init__.py', 'werewolf/artifact_io/canonical.py',
             'werewolf/artifact_io/tensor_state.py', 'werewolf/tom/temporal.py',
             'werewolf/tom/final_capacity.py', 'werewolf/canonical_collection/pre.py',
             'werewolf/canonical_collection/public_history.py', 'werewolf/structured_history.py',
             'configs/paper/tom-backbone-study-v1.json')
    hashes = {}
    for name in paths:
        current = (ROOT / name).read_bytes()
        require(current == (Path(checkout) / name).read_bytes(), 'audit_reader_source_mismatch')
        hashes[name] = sha256_bytes(current)
    return {'status': 'PASS_BYTES_MATCH_PINNED_CHECKOUT', 'sha256': hashes}


def audit_final_tom(fit_path, predictor_checkout, *, chain_files=False, _pins=None):
    """Return redacted evidence, including failures. _pins is for synthetic tests only.

    CLI never accepts replacement pins. A fixture report is visibly non-production.
    This is not a formal execution guard or permission to deploy/infer.
    """
    pins = _pins or _Pins()
    report = {'schema_version': VERSION, 'scope': 'CONTROLLED_FIXTURE' if _pins else 'PINNED_FINAL_PROVENANCE_ONLY',
              'mode': 'chain-files' if chain_files else 'metadata', 'status': 'FAILED',
              'checks': {k: {'status': 'NOT_RUN'} for k in ('fit', 'research_links', 'temporal', 'source',
                  'reader_sources', 'recovery', 'concurrent_manifest_change')},
              'not_authorized': ['model_forward', 'model_restore', 'optimizer_restore',
                  'cal_test_execution', 'deployment_qualification']}
    stage = 'fit'
    try:
        fit = _fit(fit_path, pins)
        p = fit.manifest['protocol_inputs']
        report['checks']['fit'] = {'status': 'PASS_MANIFEST_AND_FIT_FILE_HASHES', 'manifest_digest': fit.manifest_digest,
            'protocol_digest': fit.manifest['protocol_digest'], 'game_count': len(fit.manifest['game_ids']),
            'optimizer_steps': 31500, 'fit_manifest_raw_sha256': sha256_bytes((fit.path / fit.manifest_name).read_bytes()),
            'graph_digest': p['graph']['graph_digest'], 'schedule_digest': fit.manifest['schedule_digest'],
            'schedule_reconstruction': 'NOT_RUN', 'primary_reconstruction': 'NOT_RUN'}
        stage = 'research_links'
        report['checks'][stage] = _research_links(fit)
        stage = 'temporal'
        report['checks'][stage], temporal = _temporal(fit)
        stage = 'source'
        observed = checkout_identity(predictor_checkout)
        require(observed['source_revision'] == pins.revision and observed['clean']
                and observed['head_blobs_match'], 'predictor_checkout')
        require(p['source'] == {k: observed[k] for k in ('source_revision', 'implementation_digest')}
                and p['runtime']['implementation_digest'] == observed['implementation_digest'], 'fit_source_runtime')
        require(p['runtime']['source_revision'] == p['config']['source_revision'], 'config_runtime_source_revision')
        report['checks'][stage] = {'status': 'PASS_FROZEN_CHECKOUT_AND_SAVED_SOURCE', **observed,
            'runtime': _runtime(p['runtime'])}
        stage = 'reader_sources'
        report['checks'][stage] = (_reader_sources(predictor_checkout) if _pins is None
                                   else {'status': 'NOT_RUN_SYNTHETIC_SOURCE_ONLY'})
        stage = 'recovery'
        report['checks'][stage], artifacts = _chain(fit, pins, chain_files=chain_files)
        stage = 'concurrent_manifest_change'
        # Re-open envelopes after the read. This detects changed seals/manifests;
        # it is not a filesystem snapshot or protection against adversarial writes.
        for a in [fit, temporal, *artifacts]:
            again = _envelope(a.path, a.manifest['artifact_type'], a.manifest['schema_version'], a.manifest_name)
            require(again.manifest_digest == a.manifest_digest, 'manifest_changed_during_audit')
        require(checkout_identity(predictor_checkout) == observed, 'source_changed_during_audit')
        report['checks'][stage] = {'status': 'PASS_REOPENED_MANIFESTS_AND_SOURCE',
                                  'filesystem_snapshot_or_write_exclusion': 'NOT_ESTABLISHED'}
        report['status'] = 'PASS_WITH_EXPLICIT_SCOPE_LIMITS'
    except (OSError, ValueError, KeyError, TypeError, ImportError, subprocess.SubprocessError) as error:
        report['checks'][stage] = {'status': 'FAILED'}
        report['failure'] = {'stage': stage, 'code': str(error) if isinstance(error, AuditMismatch)
                             else 'invalid_or_unavailable_evidence', 'exception_type': type(error).__name__}
    report['full_semantic_recovery'] = 'NOT_RUN'
    report['inference_qualification'] = 'NOT_RUN'
    report['existing_running_predictor_process_identity'] = 'NOT_OBSERVED'
    report['evidence_digest'] = digest(report)
    return report


def _output_protection_root(value):
    """Resolve known ancestors strictly; permit only genuinely missing suffixes.

    Missing inputs must still support an external failure report. Dangling
    symlinks, loops, inaccessible ancestors and unresolved '..' are not safe
    substitutes for a known protected root.
    """
    path, missing = Path(value).absolute(), []
    while True:
        try:
            return path.resolve(strict=True).joinpath(*reversed(missing))
        except FileNotFoundError:
            require(not path.is_symlink() and path.parent != path and path.name != '..',
                    'output_protection_root_unresolved')
            missing.append(path.name)
            path = path.parent
        except RuntimeError:
            raise AuditMismatch('output_protection_root_unresolved') from None


def write_evidence(output, report, *, fit_path, predictor_checkout):
    """Exclusive new output, outside artifacts and both code checkouts. No mkdir."""
    path = Path(output).absolute()
    try:
        require(path.parent.resolve(strict=True) == path.parent and path.parent.is_dir(), 'output_parent')
    except RuntimeError:
        raise AuditMismatch('output_parent') from None
    fit = _output_protection_root(fit_path)
    # This guard runs even after input validation failed. Protect the resolved
    # fit's lifecycle, not the alias's lexical parents or the shared data root.
    protected = (fit, _output_protection_root(fit.parent.parent / 'runs'),
                 _output_protection_root(predictor_checkout), _output_protection_root(ROOT))
    require(not any(path == p or p in path.parents for p in protected), 'output_inside_audited_tree')
    identities = set()
    for root in protected:
        try:
            info = root.stat()
        except FileNotFoundError:
            continue
        identities.add((info.st_dev, info.st_ino))
    parent_info = path.parent.stat()
    for parent in path.parents:
        info = parent.stat()
        require((info.st_dev, info.st_ino) not in identities, 'output_inside_audited_tree')
        for name in ('manifest.json', 'qwen3_final_manifest.json', 'phase_codebook.manifest.json', '.git'):
            try:
                (parent / name).lstat()
            except FileNotFoundError:
                continue
            raise AuditMismatch('output_inside_artifact_or_source')
    data = canonical_json_bytes(report) + b'\n'
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(parent_fd)
        require((opened.st_dev, opened.st_ino) == (parent_info.st_dev, parent_info.st_ino), 'output_parent_changed')
        fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent_fd)
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return hashlib.sha256(data).hexdigest()
