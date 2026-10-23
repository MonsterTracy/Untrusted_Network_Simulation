"""Offline inference-only export from externally audited final-ToM artifacts.

Reads envelopes and the last model + temporal payloads only. Does not call
open_fit/verify_seal, hash Primary, reconstruct reports, or open optimizer/RNG.
Historical ancestry logic stays in the already reviewed read-only audit tool.
"""
import json
from pathlib import Path

import torch

from werewolf.artifact_io import canonical_json_bytes, publish_artifact, read_artifact_file, sha256_bytes
from werewolf.observer_dtrs import final_tom_audit as audit
from werewolf.observer_dtrs.final_tom_capsule import (
    TYPE, VERSION, ROOT, TEMPORAL_FILES, digest, sha, require, public_contract,
    loader_source, source_observation, runtime_identity, validate_origin,
)
from werewolf.observer_dtrs.final_tom_graph import InferenceGraph, graph_identity
from werewolf.artifact_io.tensor_state import decode_tensor_container
from werewolf.tom.state import restore_model_tensors
from werewolf.tom.temporal import TemporalCodeProvider

REUSED_AUDIT_READERS = (
    'werewolf/artifact_io/__init__.py', 'werewolf/artifact_io/canonical.py',
    'werewolf/artifact_io/tensor_state.py', 'werewolf/tom/temporal.py',
    'werewolf/tom/final_capacity.py', 'werewolf/canonical_collection/pre.py',
    'werewolf/canonical_collection/public_history.py', 'werewolf/structured_history.py',
    'configs/paper/tom-backbone-study-v1.json',
)


def _audit_packet(path, expected_raw_sha256, pins, *, controlled):
    sha(expected_raw_sha256)
    raw = Path(path).read_bytes()
    require(sha256_bytes(raw) == expected_raw_sha256, 'external audit file SHA mismatch')
    report = json.loads(raw)
    require(raw in (canonical_json_bytes(report), canonical_json_bytes(report)+b'\n'),
            'canonical audit evidence required')
    require(type(report) is dict and report.get('evidence_digest') == digest(
        {k: v for k, v in report.items() if k != 'evidence_digest'}), 'audit evidence digest mismatch')
    require(report.get('schema_version') == audit.VERSION and report.get('mode') == 'chain-files'
            and report.get('status') == 'PASS_WITH_EXPLICIT_SCOPE_LIMITS'
            and report.get('scope') == ('CONTROLLED_FIXTURE' if controlled else 'PINNED_FINAL_PROVENANCE_ONLY'),
            'complete audited chain-files evidence required')
    c = report['checks']
    require(c['fit']['manifest_digest'] == pins.fit and c['source']['source_revision'] == pins.revision
            and c['source']['clean'] is True and c['source']['head_blobs_match'] is True
            and c['recovery']['terminal_digest'] == pins.terminal
            and c['recovery']['seal_digest'] == pins.seal
            and c['recovery']['model_container_descriptor_sha256'] == pins.model_descriptor
            and c['recovery']['checkpoint_count'] == 32
            and c['recovery']['all_recovery_payloads'] == 'PASS_BYTES_AND_CONTAINER_CODECS'
            and c['concurrent_manifest_change']['status'] == 'PASS_REOPENED_MANIFESTS_AND_SOURCE',
            'audit provenance identity mismatch')
    statuses = {'fit': 'PASS_MANIFEST_AND_FIT_FILE_HASHES', 'research_links': 'PASS_ENVELOPES_ONLY',
                'temporal': 'PASS_EXISTING_TEMPORAL_READER', 'source': 'PASS_FROZEN_CHECKOUT_AND_SAVED_SOURCE',
                'recovery': 'PASS_MANIFEST_ANCESTRY_AND_LAST_MODEL_BYTES'}
    require(all(c[k]['status'] == v for k, v in statuses.items())
            and (controlled or c['reader_sources']['status'] == 'PASS_BYTES_MATCH_PINNED_CHECKOUT')
            and report['full_semantic_recovery'] == report['inference_qualification'] == 'NOT_RUN',
            'audit scope/status mismatch')
    return report


def _destination(destination, fit, runs):
    p = Path(destination).absolute()
    require('..' not in p.parts and p.parent.resolve(strict=True) == p.parent,
            'new canonical destination parent required')
    require(not p.exists() and not p.is_symlink(), 'capsule destination must not exist')
    protected = [fit.resolve(strict=True), runs.resolve(strict=True), ROOT.resolve(strict=True)]
    identities = {(q.stat().st_dev, q.stat().st_ino) for q in protected}
    for parent in p.parents:
        info = parent.stat()
        require(parent not in protected and (info.st_dev, info.st_ino) not in identities,
                'destination inside protected tree')
        for marker in ('.git', 'manifest.json', 'qwen3_final_manifest.json', 'phase_codebook.manifest.json'):
            require(not (parent / marker).exists() and not (parent / marker).is_symlink(),
                    'destination inside artifact or checkout')
    return p


def compile_capsule(fit_path, destination, *, audit_path, expected_audit_raw_sha256, _pins=None):
    """Controlled pins are Python-test-only; the CLI has no origin pin override.

    The chain-files report is externally anchored evidence, not a fresh full
    recovery. It is checked against re-opened original manifests/last weights.
    No inference is performed by compilation. No restricted payload is opened.
    """
    pins = _pins or audit._Pins()
    report = _audit_packet(audit_path, expected_audit_raw_sha256, pins, controlled=_pins is not None)
    c = report['checks']
    if _pins is None:
        readers = c['reader_sources']['sha256']
        require(set(readers) == set(REUSED_AUDIT_READERS)
                and all(sha256_bytes((ROOT/n).read_bytes()) == readers[n] for n in REUSED_AUDIT_READERS),
                'reused historical readers differ from audited bytes')
    fit = audit._envelope(fit_path, 'qwen3_all_development_final', audit.FIT_VERSION,
                          'qwen3_final_manifest.json')
    m, p = fit.manifest, fit.manifest['protocol_inputs']
    require(fit.manifest_digest == pins.fit and digest(p) == m['protocol_digest']
            and m['protocol_digest'] == c['fit']['protocol_digest']
            and sha256_bytes((fit.path / fit.manifest_name).read_bytes()) == c['fit']['fit_manifest_raw_sha256']
            and p['graph'] == graph_identity() and p['capacity'] == public_contract()['capacity']
            and p['source'] == {k: c['source'][k] for k in ('source_revision', 'implementation_digest')}
            and digest(p['runtime']) == c['source']['runtime']['saved_runtime_digest'], 'fit source/graph identity mismatch')
    require(p['study_digest'] == c['research_links']['study_digest']
            and p['publication_digest'] == c['research_links']['publication_digest']
            and p['initial_state_digest'] == c['research_links']['initial_state_digest'], 'research origin mismatch')
    runs = fit.path.parent.parent / 'runs' / fit.manifest_digest
    output = _destination(destination, fit.path, runs)
    recovery, artifacts = audit._chain(fit, pins, chain_files=False)
    for key in ('last_recovery_digest', 'ancestry_digest', 'model_bin_raw_sha256',
                'model_container_descriptor_sha256', 'model_bin_bytes', 'model_tensor_count'):
        require(recovery[key] == c['recovery'][key], 'last weights/ancestry differ from audit')
    temporal_evidence, temporal_artifact = audit._temporal(fit)
    require(temporal_evidence == c['temporal'], 'temporal differs from audited original bytes')
    last = artifacts[-1]
    container = last.manifest['containers']['model']
    payload = read_artifact_file(last, 'model.bin')
    require(sha256_bytes(payload) == c['recovery']['model_bin_raw_sha256'], 'model changed during compilation')
    temporal = TemporalCodeProvider.explicit(temporal_artifact.path, m['protocol_digest'])
    model = InferenceGraph(temporal)
    restore_model_tensors(model, decode_tensor_container(container, payload, payload_path='model.bin'))
    del model  # validation only; no forward, optimizer or training
    origin = {**p['source'], 'saved_runtime_digest': digest(p['runtime']),
        'fit_digest': fit.manifest_digest, 'protocol_digest': m['protocol_digest'],
        **{n: p[n] for n in ('study_digest', 'publication_digest', 'initial_state_digest')},
        **{n: recovery[n] for n in ('terminal_digest', 'seal_digest', 'last_recovery_digest',
            'ancestry_digest', 'model_container_descriptor_sha256', 'model_bin_raw_sha256')},
        'temporal_manifest_digest': temporal_artifact.manifest_digest,
        'audit_evidence_digest': report['evidence_digest'], 'audit_file_raw_sha256': expected_audit_raw_sha256,
        'full_semantic_recovery': 'NOT_RUN', 'inference_qualification': 'NOT_RUN'}
    scope = 'CONTROLLED_FIXTURE' if _pins is not None else 'PINNED_FINAL_INFERENCE_UNQUALIFIED'
    validate_origin(origin, scope)
    spec = {'graph': graph_identity(), 'public_contract': public_contract(), 'model_container': container,
            'temporal_digest': temporal_artifact.manifest_digest, 'temporal_parent_protocol': m['protocol_digest'],
            'loader_source': loader_source()}
    files = {'model.bin': payload, 'inference.json': canonical_json_bytes(spec)}
    files.update({'temporal/' + n: (temporal_artifact.path / n).read_bytes() for n in TEMPORAL_FILES})
    require(sha256_bytes(files['temporal/phase_codebook.manifest.json'])
            == temporal_evidence['phase_manifest_raw_sha256'], 'temporal manifest changed during copy')
    for n in TEMPORAL_FILES:
        if n != 'phase_codebook.manifest.json':
            require(sha256_bytes(files['temporal/'+n]) == temporal_artifact.manifest['file_table'][n]['sha256'],
                    'temporal payload changed during copy')
    # Re-open source metadata after copying. This is not an adversarial filesystem snapshot.
    for a in (fit, temporal_artifact, *artifacts):
        again = audit._envelope(a.path, a.manifest['artifact_type'], a.manifest['schema_version'], a.manifest_name)
        require(again.manifest_digest == a.manifest_digest, 'source manifest changed during compilation')
    require(_destination(output, fit.path, runs) == output, 'output path changed')
    compiler = {'source': source_observation(), 'runtime': runtime_identity(torch.device('cpu')),
                'module_raw_sha256': sha256_bytes(Path(__file__).read_bytes()),
                'audit_reader_raw_sha256': sha256_bytes(Path(audit.__file__).read_bytes()),
                'forward': 'NOT_RUN', 'scope': 'INFERENCE_ONLY_EXPORT_NOT_QUALIFICATION'}
    return publish_artifact(output, manifest_fields={
        'artifact_type': TYPE, 'schema_version': VERSION, 'scope': scope,
        'inference_digest': digest(spec), 'origin': origin, 'compiler': compiler}, files=files)
