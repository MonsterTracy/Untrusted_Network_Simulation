"""Read-only final-ToM inference. No fit, seal, training-data or audit imports.

An externally pinned capsule is an input, not authorization for CAL/TEST.
The compiler and original provenance verifier are deliberately separate modules.
"""
from dataclasses import dataclass
from importlib import metadata
import json
import hashlib
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

import torch

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes, verify_artifact
from werewolf.artifact_io.tensor_state import decode_tensor_container
from werewolf.canonical_collection.public_history import PLAYER_IDS, PUBLIC_PHASES
from werewolf.canonical_collection.speech import V1_ACTIONS
from werewolf.structured_history import STRUCTURED_TOKEN_TYPES
from werewolf.tom.final_capacity import derived_capacity, validate_final_pre
from werewolf.tom.state import restore_model_tensors
from werewolf.tom.temporal import TemporalCodeProvider
from werewolf.observer_dtrs.final_tom_graph import InferenceGraph, graph_identity

TYPE = 'observer_dtrs_final_tom_inference_capsule'
VERSION = 'observer_dtrs_final_tom_inference_capsule_v1'
ROOT = Path(__file__).resolve().parents[2]
TEMPORAL_FILES = ('phase_codebook.manifest.json', 'phase_codebook.bin',
                  'canonical_day_code_table.manifest.json', 'canonical_day_code_table.bin')
FILES = {'model.bin', 'inference.json'} | {'temporal/' + n for n in TEMPORAL_FILES}
SCOPES = ('CONTROLLED_FIXTURE', 'PINNED_FINAL_INFERENCE_UNQUALIFIED')


def digest(value):
    return sha256_bytes(canonical_json_bytes(value))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None,
            'SHA-256 trust anchor required')


def public_contract():
    return {'seats': list(PLAYER_IDS), 'phases': list(PUBLIC_PHASES),
            'actions': list(V1_ACTIONS), 'token_types': list(STRUCTURED_TOKEN_TYPES),
            'capacity': derived_capacity(1024), 'encoding': 'original_public_tensor_ids_v1',
            'padding': 'right_zero_to_1024', 'temporal': 'original_explicit_bytes_no_regeneration',
            'output': 'absolute_seat_7x7_nonself_probability_mass',
            'dead_target_mass': 'retain', 'self_mass': 0,
            'conversion': 'log_Q_exp_once_no_alive_renormalization'}


def runtime_identity(device):
    """Observe the new process, without pretending equality with training runtime."""
    result = {'python': platform.python_version(), 'platform': platform.platform(),
              'packages': {n: metadata.version(n) for n in ('numpy', 'torch', 'transformers')},
              'device': str(device), 'dtype': 'float32', 'attention_backend': 'eager',
              'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
              'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
              'cuda_build': torch.version.cuda}
    if device.type == 'cuda':
        result['gpu_name'] = torch.cuda.get_device_name(device)
        result['gpu_capability'] = list(torch.cuda.get_device_capability(device))
    return result


def loader_source():
    """Explicit runtime closure, including package initializers and reused readers.

    state.py's lazy validation/training helpers are never called. Its imported
    restore_model_tensors is the original strict tensor restore, unchanged.
    """
    names = [
        'werewolf/__init__.py', 'werewolf/observer_dtrs/__init__.py',
        'werewolf/observer_dtrs/final_tom_capsule.py', 'werewolf/observer_dtrs/final_tom_graph.py',
        'werewolf/artifact_io/__init__.py', 'werewolf/artifact_io/canonical.py',
        'werewolf/artifact_io/tensor_state.py', 'werewolf/structured_history.py',
        'werewolf/tom/__init__.py', 'werewolf/tom/final_capacity.py',
        'werewolf/tom/temporal.py', 'werewolf/tom/state.py',
    ] + ['werewolf/canonical_collection/' + n + '.py' for n in (
        '__init__', 'attempt_ledger', 'collector', 'failure_evidence', 'frozen_json',
        'game_bundle', 'pre', 'public_history', 'speech', 'trajectory_evidence')]
    hashes = {}
    for n in sorted(names):
        module_name = n.removesuffix('.py').replace('/', '.').removesuffix('.__init__')
        module = sys.modules.get(module_name)
        expected = ROOT / n
        require(module is not None and Path(module.__file__).resolve(strict=True) == expected
                and not expected.is_symlink(), 'runtime module origin mismatch')
        hashes[n] = sha256_bytes(expected.read_bytes())
    return {'file_sha256': hashes, 'implementation_digest': digest(hashes)}


def source_observation():
    sources = loader_source()
    revision, blobs_match = None, False
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    env['GIT_OPTIONAL_LOCKS'] = '0'
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], env=env,
                                       stderr=subprocess.DEVNULL, timeout=5)
    try:
        require(Path(git('rev-parse', '--show-toplevel').decode().strip()).resolve() == ROOT, 'source root mismatch')
        revision = git('rev-parse', 'HEAD').decode().strip()
        require(re.fullmatch('[0-9a-f]{40}', revision) is not None, 'source revision mismatch')
        entries = git('ls-tree', '-rz', 'HEAD', '--', *sources['file_sha256']).split(b'\0')
        blobs = {name.decode(): info.split()[2].decode()
                 for e in entries if e for info, name in (e.split(b'\t', 1),)}
        actual = {}
        for n in sources['file_sha256']:
            data = (ROOT / n).read_bytes()
            actual[n] = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        blobs_match = blobs == actual
    except (OSError, subprocess.SubprocessError):
        pass
    # No claim that a revision alone identifies these new, possibly uncommitted bytes.
    return {'checkout_revision': revision, 'all_loader_bytes_match_head': blobs_match,
            'formal_execution_guard': 'NOT_ESTABLISHED', 'loader': sources}


def encode_public_pre(prefix):
    """Public-only projection of dataset.tensorize_public_pre, fixed capacity."""
    plan = validate_final_pre(prefix, derived_capacity(1024))
    arrays = {n: torch.zeros(1024, dtype=torch.long) for n in (
        'event_ids', 'source_ids', 'action_ids', 'target_ids', 'day_ids', 'phase_ids')}
    for i, token in enumerate(plan.tokens):
        arrays['event_ids'][i] = STRUCTURED_TOKEN_TYPES.index(token.token_type) + 1
        arrays['source_ids'][i] = 0 if token.source is None else PLAYER_IDS.index(token.source) + 1
        arrays['target_ids'][i] = 0 if token.target is None else PLAYER_IDS.index(token.target) + 1
        arrays['action_ids'][i] = 0 if token.action is None else V1_ACTIONS.index(token.action) + 1
        arrays['day_ids'][i] = token.day
        arrays['phase_ids'][i] = PUBLIC_PHASES.index(token.phase)
    arrays['attention_mask'] = torch.arange(1024) < plan.token_count
    return plan, {n: a[None] for n, a in arrays.items()}


@dataclass(frozen=True)
class CapsulePrediction:
    game_id: str
    boundary_id: str
    prefix_digest: str
    plan_digest: str
    token_count: int
    probabilities: tuple[tuple[float, ...], ...]
    identity_bytes: bytes

    @property
    def identity(self):
        return json.loads(self.identity_bytes)


class CapsulePredictor:
    """Load only pinned inference bytes. No original fit path is retained/opened.

    CPU is default; GPU and other runtime changes require subsequent real QUAL.
    This synchronous core provides no timeout/cancellation/service guarantees.
    """

    def __init__(self, path, *, expected_manifest_digest, expected_scope, device='cpu'):
        sha(expected_manifest_digest)
        require(expected_scope in SCOPES, 'explicit capsule scope required')
        path = Path(path).absolute()
        require(path.resolve(strict=True) == path, 'capsule aliases are not accepted')
        a = verify_artifact(path, expected_artifact_type=TYPE, expected_schema_version=VERSION)
        m = a.manifest
        require(a.manifest_digest == expected_manifest_digest and set(m) == {
            'artifact_type', 'schema_version', 'scope', 'inference_digest', 'origin',
            'compiler', 'file_table', 'manifest_digest'}, 'capsule identity/schema mismatch')
        require(m['scope'] == expected_scope and set(m['file_table']) == FILES, 'capsule scope/inventory mismatch')
        raw = (path / 'inference.json').read_bytes()
        spec = json.loads(raw)
        require(canonical_json_bytes(spec) == raw and digest(spec) == m['inference_digest'], 'inference digest mismatch')
        require(set(spec) == {'graph', 'public_contract', 'model_container', 'temporal_digest',
                              'temporal_parent_protocol', 'loader_source'}, 'inference schema mismatch')
        require(spec['graph'] == graph_identity() and spec['public_contract'] == public_contract(),
                'inference graph/public contract mismatch')
        require(spec['loader_source'] == loader_source(), 'loader source bytes mismatch')
        from_origin = m['origin']
        validate_origin(from_origin, expected_scope)
        require(spec['temporal_parent_protocol'] == from_origin['protocol_digest']
                and spec['temporal_digest'] == from_origin['temporal_manifest_digest']
                and digest(spec['model_container']) == from_origin['model_container_descriptor_sha256'],
                'inference/origin mismatch')
        payload = (path / 'model.bin').read_bytes()
        require(sha256_bytes(payload) == from_origin['model_bin_raw_sha256'], 'raw model SHA mismatch')
        tensors = decode_tensor_container(spec['model_container'], payload, payload_path='model.bin')
        temporal = TemporalCodeProvider.explicit(path / 'temporal', spec['temporal_parent_protocol'])
        require(temporal.artifact_digest == spec['temporal_digest'] and temporal.day.shape == (257, 128),
                'temporal identity/capacity mismatch')
        self._device = torch.device(device)
        require(self._device.type in ('cpu', 'cuda'), 'only CPU/CUDA FP32 runtimes are candidates')
        model = InferenceGraph(temporal)
        restore_model_tensors(model, tensors)
        self._model = model.to(self._device).requires_grad_(False).eval()
        self._identity_bytes = canonical_json_bytes({
            'family': 'final_tom_capsule', 'scope': expected_scope,
            'capsule_digest': a.manifest_digest, 'origin': from_origin,
            'loader_source': source_observation(), 'runtime': runtime_identity(self._device),
            'qualification': 'NOT_ESTABLISHED', 'original_f199_execution': False})

    @property
    def identity(self):
        return json.loads(self._identity_bytes)

    def predict(self, prefix):
        plan, arrays = encode_public_pre(prefix)
        with torch.inference_mode():
            log_q = self._model(**{n: a.to(self._device) for n, a in arrays.items()})[0]
            q = log_q.exp().cpu()  # exactly one log-Q -> Q conversion; retain all seven columns
        require(q.shape == (7, 7) and torch.isfinite(q).all().item()
                and (q >= 0).all().item() and (q <= 1).all().item()
                and torch.equal(q.diag(), torch.zeros(7))
                and torch.allclose(q.sum(-1), torch.ones(7), atol=1e-6, rtol=0), 'invalid Q; no fallback')
        return CapsulePrediction(prefix.game_id, prefix.boundary_id, prefix.prefix_digest, plan.plan_digest,
                                 plan.token_count, tuple(tuple(row) for row in q.tolist()), self._identity_bytes)


ORIGIN_FIELDS = {
    'source_revision', 'implementation_digest', 'saved_runtime_digest', 'fit_digest', 'protocol_digest',
    'study_digest', 'publication_digest', 'initial_state_digest', 'terminal_digest', 'seal_digest',
    'last_recovery_digest', 'ancestry_digest', 'model_container_descriptor_sha256', 'model_bin_raw_sha256',
    'temporal_manifest_digest', 'audit_evidence_digest', 'audit_file_raw_sha256',
    'full_semantic_recovery', 'inference_qualification',
}


def validate_origin(origin, scope):
    require(type(origin) is dict and set(origin) == ORIGIN_FIELDS, 'origin whitelist mismatch')
    for n, v in origin.items():
        if n == 'source_revision':
            require(type(v) is str and re.fullmatch('[0-9a-f]{40}', v) is not None, 'original source revision')
        elif n not in ('full_semantic_recovery', 'inference_qualification'):
            sha(v)
    require(origin['full_semantic_recovery'] == 'NOT_RUN' and origin['inference_qualification'] == 'NOT_RUN',
            'capsule compilation does not establish semantic recovery or QUAL')
    if scope == 'PINNED_FINAL_INFERENCE_UNQUALIFIED':
        # Origin is immutable; this independent loader is NOT the original predictor checkout.
        expected = {
            'source_revision': 'f199a9e6c64168a412de91ff8ca04d0c795c57e2',
            'fit_digest': '87c2485acf48443034c59430d79f098a13151677852988bf691599ca2d2f8248',
            'terminal_digest': 'b977ec24d6292ad329e664c931a1aae39dc4738b4b33685b19d92fd4fb1921af',
            'seal_digest': 'fcbb81ee43050142ebdf01c0aee576156221d725598ff5ee2c02055049f72718',
            'model_container_descriptor_sha256': 'fdf858b5e21a1951da9ed2c2b9e988f8a1f1edcf99d2bb84f54faf055dc7e3e9',
        }
        require(all(origin[n] == v for n, v in expected.items()), 'pinned final origin mismatch')
