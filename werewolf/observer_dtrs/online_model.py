"""Restricted-reader-free deployment values and immutable fixture package reader.

Only controlled/verified-D2 fixtures are enabled in D4b. No production final-ToM
calibration claim or implicit OOF -> final transfer is supported.
"""
from bisect import bisect_left
from dataclasses import asdict, dataclass
import json
import math
import re

from werewolf.artifact_io import canonical_json_bytes, read_artifact_file, verify_artifact, sha256_bytes
from werewolf.observer_dtrs.estimation import condition
from werewolf.observer_dtrs.online_context import digest

VERSION = 'observer_dtrs_shadow_deployment_fixture_v1'
TYPE = 'observer_dtrs_shadow_deployment_fixture'
STATES = ('discussion/1', 'discussion/2', 'pk_discussion/1', 'pk_discussion/2')


def sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('explicit lowercase SHA-256 required')


def probability(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('invalid event probability; no clipping')


@dataclass(frozen=True)
class TomIdentity:
    family: str
    source_revision: str | None
    fit_or_execution_digest: str
    seal_digest: str
    terminal_digest: str
    weights_digest: str
    fold_index: int | None
    max_seq_len: int = 1024
    temporal_condition: str = 'explicit_day_phase'
    planner_version: str = 'classic7_structured_token_planner_v1'

    def __post_init__(self):
        if self.family not in ('controlled_fixture', 'd2_oof', 'final_tom'):
            raise ValueError('unknown ToM source family')
        for v in (self.fit_or_execution_digest, self.seal_digest, self.terminal_digest, self.weights_digest):
            sha(v)
        if (type(self.max_seq_len) is not int or self.max_seq_len < 4
                or self.temporal_condition != 'explicit_day_phase'
                or self.planner_version != 'classic7_structured_token_planner_v1'):
            raise ValueError('unsupported ToM public input contract')
        if self.family == 'controlled_fixture':
            if self.source_revision is not None or self.fold_index is not None:
                raise ValueError('artificial sources must not claim a real source commit/fold')
        elif (type(self.source_revision) is not str
                or re.fullmatch('[0-9a-f]{40}', self.source_revision) is None
                or (self.family == 'd2_oof' and (type(self.fold_index) is not int or self.fold_index not in range(5)))
                or (self.family == 'final_tom' and self.fold_index is not None)):
            raise ValueError('invalid ToM revision/fold')

    def to_record(self):
        return asdict(self)


def validate_payload(record):
    keys = {'schema_version', 'scope', 'model_id', 'weighting', 'tom', 'source_model_digest',
            'source_evidence_digest', 'selection_rule_digest', 'inference'}
    if (set(record) != keys or record['schema_version'] != VERSION
            or record['scope'] not in ('controlled_fixture', 'verified_d2_fixture')
            or record['model_id'] not in ('OBS-EQ0', 'OBS-EQ1', 'OBS-EF', 'OBS-LR-r')
            or record['weighting'] not in ('game_equal', 'record_equal')):
        raise ValueError('invalid deployment fixture schema')
    tom = TomIdentity(**record['tom'])
    expected = 'controlled_fixture' if record['scope'] == 'controlled_fixture' else 'd2_oof'
    if tom.family != expected:
        raise ValueError('fixture scope/ToM source mismatch; final deployment is not enabled')
    for k in ('source_model_digest', 'source_evidence_digest', 'selection_rule_digest'):
        sha(record[k])
    m = record['inference']
    if record['model_id'] == 'OBS-LR-r':
        if set(m) != {'mean_r', 'std_r', 'coefficients', 'constant'}:
            raise ValueError('invalid Logistic inference schema')
        if (type(m['mean_r']) not in (int, float) or not math.isfinite(m['mean_r'])
                or not 0 <= m['mean_r'] <= 1+1e-6
                or type(m['std_r']) not in (int, float) or not math.isfinite(m['std_r']) or m['std_r'] < 0):
            raise ValueError('invalid frozen Logistic standardization')
        if m['coefficients'] is None:
            probability(m['constant'])
        elif (m['constant'] is not None or type(m['coefficients']) is not list or len(m['coefficients']) != 6
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in m['coefficients'])):
            raise ValueError('invalid Logistic coefficients')
    else:
        if set(m) != {'root', 'wolf_nodes', 'state_nodes'}:
            raise ValueError('invalid inference table schema')
        probability(m['root'])
        if not set(m['wolf_nodes']) <= {'1', '2'} or not set(m['state_nodes']) <= set(STATES):
            raise ValueError('invalid frozen probability stratum')
        for p in m['wolf_nodes'].values():
            probability(p)
        for state, node in m['state_nodes'].items():
            if set(node) != {'probability', 'edges', 'leaves'} or state[-1] not in m['wolf_nodes']:
                raise ValueError('invalid inference node')
            probability(node['probability'])
            if (type(node['edges']) is not list or type(node['leaves']) is not list
                    or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1+1e-6 for v in node['edges'])
                    or node['edges'] != sorted(set(node['edges']))
                    or bool(node['edges']) != bool(node['leaves'])
                    or (node['edges'] and len(node['leaves']) != len(node['edges'])+1)
                    or (record['model_id'] == 'OBS-EQ0' and node['edges'])):
                raise ValueError('invalid frozen bin boundaries')
            for p in node['leaves']:
                probability(p)


@dataclass(frozen=True)
class FrozenEventEstimator:
    payload: bytes
    manifest_digest: str
    manifest_payload: bytes

    def __post_init__(self):
        sha(self.manifest_digest)
        r = self.record
        if self.payload != canonical_json_bytes(r):
            raise ValueError('noncanonical inference payload')
        validate_payload(r)
        m = json.loads(self.manifest_payload)
        if (self.manifest_payload != canonical_json_bytes(m)
                or set(m) != {'artifact_type', 'schema_version', 'scope', 'inference_digest', 'file_table', 'manifest_digest'}
                or m['artifact_type'] != TYPE or m['schema_version'] != VERSION or m['scope'] != r['scope']
                or m['inference_digest'] != self.digest or m['manifest_digest'] != self.manifest_digest
                or digest({k: v for k, v in m.items() if k != 'manifest_digest'}) != self.manifest_digest
                or m['file_table'] != {'inference.json': {'byte_size': len(self.payload), 'sha256': sha256_bytes(self.payload)}}):
            raise ValueError('inference bytes are not bound to the pinned manifest')

    @property
    def record(self):
        return json.loads(self.payload)

    @property
    def tom(self):
        return TomIdentity(**self.record['tom'])

    @property
    def digest(self):
        return digest(self.record)

    def predict(self, features):
        phase, n = condition(features)
        r = self.record
        m = r['inference']
        if r['model_id'] == 'OBS-LR-r':
            # Reuse the original deterministic vector/float64 implementation.
            from werewolf.observer_dtrs.baselines import predict_many
            return predict_many({'model_id': 'OBS-LR-r', **m}, [features])[0]
        p, level = m['root'], 'global'
        if str(n) in m['wolf_nodes']:
            p, level = m['wolf_nodes'][str(n)], 'nW'
        node = m['state_nodes'].get(f'{phase}/{n}')
        if node:
            p, level = node['probability'], 'phase_nW'
            if node['edges']:
                p, level = node['leaves'][bisect_left(node['edges'], features.r)], 'r_bin'
        probability(p)
        return {'probability': p, 'level': level}


def read_deployment(path, *, expected_manifest_digest, expected_tom):
    sha(expected_manifest_digest)
    if not isinstance(expected_tom, TomIdentity):
        raise TypeError('explicit ToM trust anchor required')
    a = verify_artifact(path, expected_artifact_type=TYPE, expected_schema_version=VERSION)
    if a.manifest_digest != expected_manifest_digest or set(a.manifest['file_table']) != {'inference.json'}:
        raise ValueError('deployment artifact identity/inventory mismatch')
    model = FrozenEventEstimator(read_artifact_file(a, 'inference.json'), a.manifest_digest, canonical_json_bytes(a.manifest))
    if (model.tom != expected_tom or a.manifest['inference_digest'] != model.digest
            or a.manifest['scope'] != model.record['scope']):
        raise ValueError('deployment ToM/source mismatch')
    return model
