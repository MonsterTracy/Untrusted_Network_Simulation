"""Read-only D2 complete/contract/block adapter with explicit supervised access.

Validates the sealed minimal package, not inner selection, fitted outer models or
independent D1c labels. Stored source text is hashed, never imported or executed.
"""
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path, PurePosixPath
import re

from werewolf.artifact_io import (
    ArtifactValidationError, canonical_json_bytes, read_artifact_file, sha256_bytes, verify_artifact,
)
from werewolf.observer_dtrs.d2_protocol import FrozenProtocol
from werewolf.observer_dtrs.decision_offline import ProbabilityInput, evaluate
from werewolf.observer_dtrs.splits import build_splits, game_partition, partition_games
from werewolf.observer_dtrs.statistical_analysis import MODELS, WEIGHTINGS, bootstrap_plan, VERSION as ANALYSIS_VERSION
from werewolf.observer_dtrs.validation_artifacts import VERSION as D2_VERSION, digest

VERSION = 'observer_dtrs_sealed_d2_adapter_v1'


def _check(condition, message):
    if not condition:
        raise ArtifactValidationError(message)


def _sha(value):
    _check(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None,
           'invalid SHA-256 source identity')
    return value


@dataclass(frozen=True)
class SourceAnchors:
    complete_digest: str
    source_revision: str
    dataset_digest: str
    split_digest: str

    def __post_init__(self):
        for value in (self.complete_digest, self.dataset_digest, self.split_digest):
            _sha(value)
        _check(isinstance(self.source_revision, str)
               and re.fullmatch(r'[0-9a-f]{40}', self.source_revision) is not None,
               'explicit full source commit OID required')


def _json(raw):
    value = json.loads(raw)
    _check(raw == canonical_json_bytes(value), 'noncanonical sealed JSON')
    return value


def _unit(path, kind, expected_digest, *, identity=None, extras=()):
    a = verify_artifact(path, expected_artifact_type='observer_dtrs_'+kind,
                        expected_schema_version=D2_VERSION)
    _check(a.manifest_digest == expected_digest, 'sealed artifact trust anchor mismatch')
    _check(set(a.manifest) == {'artifact_type', 'schema_version', 'file_table',
           'manifest_digest', 'identity', 'identity_digest'}, 'unexpected sealed envelope fields')
    _check(set(a.manifest['file_table']) == {'record.json', *extras}, 'sealed payload inventory mismatch')
    _check(a.manifest['identity_digest'] == digest(a.manifest['identity']), 'unit identity digest mismatch')
    if identity is not None:
        _check(a.manifest['identity'] == identity, 'sealed unit source/fold identity mismatch')
    return a, _json(read_artifact_file(a, 'record.json'))


@dataclass(frozen=True)
class SampleOrigin:
    sample_id: str
    game_id: str
    phase: str
    nW: int
    tom_fold: int
    outer_fold: int
    block_digest: str


@dataclass(frozen=True)
class LegalSample:
    origin: SampleOrigin
    probabilities: tuple[float, ...]  # WEIGHTINGS-major then MODELS; no target fields.


def _rows_digest(rows):
    return digest(sorted([r.sample_id, r.game_id, float(r.probability)] for r in rows))


@dataclass(frozen=True)
class PredictionBatch:
    rows: tuple[ProbabilityInput, ...]
    origins: tuple[SampleOrigin, ...]
    identity_payload: bytes

    @property
    def identity(self):
        return json.loads(self.identity_payload)


@dataclass(frozen=True)
class SealedTargets:
    run_identity_digest: str
    payload: bytes

    @property
    def labels(self):
        return {sid: row['y'] for sid, row in json.loads(self.payload).items()}

    @property
    def successful_empty(self):
        return {sid: row['successful_empty'] for sid, row in json.loads(self.payload).items()}


@dataclass(frozen=True)
class SealedRun:
    identity_payload: bytes
    samples: tuple[LegalSample, ...]
    _targets_payload: bytes

    @property
    def identity(self):
        return json.loads(self.identity_payload)

    @property
    def identity_digest(self):
        return digest(self.identity)

    def batch(self, model_id, weighting):
        _check(model_id in MODELS and weighting in WEIGHTINGS, 'unknown model or weighting')
        offset = WEIGHTINGS.index(weighting)*len(MODELS)+MODELS.index(model_id)
        rows = tuple(ProbabilityInput(s.origin.sample_id, s.origin.game_id, s.probabilities[offset])
                     for s in self.samples)
        identity = {'schema_version': VERSION, 'run_identity_digest': self.identity_digest,
                    'run': self.identity, 'model_id': model_id, 'weighting': weighting,
                    'predictions_digest': _rows_digest(rows),
                    'origins_digest': self.identity['sample_origins_digest'],
                    'outer_prediction_digest_by_block': {
                        b: refs[weighting]['predictions'] for b, refs in self.identity['outer_references'].items()}}
        return PredictionBatch(rows, tuple(s.origin for s in self.samples), canonical_json_bytes(identity))


def open_sealed_targets(view):
    """Explicit offline opt-in; no D1c/raw-report verification is implied."""
    if not isinstance(view, SealedRun):
        raise TypeError('opened SealedRun required')
    _check(digest(_json(view._targets_payload)) == view.identity['sealed_targets_digest'],
           'sealed supervision binding mismatch')
    return SealedTargets(view.identity_digest, view._targets_payload)


def evaluate_batch(batch, targets, *, loss, binary_tie_policy, subgroup_ids=None):
    """Explicit bridge, never invoked by loading/auditing a real package."""
    if not isinstance(batch, PredictionBatch) or not isinstance(targets, SealedTargets):
        raise TypeError('verified probability batch and explicitly opened sealed targets required')
    identity = _json(batch.identity_payload)
    run = identity['run']
    _check(set(identity) == {'schema_version', 'run_identity_digest', 'run', 'model_id', 'weighting',
                            'predictions_digest', 'origins_digest', 'outer_prediction_digest_by_block'}
           and identity['schema_version'] == VERSION, 'batch schema mismatch')
    _check(identity['model_id'] in MODELS and identity['weighting'] in WEIGHTINGS,
           'batch model/weighting identity mismatch')
    _check(identity['run_identity_digest'] == targets.run_identity_digest == digest(run),
           'cross-run targets/batch mismatch')
    raw_targets = _json(targets.payload)
    _check(digest(raw_targets) == run['sealed_targets_digest'], 'changed sealed targets')
    _check(_rows_digest(batch.rows) == identity['predictions_digest']
           == run['prediction_digests'][identity['weighting']+'/'+identity['model_id']],
           'changed probability batch/model/weighting binding')
    _check(digest(sorted((asdict(o) for o in batch.origins), key=lambda o: o['sample_id']))
           == identity['origins_digest'] == run['sample_origins_digest'], 'changed sample origins')
    _check(identity['outer_prediction_digest_by_block'] == {
        b: refs[identity['weighting']]['predictions'] for b, refs in run['outer_references'].items()},
        'changed outer model artifact references')
    _check({r.sample_id: r.game_id for r in batch.rows}
           == {s: v['game_id'] for s, v in raw_targets.items()}, 'batch/targets sample identity mismatch')
    result = evaluate(batch.rows, targets.labels, loss=loss, game_sizes=run['game_sizes'],
                      weighting=identity['weighting'], binary_tie_policy=binary_tie_policy,
                      subgroup_ids=subgroup_ids)
    chosen_ids = {r['sample_id'] for r in result['decisions']}
    return {'source': identity, 'loss_digest': loss.digest, 'research_scope': run['research_scope'],
            'verification_level': run['verification_level'],
            'sample_origins': sorted((asdict(o) for o in batch.origins if o.sample_id in chosen_ids),
                                     key=lambda o: o['sample_id']),
            'evaluation': result}


def _validate_contract(c, sources, protocol, anchors, scope):
    _check(set(c) == {'scope', 'protocol_digest', 'dataset_digest', 'split_digest', 'source_snapshot_digest',
           'origins', 'game_sizes', 'game_folds', 'models', 'weightings', 'split_record',
           'sample_ids_digest_by_game', 'known_source_limitations', 'research_scope'}, 'contract schema mismatch')
    _check(c['scope'] == scope, 'formal/controlled scope mismatch')
    _check(c['protocol_digest'] == protocol.digest and c['dataset_digest'] == anchors.dataset_digest
           and c['split_digest'] == anchors.split_digest, 'protocol/dataset/split source anchor mismatch')
    _check(c['models'] == list(MODELS) and c['weightings'] == list(WEIGHTINGS), 'contract model/weighting mismatch')
    _check(c['known_source_limitations'] == protocol.record['known_source_limitations']
           and c['research_scope'] == protocol.record['research_scope'], 'research limitations/scope mismatch')
    _check(set(sources) == {'head', 'files', 'runtime'} and digest(sources) == c['source_snapshot_digest']
           and sources['head'] == anchors.source_revision, 'source snapshot/OID mismatch')
    _check(isinstance(sources['files'], dict) and bool(sources['files']), 'empty source snapshot')
    for name, text in sources['files'].items():
        path = PurePosixPath(name)
        _check(isinstance(text, str) and not path.is_absolute() and '..' not in path.parts
               and '\n' not in name and '\r' not in name, 'invalid stored source name/text')
    sizes, folds, sample_digests = c['game_sizes'], c['game_folds'], c['sample_ids_digest_by_game']
    _check(bool(sizes) and set(sizes) == set(folds) == set(sample_digests), 'contract game domain mismatch')
    _check(all(isinstance(g, str) and g and type(n) is int and n > 0 for g, n in sizes.items())
           and all(type(f) is int and f in range(5) for f in folds.values()), 'invalid game size/fold')
    for value in sample_digests.values():
        _sha(value)
    if scope == 'formal_development_validation':
        _check(c['dataset_digest'] == protocol.record['dataset_digest'], 'not frozen formal dataset')
        split = c['split_record']
        games = split['source_games']
        _check({g['game_id']: g['row_count'] for g in games} == sizes
               and {g['game_id']: g['fold_index'] for g in games} == folds
               and {g['game_id']: g['sample_ids_digest'] for g in games} == sample_digests,
               'split source games do not match contract')
        rebuilt = build_splits({'manifest_digest': c['dataset_digest'], 'games': games,
            'origins': c['origins'], 'coverage': {'sample_count': sum(sizes.values())}}, protocol)
    else:
        records = []
        for k in range(5):
            group = sorted(g for g, f in folds.items() if f == k)
            outer = game_partition(group, protocol=protocol, tom_fold=k, folds=5)
            nested = [{'outer_fold': j, 'inner_assignment': game_partition(
                [g for g in group if outer[g] != j], protocol=protocol, tom_fold=k, folds=3, parent_outer=j)}
                for j in range(5)]
            records.append({'tom_fold': k, 'outer_assignment': outer, 'nested': nested})
        rebuilt = {'folds': records, 'coverage': {'games': len(sizes), 'samples': sum(sizes.values())},
                   'scope': 'controlled_fixture'}
    _check(c['split_record'] == rebuilt, 'label-independent split reconstruction mismatch')
    if scope == 'controlled_fixture':
        _check(digest(rebuilt) == c['split_digest'], 'controlled split digest mismatch')


def _read(root, anchors, scope):
    root = Path(root)
    final_a, final = _unit(root/'complete', 'validation_complete', anchors.complete_digest,
                           extras=('bootstrap_indices.int64le',))
    _check(set(final) == {'status', 'block_digests', 'summary'} and final['status'] == 'ALL_BLOCKS_COMPLETE',
           'missing full completion seal')
    fid = final_a.manifest['identity']
    _check(set(fid) == {'contract_digest', 'block_digests'} and final['block_digests'] == fid['block_digests']
           and set(final['block_digests']) == {f'{k}-{j}' for k in range(5) for j in range(5)},
           'all 25 sealed blocks required')
    ca, c = _unit(root/'contract', 'validation_contract', _sha(fid['contract_digest']),
                  extras=('sources.json', 'protocol.json'))
    _check(ca.manifest['identity'] == {'run_identity': digest(c)}, 'contract run identity mismatch')
    protocol = FrozenProtocol(read_artifact_file(ca, 'protocol.json'))
    _validate_contract(c, _json(read_artifact_file(ca, 'sources.json')), protocol, anchors, scope)
    samples, targets, outer_refs, blocks = {}, {}, {}, {}
    for k in range(5):
        for j in range(5):
            key = f'{k}-{j}'
            block_digest = _sha(final['block_digests'][key])
            _, b = _unit(root/'blocks'/key/'complete', 'validation_block_complete', block_digest,
                identity={'contract_digest': ca.manifest_digest, 'tom_fold': k, 'outer_fold': j})
            train, test = partition_games(c['split_record'], tom_fold=k, outer_fold=j)
            _check(set(b) == {'status', 'references', 'rows', 'train_games', 'test_games'}
                   and b['status'] == 'PAIRED_BLOCK_COMPLETE'
                   and b['train_games'] == list(train) and b['test_games'] == list(test)
                   and not set(train)&set(test) and all(c['game_folds'][g] == k for g in (*train, *test)),
                   'block game/source fold mismatch')
            _check(set(b['references']) == set(WEIGHTINGS), 'missing weighting references')
            for refs in b['references'].values():
                _check(set(refs) == {'selection', 'predictions', 'evaluation'}, 'outer reference schema mismatch')
                for value in refs.values():
                    _sha(value)
            _check(Counter(r['game_id'] for r in b['rows']) == Counter({g: c['game_sizes'][g] for g in test}),
                   'missing/cross-game block rows')
            for g in test:
                _check(digest([r['sample_id'] for r in b['rows'] if r['game_id'] == g])
                       == c['sample_ids_digest_by_game'][g], 'sealed per-game sample identity mismatch')
            for row in b['rows']:
                _check(set(row) == {'sample_id', 'game_id', 'phase', 'nW', 'y', 'successful_empty', 'predictions'},
                       'sealed row schema mismatch')
                sid, game = row['sample_id'], row['game_id']
                _check(isinstance(sid, str) and bool(sid) and sid not in samples, 'duplicate/invalid sample ID')
                if scope == 'formal_development_validation':
                    _sha(sid)
                _check(row['phase'] in ('discussion', 'pk_discussion') and type(row['nW']) is int
                       and row['nW'] in (1, 2), 'invalid phase/nW')
                _check(type(row['y']) is int and row['y'] in (0, 1) and type(row['successful_empty']) is bool
                       and (not row['successful_empty'] or row['y'] == 1), 'invalid sealed target/empty semantics')
                _check(set(row['predictions']) == set(WEIGHTINGS), 'row weighting mismatch')
                probabilities = []
                for w in WEIGHTINGS:
                    _check(set(row['predictions'][w]) == set(MODELS), 'row model domain mismatch')
                    for model in MODELS:
                        value = row['predictions'][w][model]
                        ProbabilityInput(sid, game, value)  # Reuse D3b's strict scalar probability contract.
                        probabilities.append(value)
                origin = SampleOrigin(sid, game, row['phase'], row['nW'], k, j, block_digest)
                samples[sid] = LegalSample(origin, tuple(probabilities))
                targets[sid] = {'game_id': game, 'y': row['y'], 'successful_empty': row['successful_empty']}
            outer_refs[key], blocks[(k, j)] = b['references'], test
    _check(Counter(s.origin.game_id for s in samples.values()) == Counter(c['game_sizes']),
           'global sample population mismatch')
    summary = final['summary']
    _check(summary['schema_version'] == ANALYSIS_VERSION and summary['scope'] == scope and summary['protocol_digest'] == protocol.digest
           and summary['research_scope'] == c['research_scope']
           and summary['known_source_limitations'] == c['known_source_limitations']
           and summary['games'] == len(c['game_sizes']) and summary['rows'] == len(samples),
           'complete summary population/scope mismatch')
    _check(set(summary['models']) == set(WEIGHTINGS), 'summary weighting domain mismatch')
    for model_map in summary['models'].values():
        _check(set(model_map) == set(MODELS), 'summary model domain mismatch')
        for model in model_map.values():
            for group, keep in {'overall': lambda sid: True,
                'successful_empty': lambda sid: targets[sid]['successful_empty'],
                'nonempty': lambda sid: not targets[sid]['successful_empty'],
                **{f'{p}/{n}': (lambda sid, p=p, n=n: samples[sid].origin.phase == p
                     and samples[sid].origin.nW == n)
                   for p in ('discussion', 'pk_discussion') for n in (1, 2)}}.items():
                selected = [sid for sid in samples if keep(sid)]
                stats = model['groups'][group]
                _check(stats['rows'] == len(selected)
                       and stats['games'] == len({samples[sid].origin.game_id for sid in selected})
                       and stats['positive'] == sum(targets[sid]['y'] for sid in selected),
                       'sealed targets/summary coverage mismatch')
    order, draws = bootstrap_plan(blocks, protocol=protocol, formal=scope == 'formal_development_validation')
    _check(summary['game_order'] == order and summary['bootstrap'] == {
        **protocol.record['uncertainty'], 'indices_sha256': sha256_bytes(draws.astype('<i8').tobytes()),
        'game_order_digest': digest(order)}, 'bootstrap specification/game identity mismatch')
    _check(read_artifact_file(final_a, 'bootstrap_indices.int64le') == draws.astype('<i8').tobytes(),
           'sealed bootstrap indices do not reconstruct')
    identity = {'schema_version': VERSION, 'verification_level': 'sealed_complete_contract_and_25_blocks',
        'scope': scope, 'complete_digest': anchors.complete_digest, 'contract_digest': ca.manifest_digest,
        'source_revision': anchors.source_revision, 'source_snapshot_digest': c['source_snapshot_digest'],
        'dataset_digest': c['dataset_digest'], 'split_digest': c['split_digest'],
        'protocol_digest': protocol.digest, 'research_scope': c['research_scope'],
        'known_source_limitations': c['known_source_limitations'], 'game_sizes': c['game_sizes'],
        'samples': len(samples), 'blocks': 25, 'block_digests': final['block_digests'],
        'outer_references': outer_refs, 'sealed_targets_digest': digest(targets),
        'independent_D1c_labels_verified': False, 'outer_models_verified': False,
        'inner_selection_replayed': False, 'full_run_replayed': False}
    identity['sample_origins_digest'] = digest([asdict(samples[sid].origin) for sid in sorted(samples)])
    identity['prediction_digests'] = {
        w+'/'+m: digest(sorted([sid, s.origin.game_id, float(s.probabilities[wi*len(MODELS)+mi])]
                              for sid, s in samples.items()))
        for wi, w in enumerate(WEIGHTINGS) for mi, m in enumerate(MODELS)}
    return SealedRun(canonical_json_bytes(identity), tuple(samples[sid] for sid in sorted(samples)),
                     canonical_json_bytes(targets))


def _read_checked(root, anchors, scope):
    if not isinstance(anchors, SourceAnchors):
        raise TypeError('explicit SourceAnchors required')
    try:
        return _read(root, anchors, scope)
    except ArtifactValidationError:
        raise
    except (OSError, KeyError, TypeError, ValueError, IndexError) as error:
        raise ArtifactValidationError(f'invalid sealed D2 package: {error}') from error


def read_formal_d2(root, *, anchors):
    return _read_checked(root, anchors, 'formal_development_validation')


def read_controlled_d2(root, *, anchors):
    """Explicitly synthetic fixture reader; cannot certify formal scope."""
    return _read_checked(root, anchors, 'controlled_fixture')
