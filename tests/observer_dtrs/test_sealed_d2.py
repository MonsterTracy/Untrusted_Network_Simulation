"""Synthetic sealed artifacts only; no formal model fitting or external data."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import json
import shutil

import pytest

from werewolf.artifact_io import ArtifactValidationError, canonical_json_bytes, sha256_bytes
from werewolf.observer_dtrs.d2_protocol import frozen_protocol
from werewolf.observer_dtrs.decision import LossSpec, TiePolicy
from werewolf.observer_dtrs.sealed_d2_offline import (
    SourceAnchors, evaluate_batch, open_sealed_targets, read_controlled_d2, read_formal_d2,
)
from werewolf.observer_dtrs.splits import partition_games
from werewolf.observer_dtrs.statistical_analysis import MODELS, WEIGHTINGS, analyze
from werewolf.observer_dtrs.validation_artifacts import digest, publish_unit
from werewolf.observer_dtrs.validation_data import controlled_fixture


@pytest.fixture(scope='module')
def controlled_package(tmp_path_factory):
    root = tmp_path_factory.mktemp('sealed-controlled')
    protocol = frozen_protocol()
    data = controlled_fixture(protocol)
    sources = {'head': 'a'*40, 'files': {'controlled_only.txt': 'synthetic, no ToM source'},
               'runtime': {'scope': 'controlled_fixture'}}
    c = {'scope': data.scope, 'protocol_digest': protocol.digest, 'dataset_digest': data.dataset_digest,
         'split_digest': data.split_digest, 'source_snapshot_digest': digest(sources), 'origins': data.origins,
         'game_sizes': data.sizes, 'game_folds': data.folds, 'models': list(MODELS), 'weightings': list(WEIGHTINGS),
         'split_record': data.split, 'sample_ids_digest_by_game': {
             g: digest([r.sample_id for r in data.legal((g,))]) for g in data.sizes},
         'known_source_limitations': protocol.record['known_source_limitations'],
         'research_scope': protocol.record['research_scope']}
    ca = publish_unit(root/'contract', 'validation_contract', {'run_identity': digest(c)}, c,
                      {'sources.json': canonical_json_bytes(sources), 'protocol.json': protocol.payload})
    refs, blocks, all_rows = {}, {}, []
    for k in range(5):
        for j in range(5):
            train, test = partition_games(data.split, tom_fold=k, outer_fold=j)
            legal, targets = data.legal(test), data.targets(test)
            rows = [{'sample_id': r.sample_id, 'game_id': r.game_id, 'phase': r.features.phase,
                'nW': r.features.nW, **targets[r.sample_id], 'predictions': {
                    w: {m: (mi+1+wi*.1)/5 for mi, m in enumerate(MODELS)}
                    for wi, w in enumerate(WEIGHTINGS)}} for r in legal]
            references = {w: {kind: digest(['controlled_only', k, j, w, kind])
                             for kind in ('selection', 'predictions', 'evaluation')} for w in WEIGHTINGS}
            b = publish_unit(root/'blocks'/f'{k}-{j}'/'complete', 'validation_block_complete',
                {'contract_digest': ca.manifest_digest, 'tom_fold': k, 'outer_fold': j},
                {'status': 'PAIRED_BLOCK_COMPLETE', 'references': references, 'rows': rows,
                 'train_games': list(train), 'test_games': list(test)})
            refs[f'{k}-{j}'], blocks[(k,j)] = b.manifest_digest, test
            all_rows.extend(rows)
    summary, indices = analyze(all_rows, blocks=blocks, protocol=protocol, scope='controlled_fixture')
    final = publish_unit(root/'complete', 'validation_complete',
        {'contract_digest': ca.manifest_digest, 'block_digests': refs},
        {'status': 'ALL_BLOCKS_COMPLETE', 'block_digests': refs, 'summary': summary},
        {'bootstrap_indices.int64le': indices})
    return root, SourceAnchors(final.manifest_digest, sources['head'], data.dataset_digest, data.split_digest)


@pytest.fixture
def package(controlled_package, tmp_path):
    root, anchors = controlled_package
    destination = tmp_path/'copied-fixture'
    shutil.copytree(root, destination)
    return destination, anchors


def rewrite(directory, *, transform=lambda r: None, transform_identity=lambda r: None):
    m = json.loads((directory/'manifest.json').read_bytes())
    r = json.loads((directory/'record.json').read_bytes())
    transform(r)
    transform_identity(m['identity'])
    raw = canonical_json_bytes(r)
    (directory/'record.json').write_bytes(raw)
    m['file_table']['record.json'] = {'byte_size': len(raw), 'sha256': sha256_bytes(raw)}
    m['identity_digest'] = digest(m['identity'])
    m['manifest_digest'] = digest({k:v for k,v in m.items() if k != 'manifest_digest'})
    (directory/'manifest.json').write_bytes(canonical_json_bytes(m))
    return m['manifest_digest']


def reseal_block(root, anchors, *, transform=lambda r: None, transform_identity=lambda r: None):
    h = rewrite(root/'blocks/0-0/complete', transform=transform, transform_identity=transform_identity)
    complete = rewrite(root/'complete', transform=lambda r: r['block_digests'].__setitem__('0-0', h),
                       transform_identity=lambda i: i['block_digests'].__setitem__('0-0', h))
    return replace(anchors, complete_digest=complete)


def reseal_contract(root, anchors, transform):
    record = json.loads((root/'contract/record.json').read_bytes())
    transform(record)
    h = rewrite(root/'contract', transform=lambda r: r.update(record),
                transform_identity=lambda i: i.update(run_identity=digest(record)))
    complete = rewrite(root/'complete', transform_identity=lambda i: i.update(contract_digest=h))
    return replace(anchors, complete_digest=complete)


def test_split_inputs_and_all_eight_model_bindings(controlled_package, monkeypatch):
    root, anchors = controlled_package
    import werewolf.observer_dtrs.decision_offline as offline
    def forbid(*args, **kwargs):
        raise AssertionError('loading real or synthetic sources must never classify')
    monkeypatch.setattr(offline, 'decide', forbid)
    view = read_controlled_d2(root, anchors=anchors)
    assert view.identity['scope'] == 'controlled_fixture'
    assert view.identity['samples'] == 150 and len(view.identity['game_sizes']) == 75
    assert view.identity['independent_D1c_labels_verified'] is False
    for wi,w in enumerate(WEIGHTINGS):
        for mi,m in enumerate(MODELS):
            batch = view.batch(m,w)
            assert len(batch.rows) == 150
            assert all(float(r.probability) == (mi+1+wi*.1)/5 for r in batch.rows)
            assert set(vars(batch.rows[0])) == {'sample_id', 'game_id', 'probability'}
            assert not hasattr(batch.origins[0], 'y') and not hasattr(batch.origins[0], 'successful_empty')
    with pytest.raises(ArtifactValidationError):
        view.batch('old-Theta_AC', 'game_equal')
    with pytest.raises(ArtifactValidationError):
        read_formal_d2(root, anchors=anchors)


def test_supervision_is_explicit_id_join_and_loss_policy_is_preserved(controlled_package):
    view = read_controlled_d2(controlled_package[0], anchors=controlled_package[1])
    batch = view.batch('OBS-EQ1', 'game_equal')
    targets = open_sealed_targets(view)
    loss = LossSpec(1,1,Fraction(1,4),Fraction(1,4),TiePolicy.TERMINAL_NEG_FIRST)
    first = evaluate_batch(batch, targets, loss=loss, binary_tie_policy=TiePolicy.TERMINAL_NEG_FIRST)
    second = evaluate_batch(replace(batch, rows=tuple(reversed(batch.rows))), targets,
                            loss=loss, binary_tie_policy=TiePolicy.TERMINAL_NEG_FIRST)
    assert first == second
    assert first['loss_digest'] == loss.digest
    assert first['source']['model_id'] == 'OBS-EQ1'
    assert len(first['sample_origins']) == 150 and all('block_digest' in r for r in first['sample_origins'])
    assert all('y' not in r and 'successful_empty' not in r for r in first['evaluation']['decisions'])
    changed = json.loads(targets.payload)
    sid = next(iter(changed)); changed[sid]['y'] = 1-changed[sid]['y']
    with pytest.raises(ArtifactValidationError, match='changed sealed targets'):
        evaluate_batch(batch, replace(targets, payload=canonical_json_bytes(changed)),
                       loss=loss, binary_tie_policy=TiePolicy.TERMINAL_NEG_FIRST)
    with pytest.raises(ArtifactValidationError, match='cross-run'):
        evaluate_batch(batch, replace(targets, run_identity_digest='0'*64),
                       loss=loss, binary_tie_policy=TiePolicy.TERMINAL_NEG_FIRST)


@pytest.mark.parametrize('mutate', [
    lambda b: b['rows'].pop(),
    lambda b: b['rows'].append(deepcopy(b['rows'][0])),
    lambda b: b['rows'][0].__setitem__('game_id', 'cross-game'),
    lambda b: b['rows'][0].__setitem__('sample_id', 'mismatched-sample'),
    lambda b: b['rows'][0]['predictions']['game_equal'].pop('OBS-EF'),
    lambda b: b['rows'][0]['predictions'].__setitem__('old-weight', b['rows'][0]['predictions'].pop('record_equal')),
    lambda b: b['rows'][0]['predictions']['game_equal'].__setitem__('OBS-EQ1', -1),
    lambda b: b['rows'][0]['predictions']['game_equal'].__setitem__('OBS-EQ1', True),
    lambda b: b['rows'][0].update(y=0, successful_empty=True),
    lambda b: b['rows'][0].__setitem__('supervision', {'leak': True}),
    lambda b: b['references']['game_equal'].__setitem__('predictions', 'not-a-digest'),
    lambda b: b['test_games'].append(b['train_games'][0]),
])
def test_rehashed_semantically_invalid_blocks_are_rejected(package, mutate):
    root, anchors = package
    anchors = reseal_block(root, anchors, transform=mutate)
    with pytest.raises(ArtifactValidationError):
        read_controlled_d2(root, anchors=anchors)


def test_rehashed_wrong_fold_and_model_binding_rejected(package):
    root, anchors = package
    anchors = reseal_block(root, anchors, transform_identity=lambda i: i.__setitem__('tom_fold', 1))
    with pytest.raises(ArtifactValidationError, match='identity'):
        read_controlled_d2(root, anchors=anchors)


def test_model_and_weight_metadata_cannot_reinterpret_probability_map(controlled_package):
    view = read_controlled_d2(controlled_package[0], anchors=controlled_package[1])
    batch, targets = view.batch('OBS-EQ1','game_equal'), open_sealed_targets(view)
    loss = LossSpec(1,1,Fraction(1,4),Fraction(1,4),TiePolicy.DEFER_FIRST)
    for key, value in [('model_id','OBS-EQ0'), ('weighting','record_equal')]:
        ident = batch.identity; ident[key] = value
        with pytest.raises(ArtifactValidationError, match='binding'):
            evaluate_batch(replace(batch,identity_payload=canonical_json_bytes(ident)), targets,
                           loss=loss,binary_tie_policy=TiePolicy.TERMINAL_NEG_FIRST)
    origins = (replace(batch.origins[0], outer_fold=99), *batch.origins[1:])
    with pytest.raises(ArtifactValidationError, match='origins'):
        evaluate_batch(replace(batch,origins=origins),targets,loss=loss,
                       binary_tie_policy=TiePolicy.TERMINAL_NEG_FIRST)
    ident = batch.identity
    ident['outer_prediction_digest_by_block']['0-0'] = '0'*64
    with pytest.raises(ArtifactValidationError, match='references'):
        evaluate_batch(replace(batch,identity_payload=canonical_json_bytes(ident)),targets,
                       loss=loss,binary_tie_policy=TiePolicy.TERMINAL_NEG_FIRST)


@pytest.mark.parametrize('relative', ['contract/sources.json', 'blocks/4-4/complete/manifest.json',
                                    'complete/bootstrap_indices.int64le'])
def test_missing_or_corrupt_files_stop_without_repair(package, relative):
    root, anchors = package
    p = root/relative
    p.write_bytes(b'controlled corruption')
    with pytest.raises(ArtifactValidationError):
        read_controlled_d2(root, anchors=anchors)
    assert p.read_bytes() == b'controlled corruption'


@pytest.mark.parametrize('key,value', [('source_revision','b'*40),('dataset_digest','b'*64),
                                     ('split_digest','b'*64),('complete_digest','b'*64)])
def test_explicit_source_anchor_mismatch(controlled_package,key,value):
    root, anchors = controlled_package
    with pytest.raises(ArtifactValidationError):
        read_controlled_d2(root, anchors=replace(anchors, **{key:value}))


@pytest.mark.parametrize('relative', ['contract/manifest.json', 'blocks/3-2/complete/record.json'])
def test_missing_artifacts_do_not_downgrade_to_partial(package, relative):
    root, anchors = package
    (root/relative).unlink()
    with pytest.raises(ArtifactValidationError):
        read_controlled_d2(root, anchors=anchors)
    assert not (root/relative).exists()


@pytest.mark.parametrize('mutate', [
    lambda c: c.__setitem__('models', ['old-Theta_AC']),
    lambda c: c.__setitem__('known_source_limitations', []),
    lambda c: c.__setitem__('source_snapshot_digest', '0'*64),
    lambda c: c['split_record']['folds'][0]['outer_assignment'].__setitem__('cross-source-game', 0),
    lambda c: c['game_folds'].__setitem__(next(iter(c['game_folds'])), 99),
])
def test_rehashed_contract_semantic_conflicts(package, mutate):
    root, anchors = package
    anchors = reseal_contract(root, anchors, mutate)
    with pytest.raises(ArtifactValidationError):
        read_controlled_d2(root, anchors=anchors)


def test_identity_and_target_accessors_do_not_expose_mutable_internal_state(controlled_package):
    view = read_controlled_d2(controlled_package[0], anchors=controlled_package[1])
    identity = view.identity
    identity['game_sizes'].clear()
    assert len(view.identity['game_sizes']) == 75
    targets = open_sealed_targets(view)
    labels = targets.labels; labels.clear()
    assert len(targets.labels) == 150
