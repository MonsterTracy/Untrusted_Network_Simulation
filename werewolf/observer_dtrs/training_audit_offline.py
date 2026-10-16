"""Restricted, read-only D1c crosschecks and saved outer-model support audit.

Never fits, learns bins, classifies, or supplies support/targets to decide.
Independent D1c verification is not original-canonical reconstruction; recorded
inner selection is checked without claiming candidate-fit replay.
"""
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from functools import wraps
import json
import math
from pathlib import Path
from types import SimpleNamespace

from werewolf.artifact_io import ArtifactValidationError, canonical_json_bytes, read_artifact_file
from werewolf.observer_dtrs.baselines import SOLVER, VERSION as BASELINE_VERSION, empirical_model, predict_many
from werewolf.observer_dtrs.d2_protocol import FrozenProtocol
from werewolf.observer_dtrs.dataset import LegalDatasetRow, open_dataset, sample_id
from werewolf.observer_dtrs.dataset_offline import open_supervision, verify_dataset
from werewolf.observer_dtrs.estimation import FittedModel, condition, validate_model
from werewolf.observer_dtrs.nested_validation import candidates, _select
from werewolf.observer_dtrs.scoring import score_predictions
from werewolf.observer_dtrs.sealed_d2_offline import (
    SealedRun, SealedTargets, SampleOrigin, _check, _json, _sha, _unit, open_sealed_targets,
)
from werewolf.observer_dtrs.splits import partition_games
from werewolf.observer_dtrs.statistical_analysis import MODELS, WEIGHTINGS
from werewolf.observer_dtrs.validation_artifacts import digest

VERSION = 'observer_dtrs_training_audit_v1'
INFERENCE_ABSOLUTE_TOLERANCE = 1e-12  # Cross-platform saved Logistic inference, never probability clipping.


def _guarded(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ArtifactValidationError:
            raise
        except (OSError, KeyError, TypeError, ValueError, IndexError, OverflowError) as error:
            raise ArtifactValidationError(f'invalid D3 training audit: {error}') from error
    return checked


def _legal_digest(rows):
    return digest([r.to_record() for r in sorted(rows, key=lambda r: r.sample_id)])


@dataclass(frozen=True)
class D1cLink:
    """Legal canonical contexts/features, with separately gated target bytes."""
    legal_rows: tuple[LegalDatasetRow, ...]
    identity_payload: bytes
    _targets_payload: bytes

    @property
    def identity(self):
        return _json(self.identity_payload)


def _validate_link(view, link):
    _check(isinstance(view, SealedRun) and isinstance(link, D1cLink), 'opened run and D1c link required')
    i = link.identity
    _check(i['run_identity_digest'] == view.identity_digest
           and i['dataset_digest'] == view.identity['dataset_digest'], 'cross-run D1c link')
    _check(len({r.sample_id for r in link.legal_rows}) == len(link.legal_rows)
           and _legal_digest(link.legal_rows) == i['legal_rows_digest'], 'changed D1c legal link')
    _check(digest(_json(link._targets_payload)) == i['targets_digest']
           == view.identity['sealed_targets_digest'], 'changed D1c target link')


@_guarded
def crosscheck_d1c(view, dataset_path, *, expected_dataset_digest):
    """Explicit offline opt-in: full official reader/audit, then SID/PRE/fold joins."""
    _check(isinstance(view, SealedRun), 'opened sealed run required')
    _sha(expected_dataset_digest)
    _check(expected_dataset_digest == view.identity['dataset_digest'], 'D1c/run dataset anchor mismatch')
    # Full file/capsule audit, not audit_dataset_sources (which needs original trajectories).
    verify_dataset(dataset_path, expected_digest=expected_dataset_digest)
    legal = open_dataset(dataset_path, expected_digest=expected_dataset_digest)
    offline = open_supervision(legal)
    manifest = legal.artifact.manifest
    sealed = open_sealed_targets(view)
    origins = {s.origin.sample_id: s.origin for s in view.samples}
    _check(len(legal.rows) == len(offline) == len(origins)
           and {r.sample_id for r in legal.rows} == set(origins), 'D1c missing/duplicate sample population')
    games = manifest['games']
    _check({g['game_id']: g['row_count'] for g in games} == view.identity['game_sizes'],
           'D1c game population mismatch')
    targets, seen_pre = {}, set()
    labels, empty = sealed.labels, sealed.successful_empty
    folds = {g['game_id']: g['fold_index'] for g in games}
    for row in offline:
        r, source, q = row.legal, row.sample_source, row.q_source
        c, o = r.context, origins[row.legal.sample_id]
        canonical = (c.game_id, c.boundary_id, c.prefix_digest, c.acting_wolf, c.observer_id)
        _check(canonical not in seen_pre and sample_id(c) == r.sample_id, 'duplicate/mismatched canonical PRE')
        seen_pre.add(canonical)
        _check((c.game_id, c.phase, r.features.nW) == (o.game_id, o.phase, o.nW)
               and folds[c.game_id] == source.fold_index == q.fold_index == o.tom_fold,
               'D1c cross-game/PRE/source fold mismatch')
        y, is_empty = row.supervision.event_label_y, not row.supervision.raw_suspicion_support
        _check(y == labels[r.sample_id] and is_empty == empty[r.sample_id], 'D1c/sealed label or empty-report mismatch')
        targets[r.sample_id] = {'game_id': c.game_id, 'y': y, 'successful_empty': is_empty}
    _check(digest(targets) == view.identity['sealed_targets_digest'], 'D1c target population digest mismatch')
    identity = {'schema_version': VERSION, 'run_identity_digest': view.identity_digest,
        'dataset_digest': expected_dataset_digest, 'legal_rows_digest': _legal_digest(legal.rows),
        'targets_digest': digest(targets), 'games': len(games), 'samples': len(legal.rows),
        'canonical_identities': len(seen_pre), 'independent_D1c_labels_verified': True,
        'original_canonical_reconstruction': False,
        'dataset_origins': manifest['origins'], 'game_records': games,
        'verification_level': 'full_D1c_reader_capsules_and_sealed_SID_PRE_fold_target_crosscheck'}
    return D1cLink(legal.rows, canonical_json_bytes(identity), canonical_json_bytes(targets))


@_guarded
def open_d1c_targets(view, link):
    """Gated independently checked targets compatible with the existing evaluator."""
    _validate_link(view, link)
    return SealedTargets(view.identity_digest, link._targets_payload)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _close_tree(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_close_tree(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close_tree(x, y) for x, y in zip(a, b))
    if _finite(a) and _finite(b):
        return math.isclose(a, b, rel_tol=0, abs_tol=INFERENCE_ABSOLUTE_TOLERANCE)
    return type(a) is type(b) and a == b


def _check_logistic(m, source, protocol, parameters, weighting, fingerprint):
    _check(set(m) == {'schema_version', 'model_id', 'protocol_digest', 'parameters', 'source',
        'training_fingerprint', 'mean_r', 'std_r', 'solver', 'constant', 'coefficients', 'convergence'}
        and m['schema_version'] == BASELINE_VERSION and m['model_id'] == 'OBS-LR-r'
        and m['protocol_digest'] == protocol.digest and m['source'] == source
        and m['parameters'] == {'tau': parameters['tau'], 'weighting': weighting}
        and m['training_fingerprint'] == fingerprint and m['solver'] == SOLVER,
        'Logistic model/source/parameter mismatch')
    _check(_finite(m['mean_r']) and 0 <= m['mean_r'] <= 1+1e-6
           and _finite(m['std_r']) and m['std_r'] >= 0, 'invalid Logistic standardization')
    conv = m['convergence']
    _check(type(conv['iterations']) is int and 0 <= conv['iterations'] <= SOLVER['max_iterations'],
           'invalid Logistic convergence evidence')
    if m['coefficients'] is None:
        _check(_finite(m['constant']) and 0 < m['constant'] < 1 and conv == {
            'reason': 'single_class_root_constant', 'iterations': 0}, 'invalid Logistic constant branch')
    else:
        _check(m['constant'] is None and isinstance(m['coefficients'], list)
               and len(m['coefficients']) == 6 and all(_finite(v) for v in m['coefficients'])
               and set(conv) == {'reason', 'iterations', 'gradient_inf', 'objective'}
               and conv['reason'] == 'gradient_tolerance' and _finite(conv['gradient_inf'])
               and 0 <= conv['gradient_inf'] <= SOLVER['gradient_tolerance']
               and _finite(conv['objective']), 'invalid Logistic fitted coefficients/convergence')


@dataclass(frozen=True)
class VerifiedOuterModel:
    """Restricted training evidence, never a decision input."""
    model_payload: bytes
    identity_payload: bytes

    @property
    def identity(self):
        return _json(self.identity_payload)

    @property
    def record(self):
        m = _json(self.model_payload)
        _check(digest(m) == self.identity['model_digest']
               and m['model_id'] == self.identity['model_id']
               and m['parameters']['weighting'] == self.identity['weighting'],
               'changed outer model bytes/model/weighting')
        return m


@dataclass(frozen=True)
class OuterAudit:
    models: tuple[VerifiedOuterModel, ...]
    report_payload: bytes

    @property
    def report(self):
        return _json(self.report_payload)

    def model(self, *, model_id, weighting, tom_fold, outer_fold):
        found = [m for m in self.models if all(m.identity[k] == v for k, v in {
            'model_id': model_id, 'weighting': weighting, 'tom_fold': tom_fold, 'outer_fold': outer_fold}.items())]
        _check(len(found) == 1, 'missing/duplicate verified outer model')
        return found[0]


def _support(m, identity, origin, legal):
    _check(isinstance(origin, SampleOrigin) and isinstance(legal, LegalDatasetRow), 'legal context and sample origin required')
    _check(sample_id(legal.context) == legal.sample_id == origin.sample_id
           and (legal.context.game_id, legal.features.phase, legal.features.nW) == (origin.game_id, origin.phase, origin.nW)
           and (origin.tom_fold, origin.outer_fold, origin.block_digest) == (
               identity['tom_fold'], identity['outer_fold'], identity['block_digest'])
           and origin.game_id in identity['test_games'] and origin.game_id not in m['source']['game_sizes']
           and identity['test_legal_by_id'].get(legal.sample_id) == digest(legal.to_record()),
           'support sample/canonical context/test partition mismatch')
    condition(legal.features)
    base = {'sample_id': origin.sample_id, 'game_id': origin.game_id, 'model_id': m['model_id'],
        'weighting': identity['weighting'], 'tom_fold': origin.tom_fold, 'outer_fold': origin.outer_fold,
        'model_digest': identity['model_digest'], 'outer_predictions_digest': identity['outer_predictions_digest'],
        'training_source_digest': digest(m['source']), 'scope': 'restricted_training_support_not_decision_input'}
    if m['model_id'] == 'OBS-LR-r':
        return {**base, 'applicability': 'NOT_APPLICABLE', 'reason': 'continuous_Logistic_has_no_equivalence_grain'}
    node, path = m['root'], ['root']
    wolf = m['wolf_nodes'].get(str(legal.features.nW))
    if wolf:
        node = wolf; path.append('nW/'+str(legal.features.nW))
    key = f'{legal.features.phase}/{legal.features.nW}'
    state = m['state_nodes'].get(key)
    index = None
    if state:
        node = state; path.append('phase_nW/'+key)
        if state['leaves']:
            index = bisect_left(state['edges'], legal.features.r)
            node = state['leaves'][index]; path.append('r_bin/'+str(index))
    level = ('r_bin' if index is not None else 'phase_nW' if state else 'nW' if wolf else 'global')
    return {**base, 'applicability': 'APPLICABLE', 'path': path, 'level': level,
        'fallback': level in ('nW', 'global'), 'state_observed': state is not None,
        'requested_bins': m['parameters']['requested_bins'],
        'actual_bins_in_observed_state': len(state['edges'])+1 if state else None, 'bin_index': index,
        'edges': state['edges'] if state else [], 'training_stats': node['stats'],
        'support_meaning': 'saved_training_node_game_cluster_Kish_not_probability_uncertainty'}


@_guarded
def resolve_training_support(origin, legal_row, model):
    _check(isinstance(model, VerifiedOuterModel), 'verified outer model required')
    return _support(model.record, model.identity, origin, legal_row)


@_guarded
def verify_outer_artifacts(root, view, link):
    """Verify all 150 outer units; no training, bin learning, or inner-fit replay."""
    _validate_link(view, link)
    root = Path(root)
    ca, contract = _unit(root/'contract', 'validation_contract', view.identity['contract_digest'],
                         extras=('sources.json', 'protocol.json'))
    _check(digest(contract) == ca.manifest['identity']['run_identity'], 'contract identity changed')
    protocol = FrozenProtocol(read_artifact_file(ca, 'protocol.json'))
    _check(contract['origins'] == link.identity['dataset_origins']
           and {g['game_id']: g for g in link.identity['game_records']}
           == {g['game_id']: g for g in contract.get('split_record', {}).get('source_games', link.identity['game_records'])},
           'D1c/contract origin or game-index mismatch')
    legal_by_game = {}
    for r in link.legal_rows:
        legal_by_game.setdefault(r.context.game_id, []).append(r)
    origins = {s.origin.sample_id: s.origin for s in view.samples}
    probabilities = {s.origin.sample_id: s.probabilities for s in view.samples}
    targets = _json(link._targets_payload)
    models, evidence, support_counts = [], {}, Counter()
    support_digests = {}
    for k in range(5):
        for j in range(5):
            block = f'{k}-{j}'
            train, test = partition_games(contract['split_record'], tom_fold=k, outer_fold=j)
            train_rows = [r for g in train for r in legal_by_game[g]]
            test_rows = [r for g in test for r in legal_by_game[g]]
            sorted_train = sorted(train_rows, key=lambda r: (r.context.game_id, r.sample_id))
            training_values = [[r.sample_id, r.context.game_id, r.features.to_record(), targets[r.sample_id]['y']]
                               for r in train_rows]
            eq_fingerprint = digest([{'sample_id': r.sample_id, 'game_id': r.context.game_id,
                'conditions': r.features.to_record(), 'y': targets[r.sample_id]['y']} for r in sorted_train])
            lr_fingerprint = digest([[r.sample_id, r.context.game_id, r.features.to_record(), targets[r.sample_id]['y']]
                                     for r in sorted_train])
            legal_values = [[r.sample_id, r.context.game_id, r.features.to_record()] for r in test_rows]
            expected_targets = {r.sample_id: {key: targets[r.sample_id][key] for key in ('y', 'successful_empty')}
                                for r in test_rows}
            common = {'contract_digest': ca.manifest_digest, 'tom_fold': k, 'outer_fold': j}
            for wi, w in enumerate(WEIGHTINGS):
                refs = view.identity['outer_references'][block][w]
                bank = root/'blocks'/block/w
                sa, selection = _unit(bank/'selection', 'validation_selection', refs['selection'])
                candidate_refs = selection['candidate_artifact_digests']
                grid = candidates(protocol)
                _check(set(selection) == {'selected', 'candidate_game_brier', 'candidate_artifact_digests'}
                    and len(candidate_refs) == 3*len(grid) and len(set(candidate_refs)) == len(candidate_refs),
                    'selection candidate inventory mismatch')
                for h in candidate_refs:
                    _sha(h)
                _check(sa.manifest['identity'] == {**common, 'weighting': w, 'candidate_artifact_digests': candidate_refs},
                       'selection fold/weight/source mismatch')
                table = selection['candidate_game_brier']
                _check(len(table) == len(grid), 'selection grid size mismatch')
                for entry, (name, parameters) in zip(table, grid, strict=True):
                    _check(set(entry) == {'model', 'parameters', 'games'} and entry['model'] == name
                        and entry['parameters'] == parameters and set(entry['games']) == set(train)
                        and all(_finite(v) and 0 <= v <= 1 for v in entry['games'].values()),
                        'selection candidate grid/game/score mismatch')
                chosen = _select(table, SimpleNamespace(protocol=protocol, sizes=contract['game_sizes']), train, w)
                _check(selection['selected'] == chosen, 'recorded selection violates frozen tie/grid rule')
                formal = view.identity['scope'] == 'formal_development_validation'
                source = {'scope': 'formal_training_partition' if formal else 'controlled_fixture',
                    'protocol_digest': protocol.digest, 'dataset_digest': contract['dataset_digest'] if formal else None,
                    'split_manifest_digest': contract['split_digest'] if formal else None,
                    'tom_fold': k if formal else None, 'outer_fold': j if formal else None, 'inner_fold': None,
                    'game_sizes': {g: contract['game_sizes'][g] for g in train}, 'origins': contract['origins'],
                    'implementation_digest': contract['source_snapshot_digest']}
                pa, outer = _unit(bank/'outer_predictions', 'validation_outer_predictions', refs['predictions'], identity={
                    **common, 'weighting': w, 'selection_digest': sa.manifest_digest, 'source': source,
                    'training_digest': digest(training_values), 'test_legal_digest': digest(legal_values)})
                _check(set(outer) == {'models', 'predictions', 'bin_audit'} and set(outer['models']) == set(MODELS)
                       and set(outer['predictions']) == set(MODELS) and set(outer['bin_audit']) == {'OBS-EQ0', 'OBS-EQ1'},
                       'outer model inventory mismatch')
                for name in ('OBS-EQ0', 'OBS-EQ1'):
                    m = outer['models'][name]
                    validate_model(FittedModel(canonical_json_bytes(m)), protocol)
                    _check(m['model_id'] == name and m['source'] == source
                        and m['training_fingerprint'] == eq_fingerprint
                        and m['parameters'] == {'requested_bins': chosen[name]['bins'],
                            'lambda': chosen[name]['smoothing'], 'weighting': w}, 'EQ model/weight/source/fingerprint mismatch')
                    audits = outer['bin_audit'][name]
                    _check(set(audits) == set(m['state_nodes']), 'bin audit state mismatch')
                    for state_key, audit in audits.items():
                        _check(set(audit) == {'initial_edges', 'final_edges', 'merges'}
                            and audit['final_edges'] == m['state_nodes'][state_key]['edges']
                            and audit['initial_edges'] == sorted(set(audit['initial_edges']))
                            and all(_finite(e) and 0 <= e <= 1+1e-6 for e in audit['initial_edges']), 'bin audit edges mismatch')
                        remaining = list(audit['initial_edges'])
                        for merge in audit['merges']:
                            _check(set(merge) == {'failing_bin', 'games', 'kish', 'removed_edge'}
                                and merge['removed_edge'] in remaining and type(merge['failing_bin']) is int
                                and 0 <= merge['failing_bin'] <= len(remaining)
                                and type(merge['games']) is int and merge['games'] >= 0
                                and _finite(merge['kish']) and merge['kish'] >= 0, 'invalid saved bin merge trace')
                            remaining.remove(merge['removed_edge'])
                        _check(remaining == audit['final_edges'], 'saved merge trace/final partition mismatch')
                _check(outer['models']['OBS-EF'] == empirical_model(FittedModel(canonical_json_bytes(outer['models']['OBS-EQ1']))),
                       'EF not bound to original EQ1 bins/stats/weights')
                _check_logistic(outer['models']['OBS-LR-r'], source, protocol, chosen['OBS-LR-r'], w, lr_fingerprint)
                ea, evaluation = _unit(bank/'evaluation', 'validation_outer_evaluation', refs['evaluation'], identity={
                    **common, 'weighting': w, 'predictions_digest': pa.manifest_digest,
                    'test_target_digest': digest(expected_targets)})
                _check(set(evaluation) == {'scores', 'test_target_digest', 'predictions_digest'}
                    and evaluation['test_target_digest'] == digest(expected_targets)
                    and evaluation['predictions_digest'] == pa.manifest_digest
                    and set(evaluation['scores']) == set(MODELS), 'evaluation target/prediction identity mismatch')
                scores_rows = [SimpleNamespace(sample_id=r.sample_id, game_id=r.context.game_id,
                    y=targets[r.sample_id]['y']) for r in test_rows]
                for mi, name in enumerate(MODELS):
                    m = outer['models'][name]
                    raw = outer['predictions'][name]
                    _check(len(raw) == len(test_rows) and len({r['sample_id'] for r in raw}) == len(raw),
                           'outer missing/duplicate predictions')
                    pred = {r['sample_id']: r for r in raw}
                    _check(set(pred) == {r.sample_id for r in test_rows}, 'outer cross-game/missing sample IDs')
                    inferred = predict_many(m, [r.features for r in test_rows])  # Saved model only, no fit.
                    model_identity = {**common, 'run_identity_digest': view.identity_digest,
                        'dataset_digest': link.identity['dataset_digest'], 'model_id': name, 'weighting': w,
                        'model_digest': digest(m), 'outer_predictions_digest': pa.manifest_digest,
                        'selection_digest': sa.manifest_digest, 'evaluation_digest': ea.manifest_digest,
                        'test_games': list(test), 'block_digest': view.identity['block_digests'][block],
                        'test_legal_by_id': {r.sample_id: digest(r.to_record()) for r in test_rows}}
                    saved_support = []
                    for r, detail in zip(test_rows, inferred, strict=True):
                        p = pred[r.sample_id]
                        _check(set(p) == {'sample_id', 'game_id', 'phase', 'nW', 'probability', 'level'}
                            and type(p['nW']) is int
                            and (p['game_id'], p['phase'], p['nW']) == (r.context.game_id, r.features.phase, r.features.nW)
                            and _finite(p['probability']) and 0 <= p['probability'] <= 1
                            and p['probability'] == probabilities[r.sample_id][wi*len(MODELS)+mi]
                            and math.isclose(p['probability'], detail['probability'], rel_tol=0,
                                             abs_tol=INFERENCE_ABSOLUTE_TOLERANCE)
                            and p['level'] == detail['level'], 'outer prediction/model/sealed probability mismatch')
                        support = _support(m, model_identity, origins[r.sample_id], r)
                        if name != 'OBS-LR-r':
                            _check(support['level'] == p['level'], 'support path/prediction level mismatch')
                        support_counts[w+'/'+name+'/'+support.get('level', 'NOT_APPLICABLE')] += 1
                        saved_support.append(support)
                    _check(_close_tree(evaluation['scores'][name], score_predictions(scores_rows,
                        {sid: r['probability'] for sid, r in pred.items()})), 'saved D2 evaluation score mismatch')
                    key = block+'/'+w+'/'+name
                    support_digests[key] = digest(sorted(saved_support, key=lambda s: s['sample_id']))
                    models.append(VerifiedOuterModel(canonical_json_bytes(m), canonical_json_bytes(model_identity)))
                evidence[block+'/'+w] = {**refs, 'training_games': len(train), 'test_games': len(test),
                    'training_digest': digest(training_values), 'test_legal_digest': digest(legal_values),
                    'test_target_digest': digest(expected_targets)}
    report = {'schema_version': VERSION, 'run_identity_digest': view.identity_digest,
        'complete_digest': view.identity['complete_digest'], 'dataset_digest': link.identity['dataset_digest'],
        'D1c_legal_rows_digest': link.identity['legal_rows_digest'], 'independent_D1c_labels_verified': True,
        'outer_models_verified': True, 'saved_model_inference_verified': True,
        'recorded_selection_rule_verified': True, 'inner_candidate_fits_replayed': False,
        'original_canonical_reconstruction': False, 'full_run_replayed': False,
        'blocks': 25, 'outer_units': len(evidence)*3, 'fitted_models': len(models),
        'samples': len(link.legal_rows), 'probability_bindings_verified': len(link.legal_rows)*8,
        'support_counts': dict(sorted(support_counts.items())), 'support_digests_by_model': support_digests,
        'outer_evidence': evidence, 'inference_absolute_tolerance': INFERENCE_ABSOLUTE_TOLERANCE,
        'training_support_source': 'saved_nodes_only_no_rebin_or_refit_or_test_label_support',
        'research_scope': view.identity['research_scope'], 'known_source_limitations': view.identity['known_source_limitations'],
        'd3_regions_or_six_losses_computed': False, 'loss_or_tie_policy_selected': False}
    return OuterAudit(tuple(models), canonical_json_bytes(report))
