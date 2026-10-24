"""Explicit offline fixture compiler. Never imported by Shadow or online readers.

No fitting or model selection. A D2 export requires a previously verified outer
audit and externally pinned audit/model digests. Only inference fields leave it.
"""
from pathlib import Path

from werewolf.artifact_io import canonical_json_bytes, publish_artifact
from werewolf.observer_dtrs.d2_protocol import frozen_protocol
from werewolf.observer_dtrs.estimation import FittedModel, validate_model
from werewolf.observer_dtrs.online_context import digest
from werewolf.observer_dtrs.online_model import TYPE, VERSION, TomIdentity, validate_payload, sha


def _compile(model):
    if model['model_id'] == 'OBS-LR-r':
        return {k: model[k] for k in ('mean_r', 'std_r', 'coefficients', 'constant')}
    return {'root': model['root']['probability'],
            'wolf_nodes': {k: n['probability'] for k, n in model['wolf_nodes'].items()},
            'state_nodes': {k: {'probability': n['probability'], 'edges': n['edges'],
                'leaves': [leaf['probability'] for leaf in n['leaves']]}
                for k, n in model['state_nodes'].items()}}


def _publish(destination, model, *, scope, tom, evidence_digest, selection_rule_digest):
    if not isinstance(tom, TomIdentity):
        raise TypeError('explicit compatible ToM identity required')
    for h in (evidence_digest, selection_rule_digest):
        sha(h)
    record = {'schema_version': VERSION, 'scope': scope, 'model_id': model['model_id'],
        'weighting': model['parameters']['weighting'], 'tom': tom.to_record(),
        'source_model_digest': digest(model), 'source_evidence_digest': evidence_digest,
        'selection_rule_digest': selection_rule_digest, 'inference': _compile(model)}
    validate_payload(record)
    destination = Path(destination)
    if any((p/'manifest.json').exists() for p in destination.parents):
        raise ValueError('cannot publish inside an existing immutable artifact')
    return publish_artifact(destination, manifest_fields={'artifact_type': TYPE, 'schema_version': VERSION,
        'scope': scope, 'inference_digest': digest(record)}, files={'inference.json': canonical_json_bytes(record)})


def publish_controlled_estimator(destination, model, *, tom, selection_rule_digest):
    """Explicitly artificial EQ or secondary-model fixture, never a formal fit."""
    m = model.record if isinstance(model, FittedModel) else model
    if m['source']['scope'] != 'controlled_fixture' or tom.family != 'controlled_fixture':
        raise ValueError('controlled exporter refuses real model/source claims')
    if m['model_id'] in ('OBS-EQ0', 'OBS-EQ1'):
        validate_model(FittedModel(canonical_json_bytes(m)), frozen_protocol())
    return _publish(destination, m, scope='controlled_fixture', tom=tom,
                    evidence_digest=digest(m), selection_rule_digest=selection_rule_digest)


def export_verified_outer_fixture(destination, *, audit, expected_audit_digest,
                                  expected_model_digest, model_id, weighting,
                                  tom_fold, outer_fold, max_seq_len, selection_rule_digest,
                                  controlled_tom=None):
    # This import and its restricted dependency closure are OFFLINE ONLY.
    from werewolf.observer_dtrs.training_audit_offline import OuterAudit
    if (not isinstance(audit, OuterAudit) or digest(audit.report) != expected_audit_digest
            or any(audit.report.get(k) is not True for k in ('outer_models_verified',
                'saved_model_inference_verified', 'recorded_selection_rule_verified', 'independent_D1c_labels_verified'))):
        raise ValueError('explicit verified outer-audit trust anchor mismatch')
    sha(expected_model_digest)
    item = audit.model(model_id=model_id, weighting=weighting, tom_fold=tom_fold, outer_fold=outer_fold)
    m = item.record
    if digest(m) != expected_model_digest:
        raise ValueError('outer model trust anchor mismatch')
    identity = item.identity
    refs = audit.report['outer_evidence'][f'{tom_fold}-{outer_fold}/{weighting}']
    if (identity['outer_predictions_digest'] != refs['predictions']
            or identity['selection_digest'] != refs['selection']
            or identity['evaluation_digest'] != refs['evaluation']
            or identity['dataset_digest'] != audit.report['dataset_digest']):
        raise ValueError('model is not the declared audited D2 outer partition')
    if m['source']['scope'] == 'controlled_fixture':
        if not isinstance(controlled_tom, TomIdentity) or controlled_tom.family != 'controlled_fixture':
            raise ValueError('artificial audit requires explicitly artificial ToM')
        return _publish(destination, m, scope='controlled_fixture', tom=controlled_tom,
                        evidence_digest=expected_audit_digest, selection_rule_digest=selection_rule_digest)
    if (controlled_tom is not None or m['source']['scope'] != 'formal_training_partition'
            or (m['source']['tom_fold'], m['source']['outer_fold'], m['source']['inner_fold']) != (tom_fold, outer_fold, None)
            or m['source']['dataset_digest'] != audit.report['dataset_digest']):
        raise ValueError('D2 model source partition mismatch')
    origins = [s for s in m['source']['origins']['oof'] if s['fold_index'] == tom_fold]
    if len(origins) != 1 or origins[0]['temporal_condition'] != 'explicit_day_phase' or origins[0]['architecture'] != 'qwen3':
        raise ValueError('missing/ambiguous compatible ToM source')
    q = origins[0]
    tom = TomIdentity('d2_oof', q['training_revision'], q['training_execution_digest'],
        q['training_seal_digest'], q['terminal_digest'], q['checkpoint_digest'], tom_fold, max_seq_len)
    return _publish(destination, m, scope='verified_d2_fixture', tom=tom,
                    evidence_digest=expected_audit_digest, selection_rule_digest=selection_rule_digest)
