"""Deployment compilation is offline; online reading never opens supervision."""
from dataclasses import replace
import json
import math
import random

import pytest

from tests.observer_dtrs.test_estimation import fit, features
from tests.observer_dtrs.test_shadow import tom_identity, ArtificialTom, deployed
from werewolf.artifact_io import ArtifactValidationError, canonical_json_bytes, publish_artifact
from werewolf.observer_dtrs.baselines import fit_logistic, empirical_model, predict_many
from werewolf.observer_dtrs.d2_protocol import frozen_protocol
from werewolf.observer_dtrs.deployment_offline import publish_controlled_estimator, export_verified_outer_fixture
from werewolf.observer_dtrs.online_context import digest
from werewolf.observer_dtrs.online_model import VERSION, TYPE, TomIdentity, read_deployment
from werewolf.observer_dtrs.shadow import ShadowEngine
from werewolf.observer_dtrs.validation_data import controlled_fixture, training_rows


@pytest.fixture(scope='module')
def fitted_models():
    p = frozen_protocol();data = controlled_fixture(p)
    rows = training_rows(data, tuple(data.sizes))
    eq0 = fit(rows, p, model='OBS-EQ0', bins=1)
    eq1 = fit(rows, p, model='OBS-EQ1', bins=3)
    lr = fit_logistic(rows, protocol=p, tau=.1, weighting='game_equal', source=eq1.record['source'])
    return {'OBS-EQ0': eq0, 'OBS-EQ1': eq1, 'OBS-EF': empirical_model(eq1), 'OBS-LR-r': lr}


@pytest.mark.parametrize('name', ['OBS-EQ0', 'OBS-EQ1', 'OBS-EF', 'OBS-LR-r'])
def test_compiled_predictions_exactly_match_original_models(tmp_path, fitted_models, name):
    m = fitted_models[name]
    a = publish_controlled_estimator(tmp_path/'deployment', m, tom=tom_identity(), selection_rule_digest='6'*64)
    online = read_deployment(a.path, expected_manifest_digest=a.manifest_digest, expected_tom=tom_identity())
    record = m.record if hasattr(m, 'record') else m
    r_values = {0., .1, .3, .6, .9, 1.}
    for node in record.get('state_nodes', {}).values():
        for edge in node['edges']:
            r_values.update(x for x in (edge, math.nextafter(edge, 0.), math.nextafter(edge, 1.)) if 0 <= x <= 1)
    rng = random.Random(20261017)
    r_values.update(rng.random() for _ in range(30))
    legal = [features(r, phase, n) for r in sorted(r_values)
             for phase in ('discussion', 'pk_discussion') for n in (1, 2)]
    assert [online.predict(f) for f in legal] == predict_many(m, legal)
    raw = (a.path/'inference.json').read_text()
    assert not set(json.loads(raw)) & {'stats', 'targets', 'support', 'game_sizes', 'source'}
    assert 'positive_mass' not in raw and 'training_fingerprint' not in raw
    # Fallback remains the saved observed parent, never a BND or online refit.
    assert online.predict(features(.5))['level'] in ('phase_nW', 'r_bin', 'logistic')
    assert 'probability' in online.predict(features(.5))


@pytest.mark.parametrize('field', ['weights_digest', 'terminal_digest', 'seal_digest', 'max_seq_len'])
def test_all_tom_source_fields_are_bound(tmp_path, deployed, field):
    wrong = replace(tom_identity(), **{field: 512 if field == 'max_seq_len' else 'f'*64})
    with pytest.raises(ValueError, match='matching'):
        ShadowEngine(tom_provider=ArtificialTom(wrong), probability_provider=deployed,
            expected_deployment_digest=deployed.manifest_digest, journal_root=tmp_path/'refused')
    assert not (tmp_path/'refused').exists()


def test_oof_and_final_are_never_implicitly_interchangeable(tmp_path, deployed):
    final = TomIdentity('final_tom', 'a'*40, '1'*64, '2'*64, '3'*64, '4'*64, None)
    oof = TomIdentity('d2_oof', 'a'*40, '1'*64, '2'*64, '3'*64, '4'*64, 0)
    assert final != oof
    for wrong in (final, oof):
        with pytest.raises(ValueError):
            read_deployment(tmp_path/'model', expected_manifest_digest=deployed.manifest_digest, expected_tom=wrong)
    # Even a fully rehashed payload cannot masquerade as a final deployment.
    r = deployed.record;r['tom'] = final.to_record()
    a = publish_artifact(tmp_path/'fake-final',
        manifest_fields={'artifact_type': TYPE, 'schema_version': VERSION, 'scope': r['scope'], 'inference_digest': digest(r)},
        files={'inference.json': canonical_json_bytes(r)})
    with pytest.raises(ValueError, match='source mismatch'):
        read_deployment(a.path, expected_manifest_digest=a.manifest_digest, expected_tom=final)


@pytest.mark.parametrize('kind', ['nan', 'edges', 'extra_target', 'scope', 'manifest', 'missing'])
def test_corrupt_or_rehashed_deployment_rejected(tmp_path, deployed, kind):
    path = tmp_path/'model'
    if kind == 'missing':
        (path/'inference.json').unlink()
        expected = deployed.manifest_digest
    elif kind == 'manifest':
        expected = 'f'*64
    else:
        record = deployed.record
        if kind == 'nan':
            # Non-JSON finite probability violation; canonical JSON itself refuses NaN.
            record['inference']['root'] = -1.
        elif kind == 'edges':
            node = record['inference']['state_nodes']['discussion/2']
            node.update(edges=[.8, .2], leaves=[.1, .5, .9])
        elif kind == 'extra_target': record['Y'] = 1
        else: record['scope'] = 'formal_deployment'
        path = tmp_path/'rehashed-deployment'
        a = publish_artifact(path,
            manifest_fields={'artifact_type': TYPE, 'schema_version': VERSION,
                             'scope': record['scope'], 'inference_digest': digest(record)},
            files={'inference.json': canonical_json_bytes(record)})
        expected = a.manifest_digest
    with pytest.raises((ValueError, ArtifactValidationError, OSError)):
        read_deployment(path, expected_manifest_digest=expected, expected_tom=tom_identity())


def test_exporter_requires_external_audit_and_model_anchors(tmp_path):
    with pytest.raises(ValueError):
        export_verified_outer_fixture(tmp_path/'blocked', audit=None, expected_audit_digest='a'*64,
            expected_model_digest='b'*64, model_id='OBS-EQ1', weighting='game_equal',
            tom_fold=0, outer_fold=0, max_seq_len=1024, selection_rule_digest='c'*64)
    assert not (tmp_path/'blocked').exists()


def test_payload_cannot_be_swapped_behind_manifest_anchor(deployed):
    r = deployed.record
    r['inference']['root'] = .123
    with pytest.raises(ValueError, match='pinned manifest'):
        replace(deployed, payload=canonical_json_bytes(r))


def test_fallback_uses_saved_parent_and_never_support_threshold(tmp_path):
    from tests.observer_dtrs.test_estimation import row
    m = fit([row('a', 0), row('b', 1)], frozen_protocol())
    a = publish_controlled_estimator(tmp_path/'fallback', m, tom=tom_identity(), selection_rule_digest='6'*64)
    online = read_deployment(a.path, expected_manifest_digest=a.manifest_digest, expected_tom=tom_identity())
    assert online.predict(features(.5, 'pk_discussion', 2))['level'] == 'nW'
    assert online.predict(features(.5, 'pk_discussion', 1))['level'] == 'global'
    assert online.predict(features(.5, 'pk_discussion', 1)) == predict_many(m, [features(.5, 'pk_discussion', 1)])[0]


def test_audited_controlled_outer_export_for_all_four_models(tmp_path):
    # Real old audit functions, but the inputs and all fits are deliberately synthetic.
    from tests.observer_dtrs.test_training_audit import _make_package
    from werewolf.observer_dtrs.sealed_d2_offline import read_controlled_d2
    from werewolf.observer_dtrs.training_audit_offline import verify_outer_artifacts
    run = tmp_path/'artificial-old-run'
    anchors, link = _make_package(run)
    view = read_controlled_d2(run, anchors=anchors)
    audit = verify_outer_artifacts(run, view, link)
    for name in ('OBS-EQ0', 'OBS-EQ1', 'OBS-LR-r', 'OBS-EF'):
        model = audit.model(model_id=name, weighting='game_equal', tom_fold=0, outer_fold=0)
        args = dict(audit=audit, expected_audit_digest=digest(audit.report),
            expected_model_digest=model.identity['model_digest'], model_id=name, weighting='game_equal',
            tom_fold=0, outer_fold=0, max_seq_len=1024, selection_rule_digest='e'*64, controlled_tom=tom_identity())
        a = export_verified_outer_fixture(tmp_path/name, **args)
        online = read_deployment(a.path, expected_manifest_digest=a.manifest_digest, expected_tom=tom_identity())
        assert online.record['scope'] == 'controlled_fixture'
        assert online.record['source_evidence_digest'] == digest(audit.report)
        points = [features(r, phase, n) for r in (0., .1, .3, .9, 1.)
                  for phase in ('discussion', 'pk_discussion') for n in (1, 2)]
        assert [online.predict(f) for f in points] == predict_many(model.record, points)
        assert not set(online.record) & {'test_games', 'test_legal_by_id', 'training_fingerprint', 'targets'}
    args['expected_model_digest'] = 'f'*64
    with pytest.raises(ValueError, match='model trust'):
        export_verified_outer_fixture(tmp_path/'wrong-model', **args)
    assert not (tmp_path/'wrong-model').exists()
