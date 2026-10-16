"""Controlled D1c gates and saved-model audits; never reads formal research data."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace
import shutil

import pytest

from tests.observer_dtrs.test_oof import oof_factory
from tests.observer_dtrs.test_dataset import published, republish
from tests.observer_dtrs.test_sealed_d2 import rewrite
from werewolf.artifact_io import ArtifactValidationError, canonical_json_bytes
from werewolf.observer_dtrs.baselines import fit_logistic, empirical_model, predict_many, bin_audit
from werewolf.observer_dtrs.context import ObserverContext
from werewolf.observer_dtrs.dataset import LegalDatasetRow, open_dataset, sample_id
from werewolf.observer_dtrs.dataset_offline import open_supervision
from werewolf.observer_dtrs.d2_protocol import frozen_protocol
from werewolf.observer_dtrs.estimation import FittedModel, fit_model
from werewolf.observer_dtrs.features import observer_features
from werewolf.observer_dtrs.probabilities import ProbabilityRow
from werewolf.observer_dtrs.nested_validation import candidates, _select
from werewolf.observer_dtrs.scoring import score_predictions
from werewolf.observer_dtrs.sealed_d2_offline import (
    SealedRun, SampleOrigin, LegalSample, SourceAnchors, read_controlled_d2, open_sealed_targets,
)
from werewolf.observer_dtrs.splits import partition_games
from werewolf.observer_dtrs.statistical_analysis import MODELS, WEIGHTINGS, analyze
from werewolf.observer_dtrs.training_audit_offline import (
    D1cLink, VerifiedOuterModel, crosscheck_d1c, open_d1c_targets,
    verify_outer_artifacts, resolve_training_support,
)
from werewolf.observer_dtrs.validation_artifacts import digest, publish_unit
from werewolf.observer_dtrs.validation_data import ControlledData, PredictionInput, controlled_fixture, fit_source, training_rows


def sealed_dataset(artifact):
    """Small explicit sealed view for official D1c-reader unit tests, not a formal run."""
    legal = open_dataset(artifact.path)
    offline = open_supervision(legal)
    samples, targets = [], {}
    for r in offline:
        c = r.legal.context
        o = SampleOrigin(r.legal.sample_id, c.game_id, c.phase, c.wolf_count, r.q_source.fold_index, 0, 'b'*64)
        samples.append(LegalSample(o, (.5,)*8))
        targets[o.sample_id] = {'game_id': c.game_id, 'y': r.supervision.event_label_y,
                              'successful_empty': not r.supervision.raw_suspicion_support}
    i = {'scope': 'controlled_fixture', 'dataset_digest': artifact.manifest_digest,
         'game_sizes': {g['game_id']: g['row_count'] for g in artifact.manifest['games']},
         'sealed_targets_digest': digest(targets)}
    return SealedRun(canonical_json_bytes(i), tuple(samples), canonical_json_bytes(targets))


def test_official_reader_canonical_labels_and_gated_access(published):
    artifact, _ = published()
    view = sealed_dataset(artifact)
    link = crosscheck_d1c(view, artifact.path, expected_dataset_digest=artifact.manifest_digest)
    assert link.identity['samples'] == 25
    assert link.identity['independent_D1c_labels_verified'] is True
    assert link.identity['original_canonical_reconstruction'] is False
    assert open_d1c_targets(view, link).labels == open_sealed_targets(view).labels
    assert all(not hasattr(r, 'supervision') and not hasattr(r, 'y') for r in link.legal_rows)
    changed = json.loads(link._targets_payload)
    sid = next(iter(changed)); changed[sid]['y'] = 1-changed[sid]['y']
    with pytest.raises(ArtifactValidationError, match='target link'):
        open_d1c_targets(view, replace(link, _targets_payload=canonical_json_bytes(changed)))
    with pytest.raises(ArtifactValidationError, match='legal link'):
        open_d1c_targets(view, replace(link, legal_rows=link.legal_rows[:-1]))


@pytest.mark.parametrize('kind', ['label', 'empty', 'fold', 'missing', 'cross_game', 'anchor'])
def test_sealed_vs_independent_d1c_mismatch_stops(published, kind):
    artifact, _ = published()
    view = sealed_dataset(artifact)
    sid = view.samples[0].origin.sample_id
    if kind in ('label', 'empty'):
        targets = json.loads(view._targets_payload)
        key = 'y' if kind == 'label' else 'successful_empty'
        targets[sid][key] = 1-targets[sid][key] if kind == 'label' else not targets[sid][key]
        i = view.identity; i['sealed_targets_digest'] = digest(targets)
        view = replace(view, identity_payload=canonical_json_bytes(i), _targets_payload=canonical_json_bytes(targets))
    elif kind in ('fold', 'cross_game'):
        first = view.samples[0]
        o = replace(first.origin, **({'tom_fold': (first.origin.tom_fold+1)%5} if kind == 'fold' else {'game_id': 'wrong-game'}))
        view = replace(view, samples=(replace(first, origin=o), *view.samples[1:]))
    elif kind == 'missing':
        view = replace(view, samples=view.samples[:-1])
    with pytest.raises(ArtifactValidationError):
        crosscheck_d1c(view, artifact.path, expected_dataset_digest='f'*64 if kind=='anchor' else artifact.manifest_digest)


@pytest.mark.parametrize('part', ['legal/games', 'restricted/supervision', 'restricted/provenance'])
def test_rehashed_d1c_corruption_uses_official_validator(published, tmp_path, part):
    artifact, _ = published()
    game = artifact.manifest['games'][0]['game_id']
    bad = republish(artifact, tmp_path/'bad', filename=f'{part}/{game}.jsonl',
                    edit=lambda records: records.append(deepcopy(records[0])))
    view = sealed_dataset(artifact)
    identity = view.identity; identity['dataset_digest'] = bad.manifest_digest
    view = replace(view, identity_payload=canonical_json_bytes(identity))
    with pytest.raises(ArtifactValidationError):
        crosscheck_d1c(view, bad.path, expected_dataset_digest=bad.manifest_digest)


def _make_package(root):
    protocol = frozen_protocol()
    original = controlled_fixture(protocol)
    legal, inputs, targets = [], [], {}
    for number, old in enumerate(original.rows):
        n = old.features.nW
        c = ObserverContext(old.game_id, 'PRE-'+old.sample_id, digest(['prefix', old.sample_id]), number,
            'player1', 'player3', 1, old.features.phase,
            ('player1', 'player2', 'player3') if n==2 else ('player1', 'player3'), ('player1', 'player2'))
        q = ProbabilityRow(c.game_id, c.boundary_id, c.prefix_digest, c.observer_id,
            (old.features.s, old.features.m, 0., 0., 0., 0., 1-old.features.r))
        f = observer_features(c, q); sid = sample_id(c)
        r = LegalDatasetRow(sid, c, q, f)
        legal.append(r); inputs.append(PredictionInput(sid, c.game_id, f))
        targets[sid] = original.target_values[old.sample_id]
    data = ControlledData(inputs, targets, protocol=protocol)
    sources = {'head': 'a'*40, 'files': {'synthetic.txt': 'controlled'}, 'runtime': {}}
    c = {'scope': data.scope, 'protocol_digest': protocol.digest, 'dataset_digest': data.dataset_digest,
        'split_digest': data.split_digest, 'source_snapshot_digest': digest(sources), 'origins': data.origins,
        'game_sizes': data.sizes, 'game_folds': data.folds, 'models': list(MODELS), 'weightings': list(WEIGHTINGS),
        'split_record': data.split, 'sample_ids_digest_by_game': {g: digest([r.sample_id for r in data.legal((g,))]) for g in data.sizes},
        'known_source_limitations': protocol.record['known_source_limitations'], 'research_scope': protocol.record['research_scope']}
    ca = publish_unit(root/'contract', 'validation_contract', {'run_identity': digest(c)}, c,
                      {'sources.json': canonical_json_bytes(sources), 'protocol.json': protocol.payload})
    refs, blocks, all_rows = {}, {}, []
    for k in range(5):
        for j in range(5):
            train, test = partition_games(data.split, tom_fold=k, outer_fold=j)
            fit_rows, test_rows = training_rows(data, train), data.legal(test)
            target = data.targets(test)
            common = {'contract_digest': ca.manifest_digest, 'tom_fold': k, 'outer_fold': j}
            references, maps = {}, {}
            for w in WEIGHTINGS:
                bank = root/'blocks'/f'{k}-{j}'/w
                table = [{'model': m, 'parameters': p, 'games': {g:.2 for g in train}} for m,p in candidates(protocol)]
                chosen = _select(table, data, train, w)
                candidate_refs = [digest(['synthetic-inner',k,j,w,h,n]) for h in range(3) for n in range(len(table))]
                sa = publish_unit(bank/'selection','validation_selection',
                    {**common,'weighting':w,'candidate_artifact_digests':candidate_refs},
                    {'selected':chosen,'candidate_game_brier':table,'candidate_artifact_digests':candidate_refs})
                source = fit_source(data,train,tom_fold=k,outer_fold=j,inner_fold=None,implementation_digest=digest(sources))
                models = {name:fit_model(fit_rows, protocol=protocol, model_id=name, weighting=w,
                    source=source, **chosen[name]).record for name in ('OBS-EQ0','OBS-EQ1')}
                models['OBS-LR-r'] = fit_logistic(fit_rows,protocol=protocol,weighting=w,source=source,**chosen['OBS-LR-r'])
                models['OBS-EF'] = empirical_model(FittedModel(canonical_json_bytes(models['OBS-EQ1'])))
                predictions = {name:[{'sample_id':r.sample_id,'game_id':r.game_id,'phase':r.features.phase,'nW':r.features.nW,**d}
                    for r,d in zip(test_rows,predict_many(m,[r.features for r in test_rows]),strict=True)] for name,m in models.items()}
                pa = publish_unit(bank/'outer_predictions','validation_outer_predictions',
                    {**common,'weighting':w,'selection_digest':sa.manifest_digest,'source':source,
                     'training_digest':digest([[r.sample_id,r.game_id,r.features.to_record(),r.y] for r in fit_rows]),
                     'test_legal_digest':digest([r.record() for r in test_rows])},
                    {'models':models,'predictions':predictions,'bin_audit':{name:bin_audit(FittedModel(canonical_json_bytes(models[name])),fit_rows,protocol)
                                                                        for name in ('OBS-EQ0','OBS-EQ1')}})
                scores_rows = [SimpleNamespace(sample_id=r.sample_id,game_id=r.game_id,y=target[r.sample_id]['y']) for r in test_rows]
                ea = publish_unit(bank/'evaluation','validation_outer_evaluation',
                    {**common,'weighting':w,'predictions_digest':pa.manifest_digest,'test_target_digest':digest(target)},
                    {'scores':{name:score_predictions(scores_rows,{r['sample_id']:r['probability'] for r in pred}) for name,pred in predictions.items()},
                     'test_target_digest':digest(target),'predictions_digest':pa.manifest_digest})
                references[w] = {'selection':sa.manifest_digest,'predictions':pa.manifest_digest,'evaluation':ea.manifest_digest}
                maps[w] = {name:{r['sample_id']:r['probability'] for r in p} for name,p in predictions.items()}
            rows = [{'sample_id':r.sample_id,'game_id':r.game_id,'phase':r.features.phase,'nW':r.features.nW,
                **target[r.sample_id], 'predictions':{w:{m:maps[w][m][r.sample_id] for m in MODELS} for w in WEIGHTINGS}} for r in test_rows]
            ba = publish_unit(root/'blocks'/f'{k}-{j}'/'complete','validation_block_complete',common,
                {'status':'PAIRED_BLOCK_COMPLETE','references':references,'rows':rows,'train_games':train,'test_games':test})
            refs[f'{k}-{j}'] = ba.manifest_digest; blocks[k,j] = test; all_rows.extend(rows)
    summary, indices = analyze(all_rows,blocks=blocks,protocol=protocol,scope=data.scope)
    final = publish_unit(root/'complete','validation_complete',{'contract_digest':ca.manifest_digest,'block_digests':refs},
        {'status':'ALL_BLOCKS_COMPLETE','block_digests':refs,'summary':summary},{'bootstrap_indices.int64le':indices})
    anchors = SourceAnchors(final.manifest_digest,sources['head'],data.dataset_digest,data.split_digest)
    view = read_controlled_d2(root,anchors=anchors)
    payload = canonical_json_bytes({sid:{'game_id':r.game_id,**targets[sid]} for sid,r in {r.sample_id:r for r in inputs}.items()})
    from werewolf.observer_dtrs.training_audit_offline import _legal_digest
    link = D1cLink(tuple(legal),canonical_json_bytes({'run_identity_digest':view.identity_digest,
        'dataset_digest':data.dataset_digest,'targets_digest':digest(json.loads(payload)),
        'legal_rows_digest':_legal_digest(legal),'dataset_origins':data.origins,'game_records':[]}),payload)
    return anchors, link


@pytest.fixture(scope='module')
def outer_package(tmp_path_factory):
    root = tmp_path_factory.mktemp('saved-outer')
    anchors, link = _make_package(root)
    return root, anchors, link


def forbid_fitting(monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError('audit must not fit/rebin/classify')
    import werewolf.observer_dtrs.estimation as eq
    import werewolf.observer_dtrs.baselines as base
    import werewolf.observer_dtrs.decision as dec
    for module,name in ((eq,'fit_model'),(eq,'train_edges'),(base,'fit_logistic'),(base,'train_edges'),(dec,'decide')):
        monkeypatch.setattr(module,name,forbidden)


def test_all_saved_models_and_support_no_fitting(outer_package, monkeypatch):
    root,anchors,link = outer_package
    view = read_controlled_d2(root,anchors=anchors)
    forbid_fitting(monkeypatch)
    audit = verify_outer_artifacts(root,view,link)
    assert audit.report['outer_units'] == 150 and len(audit.models) == 200
    assert audit.report['probability_bindings_verified'] == 1200
    assert audit.report['inner_candidate_fits_replayed'] is False
    assert audit.report['d3_regions_or_six_losses_computed'] is False
    row = link.legal_rows[0]
    origin = next(s.origin for s in view.samples if s.origin.sample_id == row.sample_id)
    for w in WEIGHTINGS:
        for name in MODELS:
            model = audit.model(model_id=name,weighting=w,tom_fold=origin.tom_fold,outer_fold=origin.outer_fold)
            support = resolve_training_support(origin,row,model)
            if name == 'OBS-LR-r':
                assert support['applicability'] == 'NOT_APPLICABLE' and 'training_stats' not in support
            else:
                assert support['training_stats']['rows'] > 0
                assert 'y' not in support and 'successful_empty' not in support
            with pytest.raises(ArtifactValidationError,match='partition'):
                resolve_training_support(replace(origin,outer_fold=(origin.outer_fold+1)%5),row,model)


def _reanchor(root, anchors, w, kind, mutate):
    h = rewrite(root/f'blocks/0-0/{w}/{kind}', transform=mutate)
    refkind = {'outer_predictions':'predictions','selection':'selection','evaluation':'evaluation'}[kind]
    block_h = rewrite(root/'blocks/0-0/complete',transform=lambda b:b['references'][w].__setitem__(refkind,h))
    complete_h = rewrite(root/'complete',transform=lambda r:r['block_digests'].__setitem__('0-0',block_h),
                         transform_identity=lambda i:i['block_digests'].__setitem__('0-0',block_h))
    return replace(anchors,complete_digest=complete_h)


@pytest.mark.parametrize('kind,mutate', [
    ('outer_predictions',lambda v:v['models']['OBS-EQ1'].__setitem__('model_id','OBS-EQ0')),
    ('outer_predictions',lambda v:v['models']['OBS-EQ1']['parameters'].__setitem__('weighting','record_equal')),
    ('outer_predictions',lambda v:v['models']['OBS-EQ1']['source'].__setitem__('implementation_digest','f'*64)),
    ('outer_predictions',lambda v:v['models']['OBS-EQ1']['source'].__setitem__('game_sizes',{})),
    ('outer_predictions',lambda v:v['models']['OBS-LR-r'].__setitem__('training_fingerprint','f'*64)),
    ('outer_predictions',lambda v:v['predictions']['OBS-EF'].pop()),
    ('outer_predictions',lambda v:v['predictions']['OBS-EF'].append(deepcopy(v['predictions']['OBS-EF'][0]))),
    ('outer_predictions',lambda v:v['predictions']['OBS-EF'][0].__setitem__('game_id','wrong-game')),
    ('outer_predictions',lambda v:v['predictions']['OBS-EF'][0].__setitem__('probability',.123)),
    ('outer_predictions',lambda v:v['bin_audit']['OBS-EQ1'][next(iter(v['bin_audit']['OBS-EQ1']))].__setitem__('final_edges',[.1])),
    ('selection',lambda v:v['selected']['OBS-EQ1'].__setitem__('bins',999)),
    ('selection',lambda v:v['candidate_game_brier'][0]['games'].__setitem__('wrong-game',.2)),
    ('evaluation',lambda v:v.__setitem__('test_target_digest','f'*64)),
    ('evaluation',lambda v:v['scores']['OBS-EQ0']['game_equal'].__setitem__('brier',.123)),
])
def test_rehashed_outer_semantic_conflicts(outer_package,tmp_path,kind,mutate):
    source,anchors,old_link = outer_package
    root = tmp_path/'copy';shutil.copytree(source,root)
    anchors = _reanchor(root,anchors,'game_equal',kind,mutate)
    view = read_controlled_d2(root,anchors=anchors)
    i=old_link.identity;i['run_identity_digest']=view.identity_digest
    link=replace(old_link,identity_payload=canonical_json_bytes(i))
    with pytest.raises(ArtifactValidationError):
        verify_outer_artifacts(root,view,link)


@pytest.mark.parametrize('relative',['blocks/2-3/record_equal/outer_predictions/record.json',
    'blocks/4-4/game_equal/selection/manifest.json','blocks/0-1/game_equal/evaluation/record.json'])
def test_missing_outer_files_fail_without_repair(outer_package,tmp_path,relative):
    source,anchors,link=outer_package
    root=tmp_path/'copy';shutil.copytree(source,root);(root/relative).unlink()
    with pytest.raises(ArtifactValidationError):
        verify_outer_artifacts(root,read_controlled_d2(root,anchors=anchors),link)
    assert not (root/relative).exists()


def test_saved_leaf_boundary_and_hierarchical_fallback_are_label_independent():
    from werewolf.observer_dtrs.estimation import TrainingRow
    from werewolf.observer_dtrs.features import ObserverFeatures
    protocol = frozen_protocol()
    rows = [TrainingRow(f'train-{g}-{b}',f'train-{g}',ObserverFeatures(r/2,r/2,r,0.,'discussion',2),g%2)
            for g in range(20) for b,r in enumerate((.2,.8))]
    source = {'scope':'controlled_fixture','protocol_digest':protocol.digest,'dataset_digest':None,
        'split_manifest_digest':None,'tom_fold':None,'outer_fold':None,'inner_fold':None,
        'game_sizes':{f'train-{g}':2 for g in range(20)},'origins':{'fixture':'support'},'implementation_digest':'a'*64}
    fitted = fit_model(rows,protocol=protocol,model_id='OBS-EQ1',bins=5,smoothing=1,source=source)
    m = fitted.record
    assert m['state_nodes']['discussion/2']['edges'] == [.2]
    cases = [('discussion',2,.2,'r_bin',0),('discussion',2,.20000001,'r_bin',1),
             ('pk_discussion',2,.2,'nW',None),('discussion',1,.2,'global',None)]
    for phase,n,r,level,index in cases:
        c=ObserverContext('test','PRE',digest([phase,n,r]),0,'player1','player3',1,phase,
            ('player1','player2','player3') if n==2 else ('player1','player3'),('player1','player2'))
        q=ProbabilityRow(c.game_id,c.boundary_id,c.prefix_digest,c.observer_id,
            (r/2 if n==2 else r,r/2 if n==2 else 0.,0.,0.,0.,0.,1-r))
        legal=LegalDatasetRow(sample_id(c),c,q,observer_features(c,q))
        origin=SampleOrigin(legal.sample_id,'test',phase,n,0,0,'b'*64)
        identity={'model_id':'OBS-EQ1','weighting':'game_equal','tom_fold':0,'outer_fold':0,
            'model_digest':digest(m),'block_digest':'b'*64,'outer_predictions_digest':'c'*64,
            'test_games':['test'],'test_legal_by_id':{legal.sample_id:digest(legal.to_record())}}
        model=VerifiedOuterModel(fitted.payload,canonical_json_bytes(identity))
        support=resolve_training_support(origin,legal,model)
        assert (support['level'],support['bin_index']) == (level,index)
        node=m['root'] if level=='global' else m['wolf_nodes']['2'] if level=='nW' else m['state_nodes']['discussion/2']['leaves'][index]
        assert support['training_stats']==node['stats']
        assert support['fallback'] == (level in ('nW','global'))
        # No test Y argument or target handle exists. Changing an external label cannot affect support.
        external_target={'y':0}; first=resolve_training_support(origin,legal,model)
        external_target['y']=1
        assert resolve_training_support(origin,legal,model)==first
        with pytest.raises(ArtifactValidationError):
            resolve_training_support(origin,replace(legal,features=replace(legal.features,r=.5)),model)
        wrong=model.identity;wrong['weighting']='record_equal'
        with pytest.raises(ArtifactValidationError,match='model'):
            resolve_training_support(origin,legal,replace(model,identity_payload=canonical_json_bytes(wrong)))


@pytest.mark.parametrize('mutate',[
    lambda i:i.__setitem__('weighting','record_equal'),
    lambda i:i.__setitem__('tom_fold',4),
    lambda i:i.__setitem__('source',{**i['source'],'outer_fold':3}),
])
def test_rehashed_outer_envelope_identity_is_checked(outer_package,tmp_path,mutate):
    src,anchors,old_link=outer_package
    root=tmp_path/'copy';shutil.copytree(src,root)
    h=rewrite(root/'blocks/0-0/game_equal/outer_predictions',transform_identity=mutate)
    bh=rewrite(root/'blocks/0-0/complete',transform=lambda b:b['references']['game_equal'].__setitem__('predictions',h))
    fh=rewrite(root/'complete',transform=lambda r:r['block_digests'].__setitem__('0-0',bh),
               transform_identity=lambda i:i['block_digests'].__setitem__('0-0',bh))
    view=read_controlled_d2(root,anchors=replace(anchors,complete_digest=fh))
    identity=old_link.identity;identity['run_identity_digest']=view.identity_digest
    with pytest.raises(ArtifactValidationError,match='identity'):
        verify_outer_artifacts(root,view,replace(old_link,identity_payload=canonical_json_bytes(identity)))


def test_outer_prediction_order_is_not_a_join_key(outer_package,tmp_path):
    src,anchors,old_link=outer_package
    root=tmp_path/'copy';shutil.copytree(src,root)
    h=rewrite(root/'blocks/0-0/game_equal/outer_predictions',
        transform=lambda v:[v['predictions'][m].reverse() for m in MODELS])
    e=rewrite(root/'blocks/0-0/game_equal/evaluation',transform=lambda v:v.__setitem__('predictions_digest',h),
        transform_identity=lambda i:i.__setitem__('predictions_digest',h))
    bh=rewrite(root/'blocks/0-0/complete',transform=lambda b:b['references']['game_equal'].update(predictions=h,evaluation=e))
    fh=rewrite(root/'complete',transform=lambda r:r['block_digests'].__setitem__('0-0',bh),
               transform_identity=lambda i:i['block_digests'].__setitem__('0-0',bh))
    view=read_controlled_d2(root,anchors=replace(anchors,complete_digest=fh))
    i=old_link.identity;i['run_identity_digest']=view.identity_digest
    audit=verify_outer_artifacts(root,view,replace(old_link,identity_payload=canonical_json_bytes(i)))
    assert audit.report['probability_bindings_verified']==1200


def test_cli_refuses_to_write_inside_inputs(tmp_path):
    from scripts.inspect_observer_dtrs_training_audit import main
    with pytest.raises(SystemExit):
        main(['--run',str(tmp_path/'run'),'--dataset',str(tmp_path/'dataset'),
            '--expected-complete-digest','a'*64,'--expected-source-revision','b'*40,
            '--expected-dataset-digest','c'*64,'--expected-split-digest','d'*64,
            '--output',str(tmp_path/'dataset/report.json')])
    assert not (tmp_path/'dataset').exists()
