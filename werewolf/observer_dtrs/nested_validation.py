"""Paired nested validation with immutable per-candidate recovery checkpoints."""
from collections import Counter
from pathlib import Path

from werewolf.artifact_io import canonical_json_bytes
from werewolf.observer_dtrs.baselines import fit_logistic, empirical_model, predict_many, bin_audit
from werewolf.observer_dtrs.estimation import fit_model, FittedModel, TrainingRow, validate_model
from werewolf.observer_dtrs.scoring import score_predictions, select_parameters
from werewolf.observer_dtrs.splits import partition_games
from werewolf.observer_dtrs.statistical_analysis import analyze, MODELS, WEIGHTINGS
from werewolf.observer_dtrs.validation_artifacts import (
    digest, source_snapshot, publish_unit, read_unit, run_lock, record_failure,
)
from werewolf.observer_dtrs.validation_data import training_rows, fit_source


def candidates(protocol):
    p=protocol.record
    return [('OBS-EQ0',{'bins':1,'smoothing':l}) for l in p['smoothing']['lambda_grid_game_equivalent_mass']] + [
        ('OBS-EQ1',{'bins':b,'smoothing':l}) for b in p['histogram']['requested_bin_counts']
            for l in p['smoothing']['lambda_grid_game_equivalent_mass']] + [
        ('OBS-LR-r',{'tau':t}) for t in p['secondary_logistic']['tau_grid']]


def _fit(rows, model, parameters, *, data, source, weighting):
    if model == 'OBS-LR-r':
        return fit_logistic(rows,protocol=data.protocol,tau=parameters['tau'],weighting=weighting,source=source)
    return fit_model(rows,protocol=data.protocol,model_id=model,source=source,weighting=weighting,**parameters).record


def _prediction_records(model, legal):
    if len({r.sample_id for r in legal}) != len(legal):
        raise ValueError('duplicate legal prediction sample')
    details=predict_many(model,[r.features for r in legal])
    return [{'sample_id':r.sample_id,'game_id':r.game_id,'phase':r.features.phase,'nW':r.features.nW,
             **d} for r,d in zip(legal,details,strict=True)]


def _score(legal, targets, predictions):
    if (len(predictions) != len(legal) or len({r['sample_id'] for r in predictions}) != len(legal)
            or {r['sample_id'] for r in predictions} != {r.sample_id for r in legal}
            or set(targets) != {r.sample_id for r in legal}):
        raise ValueError('missing/duplicate/cross-partition predictions or supervision')
    rows=tuple(TrainingRow(
        r.sample_id,r.game_id,r.features,targets[r.sample_id]['y']) for r in legal)
    return score_predictions(rows,{r['sample_id']:r['probability'] for r in predictions})


def _checked_checkpoint(path, kind, identity, compute, verify):
    if path.exists():
        a,value=read_unit(path,kind,identity)
    else:
        value=compute()
        verify(value)
        a=publish_unit(path,kind,identity,value)
    verify(value)
    return a,value


def _validate_fitted(m, model, parameters, source, protocol, weighting):
    if (m['model_id'] != model or m['protocol_digest'] != protocol.digest or m['source'] != source):
        raise ValueError('fitted model identity mismatch')
    if model.startswith('OBS-EQ'):
        validate_model(FittedModel(canonical_json_bytes(m)),protocol)
        expected={'requested_bins':parameters['bins'],'lambda':parameters['smoothing'],
                  'weighting':weighting}
        if m['parameters'] != expected:
            raise ValueError('EQ parameter mismatch')
    elif m['parameters'] != {'tau':parameters['tau'],'weighting':weighting}:
        raise ValueError('Logistic parameter mismatch')


def _select(all_inner, data, train, weighting):
    selected={}
    for model in ('OBS-EQ0','OBS-EQ1'):
        table={}
        for entry in all_inner:
            if entry['model'] == model:
                key=(entry['parameters']['bins'],entry['parameters']['smoothing'])
                table[key]=entry['games']
        b,l=select_parameters(table,protocol=data.protocol,model_id=model,weighting=weighting,
                             game_sizes={g:data.sizes[g] for g in train})
        selected[model]={'bins':b,'smoothing':l}
    values=[r for r in all_inner if r['model']=='OBS-LR-r']
    if {r['parameters']['tau'] for r in values} != set(data.protocol.record['secondary_logistic']['tau_grid']):
        raise ValueError('incomplete Logistic grid')
    scores={}
    for entry in values:
        if set(entry['games']) != set(train):
            raise ValueError('Logistic candidates do not cover identical inner games')
        weights={g:1 if weighting=='game_equal' else data.sizes[g] for g in train}
        scores[entry['parameters']['tau']]=sum(weights[g]*entry['games'][g] for g in train)/sum(weights.values())
    best=min(scores.values())
    selected['OBS-LR-r']={'tau':max(t for t,s in scores.items() if s<=best+data.protocol.record['selection']['tie_absolute_tolerance'])}
    return selected


def run_block(root, *, data, contract_digest, implementation_digest, tom_fold, outer_fold, hook=None):
    root=Path(root)
    train,test=partition_games(data.split,tom_fold=tom_fold,outer_fold=outer_fold)
    if set(train)&set(test) or any(data.folds[g]!=tom_fold for g in (*train,*test)):
        raise ValueError('outer partition source/leakage mismatch')
    block_id={'contract_digest':contract_digest,'tom_fold':tom_fold,'outer_fold':outer_fold}
    block_root=root/'blocks'/f'{tom_fold}-{outer_fold}'
    result_rows={}
    references={}
    pending={}
    for weighting in WEIGHTINGS:
        bank=block_root/weighting
        all_inner=[]
        candidate_refs=[]
        for model,parameters in candidates(data.protocol):
            all_inner.append({'model':model,'parameters':parameters,'games':{}})
        for h in range(3):
            fitting,validation=partition_games(data.split,tom_fold=tom_fold,outer_fold=outer_fold,inner_fold=h)
            if set(fitting)&set(validation) or set(fitting)|set(validation)!=set(train):
                raise ValueError('inner partition isolation mismatch')
            fit_rows=training_rows(data,fitting)
            source=fit_source(data,fitting,tom_fold=tom_fold,outer_fold=outer_fold,inner_fold=h,
                              implementation_digest=implementation_digest)
            legal=data.legal(validation)
            targets=data.targets(validation)  # INNER validation only, never outer test.
            common={**block_id,'weighting':weighting,'inner_fold':h,'source':source,
                'training_digest':digest([[r.sample_id,r.game_id,r.features.to_record(),r.y] for r in fit_rows]),
                'validation_legal_digest':digest([r.record() for r in legal]),'validation_target_digest':digest(targets)}
            for number,(model,parameters) in enumerate(candidates(data.protocol)):
                identity={**common,'model':model,'parameters':parameters}
                def compute(model=model,parameters=parameters):
                    m=_fit(fit_rows,model,parameters,data=data,source=source,weighting=weighting)
                    pred=_prediction_records(m,legal)
                    return {'model':m,'predictions':pred,'scores':_score(legal,targets,pred),
                        'bin_audit':bin_audit(FittedModel(canonical_json_bytes(m)),fit_rows,data.protocol) if model.startswith('OBS-EQ') else None}
                def check(value,model=model,parameters=parameters):
                    _validate_fitted(value['model'],model,parameters,source,data.protocol,weighting)
                    if value['predictions']!=_prediction_records(value['model'],legal) or value['scores']!=_score(legal,targets,value['predictions']):
                        raise ValueError('inner checkpoint predictions/scores inconsistent')
                a,value=_checked_checkpoint(bank/'inner'/str(h)/str(number),'validation_candidate',identity,compute,check)
                scores=value['scores']['by_game']
                if set(scores)!=set(validation) or set(all_inner[number]['games'])&set(scores):
                    raise ValueError('inner duplicate/missing validation game')
                all_inner[number]['games'].update({g:s['brier'] for g,s in scores.items()})
                candidate_refs.append(a.manifest_digest)
                if hook: hook('candidate',a.path)
        selected=_select(all_inner,data,train,weighting)
        selection={'selected':selected,'candidate_game_brier':all_inner,'candidate_artifact_digests':candidate_refs}
        selection_id={**block_id,'weighting':weighting,'candidate_artifact_digests':candidate_refs}
        sa=publish_unit(bank/'selection','validation_selection',selection_id,selection)
        rows=training_rows(data,train)
        source=fit_source(data,train,tom_fold=tom_fold,outer_fold=outer_fold,inner_fold=None,
                          implementation_digest=implementation_digest)
        legal=data.legal(test)  # LEGAL only until ALL models predict and artifact seals.
        identity={**block_id,'weighting':weighting,'selection_digest':sa.manifest_digest,'source':source,
                  'training_digest':digest([[r.sample_id,r.game_id,r.features.to_record(),r.y] for r in rows]),
                  'test_legal_digest':digest([r.record() for r in legal])}
        def outer_compute():
            models={name:_fit(rows,name,parameters,data=data,source=source,weighting=weighting) for name,parameters in selected.items()}
            models['OBS-EF']=empirical_model(FittedModel(canonical_json_bytes(models['OBS-EQ1'])))
            return {'models':models,'predictions':{name:_prediction_records(models[name],legal) for name in MODELS},
                'bin_audit':{name:bin_audit(FittedModel(canonical_json_bytes(models[name])),rows,data.protocol) for name in ('OBS-EQ0','OBS-EQ1')}}
        def outer_check(value):
            if set(value['models'])!=set(MODELS) or set(value['predictions'])!=set(MODELS):
                raise ValueError('outer model coverage mismatch')
            for name,parameters in selected.items():
                _validate_fitted(value['models'][name],name,parameters,source,data.protocol,weighting)
            if value['models']['OBS-EF']!=empirical_model(FittedModel(canonical_json_bytes(value['models']['OBS-EQ1']))):
                raise ValueError('EF not bound to selected EQ1')
            for name in MODELS:
                if value['predictions'][name]!=_prediction_records(value['models'][name],legal):
                    raise ValueError('outer missing/duplicate/inconsistent predictions')
        pa,prediction=_checked_checkpoint(bank/'outer_predictions','validation_outer_predictions',identity,outer_compute,outer_check)
        if hook: hook('outer_predictions',pa.path)
        pending[weighting]=(bank,legal,pa,prediction)
        references[weighting]={'selection':sa.manifest_digest,'predictions':pa.manifest_digest}
    # Both weighting pipelines and all models are sealed before test Y is opened.
    for weighting,(bank,legal,pa,prediction) in pending.items():
        targets=data.targets(test)  # Only AFTER selection/refit/prediction persistence.
        scores={name:_score(legal,targets,prediction['predictions'][name]) for name in MODELS}
        evaluation={'scores':scores,'test_target_digest':digest(targets),'predictions_digest':pa.manifest_digest}
        ea=publish_unit(bank/'evaluation','validation_outer_evaluation',
            {**block_id,'weighting':weighting,'predictions_digest':pa.manifest_digest,'test_target_digest':digest(targets)},evaluation)
        references[weighting]['evaluation']=ea.manifest_digest
        for j,r in enumerate(legal):
            if r.sample_id not in result_rows:
                result_rows[r.sample_id]={'sample_id':r.sample_id,'game_id':r.game_id,'phase':r.features.phase,'nW':r.features.nW,
                    **targets[r.sample_id],'predictions':{}}
            result_rows[r.sample_id]['predictions'][weighting]={name:prediction['predictions'][name][j]['probability'] for name in MODELS}
    record={'status':'PAIRED_BLOCK_COMPLETE','references':references,'rows':list(result_rows.values()),
            'train_games':train,'test_games':test}
    a=publish_unit(block_root/'complete','validation_block_complete',block_id,record)
    return a,record


def execute(root, *, data, expected_revision=None, block=None, hook=None):
    root=Path(root).resolve()
    formal=data.scope=='formal_development_validation'
    snapshot=source_snapshot(formal=formal,expected_revision=expected_revision)
    code_digest=digest(snapshot)
    contract={'scope':data.scope,'protocol_digest':data.protocol.digest,'dataset_digest':data.dataset_digest,
        'split_digest':data.split_digest,'source_snapshot_digest':code_digest,'origins':data.origins,
        'game_sizes':data.sizes,'game_folds':data.folds,'models':MODELS,'weightings':WEIGHTINGS,
        'split_record':data.split,
        'sample_ids_digest_by_game':({g['game_id']:g['sample_ids_digest'] for g in data.manifest['games']}
            if formal else {g:digest([r.sample_id for r in data.legal((g,))]) for g in data.sizes}),
        'known_source_limitations':data.protocol.record['known_source_limitations'],
        'research_scope':data.protocol.record['research_scope']}
    contract_id={'run_identity':digest(contract)}
    with run_lock(root):
        ca=publish_unit(root/'contract','validation_contract',contract_id,contract,
                        {'sources.json':canonical_json_bytes(snapshot),'protocol.json':data.protocol.payload})
        stage='start'
        try:
            all_rows=[]
            block_refs={}
            blocks={}
            requested=[block] if block is not None else [(k,j) for k in range(5) for j in range(5)]
            for k,j in requested:
                stage=f'block-{k}-{j}'
                a,record=run_block(root,data=data,contract_digest=ca.manifest_digest,
                    implementation_digest=code_digest,tom_fold=k,outer_fold=j,hook=hook)
                block_refs[f'{k}-{j}']=a.manifest_digest
                blocks[(k,j)]=record['test_games']
                all_rows.extend(record['rows'])
                if hasattr(data,'views'):
                    data.views.clear()
            if source_snapshot(formal=formal,expected_revision=expected_revision)!=snapshot:
                raise ValueError('source/runtime changed during execution; no completion seal')
            if block is not None:
                return {'status':'PARTIAL_NOT_FULL_EXPERIMENT','block':list(block),'block_digest':a.manifest_digest}
            if (set(r['game_id'] for r in all_rows)!=set(data.sizes) or len(all_rows)!=sum(data.sizes.values())
                    or len({r['sample_id'] for r in all_rows})!=len(all_rows)
                    or Counter(r['game_id'] for r in all_rows)!=Counter(data.sizes)):
                raise ValueError('all outer predictions must cover every eligible sample/game exactly once')
            stage='fixed_prediction_analysis'
            summary,indices=analyze(all_rows,blocks=blocks,protocol=data.protocol,scope=data.scope)
            final=publish_unit(root/'complete','validation_complete',
                {'contract_digest':ca.manifest_digest,'block_digests':block_refs},
                {'status':'ALL_BLOCKS_COMPLETE','block_digests':block_refs,'summary':summary},
                {'bootstrap_indices.int64le':indices})
            return {'status':'ALL_BLOCKS_COMPLETE','manifest_digest':final.manifest_digest,
                    'contract_digest':ca.manifest_digest,'rows':len(all_rows),'games':len(data.sizes),'scope':data.scope}
        except BaseException as error:
            record_failure(root,ca.manifest_digest,stage,error)
            raise


def verify_result(root):
    """Read-only cross-reference/coverage/summary reconstruction; never repairs.

    Resume additionally rechecks original training/validation partition identities
    and predictions. This standalone audit does not execute stored source text.
    """
    import json
    from types import SimpleNamespace
    from werewolf.artifact_io import verify_artifact, read_artifact_file
    from werewolf.observer_dtrs.validation_artifacts import VERSION
    from werewolf.observer_dtrs.d2_protocol import FrozenProtocol
    root=Path(root)
    def load(path,kind):
        a=verify_artifact(path,expected_artifact_type='observer_dtrs_'+kind,expected_schema_version=VERSION)
        record=json.loads(read_artifact_file(a,'record.json'))
        if a.manifest['identity_digest']!=digest(a.manifest['identity']):
            raise ValueError('unit identity digest mismatch')
        return a,record
    ca,c=load(root/'contract','validation_contract')
    protocol=FrozenProtocol(read_artifact_file(ca,'protocol.json'))
    if (digest(json.loads(read_artifact_file(ca,'sources.json')))!=c['source_snapshot_digest']
            or ca.manifest['identity']!={'run_identity':digest(c)} or c['protocol_digest']!=protocol.digest):
        raise ValueError('contract source/protocol identity mismatch')
    data=SimpleNamespace(protocol=protocol,sizes=c['game_sizes'])
    refs,blocks,all_rows={}, {}, []
    for k in range(5):
        for j in range(5):
            base=root/'blocks'/f'{k}-{j}'
            if not (base/'complete').exists():
                continue
            ba,block=load(base/'complete','validation_block_complete')
            identity={'contract_digest':ca.manifest_digest,'tom_fold':k,'outer_fold':j}
            train,test=partition_games(c['split_record'],tom_fold=k,outer_fold=j)
            if (ba.manifest['identity']!=identity or block['train_games']!=list(train) or block['test_games']!=list(test)
                    or block['status']!='PAIRED_BLOCK_COMPLETE'):
                raise ValueError('sealed block source/partition mismatch')
            by_sample={r['sample_id']:r for r in block['rows']}
            if len(by_sample)!=len(block['rows']) or Counter(r['game_id'] for r in block['rows'])!=Counter({g:c['game_sizes'][g] for g in test}):
                raise ValueError('sealed outer block population mismatch')
            for g in test:
                if digest([r['sample_id'] for r in block['rows'] if r['game_id']==g])!=c['sample_ids_digest_by_game'][g]:
                    raise ValueError('outer sample identity mismatch')
            for weighting in WEIGHTINGS:
                bank=base/weighting
                sa,selection=load(bank/'selection','validation_selection')
                collected=[{'model':m,'parameters':p,'games':{}} for m,p in candidates(protocol)]
                inner_refs=[]
                for h in range(3):
                    fitting,validation=partition_games(c['split_record'],tom_fold=k,outer_fold=j,inner_fold=h)
                    for number,(model,parameters) in enumerate(candidates(protocol)):
                        a,value=load(bank/'inner'/str(h)/str(number),'validation_candidate')
                        ident=a.manifest['identity']
                        if any(ident[key]!=v for key,v in {**identity,'weighting':weighting,'inner_fold':h,'model':model,'parameters':parameters}.items()):
                            raise ValueError('candidate grid/source mismatch')
                        source=ident['source']
                        if source['game_sizes']!={g:c['game_sizes'][g] for g in fitting}:
                            raise ValueError('candidate training partition mismatch')
                        _validate_fitted(value['model'],model,parameters,source,protocol,weighting)
                        scores=value['scores']['by_game']
                        predictions=value['predictions']
                        if (set(scores)!=set(validation) or any(scores[g]['rows']!=c['game_sizes'][g] for g in validation)
                                or len({r['sample_id'] for r in predictions})!=len(predictions)
                                or Counter(r['game_id'] for r in predictions)!=Counter({g:c['game_sizes'][g] for g in validation})):
                            raise ValueError('candidate prediction/score population mismatch')
                        collected[number]['games'].update({g:s['brier'] for g,s in scores.items()})
                        inner_refs.append(a.manifest_digest)
                if (selection['candidate_artifact_digests']!=inner_refs or selection['candidate_game_brier']!=collected
                        or selection['selected']!=_select(collected,data,train,weighting)
                        or sa.manifest['identity']!={**identity,'weighting':weighting,'candidate_artifact_digests':inner_refs}):
                    raise ValueError('selection is not frozen inner selection')
                pa,prediction=load(bank/'outer_predictions','validation_outer_predictions')
                ea,evaluation=load(bank/'evaluation','validation_outer_evaluation')
                if (block['references'][weighting]!={'selection':sa.manifest_digest,'predictions':pa.manifest_digest,'evaluation':ea.manifest_digest}
                        or pa.manifest['identity']['selection_digest']!=sa.manifest_digest
                        or evaluation['predictions_digest']!=pa.manifest_digest):
                    raise ValueError('outer artifact cross-reference mismatch')
                targets={sid:{'y':r['y'],'successful_empty':r['successful_empty']} for sid,r in by_sample.items()}
                if evaluation['test_target_digest']!=digest(targets):
                    raise ValueError('outer supervision fingerprint mismatch')
                for model in MODELS:
                    pred=prediction['predictions'][model]
                    if (len({r['sample_id'] for r in pred})!=len(pred) or set(r['sample_id'] for r in pred)!=set(by_sample)
                            or any(r['probability']!=by_sample[r['sample_id']]['predictions'][weighting][model] for r in pred)):
                        raise ValueError('outer prediction mismatch')
                    scored=score_predictions([SimpleNamespace(sample_id=sid,game_id=r['game_id'],y=r['y']) for sid,r in by_sample.items()],
                        {r['sample_id']:r['probability'] for r in pred})
                    if scored!=evaluation['scores'][model]:
                        raise ValueError('outer scores do not reconstruct')
            refs[f'{k}-{j}']=ba.manifest_digest
            blocks[(k,j)]=test
            all_rows.extend(block['rows'])
    if not (root/'complete').exists():
        return {'scope':c['scope'],'status':'PARTIAL_NOT_FULL_EXPERIMENT','verified_blocks':len(refs)}
    fa,final=load(root/'complete','validation_complete')
    if (len(refs)!=25 or fa.manifest['identity']!={'contract_digest':ca.manifest_digest,'block_digests':refs}
            or final['block_digests']!=refs or final['status']!='ALL_BLOCKS_COMPLETE'
            or len(all_rows)!=sum(c['game_sizes'].values())
            or len({r['sample_id'] for r in all_rows})!=len(all_rows)):
        raise ValueError('invalid all-block completion seal')
    summary,draws=analyze(all_rows,blocks=blocks,protocol=protocol,scope=c['scope'])
    if summary!=final['summary'] or draws!=read_artifact_file(fa,'bootstrap_indices.int64le'):
        raise ValueError('statistical summary/bootstrap do not reconstruct')
    return {'scope':c['scope'],'status':'ALL_BLOCKS_COMPLETE','verified_blocks':25,'manifest_digest':fa.manifest_digest}
