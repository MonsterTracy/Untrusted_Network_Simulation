"""One completed historical CPU study of O, conditional Z and complete-window T."""
from argparse import ArgumentParser
from collections import Counter,defaultdict
from pathlib import Path
import datetime,gzip,json,sys,time,zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.analyze_observer_dtrs_public_behavior import records,manifest,sha,save,table,interval
from werewolf.artifact_io import canonical_json_bytes
from werewolf.canonical_collection.game_bundle import _prefix_from_record
from werewolf.canonical_collection.pre import validate_authoritative_pre_prefix
from werewolf.observer_dtrs.public_behavior_research import (PUBLIC_NAMES,public_features,fit_target,target_logits,row_scores,paired_game_cv,
    parse_public_game,public_day_witness,PUBLIC_AUDIT_ORIGIN_SHA256)
from werewolf.observer_dtrs.opportunity_risk_research import VERSION,observe_complete_window,fit_conditional,product_scores
from werewolf.observer_dtrs.cognition_gate_research import profile,D3_DIGEST
from werewolf.observer_dtrs.decision import TiePolicy
from werewolf.observer_dtrs.d2_protocol import FrozenProtocol
from werewolf.observer_dtrs.statistical_analysis import bootstrap_plan

# The archive's internal publication name is fixed by OLD_SHA, not by its local directory name.
PRIOR_ARCHIVE_ROOT='observer-dtrs-public-behavior-20261030T014526+0800'
OLD_SHA='ad10764afe774d23cb5eb69e0fe79e0116b75b62882938e4f2cf78a650dce4b1'
PINNED={'features-before-future.npz':'f1cbe51d25ca5482d82ac5443c7de12f55c8fddd7f935f8d9628a8a015993b17',
 'feature-identities.jsonl.gz':'8b355cfdb54e142eba7d3b27e67c48907428c549366ee64e973c1c2ccb20f3b6',
 'outcome-evidence.jsonl.gz':'a3fcf960d57dafa82878025bef8f62f275014c810d4c183af8d0aa0e90018af1',
 'predictions.npz':'0852242a2d20fcb6c9703d5373eff77335d441b8e62afb1b6cb01a2b23621d13'}


def argument_parser():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--prior-run',type=Path,required=True,help='Verified prior public-behavior result directory')
    parser.add_argument('--prior-archive',type=Path,required=True,help='Original prior-run ZIP with the pinned SHA')
    parser.add_argument('--public-pre-file',type=Path,required=True,help='Admitted canonical public PRE JSONL')
    parser.add_argument('--public-archive',type=Path,required=True,help='Admitted public event/V1 ZIP archive')
    parser.add_argument('--d2-run',type=Path,required=True,help='Verified formal D2 run directory')
    parser.add_argument('--plan',type=Path,required=True,help='Previously reviewed analysis plan')
    parser.add_argument('--output',type=Path,required=True,help='New independent output directory')
    return parser


def main():
    a=argument_parser().parse_args();out=a.output.absolute()
    prior_run=a.prior_run.absolute();prior_archive=a.prior_archive.absolute();d2_run=a.d2_run.absolute()
    public_pre_file=a.public_pre_file.absolute();public_archive=a.public_archive.absolute()
    if (out.exists() or not out.parent.is_dir() or '..' in out.parts or out.parent.resolve()!=out.parent
            or any(out.is_relative_to(x.resolve()) for x in (ROOT,prior_run,prior_archive,d2_run,public_pre_file.parent,public_archive.parent))):
        raise ValueError('new independent canonical output directory required')
    out.mkdir(mode=0o700);start=time.monotonic();plan=a.plan.read_bytes();(out/'analysis-plan.zh.md').write_bytes(plan)
    save(out/'start.json',{'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan_sha256':sha(plan),
        'public_validator_origin_sha256':PUBLIC_AUDIT_ORIGIN_SHA256,
        'primary':'direct_s-minus-direct_PUBLIC_game_equal_Log_Loss','version':VERSION,
        'new_source_sha256':{str(f.relative_to(ROOT)):sha(f.read_bytes()) for f in [
            ROOT/'werewolf/observer_dtrs/opportunity_risk_research.py',ROOT/'scripts/analyze_observer_dtrs_opportunity_risk.py']}})
    # Independent old archive anchor and every original persisted member are verified before reuse.
    archive=prior_archive
    if sha(archive.read_bytes())!=OLD_SHA:raise ValueError('old result external ZIP SHA mismatch')
    with zipfile.ZipFile(archive) as z:
        for item in json.loads((prior_run/'SHA256-MANIFEST.json').read_text())['files']:
            path=prior_run/item['path'];raw=path.read_bytes()
            if len(raw)!=item['bytes'] or sha(raw)!=item['sha256'] or z.read(PRIOR_ARCHIVE_ROOT+'/'+item['path'])!=raw:
                raise ValueError('old result byte/manifest mismatch')
    for name,h in PINNED.items():
        if sha((prior_run/'real-analysis'/name).read_bytes())!=h:raise ValueError('old source snapshot SHA mismatch')
    meta=[json.loads(x) for x in gzip.open(prior_run/'real-analysis/feature-identities.jsonl.gz','rt')]
    previous={x['sample_id']:x for x in map(json.loads,gzip.open(prior_run/'real-analysis/outcome-evidence.jsonl.gz','rt'))}
    data=np.load(prior_run/'real-analysis/features-before-future.npz',allow_pickle=False)
    if len(meta)!=22685 or len({m['sample_id'] for m in meta})!=22685 or set(previous)!={m['sample_id'] for m in meta}:
        raise ValueError('fixed sample population mismatch')
    if list(data['sample_ids'])!=[m['sample_id'] for m in meta]:raise ValueError('sample snapshot identity mismatch')
    X0=data['public'];cog=data['cognition_s_r_p'];data.close()
    if X0.shape!=(22685,len(PUBLIC_NAMES)) or not np.array_equal(cog,np.array([[m['s'],m['r'],m['p']] for m in meta])):
        raise ValueError('feature snapshot/source mismatch')
    by_pre=defaultdict(list);by_game=defaultdict(list)
    for i,m in enumerate(meta):by_pre[m['game_id'],m['boundary_id']].append(i);by_game[m['game_id']].append(i)
    raw=public_pre_file.read_bytes()
    if sha(raw)!='ea40e0c4307105fe4e38a8ce1a2844b18cf9195fcb55c3a096ea9a3e1973f50a':raise ValueError('public PRE SHA')
    pres={}
    for r in records(raw):
        pre=validate_authoritative_pre_prefix(_prefix_from_record(r));key=pre.game_id,pre.boundary_id
        if key not in by_pre or key in pres:raise ValueError('foreign/duplicate PRE')
        pres[key]=pre
        rows=[meta[i] for i in by_pre[key]]
        if len(rows)!=len(pre.alive_observer_ids)-rows[0]['nW']:raise ValueError('complete observer population mismatch')
        for i in by_pre[key]:
            m=meta[i]
            if (m['prefix_digest'],m['step'],m['actor'],m['phase'],m['day'])!=(pre.prefix_digest,pre.step_index,pre.current_speaker,pre.public_temporal_state.phase.value,pre.public_temporal_state.day):
                raise ValueError('full PRE metadata identity mismatch')
            if not np.array_equal(X0[i],public_features(pre,m['observer'])):raise ValueError('PRE-only features changed')
    if set(pres)!=set(by_pre) or len(pres)!=5835 or len(by_game)!=1500:raise ValueError('missing PRE/game population')
    identity=sorted((p.game_id,p.boundary_id,p.prefix_digest,p.step_index,p.current_speaker) for p in pres.values())
    if sha(canonical_json_bytes(identity))!='2e151e0ac08d794368f32782eab4943b9c23d56640e3a22c9df006a68c30b464':raise ValueError('fixed identity set mismatch')
    save(out/'feature-proof.json',{'prior_result_zip_sha256':OLD_SHA,'prior_snapshot_files':PINNED,
        'all_22685_public_feature_rows_recomputed_from_PRE_exactly':True,'rows':22685,'PREs':5835,'games':1500,
        'public_names':PUBLIC_NAMES,'future_not_a_feature':True,'raw_Q_not_reread':'authenticated prior s/r/p snapshot reused'})
    print('FEATURES_REVERIFIED',len(meta),flush=True)

    raw=public_archive.read_bytes()
    if sha(raw)!='e3d9dfdc095396346e97ec7d712417ffd37cff5515f9e0016afd58085e065947':raise ValueError('public future external SHA')
    targets=[None]*len(meta);source_files=[];day_witnesses={}
    with zipfile.ZipFile(public_archive) as z:
        if len(z.namelist())!=3004 or len(set(z.namelist()))!=3004:raise ValueError('public ZIP file set mismatch')
        pm=manifest(z.read('source/publication-manifest.json'));fm=manifest(z.read('source/fold-manifest.json'))
        if (pm['manifest_digest']!='396ea0fa3f3f03dd1dfce8b25c0162ca489b992321d0c9e6f772eb2b0c4a2acd'
            or fm['manifest_digest']!=pm['development_fold_manifest_digest']
            or fm['manifest_digest']!='754fe0e517aab9e9237f0cb5097bde0daf8a43a09518b9691cfca67724539e69'):raise ValueError('publication/fold identity')
        expected={g['game_id']:g['bundle_digest'] for g in pm['development_games']};bundles=records(z.read('source/bundle-manifests.jsonl'))
        if len(bundles)!=1500 or set(expected)!=set(by_game):raise ValueError('public game population')
        seen_games=set()
        for record in bundles:
            m=manifest(canonical_json_bytes(record));gid=m['game_id']
            if gid in seen_games or m['manifest_digest']!=expected[gid] or m['source_revision']!=pm['source_revision'] or m['canonical_eligibility'] is not True:
                raise ValueError('bundle source mismatch')
            seen_games.add(gid);payload={}
            for name in ('public_event_stream.jsonl','speech_annotations_v1.jsonl'):
                b=z.read(f'games/{gid}/{name}');spec=m['file_table']['public/'+name]
                if len(b)!=spec['byte_size'] or sha(b)!=spec['sha256']:raise ValueError('raw public file mismatch')
                source_files.append({'game_id':gid,'member':name,'bytes':len(b),'sha256':sha(b),'bundle_digest':m['manifest_digest']});payload[name]=records(b)
            h,a=parse_public_game(payload['public_event_stream.jsonl'],payload['speech_annotations_v1.jsonl'])
            if set(a)!={e.event_id for e in h.events if e.event_type=='public_speech'}:raise ValueError('public speech/V1 coverage')
            for i in by_game[gid]:
                row=meta[i];key=gid,row['day']
                if key not in day_witnesses:day_witnesses[key]={'game_id':gid,**public_day_witness(h,row['day'])}
                outcome=observe_complete_window(pres[gid,row['boundary_id']],row['observer'],h,a,day_witnesses[key])
                old={k:v for k,v in outcome.items() if k not in ('O','T','new_status')}
                if {'sample_id':row['sample_id'],**old}!=previous[row['sample_id']]:raise ValueError('old conditional evidence changed')
                targets[i]={'sample_id':row['sample_id'],**outcome}
    if seen_games!=set(by_game) or any(t is None for t in targets):raise ValueError('missing outcome rows')
    with gzip.open(out/'O-Z-T-evidence.jsonl.gz','wt') as f:
        for t in targets:f.write(json.dumps(t,ensure_ascii=False)+'\n')
    save(out/'source-public-files.json',source_files)
    valid=np.array([t['T'] is not None for t in targets]);idx=np.flatnonzero(valid)
    T=np.array([targets[i]['T'] for i in idx],int);O=np.array([targets[i]['O'] for i in idx],int)
    if not np.all(T<=O):raise ValueError('event implies opportunity')
    M=[meta[i] for i in idx];G=[m['game_id'] for m in M];B=np.array([[m['tom_fold'],m['outer_fold']] for m in M],int)
    folds=list(paired_game_cv(B,G));coverage=[]
    raw_groups={'all':np.ones(len(meta),bool),'first':np.array([m['first'] for m in meta])}
    for phase in ('discussion','pk_discussion'):
        for nw in (1,2):raw_groups[f'{phase}/{nw}']=np.array([m['phase']==phase and m['nW']==nw for m in meta])
    for group,mask in raw_groups.items():
        ids=np.flatnonzero(mask);known=ids[valid[ids]];o=[targets[i]['O'] for i in ids if targets[i]['O'] is not None]
        coverage.append({'group':group,'original_rows':len(ids),'PREs':len({(meta[i]['game_id'],meta[i]['boundary_id']) for i in ids}),
            'original_games':len({meta[i]['game_id'] for i in ids}),'O_known':len(o),'O1':sum(o),'O0':len(o)-sum(o),
            'T_known':len(known),'T_positive':sum(targets[i]['T'] for i in known),'T_negative':sum(1-targets[i]['T'] for i in known),
            'T_games':len({meta[i]['game_id'] for i in known}),'T_unknown':len(ids)-len(known)})
    table(out/'coverage.csv',coverage);print('COVERAGE',json.dumps(coverage[0]),flush=True)
    design={'PUBLIC':X0[idx],'s':np.column_stack([X0,cog[:,0]])[idx],
        'r':np.column_stack([X0,cog[:,1]])[idx],'p':np.column_stack([X0,cog[:,2]])[idx]}
    codes=np.array([('POS','BND','NEG').index(profile(float(p),22,TiePolicy.TERMINAL_NEG_FIRST)[0].region.value) for p in cog[:,2]])
    design['C22']=np.column_stack([X0,codes==0,codes==1])[idx]
    model_records=[];predictions={};split=[]
    for (source,outer),train,test in folds:
        if set(G[i] for i in train)&set(G[i] for i in test):raise ValueError('game leakage')
        split.append({'source_fold':source,'outer_fold':outer,'train_games':sorted({G[i] for i in train}),
            'test_games':sorted({G[i] for i in test}),'train_rows':len(train),'test_rows':len(test),
            'train_O1':int(O[train].sum()),'train_T1':int(T[train].sum()),'test_O1':int(O[test].sum()),'test_T1':int(T[test].sum())})
    save(out/'split-audit.json',split)
    for name,X in design.items():
        for component in ('direct','opportunity','conditional'):
            oof=np.full(len(idx),np.nan);seen=np.zeros(len(idx),int)
            for (source,outer),train,test in folds:
                games=[G[i] for i in train]
                if component=='conditional':model=fit_conditional(X[train],O[train],T[train],games)
                else:model=fit_target(X[train],T[train] if component=='direct' else O[train],games)
                model.update(component=component,representation=name,source_fold=source,outer_fold=outer,
                    full_reference_training_fingerprint=sha(canonical_json_bytes([[M[i]['sample_id'],int(O[i]),int(T[i]),X[i].tolist()] for i in train])),
                    test_sample_identity_digest=sha(canonical_json_bytes(sorted(M[i]['sample_id'] for i in test))))
                oof[test]=target_logits(model,X[test]);seen[test]+=1;model_records.append(model)
            if not np.all(seen==1) or not np.isfinite(oof).all():raise ValueError('OOF coverage failure')
            predictions[component+'/'+name]=oof
        print('REPRESENTATION_COMPLETE',name,flush=True)
    save(out/'fit-records.json',model_records)
    np.savez_compressed(out/'new-predictions.npz',sample_ids=np.array([m['sample_id'] for m in M]),O=O,T=T,
        component_ids=np.array(list(predictions)),component_logits=np.stack(list(predictions.values())),all_sample_ids=np.array([m['sample_id'] for m in meta]),valid_indices=idx)
    # Every test PRE gets both learned components, including actual O=0 rows.
    scores={}
    for name in design:
        scores['T/direct/'+name]=('T',row_scores(predictions['direct/'+name],T),np.ones(len(T),bool))
        scores['T/two/'+name]=('T',product_scores(predictions['opportunity/'+name],predictions['conditional/'+name],T),np.ones(len(T),bool))
        scores['O/'+name]=('O',row_scores(predictions['opportunity/'+name],O),np.ones(len(T),bool))
        keep=O==1;scores['C/new/'+name]=('C',row_scores(predictions['conditional/'+name][keep],T[keep]),keep)
    scores['T/hybrid/sO_publicC']=('T',product_scores(predictions['opportunity/s'],predictions['conditional/PUBLIC'],T),np.ones(len(T),bool))
    scores['T/hybrid/publicO_sC']=('T',product_scores(predictions['opportunity/PUBLIC'],predictions['conditional/s'],T),np.ones(len(T),bool))
    # Historical conditional predictions are read-only controls, not refitted or relabeled.
    legacy=np.load(prior_run/'real-analysis/predictions.npz',allow_pickle=False);old_ids=legacy['sample_ids'];old_order={s:j for j,s in enumerate(old_ids)}
    keep=O==1;conditional_ids=[m['sample_id'] for m,k in zip(M,keep) if k]
    if set(conditional_ids)!=set(old_ids):raise ValueError('old conditional population changed')
    order=np.array([old_order[s] for s in conditional_ids]);legacy_names=list(legacy['model_ids'])
    for name,old_name in [('PUBLIC','M0_PUBLIC'),('s','M1_s'),('r','M1_r'),('p','M2_p'),('C22','M3_C22')]:
        z=legacy['logits'][legacy_names.index(old_name)][order]
        if not np.array_equal(legacy['Z'][order],T[keep]):raise ValueError('old Z changed')
        scores['C/historical/'+name]=('C',row_scores(z,T[keep]),keep)
    legacy.close()
    d2=FrozenProtocol((d2_run/'contract/protocol.json').read_bytes());plans={}
    for scope,mask in [('full',np.ones(len(T),bool)),('conditional',keep)]:
        blocks=defaultdict(set)
        for m,k in zip(M,mask):
            if k:blocks[m['tom_fold'],m['outer_fold']].add(m['game_id'])
        games,draws=bootstrap_plan(blocks,protocol=d2,formal=scope=='full')
        if scope=='full' and sha(draws.astype('<i8').tobytes())!='33e94ab2b2875ff7f47de212062a628de613de37e4b8f3ff0b887eb479f0a3da':
            raise ValueError('full original bootstrap bytes mismatch')
        if scope=='conditional' and not np.array_equal(draws,np.load(prior_run/'real-analysis/bootstrap-indices.npy')):raise ValueError('historical conditional bootstrap differs')
        plans[scope]=(games,draws);np.save(out/('bootstrap-'+scope+'.npy'),draws);save(out/('game-order-'+scope+'.json'),games)
    results=[];calibration=[];subgroups=[];game_tables=[];model_stats={};group_masks={k:v[idx] for k,v in raw_groups.items()}
    for name,(target,values,mask) in scores.items():
        probability,bs,ll=values;y=(O if target=='O' else T)[mask];models_meta=[m for m,k in zip(M,mask) if k]
        scope='conditional' if target=='C' else 'full';games,draws=plans[scope];gindex={g:i for i,g in enumerate(games)}
        gi=np.array([gindex[m['game_id']] for m in models_meta]);sizes=np.bincount(gi,minlength=len(games));weights=1/(len(games)*sizes[gi])
        matrix=np.column_stack([np.bincount(gi,weights=v,minlength=len(games))/sizes for v in (bs,ll,probability-y)])
        model_stats[name]=(scope,matrix);counts=np.array([np.bincount(r,minlength=len(games)) for r in draws],float);boots=counts@matrix/len(games)
        row={'model':name,'target':target,'rows':len(y),'games':len(games),'positive':int(y.sum()),'negative':int((1-y).sum()),
            'brier':float(matrix[:,0].mean()),'log_loss':float(matrix[:,1].mean()),'bias':float(matrix[:,2].mean()),
            'row_equal_brier':float(bs.mean()),'row_equal_log_loss':float(ll.mean()),'ECE_10':0.}
        for j,key in enumerate(('brier','log_loss')):row[key+'_ci_low'],row[key+'_ci_high']=interval(boots[:,j])
        bins=np.minimum(9,(probability*10).astype(int))
        for k in range(10):
            b=bins==k;mass=float(weights[b].sum());mp=float(weights[b]@probability[b]/mass) if mass else None;rate=float(weights[b]@y[b]/mass) if mass else None
            calibration.append({'model':name,'bin':k,'rows':int(b.sum()),'games':len(set(gi[b])),'mass':mass,'mean_prediction':mp,'observed_rate':rate})
            if mass:row['ECE_10']+=mass*abs(mp-rate)
        results.append(row)
        for i,g in enumerate(games):game_tables.append({'model':name,'game_id':g,'rows':int(sizes[i]),'brier':float(matrix[i,0]),'log_loss':float(matrix[i,1])})
        for group,gm in group_masks.items():
            b=gm[mask];den=np.bincount(gi[b],minlength=len(games));active=den>0
            subgroups.append({'model':name,'group':group,'rows':int(b.sum()),'games':int(active.sum()),'positive':int(y[b].sum()),
                **{key:float((np.bincount(gi[b],weights=v[b],minlength=len(games))[active]/den[active]).mean()) if active.any() else None for key,v in [('brier',bs),('log_loss',ll)]}})
    table(out/'model-results.csv',results);table(out/'calibration.csv',calibration);table(out/'subgroup-results.csv',subgroups);table(out/'game-metrics.csv.gz',game_tables)
    comparisons=[];pairs=[]
    for prefix in ('T/direct/','T/two/','O/','C/new/','C/historical/'):
        pairs.extend((prefix+n,prefix+'PUBLIC') for n in ('s','r','p','C22'))
    pairs.extend((f'T/two/{n}',f'T/direct/{n}') for n in design)
    pairs.extend((f'C/new/{n}',f'C/historical/{n}') for n in design)
    pairs.extend([('T/hybrid/sO_publicC','T/two/PUBLIC'),('T/hybrid/publicO_sC','T/two/PUBLIC'),
        ('T/two/s','T/hybrid/sO_publicC'),('T/two/s','T/hybrid/publicO_sC')])
    for a,b in pairs:
        scope,x=model_stats[a];other,y=model_stats[b]
        if scope!=other:raise ValueError('comparison populations differ')
        games,draws=plans[scope];delta=x-y;counts=np.array([np.bincount(r,minlength=len(games)) for r in draws],float);boots=counts@delta/len(games)
        row={'model':a,'reference':b,'population':scope}
        for j,key in enumerate(('brier','log_loss')):row[key+'_difference']=float(delta[:,j].mean());row[key+'_ci_low'],row[key+'_ci_high']=interval(boots[:,j])
        comparisons.append(row)
    table(out/'paired-comparisons.csv',comparisons)
    primary=next(x for x in comparisons if x['model']=='T/direct/s' and x['reference']=='T/direct/PUBLIC')
    summary={'status':'EXPLORATORY_COMPLETE_PRE_OPPORTUNITY_RISK_ANALYSIS_COMPLETED','version':VERSION,'coverage':coverage,
        'primary_predeclared':primary,'reference_population':'T-known complete public windows; no-opportunity T0 is a new label only',
        'new_model_fits':len(model_records),'old_conditional_fits_reused_not_refit':125,'outer_blocks':len(folds),'conditional_weight_rule':'inherit full-reference S game weights then normalize',
        'all_test_rows_receive_both_components_not_future_O_gate':True,'D3_protocol_digest':D3_DIGEST,'cost':'C22 fixed secondary only',
        'old_archive_sha256':OLD_SHA,'public_future_sha256':'e3d9dfdc095396346e97ec7d712417ffd37cff5515f9e0016afd58085e065947',
        'bootstrap':{name:{'games':len(g),'replicates':len(d),'raw_indices_sha256':sha(d.astype('<i8').tobytes())} for name,(g,d) in plans.items()},
        'seconds':time.monotonic()-start,'scope':'conditional repeated historical development; not independent TEST, action utility or causal effect'}
    save(out/'summary.json',summary);print('COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
