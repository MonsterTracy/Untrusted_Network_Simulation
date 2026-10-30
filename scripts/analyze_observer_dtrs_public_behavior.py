"""New historical behavior target: PRE-only features, authenticated public outcomes.

Research-only CPU entry; no original artifact or production module is changed.
"""
from argparse import ArgumentParser
from collections import Counter,defaultdict
from dataclasses import asdict
import csv,datetime,gzip,hashlib,json,os
from pathlib import Path
import sys,time,zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from werewolf.artifact_io import canonical_json_bytes
from werewolf.canonical_collection.game_bundle import _prefix_from_record
from werewolf.canonical_collection.pre import validate_authoritative_pre_prefix
from werewolf.observer_dtrs.cognition_gate_research import bind_inputs,validate_pre_inputs,profile,D3_DIGEST
from werewolf.observer_dtrs.dataset import open_dataset
from werewolf.observer_dtrs.sealed_d2_offline import SourceAnchors,read_formal_d2
from werewolf.observer_dtrs.decision import TiePolicy,Region
from werewolf.observer_dtrs.statistical_analysis import bootstrap_plan
from werewolf.observer_dtrs.d2_protocol import FrozenProtocol
from werewolf.observer_dtrs.public_behavior_research import (VERSION,PUBLIC_NAMES,LEGAL_NAMES,
    public_features,legal_features,observe_after_speech,fit_target,target_logits,row_scores,paired_game_cv,
    parse_public_game,public_day_witness,PUBLIC_AUDIT_ORIGIN_SHA256)

ANCHORS=SourceAnchors('870e75455572f4ce51dc3a0fc6d6ac050f658363d21ad130ba6616480bc831d1',
    '3ff578f97b9d4c3620a15a67357ddabd848d10fc','d25537be18da93ddc2f49be695f876db2edef5f31904fea0999a292f491abfa3',
    '32e30246cbadec8240f47291422a791b4419482e1386b4d8b570344a2e45acc5')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def save(path,value):
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def table(path,rows):
    rows=list(rows)
    if not rows:return
    with (gzip.open(path,'wt',newline='') if path.suffix=='.gz' else path.open('x',newline='')) as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def interval(x):return list(map(float,np.quantile(x,[.025,.975],method='linear')))
def records(raw):
    lines=raw.splitlines();r=[json.loads(x) for x in lines]
    if not raw.endswith(b'\n') or any(canonical_json_bytes(x)!=line for x,line in zip(r,lines)):
        raise ValueError('noncanonical source JSONL')
    return r
def manifest(raw):
    m=json.loads(raw)
    if raw!=canonical_json_bytes(m) or sha(canonical_json_bytes({k:v for k,v in m.items() if k!='manifest_digest'}))!=m['manifest_digest']:
        raise ValueError('source manifest integrity mismatch')
    return m


def argument_parser():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True,help='Verified formal D1c dataset directory')
    parser.add_argument('--d2-run',type=Path,required=True,help='Verified formal D2 run directory')
    parser.add_argument('--public-pre-file',type=Path,required=True,help='Admitted canonical public PRE JSONL')
    parser.add_argument('--public-archive',type=Path,required=True,help='Admitted public event/V1 ZIP archive')
    parser.add_argument('--plan',type=Path,required=True,help='Previously reviewed analysis plan')
    parser.add_argument('--output',type=Path,required=True,help='New independent output directory')
    return parser


def main():
    args=argument_parser().parse_args();out=args.output.absolute()
    dataset_root=args.dataset.absolute();d2_run=args.d2_run.absolute()
    public_pre_file=args.public_pre_file.absolute();public_archive=args.public_archive.absolute()
    protected=(ROOT,dataset_root,d2_run,public_pre_file.parent,public_archive.parent)
    if (out.exists() or not out.parent.is_dir() or '..' in out.parts or out.parent.resolve()!=out.parent
            or any(out.is_relative_to(x.resolve()) for x in protected)):raise ValueError('new external canonical output required')
    out.mkdir(mode=0o700);start=time.monotonic()
    plan=args.plan.read_bytes();(out/'analysis-plan.zh.md').write_bytes(plan)
    save(out/'start.json',{'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan_sha256':sha(plan),
        'public_validator_origin_sha256':PUBLIC_AUDIT_ORIGIN_SHA256,
        'scope':'CONDITIONAL_FIXED_HISTORICAL_POSTHOC_BEHAVIOR_PREDICTION_NOT_ACTION_EFFECT',
        'new_source_sha256':{str(f.relative_to(ROOT)):sha(f.read_bytes()) for f in
            [ROOT/'werewolf/observer_dtrs/public_behavior_research.py',ROOT/'scripts/analyze_observer_dtrs_public_behavior.py']}})
    d=open_dataset(dataset_root,expected_digest=ANCHORS.dataset_digest)
    v=read_formal_d2(d2_run,anchors=ANCHORS);b=v.batch('OBS-EQ1','game_equal');del v
    grouped=bind_inputs(d,b)
    if (len(d.rows),len(grouped),len(d.game_ids))!=(22685,5835,1500):raise ValueError('fixed D1 population mismatch')
    raw=public_pre_file.read_bytes()
    if sha(raw)!='ea40e0c4307105fe4e38a8ce1a2844b18cf9195fcb55c3a096ea9a3e1973f50a':raise ValueError('PRE external SHA mismatch')
    pres={};meta=[];pub=[];legal=[];cog=[];groups=defaultdict(list)
    for record in records(raw):
        pre=validate_authoritative_pre_prefix(_prefix_from_record(record));key=(pre.game_id,pre.boundary_id)
        if key in pres or key not in grouped:raise ValueError('PRE population duplicate/foreign')
        rows=grouped[key];validate_pre_inputs(pre,rows);pres[key]=pre
        for row in rows:
            c=row.legal.context;idx=len(meta);groups[c.game_id].append(idx)
            meta.append({'sample_id':row.legal.sample_id,'game_id':c.game_id,'boundary_id':c.boundary_id,'prefix_digest':c.prefix_digest,
                'step':c.step_index,'actor':c.acting_wolf,'observer':c.observer_id,'day':c.day,'phase':c.phase,'nW':c.wolf_count,
                'tom_fold':row.origin.tom_fold,'outer_fold':row.origin.outer_fold,'outer_reference':row.prediction_reference,
                'p':row.probability,'s':row.legal.features.s,'r':row.legal.features.r,
                'Q_row_digest':sha(canonical_json_bytes(asdict(row.legal.probability)))})
            pub.append(public_features(pre,c.observer_id));legal.append(legal_features(c));cog.append([row.legal.features.s,row.legal.features.r,row.probability])
    if set(pres)!=set(grouped) or len(meta)!=22685 or len({m['sample_id'] for m in meta})!=22685:raise ValueError('incomplete fixed population')
    ident=sorted((p.game_id,p.boundary_id,p.prefix_digest,p.step_index,p.current_speaker) for p in pres.values())
    if sha(canonical_json_bytes(ident))!='2e151e0ac08d794368f32782eab4943b9c23d56640e3a22c9df006a68c30b464':raise ValueError('fixed PRE identity set mismatch')
    first={g:min(meta[i]['step'] for i in ids) for g,ids in groups.items()}
    for m in meta:m['first']=m['step']==first[m['game_id']]
    X0=np.asarray(pub);XL=np.asarray(legal);C=np.asarray(cog);block=np.array([[m['tom_fold'],m['outer_fold']] for m in meta],int)
    # Public feature and legal cognition snapshots are persisted before reading any full future stream.
    np.savez_compressed(out/'features-before-future.npz',sample_ids=np.array([m['sample_id'] for m in meta]),public=X0,legal=XL,cognition_s_r_p=C)
    with gzip.open(out/'feature-identities.jsonl.gz','wt') as f:
        for m in meta:f.write(json.dumps(m,ensure_ascii=False)+'\n')
    save(out/'feature-boundary.json',{'feature_sha256':sha((out/'features-before-future.npz').read_bytes()),'rows':len(meta),
        'public_names':PUBLIC_NAMES,'legal_names':LEGAL_NAMES,'cognition_columns':['s','r','D1_event_p'],
        'created_before_opening_full_future_archive':True,'no_Y_role_sidecar_report_or_future_feature':True})
    print('FEATURES_FROZEN',len(meta),flush=True)
    archive=public_archive
    if sha(archive.read_bytes())!='e3d9dfdc095396346e97ec7d712417ffd37cff5515f9e0016afd58085e065947':raise ValueError('future public archive external SHA')
    outcomes={};days={};source_files=[]
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist())!=3004 or len(set(z.namelist()))!=3004:raise ValueError('archive file set mismatch')
        pm=manifest(z.read('source/publication-manifest.json'));fm=manifest(z.read('source/fold-manifest.json'))
        if (pm['manifest_digest']!='396ea0fa3f3f03dd1dfce8b25c0162ca489b992321d0c9e6f772eb2b0c4a2acd'
            or fm['manifest_digest']!=pm['development_fold_manifest_digest']
            or fm['manifest_digest']!='754fe0e517aab9e9237f0cb5097bde0daf8a43a09518b9691cfca67724539e69'):
            raise ValueError('publication/fold identity mismatch')
        expected={x['game_id']:x['bundle_digest'] for x in pm['development_games']}
        bundles=records(z.read('source/bundle-manifests.jsonl'))
        if len(bundles)!=1500 or set(expected)!=set(groups):raise ValueError('full public game population mismatch')
        for bundle in bundles:
            m=manifest(canonical_json_bytes(bundle));gid=m['game_id']
            if m['manifest_digest']!=expected[gid] or m['source_revision']!=pm['source_revision'] or not m['canonical_eligibility']:
                raise ValueError('bundle provenance mismatch')
            payload={}
            for name in ('public_event_stream.jsonl','speech_annotations_v1.jsonl'):
                raw=z.read(f'games/{gid}/{name}');spec=m['file_table']['public/'+name]
                if len(raw)!=spec['byte_size'] or sha(raw)!=spec['sha256']:raise ValueError('raw public file mismatch')
                source_files.append({'game_id':gid,'member':name,'bytes':len(raw),'sha256':sha(raw),'bundle_digest':m['manifest_digest']})
                payload[name]=records(raw)
            h,a=parse_public_game(payload['public_event_stream.jsonl'],payload['speech_annotations_v1.jsonl'])
            if set(a)!={e.event_id for e in h.events if e.event_type=='public_speech'}:raise ValueError('full public speech/V1 mismatch')
            for i in groups[gid]:
                r=meta[i];daykey=(gid,r['day'])
                if daykey not in days:days[daykey]={'game_id':gid,**public_day_witness(h,r['day'])}
                target=observe_after_speech(pres[gid,r['boundary_id']],r['observer'],h,a,days[daykey])
                sid=r['sample_id']
                if sid in outcomes:raise ValueError('duplicate outcome SID')
                outcomes[sid]={'sample_id':sid,**target}
    if set(outcomes)!={r['sample_id'] for r in meta}:raise ValueError('missing target identity')
    targets=[outcomes[r['sample_id']] for r in meta]
    with gzip.open(out/'outcome-evidence.jsonl.gz','wt') as f:
        for t in targets:f.write(json.dumps(t,ensure_ascii=False)+'\n')
    save(out/'public-source-files.json',source_files)
    mask=np.array([t['status']=='OBSERVED' for t in targets]);idx=np.flatnonzero(mask)
    Z=np.array([targets[i]['Z'] for i in idx],dtype=int);M=[meta[i] for i in idx];G=[m['game_id'] for m in M];B=block[idx]
    coverage=[]
    populations={'all':np.ones(len(meta),bool),'first':np.array([m['first'] for m in meta])}
    for phase in ('discussion','pk_discussion'):
        for nw in (1,2):populations[f'{phase}/{nw}']=np.array([m['phase']==phase and m['nW']==nw for m in meta])
    for name,sel in populations.items():
        counts=Counter(targets[i]['status'] for i in np.flatnonzero(sel));observed=np.flatnonzero(sel&mask)
        coverage.append({'population':name,'rows':int(sel.sum()),'PREs':len({(meta[i]['game_id'],meta[i]['boundary_id']) for i in np.flatnonzero(sel)}),
            'games':len({meta[i]['game_id'] for i in np.flatnonzero(sel)}),'observed':len(observed),
            'observed_games':len({meta[i]['game_id'] for i in observed}),'positive':sum(targets[i]['Z'] for i in observed),
            'negative':sum(1-targets[i]['Z'] for i in observed),'status_counts':json.dumps(dict(counts),sort_keys=True)})
    table(out/'coverage.csv',coverage)
    print('COVERAGE',json.dumps(coverage[0]),flush=True)
    if len(set(Z))!=2:raise ValueError('UNSUPPORTED_GLOBAL_CLASS_SUPPORT')
    # The same games/rows and source fold restrictions apply to every representation.
    folds=list(paired_game_cv(B,G))
    design={};names={}
    def add(name,arrays,columns):
        design[name]=np.column_stack(arrays)[idx];names[name]=list(columns)
    add('M0_PUBLIC',[X0],PUBLIC_NAMES)
    add('L0_LEGAL',[X0,XL],PUBLIC_NAMES+LEGAL_NAMES)
    for prefix,base,base_names in [('M',[X0],PUBLIC_NAMES),('L',[X0,XL],PUBLIC_NAMES+LEGAL_NAMES)]:
        add(prefix+'1_s',base+[C[:,0]],base_names+('s',))
        add(prefix+'1_r',base+[C[:,1]],base_names+('r',))
        add(prefix+'1_sr',base+[C[:,:2]],base_names+('s','r'))
        add(prefix+'2_p',base+[C[:,2]],base_names+('D1_event_p',))
    region_codes=[];defer_codes=[]
    for sc in range(45):
        regions=np.array([('POS','BND','NEG').index(profile(float(p),sc,TiePolicy.TERMINAL_NEG_FIRST)[0].region.value) for p in C[:,2]],dtype=np.uint8)
        region_codes.append(regions)
        defer_codes.append(np.array([('POS','BND','NEG').index(profile(float(p),sc,TiePolicy.DEFER_FIRST)[0].region.value) for p in C[:,2]],dtype=np.uint8))
        dummy=np.column_stack([regions==0,regions==1]).astype(float)
        add(f'M3_C{sc:02}',[X0,dummy],PUBLIC_NAMES+('region_POS','region_BND'))
        if sc==22:
            add('M2_p_rich_C22',[X0,C[:,2],dummy],PUBLIC_NAMES+('D1_event_p','region_POS','region_BND'))
            add('L3_C22',[X0,XL,dummy],PUBLIC_NAMES+LEGAL_NAMES+('region_POS','region_BND'))
            add('L2_p_rich_C22',[X0,XL,C[:,2],dummy],PUBLIC_NAMES+LEGAL_NAMES+('D1_event_p','region_POS','region_BND'))
    predictions={};logits={};fit_records=[];folds_audit=[]
    for (source,outer),train,test in folds:
        if set(G[i] for i in train)&set(G[i] for i in test):raise ValueError('game split leakage')
        folds_audit.append({'source_fold':source,'outer_fold':outer,'train_games':sorted({G[i] for i in train}),
            'test_games':sorted({G[i] for i in test}),'train_rows':len(train),'test_rows':len(test),
            'train_positive':int(Z[train].sum()),'test_positive':int(Z[test].sum())})
    save(out/'split-audit.json',folds_audit)
    for name,X in design.items():
        oof=np.full(len(idx),np.nan);seen=np.zeros(len(idx),int)
        for (source,outer),train,test in folds:
            model=fit_target(X[train],Z[train],[G[i] for i in train])
            model.update(model_id=name,source_fold=source,outer_fold=outer,feature_names=names[name],
                training_fingerprint=sha(canonical_json_bytes([[M[i]['sample_id'],int(Z[i]),X[i].tolist()] for i in train])),
                test_sample_identity_digest=sha(canonical_json_bytes(sorted(M[i]['sample_id'] for i in test))))
            oof[test]=target_logits(model,X[test]);seen[test]+=1;fit_records.append(model)
        if not np.all(seen==1) or not np.isfinite(oof).all():raise ValueError('missing/duplicate new-Z OOF prediction')
        logits[name]=oof;predictions[name]=row_scores(oof,Z)[0]
        print('MODEL_COMPLETE',name,flush=True)
    save(out/'fit-records.json',fit_records)
    np.savez_compressed(out/'predictions.npz',sample_ids=np.array([m['sample_id'] for m in M]),Z=Z,
        model_ids=np.array(list(predictions)),logits=np.stack(list(logits.values())),probabilities=np.stack(list(predictions.values())),
        all_sample_ids=np.array([m['sample_id'] for m in meta]),eligible_indices=idx,region_codes=np.asarray(region_codes),defer_codes=np.asarray(defer_codes))
    eligible_blocks=defaultdict(set)
    for m in M:eligible_blocks[m['tom_fold'],m['outer_fold']].add(m['game_id'])
    d2=FrozenProtocol((d2_run/'contract/protocol.json').read_bytes())
    games,draws=bootstrap_plan(eligible_blocks,protocol=d2,formal=False)
    np.save(out/'bootstrap-indices.npy',draws);save(out/'bootstrap-game-order.json',games)
    order={g:j for j,g in enumerate(games)};gi=np.array([order[g] for g in G]);sizes=np.bincount(gi,minlength=len(games))
    bootstrap_counts=np.array([np.bincount(x,minlength=len(games)) for x in draws],float)
    metrics={};results=[];calibration=[];subgroup=[]
    eval_groups={'all':np.ones(len(idx),bool),'first':np.array([m['first'] for m in M]),
        **{f'{ph}/{nw}':np.array([m['phase']==ph and m['nW']==nw for m in M]) for ph in ('discussion','pk_discussion') for nw in (1,2)}}
    for name,z in logits.items():
        p,bs,ll=row_scores(z,Z)
        matrix=np.column_stack([np.bincount(gi,weights=x,minlength=len(games))/sizes for x in (bs,ll,p-Z)])
        metrics[name]=matrix;boots=bootstrap_counts@matrix/len(games)
        result={'model':name,'rows':len(Z),'games':len(games),'positive':int(Z.sum()),'brier':float(matrix[:,0].mean()),
            'log_loss':float(matrix[:,1].mean()),'bias':float(matrix[:,2].mean()),'row_equal_brier':float(bs.mean()),'row_equal_log_loss':float(ll.mean()),
            'brier_ci_low':interval(boots[:,0])[0],'brier_ci_high':interval(boots[:,0])[1],
            'log_loss_ci_low':interval(boots[:,1])[0],'log_loss_ci_high':interval(boots[:,1])[1],'ECE_10':0.}
        weights=1/(len(games)*sizes[gi]);bins=np.minimum(9,(p*10).astype(int))
        for k in range(10):
            keep=bins==k;mass=float(weights[keep].sum())
            mp=float(weights[keep]@p[keep]/mass) if mass else None;rate=float(weights[keep]@Z[keep]/mass) if mass else None
            calibration.append({'model':name,'bin':k,'rows':int(keep.sum()),'games':len(set(np.array(G)[keep])),
                'mass':mass,'mean_prediction':mp,'observed_rate':rate})
            if mass:result['ECE_10']+=mass*abs(mp-rate)
        results.append(result)
        for group,keep in eval_groups.items():
            den=np.bincount(gi[keep],minlength=len(games));active=den>0
            subgroup.append({'model':name,'group':group,'rows':int(keep.sum()),'games':int(active.sum()),'positive':int(Z[keep].sum()),
                'brier':float((np.bincount(gi[keep],weights=bs[keep],minlength=len(games))[active]/den[active]).mean()) if active.any() else None,
                'log_loss':float((np.bincount(gi[keep],weights=ll[keep],minlength=len(games))[active]/den[active]).mean()) if active.any() else None})
    table(out/'model-results.csv',results);table(out/'calibration.csv',calibration);table(out/'subgroup-results.csv',subgroup)
    comparisons=[]
    pairs=[(n,'M0_PUBLIC') for n in metrics if n!='M0_PUBLIC']
    pairs += [(n,'L0_LEGAL') for n in metrics if n.startswith('L') and n!='L0_LEGAL']
    pairs += [('M1_s','M1_r'),('M2_p','M1_s'),('M2_p','M1_sr'),('M3_C22','M2_p'),('M3_C22','M2_p_rich_C22'),
        ('L1_s','L1_r'),('L2_p','L1_s'),('L2_p','L1_sr'),('L3_C22','L2_p'),('L3_C22','L2_p_rich_C22')]
    for a,b in pairs:
        difference=metrics[a]-metrics[b];boots=bootstrap_counts@difference/len(games)
        comparisons.append({'model':a,'reference':b,'brier_difference':float(difference[:,0].mean()),
            'brier_ci_low':interval(boots[:,0])[0],'brier_ci_high':interval(boots[:,0])[1],
            'log_loss_difference':float(difference[:,1].mean()),'log_loss_ci_low':interval(boots[:,1])[0],'log_loss_ci_high':interval(boots[:,1])[1]})
    table(out/'paired-comparisons.csv',comparisons)
    # Descriptive panel outcomes: panel H uses ALL legal observers before observing their future opportunities.
    panel_groups=defaultdict(list)
    for i,m in enumerate(meta):panel_groups[m['game_id'],m['boundary_id']].append(i)
    panel=[]
    codes=region_codes[22]
    for key,all_ids in panel_groups.items():
        available=[i for i in all_ids if mask[i]];m=meta[all_ids[0]]
        panel.append({'game_id':key[0],'boundary_id':key[1],'phase':m['phase'],'nW':m['nW'],'first':m['first'],
            'legal_observers':len(all_ids),'observed_observers':len(available),'panel_C22_mixed':len(set(codes[all_ids]))>1,
            'p_range':float(np.ptp(C[all_ids,2])),'behavior_mixed':len({targets[i]['Z'] for i in available})>1 if len(available)>=2 else None,
            'positive_count':sum(targets[i]['Z'] for i in available),'candidate_value':'NOT_DEFINED'})
    table(out/'panel-behavior.csv',panel)
    summary={'status':'HISTORICAL_PUBLIC_BEHAVIOR_PREDICTION_COMPLETED','version':VERSION,'source_anchors':asdict(ANCHORS),
        'D3_digest':D3_DIGEST,'scope':'fixed_historical_development_not_causal_not_external_TEST_not_action_utility',
        'coverage':coverage,'models':len(predictions),'fits':len(fit_records),'outer_blocks':len(folds),
        'bootstrap':{'replicates':len(draws),'games':len(games),'index_raw_sha256':sha(draws.astype('<i8').tobytes()),
            'strata':'original_ToM_source_x_D2_outer; eligible_games_only','seed':d2.record['uncertainty']['seed'],'not_original_full_population_indices':True},
        'same_class_support_all_models':True,'single_wolf_s_equals_r':bool(np.all(C[np.array([m['nW']==1 for m in meta]),0]==C[np.array([m['nW']==1 for m in meta]),1])),
        'D3_tie_policy_real_row_flips':int(np.sum(np.asarray(region_codes)!=np.asarray(defer_codes))),
        'private_D1_supervision_opened':False,'D2_reader_internal_mixed_rows_note':'only extracted label-free batch used as predictors',
        'all_pre_features_frozen_before_future_archive_read':True,'seconds':time.monotonic()-start}
    save(out/'summary.json',summary);print('COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
