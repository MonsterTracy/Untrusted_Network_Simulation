"""Fixed historical PRE cognition/basis diagnostics; no labels, values or policy execution.

Outputs derived research evidence only. All public-cue gate results are explicitly
factorized, checked against bounded calls to the unchanged original G/P0 functions.
"""
from argparse import ArgumentParser
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from werewolf.observer_dtrs.cognition_gate_research import (ARMS, PUBLIC_RULE, bind_inputs,
    public_reply_catalog, profile, audit_pre, digest, D3_DIGEST)
from scripts.speech_planning import generate_candidates
from werewolf.artifact_io import canonical_json_bytes
from werewolf.canonical_collection.game_bundle import _prefix_from_record
from werewolf.canonical_collection.pre import validate_authoritative_pre_prefix
from werewolf.observer_dtrs.d2_protocol import FrozenProtocol
from werewolf.observer_dtrs.d3_protocol import load_protocol, cost_edges
from werewolf.observer_dtrs.dataset import open_dataset
from werewolf.observer_dtrs.decision import Region, TiePolicy
from werewolf.observer_dtrs.sealed_d2_offline import SourceAnchors, read_formal_d2
from werewolf.observer_dtrs.statistical_analysis import bootstrap_plan
from werewolf.observer_dtrs.speech_evidence import select_observer

ANCHORS = SourceAnchors('870e75455572f4ce51dc3a0fc6d6ac050f658363d21ad130ba6616480bc831d1',
    '3ff578f97b9d4c3620a15a67357ddabd848d10fc',
    'd25537be18da93ddc2f49be695f876db2edef5f31904fea0999a292f491abfa3',
    '32e30246cbadec8240f47291422a791b4419482e1386b4d8b570344a2e45acc5')
QSHA = 'ea40e0c4307105fe4e38a8ce1a2844b18cf9195fcb55c3a096ea9a3e1973f50a'
BSHA = '33e94ab2b2875ff7f47de212062a628de613de37e4b8f3ff0b887eb479f0a3da'
CELLS = {'discussion/1':1238, 'discussion/2':4158, 'pk_discussion/1':154, 'pk_discussion/2':285}
POLICIES = (TiePolicy.TERMINAL_NEG_FIRST, TiePolicy.DEFER_FIRST)
REGIONS = (Region.POS, Region.BND, Region.NEG)
ARM_NAMES = ('A0_RECORD_ONLY', 'A1_FORCED_BINARY', 'A2_THREE_WAY', 'A3_CONTINUOUS_EQUIVALENCE')


def sha(raw): return hashlib.sha256(raw).hexdigest()


def save(root, name, value):
    path=root/name
    with path.open('x') as f: json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n')


def table(path, rows):
    rows=list(rows)
    if not rows: return
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path, 'wt', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def identity(c): return (c.game_id,c.boundary_id,c.prefix_digest,c.step_index,c.acting_wolf)


def argument_parser():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True,help='Verified formal D1c dataset directory')
    parser.add_argument('--d2-run',type=Path,required=True,help='Verified formal D2 run directory')
    parser.add_argument('--public-pre-file',type=Path,required=True,help='Admitted canonical public PRE JSONL')
    parser.add_argument('--output',type=Path,required=True,help='New independent output directory')
    return parser


def main():
    args=argument_parser().parse_args();out=args.output.absolute()
    dataset_root=args.dataset.absolute();d2_run=args.d2_run.absolute();public_pre_file=args.public_pre_file.absolute()
    protected=(ROOT,dataset_root,d2_run,public_pre_file.parent)
    if (out.exists() or not out.parent.is_dir() or '..' in out.parts or out.parent.resolve()!=out.parent
            or any(out.is_relative_to(x.resolve()) for x in protected)): raise ValueError('new canonical external output directory required')
    out.mkdir(mode=0o700);started=time.monotonic()
    save(out,'analysis-plan.json',{'scope':'POSTHOC_FIXED_HISTORICAL_PREDICTIONS_NOT_ACTION_EFFECT',
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'population':{'games':1500,'PREs':5835,'observer_rows':22685,'candidates':96200,'cells':CELLS},
        'model':'OBS-EQ1','weighting':'game_equal','costs':'all45 frozen','tie_policies':[p.value for p in POLICIES],
        'arms':list(ARM_NAMES),'basis_rule':PUBLIC_RULE,
        'basis_semantics':'verified public accusation reply cue only; no sufficient argument or Y dependency claimed',
        'cognitive_bases':'UNSUPPORTED_NO_VERIFIED_ACTION_PREMISE_GENERATOR',
        'action_values':'UNAVAILABLE','selected_plan':None,'diagnostic_candidate_order_selection':'NOT_USED',
        'public_gate_expansion':'PUBLIC_ONLY factorized all45 x2 ties x2 panels x4 arms; not 1M original P0 calls',
        'bounded_original_check':'identity-first PRE of each phase/nW cell: C00,C22 both ties; shortest raw PRE: all45 both ties; no outcome/probability selection',
        'first_opportunity':'minimum canonical step per game, independent of p/V1/outcome',
        'panel_metrics':'per-PRE fractions/counts, no probability average used as policy input',
        'inference':'whole-game equal mean of PRE diagnostics; first has one PRE/game; 2000 original source/outer-stratified game draws; no refits',
        'CI_scope':'overall/all and first only; phase/nW/seat descriptive; no absent group games filled with zero',
        'labels_roles_support_future_results_used':False,'scientific_claim':'representation and admissibility diagnostic, not causal/utility evidence'})
    protocol=load_protocol();protocol.require_frozen(D3_DIGEST)
    dataset=open_dataset(dataset_root,expected_digest=ANCHORS.dataset_digest)
    view=read_formal_d2(d2_run,anchors=ANCHORS);batch=view.batch('OBS-EQ1','game_equal')
    del view  # Drop the mixed sealed reader after extracting its label-free batch.
    grouped=bind_inputs(dataset,batch)
    if len(grouped)!=5835 or len(dataset.rows)!=22685 or len(dataset.game_ids)!=1500: raise ValueError('fixed D1 population mismatch')
    first={}
    for rows in grouped.values():
        c=rows[0].legal.context
        if c.game_id not in first or c.step_index<first[c.game_id][0]: first[c.game_id]=(c.step_index,c.boundary_id)
    payload=public_pre_file.read_bytes()
    if sha(payload)!=QSHA: raise ValueError('public PRE external SHA mismatch')
    meta=[];sample_rows=[];sample_pre=[];seen=set();facts_count=Counter();cell_count=Counter();bounded={};shortest=None
    with gzip.open(out/'pre-evidence.jsonl.gz','wt') as writer:
        for line in payload.splitlines():
            pre=validate_authoritative_pre_prefix(_prefix_from_record(json.loads(line)))
            if canonical_json_bytes(pre.to_record())!=line: raise ValueError('canonical PRE roundtrip mismatch')
            key=(pre.game_id,pre.boundary_id)
            if key not in grouped or key in seen: raise ValueError('missing/duplicate/foreign PRE')
            seen.add(key);rows=grouped[key];c=rows[0].legal.context
            catalog,facts=public_reply_catalog(pre,rows);cell=f'{c.phase}/{c.wolf_count}';cell_count[cell]+=1
            selection=select_observer(pre,c.known_wolf_ids)
            candidates=catalog.candidates;cue=len(catalog.bases);index=len(meta)
            selected=next(i for i,r in enumerate(rows) if r.legal.context.observer_id==selection.observer_id)
            indices=[]
            for row in rows: indices.append(len(sample_rows));sample_rows.append(row);sample_pre.append(index)
            info={'identity':identity(c),'phase':c.phase,'nW':c.wolf_count,'day':c.day,
                'first':first[c.game_id][1]==c.boundary_id,'legal_candidates':len(candidates),'public_bases':cue,
                'assessable_candidates':int(cue>0),'no_declared_basis_candidates':len(candidates)-int(cue>0),
                'admitted_basis_ids':[b.basis_id for b in catalog.bases],
                'assessable_plans':[catalog.bases[0].plan.public_payload()] if cue else [],
                'selected_observer':selection.observer_id,'selection_reason':selection.reason,'selection_relation':selection.relation,
                'observer_ids':[r.legal.context.observer_id for r in rows],
                'sample_ids':[r.legal.sample_id for r in rows],'prediction_reference':rows[0].prediction_reference,
                'tom_fold':rows[0].origin.tom_fold,'outer_fold':rows[0].origin.outer_fold,
                'block_digest':rows[0].origin.block_digest,
                'event_probabilities':[r.probability for r in rows],
                'Q_row_digests':[digest(asdict(r.legal.probability)) for r in rows],
                'sample_indices':indices,'selected_sample_index':indices[selected],
                'p_min':min(r.probability for r in rows),'p_max':max(r.probability for r in rows),
                'public_facts':facts,'catalog_digest':catalog.digest,'public_PRE_record_sha256':sha(line),
                'status':'ACTION_VALUE_UNAVAILABLE' if cue else 'NO_ADMISSIBLE_BASIS','selected_plan':None}
            meta.append(info);facts_count.update(f['status'] for f in facts)
            writer.write(json.dumps({k:v for k,v in info.items() if k not in ('sample_indices','selected_sample_index')},ensure_ascii=False)+'\n')
            if cell not in bounded or info['identity']<bounded[cell][0]: bounded[cell]=(info['identity'],pre,rows)
            sk=(len(line),info['identity'])
            if shortest is None or sk<shortest[0]: shortest=(sk,pre,rows)
    del payload
    if seen!=set(grouped) or dict(cell_count)!=CELLS or sum(m['legal_candidates'] for m in meta)!=96200: raise ValueError('full PRE/candidate population mismatch')
    if sum(m['first'] for m in meta)!=1500: raise ValueError('first-PRE population mismatch')
    if digest(sorted(m['identity'] for m in meta))!='2e151e0ac08d794368f32782eab4943b9c23d56640e3a22c9df006a68c30b464':
        raise ValueError('fixed original D1 PRE identity set mismatch')
    # No future event/report/label is consulted by the above prebinding.
    cases=[]
    save(out,'bounded-original-check-plan.json',{'cells':[{'identity':x[0],'scenarios':[0,22]} for x in bounded.values()],
        'shortest_PRE':shortest[0][1],'shortest_scenarios':list(range(45)),'policies':[p.value for p in POLICIES]})
    checks=[(p,r,(0,22)) for _,p,r in bounded.values()]+[(shortest[1],shortest[2],range(45))]
    for pre,rows,scenarios in checks:
        catalog,_=public_reply_catalog(pre,rows);count=int(bool(catalog.bases));basis_ids=[b.basis_id for b in catalog.bases]
        for sc in scenarios:
            for policy in POLICIES:
                result=audit_pre(pre,rows,scenario=sc,policy=policy)
                for panel in ('full','selected'):
                    for mode in ARMS:
                        a=result[panel][mode.value]
                        assert a['selected_plan'] is None and a['assessable_candidate_count']==count
                        assert a['admitted_basis_count']==len(basis_ids) and a['legal_candidate_count']==len(catalog.candidates)
                        assert [x['plan'] for x in a['candidates'] if x['assessable']]==([catalog.bases[0].plan.public_payload()] if count else [])
                cases.append({'identity':identity(rows[0].legal.context),'scenario':f'C{sc:02}','policy':policy.value,
                    'all8_original_calls_match_PUBLIC_ONLY_factorization':True,'assessable_candidates':count,'actual_selected_plan':None})
    save(out,'bounded-original-checks.json',{'checks':cases,'original_appraisal_calls':8*len(cases),'mismatches':0})
    # Freeze whole-game bootstrap from the original D2 protocol and verify byte identity.
    blocks=defaultdict(set)
    for r in sample_rows: blocks[(r.origin.tom_fold,r.origin.outer_fold)].add(r.origin.game_id)
    d2=FrozenProtocol((d2_run/'contract/protocol.json').read_bytes())
    games,draws=bootstrap_plan(blocks,protocol=d2,formal=True)
    if sha(draws.astype('<i8').tobytes())!=BSHA: raise ValueError('original bootstrap draw mismatch')
    if (d2_run/'complete/bootstrap_indices.int64le').read_bytes()!=draws.astype('<i8').tobytes(): raise ValueError('sealed/bootstrap mismatch')
    bootstrap_counts=np.asarray([np.bincount(row,minlength=len(games)) for row in draws],dtype=float)
    gi={g:j for j,g in enumerate(games)};pre_game=np.array([gi[m['identity'][0]] for m in meta]);sp=np.array(sample_pre)
    nobs=np.bincount(sp,minlength=len(meta));assert np.all(nobs>1), 'no other-observer denominator'
    sel=np.array([m['selected_sample_index'] for m in meta]);p=np.array([r.probability for r in sample_rows])
    unique,inv=np.unique(p,return_inverse=True);first_mask=np.array([m['first'] for m in meta]);N=len(meta)
    groups={'all':np.ones(N,dtype=bool),'first':first_mask}
    for cell in CELLS: groups['phase_nW/'+cell]=np.array([f'{m["phase"]}/{m["nW"]}'==cell for m in meta])
    for name,mask in list(groups.items()):
        if name.startswith('phase_nW/'): groups['first/'+name]=mask&first_mask
    cost_rows=[];stratum_rows=[];seat_rows=[];all_codes={};all_binary={};ties_map={};game_rows=[]
    metrics_ci=('panel_mixed_regions','selected_rest_disagreement_fraction','PRE_has_BND','any_three_binary_difference','BND_observer_fraction','tie_PRE')
    for policy in POLICIES:
        for sc in range(45):
            decisions=[profile(float(q),sc,policy) for q in unique]
            code=np.array([REGIONS.index(d.region) for d,b in decisions],dtype=np.uint8)[inv]
            binary=np.array([REGIONS.index(b) for d,b in decisions],dtype=np.uint8)[inv]
            ties=np.array([len(d.minimizing_actions)>1 for d,b in decisions],dtype=bool)[inv]
            all_codes[policy,sc]=code;all_binary[policy,sc]=binary;ties_map[policy,sc]=ties
            counts=np.stack([np.bincount(sp,weights=code==k,minlength=N) for k in range(3)])
            selected=code[sel];delta=code!=binary
            fields={'panel_mixed_regions':np.sum(counts>0,axis=0)>=2,
                'selected_disagrees_with_some':counts[selected,np.arange(N)]<nobs,
                'selected_rest_disagreement_fraction':(nobs-counts[selected,np.arange(N)])/(nobs-1),
                'PRE_has_BND':counts[1]>0,'any_three_binary_difference':np.bincount(sp,weights=delta,minlength=N)>0,
                'BND_observer_fraction':counts[1]/nobs,'tie_PRE':np.bincount(sp,weights=ties,minlength=N)>0,
                'POS_observer_fraction':counts[0]/nobs,'NEG_observer_fraction':counts[2]/nobs,
                'selected_BND':selected==1,'public_basis_available':np.array([m['public_bases']>0 for m in meta])}
            for group,mask in groups.items():
                row={'scenario':f'C{sc:02}','policy':policy.value,'population':group,
                    'PREs':int(mask.sum()),'observer_rows':int(nobs[mask].sum()),'games':len(set(pre_game[mask]))}
                for name,values in fields.items(): row[name+'_PRE_mean']=float(np.mean(values[mask])) if mask.any() else None
                row.update({r.value+'_observer_rows':int(counts[k,mask].sum()) for k,r in enumerate(REGIONS)})
                stratum_rows.append(row)
                if group not in ('all','first'):continue
                denom=np.bincount(pre_game[mask],minlength=len(games));assert np.all(denom>0)
                game_matrix=np.column_stack([np.bincount(pre_game[mask],weights=np.asarray(fields[name],float)[mask],minlength=len(games))/denom for name in metrics_ci])
                boots=bootstrap_counts@game_matrix/len(games);cr=dict(row)
                for j,name in enumerate(metrics_ci):
                    cr[name+'_game_equal']=float(game_matrix[:,j].mean());cr[name+'_ci_low'],cr[name+'_ci_high']=map(float,np.quantile(boots[:,j],[.025,.975],method='linear'))
                cost_rows.append(cr)
                for i,g in enumerate(games): game_rows.append({'scenario':f'C{sc:02}','policy':policy.value,'population':group,'game_id':g,'PREs':int(denom[i]),**{name:float(game_matrix[i,j]) for j,name in enumerate(metrics_ci)}})
            for scope,pm in [('all',np.ones(N,bool)),('first',first_mask)]:
                for seat in sorted({r.legal.context.observer_id for r in sample_rows}):
                    sm=np.array([r.legal.context.observer_id==seat for r in sample_rows])&pm[sp]
                    seat_rows.append({'scenario':f'C{sc:02}','policy':policy.value,'population':scope,'observer':seat,
                        'rows':int(sm.sum()),'games':len({sample_rows[j].origin.game_id for j in np.flatnonzero(sm)}),
                        **{r.value:int(np.sum(code[sm]==k)) for k,r in enumerate(REGIONS)},'ties':int(ties[sm].sum()),'three_binary_diff':int(delta[sm].sum())})
            print(json.dumps({'done':f'{policy.value}/C{sc:02}','seconds':time.monotonic()-started}),flush=True)
    # Dense derived decisions, not published formal D3 rows; no targets/labels.
    np.savez_compressed(out/'derived-observer-decisions.npz',
        sample_ids=np.asarray([r.legal.sample_id for r in sample_rows]),
        probabilities=p,pre_index=sp,
        regions=np.stack([[all_codes[policy,sc] for sc in range(45)] for policy in POLICIES]),
        binary_regions=np.stack([[all_binary[policy,sc] for sc in range(45)] for policy in POLICIES]),
        exact_ties=np.stack([[ties_map[policy,sc] for sc in range(45)] for policy in POLICIES]),
        policies=np.asarray([p.value for p in POLICIES]),region_codes=np.asarray([r.value for r in REGIONS]))
    ranges=np.asarray([m['p_max']-m['p_min'] for m in meta]);panel_ranges=[]
    for group,mask in groups.items():
        v=ranges[mask]
        panel_ranges.append({'population':group,'PREs':int(mask.sum()),'games':len(set(pre_game[mask])),
            'nonidentical_p_PREs':int(np.sum(v>0)),
            'p_range_q25':float(np.quantile(v,.25)) if len(v) else None,
            'p_range_median':float(np.median(v)) if len(v) else None,
            'p_range_q75':float(np.quantile(v,.75)) if len(v) else None,
            'p_range_max':float(v.max()) if len(v) else None})
    table(out/'panel-probability-ranges.csv',panel_ranges)
    edge_rows=[]
    for policy in POLICIES:
        for a,b,axis in cost_edges():
            flip=all_codes[policy,a]!=all_codes[policy,b];preflip=np.bincount(sp,weights=flip,minlength=N)>0
            for group in ('all','first'):
                mask=groups[group];edge_rows.append({'policy':policy.value,'population':group,'from':f'C{a:02}','to':f'C{b:02}','axis':axis,
                    'PREs':int(mask.sum()),'observer_rows':int(nobs[mask].sum()),'observer_region_flips':int(flip[mask[sp]].sum()),
                    'PRE_any_region_flip':int(preflip[mask].sum()),'selected_observer_flips':int((all_codes[policy,a][sel][mask]!=all_codes[policy,b][sel][mask]).sum()),
                    'gate_admission_changes':0,'actual_action_changes':'UNKNOWN_NO_ACTION_VALUE_OR_SELECTION'})
    tie_rows=[]
    for sc in range(45):
        flip=all_codes[POLICIES[0],sc]!=all_codes[POLICIES[1],sc]
        for group in ('all','first'):
            mask=groups[group];tie_rows.append({'scenario':f'C{sc:02}','population':group,'PREs':int(mask.sum()),'observer_rows':int(nobs[mask].sum()),
                'observer_region_flips':int(flip[mask[sp]].sum()),'PRE_any_region_flip':int(np.sum(np.bincount(sp,weights=flip,minlength=N)[mask]>0)),
                'selected_observer_flips':int(flip[sel][mask].sum()),'gate_admission_changes':0})
    gate_rows=[];cue=np.array([m['public_bases'] for m in meta]);legal=np.array([m['legal_candidates'] for m in meta]);yes=cue>0
    for group,mask in groups.items():
        for sc in range(45):
            for policy in POLICIES:
                for panel in ('selected','full'):
                    for arm in ARM_NAMES:
                        gate_rows.append({'population':group,'scenario':f'C{sc:02}','policy':policy.value,'panel':panel,'arm':arm,
                            'PREs':int(mask.sum()),'legal_candidates':int(legal[mask].sum()),'admitted_plans':int(yes[mask].sum()),'admitted_bases':int(cue[mask].sum()),
                            'ACTION_VALUE_UNAVAILABLE_PREs':int((yes&mask).sum()),'NO_ADMISSIBLE_BASIS_PREs':int((~yes&mask).sum()),
                            'selected_plan_count':0,'cognitive_basis_count':0,'computation':'PUBLIC_ONLY_ALGEBRAIC_EXPANSION_BOUNDED_ORIGINAL_CHECKED'})
    table(out/'costs.csv',cost_rows);table(out/'strata.csv',stratum_rows);table(out/'observer-seats.csv',seat_rows)
    table(out/'game-metrics.csv.gz',game_rows);table(out/'cost-edges.csv',edge_rows);table(out/'tie-sensitivity.csv',tie_rows);table(out/'gate-arms.csv.gz',gate_rows)
    static={'source_anchors':asdict(ANCHORS),'D3_protocol':protocol.digest,'public_PRE_sha256':QSHA,'D1_legal_rows':len(dataset.rows),
        'public_PREs':N,'games':len(games),'legal_candidates':sum(m['legal_candidates'] for m in meta),'first_PREs':int(first_mask.sum()),'four_cells':dict(cell_count),
        'complete_identity_set_digest':digest(sorted(m['identity'] for m in meta)),
        'expected_identity_set_digest':'2e151e0ac08d794368f32782eab4943b9c23d56640e3a22c9df006a68c30b464',
        'sample_set_digest':digest(sorted(r.legal.sample_id for r in sample_rows)),
        'public_basis_PREs':sum(m['public_bases']>0 for m in meta),'public_bases':sum(m['public_bases'] for m in meta),'facts_statuses':dict(facts_count),
        'cognitive_bases':'UNSUPPORTED','actual_plan_selection':'NOT_PERFORMED_ACTION_VALUE_UNAVAILABLE','actual_action_or_target_difference':'UNKNOWN_NOT_ZERO',
        'gate_admission_difference_across_arms_costs_panels':0,'basis_identity_difference_across_arms_costs_panels':0,
        'A2_A3_region_mismatches':0,'A2_A3_scope':'original exact risk argmin per cached identical p/scenario/tie; not novel information',
        'bootstrap':{'sha256':BSHA,'replicates':2000,'strata':'ToM_source_fold_x_outer_fold','game_order_digest':digest(games),'estimand':'game equal mean of within-game PRE diagnostic rates, not D2 scoring population weights'},
        'prebinding_order':'plan persisted before input result computation','supervision_targets_handle_opened':False,'D1_restricted_opened':False,
        'reader_note':'read_formal_d2 internally validates mixed sealed D2 rows; only its label-free batch is consumed, no target/support handler',
        'D2_batch_identity':{k:batch.identity[k] for k in ('model_id','weighting','run_identity_digest','predictions_digest','origins_digest')},
        'new_source_sha256':{str(Path(__file__).relative_to(ROOT)):sha(Path(__file__).read_bytes()),
            'werewolf/observer_dtrs/cognition_gate_research.py':sha((ROOT/'werewolf/observer_dtrs/cognition_gate_research.py').read_bytes())},
        'dense_decision_scope':'DERIVED_THIS_STUDY_NOT_FORMAL_D3_ARTIFACT; policy x scenario x sample; no Y',
        'profile_cache':profile.cache_info()._asdict(),'seconds':time.monotonic()-started,'scope':'FIXED_HISTORICAL_DEV_POSTHOC_NOT_CAUSAL_OR_ACTION_UTILITY'}
    assert static['complete_identity_set_digest']==static['expected_identity_set_digest']
    save(out,'summary.json',static);print(json.dumps(static,ensure_ascii=False),flush=True)


if __name__=='__main__': main()
