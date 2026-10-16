"""Resumable fixed-probability D3 evaluation. No estimator fitting or cost selection."""
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
import math
from pathlib import Path

import numpy as np

from werewolf.artifact_io import canonical_json_bytes, read_artifact_file, verify_artifact
from werewolf.observer_dtrs.decision import (
    Region, TiePolicy, LossSpec, decide, forced_binary, thresholds as loss_thresholds,
)
from werewolf.observer_dtrs.decision_offline import realized_loss
from werewolf.observer_dtrs.d3_stability import interval_summary_engine
from werewolf.observer_dtrs.d3_inputs_offline import D3Inputs, controlled_inputs, prepare_formal
from werewolf.observer_dtrs.d3_protocol import load_protocol, checked_source, cost_edges
from werewolf.observer_dtrs.statistical_analysis import MODELS, WEIGHTINGS
from werewolf.observer_dtrs.validation_artifacts import (
    digest, publish_unit, read_unit, run_lock, record_failure, VERSION as ENVELOPE_VERSION,
)

VERSION='observer_dtrs_offline_decision_run_v1'
# Sufficient-statistic order: mass, actual loss, P/B/N mass, FP/FN, Y=1 mass by P/B/N,
# P/B/N loss, forced-binary loss, 3 fixed-action losses, optimal-tie mass, predicted minimum risk.
BASE=21


def _ratio(n,d):
    return n/d if d else None


def _band(value,edges):
    return next((f'[{a},{b})' for a,b in zip(edges,edges[1:]) if a<=value<b), f'[{edges[-1]},inf)')


def _point(v,*,stability,radii):
    mass=v[0];pos,bnd,neg=v[2:5];fp,fn=v[5:7];decided=pos+neg
    regions={a.value:{'mass':v[2+i],'proportion':_ratio(v[2+i],mass),
        'event_rate':_ratio(v[7+i],v[2+i]),'loss_contribution':_ratio(v[10+i],mass)} for i,a in enumerate(Region)}
    return {'mass':mass,'realized_risk':_ratio(v[1],mass),'regions':regions,
        'decided_coverage':_ratio(decided,mass),'false_positive_joint':_ratio(fp,mass),'false_negative_joint':_ratio(fn,mass),
        'false_positive_rate_given_Y0':_ratio(fp,v[19]),'false_negative_rate_given_Y1':_ratio(fn,v[20]),
        'POS_error_rate':_ratio(fp,pos),'NEG_error_rate':_ratio(fn,neg),'selective_error_rate':_ratio(fp+fn,decided),
        'forced_binary_risk':_ratio(v[13],mass),'fixed_action_risks':{a.value:_ratio(v[14+i],mass) for i,a in enumerate(Region)},
        'optimal_tie_mass_fraction':_ratio(v[17],mass),'predicted_minimum_risk':_ratio(v[18],mass),
        'probability_stability':({r:{'unstable_mass_fraction':_ratio(v[BASE+2*i],mass),
            'mean_scenario_worst_regret':_ratio(v[BASE+2*i+1],mass)} for i,r in enumerate(radii)} if stability else None)}


class _Prepared:
    """Run-local decoded views. Never returned through the legal probability API."""
    def __init__(self,inputs):
        self.identity=inputs.identity
        self.legal=inputs.legal
        self.targets=inputs.open_targets()
        self.support=inputs.open_support()
        from werewolf.observer_dtrs.decision_offline import ProbabilityInput
        self.rows={(m,w):tuple(ProbabilityInput(sid,r['game_id'],r['p'][w][m]) for sid,r in self.legal.items())
                   for w in WEIGHTINGS for m in MODELS}
        self.bootstrap_cache={}


def bootstrap_risks(by_game,*,inputs,other=None,_prepared=None):
    """Paired ratio bootstrap, caching replicate estimates rather than subtracting CIs."""
    identity=inputs.identity if _prepared is None else _prepared.identity
    order=identity['game_order']
    if set(by_game)!=set(order) or (other is not None and set(other)!=set(order)):
        raise ValueError('bootstrap requires all original games, including single-class games')
    matrix=np.array([[by_game[g]['loss_sum'],by_game[g]['mass']] for g in order])
    second=None if other is None else np.array([[other[g]['loss_sum'],other[g]['mass']] for g in order])
    if (not np.isfinite(matrix).all() or np.any(matrix[:,1]<0)
            or (second is not None and (not np.isfinite(second).all() or not np.array_equal(matrix[:,1],second[:,1])))):
        raise ValueError('nonfinite bootstrap statistics or unpaired denominators')
    draws=np.frombuffer(inputs.bootstrap_payload,dtype='<i8').reshape(2000,len(order))
    def estimates(values):
        key=(identity['bootstrap_sha256'],values.tobytes())
        cache={} if _prepared is None else _prepared.bootstrap_cache
        if key not in cache:
            result=[]
            for start in range(0,2000,64):
                total=values[draws[start:start+64]].sum(axis=1)
                with np.errstate(divide='ignore',invalid='ignore'):
                    result.extend((total[:,0]/total[:,1]).tolist())
            cache[key]=np.asarray(result)
        return cache[key]
    values=estimates(matrix)
    if second is not None:
        values=estimates(second)-values
    valid=values[np.isfinite(values)]
    return {'replicates':2000,'valid_replicates':len(valid),
        'interval':np.quantile(valid,[.025,.975],method='linear').tolist() if len(valid)>=1900 else None,
        'draws_digest':identity['bootstrap_sha256'],'scope':'fixed_predictions_case_composition_not_retraining',
        'interval_role':'pointwise_normative_sensitivity_not_independent_confirmatory_test'}


def _case(inputs,protocol,*,scenario,policy,model,weighting,_prepared=None):
    loss=protocol.loss(scenario,policy)
    spec=protocol.record
    prepared=_Prepared(inputs) if _prepared is None else _prepared
    binary_policy=TiePolicy(spec['binary_tie_policy'])
    rows=prepared.rows[model,weighting]  # Only ProbabilityInput enters pure calls.
    legal=prepared.legal;targets=prepared.targets;support=prepared.support[weighting+'/'+model]
    sizes=prepared.identity['game_sizes'];radii=spec['probability_stability']['radii_exact']
    stability=(model in spec['probability_stability']['models']
               and policy.value in spec['probability_stability']['policies'])
    costs={(a,y):float(realized_loss(a,y,loss)) for a in Region for y in (0,1)}
    interval_summary=interval_summary_engine(loss)
    @lru_cache(maxsize=None)
    def pure(p):
        d=decide(p,loss)
        b=forced_binary(p,loss,tie_policy=binary_policy)
        sens={r:interval_summary(d,Fraction(r)) for r in radii} if stability else {}
        exact={'minimizing_actions':[a.value for a in d.minimizing_actions],
               'risks_exact':{a.value:str(value) for a,value in d.risks.items()},'risk_gap_exact':str(d.risk_gap)}
        return d,b,sens,exact
    buckets=defaultdict(list);records=[];game_vectors=defaultdict(list);group_games=defaultdict(set)
    max_regret=defaultdict(float);region_rows=defaultdict(lambda:defaultdict(int));region_games=defaultdict(lambda:defaultdict(set))
    # All predeclared groups retained even if empty. Training support is descriptive only.
    group_names=['overall',*[f'phase_nW/{p}/{n}' for p in ('discussion','pk_discussion') for n in (1,2)],
        'report/successful_empty','report/nonempty',*[f'tom/{k}' for k in range(5)],
        *[f'block/{k}-{j}' for k in range(5) for j in range(5)],
        *[f'level/{l}' for l in spec['subgroups']['fallback_levels']],
        'support/NA','fallback/yes','fallback/no']
    for attr,edges in (('kish','kish_edges'),('games','games_edges')):
        e=spec['subgroups'][edges]
        group_names += [f'{attr}/{_band(v,e)}' for v in e]
    for name in group_names:
        buckets[name]=[]
    for row in sorted(rows,key=lambda r:(r.game_id,r.sample_id)):
        sid,g=row.sample_id,row.game_id
        d,b,sens,exact=pure(row.probability)
        y=targets[sid]['y']  # Supervision used AFTER pure decision, only for realized loss.
        a=list(Region).index(d.region);w=1/sizes[g] if weighting=='game_equal' else 1.
        v=[0.]*(BASE+2*len(radii));v[0]=w;v[1]=w*costs[d.region,y];v[2+a]=w
        v[5]=w*int(d.region==Region.POS and y==0);v[6]=w*int(d.region==Region.NEG and y==1)
        v[7+a]=w*y;v[10+a]=v[1];v[13]=w*costs[b,y]
        for i,fixed in enumerate(Region):
            v[14+i]=w*costs[fixed,y]
        v[17]=w*int(len(d.minimizing_actions)>1);v[18]=w*float(d.risks[d.region])
        v[19]=w*(1-y);v[20]=w*y
        compact_sensitivity={}
        for i,radius in enumerate(radii if stability else ()):
            s=sens[radius];v[BASE+2*i]=w*int(not s['stable_under_selected_tie_policy'])
            v[BASE+2*i+1]=w*s['worst_case_regret']
            max_regret[radius]=max(max_regret[radius],s['worst_case_regret'])
            compact_sensitivity[radius]={'unstable':not s['stable_under_selected_tie_policy'],'regret':s['worst_case_regret']}
        records.append({'sample_id':sid,'game_id':g,'region':d.region.value,
            'minimizing_actions':list(exact['minimizing_actions']),
            'risks_exact':dict(exact['risks_exact']),'risk_gap_exact':exact['risk_gap_exact'],
            'binary_region':b.value,'probability_stability':compact_sensitivity})
        info=legal[sid];saved=support[sid]
        names=['overall',f'phase_nW/{info["phase"]}/{info["nW"]}',
            'report/'+('successful_empty' if targets[sid]['successful_empty'] else 'nonempty'),
            f'tom/{info["tom_fold"]}',f'block/{info["tom_fold"]}-{info["outer_fold"]}',f'level/{saved["level"]}']
        if saved['level']=='NOT_APPLICABLE':
            names.append('support/NA')
        else:
            names += ['kish/'+_band(saved['kish'],spec['subgroups']['kish_edges']),
                'games/'+_band(saved['games'],spec['subgroups']['games_edges']),
                'fallback/'+('yes' if saved['fallback'] else 'no')]
        for name in names:
            buckets[name].append(v);group_games[name].add(g)
            region_rows[name][d.region.value]+=1;region_games[name][d.region.value].add(g)
        game_vectors[g].append(v)
    def total(vectors):
        return [math.fsum(v[i] for v in vectors) for i in range(BASE+2*len(radii))]
    groups={name:{'rows':len(values),'games':len(group_games[name]),
                   **_point(total(values),stability=stability,radii=radii)} for name,values in sorted(buckets.items())}
    for name,group in groups.items():
        for a in Region:
            group['regions'][a.value].update(rows=region_rows[name][a.value],games=len(region_games[name][a.value]))
    by_game={g:{'loss_sum':total(game_vectors[g])[1],'mass':total(game_vectors[g])[0]} for g in sizes}
    return {'schema_version':VERSION,'status':'CASE_COMPLETE','loss':loss.to_record(),'loss_digest':loss.digest,
        'thresholds':d.thresholds.to_record(),'model':model,'weighting':weighting,'scenario':scenario,'policy':policy.value,
        'binary_tie_policy':binary_policy.value,'rows':len(rows),'games':len(sizes),
        'decisions':records,'groups':groups,'by_game':by_game,'risk_interval':bootstrap_risks(by_game,inputs=inputs,_prepared=prepared),
        'maximum_scenario_worst_regret':dict(max_regret) if stability else None,
        'contrasts_to_baselines':{
            'three_way_minus_forced_binary':groups['overall']['realized_risk']-groups['overall']['forced_binary_risk'],
            **{f'three_way_minus_always_{a}':groups['overall']['realized_risk']-risk
               for a,risk in groups['overall']['fixed_action_risks'].items()}},
        'support_used_for_decision':False}


def _checkpoint(path,kind,identity,compute):
    exists=path.exists()
    if exists:
        a,value=read_unit(path,kind,identity)  # Reject bytes/identity before expensive replay.
    expected=compute()  # Fixed p semantic replay; never re-fit.
    if exists:
        if canonical_json_bytes(value)!=canonical_json_bytes(expected):
            raise ValueError('D3 checkpoint content inconsistent; stop without repair')
        return a,value
    return publish_unit(path,kind,identity,expected),expected


def _degenerate_diagnostics():
    values=[]
    for costs in ((Fraction(1,2),Fraction(1,2),Fraction(1,4),Fraction(1,4)),(1,1,0,0),(1,1,1,0),(1,1,0,1)):
        for policy in TiePolicy:
            loss=LossSpec(*costs,policy)
            values.append({'loss':loss.to_record(),'decisions':[decide(p,loss).to_record() for p in (0,Fraction(1,2),1)]})
    return {'scope':'label_free_degenerate_diagnostic_not_formal_cost_grid','cases':values}


def _execute(root,*,inputs,protocol,source,hook=None):
    inputs.validate()
    prepared=_Prepared(inputs);input_identity=prepared.identity
    root=Path(root)
    if any((parent/'manifest.json').exists() for parent in (root,*root.parents)):
        raise ValueError('D3 output cannot be inside an existing immutable artifact')
    if (root/'contract/manifest.json').exists():
        import json
        if json.loads((root/'contract/manifest.json').read_bytes()).get('artifact_type')!='observer_dtrs_D3_contract':
            raise ValueError('destination already belongs to a different experiment')
    contract={'schema_version':VERSION,'scope':input_identity['scope'],'inputs':input_identity,
        'protocol':protocol.record,'protocol_digest':protocol.digest,'source_snapshot_digest':digest(source)}
    with run_lock(root):
        ca,_=_checkpoint(root/'contract','D3_contract',{'run_identity':digest(contract)},lambda:contract)
        source_unit,_=_checkpoint(root/'source','D3_source',{'contract':ca.manifest_digest},lambda:source)
        audit_record={'support':prepared.support,'targets_digest':input_identity['targets_digest'],
                      'legal_digest':input_identity['legal_digest'],'scope':'restricted_audit_only_not_decision_input'}
        audit_unit,_=_checkpoint(root/'restricted_support','D3_support',{'contract':ca.manifest_digest},lambda:audit_record)
        boot_identity={'contract':ca.manifest_digest,'bootstrap_sha256':input_identity['bootstrap_sha256']}
        boot_record={'game_order':input_identity['game_order'],'game_strata':input_identity['game_strata']}
        if (root/'bootstrap').exists():
            boot_unit,stored=read_unit(root/'bootstrap','D3_bootstrap',boot_identity,extra_names=('indices.int64le',))
            if stored!=boot_record or read_artifact_file(boot_unit,'indices.int64le')!=inputs.bootstrap_payload:
                raise ValueError('changed paired bootstrap source; stop without repair')
        else:
            boot_unit=publish_unit(root/'bootstrap','D3_bootstrap',boot_identity,boot_record,{'indices.int64le':inputs.bootstrap_payload})
        stage='start';case_refs={};comparisons={};cases={};tie_refs={}
        try:
            for policy in protocol.policies:
                for scenario in range(45):
                    for weighting in WEIGHTINGS:
                        for model in MODELS:
                            key=f'{policy.name}/C{scenario:02}/{weighting}/{model}'
                            stage=key
                            identity={'contract':ca.manifest_digest,'model':model,'weighting':weighting,
                                      'loss_digest':protocol.loss(scenario,policy).digest,'scenario':scenario,'policy':policy.value}
                            a,value=_checkpoint(root/'cases'/key,'D3_case',identity,
                                lambda:_case(inputs,protocol,scenario=scenario,policy=policy,model=model,weighting=weighting,_prepared=prepared))
                            case_refs[key]=a.manifest_digest
                            # Only comparison/game/compact-region fields retained in memory across cases.
                            cases[key]={'by_game':value['by_game'],'risk':value['groups']['overall']['realized_risk'],
                                        'regions':({r['sample_id']:r['region'] for r in value['decisions']}
                                                   if model=='OBS-EQ1' and policy==TiePolicy.TERMINAL_NEG_FIRST else {})}
                            if policy==TiePolicy.DEFER_FIRST:
                                primary_key=f'TERMINAL_NEG_FIRST/C{scenario:02}/{weighting}/{model}'
                                primary_identity={**identity,'policy':TiePolicy.TERMINAL_NEG_FIRST.value,
                                    'loss_digest':protocol.loss(scenario,TiePolicy.TERMINAL_NEG_FIRST).digest}
                                primary_unit,primary=read_unit(root/'cases'/primary_key,'D3_case',primary_identity)
                                before={r['sample_id']:r for r in primary['decisions']}
                                after={r['sample_id']:r for r in value['decisions']}
                                if set(before)!=set(after):raise ValueError('unpaired policy sensitivity population')
                                changed=[];weights=[];sizes=input_identity['game_sizes']
                                for sid,r in after.items():
                                    old=before[sid]
                                    if old['minimizing_actions']!=r['minimizing_actions'] or old['risks_exact']!=r['risks_exact']:
                                        raise ValueError('policy sensitivity changed costs/probabilities')
                                    weight=1/sizes[r['game_id']] if weighting=='game_equal' else 1.
                                    weights.append(weight)
                                    if old['region']!=r['region']:
                                        if len(r['minimizing_actions'])<2:raise ValueError('policy changed unique minimizer')
                                        changed.append(weight)
                                tie_record={'scope':'exact_tie_sensitivity_only_no_near_tie_tolerance',
                                    'changed_rows':len(changed),'changed_mass_fraction':math.fsum(changed)/math.fsum(weights),
                                    'realized_risk_DEFER_minus_NEG':value['groups']['overall']['realized_risk']-primary['groups']['overall']['realized_risk']}
                                tie_name=f'C{scenario:02}/{weighting}/{model}'
                                stage='tie_sensitivity/'+tie_name
                                ta,_=_checkpoint(root/'tie_sensitivity'/tie_name,'D3_tie_sensitivity',
                                    {'contract':ca.manifest_digest,'cases':[primary_unit.manifest_digest,a.manifest_digest]},lambda:tie_record)
                                tie_refs[tie_name]=ta.manifest_digest
                            if hook:hook('case',a.path)
                        base=f'{policy.name}/C{scenario:02}/{weighting}/'
                        pairs=(('OBS-EQ1','OBS-EQ0','primary'),
                               ('OBS-LR-r','OBS-EQ1','key_secondary'),('OBS-EF','OBS-EQ1','key_secondary'),
                               ('OBS-LR-r','OBS-EQ0','supplementary'),('OBS-EF','OBS-EQ0','supplementary'))
                        contrast={a+'_minus_'+b:{'role':role,'point':cases[base+a]['risk']-cases[base+b]['risk'],
                            'interval':bootstrap_risks(cases[base+b]['by_game'],inputs=inputs,
                                other=cases[base+a]['by_game'],_prepared=prepared)} for a,b,role in pairs}
                        name=f'{policy.name}/C{scenario:02}/{weighting}'
                        stage='comparisons/'+name
                        a,_=_checkpoint(root/'comparisons'/name,'D3_comparison',
                            {'contract':ca.manifest_digest,'cases':{m:case_refs[base+m] for m in MODELS}},lambda:contrast)
                        comparisons[name]=a.manifest_digest
                        for m in MODELS:
                            key=base+m
                            if m=='OBS-EQ1' and policy==TiePolicy.TERMINAL_NEG_FIRST:
                                cases[key]={'regions':cases[key]['regions']}
                            else:
                                del cases[key]
                        prepared.bootstrap_cache.clear()  # Only four paired model estimates are needed at once.
            cost_refs={}
            for w in WEIGHTINGS:
                for a,b,axis in cost_edges():
                    ka=f'TERMINAL_NEG_FIRST/C{a:02}/{w}/OBS-EQ1';kb=f'TERMINAL_NEG_FIRST/C{b:02}/{w}/OBS-EQ1'
                    ra,rb=cases[ka]['regions'],cases[kb]['regions']
                    mass=defaultdict(list)
                    legal=prepared.legal;sizes=input_identity['game_sizes']
                    for sid in sorted(ra):
                        weight=1/sizes[legal[sid]['game_id']] if w=='game_equal' else 1.
                        mass[ra[sid]+'->'+rb[sid]].append(weight)
                    total=math.fsum(x for v in mass.values() for x in v)
                    transition={x+'->'+y:math.fsum(mass[x+'->'+y])/total for x in ('POS','BND','NEG') for y in ('POS','BND','NEG')}
                    result={'scope':'adjacent_normative_cost_sensitivity_not_cost_selection','axis':axis,'first':a,'second':b,
                        'weighting':w,'transition_mass_fraction':transition,
                        'flipped_mass_fraction':math.fsum(v for k,v in transition.items() if k.split('->')[0]!=k.split('->')[1])}
                    key=f'{w}/{a:02}-{b:02}'
                    stage='cost_edges/'+key
                    unit,_=_checkpoint(root/'cost_edges'/key,'D3_cost_edge',
                        {'contract':ca.manifest_digest,'cases':[case_refs[ka],case_refs[kb]]},lambda:result)
                    cost_refs[key]=unit.manifest_digest
            stage='diagnostics'
            diagnostic,_=_checkpoint(root/'diagnostics','D3_degenerate',{'contract':ca.manifest_digest},_degenerate_diagnostics)
            if len(case_refs)!=720 or len(comparisons)!=180 or len(cost_refs)!=192 or len(tie_refs)!=360:
                raise ValueError('D3 incomplete scenario/model/policy coverage')
            final={'status':'D3_ALL_CASES_COMPLETE','schema_version':VERSION,'scope':input_identity['scope'],
                'contract':ca.manifest_digest,'source':source_unit.manifest_digest,'restricted_support':audit_unit.manifest_digest,
                'bootstrap':boot_unit.manifest_digest,
                'cases':case_refs,'comparisons':comparisons,'cost_edges':cost_refs,'tie_sensitivity':tie_refs,'diagnostics':diagnostic.manifest_digest,
                'samples':input_identity['samples'],'games':len(input_identity['game_sizes']),
                'bootstrap_sha256':input_identity['bootstrap_sha256'],'research_scope':protocol.record['research_scope'],
                'limitations':protocol.record['limitations'],'no_model_fitting':True}
            stage='complete'
            final_unit,_=_checkpoint(root/'complete','D3_complete',{'contract':ca.manifest_digest},lambda:final)
            return {**final,'manifest_digest':final_unit.manifest_digest}
        except BaseException as error:
            record_failure(root,ca.manifest_digest,stage,error)
            raise


def execute_controlled(destination,*,inputs=None,protocol=None,hook=None):
    inputs=controlled_inputs() if inputs is None else inputs
    if inputs.identity['scope']!='controlled_fixture':
        raise ValueError('controlled runner must never consume formal material')
    return _execute(destination,inputs=inputs,protocol=load_protocol() if protocol is None else protocol,
                    source={'scope':'controlled_fixture','identity':'synthetic_engineering_acceptance_no_formal_source_claim'},hook=hook)


def execute_formal(destination,*,run_path,dataset_path,protocol,expected_protocol_digest,expected_revision,artifact_root):
    # Guard and explicit protocol approval BEFORE input reading, destination creation or computation.
    source=checked_source(expected_revision)
    protocol.require_frozen(expected_protocol_digest)
    dest=Path(destination).resolve();allowed=Path(artifact_root).resolve()
    protected=[Path(run_path).resolve(),Path(dataset_path).resolve()]
    if not dest.is_relative_to(allowed) or any(dest.is_relative_to(p) or p.is_relative_to(dest) for p in protected):
        raise ValueError('D3 output must be separate from frozen inputs and inside artifact_root')
    inputs=prepare_formal(run_path,dataset_path,protocol=protocol)
    return _execute(dest,inputs=inputs,protocol=protocol,source=source)


def _inspect_result(root,*,expected_manifest_digest=None):
    """Verify bytes, complete identities and cross-references; not raw-input replay."""
    import json
    from collections import Counter
    from werewolf.observer_dtrs.d3_protocol import D3Protocol
    root=Path(root)
    def load(relative,kind,expected,identity=None,extras=()):
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('unsafe D3 reference')
        a=verify_artifact(root/path,expected_artifact_type='observer_dtrs_'+kind,expected_schema_version=ENVELOPE_VERSION)
        if (expected is not None and a.manifest_digest!=expected) or a.manifest['identity_digest']!=digest(a.manifest['identity']):
            raise ValueError('D3 digest/identity mismatch')
        if identity is not None and a.manifest['identity']!=identity:
            raise ValueError('D3 complete checkpoint identity mismatch')
        if set(a.manifest['file_table'])!={'record.json',*extras}:
            raise ValueError('D3 payload inventory mismatch')
        raw=read_artifact_file(a,'record.json');r=json.loads(raw)
        if raw!=canonical_json_bytes(r):raise ValueError('noncanonical D3 record')
        return a,r
    if expected_manifest_digest is not None and (len(expected_manifest_digest)!=64 or any(c not in '0123456789abcdef' for c in expected_manifest_digest)):
        raise ValueError('explicit complete SHA-256 required')
    fa,final=load('complete','D3_complete',expected_manifest_digest)
    required={'status','schema_version','scope','contract','source','restricted_support','bootstrap','cases','comparisons',
              'cost_edges','tie_sensitivity','diagnostics','samples','games','bootstrap_sha256','research_scope','limitations','no_model_fitting'}
    if (set(final)!=required or final['status']!='D3_ALL_CASES_COMPLETE' or final['schema_version']!=VERSION
            or fa.manifest['identity']!={'contract':final['contract']}):
        raise ValueError('invalid/incomplete D3 seal')
    ca,c=load('contract','D3_contract',final['contract'])
    if ca.manifest['identity']!={'run_identity':digest(c)}:raise ValueError('D3 contract identity mismatch')
    p=D3Protocol(canonical_json_bytes(c['protocol']));i=c['inputs'];cid=ca.manifest_digest
    if (c['schema_version']!=VERSION or c['protocol_digest']!=p.digest or c['scope']!=i['scope']
            or final['scope']!=c['scope'] or final['samples']!=i['samples'] or final['games']!=len(i['game_sizes'])
            or final['bootstrap_sha256']!=i['bootstrap_sha256'] or final['research_scope']!=p.record['research_scope']
            or final['limitations']!=p.record['limitations'] or final['no_model_fitting'] is not True):
        raise ValueError('D3 seal/contract/source population mismatch')
    expected_cases={f'{policy.name}/C{s:02}/{w}/{m}' for policy in p.policies for s in range(45) for w in WEIGHTINGS for m in MODELS}
    expected_comparisons={key.rsplit('/',1)[0] for key in expected_cases}
    expected_edges={f'{w}/{a:02}-{b:02}':axis for w in WEIGHTINGS for a,b,axis in cost_edges()}
    expected_ties={f'C{s:02}/{w}/{m}' for s in range(45) for w in WEIGHTINGS for m in MODELS}
    if (set(final['cases'])!=expected_cases or set(final['comparisons'])!=expected_comparisons
            or set(final['cost_edges'])!=set(expected_edges) or set(final['tie_sensitivity'])!=expected_ties):
        raise ValueError('D3 incomplete or renamed registered units')
    _,source=load('source','D3_source',final['source'],{'contract':cid})
    if digest(source)!=c['source_snapshot_digest']:raise ValueError('D3 source snapshot mismatch')
    if final['scope']=='formal_D3_posthoc_development':
        p.require_frozen(p.digest)
        if (len(source.get('head',''))!=40 or not source.get('files') or not source.get('D3_execution_files')
                or i['D2']['source_revision']!=p.record['inputs']['D2_source_revision']
                or i['D2']['complete_digest']!=p.record['inputs']['D2_complete_digest']
                or i['D2']['dataset_digest']!=p.record['inputs']['D1c_digest']
                or i['D2']['split_digest']!=p.record['inputs']['D2_split_digest']
                or i['bootstrap_sha256']!=p.record['inputs']['D2_bootstrap_sha256']
                or final['samples']!=p.record['inputs']['samples'] or final['games']!=p.record['inputs']['games']):
            raise ValueError('formal D3 source/dataset provenance mismatch')
    elif final['scope']!='controlled_fixture' or source.get('scope')!='controlled_fixture':
        raise ValueError('invalid formal/controlled source scope')
    _,audit=load('restricted_support','D3_support',final['restricted_support'],{'contract':cid})
    if (digest(audit['support'])!=i['support_digest'] or audit['targets_digest']!=i['targets_digest']
            or audit['legal_digest']!=i['legal_digest']):raise ValueError('D3 restricted audit binding mismatch')
    ba,boot=load('bootstrap','D3_bootstrap',final['bootstrap'],{'contract':cid,'bootstrap_sha256':i['bootstrap_sha256']},('indices.int64le',))
    raw=read_artifact_file(ba,'indices.int64le')
    from werewolf.artifact_io import sha256_bytes
    if (sha256_bytes(raw)!=i['bootstrap_sha256'] or boot!={'game_order':i['game_order'],'game_strata':i['game_strata']}
            or len(raw)!=2000*len(i['game_order'])*8):raise ValueError('D3 bootstrap bytes/source mismatch')
    draws=np.frombuffer(raw,dtype='<i8').reshape(2000,-1)
    strata=[i['game_strata'][g] for g in i['game_order']]
    if len(set(i['game_order']))!=len(i['game_order']) or set(i['game_order'])!=set(i['game_sizes']):
        raise ValueError('D3 bootstrap game order mismatch')
    for col in range(len(strata)):
        values=draws[:,col]
        if np.any(values<0) or np.any(values>=len(strata)) or any(strata[int(v)]!=strata[col] for v in set(values)):
            raise ValueError('D3 bootstrap crossed strata')
    risks={};population_digest=None
    def interval(r):
        ci=r['interval']
        if (r['replicates']!=2000 or type(r['valid_replicates']) is not int or not 0<=r['valid_replicates']<=2000
                or r['draws_digest']!=i['bootstrap_sha256'] or (ci is not None and (
                    len(ci)!=2 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in ci) or ci[0]>ci[1]))
                or (r['valid_replicates']<1900 and ci is not None)):
            raise ValueError('D3 interval provenance/shape mismatch')
    for key,h in final['cases'].items():
        policy_name,scenario_name,w,m=key.split('/');policy=TiePolicy[policy_name];scenario=int(scenario_name[1:]);loss=p.loss(scenario,policy)
        _,v=load('cases/'+key,'D3_case',h,{'contract':cid,'model':m,'weighting':w,'loss_digest':loss.digest,'scenario':scenario,'policy':policy.value})
        if (v['schema_version']!=VERSION or v['status']!='CASE_COMPLETE' or v['model']!=m or v['weighting']!=w
                or v['scenario']!=scenario or v['policy']!=policy.value or v['loss_digest']!=loss.digest
                or v['loss']!=loss.to_record() or v['binary_tie_policy']!=p.record['binary_tie_policy']
                or v['rows']!=i['samples'] or v['games']!=len(i['game_sizes']) or v['support_used_for_decision'] is not False):
            raise ValueError('D3 case payload identity mismatch')
        rows=v['decisions'];ids={r['sample_id']:r['game_id'] for r in rows}
        if (len(rows)!=i['samples'] or len(ids)!=len(rows) or dict(Counter(ids.values()))!=i['game_sizes']
                or any(r['region'] not in ('POS','BND','NEG') or 'y' in r or 'support' in r for r in rows)
                or set(v['by_game'])!=set(i['game_sizes'])):raise ValueError('D3 case sample/game population mismatch')
        pop=digest(sorted(ids.items()))
        if population_digest is None:population_digest=pop
        if pop!=population_digest:raise ValueError('D3 cross-case population mismatch')
        if v['thresholds']!=loss_thresholds(loss).to_record():raise ValueError('D3 thresholds mismatch')
        interval(v['risk_interval']);risks[key]=v['groups']['overall']['realized_risk']
        total_mass=math.fsum(g['mass'] for g in v['by_game'].values())
        total_loss=math.fsum(g['loss_sum'] for g in v['by_game'].values())
        if (total_mass<=0 or not math.isclose(total_loss/total_mass,risks[key],abs_tol=1e-12,rel_tol=0)
                or any(not math.isclose(r['mass'],1 if w=='game_equal' else i['game_sizes'][g],abs_tol=1e-12,rel_tol=0)
                       for g,r in v['by_game'].items())):raise ValueError('D3 weighted game statistics mismatch')
    pairs=(('OBS-EQ1','OBS-EQ0','primary'),('OBS-LR-r','OBS-EQ1','key_secondary'),('OBS-EF','OBS-EQ1','key_secondary'),
           ('OBS-LR-r','OBS-EQ0','supplementary'),('OBS-EF','OBS-EQ0','supplementary'))
    for key,h in final['comparisons'].items():
        base=key+'/'
        _,v=load('comparisons/'+key,'D3_comparison',h,{'contract':cid,'cases':{m:final['cases'][base+m] for m in MODELS}})
        if set(v)!={a+'_minus_'+b for a,b,_ in pairs}:raise ValueError('missing direct registered comparison')
        for a,b,role in pairs:
            r=v[a+'_minus_'+b];interval(r['interval'])
            if r['role']!=role or r['point']!=risks[base+a]-risks[base+b]:raise ValueError('D3 contrast identity/point mismatch')
    for key,h in final['tie_sensitivity'].items():
        scenario,w,m=key.split('/');primary=f'TERMINAL_NEG_FIRST/{key}';defer=f'DEFER_FIRST/{key}'
        _,v=load('tie_sensitivity/'+key,'D3_tie_sensitivity',h,{'contract':cid,'cases':[final['cases'][primary],final['cases'][defer]]})
        if (not 0<=v['changed_rows']<=i['samples'] or not 0<=v['changed_mass_fraction']<=1
                or v['realized_risk_DEFER_minus_NEG']!=risks[defer]-risks[primary]):raise ValueError('D3 tie statistic mismatch')
    for key,h in final['cost_edges'].items():
        w,edge=key.split('/');a,b=map(int,edge.split('-'))
        ka=f'TERMINAL_NEG_FIRST/C{a:02}/{w}/OBS-EQ1';kb=f'TERMINAL_NEG_FIRST/C{b:02}/{w}/OBS-EQ1'
        _,v=load('cost_edges/'+key,'D3_cost_edge',h,{'contract':cid,'cases':[final['cases'][ka],final['cases'][kb]]})
        transition=v['transition_mass_fraction']
        expected={a+'->'+b for a in ('POS','BND','NEG') for b in ('POS','BND','NEG')}
        if (v['axis']!=expected_edges[key] or v['first']!=a or v['second']!=b or v['weighting']!=w
                or set(transition)!=expected or any(not 0<=x<=1 for x in transition.values())
                or not math.isclose(math.fsum(transition.values()),1,abs_tol=1e-12)
                or v['flipped_mass_fraction']!=math.fsum(x for k,x in transition.items() if k.split('->')[0]!=k.split('->')[1])):
            raise ValueError('D3 cost-edge identity/statistic mismatch')
    load('diagnostics','D3_degenerate',final['diagnostics'],{'contract':cid})
    return {'status':final['status'],'scope':final['scope'],'cases':720,'comparisons':180,'cost_edges':192,
        'tie_sensitivity':360,'manifest_digest':fa.manifest_digest,'external_complete_anchor_verified':expected_manifest_digest is not None,
        'verification_level':'sealed_bytes_full_identities_and_cross_references_not_input_semantic_replay'}


def inspect_result(root,*,expected_manifest_digest=None):
    from werewolf.artifact_io import ArtifactValidationError
    try:
        return _inspect_result(root,expected_manifest_digest=expected_manifest_digest)
    except (OSError,KeyError,TypeError,IndexError,OverflowError) as error:
        raise ArtifactValidationError(f'invalid D3 inspection package: {error}') from error
