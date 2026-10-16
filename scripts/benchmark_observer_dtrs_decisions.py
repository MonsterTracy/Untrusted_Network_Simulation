"""Bounded, entirely artificial D3 pressure probe. Never reads D1c/D2 data."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import resource
import sys
import tempfile
import time

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from werewolf.artifact_io import canonical_json_bytes
from werewolf.observer_dtrs.d3_inputs_offline import _assemble
from werewolf.observer_dtrs.d3_protocol import load_protocol
from werewolf.observer_dtrs.statistical_analysis import MODELS,WEIGHTINGS,bootstrap_plan
from werewolf.observer_dtrs.d2_protocol import frozen_protocol
from werewolf.observer_dtrs.validation_artifacts import publish_unit,read_unit
import werewolf.observer_dtrs.d3_runner as runner


def artificial_inputs(games=1500,samples=22685,pattern='unique'):
    if (type(games) is not int or games<25 or games>1500 or games%25 or type(samples) is not int
            or not games<=samples<=22685 or pattern not in ('unique','pooled')):
        raise ValueError('bounded artificial probe only: 25..1500 games, at most 22685 rows')
    rng=np.random.Generator(np.random.PCG64(20261016))
    legal={};targets={};support={w+'/'+m:{} for w in WEIGHTINGS for m in MODELS};strata={}
    for g in range(games):
        game='artificial-pressure-'+hashlib.sha256(str(g).encode()).hexdigest()[:52]
        k,j=divmod(g%25,5);strata[game]=[k,j]
        for n in range(samples//games+int(g<samples%games)):
            sid=hashlib.sha256(f'artificial/{g}/{n}'.encode()).hexdigest()
            value=float(rng.uniform()) if pattern=='unique' else (.02+.96*((g+n)%32)/31)
            legal[sid]={'game_id':game,'phase':'discussion' if g%2 else 'pk_discussion','nW':1+g%2,
                'tom_fold':k,'outer_fold':j,'p':{w:{m:(.6+g%4/20 if m=='OBS-EQ0' else value) for m in MODELS} for w in WEIGHTINGS}}
            y=int(rng.integers(0,2));targets[sid]={'y':y,'successful_empty':bool(y and n%11==0)}
            for key in support:
                support[key][sid]=({'level':'NOT_APPLICABLE','games':None,'kish':None,'fallback':None} if key.endswith('OBS-LR-r')
                    else {'level':'r_bin','games':20+g%100,'kish':10.+g%50,'fallback':False})
    blocks={(k,j):[g for g,s in strata.items() if s==[k,j]] for k in range(5) for j in range(5)}
    order,draws=bootstrap_plan(blocks,protocol=frozen_protocol(),formal=False)
    identity={'scope':'controlled_fixture','samples':samples,'game_sizes':dict(Counter(r['game_id'] for r in legal.values())),
        'game_order':order,'game_strata':strata,'research_scope':'artificial_pressure_only_not_formal_results',
        'limitations':['synthetic_not_ToM_or_D1c'],'source':'hand_generated_fixed_seed_20261016','pattern':pattern}
    return _assemble(legal,targets,support,identity,draws.astype('<i8').tobytes())


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--games',type=int,default=1500);p.add_argument('--samples',type=int,default=22685)
    p.add_argument('--pattern',choices=('unique','pooled'),default='unique')
    p.add_argument('--model',choices=MODELS,default='OBS-EQ1');p.add_argument('--scenario',type=int,choices=range(45),default=22)
    p.add_argument('--policy',choices=('main','defer'),default='main')
    p.add_argument('--weighting',choices=WEIGHTINGS,default='game_equal')
    p.add_argument('--reference-source',type=Path,help='Optional trusted pre-edit local source copy, artificial regression only')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',type=Path)
    p.add_argument('--retain-region-maps',type=int,choices=range(91),default=0,
                   help='Bounded emulator for the 90 EQ1 region maps retained by the serial runner')
    args=p.parse_args(argv);module=runner
    if args.reference_source:
        spec=importlib.util.spec_from_file_location('trusted_preedit_d3_runner',args.reference_source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    start=time.perf_counter();inputs=artificial_inputs(args.games,args.samples,args.pattern)
    build_seconds=time.perf_counter()-start;protocol=load_protocol()
    kwargs={'scenario':args.scenario,'policy':protocol.policies[args.policy=='defer'],'model':args.model,'weighting':args.weighting}
    prepare_seconds=0.
    if module is runner:
        start=time.perf_counter();prepared=runner._Prepared(inputs);prepare_seconds=time.perf_counter()-start
        kwargs['_prepared']=prepared
    source_ids=tuple(inputs.legal)
    retained=[{sid:('POS','BND','NEG')[j%3] for sid in source_ids} for j in range(args.retain_region_maps)]
    profile=None
    if args.profile:
        import cProfile
        profile=cProfile.Profile();profile.enable()
    wall=time.perf_counter();cpu=time.process_time();result=module._case(inputs,protocol,**kwargs)
    case_wall=time.perf_counter()-wall;case_cpu=time.process_time()-cpu
    if profile:
        profile.disable();profile.dump_stats(args.profile)
    start=time.perf_counter();raw=canonical_json_bytes(result);serialize=time.perf_counter()-start
    with tempfile.TemporaryDirectory(prefix='d3-artificial-pressure-') as tmp:
        root=Path(tmp).resolve()/'unit';identity={'scope':'controlled_pressure','protocol_digest':protocol.digest,
              'rows':args.samples,'case':{k:(v.value if hasattr(v,'value') else v) for k,v in kwargs.items() if not k.startswith('_')}}
        start=time.perf_counter();a=publish_unit(root,'D3_case',identity,result);write=time.perf_counter()-start
        start=time.perf_counter();read_unit(root,'D3_case',identity);read=time.perf_counter()-start
        start=time.perf_counter()
        a,_=module._checkpoint(root,'D3_case',identity,lambda:module._case(inputs,protocol,**kwargs))
        recovery=time.perf_counter()-start
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    report={'scope':'bounded_artificial_pressure_not_formal_D3','games':args.games,'samples':args.samples,'pattern':args.pattern,
        'model':args.model,'policy':args.policy,'scenario':args.scenario,'weighting':args.weighting,
        'retained_region_maps':len(retained),'build_seconds':build_seconds,'prepare_seconds':prepare_seconds,'profile_enabled':bool(args.profile),'case_wall_seconds':case_wall,'case_cpu_seconds':case_cpu,'serialization_seconds':serialize,
        'durable_publish_seconds':write,'verified_read_seconds':read,'semantic_recovery_seconds':recovery,
        'case_payload_bytes':len(raw),'peak_process_rss_bytes':peak,
        'python':platform.python_version(),'numpy':np.__version__,'machine':platform.machine(),
        'protocol_digest':protocol.digest,'reference_source':str(args.reference_source) if args.reference_source else None,
        'result_sha256':hashlib.sha256(raw).hexdigest(),'real_inputs_read':False}
    with args.output.open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(report,sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
