"""Immutable D3 inputs: legal probabilities, gated targets, separate support audit."""
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from werewolf.artifact_io import canonical_json_bytes, read_artifact_file, sha256_bytes
from werewolf.observer_dtrs.decision_offline import ProbabilityInput
from werewolf.observer_dtrs.sealed_d2_offline import SourceAnchors, read_formal_d2, _unit
from werewolf.observer_dtrs.training_audit_offline import (
    crosscheck_d1c, verify_outer_artifacts, open_d1c_targets, _support,
)
from werewolf.observer_dtrs.statistical_analysis import MODELS, WEIGHTINGS
from werewolf.observer_dtrs.validation_artifacts import digest


@dataclass(frozen=True)
class D3Inputs:
    legal_payload: bytes
    _targets_payload: bytes
    _support_payload: bytes
    identity_payload: bytes
    bootstrap_payload: bytes

    @property
    def identity(self):
        return json.loads(self.identity_payload)

    @property
    def legal(self):
        return json.loads(self.legal_payload)

    def rows(self, model, weighting):
        if model not in MODELS or weighting not in WEIGHTINGS:
            raise ValueError('unregistered probability mapping')
        return tuple(ProbabilityInput(sid,r['game_id'],r['p'][weighting][model]) for sid,r in self.legal.items())

    def validate(self):
        i=self.identity
        for name,raw in (('legal',self.legal_payload),('targets',self._targets_payload),('support',self._support_payload)):
            value=json.loads(raw)
            if raw!=canonical_json_bytes(value) or digest(value)!=i[name+'_digest']:
                raise ValueError('changed D3 '+name+' binding')
        legal=self.legal;targets=self.open_targets();support=self.open_support()
        if (set(legal)!=set(targets) or len(legal)!=i['samples']
                or dict(Counter(r['game_id'] for r in legal.values()))!=i['game_sizes']
                or set(i['game_order'])!=set(i['game_sizes']) or len(i['game_order'])!=len(i['game_sizes'])):
            raise ValueError('D3 missing/duplicate sample/game population')
        for sid,r in legal.items():
            if set(r)!= {'game_id','phase','nW','tom_fold','outer_fold','p'} or set(r['p'])!=set(WEIGHTINGS):
                raise ValueError('illegal probability input schema')
            if (r['phase'] not in ('discussion','pk_discussion') or type(r['nW']) is not int or r['nW'] not in (1,2)
                    or any(type(r[k]) is not int or r[k] not in range(5) for k in ('tom_fold','outer_fold'))
                    or i['game_strata'][r['game_id']]!=[r['tom_fold'],r['outer_fold']]):
                raise ValueError('D3 source fold/phase mismatch')
            t=targets[sid]
            if (set(t)!= {'y','successful_empty'} or type(t['y']) is not int or t['y'] not in (0,1)
                    or type(t['successful_empty']) is not bool or (t['successful_empty'] and t['y']!=1)):
                raise ValueError('invalid D3 offline target')
            for w in WEIGHTINGS:
                if set(r['p'][w])!=set(MODELS):
                    raise ValueError('incomplete D3 models')
                for m in MODELS:
                    ProbabilityInput(sid,r['game_id'],r['p'][w][m])
        if set(support)!={w+'/'+m for w in WEIGHTINGS for m in MODELS}:
            raise ValueError('incomplete support mappings')
        for key,values in support.items():
            if set(values)!=set(legal):
                raise ValueError('missing support audit IDs')
            for s in values.values():
                if key.endswith('OBS-LR-r'):
                    if s!={'level':'NOT_APPLICABLE','games':None,'kish':None,'fallback':None}:
                        raise ValueError('Logistic equivalence support must be NA')
                elif (set(s)!= {'level','games','kish','fallback'} or s['level'] not in ('global','nW','phase_nW','r_bin')
                        or type(s['games']) is not int or s['games']<=0 or not isinstance(s['kish'],(int,float))
                        or isinstance(s['kish'],bool) or not np.isfinite(s['kish']) or not 0<s['kish']<=s['games']+1e-9
                        or s['fallback'] is not (s['level'] in ('global','nW'))):
                    raise ValueError('invalid saved support audit')
        draws=np.frombuffer(self.bootstrap_payload,dtype='<i8')
        if sha256_bytes(self.bootstrap_payload)!=i['bootstrap_sha256'] or draws.size!=2000*len(i['game_order']):
            raise ValueError('D3 bootstrap digest/shape mismatch')
        draws=draws.reshape(2000,-1)
        strata=[i['game_strata'][g] for g in i['game_order']]
        for col in range(len(strata)):
            values=draws[:,col]
            if np.any(values<0) or np.any(values>=len(strata)) or any(strata[int(v)]!=strata[col] for v in set(values)):
                raise ValueError('bootstrap crossed source/outer strata')
        return self

    def open_targets(self):
        """Explicit offline access; never part of rows()."""
        return json.loads(self._targets_payload)

    def open_support(self):
        """Restricted diagnostics; never part of rows() or decide()."""
        return json.loads(self._support_payload)


def _assemble(legal,targets,support,identity,draws):
    identity={**identity,'legal_digest':digest(legal),'targets_digest':digest(targets),
              'support_digest':digest(support),'bootstrap_sha256':sha256_bytes(draws)}
    return D3Inputs(*(canonical_json_bytes(v) for v in (legal,targets,support,identity)),draws).validate()


def prepare_formal(run_path,dataset_path,*,protocol):
    p=protocol.record['inputs']
    anchors=SourceAnchors(p['D2_complete_digest'],p['D2_source_revision'],p['D1c_digest'],p['D2_split_digest'])
    view=read_formal_d2(run_path,anchors=anchors)
    link=crosscheck_d1c(view,dataset_path,expected_dataset_digest=anchors.dataset_digest)
    outer=verify_outer_artifacts(run_path,view,link)
    target=open_d1c_targets(view,link)
    legal={};sources={s.origin.sample_id:s.origin for s in view.samples}
    for s in view.samples:
        o=s.origin
        legal[o.sample_id]={'game_id':o.game_id,'phase':o.phase,'nW':o.nW,'tom_fold':o.tom_fold,'outer_fold':o.outer_fold,
            'p':{w:{m:s.probabilities[wi*4+mi] for mi,m in enumerate(MODELS)} for wi,w in enumerate(WEIGHTINGS)}}
    rows={r.sample_id:r for r in link.legal_rows}
    support={w+'/'+m:{} for w in WEIGHTINGS for m in MODELS}
    # Bulk parsing uses the same resolver as resolve_training_support; no repeated model JSON parsing per sample.
    for model in outer.models:
        m,i=model.record,model.identity
        for sid in i['test_legal_by_id']:
            s=_support(m,i,sources[sid],rows[sid])
            support[i['weighting']+'/'+i['model_id']][sid]=({'level':'NOT_APPLICABLE','games':None,'kish':None,'fallback':None}
                if s['applicability']=='NOT_APPLICABLE' else {'level':s['level'],'games':s['training_stats']['games'],
                    'kish':s['training_stats']['kish_games'],'fallback':s['fallback']})
    final,record=_unit(Path(run_path)/'complete','validation_complete',anchors.complete_digest,
                       extras=('bootstrap_indices.int64le',))
    order=record['summary']['game_order'];strata={r['game_id']:[r['tom_fold'],r['outer_fold']] for r in legal.values()}
    if len(order)!=p['games'] or len(legal)!=p['samples'] or len(set(map(tuple,strata.values())))!=p['outer_blocks']:
        raise ValueError('formal D3 population differs from registered D1/D2')
    identity={'scope':'formal_D3_posthoc_development','samples':len(legal),'game_sizes':view.identity['game_sizes'],
        'game_order':order,'game_strata':strata,'D2':view.identity,'D1c_link_digest':digest(link.identity),
        'outer_audit':outer.report,'research_scope':protocol.record['research_scope'],
        'limitations':protocol.record['limitations']}
    labels, empty=target.labels, target.successful_empty
    return _assemble(legal,{sid:{'y':labels[sid],'successful_empty':empty[sid]} for sid in legal},
                     support,identity,read_artifact_file(final,'bootstrap_indices.int64le'))


def controlled_inputs():
    """26 artificial games in 25 strata; unequal lengths, no formal source claim."""
    legal={};targets={};strata={};support={w+'/'+m:{} for w in WEIGHTINGS for m in MODELS}
    for k in range(5):
        for j in range(5):
            for g in range(2 if (k,j)==(0,0) else 1):
                game=f'artificial-{k}-{j}-{g}';strata[game]=[k,j]
                for n in range(g+1):
                    sid=game+'-'+str(n);phase='discussion' if j%2 else 'pk_discussion';nw=1+k%2
                    # All models have controlled probabilities, including exact binary endpoints/boundaries.
                    legal[sid]={'game_id':game,'phase':phase,'nW':nw,'tom_fold':k,'outer_fold':j,
                        'p':{w:{m:(.5 if m=='OBS-EQ0' else (.0,.25,.5,.75,1.)[(j+n)%5]) for m in MODELS} for w in WEIGHTINGS}}
                    targets[sid]={'y':(j+n+k)%2,'successful_empty':bool((j+n+k)%2 and j==1)}
                    for w in WEIGHTINGS:
                        for m in MODELS:
                            support[w+'/'+m][sid]=({'level':'NOT_APPLICABLE','games':None,'kish':None,'fallback':None} if m=='OBS-LR-r'
                                else {'level':('global','nW','phase_nW','r_bin','r_bin')[j], 'games':4+j*12,'kish':2.+j*10,
                                      'fallback':j<2})
    from werewolf.observer_dtrs.d2_protocol import frozen_protocol
    from werewolf.observer_dtrs.statistical_analysis import bootstrap_plan
    blocks={(k,j):[g for g,s in strata.items() if s==[k,j]] for k in range(5) for j in range(5)}
    order,draws=bootstrap_plan(blocks,protocol=frozen_protocol(),formal=False)
    identity={'scope':'controlled_fixture','samples':len(legal),'game_sizes':dict(Counter(r['game_id'] for r in legal.values())),
        'game_order':order,'game_strata':strata,'research_scope':'artificial_engineering_acceptance_only',
        'limitations':['not_formal_ToM_D1c_D2_data'],'source':'hand_specified_fixture_not_a_publication'}
    return _assemble(legal,targets,support,identity,draws.astype('<i8').tobytes())
