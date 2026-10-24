"""Reviewed D3 execution candidate; explicit freeze and committed source still required."""
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path

from werewolf.artifact_io import canonical_json_bytes
from werewolf.observer_dtrs.decision import LossSpec, TiePolicy, candidate_losses, thresholds
from werewolf.observer_dtrs.statistical_analysis import MODELS, WEIGHTINGS
from werewolf.observer_dtrs.validation_artifacts import digest, source_snapshot

CONFIG_PATH = Path(__file__).resolve().parents[2]/'configs/observer_dtrs/d3-execution-v1.json'
VERSION = 'observer_dtrs_d3_execution_protocol_v1'


@dataclass(frozen=True)
class D3Protocol:
    payload: bytes

    def __post_init__(self):
        p = self.record
        if (self.payload != canonical_json_bytes(p) or p['schema_version'] != VERSION
                or p['status'] not in ('READY_FOR_REVIEW_NOT_FROZEN','FROZEN_FOR_EXECUTION')
                or p['models'] != list(MODELS) or p['weightings'] != list(WEIGHTINGS)
                or p['primary']['tie_policy'] != TiePolicy.TERMINAL_NEG_FIRST.value
                or p['sensitivity_tie_policy'] != TiePolicy.DEFER_FIRST.value
                or p['binary_tie_policy'] != TiePolicy.TERMINAL_NEG_FIRST.value
                or p['bootstrap']['replicates'] != 2000):
            raise ValueError('invalid D3 execution contract')
        expected = [{'id':f'C{i:02}', 'exact_costs':l.to_record()['exact_costs']}
                    for i,l in enumerate(candidate_losses(tie_policy=TiePolicy.TERMINAL_NEG_FIRST))]
        if p['loss_scenarios'] != expected:
            raise ValueError('changed registered 45 costs')
        # Executed rules must match the reviewed implementation, not decorative metadata.
        reference = json.loads(CONFIG_PATH.read_bytes())
        if {k:v for k,v in p.items() if k!='status'} != {k:v for k,v in reference.items() if k!='status'}:
            raise ValueError('unsupported D3 rule change; review implementation before execution')

    @property
    def record(self):
        return json.loads(self.payload)

    @property
    def digest(self):
        return digest(self.record)

    @property
    def policies(self):
        return (TiePolicy(self.record['primary']['tie_policy']), TiePolicy(self.record['sensitivity_tie_policy']))

    def loss(self, scenario, policy):
        if policy not in self.policies or type(scenario) is not int or scenario not in range(45):
            raise ValueError('unregistered scenario/policy')
        r = self.record['loss_scenarios'][scenario]['exact_costs']
        loss = LossSpec(**{k:Fraction(v) for k,v in r.items()},tie_policy=policy)
        if thresholds(loss).regime != 'NONDEGENERATE':
            raise ValueError('main scenario must be nondegenerate')
        return loss

    def require_frozen(self, expected_digest):
        if self.record['status'] != 'FROZEN_FOR_EXECUTION' or expected_digest != self.digest:
            raise ValueError('formal D3 requires separately reviewed frozen protocol and explicit digest')


def load_protocol(path=CONFIG_PATH):
    return D3Protocol(canonical_json_bytes(json.loads(Path(path).read_bytes())))


def cost_edges():
    edges=[]
    for a in range(5):
        for b in range(3):
            for c in range(3):
                i=a*9+b*3+c
                for axis,valid,j in (('rho',a<4,i+9),('kappa',b<2,i+3),('h',c<2,i+1)):
                    if valid:
                        edges.append((i,j,axis))
    return tuple(edges)


def checked_source(expected_revision):
    """Keep D2's production guard, and extend tracking checks to the D3 entry/config."""
    snapshot=source_snapshot(formal=True,expected_revision=expected_revision)
    import subprocess
    root=Path(__file__).resolve().parents[2]
    extra=('scripts/run_observer_dtrs_decisions.py','configs/observer_dtrs/d3-execution-v1.json')
    tracked=set(subprocess.check_output(['git','ls-files'],cwd=root,text=True).splitlines())
    dirty=subprocess.check_output(['git','diff','HEAD','--',*extra],cwd=root,text=True)
    if not set(extra)<=tracked or dirty:
        raise ValueError('formal D3 entry/config must be committed and unchanged')
    captured={**snapshot,'D3_execution_files':{p:(root/p).read_text() for p in extra}}
    if snapshot['head']!=expected_revision:raise ValueError('D3 source OID mismatch')
    files={**snapshot['files'],**captured['D3_execution_files']}
    if any('\n' in p or '\r' in p or Path(p).is_absolute() or '..' in Path(p).parts for p in files):
        raise ValueError('unsafe D3 source path')
    names=sorted(files)
    query=''.join(f'{expected_revision}:{p}\n' for p in names).encode()
    raw=subprocess.check_output(['git','cat-file','--batch'],cwd=root,input=query)
    offset=0
    for name in names:
        end=raw.find(b'\n',offset)
        header=raw[offset:end].split()
        if end<0 or len(header)!=3 or header[1]!=b'blob':raise ValueError('missing D3 committed source blob')
        size=int(header[2]);offset=end+1;payload=raw[offset:offset+size];offset+=size
        if payload!=files[name].encode() or raw[offset:offset+1]!=b'\n':
            raise ValueError('captured D3 source differs from committed blob')
        offset+=1
    if offset!=len(raw) or subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()!=expected_revision:
        raise ValueError('D3 source HEAD changed during verification')
    captured['D3_blob_verification']={'revision':expected_revision,'files':len(files),'status':'exact_match'}
    return captured
