"""Isolated historical CPU GRAPH reference. NOT original sealed lifecycle replay.

Invoked by the local QUAL coordinator, never by Shadow or the resident process.
Reads only a pinned inference capsule and allowlisted public PRE packet. The
original open_fit/verify_seal are deliberately not called; semantic recovery
remains NOT_RUN and this output cannot certify original predictor initialization.
"""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

sys.dont_write_bytecode = True


def require(condition):
    if not condition:
        raise ValueError("REFERENCE_IDENTITY_OR_NUMERICAL_INPUT_REJECTED")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkout', required=True, type=Path)
    p.add_argument('--revision', required=True)
    p.add_argument('--implementation-digest', required=True)
    p.add_argument('--capsule', required=True, type=Path)
    p.add_argument('--capsule-digest', required=True)
    p.add_argument('--inputs', required=True, type=Path)
    p.add_argument('--inputs-sha256', required=True)
    p.add_argument('--threads', required=True, type=int)
    a = p.parse_args()
    root = a.checkout.absolute()
    require(root.resolve(strict=True) == root and 1 <= a.threads <= 64)
    env = {k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
    env['GIT_OPTIONAL_LOCKS'] = '0'
    def git(*args):
        return subprocess.check_output(['git','-C',str(root),*args], env=env, timeout=10).strip()
    require(git('rev-parse','HEAD').decode() == a.revision)
    require(not git('status','--porcelain=v1','--untracked-files=all'))
    paths = sorted((root/'werewolf').rglob('*.py'))+[root/'run_random.py']
    hashes = {}
    for path in paths:
        require(path.resolve(strict=True) == path)
        name = path.relative_to(root).as_posix();raw = path.read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        require(git('rev-parse','HEAD:'+name).decode() == hashlib.sha1(
            b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())
    # Drop the caller project from import resolution before any project imports.
    sys.path[:] = [str(root)]+[s for s in sys.path if s and 'Untrusted_Network_Simulation' not in s
                              and s != str(Path(__file__).resolve().parent)]
    from werewolf.artifact_io import canonical_json_bytes, read_artifact_file, sha256_bytes, verify_artifact
    digest = lambda r: sha256_bytes(canonical_json_bytes(r))
    require(digest(hashes) == a.implementation_digest)
    import torch
    from werewolf.tom.backbone_model import BackboneToM, graph_identity
    from werewolf.tom.dataset import ExperimentCapacity, tensorize_public_pre
    from werewolf.tom.state import restore_model_tensors
    from werewolf.tom.temporal import TemporalCodeProvider
    from werewolf.artifact_io.tensor_state import decode_tensor_container
    from werewolf.canonical_collection.game_bundle import _prefix_from_record
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    artifact = verify_artifact(a.capsule,
        expected_artifact_type='observer_dtrs_final_tom_inference_capsule',
        expected_schema_version='observer_dtrs_final_tom_inference_capsule_v1')
    require(artifact.manifest_digest == a.capsule_digest)
    spec = json.loads(read_artifact_file(artifact,'inference.json'))
    require(spec['graph'] == graph_identity('qwen3'))
    temporal = TemporalCodeProvider.explicit(artifact.path/'temporal',spec['temporal_parent_protocol'])
    model = BackboneToM('qwen3',temporal,study_seed=0,fold=0).eval()
    restore_model_tensors(model,decode_tensor_container(spec['model_container'],
        read_artifact_file(artifact,'model.bin'),payload_path='model.bin'))
    require(a.inputs.stat().st_size <= 16*1024*1024)
    raw = a.inputs.read_bytes();require(sha256_bytes(raw) == a.inputs_sha256)
    packet = json.loads(raw)
    require(canonical_json_bytes(packet) == raw)
    require(set(packet) == {'schema_version','scope','cases'})
    require(packet['schema_version'] == 'observer_dtrs_qual_public_pres_v1' and 1 <= len(packet['cases']) <= 32)
    def tensor(t):
        x = t.detach().cpu().numpy()
        return {'dtype':x.dtype.str,'shape':list(x.shape),'sha256':sha256_bytes(x.tobytes(order='C'))}
    rows=[]
    for row in packet['cases']:
        require(set(row) == {'case_id','pre'})
        prefix = _prefix_from_record(row['pre']);require(prefix.to_record() == row['pre'])
        plan, x = tensorize_public_pre(prefix,ExperimentCapacity(1024))
        arrays=x.kwargs()
        with torch.inference_mode(): logq=model(**{k:v[None] for k,v in arrays.items()})[0]
        nonself=[[float(logq[i,j]) for j in range(7) if i != j] for i in range(7)]
        rows.append({'case_id':row['case_id'],'pre_sha256':digest(row['pre']),
            'plan_digest':plan.plan_digest,'token_count':plan.token_count,
            'inputs':{k:tensor(v) for k,v in arrays.items()},
            'log_q_nonself':nonself,'q':logq.exp().tolist()})
    modules={n:m for n,m in sys.modules.items() if n=='werewolf' or n.startswith('werewolf.')}
    require(all(Path(m.__file__).resolve().is_relative_to(root) for m in modules.values()))
    require(hashes == {str(path.relative_to(root)):sha256_bytes(path.read_bytes()) for path in paths})
    report={'schema_version':'observer_dtrs_qual_reference_v1',
        'reference_kind':'PINNED_ORIGINAL_GRAPH_CPU_SAME_CAPSULE_NOT_SEALED_LIFECYCLE',
        'source_revision':a.revision,'implementation_digest':digest(hashes),
        'runtime':{'python':platform.python_version(),
            'packages':{n:metadata.version(n) for n in ('numpy','torch','transformers')},
            'device':'cpu','dtype':'float32','attention_backend':'eager',
            'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads()},
        'capsule_digest':a.capsule_digest,'inputs_raw_sha256':a.inputs_sha256,
        'model_tensors':{k:tensor(v) for k,v in model.state_dict().items()},
        'temporal':{k:sha256_bytes(getattr(temporal,k).tobytes()) for k in ('day','phase')},
        'full_semantic_recovery':'NOT_RUN','original_sealed_predictor':'NOT_RUN',
        'cases':rows}
    sys.stdout.buffer.write(canonical_json_bytes(report))
    return 0


if __name__ == '__main__':
    try:
        code = main()
    except Exception:
        sys.stderr.write('REFERENCE_FAILED_NO_FALLBACK\n');code=2
    raise SystemExit(code)
