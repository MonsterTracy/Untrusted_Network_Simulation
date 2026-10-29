"""CPU-only physical assets/environment receipts; never imports model runtimes."""

import hashlib
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys

from werewolf.artifact_io import publish_artifact, verify_artifact, canonical_json_bytes
from werewolf.tom_llm.common import digest, require, require_digest, strict_json
from werewolf.tom_llm.model_spec import ModelSpec, ROOT

VERSION = 'classic7_multimodel_physical_receipt_v1'
ASSET_PATH = ROOT/'configs/paper/tom-llm-physical-assets-v1.json'
ASSET_SHA = '52b32dba9721911932176bd4c4aef3fc6607483a97f7eb122390c571bfd624c1'
CORE_PACKAGES = ('vllm', 'torch', 'transformers', 'tokenizers', 'openai', 'httpx2',
                 'flashinfer-python', 'apache-tvm-ffi', 'triton', 'Jinja2', 'numpy')


def regular_root(path):
    p = Path(path).absolute()
    require(p.is_dir() and not any(x.is_symlink() for x in (p,*p.parents)), 'physical directory missing/symlink')
    return p


def file_identity(path, *, git_blob=False):
    p = Path(path)
    require(p.is_file() and not any(x.is_symlink() for x in (p,*p.parents)), 'physical file missing/symlink')
    a = p.stat(); sha = hashlib.sha256()
    oid = hashlib.sha1(f'blob {a.st_size}\0'.encode()) if git_blob else None
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            sha.update(block)
            if oid is not None: oid.update(block)
    b = p.stat()
    require((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns), 'file changed while hashing')
    return {'bytes':b.st_size,'sha256':sha.hexdigest(),'git_blob_oid':oid.hexdigest() if oid else None}


def asset_spec(spec):
    require(type(spec) is ModelSpec, 'explicit model required')
    raw = ASSET_PATH.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==ASSET_SHA, 'official physical asset specification drift')
    result = strict_json(raw)[spec.key]
    require(result['model']==spec.record()['model'] and result['revision']==spec.record()['revision'], 'physical ModelSpec drift')
    return result


def verify_model(spec, directory):
    """Stream official shards; require exact index/shard coverage, never load tensors."""
    root = regular_root(directory); expected = asset_spec(spec)
    inventory = expected['files']
    entries = list(root.rglob('*'))
    require(not any(p.is_symlink() for p in entries) and
            {p.relative_to(root).as_posix() for p in entries if p.is_file()}==set(inventory)|{'HF_REVISION'},
            'physical model inventory mismatch')
    require((root/'HF_REVISION').read_text()==expected['revision']+'\n', 'physical model revision marker drift')
    files = {}
    for name,want in inventory.items():
        observed = file_identity(root/name,git_blob=want['git_blob_oid'] is not None)
        require(observed['bytes']==want['bytes'] and
                (want['sha256'] is None or observed['sha256']==want['sha256']) and
                observed['git_blob_oid']==want['git_blob_oid'], 'official physical bytes drift: '+name)
        files[name] = observed
    files['HF_REVISION'] = file_identity(root/'HF_REVISION')
    index = strict_json((root/'model.safetensors.index.json').read_bytes())
    require(type(index.get('weight_map')) is dict and index['weight_map'] and
            set(index['weight_map'].values())=={n for n in inventory if n.endswith('.safetensors')},
            'weight index/shard coverage mismatch')
    return {'schema_version':VERSION+'_model','evidence_kind':'physical_cpu','model_spec_digest':spec.digest,
            'model':expected['model'],'revision':expected['revision'],'asset_spec_sha256':ASSET_SHA,
            'directory':str(root),'files':files,'weights_loaded':False,'gpu_qualification':False}


def capture_environment(prefix, *, prior_audit_digest):
    """Bind actual installed metadata and critical code, retain prior RECORD deviations."""
    require_digest(prior_audit_digest)
    require(sys.version_info[:2]==(3,12),'qualification recipe requires the isolated Python 3.12 environment')
    prefix = regular_root(prefix)
    require(Path(sys.prefix).resolve()==prefix and Path(sys.executable).resolve().is_relative_to(prefix),
            'CPU environment audit must use its own isolated interpreter')
    versions, metadata = {}, {}
    for name in CORE_PACKAGES:
        dist = importlib.metadata.distribution(name)
        require(Path(dist.locate_file('')).resolve().is_relative_to(prefix), 'package escaped isolated prefix')
        versions[name] = dist.version
        names = [p for p in (dist.files or []) if p.name in ('METADATA','RECORD') and '.dist-info' in str(p)]
        require(len(names)==2, 'installed package metadata/RECORD absent: '+name)
        metadata[name] = {str(p):file_identity(dist.locate_file(p)) for p in names}
    from werewolf.tom_llm.deployment import recipe
    require(versions['vllm']==recipe()['vllm_version'], 'unsupported vLLM version')
    critical = {}
    for relative,wanted in recipe()['vllm_source_sha256'].items():
        p = importlib.metadata.distribution('vllm').locate_file('vllm/'+relative)
        critical['vllm/'+relative] = file_identity(p)
        require(critical['vllm/'+relative]['sha256']==wanted, 'vLLM instrumentation seam code drift')
    for dist_name,path in [('openai','openai/_base_client.py'),('torch','torch/__init__.py'),
                           ('flashinfer-python','flashinfer/__init__.py'),('apache-tvm-ffi','tvm_ffi/__init__.py')]:
        critical[path] = file_identity(importlib.metadata.distribution(dist_name).locate_file(path))
    check = subprocess.run([sys.executable,'-I','-B','-m','pip','check'],capture_output=True,text=True,timeout=120)
    if check.returncode!=0:
        raise ValueError(f'dependency integrity failure (exit {check.returncode}); do not repair existing environment\n'
                         f'stdout:\n{check.stdout}\nstderr:\n{check.stderr}')
    return {'schema_version':VERSION+'_environment','evidence_kind':'physical_cpu','prefix':str(prefix),
            'python':sys.version,'executable':file_identity(Path(sys.executable).resolve()),
            'versions':versions,'distribution_metadata':metadata,'critical_files':critical,
            'pip_check':check.stdout,'prior_audit_digest':prior_audit_digest,
            'record_integrity':'PRIOR_STRICT_FAILURES_RETAINED_NOT_REPAIRED','gpu_runtime_imported':False}


def publish_receipt(destination, kind, record):
    require(kind in ('model','environment') and record['schema_version']==VERSION+'_'+kind
            and record['evidence_kind']=='physical_cpu', 'physical CPU receipt domain mismatch')
    return publish_artifact(destination,manifest_fields={'artifact_type':'mm_physical_'+kind,
        'schema_version':VERSION,'record_digest':digest(record),'qualification':'CPU_ONLY_NOT_GPU_ADMISSION'},
        files={'record.json':canonical_json_bytes(record)})


def read_receipt(root, kind, expected_digest):
    require_digest(expected_digest)
    a = verify_artifact(root,expected_artifact_type='mm_physical_'+kind,expected_schema_version=VERSION)
    require(a.manifest_digest==expected_digest and set(a.manifest['file_table'])=={'record.json'}, 'CPU receipt identity/inventory drift')
    record = strict_json((a.path/'record.json').read_bytes())
    require(digest(record)==a.manifest['record_digest'] and record['evidence_kind']=='physical_cpu'
            and record['schema_version']==VERSION+'_'+kind, 'CPU receipt payload drift')
    return record
