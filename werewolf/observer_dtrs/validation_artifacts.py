"""Immutable nested-run contracts/checkpoints; no mutable completion flag."""
from contextlib import contextmanager, redirect_stdout
import fcntl
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

from werewolf.artifact_io import canonical_json_bytes, publish_artifact, read_artifact_file, sha256_bytes, verify_artifact

VERSION = 'observer_dtrs_nested_validation_v1'


def digest(value):
    return sha256_bytes(canonical_json_bytes(value))


def source_snapshot(*, formal, expected_revision=None):
    root = Path(__file__).resolve().parents[2]
    paths = sorted([*root.joinpath('werewolf').rglob('*.py'),
        root/'scripts/run_observer_dtrs_validation.py',root/'configs/observer_dtrs/d2-execution-v1.json'])
    head = subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True,check=True).stdout.strip()
    names=[str(p.relative_to(root)) for p in paths]
    if formal:
        if expected_revision != head:
            raise ValueError('formal execution requires explicitly pinned current source commit')
        tracked=set(subprocess.run(['git','ls-files'],cwd=root,capture_output=True,text=True,check=True).stdout.splitlines())
        dirty=subprocess.run(['git','diff','HEAD','--','werewolf','scripts/run_observer_dtrs_validation.py','configs/observer_dtrs/d2-execution-v1.json'],cwd=root,capture_output=True,text=True,check=True).stdout
        if not set(names) <= tracked or dirty:
            raise ValueError('formal experiment requires committed, unchanged production sources; no automatic commit')
    stream=io.StringIO()
    with redirect_stdout(stream):
        np.show_config()
    return {'head':head,'files':{str(p.relative_to(root)):p.read_text() for p in paths},
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'machine':platform.machine(),
            'numpy_configuration':stream.getvalue(),
            'thread_environment':{key:os.environ.get(key) for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}}}


def publish_unit(path, kind, identity, record, extra=None):
    return publish_artifact(path,manifest_fields={'artifact_type':'observer_dtrs_'+kind,
        'schema_version':VERSION,'identity':identity,'identity_digest':digest(identity)},
        files={'record.json':canonical_json_bytes(record),**(extra or {})})


def read_unit(path, kind, identity, extra_names=()):
    a=verify_artifact(path,expected_artifact_type='observer_dtrs_'+kind,expected_schema_version=VERSION)
    if (a.manifest['identity_digest'] != digest(identity) or a.manifest['identity'] != identity
            or set(a.manifest['file_table']) != {'record.json',*extra_names}):
        raise ValueError('checkpoint identity/inventory mismatch; stop without overwriting')
    raw=read_artifact_file(a,'record.json')
    value=json.loads(raw)
    if raw != canonical_json_bytes(value):
        raise ValueError('noncanonical run checkpoint')
    return a,value


@contextmanager
def run_lock(root):
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    with (root/'runner.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('another nested executor owns this run') from error
        try:
            yield
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN)


def record_failure(root, contract_digest, stage, error):
    directory=Path(root)/'failures'
    count=0
    while (directory/f'{count:06}').exists():
        count+=1
    return publish_unit(directory/f'{count:06}','validation_failure',
        {'contract_digest':contract_digest,'attempt':count},
        {'status':'FAILED_NOT_COMPLETE','stage':stage,'exception':type(error).__name__,'message':str(error)[:2000]})
