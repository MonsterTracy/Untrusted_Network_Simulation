"""Reviewable working-byte snapshots. A snapshot is not a deployment approval/OID."""
import json
import os
from pathlib import Path
import stat
import subprocess

from werewolf.artifact_io import canonical_json_bytes, publish_artifact, read_artifact_file, sha256_bytes, verify_artifact

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'observer_dtrs_qual_source_snapshot_v1'
ENTRIES = ('run_random.py', 'scripts/build_final_tom_capsule.py',
           'scripts/serve_final_tom_capsule.py', 'scripts/qualify_final_tom_capsule.py',
           'scripts/export_final_tom_qual_reference.py')


def digest(value):
    return sha256_bytes(canonical_json_bytes(value))


def require(value, reason):
    if not value:
        raise ValueError(reason)


def new_external_path(path, *, protected=()):
    """Existing canonical parent; never write inside a checkout or artifact."""
    p = Path(path).absolute()
    require('..' not in p.parts and p.parent.resolve(strict=True) == p.parent,
            'canonical output parent required')
    require(not p.exists() and not p.is_symlink(), 'output already exists')
    roots = [ROOT.resolve(strict=True)]
    for root in protected:
        q = Path(root).resolve(strict=True)
        roots.append(q if q.is_dir() else q.parent)
    identities = {(q.stat().st_dev, q.stat().st_ino) for q in roots}
    for parent in p.parents:
        info = parent.stat()
        require(parent not in roots and (info.st_dev, info.st_ino) not in identities,
                'protected output tree')
        require(not any((parent/n).exists() or (parent/n).is_symlink() for n in
                        ('.git', 'manifest.json', 'qwen3_final_manifest.json',
                         'phase_codebook.manifest.json')), 'output inside source/artifact')
    return p


def capture_sources(root=ROOT):
    """All project runtime Python + explicit CLIs/configs, not data or test output."""
    root = Path(root).absolute()
    require(root.resolve(strict=True) == root, 'source alias')
    paths = list((root/'werewolf').rglob('*.py')) + [root/n for n in ENTRIES]
    paths += list((root/'configs/observer_dtrs').glob('*.json'))
    paths.append(root/'configs/paper/tom-backbone-study-v1.json')
    payload, inventory = {}, {}
    for path in sorted(paths):
        require(path.is_file() and path.resolve(strict=True) == path, 'missing/aliased source')
        name = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        inventory[name] = {'sha256': sha256_bytes(raw), 'bytes': len(raw),
                           'mode': stat.S_IMODE(path.stat().st_mode)}
        payload[name] = raw
    return inventory, payload


def _git(root, *args):
    env = {k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
    env['GIT_OPTIONAL_LOCKS'] = '0'
    return subprocess.check_output(['git', '-C', str(root), *args], env=env, timeout=10).decode().strip()


def publish_snapshot(destination):
    output = new_external_path(destination)
    before, payload = capture_sources()
    record = {'schema_version': VERSION, 'base_head': _git(ROOT, 'rev-parse', 'HEAD'),
              'branch': _git(ROOT, 'branch', '--show-current'),
              'scope': 'REVIEWABLE_WORKING_BYTES_NOT_FORMAL_SOURCE_AUTHORIZATION',
              'inventory': before, 'inventory_digest': digest(before),
              'git_head_is_complete_D4_identity': False}
    require(capture_sources()[0] == before, 'source changed during snapshot')
    artifact = publish_artifact(output, manifest_fields={'artifact_type': VERSION,
        'schema_version': VERSION, 'inventory_digest': record['inventory_digest']},
        files={'snapshot.json': canonical_json_bytes(record),
               **{'sources/'+n:v for n,v in payload.items()}})
    require(capture_sources()[0] == before, 'source changed; snapshot not usable for execution')
    return artifact


def verify_snapshot(path, expected_digest, *, root=ROOT):
    a = verify_artifact(path, expected_artifact_type=VERSION, expected_schema_version=VERSION)
    require(a.manifest_digest == expected_digest, 'external source snapshot digest mismatch')
    record = json.loads(read_artifact_file(a, 'snapshot.json'))
    require(set(record) == {'schema_version', 'base_head', 'branch', 'scope', 'inventory',
                           'inventory_digest', 'git_head_is_complete_D4_identity'}
            and record['schema_version'] == VERSION
            and record['scope'] == 'REVIEWABLE_WORKING_BYTES_NOT_FORMAL_SOURCE_AUTHORIZATION'
            and record['git_head_is_complete_D4_identity'] is False, 'snapshot scope/schema')
    inventory, payload = capture_sources(root)
    require(inventory == record['inventory'] and digest(inventory) == record['inventory_digest']
            == a.manifest['inventory_digest'], 'runtime source inventory mismatch')
    require(set(a.manifest['file_table']) == {'snapshot.json'} | {'sources/'+n for n in inventory},
            'snapshot payload inventory')
    for n, raw in payload.items():
        require(read_artifact_file(a, 'sources/'+n) == raw, 'source snapshot byte mismatch')
    return record
