"""共享公开输入与一次性软件身份；不包含任何推理执行器。"""

import os
from pathlib import Path
import subprocess

from werewolf.artifact_io import publish_artifact, verify_artifact, canonical_jsonl_bytes, sha256_bytes
from werewolf.tom_llm.common import digest, identity, require, require_digest, strict_json
from werewolf.tom_llm.task import SET_VERSION, task_binding, validate_task

COHORT_VERSION = 'classic7_multimodel_public_cohort_v1'


def source_identity():
    """每次运行开始和结束各核验一次实际源码，不在逐请求路径扫描。"""
    root = Path(__file__).resolve().parents[2]
    paths = list((root/'werewolf').rglob('*.py'))
    paths += list((root/'werewolf/tom_llm/assets').iterdir())
    paths += list((root/'werewolf/tom_llm').rglob('*.jinja'))
    paths += list((root/'configs/paper').glob('tom-llm-*.json'))
    paths += [root/'setup.py']
    files = {}
    for path in sorted(set(paths)):
        require(path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)), 'Source symlink')
        files[path.relative_to(root).as_posix()] = sha256_bytes(path.read_bytes())
    env = {k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_OPTIONAL_LOCKS='0',GIT_NO_LAZY_FETCH='1')
    def git(*args):
        return subprocess.check_output(['git','--no-optional-locks','-c','core.fsmonitor=false',
                                        '-C',str(root),*args],env=env).decode().strip()
    status = git('status','--porcelain=v1','--untracked-files=all','--',
                 'werewolf','configs/paper/tom-llm-*.json','setup.py')
    return {'schema_version':'classic7_tom_baseline_source_v1',
            'source_revision':git('rev-parse','HEAD'),'commit_tree':git('rev-parse','HEAD^{tree}'),
            'file_digests':files,'implementation_digest':digest(files),
            'source_inventory_clean':not bool(status),'source_status_digest':digest(status)}


def read_cohort(root, expected_digest):
    """旧 Cohort 作为不可变科学输入复用，其 Source 不再充当执行资格。"""
    require_digest(expected_digest)
    artifact = verify_artifact(root,expected_artifact_type='llm_multimodel_public_cohort',
                               expected_schema_version=COHORT_VERSION)
    m = artifact.manifest
    require(artifact.manifest_digest==expected_digest and set(m['file_table'])=={'requests.jsonl'},
            'cohort anchor/inventory drift')
    require(m['task']==task_binding(SET_VERSION),'only the frozen suspicion-set task is executable')
    selection = verify_artifact(m['selection_root'],expected_artifact_type='llm_game_selection',
                                expected_schema_version='classic7_llm_game_selection_v1')
    require(selection.manifest_digest==m['selection']['selection_artifact_manifest_digest'],'Selection anchor drift')
    selected = strict_json((selection.path/'selection.json').read_bytes())
    require(selected['game_ids']==m['selection']['game_ids']
            and selected['selection_digest']==m['selection']['selection_digest'],'Selection payload drift')
    assignments = [(g,f['fold']) for f in selected['folds'] for g in f['game_ids']]
    fold_of_game = dict(assignments)
    require(len(assignments)==len(fold_of_game) and set(fold_of_game)==set(selected['game_ids']),
            'Selection fold/game coverage drift')
    tasks = [strict_json(line) for line in (artifact.path/'requests.jsonl').read_bytes().splitlines()]
    for task in tasks:
        validate_task(task)
        require(task.get('schema_version')==SET_VERSION,'mixed cohort task versions')
        require(task['fold']==fold_of_game.get(task['game_id']),'task/Selection fold mismatch')
    require(tasks and len({identity(t) for t in tasks})==len(tasks)
            and digest(tasks)==m['requests_digest']
            and digest([identity(t) for t in tasks])==m['row_identity_digest'],'cohort coverage drift')
    require(set(t['game_id'] for t in tasks)==set(m['selection']['game_ids']),'cohort games drift')
    return artifact,tasks


def publish_cohort(destination, tasks, reference, *, selection_root, expected, provenance, prepared_manifest):
    """只发布独立构造的公开 PRE；不把标签放入请求。"""
    from werewolf.tom_llm.selection_binding import open_selection_binding
    from werewolf.tom_llm.multimodel_evaluation import reference_binding,validate_public_coverage
    selection = open_selection_binding(selection_root,reference.folds,expected=expected,
        provenance=provenance,prepared_manifest=prepared_manifest)
    require(selection['publication_digest']==reference.publication_digest
            and selection['fold_manifest_digest']==reference.fold_manifest_digest,'Selection/Reference mismatch')
    selected = strict_json((Path(selection_root)/'selection.json').read_bytes())
    validate_public_coverage(tasks,reference,selected['game_ids'])
    require(all(t.get('schema_version')==SET_VERSION for t in tasks),'only set tasks may be published')
    return publish_artifact(destination,manifest_fields={
        'artifact_type':'llm_multimodel_public_cohort','schema_version':COHORT_VERSION,'engineering_only':True,
        'task':task_binding(SET_VERSION),'source':source_identity(),'reference':reference_binding(reference),
        'selection':{**selection,'game_ids':selected['game_ids']},'selection_root':str(Path(selection_root).absolute()),
        'requests_digest':digest(tasks),'row_identity_digest':digest([identity(t) for t in tasks])},
        files={'requests.jsonl':canonical_jsonl_bytes(tasks)})
