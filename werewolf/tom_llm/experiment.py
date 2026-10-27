"""Shared experiment plans and Source v3; frozen task/metric implementations are reused."""

from pathlib import Path
import os
import re
import subprocess

from werewolf.artifact_io import publish_artifact, verify_artifact, canonical_json_bytes
from werewolf.tom_llm.common import digest, identity, require, strict_json
from werewolf.tom_llm.model_spec import ROOT, ModelSpec
from werewolf.tom_llm.multimodel import source_identity, read_cohort

VERSION = 'classic7_multimodel_experiment_v1'
POLICY = {'schema_version':VERSION,'formal_games':150,
    'technical_selection':{'version':'tom_llm_technical_subset_v1','seed':20261026},
    'metrics':['KL','TV','Top-1_NU'],'difference_direction':'left_minus_right',
    'transport_attempts':2,'unknown_outcome':'stop_without_resend',
    'efficiency_scope':'per_attempt_client_wall_includes_request_conversion_http_service; request_loop_wall_includes_per_row_artifacts; excludes_startup_physical_preparation_final_seal_analysis; TTFT_allocator_peak_not_measured'}


def protocol():
    require(strict_json((ROOT/'configs/paper/tom-llm-experiment-v1.json').read_bytes()) == POLICY,
            'experiment policy drift')
    return POLICY


def source_v3():
    old = source_identity()
    files = dict(old['file_digests'])
    for relative in ('scripts/run_tom_multimodel_baseline.py','scripts/analyze_tom_multimodel_baseline.py',
                     'configs/paper/tom-llm-experiment-v1.json'):
        path = ROOT/relative
        require(not any(p.is_symlink() for p in (path,*path.parents)), 'Source v3 symlink')
        from werewolf.artifact_io import sha256_bytes
        files[relative] = sha256_bytes(path.read_bytes())
    env = {k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_OPTIONAL_LOCKS='0',GIT_NO_LAZY_FETCH='1')
    status = subprocess.check_output(['git','--no-optional-locks','-c','core.fsmonitor=false','-C',str(ROOT),
        'status','--porcelain=v1','--untracked-files=all','--',*sorted(files)],env=env).decode().strip()
    return {**old,'schema_version':'classic7_multimodel_source_v3','file_digests':files,
            'implementation_digest':digest(files),'source_inventory_clean':not bool(status),
            'source_status_digest':digest(status),'qualification':'SOFTWARE_IDENTITY_ONLY_NOT_ADMISSION'}


def safe_path(path):
    p = Path(path).absolute()
    require('..' not in p.parts, 'experiment path contains parent traversal')
    require(not any(q.is_symlink() for q in (p,*p.parents)), 'experiment path traverses symlink')
    return p


def publish_record(path, kind, record):
    path = safe_path(path)
    require(not path.exists(), 'artifact exists; no replacement')
    return publish_artifact(path,manifest_fields={'artifact_type':kind,'schema_version':VERSION,
        'record_digest':digest(record)},files={'record.json':canonical_json_bytes(record)})


def read_record(path, kind, expected=None):
    a = verify_artifact(safe_path(path),expected_artifact_type=kind,expected_schema_version=VERSION)
    require(expected is None or a.manifest_digest == expected, 'external artifact anchor drift')
    require(set(a.manifest['file_table']) == {'record.json'}, 'record inventory drift')
    r = strict_json((a.path/'record.json').read_bytes())
    require(digest(r) == a.manifest['record_digest'], 'record payload drift')
    return a,r


def choose_games(selection, count, mode):
    require(mode in ('formal','technical','fake'), 'unknown experiment mode')
    games = selection['game_ids']
    require(type(count) is int and 0 < count <= len(games), 'games counts whole games in existing Selection')
    if mode == 'formal':
        require(count == len(games) == protocol()['formal_games'] and selection['scope']=='subset150'
                and [len(f['game_ids']) for f in selection['folds']] == [30]*5,
                'formal requires the complete frozen 150 games, 30/fold')
        return sorted(games)
    rule = protocol()['technical_selection']
    return sorted(sorted(games,key=lambda g:(digest([rule['version'],rule['seed'],g]),g))[:count])


def prepare_run(destination, *, run_id, models, games, mode, cohort_root, expected_cohort,
                publication, evaluation, expected_reference, configuration=None):
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',run_id) is not None, 'invalid run-id')
    require(models and len(set(models)) == len(models), 'empty/duplicate model list')
    cohort,tasks = read_cohort(cohort_root,expected_cohort,current_source=False)
    selection = strict_json((Path(cohort.manifest['selection_root'])/'selection.json').read_bytes())
    selected = choose_games(selection,games,mode)
    tasks = [t for t in tasks if t['game_id'] in set(selected)]
    require({t['game_id'] for t in tasks} == set(selected), 'whole-game public request coverage mismatch')
    require(cohort.manifest['reference']['seal_digest'] == expected_reference, 'Reference anchor drift')
    record = {'run_id':run_id,'mode':mode,'models':{k:ModelSpec(k).digest for k in models},
        'model_order':list(models),'configuration':configuration or {},'games':selected,'game_count':games,'cohort_root':str(cohort.path),'cohort_digest':cohort.manifest_digest,
        'selection':cohort.manifest['selection'],'reference':cohort.manifest['reference'],
        'publication':str(Path(publication).absolute()),'evaluation':str(Path(evaluation).absolute()),
        'source':source_v3(),'protocol':protocol(),'task_digests':[digest(t) for t in tasks],
        'row_identity_digest':digest([identity(t) for t in tasks]),'request_count':len(tasks)}
    return publish_record(Path(destination)/'contract','mm_baseline_experiment',record)


def open_run(root, expected=None, *, current_source=True):
    a,r = read_record(Path(root)/'contract','mm_baseline_experiment',expected)
    require(r['protocol'] == protocol(), 'fixed metrics/protocol drift')
    if current_source: require(r['source'] == source_v3(), 'experiment Source v3 drift; rebuild, never re-sign')
    from werewolf.tom_llm.deployment import validate_source_evidence
    validate_source_evidence(r['source'])
    c,tasks = read_cohort(r['cohort_root'],r['cohort_digest'],current_source=False)
    require(r['selection'] == c.manifest['selection'] and r['reference'] == c.manifest['reference'], 'common cohort anchors drift')
    selection = strict_json((Path(c.manifest['selection_root'])/'selection.json').read_bytes())
    require(r['games'] == choose_games(selection,r['game_count'],r['mode']), 'game selection drift')
    tasks = [t for t in tasks if t['game_id'] in set(r['games'])]
    require(r['task_digests'] == [digest(t) for t in tasks] and r['request_count']==len(tasks)
            and r['row_identity_digest']==digest([identity(t) for t in tasks]), 'ordered task coverage drift')
    for key,want in r['models'].items(): require(ModelSpec(key).digest==want, 'ModelSpec revision drift')
    return a,r,tasks
