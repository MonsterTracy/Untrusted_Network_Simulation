"""Shared experiment plans and Source v3; frozen task/metric implementations are reused."""

from pathlib import Path
import os
import re
import subprocess

from werewolf.artifact_io import publish_artifact, verify_artifact, canonical_json_bytes
from werewolf.tom_llm.common import digest, identity, require, strict_json
from werewolf.tom_llm.model_spec import ROOT, ModelSpec
from werewolf.tom_llm.multimodel import source_identity, read_cohort
from werewolf.tom_llm.task import VERSION as TASK_VERSION, SET_VERSION, task_binding

VERSION = 'classic7_multimodel_experiment_v1'
APPROVED_PROTOCOL_VERSION = 'classic7_suspicion_set_research_protocol_v2'
LIVE_SET_VERSION = 'classic7_suspicion_set_live_execution_v1'
FORMAL_PUBLIC_REQUESTS = 10714
FORMAL_PUBLIC_ROWS_DIGEST = '8fd129c892d9657ed429d984c83acc278792637a5e96e06eb9c148f3b98e80c8'
POLICY = {'schema_version':VERSION,'formal_games':150,
    'technical_selection':{'version':'tom_llm_technical_subset_v1','seed':20261026},
    'metrics':['KL','TV','Top-1_NU'],'difference_direction':'left_minus_right',
    'transport_attempts':2,'unknown_outcome':'stop_without_resend',
    'efficiency_scope':'per_attempt_client_wall_includes_request_conversion_http_service; request_loop_wall_includes_per_row_artifacts; excludes_startup_physical_preparation_final_seal_analysis; TTFT_allocator_peak_not_measured'}


def protocol(task_schema_version=TASK_VERSION, *, protocol_version=None):
    require(strict_json((ROOT/'configs/paper/tom-llm-experiment-v1.json').read_bytes()) == POLICY,
            'experiment policy drift')
    task_binding(task_schema_version)
    require(protocol_version is None or
            (task_schema_version==SET_VERSION and protocol_version==APPROVED_PROTOCOL_VERSION),
            'unsupported research protocol/task binding')
    if task_schema_version == TASK_VERSION:
        return POLICY
    from copy import deepcopy
    result = deepcopy(POLICY)
    result.update(schema_version='classic7_multimodel_suspicion_set_experiment_v2',
        status='CPU_MOCK_ONLY_NOT_FROZEN_NOT_EXECUTABLE', task_schema_version=SET_VERSION,
        metrics=['Set-F1', 'Set-Precision', 'Set-Recall', 'Jaccard', 'Exact-Set-Match', 'Valid-Output-Rate'],
        supplementary_metrics=['KL', 'TV', 'Top-1_NU'],
        set_metrics={'both_empty':1, 'exactly_one_empty':0, 'invalid_prediction':0,
                     'missing_prediction':'incomplete_run_no_analysis',
                     'unobserved_gold':'excluded_with_explicit_label_coverage'},
        probability_conversion={'nonempty':'uniform_on_predicted_set', 'empty':'uniform_non_self',
                                'epsilon':1e-6, 'invalid':'failure_uniform_not_a_predicted_set'},
        probability_to_set='BLOCKED_UNTIL_INDEPENDENT_DEVELOPMENT_RULE_SUPPLIED',
        calibration='independent_game_ids_and_source_qualification_required',
        comparison_scope='previously_observed_formal_games_method_comparison_not_independent_generalization_or_causal_effect',
        statistical_inference={'bootstrap':10000, 'confidence':.95, 'unit':'game',
                               'interval_method':'percentile_linear', 'simultaneous_intervals':False})
    if protocol_version==APPROVED_PROTOCOL_VERSION:
        from werewolf.tom_llm.output import fixed_projection_rule
        from werewolf.tom_llm.reference import fixed_oof_contract
        result.update(schema_version=APPROVED_PROTOCOL_VERSION,
            status='RESEARCH_APPROVED_CPU_ONLY_FORMAL_EXECUTION_BLOCKED',
            probability_to_set=fixed_projection_rule(), fixed_oof=fixed_oof_contract(),
            calibration='none_no_1350_game_fitting',
            primary_endpoint='full_primary_raw/set_f1', key_comparison='primary_comparable_nu/set_f1',
            baselines=['Always Empty','Always All-Six'],
            comparisons={'full_primary_raw':{'simple_baseline':6,'exploratory':4},
                         'primary_comparable_nu':{'key_method':3,'simple_baseline':8,'exploratory':4}},
            inference='descriptive_exploratory_pointwise_only_no_joint_superiority')
    return result


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


def live_set_binding(run,cohort,tasks):
    """验证真实 V2 输入及 Source；执行绑定不改冻结科研协议。"""
    from werewolf.tom_llm.deployment import validate_source_evidence
    from werewolf.tom_llm.qualification_inputs import open_budget
    m=cohort.manifest;source=run['source'];qualified=m['source']
    require(run['protocol']==protocol(SET_VERSION,protocol_version=APPROVED_PROTOCOL_VERSION),
            'V2 real execution requires the approved research protocol')
    require(run['configuration'].get('input_config',{}).get('task_version')==SET_VERSION,
            'V2 real execution requires the explicit configured set task')
    require(m['task']==task_binding(SET_VERSION), 'V2 live cohort task mismatch')
    for value in (source,qualified):
        validate_source_evidence(value)
        require(value['source_inventory_clean'] and value['source_status_digest']==digest(''),
                'V2 real execution requires committed clean Source')
    require(source['schema_version']=='classic7_multimodel_source_v3'
            and qualified['schema_version']=='classic7_multimodel_source_v2', 'V2 live Source schema mismatch')
    require(all(source[k]==qualified[k] for k in ('source_revision','commit_tree','integration_base_commit')),
            'V2 run/cohort Source revision mismatch')
    extra={'scripts/run_tom_multimodel_baseline.py','scripts/analyze_tom_multimodel_baseline.py',
           'configs/paper/tom-llm-experiment-v1.json'}
    require(set(source['file_digests'])-set(qualified['file_digests'])==extra
            and all(source['file_digests'].get(p)==h for p,h in qualified['file_digests'].items()),
            'V2 run/cohort Source inventory mismatch')
    require(m['reference']==run['protocol']['fixed_oof']['reference'], 'V2 live fixed OOF mismatch')
    require(len(m['selection']['game_ids'])==150 and len(tasks)==FORMAL_PUBLIC_REQUESTS
            and digest([identity(t) for t in tasks])==m['row_identity_digest']==FORMAL_PUBLIC_ROWS_DIGEST,
            'V2 live requires the complete frozen 150-game/10714-row cohort')
    budgets=run['configuration'].get('budgets',{})
    require(set(budgets)==set(run['models']), 'V2 live requires every planned model budget')
    for key,item in budgets.items():
        require(type(item) is dict and set(item)=={'path','digest'}, 'V2 live budget anchor required')
        budget=open_budget(item['path'],ModelSpec(key),item['digest'],
            task_schema_version=SET_VERSION,expected_source=qualified)
        require(budget['cohort_digest']==run['cohort_digest'] and budget['cohort_root']==str(cohort.path)
                and budget['request_count']==FORMAL_PUBLIC_REQUESTS
                and budget['row_identity_digest']==m['row_identity_digest'], 'V2 live budget/cohort coverage mismatch')
    return {'schema_version':LIVE_SET_VERSION,'task':m['task'],'research_protocol_digest':digest(run['protocol']),
            'qualification_source_digest':digest(qualified),'cohort_digest':run['cohort_digest'],
            'budget_manifests':{key:item['digest'] for key,item in budgets.items()},
            'authority':'physical_admission_and_per_model_execution_permit_required'}


def prepare_run(destination, *, run_id, models, games, mode, cohort_root, expected_cohort,
                publication, evaluation, expected_reference, configuration=None):
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',run_id) is not None, 'invalid run-id')
    require(models and len(set(models)) == len(models), 'empty/duplicate model list')
    cohort,tasks = read_cohort(cohort_root,expected_cohort,current_source=False)
    all_tasks=tasks
    task_schema_version = cohort.manifest['task']['schema_version']
    selection = strict_json((Path(cohort.manifest['selection_root'])/'selection.json').read_bytes())
    selected = choose_games(selection,games,mode)
    tasks = [t for t in tasks if t['game_id'] in set(selected)]
    require({t['game_id'] for t in tasks} == set(selected), 'whole-game public request coverage mismatch')
    require(cohort.manifest['reference']['seal_digest'] == expected_reference, 'Reference anchor drift')
    selected_protocol = protocol(task_schema_version,protocol_version=(configuration or {}).get('input_config',{}).get('protocol_version'))
    if selected_protocol['schema_version']==APPROVED_PROTOCOL_VERSION:
        require((configuration or {}).get('input_config',{}).get('probability_to_set') is None,
                'approved protocol forbids a calibrated or legacy threshold override')
    record = {'run_id':run_id,'mode':mode,'models':{k:ModelSpec(k).digest for k in models},
        'model_order':list(models),'configuration':configuration or {},'games':selected,'game_count':games,'cohort_root':str(cohort.path),'cohort_digest':cohort.manifest_digest,
        'selection':cohort.manifest['selection'],'reference':cohort.manifest['reference'],
        'publication':str(Path(publication).absolute()),'evaluation':str(Path(evaluation).absolute()),
        'source':source_v3(),'protocol':selected_protocol,'task_digests':[digest(t) for t in tasks],
        'row_identity_digest':digest([identity(t) for t in tasks]),'request_count':len(tasks)}
    if task_schema_version==SET_VERSION and mode!='fake':
        record['live_execution']=live_set_binding(record,cohort,all_tasks)
    return publish_record(Path(destination)/'contract','mm_baseline_experiment',record)


def open_run(root, expected=None, *, current_source=True):
    a,r = read_record(Path(root)/'contract','mm_baseline_experiment',expected)
    if current_source: require(r['source'] == source_v3(), 'experiment Source v3 drift; rebuild, never re-sign')
    from werewolf.tom_llm.deployment import validate_source_evidence
    validate_source_evidence(r['source'])
    c,tasks = read_cohort(r['cohort_root'],r['cohort_digest'],current_source=False)
    version = r['protocol'].get('schema_version')
    require(r['protocol'] == protocol(c.manifest['task']['schema_version'],
            protocol_version=version if version==APPROVED_PROTOCOL_VERSION else None), 'fixed metrics/protocol drift')
    if c.manifest['task']['schema_version']==SET_VERSION and r['mode']!='fake':
        require(r.get('live_execution')==live_set_binding(r,c,tasks), 'V2 live execution binding drift')
    require(r['selection'] == c.manifest['selection'] and r['reference'] == c.manifest['reference'], 'common cohort anchors drift')
    selection = strict_json((Path(c.manifest['selection_root'])/'selection.json').read_bytes())
    require(r['games'] == choose_games(selection,r['game_count'],r['mode']), 'game selection drift')
    tasks = [t for t in tasks if t['game_id'] in set(r['games'])]
    require(r['task_digests'] == [digest(t) for t in tasks] and r['request_count']==len(tasks)
            and r['row_identity_digest']==digest([identity(t) for t in tasks]), 'ordered task coverage drift')
    for key,want in r['models'].items(): require(ModelSpec(key).digest==want, 'ModelSpec revision drift')
    return a,r,tasks
