"""Frozen synthetic public PREs, native expected token IDs, and full-cohort CPU budgets."""

from pathlib import Path
import numpy as np

from werewolf.artifact_io import publish_artifact, verify_artifact, canonical_json_bytes, sha256_bytes
from werewolf.tom_llm.common import digest, identity, require, require_digest, strict_json
from werewolf.tom_llm.model_spec import ModelSpec, load_cpu_tokenizer, render, wire
from werewolf.tom_llm.task import from_legacy, validate_task

VERSION = 'classic7_multimodel_qualification_inputs_v1'
LEGACY_SHA = 'f1052efa861e458606d59b33793649fc81772d13958aad3c38ddffb7acf0f98f'
LEGACY_PATH = Path(__file__).parent/'qualification_assets/qwen3-suite-v1.json'
SCHEDULE = [['warmup',i,0] for i in (0,1)] + [['measurement',i,r] for r in range(3) for i in range(12)]


def legacy_suite():
    raw = LEGACY_PATH.read_bytes()
    require(sha256_bytes(raw)==LEGACY_SHA, 'frozen 76-call Qwen synthetic suite drift')
    suite = strict_json(raw)
    require(suite['schedule']==SCHEDULE and len(suite['cases'])==12, 'legacy synthetic schedule drift')
    return suite


def synthetic_tasks():
    return [from_legacy(c['request']) for c in legacy_suite()['cases']]


def build_suite(spec, tokenizer_assets):
    tokenizer = load_cpu_tokenizer(spec,tokenizer_assets)
    tasks = synthetic_tasks(); cases = []
    for i,task in enumerate(tasks):
        text = render(task,spec)['text']
        ids = tokenizer.encode(text,add_special_tokens=False)
        require(ids==tokenizer.apply_chat_template(task['messages'],tokenize=True,return_dict=False,
                add_generation_prompt=True,**spec.record()['chat_template']['kwargs']), 'native suite input IDs drift')
        cases.append({'case_index':i,'task':task,'input_ids':ids,'wire_digest':digest(wire(task,spec)),
                      'rendered_sha256':sha256_bytes(text.encode())})
    suite = {'schema_version':VERSION,'purpose':'synthetic_gpu_qualification_only','model_spec_digest':spec.digest,
             'key':spec.key,'legacy_public_suite_sha256':LEGACY_SHA,'cases':cases,'schedule':SCHEDULE,
             'vocabulary_size':len(tokenizer),'cpu_native_tokenizer_verified':True}
    validate_suite(suite,spec)
    return suite


def validate_suite(suite,spec):
    require(set(suite)=={'schema_version','purpose','model_spec_digest','key','legacy_public_suite_sha256',
            'cases','schedule','vocabulary_size','cpu_native_tokenizer_verified'} and
            suite['schema_version']==VERSION and suite['purpose']=='synthetic_gpu_qualification_only'
            and suite['model_spec_digest']==spec.digest and suite['key']==spec.key
            and suite['legacy_public_suite_sha256']==LEGACY_SHA and suite['schedule']==SCHEDULE
            and suite['cpu_native_tokenizer_verified'] is True and type(suite['vocabulary_size']) is int
            and suite['vocabulary_size']>0 and len(suite['cases'])==12, 'model native suite identity drift')
    for i,(case,task) in enumerate(zip(suite['cases'],synthetic_tasks(),strict=True)):
        require(set(case)=={'case_index','task','input_ids','wire_digest','rendered_sha256'}
                and case['case_index']==i and case['task']==task
                and case['wire_digest']==digest(wire(task,spec))
                and case['rendered_sha256']==render(task,spec)['text_sha256'], 'synthetic case/task/wire drift')
        ids = case['input_ids']
        require(type(ids) is list and 0<len(ids)<=7680 and all(type(n) is int and 0<=n<suite['vocabulary_size'] for n in ids),
                'synthetic IDs/context invalid')
        if spec.key=='qwen3':
            old = legacy_suite()['cases'][i]
            require(ids==old['input_ids'] and case['wire_digest']==old['wire_digest'], 'original Qwen 76-call protocol changed')
    require(len({digest(c['input_ids']) for c in suite['cases']})==12, 'synthetic IDs must distinguish cases')
    return suite


def publish_suite(destination,spec,tokenizer_assets):
    suite = build_suite(spec,tokenizer_assets)
    return publish_artifact(destination,manifest_fields={'artifact_type':'mm_native_suite','schema_version':VERSION,
        'model_spec_digest':spec.digest,'suite_digest':digest(suite),'qualification':'INPUTS_ONLY_NOT_GPU_ADMISSION'},
        files={'suite.json':canonical_json_bytes(suite)})


def open_suite(root,spec,expected_digest):
    require_digest(expected_digest)
    a = verify_artifact(root,expected_artifact_type='mm_native_suite',expected_schema_version=VERSION)
    require(a.manifest_digest==expected_digest and set(a.manifest['file_table'])=={'suite.json'}
            and a.manifest['model_spec_digest']==spec.digest, 'native suite artifact drift')
    suite = validate_suite(strict_json((a.path/'suite.json').read_bytes()),spec)
    require(digest(suite)==a.manifest['suite_digest'], 'native suite payload drift')
    return suite


def budget_record(tasks,spec,tokenizer):
    """Every row, full PRE; never an intersection, Primary filter or truncation."""
    require(tasks and len({identity(t) for t in tasks})==len(tasks), 'empty/duplicate budget requests')
    counts = []
    for task in tasks:
        validate_task(task)
        text = render(task,spec)['text']; ids = tokenizer.encode(text,add_special_tokens=False)
        require(ids==tokenizer.apply_chat_template(task['messages'],tokenize=True,return_dict=False,
                add_generation_prompt=True,**spec.record()['chat_template']['kwargs']), 'budget native input IDs drift')
        counts.append(len(ids))
    require(all(0<n<=7680 for n in counts), 'complete PRE exceeds 8192 including output; no row filtering/truncation')
    return {'request_count':len(counts),'row_identity_digest':digest([identity(t) for t in tasks]),
            'input_token_counts':counts,'input_tokens_total':sum(counts),'maximum_input_tokens':max(counts),
            'p50':float(np.quantile(counts,.5)),'p95':float(np.quantile(counts,.95)),
            'p99':float(np.quantile(counts,.99)),'output_cap':512,'context_length':8192}


def publish_budget(destination,cohort_root,expected_cohort,spec,tokenizer_assets):
    from werewolf.tom_llm.multimodel import read_cohort
    cohort,tasks = read_cohort(cohort_root,expected_cohort)
    selection = cohort.manifest['selection']
    # All selection/provenance validation happens in the existing common cohort reader.
    require(len(selection['game_ids'])==150, 'qualification budget requires the complete fixed 150-game cohort')
    folds = {}
    for task in tasks:folds.setdefault(task['fold'],set()).add(task['game_id'])
    require(set(folds)==set(range(5)) and all(len(g)==30 for g in folds.values()), 'fixed 30 games/fold required')
    report = budget_record(tasks,spec,load_cpu_tokenizer(spec,tokenizer_assets))
    record = {'schema_version':VERSION+'_budget','model_spec_digest':spec.digest,
              'cohort_digest':cohort.manifest_digest,'cohort_root':str(cohort.path),
              'selection':selection,'reference':cohort.manifest['reference'],**report}
    return publish_artifact(destination,manifest_fields={'artifact_type':'mm_cpu_budget','schema_version':VERSION,
        'record_digest':digest(record),'model_spec_digest':spec.digest,'qualification':'CPU_BUDGET_ONLY'},
        files={'budget.json':canonical_json_bytes(record)})


def open_budget(root,spec,expected_digest):
    from werewolf.tom_llm.multimodel import read_cohort
    require_digest(expected_digest)
    a = verify_artifact(root,expected_artifact_type='mm_cpu_budget',expected_schema_version=VERSION)
    require(a.manifest_digest==expected_digest and set(a.manifest['file_table'])=={'budget.json'}, 'CPU budget identity drift')
    record = strict_json((a.path/'budget.json').read_bytes())
    require(record['model_spec_digest']==spec.digest and a.manifest['model_spec_digest']==spec.digest
            and digest(record)==a.manifest['record_digest'], 'CPU budget ModelSpec/payload drift')
    cohort,tasks = read_cohort(record['cohort_root'],record['cohort_digest'])
    require(record['selection']==cohort.manifest['selection'] and record['reference']==cohort.manifest['reference']
            and len(record['selection']['game_ids'])==150 and record['request_count']==len(tasks)
            and record['row_identity_digest']==digest([identity(t) for t in tasks])
            and len(record['input_token_counts'])==len(tasks) and all(type(n) is int and 0<n<=7680 for n in record['input_token_counts']),
            'CPU budget common-cohort/coverage drift')
    require(record['input_tokens_total']==sum(record['input_token_counts'])
            and record['maximum_input_tokens']==max(record['input_token_counts'])
            and record['context_length']==8192 and record['output_cap']==512,'CPU budget summary/context drift')
    return record


def budget_matrix(records):
    require(set(records)=={'qwen3','yi','mistral'}, 'complete three-model CPU budget matrix required')
    keys=('cohort_digest','selection','reference','row_identity_digest','request_count')
    common={k:records['qwen3'][k] for k in keys}
    for key,r in records.items():
        require(r['model_spec_digest']==ModelSpec(key).digest and {k:r[k] for k in keys}==common,
                'cross-model CPU cohort/Selection/Reference mismatch')
    return {'common':common,'models':{k:{n:r[n] for n in ('input_tokens_total','p50','p95','p99','maximum_input_tokens')}
            for k,r in records.items()},'gpu_qualification':False}


def export_cpu_cohort(destination,*,publication_root,evaluation_root,expected_reference,selection_root,
                      expected_selection,selection_provenance,prepared_root):
    """Reuse the fixed Selection; build public tasks before opening prediction rows."""
    from werewolf.development_publication import open_publication
    from werewolf.tom_llm.selection_binding import open_selection_binding
    from werewolf.tom_llm.reference import open_reference
    from werewolf.tom_llm.task import public_tasks
    from werewolf.tom_llm.multimodel_evaluation import publish_mock_cohort
    prepared=verify_artifact(prepared_root,expected_artifact_type='backbone_tom_study',expected_schema_version='classic7_backbone_study_v1')
    folds=prepared.manifest['folds']
    open_selection_binding(selection_root,folds,expected=expected_selection,provenance=selection_provenance,
                           prepared_manifest=prepared.manifest)
    selected=strict_json((Path(selection_root)/'selection.json').read_bytes())['game_ids']
    require(len(selected)==150,'only the existing fixed 150-game Selection is accepted')
    publication=open_publication(publication_root)
    membership={g:r['fold'] for r in folds for g in r['held_out_game_ids']}
    tasks=public_tasks([p for g in publication.public_view.games if g.game_id in set(selected)
                       for p in g.authoritative_pre_prefixes],membership)
    reference=open_reference(publication_root,evaluation_root,expected_seal_digest=expected_reference)
    return publish_mock_cohort(destination,tasks,reference,selection_root=selection_root,
        expected=expected_selection,provenance=selection_provenance,prepared_manifest=prepared.manifest)
