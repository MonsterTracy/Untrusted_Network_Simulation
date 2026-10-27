"""Versioned shared cohort and strictly model/service-isolated Mock executions."""

from dataclasses import dataclass
import fcntl
import os
import subprocess
from pathlib import Path

from werewolf.artifact_io import (publish_artifact, verify_artifact, canonical_json_bytes,
                                  canonical_jsonl_bytes, sha256_bytes)
from werewolf.tom_llm.common import digest, identity, require, require_digest, strict_json
from werewolf.tom_llm.development_source import development_source_identity
from werewolf.tom_llm.inference import MockBackend, InfrastructureFailure
from werewolf.tom_llm.model_spec import ModelSpec, registry, render, wire, token_preflight
from werewolf.tom_llm.output import parse_output
from werewolf.tom_llm.provider_semantics import NotExecuted, OutcomeUnknown
from werewolf.tom_llm.task import task_binding, validate_task

VERSION = 'classic7_multimodel_mock_v1'
COHORT_VERSION = 'classic7_multimodel_public_cohort_v1'


def source_identity():
    source = development_source_identity()
    root = Path(__file__).resolve().parents[2]
    files = dict(source['file_digests'])
    for name in ['configs/paper/tom-llm-models-v1.json', 'scripts/check_tom_multimodel_cpu.py',
                 'scripts/tom_multimodel_admission.py',
                 'configs/paper/tom-llm-physical-assets-v1.json',
                 'configs/paper/tom-llm-qualification-recipe-v1.json',
                 'werewolf/tom_llm/qualification_assets/qwen3-suite-v1.json',
                 *['werewolf/tom_llm/' + r['chat_template']['path'] for r in registry().values()]]:
        p = root / name
        require(not any(x.is_symlink() for x in (p, *p.parents)), 'multimodel source symlink')
        files[name] = sha256_bytes(p.read_bytes())
    # Recompute Git metadata over the complete extended runtime inventory.
    # The legacy Qwen development_source contract remains untouched.
    env = {k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_OPTIONAL_LOCKS='0', GIT_NO_LAZY_FETCH='1')
    status = subprocess.check_output(['git','-C',str(root),'status','--porcelain=v1',
        '--untracked-files=all','--',*sorted(files)],env=env,stderr=subprocess.PIPE).decode().strip()
    return {**source, 'schema_version': 'classic7_multimodel_source_v2', 'file_digests': files,
            'source_inventory_clean': not bool(status), 'source_status_digest': digest(status),
            'implementation_digest': digest(files), 'qualification': 'CPU_MOCK_ONLY_NOT_ADMITTED'}


def read_cohort(root, expected_digest, *, current_source=False):
    require_digest(expected_digest)
    artifact = verify_artifact(root, expected_artifact_type='llm_multimodel_public_cohort',
                               expected_schema_version=COHORT_VERSION)
    m = artifact.manifest
    require(artifact.manifest_digest == expected_digest and m['engineering_only'] is True
            and (not current_source or m['source'] == source_identity()) and m['task'] == task_binding(), 'cohort identity/source drift')
    from werewolf.tom_llm.deployment import validate_source_evidence
    validate_source_evidence(m['source'])
    require(set(m['file_table']) == {'requests.jsonl'}, 'cohort file inventory drift')
    selection = verify_artifact(m['selection_root'], expected_artifact_type='llm_game_selection',
                                expected_schema_version='classic7_llm_game_selection_v1')
    require(selection.manifest_digest == m['selection']['selection_artifact_manifest_digest'], 'Selection artifact drift')
    selected = strict_json((selection.path/'selection.json').read_bytes())
    require(selected['game_ids'] == m['selection']['game_ids']
            and selected['selection_digest'] == m['selection']['selection_digest'], 'Selection payload drift')
    tasks = [strict_json(line) for line in (artifact.path/'requests.jsonl').read_bytes().splitlines()]
    for task in tasks:
        validate_task(task)
    require(tasks and len({identity(t) for t in tasks}) == len(tasks)
            and digest(tasks) == m['requests_digest']
            and digest([identity(t) for t in tasks]) == m['row_identity_digest'], 'cohort request coverage drift')
    require(set(t['game_id'] for t in tasks) == set(m['selection']['game_ids']), 'cohort game coverage drift')
    return artifact, tasks


@dataclass(frozen=True)
class ModelExecution:
    spec: ModelSpec
    service_id: str

    def binding(self):
        require(type(self.spec) is ModelSpec and type(self.service_id) is str
                and self.service_id.startswith('synthetic:') and len(self.service_id) > 10,
                'only explicit synthetic model/service execution is admitted')
        return {'schema_version': VERSION + '_execution', 'model_spec': self.spec.record(),
                'model_spec_digest': self.spec.digest, 'service_id': self.service_id,
                'task': task_binding(), 'source': source_identity(),
                'engineering_only': True, 'physical_admission': None}


def prepare_model_mock(destination, cohort_root, *, expected_cohort_digest, execution,
                       tokenizer_identity, count_tokens):
    require(type(execution) is ModelExecution, 'explicit model execution required')
    require(type(tokenizer_identity) is str and tokenizer_identity.startswith('synthetic:'), 'Mock token counter required')
    cohort, tasks = read_cohort(cohort_root, expected_cohort_digest)
    bound = execution.binding()
    preflight = token_preflight(tasks, execution.spec, count_tokens=count_tokens)
    return publish_artifact(Path(destination)/'contract', manifest_fields={
        'artifact_type': 'llm_multimodel_mock_contract', 'schema_version': VERSION,
        'engineering_only': True, 'execution': bound, 'execution_digest': digest(bound),
        'cohort_path': str(cohort.path), 'cohort_digest': cohort.manifest_digest,
        'tokenizer_identity': tokenizer_identity, 'preflight': preflight}, files={})


def open_model_mock(root, execution):
    require(type(execution) is ModelExecution, 'explicit model execution required')
    contract = verify_artifact(Path(root)/'contract', expected_artifact_type='llm_multimodel_mock_contract',
                               expected_schema_version=VERSION)
    m = contract.manifest
    bound = execution.binding()
    require(m['engineering_only'] is True and m['execution'] == bound and m['execution_digest'] == digest(bound),
            'model/revision/service/source execution drift')
    cohort, tasks = read_cohort(m['cohort_path'], m['cohort_digest'])
    return contract, cohort, tasks


def owned(root, contract, task):
    artifact = verify_artifact(root, expected_artifact_type='llm_multimodel_mock_event', expected_schema_version=VERSION)
    m = artifact.manifest
    require(m['contract_digest'] == contract.manifest_digest
            and m['execution_digest'] == contract.manifest['execution_digest']
            and m['task_digest'] == digest(task), 'cross-model/event ownership mismatch')
    return artifact


def event(path, contract, task, fields, *, files=None):
    return publish_artifact(path, manifest_fields={'artifact_type':'llm_multimodel_mock_event',
        'schema_version':VERSION, 'contract_digest':contract.manifest_digest,
        'execution_digest':contract.manifest['execution_digest'], 'task_digest':digest(task), **fields},
        files={} if files is None else files)


def result_row(contract, task, spec, attempts, reply):
    return {k:task[k] for k in ('game_id','boundary_id','prefix_digest','plan_digest','observer','fold')} | {
        'canonical_shift':0, 'temporal_condition':'explicit_day_phase',
        'model_spec_digest':spec.digest, 'execution_digest':contract.manifest['execution_digest'],
        'cohort_digest':contract.manifest['cohort_digest'], 'task_digest':digest(task),
        'wire_digest':digest(wire(task,spec)), 'rendered_prompt_digest':render(task,spec)['text_sha256'],
        'attempt_digests':[a.manifest_digest for a in attempts], 'usage':reply['usage'],
        'prediction':parse_output(reply['content'],task['observer'],finish_reason=reply['finish_reason']).to_record()}


def verify_result(root, contract, task, spec):
    terminal = owned(root/'result', contract, task)
    require(set(terminal.manifest['file_table']) == {'result.json'}, 'terminal inventory drift')
    row = strict_json((terminal.path/'result.json').read_bytes())
    require(1 <= len(row['attempt_digests']) <= 2, 'attempt budget mismatch')
    attempts = []
    for number, wanted in enumerate(row['attempt_digests'],1):
        start = owned(root/f'dispatch-{number}',contract,task)
        attempt = owned(root/f'attempt-{number}',contract,task)
        require(start.manifest['number'] == attempt.manifest['number'] == number
                and start.manifest['outcome'] == 'dispatched' and attempt.manifest_digest == wanted,
                'dispatch/attempt chain drift')
        last = number == len(row['attempt_digests'])
        require(attempt.manifest['outcome'] == ('response' if last else 'not_executed'), 'attempt outcome drift')
        require(set(attempt.manifest['file_table']) == ({'reply.json'} if last else set()), 'attempt file drift')
        attempts.append(attempt)
    reply = strict_json((attempts[-1].path/'reply.json').read_bytes())
    require(row == result_row(contract,task,spec,attempts,reply), 'result differs from durable response/model')
    expected_paths = {'result', *[f'{kind}-{i}' for i in range(1,len(attempts)+1) for kind in ('dispatch','attempt')]}
    require({p.name for p in root.iterdir()} == expected_paths, 'extra request events; no silent recovery')
    return terminal, row



def inspect_pending(lineage, contract, task, spec):
    """Validate the entire durable prefix before permitting any new dispatch."""
    names = {p.name for p in lineage.iterdir()}
    expected = set()
    responded = False
    for number in (1, 2):
        dispatch, finish = lineage/f'dispatch-{number}', lineage/f'attempt-{number}'
        if not dispatch.exists():
            break
        start = owned(dispatch, contract, task)
        require(start.manifest['number'] == number and start.manifest['outcome'] == 'dispatched'
                and not start.manifest['file_table'], 'dispatch chain drift')
        require(finish.exists(), 'uncertain in-flight outcome; automatic resend forbidden')
        attempt = owned(finish, contract, task)
        outcome = attempt.manifest['outcome']
        require(attempt.manifest['number'] == number and outcome in ('response', 'not_executed'),
                'attempt chain drift')
        require(set(attempt.manifest['file_table']) == ({'reply.json'} if outcome == 'response' else set()),
                'attempt file drift')
        expected.update((dispatch.name, finish.name))
        if outcome == 'response':
            reply = strict_json((finish/'reply.json').read_bytes())
            result_row(contract, task, spec, [attempt], reply)
            responded = True
            break
    require(names == expected, 'extra or noncontiguous request events; no silent recovery')
    if len(expected) == 4 and not responded:
        raise InfrastructureFailure('two not-executed attempts exhausted; no fabricated distribution')


def run_model_mock(root, execution, backend, *, after_response=None):
    """Exactly MockBackend. An uncertain dispatch permanently blocks automatic resend."""
    require(type(backend) is MockBackend, 'only MockBackend; no live execution factory')
    root = Path(root)
    contract, _, tasks = open_model_mock(root,execution)
    with (root/'mock.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (root/'seal').exists(), 'sealed model run is closed')
        request_root = root/'requests'
        if request_root.exists():
            require({p.name for p in request_root.iterdir()} <= {digest(identity(t)) for t in tasks},
                    'extra request outside cohort')
        # A later contaminated/unknown lineage must block the whole run before any calls.
        for task in tasks:
            lineage = request_root/digest(identity(task))
            if (lineage/'result').exists():
                verify_result(lineage,contract,task,execution.spec)
            elif lineage.exists():
                inspect_pending(lineage,contract,task,execution.spec)
        for task in tasks:
            open_model_mock(root,execution)
            lineage = root/'requests'/digest(identity(task))
            if (lineage/'result').exists():
                verify_result(lineage,contract,task,execution.spec)
                continue
            attempts = []
            for number in (1,2):
                finish = lineage/f'attempt-{number}'
                if finish.exists():
                    attempt = owned(finish,contract,task)
                else:
                    require(not (lineage/f'dispatch-{number}').exists(),
                            'uncertain in-flight outcome; automatic resend forbidden')
                    event(lineage/f'dispatch-{number}',contract,task,{'number':number,'outcome':'dispatched'})
                    try:
                        reply = backend.chat(wire(task,execution.spec))
                    except NotExecuted:
                        attempt = event(finish,contract,task,{'number':number,'outcome':'not_executed'})
                    except Exception as error:
                        raise OutcomeUnknown('dispatch unresolved; no automatic retry/fallback') from error
                    else:
                        open_model_mock(root,execution)
                        attempt = event(finish,contract,task,{'number':number,'outcome':'response'},files={
                            'reply.json':canonical_json_bytes({'content':reply.content,'finish_reason':reply.finish_reason,'usage':reply.usage})})
                        if after_response is not None:
                            after_response()
                attempts.append(attempt)
                if attempt.manifest['outcome'] == 'response':
                    break
                require(attempt.manifest['outcome'] == 'not_executed', 'invalid Mock attempt')
            if attempts[-1].manifest['outcome'] != 'response':
                raise InfrastructureFailure('two not-executed attempts exhausted; no fabricated distribution')
            reply = strict_json((attempts[-1].path/'reply.json').read_bytes())
            row = result_row(contract,task,execution.spec,attempts,reply)
            event(lineage/'result',contract,task,{'outcome':'terminal'},files={'result.json':canonical_json_bytes(row)})
            verify_result(lineage,contract,task,execution.spec)


def seal_model_mock(root, execution):
    root = Path(root)
    contract, cohort, tasks = open_model_mock(root,execution)
    with (root/'mock.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        require((root/'requests').is_dir() and {p.name for p in (root/'requests').iterdir()}
                == {digest(identity(t)) for t in tasks}, 'model request coverage mismatch')
        rows, terminals = [], []
        for task in tasks:
            terminal,row = verify_result(root/'requests'/digest(identity(task)),contract,task,execution.spec)
            rows.append(row); terminals.append(terminal.manifest_digest)
        return publish_artifact(root/'seal',manifest_fields={'artifact_type':'llm_multimodel_mock_seal',
            'schema_version':VERSION,'engineering_only':True,'contract_digest':contract.manifest_digest,
            'execution_digest':contract.manifest['execution_digest'],'cohort_digest':cohort.manifest_digest,
            'model_spec_digest':execution.spec.digest,'terminal_digests':terminals},
            files={'predictions.jsonl':canonical_jsonl_bytes(rows)})


def open_model_predictions(root, execution):
    root = Path(root)
    contract, cohort, tasks = open_model_mock(root,execution)
    seal = verify_artifact(root/'seal', expected_artifact_type='llm_multimodel_mock_seal',expected_schema_version=VERSION)
    m = seal.manifest
    require(m['engineering_only'] is True and m['contract_digest'] == contract.manifest_digest
            and m['execution_digest'] == contract.manifest['execution_digest']
            and m['cohort_digest'] == cohort.manifest_digest and m['model_spec_digest'] == execution.spec.digest,
            'model seal execution/cohort mismatch')
    rows = [strict_json(line) for line in (seal.path/'predictions.jsonl').read_bytes().splitlines()]
    verified = [verify_result(root/'requests'/digest(identity(t)),contract,t,execution.spec) for t in tasks]
    require(rows == [row for _,row in verified] and m['terminal_digests'] == [a.manifest_digest for a,_ in verified],
            'sealed predictions differ from model terminals')
    require({p.name for p in (root/'requests').iterdir()} == {digest(identity(t)) for t in tasks}, 'extra sealed request')
    return seal, rows
