"""Public-only task contract, independent of model rendering and execution."""

from werewolf.structured_history import StructuredTokenPlan, StructuredTokenDescriptor
from werewolf.tom_llm.common import digest, require
from werewolf.tom_llm.policy import policy_binding
from werewolf.tom_llm.prompt import render_prompt, render_set_prompt, SET_PROMPT_VERSION, set_system_text
from werewolf.tom_llm.study import public_requests, validate_request

VERSION = 'classic7_llm_shared_public_task_v1'
SET_VERSION = 'classic7_llm_suspicion_set_task_v2'


def task_version(task):
    require(type(task) is dict, 'public task must be a mapping')
    version = task.get('schema_version', VERSION)
    require(version in (VERSION, SET_VERSION), 'unsupported public task protocol')
    return version


def task_binding(version=VERSION):
    original = policy_binding()
    keys = ('prompt_version', 'normalization', 'max_structured_tokens', 'dispatch_population',
            'evaluation_population', 'reference_temporal_condition', 'canonical_shift', 'subset', 'bootstrap')
    binding = {'schema_version': VERSION, 'policy': {k: original['config'][k] for k in keys},
               'system_sha256': original['asset_digests']['system.txt']}
    require(version in (VERSION, SET_VERSION), 'unsupported public task protocol')
    if version == SET_VERSION:
        from werewolf.artifact_io import sha256_bytes
        from werewolf.tom_llm.output import SET_OUTPUT_VERSION
        binding['schema_version'] = version
        binding['policy']['prompt_version'] = SET_PROMPT_VERSION
        binding['output_version'] = SET_OUTPUT_VERSION
        binding['system_sha256'] = sha256_bytes(set_system_text().encode('utf-8'))
    return binding


def from_legacy(request, *, version=VERSION):
    validate_request(request)
    task_binding(version)
    task = {k: v for k, v in request.items() if k not in ('rendered_prompt', 'prompt_digest')}
    if version == SET_VERSION:
        record = dict(task['public_plan'])
        record['tokens'] = tuple(StructuredTokenDescriptor(**t) for t in record['tokens'])
        prompt = render_set_prompt(StructuredTokenPlan(**record), task['observer'])
        task.update(schema_version=version, messages=list(prompt.messages), response_schema=prompt.schema)
    return task


def to_legacy(task):
    fields = {'game_id','boundary_id','prefix_digest','plan_digest','observer','fold','public_plan','messages','response_schema'}
    require(type(task) is dict and set(task) == fields, 'public task forbids labels/private/extra fields')
    record = dict(task['public_plan'])
    record['tokens'] = tuple(StructuredTokenDescriptor(**t) for t in record['tokens'])
    prompt = render_prompt(StructuredTokenPlan(**record), task['observer'])
    request = {**task, 'rendered_prompt': prompt.rendered, 'prompt_digest': prompt.prompt_digest}
    validate_request(request)
    return request


def validate_task(task):
    if task_version(task) == VERSION:
        to_legacy(task)
        return task
    fields = {'schema_version', 'game_id', 'boundary_id', 'prefix_digest', 'plan_digest',
              'observer', 'fold', 'public_plan', 'messages', 'response_schema'}
    require(set(task) == fields, 'public task forbids labels/private/extra fields')
    record = dict(task['public_plan'])
    record['tokens'] = tuple(StructuredTokenDescriptor(**t) for t in record['tokens'])
    plan = StructuredTokenPlan(**record)
    original = render_prompt(plan, task['observer'])
    # 复用原任务的身份及完整 PRE 校验，集合消息再按其自身协议精确核对。
    legacy = {k: v for k, v in task.items() if k != 'schema_version'}
    legacy.update(messages=list(original.messages), response_schema=original.schema)
    to_legacy(legacy)
    prompt = render_set_prompt(plan, task['observer'])
    require(task['messages'] == list(prompt.messages) and task['response_schema'] == prompt.schema,
            'set task differs from public plan/versioned renderer')
    return task


def public_tasks(prefixes, fold_of_game, *, version=VERSION):
    return [from_legacy(r, version=version) for r in public_requests(prefixes, fold_of_game)]
