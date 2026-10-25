"""Public-only task contract, independent of model rendering and execution."""

from werewolf.structured_history import StructuredTokenPlan, StructuredTokenDescriptor
from werewolf.tom_llm.common import digest, require
from werewolf.tom_llm.policy import policy_binding
from werewolf.tom_llm.prompt import render_prompt
from werewolf.tom_llm.study import public_requests, validate_request

VERSION = 'classic7_llm_shared_public_task_v1'


def task_binding():
    original = policy_binding()
    keys = ('prompt_version', 'normalization', 'max_structured_tokens', 'dispatch_population',
            'evaluation_population', 'reference_temporal_condition', 'canonical_shift', 'subset', 'bootstrap')
    return {'schema_version': VERSION, 'policy': {k: original['config'][k] for k in keys},
            'system_sha256': original['asset_digests']['system.txt']}


def from_legacy(request):
    validate_request(request)
    return {k: v for k, v in request.items() if k not in ('rendered_prompt', 'prompt_digest')}


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
    to_legacy(task)
    return task


def public_tasks(prefixes, fold_of_game):
    return [from_legacy(r) for r in public_requests(prefixes, fold_of_game)]
