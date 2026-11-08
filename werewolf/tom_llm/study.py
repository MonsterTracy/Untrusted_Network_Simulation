"""公开 PRE 请求构造与验证；执行流程统一由 live_runner 负责。"""
from werewolf.structured_history import plan_structured_history, StructuredTokenPlan, StructuredTokenDescriptor
from werewolf.tom_llm.common import digest, identity, require
from werewolf.tom_llm.prompt import render_prompt


def make_request(prefix, observer, *, fold):
    require(type(fold) is int and 0 <= fold < 5, "invalid request fold")
    plan = plan_structured_history(prefix)
    require(observer in [int(p.removeprefix("player")) - 1 for p in prefix.alive_observer_ids],
            "dispatch must use a public alive observer")
    prompt = render_prompt(plan, observer)
    return {"game_id": prefix.game_id, "boundary_id": prefix.boundary_id, "prefix_digest": plan.prefix_digest,
            "plan_digest": plan.plan_digest, "observer": observer, "fold": fold, "public_plan": plan.to_record(),
            "messages": list(prompt.messages), "rendered_prompt": prompt.rendered,
            "prompt_digest": prompt.prompt_digest, "response_schema": prompt.schema}


def public_requests(prefixes, fold_of_game):
    requests = [make_request(prefix, int(seat.removeprefix("player")) - 1, fold=fold_of_game[prefix.game_id])
                for prefix in prefixes for seat in prefix.alive_observer_ids]
    require(requests and len({identity(r) for r in requests}) == len(requests), "duplicate or empty public requests")
    return requests


def validate_request(request):
    fields = {"game_id", "boundary_id", "prefix_digest", "plan_digest", "observer", "fold",
              "public_plan", "messages", "rendered_prompt", "prompt_digest", "response_schema"}
    require(type(request) is dict and set(request) == fields, "request schema forbids labels/private fields")
    require(type(request["game_id"]) is str and request["game_id"] and type(request["boundary_id"]) is str
            and request["boundary_id"] and type(request["fold"]) is int and 0 <= request["fold"] < 5, "invalid request identity")
    value = dict(request["public_plan"])
    value["tokens"] = tuple(StructuredTokenDescriptor(**t) for t in value["tokens"])
    plan = StructuredTokenPlan(**value)
    prompt = render_prompt(plan, request["observer"])
    require(request["prefix_digest"] == plan.prefix_digest and request["plan_digest"] == plan.plan_digest
            and request["messages"] == list(prompt.messages) and request["rendered_prompt"] == prompt.rendered
            and request["response_schema"] == prompt.schema and request["prompt_digest"] == prompt.prompt_digest,
            "request differs from public plan/frozen renderer")
    return request
