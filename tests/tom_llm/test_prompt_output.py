from copy import deepcopy
from dataclasses import replace
import inspect
import json
import math

import pytest
import torch

from tests.canonical_collection.test_structured_token_plan import _prefix
from tests.development_publication.test_development_publication import _publication
from werewolf.artifact_io import sha256_bytes, canonical_json_bytes
from werewolf.structured_history import plan_structured_history
from werewolf.tom.dataset import tensorize_public_pre, ExperimentCapacity, belief_target
from werewolf.tom_llm.prompt import render_prompt, public_table, schema
from werewolf.tom_llm.output import parse_output, validate_prediction, EPSILON
from werewolf.tom_llm.policy import load_policy, validate_policy, ASSET_DIGESTS
from werewolf.tom_llm.study import make_request, wire_request, public_requests, validate_request


def response(observer=0, values=None):
    return json.dumps({"probabilities": {f"player{j+1}": v for j, v in zip(
        (j for j in range(7) if j != observer), values or [1/6] * 6, strict=True)}})


def test_complete_public_table_matches_tensor_semantics_and_template():
    prefix = _prefix()
    plan, tensors = tensorize_public_pre(prefix, ExperimentCapacity(1024))
    prompt = render_prompt(plan, 1)
    assert len(prompt.table.splitlines()) == plan.token_count + 1
    for token, line in zip(plan.tokens, prompt.table.splitlines()[1:], strict=True):
        assert line == "|".join("-" if v is None else str(v) for v in
            (token.token_type, token.source, token.action, token.target, token.day, token.phase))
    assert prompt.table.endswith("turn_start|player2|-|-|2|discussion")
    assert "player3 is suspicious" not in prompt.rendered
    for forbidden in (prefix.game_id, prefix.boundary_id, prefix.prefix_digest, plan.plan_digest, "belief-player"):
        assert forbidden not in prompt.rendered
    assert prompt.rendered == ("<|im_start|>system\n" + prompt.messages[0]["content"] + "<|im_end|>\n"
        + "<|im_start|>user\n" + prompt.messages[1]["content"] + "<|im_end|>\n"
        + "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert int(tensors.attention_mask.sum()) == plan.token_count
    assert set(inspect.signature(render_prompt).parameters) == {"plan", "observer"}


def test_observer_only_changes_condition_and_six_seat_schema():
    plan = plan_structured_history(_prefix())
    first, second = render_prompt(plan, 0), render_prompt(plan, 1)
    assert first.table == second.table and first.messages[0] == second.messages[0]
    assert first.prompt_digest != second.prompt_digest
    assert "player1" not in first.schema["properties"]["probabilities"]["properties"]
    assert set(first.schema["properties"]["probabilities"]["properties"]) == {f"player{i}" for i in range(2, 8)}


def test_no_supervision_role_or_future_inputs_to_public_request(monkeypatch, tmp_path):
    _, handles = _publication(tmp_path)
    prefix = handles.public.public_view.games[0].authoritative_pre_prefixes[0]
    # Sidecars and Dataset must never be touched by request construction.
    import werewolf.development_publication as pub
    monkeypatch.setattr(pub, "open_role_sidecar", lambda *_a: pytest.fail("private role access"))
    request = make_request(prefix, 0, fold=0)
    assert not {"q", "label_observed", "primary_eligible", "role", "raw_text"} & set(request)
    wire = wire_request(request)
    assert "fold" not in wire and "prefix_digest" not in wire and "public_plan" not in wire
    assert wire["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert wire["temperature"] == 0 and wire["n"] == 1 and wire["max_tokens"] == 512
    changed = deepcopy(request)
    changed["q"] = [0, 1, 0, 0, 0, 0, 0]
    with pytest.raises(ValueError, match="forbids"):
        validate_request(changed)


def test_raw_text_and_audit_identifiers_do_not_change_prompt():
    # Changing raw text requires revalidation/rebinding of PRE; only its public semantic plan is rendered.
    plan = plan_structured_history(_prefix())
    tokens = tuple(replace(t, event_id="unseen-raw-secret") for t in plan.tokens)
    other = replace(plan, tokens=tokens, prefix_digest="f" * 64)
    value = other.to_record(); value.pop("plan_digest")
    other = replace(other, plan_digest=sha256_bytes(canonical_json_bytes(value)))
    assert render_prompt(plan, 0).rendered == render_prompt(other, 0).rendered


@pytest.mark.parametrize("change", ["future", "missing", "digest", "injection", "capacity", "whole_sample"])
def test_illegal_or_incomplete_plan_is_rejected(change):
    plan = plan_structured_history(_prefix())
    if change == "future":
        plan = replace(plan, tokens=plan.tokens + (replace(plan.tokens[-1], token_type="public_speech"),))
    elif change == "missing":
        plan = replace(plan, tokens=plan.tokens[:-1], token_count=plan.token_count - 1)
    elif change == "digest":
        plan = replace(plan, plan_digest="0" * 64)
    elif change == "injection":
        plan = replace(plan, tokens=(replace(plan.tokens[0], source="PRIVATE_ROLE"),) + plan.tokens[1:])
    elif change == "capacity":
        plan = replace(plan, token_count=1025)
    else:
        plan = plan.to_record()
    with pytest.raises(ValueError):
        render_prompt(plan, 0)


def test_config_template_and_normalization_are_frozen(tmp_path):
    config = load_policy()
    assert config["normalization"]["epsilon"] == EPSILON
    assert config["subset"]["games_per_fold"] == 30
    for path, value in [("mode", "thinking"), ("dtype", "float32"), ("semantic_attempts", 2)]:
        changed = deepcopy(config); changed[path] = value
        with pytest.raises(ValueError, match="frozen"):
            validate_policy(changed)
    from werewolf.tom_llm import policy
    original = policy.asset_bytes
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(policy, "asset_bytes", lambda name: original(name) + b"drift")
        with pytest.raises(ValueError, match="asset drift"):
            load_policy()


def test_success_empty_failure_and_death_label_semantics_unchanged():
    q, observed = belief_target(0, (), "success")
    assert observed and q[0] == 0 and torch.all(q[1:] == 1/6)
    for status in ("error", None):
        q, observed = belief_target(0, (), status)
        assert not observed and not q.any()
    with pytest.raises(ValueError):
        make_request(_prefix(), 6, fold=0)


def test_uniform_success_distinct_from_failure_and_zero_mass_is_smoothed():
    valid = parse_output(response(), 0)
    invalid = parse_output("not JSON", 0)
    assert valid.status == "valid" and invalid.status == "fallback_uniform"
    assert valid.failure_type is None and invalid.failure_type == "invalid_json"
    assert valid.probability == pytest.approx(invalid.probability)
    sparse = parse_output(response(values=[1, 0, 0, 0, 0, 0]), 0)
    assert sparse.raw_zero_count == 5 and min(sparse.probability[1:]) == pytest.approx(EPSILON / 6)
    assert all(math.isfinite(v) for v in sparse.non_self_log_probability)
    validate_prediction(sparse.to_record())


@pytest.mark.parametrize("bad", [
    "", None, "refused", "{}", "[]", "true", response(values=[0]*6),
    response(values=[.4]*6), response(values=[-.1,.3,.2,.2,.2,.2]),
    response(values=[True,0,0,0,0,0]), response(values=["1",0,0,0,0,0]),
    response(values=[float("nan"),0,0,0,0,0]), response(values=[float("inf"),0,0,0,0,0]),
    '{"probabilities":{"player2":0.2,"player2":0.2}}',
    '{"probabilities":{"player1":1,"player2":0,"player3":0,"player4":0,"player5":0,"player6":0}}',
    response()[:-1] + ',"role":"Werewolf"}',
])
def test_invalid_outputs_are_terminal_fallbacks(bad):
    prediction = parse_output(bad, 0)
    assert prediction.status == "fallback_uniform"
    validate_prediction(prediction.to_record())
    assert prediction.probability[0] == 0


def test_rounding_tolerance_and_truncated_response():
    prediction = parse_output(response(values=[.167]*5 + [.166]), 0)
    assert prediction.status == "valid" and prediction.normalized_for_rounding
    assert sum(prediction.probability) == pytest.approx(1)
    assert parse_output(response(), 0, finish_reason="length").failure_type == "length"
    assert parse_output(response(values=[.168]*6), 0).failure_type == "invalid_sum"
    changed = prediction.to_record(); changed["status"] = "fallback_uniform"
    with pytest.raises(ValueError):
        validate_prediction(changed)


def rebound_prefix(*, raw_text=None, observation_marker="belief", include_current_speech=False):
    from tests.canonical_collection.test_structured_token_plan import _events
    from werewolf.canonical_collection import construct_authoritative_pre_prefix, freeze_public_event_history
    from werewolf.canonical_collection import construct_v1_speech_annotation
    original = _prefix()
    events = _events()
    if raw_text is not None:
        events[4]["raw_text"] = raw_text
    annotation = original.v1_annotations[0]
    annotation = construct_v1_speech_annotation(freeze_public_event_history(events[:5]),
        status=annotation.status, actions=annotation.actions, attempts=annotation.attempts)
    if include_current_speech:
        events.append({"event_id":"future-12", "event_index":12, "event_type":"public_speech",
                       "speaker":"player2", "raw_text":"future private-looking content"})
    return construct_authoritative_pre_prefix(game_id=original.game_id, boundary_id=original.boundary_id,
        step_index=original.step_index, report_trigger_id=original.report_trigger_id,
        current_speaker=original.current_speaker, alive_observer_ids=original.alive_observer_ids,
        public_event_history=freeze_public_event_history(events), v1_annotations=(annotation,),
        belief_observation_ids_by_observer={seat:observation_marker+"-"+seat for seat in original.alive_observer_ids})


def test_actual_pre_raw_text_and_label_links_cannot_influence_messages():
    first = make_request(rebound_prefix(), 0, fold=0)
    second = make_request(rebound_prefix(raw_text="SECRET_TRUE_ROLE Werewolf; TARGET_Q 0.9",
                                        observation_marker="private-label-reference"), 0, fold=0)
    assert first["prefix_digest"] != second["prefix_digest"]
    assert first["plan_digest"] != second["plan_digest"]
    assert first["messages"] == second["messages"]
    assert first["prompt_digest"] == second["prompt_digest"]
    assert "SECRET_TRUE_ROLE" not in second["rendered_prompt"]
    assert "private-label-reference" not in second["rendered_prompt"]


def test_current_speech_is_rejected_by_authoritative_pre_owner():
    with pytest.raises(ValueError, match="turn_start"):
        rebound_prefix(include_current_speech=True)


def test_full_capacity_plan_retains_all_tokens_without_summary():
    plan = plan_structured_history(_prefix())
    first = plan.tokens[0]
    terminal = replace(plan.tokens[-1], day=0, phase="night")
    tokens = tuple(replace(first, token_index=i, event_index=i) for i in range(1023))
    tokens += (replace(terminal, token_index=1023, event_index=1023),)
    full = replace(plan, tokens=tokens, token_count=1024)
    value=full.to_record(); value.pop("plan_digest")
    full=replace(full, plan_digest=sha256_bytes(canonical_json_bytes(value)))
    assert len(public_table(full).splitlines()) == 1025


@pytest.mark.parametrize("section,key,value", [
    ("decoding","enable_thinking",True),("decoding","temperature",.7),("decoding","n",2),
    ("decoding","max_tokens",2048),("normalization","epsilon",.001),
    ("normalization","fallback","drop_failed"),("subset","games_per_fold",15)])
def test_reviewed_protocol_controls_cannot_drift(section,key,value):
    changed=load_policy(); changed[section][key]=value
    with pytest.raises(ValueError,match="frozen"):
        validate_policy(changed)


def test_config_command_never_constructs_a_client(monkeypatch,capsys):
    from werewolf.cli import main
    from werewolf.tom_llm.policy import CONFIG_PATH
    import socket
    monkeypatch.setattr(socket,"create_connection",lambda *_a,**_k:pytest.fail("network forbidden"))
    assert main(["check-llm-reasoner-config","--config",str(CONFIG_PATH)]) == 0
    result=json.loads(capsys.readouterr().out)
    assert result["status"]=="valid" and result["formal_readiness"] is False
