"""外部投票前代理监督的边界、排除规则和原数据隔离测试。"""

from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from werewolf.artifact_io import ArtifactConflictError, canonical_json_bytes, sha256_bytes
from werewolf.tom import makto
from werewolf.tom.dataset import ExperimentCapacity


def _label(actor=3):
    return {"player": actor, "day": "1-1", "reason": "不能进入公开输入的判断理由",
            "role_prediction": {str(j): ["werewolf" if j in (1, 2) else "simple_villager"]
                                for j in range(1, 8)}}


def _events():
    roles = ["werewolf", "werewolf", "seer", "witch"] + ["simple_villager"] * 3
    return [*({"event": "roles", "content": {"player": j, "role": role}}
              for j, role in enumerate(roles, 1)),
            {"event": "cycle_round", "content": {"round": 1, "status": "night"}},
            {"event": "wolf_kill", "content": {"secret": "不能进入输入"}},
            {"event": "cycle_round", "content": {"round": 1, "status": "day", "context": "今天是平安夜"}},
            {"event": "speech", "content": {"player": 3, "day": "1-1", "context": "我是预言家，1号查杀。"}},
            {"event": "speech_summary", "content": {"identity_label": "不能进入输入"}},
            {"event": "vote_start", "content": None},
            {"event": "voted", "content": _label()},
            {"event": "voted", "content": _label(1)},
            {"event": "vote_results", "content": {"1": 2, "2": 2, "3": 2}},
            {"event": "game_end", "content": {"winner": "不能进入输入"}}]


def _annotations(sources):
    return {row["key"]: {"text_sha256": row["text_sha256"],
                         "actions": [["player3", "point_as_seer", "player3"],
                                     ["player3", "check_as_werewolf", "player1"]],
                         "origin": "test_fixture", "independently_reviewed": False}
            for row in makto.speech_records(sources)}


def _extract(events=None):
    path = makto.source_paths()[0]
    payload = canonical_json_bytes(_events() if events is None else events)
    return makto.extract_game(path, payload, _annotations({path: payload}))


def _role_fixture(events=None):
    """仅模拟已经完成的夜晚；角色卡与公开发言严格分离。"""
    if events is None:
        events = _events()
        events[9:9] = [
            {"event": "inquired", "content": {"night": 1, "player": 1, "is_werewolf": True}},
            {"event": "werewolf_kill", "content": {"night": 1, "target_player": 7}},
            {"event": "healed", "content": {"night": 1, "player": 7}},
            {"event": "poison", "content": {"night": 1, "player": None}},
        ]
    path = makto.source_paths()[0]
    payload = canonical_json_bytes(events)
    annotations = _annotations({path: payload})
    for row in annotations.values():
        row["actions"] = []
    boundaries = makto.speech_pre_snapshots(path, payload, annotations)
    return events, payload, boundaries


def _role_request(events, observer):
    _, payload, boundaries = _role_fixture(events)
    view = makto.role_belief_view(payload, boundaries[0], observer)
    return view, makto.build_role_belief_request(view)


def test_role_views_are_independent_of_other_private_information_and_global_truth():
    events, _, _ = _role_fixture()
    seer, seer_request = _role_request(events, 3)
    witch, witch_request = _role_request(events, 4)
    villager, villager_request = _role_request(events, 5)
    assert seer["known_werewolves"] == ["player1"]
    assert seer["private_facts"] == [{"source_index": 9, "night": 1, "event": "inquired",
                                      "target": "player1", "is_werewolf": True}]
    assert witch["known_non_werewolves"] == ["player4"]  # 银水7不是确定好人。
    assert [f["event"] for f in witch["private_facts"]] == ["healed", "poison", "privately_observed_attack_target"]
    assert villager["private_facts"] == []
    altered = deepcopy(events)
    altered[9]["content"].update(player=6, is_werewolf=False)
    assert _role_request(altered, 4)[1] == witch_request
    assert _role_request(altered, 5)[1] == villager_request
    altered = deepcopy(events)
    altered[10]["content"]["target_player"] = 6
    altered[11]["content"]["player"] = 6
    assert _role_request(altered, 3)[1] == seer_request
    altered[0]["content"]["role"], altered[4]["content"]["role"] = "simple_villager", "werewolf"
    assert _role_request(altered, 3)[1] == seer_request
    seer_request["messages"][0]["content"] = "被其他调用方修改"
    assert _role_request(events, 3)[1] != seer_request
    assert _role_request(events, 4)[1] == witch_request
    assert all(m["role"] in {"system", "user"} for m in witch_request["messages"])


def test_speech_pre_excludes_current_speech_gold_and_future_private_events():
    events, payload, boundaries = _role_fixture()
    boundary = boundaries[0]
    assert boundary["public_events"][-1]["event_type"] == "turn_start"
    assert not any(e["event_type"] == "public_speech" for e in boundary["public_events"])
    original = makto.build_role_belief_request(makto.role_belief_view(payload, boundary, 3))
    altered = deepcopy(events)
    altered[boundary["source_index"]]["content"]["context"] = "尚未说出的当前发言"
    altered.append({"event": "inquired", "content": {"night": 2, "player": 2, "is_werewolf": True}})
    for event in altered:
        if event["event"] == "voted":
            event["content"] = {"role_prediction": "不能读取的标签"}
    assert _role_request(altered, 3)[1] == original
    body = json.dumps(original, ensure_ascii=False)
    assert "role_prediction" not in body and "不能进入" not in body
    assert "我是预言家，1号查杀。" not in body
    assert original["prompt_version"] == makto.ROLE_BELIEF_PROMPT
    assert original["response_format"]["json_schema"]["schema"]["properties"]["suspected_werewolves"]["minItems"] == 1


def test_later_pre_contains_only_completed_public_speech_and_snapshots_do_not_alias():
    events, _, _ = _role_fixture()
    speech_index = next(i for i, e in enumerate(events) if e["event"] == "speech")
    events.insert(speech_index + 1, {"event": "speech", "content": {
        "player": 3, "day": "1-1", "context": "尚未发生的第二次发言"}})
    _, payload, boundaries = _role_fixture(events)
    assert len(boundaries) == 2
    later = makto.role_belief_view(payload, boundaries[1], 5)
    body = json.dumps(makto.build_role_belief_request(later), ensure_ascii=False)
    assert "我是预言家，1号查杀。" in body
    assert "尚未发生的第二次发言" not in body
    later["public_events"][0]["day"] = 999
    assert boundaries[0]["public_events"][0]["day"] == 0
    assert boundaries[1]["public_events"][0]["day"] == 0
    other_payload = canonical_json_bytes(events + [{"event": "game_end", "content": None}])
    with pytest.raises(ValueError, match="source digest mismatch"):
        makto.role_belief_view(other_payload, boundaries[0], 5)


def test_shared_renderer_preserves_original_online_prompt_bytes():
    _, _, boundaries = _role_fixture()
    boundary = boundaries[0]
    prefix = SimpleNamespace(public_event_history=makto.freeze_public_event_history(boundary["public_events"]),
                             public_event_digest=boundary["prefix_digest"])
    prompt = makto.PlayingAgentBeliefReporter.build_prompt(observer_id="player3", pre_prefix=prefix,
        known_werewolves=["player1"], known_non_werewolves=["player3"])
    # 对应 HEAD 02687f2 中的原函数实际输出；抽取纯渲染不改变在线任务身份。
    assert sha256_bytes(prompt.encode()) == "fe9878196affcd65ee2d20beca331d63d1f16d6c9d21ec81c0f9744ee14a48f7"


def test_role_prompt_sends_only_player_visible_facts_and_keeps_internal_audit():
    events, _, _ = _role_fixture()
    view, request = _role_request(events, 4)
    altered = deepcopy(view)
    altered["omitted_private_information"] = [{"reason": "内部缺失审计，不是玩家知识"}]
    altered["public_history_digest"] = "changed-internal-digest"
    for fact in altered["private_facts"]:
        fact.update(source_index=999, audit_note="内部审计记录")
    for event in altered["public_events"]:
        event.update(event_id="changed-internal-id", event_index=999, audit_note="内部审计记录")
    assert makto.build_role_belief_request(altered) == request
    assert view["private_facts"][0]["source_index"] == 11
    assert "omitted_private_information" in view
    body = json.dumps(request["messages"], ensure_ascii=False)
    for field in ("source_index", "omitted_private_information", "event_id", "event_index",
                  "public_history_digest", "public_action_count", "prompt_version", "audit_note"):
        assert field not in body
    private = json.loads(request["messages"][1]["content"].split("\n", 1)[1].split("\n\n", 1)[0])
    assert private["private_facts"] == [
        {"event": "healed", "night": 1, "target": "player7"},
        {"event": "poison", "night": 1, "target": None},
        {"event": "privately_observed_attack_target", "night": 1, "target": "player7"},
    ]
    assert private["witch_medicines"] == view["witch_medicines"]
    # 公共投票也是玩家事实，但其嵌套审计字段不能随 JSON 一并发送。
    view["public_events"].insert(-1, {"event_type": "vote_result", "votes": [
        {"voter": "player1", "target": "player2", "source_index": 999}]})
    body = json.dumps(makto.build_role_belief_request(view)["messages"], ensure_ascii=False)
    assert "source_index" not in body and "vote_result" in body


@pytest.mark.parametrize("change", ["missing", "null", "wrong_night", "daytime"])
def test_seer_missing_or_unconfirmed_private_result_is_never_inferred(change):
    events, _, _ = _role_fixture()
    if change == "missing":
        del events[9]
    elif change == "null":
        events[9]["content"]["player"] = None
    elif change == "wrong_night":
        events[9]["content"]["night"] = 2
    else:
        event = events.pop(9)
        speech_index = next(i for i, e in enumerate(events) if e["event"] == "speech")
        events.insert(speech_index, event)
    view, _ = _role_request(events, 3)
    assert view["private_facts"] == [] and view["known_werewolves"] == []
    assert view["known_non_werewolves"] == ["player3"]


def test_witch_null_missing_and_completed_night_skill_state_are_distinct():
    events, _, _ = _role_fixture()
    view, _ = _role_request(events, 4)
    assert view["witch_medicines"] == {"healed": "used", "poison": "available"}
    del events[12]  # 缺少毒药记录，不自动当作本夜压毒。
    view, _ = _role_request(events, 4)
    assert view["witch_medicines"] == {"healed": "used", "poison": "unknown"}
    events = events[:-1] + [
        {"event": "cycle_round", "content": {"round": 2, "status": "night"}},
        {"event": "werewolf_kill", "content": {"night": 2, "target_player": 5}},
        {"event": "poison", "content": {"night": 2, "player": 1}},
        {"event": "cycle_round", "content": {"round": 2, "status": "day", "context": "今天是平安夜"}},
        {"event": "speech", "content": {"day": "2-1", "player": 3, "context": "新一天"}},
    ]
    _, payload, boundaries = _role_fixture(events)
    later = makto.role_belief_view(payload, boundaries[-1], 4)
    assert later["witch_medicines"] == {"healed": "used", "poison": "used"}
    assert later["private_facts"][-1]["target"] == "player5"
    assert view["private_facts"] != later["private_facts"]
    # 两条本人技能记录都缺失时，即使原始数据有狼刀，也不能给女巫补造刀口。
    events, _, _ = _role_fixture()
    del events[11:13]
    view, _ = _role_request(events, 4)
    assert not view["private_facts"]
    assert set(view["witch_medicines"].values()) == {"unknown"}


def test_primary_observers_and_missing_public_history_remain_strict():
    events, payload, boundaries = _role_fixture()
    with pytest.raises(ValueError, match="non-Werewolf"):
        makto.role_belief_view(payload, boundaries[0], 1)
    events[13]["content"]["context"] = "死亡玩家：5号玩家"
    _, payload, boundaries = _role_fixture(events)
    with pytest.raises(ValueError, match="alive"):
        makto.role_belief_view(payload, boundaries[0], 5)
    del events[13]["content"]["context"]
    _, payload, boundaries = _role_fixture(events)
    assert boundaries[0]["plan"] is None
    with pytest.raises(ValueError, match="complete public history"):
        makto.role_belief_view(payload, boundaries[0], 3)


@pytest.mark.parametrize("raw", [None, "not JSON", '[]', '{"suspected_werewolves":["player5","player5"]}',
    '{"suspected_werewolves":["player5"]}', '{"suspected_werewolves":["player3"]}', '{"suspected_werewolves":["player8"]}',
    '{"suspected_werewolves":[],"extra":1}', '{"suspected_werewolves":[],"suspected_werewolves":["player1"]}'])
def test_failed_role_generation_never_becomes_successful_uniform(raw):
    events, _, _ = _role_fixture()
    view, _ = _role_request(events, 5)
    # 注入一项受控的好人硬知识，单独测试集合校验，不借助角色真值。
    view["known_non_werewolves"].append("player3")
    result = makto.parse_role_belief_response(raw, view)
    assert result["status"] == "error" and not result["observed"]
    assert result["suspected_werewolves"] is None and result["q"] == [0] * 7


def test_role_successful_empty_full_and_hard_knowledge_use_existing_targets():
    events, _, _ = _role_fixture()
    villager, _ = _role_request(events, 5)
    empty = makto.parse_role_belief_response('{"suspected_werewolves":[]}', villager)
    assert empty["observed"] and empty["suspected_werewolves"] == []
    assert empty["q"][4] == 0 and sum(empty["q"]) == pytest.approx(1)
    full = [p for p in makto.PLAYER_IDS if p != "player5"]
    result = makto.parse_role_belief_response(json.dumps({"suspected_werewolves": full}), villager)
    assert result["q"] == empty["q"]
    seer, _ = _role_request(events, 3)
    assert not makto.parse_role_belief_response('{"suspected_werewolves":[]}', seer)["observed"]
    assert makto.parse_role_belief_response('{"suspected_werewolves":["player1"]}', seer)["q"] == [1, 0, 0, 0, 0, 0, 0]


def _teacher_reply(content, *, finish="stop", model="Qwen/Qwen3-8B", refusal=None, tokens=100):
    return json.dumps({"model": model, "choices": [{"finish_reason": finish,
        "message": {"role": "assistant", "content": content, "refusal": refusal}}],
        "usage": {"prompt_tokens": tokens, "completion_tokens": 10, "total_tokens": tokens + 10}})


@pytest.mark.parametrize("change", ["length", "refusal", "model", "tokens", "JSON", "hard_knowledge"])
def test_trial_reply_checks_envelope_and_never_labels_failed_generations(change):
    events, _, _ = _role_fixture()
    view, _ = _role_request(events, 3)
    kwargs = {}
    content = '{"suspected_werewolves":["player1"]}'
    if change == "length":
        kwargs["finish"] = "length"
    elif change == "refusal":
        kwargs["refusal"] = "refused"
    elif change == "model":
        kwargs["model"] = "wrong-model"
    elif change == "tokens":
        kwargs["tokens"] = 101
    elif change == "JSON":
        content = "invalid JSON"
    else:
        content = '{"suspected_werewolves":[]}'
    result = makto.role_trial_reply(_teacher_reply(content, **kwargs), view, "Qwen/Qwen3-8B", 100)
    assert not result["observed"] and result["q"] == [0] * 7
    assert result["hard_knowledge_compliant"] is (False if change == "hard_knowledge" else None)


@pytest.fixture
def trial_wiring(tmp_path, monkeypatch):
    """只替换物理资源；执行真实的请求、日志、解析和一次性保护代码。"""
    from werewolf.tom_llm import deployment, experiment, live_runner, physical, service_manager
    events, _, _ = _role_fixture()
    inputs = []
    for i in range(12):
        view, _ = _role_request(events, (5, 3, 4)[i % 3])
        view["boundary_id"] += f"-fixture-{i // 3}"
        inputs.append({"view": view, "request": makto.build_role_belief_request(view)})
    resources = {"model_directory": str(tmp_path / "model"), "tokenizer_assets": str(tmp_path / "tokenizer"),
                 "environment_prefix": str(tmp_path / "environment")}
    runtime = {"environment_prefix": resources["environment_prefix"], "gpu_uuid": "GPU-MOCK",
               "argv_options": {"port": 18581, "structured-outputs-config": {"backend": "xgrammar"}}}
    calls = []
    class Manager:
        def __init__(self, *, root, spec, runtime, model_evidence, owner):
            assert root == output.parent and spec.key == "qwen3"
            assert model_evidence == {"directory": resources["model_directory"], "verification": "MOCK"}
            assert set(owner) == {"request_digest"}
        def status(self):
            return {"state": "not_started"}
        def start(self, *, execute):
            assert execute
            calls.append("start")
        def stop(self, *, execute):
            assert execute
            calls.append("stop")
        def guard(self, *args):
            return {"owned": "mock"}
    class Transport:
        failure = None
        def __init__(self, endpoint, *, alive_check):
            assert endpoint == "http://127.0.0.1:18581" and alive_check()["owned"] == "mock"
            self.count = 0
        def send(self, wire, identity):
            assert output.exists()  # dispatch 在调用前已经持久化。
            last = json.loads(output.read_bytes().splitlines()[-1])
            assert last["event"] == "dispatch" and last["request_digest"] == identity
            assert wire["max_tokens"] == 96 and wire["temperature"] == 0
            assert wire["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
            assert "game_id" not in wire and all(m["role"] in {"system", "user"} for m in wire["messages"])
            calls.append("send")
            index = self.count
            self.count += 1
            if self.failure is not None and index == 1:
                raise self.failure
            content = "invalid JSON" if index == 2 else json.dumps({
                "suspected_werewolves": inputs[index]["view"]["known_werewolves"]})
            return _teacher_reply(content)
    monkeypatch.setattr(experiment, "open_run", lambda *a, **kw: pytest.fail("MaKTO must not open a TomBaseline Run"))
    def profile(spec, **kwargs):
        assert spec.key == "qwen3" and kwargs["root"] == output.parent
        if kwargs["gpu_uuid"] != "GPU-MOCK":
            raise ValueError("explicit GPU UUID required")
        return runtime
    monkeypatch.setattr(deployment, "profile", profile)
    monkeypatch.setattr(physical, "verify_model", lambda spec, path: {"directory": path, "verification": "MOCK"})
    monkeypatch.setattr(service_manager, "Service", Manager)
    monkeypatch.setattr(live_runner, "HttpTransport", Transport)
    monkeypatch.setattr(makto, "role_trial_inputs", lambda *a: deepcopy(inputs))
    monkeypatch.setattr(makto, "check_role_trial_inputs", lambda *a: {
        "versions": {"vllm": "MOCK_NOT_REAL_PARITY"}, "input_token_counts": [100] * 12})
    monkeypatch.setattr(makto.sys, "platform", "linux")
    output = tmp_path / "trial.jsonl"
    return output, resources, inputs, calls, Transport


def test_trial_prepare_then_twelve_independent_calls_save_raw_and_invalid_results(trial_wiring):
    output, resources, inputs, calls, _ = trial_wiring
    prepared = makto.role_trial("mock_data", output, "mock_digest", **resources)
    assert prepared["status"] == "PREPARED_NOT_EXECUTED" and not calls
    result = makto.role_trial("mock_data", output, "mock_digest", **resources, gpu_uuid="GPU-MOCK", execute_live=True)
    assert result["request_count"] == 12 and result["failed"] == 1 and result["not_attempted"] == 0
    assert result["valid_nonempty"] == 4 and result["valid_empty"] == 7
    assert result["hard_knowledge_compliant"] == 11
    assert calls == ["start", *(["send"] * 12), "stop"]
    records = [json.loads(line) for line in output.read_bytes().splitlines()]
    assert sum(r["event"] == "received" and "raw_response" in r for r in records) == 12
    assert sum(r["event"] == "result" for r in records) == 12
    assert result["results"][2]["q"] == [0] * 7
    assert result["pre_changes"]
    before = output.read_bytes()
    with pytest.raises(ValueError, match="no automatic replay"):
        makto.role_trial("mock_data", output, "mock_digest", **resources, gpu_uuid="GPU-MOCK", execute_live=True)
    assert output.read_bytes() == before and calls.count("send") == 12


def test_trial_unknown_keeps_raw_error_and_never_accepts_or_resends(trial_wiring):
    from werewolf.tom_llm.provider_semantics import OutcomeUnknown
    output, resources, _, calls, transport = trial_wiring
    makto.role_trial("mock_data", output, "mock_digest", **resources)
    error = OutcomeUnknown("ownership lost after response")
    error.response_body = _teacher_reply('{"suspected_werewolves":["player1"]}').encode()
    error.http_status = 200
    transport.failure = error
    result = makto.role_trial("mock_data", output, "mock_digest", **resources, gpu_uuid="GPU-MOCK", execute_live=True)
    assert result["failed"] == 1 and result["not_attempted"] == 10
    assert result["results"][1]["status"] == "unknown" and not result["results"][1]["observed"]
    assert calls.count("send") == 2 and calls[-1] == "stop"
    record = next(json.loads(line) for line in output.read_bytes().splitlines()
                  if json.loads(line)["event"] == "transport_error")
    assert bytes.fromhex(record["raw_response_hex"]) == error.response_body
    with pytest.raises(ValueError, match="no automatic replay"):
        makto.role_trial("mock_data", output, "mock_digest", **resources, gpu_uuid="GPU-MOCK", execute_live=True)
    assert calls.count("send") == 2


def test_missing_real_tokenizer_or_schema_assets_cannot_publish_prepared_trial(trial_wiring, monkeypatch):
    output, resources, _, calls, _ = trial_wiring
    def missing(*args):
        raise FileNotFoundError("real tokenizer or xgrammar unavailable")
    monkeypatch.setattr(makto, "check_role_trial_inputs", missing)
    with pytest.raises(FileNotFoundError):
        makto.role_trial("mock_data", output, "mock_digest", **resources)
    assert not output.exists() and calls == []


def test_trial_execution_needs_explicit_gpu_but_cpu_preparation_does_not(trial_wiring):
    output, resources, _, calls, _ = trial_wiring
    makto.role_trial("mock_data", output, "mock_digest", **resources)
    before = output.read_bytes()
    with pytest.raises(ValueError, match="explicit GPU UUID"):
        makto.role_trial("mock_data", output, "mock_digest", **resources, execute_live=True)
    assert output.read_bytes() == before and not calls


def test_trial_resource_binding_changes_cannot_reuse_preparation(trial_wiring):
    output, resources, _, calls, _ = trial_wiring
    makto.role_trial("mock_data", output, "mock_digest", **resources)
    altered = {**resources, "tokenizer_assets": str(output.parent / "different-tokenizer")}
    with pytest.raises(ValueError, match="binding changed"):
        makto.role_trial("mock_data", output, "mock_digest", **altered, gpu_uuid="GPU-MOCK", execute_live=True)
    assert not calls


def test_trial_cli_uses_registered_teacher_paths_without_service_run(monkeypatch, tmp_path):
    calls = []
    def trial(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "PREPARED_NOT_EXECUTED"}
    monkeypatch.setattr(makto, "role_trial", trial)
    args = ["--role-trial", "--data", str(tmp_path / "data"), "--output", str(tmp_path / "trial.jsonl"),
            "--expected-data-digest", "mock-digest"]
    makto.main(args)
    assert calls[0][0] == (tmp_path / "data", tmp_path / "trial.jsonl", "mock-digest")
    assert calls[0][1] == {"model_directory": makto.TEACHER_MODEL_DIRECTORY,
        "tokenizer_assets": makto.TEACHER_TOKENIZER_ASSETS, "environment_prefix": makto.TEACHER_ENVIRONMENT_PREFIX,
        "gpu_uuid": None, "execute_live": False}
    with pytest.raises(SystemExit):
        makto.main([*args, "--service-run", str(tmp_path / "obsolete-run")])
    assert len(calls) == 1


@pytest.mark.parametrize("failure", [None, "template", "budget", "schema"])
def test_trial_cpu_checks_actual_message_path_with_controlled_tokenizer_and_schema(monkeypatch, tmp_path, failure):
    import importlib.metadata
    import sys
    from jinja2 import Environment
    from werewolf.tom_llm import deployment, model_spec
    events, _, _ = _role_fixture()
    view, request = _role_request(events, 3)
    spec = model_spec.ModelSpec("qwen3")
    row = spec.record()
    template = Environment().from_string(model_spec.template_bytes(spec).decode())
    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, **kwargs):
            assert messages == request["messages"] and kwargs["enable_thinking"] is False
            text = template.render(messages=messages, bos_token=None, eos_token=row["tokenizer"]["eos_token"], **kwargs)
            if tokenize:
                return self.encode(text)
            return text + ("wrong template" if failure == "template" else "")
        def encode(self, text, **kwargs):
            return [1] * (8000 if failure == "budget" else len(text))
    schemas = []
    def validate(params):
        schemas.append(params.structured_outputs.json)
        if failure == "schema":
            raise ValueError("unsupported actual schema")
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", SimpleNamespace(
        SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs),
        StructuredOutputsParams=lambda **kwargs: SimpleNamespace(**kwargs)))
    monkeypatch.setitem(sys.modules, "vllm.v1.structured_output.backend_xgrammar", SimpleNamespace(validate_xgrammar_grammar=validate))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.27.0" if name == "vllm" else "MOCK")
    monkeypatch.setattr(model_spec, "load_cpu_tokenizer", lambda *args: Tokenizer())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(deployment, "service_environment", lambda *a, **kw: {
        "CUDA_VISIBLE_DEVICES": "", "TMPDIR": str(tmp_path)})
    prefix = Path(sys.executable).parent.parent
    inputs = [{"view": view, "request": request}] * 12
    if failure:
        with pytest.raises(ValueError, match={"template": "template/token", "budget": "token budget", "schema": "actual schema"}[failure]):
            makto.check_role_trial_inputs(inputs, spec, "mock-small-assets", prefix, tmp_path)
    else:
        result = makto.check_role_trial_inputs(inputs, spec, "mock-small-assets", prefix, tmp_path)
        assert len(result["input_token_counts"]) == 12 and len(result["input_ids_digests"]) == 12
        assert len(schemas) == 1 and schemas[0] == request["response_format"]["json_schema"]["schema"]


def test_label_conversion_uses_candidates_not_actual_identity_or_reason():
    label = _label()
    result = makto.convert_label(label)
    assert result["label_usable"]
    assert result["wolf_support"] == ["player1", "player2"]
    assert result["q"] == [.5, .5, 0, 0, 0, 0, 0]
    label["reason"] = "相反的理由也不能自动改写标签"
    assert makto.convert_label(label) == result
    label["role_prediction"]["4"] = ["seer", "witch"]
    assert makto.convert_label(label)["label_usable"]
    label["role_prediction"]["4"] = ["werewolf"]
    converted = makto.convert_label(label)
    assert converted["label_usable"] and len(converted["wolf_support"]) == 3
    assert converted["q"][2] == 0
    assert sum(converted["q"]) == pytest.approx(1)


@pytest.mark.parametrize("change,reason", [
    (lambda p: p.pop("4"), "missing_target"),
    (lambda p: p.update({"4": ["NA"]}), "NA"),
    (lambda p: p.update({"4": ["werewolf", "seer"]}), "mixed_wolf_nonwolf"),
    (lambda p: p.update({"4": []}), "invalid_candidates"),
    (lambda p: p.update({"4": ["guard"]}), "invalid_candidates"),
    (lambda p: p.update({"1": ["seer"], "2": ["witch"]}), "empty_support"),
])
def test_unconvertible_label_never_becomes_uniform_or_successful_empty(change, reason):
    label = _label()
    change(label["role_prediction"])
    result = makto.convert_label(label)
    assert reason in result["exclusion_reasons"]
    assert not result["label_usable"] and not any(result["q"])
    assert result["raw_role_prediction"] == label["role_prediction"]


def test_confirmed_conflict_quarantines_without_rewriting():
    label = _label()
    result = makto.convert_label(label, quarantined=True)
    assert result["raw_role_prediction"] == label["role_prediction"]
    assert result["quarantined"] and not result["label_usable"]
    assert result["wolf_support"] == ["player1", "player2"]
    assert result["exclusion_reasons"] == ["confirmed_text_label_conflict"]
    assert all(split == "train" for split, _, _ in makto.CONFLICTS)


def test_vote_start_input_excludes_current_votes_gold_and_private_or_future_events():
    game, inputs, labels = _extract()
    boundary = inputs[0]
    serialized = json.dumps(boundary, ensure_ascii=False)
    assert "不能进入" not in serialized
    assert "role_prediction" not in serialized
    assert "raw_role_prediction" not in json.dumps(game)
    assert not any(e["event_type"] in {"vote_result", "exile_result"} for e in boundary["public_events"])
    assert boundary["public_events"][-1]["phase"] == "vote"
    assert boundary["plan"]["tokens"][-1]["token_type"] == "phase_change"
    assert labels[0]["effective"] and not labels[1]["effective"]
    altered = _events()
    # 角色真值只影响外围人口；即使改变私人信息和后续选票，公开PRE完全不变。
    altered[0]["content"]["role"], altered[2]["content"]["role"] = "seer", "werewolf"
    altered[8]["content"] = {"secret": "其他秘密"}
    altered[-2]["content"] = {"1": 4, "2": 4, "3": 4}
    _, other_inputs, other_labels = _extract(altered)
    assert other_inputs == inputs
    assert other_labels[0]["q"] == labels[0]["q"]
    assert not other_labels[0]["effective"]


def test_missing_public_deaths_never_reconstructed_from_private_actions():
    events = _events()
    del events[9]["content"]["context"]
    _, inputs, labels = _extract(events)
    assert inputs[0]["plan"] is None and inputs[0]["alive"] is None
    assert labels[0]["label_usable"] and not labels[0]["effective"]
    assert "missing_public_death_announcement" in labels[0]["exclusion_reasons"]


def test_pk_vote_includes_previous_public_votes_but_not_current_votes():
    events = _events()[:-1]
    events[-1]["content"] = {"1": 1, "2": 2, "3": None}
    events.extend([
        {"event": "speech", "content": {"player": 3, "day": "1-2", "context": "我仍然认为1号是狼。"}},
        {"event": "vote_start", "content": None},
        {"event": "voted", "content": {**_label(), "day": "1-2"}},
        {"event": "vote_results", "content": {"1": 2, "2": 1, "3": 1}},
    ])
    _, inputs, _ = _extract(events)
    assert len(inputs) == 2
    assert inputs[1]["phase"] == "pk_vote"
    assert sum(e["event_type"] == "vote_result" for e in inputs[1]["public_events"]) == 1
    assert any(t["token_type"] == "speech_action" for t in inputs[1]["plan"]["tokens"])
    assert all(t["day"] in (0, 1) for t in inputs[1]["plan"]["tokens"])


def test_annotations_bound_to_public_text_and_fixed_source_coverage():
    sources = {p: canonical_json_bytes(_events()) for p in makto.source_paths()}
    annotations = _annotations(sources)
    with pytest.raises(ValueError, match="fixed 20 Train and 8 Test"):
        makto.prepare(dict(list(sources.items())[:-1]), annotations)
    with pytest.raises(ValueError, match="annotation coverage"):
        makto.prepare(sources, {})
    first = next(iter(annotations))
    bad = deepcopy(annotations)
    bad[first]["text_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="text identity"):
        makto.prepare(sources, bad)
    bad = deepcopy(annotations)
    bad[first]["actions"][0][0] = "player4"
    with pytest.raises(ValueError, match="subject"):
        makto.prepare(sources, bad)


def test_artifact_split_reader_partial_mask_tensors_and_no_overwrite(tmp_path, monkeypatch):
    sources = {p: canonical_json_bytes(_events()) for p in makto.source_paths()}
    output = tmp_path / "makto"
    artifact = makto.publish(output, sources, _annotations(sources))
    assert artifact.manifest["revision"] == makto.REVISION
    assert artifact.manifest["label_timing"] == "UNKNOWN"
    opened = []
    read = makto.read_artifact_file

    def record_read(envelope, path):
        opened.append(path)
        return read(envelope, path)

    monkeypatch.setattr(makto, "read_artifact_file", record_read)
    data = makto.MaKTOToMDataset(output, "train", capacity=ExperimentCapacity(64))
    assert opened == ["public/train.jsonl", "supervision/train.jsonl"]
    assert len(data) == 20
    sample = data[0]
    assert sample.public.attention_mask.sum() == sample.token_count
    assert sample.q.shape == (7, 7) and sample.public.event_ids.shape == (64,)
    assert data.effective_mask(sample).tolist() == [False, False, True, False, False, False, False]
    assert torch.all(sample.q.diagonal() == 0)
    with pytest.raises(ValueError, match="capacity"):
        makto.MaKTOToMDataset(output, "train", capacity=ExperimentCapacity(1))
    before = sha256_bytes((output / "manifest.json").read_bytes())
    assert makto.publish(output, sources, _annotations(sources)).manifest_digest == artifact.manifest_digest
    changed = _annotations(sources)
    next(iter(changed.values()))["origin"] = "changed"
    with pytest.raises(ArtifactConflictError):
        makto.publish(output, sources, changed)
    assert sha256_bytes((output / "manifest.json").read_bytes()) == before


def _holdout_data():
    """仿真同样的游戏行数分布；不读取官方Test或真实Checkpoint。"""
    from werewolf.tom.dataset import CanonicalToMSample, PublicTensors

    counts = [0, 1, 3, 2, 2, 2, 3, 5, 3, 4, 2, 1, 2, 3, 2, 7, 1, 4, 4, 3]
    samples = []
    for game, count in zip(range(21, 41), counts):
        game_id = f"makto/train/seer_witch/game_{game}"
        for row in range(max(1, count)):
            arrays = {name: torch.tensor([1, 2]) for name in (
                "event_ids", "source_ids", "action_ids", "target_ids", "day_ids", "phase_ids")}
            public = PublicTensors(**arrays, attention_mask=torch.ones(2, dtype=torch.bool))
            q = torch.zeros(7, 7)
            # 不同游戏具有不同目标熵，能区分游戏宏平均与行平均。
            support = 1 + (game % 3)
            q[0, 1:1 + support] = 1 / support
            observed = torch.tensor([bool(count), False, False, False, False, False, False])
            samples.append(CanonicalToMSample(game_id, f"{game_id}/{row}", "a" * 64, "b" * 64,
                                             2, public, q, observed, torch.ones(7, dtype=torch.bool)))
    return SimpleNamespace(split="train", source_digest="d" * 64, samples=samples,
                           effective_mask=lambda sample: sample.label_observed & sample.observer_alive)


def test_internal_holdout_is_fixed_whole_game_and_official_test_is_forbidden():
    data = _holdout_data()
    split = makto.internal_split(data)
    assert split["validation_game_ids"] == [f"makto/train/seer_witch/game_{i}" for i in (31, 34, 37, 38)]
    assert len(split["training_game_ids"]) == 15
    assert len(split["validation_game_ids"]) == 4
    counts = split["effective_rows_by_game"]
    assert sum(counts[g] for g in split["training_game_ids"]) == 44
    assert sum(counts[g] for g in split["validation_game_ids"]) == 10
    assert split["unsupervised_game_ids"] == ["makto/train/seer_witch/game_21"]
    data.samples.reverse()
    assert makto.internal_split(data) == split
    data.split = "test"
    with pytest.raises(ValueError, match="original Train"):
        makto.internal_split(data)


def test_validation_reuses_masks_and_game_macro_and_restores_model_mode():
    data = _holdout_data()
    split = makto.internal_split(data)

    class UniformModel(torch.nn.Module):
        def forward(self, **public):
            assert not self.training and not torch.is_grad_enabled()
            assert set(public) == {"event_ids", "source_ids", "action_ids", "target_ids",
                                   "day_ids", "phase_ids", "attention_mask"}
            result = torch.full((1, 7, 7), -math.log(6), dtype=torch.float64)
            return result.masked_fill(torch.eye(7, dtype=torch.bool), -torch.inf)

    model = UniformModel().train()
    result = makto.validation_scores(data, split, model)
    assert model.training
    assert result["uniform_kl"] == result["epoch35_kl"]
    assert result["absolute_difference_epoch35_minus_uniform"] == 0
    assert result["relative_difference"] == 0
    assert sum(g["effective_rows"] for g in result["games"]) == 10
    macro = sum(g["uniform_kl"] for g in result["games"]) / 4
    pooled = sum(g["uniform_kl"] * g["effective_rows"] for g in result["games"]) / 10
    assert result["uniform_kl"] == pytest.approx(macro)
    assert abs(macro - pooled) > .01
    without_model = makto.validation_scores(data, split)
    assert without_model["row_identity_digest"] == result["row_identity_digest"]
    assert without_model["epoch35_kl"] is None
    split["validation_game_ids"].reverse()
    with pytest.raises(ValueError, match="split identity"):
        makto.validation_scores(data, split)


@pytest.mark.parametrize("invalid", ["digest", "epoch", "graph", "temporal"])
def test_epoch35_loader_rejects_wrong_identity_before_model_construction(monkeypatch, invalid):
    from werewolf.tom import qwen3_large

    checkpoint = SimpleNamespace(manifest_digest=makto.EPOCH35_DIGEST, manifest={
        "completed_epoch": 35, "selection": {"best_epoch": 35},
        "binding": {"graph": qwen3_large.graph_identity(), "temporal_artifact_digest": "f" * 64}})
    temporal = SimpleNamespace(manifest_digest="f" * 64)
    if invalid == "digest":
        checkpoint.manifest_digest = "0" * 64
    elif invalid == "epoch":
        checkpoint.manifest["completed_epoch"] = 34
    elif invalid == "graph":
        checkpoint.manifest["binding"]["graph"]["total_parameters"] = 1
    else:
        temporal.manifest_digest = "0" * 64
    monkeypatch.setattr(makto, "open_artifact_envelope", lambda path, **kwargs:
                        checkpoint if path == "checkpoint" else temporal)
    monkeypatch.setattr(qwen3_large, "Qwen3ToMLarge", lambda *a, **k: pytest.fail("invalid identity reached construction"))
    with pytest.raises(ValueError, match="identity|mismatch"):
        makto.load_epoch35_model("checkpoint", "temporal")


def test_epoch35_reader_restores_only_model_without_old_optimizer_or_rng(monkeypatch):
    from werewolf.artifact_io import TensorValue, tensor_state
    from werewolf.tom import qwen3_large, temporal

    checkpoint = SimpleNamespace(manifest_digest=makto.EPOCH35_DIGEST, manifest={
        "completed_epoch": 35, "selection": {"best_epoch": 35}, "containers": {"model": {}},
        "binding": {"graph": qwen3_large.graph_identity(), "temporal_artifact_digest": "f" * 64,
                    "protocol": {"initialization_seed": 17}}})
    time = SimpleNamespace(manifest_digest="f" * 64, manifest={"parent_digest": "a" * 64})
    monkeypatch.setattr(makto, "open_artifact_envelope", lambda path, **kwargs:
                        checkpoint if path == "checkpoint" else time)
    monkeypatch.setattr(temporal.TemporalCodeProvider, "explicit", lambda *a: "verified-time")
    # 仅缩小模型规模；状态恢复仍使用原有严格键、形状、dtype及逐字节校验。
    model = torch.nn.Linear(2, 7)
    tensors = {name: TensorValue(torch.full_like(value, .25).numpy(), True)
               for name, value in model.state_dict().items()}
    monkeypatch.setattr(qwen3_large, "Qwen3ToMLarge", lambda *a, **k: model)
    monkeypatch.setattr(tensor_state, "decode_tensor_container", lambda *a, **k: tensors)
    reads = []

    def read(envelope, name):
        reads.append(name)
        return b"model bytes"

    monkeypatch.setattr(makto, "read_artifact_file", read)
    restored = makto.load_epoch35_model("checkpoint", "temporal")
    assert reads == ["model.bin"]
    assert not restored.training
    assert all(torch.all(value == .25) for value in restored.state_dict().values())


def test_baseline_cli_never_downloads_or_marks_missing_checkpoint_as_pass(tmp_path, monkeypatch, capsys):
    data = _holdout_data()
    data.source_digest = makto.PREPARED_DIGEST
    monkeypatch.setattr(makto, "MaKTOToMDataset", lambda path, split: data)
    monkeypatch.setattr(makto, "download_sources", lambda: pytest.fail("baseline cannot redownload data"))
    split_path, result_path = tmp_path / "split.json", tmp_path / "baseline.json"
    makto.main(["--baseline", "--data", "existing-data", "--split-record", str(split_path),
                "--output", str(result_path)])
    result = json.loads(result_path.read_bytes())
    assert result["epoch35_status"] == "NOT_RUN" and result["epoch35_kl"] is None
    assert result["quality_status"] == "ANNOTATION_SEMANTICS_REVIEW_REQUIRED"
    assert json.loads(split_path.read_bytes())["dataset_digest"] == makto.PREPARED_DIGEST
    assert json.loads(capsys.readouterr().out) == result


def test_new_records_cannot_modify_an_existing_artifact_or_record(tmp_path):
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="inside an immutable artifact"):
        makto._write_record(frozen / "split.json", {"new": True})
    path = tmp_path / "record.json"
    makto._write_record(path, {"new": True})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        makto._write_record(path, {"new": False})
    assert path.read_bytes() == before


def test_annotation_revision_preserves_sources_gold_and_public_boundary(tmp_path, monkeypatch):
    sources = {p: canonical_json_bytes(_events()) for p in makto.source_paths()}
    original = _annotations(sources)
    key = next(iter(original))
    original[key]["actions"][1][2] = "player2"  # 原公开文本明确说1号，故模拟旧标注编号错误。
    parent = makto.publish(tmp_path / "parent", sources, original)
    monkeypatch.setattr(makto, "PREPARED_DIGEST", parent.manifest_digest)
    corrected = deepcopy(original)
    corrected[key]["actions"][1][2] = "player1"
    correction = {"key": key, "text_sha256": original[key]["text_sha256"],
                  "before": original[key]["actions"], "after": corrected[key]["actions"],
                  "reason": "公开原文目标为1号", "evidence": "1号查杀", "rule": "explicit_target"}
    revision = makto.publish(tmp_path / "revision", sources, corrected,
                             previous=parent.path, corrections=[correction])
    assert revision.manifest["annotation_correction"]["parent_digest"] == parent.manifest_digest
    assert revision.manifest["source_digests"] == parent.manifest["source_digests"]
    for split in ("train", "test"):
        assert (revision.path / f"supervision/{split}.jsonl").read_bytes() == (parent.path / f"supervision/{split}.jsonl").read_bytes()
    assert (revision.path / "corrections.jsonl").is_file()
    before_inputs = [json.loads(x) for x in (parent.path / "public/train.jsonl").read_bytes().splitlines()]
    after_inputs = [json.loads(x) for x in (revision.path / "public/train.jsonl").read_bytes().splitlines()]
    assert before_inputs[0]["prefix_digest"] == after_inputs[0]["prefix_digest"]
    assert before_inputs[0]["plan"]["plan_digest"] != after_inputs[0]["plan"]["plan_digest"]
    with pytest.raises(ValueError, match="exactly the changed"):
        makto.publish(tmp_path / "missing-log", sources, corrected, previous=parent.path)
    bad = {**correction, "before": []}
    with pytest.raises(ValueError, match="evidence identity"):
        makto.publish(tmp_path / "bad-log", sources, corrected, previous=parent.path, corrections=[bad])
    changed_sources = dict(sources)
    events = _events()
    events[8]["content"]["secret"] = "修改私人内容仍属于改变原始来源"
    changed_sources[next(iter(changed_sources))] = canonical_json_bytes(events)
    with pytest.raises(ValueError, match="original sources"):
        makto.publish(tmp_path / "bad-source", changed_sources, corrected, previous=parent.path, corrections=[correction])


@pytest.mark.parametrize("field,match", [("labels", "supervision"), ("inputs", "public boundaries")])
def test_annotation_revision_rejects_label_or_temporal_drift(tmp_path, monkeypatch, field, match):
    sources = {p: canonical_json_bytes(_events()) for p in makto.source_paths()}
    annotations = _annotations(sources)
    parent = makto.publish(tmp_path / "parent", sources, annotations)
    monkeypatch.setattr(makto, "PREPARED_DIGEST", parent.manifest_digest)
    prepared = makto.prepare(sources, annotations)
    if field == "labels":
        prepared[2][0]["q"] = [0] * 7
    else:
        prepared[1][0]["day"] = 99
    monkeypatch.setattr(makto, "prepare", lambda *a: prepared)
    with pytest.raises(ValueError, match=match):
        makto.publish(tmp_path / "invalid", sources, annotations, previous=parent.path)


def test_baseline_requires_explicit_revised_digest_and_keeps_same_game_split(tmp_path, monkeypatch, capsys):
    data = _holdout_data()
    original_split = makto.internal_split(data)
    data.source_digest = "e" * 64
    data.annotation_correction = {"parent_digest": makto.PREPARED_DIGEST}
    revised_split = makto.internal_split(data)
    assert {k: v for k, v in original_split.items() if k not in {"dataset_digest", "split_digest"}} == {
        k: v for k, v in revised_split.items() if k not in {"dataset_digest", "split_digest"}}
    monkeypatch.setattr(makto, "MaKTOToMDataset", lambda path, split: data)
    args = ["--baseline", "--data", "existing-data", "--split-record", str(tmp_path / "split.json"),
            "--output", str(tmp_path / "baseline.json")]
    with pytest.raises(ValueError, match="artifact digest mismatch"):
        makto.main(args)
    assert not (tmp_path / "split.json").exists()
    makto.main([*args, "--expected-data-digest", data.source_digest])
    result = json.loads(capsys.readouterr().out)
    assert result["dataset_digest"] == data.source_digest
    assert result["epoch35_status"] == "NOT_RUN"
    assert result["quality_status"] == "TARGETED_CORRECTIONS_APPLIED_NOT_FULL_CERTIFICATION"
