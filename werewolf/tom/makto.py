"""MaKTO 七人 Seer-Witch 外部代理标签；不进入 Canonical Development。"""

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.request import urlopen

import torch

from werewolf.artifact_io import (
    canonical_json_bytes, canonical_jsonl_bytes, publish_artifact,
    open_artifact_envelope, read_artifact_file, sha256_bytes,
)
from werewolf.canonical_collection.public_history import PLAYER_IDS, freeze_public_event_history
from werewolf.canonical_collection.speech import V1SpeechAction, V1_SPEECH_ONTOLOGY_VERSION
from werewolf.speech.speech_perceiver import SpeechPerceiver
from werewolf.speech.private_belief_perceiver import (
    PRIVATE_BELIEF_MAX_TOKENS, PlayingAgentBeliefReporter, private_belief_response_format,
)
from werewolf.speech.validation import close_hard_knowledge, normalize_player, validate_player_suspicion
from werewolf.structured_history import (
    STRUCTURED_TOKEN_PLANNER_VERSION, StructuredTokenDescriptor, StructuredTokenPlan,
    plan_public_history,
)
from werewolf.tom.dataset import (
    CanonicalToMSample, ExperimentCapacity, belief_target, tensorize_structured_plan,
)


REVISION = "7d7d516f780879dbe9f512ef29655f51548f7578"
PREPARED_DIGEST = "eadb9dc8c5d31a0594f734561c8144485c1f8407432dbebc928d050cb17be8ad"
EPOCH35_DIGEST = "6229037415e6aec74f3e615f3f4838f269a6ff1634b6bd9e83dbd291b9ef1609"
DATASET = "ReneeYe/werewolf_game_reasoning"
VERSION = "makto_seer_witch_proxy_belief_v1"
SPLITS = {"train": range(21, 41), "test": range(6, 14)}
# 只隔离 Train 审阅已经确认的冲突；不根据 Test 结果修改规则或补标签。
CONFLICTS = {
    ("train", 23, 34): "reason称1号狼人，role_prediction将1号标为seer",
    ("train", 40, 36): "reason称6号狼人，role_prediction将6号标为seer",
    ("train", 36, 33): "reason称2号seer，role_prediction将2号标为witch",
}
ROLES = {"werewolf", "simple_villager", "seer", "witch"}
ROLE_BELIEF_PROMPT = "makto_pre_speech_role_belief"
# 真实调用前固定的小规模质量试验，均来自既有内部 Training；不用于性能评价。
ROLE_BELIEF_TRIAL_REQUESTS = (
    (22, 15, "player3"), (22, 15, "player2"), (22, 15, "player4"),
    (22, 27, "player3"), (22, 27, "player2"), (22, 27, "player4"),
    (36, 44, "player3"), (36, 44, "player1"), (36, 48, "player1"),
    (29, 44, "player7"), (29, 44, "player6"), (29, 46, "player6"),
)
ROLE_BELIEF_TRIAL_DIGEST = "5d33f9eaea0057469169b9374fae33e392b5d46a9e1b836090f2918123332f4b"


def source_paths():
    return [f"raw/{split}/7_player_game/seer_witch/game_{game}/event_zh.json"
            for split, games in SPLITS.items() for game in games]


def download_sources():
    """固定版本及28个明确文件，无隐式缓存、模型下载或其他游戏读取。"""
    root = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/"
    return {path: urlopen(root + path, timeout=60).read() for path in source_paths()}


def speech_records(sources):
    """给标注者的视图只含公开发言及位置，不含角色或任何监督。"""
    rows = []
    for path, payload in sources.items():
        for index, event in enumerate(json.loads(payload)):
            if event["event"] == "speech":
                content = event["content"]
                rows.append({"key": path.rsplit("/", 1)[0] + f"/{index}",
                             "speaker": content["player"], "day": content["day"],
                             "text": content["context"],
                             "text_sha256": sha256_bytes(content["context"].encode())})
    return rows


def convert_label(content, *, quarantined=False):
    """候选角色到支持集的保守转换；真实角色不进入本函数。"""
    actor = content["player"]
    labels = content.get("role_prediction")
    reasons = []
    support = []
    values = []
    if not isinstance(labels, dict) or any(str(j) not in labels for j in range(1, 8) if j != actor):
        reasons.append("missing_target")
    else:
        values = [labels[str(j)] for j in range(1, 8) if j != actor]
        if any(not isinstance(v, list) or not v or len(v) != len(set(v))
               or set(v) - ROLES - {"NA"} for v in values):
            reasons.append("invalid_candidates")
        else:
            support = [f"player{j}" for j in range(1, 8)
                       if j != actor and labels[str(j)] == ["werewolf"]]
            if any("NA" in v for v in values):
                reasons.append("NA")
            if any("werewolf" in v and set(v) != {"werewolf"} for v in values):
                reasons.append("mixed_wolf_nonwolf")
            if not support:
                reasons.append("empty_support")
    if quarantined:
        reasons.append("confirmed_text_label_conflict")
    q, observed = belief_target(actor - 1, tuple(support) if not reasons else (),
                                "success" if not reasons else None)
    return {"raw_role_prediction": labels, "wolf_support": support,
            "label_usable": observed, "q": q.tolist(), "exclusion_reasons": reasons,
            "quarantined": quarantined}


def extract_game(path, payload, annotations, *, speech_pre=False):
    """共享公开时间线；默认保持原投票代理任务，发言模式不读取投票标签。"""
    split, game_number = path.split("/")[1], int(path.split("/")[-2].split("_")[1])
    game_id = f"makto/{split}/seer_witch/game_{game_number}"
    raw = json.loads(payload)
    roles = {e["content"]["player"]: e["content"]["role"] for e in raw if e["event"] == "roles"}
    if Counter(roles.values()) != Counter(werewolf=2, simple_villager=3, seer=1, witch=1):
        raise ValueError("not a seven-player Seer-Witch source")
    public, actions, boundaries, labels = [], {}, [], []
    alive, problems = set(range(1, 8)), []
    current_boundary = None
    day, phase, latest_speech_round = 0, "night", None

    def emit(kind, **fields):
        record = {"event_id": f"event-{len(public):06d}", "event_index": len(public),
                  "event_type": kind, **fields}
        public.append(record)
        return record["event_id"]

    def change(next_day, next_phase):
        nonlocal day, phase
        day, phase = next_day, next_phase
        emit("phase_change", day=day, phase=phase)

    def snapshot(index, boundary_id):
        events = deepcopy(public)
        digest = sha256_bytes(canonical_json_bytes({"game_id": game_id,
            "boundary_id": boundary_id, "public_events": events}))
        plan = None
        if not problems:
            plan = plan_public_history(freeze_public_event_history(events), actions,
                                       boundary_digest=digest)
            if speech_pre and plan.token_count > 1024:
                raise ValueError("complete public input exceeds model capacity; truncation forbidden")
        return {"game_id": game_id, "split": split, "boundary_id": boundary_id,
                "source_index": index, "day": day, "phase": phase,
                "prefix_digest": digest, "public_events": events,
                "input_problems": deepcopy(problems),
                "alive": [j in alive for j in range(1, 8)] if not problems else None,
                "plan": plan.to_record() if plan else None}

    for index, event in enumerate(raw):
        kind, c = event["event"], event.get("content")
        if kind == "cycle_round" and c["status"] == "night":
            change(c["round"] - 1, "night")
        elif kind == "cycle_round" and c["status"] == "day":
            text = c.get("context")
            if text is None:
                problems.append({"reason": "missing_public_death_announcement", "source_index": index})
            elif text == "今天是平安夜" or text.startswith("死亡玩家："):
                dead = sorted(set(map(int, re.findall(r"(\d+)号", text))))
                if any(j not in alive for j in dead):
                    raise ValueError("public death announcement repeats a dead seat")
                emit("death_announcement", dead_players=[f"player{j}" for j in dead])
                alive.difference_update(dead)
            else:
                raise ValueError("unrecognized public death announcement")
            change(c["round"], "discussion")
        elif kind == "speech":
            round_day, round_stage = map(int, c["day"].split("-"))
            if round_day != day:
                raise ValueError("speech day differs from public phase")
            if round_stage > 1 and phase in {"vote", "pk_vote"}:
                change(day, "pk_discussion")
            latest_speech_round = c["day"]
            speaker = f"player{c['player']}"
            emit("turn_start", speaker=speaker)
            if speech_pre:
                if not problems and c["player"] not in alive:
                    raise ValueError("a dead player cannot have a speech PRE")
                boundaries.append({**snapshot(index, f"{game_id}/speech-pre-{index}"),
                                   "source_sha256": sha256_bytes(payload)})
            event_id = emit("public_speech", speaker=speaker, raw_text=c["context"])
            key = path.rsplit("/", 1)[0] + f"/{index}"
            annotation = annotations[key]
            if annotation["text_sha256"] != sha256_bytes(c["context"].encode()):
                raise ValueError("speech annotation text identity mismatch")
            triples = SpeechPerceiver._normalize(parsed=annotation["actions"], speaker=c["player"])
            actions[event_id] = [V1SpeechAction(*triple) for triple in triples]
        elif kind == "vote_start":
            change(day, "pk_vote" if phase == "pk_discussion" else "vote")
            boundary_id = f"{game_id}/vote-start-{index}"
            if not speech_pre:
                current_boundary = snapshot(index, boundary_id)
                boundaries.append(current_boundary)
        elif kind == "voted" and not speech_pre:
            if current_boundary is None or c["day"] != latest_speech_round:
                raise ValueError("vote has no matching current public boundary")
            converted = convert_label(c, quarantined=(split, game_number, index) in CONFLICTS)
            actor = c["player"]
            nonwolf = roles[actor] != "werewolf"
            if not problems and actor not in alive:
                raise ValueError("a dead observer submitted a vote label")
            eligible = nonwolf and not problems
            reasons = list(converted["exclusion_reasons"])
            if not nonwolf:
                reasons.append("wolf_observer")
            reasons.extend(p["reason"] for p in problems)
            labels.append({"game_id": game_id, "split": split,
                "boundary_id": current_boundary["boundary_id"],
                "source_index": index, "source_sha256": sha256_bytes(payload),
                "observer_id": f"player{actor}", "day": c["day"],
                **converted, "non_wolf_observer": nonwolf,
                "quarantine_reason": CONFLICTS.get((split, game_number, index)),
                "primary_eligible": eligible,
                "effective": eligible and converted["label_usable"],
                "exclusion_reasons": list(dict.fromkeys(reasons))})
        elif kind == "vote_results":
            votes = [{"voter": f"player{k}", "target": None if v is None else f"player{v}"}
                     for k, v in sorted(c.items(), key=lambda pair: int(pair[0]))]
            emit("vote_result", votes=votes)
            counts = Counter(v for v in c.values() if v is not None)
            leaders = [j for j, n in counts.items() if n == max(counts.values())] if counts else []
            exiled = leaders if len(leaders) == 1 else []
            emit("exile_result", exiled_players=[f"player{j}" for j in sorted(exiled)])
            alive.difference_update(exiled)
            current_boundary = None
        # roles只供外围Mask；所有私人夜间行动、notes、summary、review、winner均无输入路径。
    return {"game_id": game_id, "split": split, "source_path": path,
            "source_sha256": sha256_bytes(payload), "public_events": public,
            "speech_actions": {key: [a.to_record() for a in value] for key, value in actions.items()}}, boundaries, labels


def speech_pre_snapshots(path, payload, annotations):
    """只在当前发言写入前截取；保留缺失公共公告的边界及其排除原因。"""
    _, boundaries, _ = extract_game(path, payload, annotations, speech_pre=True)
    return boundaries


def role_belief_view(payload, boundary, observer_id):
    """原始事件先投影成单人合法视图；生成器永远不接收原始全局记录。"""
    observer = normalize_player(observer_id)
    seat = PLAYER_IDS.index(observer)
    if boundary["input_problems"] or boundary["alive"] is None:
        raise ValueError("speech PRE lacks complete public history")
    if not boundary["alive"][seat]:
        raise ValueError("belief observer must be alive")
    if boundary["source_sha256"] != sha256_bytes(payload):
        raise ValueError("speech PRE source digest mismatch")
    raw = json.loads(payload)
    cutoff = boundary["source_index"]
    current = raw[cutoff]
    last = boundary["public_events"][-1]
    if (current["event"] != "speech" or last["event_type"] != "turn_start"
            or normalize_player(current["content"]["player"]) != last["speaker"]):
        raise ValueError("speech PRE source mismatch")
    # 只读取本人身份卡；其他角色只由调用方用于外围 Primary 人口选择。
    own_roles = [e["content"]["role"] for e in raw[:cutoff]
                 if e["event"] == "roles" and e["content"]["player"] == seat + 1]
    if len(own_roles) != 1 or own_roles[0] not in ROLES - {"werewolf"}:
        raise ValueError("belief observer must have one non-Werewolf role card")
    role = own_roles[0]
    facts, omitted = [], []
    known_wolves, known_non_wolves = set(), {observer}
    medicines = {"healed": "available", "poison": "available"} if role == "witch" else {}
    night, events = None, []

    def completed_night(number):
        # 官方 process_script/utils.py 的 night_event/night_action 按角色路由这些事件。
        # 只采用已结束夜晚、显式夜次和唯一动作；缺失不等同于显式 null。
        if role == "simple_villager":
            return
        relevant = ("inquired",) if role == "seer" else ("healed", "poison", "werewolf_kill")
        selected = {}
        for kind in relevant:
            rows = [(i, c) for i, k, c in events if k == kind]
            if (len(rows) != 1 or not isinstance(rows[0][1], dict)
                    or type(rows[0][1].get("night")) is not int or rows[0][1]["night"] != number):
                omitted.append({"night": number, "event": kind, "reason": "missing_or_ambiguous"})
                continue
            i, c = rows[0]
            field = "target_player" if kind == "werewolf_kill" else "player"
            if field not in c:
                omitted.append({"night": number, "event": kind, "reason": "missing_target"})
                continue
            target = c[field]
            if target is not None and (type(target) is not int or target not in range(1, 8)):
                omitted.append({"night": number, "event": kind, "reason": "invalid_target"})
                continue
            if kind == "inquired" and (target is None or type(c.get("is_werewolf")) is not bool):
                omitted.append({"night": number, "event": kind, "reason": "missing_check_result"})
                continue
            selected[kind] = {"source_index": i, "night": number, "event": kind,
                              "target": None if target is None else normalize_player(target)}
            if kind == "inquired":
                selected[kind]["is_werewolf"] = c["is_werewolf"]
                (known_wolves if c["is_werewolf"] else known_non_wolves).add(normalize_player(target))
        if role == "seer":
            facts.extend(selected.values())
            return
        for kind in medicines:
            if kind not in selected:
                if medicines[kind] != "used":
                    medicines[kind] = "unknown"
            elif selected[kind]["target"] is not None:
                medicines[kind] = "used"
            if kind in selected:
                facts.append(selected[kind])
        # 官方女巫输入只在存在本人技能记录时提供当夜刀口；银水不是好人硬知识。
        if any(k in selected for k in medicines) and "werewolf_kill" in selected:
            facts.append({**selected["werewolf_kill"], "event": "privately_observed_attack_target"})

    for i, event in enumerate(raw[:cutoff]):
        kind, c = event["event"], event.get("content")
        if kind == "cycle_round":
            if c["status"] == "night":
                night, events = c["round"], []
            elif c["status"] == "day":
                if night == c["round"]:
                    completed_night(night)
                elif role != "simple_villager":
                    omitted.append({"night": c["round"], "reason": "unconfirmed_private_timing"})
                    medicines = {k: "used" if v == "used" else "unknown" for k, v in medicines.items()}
                night, events = None, []
        elif night is not None and kind in {"inquired", "healed", "poison", "werewolf_kill"}:
            events.append((i, kind, c))
    wolves, non_wolves = close_hard_knowledge(sorted(known_wolves), sorted(known_non_wolves))
    return {"game_id": boundary["game_id"], "boundary_id": boundary["boundary_id"],
            "observer_id": observer, "self_role": role, "day": boundary["day"], "phase": boundary["phase"],
            "current_public_actor": last["speaker"],
            "alive_players": [p for p, alive in zip(PLAYER_IDS, boundary["alive"]) if alive],
            "public_events": deepcopy(boundary["public_events"]),
            "public_history_digest": boundary["prefix_digest"], "private_facts": facts,
            "witch_medicines": medicines, "omitted_private_information": omitted,
            "known_werewolves": wolves, "known_non_werewolves": non_wolves}


def build_role_belief_request(view):
    """每个三元组重新构造消息；无 Agent、会话、私人记忆或模型调用。"""
    observer = view["observer_id"]
    private = {k: deepcopy(view[k]) for k in (
        "observer_id", "self_role", "day", "phase", "current_public_actor", "alive_players", "witch_medicines")}
    # 模型只看到玩家事实；位置、缺失原因等审计字段继续留在内部 view。
    private["private_facts"] = [{k: deepcopy(fact[k]) for k in (
        "event", "night", "target", "is_werewolf") if k in fact} for fact in view["private_facts"]]
    public = [{k: deepcopy(event[k]) for k in (
        "event_type", "day", "phase", "speaker", "raw_text", "dead_players", "votes", "exiled_players")
        if k in event} for event in view["public_events"]]
    for event in public:
        if "votes" in event:
            event["votes"] = [{k: vote[k] for k in ("voter", "target")} for vote in event["votes"]]
    prompt = PlayingAgentBeliefReporter.render_prompt(
        observer_id=observer, public_events=public,
        public_history_digest=view["public_history_digest"],
        known_werewolves=view["known_werewolves"], known_non_werewolves=view["known_non_werewolves"],
        prompt_version=ROLE_BELIEF_PROMPT,
        include_audit_metadata=False,
        report_context="这是独立重建请求；没有此前对话或私人记忆，回答不会用于公开发言或回写游戏。")
    legal = PlayingAgentBeliefReporter.legal_candidates(
        observer_id=observer, known_werewolves=view["known_werewolves"],
        known_non_werewolves=view["known_non_werewolves"])
    messages = [{"role": "system", "content": (
        "你是这局七人狼人杀中的当前观察者。阵容为两狼、三村民、一预言家、一女巫。"
        "请以第一人称代入本人身份，报告此刻的主观狼人怀疑，不是识别真实全局身份或制定发言策略。"
        "公开发言中的‘我’属于该条发言者，不代表当前观察者；其中的角色声明可能是谎言。"
        "缺失或时点不确定的私人信息保持未知；null 仅表示该条记录明确没有目标。"
        "不推测缺失技能或刀口，女巫救过的玩家不因此成为确定好人。")},
        {"role": "user", "content": "你本人合法可见的状态：\n" + json.dumps(private, ensure_ascii=False, sort_keys=True)
         + "\n\n" + prompt}]
    return {"game_id": view["game_id"], "boundary_id": view["boundary_id"], "observer_id": observer,
            "prompt_version": ROLE_BELIEF_PROMPT,
            "messages": messages, "temperature": 0.0, "max_tokens": PRIVATE_BELIEF_MAX_TOKENS,
            "response_format": private_belief_response_format(supports_json_schema=True,
                legal_candidates=legal, required_candidates=view["known_werewolves"])}


def parse_role_belief_response(raw_response, view):
    """严格沿用私人集合协议；失败保持未监督，绝不转成成功均匀标签。"""
    support, error = None, None
    try:
        support = validate_player_suspicion(PlayingAgentBeliefReporter.parse_response(raw_response),
            view["known_werewolves"], view["known_non_werewolves"], observer_id=view["observer_id"])
    except (ValueError, TypeError) as exc:
        error = str(exc)
    q, observed = belief_target(PLAYER_IDS.index(view["observer_id"]),
                                tuple(support) if error is None else (), "success" if error is None else "error")
    return {"game_id": view["game_id"], "boundary_id": view["boundary_id"],
            "observer_id": view["observer_id"], "status": "success" if observed else "error",
            "suspected_werewolves": support if observed else None,
            "q": q.tolist(), "observed": observed, "error": error}


def role_trial_inputs(data, expected_digest):
    """只读取固定三个 Training 游戏；原投票标签、Test 与 Baseline 均不打开。"""
    envelope = open_artifact_envelope(data, expected_artifact_type="makto_proxy_belief", expected_schema_version=VERSION)
    if envelope.manifest_digest != expected_digest or envelope.manifest["revision"] != REVISION:
        raise ValueError("MaKTO trial data identity mismatch")
    annotations = {}
    for line in read_artifact_file(envelope, "annotations.jsonl").splitlines():
        row = json.loads(line)
        annotations[row.pop("key")] = row
    games, inputs = {}, []
    for game, index, observer in ROLE_BELIEF_TRIAL_REQUESTS:
        if game not in games:
            path = f"raw/train/7_player_game/seer_witch/game_{game}/event_zh.json"
            payload = read_artifact_file(envelope, path)
            games[game] = payload, {b["source_index"]: b for b in speech_pre_snapshots(path, payload, annotations)}
        payload, boundaries = games[game]
        view = role_belief_view(payload, boundaries[index], observer)
        inputs.append({"view": view, "request": build_role_belief_request(view)})
    if sha256_bytes(canonical_json_bytes([i["request"] for i in inputs])) != ROLE_BELIEF_TRIAL_DIGEST:
        raise ValueError("approved twelve requests changed")
    return inputs


def check_role_trial_inputs(inputs, run):
    """实际环境的 CPU Tokenizer/模板与 xgrammar 校验；不加载模型、不调用服务。"""
    from importlib.metadata import version
    from jinja2 import Environment, StrictUndefined
    from werewolf.tom_llm.model_spec import load_cpu_tokenizer, template_bytes

    # CPU 准备沿用服务已登记缓存位置，并显式隐藏 GPU；不会初始化模型。
    os.environ.update(run.record["runtime"]["environment"])
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from vllm.sampling_params import SamplingParams, StructuredOutputsParams
    from vllm.v1.structured_output.backend_xgrammar import validate_xgrammar_grammar

    versions = {name: version(name) for name in ("vllm", "xgrammar", "transformers")}
    if versions["vllm"] != "0.27.0":
        raise ValueError("existing vLLM 0.27.0 environment required")
    if Path(sys.executable).resolve() != Path(run.record["runtime"]["environment_prefix"], "bin/python").resolve():
        raise ValueError("use the registered service environment Python")
    row = run.spec.record()
    tokenizer = load_cpu_tokenizer(run.spec, run.record["tokenizer_assets"])
    env = Environment(undefined=StrictUndefined, autoescape=False)
    def refuse(message):
        raise ValueError(message)
    env.globals["raise_exception"] = refuse
    template = env.from_string(template_bytes(run.spec).decode())
    counts, ids_digests, checked_schemas = [], [], set()
    for item in inputs:
        request = item["request"]
        kwargs = {"add_generation_prompt": True, **row["chat_template"]["kwargs"]}
        text = template.render(messages=request["messages"], bos_token=row["tokenizer"]["bos_token"],
                               eos_token=row["tokenizer"]["eos_token"], **kwargs)
        native = tokenizer.apply_chat_template(request["messages"], tokenize=False, **kwargs)
        ids = tokenizer.apply_chat_template(request["messages"], tokenize=True, return_dict=False, **kwargs)
        if text != native or ids != tokenizer.encode(text, add_special_tokens=False):
            raise ValueError("MaKTO native template/token IDs mismatch")
        if not 0 < len(ids) <= min(7680, row["context_length"] - PRIVATE_BELIEF_MAX_TOKENS):
            raise ValueError("complete trial prompt exceeds token budget; no truncation")
        counts.append(len(ids)); ids_digests.append(sha256_bytes(canonical_json_bytes(ids)))
        schema = request["response_format"]["json_schema"]["schema"]
        schema_digest = sha256_bytes(canonical_json_bytes(schema))
        if schema_digest not in checked_schemas:
            validate_xgrammar_grammar(SamplingParams(temperature=0, max_tokens=PRIVATE_BELIEF_MAX_TOKENS,
                structured_outputs=StructuredOutputsParams(json=deepcopy(schema))))
            checked_schemas.add(schema_digest)
    return {"versions": versions, "input_token_counts": counts, "input_ids_digests": ids_digests,
            "maximum_input_tokens": max(counts), "schema_digests": sorted(checked_schemas)}


def role_trial_reply(raw, view, model, input_tokens):
    """保留完整信封；拒答、截断、身份或 Token 数不符均不能成为监督。"""
    from werewolf.tom_llm.common import strict_json
    try:
        body = strict_json(raw)
        if body["model"] != model or len(body["choices"]) != 1:
            raise ValueError("teacher model/choices mismatch")
        choice = body["choices"][0]
        if choice["finish_reason"] != "stop" or choice["message"].get("refusal") is not None:
            raise ValueError("teacher refusal or incomplete generation")
        usage = body["usage"]
        if (type(usage["prompt_tokens"]) is not int or usage["prompt_tokens"] != input_tokens
                or type(usage["completion_tokens"]) is not int or not 0 <= usage["completion_tokens"] <= 96
                or type(usage["total_tokens"]) is not int
                or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]):
            raise ValueError("actual teacher token usage differs from preparation")
        content = choice["message"]["content"]
        result = parse_role_belief_response(content, view)
        try:
            PlayingAgentBeliefReporter.parse_response(content)
        except (ValueError, TypeError):
            compliant = None
        else:
            compliant = result["observed"]
        return {**result, "hard_knowledge_compliant": compliant}
    except (ValueError, TypeError, KeyError, IndexError) as error:
        return {**parse_role_belief_response(None, view), "error": str(error), "hard_knowledge_compliant": None}


def role_trial_summary(inputs, results):
    """只汇总质量试验与同人跨 PRE 变化，不计算 KL 或模型性能。"""
    rows = []
    for ordinal, item in enumerate(inputs):
        view = item["view"]
        rows.append({"ordinal": ordinal, "game_id": view["game_id"], "boundary_id": view["boundary_id"],
            "observer_id": view["observer_id"], "role": view["self_role"],
            **results.get(ordinal, {"status": "not_attempted", "observed": False, "suspected_werewolves": None})})
    valid = [row for row in rows if row["observed"]]
    previous, changes = {}, []
    for row in valid:
        key = row["game_id"], row["observer_id"]
        if key in previous:
            before = previous[key]
            changes.append({"game_id": key[0], "observer_id": key[1], "from": before["boundary_id"],
                "to": row["boundary_id"], "before": before["suspected_werewolves"],
                "after": row["suspected_werewolves"],
                "added": sorted(set(row["suspected_werewolves"]) - set(before["suspected_werewolves"])),
                "removed": sorted(set(before["suspected_werewolves"]) - set(row["suspected_werewolves"]))})
        previous[key] = row
    return {"request_count": len(rows), "valid_nonempty": sum(bool(r["suspected_werewolves"]) for r in valid),
            "valid_empty": sum(not r["suspected_werewolves"] for r in valid),
            "failed": sum(not r["observed"] and r["status"] != "not_attempted" for r in rows),
            "not_attempted": sum(r["status"] == "not_attempted" for r in rows),
            "hard_knowledge_compliant": len(valid),
            "hard_knowledge_violations": sum(r.get("hard_knowledge_compliant") is False for r in rows),
            "results": rows, "pre_changes": changes}


def role_trial(data, output, service_run, expected_digest, *, execute_live=False):
    """固定12条的一次性入口；共用既有服务和 HTTP，不创建 Baseline Run 或 Seal。"""
    import fcntl
    from werewolf.tom_llm.common import strict_json
    from werewolf.tom_llm.experiment import open_run
    from werewolf.tom_llm.live_runner import HttpTransport, run_lock
    from werewolf.tom_llm.service_manager import Service
    from werewolf.tom_llm.model_spec import ROOT
    from werewolf.tom_llm.deployment import profile
    from werewolf.tom_llm.provider_semantics import NotExecuted, OutcomeUnknown

    run = open_run(service_run, with_tasks=False)
    if run.key != "qwen3" or run.record["mode"] != "formal" or run.spec.record()["revision"] != "b968826d9c46dd6066d109eabc6255188de91218":
        raise ValueError("registered Qwen3-8B service resource contract required")
    runtime = run.record["runtime"]
    if runtime != profile(run.spec, root=run.root, model_directory=runtime["argv_options"]["model"],
                          environment_prefix=runtime["environment_prefix"], gpu_uuid=runtime["gpu_uuid"]):
        raise ValueError("registered service runtime settings changed")
    if runtime["argv_options"]["structured-outputs-config"] != {"backend": "xgrammar"}:
        raise ValueError("existing xgrammar service required")
    inputs = role_trial_inputs(data, expected_digest)
    output = Path(output).absolute()
    if any((parent / "manifest.json").exists() for parent in output.parents):
        raise ValueError("trial output cannot be inside an existing artifact")
    source = {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "files": {name: sha256_bytes((ROOT / name).read_bytes()) for name in (
                  "werewolf/tom/makto.py", "werewolf/speech/private_belief_perceiver.py",
                  "werewolf/tom_llm/live_runner.py", "werewolf/tom_llm/model_spec.py",
                  "werewolf/tom_llm/service_manager.py")}}
    binding = {"dataset_digest": expected_digest, "service_run": str(run.root),
               "service_run_digest": run.artifact.manifest_digest, "source": source,
               "model": run.spec.record()["model"], "revision": run.spec.record()["revision"],
               "model_spec_digest": run.spec.digest, "request_sequence_digest": ROLE_BELIEF_TRIAL_DIGEST}

    def append(stream, record):
        stream.write(canonical_jsonl_bytes([record])); stream.flush(); os.fsync(stream.fileno())

    if not execute_live:
        checks = check_role_trial_inputs(inputs, run)
        with output.open("xb") as stream:
            append(stream, {"event": "prepared", **binding, "checks": checks, "inputs": inputs})
            from werewolf.artifact_io.canonical import _fsync_directory
            _fsync_directory(output.parent)
        return {"status": "PREPARED_NOT_EXECUTED", "output": str(output), **checks}
    if sys.platform != "linux":
        raise ValueError("real trial requires explicit --execute-live on Ubuntu")
    results = {}
    with output.open("r+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = [strict_json(line) for line in stream.read().splitlines()]
        if len(records) != 1 or records[0].get("event") != "prepared":
            raise ValueError("trial already attempted or incomplete; no automatic replay/resend")
        prepared = records[0]
        if any(prepared.get(k) != value for k, value in binding.items()) or prepared["inputs"] != inputs:
            raise ValueError("trial preparation/source/request binding changed")
        stream.seek(0, 2)
        with run_lock(run.root):
            manager = Service(run)
            already_running = manager.status()["state"] == "running"
            manager.start(execute=True)
            transport = HttpTransport("http://127.0.0.1:" + str(runtime["argv_options"]["port"]), alive_check=manager.guard)
            try:
                row = run.spec.record()
                for ordinal, item in enumerate(inputs):
                    request, view = item["request"], item["view"]
                    wire = {k: deepcopy(request[k]) for k in ("messages", "temperature", "max_tokens", "response_format")}
                    wire.update(model=row["model"], **{k: row["decoding"][k] for k in ("top_p", "n", "seed")},
                        extra_body={"chat_template_kwargs": row["chat_template"]["kwargs"],
                                    "top_k": row["decoding"]["top_k"], "min_p": row["decoding"]["min_p"]})
                    task_id = sha256_bytes(canonical_json_bytes(wire))
                    append(stream, {"event": "dispatch", "ordinal": ordinal, "request_digest": task_id})
                    start = time.monotonic()
                    try:
                        raw = transport.send(wire, task_id)
                    except (NotExecuted, OutcomeUnknown) as error:
                        result = {**parse_role_belief_response(None, view),
                                  "status": "unknown" if isinstance(error, OutcomeUnknown) else "not_executed", "error": str(error)}
                        body = getattr(error, "response_body", None)
                        append(stream, {"event": "transport_error", "ordinal": ordinal,
                            "http_status": getattr(error, "http_status", None),
                            "raw_response": body.decode("utf-8", errors="replace") if body is not None else None,
                            "raw_response_hex": body.hex() if body is not None else None, **result})
                        results[ordinal] = result
                        break
                    append(stream, {"event": "received", "ordinal": ordinal, "raw_response": raw,
                                    "elapsed_seconds": time.monotonic() - start})
                    result = role_trial_reply(raw, view, row["model"], prepared["checks"]["input_token_counts"][ordinal])
                    results[ordinal] = result
                    append(stream, {"event": "result", "ordinal": ordinal, **result})
            finally:
                if not already_running:
                    manager.stop(execute=True)
        summary = role_trial_summary(inputs, results)
        append(stream, {"event": "summary", **summary})
    return summary


def prepare(sources, annotations):
    """一次准备，两组固定规则；原始文件与候选/排除审计全部保留。"""
    if set(sources) != set(source_paths()):
        raise ValueError("exactly the fixed 20 Train and 8 Test sources are required")
    expected = {row["key"] for row in speech_records(sources)}
    if set(annotations) != expected:
        raise ValueError("exact public speech annotation coverage is required")
    games, inputs, labels = [], [], []
    for path in source_paths():
        game, boundaries, rows = extract_game(path, sources[path], annotations)
        games.append(game)
        inputs.extend(boundaries)
        labels.extend(rows)
    report = {}
    for split in SPLITS:
        group = [g for g in games if g["split"] == split]
        rows = [row for row in labels if row["split"] == split]
        primary = [row for row in rows if row["non_wolf_observer"]]
        boundary_rows = [row for row in inputs if row["split"] == split]
        report[split] = {
            "games": len(group), "candidate_vote_rows": len(rows),
            "non_wolf_candidate_rows": len(primary),
            "label_usable_rows": sum(row["label_usable"] for row in primary),
            "quarantined_rows": sum(row["quarantined"] for row in primary),
            "effective_rows": sum(row["effective"] for row in rows),
            "effective_games": len({row["game_id"] for row in rows if row["effective"]}),
            "vote_boundaries": len(boundary_rows),
            "usable_input_boundaries": sum(row["plan"] is not None for row in boundary_rows),
            "public_speeches": sum(len(g["speech_actions"]) for g in group),
            "public_events": sum(len(g["public_events"]) for g in group),
            "speech_actions": sum(len(a) for g in group for a in g["speech_actions"].values()),
            "max_input_tokens": max((row["plan"]["token_count"] for row in boundary_rows if row["plan"]), default=0),
            "exclusion_counts_nonexclusive": dict(Counter(reason for row in primary for reason in row["exclusion_reasons"])),
            "label_exclusion_counts_exclusive": dict(Counter(
                row["exclusion_reasons"][0] for row in primary if not row["label_usable"])),
            "per_game": [{
                "game_id": game["game_id"],
                "candidate_vote_rows": sum(row["game_id"] == game["game_id"] for row in rows),
                "effective_rows": sum(row["game_id"] == game["game_id"] and row["effective"] for row in rows),
            } for game in group],
        }
    if any(row["plan"] and row["plan"]["token_count"] > 1024 for row in inputs):
        raise ValueError("complete public input exceeds model capacity; truncation forbidden")
    return games, inputs, labels, report


def publish(destination, sources, annotations, *, previous=None, corrections=()):
    games, inputs, labels, report = prepare(sources, annotations)
    revision_fields, revision_files = {}, {}
    if previous is not None:
        parent = open_artifact_envelope(previous, expected_artifact_type="makto_proxy_belief",
                                        expected_schema_version=VERSION)
        if parent.manifest_digest != PREPARED_DIGEST:
            raise ValueError("annotation correction parent identity mismatch")
        if parent.manifest["source_digests"] != {p: sha256_bytes(b) for p, b in sources.items()}:
            raise ValueError("annotation correction cannot change original sources")
        old = {r["key"]: r for r in map(json.loads, read_artifact_file(parent, "annotations.jsonl").splitlines())}
        changed = {k for k in annotations if old[k]["actions"] != annotations[k]["actions"]}
        if len({c["key"] for c in corrections}) != len(corrections) or changed != {c["key"] for c in corrections}:
            raise ValueError("correction log must cover exactly the changed annotations")
        for c in corrections:
            if (c["before"] != old[c["key"]]["actions"] or c["after"] != annotations[c["key"]]["actions"]
                    or c["text_sha256"] != old[c["key"]]["text_sha256"] or not c["reason"] or not c["evidence"]):
                raise ValueError("correction evidence identity mismatch")
        for split in SPLITS:
            if canonical_jsonl_bytes(r for r in labels if r["split"] == split) != read_artifact_file(parent, f"supervision/{split}.jsonl"):
                raise ValueError("annotation correction cannot change supervision")
            old_inputs = [json.loads(line) for line in read_artifact_file(parent, f"public/{split}.jsonl").splitlines()]
            new_inputs = [r for r in inputs if r["split"] == split]
            if ([{k: v for k, v in r.items() if k != "plan"} for r in old_inputs]
                    != [{k: v for k, v in r.items() if k != "plan"} for r in new_inputs]):
                raise ValueError("annotation correction cannot change public boundaries")
        revision_fields = {"annotation_correction": {
            "parent_digest": parent.manifest_digest, "changed_speeches": len(corrections),
            "scope": "targeted_V1_public_speech_semantics_not_full_independent_review",
            "rules_selected_on": "original_train_only", "official_test_rule_selection": False,
            "supervision_and_boundaries": "unchanged",
            "v1_parser_source_digest": sha256_bytes(Path(__file__).parents[1].joinpath("speech/speech_perceiver.py").read_bytes()),
        }}
        revision_files = {"corrections.jsonl": canonical_jsonl_bytes(sorted(corrections, key=lambda c: c["key"]))}
    elif corrections:
        raise ValueError("correction log requires the original artifact")
    package = Path(__file__).parents[1]
    return publish_artifact(destination, manifest_fields={
        "artifact_type": "makto_proxy_belief", "schema_version": VERSION,
        "dataset": DATASET, "revision": REVISION, "license": "MIT (dataset card)",
        "target": "human_expert_role_judgment_proxy_not_verified_private_belief",
        "label_rule": "six_complete_no_NA_no_wolf_nonwolf_mix_nonempty_support_uniform",
        "boundary": "before_current_vote_results_at_vote_start",
        "annotation_origin": "Codex_public_text_V1_annotation_not_independently_reviewed",
        "speech_ontology": V1_SPEECH_ONTOLOGY_VERSION,
        "token_planner": STRUCTURED_TOKEN_PLANNER_VERSION,
        "label_timing": "UNKNOWN", "report": report,
        "implementation_digests": {name: sha256_bytes((package / name).read_bytes())
                                   for name in ("tom/makto.py", "tom/dataset.py", "structured_history.py")},
        "source_digests": {p: sha256_bytes(b) for p, b in sources.items()},
        **revision_fields,
    }, files={
        **sources,
        **revision_files,
        "public/games.jsonl": canonical_jsonl_bytes(games),
        **{f"public/{split}.jsonl": canonical_jsonl_bytes(row for row in inputs if row["split"] == split)
           for split in SPLITS},
        **{f"supervision/{split}.jsonl": canonical_jsonl_bytes(row for row in labels if row["split"] == split)
           for split in SPLITS},
        "annotations.jsonl": canonical_jsonl_bytes({"key": k, **v} for k, v in sorted(annotations.items())),
        "report.json": canonical_json_bytes(report),
    })


class MaKTOToMDataset:
    """只打开指定划分的公开输入与标签；不读取原始角色和私人事件。"""

    def __init__(self, path, split, *, capacity=ExperimentCapacity(1024)):
        if split not in SPLITS:
            raise ValueError("split must be train or test")
        envelope = open_artifact_envelope(path, expected_artifact_type="makto_proxy_belief", expected_schema_version=VERSION)
        if envelope.manifest["revision"] != REVISION:
            raise ValueError("MaKTO revision mismatch")
        inputs = [json.loads(line) for line in read_artifact_file(envelope, f"public/{split}.jsonl").splitlines()]
        labels = [json.loads(line) for line in read_artifact_file(envelope, f"supervision/{split}.jsonl").splitlines()]
        self.samples, self.primary_masks = [], {}
        self.split = split
        self.source_digest = envelope.manifest_digest
        self.annotation_correction = envelope.manifest.get("annotation_correction")
        boundaries = set()
        for row in inputs:
            if row["split"] != split:
                raise ValueError("input split identity mismatch")
            if row["boundary_id"] in boundaries:
                raise ValueError("duplicate vote boundary")
            boundaries.add(row["boundary_id"])
            if row["plan"] is None:
                continue
            value = row["plan"]
            if (value["prefix_digest"] != row["prefix_digest"]
                    or value["token_count"] != len(value["tokens"])
                    or sha256_bytes(canonical_json_bytes({k: v for k, v in value.items()
                                                         if k != "plan_digest"})) != value["plan_digest"]):
                raise ValueError("structured plan identity mismatch")
            plan = StructuredTokenPlan(value["planner_version"], value["prefix_digest"],
                tuple(StructuredTokenDescriptor(**token) for token in value["tokens"]),
                value["token_count"], value["plan_digest"])
            q, observed, primary = torch.zeros(7, 7), torch.zeros(7, dtype=torch.bool), torch.zeros(7, dtype=torch.bool)
            seats = set()
            for label in labels:
                if label["boundary_id"] != row["boundary_id"]:
                    continue
                if label["split"] != split or label["game_id"] != row["game_id"]:
                    raise ValueError("label identity mismatch")
                seat = PLAYER_IDS.index(label["observer_id"])
                if seat in seats:
                    raise ValueError("duplicate observer at vote boundary")
                seats.add(seat)
                expected_q, _ = belief_target(seat, tuple(label["wolf_support"]) if label["label_usable"] else (),
                                              "success" if label["label_usable"] else None)
                if label["q"] != expected_q.tolist():
                    raise ValueError("proxy target does not match conservative label conversion")
                q[seat] = expected_q
                observed[seat], primary[seat] = label["label_usable"], label["primary_eligible"]
            self.primary_masks[row["boundary_id"]] = primary
            self.samples.append(CanonicalToMSample(row["game_id"], row["boundary_id"],
                row["prefix_digest"], plan.plan_digest, plan.token_count,
                tensorize_structured_plan(plan, capacity), q, observed, torch.tensor(row["alive"])))
        self.effective_game_ids = tuple(sorted({sample.game_id for sample in self.samples
                                               if self.effective_mask(sample).any()}))

    def effective_mask(self, sample):
        """部分监督只作用于非狼人、存活且标签通过规则的观察者。"""
        return self.primary_masks[sample.boundary_id] & sample.label_observed & sample.observer_alive

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def internal_split(data):
    """仅按Train的监督行数分层，再用固定哈希一次选定整局Validation。"""
    if data.split != "train":
        raise ValueError("internal split must use original Train only")
    counts = Counter()
    for sample in data.samples:
        counts[sample.game_id] += int(data.effective_mask(sample).sum())
    strata = {
        "one_or_two": ([g for g, n in counts.items() if 1 <= n <= 2], 2),
        "three": ([g for g, n in counts.items() if n == 3], 1),
        "four_or_more": ([g for g, n in counts.items() if n >= 4], 1),
    }
    validation = []
    for name, (games, take) in strata.items():
        if len(games) < take:
            raise ValueError("insufficient supervised games for the fixed split strata")
        ordered = sorted(games, key=lambda g: sha256_bytes(
            canonical_json_bytes(["makto_internal_holdout", 42, name, g])))
        validation.extend(ordered[:take])
    record = {
        "schema_version": "makto_internal_holdout_v1", "dataset_digest": data.source_digest,
        "seed": 42, "selection": "sha256_within_effective_row_count_strata_2_1_1",
        "training_game_ids": sorted(g for g, n in counts.items() if n and g not in validation),
        "validation_game_ids": sorted(validation),
        "unsupervised_game_ids": sorted(g for g, n in counts.items() if not n),
        "effective_rows_by_game": dict(sorted(counts.items())),
        "official_test_used": False,
    }
    return {**record, "split_digest": sha256_bytes(canonical_json_bytes(record))}


def validation_scores(data, split, model=None):
    """同一Mask逐行KL、局内平均后游戏等权；不更新模型或优化器。"""
    from werewolf.tom.scoring import row_kl

    if split != internal_split(data):
        raise ValueError("internal split identity mismatch")
    totals = {g: {"rows": 0, "uniform": 0., "model": 0.}
              for g in split["validation_game_ids"]}
    identity = []
    was_training = model.training if model is not None else None
    try:
        if model is not None:
            model.eval()
        with torch.inference_mode():
            for sample in data.samples:
                if sample.game_id not in totals:
                    continue
                mask = data.effective_mask(sample)
                if not mask.any():
                    continue
                q = sample.q.double()
                if not torch.allclose(q.sum(-1)[mask], torch.ones_like(q.sum(-1)[mask])):
                    raise ValueError("effective proxy target must sum to one")
                log_uniform = torch.full((7, 7), -math.log(6), dtype=torch.float64)
                log_uniform.fill_diagonal_(-torch.inf)
                result = totals[sample.game_id]
                result["rows"] += int(mask.sum())
                result["uniform"] += float(row_kl(log_uniform, q)[mask].sum())
                if model is not None:
                    public = {name: value[:sample.token_count].unsqueeze(0)
                              for name, value in sample.public.kwargs().items()}
                    logp = model(**public).squeeze(0).double()
                    result["model"] += float(row_kl(logp, q)[mask].sum())
                identity.extend({"game_id": sample.game_id, "boundary_id": sample.boundary_id,
                                 "prefix_digest": sample.prefix_digest, "plan_digest": sample.plan_digest,
                                 "observer_id": PLAYER_IDS[i]}
                                for i in mask.nonzero().flatten().tolist())
    finally:
        if model is not None:
            model.train(was_training)
    games = [{"game_id": g, "effective_rows": v["rows"],
              "uniform_kl": v["uniform"] / v["rows"],
              "epoch35_kl": v["model"] / v["rows"] if model is not None else None}
             for g, v in totals.items()]
    uniform = math.fsum(g["uniform_kl"] for g in games) / len(games)
    original = math.fsum(g["epoch35_kl"] for g in games) / len(games) if model is not None else None
    difference = original - uniform if original is not None else None
    return {"aggregation": "game_macro", "direction": "KL(q||p)", "unit": "nats",
            "row_identity_digest": sha256_bytes(canonical_json_bytes(sorted(
                identity, key=lambda r: (r["game_id"], r["boundary_id"], r["observer_id"])))),
            "games": games, "uniform_kl": uniform, "epoch35_kl": original,
            "absolute_difference_epoch35_minus_uniform": difference,
            "relative_difference": difference / uniform if difference is not None and uniform else None}


def load_epoch35_model(checkpoint_path, temporal_path):
    """只读原模型容器及绑定的时间表；不读取或恢复旧优化器、RNG和AI数据。"""
    from werewolf.artifact_io.tensor_state import decode_tensor_container
    from werewolf.tom.qwen3_large import Qwen3ToMLarge, graph_identity
    from werewolf.tom.state import restore_model_tensors
    from werewolf.tom.temporal import TemporalCodeProvider, TEMPORAL_VERSION

    checkpoint = open_artifact_envelope(checkpoint_path,
        expected_artifact_type="qwen3_large_training_checkpoint",
        expected_schema_version="classic7_qwen3_large_epoch_training_v1")
    m = checkpoint.manifest
    if (checkpoint.manifest_digest != EPOCH35_DIGEST or m["completed_epoch"] != 35
            or m["selection"]["best_epoch"] != 35 or m["binding"]["graph"] != graph_identity()):
        raise ValueError("Epoch 35 checkpoint digest/epoch/graph mismatch")
    temporal_envelope = open_artifact_envelope(temporal_path,
        expected_artifact_type="deterministic_temporal_code", expected_schema_version=TEMPORAL_VERSION,
        manifest_name="phase_codebook.manifest.json")
    if temporal_envelope.manifest_digest != m["binding"]["temporal_artifact_digest"]:
        raise ValueError("Epoch 35 temporal artifact identity mismatch")
    temporal = TemporalCodeProvider.explicit(temporal_path, temporal_envelope.manifest["parent_digest"])
    with torch.device("cpu"):
        model = Qwen3ToMLarge(temporal,
                             initialization_seed=m["binding"]["protocol"]["initialization_seed"])
    tensors = decode_tensor_container(m["containers"]["model"],
                                     read_artifact_file(checkpoint, "model.bin"), payload_path="model.bin")
    restore_model_tensors(model, tensors)
    return model.eval()


def _write_record(path, record):
    """小型实验记录只写一次；不向已有数据Artifact内部增加文件。"""
    path = Path(path).absolute()
    if any((parent / "manifest.json").exists() for parent in path.parents):
        raise ValueError("record cannot be written inside an immutable artifact")
    payload = canonical_json_bytes(record)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise FileExistsError("different experiment record already exists")
    with path.open("xb") as handle:
        handle.write(payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--role-trial", action="store_true")
    parser.add_argument("--service-run", type=Path)
    parser.add_argument("--execute-live", action="store_true")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--split-record", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--temporal", type=Path)
    parser.add_argument("--expected-data-digest")
    args = parser.parse_args(argv)
    if args.role_trial:
        if (args.baseline or args.annotations or args.split_record or args.checkpoint or args.temporal
                or not args.data or not args.service_run or not args.expected_data_digest):
            parser.error("role trial requires --data/--service-run/--expected-data-digest only")
        result = role_trial(args.data, args.output, args.service_run, args.expected_data_digest,
                            execute_live=args.execute_live)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result.get("failed", 0) or result.get("not_attempted", 0):
            raise SystemExit(2)
        return
    if args.service_run or args.execute_live:
        parser.error("--service-run/--execute-live require --role-trial")
    if args.baseline:
        if args.annotations is not None or args.data is None or args.split_record is None:
            parser.error("baseline requires --data/--split-record and forbids --annotations")
        if bool(args.checkpoint) != bool(args.temporal):
            parser.error("--checkpoint and --temporal must be supplied together")
        if args.output.absolute() == args.split_record.absolute():
            parser.error("split and baseline records must use different paths")
        data = MaKTOToMDataset(args.data, "train")
        if data.source_digest != (args.expected_data_digest or PREPARED_DIGEST):
            raise ValueError("prepared MaKTO artifact digest mismatch")
        split = internal_split(data)
        _write_record(args.split_record, split)
        model = load_epoch35_model(args.checkpoint, args.temporal) if args.checkpoint else None
        result = {"schema_version": "makto_validation_baseline_v1", "dataset_digest": data.source_digest,
                  "split_digest": split["split_digest"], "expected_checkpoint_digest": EPOCH35_DIGEST,
                  "checkpoint_digest": EPOCH35_DIGEST if model is not None else None,
                  "epoch35_status": "PASS" if model is not None else "NOT_RUN",
                  "epoch35_not_run_reason": None if model is not None else "checkpoint_and_temporal_paths_not_provided",
                  "evaluation_code_digest": sha256_bytes(Path(__file__).read_bytes()),
                  "quality_status": ("TARGETED_CORRECTIONS_APPLIED_NOT_FULL_CERTIFICATION"
                                     if getattr(data, "annotation_correction", None) else "ANNOTATION_SEMANTICS_REVIEW_REQUIRED"),
                  "interpretation": "diagnostic_on_existing_frozen_proxy_data",
                  **validation_scores(data, split, model)}
        _write_record(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.annotations is None or any((args.data, args.split_record, args.checkpoint, args.temporal, args.expected_data_digest)):
        parser.error("extraction requires --annotations and forbids baseline-only arguments")
    annotations = {}
    for line in args.annotations.read_bytes().splitlines():
        row = json.loads(line)
        key = row.pop("key")
        if key in annotations:
            raise ValueError("duplicate annotation key")
        annotations[key] = row
    artifact = publish(args.output, download_sources(), annotations)
    print(json.dumps({"path": str(artifact.path), "digest": artifact.manifest_digest,
                      "report": artifact.manifest["report"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
