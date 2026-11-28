from copy import deepcopy
from dataclasses import asdict, replace
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.observer_dtrs.test_speech_runtime import runtime
from tests.tom.test_counterfactual_consumer import P, PublicWitness
from tests.tom.test_verified_speech_commit import snapshot
from werewolf.agents.gpt_agent import GPTAgent, GameplayActionValidationError, GameplaySpeechQualityError
from werewolf.canonical_collection.speech import V1AnnotationStatus, V1SpeechAction
from werewolf.observer_dtrs.probabilities import ProbabilityMatrix
from werewolf.observer_dtrs.speech_runtime import _legal_team
from werewolf.speech.commit_workflow import commit_public_speech


TOPIC = "如何处理当前公开讨论中的证据分歧"


def candidate(function, text, addressee=None, *, premises=(), evidence=()):
    return {"function": function, "text": text, "addressee": addressee,
            "required_premises": list(premises), "public_evidence_refs": list(evidence)}


def mock_agent(rows, *, raw=None, finish_reason="stop"):
    content = json.dumps({"topic": TOPIC, "candidates": rows}) if raw is None else raw
    chat = Mock(return_value=(content, {"finish_reason": finish_reason}))
    backend = SimpleNamespace(supports_json_schema=True, chat_with_metadata=chat)
    return GPTAgent(backend=backend, model_name="fixture"), chat


def request_context(chat):
    prompt = chat.call_args.kwargs["messages"][0]["content"]
    return json.loads(prompt.rsplit("\n\n", 1)[1])


def discussion_pre(alive=P, prior=(P[-1],)):
    witness = PublicWitness(alive=alive)
    del witness.events[3:]
    witness.annotations.clear()
    for seat in prior:
        witness.event("turn_start", speaker=seat)
        witness.speech(seat, "no_commitment", None, "我暂时不作明确表态。")
    witness.event("turn_start", speaker=P[0])
    return witness.pre()


def frozen_q(pre):
    rows = []
    for observer in P:
        weights = [0 if seat == observer else col + 1 for col, seat in enumerate(P)]
        total = sum(weights)
        rows.append(tuple(weight / total for weight in weights))
    return ProbabilityMatrix(pre.game_id, pre.boundary_id, pre.prefix_digest, tuple(rows))


@pytest.mark.parametrize("alive,team,wolves,observers", [
    (P[:6], (P[0], P[3]), (P[0], P[3]), (P[1], P[2], P[4], P[5])),
    ((P[0], *P[2:6]), P[:2], (P[0],), P[2:6]),
])
def test_generation_projects_frozen_q_for_double_and_single_wolves(alive, team, wolves, observers):
    pre = discussion_pre(alive=alive, prior=(P[5],))
    q = frozen_q(pre)
    rows = [candidate("INQUIRE", "3号，你怎么看这段公开发言？", P[2])]
    agent, chat = mock_agent(rows)

    batch = agent.generate_speech_candidates(pre, known_wolf_ids=team, q=q)

    chat.assert_called_once()
    context = request_context(chat)
    assert set(context) == {"speaker", "day", "phase", "alive_players", "known_wolf_team",
                            "public_events", "remaining_speakers", "replyable_observers", "wolf_suspicion_mass"}
    assert context["speaker"] == P[0] and context["phase"] == "discussion" and context["day"] == 1
    assert context["public_events"] == pre.public_event_history.to_records()
    assert context["alive_players"] == list(alive)
    assert context["known_wolf_team"] == list(team)
    assert context["remaining_speakers"] == [seat for seat in alive if seat not in (P[0], P[5])]
    assert context["replyable_observers"] == [seat for seat in observers if seat != P[5]]
    expected_values = [[q.row(seat).values[P.index(wolf)] for wolf in wolves] for seat in observers]
    assert context["wolf_suspicion_mass"] == {
        "observer_ids": list(observers), "wolf_ids": list(wolves), "values": expected_values,
    }
    assert all(sum(row) < 1 for row in expected_values)
    assert batch.pre is pre and batch.topic == TOPIC
    assert batch.candidates[0].text == rows[0]["text"]


@pytest.mark.parametrize("field", [None, "game_id", "boundary_id", "prefix_digest"])
def test_missing_or_mismatched_frozen_q_precedes_backend(field):
    pre = discussion_pre()
    q = None if field is None else replace(frozen_q(pre), **{field: "f" * 64})
    agent, chat = mock_agent([])
    with pytest.raises(ValueError, match="冻结 ProbabilityMatrix"):
        agent.generate_speech_candidates(pre, known_wolf_ids=P[:2], q=q)
    chat.assert_not_called()


def test_current_speaker_must_belong_to_known_wolf_team():
    pre = discussion_pre()
    agent, chat = mock_agent([])
    with pytest.raises(ValueError, match="合法狼人发言 PRE"):
        agent.generate_speech_candidates(pre, known_wolf_ids=P[1:3], q=frozen_q(pre))
    chat.assert_not_called()


@pytest.mark.parametrize("phase,last,expected", [
    ("discussion", False, P[2:6]),
    ("discussion", True, ()),
    ("pk_discussion", False, (P[1],)),
    ("pk_discussion", True, ()),
])
def test_reply_window_before_next_vote(phase, last, expected):
    if phase == "discussion":
        pre = discussion_pre(prior=P[1:] if last else (P[-1],))
        team = P[:2]
    else:
        witness = PublicWitness(phase)
        if last:
            witness.speech(P[0], "no_commitment", None, "我暂时不作明确表态。")
            witness.event("turn_start", speaker=P[1])
        pre = witness.pre()
        team = (pre.current_speaker, P[3])
    if expected:
        row = candidate("INQUIRE", f"{expected[0]}，你能说明依据吗？", expected[0])
    else:
        row = candidate("INFORM", "请核对刚才公开的记录。")
    agent, chat = mock_agent([row])
    batch = agent.generate_speech_candidates(pre, known_wolf_ids=team, q=frozen_q(pre))
    assert request_context(chat)["replyable_observers"] == list(expected)
    properties = chat.call_args.kwargs["response_format"]["json_schema"]["schema"]["properties"]
    item = properties["candidates"]["items"]["properties"]
    assert ("INQUIRE" in item["function"]["enum"]) == bool(expected)
    assert item["addressee"]["enum"] == [*expected, None]
    assert batch.candidates[0].addressee == row["addressee"]


@pytest.mark.parametrize("extra,omit", [
    ("ADVOCATE", None), ("INQUIRE", None), ("INFORM", None), ("ADVOCATE", "INQUIRE"),
])
def test_one_supplement_can_belong_to_any_function(extra, omit):
    pre = discussion_pre()
    rows = [
        candidate("INFORM", "我想核对刚才的转述。"),
        candidate("ADVOCATE", "我主张先听完其他人的发言。"),
        candidate("INQUIRE", "3号，你的判断依据是什么？", P[2]),
    ]
    supplements = {
        "INFORM": candidate("INFORM", "先前一句发言仍是待核实的主张。"),
        "ADVOCATE": candidate("ADVOCATE", "我现在倾向于投4号。"),
        "INQUIRE": candidate("INQUIRE", "4号，哪句话改变了你的判断？", P[3]),
    }
    rows = [row for row in rows if row["function"] != omit] + [supplements[extra]]
    agent, chat = mock_agent(rows)
    batch = agent.generate_speech_candidates(pre, known_wolf_ids=P[:2], q=frozen_q(pre))
    chat.assert_called_once()
    assert json.loads(json.dumps([asdict(item) for item in batch.candidates])) == rows


@pytest.mark.parametrize("functions", [("INFORM", "INFORM", "ADVOCATE", "ADVOCATE"),
                                       ("ADVOCATE", "ADVOCATE", "ADVOCATE", "INQUIRE")])
def test_rejects_more_than_one_supplement(functions):
    pre = discussion_pre()
    rows = [candidate(function, f"3号，请说明第{i + 1}条依据。" if function == "INQUIRE" else f"陈述{i + 1}。",
                      P[2] if function == "INQUIRE" else None) for i, function in enumerate(functions)]
    agent, chat = mock_agent(rows)
    with pytest.raises(GameplayActionValidationError, match="至多补充一份"):
        agent.generate_speech_candidates(pre, known_wolf_ids=P[:2], q=frozen_q(pre))
    chat.assert_called_once()


@pytest.mark.parametrize("seat", [P[0], P[1], P[5], P[6]])
def test_rejects_self_wolf_already_spoken_or_dead_addressee(seat):
    pre = discussion_pre(alive=P[:6], prior=(P[5],))
    agent, chat = mock_agent([candidate("INQUIRE", f"{seat}，你能解释依据吗？", seat)])
    with pytest.raises(GameplayActionValidationError, match="投票前可以回复"):
        agent.generate_speech_candidates(pre, known_wolf_ids=P[:2], q=frozen_q(pre))
    chat.assert_called_once()


@pytest.mark.parametrize("rows,raw,finish_reason,error,match", [
    ([], None, "stop", GameplayActionValidationError, "1 至 4"),
    ([candidate("INFORM", "请核对公开记录。")] * 5, None, "stop", GameplayActionValidationError, "1 至 4"),
    ([], "{", "stop", GameplayActionValidationError, "JSON"),
    ([candidate("INFORM", "请核对公开记录。")], None, "length", GameplayActionValidationError, "截断"),
    ([candidate("DEFER", "我暂不表态。")], None, "stop", GameplayActionValidationError, "未知"),
    ([candidate("INFORM", " ")], None, "stop", GameplaySpeechQualityError, "empty"),
    ([candidate("INFORM", "请核对公开记录。"), candidate("ADVOCATE", " 请核对公开记录。\n")],
     None, "stop", GameplayActionValidationError, "重复"),
    ([candidate("INFORM", "核对3号发言。", P[2])], None, "stop", GameplayActionValidationError, "仅 INQUIRE"),
    ([candidate("INQUIRE", "4号，能解释吗？", P[2])], None, "stop", GameplaySpeechQualityError, "omitted"),
])
def test_rejects_invalid_responses_without_another_call(rows, raw, finish_reason, error, match):
    pre = discussion_pre()
    agent, chat = mock_agent(rows, raw=raw, finish_reason=finish_reason)
    with pytest.raises(error, match=match):
        agent.generate_speech_candidates(pre, known_wolf_ids=P[:2], q=frozen_q(pre))
    chat.assert_called_once()


@pytest.mark.parametrize("phase,row,actions", [
    ("discussion", candidate("INFORM", "  我们先核对公开原话。\n这两句话涉及不同的事。  "), ()),
    ("discussion", candidate("INQUIRE", "3号，你能说明你的公开依据吗？\n", P[2]), (("oppose", P[2]),)),
    ("pk_discussion", candidate("ADVOCATE", "我反对3号刚才的说法。\n本轮我会投3号。"),
     (("oppose", P[2]), ("vote_intent", P[2]))),
])
def test_explicit_candidate_commits_original_text_and_actual_v1_result(monkeypatch, phase, row, actions):
    env, recorder, parser_backend = runtime(phase, native=True)
    handoff = recorder._pending.handoff
    pre = recorder.public_pre_for_handoff(handoff)
    agent, chat = mock_agent([candidate("INFORM", "我说的是另一条公开记录。"), row])
    team = _legal_team(pre, env.get_observation())
    batch = agent.generate_speech_candidates(pre, known_wolf_ids=team, q=frozen_q(pre))
    speech = batch.candidates[1].text
    parser_backend.perception_response = "\n".join(
        f"{pre.current_speaker} | {action} | {target}" for action, target in actions
    ) if actions else "NONE"
    parse = Mock(wraps=env.speech_perceiver.parse_with_audit)
    monkeypatch.setattr(env.speech_perceiver, "parse_with_audit", parse)
    previous_events, previous_votes = len(env.public_events), deepcopy(env.vote_target)
    speaker_id, day, env_phase = env.current_act_idx + 1, env.day, env.phase

    commit_public_speech(speech, pre=batch.pre, env=env, recorder=recorder, handoff=handoff)

    chat.assert_called_once()
    parse.assert_called_once_with(speaker=speaker_id, speech=row["text"], day=day, phase=env_phase)
    speeches = [event for event in env.public_events[previous_events:] if event["event_type"] == "public_speech"]
    assert len(speeches) == len(recorder.submitted_gameplay_actions) == 1
    submitted = recorder.submitted_gameplay_actions[0]
    annotation = env.speech_annotations[-1]
    assert speeches[0]["raw_text"] == submitted.action_payload.to_value() == row["text"]
    assert submitted.resulting_public_event_ids == (annotation.event_id,) == (speeches[0]["event_id"],)
    assert annotation.status is (V1AnnotationStatus.OK if actions else V1AnnotationStatus.NO_ACTION)
    assert annotation.actions == tuple(V1SpeechAction(pre.current_speaker, action, target) for action, target in actions)
    assert annotation.attempts[-1].raw_response == parser_backend.perception_response
    assert env.vote_target == previous_votes
    assert recorder._pending is None

    committed = snapshot(env, recorder)
    with pytest.raises(ValueError, match="unconsumed"):
        commit_public_speech(speech, pre=pre, env=env, recorder=recorder, handoff=handoff)
    assert parse.call_count == 1 and snapshot(env, recorder) == committed


@pytest.mark.parametrize("problem,match", [("pre", "pending PRE"), ("speaker", "偏离 PRE"), ("parser", "incomplete")])
def test_failed_commit_leaves_native_pending_state_unchanged(monkeypatch, problem, match):
    env, recorder, backend = runtime(native=True)
    handoff = recorder._pending.handoff
    pre = recorder.public_pre_for_handoff(handoff)
    if problem == "pre":
        pre = replace(pre, prefix_digest="f" * 64)
    elif problem == "speaker":
        env.current_act_idx += 1
    else:
        backend.perception_response = "无效解析响应"
    parse = Mock(wraps=env.speech_perceiver.parse_with_audit)
    monkeypatch.setattr(env.speech_perceiver, "parse_with_audit", parse)
    before = snapshot(env, recorder)
    with pytest.raises(ValueError, match=match):
        commit_public_speech("请核对公开记录。", pre=pre, env=env, recorder=recorder, handoff=handoff)
    assert parse.call_count == int(problem == "parser")
    assert snapshot(env, recorder) == before
