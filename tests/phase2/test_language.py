"""Synthetic Phase-2 speech execution tests; no gameplay or real LLM calls."""

from dataclasses import replace
import json

import pytest

from scripts.phase2_language_realization import realize_verify_action
from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.phase2_actions import (
    ActionContextV1, InformationRequestV1, probe_plan, push_plan, redirect_plan,
)
from werewolf.phase2_language import (
    OTHER_REQUEST, Phase2LanguageActorV1, Phase2LanguageError,
    Phase2SemanticPerceiverV1, Phase2SpeechSemanticV1, PublicLanguageContextV1,
    build_perception_prompt, build_realization_prompt, parse_perceived_semantics,
    verify_language_execution,
)


def contexts():
    alive = tuple(f"player{i}" for i in range(1, 8))
    legal = ActionContextV1("g", "b", "prefix", "public-digest", "speech",
                            "player1", alive, frozenset(("player1", "player5")),
                            alive, alive)
    public = PublicLanguageContextV1("speech", "player1", alive, alive, "public-digest")
    return legal, public


def semantic(*, commitments=(), rejections=(), votes=(), requests=(),
             abstain=False, private=False):
    return Phase2SpeechSemanticV1("player1", "speech", tuple(commitments),
                                  tuple(rejections), tuple(votes), tuple(requests),
                                  abstain, private)


@pytest.mark.parametrize("perceived,expected", [
    (semantic(commitments=("player2",), votes=("player2",)), None),
    (semantic(commitments=("player3",), votes=("player3",)), "PUSH_COMMITMENT_MISMATCH"),
    (semantic(commitments=("player2", "player3"), votes=("player2",)), "PUSH_COMMITMENT_MISMATCH"),
    (semantic(commitments=("player2",), votes=("player2",), private=True), "PRIVATE_FACT_CLAIM"),
])
def test_push_perception_cases(perceived, expected):
    legal, _ = contexts()
    result = verify_language_execution(push_plan(legal, "player2"), perceived, legal)
    assert result.invalid_reason == expected
    assert result.valid is (expected is None)


@pytest.mark.parametrize("perceived,expected", [
    (semantic(commitments=("player3",), rejections=("player2",), votes=("player3",)), None),
    (semantic(commitments=("player3",), votes=("player3",)), "REDIRECT_REJECTION_MISMATCH"),
    (semantic(commitments=("player3",), rejections=("player4",), votes=("player3",)),
     "REDIRECT_REJECTION_MISMATCH"),
    (semantic(commitments=("player3",), rejections=("player2",), votes=("player3",),
              requests=(InformationRequestV1("player2", "player2"),)), "REDIRECT_REQUEST_CONFLICT"),
])
def test_redirect_requires_reject_and_commit_but_no_role_assertion(perceived, expected):
    legal, public = contexts()
    plan = redirect_plan(legal, "player2", {k: .8 if k == "player3" else .1
                                            for k in legal.legal_targets})
    assert "好人" in build_realization_prompt(plan, public)
    assert "不等于断言" in build_realization_prompt(plan, public)
    result = verify_language_execution(plan, perceived, legal)
    assert result.invalid_reason == expected
    assert result.valid is (expected is None)


@pytest.mark.parametrize("perceived,expected", [
    (semantic(requests=(InformationRequestV1("player2", "player2"),)), None),
    (semantic(), "PROBE_REQUEST_MISMATCH"),
    (semantic(requests=(InformationRequestV1("player2", "player3"),)), "PROBE_REQUEST_MISMATCH"),
    (semantic(requests=(InformationRequestV1("player2", "player2"),),
              commitments=("player2",)), "PROBE_COMMITMENT_CONFLICT"),
    (semantic(requests=(InformationRequestV1("player2", "player2", OTHER_REQUEST),)),
     "PROBE_REQUEST_MISMATCH"),
])
def test_probe_request_is_distinct_from_no_stance(perceived, expected):
    legal, _ = contexts()
    result = verify_language_execution(probe_plan(legal, "player2"), perceived, legal)
    assert result.invalid_reason == expected
    assert result.valid is (expected is None)


def test_structural_invalidity_blocks_language_comparison():
    legal, _ = contexts()
    bad = replace(push_plan(legal, "player2"), vote_intent="player3")
    result = verify_language_execution(bad,
        semantic(commitments=("player2",), votes=("player2",)), legal)
    assert result.invalid_reason == "STRUCTURED_PLAN_INVALID:PUSH_SEMANTICS_MISMATCH"


class ControlledActor:
    def __init__(self, texts):
        self.texts = iter(texts)
        self.calls = []

    def realize(self, plan, public_context, *, failure_reason=None):
        self.calls.append((plan, public_context, failure_reason))
        return next(self.texts)


class ControlledPerceiver:
    def __init__(self, semantics):
        self.semantics = iter(semantics)
        self.calls = []

    def perceive(self, text, public_context):
        self.calls.append((text, public_context))
        value = next(self.semantics)
        if isinstance(value, Exception):
            raise value
        return value


@pytest.mark.parametrize("plan_kind,text,perceived", [
    ("push", "我建议今天投player2。",
     semantic(commitments=("player2",), votes=("player2",))),
    ("redirect", "今天不把player2作为主放逐目标，转而投player3。",
     semantic(commitments=("player3",), rejections=("player2",), votes=("player3",))),
    ("probe", "player2，请说明你现在主要怀疑谁，以及有哪些公开依据？",
     semantic(requests=(InformationRequestV1("player2", "player2"),))),
])
def test_controlled_text_pipeline_for_each_action(plan_kind, text, perceived):
    legal, public = contexts()
    if plan_kind == "push":
        plan = push_plan(legal, "player2")
    elif plan_kind == "redirect":
        plan = redirect_plan(legal, "player2", {k: .8 if k == "player3" else .1
                                                for k in legal.legal_targets})
    else:
        plan = probe_plan(legal, "player2")
    result = realize_verify_action(plan, legal, public,
                                   actor=ControlledActor([text]),
                                   perceiver=ControlledPerceiver([perceived]))
    assert result.speech == text
    assert result.audit.structured_execution_valid
    assert result.audit.language_execution_valid
    assert result.audit.generated_text_digest == sha256_bytes(text.encode("utf-8"))
    assert result.audit.perceived_semantics["action_identity"] == plan.action.value


def test_retry_success_preserves_plan_and_hides_it_from_perceiver():
    legal, public = contexts()
    plan = push_plan(legal, "player2")
    actor = ControlledActor(["今天投player3。", "今天投player2。"])
    perceiver = ControlledPerceiver([
        semantic(commitments=("player3",), votes=("player3",)),
        semantic(commitments=("player2",), votes=("player2",))])
    result = realize_verify_action(plan, legal, public, actor=actor, perceiver=perceiver)
    assert result.speech == "今天投player2。"
    assert result.audit.structured_execution_valid and result.audit.language_execution_valid
    assert [call[2] for call in actor.calls] == [None, "PUSH_COMMITMENT_MISMATCH"]
    assert all(call[0] is plan for call in actor.calls)
    assert perceiver.calls == [("今天投player3。", public), ("今天投player2。", public)]
    assert len(result.audit.attempts) == 2
    assert result.audit.response_opportunity_reached is None
    assert result.audit.information_gain is None


def test_retry_exhausted_returns_no_committable_speech():
    legal, public = contexts()
    plan = push_plan(legal, "player2")
    actor = ControlledActor(["今天投player3。", "还是投player3。"])
    perceiver = ControlledPerceiver([
        semantic(commitments=("player3",), votes=("player3",)),
        semantic(commitments=("player3",), votes=("player3",))])
    result = realize_verify_action(plan, legal, public, actor=actor, perceiver=perceiver)
    assert result.speech is None and result.perceived_semantics is None
    assert not result.audit.language_execution_valid
    assert result.audit.language_invalid_reason == "PUSH_COMMITMENT_MISMATCH"
    assert len(actor.calls) == len(perceiver.calls) == 2
    assert result.audit.generated_text == "还是投player3。"


def test_perception_parse_failure_gets_one_repair_attempt():
    legal, public = contexts()
    plan = push_plan(legal, "player2")
    actor = ControlledActor(["投player2。", "今天明确投player2。"])
    perceiver = ControlledPerceiver([
        Phase2LanguageError("PERCEPTION_JSON_INVALID"),
        semantic(commitments=("player2",), votes=("player2",))])
    result = realize_verify_action(plan, legal, public, actor=actor, perceiver=perceiver)
    assert result.audit.language_execution_valid
    assert [a.language_invalid_reason for a in result.audit.attempts] == [
        "PERCEPTION_JSON_INVALID", None]
    assert actor.calls[1][2] == "PERCEPTION_JSON_INVALID"


def test_invalid_plan_makes_no_language_call_and_audit_serializes():
    legal, public = contexts()
    bad = replace(probe_plan(legal, "player2"), information_request=None)
    actor, perceiver = ControlledActor([]), ControlledPerceiver([])
    result = realize_verify_action(bad, legal, public, actor=actor, perceiver=perceiver)
    assert not actor.calls and not perceiver.calls
    assert not result.audit.structured_execution_valid
    assert result.audit.language_invalid_reason == "STRUCTURED_PLAN_INVALID:INVALID_INFORMATION_REQUEST"
    assert result.audit.canonical_bytes() == canonical_json_bytes(result.audit.to_record())
    assert result.audit.requested_plan_digest == sha256_bytes(canonical_json_bytes(bad.to_record()))


class ScriptedBackend:
    supports_json_schema = True

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def chat_with_metadata(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.responses), {"finish_reason": "stop"}


def test_independent_perception_adapter_sees_text_and_public_context_only():
    _, public = contexts()
    source = semantic(commitments=("player2",), votes=("player2",))
    payload = source.to_record()
    payload.pop("schema_version")
    payload.pop("action_identity")
    payload.pop("speaker")
    payload.pop("phase")
    backend = ScriptedBackend([json.dumps(payload)])
    perceiver = Phase2SemanticPerceiverV1(backend, "mock")
    result = perceiver.perceive("今天投player2。", public)
    assert result == source
    request = backend.calls[0]
    prompt = request["messages"][0]["content"]
    assert "今天投player2。" in prompt
    assert "PUSH" not in prompt and "requested_plan" not in prompt
    assert "known_wolves" not in prompt and "Werewolf" not in prompt
    assert request["response_format"]["json_schema"]["strict"] is True
    assert "speaker" not in request["response_format"]["json_schema"]["schema"]["properties"]
    assert "phase" not in request["response_format"]["json_schema"]["schema"]["properties"]


def test_realization_adapter_uses_frozen_plan_and_bounded_repair_prompt():
    legal, public = contexts()
    plan = push_plan(legal, "player2")
    backend = ScriptedBackend(["今天投player2。"])
    actor = Phase2LanguageActorV1(backend, "mock")
    assert actor.realize(plan, public, failure_reason="PUSH_COMMITMENT_MISMATCH") == "今天投player2。"
    prompt = backend.calls[0]["messages"][0]["content"]
    assert "player2" in prompt and "PUSH_COMMITMENT_MISMATCH" in prompt
    assert prompt.count("上游已冻结唯一目标player2") == 2
    assert "不得改选目标" in prompt and "不得加入其他承诺" in prompt
    assert "优先只输出1至2句" in prompt
    assert "上游已冻结唯一目标player3" not in prompt


def test_perception_json_rejects_extra_or_invalid_fields():
    _, public = contexts()
    source = semantic(requests=(InformationRequestV1("player2", "player2"),))
    record = source.to_record()
    record.pop("schema_version")
    record.pop("action_identity")
    record.pop("speaker")
    record.pop("phase")
    assert parse_perceived_semantics(json.dumps(record), public) == source
    with pytest.raises(Phase2LanguageError, match="PERCEPTION_FIELDS_INVALID"):
        parse_perceived_semantics(json.dumps({**record, "requested_action": "PROBE"}), public)
    with pytest.raises(Phase2LanguageError, match="PERCEPTION_FIELDS_INVALID"):
        parse_perceived_semantics(json.dumps({**record, "speaker": "player2"}), public)
    with pytest.raises(Phase2LanguageError, match="PERCEPTION_SEMANTICS_INVALID"):
        parse_perceived_semantics(json.dumps({**record, "commitment_targets": ["player2", "player2"]}), public)


def test_addressed_player_never_replaces_trusted_public_speaker():
    legal, public = contexts()
    request = InformationRequestV1("player2", "player2")
    payload = {"commitment_targets": [], "rejected_targets": [],
               "vote_intent_targets": [], "information_requests": [request.to_record()],
               "abstain_intent": False, "private_fact_claim": False}
    text = "player2，你当前主要怀疑或投票判断是谁？依据哪些已公开信息？"
    backend = ScriptedBackend([json.dumps(payload)])
    perceived = Phase2SemanticPerceiverV1(backend, "mock").perceive(text, public)
    assert (perceived.speaker, perceived.phase) == ("player1", "speech")
    assert perceived.information_requests == (request,)
    assert verify_language_execution(probe_plan(legal, "player2"), perceived, legal).valid
    prompt = backend.calls[0]["messages"][0]["content"]
    assert "被称呼或被推动的玩家不是发言者" in prompt
    assert "requested_plan" not in prompt and "known_wolves" not in prompt


def test_push_and_probe_extraction_examples_do_not_turn_mention_into_rejection():
    _, public = contexts()
    push_text = "player3的发言我也听到了，但本轮坚决将放逐票投给player2。"
    push_prompt = build_perception_prompt(push_text, public)
    assert "坚决将放逐票投给X" in push_prompt
    assert "仅提及、比较或讨论其他玩家，不等于拒绝" in push_prompt
    probe_prompt = build_perception_prompt(
        "player2，你当前主要怀疑或投票判断是谁？依据哪些已公开信息？", public)
    assert "target_j=player3、addressee_j=player3" in probe_prompt
    assert "CURRENT_SUSPICION_BASIS" in probe_prompt
    assert "requested_action" not in push_prompt + probe_prompt


def test_push_target_drift_and_unrequested_abstention_remain_invalid():
    legal, public = contexts()
    plan = push_plan(legal, "player2")
    drift = semantic(commitments=("player3",), votes=("player3",))
    abstain = semantic(commitments=("player2",), votes=("player2",), abstain=True)
    assert verify_language_execution(plan, drift, legal).invalid_reason == "PUSH_COMMITMENT_MISMATCH"
    assert verify_language_execution(plan, abstain, legal).invalid_reason == "UNREQUESTED_ABSTAIN"
    prompt = build_realization_prompt(plan, public, "PUSH_COMMITMENT_MISMATCH")
    assert prompt.count("上游已冻结唯一目标player2") == 2
    assert "不得改选目标" in prompt and "弃票表态" in prompt


def test_repair_repeats_redirect_and_probe_frozen_requirements():
    legal, public = contexts()
    redirect = redirect_plan(legal, "player2", {k: .8 if k == "player3" else .1
                                                for k in legal.legal_targets})
    redirect_prompt = build_realization_prompt(redirect, public, "REDIRECT_COMMITMENT_MISMATCH")
    assert redirect_prompt.count("转而推动player3") == 2
    probe_prompt = build_realization_prompt(probe_plan(legal, "player2"), public,
                                            "PROBE_REQUEST_MISMATCH")
    assert probe_prompt.count("直接向player2提问") == 2
