"""Phase 3A uses existing Actor/parser with scripted backends, never a server."""
from copy import deepcopy
from unittest.mock import Mock
import subprocess
import sys

import pytest

from scripts import speech_realization as r
from scripts.speech_planning import SpeechPlan, PlanningAction as A
from werewolf.agents.gpt_agent import GPTAgent
from werewolf.speech.speech_perceiver import SpeechPerceiver


CASES = [
    (A.ACCUSE_WOLF, "player2", "point_as_werewolf", "player2", "我认为player2是狼人。"),
    (A.CLEAR, "player2", "point_as_non_werewolf", "player2", "我认为player2不是狼人。"),
    (A.SUPPORT, "player2", "support", "player2", "我支持player2的观点。"),
    (A.OPPOSE, "player2", "oppose", "player2", "我反对player2的观点。"),
    (A.SELF_DEFEND, None, "point_as_non_werewolf", "player1", "我认为player1不是狼人。"),
    (A.NO_COMMITMENT, None, "no_commitment", None, "我暂时不作明确表态。"),
]


class ActorBackend:
    def __init__(self, speech):
        self.speech = speech
        self.calls = []

    def chat_with_metadata(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        return self.speech, {"finish_reason": "stop"}


class ParserBackend:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        return self.response


def parser(response):
    return SpeechPerceiver(backend=ParserBackend(response), model_name="test-parser")


@pytest.mark.parametrize("action,target,semantic,obj,speech", CASES)
def test_six_plans_existing_actor_exact_perception_and_request_identity(action, target, semantic, obj, speech):
    plan = SpeechPlan(action, target)
    payload = plan.public_payload()
    context = r.PublicRealizationContext("player1", 1, "discussion")
    before = deepcopy(payload)
    expected = r.expected_action(payload, context.speaker)
    assert expected.to_record() == ["player1", semantic, obj]
    intent = r.actor_intent(payload, context.speaker)
    assert intent == r.actor_intent(payload, context.speaker)
    assert len(intent) == 1 and intent[0].action == semantic
    assert intent[0].target == (None if obj is None else int(obj[-1]))
    requests = []
    # Selection source is deliberately absent from the Actor API.
    for selection_source in ("No-ToM", "+ToM"):
        backend = ActorBackend(speech)
        actor = GPTAgent(backend=backend, model_name="test-actor")
        actor._generate_day_cognition = Mock(side_effect=AssertionError("cognition forbidden"))
        perceiver = parser(f"player1 | {semantic} | {obj or 'NONE'}")
        captured = []
        parse = perceiver.parse_with_audit
        def once(**kwargs):
            audit = parse(**kwargs)
            captured.append(audit)
            return audit
        perceiver.parse_with_audit = Mock(side_effect=once)
        result = r.realize_parse_verify(payload, context, actor=actor, perceiver=perceiver)
        assert len(backend.calls) == 1
        assert perceiver.parse_with_audit.call_count == 1
        assert result.perception is captured[0]
        actor._generate_day_cognition.assert_not_called()
        snapshot = deepcopy(result.perception)
        assert r.verify_semantics(payload, "player1", result.perception) is result.perception
        assert result.perception == snapshot
        assert result.expected == expected and result.speech == speech
        requests.append(backend.calls)
    assert requests[0] == requests[1]
    # Direct Original Agent realization must issue the identical request.
    backend = ActorBackend(speech)
    original = GPTAgent(backend=backend, model_name="test-actor")
    temperature, max_tokens = original._request_limits()
    assert original._generate_public_speech(context.observation(), discussion_acts=intent,
        claim_catalog=context.claims, temperature=temperature, max_tokens=max_tokens) == speech
    assert backend.calls == requests[0]
    assert payload == before and plan == SpeechPlan(action, target)


@pytest.mark.parametrize("response", [
    "player1 | oppose | player2",  # wrong action
    "player1 | point_as_werewolf | player3",  # wrong target
    "player1 | point_as_werewolf | player2\nplayer1 | support | player3",
    "player1 | point_as_seer | player1",
    "player1 | vote_intent | player2",
    "player1 | poison | player2",
    "NONE", "invalid parser output",
])
def test_failure_is_terminal_with_one_perception_and_no_regeneration(response):
    payload = SpeechPlan(A.ACCUSE_WOLF, "player2").public_payload()
    backend = ActorBackend("我认为player2是狼人。")
    actor = GPTAgent(backend=backend, model_name="test-actor")
    perceiver = parser(response)
    captured = []
    parse = perceiver.parse_with_audit
    def once(**kwargs):
        result = parse(**kwargs)
        captured.append(result)
        return result
    perceiver.parse_with_audit = Mock(side_effect=once)
    sentinel = object()
    result = sentinel
    with pytest.raises(r.SemanticVerificationError) as failure:
        result = r.realize_parse_verify(payload, r.PublicRealizationContext("player1", 1, "discussion"),
                                         actor=actor, perceiver=perceiver)
    assert result is sentinel
    assert len(backend.calls) == 1
    assert perceiver.parse_with_audit.call_count == 1
    assert failure.value.perception is captured[0]
    snapshot = deepcopy(captured[0])
    with pytest.raises(r.SemanticVerificationError):
        r.verify_semantics(payload, "player1", captured[0])
    assert captured[0] == snapshot
    if response == "invalid parser output":
        assert captured[0].parse_status == "parser_error"


@pytest.mark.parametrize("action,response", [
    (A.SELF_DEFEND, "player1 | point_as_non_werewolf | player2"),
    (A.NO_COMMITMENT, "NONE"),
])
def test_self_defend_and_no_commitment_are_exact(action, response):
    perception = parser(response).parse_with_audit(speaker=1, speech="我暂时不作明确表态。", day=1, phase="discussion")
    with pytest.raises(r.SemanticVerificationError):
        r.verify_semantics(SpeechPlan(action).public_payload(), "player1", perception)
    if response == "NONE":
        assert perception.normalized_actions == []


@pytest.mark.parametrize("private_field", ["probability", "score", "Delta", "wolf_identities",
    "diagnostics", "candidate_ranking", "provenance", "checkpoint", "seal"])
def test_private_fields_rejected_before_actor(private_field):
    payload = SpeechPlan(A.NO_COMMITMENT).public_payload()
    payload[private_field] = "PRIVATE_SENTINEL"
    backend = ActorBackend("我暂时不作明确表态。")
    with pytest.raises(ValueError, match="public payload only"):
        r.realize_parse_verify(payload, r.PublicRealizationContext("player1", 1, "discussion"),
            actor=GPTAgent(backend=backend, model_name="test-actor"), perceiver=parser("NONE"))
    assert backend.calls == []


def test_actor_module_does_not_import_sealed_tom_or_planner():
    # Fresh interpreter avoids altering pytest's module identities.
    subprocess.run([sys.executable, "-c", "import sys; import scripts.speech_realization; "
        "assert 'scripts.counterfactual_tom' not in sys.modules; "
        "assert 'scripts.speech_planning' not in sys.modules; "
        "assert 'werewolf.tom.final_evaluation' not in sys.modules"], check=True)
