"""Phase-2 calls use a separate sidecar and canonical RUNTIME contexts."""

from contextlib import contextmanager
from types import SimpleNamespace
import json

import pytest

from tests.phase2.test_decision_opportunity import opportunity
from tests.phase2.test_language import contexts
from werewolf.phase2_actions import Action
from werewolf.phase2_backend_audit import (
    Phase2BackendAuditError, Phase2BackendCallAuditV1,
)
from werewolf.phase2_language import (
    Phase2LanguageActorV1, Phase2SemanticPerceiverV1,
    build_perception_prompt, perception_response_format,
)
from werewolf.phase2_treatment import build_phase2_treatment
from werewolf.canonical_collection.trajectory_evidence import BackendCallPurpose


class FakeCanonicalAudit:
    def __init__(self):
        self.records = []
        self.current = None
        self.contexts = []

    @contextmanager
    def context(self, **kwargs):
        assert self.current is None
        self.current = kwargs
        self.contexts.append(kwargs)
        try:
            yield
        finally:
            self.current = None


class FakeAuditedBackend:
    supports_json_schema = True
    canonical_backend_identity = "scripted-canonical"

    def __init__(self, audit, responses):
        self._audit = audit
        self.responses = iter(responses)
        self.calls = []

    def chat_with_metadata(self, **kwargs):
        self.calls.append(kwargs)
        operation = self._audit.current["operation_id"]
        self._audit.records.append(SimpleNamespace(
            operation_id=operation, call_id=operation + "-attempt-01"))
        return next(self.responses), {"finish_reason": "stop"}


def _setup():
    opp = opportunity()
    treatment = build_phase2_treatment(
        opp, Action.PUSH, assignment_source="randomized_pilot",
        assignment_probability=.5, randomization_key="assignment")
    canonical = FakeCanonicalAudit()
    call_audit = Phase2BackendCallAuditV1(opp, treatment, canonical)
    return opp, treatment, canonical, call_audit


def test_actor_perception_repair_calls_are_indexed_and_plan_blinded():
    _, treatment, canonical, call_audit = _setup()
    _, public = contexts()
    text = "本轮投player2。"
    payload = {"commitment_targets": ["player2"], "rejected_targets": [],
               "vote_intent_targets": ["player2"], "information_requests": [],
               "abstain_intent": False, "private_fact_claim": False}
    backend = FakeAuditedBackend(canonical, [text, json.dumps(payload), text])
    actor = Phase2LanguageActorV1(backend, "model", call_audit=call_audit)
    perceiver = Phase2SemanticPerceiverV1(backend, "model", call_audit=call_audit)
    assert actor.realize(treatment.plan, public) == text
    assert perceiver.perceive(text, public).vote_intent_targets == ("player2",)
    assert actor.realize(treatment.plan, public,
                         failure_reason="SYNTHETIC_REPAIR") == text
    assert [(row.role, row.attempt_index) for row in call_audit.records] == [
        ("realization", 1), ("perception", 1), ("repair", 2)]
    assert all(row.treatment_id == treatment.treatment_id for row in call_audit.records)
    assert all(row.request_digest and row.response_digest and row.canonical_call_id
               for row in call_audit.records)
    prompt = backend.calls[1]["messages"][0]["content"]
    assert prompt == build_perception_prompt(text, public)
    assert "requested_plan" not in prompt and "requested_action" not in prompt
    assert "known_wolves" not in prompt
    assert all(item["purpose"] is BackendCallPurpose.RUNTIME
               for item in canonical.contexts)


def test_perception_request_cannot_substitute_plan_bearing_prompt():
    _, treatment, canonical, call_audit = _setup()
    _, public = contexts()
    backend = FakeAuditedBackend(canonical, ["本轮投player2。"])
    actor = Phase2LanguageActorV1(backend, "model", call_audit=call_audit)
    actor.realize(treatment.plan, public)
    with pytest.raises(Phase2BackendAuditError, match="public-only"):
        call_audit.perception_call(
            backend=backend, model="model", prompt="requested_plan=PUSH",
            text="本轮投player2。", public_context=public,
            temperature=0.0, max_tokens=384,
            response_format=perception_response_format())
    assert len(backend.calls) == 1
