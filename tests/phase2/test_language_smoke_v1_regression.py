"""Verbatim smoke-v1 text against V1.1 perception mechanics; no live model calls."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_phase2_language_smoke as smoke
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.phase2_actions import (
    Action, ActionContextV1, InformationRequestV1, ObservationWindowV1,
    Phase2SemanticPlanV1, verify_plan,
)
from werewolf.phase2_language import (
    Phase2SemanticPerceiverV1, PublicLanguageContextV1, verify_language_execution,
)


FIXTURE = Path(__file__).parent / "fixtures/phase2-smoke-v1-regression-fixtures.json"
SHA256 = "563375684201c194c550f9642a487cf356aa4404d08210485113ae53d00c615e"
SELECTION_DIGEST = "9ab58cb73fdee18fee63795ee71df37e5ff00c2c320939c2f48221c3167e192e"


def real_fixtures():
    data = FIXTURE.read_bytes()
    assert hashlib.sha256(data).hexdigest() == SHA256
    source = json.loads(data)["source"]
    assert source == {
        "artifact": "paper-phase2-language-smoke-v1",
        "manifest_digest": "bcd838964b46b63e25cf3013ecf77936d66389fb399cc1457f74c5492a58215a",
        "case_selection_digest": SELECTION_DIGEST,
        "source_file": "case_executions.jsonl",
        "source_case_count": 58,
    }
    fixtures = json.loads(data)["fixtures"]
    assert {row["source_row_index"] for row in fixtures} == {1, 3, 16, 28, 46, 52, 53, 54}
    assert len(fixtures) == 8
    return {row["source_row_index"]: row for row in fixtures}


class ScriptedBackend:
    """Replay the recorded V1 language fields, not a desired V1.1 answer."""

    supports_json_schema = True

    def __init__(self, recorded):
        self.recorded = {key: recorded[key] for key in (
            "commitment_targets", "rejected_targets", "vote_intent_targets",
            "information_requests", "abstain_intent", "private_fact_claim")}
        self.calls = []

    def chat_with_metadata(self, **kwargs):
        self.calls.append(kwargs)
        return json.dumps(self.recorded, ensure_ascii=False), {"finish_reason": "stop"}


def fixture_contract(row):
    """Reconstruct only action-relevant PRE fields supplied in the fixture."""
    record = row["requested_plan"]
    legal = set(record["legal_targets"])
    actor = row["acting_wolf"]
    teammate = record["continuation_actor"] if record["action"] == "PROBE" else next(
        p for p in PLAYER_IDS if p not in legal and p != actor)
    wolves = frozenset((actor, teammate))
    alive = tuple(p for p in PLAYER_IDS if p in legal or p in wolves)
    competition = (alive if row["phase"] == "speech" else
                   tuple(p for p in PLAYER_IDS if p in legal or p == actor or
                         (record["action"] == "PROBE" and p == teammate)))
    start = competition.index(actor)
    queue = competition[start:] + competition[:start]
    context = ActionContextV1(
        row["game_id"], row["boundary_id"], row["prefix_digest"],
        "fixture-public-history-digest", row["phase"], actor, alive, wolves,
        competition, queue)
    request = record["information_request"]
    window = record["expected_observation_window"]
    plan = Phase2SemanticPlanV1(
        Action(record["action"]), record["candidate_j"], record["commitment_target"],
        record["rejected_target"], record["redirect_target"],
        InformationRequestV1(**request) if request else None,
        record["vote_intent"], record["phase"], record["acting_wolf"],
        tuple(record["legal_targets"]), record["continuation_actor"],
        ObservationWindowV1(window["after_boundary_id"], window["before_actor"],
                            window["phase"], tuple(window["expected_speakers"])) if window else None)
    assert plan.to_record() == record
    assert verify_plan(context, plan).valid
    public = PublicLanguageContextV1(row["phase"], actor, alive, competition,
                                     context.public_history_digest)
    return context, plan, public


def replay(row, attempt_number=2):
    attempt = next(item for item in row["attempts"] if item["attempt"] == attempt_number)
    context, plan, public = fixture_contract(row)
    backend = ScriptedBackend(attempt["perceived_semantics"])
    perceived = Phase2SemanticPerceiverV1(backend, "offline-script").perceive(
        attempt["generated_text"], public)
    prompt = backend.calls[0]["messages"][0]["content"]
    assert attempt["generated_text"] in prompt
    assert "requested_plan" not in prompt and "known_wolves" not in prompt
    assert "speaker" not in backend.calls[0]["response_format"]["json_schema"]["schema"]["properties"]
    assert "phase" not in backend.calls[0]["response_format"]["json_schema"]["schema"]["properties"]
    assert (perceived.speaker, perceived.phase) == (public.speaker, public.phase)
    return perceived, verify_language_execution(plan, perceived, context)


def test_real_fixture_provenance_and_v3_selection_lock():
    rows = real_fixtures()
    assert len(rows) == 8
    assert smoke.NAME == "paper-phase2-language-smoke-v3"
    assert smoke.VERSION == "phase2_language_smoke_v3"
    assert smoke.EXPECTED_CASE_SELECTION_DIGEST == SELECTION_DIGEST
    assert sum(quota for _, _, quota in smoke.LAYOUT) == 58


@pytest.mark.parametrize("index", [1, 3, 16, 52, 54, 46])
def test_recorded_perception_failures_are_corrected(index):
    row = real_fixtures()[index]
    assert not row["final_language_execution_valid"]
    perceived, result = replay(row)
    assert result.valid, (index, result.invalid_reason, perceived)
    if index in (1, 3, 16):
        assert perceived.information_requests == (
            InformationRequestV1(row["candidate_j"], row["candidate_j"]),)
    if index in (52, 54, 46):
        assert perceived.commitment_targets == perceived.vote_intent_targets == (row["candidate_j"],)
        assert perceived.rejected_targets == () and not perceived.abstain_intent
    if index == 3:
        assert perceived.speaker != row["candidate_j"]


def test_actual_target_drift_and_private_claim_remain_invalid():
    rows = real_fixtures()
    drift, result = replay(rows[53])
    assert drift.commitment_targets == drift.vote_intent_targets == ("player7",)
    assert result.invalid_reason == "PUSH_COMMITMENT_MISMATCH"
    private, result = replay(rows[28])
    assert private.private_fact_claim
    assert result.invalid_reason == "PRIVATE_FACT_CLAIM"


def test_real_abstention_negative_control_still_invalid():
    # Row 46 is a false abstention alarm: its text explicitly votes for j.
    # A genuinely unrequested speaker abstention must still fail unchanged.
    row = real_fixtures()[46]
    context, plan, public = fixture_contract(row)
    payload = {"commitment_targets": [row["candidate_j"]], "rejected_targets": [],
               "vote_intent_targets": [row["candidate_j"]], "information_requests": [],
               "abstain_intent": True, "private_fact_claim": False}
    backend = ScriptedBackend(payload)
    perceived = Phase2SemanticPerceiverV1(backend, "offline-script").perceive(
        f"本轮我弃票，不投任何人。", public)
    assert perceived.abstain_intent
    assert verify_language_execution(plan, perceived, context).invalid_reason == "UNREQUESTED_ABSTAIN"


def test_explicit_conflicting_votes_are_not_collapsed_to_frozen_target():
    context, plan, public = fixture_contract(real_fixtures()[52])
    backend = ScriptedBackend({"commitment_targets": [], "rejected_targets": [],
                               "vote_intent_targets": [], "information_requests": [],
                               "abstain_intent": False, "private_fact_claim": False})
    perceived = Phase2SemanticPerceiverV1(backend, "offline-script").perceive(
        "本轮放逐票投player3。今天放逐票投player5。", public)
    assert perceived.commitment_targets == perceived.vote_intent_targets == ("player3", "player5")
    assert verify_language_execution(plan, perceived, context).invalid_reason == "PUSH_COMMITMENT_MISMATCH"
