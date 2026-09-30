"""Replay the seven verbatim final-invalid smoke-v2 cases without an LLM."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_phase2_language_smoke as smoke
from tests.phase2.test_language_smoke_v1_regression import (
    ScriptedBackend, fixture_contract,
)
from werewolf.phase2_language import (
    Phase2SemanticPerceiverV1, verify_language_execution,
)


FIXTURE = Path(__file__).parent / "fixtures/phase2-smoke-v2-final-invalid-regression-fixtures.json"
SHA256 = "33480c195d6907797a337d3e8f8f1ba4756b97eaa32bf32cb4345974056d0d5d"
SELECTION_DIGEST = "9ab58cb73fdee18fee63795ee71df37e5ff00c2c320939c2f48221c3167e192e"


def real_fixtures():
    raw = FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHA256
    artifact = json.loads(raw)
    assert artifact["source"] == {
        "artifact": "paper-phase2-language-smoke-v2",
        "manifest_digest": "a7d4d31a7af49858956262c67e7f8b1d4b86777abbf960c0ef0af9cb6c52cfdd",
        "case_selection_digest": SELECTION_DIGEST,
        "source_case_count": 58,
    }
    rows = {row["source_row_index"]: row for row in artifact["fixtures"]}
    assert set(rows) == {7, 18, 23, 25, 28, 34, 36}
    assert all(len(row["attempts"]) == 2 for row in rows.values())
    return rows


def replay(row, attempt_number):
    attempt = next(item for item in row["attempts"] if item["attempt"] == attempt_number)
    legal, plan, public = fixture_contract(row)
    backend = ScriptedBackend(attempt["perceived_semantics"])
    perceived = Phase2SemanticPerceiverV1(backend, "offline-script").perceive(
        attempt["generated_text"], public)
    prompt = backend.calls[0]["messages"][0]["content"]
    assert attempt["generated_text"] in prompt
    assert "requested_plan" not in prompt
    assert "requested_action" not in prompt
    assert "redirect_target" not in prompt
    assert "known_wolves" not in prompt
    assert (perceived.speaker, perceived.phase) == (public.speaker, public.phase)
    return perceived, verify_language_execution(plan, perceived, legal)


def test_real_v2_fixture_provenance_and_v3_selection_lock():
    assert len(real_fixtures()) == 7
    assert smoke.NAME == "paper-phase2-language-smoke-v3"
    assert smoke.EXPECTED_CASE_SELECTION_DIGEST == SELECTION_DIGEST
    assert sum(quota for _, _, quota in smoke.LAYOUT) == 58


@pytest.mark.parametrize("row_index,attempt_number", [
    (18, 1), (23, 2), (25, 1), (25, 2), (34, 1), (34, 2), (36, 1), (36, 2),
])
def test_v2_perception_false_negatives_are_corrected(row_index, attempt_number):
    row = real_fixtures()[row_index]
    perceived, verdict = replay(row, attempt_number)
    assert verdict.valid, (row_index, attempt_number, verdict.invalid_reason, perceived)
    j = row["candidate_j"]
    k = row["requested_plan"]["redirect_target"]
    assert perceived.rejected_targets == (j,)
    assert perceived.commitment_targets == perceived.vote_intent_targets == (k,)
    assert not perceived.abstain_intent


def test_v2_probe_question_does_not_address_mentioned_suspect():
    row = real_fixtures()[7]
    perceived, verdict = replay(row, 1)
    assert verdict.valid
    assert [request.to_record() for request in perceived.information_requests] == [
        row["requested_plan"]["information_request"]]
    assert perceived.information_requests[0].addressee_j == "player6"
    assert perceived.information_requests[0].target_j == "player6"


@pytest.mark.parametrize("row_index,attempt_number,reason", [
    (7, 2, "PROBE_REQUEST_MISMATCH"),
    (18, 2, "UNREQUESTED_ABSTAIN"),
    (23, 1, "REDIRECT_REJECTION_MISMATCH"),
    (28, 1, "PRIVATE_FACT_CLAIM"),
    (28, 2, "PRIVATE_FACT_CLAIM"),
])
def test_v2_genuine_realization_violations_remain_invalid(row_index, attempt_number, reason):
    row = real_fixtures()[row_index]
    _, verdict = replay(row, attempt_number)
    assert not verdict.valid
    assert verdict.invalid_reason == reason
