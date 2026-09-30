"""Canonical V1 parser audit is preserved for Phase-2 multi/no-action text."""

from dataclasses import replace

import pytest

from werewolf.speech.speech_perceiver import SpeechParseAuditResult
from werewolf.speech.verified_commit import bind_audited_public_speech


class Perceiver:
    backend_id = "test-parser"
    model_name = "test-model"


def audit(actions):
    return SpeechParseAuditResult(actions, "NONE" if not actions else "parsed",
        "ok", None, None, ({"generation_attempt": 1, "status": "ok",
                            "raw_response": "NONE" if not actions else "parsed",
                            "error_type": None, "error_message": None},))


@pytest.mark.parametrize("actions", [
    [],
    [["player1", "vote_intent", "player3"]],
    [["player1", "oppose", "player2"], ["player1", "vote_intent", "player3"]],
])
def test_canonical_audit_preserves_zero_one_or_multiple_v1_actions(actions):
    envelope = bind_audited_public_speech(
        speech="公开文本逐字保留", speaker="player1", perception=audit(actions),
        day=1, phase="speech", public_history_digest="history",
        perceiver=Perceiver())
    assert envelope.validated_perception().normalized_actions == actions
    assert envelope.speech == "公开文本逐字保留"
    with pytest.raises(ValueError, match="binding mismatch"):
        replace(envelope, speech="篡改").validated_perception()


def test_canonical_audit_rejects_wrong_speaker_or_failed_parser():
    with pytest.raises(ValueError, match="wrong speaker"):
        bind_audited_public_speech(
            speech="文本", speaker="player1",
            perception=audit([["player2", "vote_intent", "player3"]]),
            day=1, phase="speech", public_history_digest="history",
            perceiver=Perceiver())
    failed = replace(audit([]), parse_status="parser_error")
    with pytest.raises(ValueError, match="incomplete"):
        bind_audited_public_speech(
            speech="文本", speaker="player1", perception=failed,
            day=1, phase="speech", public_history_digest="history",
            perceiver=Perceiver())
