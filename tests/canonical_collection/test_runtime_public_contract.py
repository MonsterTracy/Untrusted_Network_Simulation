from __future__ import annotations

from dataclasses import dataclass

import pytest

from werewolf.canonical_collection import V1AnnotationStatus, V1SpeechAnnotation
from werewolf.canonical_collection.public_history import freeze_public_event_history
from werewolf.envs.werewolf_text_env_v0 import (
    V1SpeechPerceptionExhausted,
    WerewolfTextEnvV0,
)
from werewolf.speech.speech_perceiver import SpeechParseAuditResult


ROLES = (
    "Werewolf",
    "Werewolf",
    "Villager",
    "Villager",
    "Villager",
    "Seer",
    "Witch",
)


@dataclass
class _Perceiver:
    fail: bool = False
    model_name: str = "fixture-model-v1"
    backend_id: str = "deterministic-fixture-backend-v1"

    def parse_with_audit(self, *, speaker, speech, day, phase):
        del speaker, speech, day, phase
        if self.fail:
            return SpeechParseAuditResult(
                normalized_actions=[],
                raw_response="not valid",
                parse_status="parser_error",
                error_type="SpeechActionValidationError",
                error_message="invalid action",
                generation_attempts=tuple(
                    {
                        "generation_attempt": index,
                        "status": "parser_error",
                        "raw_response": "not valid",
                        "error_type": "SpeechActionValidationError",
                        "error_message": "invalid action",
                    }
                    for index in range(1, 4)
                ),
            )
        return SpeechParseAuditResult(
            normalized_actions=[],
            raw_response="NONE",
            parse_status="ok",
            error_type=None,
            error_message=None,
            generation_attempts=(
                {
                    "generation_attempt": 1,
                    "status": "ok",
                    "raw_response": "NONE",
                    "error_type": None,
                    "error_message": None,
                },
            ),
        )


def _environment(*, fail_perception=False):
    return WerewolfTextEnvV0(
        speech_perceiver=_Perceiver(fail=fail_perception),
        random_seed=17,
        log_save_path=None,
    )


def _advance_to_first_speech(env):
    observation = env.reset(roles=ROLES)
    for _ in range(16):
        if env.phase == "speech":
            return observation
        action = observation["valid_action"][0]
        observation, _, done, _ = env.step(action)
        assert done is False
    raise AssertionError("fixture did not reach Day1 discussion")


def test_runtime_emits_initial_night0_and_only_coarse_public_phases():
    env = _environment()

    _advance_to_first_speech(env)
    history = freeze_public_event_history(env.public_events)

    assert history.to_records()[0] == {
        "event_id": "event-000000",
        "event_index": 0,
        "event_type": "phase_change",
        "day": 0,
        "phase": "night",
    }
    assert history.to_records()[-1]["event_type"] == "turn_start"
    phases = {
        record["phase"]
        for record in history.to_records()
        if record["event_type"] == "phase_change"
    }
    assert phases == {"night", "discussion"}
    assert not phases.intersection({"skill_wolf", "skill_seer", "skill_witch"})


def test_runtime_v1_no_action_is_success_bound_to_public_speech():
    env = _environment()
    _advance_to_first_speech(env)

    env.step(("speech", "I make no accusation."))

    annotation = env.speech_annotations[-1]
    assert isinstance(annotation, V1SpeechAnnotation)
    assert annotation.status is V1AnnotationStatus.NO_ACTION
    assert annotation.actions == ()
    speech_event = next(
        event
        for event in env.public_events
        if event["event_type"] == "public_speech"
    )
    assert annotation.event_id == speech_event["event_id"]


def test_exhausted_v1_error_is_explicit_and_stops_the_game():
    env = _environment(fail_perception=True)
    _advance_to_first_speech(env)

    with pytest.raises(V1SpeechPerceptionExhausted, match="exhausted"):
        env.step(("speech", "malformed semantic response"))

    annotation = env.speech_annotations[-1]
    assert annotation.status is V1AnnotationStatus.ERROR
    assert len(annotation.attempts) == 3
    assert env.public_events[-1]["event_type"] == "public_speech"
