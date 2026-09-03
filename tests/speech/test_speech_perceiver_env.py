import unittest

from werewolf.envs.werewolf_text_env_v0 import (
    WerewolfTextEnvV0,
)
from werewolf.speech.speech_perceiver import SpeechParseAuditResult


ROLES = [
    "Werewolf",
    "Werewolf",
    "Seer",
    "Witch",
    "Villager",
    "Villager",
    "Villager",
]


class RecordingPerceiver:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def parse_with_audit(
        self,
        speaker,
        speech,
        day,
        phase,
    ):
        self.calls.append(
            {
                "speaker": speaker,
                "speech": speech,
                "day": day,
                "phase": phase,
            }
        )

        return SpeechParseAuditResult(
            normalized_actions=self.result,
            raw_response="synthetic parser response",
            parse_status="ok",
            error_type=None,
            error_message=None,
            generation_attempts=(
                {
                    "generation_attempt": 1,
                    "status": "ok",
                    "raw_response": "synthetic parser response",
                    "error_type": None,
                    "error_message": None,
                },
            ),
        )


class RaisingPerceiver:
    def parse_with_audit(
        self,
        speaker,
        speech,
        day,
        phase,
    ):
        raise RuntimeError("parse failed")


class NonListPerceiver:
    def parse_with_audit(
        self,
        speaker,
        speech,
        day,
        phase,
    ):
        return SpeechParseAuditResult(
            normalized_actions={"action": "support", "object": "player2"},
            raw_response="synthetic parser response",
            parse_status="ok",
            error_type=None,
            error_message=None,
            generation_attempts=(
                {
                    "generation_attempt": 1,
                    "status": "ok",
                    "raw_response": "synthetic parser response",
                    "error_type": None,
                    "error_message": None,
                },
            ),
        )


class SpeechPerceiverEnvironmentTest(unittest.TestCase):
    def make_env(self, perceiver=None):
        kwargs = {
            "log_save_path": None,
        }

        if perceiver is not None:
            kwargs["speech_perceiver"] = perceiver

        env = WerewolfTextEnvV0(**kwargs)
        env.reset(roles=ROLES)

        return env

    @staticmethod
    def set_speech_state(
        env,
        phase,
        current_act_idx,
    ):
        env.phase = phase
        env._append_public_event("death_announcement", dead_players=[])
        env.day = 1
        env.day_or_night = "day"
        if phase == "speech_pk":
            env.phase = "speech"
            env._append_public_phase()
            env.phase = "vote"
            env._append_public_phase()
            env._append_public_event("vote_result", votes=[])
            env._append_public_event("exile_result", exiled_players=[])
        env.phase = phase
        env.current_act_idx = current_act_idx
        env.alive = [
            1
            for _ in range(env.n_player)
        ]
        env.speech_queue = [
            (current_act_idx + 1)
            % env.n_player
        ]
        env.vote_queue = []
        env._append_public_phase()
        env._append_turn_start()

    def test_visible_observation_contains_only_raw_speech_and_sidecar_has_actions(self):
        actions = [
            [
                "player2",
                "point_as_werewolf",
                "player3",
            ]
        ]

        perceiver = RecordingPerceiver(actions)
        env = self.make_env(perceiver)

        self.set_speech_state(
            env,
            phase="speech",
            current_act_idx=1,
        )

        observation, _, done, _ = env.step(
            (
                "speech",
                "我认为3号是狼人",
            )
        )

        self.assertFalse(done)
        self.assertEqual(
            observation["current_act_idx"],
            3,
        )

        self.assertEqual(
            perceiver.calls,
            [
                {
                    "speaker": 2,
                    "speech": "我认为3号是狼人",
                    "day": 1,
                    "phase": "speech",
                }
            ],
        )

        speech_log = next(
            log
            for log in reversed(env.game_log)
            if log.event == "speech"
        )

        self.assertEqual(
            speech_log.content,
            {
                "speech_content": "我认为3号是狼人",
            },
        )

        observed_log = next(
            log
            for log in reversed(
                observation["game_log"]
            )
            if log.event == "speech"
        )

        self.assertEqual(
            observed_log.source,
            2,
        )
        self.assertNotIn("sp_actions", observed_log.content)
        self.assertEqual(
            [action.to_record() for action in env.speech_annotations[-1].actions],
            actions,
        )

    def test_speech_pk_actions_are_only_in_annotation_sidecar(self):
        actions = [
            [
                "player4",
                "oppose",
                "player2",
            ]
        ]

        perceiver = RecordingPerceiver(actions)
        env = self.make_env(perceiver)

        self.set_speech_state(
            env,
            phase="speech_pk",
            current_act_idx=3,
        )

        env.step(
            (
                "speech_pk",
                "我不信2号",
            )
        )

        self.assertEqual(
            perceiver.calls[0]["speaker"],
            4,
        )
        self.assertEqual(
            perceiver.calls[0]["phase"],
            "speech_pk",
        )

        speech_log = next(
            log
            for log in reversed(env.game_log)
            if log.event == "speech_pk"
        )

        self.assertNotIn("sp_actions", speech_log.content)
        self.assertEqual(
            [action.to_record() for action in env.speech_annotations[-1].actions],
            actions,
        )

    def test_parser_exception_prevents_raw_speech_commit(self):
        env = self.make_env(
            RaisingPerceiver()
        )

        self.set_speech_state(
            env,
            phase="speech",
            current_act_idx=0,
        )

        before_events = list(env.public_events)
        with self.assertRaisesRegex(RuntimeError, "parse failed"):
            env.step(("speech", "发言"))
        self.assertEqual(env.public_events, before_events)

    def test_non_sequence_result_fails_closed_after_public_speech_boundary(self):
        env = self.make_env(
            NonListPerceiver()
        )

        self.set_speech_state(
            env,
            phase="speech",
            current_act_idx=0,
        )

        before_events = list(env.public_events)
        with self.assertRaisesRegex(
            TypeError,
            "V1SpeechAction",
        ):
            env.step(("speech", "发言"))
        self.assertEqual(env.public_events[: len(before_events)], before_events)
        self.assertEqual(env.public_events[-1]["event_type"], "public_speech")

if __name__ == "__main__":
    unittest.main()
