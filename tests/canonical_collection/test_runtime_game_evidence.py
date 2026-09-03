from __future__ import annotations

import json

import pytest

from run_random import eval as run_game
from tests.canonical_collection.test_game_bundle import _claim, _plan
from werewolf.canonical_collection import (
    BackendCallPurpose,
    CanonicalFailureStage,
    CollectionSeedPoolExhausted,
    SpeakerPREBeliefHandoff,
    TerminalOutcome,
    V1_SPEECH_PARSER_VERSION,
    V1_SPEECH_PROMPT_VERSION,
    collect,
    validate_attempt_ledger,
    validate_canonical_failure_evidence,
    validate_canonical_game_bundle,
    validate_canonical_game_evidence,
    validate_canonical_partial_evidence,
)
from werewolf.canonical_collection.collector import CanonicalGameProduct
from werewolf.canonical_collection.call_audit import (
    CanonicalCallAudit,
    audited_backends,
)
from werewolf.canonical_collection.runtime import (
    CanonicalGameRecorder,
    PlayingAgentBeliefObservationCollector,
    make_classic7_replay_executor,
)
from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
from werewolf.agents.llm_agent import GameplayGenerationExhausted
from werewolf.speech.private_belief_perceiver import PlayingAgentBeliefReporter
from werewolf.speech.speech_perceiver import SpeechPerceiver


ROLES = (
    "Werewolf",
    "Werewolf",
    "Villager",
    "Villager",
    "Villager",
    "Seer",
    "Witch",
)


class _Backend:
    def __init__(self, *, perception_response="NONE"):
        self.requests = []
        self.perception_response = perception_response

    def chat(self, *, messages, **_kwargs):
        prompt = messages[-1]["content"]
        self.requests.append(prompt)
        if "suspected_werewolves" in prompt:
            marker = "MUST INCLUDE: "
            required = prompt.split(marker, 1)[1].splitlines()[0]
            support = [] if required == "<none>" else required.split(", ")
            return json.dumps({"suspected_werewolves": support})
        return self.perception_response


class _Agent:
    def __init__(self, backend, seat):
        self.backend = backend
        self.backend_id = "deterministic-fixture-backend-v1"
        self.model_name = "fixture-model-v1"
        self.seat = seat

    def reset(self):
        return None

    def report_suspected_werewolves_readonly(
        self,
        *,
        observation,
        report_prompt,
        legal_candidates,
        required_candidates,
    ):
        del observation, legal_candidates, required_candidates
        return self.backend.chat(messages=[{"role": "user", "content": report_prompt}])

    def act(self, observation):
        phase = observation["phase"]
        actions = observation["valid_action"]
        if "skill_witch" in phase:
            return next(action for action in actions if "pass" in action[0])
        if "skill_wolf" in phase:
            return next(action for action in actions if action[1] != -1)
        if "vote" in phase:
            return next(
                (action for action in actions if action[1] == 0),
                actions[0],
            )
        return actions[0]

    def act_with_pre_speech_belief(self, observation, *, pre_speech_belief):
        assert isinstance(pre_speech_belief, SpeakerPREBeliefHandoff)
        assert pre_speech_belief.observer_id == f"player{self.seat}"
        speech_kind = "speech_pk" if "speech_pk" in observation["phase"] else "speech"
        return (speech_kind, "I make no public accusation.")


class _OneSpeechEnvironment(WerewolfTextEnvV0):
    """Controlled runtime fixture that terminates after one real speech step."""

    def step(self, action):
        observation, reward, done, info = super().step(action)
        if action[0] in {"speech", "speech_pk"} and not done:
            return observation, reward, True, {"Werewolf": -1}
        return observation, reward, done, info


def _runtime(
    *,
    max_calls=10000,
    perception_response="NONE",
    plan=None,
    claim=None,
):
    plan = plan or _plan(
        parser_identity=V1_SPEECH_PARSER_VERSION,
        prompt_identity=V1_SPEECH_PROMPT_VERSION,
        call_budget_identity="fixture-call-budget-v1",
    )
    claim = claim or _claim(plan)
    raw_backend = _Backend(perception_response=perception_response)
    audit = CanonicalCallAudit(plan=plan, configured_call_limit=max_calls)
    backend = audited_backends({"fixture": raw_backend}, audit)["fixture"]
    agents = [_Agent(backend, seat) for seat in range(1, 8)]
    env = _OneSpeechEnvironment(
        speech_perceiver=SpeechPerceiver(
            backend=backend,
            model_name=plan.model_identity,
        ),
        random_seed=claim.seed,
        log_save_path=None,
    )
    belief_collector = PlayingAgentBeliefObservationCollector(
        plan=plan,
        reporter=PlayingAgentBeliefReporter(audit_hook=audit),
        agents=agents,
    )
    recorder = CanonicalGameRecorder(
        plan=plan,
        claim=claim,
        game_id="game-000",
        belief_collector=belief_collector,
        call_audit=audit,
        runtime_configuration={"night0": True, "fixture": True},
    )
    return plan, claim, env, agents, audit, recorder, raw_backend


def _replay_executor():
    class NoActionPerceiver:
        def parse_with_audit(self, **_kwargs):
            from werewolf.speech.speech_perceiver import SpeechParseAuditResult

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

    return make_classic7_replay_executor(
        identity="controlled-one-speech-replay-v1",
        environment_factory=lambda seed: _OneSpeechEnvironment(
            speech_perceiver=NoActionPerceiver(),
            random_seed=seed,
            log_save_path=None,
        ),
    )


def test_real_runtime_constructs_complete_pre_v1_handoff_evidence():
    plan, claim, env, agents, audit, recorder, raw_backend = _runtime()

    result = run_game(
        env,
        agents,
        ROLES,
        canonical_recorder=recorder,
        call_audit=audit,
    )
    evidence = recorder.complete_evidence()

    assert result in {"Werewolf win", "Villager win"}
    assert validate_canonical_game_evidence(evidence) is evidence
    assert evidence.authoritative_pre_prefixes
    for prefix in evidence.authoritative_pre_prefixes:
        terminal = prefix.public_event_history.events[-1]
        assert terminal.event_type == "turn_start"
        assert terminal.speaker == prefix.current_speaker
        assert all(
            event.event_index <= terminal.event_index
            for event in prefix.public_event_history.events
        )
    assert len(evidence.belief_observations) == sum(
        len(prefix.alive_observer_ids)
        for prefix in evidence.authoritative_pre_prefixes
    )
    assert evidence.call_budget_summary.second_speaker_belief_count == 0
    assert evidence.call_budget_summary.fallback_action_count == 0
    assert len(raw_backend.requests) == evidence.call_budget_summary.used_calls
    belief_calls = [
        call
        for call in evidence.backend_call_evidence
        if call.purpose is BackendCallPurpose.BELIEF_OBSERVATION
    ]
    assert len(belief_calls) == len(evidence.belief_observations)
    assert len({call.operation_id for call in belief_calls}) == len(belief_calls)
    assert all(
        action.speaker_pre_belief_handoff is not None
        for action in evidence.submitted_gameplay_actions
        if action.action_type == "public_speech"
    )


def test_production_collector_executes_real_game_to_verified_bundle(tmp_path):
    plan = _plan(
        target_canonical_success_count=1,
        parser_identity=V1_SPEECH_PARSER_VERSION,
        prompt_identity=V1_SPEECH_PROMPT_VERSION,
        call_budget_identity="fixture-call-budget-v1",
    )

    class Runtime:
        def __init__(self, recorder, env, agents, audit):
            self.recorder = recorder
            self.env = env
            self.agents = agents
            self.audit = audit

        def run(self):
            run_game(
                self.env,
                self.agents,
                ROLES,
                canonical_recorder=self.recorder,
                call_audit=self.audit,
            )
            return CanonicalGameProduct(
                self.recorder.complete_evidence(),
                _replay_executor(),
            )

    def factory(*, plan, claim):
        _, _, env, agents, audit, recorder, _ = _runtime(
            plan=plan,
            claim=claim,
        )
        return Runtime(recorder, env, agents, audit)

    result = collect(
        plan=plan,
        runtime_factory=factory,
        destination=tmp_path,
        timestamp_utc=lambda: "2026-09-03T03:00:00Z",
    )

    claim = result.ledger_state.claims[0]
    terminal = result.ledger_state.terminals[0]
    verified = validate_canonical_game_bundle(
        tmp_path / "games" / terminal.canonical_game_bundle_id,
        plan=plan,
        claim=claim,
        replay_executor=_replay_executor(),
    )
    assert verified.manifest_digest == terminal.canonical_game_bundle_digest


def test_failed_real_observation_publishes_complete_partial_evidence(tmp_path):
    plan = _plan(
        ordered_seed_pool=(101,),
        target_canonical_success_count=1,
        parser_identity=V1_SPEECH_PARSER_VERSION,
        prompt_identity=V1_SPEECH_PROMPT_VERSION,
        call_budget_identity="fixture-call-budget-v1",
    )

    class Runtime:
        def __init__(self, recorder, env, agents, audit):
            self.recorder = recorder
            self.env = env
            self.agents = agents
            self.audit = audit

        def run(self):
            run_game(
                self.env,
                self.agents,
                ROLES,
                canonical_recorder=self.recorder,
                call_audit=self.audit,
            )
            raise AssertionError("failed observation must stop the game")

    def factory(*, plan, claim):
        _, _, env, agents, audit, recorder, _ = _runtime(
            plan=plan,
            claim=claim,
        )
        original = agents[0].report_suspected_werewolves_readonly

        def self_suspicion(**kwargs):
            original(**kwargs)
            return '{"suspected_werewolves":["player1"]}'

        agents[0].report_suspected_werewolves_readonly = self_suspicion
        return Runtime(recorder, env, agents, audit)

    with pytest.raises(CollectionSeedPoolExhausted):
        collect(
            plan=plan,
            runtime_factory=factory,
            destination=tmp_path,
            timestamp_utc=lambda: "2026-09-03T03:00:00Z",
        )

    state = validate_attempt_ledger(tmp_path / "attempt_ledger", plan)
    terminal = state.terminals[0]
    assert terminal.outcome is TerminalOutcome.CANONICAL_FAILURE
    failure = validate_canonical_failure_evidence(
        tmp_path
        / "attempts"
        / state.claims[0].attempt_id
        / "failure_evidence.json",
        plan=plan,
        claim=state.claims[0],
    ).evidence
    assert failure.stage is CanonicalFailureStage.BELIEF_OBSERVATION
    assert len(failure.partial_evidence) == 1
    partial = validate_canonical_partial_evidence(
        tmp_path / failure.partial_evidence[0].relative_path,
        plan=plan,
        claim=state.claims[0],
    ).evidence.payload.to_value()
    assert partial["authoritative_pre_prefixes"]
    assert partial["backend_calls"]
    assert partial["public_events"][-1]["event_type"] == "turn_start"


def test_belief_failure_stops_before_speaker_action_and_is_canonical_failure():
    plan, claim, env, agents, audit, recorder, _ = _runtime()
    original = agents[0].report_suspected_werewolves_readonly

    def self_suspicion(**kwargs):
        original(**kwargs)
        return '{"suspected_werewolves":["player1"]}'

    agents[0].report_suspected_werewolves_readonly = self_suspicion

    with pytest.raises(Exception) as captured:
        run_game(
            env,
            agents,
            ROLES,
            canonical_recorder=recorder,
            call_audit=audit,
        )

    failure = recorder.failure_from_exception(captured.value)
    assert failure.stage is CanonicalFailureStage.BELIEF_OBSERVATION
    assert failure.retry_exhausted is True
    assert failure.observer_id == "player1"
    assert failure.partial_payload["public_events"][-1]["event_type"] == (
        "turn_start"
    )
    assert failure.partial_payload["backend_calls"]
    assert not any(
        action.action_type == "public_speech"
        for action in recorder.submitted_gameplay_actions
    )


def test_exhausted_perception_is_canonical_failure_not_no_action():
    _, _, env, agents, audit, recorder, _ = _runtime(
        perception_response="invalid parser output",
    )

    with pytest.raises(Exception) as captured:
        run_game(
            env,
            agents,
            ROLES,
            canonical_recorder=recorder,
            call_audit=audit,
        )

    failure = recorder.failure_from_exception(captured.value)
    assert failure.stage is CanonicalFailureStage.SPEECH_PERCEPTION
    assert failure.retry_exhausted is True
    assert len(failure.attempt_evidence) == 3
    assert failure.partial_payload["public_events"][-1]["event_type"] == (
        "public_speech"
    )
    assert failure.partial_payload["pending_submitted_action"] == (
        "speech",
        "I make no public accusation.",
    )
    assert failure.partial_payload["backend_calls"]


def test_gameplay_exhaustion_has_no_fallback_action():
    _, _, env, agents, audit, recorder, _ = _runtime()

    def exhausted(_observation):
        raise GameplayGenerationExhausted(
            stage="night_action",
            attempts=3,
            last_error=ValueError("invalid action"),
        )

    agents[0].act = exhausted

    with pytest.raises(Exception) as captured:
        run_game(
            env,
            agents,
            ROLES,
            canonical_recorder=recorder,
            call_audit=audit,
        )

    failure = recorder.failure_from_exception(captured.value)
    assert failure.stage is CanonicalFailureStage.GAMEPLAY_ACTION
    assert failure.retry_exhausted is True
    assert recorder.submitted_gameplay_actions == []
