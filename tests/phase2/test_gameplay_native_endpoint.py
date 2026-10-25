"""CPU integration regression: real Classic7 environment, recorder and replay.

No GPU, Q worker or external LLM is used. Missing native dependencies must be
reported as an environment error, not converted into a passing/skipped test.
"""
from dataclasses import replace

import pytest

from run_random import eval as run_game
from tests.canonical_collection.test_runtime_game_evidence import ROLES, _runtime
from tests.phase2.test_gameplay_campaign import GameplayLedger, no_router, plan
from tests.runtime.test_runtime_config import new_config
from werewolf.artifact_io import ArtifactValidationError
from werewolf.canonical_collection.attempt_ledger import collection_plan_from_record
from werewolf.canonical_collection.game_bundle import publish_canonical_game_bundle
from werewolf.canonical_collection.production_runtime import (
    _ReplaySpeechPerceiver, classic7_full_game_replay_executor, classic7_replay_executor,
)
from werewolf.canonical_collection.runtime import make_classic7_replay_executor
from werewolf.canonical_collection.trajectory_evidence import construct_private_replay_evidence
from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
from werewolf.phase2_gameplay import publish_campaign, validate_publication, verify_results


def native_evidence(game, claim, game_id, *, winner):
    _, _, setup_env, agents, audit, recorder, _ = _runtime(plan=game, claim=claim)
    recorder.game_id = game_id
    # Use the unmodified native class, not the one-speech termination fixture.
    env = WerewolfTextEnvV0(random_seed=claim.seed, log_save_path=None,
                           speech_perceiver=setup_env.speech_perceiver)
    for agent in agents:
        original = agent.act
        def act(observation, original=original):
            # Current public actions use 1-based seats; 0 is no-kill/abstain.
            # Match Mainline's native-terminal fixture, not the old one-speech
            # agent whose 0 actions deliberately never finish a native game.
            actions = observation["valid_action"]
            if "vote" in observation["phase"] and winner == "Werewolf":
                return max(actions, key=lambda action: action[1])
            if "skill_wolf" in observation["phase"] or "vote" in observation["phase"]:
                return next(action for action in actions if action[1] != 0)
            return original(observation)
        agent.act = act
    actual = run_game(env, agents, ROLES, canonical_recorder=recorder, call_audit=audit)
    assert actual == winner + " win"
    assert type(env) is WerewolfTextEnvV0 and env.phase == "end_game"
    assert env.is_done()[1:] == (True, {"Werewolf": 1 if winner == "Werewolf" else -1})
    return recorder.complete_evidence()


def changed_trace(evidence, *, raw=None, actions=None, winner=None):
    old = evidence.private_replay_evidence
    inputs = old.replay_inputs.to_value()
    if raw is not None:
        inputs["submitted_actions"] = raw
    if winner is not None:
        inputs["winner"] = winner
    private = construct_private_replay_evidence(game_id=evidence.game_id, seed=old.seed,
        role_assignment=dict(old.role_assignment), runtime_configuration=old.runtime_configuration,
        initial_runtime_state=old.initial_runtime_state, replay_inputs=inputs,
        expected_public_event_digest=old.expected_public_event_digest)
    return replace(evidence, private_replay_evidence=private,
                   submitted_gameplay_actions=actions if actions is not None else evidence.submitted_gameplay_actions)


@pytest.mark.parametrize("winner", ["Werewolf", "Villager"])
def test_native_full_winner_for_all_assignments_including_no_router(tmp_path, winner):
    ledger = GameplayLedger(tmp_path / "work", plan())
    ledger.initialize()
    game = collection_plan_from_record(ledger.plan["game_plan"])
    replay = classic7_full_game_replay_executor(new_config())
    generic = classic7_replay_executor(new_config())
    for allocation in ledger.plan["allocations"]:
        claim = ledger.start(allocation)
        evidence = native_evidence(game, claim, allocation["game_id"], winner=winner)
        # Historical public replay retains its original identity/output contract.
        assert generic.identity == "classic7-runtime-action-replay-v1"
        assert generic.execute(evidence.private_replay_evidence, evidence.submitted_gameplay_actions) == \
            replay.execute(evidence.private_replay_evidence, evidence.submitted_gameplay_actions)
        publish_canonical_game_bundle(ledger.work / "games" / allocation["game_id"],
            plan=game, claim=claim, evidence=evidence, replay_executor=replay)
        result = verify_results(ledger, replay, no_router)[-1][1]
        assert result["winner"] == winner
        ledger._append("RESULT", result)
    publication = publish_campaign(ledger, tmp_path / "publication", replay, no_router)
    validate_publication(publication.path, publication.manifest_digest, replay, no_router, expected_plan=ledger.plan)
    assert ledger.status()["complete_games"] == ledger.plan["randomized_games"]


@pytest.mark.parametrize("violation", ["unfinished", "wrong_winner", "extra_action"])
def test_native_replay_rejects_incomplete_or_false_terminal_evidence(tmp_path, violation):
    ledger = GameplayLedger(tmp_path / "work", plan())
    ledger.initialize()
    allocation = ledger.plan["allocations"][0]
    claim = ledger.start(allocation)
    game = collection_plan_from_record(ledger.plan["game_plan"])
    evidence = native_evidence(game, claim, allocation["game_id"], winner="Werewolf")
    raw = evidence.private_replay_evidence.replay_inputs.to_value()["submitted_actions"]
    actions = evidence.submitted_gameplay_actions
    if violation == "unfinished":
        bad = changed_trace(evidence, raw=raw[:-1], actions=actions[:-1])
    elif violation == "wrong_winner":
        bad = changed_trace(evidence, winner="Villager")
    else:
        bad = changed_trace(evidence, raw=raw + [raw[-1]], actions=actions + actions[-1:])
    replay = classic7_full_game_replay_executor(new_config())
    with pytest.raises(ValueError, match="terminal|winner"):
        replay.execute(bad.private_replay_evidence, bad.submitted_gameplay_actions)
    if violation == "wrong_winner":
        # Even an internally consistent, rehashed canonical bundle must fail the
        # independent Gameplay check. Ordinary public replay cannot check winner.
        public_only = make_classic7_replay_executor(identity=replay.identity,
            environment_factory=lambda seed: WerewolfTextEnvV0(log_save_path=None,
                random_seed=seed, speech_perceiver=_ReplaySpeechPerceiver()))
        publish_canonical_game_bundle(ledger.work / "games" / allocation["game_id"],
            plan=game, claim=claim, evidence=bad, replay_executor=public_only)
        before = ledger.path.read_bytes()
        with pytest.raises(ArtifactValidationError, match="winner"):
            verify_results(ledger, replay, no_router)
        assert ledger.path.read_bytes() == before
        assert ledger.status()["status"] == "INCOMPLETE"


def test_native_early_done_cannot_substitute_for_terminal_win_rule(tmp_path):
    from tests.canonical_collection.test_runtime_game_evidence import _OneSpeechEnvironment
    ledger = GameplayLedger(tmp_path / "work", plan())
    ledger.initialize()
    allocation = ledger.plan["allocations"][0]
    claim = ledger.start(allocation)
    game = collection_plan_from_record(ledger.plan["game_plan"])
    _, _, env, agents, audit, recorder, _ = _runtime(plan=game, claim=claim)
    recorder.game_id = allocation["game_id"]
    run_game(env, agents, ROLES, canonical_recorder=recorder, call_audit=audit)
    evidence = recorder.complete_evidence()
    # Partial public replay is still accepted by the unchanged default mode.
    public_events = classic7_replay_executor(new_config()).execute(
        evidence.private_replay_evidence, evidence.submitted_gameplay_actions)
    assert public_events == env.public_events
    with pytest.raises(ValueError, match="native game terminal"):
        classic7_full_game_replay_executor(new_config()).execute(
            evidence.private_replay_evidence, evidence.submitted_gameplay_actions)
    replay = make_classic7_replay_executor(identity="negative-early-done-v1", require_full_game=True,
        environment_factory=lambda seed: _OneSpeechEnvironment(random_seed=seed, log_save_path=None,
                                                               speech_perceiver=_ReplaySpeechPerceiver()))
    with pytest.raises(ValueError, match="native game terminal"):
        replay.execute(evidence.private_replay_evidence, evidence.submitted_gameplay_actions)
