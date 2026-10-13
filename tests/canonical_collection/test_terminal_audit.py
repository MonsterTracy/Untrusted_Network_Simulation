"""Real local engine plus deterministic Fixture agents; no external inference."""

from dataclasses import replace

import pytest

from run_random import eval as run_game
from tests.canonical_collection.test_game_bundle import _plan, _claim
from tests.canonical_collection.test_runtime_game_evidence import _runtime, _Agent, _Backend, ROLES, _replay_executor
from tests.runtime.test_runtime_config import new_config
from werewolf.artifact_io import ArtifactValidationError
from werewolf.canonical_collection import (
    construct_private_replay_evidence, initialize_attempt_ledger, publish_ledger_record,
    publish_canonical_game_bundle, construct_attempt_terminal, TerminalOutcome,
)
from werewolf.canonical_collection.call_audit import CanonicalCallAudit, audited_backends
from werewolf.canonical_collection.runtime import CanonicalGameRecorder, PlayingAgentBeliefObservationCollector
from werewolf.canonical_collection.production_runtime import classic7_replay_executor
from werewolf.canonical_terminal_audit import audit_terminal_bundle, publish_collection_terminal_audit, verify_collection_terminal_audit
from werewolf.development_publication import open_verified_collection
from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
from werewolf.runtime_config import normalize_runtime_config
from werewolf.speech.private_belief_perceiver import PlayingAgentBeliefReporter
from werewolf.speech.speech_perceiver import SpeechPerceiver


class _NativeAgent(_Agent):
    def act(self, observation):
        # Native public actions are 1-based; zero means no-kill / abstain.
        # The older one-speech Fixture intentionally selected zero forever.
        if "skill_wolf" in observation["phase"] or "vote" in observation["phase"]:
            return next(a for a in observation["valid_action"] if a[1] != 0)
        return super().act(observation)


@pytest.fixture(scope="module")
def native_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("native-terminal-fixture")
    return _native_collection(root, _NativeAgent)


def _native_collection(root, agent_type):
    config = normalize_runtime_config(new_config())
    plan = _plan()
    claim = _claim(plan)
    bundle = _native_game(root, agent_type, plan=plan, claim=claim, config=config, game_id="native-game")
    collection = open_verified_collection(root, replay_executor=classic7_replay_executor(config))
    return bundle, config, collection


def _native_game(root, agent_type, *, plan, claim, config, game_id):
    ledger = initialize_attempt_ledger(root, plan)
    publish_ledger_record(ledger, plan, claim)
    audit = CanonicalCallAudit(plan=plan, configured_call_limit=10000)
    backend = audited_backends({"fixture": _Backend()}, audit)["fixture"]
    agents = [agent_type(backend, seat) for seat in range(1, 8)]
    env = WerewolfTextEnvV0(**config["env_config"], random_seed=claim.seed, log_save_path=None,
        speech_perceiver=SpeechPerceiver(backend=backend, model_name=plan.model_identity))
    recorder = CanonicalGameRecorder(plan=plan, claim=claim, game_id=game_id, runtime_configuration=config,
        belief_collector=PlayingAgentBeliefObservationCollector(plan=plan,
            reporter=PlayingAgentBeliefReporter(audit_hook=audit), agents=agents), call_audit=audit)
    run_game(env, agents, ROLES, canonical_recorder=recorder, call_audit=audit)
    replay = classic7_replay_executor(config)
    bundle = publish_canonical_game_bundle(root / "games" / game_id, plan=plan, claim=claim,
        evidence=recorder.complete_evidence(), replay_executor=replay)
    terminal = construct_attempt_terminal(plan, claim, outcome=TerminalOutcome.CANONICAL_SUCCESS,
        canonical_game_bundle_id=bundle.game_id, canonical_game_bundle_digest=bundle.manifest_digest,
        terminal_timestamp_utc="2026-10-12T00:00:00Z")
    publish_ledger_record(ledger, plan, terminal)
    return bundle


def test_werewolf_native_winner_branch(tmp_path):
    class LastTargetAgent(_NativeAgent):
        def act(self, observation):
            if "vote" in observation["phase"]:
                return max(observation["valid_action"], key=lambda action: action[1])
            return super().act(observation)
    bundle, config, _ = _native_collection(tmp_path, LastTargetAgent)
    assert audit_terminal_bundle(bundle, runtime_config=config)["winner"] == "Werewolf"


def _private(bundle, **changes):
    original = bundle.private_replay_evidence
    fields = dict(game_id=original.game_id, seed=original.seed, role_assignment=dict(original.role_assignment),
        runtime_configuration=original.runtime_configuration.to_value(), initial_runtime_state=original.initial_runtime_state.to_value(),
        replay_inputs=original.replay_inputs.to_value(), expected_public_event_digest=original.expected_public_event_digest)
    fields.update(changes)
    return construct_private_replay_evidence(**fields)


def test_native_terminal_and_immutable_report(native_bundle, tmp_path):
    bundle, config, collection = native_bundle
    row = audit_terminal_bundle(bundle, runtime_config=config)
    assert row["native_phase"] == "end_game" and row["winner"] == "Villager"
    assert row["terminal_step"] == len(bundle.submitted_gameplay_actions)
    assert row["public_event_digest"] == bundle.public_event_stream.digest
    first = publish_collection_terminal_audit(collection, tmp_path / "audit", runtime_config=config)
    assert set(first.manifest["file_table"]) == {"restricted/terminal_audit.json"}
    assert "winner" not in first.manifest
    assert verify_collection_terminal_audit(first.path, collection, runtime_config=config).manifest_digest == first.manifest_digest
    assert publish_collection_terminal_audit(collection, first.path, runtime_config=config).manifest_digest == first.manifest_digest
    with pytest.raises(ArtifactValidationError):
        verify_collection_terminal_audit(first.path, replace(collection, plan=replace(collection.plan, source_revision="different")), runtime_config=config)


def test_wrong_winner_passes_legacy_replay_but_fails_terminal_audit(native_bundle):
    from werewolf.canonical_collection.game_bundle import validate_deterministic_replay
    bundle, config, _ = native_bundle
    replay = bundle.private_replay_evidence.replay_inputs.to_value()
    replay["winner"] = "Villager" if replay["winner"] == "Werewolf" else "Werewolf"
    private = _private(bundle, replay_inputs=replay)
    validate_deterministic_replay(expected_public_event_stream=bundle.public_event_stream,
        private_replay_evidence=private, submitted_gameplay_actions=bundle.submitted_gameplay_actions,
        replay_executor=classic7_replay_executor(config))
    with pytest.raises(ArtifactValidationError, match="winner mismatch"):
        audit_terminal_bundle(replace(bundle, private_replay_evidence=private), runtime_config=config)


def test_cli_terminal_audit_and_semantic_validation_require_owning_collection(native_bundle, tmp_path, capsys):
    import yaml
    from werewolf.cli import main, validate_artifact
    from werewolf.artifact_io import canonical_json_bytes
    _, config, collection = native_bundle
    profile, runtime = tmp_path / "storage.json", tmp_path / "runtime.yaml"
    profile.write_bytes(canonical_json_bytes({"artifact_root": str(collection.path.parent)}))
    runtime.write_text(yaml.safe_dump(config))
    destination = tmp_path / "audit"
    prefix = ["--storage-profile", str(profile)]
    assert main(prefix + ["audit-canonical-terminals", "--collection", str(collection.path),
        "--runtime-config", str(runtime), "--destination", str(destination)]) == 0
    digest = capsys.readouterr().out.strip()
    with pytest.raises(ValueError, match="requires --collection"):
        validate_artifact(destination, runtime_config=runtime)
    assert main(prefix + ["validate-artifact", str(destination), "--collection", str(collection.path),
        "--runtime-config", str(runtime)]) == 0
    assert capsys.readouterr().out.strip() == digest


@pytest.mark.parametrize("change", ["truncated", "after_terminal", "payload", "actor", "outcomes", "public_history", "roles", "config"])
def test_native_audit_rejects_drift(native_bundle, change):
    bundle, config, _ = native_bundle
    actions = bundle.submitted_gameplay_actions
    replay = bundle.private_replay_evidence.replay_inputs.to_value()
    if change == "truncated":
        replay["submitted_actions"] = replay["submitted_actions"][:-1]
        bundle = replace(bundle, submitted_gameplay_actions=actions[:-1], private_replay_evidence=_private(bundle, replay_inputs=replay))
    elif change == "after_terminal":
        replay["submitted_actions"].append(replay["submitted_actions"][-1])
        bundle = replace(bundle, submitted_gameplay_actions=actions + (actions[-1],), private_replay_evidence=_private(bundle, replay_inputs=replay))
    elif change == "payload":
        from werewolf.canonical_collection.frozen_json import freeze_json_value
        bundle = replace(bundle, submitted_gameplay_actions=(replace(actions[0], action_payload=freeze_json_value(["skill_wolf", -1])), *actions[1:]))
    elif change == "actor":
        bundle = replace(bundle, submitted_gameplay_actions=(replace(actions[0], actor_id="player7"), *actions[1:]))
    elif change == "outcomes":
        bundle = replace(bundle, submitted_gameplay_actions=(replace(actions[0], resulting_public_event_ids=("made-up",)), *actions[1:]))
    elif change == "public_history":
        bundle = replace(bundle, public_event_stream=replace(bundle.public_event_stream, digest="0" * 64))
    elif change == "roles":
        roles = dict(bundle.private_replay_evidence.role_assignment)
        roles["player1"] = "Villager"
        bundle = replace(bundle, private_replay_evidence=replace(bundle.private_replay_evidence, role_assignment=tuple(roles.items())))
    else:
        config = {**config, "env_config": {**config["env_config"], "n_player": 8}}
    with pytest.raises((ArtifactValidationError, ValueError)):
        audit_terminal_bundle(bundle, runtime_config=config)


def test_controlled_one_speech_fixture_is_not_a_native_completed_game(tmp_path):
    config = normalize_runtime_config(new_config())
    plan, claim, env, agents, audit, recorder, _ = _runtime()
    recorder.runtime_configuration = config
    run_game(env, agents, ROLES, canonical_recorder=recorder, call_audit=audit)
    bundle = publish_canonical_game_bundle(tmp_path / recorder.game_id, plan=plan, claim=claim,
        evidence=recorder.complete_evidence(), replay_executor=_replay_executor())
    with pytest.raises(ArtifactValidationError, match="native terminal"):
        audit_terminal_bundle(bundle, runtime_config=config)
