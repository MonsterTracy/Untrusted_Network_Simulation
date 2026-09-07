"""The production path with deterministic external backends, not a pilot collector."""

from types import SimpleNamespace

from tests.canonical_collection.test_game_bundle import _plan
from tests.canonical_collection.test_runtime_game_evidence import _Agent, _Backend, ROLES
from tests.tom.test_experiment import experiment_config


class CompletingAgent(_Agent):
    def act(self, observation):
        if "skill_wolf" in observation["phase"] or "vote" in observation["phase"]:
            return next(action for action in observation["valid_action"] if action[1] > 0)
        return super().act(observation)


def test_runtime_to_collection_publication_and_paired_oof(tmp_path):
    from run_random import eval as run_game
    from werewolf.canonical_collection import collect, V1_SPEECH_PARSER_VERSION, V1_SPEECH_PROMPT_VERSION
    from werewolf.canonical_collection.call_audit import CanonicalCallAudit, audited_backends
    from werewolf.canonical_collection.collector import CanonicalGameProduct
    from werewolf.canonical_collection.production_runtime import _ReplaySpeechPerceiver
    from werewolf.canonical_collection.runtime import CanonicalGameRecorder, PlayingAgentBeliefObservationCollector, make_classic7_replay_executor
    from werewolf.development_publication import open_verified_collection, publish_development
    from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
    from werewolf.speech.private_belief_perceiver import PlayingAgentBeliefReporter
    from werewolf.speech.speech_perceiver import SpeechPerceiver
    from werewolf.tom.experiment import prepare_experiment
    from werewolf.tom.reporting import run_development_oof

    plan = _plan(ordered_seed_pool=(1, 2, 3, 4, 5), target_canonical_success_count=5,
        parser_identity=V1_SPEECH_PARSER_VERSION, prompt_identity=V1_SPEECH_PROMPT_VERSION)
    replay = make_classic7_replay_executor(identity="full-classic7-test-replay-v1",
        environment_factory=lambda seed: WerewolfTextEnvV0(speech_perceiver=_ReplaySpeechPerceiver(), random_seed=seed, log_save_path=None))

    def factory(*, plan, claim):
        audit = CanonicalCallAudit(plan=plan, configured_call_limit=10000)
        backend = audited_backends({"fixture": _Backend()}, audit)["fixture"]
        agents = [CompletingAgent(backend, seat) for seat in range(1, 8)]
        env = WerewolfTextEnvV0(speech_perceiver=SpeechPerceiver(backend=backend, model_name=plan.model_identity),
            random_seed=claim.seed, log_save_path=None)
        recorder = CanonicalGameRecorder(plan=plan, claim=claim, game_id=f"game-{claim.ordinal}",
            belief_collector=PlayingAgentBeliefObservationCollector(plan=plan, reporter=PlayingAgentBeliefReporter(audit_hook=audit), agents=agents),
            call_audit=audit, runtime_configuration={"test_external_backend": "deterministic-v1"})

        def run():
            run_game(env, agents, ROLES, canonical_recorder=recorder, call_audit=audit)
            return CanonicalGameProduct(recorder.complete_evidence(), replay)
        return SimpleNamespace(run=run)

    result = collect(plan=plan, runtime_factory=factory, destination=tmp_path / "collection")
    assert result.ledger_state.canonical_success_count == 5
    collection = open_verified_collection(result.collection_directory, replay_executor=replay)
    publication = publish_development(collection,
        tmp_path / "publications" / "development", publication_id="development").public
    assert publication.manifest["collection_plan_digest"] == plan.plan_digest
    for row, bundle, claim, terminal in zip(publication.manifest["development_games"], collection.games,
            collection.ledger_state.claims, collection.ledger_state.terminals, strict=True):
        assert row["claim_record_digest"] == claim.record_digest
        assert row["terminal_record_digest"] == terminal.record_digest
        assert row["bundle_digest"] == bundle.manifest_digest
    assert publication.public_view.max_observed_day >= 2
    assert all(len(g.authoritative_pre_prefixes) > 1 for g in publication.public_view.games)
    experiment = prepare_experiment(publication, experiment_config(publication.public_view.max_structured_token_count),
        tmp_path / "experiments" / "experiment")
    assert experiment.manifest["publication_digest"] == publication.manifest_digest
    assert experiment.manifest["primary_sidecar_digest"] == publication.manifest["role_sidecar_digest"]
    assert experiment.manifest["fold_manifest_digest"] == publication.public_view.fold_manifest.manifest_digest
    reports = run_development_oof(experiment)
    from werewolf.tom.training import verify_checkpoint_set, verify_terminal
    from werewolf.tom.run_records import read_record
    from werewolf.cli import validate_artifact
    seal = verify_checkpoint_set(experiment)
    assert reports["checkpoint_set_digest"] == seal["record_digest"]
    for checkpoint in seal["checkpoints"]:
        fold, condition = checkpoint["fold"], checkpoint["temporal_condition"]
        terminal = verify_terminal(experiment, fold, condition)
        assert terminal.manifest_digest == checkpoint["checkpoint_digest"]
        assert terminal.manifest["provenance"]["paired_initial_state_digest"] == experiment.fold(fold)["paired_initial_state_digest"]
        prediction = read_record(terminal.path.parent / "prediction_manifest.json")
        for operation in ("primary_development_oof", "all_alive_identifiability_stress"):
            report = read_record(terminal.path.parent / operation / "fold_report.json")
            assert report["checkpoint_digest"] == terminal.manifest_digest
            assert report["prediction_digest"] == prediction["prediction_digest"]
            assert report["checkpoint_set_digest"] == seal["record_digest"]
            assert report["record_digest"] in reports["cells"][f"{condition}/{operation}"]["fold_report_digests"]
    assert validate_artifact(experiment.path) == experiment.digest  # sealed audit is legal
    assert len(reports["cells"]) == 4
    assert len(list(experiment.runs_path.glob("*/*/terminal_checkpoint/manifest.json"))) == 10
    assert all(len(cell["game_scores"]) == 5 for cell in reports["cells"].values())
