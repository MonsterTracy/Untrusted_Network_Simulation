"""Phase 1: synthetic public witnesses, no gameplay integration or remote LLM.

Oracle annotations are created ONLY in tests using actual SpeechPerceiver calls
to a scripted test backend. Production continuation code creates none of them.
"""

from dataclasses import FrozenInstanceError, asdict, fields, replace
import inspect
from pathlib import Path

import pytest
import torch

import scripts.counterfactual_tom as cf
from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection import (
    construct_authoritative_pre_prefix, freeze_public_event_history,
    construct_v1_speech_annotation, V1AnnotationStatus, V1SpeechAction,
    V1PerceptionAttempt, V1_SPEECH_PROMPT_VERSION, V1_SPEECH_PARSER_VERSION,
)
from werewolf.canonical_collection.pre import validate_authoritative_pre_prefix
from werewolf.structured_history import plan_structured_history
from werewolf.speech.speech_perceiver import SpeechPerceiver
from werewolf.tom.dataset import ExperimentCapacity, PublicTensors, tensorize_public_pre


P = tuple(f"player{i}" for i in range(1, 8))
CAPACITY = ExperimentCapacity(96)
# Independent oracle mapping; never use production _SEMANTICS in the oracle.
ORACLE = {
    "ACCUSE_WOLF": ("point_as_werewolf", "player3", "player3 是狼人。"),
    "CLEAR": ("point_as_non_werewolf", "player3", "player3 不是狼人。"),
    "SUPPORT": ("support", "player3", "我支持 player3。"),
    "OPPOSE": ("oppose", "player3", "我反对 player3。"),
    "SELF_DEFEND": ("point_as_non_werewolf", "player1", "我不是狼人。"),
    "NO_COMMITMENT": ("no_commitment", None, "本轮我暂不作明确表态。"),
}


class ScriptedBackend:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class PublicWitness:
    def __init__(self, phase="discussion", alive=P):
        self.events, self.annotations = [], []
        self.alive = alive
        self.event("phase_change", day=0, phase="night")
        self.event("death_announcement", dead_players=[p for p in P if p not in alive])
        self.event("phase_change", day=1, phase="discussion")
        if phase == "pk_discussion":
            for p in alive:
                self.event("turn_start", speaker=p)
                self.speech(p, "no_commitment", None, "本轮我暂不作明确表态。")
            self.event("phase_change", day=1, phase="vote")
            # Three tied players, two votes each and one abstention.
            self.event("vote_result", votes=[{"voter": p, "target": t} for p, t in zip(
                P, (P[1], P[2], P[0], P[0], P[1], P[2], None), strict=True)])
            self.event("exile_result", exiled_players=[])
            self.event("phase_change", day=1, phase="pk_discussion")
        prior = ("player3",) if phase == "pk_discussion" else tuple(p for p in alive if p not in P[:2])
        for p in prior:
            self.event("turn_start", speaker=p)
            self.speech(p, "no_commitment", None, "本轮我暂不作明确表态。")
        self.event("turn_start", speaker="player1")

    def event(self, event_type, **payload):
        self.events.append(dict(event_id=f"e{len(self.events)}", event_index=len(self.events),
                                event_type=event_type, **payload))

    def speech(self, speaker, action, target, text):
        response = f"{speaker} | {action} | {target or 'NONE'}"
        backend = ScriptedBackend(response)
        parser = SpeechPerceiver(backend=backend, model_name="synthetic-oracle")
        history = freeze_public_event_history(self.events)
        audit = parser.parse_with_audit(speaker=P.index(speaker) + 1, speech=text,
            day=history.current_state.day, phase=history.current_state.phase.value)
        assert len(backend.calls) == 1 and audit.parse_status == "ok"
        self.event("public_speech", speaker=speaker, raw_text=text)
        attempts = tuple(V1PerceptionAttempt(
            attempt_index=a["generation_attempt"], call_id=f"test-{len(self.events)}-{n}",
            backend_id="scripted-test-backend", model_id="synthetic-oracle",
            prompt_version=V1_SPEECH_PROMPT_VERSION, parser_version=V1_SPEECH_PARSER_VERSION,
            status=V1AnnotationStatus.OK, raw_response=a["raw_response"],
            error_category=None, error_message=None) for n, a in enumerate(audit.generation_attempts))
        self.annotations.append(construct_v1_speech_annotation(
            freeze_public_event_history(self.events), status=V1AnnotationStatus.OK,
            actions=tuple(V1SpeechAction(*a) for a in audit.normalized_actions), attempts=attempts))

    def pre(self):
        return construct_authoritative_pre_prefix(game_id="synthetic-oracle",
            boundary_id=f"b{len(self.events)}", step_index=len(self.events) - 1,
            report_trigger_id=f"r{len(self.events)}", current_speaker=self.events[-1]["speaker"],
            alive_observer_ids=self.alive, public_event_history=freeze_public_event_history(self.events),
            v1_annotations=tuple(self.annotations),
            belief_observation_ids_by_observer={p: f"synthetic-observation-{len(self.events)}-{p}" for p in self.alive})


def queue(parent):
    order = ("player3", "player1", "player2")
    if parent.public_temporal_state.phase.value == "discussion":
        order = tuple(p for p in parent.alive_observer_ids if p not in P[:2]) + P[:2]
    return cf.PublicQueueEvidence(parent.prefix_digest, parent.public_temporal_state.day,
                                  parent.public_temporal_state.phase.value, order,
                                  "synthetic-public-queue-witness")


def candidate(name="ACCUSE_WOLF"):
    target = None if name in ("SELF_DEFEND", "NO_COMMITMENT") else "player3"
    return cf.CounterfactualSpeechAction("player1", cf.PlanningAction(name), target)


def tensorize(parent, action=None, **kwargs):
    options = dict(next_speaker="player2", queue_evidence=queue(parent), capacity=CAPACITY)
    options.update(kwargs)
    return cf.tensorize_counterfactual(parent, action or candidate(), **options)


@pytest.mark.parametrize("phase", ["discussion", "pk_discussion"])
@pytest.mark.parametrize("name", list(ORACLE))
def test_real_hypothetical_tensor_equivalence(name, phase):
    witness = PublicWitness(phase)
    parent = witness.pre()
    before = canonical_json_bytes(parent.to_record())
    continuation, hypothetical = tensorize(parent, candidate(name))
    semantic, target, text = ORACLE[name]
    witness.speech("player1", semantic, target, text)
    witness.event("turn_start", speaker="player2")
    successor = witness.pre()
    validate_authoritative_pre_prefix(successor)
    _, real = tensorize_public_pre(successor, CAPACITY)
    assert set(real.kwargs()) == {f.name for f in fields(PublicTensors)}
    for key, expected in real.kwargs().items():
        assert torch.equal(hypothetical.kwargs()[key], expected), (name, phase, key)
        assert hypothetical.kwargs()[key].dtype == expected.dtype
    parent_plan = plan_structured_history(parent)
    assert continuation.tokens[:-3] == parent_plan.tokens
    assert continuation.token_count == parent_plan.token_count + 3
    assert [t.token_type for t in continuation.tokens[-3:]] == ["public_speech", "speech_action", "turn_start"]
    assert all((t.day, t.phase) == (1, phase) for t in continuation.tokens[-3:])
    assert continuation.tokens[-2].event_index == continuation.tokens[-3].event_index
    assert continuation.tokens[-2].semantic_index == 1
    assert canonical_json_bytes(parent.to_record()) == before
    assert tensorize(parent, candidate(name))[0] == continuation
    record = asdict(continuation)
    digest = record.pop("continuation_digest")
    assert sha256_bytes(canonical_json_bytes(record)) == digest
    with pytest.raises(FrozenInstanceError):
        continuation.target = "player7"
    # Mutation of a returned tensor cannot contaminate a later invocation.
    hypothetical.event_ids.zero_()
    assert torch.equal(tensorize(parent, candidate(name))[1].event_ids, real.event_ids)


def test_parent_type_digest_phase_and_speaker_fail_closed():
    p = PublicWitness().pre()
    for wrong in (None, {}, p.to_record()):
        with pytest.raises(TypeError, match="AuthoritativePREPrefix"):
            cf.build_continuation(wrong, candidate(), next_speaker="player2",
                                 queue_evidence=queue(p), capacity=CAPACITY)
    with pytest.raises(ValueError, match="digest mismatch"):
        tensorize(replace(p, prefix_digest="0" * 64))
    with pytest.raises(ValueError, match="wrong current speaker"):
        tensorize(p, cf.CounterfactualSpeechAction("player2", cf.PlanningAction.CLEAR, "player3"))
    # The original public validator already forbids a speech turn in vote/night.
    for phase in ("vote", "night"):
        from werewolf.canonical_collection.public_history import PublicPhase
        history = p.public_event_history
        terminal = replace(history.events[-1], temporal_state=replace(history.current_state, phase=PublicPhase(phase)))
        bad = replace(p, public_event_history=replace(history, events=history.events[:-1] + (terminal,)))
        with pytest.raises(ValueError):
            tensorize(bad, queue_evidence=queue(p))


@pytest.mark.parametrize("next_speaker", [None, "player1", "player4", "player8"])
def test_invalid_next_speaker(next_speaker):
    with pytest.raises(cf.UnsupportedOpportunityError):
        tensorize(PublicWitness().pre(), next_speaker=next_speaker)


def test_dead_next_and_last_speaker():
    p = PublicWitness(alive=P[:-1]).pre()
    with pytest.raises(cf.UnsupportedOpportunityError, match="not alive"):
        tensorize(p, next_speaker="player7")
    w = PublicWitness("pk_discussion")
    w.speech("player1", "no_commitment", None, "我暂不表态。")
    w.event("turn_start", speaker="player2")
    p = w.pre()
    c = cf.CounterfactualSpeechAction("player2", cf.PlanningAction.NO_COMMITMENT)
    with pytest.raises(cf.UnsupportedOpportunityError, match="last phase speaker"):
        tensorize(p, c, next_speaker="player1")


def test_queue_binding_order_and_caller_contract():
    p = PublicWitness().pre()
    q = queue(p)
    for change in (dict(parent_prefix_digest="0" * 64), dict(day=2), dict(phase="pk_discussion"),
                   dict(speaker_order=P), dict(speaker_order=("player3", "player1", "player2"))):
        with pytest.raises(ValueError):
            tensorize(p, queue_evidence=replace(q, **change))
    with pytest.raises(TypeError, match="queue evidence"):
        tensorize(p, queue_evidence=None)
    # At the first turn, PRE alone cannot distinguish caller-attested suffixes.
    # This test does not assert that both orders follow the real game's rules.
    witness = PublicWitness()
    witness.events = witness.events[:4]  # first turn_start(player3)
    witness.annotations = []
    p = witness.pre()
    q = queue(p)
    alternate = replace(q, speaker_order=("player3", "player5", "player4", "player6", "player7", "player1", "player2"))
    action = cf.CounterfactualSpeechAction("player3", cf.PlanningAction.CLEAR, "player1")
    c, _ = tensorize(p, action, next_speaker="player5", queue_evidence=alternate)
    assert c.queue_evidence_digest == alternate.digest != q.digest
    with pytest.raises(cf.UnsupportedOpportunityError, match="queue evidence"):
        tensorize(p, action, next_speaker="player5", queue_evidence=q)


@pytest.mark.parametrize("name,target", [("vote_intent", "player3"), ("ACCUSE_WOLF", "player3"),
    (cf.PlanningAction.CLEAR, "player1"), (cf.PlanningAction.CLEAR, "player8"),
    (cf.PlanningAction.SELF_DEFEND, "player1"), (cf.PlanningAction.NO_COMMITMENT, "player3")])
def test_unsupported_action_or_illegal_target(name, target):
    with pytest.raises(ValueError):
        cf.CounterfactualSpeechAction("player1", name, target)


def test_alive_and_today_target_rules_capacity_and_no_private_input():
    p = PublicWitness(alive=P[:-1]).pre()
    with pytest.raises(ValueError, match="not alive"):
        tensorize(p, cf.CounterfactualSpeechAction("player1", cf.PlanningAction.CLEAR, "player7"))
    for action in (cf.PlanningAction.SUPPORT, cf.PlanningAction.OPPOSE):
        with pytest.raises(ValueError, match="not spoken"):
            tensorize(p, cf.CounterfactualSpeechAction("player1", action, "player2"))
    count = plan_structured_history(p).token_count
    with pytest.raises(ValueError, match="capacity"):
        tensorize(p, capacity=ExperimentCapacity(count + 2))
    assert tensorize(p, capacity=ExperimentCapacity(count + 3))[1].attention_mask.all()
    with pytest.raises(TypeError):
        cf.CounterfactualSpeechAction("player1", cf.PlanningAction.CLEAR, "player3", role_truth={})
    assert set(inspect.signature(cf.CounterfactualSpeechAction).parameters) == {"speaker", "action", "target"}
    assert "wolf" not in inspect.signature(cf.CounterfactualToMConsumer.log_probabilities).parameters
    with pytest.raises(TypeError):
        tensorize(p, wolf_team=("player1", "player3"))


def test_hypothetical_cannot_enter_canonical_writer(tmp_path):
    from tests.canonical_collection.test_game_bundle import _fixture
    from werewolf.canonical_collection import publish_canonical_game_bundle
    plan, claim, evidence, replay = _fixture()
    continuation, _ = tensorize(PublicWitness().pre())
    with pytest.raises(TypeError, match="AuthoritativePREPrefix"):
        plan_structured_history(continuation)
    bad = replace(evidence, authoritative_pre_prefixes=(continuation,))
    with pytest.raises((TypeError, ValueError)):
        publish_canonical_game_bundle(tmp_path / "rejected", plan=plan, claim=claim,
                                     evidence=bad, replay_executor=replay)
    assert not (tmp_path / "rejected").exists()


@pytest.fixture(scope="module")
def sealed(tmp_path_factory):
    """Tiny synthetic local lifecycle, never reads/writes existing experiments."""
    from tests.development_publication.test_development_publication import _publication
    from tests.tom.test_experiment import experiment_config
    import werewolf.tom.final_experiment as fe
    from werewolf.tom.final_training import _final_worker, seal_final_models
    root = tmp_path_factory.mktemp("counterfactual-synthetic-seal")
    _, handles = _publication(root / "development")
    # Existing lifecycle tests isolate only the clean-checkout identity. Seal,
    # runtime digest, checkpoint verification, loading and model are all real.
    with pytest.MonkeyPatch.context() as m:
        m.setattr(fe, "actual_clean_revision", lambda: "a" * 40)
        experiment = fe.prepare_final_experiment(handles.public,
            replace(experiment_config(CAPACITY.max_seq_len), game_batch_size=5),
            root / "final/experiments/experiment")
        for condition in experiment.manifest["temporal_conditions"]:
            _final_worker(experiment.path, experiment.digest, condition, False)
        seal_final_models(experiment)
    return experiment


@pytest.mark.parametrize("condition", ["implicit", "explicit_day_phase"])
def test_sealed_inference_matches_real_successor_and_preserves_artifacts(sealed, condition):
    from werewolf.tom.final_evaluation import SealedFinalPredictor
    before = {p: p.read_bytes() for root in (sealed.path, sealed.runs_path)
              for p in root.rglob("*") if p.is_file()}
    consumer = cf.CounterfactualToMConsumer(sealed, condition)
    real_predictor = SealedFinalPredictor(sealed, condition)
    provenance = consumer.provenance.to_record()
    assert set(provenance) == {"consumer_schema_version", "consumer_implementation_version", "consumer_source_digest",
        "parent_experiment_digest", "parent_final_seal_digest", "condition", "checkpoint_digest", "counterfactual_builder_version"}
    assert provenance["consumer_source_digest"] == sha256_bytes(Path(cf.__file__).read_bytes())
    assert provenance["parent_experiment_digest"] == sealed.digest
    assert provenance["parent_final_seal_digest"] == real_predictor.seal["record_digest"]
    assert provenance["checkpoint_digest"] == real_predictor.checkpoint.manifest_digest
    for phase in ("discussion", "pk_discussion"):
        for name, (semantic, target, text) in ORACLE.items():
            w = PublicWitness(phase)
            p = w.pre()
            parent_before = real_predictor.log_probabilities(p)
            logp = consumer.log_probabilities(p, candidate(name), next_speaker="player2", queue_evidence=queue(p))
            w.speech("player1", semantic, target, text)
            w.event("turn_start", speaker="player2")
            assert torch.equal(logp, real_predictor.log_probabilities(w.pre()))
            assert torch.equal(parent_before, real_predictor.log_probabilities(p))
            assert logp.shape == (7, 7) and torch.isneginf(logp.diagonal()).all()
            assert torch.equal(logp.exp(), consumer.probabilities(p, candidate(name), next_speaker="player2", queue_evidence=queue(p)))
            assert torch.allclose(logp.exp().sum(-1), torch.ones(7), atol=1e-6)
    after = {p: p.read_bytes() for root in (sealed.path, sealed.runs_path)
             for p in root.rglob("*") if p.is_file()}
    assert after == before


def test_sealed_gates_run_before_model_loading(sealed, monkeypatch):
    import werewolf.tom.final_evaluation as fe
    calls = []
    for name in ("verify_final_seal", "validate_runtime", "verify_final_terminal", "build_model", "load_model_state"):
        original = getattr(fe, name)
        def spy(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)
        monkeypatch.setattr(fe, name, spy)
    rng = torch.get_rng_state().clone()
    cf.CounterfactualToMConsumer(sealed, "implicit")
    assert torch.equal(torch.get_rng_state(), rng)
    assert calls == ["verify_final_seal", "validate_runtime", "verify_final_terminal", "build_model", "load_model_state"]


@pytest.mark.parametrize("gate", ["verify_final_seal", "validate_runtime", "verify_final_terminal"])
def test_gate_failure_never_loads_model(sealed, monkeypatch, gate):
    import werewolf.tom.final_evaluation as fe
    def reject(*args):
        raise ValueError(f"rejected {gate}")
    def forbidden(*args):
        pytest.fail("model loaded after rejected gate")
    monkeypatch.setattr(fe, gate, reject)
    monkeypatch.setattr(fe, "build_model", forbidden)
    with pytest.raises(ValueError, match=f"rejected {gate}"):
        cf.CounterfactualToMConsumer(sealed, "implicit")


def test_real_runtime_mismatch_rejected(sealed, monkeypatch):
    import werewolf.tom.experiment as experiment
    original = experiment.runtime_provenance
    monkeypatch.setattr(experiment, "runtime_provenance", lambda config:
                        {**original(config), "implementation_digest": "0" * 64})
    with pytest.raises(ValueError, match="runtime environment/source mismatch"):
        cf.CounterfactualToMConsumer(sealed, "implicit")


@pytest.mark.parametrize("corruption", ["missing_seal", "seal_bytes", "seal_digest", "checkpoint_bytes"])
def test_actual_artifact_corruption_rejected(sealed, tmp_path, corruption):
    import shutil
    from werewolf.tom.final_experiment import open_final_experiment
    # Mutate only a disposable COPY of synthetic test artifacts, never a
    # preexisting seal or the shared fixture. No manifest/hash repair is used.
    destination = tmp_path / "copied-final"
    shutil.copytree(sealed.path.parent.parent, destination)
    copied = open_final_experiment(destination / "experiments/experiment")
    seal_path = copied.runs_path / "final_model_seal.json"
    if corruption == "missing_seal":
        seal_path.unlink()
    elif corruption == "seal_bytes":
        seal_path.write_bytes(b"{}")
    elif corruption == "seal_digest":
        import json
        record = json.loads(seal_path.read_bytes())
        record["record_digest"] = "0" * 64
        seal_path.write_bytes(canonical_json_bytes(record))
    else:
        path = copied.runs_path / "implicit/terminal_checkpoint/tensors.bin"
        path.write_bytes(path.read_bytes() + b"corrupt")
    # Preserve the old validator's actual failure: missing record_digest is a
    # KeyError, whereas digest/content rejection is ValueError. Do not repair
    # or wrap the original sealed contract just to standardize test errors.
    expected_error = KeyError if corruption == "seal_bytes" else ValueError
    with pytest.raises(expected_error):
        cf.CounterfactualToMConsumer(copied, "implicit")


def test_model_receives_only_public_tensors(sealed, monkeypatch):
    consumer = cf.CounterfactualToMConsumer(sealed, "implicit")
    original = consumer._predictor.model.forward
    captured = []
    def public_only(**kwargs):
        assert set(kwargs) == {f.name for f in fields(PublicTensors)}
        captured.append({k: v.clone() for k, v in kwargs.items()})
        return original(**kwargs)
    monkeypatch.setattr(consumer._predictor.model, "forward", public_only)
    p = PublicWitness().pre()
    first = consumer.log_probabilities(p, candidate(), next_speaker="player2", queue_evidence=queue(p))
    changed_source = replace(queue(p), source_reference="another-public-source-reference")
    second = consumer.log_probabilities(p, candidate(), next_speaker="player2", queue_evidence=changed_source)
    assert torch.equal(first, second)
    assert all(torch.equal(captured[0][k], captured[1][k]) for k in captured[0])
    continuation, _ = tensorize(p)
    with pytest.raises(TypeError, match="AuthoritativePREPrefix"):
        consumer._predictor.log_probabilities(continuation)
