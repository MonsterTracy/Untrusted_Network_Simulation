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


def candidate(name="ACCUSE_WOLF"):
    target = None if name in ("SELF_DEFEND", "NO_COMMITMENT") else "player3"
    return cf.CounterfactualSpeechAction("player1", cf.PlanningAction(name), target)


def tensorize(parent, action=None, **kwargs):
    options = dict(capacity=CAPACITY)
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
            cf.build_continuation(wrong, candidate(), capacity=CAPACITY)
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
            tensorize(bad)


@pytest.mark.parametrize("phase", ["discussion", "pk_discussion"])
def test_public_order_and_next_are_derived(phase):
    p = PublicWitness(phase).pre()
    expected = P[2:] + P[:2] if phase == "discussion" else (P[2], P[0], P[1])
    assert cf.derive_public_phase_speaker_order(p) == expected
    assert tensorize(p)[0].next_speaker == "player2"
    assert tensorize(p)[0].queue_rule_version == cf.QUEUE_RULE_VERSION


@pytest.mark.parametrize("first", P)
def test_normal_first_public_turn_determines_every_rotation(first):
    w = PublicWitness()
    w.events = w.events[:3]
    w.annotations = []
    w.event("turn_start", speaker=first)
    p = w.pre()
    index = P.index(first)
    expected = P[index:] + P[:index]
    assert cf.derive_public_phase_speaker_order(p) == expected
    action = cf.CounterfactualSpeechAction(first, cf.PlanningAction.NO_COMMITMENT)
    assert tensorize(p, action)[0].next_speaker == expected[1]


def test_missing_first_public_turn_rejected_by_pre_contract():
    w = PublicWitness()
    parent = w.pre()
    history = freeze_public_event_history(w.events[:3])
    broken = replace(parent, public_event_history=history, public_event_digest=history.digest)
    with pytest.raises(ValueError, match="turn_start"):
        cf.derive_public_phase_speaker_order(broken)


@pytest.mark.parametrize("field,value", [("next_speaker", None), ("next_speaker", "player1"),
    ("next_speaker", "player4"), ("next_speaker", "player8"),
    ("speaker_order", P), ("queue_evidence", object())])
def test_caller_cannot_override_future_order(field, value):
    with pytest.raises(TypeError, match="unexpected keyword"):
        tensorize(PublicWitness().pre(), **{field: value})
    for method in (cf.CounterfactualToMConsumer.log_probabilities,
                   cf.CounterfactualToMConsumer.probabilities):
        assert set(inspect.signature(method).parameters) == {"self", "parent", "candidate"}
    assert not hasattr(cf, "PublicQueueEvidence")


@pytest.mark.parametrize("phase", ["discussion", "pk_discussion"])
def test_last_speaker_unsupported(phase):
    w = PublicWitness(phase)
    w.speech("player1", "no_commitment", None, "我暂不表态。")
    w.event("turn_start", speaker="player2")
    c = cf.CounterfactualSpeechAction("player2", cf.PlanningAction.NO_COMMITMENT)
    with pytest.raises(cf.UnsupportedOpportunityError, match="last phase speaker"):
        tensorize(w.pre(), c)


def test_dead_players_excluded_and_public_alive_consistency():
    w = PublicWitness(alive=P[:-1])
    assert cf.derive_public_phase_speaker_order(w.pre()) == P[2:-1] + P[:2]
    w.alive = P  # Contradict public elimination without changing annotation bindings.
    with pytest.raises(ValueError, match="alive players disagree"):
        tensorize(w.pre())


def test_observed_order_cannot_declare_arbitrary_rotation_suffix():
    w = PublicWitness()
    w.events = w.events[:4]
    w.annotations = []
    p = w.pre()
    assert cf.derive_public_phase_speaker_order(p) == P[2:] + P[:2]
    c = cf.CounterfactualSpeechAction(P[2], cf.PlanningAction.CLEAR, P[0])
    assert tensorize(p, c)[0].next_speaker == P[3]
    w.speech(P[2], "no_commitment", None, "我暂不表态。")
    w.event("turn_start", speaker=P[4])  # skips publicly determined P4
    with pytest.raises(ValueError, match="cyclic seat-order"):
        cf.derive_public_phase_speaker_order(w.pre())


def test_no_env_or_queue_dependency():
    import ast
    tree = ast.parse(Path(cf.__file__).read_text())
    assert not any(isinstance(n, ast.Attribute) and n.attr == "speech_queue" for n in ast.walk(tree))
    assert not any(isinstance(n, ast.ImportFrom) and n.module and "env" in n.module.split(".")
                   for n in ast.walk(tree))
    assert tuple(inspect.signature(cf.derive_public_phase_speaker_order).parameters) == ("parent_pre",)


@pytest.mark.parametrize("malformation", ["missing_vote", "missing_exile", "duplicate_vote",
    "incomplete_voters", "all_abstain", "unique_winner", "nonempty_exile", "self_vote",
    "noncandidate_first"])
def test_pk_insufficient_or_malformed_public_evidence_fails(malformation):
    from copy import deepcopy
    w = PublicWitness("pk_discussion")
    start = next(i for i, e in enumerate(w.events)
                 if e.get("phase") == "pk_discussion")
    w.events = w.events[:start + 2]  # retain only the first PK turn
    w.annotations = [a for a in w.annotations if a.event_index < start]
    vote = next(e for e in w.events if e["event_type"] == "vote_result")
    if malformation.startswith("missing_"):
        kind = "vote_result" if malformation == "missing_vote" else "exile_result"
        w.events = [e for e in w.events if e["event_type"] != kind]
    elif malformation == "duplicate_vote":
        w.events.insert(w.events.index(vote) + 1, deepcopy(vote))
    elif malformation == "incomplete_voters":
        vote["votes"].pop()
    elif malformation == "all_abstain":
        for ballot in vote["votes"]:
            ballot["target"] = None
    elif malformation == "unique_winner":
        for ballot in vote["votes"]:
            ballot["target"] = None if ballot["voter"] == P[0] else P[0]
    elif malformation == "nonempty_exile":
        next(e for e in w.events if e["event_type"] == "exile_result")["exiled_players"] = [P[6]]
        w.alive = P[:-1]
    elif malformation == "self_vote":
        vote["votes"][0]["target"] = P[0]
    else:
        w.events[-1]["speaker"] = P[3]
    # Only events after the prior real speech annotations move; their bindings
    # remain intact. These are explicit negative test fixtures, not production.
    for i, event in enumerate(w.events):
        event["event_index"], event["event_id"] = i, f"e{i}"
    parent = w.pre()
    validate_authoritative_pre_prefix(parent)
    with pytest.raises(ValueError):
        cf.derive_public_phase_speaker_order(parent)


@pytest.mark.parametrize("action", [cf.PlanningAction.SUPPORT, cf.PlanningAction.OPPOSE])
def test_same_day_normal_speech_remains_eligible_in_pk(action):
    w = PublicWitness("pk_discussion")
    p = w.pre()
    # P4 spoke in normal discussion, is not a PK speaker, and is still alive.
    current_phase_speakers = {e.speaker for e in p.public_event_history.events
                              if e.event_type == "public_speech"
                              and e.temporal_state.phase.value == "pk_discussion"}
    assert P[3] not in current_phase_speakers
    c = cf.CounterfactualSpeechAction(P[0], action, P[3])
    continuation, _ = tensorize(p, c)
    assert continuation.tokens[-2].target == P[3]
    assert continuation.tokens[-2].action == action.value.lower()


@pytest.mark.parametrize("action", [cf.PlanningAction.SUPPORT, cf.PlanningAction.OPPOSE])
def test_previous_day_speech_does_not_satisfy_spoken_today(action):
    w = PublicWitness()
    for player in P[:2]:
        if w.events[-1].get("speaker") != player:
            w.event("turn_start", speaker=player)
        w.speech(player, "no_commitment", None, "我暂不表态。")
    w.event("phase_change", day=1, phase="vote")
    w.event("vote_result", votes=[{"voter": p, "target": None} for p in P])
    w.event("exile_result", exiled_players=[])
    w.event("phase_change", day=1, phase="night")
    w.event("death_announcement", dead_players=[])
    w.event("phase_change", day=2, phase="discussion")
    w.event("turn_start", speaker=P[0])
    parent = w.pre()
    assert cf.derive_public_phase_speaker_order(parent) == P
    assert any(e.event_type == "public_speech" and e.speaker == P[2]
               for e in parent.public_event_history.events)
    with pytest.raises(ValueError, match="not spoken on the current day"):
        tensorize(parent, cf.CounterfactualSpeechAction(P[0], action, P[2]))


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
            logp = consumer.log_probabilities(p, candidate(name))
            w.speech("player1", semantic, target, text)
            w.event("turn_start", speaker="player2")
            assert torch.equal(logp, real_predictor.log_probabilities(w.pre()))
            assert torch.equal(parent_before, real_predictor.log_probabilities(p))
            assert logp.shape == (7, 7) and torch.isneginf(logp.diagonal()).all()
            assert torch.equal(logp.exp(), consumer.probabilities(p, candidate(name)))
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
    first = consumer.log_probabilities(p, candidate())
    second = consumer.log_probabilities(p, candidate())
    assert torch.equal(first, second)
    assert all(torch.equal(captured[0][k], captured[1][k]) for k in captured[0])
    continuation, _ = tensorize(p)
    with pytest.raises(TypeError, match="AuthoritativePREPrefix"):
        consumer._predictor.log_probabilities(continuation)
