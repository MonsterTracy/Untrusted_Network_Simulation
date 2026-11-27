"""MultiMind-Text 的本地合成见证，使用现有 V1 Parser 与 PRE 合同。"""

import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tests.tom.test_counterfactual_consumer import ScriptedBackend
from werewolf.artifact_io.tensor_state import decode_tensor_container, encode_tensor_container
from werewolf.canonical_collection.pre import construct_authoritative_pre_prefix
from werewolf.canonical_collection.public_history import PLAYER_IDS, freeze_public_event_history
from werewolf.canonical_collection.speech import (
    V1_ACTIONS, V1_SPEECH_PARSER_VERSION, V1_SPEECH_PROMPT_VERSION,
    V1AnnotationStatus, V1PerceptionAttempt, V1SpeechAction, construct_v1_speech_annotation,
)
from werewolf.canonical_collection.trajectory_evidence import (
    BeliefObservationStatus, construct_belief_observation, construct_call_budget_summary,
    construct_parser_summary,
)
from werewolf.development_publication import (
    PUBLISHED_GAME_SCHEMA_VERSION, PublicationPublicView, PublishedGame, RoleSidecarGame,
    VerifiedDevelopmentPublication, VerifiedRoleSidecar, _development_game_set_digest, assign_development_folds,
)
from werewolf.speech.speech_perceiver import SpeechPerceiver
from werewolf.tom import population
from werewolf.tom.dataset import ExperimentCapacity, PublicTensors, rotate
from werewolf.tom.multimind import (
    MultiMindReasoner, encode_candidate, encode_history, prepare_training_samples,
)
from werewolf.tom.state import model_tensor_values, restore_model_tensors
from werewolf.tom.training import fit_steps


P = PLAYER_IDS
GAME = "synthetic-game"
ATTEMPT = "synthetic-attempt"
CAPACITY = ExperimentCapacity(96)


class PublicWitness:
    def __init__(self, game_id=GAME, attempt_id=ATTEMPT):
        self.game_id, self.attempt_id = game_id, attempt_id
        self.events, self.annotations, self.prefixes, self.observations = [], [], [], []
        self.alive = P[:-1]
        self.event("phase_change", day=0, phase="night")
        self.event("death_announcement", dead_players=[P[-1]])
        self.event("phase_change", day=1, phase="discussion")

    def event(self, event_type, **payload):
        offset = len(self.events)
        self.events.append(dict(event_id=f"e{offset}", event_index=offset, event_type=event_type, **payload))

    def history(self):
        return freeze_public_event_history(self.events)

    def speak(self, speaker, actions=(), *, supports=None):
        self.event("turn_start", speaker=speaker)
        history = self.history()
        index = len(self.events) - 1
        prefix = construct_authoritative_pre_prefix(
            game_id=self.game_id, boundary_id=f"pre-{index}", step_index=index,
            report_trigger_id=f"trigger-{index}", current_speaker=speaker,
            alive_observer_ids=self.alive, public_event_history=history,
            v1_annotations=self.annotations,
            belief_observation_ids_by_observer={seat: f"obs-{index}-{seat}" for seat in self.alive},
        )
        self.prefixes.append(prefix)
        self.observations.extend(self.make_observation(prefix, link, (supports or {}).get(link.observer_id, ()))
                                 for link in prefix.belief_observation_links)
        response = "\n".join(f"{a.subject} | {a.action} | {a.object or 'NONE'}" for a in actions) or "NONE"
        self.annotate_speech(speaker, response)
        return prefix

    def annotate_speech(self, speaker, response):
        state = self.history().current_state
        backend = ScriptedBackend(response)
        perceiver = SpeechPerceiver(backend=backend, model_name="synthetic-oracle")
        audit = perceiver.parse_with_audit(speaker=P.index(speaker) + 1, speech="合成公开发言。",
                                         day=state.day, phase=state.phase.value)
        if audit.parse_status != "ok":
            status = V1AnnotationStatus.ERROR
        else:
            status = V1AnnotationStatus.OK if audit.normalized_actions else V1AnnotationStatus.NO_ACTION
        self.event("public_speech", speaker=speaker, raw_text="合成公开发言。")
        evidence = tuple(V1PerceptionAttempt(
            attempt_index=a["generation_attempt"], call_id=f"call-{len(self.events)}-{i}",
            backend_id="scripted-test-backend", model_id="synthetic-oracle",
            prompt_version=V1_SPEECH_PROMPT_VERSION, parser_version=V1_SPEECH_PARSER_VERSION,
            status=status, raw_response=a["raw_response"],
            error_category=a["error_type"], error_message=a["error_message"],
        ) for i, a in enumerate(audit.generation_attempts))
        self.annotations.append(construct_v1_speech_annotation(
            self.history(), status=status, attempts=evidence,
            actions=tuple(V1SpeechAction(*a) for a in audit.normalized_actions),
        ))

    def make_observation(self, prefix, link, support=(), *, success=True):
        status = BeliefObservationStatus.SUCCESS if success else BeliefObservationStatus.ERROR
        return construct_belief_observation(
            game_id=self.game_id, attempt_id=self.attempt_id, boundary_id=prefix.boundary_id,
            prefix_digest=prefix.prefix_digest, observation_id=link.observation_id,
            observer_id=link.observer_id, observer_alive=True, day=prefix.public_temporal_state.day,
            phase=prefix.public_temporal_state.phase.value, status=status, suspicion_support=support,
            attempts=({"attempt_index": 1, "call_id": f"call-{link.observation_id}", "status": status.value,
                       "response_digest": "1" * 64 if success else None,
                       "error_category": None if success else "synthetic_error",
                       "error_message": None if success else "synthetic report failed"},),
        )

    def publication(self, monkeypatch):
        digest = "1" * 64
        game = PublishedGame(
            schema_version=PUBLISHED_GAME_SCHEMA_VERSION, game_id=self.game_id, bundle_digest=digest,
            public_event_stream=self.history(), authoritative_pre_prefixes=tuple(self.prefixes),
            belief_observations=tuple(self.observations), speech_annotations_v1=tuple(self.annotations),
            call_budget_summary=construct_call_budget_summary(
                configured_call_limit=0, used_calls=0, retry_calls=0, fallback_action_count=0,
                second_speaker_belief_count=0, opaque_call_digests=(),
            ),
            parser_summary=construct_parser_summary(self.annotations), record_digest=digest,
        )
        view = PublicationPublicView(
            publication_id="synthetic-publication", collection_id="synthetic-collection",
            collection_plan_digest=digest, collection_identity_digest=digest,
            development_game_set_digest=digest, games=(game,), fold_manifest=None,
            complete_case_exclusions=(), max_observed_day=game.public_event_stream.max_public_day,
            max_structured_token_count=0, structured_token_planner_version="synthetic",
        )
        sidecar = VerifiedRoleSidecar(
            path=Path("synthetic"), publication_id=view.publication_id, collection_identity_digest=digest,
            development_game_set_digest=digest, sidecar_digest=digest,
            games=(RoleSidecarGame(self.game_id, digest, tuple(zip(P, (
                "Werewolf", "Villager", "Seer", "Witch", "Villager", "Villager", "Werewolf",
            ), strict=True))),),
        )
        monkeypatch.setattr(population, "open_role_sidecar", lambda _publication: sidecar)
        return VerifiedDevelopmentPublication(
            path=Path("synthetic"), manifest_digest=digest,
            manifest={"development_games": [
                {"game_id": self.game_id, "attempt_id": self.attempt_id, "bundle_digest": digest},
            ]},
            public_view=view,
        )

    def training_samples(self, monkeypatch, capacity=CAPACITY):
        return prepare_training_samples(self.publication(monkeypatch), (self.game_id,), capacity)


def test_all_v1_predicates_and_distinct_public_events(monkeypatch):
    witness = PublicWitness()
    actions = tuple(V1SpeechAction(P[0], name, None if name in {"abstain_intent", "no_commitment"} else P[6])
                    for name in V1_ACTIONS)
    witness.speak(P[0], actions)
    witness.speak(P[1])  # 成功解析为空，与显式 no_commitment 不同。
    witness.event("phase_change", day=1, phase="vote")
    witness.event("vote_result", votes=[{"voter": P[0], "target": P[3]}, {"voter": P[1], "target": None}])
    witness.event("exile_result", exiled_players=[P[3]])
    witness.event("phase_change", day=1, phase="night")
    witness.event("death_announcement", dead_players=[])
    witness.event("phase_change", day=2, phase="discussion")
    witness.alive = tuple(seat for seat in witness.alive if seat != P[3])
    prefix = witness.speak(P[2])
    enc = encode_history(prefix.public_event_history, prefix.v1_annotations)
    assert [token.predicate for token in enc.tokens] == [
        "BOS", "phase_change:night", "death_announcement", "dead_player", "phase_change:discussion",
        "turn_start", *V1_ACTIONS, "SPEECH_END", "turn_start", "SPEECH_END",
        "phase_change:vote", "vote_result", "vote", "vote", "exile_result", "exiled_player",
        "phase_change:night", "death_announcement", "phase_change:discussion", "turn_start",
    ]
    assert tuple((t.subject, t.predicate, t.target) for t in enc.tokens[6:20]) == tuple(
        (a.subject, a.action, a.object) for a in actions
    )
    assert (enc.tokens[3].target, enc.tokens[25].subject, enc.tokens[25].target,
            enc.tokens[26].target, enc.tokens[28].target) == (P[6], P[0], P[3], None, P[3])
    assert enc.tokens[-1].day == 2 and enc.tokens[-1].subject == P[2]
    public = witness.training_samples(monkeypatch).by_game[GAME][0].public
    assert public.action_ids[public.action_ids != 0].tolist() == list(range(1, 15))
    assert public.target_ids[6:20].tolist() == [7] * 12 + [0, 0]
    assert (public.event_ids[public.action_ids != 0] == 0).all()
    witness.speak(P[4])
    data = witness.training_samples(monkeypatch)
    with_votes = data.by_game[GAME][-1]
    membership = torch.tensor(data.masks[GAME][with_votes.boundary_id])
    rotated, rotated_membership = rotate(with_votes, 2, membership)
    # 同一置换覆盖 public vote、exile、death 所用的席位以及两条监督轴。
    for index in (3, 25, 26, 28):
        for name in ("source_ids", "target_ids"):
            seat_id = getattr(with_votes.public, name)[index].item()
            assert getattr(rotated.public, name)[index] == ((seat_id + 1) % 7 + 1 if seat_id else 0)
    assert torch.equal(rotated.public.event_ids, with_votes.public.event_ids)
    assert torch.equal(rotated.q, torch.roll(with_votes.q, (2, 2), (-2, -1)))
    assert torch.equal(rotated_membership, torch.roll(membership, 2))


@pytest.mark.parametrize("failure, message", [(False, "coverage mismatch"), (True, "failed V1 annotation")])
def test_missing_and_failed_v1_annotation_are_not_no_action(failure, message):
    witness = PublicWitness()
    witness.event("turn_start", speaker=P[0])
    witness.annotate_speech(P[0], "bad syntax" if failure else "NONE")
    witness.event("turn_start", speaker=P[1])
    with pytest.raises(ValueError, match=message):
        encode_history(witness.history(), witness.annotations if failure else ())


@pytest.mark.parametrize("count", range(4))
def test_candidate_and_real_training_suffix_have_identical_encoding(monkeypatch, count):
    witness = PublicWitness()
    actions = (V1SpeechAction(P[0], "support", P[2]), V1SpeechAction(P[0], "oppose", P[3]),
               V1SpeechAction(P[0], "point_as_seer", P[0]))[:count]
    current = witness.speak(P[0], actions)
    following = witness.speak(P[1])
    history = encode_history(current.public_event_history, current.v1_annotations)
    candidate = encode_candidate(history, P[0], actions, CAPACITY)
    sample, = witness.training_samples(monkeypatch).by_game[GAME]
    length = len(history.tokens) + count + 1
    assert sample.public.attention_mask.sum() == length
    for key, tensor in candidate.kwargs().items():
        torch.testing.assert_close(tensor, sample.public.kwargs()[key][:length])
    assert sample.boundary_id == following.boundary_id and sample.source_boundary_id == current.boundary_id
    assert sample.speech_event_id == witness.annotations[0].event_id and sample.attempt_id == ATTEMPT
    assert candidate.source_ids[-1] == 1 and candidate.target_ids[-1] == 0
    assert candidate.action_ids[-1] == 0 and candidate.event_ids[-1] != 0
    assert (candidate.source_ids < 2).all()


def test_only_adjacent_same_phase_samples_and_last_speaker_counts(monkeypatch):
    witness = PublicWitness()
    ordinary = witness.speak(P[0]), witness.speak(P[1])
    witness.event("phase_change", day=1, phase="vote")
    witness.event("vote_result", votes=[{"voter": P[0], "target": P[1]}, {"voter": P[1], "target": P[0]}])
    witness.event("exile_result", exiled_players=[])
    witness.event("phase_change", day=1, phase="pk_discussion")
    pk = witness.speak(P[0]), witness.speak(P[1])
    witness.event("phase_change", day=1, phase="pk_vote")
    witness.event("vote_result", votes=[{"voter": P[2], "target": P[1]}])
    witness.event("exile_result", exiled_players=[P[1]])
    witness.event("phase_change", day=1, phase="night")
    witness.event("death_announcement", dead_players=[P[4]])
    witness.event("phase_change", day=2, phase="discussion")
    witness.alive = (P[0], P[2], P[3], P[5])
    later = witness.speak(P[0]), witness.speak(P[2])

    data = witness.training_samples(monkeypatch)
    assert [(s.source_boundary_id, s.boundary_id) for s in data.by_game[GAME]] == [
        (a.boundary_id, b.boundary_id) for a, b in (ordinary, pk, later)
    ]
    assert data.excluded == {("discussion", "last_speaker"): 2, ("pk_discussion", "last_speaker"): 1}
    assert data.masks[GAME][later[1].boundary_id] == [False, False, True, True, False, True, False]


def test_intervening_public_speech_is_never_skipped(monkeypatch):
    witness = PublicWitness()
    witness.speak(P[0])
    witness.speak(P[1])
    witness.speak(P[2])
    witness.prefixes.pop(1)  # 合成缺口，实际 Development reader 会拒绝。
    data = witness.training_samples(monkeypatch)
    assert not data.by_game
    assert data.excluded == {("discussion", "not_adjacent"): 1, ("discussion", "last_speaker"): 1}


@pytest.mark.parametrize("mismatch, message", [
    ("game", "observer or game identity mismatch"),
    ("attempt", "game or attempt identity mismatch"),
    ("observers", "observer or game identity mismatch"),
    ("label", "label does not belong"),
])
def test_requires_same_game_attempt_and_observer_identity(monkeypatch, mismatch, message):
    witness = PublicWitness()
    witness.speak(P[0])
    following = witness.speak(P[1])
    if mismatch == "game":
        witness.prefixes[-1] = replace(following, game_id="different-game")
    elif mismatch == "attempt":
        witness.observations[-1] = replace(witness.observations[-1], attempt_id="different-attempt")
    elif mismatch == "observers":
        witness.prefixes[-1] = replace(following, alive_observer_ids=following.alive_observer_ids[:-1])
    elif mismatch == "label":
        witness.observations[-1] = replace(witness.observations[-1], boundary_id="different-boundary")
    with pytest.raises(ValueError, match=message):
        witness.training_samples(monkeypatch)


def test_post_labels_empty_missing_failed_and_private_input_independence(monkeypatch):
    witness = PublicWitness()
    current = witness.speak(P[0], supports={P[2]: (P[1],)})
    following = witness.speak(P[1], supports={P[2]: (P[0], P[6])})
    original, = witness.training_samples(monkeypatch).by_game[GAME]
    assert original.q[2].tolist() == [0.5, 0, 0, 0, 0, 0, 0.5]
    assert original.q[3, 3] == 0 and original.q[3, 6] == pytest.approx(1 / 6)
    assert original.label_observed.tolist() == [True] * 6 + [False]

    old_link = current.belief_observation_links[2]
    witness.observations[2] = witness.make_observation(current, old_link, (P[4],))
    unchanged, = witness.training_samples(monkeypatch).by_game[GAME]
    torch.testing.assert_close(unchanged.q, original.q)

    # 无报告与失败均无监督。后续 PRE 报告改变时输入序列保持相同。
    keep = [obs for obs in witness.observations if obs.observation_id != following.belief_observation_links[3].observation_id]
    failed_link = following.belief_observation_links[-1]
    keep[-1] = witness.make_observation(following, failed_link, success=False)
    witness.observations = keep
    data = witness.training_samples(monkeypatch)
    changed, = data.by_game[GAME]
    assert changed.label_observed.tolist() == [True, True, True, False, True, False, False]
    assert not changed.q[[3, 5, 6]].any()
    assert data.masks[GAME][following.boundary_id] == [False, True, True, True, True, True, False]
    for name, tensor in original.public.kwargs().items():
        torch.testing.assert_close(tensor, changed.public.kwargs()[name])

    witness.observations = [obs for obs in keep if obs.boundary_id == current.boundary_id]
    empty = witness.training_samples(monkeypatch)
    assert not empty.by_game and not empty.masks
    assert empty.excluded["discussion", "no_effective_observer"] == 1


def test_partition_capacity_and_candidate_constraints(monkeypatch):
    witness = PublicWitness()
    current = witness.speak(P[0])
    witness.speak(P[1])
    public = witness.publication(monkeypatch)
    with pytest.raises(ValueError, match="unique Development partition"):
        prepare_training_samples(public, ("outside",), CAPACITY)
    with pytest.raises(TypeError, match="verified Development"):
        prepare_training_samples(public.public_view, (GAME,), CAPACITY)
    history = encode_history(current.public_event_history, current.v1_annotations)
    limit = ExperimentCapacity(len(history.tokens))
    with pytest.raises(ValueError, match="sequence needs .* capacity"):
        encode_candidate(history, P[0], (), limit)
    with pytest.raises(ValueError, match="sequence needs .* capacity"):
        witness.training_samples(monkeypatch, limit)
    action = V1SpeechAction(P[0], "support", P[2])
    with pytest.raises(ValueError, match="at most three"):
        encode_candidate(history, P[0], (action,) * 4, CAPACITY)
    with pytest.raises(ValueError, match="speaker differs from current turn"):
        encode_candidate(history, P[1], (), CAPACITY)
    with pytest.raises(ValueError, match="subject differs from speaker"):
        encode_candidate(history, P[0], (V1SpeechAction(P[1], "support", P[2]),), CAPACITY)
    with pytest.raises(TypeError, match="V1SpeechAction"):
        encode_candidate(history, P[0], ((P[0], "support", P[2]),), CAPACITY)


def test_reasoner_default_cpu_forward(monkeypatch):
    witness = PublicWitness()
    current = witness.speak(P[0])
    witness.speak(P[1])
    capacity = ExperimentCapacity(64)
    sample, = witness.training_samples(monkeypatch, capacity).by_game[GAME]
    model = MultiMindReasoner(capacity).eval()
    assert len(model.layers) == 8 and model.layers[0].self_attn.num_heads == 8
    assert model.head.in_features == 512 and model.input_dropout.p == 0.1
    assert sum(p.numel() for p in model.parameters()) == 25_268_273 + 512 * capacity.max_seq_len
    history = encode_history(current.public_event_history, current.v1_annotations)
    score = model.score_candidate(history, P[0], ())
    assert score.shape == (7, 7) and score.device.type == "cpu"
    assert torch.isfinite(score).all() and (score >= 0).all() and torch.count_nonzero(score.diagonal()) == 0
    assert (score[:-1, -1] > 0).all()  # player7 已死亡，但对应嫌疑列仍然存在。
    torch.testing.assert_close(score.sum(-1), torch.ones(7))
    with torch.inference_mode():
        torch.testing.assert_close(model(**PublicTensors.stack([sample.public]).kwargs())[0].exp(), score)


def test_causal_mask_padding_and_readout(monkeypatch):
    witness = PublicWitness()
    current = witness.speak(P[0], (V1SpeechAction(P[0], "support", P[2]),))
    witness.speak(P[1])
    sample, = witness.training_samples(monkeypatch).by_game[GAME]
    model = MultiMindReasoner(CAPACITY, hidden_size=24, num_layers=2, num_heads=4, dropout=0).eval()
    history = encode_history(current.public_event_history, current.v1_annotations)
    first = encode_candidate(history, P[0], (V1SpeechAction(P[0], "support", P[2]),), CAPACITY)
    second = encode_candidate(history, P[0], (V1SpeechAction(P[0], "oppose", P[4]),), CAPACITY)
    candidates = PublicTensors.stack([first, second])
    inputs = PublicTensors.stack([sample.public])
    dirty_tail = replace(inputs, target_ids=inputs.target_ids.masked_fill(~inputs.attention_mask, 7),
                         source_ids=inputs.source_ids.masked_fill(~inputs.attention_mask, 3))
    with torch.inference_mode():
        hidden = model._hidden_states(candidates.event_ids, candidates.source_ids, candidates.action_ids,
                                      candidates.target_ids, candidates.attention_mask)
        torch.testing.assert_close(hidden[0, :-2], hidden[1, :-2])
        assert not torch.allclose(hidden[0, -1], hidden[1, -1])
        scores = model(**candidates.kwargs()).exp()
        assert scores.shape == (2, 7, 7)
        torch.testing.assert_close(model(**dirty_tail.kwargs()), model(**inputs.kwargs()))
        torch.testing.assert_close(model(**inputs.kwargs())[0].exp(), scores[0])
    for mask in (torch.zeros_like(inputs.attention_mask), ~inputs.attention_mask,
                 inputs.attention_mask.clone().index_fill(1, torch.tensor([1]), False)):
        with pytest.raises(ValueError, match="nonempty right-padded"):
            model(**replace(inputs, attention_mask=mask).kwargs())
    mask = inputs.attention_mask.clone()
    mask[0, int(mask.sum()) - 1] = False
    with pytest.raises(ValueError, match="readout must be at SPEECH_END"):
        model(**replace(inputs, attention_mask=mask).kwargs())


def test_reasoner_reuses_game_balanced_training_and_seat_rotation(monkeypatch):
    witness = PublicWitness()
    current = witness.speak(P[0])
    following = witness.speak(P[1], supports={P[2]: (P[0], P[6])})
    data = witness.training_samples(monkeypatch)
    sample, = data.by_game[GAME]
    eligibility = torch.tensor(data.masks[GAME][following.boundary_id])
    rotated, rotated_mask = rotate(sample, 3, eligibility)
    assert torch.equal(rotated.q, torch.roll(sample.q, (3, 3), (-2, -1)))
    assert torch.equal(rotated_mask, torch.roll(eligibility, 3))
    assert rotated.public.source_ids[sample.public.source_ids == 1].tolist() == [4, 4]
    model = MultiMindReasoner(CAPACITY, hidden_size=24, num_layers=2, num_heads=4)
    history = encode_history(current.public_event_history, current.v1_annotations)
    with pytest.raises(ValueError, match="requires eval mode"):
        model.score_candidate(history, P[0], ())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    old_head = model.head.weight.detach().clone()
    logs = []
    schedule = {"optimizer_steps": 1, "batches": [[{"game_id": GAME, "shift": 3}]]}
    assert fit_steps(model, optimizer, data.by_game, data.masks, schedule,
                     device="cpu", learning_rate=0.001, start=0, logs=logs,
                     gate=lambda: None, on_step=lambda *_: None) == 1
    assert math.isfinite(logs[0]["loss"]) and not torch.equal(old_head, model.head.weight)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

    recovered = MultiMindReasoner(CAPACITY, hidden_size=24, num_layers=2, num_heads=4)
    container, raw = encode_tensor_container(model_tensor_values(model), payload_path="tensors.bin")
    restore_model_tensors(recovered, decode_tensor_container(container, raw, payload_path="tensors.bin"))
    torch.testing.assert_close(recovered.eval().score_candidate(history, P[0], ()),
                               model.eval().score_candidate(history, P[0], ()), rtol=0, atol=0)


@pytest.fixture
def multimind_checkpoint(tmp_path, monkeypatch, capsys):
    """五个只在内存中的 Development 见证，真实训练和张量发布使用临时小模型。"""
    import json

    from werewolf.cli import main
    from werewolf.tom import development_input, multimind, training

    games, members, sidecar_games = [], [], []
    witness = None
    for index in range(5):
        witness = PublicWitness(f"synthetic-game-{index}", f"synthetic-attempt-{index}")
        current = witness.speak(P[0], (V1SpeechAction(P[0], "point_as_werewolf", P[2]),))
        witness.speak(P[1], supports={P[2]: (P[0],)})
        publication = witness.publication(monkeypatch)
        games.extend(publication.public_view.games)
        members.extend(publication.manifest["development_games"])
        sidecar = population.open_role_sidecar(publication)
        sidecar_games.extend(sidecar.games)
    pairs = [(game.game_id, game.bundle_digest) for game in games]
    game_set = _development_game_set_digest(pairs)
    folds = assign_development_folds(pairs, development_game_set_digest=game_set)
    view = replace(publication.public_view, games=tuple(games), fold_manifest=folds,
                   development_game_set_digest=game_set)
    publication = replace(publication, public_view=view, manifest={
        "development_games": members, "split_manifest_digest": "2" * 64,
    })
    monkeypatch.setattr(population, "open_role_sidecar", lambda _p: replace(sidecar, games=tuple(sidecar_games)))

    root = tmp_path.resolve()
    profile = root / "storage.json"
    profile.write_text(json.dumps({"artifact_root": str(root)}), encoding="utf-8")

    def open_publication(path, *, expected_publication_digest, expected_split_digest):
        assert path == root / "publications" / "synthetic"
        assert (expected_publication_digest, expected_split_digest) == (publication.manifest_digest, "2" * 64)
        return publication

    monkeypatch.setattr(development_input, "open_development_input", open_publication)
    for key, value in {"hidden_size": 16, "num_layers": 1, "num_heads": 2}.items():
        monkeypatch.setitem(multimind._REASONER_GRAPH, key, value)

    args = ["--storage-profile", str(profile), "train-multimind", "--publication", "synthetic",
            "--publication-digest", publication.manifest_digest, "--split-digest", "2" * 64,
            "--validation-fold", "0"]
    assert main([*args, "--inspect-only"]) == 0
    inspection = json.loads(capsys.readouterr().out)
    history = encode_history(current.public_event_history, current.v1_annotations)
    validate = multimind._validation_loss
    fit = training.fit_steps
    predictions = []

    def fit_training_only(model, optimizer, by_game, masks, schedule, **kwargs):
        assert set(by_game) == set(masks) == set(inspection["source"]["training_game_ids"])
        assert all(item["game_id"] in by_game and 0 <= item["shift"] < 7
                   for batch in schedule["batches"] for item in batch)
        assert len(schedule["batches"]) == schedule["optimizer_steps"] == 2
        return fit(model, optimizer, by_game, masks, schedule, **kwargs)

    monkeypatch.setattr(training, "fit_steps", fit_training_only)

    def validation_loss(model, validation, device):
        assert set(validation.by_game) == set(folds.folds[0].game_ids)
        assert math.isfinite(validate(model, validation, device))
        predictions.append(model.score_candidate(history, P[0], ()))
        # 固定选择第一步，以覆盖验证最优权重和最后一步权重不同的流程。
        return float(len(predictions))

    monkeypatch.setattr(multimind, "_validation_loss", validation_loss)
    destination = root / "reasoner"
    assert main([*args, "--destination", str(destination), "--max-seq-len", "192", "--learning-rate", "0.001",
                 "--game-batch-size", "1", "--optimizer-steps", "2", "--validate-every", "1", "--seed", "3",
                 "--device", "cpu"]) == 0
    checkpoint_digest = capsys.readouterr().out.strip()
    assert len(predictions) == 2 and not torch.equal(*predictions)
    return destination, checkpoint_digest, history, predictions[0], inspection


def test_multimind_train_cli_stats_holdout_and_checkpoint_roundtrip(multimind_checkpoint):
    from werewolf.tom.multimind import load_multimind_checkpoint, open_multimind_checkpoint

    path, identity, history, expected, inspection = multimind_checkpoint
    source = inspection["source"]
    assert len(source["training_game_ids"]) == 4 and len(source["validation_game_ids"]) == 1
    assert not set(source["training_game_ids"]) & set(source["validation_game_ids"])
    for section, games in (("training", 4), ("validation", 1)):
        counts = inspection[section]
        ordinary = counts["phases"]["discussion"]
        assert counts["games"] == counts["supervised_games"] == games
        assert ordinary["strict_pairs"] == ordinary["last_speaker"] == games
        assert ordinary["effective_observer_rows"] == 5 * games
        assert ordinary["successful_empty_reports"] == 4 * games
        assert len(counts["v1_predicate_counts"]) == 14 and counts["v1_predicate_counts"]["point_as_werewolf"] == games
        assert counts["max_candidate_sequence_length"] == max(map(int, counts["prefix_lengths"])) + 4

    manifest = open_multimind_checkpoint(path, identity).manifest
    assert manifest["model"]["max_seq_len"] == 192 and len(manifest["model"]["vocabulary"]["v1_action_ids"]) == 14
    assert manifest["training"]["selected_step"] == 1 and manifest["training"]["optimizer_steps"] == 2
    assert manifest["training"]["source"] == source and manifest["training"]["training_coverage"] == inspection["training"]
    model = load_multimind_checkpoint(path, identity, device="cpu")
    assert not model.training and next(model.parameters()).device.type == "cpu"
    torch.testing.assert_close(model.score_candidate(history, P[0], ()), expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        load_multimind_checkpoint(path, "a" * 64, device="cpu")


@pytest.mark.parametrize("fault", ("vocabulary", "shape", "missing"))
def test_multimind_checkpoint_rejects_invalid_graph_or_payload(multimind_checkpoint, tmp_path, fault):
    from shutil import copytree

    from werewolf.artifact_io import TensorValue, publish_tensor_state
    from werewolf.tom.multimind import load_multimind_checkpoint, open_multimind_checkpoint

    path, identity, *_ = multimind_checkpoint
    wrong = tmp_path.resolve() / "invalid-reasoner"
    if fault == "missing":
        copytree(path, wrong)
        (wrong / "tensors.bin").unlink()
        with pytest.raises(ValueError):
            load_multimind_checkpoint(wrong, identity, device="cpu")
        return

    model = load_multimind_checkpoint(path, identity, device="cpu")
    tensors = model_tensor_values(model)
    manifest = open_multimind_checkpoint(path, identity).manifest
    fields = {key: manifest[key] for key in ("artifact_type", "schema_version", "model", "training")}
    if fault == "vocabulary":
        fields["model"]["vocabulary"]["padding_id"] = -1
        message = "configuration or vocabulary mismatch"
    else:
        tensors["head.bias"] = TensorValue(tensors["head.bias"].array[:-1], True)
        message = "tensor shape"
    broken = publish_tensor_state(wrong, manifest_fields=fields, tensors=tensors)
    with pytest.raises(ValueError, match=message):
        load_multimind_checkpoint(wrong, broken.artifact.manifest_digest, device="cpu")
