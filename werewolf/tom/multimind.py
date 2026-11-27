"""MultiMind-Text 的公开序列、句末代理监督与七座位 Reasoner。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, isfinite
from pathlib import Path
from statistics import fmean

import torch
from torch import nn

from werewolf.artifact_io import (
    canonical_json_bytes, open_artifact_envelope, publish_tensor_state, read_artifact_file, sha256_bytes,
)
from werewolf.artifact_io.canonical import require_sha256
from werewolf.artifact_io.tensor_state import decode_tensor_container
from werewolf.canonical_collection.pre import AuthoritativePREPrefix
from werewolf.canonical_collection.public_history import (
    PLAYER_IDS, PUBLIC_PHASES, PublicEventHistory,
)
from werewolf.canonical_collection.speech import (
    V1_ACTIONS, V1SpeechAction, V1SpeechAnnotation, validate_v1_annotation_binding,
)
from werewolf.canonical_collection.trajectory_evidence import BeliefObservation
from werewolf.development_publication import VerifiedDevelopmentPublication
from werewolf.structured_history import plan_public_history
from werewolf.tom.dataset import ExperimentCapacity, PublicTensors, belief_target
from werewolf.tom.population import select_primary_population


_SEATS = len(PLAYER_IDS)
_PLAYER_IDS = {player: index for index, player in enumerate(PLAYER_IDS, start=1)}
_ACTION_IDS = {action: index for index, action in enumerate(V1_ACTIONS, start=1)}
_EVENT_IDS = {event: index for index, event in enumerate((
    "BOS", "turn_start", "SPEECH_END", *(f"phase_change:{phase}" for phase in PUBLIC_PHASES),
    "vote_result", "vote", "exile_result", "exiled_player", "death_announcement", "dead_player",
), start=1)}
_REASONER_GRAPH = {"hidden_size": 512, "num_layers": 8, "num_heads": 8, "dropout": 0.1}
_CHECKPOINT_TYPE = "multimind_reasoner"
_CHECKPOINT_SCHEMA = "classic7_multimind_reasoner_v1"


@dataclass(frozen=True)
class MultiMindToken:
    subject: str | None
    predicate: str
    target: str | None
    day: int
    phase: str


@dataclass(frozen=True)
class MultiMindHistory:
    """以当前发言者的 turn_start 结束的完整公开序列。"""

    tokens: tuple[MultiMindToken, ...]


def _speech_tokens(speaker, actions, day, phase):
    triples = tuple(MultiMindToken(a.subject, a.action, a.object, day, phase) for a in actions)
    return (*triples, MultiMindToken(speaker, "SPEECH_END", None, day, phase))


def encode_history(
    history: PublicEventHistory, annotations: Sequence[V1SpeechAnnotation],
) -> MultiMindHistory:
    """复用公开历史的事件展开，只消费冻结的 V1 标注与公开事件。"""
    if not isinstance(history, PublicEventHistory):
        raise TypeError("MultiMind requires frozen public history")
    if not history.events or history.events[-1].event_type != "turn_start":
        raise ValueError("MultiMind history must end at the current turn_start")
    actions_by_event = {}
    for annotation in annotations:
        validate_v1_annotation_binding(annotation, history)
        if not annotation.is_successful:
            raise ValueError("failed V1 annotation cannot enter MultiMind history")
        if annotation.event_id in actions_by_event:
            raise ValueError("duplicate V1 annotation in MultiMind history")
        actions_by_event[annotation.event_id] = annotation.actions

    plan = plan_public_history(history, actions_by_event, boundary_digest=history.digest)
    first = plan.tokens[0]
    tokens = [MultiMindToken(None, "BOS", None, first.day, first.phase)]
    for token in plan.tokens:
        if token.token_type == "public_speech":
            tokens.extend(_speech_tokens(token.source, actions_by_event[token.event_id], token.day, token.phase))
        elif token.token_type != "speech_action":
            predicate = f"phase_change:{token.phase}" if token.token_type == "phase_change" else token.token_type
            tokens.append(MultiMindToken(token.source, predicate, token.target, token.day, token.phase))
    return MultiMindHistory(tuple(tokens))


def _tensorize(tokens, capacity, *, pad):
    count = len(tokens)
    limit = capacity.max_seq_len
    if count > limit:
        raise ValueError(f"MultiMind sequence needs {count} tokens; capacity is {limit}")
    length = limit if pad else count
    rows = []
    for token in tokens:
        action_id = _ACTION_IDS.get(token.predicate, 0)
        rows.append((
            0 if action_id else _EVENT_IDS[token.predicate],
            0 if token.subject is None else _PLAYER_IDS[token.subject],
            action_id,
            0 if token.target is None else _PLAYER_IDS[token.target],
            token.day, PUBLIC_PHASES.index(token.phase),
        ))
    rows.extend([(0,) * 6] * (length - count))
    columns = torch.tensor(rows, dtype=torch.long).t().unbind()
    return PublicTensors(*columns, attention_mask=torch.arange(length) < count)


def encode_candidate(
    history: MultiMindHistory, speaker: str, actions: Sequence[V1SpeechAction],
    capacity: ExperimentCapacity,
) -> PublicTensors:
    """用真实发言的句末编码追加候选；角色相关的行动掩码由 Planner 决定。"""
    if not history.tokens:
        raise ValueError("MultiMind candidate requires a nonempty public history")
    turn = history.tokens[-1]
    if turn.predicate != "turn_start" or turn.subject != speaker:
        raise ValueError("MultiMind candidate speaker differs from current turn")
    actions = tuple(actions)
    if len(actions) > 3:
        raise ValueError("MultiMind candidate can contain at most three actions")
    if any(not isinstance(action, V1SpeechAction) for action in actions):
        raise TypeError("MultiMind candidates require V1SpeechAction values")
    if any(action.subject != speaker for action in actions):
        raise ValueError("MultiMind candidate action subject differs from speaker")
    tokens = (*history.tokens, *_speech_tokens(speaker, actions, turn.day, turn.phase))
    return _tensorize(tokens, capacity, pad=False)


@dataclass(frozen=True)
class MultiMindSample:
    game_id: str
    attempt_id: str
    source_boundary_id: str
    speech_event_id: str
    boundary_id: str  # 监督报告所属的下一 PRE
    public: PublicTensors
    q: torch.Tensor
    label_observed: torch.Tensor
    observer_alive: torch.Tensor


@dataclass(frozen=True)
class MultiMindTrainingSamples:
    by_game: dict[str, tuple[MultiMindSample, ...]]
    masks: dict[str, dict[str, list[bool]]]
    excluded: dict[tuple[str, str], int]  # (公开阶段, 原因)
    statistics: dict


def _post_speech_targets(
    prefix: AuthoritativePREPrefix, observations: dict[str, BeliefObservation],
):
    q = torch.zeros(_SEATS, _SEATS)
    observed = torch.zeros(_SEATS, dtype=torch.bool)
    empty = torch.zeros_like(observed)
    alive = torch.tensor([seat in prefix.alive_observer_ids for seat in PLAYER_IDS])
    for link in prefix.belief_observation_links:
        observation = observations.get(link.observation_id)
        if observation is None:
            continue
        if (observation.boundary_id != prefix.boundary_id
                or observation.prefix_digest != prefix.prefix_digest
                or observation.observer_id != link.observer_id
                or not observation.observer_alive
                or observation.day != prefix.public_temporal_state.day
                or observation.phase != prefix.public_temporal_state.phase.value):
            raise ValueError("MultiMind label does not belong to its next PRE observer")
        seat = _PLAYER_IDS[link.observer_id] - 1
        q[seat], observed[seat] = belief_target(seat, observation.suspicion_support, observation.status.value)
        empty[seat] = observed[seat] and not observation.suspicion_support
    return q, observed, alive, empty


def prepare_training_samples(
    publication: VerifiedDevelopmentPublication, game_ids: Sequence[str],
    capacity: ExperimentCapacity | None,
) -> MultiMindTrainingSamples:
    """提取已核验整局样本；capacity=None 时只统计，不物化训练序列。"""
    if not isinstance(publication, VerifiedDevelopmentPublication):
        raise TypeError("MultiMind training requires a verified Development publication")
    public = publication.public_view
    game_ids = tuple(game_ids)
    if not game_ids or len(set(game_ids)) != len(game_ids) or not set(game_ids) <= set(public.game_ids):
        raise ValueError("MultiMind game identities must be a unique Development partition")
    members = {row["game_id"]: row for row in publication.manifest["development_games"]}
    eligibility = select_primary_population(publication)
    by_game, masks, excluded = {}, {}, Counter()
    phase_counts = {phase: Counter() for phase in ("discussion", "pk_discussion")}
    predicates = Counter(dict.fromkeys(V1_ACTIONS, 0))
    prefix_lengths, realized_lengths = Counter(), Counter()
    supervised_games = 0
    max_training_length = 0

    for game_id in game_ids:
        game = public.load_game(game_id)
        member = members[game_id]
        attempt_id = member["attempt_id"]
        if game.bundle_digest != member["bundle_digest"] or any(
            o.game_id != game_id or o.attempt_id != attempt_id for o in game.belief_observations
        ):
            raise ValueError("MultiMind game or attempt identity mismatch")
        observations = {o.observation_id: o for o in game.belief_observations}
        annotations = {a.event_id: a for a in game.speech_annotations_v1}
        predicates.update(action.action for annotation in game.speech_annotations_v1 for action in annotation.actions)
        events = game.public_event_stream.events
        prefixes = game.authoritative_pre_prefixes
        samples, game_masks = [], {}
        supervised = False

        for current, following in zip(prefixes, (*prefixes[1:], None), strict=True):
            history = current.public_event_history
            state = history.current_state
            phase = state.phase.value
            offset = len(history.events)
            if current.game_id != game_id or history.events != events[:offset]:
                raise ValueError("MultiMind PRE does not belong to its game history")
            if offset >= len(events):
                raise ValueError("MultiMind PRE has no realized public speech")
            speech = events[offset]
            if speech.event_type != "public_speech" or speech.speaker != current.current_speaker:
                raise ValueError("MultiMind PRE does not precede its speaker's public speech")
            annotation = annotations.get(speech.event_id)
            if annotation is None:
                raise ValueError("missing V1 annotation for MultiMind training speech")
            validate_v1_annotation_binding(annotation, game.public_event_stream)
            if not annotation.is_successful:
                raise ValueError("failed V1 annotation cannot become a training speech")
            prefix = encode_history(history, current.v1_annotations)
            suffix = _speech_tokens(speech.speaker, annotation.actions, state.day, phase)
            prefix_length = len(prefix.tokens)
            realized_length = prefix_length + len(suffix)
            prefix_lengths[prefix_length] += 1
            realized_lengths[realized_length] += 1
            counts = phase_counts[phase]
            counts["speech_boundaries"] += 1

            # 真实流中同阶段的下一轮发言不存在时，当前发言是阶段末位。
            if offset + 1 == len(events) or events[offset + 1].temporal_state != state:
                excluded[phase, "last_speaker"] += 1
                continue
            # 只接受 turn_start -> public_speech -> turn_start。公开阶段相邻，
            # 且没有其他可执行步骤，所以中间也没有私有夜间技能。
            if (following is None or following.public_temporal_state != state
                    or following.step_index <= current.step_index
                    or events[offset + 1].event_type != "turn_start"
                    or following.public_event_history.events != events[:offset + 2]):
                excluded[phase, "not_adjacent"] += 1
                continue
            if (following.game_id != game_id or current.alive_observer_ids != following.alive_observer_ids
                    or following.current_speaker != events[offset + 1].speaker):
                raise ValueError("MultiMind adjacent PRE observer or game identity mismatch")
            if following.v1_annotations != (*current.v1_annotations, annotation):
                raise ValueError("MultiMind adjacent PRE V1 annotations do not match the realized speech")
            counts["strict_pairs"] += 1

            q, observed, alive, empty = _post_speech_targets(following, observations)
            mask = eligibility.mask(game_id, following.boundary_id)
            if torch.any(mask & ~alive) or not torch.equal(mask, eligibility.mask(game_id, current.boundary_id)):
                raise ValueError("MultiMind adjacent PRE eligibility mismatch")
            effective = mask & observed
            if not effective.any():
                excluded[phase, "no_effective_observer"] += 1
                continue
            counts["supervised_speeches"] += 1
            counts["effective_observer_rows"] += int(effective.sum())
            counts["successful_empty_reports"] += int((effective & empty).sum())
            supervised = True
            max_training_length = max(max_training_length, realized_length)
            if capacity is None:
                continue
            tensors = _tensorize((*prefix.tokens, *suffix), capacity, pad=True)
            samples.append(MultiMindSample(game_id, attempt_id, current.boundary_id, speech.event_id,
                                           following.boundary_id, tensors, q, observed, alive))
            game_masks[following.boundary_id] = mask.tolist()
        if samples:
            by_game[game_id] = tuple(samples)
            masks[game_id] = game_masks
        supervised_games += supervised
    fields = ("speech_boundaries", "strict_pairs", "supervised_speeches", "effective_observer_rows",
              "successful_empty_reports")
    return MultiMindTrainingSamples(by_game, masks, dict(excluded), {
        "games": len(game_ids), "supervised_games": supervised_games,
        "phases": {phase: {**{field: counts[field] for field in fields},
                          **{reason: excluded[phase, reason] for reason in
                             ("last_speaker", "not_adjacent", "no_effective_observer")}}
                   for phase, counts in phase_counts.items()},
        "v1_predicate_counts": dict(predicates),
        "prefix_lengths": {str(length): count for length, count in sorted(prefix_lengths.items())},
        "realized_speech_sequence_lengths": {str(length): count for length, count in sorted(realized_lengths.items())},
        "max_supervised_sequence_length": max_training_length,
        "max_candidate_sequence_length": max(prefix_lengths) + 4 if prefix_lengths else 0,
    })


class MultiMindReasoner(nn.Module):
    """独立因果 Transformer，从最后一个 SPEECH_END 读出七座位矩阵。"""

    def __init__(self, capacity: ExperimentCapacity, *, hidden_size=_REASONER_GRAPH["hidden_size"],
                 num_layers=_REASONER_GRAPH["num_layers"], num_heads=_REASONER_GRAPH["num_heads"],
                 dropout=_REASONER_GRAPH["dropout"]):
        super().__init__()
        if not isinstance(capacity, ExperimentCapacity):
            raise TypeError("MultiMind requires an explicit sequence capacity")
        if type(num_layers) is not int or num_layers < 1:
            raise ValueError("MultiMind requires a positive layer count")
        self.capacity = capacity
        self.subject_embedding = nn.Embedding(_SEATS + 1, hidden_size, padding_idx=0)
        self.predicate_embedding = nn.Embedding(len(_ACTION_IDS) + len(_EVENT_IDS) + 1, hidden_size, padding_idx=0)
        self.target_embedding = nn.Embedding(_SEATS + 1, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(capacity.max_seq_len, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=num_heads, dim_feedforward=4 * hidden_size,
                activation=nn.GELU(approximate="tanh"), dropout=dropout, batch_first=True, norm_first=True,
            ) for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_size, _SEATS * _SEATS)
        self.register_buffer("diagonal", torch.eye(_SEATS, dtype=torch.bool), persistent=False)

    def _hidden_states(self, event_ids, source_ids, action_ids, target_ids, attention_mask):
        length = event_ids.size(1)
        positions = torch.arange(length, device=event_ids.device)
        predicates = torch.where(event_ids == 0, action_ids, len(_ACTION_IDS) + event_ids)
        hidden = self.input_dropout(self.input_norm(
            self.subject_embedding(source_ids) + self.predicate_embedding(predicates)
            + self.target_embedding(target_ids) + self.position_embedding(positions)
        ))
        causal_mask = torch.ones(length, length, dtype=torch.bool, device=event_ids.device).triu(1)
        for layer in self.layers:
            hidden = layer(hidden, src_mask=causal_mask, src_key_padding_mask=~attention_mask)
        return hidden

    def forward(self, event_ids, source_ids, action_ids, target_ids, day_ids, phase_ids, attention_mask):
        """返回 [batch, 7, 7] 行对数概率，供既有按游戏加权的损失使用。"""
        if event_ids.ndim != 2 or any(t.shape != event_ids.shape for t in (
            source_ids, action_ids, target_ids, day_ids, phase_ids, attention_mask,
        )):
            raise ValueError("MultiMind inputs must share [batch, sequence] shape")
        if event_ids.size(1) > self.capacity.max_seq_len:
            raise ValueError("MultiMind input exceeds model sequence capacity")
        if attention_mask.dtype != torch.bool:
            raise ValueError("MultiMind attention mask must be boolean")
        lengths = attention_mask.sum(dim=1)
        if not len(lengths) or (lengths == 0).any() or (attention_mask[:, 1:] & ~attention_mask[:, :-1]).any():
            raise ValueError("MultiMind requires nonempty right-padded sequences")
        readout = lengths - 1
        batch = torch.arange(len(lengths), device=event_ids.device)
        if (event_ids[batch, readout] != _EVENT_IDS["SPEECH_END"]).any():
            raise ValueError("MultiMind readout must be at SPEECH_END")
        hidden = self._hidden_states(event_ids, source_ids, action_ids, target_ids, attention_mask)
        logits = self.head(hidden[batch, readout]).reshape(-1, _SEATS, _SEATS)
        return logits.masked_fill(self.diagonal, -torch.inf).log_softmax(dim=-1)

    @torch.inference_mode()
    def score_candidate(self, history: MultiMindHistory, speaker: str, actions: Sequence[V1SpeechAction]):
        """在 eval 模式返回句末 [7, 7] 概率矩阵，避免搜索时的 Dropout。"""
        if self.training:
            raise ValueError("MultiMind candidate scoring requires eval mode")
        public = encode_candidate(history, speaker, actions, self.capacity)
        device = self.head.weight.device
        inputs = {key: value.unsqueeze(0).to(device) for key, value in public.kwargs().items()}
        return self(**inputs)[0].exp()


def _development_partitions(publication, validation_fold, capacity):
    folds = publication.public_view.fold_manifest
    if type(validation_fold) is not int or not 0 <= validation_fold < folds.fold_count:
        raise ValueError("MultiMind requires an existing Development validation fold")
    all_games = set(publication.public_view.game_ids)
    validation_ids = sorted(folds.folds[validation_fold].game_ids)
    if not validation_ids or not set(validation_ids) < all_games:
        raise ValueError("MultiMind requires nonempty whole-game training and validation partitions")
    training_ids = sorted(all_games - set(validation_ids))
    source = {
        "publication_id": publication.public_view.publication_id,
        "publication_digest": publication.manifest_digest,
        "split_digest": publication.manifest.get("split_manifest_digest"),
        "fold_manifest_digest": folds.manifest_digest,
        "validation_fold": validation_fold,
        "training_game_ids": training_ids,
        "validation_game_ids": validation_ids,
    }
    return (source, prepare_training_samples(publication, training_ids, capacity),
            prepare_training_samples(publication, validation_ids, capacity))


def inspect_multimind_development(publication, *, validation_fold):
    """以训练相同的配对和掩码口径统计 Development 容量及覆盖。"""
    source, training, validation = _development_partitions(publication, validation_fold, None)
    return {"source": source, "training": training.statistics, "validation": validation.statistics}


def _checkpoint_model_record(capacity):
    return {
        **_REASONER_GRAPH, "max_seq_len": capacity.max_seq_len,
        "vocabulary": {"seat_ids": _PLAYER_IDS, "v1_action_ids": _ACTION_IDS, "event_ids": _EVENT_IDS,
                       "public_phases": list(PUBLIC_PHASES), "padding_id": 0},
    }


def multimind_device(device):
    """解析调用方显式选择的实际计算设备。"""
    try:
        parsed = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"invalid MultiMind device: {device!r}") from error
    if parsed.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("MultiMind requires a real CPU, CUDA or MPS device")
    return parsed


def open_multimind_checkpoint(path, expected_digest):
    """核对选中权重的 Manifest、模型图和实际编码词表。"""
    require_sha256(expected_digest, "MultiMind checkpoint digest")
    checkpoint = open_artifact_envelope(path, expected_artifact_type=_CHECKPOINT_TYPE,
                                       expected_schema_version=_CHECKPOINT_SCHEMA)
    if checkpoint.manifest_digest != expected_digest:
        raise ValueError("MultiMind checkpoint digest mismatch")
    record = checkpoint.manifest
    model = record.get("model")
    if not isinstance(model, dict):
        raise ValueError("MultiMind checkpoint is missing model configuration")
    capacity = ExperimentCapacity(model.get("max_seq_len"))
    if canonical_json_bytes(model) != canonical_json_bytes(_checkpoint_model_record(capacity)):
        raise ValueError("MultiMind checkpoint model configuration or vocabulary mismatch")
    training = record.get("training")
    if not isinstance(training, dict) or set(record["file_table"]) != {"tensors.bin"}:
        raise ValueError("MultiMind checkpoint requires an optimizer-selected state and tensor payload")
    losses = training.get("validation_losses")
    optimizer_steps = training.get("optimizer_steps")
    if type(optimizer_steps) is not int or optimizer_steps < 1 or not isinstance(losses, list) or not losses:
        raise ValueError("MultiMind checkpoint is missing Development selection evidence")
    previous = 0
    for row in losses:
        if (not isinstance(row, dict) or type(row.get("step")) is not int or not previous < row["step"] <= optimizer_steps
                or type(row.get("loss")) not in (float, int) or not isfinite(row["loss"]) or row["loss"] < 0):
            raise ValueError("invalid MultiMind Development selection evidence")
        previous = row["step"]
    best = min(losses, key=lambda row: row["loss"])
    if previous != optimizer_steps or (training.get("selected_step"), training.get("selected_validation_loss")) != (
        best["step"], best["loss"],
    ):
        raise ValueError("MultiMind checkpoint does not match the selected Development step")
    source = training.get("source")
    if not isinstance(source, dict):
        raise ValueError("MultiMind checkpoint lacks a Development source")
    require_sha256(source.get("publication_digest"), "MultiMind Development publication digest")
    require_sha256(source.get("fold_manifest_digest"), "MultiMind Development fold digest")
    return checkpoint


def load_multimind_checkpoint(path, expected_digest, *, device):
    """原样恢复已选权重，再绑定指定设备和 eval 模式。"""
    from werewolf.tom.state import restore_model_tensors

    checkpoint = open_multimind_checkpoint(path, expected_digest)
    tensors = decode_tensor_container(checkpoint.manifest["tensor_container"],
                                      read_artifact_file(checkpoint, "tensors.bin"), payload_path="tensors.bin")
    capacity = ExperimentCapacity(checkpoint.manifest["model"]["max_seq_len"])
    device = multimind_device(device)
    with torch.random.fork_rng(devices=[]):
        model = MultiMindReasoner(capacity, **_REASONER_GRAPH).cpu().float()
    restore_model_tensors(model, tensors)
    return model.to(device).eval()


@torch.inference_mode()
def _validation_loss(model, validation, device):
    from werewolf.tom.scoring import game_balanced_cross_entropy

    model.eval()
    losses = []
    for game_id, samples in validation.by_game.items():
        public = PublicTensors.stack([sample.public for sample in samples])
        logp = model(**{key: value.to(device) for key, value in public.kwargs().items()})
        q = torch.stack([sample.q for sample in samples]).to(device)
        mask = torch.tensor([validation.masks[game_id][sample.boundary_id] for sample in samples])
        mask &= torch.stack([sample.label_observed for sample in samples])
        losses.append(game_balanced_cross_entropy([(logp, q, mask.to(device))]).item())
    return fmean(losses)


def train_multimind_reasoner(publication, destination, *, validation_fold, max_seq_len, learning_rate,
                             game_batch_size, optimizer_steps, validate_every, seed, device):
    """复用席位置换和优化循环，只发布 held-out 损失最低的一份权重。"""
    from werewolf.tom.protocol import training_schedule
    from werewolf.tom.state import model_tensor_values
    from werewolf.tom.training import fit_steps

    if any(type(n) is not int or n < 1 for n in (game_batch_size, optimizer_steps, validate_every)):
        raise ValueError("MultiMind training requires positive batch, step and validation settings")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("MultiMind training requires an explicit seed in [0, 2**63)")
    if type(learning_rate) not in (float, int) or not isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("MultiMind training requires a positive finite learning rate")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"MultiMind checkpoint already exists: {destination}")
    device = multimind_device(device)
    capacity = ExperimentCapacity(max_seq_len)
    source, training, validation = _development_partitions(publication, validation_fold, capacity)
    if not training.by_game or not validation.by_game:
        raise ValueError("MultiMind requires effective supervision in both Development partitions")
    needed = max(data.statistics["max_candidate_sequence_length"] for data in (training, validation))
    if max_seq_len < needed:
        raise ValueError(f"MultiMind Development candidate needs {needed} tokens; capacity is {max_seq_len}")
    cycle_steps = len(PLAYER_IDS) * ceil(len(training.by_game) / game_batch_size)
    rotation_schedule = training_schedule(
        seed, validation_fold, training.by_game, ceil(optimizer_steps / cycle_steps), game_batch_size,
    )
    schedule = {"optimizer_steps": optimizer_steps, "batches": rotation_schedule["batches"][:optimizer_steps]}

    torch.manual_seed(seed)
    model = MultiMindReasoner(capacity, **_REASONER_GRAPH).float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    evaluations = []
    best_state, best_step, best_loss = None, None, float("inf")

    def select_checkpoint(step, _logs):
        nonlocal best_state, best_step, best_loss
        if step % validate_every and step != optimizer_steps:
            return
        loss = _validation_loss(model, validation, device)
        if not isfinite(loss):
            raise ValueError("nonfinite MultiMind Development validation loss")
        evaluations.append({"step": step, "loss": loss})
        if loss < best_loss:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_step, best_loss = step, loss
        model.train()

    fit_steps(model, optimizer, training.by_game, training.masks, schedule, device=device,
              learning_rate=learning_rate, start=0, logs=[], gate=lambda: None, on_step=select_checkpoint)
    model.load_state_dict(best_state, strict=True)
    return publish_tensor_state(destination, manifest_fields={
        "artifact_type": _CHECKPOINT_TYPE, "schema_version": _CHECKPOINT_SCHEMA,
        "model": _checkpoint_model_record(capacity),
        "training": {
            "source": source, "population": "non_wolf_alive", "objective": "game_balanced_masked_cross_entropy",
            "training_coverage": training.statistics, "validation_coverage": validation.statistics,
            "supervised_training_game_ids": sorted(training.by_game),
            "supervised_validation_game_ids": sorted(validation.by_game), "optimizer": "AdamW",
            "learning_rate": learning_rate, "game_batch_size": game_batch_size, "optimizer_steps": optimizer_steps,
            "validate_every": validate_every, "seed": seed, "device": str(device), "torch_version": str(torch.__version__),
            "schedule_version": rotation_schedule["schedule_version"],
            "batches_sha256": sha256_bytes(canonical_json_bytes(schedule["batches"])),
            "multimind_source_sha256": sha256_bytes(Path(__file__).read_bytes()),
            "selected_step": best_step, "selected_validation_loss": best_loss, "validation_losses": evaluations,
        },
    }, tensors=model_tensor_values(model))
