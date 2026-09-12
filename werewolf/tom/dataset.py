"""Full-prefix numeric materialization and the unique belief target semantics."""

from dataclasses import dataclass, fields, replace

import torch

from werewolf.canonical_collection.public_history import PLAYER_IDS, PUBLIC_PHASES
from werewolf.canonical_collection.speech import V1_ACTIONS
from werewolf.development_publication import PublicGameView
from werewolf.structured_history import STRUCTURED_TOKEN_TYPES, plan_structured_history


@dataclass(frozen=True)
class ExperimentCapacity:
    max_seq_len: int

    def __post_init__(self):
        if type(self.max_seq_len) is not int or self.max_seq_len < 1:
            raise ValueError("capacity must be a positive integer")


@dataclass(frozen=True)
class PublicTensors:
    event_ids: torch.Tensor
    source_ids: torch.Tensor
    action_ids: torch.Tensor
    target_ids: torch.Tensor
    day_ids: torch.Tensor
    phase_ids: torch.Tensor
    attention_mask: torch.Tensor

    def kwargs(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def stack(cls, samples):
        return cls(**{f.name: torch.stack([getattr(s, f.name) for s in samples]) for f in fields(cls)})


@dataclass(frozen=True)
class CanonicalToMSample:
    game_id: str
    boundary_id: str
    prefix_digest: str
    plan_digest: str
    token_count: int
    public: PublicTensors
    q: torch.Tensor
    label_observed: torch.Tensor
    observer_alive: torch.Tensor


def belief_target(observer: int, support: tuple[str, ...], status: str | None):
    if type(observer) is not int or not 0 <= observer < 7:
        raise ValueError("observer must be a zero-based seat")
    q = torch.zeros(7, dtype=torch.float32)
    if status in ("error", None):
        if support:
            raise ValueError("failed/missing observation cannot have support")
        return q, False
    if status != "success":
        raise ValueError("unsupported belief observation status")
    if any(p not in PLAYER_IDS for p in support) or len(set(support)) != len(support):
        raise ValueError("support must be a canonical player set")
    indices = [PLAYER_IDS.index(p) for p in support]
    if observer in indices:
        raise ValueError("self-suspicion is forbidden")
    if indices != sorted(indices):
        raise ValueError("support must be seat-ordered")
    indices = indices or [j for j in range(7) if j != observer]
    q[indices] = 1 / len(indices)
    return q, True


def tensorize_public_pre(prefix, capacity: ExperimentCapacity):
    """Full validated PRE to public tensors, without observation/target access."""
    plan = plan_structured_history(prefix)
    if plan.token_count > capacity.max_seq_len:
        raise ValueError("complete PRE prefix exceeds experiment capacity")
    arrays = {name: torch.zeros(capacity.max_seq_len, dtype=torch.long) for name in (
        "event_ids", "source_ids", "action_ids", "target_ids", "day_ids", "phase_ids")}
    for i, token in enumerate(plan.tokens):
        arrays["event_ids"][i] = STRUCTURED_TOKEN_TYPES.index(token.token_type) + 1
        arrays["source_ids"][i] = 0 if token.source is None else PLAYER_IDS.index(token.source) + 1
        arrays["target_ids"][i] = 0 if token.target is None else PLAYER_IDS.index(token.target) + 1
        arrays["action_ids"][i] = 0 if token.action is None else V1_ACTIONS.index(token.action) + 1
        arrays["day_ids"][i] = token.day
        arrays["phase_ids"][i] = PUBLIC_PHASES.index(token.phase)
    attention = torch.arange(capacity.max_seq_len) < plan.token_count
    return plan, PublicTensors(**arrays, attention_mask=attention)


class CanonicalToMDataset:
    def __init__(self, public_view: PublicGameView, game_ids, capacity: ExperimentCapacity):
        if not isinstance(public_view, PublicGameView) or not isinstance(capacity, ExperimentCapacity):
            raise TypeError("Dataset requires public publication and experiment capacity")
        game_ids = tuple(game_ids)
        if not game_ids or len(set(game_ids)) != len(game_ids) or not set(game_ids) <= set(public_view.game_ids):
            raise ValueError("Dataset game identities must be a unique publication partition")
        self.samples = []
        for game_id in game_ids:
            game = public_view.load_game(game_id)
            observations = {o.observation_id: o for o in game.belief_observations}
            for prefix in game.authoritative_pre_prefixes:
                plan, public = tensorize_public_pre(prefix, capacity)
                q = torch.zeros(7, 7)
                observed = torch.zeros(7, dtype=torch.bool)
                alive = torch.tensor([p in prefix.alive_observer_ids for p in PLAYER_IDS])
                for link in prefix.belief_observation_links:
                    o = observations[link.observation_id]
                    seat = PLAYER_IDS.index(o.observer_id)
                    q[seat], label = belief_target(seat, o.suspicion_support, o.status.value)
                    observed[seat] = label
                if torch.any(alive & ~observed):
                    raise ValueError("publication has unobserved alive observer")
                self.samples.append(CanonicalToMSample(
                    game_id, prefix.boundary_id, prefix.prefix_digest, plan.plan_digest,
                    plan.token_count, public, q, observed, alive))
        if not self.samples:
            raise ValueError("Dataset partition has no PRE boundaries")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def rotate(sample: CanonicalToMSample, shift: int, eligibility: torch.Tensor | None = None):
    """Permute the two player axes and ordinary references; never reinterpret roles."""
    if type(shift) is not int or not 0 <= shift < 7:
        raise ValueError("rotation shift must be in 0..6")
    def refs(ids):
        return torch.where(ids == 0, ids, (ids - 1 + shift) % 7 + 1)
    rotated = replace(sample,
        public=replace(sample.public, source_ids=refs(sample.public.source_ids), target_ids=refs(sample.public.target_ids)),
        q=torch.roll(sample.q, (shift, shift), (-2, -1)),
        label_observed=torch.roll(sample.label_observed, shift, -1),
        observer_alive=torch.roll(sample.observer_alive, shift, -1))
    if eligibility is None:
        return rotated
    if eligibility.dtype != torch.bool or eligibility.shape != (7,):
        raise ValueError("eligibility must contain seven booleans")
    return rotated, torch.roll(eligibility, shift, -1)
