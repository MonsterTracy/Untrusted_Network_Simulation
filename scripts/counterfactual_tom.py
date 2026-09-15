"""Intervention-only semantic continuations; never canonical evidence.

Phase order is derived from validated public history and the frozen Classic7
seat-order cyclic-rotation rule. No caller-supplied future order is accepted.
"""

from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import torch

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection.pre import AuthoritativePREPrefix, validate_authoritative_pre_prefix
from werewolf.canonical_collection.public_history import PLAYER_IDS, PUBLIC_PHASES
from werewolf.canonical_collection.speech import V1_ACTIONS
from werewolf.structured_history import (
    STRUCTURED_TOKEN_TYPES, StructuredTokenDescriptor, plan_structured_history,
)
from werewolf.tom.dataset import ExperimentCapacity, PublicTensors, tensorize_public_pre
from werewolf.tom.final_capacity import derived_capacity, validate_final_pre


CONSUMER_SCHEMA_VERSION = "counterfactual_tom_consumer_v1"
CONSUMER_IMPLEMENTATION_VERSION = "counterfactual_tom_consumer_2"
BUILDER_VERSION = "same_phase_semantic_continuation_v2"
CONTINUATION_SCHEMA_VERSION = "counterfactual_speech_continuation_v2"
QUEUE_RULE_VERSION = "classic7_public_cyclic_seat_order_v1"


class UnsupportedOpportunityError(ValueError):
    """There is no supported, publicly derived same-phase next speech PRE."""


class PlanningAction(str, Enum):
    ACCUSE_WOLF = "ACCUSE_WOLF"
    CLEAR = "CLEAR"
    SUPPORT = "SUPPORT"
    OPPOSE = "OPPOSE"
    SELF_DEFEND = "SELF_DEFEND"
    NO_COMMITMENT = "NO_COMMITMENT"


_SEMANTICS = {
    PlanningAction.ACCUSE_WOLF: "point_as_werewolf",
    PlanningAction.CLEAR: "point_as_non_werewolf",
    PlanningAction.SUPPORT: "support",
    PlanningAction.OPPOSE: "oppose",
    PlanningAction.SELF_DEFEND: "point_as_non_werewolf",
    PlanningAction.NO_COMMITMENT: "no_commitment",
}


@dataclass(frozen=True, slots=True)
class CounterfactualSpeechAction:
    speaker: str
    action: PlanningAction
    target: str | None = None

    def __post_init__(self):
        if self.speaker not in PLAYER_IDS:
            raise ValueError("speaker must be a canonical player")
        if not isinstance(self.action, PlanningAction):
            raise ValueError("unsupported planning action")
        if self.action in (PlanningAction.SELF_DEFEND, PlanningAction.NO_COMMITMENT):
            if self.target is not None:
                raise ValueError("targetless planning action requires target=None")
        elif self.target not in PLAYER_IDS or self.target == self.speaker:
            raise ValueError("target must be a non-self canonical player")


def derive_public_phase_speaker_order(parent_pre):
    """Derive the frozen runtime's cyclic seat order from public evidence only.

    Normal discussion uses living seats. PK uses the preceding complete public
    normal ballot's positive maximum tie and requires an empty exile result.
    The first observed turn reveals the random rotation; all observed turns
    must match its prefix. The caller still supplies authentic real PRE evidence.
    """
    validate_authoritative_pre_prefix(parent_pre)
    state = parent_pre.public_temporal_state
    phase = state.phase.value
    if phase not in ("discussion", "pk_discussion"):
        raise UnsupportedOpportunityError("parent phase is not discussion/pk_discussion")
    events = parent_pre.public_event_history.events
    alive = set(PLAYER_IDS)
    for event in events:
        if event.event_type in ("death_announcement", "exile_result"):
            if not set(event.affected_players) <= alive:
                raise ValueError("public elimination repeats a dead player")
            alive.difference_update(event.affected_players)
    if alive != set(parent_pre.alive_observer_ids):
        raise ValueError("PRE alive players disagree with public eliminations")
    phase_start = max(i for i, event in enumerate(events) if event.event_type == "phase_change")
    phase_events = events[phase_start + 1:]
    if any(e.event_type not in ("turn_start", "public_speech") for e in phase_events):
        raise ValueError("unexpected public event in discussion")
    observed = tuple(e.speaker for e in phase_events if e.event_type == "turn_start")
    if not observed:
        raise ValueError("missing first public turn_start")
    candidates = tuple(p for p in PLAYER_IDS if p in alive)
    if phase == "pk_discussion":
        previous_start = max((i for i in range(phase_start)
                              if events[i].event_type == "phase_change"), default=-1)
        segment = events[previous_start:phase_start] if previous_start >= 0 else ()
        if (tuple(e.event_type for e in segment) != ("phase_change", "vote_result", "exile_result")
                or any(e.temporal_state.day != state.day
                       or e.temporal_state.phase.value != "vote" for e in segment)):
            raise ValueError("PK requires preceding same-day public vote_result and exile_result")
        ballot, exile = segment[1:]
        if exile.affected_players:
            raise ValueError("PK requires empty public exile result")
        if {v.voter for v in ballot.votes} != alive:
            raise ValueError("PK public ballot must cover every alive voter")
        if any(v.target is not None and (v.target not in alive or v.target == v.voter)
               for v in ballot.votes):
            raise ValueError("PK public ballot contains illegal target")
        counts = Counter(v.target for v in ballot.votes if v.target is not None)
        if not counts:
            raise ValueError("all-abstention public ballot cannot produce PK")
        maximum = max(counts.values())
        candidates = tuple(p for p in PLAYER_IDS if counts[p] == maximum)
        if len(candidates) < 2:
            raise ValueError("public ballot does not establish a highest-vote tie")
    if observed[0] not in candidates:
        raise ValueError("first public speaker is not a phase candidate")
    start = candidates.index(observed[0])
    order = candidates[start:] + candidates[:start]
    if observed != order[:len(observed)]:
        raise ValueError("observed public turn order violates cyclic seat-order rule")
    return order


def _derive_next_speaker(parent):
    order = derive_public_phase_speaker_order(parent)
    index = order.index(parent.current_speaker)
    if index == len(order) - 1:
        raise UnsupportedOpportunityError("current speaker is the last phase speaker")
    return order[index + 1]


@dataclass(frozen=True, slots=True)
class CounterfactualContinuation:
    """Semantic descriptors only. This is NOT an AuthoritativePREPrefix."""

    schema_version: str
    consumer_implementation_version: str
    counterfactual_builder_version: str
    parent_prefix_digest: str
    queue_rule_version: str
    speaker: str
    action: str
    target: str | None
    next_speaker: str
    day: int
    phase: str
    token_count: int
    tokens: tuple[StructuredTokenDescriptor, ...]
    continuation_digest: str


def _validate_opportunity(parent, candidate):
    state = parent.public_temporal_state
    if not isinstance(candidate, CounterfactualSpeechAction):
        raise TypeError("candidate must be CounterfactualSpeechAction")
    candidate.__post_init__()
    if candidate.speaker != parent.current_speaker:
        raise ValueError("wrong current speaker")
    next_speaker = _derive_next_speaker(parent)
    alive = set(parent.alive_observer_ids)
    events = parent.public_event_history.events
    if candidate.target is not None and candidate.target not in alive:
        raise ValueError("target is not alive")
    if candidate.action in (PlanningAction.SUPPORT, PlanningAction.OPPOSE):
        # Day-wide, not phase-wide: earlier normal discussion qualifies in PK.
        spoken_today = {e.speaker for e in events
                        if e.event_type == "public_speech" and e.temporal_state.day == state.day}
        if candidate.target not in spoken_today:
            raise ValueError("support/oppose target has not spoken on the current day")

    return next_speaker


def build_continuation(parent, candidate, *, capacity):
    """Validate a real PRE, then append exactly three semantic descriptors."""
    if not isinstance(parent, AuthoritativePREPrefix):
        raise TypeError("parent must be AuthoritativePREPrefix")
    if not isinstance(capacity, ExperimentCapacity):
        raise TypeError("capacity must be ExperimentCapacity")
    plan = plan_structured_history(parent)
    validate_final_pre(parent, derived_capacity(capacity.max_seq_len))
    next_speaker = _validate_opportunity(parent, candidate)
    if plan.token_count + 3 > capacity.max_seq_len:
        raise ValueError("counterfactual exceeds complete sequence capacity; truncation forbidden")
    day, phase = parent.public_temporal_state.day, parent.public_temporal_state.phase.value
    semantic_target = candidate.speaker if candidate.action is PlanningAction.SELF_DEFEND else candidate.target
    start = plan.token_count
    event_index = len(parent.public_event_history.events)
    # These IDs label hypothetical token descriptors, never public events.
    stem = f"hypothetical:{parent.prefix_digest}"
    additions = (
        StructuredTokenDescriptor(start, "public_speech", candidate.speaker, None, None,
                                  day, phase, f"{stem}:speech", event_index, 0),
        StructuredTokenDescriptor(start + 1, "speech_action", candidate.speaker,
                                  _SEMANTICS[candidate.action], semantic_target,
                                  day, phase, f"{stem}:speech", event_index, 1),
        StructuredTokenDescriptor(start + 2, "turn_start", next_speaker, None, None,
                                  day, phase, f"{stem}:next", event_index + 1, 0),
    )
    values = dict(schema_version=CONTINUATION_SCHEMA_VERSION,
                  consumer_implementation_version=CONSUMER_IMPLEMENTATION_VERSION,
                  counterfactual_builder_version=BUILDER_VERSION,
                  parent_prefix_digest=parent.prefix_digest,
                  queue_rule_version=QUEUE_RULE_VERSION,
                  speaker=candidate.speaker, action=candidate.action.value, target=candidate.target,
                  next_speaker=next_speaker, day=day, phase=phase,
                  token_count=start + 3, tokens=plan.tokens + additions)
    record = {**values, "tokens": [t.to_record() for t in values["tokens"]]}
    return CounterfactualContinuation(**values,
        continuation_digest=sha256_bytes(canonical_json_bytes(record)))


def tensorize_counterfactual(parent, candidate, *, capacity):
    """Return independent tensors; no arbitrary continuation/tensor input API.

    The real prefix uses official tensorization. Only the appended descriptors
    are encoded here, using the official vocabularies (no copied numeric IDs).
    """
    continuation = build_continuation(parent, candidate, capacity=capacity)
    _, public = tensorize_public_pre(parent, capacity)
    for token in continuation.tokens[-3:]:
        i = token.token_index
        public.event_ids[i] = STRUCTURED_TOKEN_TYPES.index(token.token_type) + 1
        public.source_ids[i] = 0 if token.source is None else PLAYER_IDS.index(token.source) + 1
        public.action_ids[i] = 0 if token.action is None else V1_ACTIONS.index(token.action) + 1
        public.target_ids[i] = 0 if token.target is None else PLAYER_IDS.index(token.target) + 1
        public.day_ids[i] = token.day
        public.phase_ids[i] = PUBLIC_PHASES.index(token.phase)
        public.attention_mask[i] = True
    return continuation, public


@dataclass(frozen=True, slots=True)
class ConsumerProvenance:
    consumer_schema_version: str
    consumer_implementation_version: str
    consumer_source_digest: str
    parent_experiment_digest: str
    parent_final_seal_digest: str
    condition: str
    checkpoint_digest: str
    counterfactual_builder_version: str

    def to_record(self):
        return asdict(self)


class CounterfactualToMConsumer:
    """A NEW inference contract over a model loaded by the unchanged predictor.

    Construction invokes all original seal/runtime/checkpoint checks. No model
    injection or unchecked tensor inference is exposed. Nothing is published.
    """

    def __init__(self, experiment, condition):
        from werewolf.tom.final_evaluation import SealedFinalPredictor
        # The original loader initializes CPU parameters before loading weights.
        # Preserve the caller's CPU RNG without altering any validation gate.
        with torch.random.fork_rng(devices=[]):
            self._predictor = SealedFinalPredictor(experiment, condition)
        self._capacity = ExperimentCapacity(experiment.config.max_seq_len)
        self.provenance = ConsumerProvenance(
            CONSUMER_SCHEMA_VERSION, CONSUMER_IMPLEMENTATION_VERSION,
            sha256_bytes(Path(__file__).read_bytes()), experiment.digest,
            self._predictor.seal["record_digest"], condition,
            self._predictor.checkpoint.manifest_digest, BUILDER_VERSION)

    def log_probabilities(self, parent, candidate):
        _, public = tensorize_counterfactual(parent, candidate, capacity=self._capacity)
        inputs = PublicTensors.stack([public])
        with torch.inference_mode():
            logp = self._predictor.model(**{
                k: v.to(self._predictor.experiment.config.device) for k, v in inputs.kwargs().items()
            })[0].cpu()
        diagonal = torch.eye(7, dtype=torch.bool)
        if (logp.shape != (7, 7) or not torch.isneginf(logp[diagonal]).all()
                or not torch.isfinite(logp[~diagonal]).all()
                or not torch.allclose(logp.exp().sum(-1), torch.ones(7, dtype=logp.dtype), atol=1e-6)):
            raise ValueError("counterfactual prediction violates fixed non-self simplex")
        return logp

    def probabilities(self, parent, candidate):
        return self.log_probabilities(parent, candidate).exp()
