"""Runtime-safe Phase-2 opportunity assembled from one current wolf PRE."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.phase2_actions import (
    Action, ActionContextV1, ObservationWindowV1, context_from_pre, probe_continuation,
    select_redirect_target,
)
from werewolf.phase2_mapper_runtime import RuntimeInference, validate_final_q


VERSION = "phase2_decision_opportunity_v1"


def _sha256(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value)


class DecisionOpportunityError(ValueError):
    """Current PRE, frozen evidence, or legal actions do not agree."""


@dataclass(frozen=True)
class CandidateEvidenceV1:
    candidate_j: str
    p_tilde: float
    r2_feature_digest: str

    def __post_init__(self):
        if (not math.isfinite(self.p_tilde) or not 0 <= self.p_tilde <= 1
                or not _sha256(self.r2_feature_digest)):
            raise DecisionOpportunityError("invalid candidate evidence")


@dataclass(frozen=True)
class Phase2DecisionOpportunityV1:
    legal_context: ActionContextV1
    candidate_j: str
    s_pre: tuple[int, int]
    current_position: int
    remaining_speakers: int
    q_source_digest: str
    q_matrix_digest: str
    mapper_artifact_digest: str
    evidence_panel: tuple[CandidateEvidenceV1, ...]
    legal_actions: tuple[Action, ...]
    redirect_target: str | None
    probe_request_type: str | None
    continuation_actor: str | None
    observation_window: ObservationWindowV1 | None

    def __post_init__(self):
        context = self.legal_context
        if not isinstance(context, ActionContextV1) or self.candidate_j not in context.legal_targets:
            raise DecisionOpportunityError("candidate is not a current legal non-wolf")
        if (self.s_pre != (sum(p in context.alive for p in context.known_wolves),
                           sum(p not in context.known_wolves for p in context.alive))
                or not 0 < self.s_pre[0] < self.s_pre[1]
                or self.current_position != context.public_speaker_queue.index(context.acting_wolf)
                or self.remaining_speakers != len(context.public_speaker_queue) - self.current_position - 1):
            raise DecisionOpportunityError("PRE state or public speaking position differs")
        if any(not _sha256(value) for value in (
                self.q_source_digest, self.q_matrix_digest, self.mapper_artifact_digest)):
            raise DecisionOpportunityError("missing frozen evidence provenance")
        if (tuple(row.candidate_j for row in self.evidence_panel) != context.legal_targets
                or not self.legal_actions or self.legal_actions[0] is not Action.PUSH
                or len(self.legal_actions) != len(set(self.legal_actions))):
            raise DecisionOpportunityError("evidence panel or legal actions differ from J")
        expected_redirect = len(context.legal_targets) > 1
        continuation = probe_continuation(context, self.candidate_j)
        expected_actions = (Action.PUSH,) + ((Action.REDIRECT,) if expected_redirect else ()) + (
            (Action.PROBE,) if continuation is not None else ())
        if self.legal_actions != expected_actions:
            raise DecisionOpportunityError("legal action geometry differs")
        if expected_redirect:
            probabilities = {row.candidate_j: row.p_tilde for row in self.evidence_panel}
            if self.redirect_target != select_redirect_target(context, self.candidate_j, probabilities):
                raise DecisionOpportunityError("redirect target differs from frozen selector")
        elif self.redirect_target is not None:
            raise DecisionOpportunityError("illegal redirect target")
        if continuation is None:
            if any(value is not None for value in (
                    self.probe_request_type, self.continuation_actor, self.observation_window)):
                raise DecisionOpportunityError("illegal Probe continuation")
        elif ((self.probe_request_type, self.continuation_actor, self.observation_window)
              != ("CURRENT_SUSPICION_BASIS", *continuation)):
            raise DecisionOpportunityError("Probe request or window differs from queue")

    @property
    def identity(self) -> tuple[str, str, str, str, str, str]:
        c = self.legal_context
        return c.game_id, c.boundary_id, c.prefix_digest, c.acting_wolf, c.phase, self.candidate_j

    @property
    def p_tilde_j(self) -> float:
        return next(row.p_tilde for row in self.evidence_panel if row.candidate_j == self.candidate_j)

    def to_record(self) -> dict:
        c = self.legal_context
        return {"schema_version": VERSION,
                "identity": {"game_id": c.game_id, "boundary_id": c.boundary_id,
                             "prefix_digest": c.prefix_digest, "acting_wolf": c.acting_wolf,
                             "phase": c.phase, "candidate_j": self.candidate_j},
                "public_legal": {"alive": list(c.alive), "known_wolves": sorted(c.known_wolves),
                                 "J": list(c.legal_targets), "C": list(c.competition),
                                 "speaker_queue": list(c.public_speaker_queue),
                                 "current_position": self.current_position,
                                 "remaining_speakers": self.remaining_speakers,
                                 "public_history_digest": c.public_history_digest},
                "s_pre": list(self.s_pre),
                "evidence": {"q_source_kind": "frozen_final_qwen3_predictor",
                             "q_source_digest": self.q_source_digest,
                             "q_matrix_digest": self.q_matrix_digest,
                             "mapper_artifact_digest": self.mapper_artifact_digest,
                             "p_tilde_j": self.p_tilde_j,
                             "panel": [asdict(row) for row in self.evidence_panel]},
                "legal_actions": [action.value for action in self.legal_actions],
                "redirect_target": self.redirect_target,
                "probe": None if self.continuation_actor is None else {
                    "request_type": self.probe_request_type,
                    "continuation_actor": self.continuation_actor,
                    "observation_window": self.observation_window.to_record()}}

    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_record()))


def build_phase2_decision_opportunity(context: ActionContextV1, candidate_j: str,
                                      q, mapper, *, q_source_digest: str) -> Phase2DecisionOpportunityV1:
    """Use the final Qwen3 predictor digest; fail if any legal R2/p is absent."""
    if not isinstance(context, ActionContextV1) or candidate_j not in context.legal_targets:
        raise DecisionOpportunityError("no legal candidate at this PRE")
    if not _sha256(q_source_digest):
        raise DecisionOpportunityError("missing frozen Q source provenance")
    if not _sha256(getattr(mapper, "artifact_digest", None)) or not callable(getattr(mapper, "infer", None)):
        raise DecisionOpportunityError("frozen mapper is required")
    matrix = validate_final_q(q)
    panel = []
    for target in context.legal_targets:
        result = mapper.infer(matrix, alive=context.alive, known_wolves=context.known_wolves,
                              acting_wolf=context.acting_wolf, candidate_j=target,
                              phase=context.phase, public_speaker_queue=context.public_speaker_queue,
                              competition=context.competition, audit=True)
        if not isinstance(result, RuntimeInference):
            raise DecisionOpportunityError("mapper did not return audited R2 inference")
        observers = tuple(player for player in context.alive
                          if player not in context.known_wolves and player != target)
        if (result.observer_ids != observers or result.competition != context.competition
                or result.z.phase != context.phase
                or result.z.audience_size != len(observers)
                or result.z.competition_size != len(context.competition)
                or result.z.remaining_speakers_before_vote !=
                len(context.public_speaker_queue) - context.public_speaker_queue.index(
                    context.acting_wolf) - 1):
            raise DecisionOpportunityError("audited R2 context differs from current PRE")
        panel.append(CandidateEvidenceV1(target, result.p_tilde,
            sha256_bytes(canonical_json_bytes(asdict(result.z)))))
    probabilities = {row.candidate_j: row.p_tilde for row in panel}
    continuation = probe_continuation(context, candidate_j)
    position = context.public_speaker_queue.index(context.acting_wolf)
    return Phase2DecisionOpportunityV1(
        context, candidate_j,
        (sum(p in context.alive for p in context.known_wolves),
         sum(p not in context.known_wolves for p in context.alive)),
        position, len(context.public_speaker_queue) - position - 1,
        q_source_digest, sha256_bytes(canonical_json_bytes(matrix.tolist())),
        mapper.artifact_digest, tuple(panel),
        (Action.PUSH,) + ((Action.REDIRECT,) if len(panel) > 1 else ()) + (
            (Action.PROBE,) if continuation is not None else ()),
        select_redirect_target(context, candidate_j, probabilities) if len(panel) > 1 else None,
        "CURRENT_SUSPICION_BASIS" if continuation is not None else None,
        continuation[0] if continuation is not None else None,
        continuation[1] if continuation is not None else None)


def build_phase2_decision_opportunity_from_pre(pre, known_wolves: frozenset[str],
                                                candidate_j: str, q, mapper, *,
                                                q_source_digest: str) -> Phase2DecisionOpportunityV1:
    """Runtime entry: derive legal context from the authoritative current PRE."""
    return build_phase2_decision_opportunity(context_from_pre(pre, known_wolves),
                                             candidate_j, q, mapper,
                                             q_source_digest=q_source_digest)
