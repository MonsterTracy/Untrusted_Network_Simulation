"""Day-resolution outcome bridge to the frozen publication reference table."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.phase2_decision_opportunity import Phase2DecisionOpportunityV1
from werewolf.phase2_offline import (
    ReferenceLoss, ReferenceTables, ResolvedOutcome, classify_outcome,
)


class OutcomeError(ValueError):
    """Day resolution cannot be classified under this PRE contract."""


def reference_tables_digest(values: ReferenceTables) -> str:
    """Stable content identity for the exact frozen reference-value cells."""
    if not isinstance(values, ReferenceTables):
        raise OutcomeError("reference table is required")
    def table_record(table):
        return {"excluded_fold": table.excluded_fold,
                "training_game_count": table.training_game_count,
                "cells": [[list(state), asdict(estimate)]
                          for state, estimate in sorted(table.cells.items())]}
    return sha256_bytes(canonical_json_bytes({
        "fold_of_game": [[game, fold]
                         for game, fold in sorted(values.fold_of_game.items())],
        "cross_fit": [[fold, table_record(table)]
                      for fold, table in sorted(values.v_ref_pub_cf.items())],
        "full": table_record(values.v_ref_pub_full)}))


@dataclass(frozen=True)
class Phase2DayOutcomeV1:
    exiled_player: str | None
    category: ResolvedOutcome
    reference: ReferenceLoss

    def __post_init__(self):
        if (not isinstance(self.category, ResolvedOutcome)
                or not isinstance(self.reference, ReferenceLoss)
                or self.reference.outcome is not self.category
                or not math.isfinite(self.reference.loss)
                or not 0 <= self.reference.loss <= 1
                or not math.isfinite(self.reference.value.value)
                or not 0 <= self.reference.value.value <= 1
                or not math.isclose(self.reference.loss + self.reference.value.value,
                                    1.0, abs_tol=1e-9)):
            raise OutcomeError("reference loss and resolved outcome disagree")

    @property
    def s_plus(self) -> tuple[int, int]:
        return self.reference.s_post

    @property
    def v_ref(self) -> float:
        return self.reference.value.value

    @property
    def l_ref(self) -> float:
        return self.reference.loss


def extract_phase2_day_outcome(opportunity: Phase2DecisionOpportunityV1,
                               exiled_player: str | None, values: ReferenceTables,
                               *, reference_scope: str = "deployment") -> Phase2DayOutcomeV1:
    """Actual exile only; reuse existing tau and V_ref/L_ref implementation."""
    if not isinstance(opportunity, Phase2DecisionOpportunityV1) or not isinstance(values, ReferenceTables):
        raise OutcomeError("opportunity and frozen reference tables are required")
    if exiled_player is not None and (exiled_player not in PLAYER_IDS
                                     or exiled_player not in opportunity.legal_context.alive):
        raise OutcomeError("exile is outside the PRE alive set")
    c = opportunity.legal_context
    category = classify_outcome(exiled_player, acting_wolf=c.acting_wolf,
                                candidate_j=opportunity.candidate_j, wolves=c.known_wolves)
    if reference_scope == "deployment":
        reference = values.deployment_loss(opportunity.s_pre, category)
    elif reference_scope == "development_cross_fit":
        reference = values.development_loss(c.game_id, opportunity.s_pre, category)
    else:
        raise OutcomeError("unknown frozen reference table scope")
    return Phase2DayOutcomeV1(exiled_player, category, reference)
