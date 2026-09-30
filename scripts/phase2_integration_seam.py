"""Explicit opt-in Phase-2 speech preparation; never changes run_random."""

from __future__ import annotations

from dataclasses import dataclass

from werewolf.phase2_decision_opportunity import Phase2DecisionOpportunityV1
from werewolf.phase2_language import PublicLanguageContextV1
from werewolf.phase2_treatment import Phase2TreatmentV1
from werewolf.phase2_verified_speech import Phase2VerifiedSpeechResultV1, verified_phase2_speech


@dataclass(frozen=True)
class Phase2PreparedSpeechV1:
    opportunity_digest: str
    treatment_id: str
    verified: Phase2VerifiedSpeechResultV1
    gameplay_committed: bool = False
    actual_vote_executed: bool = False


def prepare_phase2_intervention_speech(opportunity: Phase2DecisionOpportunityV1,
                                       treatment: Phase2TreatmentV1,
                                       public: PublicLanguageContextV1, *, actor, perceiver,
                                       ) -> Phase2PreparedSpeechV1:
    """The future policy supplies a frozen treatment; only language is produced."""
    if (not isinstance(opportunity, Phase2DecisionOpportunityV1)
            or not isinstance(treatment, Phase2TreatmentV1)
            or treatment.opportunity_digest != opportunity.digest()
            or treatment.opportunity_identity != opportunity.identity):
        raise ValueError("Phase-2 treatment differs from current PRE opportunity")
    verified = verified_phase2_speech(treatment.plan, opportunity.legal_context,
                                      public, actor=actor, perceiver=perceiver)
    return Phase2PreparedSpeechV1(opportunity.digest(), treatment.treatment_id, verified)
