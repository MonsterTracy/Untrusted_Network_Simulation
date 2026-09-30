"""Sequential Probe value interface; no empirical kernel is available yet."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Protocol

from werewolf.phase2_actions import Action, InformationRequestV1
from werewolf.phase2_decision_opportunity import Phase2DecisionOpportunityV1


class ProbeValueUnavailable(ValueError):
    """An empirical observation kernel or continuation model is missing."""


class Phase2ProbeValueModel(Protocol):
    evidence_artifact_digest: str

    def observation_distribution(self, opportunity: Phase2DecisionOpportunityV1,
                                 request: InformationRequestV1) -> Mapping[str, float]: ...

    def transition_state(self, opportunity: Phase2DecisionOpportunityV1,
                         request: InformationRequestV1, observation: str) -> object: ...

    def continuation_value(self, state: object) -> float: ...


@dataclass(frozen=True)
class ProbeValueAuditV1:
    kappa: float
    contributions: tuple[tuple[str, float, float], ...]
    risk: float
    production_ready: bool


def evaluate_probe_value(opportunity: Phase2DecisionOpportunityV1,
                         request: InformationRequestV1, model: Phase2ProbeValueModel | None,
                         *, kappa: float | None, allow_synthetic: bool = False) -> ProbeValueAuditV1:
    """R_B=kappa+E_O[V_next]; synthetic use must be explicitly marked."""
    if (not isinstance(opportunity, Phase2DecisionOpportunityV1)
            or Action.PROBE not in opportunity.legal_actions
            or request != InformationRequestV1(opportunity.candidate_j,
                                               opportunity.candidate_j)):
        raise ProbeValueUnavailable("illegal Probe request or opportunity")
    if (model is None or kappa is None or type(kappa) not in (float, int)
            or not math.isfinite(kappa) or kappa < 0):
        raise ProbeValueUnavailable("Probe cost or observation/continuation model is absent")
    if not allow_synthetic:
        raise ProbeValueUnavailable("no verified pilot observation/continuation model exists")
    distribution = model.observation_distribution(opportunity, request)
    if (not isinstance(distribution, Mapping) or not distribution
            or any(not isinstance(key, str) or not key or type(probability) not in (int, float)
                   or not math.isfinite(probability) or probability < 0
                   for key, probability in distribution.items())
            or not math.isclose(math.fsum(distribution.values()), 1.0, abs_tol=1e-9)):
        raise ProbeValueUnavailable("invalid Probe observation distribution")
    contributions = []
    for observation in sorted(distribution):
        state = model.transition_state(opportunity, request, observation)
        if state is None:
            raise ProbeValueUnavailable("Probe transition state is missing")
        value = model.continuation_value(state)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ProbeValueUnavailable("invalid continuation loss")
        contributions.append((observation, float(distribution[observation]), float(value)))
    risk = float(kappa) + math.fsum(probability * value for _, probability, value in contributions)
    return ProbeValueAuditV1(float(kappa), tuple(contributions), risk, False)
