"""Pure terminal-risk arithmetic and non-production three-way comparison."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from werewolf.phase2_actions import Action
from werewolf.phase2_decision_opportunity import Phase2DecisionOpportunityV1


class RiskError(ValueError):
    """Risk inputs are missing, unsupported, or incompatible."""


@dataclass(frozen=True)
class Phase2TerminalLossTableV1:
    opportunity_digest: str
    consequence_artifact_digest: str
    lambda_p_plus: float
    lambda_p_minus: float
    lambda_n_plus: float | None
    lambda_n_minus: float | None
    source: str = "synthetic_audit"

    def __post_init__(self):
        values = (self.lambda_p_plus, self.lambda_p_minus,
                  self.lambda_n_plus, self.lambda_n_minus)
        if (not self.opportunity_digest or not self.consequence_artifact_digest
                or self.source not in ("synthetic_audit", "fitted_intervention")
                or any(value is not None and (type(value) not in (float, int)
                    or not math.isfinite(value) or value < 0) for value in values)
                or any(value is None for value in values[:2])
                or ((values[2] is None) != (values[3] is None))):
            raise RiskError("invalid action-conditioned terminal loss table")


def terminal_action_risks(p: float, table: Phase2TerminalLossTableV1, *,
                          opportunity_digest: str, redirect_legal: bool) -> dict[Action, float]:
    if (type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1
            or not isinstance(table, Phase2TerminalLossTableV1)
            or table.opportunity_digest != opportunity_digest):
        raise RiskError("p or terminal-loss context is invalid")
    result = {Action.PUSH: table.lambda_p_plus * p + table.lambda_p_minus * (1 - p)}
    if redirect_legal:
        if table.lambda_n_plus is None or table.lambda_n_minus is None:
            raise RiskError("legal Redirect lacks consequence losses")
        result[Action.REDIRECT] = table.lambda_n_plus * p + table.lambda_n_minus * (1 - p)
    return result


def load_production_terminal_loss_table(*_args, **_kwargs):
    raise RiskError("no fitted controlled-intervention consequence artifact exists")


def terminal_risks_for_opportunity(opportunity: Phase2DecisionOpportunityV1,
                                   table: Phase2TerminalLossTableV1) -> dict[Action, float]:
    """Derive p and legal Redirect from one validated current opportunity."""
    if not isinstance(opportunity, Phase2DecisionOpportunityV1):
        raise RiskError("validated opportunity required")
    return terminal_action_risks(
        opportunity.p_tilde_j, table, opportunity_digest=opportunity.digest(),
        redirect_legal=Action.REDIRECT in opportunity.legal_actions)


@dataclass(frozen=True)
class ActionRiskComparisonV1:
    selected: Action
    risks: tuple[tuple[Action, float], ...]
    tied_minimum: tuple[Action, ...]
    tie_rule: str = "PUSH, REDIRECT, PROBE"
    production_router: bool = False


def compare_available_action_risks(legal_actions: tuple[Action, ...],
                                   risks: Mapping[Action, float]) -> ActionRiskComparisonV1:
    """All legal risks required; fixed enum order breaks exact ties."""
    order = (Action.PUSH, Action.REDIRECT, Action.PROBE)
    if (not legal_actions or len(legal_actions) != len(set(legal_actions))
            or any(action not in order for action in legal_actions)
            or set(risks) != set(legal_actions)
            or any(type(risks[action]) not in (float, int)
                   or not math.isfinite(risks[action]) or risks[action] < 0
                   for action in legal_actions)):
        raise RiskError("risk comparison requires exactly the legal finite risks")
    ordered = tuple((action, float(risks[action])) for action in order if action in legal_actions)
    minimum = min(value for _, value in ordered)
    tied = tuple(action for action, value in ordered if value == minimum)
    return ActionRiskComparisonV1(tied[0], ordered, tied)


@dataclass(frozen=True)
class ClassicThresholdAuditV1:
    alpha: float
    beta: float
    gamma: float
    boundary_has_strict_region: bool
    production_probe_router: bool = False


def classic_static_threshold_reduction(*, p_plus: float, p_minus: float,
                                       b_plus: float, b_minus: float,
                                       n_plus: float, n_minus: float) -> ClassicThresholdAuditV1:
    """Algebraic six-loss diagnostic; B here is static, unlike real Probe."""
    values = (p_plus, p_minus, b_plus, b_minus, n_plus, n_minus)
    if (any(type(value) not in (float, int) or not math.isfinite(value) or value < 0
            for value in values)
            or not p_plus < b_plus < n_plus
            or not n_minus < b_minus < p_minus):
        raise RiskError("classic strict loss ordering is required")
    alpha = (p_minus - b_minus) / ((p_minus - b_minus) + (b_plus - p_plus))
    beta = (b_minus - n_minus) / ((b_minus - n_minus) + (n_plus - b_plus))
    gamma = (p_minus - n_minus) / ((p_minus - n_minus) + (n_plus - p_plus))
    return ClassicThresholdAuditV1(alpha, beta, gamma, beta < alpha)
