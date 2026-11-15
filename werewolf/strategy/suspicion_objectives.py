"""Peripheral diagnostics only; never imported by the ToM inference consumer."""

from dataclasses import dataclass

import torch

from werewolf.canonical_collection.public_history import PLAYER_IDS


@dataclass(frozen=True, slots=True)
class WolfSuspicionMass:
    alive_conditional_team_mass: float
    raw_team_mass: float
    raw_self_mass: float
    alive_conditional_self_mass: float
    per_alive_wolf_raw_mass: tuple[tuple[str, float], ...]
    per_alive_wolf_conditional_mass: tuple[tuple[str, float], ...]


def _players(values, name):
    if not isinstance(values, tuple) or any(p not in PLAYER_IDS for p in values):
        raise ValueError(f"{name} must be a tuple of canonical player IDs")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate players")
    return tuple(p for p in PLAYER_IDS if p in values)


def wolf_suspicion_mass(probabilities, *, alive_players, alive_wolves, self_player):
    """Alive-Conditional Wolf Suspicion Mass, NOT win/vote probability.

    Input is exp(log_B), on the unchanged seven-seat non-self simplex. Private
    wolf membership is used only in this peripheral pure function. No epsilon,
    repair, dead-column mutation, or model-input renormalization is performed.
    """
    alive = _players(alive_players, "alive_players")
    wolves = _players(alive_wolves, "alive_wolves")
    if not wolves:
        raise ValueError("empty W_t")
    if not set(wolves) <= set(alive) or self_player not in wolves:
        raise ValueError("wolves and self must be alive wolf players")
    observers = tuple(p for p in alive if p not in wolves)
    if not observers:
        raise ValueError("empty O_t")
    if (not isinstance(probabilities, torch.Tensor) or probabilities.shape != (7, 7)
            or not probabilities.is_floating_point()):
        raise ValueError("probabilities must be a floating [7,7] tensor")
    b = probabilities.detach().to(dtype=torch.float64)
    if (not torch.isfinite(b).all() or torch.any(b < 0)
            or torch.any(b.diagonal() != 0)
            or not torch.allclose(b.sum(-1), torch.ones(7, dtype=b.dtype, device=b.device), atol=1e-6, rtol=0)):
        raise ValueError("probabilities violate the non-self simplex")
    rows = [PLAYER_IDS.index(p) for p in observers]
    wolf_columns = [PLAYER_IDS.index(p) for p in wolves]
    # Exclusion is explicit, even though the validated self column is zero.
    denominators = torch.stack([
        b[i, [PLAYER_IDS.index(p) for p in alive if PLAYER_IDS.index(p) != i]].sum()
        for i in rows])
    if torch.any(denominators <= 0):
        raise ValueError("zero alive non-self probability mass")
    raw = b[rows][:, wolf_columns]
    conditional = raw / denominators[:, None]
    raw_per_wolf = raw.mean(0)
    conditional_per_wolf = conditional.mean(0)
    self_index = wolves.index(self_player)
    return WolfSuspicionMass(
        conditional.sum(-1).mean().item(), raw.sum(-1).mean().item(),
        raw_per_wolf[self_index].item(), conditional_per_wolf[self_index].item(),
        tuple((p, raw_per_wolf[j].item()) for j, p in enumerate(wolves)),
        tuple((p, conditional_per_wolf[j].item()) for j, p in enumerate(wolves)))
