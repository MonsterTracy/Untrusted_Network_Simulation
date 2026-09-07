"""One row-scoring implementation; game means and paired game-cluster intervals."""

import numpy as np
import torch

SCORING_VERSION = "classic7_game_macro_belief_kl_v1"


def row_cross_entropy(logp, q):
    if q.shape != logp.shape or q.shape[-1] != 7 or not torch.isfinite(q).all() or torch.any(q < 0):
        raise ValueError("invalid belief target")
    if not torch.isfinite(logp[q > 0]).all():
        raise ValueError("nonfinite model probability on target support")
    return -(q * torch.where(q > 0, logp, 0)).sum(-1)


def row_kl(logp, q):
    entropy_term = (q * torch.where(q > 0, q.log(), 0)).sum(-1)
    return row_cross_entropy(logp, q) + entropy_term


def game_balanced_cross_entropy(games):
    if not games:
        raise ValueError("zero training games")
    losses = []
    for logp, q, mask in games:
        if mask.dtype != torch.bool or mask.shape != q.shape[:-1] or not mask.any():
            raise ValueError("zero or invalid Primary effective rows")
        if not torch.allclose(q.sum(-1)[mask], torch.ones_like(q.sum(-1)[mask])):
            raise ValueError("target must sum to one")
        losses.append(row_cross_entropy(logp, q)[mask].mean())
    return torch.stack(losses).mean()


def game_macro_summary(scores, indices, confidence):
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("game scores must be a nonempty finite vector")
    if indices.ndim != 2 or indices.shape[1] != len(values) or indices.dtype.kind != "i" or np.any(indices < 0) or np.any(indices >= len(values)):
        raise ValueError("bootstrap must resample the same game identities")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    replicates = values[indices].mean(1)
    alpha = (1-confidence)/2
    return {"point": float(values.mean()), "interval": np.quantile(replicates, [alpha, 1-alpha], method="linear").tolist(),
            "aggregation": "game_macro", "sampling_unit": "game", "interval_method": "percentile_linear", "confidence": confidence}


def paired_summary(left, right, indices, confidence):
    if len(left) != len(right):
        raise ValueError("paired games must have equal coverage")
    return game_macro_summary(np.asarray(left) - np.asarray(right), indices, confidence)
