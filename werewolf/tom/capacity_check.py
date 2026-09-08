"""Synthetic engineering allocation check; never a scientific run or artifact."""

import numpy as np
import torch

from werewolf.tom.dataset import ExperimentCapacity, PublicTensors
from werewolf.tom.model import ObserverConditionedToM, MODEL_GRAPH_VERSION
from werewolf.tom.scoring import game_balanced_cross_entropy
from werewolf.tom.temporal import TemporalCodeProvider

NOTICE = "ENGINEERING CAPACITY CHECK — NOT A SCIENTIFIC RUN"


class _SyntheticTemporalCode:
    """Synthetic tables using the real lookup allocation path, without artifacts.

    These values have no public-history meaning and are never published as a
    verified temporal provider. Explicit lookup exercises both 128-wide halves.
    """

    def __init__(self):
        self.day = np.full((1, 128), .02, dtype="<f4")
        self.phase = np.full((5, 128), .02, dtype="<f4")
        self.explicit_enabled = True

    code = TemporalCodeProvider.code


def check_training_capacity(*, pre_count_per_game, max_seq_len, game_batch_size, device):
    for name, value in (("pre_count_per_game", pre_count_per_game),
                        ("max_seq_len", max_seq_len), ("game_batch_size", game_batch_size)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be an explicit positive integer")
    if device not in ("cpu", "cuda"):
        raise ValueError("capacity check device must be cpu or cuda")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("declared CUDA backend is unavailable")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    model = ObserverConditionedToM(ExperimentCapacity(max_seq_len), _SyntheticTemporalCode()).to(device)
    model.train()
    # Engineering-only scratch settings. They are not Formal Protocol values.
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, betas=(.9, .999),
        eps=1e-8, weight_decay=0., foreach=False, fused=False)
    shape = (pre_count_per_game, max_seq_len)
    public = PublicTensors(
        event_ids=torch.ones(shape, dtype=torch.long),
        source_ids=torch.ones(shape, dtype=torch.long),
        action_ids=torch.zeros(shape, dtype=torch.long),
        target_ids=torch.zeros(shape, dtype=torch.long),
        day_ids=torch.zeros(shape, dtype=torch.long),
        phase_ids=torch.zeros(shape, dtype=torch.long),
        attention_mask=torch.ones(shape, dtype=torch.bool))
    # A synthetic non-self target and mask, not observed beliefs or role selection.
    target = ((1 - torch.eye(7, dtype=torch.float32)) / 6).expand(pre_count_per_game, -1, -1)
    mask = torch.ones((pre_count_per_game, 7), dtype=torch.bool)
    # Two fixed steps include a forward/backward with Adam moments already live.
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        game_losses = []
        for _ in range(game_batch_size):
            logp = model(**{k: v.to(device) for k, v in public.kwargs().items()})
            game_losses.append((logp, target.to(device), mask.to(device)))
        loss = game_balanced_cross_entropy(game_losses)
        if not torch.isfinite(loss):
            raise ValueError("synthetic capacity execution produced a nonfinite scalar")
        loss.backward()
        optimizer.step()
        del loss, game_losses, logp
    result = {"notice": NOTICE, "status": "completed", "model_graph": MODEL_GRAPH_VERSION,
        "device": device, "pre_count_per_game": pre_count_per_game,
        "max_seq_len": max_seq_len, "game_batch_size": game_batch_size,
        "public_shape_per_game": list(shape), "output_shape_per_game": [pre_count_per_game, 7, 7],
        "optimizer_steps": 2, "parameter_dtype": str(next(model.parameters()).dtype),
        "temporal_input": "synthetic_explicit_lookup_128_plus_128",
        "scientific_artifacts_read_or_written": False}
    if device == "cuda":
        torch.cuda.synchronize()
        result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return result
