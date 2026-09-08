"""Real synthetic optimization, with scientific artifact access forbidden."""

import json
from pathlib import Path

import pytest
import torch


def test_cpu_real_step_without_artifacts(tmp_path, monkeypatch, capsys):
    from werewolf.cli import main
    from werewolf.tom.model import ObserverConditionedToM
    import werewolf.tom.capacity_check as capacity
    # Import production dependencies before installing the I/O guards.
    import werewolf.development_publication
    import werewolf.tom.experiment
    import werewolf.artifact_io
    def forbidden(*args, **kwargs):
        raise AssertionError("scientific artifact access")
    for name in ("open_publication", "open_role_sidecar", "open_verified_collection"):
        monkeypatch.setattr(werewolf.development_publication, name, forbidden)
    monkeypatch.setattr(werewolf.tom.experiment, "prepare_experiment", forbidden)
    monkeypatch.setattr(werewolf.tom.experiment, "open_experiment", forbidden)
    monkeypatch.setattr(capacity.TemporalCodeProvider, "__init__", forbidden)
    monkeypatch.setattr(werewolf.artifact_io, "publish_artifact", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UNS_STORAGE_PROFILE", "/does/not/exist")
    forwards, steps, retained = [], [], []
    real_forward = ObserverConditionedToM.forward
    real_step = torch.optim.AdamW.step
    real_loss = capacity.game_balanced_cross_entropy
    def forward(self, **kwargs):
        forwards.append(tuple(kwargs["event_ids"].shape))
        return real_forward(self, **kwargs)
    def step(self, *args, **kwargs):
        before = [p.detach().clone() for group in self.param_groups for p in group["params"]]
        result = real_step(self, *args, **kwargs)
        after = [p for group in self.param_groups for p in group["params"]]
        assert any(not torch.equal(a, b) for a, b in zip(before, after))
        steps.append(True)
        return result
    def loss(games):
        assert all(logp.grad_fn is not None for logp, _, _ in games)
        retained.append(len(games))
        return real_loss(games)
    monkeypatch.setattr(ObserverConditionedToM, "forward", forward)
    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    monkeypatch.setattr(capacity, "game_balanced_cross_entropy", loss)
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        assert main(["capacity-check", "--pre-count-per-game", "2", "--max-seq-len", "4",
                     "--game-batch-size", "2", "--device", "cpu"]) == 0
    finally:
        torch.set_num_threads(threads)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == capacity.NOTICE
    result = json.loads(lines[1])
    assert result["optimizer_steps"] == 2
    assert result["scientific_artifacts_read_or_written"] is False
    assert "loss" not in result and "accuracy" not in result
    assert "peak_allocated_bytes" not in result
    assert forwards == [(2, 4)] * 4 and retained == [2, 2] and len(steps) == 2
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("field", ["pre_count_per_game", "max_seq_len", "game_batch_size"])
def test_invalid_shapes_rejected_before_model(field, monkeypatch):
    import werewolf.tom.capacity_check as capacity
    monkeypatch.setattr(capacity, "ObserverConditionedToM", lambda *a: pytest.fail("model allocated"))
    args = dict(pre_count_per_game=1, max_seq_len=1, game_batch_size=1, device="cpu")
    args[field] = 0
    with pytest.raises(ValueError, match=field):
        capacity.check_training_capacity(**args)


def test_cuda_unavailable_and_oom_never_fallback(monkeypatch):
    import werewolf.tom.capacity_check as capacity
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = dict(pre_count_per_game=1, max_seq_len=1, game_batch_size=1)
    with pytest.raises(ValueError, match="unavailable"):
        capacity.check_training_capacity(**args, device="cuda")
    attempts = []
    def oom(*args):
        attempts.append(True)
        raise torch.OutOfMemoryError("synthetic OOM")
    monkeypatch.setattr(capacity, "ObserverConditionedToM", oom)
    with pytest.raises(torch.OutOfMemoryError):
        capacity.check_training_capacity(**args, device="cpu")
    assert len(attempts) == 1
