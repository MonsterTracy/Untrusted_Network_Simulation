"""Large 专用训练的合成 CPU 验证；临时 Checkpoint 不代表正式训练产物。"""

from copy import deepcopy
from dataclasses import replace
import math
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tests.tom.test_development_input_v2 import v2
from tests.tom.test_qwen3_large import example
from werewolf.artifact_io import ArtifactValidationError, canonical_json_bytes, publish_artifact, verify_artifact
from werewolf.tom import qwen3_large_training as t
from werewolf.tom.dataset import CanonicalToMSample, PublicTensors, rotate
from werewolf.tom.population import ObserverEligibility
from werewolf.tom.qwen3_large import Qwen3ToMLarge
from werewolf.tom.scoring import game_balanced_cross_entropy, row_kl


class TinyPublicModel(nn.Module):
    """只替换计算规模；真实公开张量、旋转、评分及状态编码仍走生产函数。"""

    def __init__(self, dropout=0.):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(49, dtype=torch.float32).reshape(7, 7) / 60)
        self.slope = nn.Parameter(torch.arange(7, dtype=torch.float32) / 80)
        self.dropout = dropout
        self.calls = []

    def forward(self, **public):
        assert set(public) == set(PublicTensors.__dataclass_fields__)
        self.calls.append((self.training, torch.is_grad_enabled(), torch.is_inference_mode_enabled(),
                           public["source_ids"].detach().clone()))
        feature = public["event_ids"].float().sum(-1)[:, None, None] / 10
        logits = self.weight[None] + feature * self.slope[None, None]
        logits = F.dropout(logits, p=self.dropout, training=self.training)
        if self.training and self.dropout:
            # 恢复测试同时覆盖 Python、NumPy 及 Torch 的 RNG。
            logits = logits + (random.random() + np.random.rand()) * self.slope[None, None] / 100
        return logits.masked_fill(torch.eye(7, dtype=torch.bool, device=logits.device), -torch.inf).log_softmax(-1)


class SyntheticData:
    def __init__(self, training=2, validation=2):
        self.split = {"training_game_ids": [f"train-{i:03}" for i in range(training)],
                      "validation_game_ids": [f"val-{i:03}" for i in range(validation)]}
        self.by_game, rows = {}, {}
        self.visits = []
        for index, game_id in enumerate(self.split["training_game_ids"] + self.split["validation_game_ids"]):
            samples, rows[game_id] = [], {}
            for pre in range(1 + index % 3):
                q = torch.zeros(7, 7)
                for observer in range(7):
                    support = [(observer + 1 + k) % 7 for k in range(1 + pre % 2)]
                    q[observer, support] = 1 / len(support)
                arrays = {name: torch.tensor([1 + pre % 3, 2, 0, 0]) for name in
                          ("event_ids", "source_ids", "action_ids", "target_ids", "day_ids", "phase_ids")}
                public = PublicTensors(**arrays, attention_mask=torch.tensor([True, True, False, False]))
                observed = torch.tensor([True, True, pre % 2 == 0, True, True, False, False])
                boundary = f"pre-{pre}"
                samples.append(CanonicalToMSample(game_id, boundary, "a" * 64, "b" * 64, 2,
                                                  public, q, observed, torch.ones(7, dtype=torch.bool)))
                rows[game_id][boundary] = [True, pre % 2 == 1, pre >= 1, pre == 2, False, False, False]
            self.by_game[game_id] = samples
        self.primary = ObserverEligibility(rows, {"test_scope": "synthetic_cpu_only"})

    def samples(self, game_id):
        self.visits.append(game_id)
        return self.by_game[game_id]


@pytest.fixture
def protocol():
    return t.LargeTrainingProtocol(initialization_seed=17, schedule_seed=23, rng_seed=31, pre_microbatch_size=1)


@pytest.fixture(autouse=True)
def preserve_rng():
    tensors, scalars = t.training._rng_capture()
    deterministic = torch.are_deterministic_algorithms_enabled()
    threads = torch.get_num_threads()
    yield
    t.training._rng_restore(tensors, scalars)
    torch.use_deterministic_algorithms(deterministic)
    torch.set_num_threads(threads)


def optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(.9, .999), eps=1e-8,
                             weight_decay=1e-5, foreach=False, fused=False)


def seed_all(seed=71):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)


def binding(data, protocol):
    return {"test_scope": "synthetic_cpu_only", "training_split": deepcopy(data.split),
            "protocol": protocol.identity(), "execution_source": {"fixture": True}}


def full_loss(model, data, batch):
    games = []
    for row in batch:
        samples = data.samples(row["game_id"])
        pairs = [rotate(sample, row["shift"], data.primary.mask(sample.game_id, sample.boundary_id)) for sample in samples]
        public = PublicTensors.stack([sample.public for sample, _ in pairs])
        q = torch.stack([sample.q for sample, _ in pairs])
        mask = torch.stack([eligible & sample.label_observed for sample, eligible in pairs])
        games.append((model(**public.kwargs()), q, mask))
    return game_balanced_cross_entropy(games)


def test_protocol_records_fixed_values_and_requires_explicit_seeds(protocol):
    p = protocol.identity()
    assert (p["training_games"], p["validation_games"], p["game_batch_size"], p["tail_batch_games"]) == (4500, 300, 32, 20)
    assert (p["learning_rate"], p["adam_betas"], p["adam_eps"], p["weight_decay"]) == (5e-5, [.9, .999], 1e-8, 1e-5)
    assert (p["max_epochs"], p["patience"], p["min_delta"]) == (80, 10, 0)
    for field in ("initialization_seed", "schedule_seed", "rng_seed"):
        for bad in (True, -1, 2**63, 1.2):
            with pytest.raises(ValueError, match="seeds"):
                replace(protocol, **{field: bad})
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="microbatch"):
            replace(protocol, pre_microbatch_size=bad)


def test_epoch_coverage_tail_order_and_balanced_rotations(protocol):
    ids = [f"development-{i:04}" for i in range(4500)]
    before = torch.get_rng_state().clone()
    batches = t.epoch_schedule(ids, 1, protocol)
    assert [len(batch) for batch in batches] == [32] * 140 + [20]
    assert sorted(row["game_id"] for batch in batches for row in batch) == ids
    assert batches == t.epoch_schedule(list(reversed(ids)), 1, protocol)
    assert batches != t.epoch_schedule(ids, 2, protocol)
    assert torch.equal(before, torch.get_rng_state())
    shifts = {game_id: [] for game_id in ids}
    for epoch in range(1, 81):
        for batch in t.epoch_schedule(ids, epoch, protocol):
            for row in batch:
                shifts[row["game_id"]].append(row["shift"])
    for values in shifts.values():
        assert all(sorted(values[start:start + 7]) == list(range(7)) for start in range(0, 74, 7))
        assert max(values.count(shift) for shift in range(7)) - min(values.count(shift) for shift in range(7)) == 1


@pytest.mark.parametrize("size", [1, 20, 32])
@pytest.mark.parametrize("microbatch", [1, 2, 8])
def test_microbatch_game_balanced_gradient_matches_full(size, microbatch):
    data = SyntheticData(training=size)
    batch = [{"game_id": g, "shift": index % 7} for index, g in enumerate(data.split["training_game_ids"])]
    full, micro = TinyPublicModel(), TinyPublicModel()
    expected = full_loss(full, data, batch)
    expected.backward()
    value = t.train_batch(micro, torch.optim.SGD(micro.parameters(), lr=0), data, batch, pre_microbatch_size=microbatch)
    assert value == pytest.approx(float(expected.detach()), rel=2e-6)
    for left, right in zip(full.parameters(), micro.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, atol=2e-7, rtol=3e-6)
    assert all(call[0] and call[1] for call in micro.calls)


def test_zero_rows_rejected_before_any_batch_update():
    data = SyntheticData()
    for row in data.primary.rows["train-001"].values():
        row[:] = [False] * 7
    model = TinyPublicModel()
    before = deepcopy(model.state_dict())
    opt = optimizer(model)
    with pytest.raises(ValueError, match="zero Primary"):
        t.train_batch(model, opt, data, [{"game_id": g, "shift": 0} for g in data.split["training_game_ids"]], pre_microbatch_size=1)
    assert not model.calls and not opt.state
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())


def test_one_optimizer_step_after_every_game_and_twenty_game_tail(protocol, monkeypatch):
    data, model = SyntheticData(training=52), TinyPublicModel()
    opt, updates = optimizer(model), []
    original = opt.step
    def step():
        updates.append(list(data.visits))
        return original()
    monkeypatch.setattr(opt, "step", step)
    batches = t.epoch_schedule(data.split["training_game_ids"], 1, protocol)
    for batch in batches:
        t.train_batch(model, opt, data, batch, pre_microbatch_size=1)
    assert [len(batch) for batch in batches] == [32, 20]
    assert [len(visited) for visited in updates] == [32, 52]
    assert all(state["step"].item() == 2 for state in opt.state.values())


def test_validation_kl_is_game_macro_readonly_and_canonical():
    data, model = SyntheticData(), TinyPublicModel()
    opt = optimizer(model)
    t.train_batch(model, opt, data, [{"game_id": "train-000", "shift": 1}], pre_microbatch_size=1)
    before = t._capture(model, opt)
    data.visits.clear()
    model.calls.clear()
    expected = []
    model.eval()
    with torch.inference_mode():
        for game_id in data.split["validation_game_ids"]:
            samples = data.by_game[game_id]
            q = torch.stack([s.q for s in samples])
            mask = torch.stack([data.primary.mask(s.game_id, s.boundary_id) & s.label_observed for s in samples])
            scores = row_kl(model(**PublicTensors.stack([s.public for s in samples]).kwargs()).double(), q.double())
            expected.append(float(scores[mask].mean()))
    model.train()
    model.calls.clear()
    measured = t.validate(model, data, pre_microbatch_size=1)
    after = t._capture(model, opt)
    assert measured == pytest.approx(sum(expected) / len(expected), abs=1e-7)
    assert before[1] == after[1]
    assert canonical_json_bytes(before[2:]) == canonical_json_bytes(after[2:])
    assert data.visits == data.split["validation_game_ids"]
    assert model.training
    assert all(not training and not grad and inference for training, grad, inference, _ in model.calls)
    expected_sources = [s.public.source_ids[:s.token_count] for game in data.visits for s in data.by_game[game]]
    assert all(torch.equal(call[3][0], source) for call, source in zip(model.calls, expected_sources, strict=True))


def test_validation_matches_uniform_minus_target_entropy():
    data, model = SyntheticData(), TinyPublicModel()
    with torch.no_grad():
        model.weight.zero_()
        model.slope.zero_()
    means = []
    for game_id in data.split["validation_game_ids"]:
        values = []
        for sample in data.by_game[game_id]:
            mask = data.primary.mask(game_id, sample.boundary_id) & sample.label_observed
            for row in sample.q[mask]:
                entropy = -sum(float(p) * math.log(float(p)) for p in row if p)
                values.append(math.log(6) - entropy)
        means.append(sum(values) / len(values))
    assert t.validate(model, data, pre_microbatch_size=2) == pytest.approx(sum(means) / len(means), abs=1e-7)


def test_best_tie_selection_and_patience_stop_at_ten(protocol, tmp_path, monkeypatch):
    data, model = SyntheticData(), TinyPublicModel()
    values = iter([.6, .4, .4] + [.5] * 20)
    monkeypatch.setattr(t, "validate", lambda *a, **kw: next(values))
    result = t._fit_epochs(model, optimizer(model), data, protocol, binding(data, protocol), tmp_path / "checkpoints")
    assert len(result["history"]) == 12 and result["stop_reason"] == "patience"
    assert result["selection"] == {"best_epoch": 2, "best_validation_kl": .4, "epochs_without_improvement": 10}
    assert result["best"]["path"].endswith("/2") and result["last"]["path"].endswith("/12")
    last = verify_artifact(result["last"]["path"], expected_artifact_type="qwen3_large_training_checkpoint",
                           expected_schema_version=t.VERSION)
    assert last.manifest["best_checkpoint_digest"] == result["best"]["manifest_digest"]


def test_maximum_eighty_epochs(protocol, tmp_path, monkeypatch):
    data, model = SyntheticData(training=1, validation=1), TinyPublicModel()
    values = iter(1 / epoch for epoch in range(1, 81))
    monkeypatch.setattr(t, "validate", lambda *a, **kw: next(values))
    result = t._fit_epochs(model, optimizer(model), data, protocol, binding(data, protocol), tmp_path / "checkpoints")
    assert len(result["history"]) == 80 and result["stop_reason"] == "max_epochs"
    assert result["selection"]["best_epoch"] == 80


def test_resume_restores_model_optimizer_rng_and_selection(protocol, tmp_path):
    data = SyntheticData()
    identity = binding(data, protocol)
    seed_all()
    full = TinyPublicModel(dropout=.3)
    full_opt = optimizer(full)
    expected = t._fit_epochs(full, full_opt, data, protocol, identity, tmp_path / "full", until_epoch=4)
    expected_state = t._capture(full, full_opt)
    seed_all()
    partial = TinyPublicModel(dropout=.3)
    first = t._fit_epochs(partial, optimizer(partial), data, protocol, identity, tmp_path / "partial", until_epoch=2)
    seed_all(996)
    restored = TinyPublicModel(dropout=.3)
    restored_opt = optimizer(restored)
    actual = t._fit_epochs(restored, restored_opt, data, protocol, identity, tmp_path / "partial", until_epoch=4,
                           resume=(first["last"]["path"], first["last"]["manifest_digest"]))
    actual_state = t._capture(restored, restored_opt)
    assert actual["history"] == expected["history"]
    assert actual["selection"] == expected["selection"]
    assert actual["best"]["manifest_digest"] == expected["best"]["manifest_digest"]
    assert actual["last"]["manifest_digest"] == expected["last"]["manifest_digest"]
    assert actual_state[1] == expected_state[1]
    assert canonical_json_bytes(actual_state[2:]) == canonical_json_bytes(expected_state[2:])


def test_interrupted_epoch_replays_only_from_last_validated_boundary(protocol, tmp_path, monkeypatch):
    data = SyntheticData(training=33, validation=1)
    identity = binding(data, protocol)
    seed_all()
    full = TinyPublicModel(dropout=.2)
    expected = t._fit_epochs(full, optimizer(full), data, protocol, identity, tmp_path / "full", until_epoch=3)
    seed_all()
    model = TinyPublicModel(dropout=.2)
    original, calls = t.train_batch, []
    def interrupt(*a, **kw):
        value = original(*a, **kw)
        calls.append(value)
        if len(calls) == 3:
            raise InterruptedError("synthetic interruption after uncheckpointed optimizer update")
        return value
    with monkeypatch.context() as patch:
        patch.setattr(t, "train_batch", interrupt)
        with pytest.raises(InterruptedError):
            t._fit_epochs(model, optimizer(model), data, protocol, identity, tmp_path / "partial", until_epoch=3)
    assert sorted(p.name for p in (tmp_path / "partial").iterdir()) == ["1"]
    first = verify_artifact(tmp_path / "partial/1", expected_artifact_type="qwen3_large_training_checkpoint",
                            expected_schema_version=t.VERSION)
    restored = TinyPublicModel(dropout=.2)
    actual = t._fit_epochs(restored, optimizer(restored), data, protocol, identity, tmp_path / "partial", until_epoch=3,
                           resume=(first.path, first.manifest_digest))
    assert actual["history"] == expected["history"] and actual["selection"] == expected["selection"]
    assert actual["last"]["manifest_digest"] == expected["last"]["manifest_digest"]


def test_failed_validation_never_publishes_best(protocol, tmp_path, monkeypatch):
    data, model = SyntheticData(), TinyPublicModel()
    def failed(*a, **kw):
        raise ValueError("synthetic failed validation")
    monkeypatch.setattr(t, "validate", failed)
    root = tmp_path / "checkpoints"
    with pytest.raises(ValueError, match="failed validation"):
        t._fit_epochs(model, optimizer(model), data, protocol, binding(data, protocol), root, until_epoch=1)
    assert not root.exists()


def test_resume_rejects_changed_identities_and_unvalidated_selection(protocol, tmp_path):
    data, model = SyntheticData(), TinyPublicModel()
    identity = binding(data, protocol)
    result = t._fit_epochs(model, optimizer(model), data, protocol, identity, tmp_path / "checkpoints", until_epoch=1)
    path, digest = result["last"]["path"], result["last"]["manifest_digest"]
    for key in ("training_split", "protocol", "execution_source"):
        changed = {**identity, key: {"different": True}}
        restored = TinyPublicModel()
        with pytest.raises(ValueError, match="identity"):
            t._read_checkpoint(path, digest, restored, optimizer(restored), changed, data, protocol)
    restored = TinyPublicModel()
    with pytest.raises(ValueError, match="identity"):
        t._read_checkpoint(path, "0" * 64, restored, optimizer(restored), identity, data, protocol)
    original = verify_artifact(path, expected_artifact_type="qwen3_large_training_checkpoint", expected_schema_version=t.VERSION)
    fields = {k: v for k, v in original.manifest.items() if k not in ("file_table", "manifest_digest")}
    files = {name: (original.path / name).read_bytes() for name in original.manifest["file_table"]}
    fake = publish_artifact(tmp_path / "wrong-selection", manifest_fields={**fields, "selection": {
        "best_epoch": 0, "best_validation_kl": None, "epochs_without_improvement": 0}}, files=files)
    with pytest.raises(ValueError, match="validation/selection"):
        t._read_checkpoint(fake.path, fake.manifest_digest, restored, optimizer(restored), identity, data, protocol)


def test_data_adapter_reuses_development_reader_and_rejects_test_and_small_publication(v2):
    _, _, split, publication = v2
    with pytest.raises(ArtifactValidationError, match="not a training input"):
        t.open_training_data(split.artifact.path, publication_digest=split.digest, split_digest=split.digest)
    with pytest.raises(ValueError, match="4800"):
        t.open_training_data(publication.path, publication_digest=publication.manifest_digest, split_digest=split.digest)


def test_data_adapter_binds_fixed_split_and_preserves_fold_metadata(monkeypatch):
    from werewolf.development_publication_v2 import VerifiedDevelopmentPublicationV2
    ids = tuple(f"verified-development-{i:04}" for i in range(4800))
    view = SimpleNamespace(game_ids=ids, max_structured_token_count=1024, fold_manifest={"unchanged": True})
    publication = VerifiedDevelopmentPublicationV2(None, {}, "a" * 64, view, {})
    calls = []
    def opened(path, **kw):
        calls.append((path, kw))
        return publication
    primary = object()
    monkeypatch.setattr(t, "open_development_input", opened)
    monkeypatch.setattr(t, "select_primary_population", lambda p: primary if p is publication else pytest.fail())
    data = t.open_training_data("synthetic-development", publication_digest="a" * 64, split_digest="b" * 64)
    assert calls == [("synthetic-development", {"expected_publication_digest": "a" * 64, "expected_split_digest": "b" * 64})]
    assert data.primary is primary
    assert len(data.split["training_game_ids"]) == 4500 and len(data.split["validation_game_ids"]) == 300
    assert view.fold_manifest == {"unchanged": True}

    validation_ids = data.split["validation_game_ids"]
    validation_view = SimpleNamespace(game_ids=tuple(validation_ids), max_structured_token_count=1024)
    publication = replace(publication, public_view=validation_view,
                          manifest={"development_games": [{"game_id": game_id} for game_id in ids]})
    subset = t.open_training_data("synthetic-development", publication_digest="a" * 64, split_digest="b" * 64,
                                  validation_game_ids=validation_ids)
    assert calls[-1][1]["game_ids"] == validation_ids and subset.split == data.split
    assert subset.publication.public_view.game_ids == tuple(validation_ids)
    validation_view.game_ids = tuple(data.split["training_game_ids"][:300])
    with pytest.raises(ValueError, match="fixed Validation game set"):
        t.open_training_data("synthetic-development", publication_digest="a" * 64, split_digest="b" * 64,
                             validation_game_ids=validation_ids)
    publication = replace(publication, public_view=view)
    view.max_structured_token_count = 1025
    with pytest.raises(ValueError, match="capacity"):
        t.open_training_data("synthetic-development", publication_digest="a" * 64, split_digest="b" * 64)


def test_actual_large_optimizer_and_checkpoint_roundtrip(example, protocol, tmp_path):
    temporal, sample, mask = example
    validation = replace(sample, game_id="synthetic-validation")
    samples = {sample.game_id: [sample], validation.game_id: [validation]}
    data = SimpleNamespace(split={"training_game_ids": [sample.game_id], "validation_game_ids": [validation.game_id]},
                           samples=lambda game_id: samples[game_id], primary=ObserverEligibility(
                               {game_id: {sample.boundary_id: mask.tolist()} for game_id in samples},
                               {"test_scope": "synthetic_cpu_only"}))
    model = Qwen3ToMLarge(temporal, initialization_seed=protocol.initialization_seed)
    opt = optimizer(model)
    result = t._fit_epochs(model, opt, data, protocol, binding(data, protocol), tmp_path / "checkpoints", until_epoch=1)
    before = t._capture(model, opt)
    with torch.no_grad():
        next(model.parameters()).add_(1)
    artifact, history = t._read_checkpoint(result["last"]["path"], result["last"]["manifest_digest"],
                                          model, opt, binding(data, protocol), data, protocol)
    after = t._capture(model, opt)
    assert len(history) == artifact.manifest["completed_epoch"] == 1
    assert artifact.manifest["optimizer_step"] == 1
    assert before[1] == after[1]
    assert all(state["step"].ndim == 0 and state["step"].item() == 1 for state in opt.state.values())


def test_epoch_ce_weights_tail_by_actual_game_count(protocol, tmp_path, monkeypatch):
    data, model = SyntheticData(training=52, validation=1), TinyPublicModel()
    original = t.train_batch
    def measured(*args, **kwargs):
        original(*args, **kwargs)
        return 1. if len(args[3]) == 32 else 3.
    monkeypatch.setattr(t, "train_batch", measured)
    result = t._fit_epochs(model, optimizer(model), data, protocol, binding(data, protocol),
                           tmp_path / "checkpoints", until_epoch=1)
    assert result["history"][0]["training_ce"] == pytest.approx((32 + 3 * 20) / 52)
    assert result["history"][0]["optimizer_step"] == 2


def test_public_entry_binds_graph_data_protocol_and_actual_source(example, protocol, tmp_path, monkeypatch):
    temporal, _, _ = example
    data = SyntheticData()
    data.publication = SimpleNamespace(manifest_digest="a" * 64)
    source = {"source_revision": "b" * 40}
    config = SimpleNamespace(device="cpu", torch_num_threads=1, source_revision=source["source_revision"])
    runtime = t.runtime_provenance(config)
    source["implementation_digest"] = runtime["implementation_digest"]
    # 与其他对话并行写源码时，合成来源夹具保持固定；不把它冒充干净 Source。
    monkeypatch.setattr(t, "runtime_provenance", lambda config: dict(runtime))
    monkeypatch.setattr(t, "attest_source", lambda: source)
    opened = []
    def data_reader(path, **kw):
        opened.append((path, kw))
        return data
    monkeypatch.setattr(t, "open_training_data", data_reader)
    captured = {}
    def fit(model, opt, inputs, p, identity, root, **kw):
        captured.update(identity)
        assert isinstance(model, Qwen3ToMLarge) and inputs is data and p == protocol
        assert all(group["lr"] == 5e-5 and group["weight_decay"] == 1e-5 for group in opt.param_groups)
        assert identity["runtime"]["device"] == "cpu"
        assert kw == {"resume": None, "until_epoch": 1}
        return {"test_scope": "synthetic_cpu_only"}
    monkeypatch.setattr(t, "_fit_epochs", fit)
    result = t.train_large("synthetic-development", publication_digest="a" * 64, split_digest="c" * 64,
                           temporal=temporal, protocol=protocol, checkpoint_root=tmp_path / "unused", until_epoch=1)
    assert result == {"test_scope": "synthetic_cpu_only"} and not (tmp_path / "unused").exists()
    assert opened == [("synthetic-development", {"publication_digest": "a" * 64, "split_digest": "c" * 64})]
    assert captured["graph"] == t.graph_identity()
    assert captured["publication_digest"] == "a" * 64 and captured["split_manifest_digest"] == "c" * 64
    assert captured["execution_source"] == source and captured["training_split"] == data.split
    assert captured["protocol"] == protocol.identity() and captured["population"] == data.primary.metadata
    assert captured["temporal_artifact_digest"] == temporal.artifact_digest
    source["implementation_digest"] = "0" * 64
    with pytest.raises(ValueError, match="Source changed"):
        t.train_large("synthetic-development", publication_digest="a" * 64, split_digest="c" * 64,
                      temporal=temporal, protocol=protocol, checkpoint_root=tmp_path / "unused", until_epoch=1)


def test_resume_does_not_silently_choose_an_older_checkpoint(protocol, tmp_path, monkeypatch):
    data, model = SyntheticData(), TinyPublicModel()
    values = iter([.4, .5])
    monkeypatch.setattr(t, "validate", lambda *a, **kw: next(values))
    root = tmp_path / "checkpoints"
    t._fit_epochs(model, optimizer(model), data, protocol, binding(data, protocol), root, until_epoch=2)
    first = verify_artifact(root / "1", expected_artifact_type="qwen3_large_training_checkpoint",
                            expected_schema_version=t.VERSION)
    restored = TinyPublicModel()
    with pytest.raises(ValueError, match="later checkpoints"):
        t._fit_epochs(restored, optimizer(restored), data, protocol, binding(data, protocol), root, until_epoch=2,
                      resume=(first.path, first.manifest_digest))


def test_historical_final_checkpoint_is_not_a_large_checkpoint(protocol, tmp_path):
    data, model = SyntheticData(), TinyPublicModel()
    old = publish_artifact(tmp_path / "old", manifest_fields={
        "artifact_type": "canonical_model_state", "schema_version": "classic7_model_state_v1"}, files={})
    with pytest.raises(ArtifactValidationError):
        t._read_checkpoint(old.path, old.manifest_digest, model, optimizer(model), binding(data, protocol), data, protocol)


@pytest.mark.parametrize("location", ["linked", "parent-traversal", "inside-artifact"])
def test_output_root_rejects_links_traversal_and_frozen_artifacts(protocol, tmp_path, location):
    data, model = SyntheticData(), TinyPublicModel()
    if location == "linked":
        root = tmp_path / "link"
        root.symlink_to(tmp_path / "target")
    elif location == "parent-traversal":
        root = tmp_path / "ordinary/../output"
    else:
        artifact = publish_artifact(tmp_path / "frozen", manifest_fields={
            "artifact_type": "synthetic_frozen_input", "schema_version": "fixture"}, files={"input": b"unchanged"})
        root = artifact.path / "checkpoints"
    before = deepcopy(model.state_dict())
    opt = optimizer(model)
    with pytest.raises(ValueError, match="checkpoint"):
        t._fit_epochs(model, opt, data, protocol, binding(data, protocol), root, until_epoch=1)
    assert not model.calls and not opt.state and not data.visits
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    if location == "inside-artifact":
        assert verify_artifact(artifact.path, expected_artifact_type="synthetic_frozen_input",
                               expected_schema_version="fixture").manifest_digest == artifact.manifest_digest


def test_microbatch_with_no_effective_rows_does_not_change_game_weight():
    data = SyntheticData(training=2)
    data.primary.rows["train-001"]["pre-0"] = [False] * 7
    batch = [{"game_id": "train-000", "shift": 0}, {"game_id": "train-001", "shift": 3}]
    full, micro = TinyPublicModel(), TinyPublicModel()
    expected = full_loss(full, data, batch)
    expected.backward()
    actual = t.train_batch(micro, torch.optim.SGD(micro.parameters(), lr=0), data, batch, pre_microbatch_size=1)
    assert actual == pytest.approx(float(expected.detach()), rel=2e-6)
    for left, right in zip(full.parameters(), micro.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, atol=2e-7, rtol=3e-6)
    assert len(micro.calls) == 2


def test_nonfinite_gradient_refuses_optimizer_update():
    data, model = SyntheticData(), TinyPublicModel()
    model.weight.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    before = deepcopy(model.state_dict())
    opt = optimizer(model)
    with pytest.raises(ValueError, match="nonfinite Large accumulated gradient"):
        t.train_batch(model, opt, data, [{"game_id": "train-000", "shift": 0}], pre_microbatch_size=1)
    assert not opt.state
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
