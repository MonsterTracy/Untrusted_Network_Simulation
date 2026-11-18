"""Large 专用的游戏等权 Epoch 训练；复用公开数据边界及既有状态编码。"""

from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch

from werewolf.artifact_io import canonical_json_bytes, publish_artifact, sha256_bytes, verify_artifact
from werewolf.artifact_io.tensor_state import decode_tensor_container
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.development_publication_v2 import VerifiedDevelopmentPublicationV2
from werewolf.tom import training
from werewolf.tom.backbone_execution import _capture, _validate_optimizer
from werewolf.tom.dataset import CanonicalToMDataset, ExperimentCapacity, PublicTensors, rotate
from werewolf.tom.development_input import open_development_input
from werewolf.tom.experiment import configure_runtime, read_json, runtime_provenance
from werewolf.tom.paper_study import attest_source
from werewolf.tom.population import select_primary_population
from werewolf.tom.protocol import FIELD_ENCODING, development_holdout_split, digest_fields
from werewolf.tom.qwen3_large import Qwen3ToMLarge, graph_identity
from werewolf.tom.scoring import paired_summary, row_cross_entropy, row_kl
from werewolf.tom.state import restore_model_tensors


VERSION = "classic7_qwen3_large_epoch_training_v1"
ORDER_VERSION = "classic7_large_epoch_hash_order_v1"
ROTATION_VERSION = "classic7_large_cyclic_seven_rotation_v1"


def _digest(value):
    return sha256_bytes(canonical_json_bytes(value))


@dataclass(frozen=True)
class LargeTrainingProtocol:
    """种子及 PRE 内存预算必须显式给出；本批不提供第二套数值协议。"""

    initialization_seed: int
    schedule_seed: int
    rng_seed: int
    pre_microbatch_size: int

    def __post_init__(self):
        for seed in (self.initialization_seed, self.schedule_seed, self.rng_seed):
            if type(seed) is not int or not 0 <= seed < 2**63:
                raise ValueError("Large training seeds must be explicit integers in 0..2**63-1")
        if type(self.pre_microbatch_size) is not int or self.pre_microbatch_size < 1:
            raise ValueError("PRE microbatch size must be a positive integer")

    def identity(self):
        return {
            "version": VERSION, **asdict(self), "field_encoding": FIELD_ENCODING,
            "order_version": ORDER_VERSION, "rotation_version": ROTATION_VERSION,
            "training_games": 4500, "validation_games": 300,
            "game_batch_size": 32, "tail_batch_games": 20,
            "learning_rate": 5e-5, "optimizer": "adamw", "adam_betas": [.9, .999],
            "adam_eps": 1e-8, "weight_decay": 1e-5, "scheduler": "constant",
            "max_epochs": 80, "patience": 10, "min_delta": 0,
            "loss": "mean_games(mean_all_primary_observed_rows(cross_entropy))",
            "validation": "once_after_complete_epoch_canonical_seats_game_macro_kl",
            "selection": "strictly_lower_validation_kl_earliest_epoch_on_tie",
            "recovery": "last_complete_validated_epoch_model_optimizer_rng",
        }


@dataclass(frozen=True)
class _DevelopmentData:
    publication: object
    split: dict
    primary: object

    def samples(self, game_id):
        # 按局张量化，避免把 4800 局的 1024 长度 Padding 同时驻留内存。
        return CanonicalToMDataset(self.publication.public_view, [game_id], ExperimentCapacity(1024)).samples


def open_training_data(path, *, publication_digest, split_digest, validation_game_ids=None):
    """锚定完整 Development V2，可只读取固定 Validation 的游戏记录。"""
    scope = {} if validation_game_ids is None else {"game_ids": validation_game_ids}
    publication = open_development_input(path, expected_publication_digest=publication_digest,
                                         expected_split_digest=split_digest, **scope)
    if not isinstance(publication, VerifiedDevelopmentPublicationV2):
        raise ValueError("Large training requires Development V2")
    view = publication.public_view
    # 验证模式只以全量清单验证划分身份，实际只加载验证局的公开记录。
    ids = view.game_ids if validation_game_ids is None else [g["game_id"] for g in publication.manifest["development_games"]]
    split = development_holdout_split(ids, development_game_ids=ids)
    if validation_game_ids is not None and set(view.game_ids) != set(split["validation_game_ids"]):
        raise ValueError("Large evaluation requires the fixed Validation game set")
    if view.max_structured_token_count > 1024:
        raise ValueError("complete Development PRE exceeds Large capacity; truncation forbidden")
    return _DevelopmentData(publication, split, select_primary_population(publication))


def epoch_schedule(game_ids, epoch, protocol):
    return game_schedule(game_ids, epoch, protocol.schedule_seed)


def game_schedule(game_ids, epoch, schedule_seed):
    """每局每 Epoch 一次；任意连续七个 Epoch 恰好覆盖该局七种旋转。"""
    games = tuple(sorted(game_ids))
    if (not games or len(set(games)) != len(games)
            or any(type(g) is not str or not g for g in games)
            or type(epoch) is not int or not 1 <= epoch <= 80):
        raise ValueError("invalid Large epoch/game identities")
    ordered = sorted(games, key=lambda g: (digest_fields(ORDER_VERSION, schedule_seed, epoch, g), g))
    rows = [{"game_id": g, "shift": (int.from_bytes(
        digest_fields(ROTATION_VERSION, schedule_seed, g), "big") + epoch - 1) % 7} for g in ordered]
    return [rows[start:start + 32] for start in range(0, len(rows), 32)]


def _primary_mask(data, sample):
    return data.primary.mask(sample.game_id, sample.boundary_id) & sample.label_observed


def _effective_masks(samples, effective_mask):
    masks = []
    for sample in samples:
        mask = effective_mask(sample)
        if not torch.allclose(sample.q.sum(-1)[mask], torch.ones_like(sample.q.sum(-1)[mask])):
            raise ValueError("effective belief targets must sum to one")
        masks.append(mask)
    count = sum(int(mask.sum()) for mask in masks)
    if not count:
        raise ValueError("game has zero Primary effective supervision rows")
    return masks, count


def _microbatches(samples, masks, size, device, shift):
    for start in range(0, len(samples), size):
        pairs = [rotate(sample, shift, mask) for sample, mask in
                 zip(samples[start:start + size], masks[start:start + size], strict=True)]
        public = PublicTensors.stack([sample.public for sample, _ in pairs])
        # 仅去掉整个微批都不用的右 Padding，完整 PRE 不截断。
        length = max(sample.token_count for sample, _ in pairs)
        inputs = {name: value[:, :length].to(device) for name, value in public.kwargs().items()}
        q = torch.stack([sample.q for sample, _ in pairs]).to(device)
        mask = torch.stack([mask for _, mask in pairs]).to(device)
        if mask.any():
            yield inputs, q, mask


def train_batch(model, optimizer, data, batch, *, pre_microbatch_size):
    return train_game_batch(model, optimizer, batch, samples_for_game=data.samples,
                            effective_mask=lambda sample: _primary_mask(data, sample),
                            pre_microbatch_size=pre_microbatch_size, training_mode=True)


def train_game_batch(model, optimizer, batch, *, samples_for_game, effective_mask,
                     pre_microbatch_size, training_mode):
    """一次更新：sum(微批 CE)/(整局有效行数 * 实际 Batch 游戏数)。"""
    if (not batch or len(batch) > 32 or len({row["game_id"] for row in batch}) != len(batch)
            or type(pre_microbatch_size) is not int or pre_microbatch_size < 1):
        raise ValueError("invalid Large game/PRE batch")
    prepared = []
    for row in batch:
        if type(row["shift"]) is not int or not 0 <= row["shift"] < 7:
            raise ValueError("invalid Large rotation")
        samples = samples_for_game(row["game_id"])
        masks, count = _effective_masks(samples, effective_mask)
        prepared.append((samples, masks, count, row["shift"]))
    optimizer.zero_grad(set_to_none=True)
    model.train(training_mode)
    device = next(model.parameters()).device
    game_losses = []
    for samples, masks, count, shift in prepared:
        ce_sum = 0.
        for public, q, mask in _microbatches(samples, masks, pre_microbatch_size, device, shift):
            rows = row_cross_entropy(model(**public), q)[mask]
            (rows.sum() / (count * len(batch))).backward()
            ce_sum += float(rows.detach().double().sum())
        game_losses.append(ce_sum / count)
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError("nonfinite Large accumulated gradient; optimizer not updated")
    optimizer.step()
    return math.fsum(game_losses) / len(batch)


def validate(model, data, *, pre_microbatch_size):
    """只遍历固定 Validation；局内有效行平均后，对游戏等权平均。"""
    if type(pre_microbatch_size) is not int or pre_microbatch_size < 1:
        raise ValueError("PRE microbatch size must be a positive integer")
    was_training = model.training
    device = next(model.parameters()).device
    values = []
    try:
        model.eval()
        with torch.inference_mode():
            for game_id in data.split["validation_game_ids"]:
                samples = data.samples(game_id)
                masks, count = _effective_masks(samples, lambda sample: _primary_mask(data, sample))
                total = 0.
                for public, q, mask in _microbatches(samples, masks, pre_microbatch_size, device, 0):
                    total += float(row_kl(model(**public).double(), q.double())[mask].sum())
                values.append(total / count)
    finally:
        model.train(was_training)
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("invalid Large Validation KL")
    return math.fsum(values) / len(values)


def validate_pair(epoch35, head_only, data, *, pre_microbatch_size):
    """同一个原始座号微批上的三套 KL，并按原始成功报告类型标记行。"""
    device = next(epoch35.parameters()).device
    diagonal = torch.eye(7, dtype=torch.bool, device=device)
    uniform = torch.full((7, 7), -math.log(6), dtype=torch.float64, device=device).masked_fill(diagonal, -torch.inf)
    epoch35.eval()
    head_only.eval()
    games = []
    with torch.inference_mode():
        for game_id in data.split["validation_game_ids"]:
            samples = data.samples(game_id)
            masks, count = _effective_masks(samples, lambda sample: _primary_mask(data, sample))
            observations = data.publication.public_view.load_game(game_id).belief_observations
            empty = {(o.boundary_id, o.observer_id) for o in observations
                     if o.status.value == "success" and not o.suspicion_support}
            empty_rows = torch.tensor([(s.boundary_id, PLAYER_IDS[i]) in empty
                                       for s, mask in zip(samples, masks, strict=True)
                                       for i in mask.nonzero().flatten().tolist()], dtype=torch.bool)
            scores = []
            for public, q, mask in _microbatches(samples, masks, pre_microbatch_size, device, 0):
                seats = mask.nonzero(as_tuple=True)[1]
                self_seats = diagonal[seats]
                q = q[mask].double()
                values = [row_kl(uniform[seats], q)]
                for model in (epoch35, head_only):
                    logp = model(**public)[mask]
                    if (logp.dtype != torch.float32 or not torch.isfinite(logp[~self_seats]).all()
                            or not torch.isneginf(logp[self_seats]).all()):
                        raise ValueError(f"invalid Validation log-probabilities for {game_id}")
                    values.append(row_kl(logp.double(), q))
                scores.append(torch.stack(values, dim=-1))
            values = torch.cat(scores).cpu()
            if not torch.isfinite(values).all():
                raise ValueError(f"nonfinite Validation KL for {game_id}")
            groups = {"successful_empty": values[empty_rows], "successful_nonempty": values[~empty_rows]}
            games.append({"game_id": game_id, "pres": len(samples), "effective_rows": count,
                          **_validation_kl_means(values), "report_types": {
                              name: {"effective_rows": len(rows), **_validation_kl_means(rows)}
                              for name, rows in groups.items()}})
    return games


def _validation_kl_means(rows):
    uniform, epoch35, head = rows.mean(0).tolist() if len(rows) else (None, None, None)
    return {"uniform_kl": uniform, "epoch35_kl": epoch35, "head_only_kl": head,
            "delta": head - epoch35 if len(rows) else None}


def summarize_validation_pair(games):
    """主指标对游戏等权；分层仅对存在该类报告的游戏等权。"""
    def macro(records):
        keys = ("uniform_kl", "epoch35_kl", "head_only_kl", "delta")
        return {k: math.fsum(r[k] for r in records) / len(records) if records else None for k in keys}

    result = macro(games)
    report_types = {}
    for name in ("successful_empty", "successful_nonempty"):
        pairs = [(g, g["report_types"][name]) for g in games if g["report_types"][name]["effective_rows"]]
        records = [r for _, r in pairs]
        # 回到完整 Primary 的局内分母，使两类贡献恰好相加为总差值。
        report_types[name] = {"games": len(records), "effective_rows": sum(r["effective_rows"] for r in records),
                              **macro(records), "contribution_to_primary_delta": math.fsum(
                                  r["delta"] * r["effective_rows"] / g["effective_rows"] for g, r in pairs) / len(games)}
    seed, replicates = 20260928, 10_000
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(len(games), size=(replicates, len(games)), dtype=np.int32)
    bootstrap = paired_summary([g["head_only_kl"] for g in games], [g["epoch35_kl"] for g in games], indices, .95)
    return {**result, "epoch35_improvement_over_uniform_percent": 100 * (result["uniform_kl"] - result["epoch35_kl"]) / result["uniform_kl"],
            "head_only_delta_percent_of_epoch35": 100 * result["delta"] / result["epoch35_kl"],
            "game_outcomes": {"improved": sum(g["delta"] < 0 for g in games),
                              "worsened": sum(g["delta"] > 0 for g in games),
                              "tied": sum(g["delta"] == 0 for g in games)},
            "paired_bootstrap": {**bootstrap, "seed": seed, "replicates": replicates, "rng": "PCG64",
                                 "indices_digest": sha256_bytes(indices.astype("<i4", copy=False).tobytes())},
            "report_types": report_types}


def _selection(history, data, protocol):
    """从完整 Epoch 日志重新推导选择状态，不信任外部 best 标志。"""
    best_epoch, best_kl, stale = None, None, 0
    steps_per_epoch = math.ceil(len(data.split["training_game_ids"]) / 32)
    for epoch, row in enumerate(history, 1):
        if (set(row) != {"epoch", "optimizer_step", "training_ce", "validation_kl", "schedule_digest"}
                or type(row["epoch"]) is not int or row["epoch"] != epoch
                or type(row["optimizer_step"]) is not int or row["optimizer_step"] != epoch * steps_per_epoch
                or row["schedule_digest"] != _digest(epoch_schedule(data.split["training_game_ids"], epoch, protocol))
                or any(type(row[k]) not in (float, int) or not math.isfinite(row[k])
                       for k in ("training_ce", "validation_kl")) or stale >= 10):
            raise ValueError("invalid or post-stopping Large epoch history")
        if best_kl is None or row["validation_kl"] < best_kl:
            best_epoch, best_kl, stale = epoch, row["validation_kl"], 0
        else:
            stale += 1
    return {"best_epoch": best_epoch, "best_validation_kl": best_kl, "epochs_without_improvement": stale}


def _save_checkpoint(path, model, optimizer, binding, history, data, protocol, previous):
    selection = _selection(history, data, protocol)
    if not history or selection["best_epoch"] is None:
        raise ValueError("unvalidated state cannot be a Large checkpoint")
    containers, files, scalars, rng = _capture(model, optimizer)
    files["history.json"] = canonical_json_bytes(history)
    best_digest = None
    if selection["best_epoch"] != len(history):
        best_digest = (previous.manifest_digest if previous.manifest["selection"]["best_epoch"] == len(history) - 1
                       else previous.manifest["best_checkpoint_digest"])
    return publish_artifact(path, manifest_fields={
        "artifact_type": "qwen3_large_training_checkpoint", "schema_version": VERSION,
        "binding": binding, "completed_epoch": len(history), "optimizer_step": history[-1]["optimizer_step"],
        "selection": selection, "containers": containers, "optimizer_scalars": scalars, "rng_scalars": rng,
        "previous_checkpoint_digest": previous.manifest_digest if previous is not None else None,
        "best_checkpoint_digest": best_digest,
    }, files=files)


def _read_checkpoint(path, expected_digest, model, optimizer, binding, data, protocol):
    artifact = verify_artifact(path, expected_artifact_type="qwen3_large_training_checkpoint", expected_schema_version=VERSION)
    m = artifact.manifest
    expected_fields = {"artifact_type", "schema_version", "binding", "completed_epoch", "optimizer_step",
                       "selection", "containers", "optimizer_scalars", "rng_scalars", "file_table", "manifest_digest",
                       "previous_checkpoint_digest", "best_checkpoint_digest"}
    if (artifact.manifest_digest != expected_digest or set(m) != expected_fields
            or canonical_json_bytes(m["binding"]) != canonical_json_bytes(binding)
            or set(m["containers"]) != {"model", "optimizer", "rng"}
            or set(m["file_table"]) != {"model.bin", "optimizer.bin", "rng.bin", "history.json"}):
        raise ValueError("Large checkpoint identity/schema mismatch")
    raw = (artifact.path / "history.json").read_bytes()
    history = read_json(raw)
    selection = _selection(history, data, protocol)
    if (not history or m["selection"] != selection
            or type(m["completed_epoch"]) is not int or m["completed_epoch"] != len(history)
            or type(m["optimizer_step"]) is not int or m["optimizer_step"] != history[-1]["optimizer_step"]):
        raise ValueError("Large checkpoint validation/selection mismatch")
    decoded = {name: decode_tensor_container(container, (artifact.path / f"{name}.bin").read_bytes(), payload_path=f"{name}.bin")
               for name, container in m["containers"].items()}
    _validate_optimizer(model, optimizer, decoded["optimizer"], m["optimizer_scalars"], m["optimizer_step"])
    expected_rng = {"torch_cpu", "numpy_state"} | {
        f"torch_cuda_{i}" for i in range(torch.cuda.device_count()) if torch.cuda.is_initialized()}
    if set(decoded["rng"]) != expected_rng or any(t.trainable for t in decoded["rng"].values()):
        raise ValueError("Large checkpoint RNG coverage mismatch")
    restore_model_tensors(model, decoded["model"])
    # 既有容器把标量编码成单元素数组；恢复 AdamW step 的原生零维形状。
    tensors = {name: replace(tensor, array=tensor.array.reshape(())) if name.endswith(".step") else tensor
               for name, tensor in decoded["optimizer"].items()}
    training._optimizer_restore(optimizer, tensors, m["optimizer_scalars"])
    training._rng_restore(decoded["rng"], m["rng_scalars"])
    containers, files, scalars, rng = _capture(model, optimizer)
    if (canonical_json_bytes(containers) != canonical_json_bytes(m["containers"])
            or canonical_json_bytes(scalars) != canonical_json_bytes(m["optimizer_scalars"])
            or canonical_json_bytes(rng) != canonical_json_bytes(m["rng_scalars"])
            or any(payload != (artifact.path / name).read_bytes() for name, payload in files.items())):
        raise ValueError("Large checkpoint restore is not byte exact")
    return artifact, history


def _best_checkpoint(last, root, history, data, protocol):
    epoch = last.manifest["selection"]["best_epoch"]
    best = verify_artifact(root / str(epoch), expected_artifact_type="qwen3_large_training_checkpoint", expected_schema_version=VERSION)
    expected = last.manifest_digest if epoch == len(history) else last.manifest["best_checkpoint_digest"]
    if (best.manifest_digest != expected or best.manifest["completed_epoch"] != epoch
            or canonical_json_bytes(best.manifest["binding"]) != canonical_json_bytes(last.manifest["binding"])
            or read_json((best.path / "history.json").read_bytes()) != history[:epoch]
            or best.manifest["selection"] != _selection(history[:epoch], data, protocol)):
        raise ValueError("Large best checkpoint does not match validated history")
    return best


def _fit_epochs(model, optimizer, data, protocol, binding, checkpoint_root, *, resume=None, until_epoch=80):
    """唯一训练循环；只发布完成 Validation 的 Epoch，不发布正式 Final Seal。"""
    if type(until_epoch) is not int or not 1 <= until_epoch <= 80:
        raise ValueError("invalid Large epoch boundary")
    root = Path(checkpoint_root).absolute()
    if ".." in root.parts or any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("Large checkpoint root cannot traverse parent paths or symbolic links")
    if any((p / "manifest.json").exists() for p in (root, *root.parents)):
        raise ValueError("Large checkpoints cannot be written inside an immutable artifact")
    history, last = [], None
    if resume is None:
        if root.exists() and any(root.iterdir()):
            raise ValueError("Large checkpoint root is not empty; explicit resume required")
    else:
        path, digest = resume
        last, history = _read_checkpoint(path, digest, model, optimizer, binding, data, protocol)
        if Path(path).absolute() != (root / str(len(history))).absolute():
            raise ValueError("Large resume checkpoint must belong to this epoch lineage")
        if {p.name for p in root.iterdir()} != {str(i) for i in range(1, len(history) + 1)}:
            raise ValueError("Large resume lineage contains missing or later checkpoints")
        _best_checkpoint(last, root, history, data, protocol)
    selection = _selection(history, data, protocol)
    if until_epoch < len(history):
        raise ValueError("Large epoch boundary precedes resume state")
    for epoch in range(len(history) + 1, until_epoch + 1):
        if selection["epochs_without_improvement"] >= 10:
            break
        batches = epoch_schedule(data.split["training_game_ids"], epoch, protocol)
        losses = [train_batch(model, optimizer, data, batch, pre_microbatch_size=protocol.pre_microbatch_size)
                  for batch in batches]
        validation_kl = validate(model, data, pre_microbatch_size=protocol.pre_microbatch_size)
        history.append({"epoch": epoch, "optimizer_step": epoch * len(batches),
                        "training_ce": math.fsum(loss * len(batch) for loss, batch in zip(losses, batches, strict=True))
                                       / len(data.split["training_game_ids"]),
                        "validation_kl": validation_kl, "schedule_digest": _digest(batches)})
        last = _save_checkpoint(root / str(epoch), model, optimizer, binding, history, data, protocol, last)
        selection = last.manifest["selection"]
    if last is None:
        raise ValueError("Large training has no validated checkpoint")
    best = _best_checkpoint(last, root, history, data, protocol)
    reason = "patience" if selection["epochs_without_improvement"] >= 10 else ("max_epochs" if len(history) == 80 else "epoch_boundary")
    return {"last": {"path": str(last.path), "manifest_digest": last.manifest_digest},
            "best": {"path": str(best.path), "manifest_digest": best.manifest_digest},
            "selection": selection, "history": history, "stop_reason": reason, "training_identity": _digest(binding)}


def train_large(publication_path, *, publication_digest, split_digest, temporal, protocol,
                checkpoint_root, resume=None, until_epoch=80, device="cpu", torch_num_threads=1):
    """正式数据适配入口；本批只用合成数据测试，不提供 CLI 或正式模型 Seal。"""
    if not isinstance(protocol, LargeTrainingProtocol):
        raise TypeError("Large training requires an explicit LargeTrainingProtocol")
    source = attest_source()
    data = open_training_data(publication_path, publication_digest=publication_digest, split_digest=split_digest)
    if not temporal.explicit_enabled:
        raise ValueError("Large training requires the existing explicit Day+Phase code")
    if device not in ("cpu", "cuda") or type(torch_num_threads) is not int or torch_num_threads < 1:
        raise ValueError("invalid Large runtime configuration")
    runtime_config = SimpleNamespace(device=device, torch_num_threads=torch_num_threads, source_revision=source["source_revision"])
    configure_runtime(runtime_config)
    runtime = runtime_provenance(runtime_config)
    if runtime["implementation_digest"] != source["implementation_digest"]:
        raise ValueError("Large execution Source changed during preparation")
    model = Qwen3ToMLarge(temporal, initialization_seed=protocol.initialization_seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(.9, .999), eps=1e-8,
                                 weight_decay=1e-5, foreach=False, fused=False)
    # 模型构造完成后再固定训练 RNG；顺序和旋转另用域分离的纯哈希。
    seed = int.from_bytes(digest_fields(VERSION, "training_rng", protocol.rng_seed), "big") % 2**63
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.random.default_generator.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    binding = {
        "graph": graph_identity(), "publication_digest": data.publication.manifest_digest,
        "split_manifest_digest": split_digest, "training_split": data.split,
        "population": data.primary.metadata, "temporal_artifact_digest": temporal.artifact_digest,
        "protocol": protocol.identity(), "execution_source": source, "runtime": runtime,
    }
    return _fit_epochs(model, optimizer, data, protocol, binding, checkpoint_root,
                       resume=resume, until_epoch=until_epoch)
