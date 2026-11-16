"""外部投票前代理监督的边界、排除规则和原数据隔离测试。"""

from copy import deepcopy
import json
import math
from types import SimpleNamespace

import pytest
import torch

from werewolf.artifact_io import ArtifactConflictError, canonical_json_bytes, sha256_bytes
from werewolf.tom import makto
from werewolf.tom.dataset import ExperimentCapacity


def _label(actor=3):
    return {"player": actor, "day": "1-1", "reason": "不能进入公开输入的判断理由",
            "role_prediction": {str(j): ["werewolf" if j in (1, 2) else "simple_villager"]
                                for j in range(1, 8)}}


def _events():
    roles = ["werewolf", "werewolf", "seer", "witch"] + ["simple_villager"] * 3
    return [*({"event": "roles", "content": {"player": j, "role": role}}
              for j, role in enumerate(roles, 1)),
            {"event": "cycle_round", "content": {"round": 1, "status": "night"}},
            {"event": "wolf_kill", "content": {"secret": "不能进入输入"}},
            {"event": "cycle_round", "content": {"round": 1, "status": "day", "context": "今天是平安夜"}},
            {"event": "speech", "content": {"player": 3, "day": "1-1", "context": "我是预言家，1号查杀。"}},
            {"event": "speech_summary", "content": {"identity_label": "不能进入输入"}},
            {"event": "vote_start", "content": None},
            {"event": "voted", "content": _label()},
            {"event": "voted", "content": _label(1)},
            {"event": "vote_results", "content": {"1": 2, "2": 2, "3": 2}},
            {"event": "game_end", "content": {"winner": "不能进入输入"}}]


def _annotations(sources):
    return {row["key"]: {"text_sha256": row["text_sha256"],
                         "actions": [["player3", "point_as_seer", "player3"],
                                     ["player3", "check_as_werewolf", "player1"]],
                         "origin": "test_fixture", "independently_reviewed": False}
            for row in makto.speech_records(sources)}


def _extract(events=None):
    path = makto.source_paths()[0]
    payload = canonical_json_bytes(_events() if events is None else events)
    return makto.extract_game(path, payload, _annotations({path: payload}))


def test_label_conversion_uses_candidates_not_actual_identity_or_reason():
    label = _label()
    result = makto.convert_label(label)
    assert result["label_usable"]
    assert result["wolf_support"] == ["player1", "player2"]
    assert result["q"] == [.5, .5, 0, 0, 0, 0, 0]
    label["reason"] = "相反的理由也不能自动改写标签"
    assert makto.convert_label(label) == result
    label["role_prediction"]["4"] = ["seer", "witch"]
    assert makto.convert_label(label)["label_usable"]
    label["role_prediction"]["4"] = ["werewolf"]
    converted = makto.convert_label(label)
    assert converted["label_usable"] and len(converted["wolf_support"]) == 3
    assert converted["q"][2] == 0
    assert sum(converted["q"]) == pytest.approx(1)


@pytest.mark.parametrize("change,reason", [
    (lambda p: p.pop("4"), "missing_target"),
    (lambda p: p.update({"4": ["NA"]}), "NA"),
    (lambda p: p.update({"4": ["werewolf", "seer"]}), "mixed_wolf_nonwolf"),
    (lambda p: p.update({"4": []}), "invalid_candidates"),
    (lambda p: p.update({"4": ["guard"]}), "invalid_candidates"),
    (lambda p: p.update({"1": ["seer"], "2": ["witch"]}), "empty_support"),
])
def test_unconvertible_label_never_becomes_uniform_or_successful_empty(change, reason):
    label = _label()
    change(label["role_prediction"])
    result = makto.convert_label(label)
    assert reason in result["exclusion_reasons"]
    assert not result["label_usable"] and not any(result["q"])
    assert result["raw_role_prediction"] == label["role_prediction"]


def test_confirmed_conflict_quarantines_without_rewriting():
    label = _label()
    result = makto.convert_label(label, quarantined=True)
    assert result["raw_role_prediction"] == label["role_prediction"]
    assert result["quarantined"] and not result["label_usable"]
    assert result["wolf_support"] == ["player1", "player2"]
    assert result["exclusion_reasons"] == ["confirmed_text_label_conflict"]
    assert all(split == "train" for split, _, _ in makto.CONFLICTS)


def test_vote_start_input_excludes_current_votes_gold_and_private_or_future_events():
    game, inputs, labels = _extract()
    boundary = inputs[0]
    serialized = json.dumps(boundary, ensure_ascii=False)
    assert "不能进入" not in serialized
    assert "role_prediction" not in serialized
    assert "raw_role_prediction" not in json.dumps(game)
    assert not any(e["event_type"] in {"vote_result", "exile_result"} for e in boundary["public_events"])
    assert boundary["public_events"][-1]["phase"] == "vote"
    assert boundary["plan"]["tokens"][-1]["token_type"] == "phase_change"
    assert labels[0]["effective"] and not labels[1]["effective"]
    altered = _events()
    # 角色真值只影响外围人口；即使改变私人信息和后续选票，公开PRE完全不变。
    altered[0]["content"]["role"], altered[2]["content"]["role"] = "seer", "werewolf"
    altered[8]["content"] = {"secret": "其他秘密"}
    altered[-2]["content"] = {"1": 4, "2": 4, "3": 4}
    _, other_inputs, other_labels = _extract(altered)
    assert other_inputs == inputs
    assert other_labels[0]["q"] == labels[0]["q"]
    assert not other_labels[0]["effective"]


def test_missing_public_deaths_never_reconstructed_from_private_actions():
    events = _events()
    del events[9]["content"]["context"]
    _, inputs, labels = _extract(events)
    assert inputs[0]["plan"] is None and inputs[0]["alive"] is None
    assert labels[0]["label_usable"] and not labels[0]["effective"]
    assert "missing_public_death_announcement" in labels[0]["exclusion_reasons"]


def test_pk_vote_includes_previous_public_votes_but_not_current_votes():
    events = _events()[:-1]
    events[-1]["content"] = {"1": 1, "2": 2, "3": None}
    events.extend([
        {"event": "speech", "content": {"player": 3, "day": "1-2", "context": "我仍然认为1号是狼。"}},
        {"event": "vote_start", "content": None},
        {"event": "voted", "content": {**_label(), "day": "1-2"}},
        {"event": "vote_results", "content": {"1": 2, "2": 1, "3": 1}},
    ])
    _, inputs, _ = _extract(events)
    assert len(inputs) == 2
    assert inputs[1]["phase"] == "pk_vote"
    assert sum(e["event_type"] == "vote_result" for e in inputs[1]["public_events"]) == 1
    assert any(t["token_type"] == "speech_action" for t in inputs[1]["plan"]["tokens"])
    assert all(t["day"] in (0, 1) for t in inputs[1]["plan"]["tokens"])


def test_annotations_bound_to_public_text_and_fixed_source_coverage():
    sources = {p: canonical_json_bytes(_events()) for p in makto.source_paths()}
    annotations = _annotations(sources)
    with pytest.raises(ValueError, match="fixed 20 Train and 8 Test"):
        makto.prepare(dict(list(sources.items())[:-1]), annotations)
    with pytest.raises(ValueError, match="annotation coverage"):
        makto.prepare(sources, {})
    first = next(iter(annotations))
    bad = deepcopy(annotations)
    bad[first]["text_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="text identity"):
        makto.prepare(sources, bad)
    bad = deepcopy(annotations)
    bad[first]["actions"][0][0] = "player4"
    with pytest.raises(ValueError, match="subject"):
        makto.prepare(sources, bad)


def test_artifact_split_reader_partial_mask_tensors_and_no_overwrite(tmp_path, monkeypatch):
    sources = {p: canonical_json_bytes(_events()) for p in makto.source_paths()}
    output = tmp_path / "makto"
    artifact = makto.publish(output, sources, _annotations(sources))
    assert artifact.manifest["revision"] == makto.REVISION
    assert artifact.manifest["label_timing"] == "UNKNOWN"
    opened = []
    read = makto.read_artifact_file

    def record_read(envelope, path):
        opened.append(path)
        return read(envelope, path)

    monkeypatch.setattr(makto, "read_artifact_file", record_read)
    data = makto.MaKTOToMDataset(output, "train", capacity=ExperimentCapacity(64))
    assert opened == ["public/train.jsonl", "supervision/train.jsonl"]
    assert len(data) == 20
    sample = data[0]
    assert sample.public.attention_mask.sum() == sample.token_count
    assert sample.q.shape == (7, 7) and sample.public.event_ids.shape == (64,)
    assert data.effective_mask(sample).tolist() == [False, False, True, False, False, False, False]
    assert torch.all(sample.q.diagonal() == 0)
    with pytest.raises(ValueError, match="capacity"):
        makto.MaKTOToMDataset(output, "train", capacity=ExperimentCapacity(1))
    before = sha256_bytes((output / "manifest.json").read_bytes())
    assert makto.publish(output, sources, _annotations(sources)).manifest_digest == artifact.manifest_digest
    changed = _annotations(sources)
    next(iter(changed.values()))["origin"] = "changed"
    with pytest.raises(ArtifactConflictError):
        makto.publish(output, sources, changed)
    assert sha256_bytes((output / "manifest.json").read_bytes()) == before


def _holdout_data():
    """仿真同样的游戏行数分布；不读取官方Test或真实Checkpoint。"""
    from werewolf.tom.dataset import CanonicalToMSample, PublicTensors

    counts = [0, 1, 3, 2, 2, 2, 3, 5, 3, 4, 2, 1, 2, 3, 2, 7, 1, 4, 4, 3]
    samples = []
    for game, count in zip(range(21, 41), counts):
        game_id = f"makto/train/seer_witch/game_{game}"
        for row in range(max(1, count)):
            arrays = {name: torch.tensor([1, 2]) for name in (
                "event_ids", "source_ids", "action_ids", "target_ids", "day_ids", "phase_ids")}
            public = PublicTensors(**arrays, attention_mask=torch.ones(2, dtype=torch.bool))
            q = torch.zeros(7, 7)
            # 不同游戏具有不同目标熵，能区分游戏宏平均与行平均。
            support = 1 + (game % 3)
            q[0, 1:1 + support] = 1 / support
            observed = torch.tensor([bool(count), False, False, False, False, False, False])
            samples.append(CanonicalToMSample(game_id, f"{game_id}/{row}", "a" * 64, "b" * 64,
                                             2, public, q, observed, torch.ones(7, dtype=torch.bool)))
    return SimpleNamespace(split="train", source_digest="d" * 64, samples=samples,
                           effective_mask=lambda sample: sample.label_observed & sample.observer_alive)


def test_internal_holdout_is_fixed_whole_game_and_official_test_is_forbidden():
    data = _holdout_data()
    split = makto.internal_split(data)
    assert split["validation_game_ids"] == [f"makto/train/seer_witch/game_{i}" for i in (31, 34, 37, 38)]
    assert len(split["training_game_ids"]) == 15
    assert len(split["validation_game_ids"]) == 4
    counts = split["effective_rows_by_game"]
    assert sum(counts[g] for g in split["training_game_ids"]) == 44
    assert sum(counts[g] for g in split["validation_game_ids"]) == 10
    assert split["unsupervised_game_ids"] == ["makto/train/seer_witch/game_21"]
    data.samples.reverse()
    assert makto.internal_split(data) == split
    data.split = "test"
    with pytest.raises(ValueError, match="original Train"):
        makto.internal_split(data)


def test_validation_reuses_masks_and_game_macro_and_restores_model_mode():
    data = _holdout_data()
    split = makto.internal_split(data)

    class UniformModel(torch.nn.Module):
        def forward(self, **public):
            assert not self.training and not torch.is_grad_enabled()
            assert set(public) == {"event_ids", "source_ids", "action_ids", "target_ids",
                                   "day_ids", "phase_ids", "attention_mask"}
            result = torch.full((1, 7, 7), -math.log(6), dtype=torch.float64)
            return result.masked_fill(torch.eye(7, dtype=torch.bool), -torch.inf)

    model = UniformModel().train()
    result = makto.validation_scores(data, split, model)
    assert model.training
    assert result["uniform_kl"] == result["epoch35_kl"]
    assert result["absolute_difference_epoch35_minus_uniform"] == 0
    assert result["relative_difference"] == 0
    assert sum(g["effective_rows"] for g in result["games"]) == 10
    macro = sum(g["uniform_kl"] for g in result["games"]) / 4
    pooled = sum(g["uniform_kl"] * g["effective_rows"] for g in result["games"]) / 10
    assert result["uniform_kl"] == pytest.approx(macro)
    assert abs(macro - pooled) > .01
    without_model = makto.validation_scores(data, split)
    assert without_model["row_identity_digest"] == result["row_identity_digest"]
    assert without_model["epoch35_kl"] is None
    split["validation_game_ids"].reverse()
    with pytest.raises(ValueError, match="split identity"):
        makto.validation_scores(data, split)


@pytest.mark.parametrize("invalid", ["digest", "epoch", "graph", "temporal"])
def test_epoch35_loader_rejects_wrong_identity_before_model_construction(monkeypatch, invalid):
    from werewolf.tom import qwen3_large

    checkpoint = SimpleNamespace(manifest_digest=makto.EPOCH35_DIGEST, manifest={
        "completed_epoch": 35, "selection": {"best_epoch": 35},
        "binding": {"graph": qwen3_large.graph_identity(), "temporal_artifact_digest": "f" * 64}})
    temporal = SimpleNamespace(manifest_digest="f" * 64)
    if invalid == "digest":
        checkpoint.manifest_digest = "0" * 64
    elif invalid == "epoch":
        checkpoint.manifest["completed_epoch"] = 34
    elif invalid == "graph":
        checkpoint.manifest["binding"]["graph"]["total_parameters"] = 1
    else:
        temporal.manifest_digest = "0" * 64
    monkeypatch.setattr(makto, "open_artifact_envelope", lambda path, **kwargs:
                        checkpoint if path == "checkpoint" else temporal)
    monkeypatch.setattr(qwen3_large, "Qwen3ToMLarge", lambda *a, **k: pytest.fail("invalid identity reached construction"))
    with pytest.raises(ValueError, match="identity|mismatch"):
        makto.load_epoch35_model("checkpoint", "temporal")


def test_epoch35_reader_restores_only_model_without_old_optimizer_or_rng(monkeypatch):
    from werewolf.artifact_io import TensorValue, tensor_state
    from werewolf.tom import qwen3_large, temporal

    checkpoint = SimpleNamespace(manifest_digest=makto.EPOCH35_DIGEST, manifest={
        "completed_epoch": 35, "selection": {"best_epoch": 35}, "containers": {"model": {}},
        "binding": {"graph": qwen3_large.graph_identity(), "temporal_artifact_digest": "f" * 64,
                    "protocol": {"initialization_seed": 17}}})
    time = SimpleNamespace(manifest_digest="f" * 64, manifest={"parent_digest": "a" * 64})
    monkeypatch.setattr(makto, "open_artifact_envelope", lambda path, **kwargs:
                        checkpoint if path == "checkpoint" else time)
    monkeypatch.setattr(temporal.TemporalCodeProvider, "explicit", lambda *a: "verified-time")
    # 仅缩小模型规模；状态恢复仍使用原有严格键、形状、dtype及逐字节校验。
    model = torch.nn.Linear(2, 7)
    tensors = {name: TensorValue(torch.full_like(value, .25).numpy(), True)
               for name, value in model.state_dict().items()}
    monkeypatch.setattr(qwen3_large, "Qwen3ToMLarge", lambda *a, **k: model)
    monkeypatch.setattr(tensor_state, "decode_tensor_container", lambda *a, **k: tensors)
    reads = []

    def read(envelope, name):
        reads.append(name)
        return b"model bytes"

    monkeypatch.setattr(makto, "read_artifact_file", read)
    restored = makto.load_epoch35_model("checkpoint", "temporal")
    assert reads == ["model.bin"]
    assert not restored.training
    assert all(torch.all(value == .25) for value in restored.state_dict().values())


def test_baseline_cli_never_downloads_or_marks_missing_checkpoint_as_pass(tmp_path, monkeypatch, capsys):
    data = _holdout_data()
    data.source_digest = makto.PREPARED_DIGEST
    monkeypatch.setattr(makto, "MaKTOToMDataset", lambda path, split: data)
    monkeypatch.setattr(makto, "download_sources", lambda: pytest.fail("baseline cannot redownload data"))
    split_path, result_path = tmp_path / "split.json", tmp_path / "baseline.json"
    makto.main(["--baseline", "--data", "existing-data", "--split-record", str(split_path),
                "--output", str(result_path)])
    result = json.loads(result_path.read_bytes())
    assert result["epoch35_status"] == "NOT_RUN" and result["epoch35_kl"] is None
    assert result["quality_status"] == "ANNOTATION_SEMANTICS_REVIEW_REQUIRED"
    assert json.loads(split_path.read_bytes())["dataset_digest"] == makto.PREPARED_DIGEST
    assert json.loads(capsys.readouterr().out) == result


def test_new_records_cannot_modify_an_existing_artifact_or_record(tmp_path):
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="inside an immutable artifact"):
        makto._write_record(frozen / "split.json", {"new": True})
    path = tmp_path / "record.json"
    makto._write_record(path, {"new": True})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        makto._write_record(path, {"new": False})
    assert path.read_bytes() == before


def test_annotation_revision_preserves_sources_gold_and_public_boundary(tmp_path, monkeypatch):
    sources = {p: canonical_json_bytes(_events()) for p in makto.source_paths()}
    original = _annotations(sources)
    key = next(iter(original))
    original[key]["actions"][1][2] = "player2"  # 原公开文本明确说1号，故模拟旧标注编号错误。
    parent = makto.publish(tmp_path / "parent", sources, original)
    monkeypatch.setattr(makto, "PREPARED_DIGEST", parent.manifest_digest)
    corrected = deepcopy(original)
    corrected[key]["actions"][1][2] = "player1"
    correction = {"key": key, "text_sha256": original[key]["text_sha256"],
                  "before": original[key]["actions"], "after": corrected[key]["actions"],
                  "reason": "公开原文目标为1号", "evidence": "1号查杀", "rule": "explicit_target"}
    revision = makto.publish(tmp_path / "revision", sources, corrected,
                             previous=parent.path, corrections=[correction])
    assert revision.manifest["annotation_correction"]["parent_digest"] == parent.manifest_digest
    assert revision.manifest["source_digests"] == parent.manifest["source_digests"]
    for split in ("train", "test"):
        assert (revision.path / f"supervision/{split}.jsonl").read_bytes() == (parent.path / f"supervision/{split}.jsonl").read_bytes()
    assert (revision.path / "corrections.jsonl").is_file()
    before_inputs = [json.loads(x) for x in (parent.path / "public/train.jsonl").read_bytes().splitlines()]
    after_inputs = [json.loads(x) for x in (revision.path / "public/train.jsonl").read_bytes().splitlines()]
    assert before_inputs[0]["prefix_digest"] == after_inputs[0]["prefix_digest"]
    assert before_inputs[0]["plan"]["plan_digest"] != after_inputs[0]["plan"]["plan_digest"]
    with pytest.raises(ValueError, match="exactly the changed"):
        makto.publish(tmp_path / "missing-log", sources, corrected, previous=parent.path)
    bad = {**correction, "before": []}
    with pytest.raises(ValueError, match="evidence identity"):
        makto.publish(tmp_path / "bad-log", sources, corrected, previous=parent.path, corrections=[bad])
    changed_sources = dict(sources)
    events = _events()
    events[8]["content"]["secret"] = "修改私人内容仍属于改变原始来源"
    changed_sources[next(iter(changed_sources))] = canonical_json_bytes(events)
    with pytest.raises(ValueError, match="original sources"):
        makto.publish(tmp_path / "bad-source", changed_sources, corrected, previous=parent.path, corrections=[correction])


@pytest.mark.parametrize("field,match", [("labels", "supervision"), ("inputs", "public boundaries")])
def test_annotation_revision_rejects_label_or_temporal_drift(tmp_path, monkeypatch, field, match):
    sources = {p: canonical_json_bytes(_events()) for p in makto.source_paths()}
    annotations = _annotations(sources)
    parent = makto.publish(tmp_path / "parent", sources, annotations)
    monkeypatch.setattr(makto, "PREPARED_DIGEST", parent.manifest_digest)
    prepared = makto.prepare(sources, annotations)
    if field == "labels":
        prepared[2][0]["q"] = [0] * 7
    else:
        prepared[1][0]["day"] = 99
    monkeypatch.setattr(makto, "prepare", lambda *a: prepared)
    with pytest.raises(ValueError, match=match):
        makto.publish(tmp_path / "invalid", sources, annotations, previous=parent.path)


def test_baseline_requires_explicit_revised_digest_and_keeps_same_game_split(tmp_path, monkeypatch, capsys):
    data = _holdout_data()
    original_split = makto.internal_split(data)
    data.source_digest = "e" * 64
    data.annotation_correction = {"parent_digest": makto.PREPARED_DIGEST}
    revised_split = makto.internal_split(data)
    assert {k: v for k, v in original_split.items() if k not in {"dataset_digest", "split_digest"}} == {
        k: v for k, v in revised_split.items() if k not in {"dataset_digest", "split_digest"}}
    monkeypatch.setattr(makto, "MaKTOToMDataset", lambda path, split: data)
    args = ["--baseline", "--data", "existing-data", "--split-record", str(tmp_path / "split.json"),
            "--output", str(tmp_path / "baseline.json")]
    with pytest.raises(ValueError, match="artifact digest mismatch"):
        makto.main(args)
    assert not (tmp_path / "split.json").exists()
    makto.main([*args, "--expected-data-digest", data.source_digest])
    result = json.loads(capsys.readouterr().out)
    assert result["dataset_digest"] == data.source_digest
    assert result["epoch35_status"] == "NOT_RUN"
    assert result["quality_status"] == "TARGETED_CORRECTIONS_APPLIED_NOT_FULL_CERTIFICATION"
