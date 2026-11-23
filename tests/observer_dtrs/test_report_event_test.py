"""验证公开 Q 先行、固定参数和新事件独立评分的连接逻辑。"""

import json
import math
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import fit_observer_dtrs_report_event as fit
from scripts import run_observer_dtrs_report_event_test as study
from tests.canonical_collection.test_game_bundle import _plan
from tests.observer_dtrs.conftest import controlled_evidence, prefix_for
from werewolf.artifact_io import canonical_jsonl_bytes, open_artifact_envelope, publish_artifact, read_artifact_file
from werewolf.canonical_collection import construct_attempt_claim, publish_canonical_game_bundle
from werewolf.canonical_collection import game_bundle
from werewolf.observer_dtrs.decision import TiePolicy


def predicted(prefix):
    q = [[.75 if j == (i + 1) % 7 else 0. if i == j else .05 for j in range(7)] for i in range(7)]
    return {**{name: getattr(prefix, name) for name in study.PRE_FIELDS},
            "plan_digest": "a" * 64, "token_count": 12, "q": q}


def fake_worker():
    return subprocess.Popen(
        [sys.executable, "-B", "-u", "-c", """
import sys
from scripts.run_observer_dtrs_report_event_test import (
    CAPSULE_DIGEST, CAPSULE_SCOPE, LOADER_DIGEST, PLAYER_IDS, RUNTIME, _prefix_from_record, read_line, write_line,
)
from tests.observer_dtrs.test_report_event_test import predicted

out = sys.stdout.buffer
write_line(out, {"seats": PLAYER_IDS, "loads": 1, "capsule": {
    "capsule_digest": CAPSULE_DIGEST, "scope": CAPSULE_SCOPE, "runtime": RUNTIME,
    "loader_source": {"loader": {"implementation_digest": LOADER_DIGEST, "file_sha256": {}}},
}})
count = 0
for raw in sys.stdin.buffer:
    write_line(out, predicted(_prefix_from_record(read_line(raw))))
    count += 1
write_line(out, {"predictions": count})
"""],
        cwd=study.ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    )


@pytest.fixture
def test_bundle(tmp_path, monkeypatch):
    plan = _plan()
    claim = construct_attempt_claim(plan, ordinal=0, attempt_id="test-attempt",
                                    claim_timestamp_utc="2026-09-04T00:00:00Z")
    evidence, replay = controlled_evidence(
        plan, claim, "test-game", phase="discussion", dead=(), actor="player1",
        supports={"player3": ("player1",), "player4": ("player2",)}, wolves=("player1", "player2"),
    )
    bundle = publish_canonical_game_bundle(tmp_path / "games/test-game", plan=plan, claim=claim,
                                           evidence=evidence, replay_executor=replay)
    members = ({name: bundle.manifest[name] for name in ("game_id", "seed", "attempt_id", "ordinal", "claim_record_digest")}
               | {"bundle_digest": bundle.manifest_digest},)
    split = SimpleNamespace(plan=plan)
    monkeypatch.setattr(study, "open_test", lambda: (split, members))
    monkeypatch.setattr(study, "COLLECTION", tmp_path)
    monkeypatch.setattr(study, "Q_PATH", tmp_path / "q")
    monkeypatch.setattr(study, "RESULT_PATH", tmp_path / "score")
    return split, members, evidence.authoritative_pre_prefixes[0], bundle


def test_predict_stream_uses_public_bundle_only(test_bundle, tmp_path, monkeypatch):
    split, members, prefix, _ = test_bundle
    snapshot = publish_artifact(tmp_path / "source", manifest_fields={
        "artifact_type": study.SNAPSHOT_TYPE, "schema_version": study.SNAPSHOT_TYPE,
    }, files={"snapshot.json": b"{}"})
    monkeypatch.setattr(study, "SNAPSHOT", snapshot.path)
    monkeypatch.setattr(study, "SNAPSHOT_DIGEST", snapshot.manifest_digest)
    monkeypatch.setattr(study, "start_worker", fake_worker)
    read = game_bundle.read_artifact_file

    def public_only(bundle, name):
        assert name == "public/authoritative_pre_prefixes.jsonl"
        return read(bundle, name)

    monkeypatch.setattr(game_bundle, "read_artifact_file", public_only)
    monkeypatch.setattr(game_bundle, "read_bundle_supervision", lambda *_: pytest.fail("supervision opened during predict"))
    study.predict()
    published = open_artifact_envelope(study.Q_PATH, expected_artifact_type=study.Q_TYPE,
                                        expected_schema_version=study.Q_TYPE)
    q, aligned = study.verify_test_q(split, members, published.manifest_digest)
    assert q.manifest["seat_order"] == list(study.PLAYER_IDS)
    assert q.manifest["public_pres"] == q.manifest["complete_q"] == 1
    assert aligned[0][1] == [predicted(prefix)]
    assert read_artifact_file(q, "predictions.jsonl") == canonical_jsonl_bytes([predicted(prefix)])


@pytest.mark.parametrize("failure", ["absent", "hash", "digest", "short", "extra", "identity", "numeric", "population"])
def test_bad_q_rejected_before_supervision(test_bundle, monkeypatch, failure):
    _, members, prefix, _ = test_bundle
    monkeypatch.setattr(game_bundle, "read_bundle_supervision", lambda *_: pytest.fail("supervision opened before Q verified"))
    record = predicted(prefix)
    expected_digest = "b" * 64
    if failure != "absent":
        if failure == "identity":
            record["step_index"] += 1
        if failure == "numeric":
            record["q"][0][0] = .25
        records = [] if failure == "short" else [record, record] if failure == "extra" else [record]
        artifact = publish_artifact(study.Q_PATH, manifest_fields={
            **study.test_binding(), "artifact_type": study.Q_TYPE, "schema_version": study.Q_TYPE,
            "artifact_identity": study.Q_PATH.name, "public_pres": 1, "complete_q": 1,
            "game_bundles": {} if failure == "population" else {"test-game": members[0]["bundle_digest"]},
        }, files={"predictions.jsonl": canonical_jsonl_bytes(records)})
        if failure != "digest":
            expected_digest = artifact.manifest_digest
        if failure == "hash":
            (study.Q_PATH / "predictions.jsonl").write_bytes(b"corrupt\n")
    with pytest.raises(ValueError):
        study.score(expected_digest)
    assert not study.RESULT_PATH.exists()


def test_raw_event_and_source_population(test_bundle):
    _, _, prefix, bundle = test_bundle
    rows, pop = study.test_samples([(bundle, [predicted(prefix)])])
    assert {row.observer_id: row.y for row in rows} == {
        "player3": 1, "player4": 0, "player5": 0, "player6": 0, "player7": 0,
    }
    assert pop["legal_wolf_pres"] == pop["pres"] == pop["source_games"] == pop["games"] == 1
    assert pop["rows"] == 5 and pop["successful_empty"] == 3 and pop["missing_or_failed_reports"] == 0


def test_fixed_coefficients_and_paired_evaluation(monkeypatch):
    monkeypatch.setattr(fit, "fit_logistic", lambda *_: pytest.fail("Test may not fit probability parameters"))
    rows = [fit.Sample(g, str(i), "player4", 1, "discussion", 2, 4,
                       (x,) + (0.,) * (len(fit.FEATURES["M2"]) - 1), y, False)
            for i, (g, x, y) in enumerate((("a", -1., 1), ("b", 0., 0), ("b", 1., 0), ("b", 1., 1)))]
    models = {}
    for model, names in fit.FEATURES.items():
        k = len(names) - len(fit.CONTEXT_FEATURES)
        models[model] = {"features": names, "standardization": {"mean": [.25] * k, "std": [.5] * k},
                         "base": {"intercept": 0., "coefficients": [1.] + [0.] * (len(names) - 1)},
                         "calibrator": {"intercept": 0., "coefficients": [2.]}}
        p = study.fixed_probabilities(rows, model, models[model])
        assert list(p["raw"].values()) == pytest.approx([1 / (1 + math.exp(-r.features[0])) for r in rows])
        assert list(p["calibrated"].values()) == pytest.approx([1 / (1 + math.exp(-2 * r.features[0])) for r in rows])
    monkeypatch.setattr(fit, "bootstrap_indices", lambda games, replicas, seed: np.array([[0, 0], [0, 1], [1, 1]]))
    result = study.evaluate_test(rows, {"models": models})
    m1 = result["decision"]["M1"]["overall"]["game_equal"]
    both = m1["three_way"], m1["forced_binary"]
    assert all(sum(r["rows"] for r in summary["regions"].values()) == len(rows) for summary in both)
    assert m1["three_way"]["realized_risk"] == pytest.approx(.5 / 2 + (.125 + .5) / 6)
    assert m1["forced_binary"]["realized_risk"] == pytest.approx(.5 / 2 + .5 / 6)
    assert m1["three_way"]["NEG_error_rate"] == 1 and m1["three_way"]["POS_error_rate"] == .5
    assert all(x == {"percentile_95": None, "valid_replicates": 2, "replicates": 3}
               for x in m1["three_way"]["error_intervals"].values())
    assert result["paired_loss_differences"]["game_equal"]["M2_minus_M1_three_way"]["percentile_95"] == [0., 0.]
    assert result["loss"]["tie_policy"] == TiePolicy.DEFER_FIRST.value


def test_q_identity_and_absolute_axis_checks():
    prefix = prefix_for()
    record = predicted(prefix)
    study.validate_q(record, prefix)
    record["q"] = record["q"][::-1]
    with pytest.raises(ValueError, match="zero-self"):
        study.validate_q(record, prefix)
