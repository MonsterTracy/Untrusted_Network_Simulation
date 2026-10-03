"""Local synthetic sealed-artifact integration, never a formal estimator fit.

Digest overrides apply only to isolated tmp_path test envelopes. Historical
private paths deliberately do not exist and are never followed by the adapter.
"""

from copy import deepcopy
import json

import pytest

from scripts import phase2_terminal_estimator as cli
from werewolf import phase2_terminal_estimator as stats
from werewolf import phase2_terminal_estimator_io as io
from werewolf.artifact_io import canonical_json_bytes, canonical_jsonl_bytes
from werewolf.artifact_io import publish_artifact, read_artifact_file, sha256_bytes, verify_artifact
from tests.phase2.terminal_estimator_test_support import synthetic_rows


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def inputs(tmp_path, monkeypatch, *, change_itt=None, change_source=None):
    """Entirely synthetic features/losses; only schema/digest plumbing mirrors V1."""
    assignments, executions, consequences = [], [], []
    recovered = None
    for index, row in enumerate(synthetic_rows()):
        failed = index == 0
        game_id = io.itt_source.FAILED_GAME if failed else row["game_id"]
        assignment = {
            "assigned_action": row["assigned_action"], "assignment_probability": .5,
            "opportunity": {"identity": {"game_id": game_id, "phase": "speech", "acting_wolf": "player2",
                                         "candidate_j": "player4", "boundary_id": "synthetic-PRE",
                                         "prefix_digest": "f" * 64},
                            "s_pre": row["S_pre"], "evidence": {"p_tilde_j": row["p_tilde_j"]}},
            "treatment": {"assignment_source": "randomized_pilot", "requested_action": row["assigned_action"]},
        }
        if change_source:
            change_source(assignment, index)
        key = sha256_bytes(canonical_json_bytes(assignment))
        assignments.append(assignment)
        executions.append({"assignment_digest": key, "game_id": game_id, "success": not failed,
                           "canonical_commit_success": not failed, "failure_reason": "PRIVATE_FACT_CLAIM" if failed else None,
                           "theta_audit_label": None})
        day = {"Y": "other_nonwolf_exiled", "s_plus": [2, 4], "v_ref": row["V_ref"], "l_ref": row["L_ref"],
               "reference_artifact_digest": io.itt_source.REFERENCE_DIGEST}
        if failed:
            recovered = day
        else:
            consequences.append({"assignment_digest": key, "game_id": game_id, "day_consequence": day,
                                 "theta_audit_label": None})
    run_digest, plan_digest = "a" * 64, "b" * 64
    formal = publish_artifact(tmp_path / "synthetic-formal", manifest_fields={
        "artifact_type": "phase2_online_terminal_pilot", "schema_version": "phase2_online_terminal_pilot_v1",
        "study_name": io.itt_source.FORMAL_NAME, "campaign_purpose": "pilot", "completion_status": "COMPLETE",
        "target_assignment_count": 120, "source_provenance": {"commit": io.itt_source.FORMAL_SOURCE},
        "reference_artifact_digest": io.itt_source.REFERENCE_DIGEST, "synthetic_test_fixture": True,
        "server_run_provenance": {"inputs_digest": run_digest, "game_plan_digest": plan_digest,
                                  "work_directory": str(tmp_path / "MUST-NOT-BE-READ")}}, files={
            "assignments.jsonl": canonical_jsonl_bytes(assignments),
            "executions.jsonl": canonical_jsonl_bytes(executions),
            "consequences.jsonl": canonical_jsonl_bytes(consequences)})
    monkeypatch.setattr(io.itt_source, "FORMAL_DIGEST", formal.manifest_digest)
    indexed, e, c = io.itt_source.exact_joins(assignments, executions, consequences)
    rows = io.itt_source.itt_rows(indexed, e, c, recovered)
    if change_itt:
        change_itt(rows)
    data = canonical_jsonl_bytes(rows)
    audit = {"recovery_rule_version": io.itt_source.RECOVERY_RULE, "game_id": io.itt_source.FAILED_GAME,
             "assignment_id": sha256_bytes(canonical_json_bytes(assignments[0])),
             "source_formal_manifest_digest": formal.manifest_digest, "reference_table_digest": io.itt_source.REFERENCE_DIGEST,
             "recovered_outcome": recovered, "canonical_partial_evidence": {
                 "execution": {"path": str(tmp_path / "MUST-NOT-BE-READ" / "execution.json"),
                               "file_sha256": io.itt_source.EXECUTION_SHA},
                 "game-result": {"path": str(tmp_path / "MUST-NOT-BE-READ" / "game-result.json"),
                                 "file_sha256": io.itt_source.GAME_RESULT_SHA}}}
    itt = publish_artifact(tmp_path / "synthetic-itt", manifest_fields={
        "artifact_type": "phase2_terminal_itt_analysis", "schema_version": io.itt_source.VERSION,
        "study_name": io.itt_source.NAME, "analysis_only": True, "primary_population": "all_120_randomized_assignments",
        "source_formal_manifest_digest": formal.manifest_digest, "source_formal_commit": io.itt_source.FORMAL_SOURCE,
        "reference_table_digest": io.itt_source.REFERENCE_DIGEST, "source_run_inputs_digest": run_digest,
        "canonical_game_plan_digest": plan_digest, "recovery_rule_version": io.itt_source.RECOVERY_RULE,
        "row_count": 120, "recovery_count": 1, "fitted_lambda": None, "synthetic_test_fixture": True}, files={
            "itt_rows.jsonl": data, "recovery_audit.json": canonical_json_bytes(audit)})
    monkeypatch.setattr(io, "ITT_DIGEST", itt.manifest_digest)
    monkeypatch.setattr(io, "ITT_ROWS_SHA256", sha256_bytes(data))
    return formal, itt


@pytest.mark.parametrize("pin,owner", [("FORMAL_DIGEST", io.itt_source), ("ITT_DIGEST", io), ("ITT_ROWS_SHA256", io)])
def test_wrong_input_pin_fails(tmp_path, monkeypatch, pin, owner):
    formal, itt = inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(owner, pin, "0" * 64)
    with pytest.raises(ValueError, match="digest mismatch"):
        io.load_inputs(formal.path, itt.path)


@pytest.mark.parametrize("fault", ["row_count", "duplicate", "wrong_counts", "missing_loss", "changed_score", "changed_outcome"])
def test_bad_itt_rows_never_skipped(tmp_path, monkeypatch, fault):
    def change(rows):
        if fault == "row_count":
            rows.pop()
        elif fault == "duplicate":
            rows[0] = deepcopy(rows[1])
        elif fault == "wrong_counts":
            rows[0]["assigned_action"] = "REDIRECT"
        elif fault == "missing_loss":
            rows[0].pop("L_ref")
        elif fault == "changed_score":
            rows[0]["p_tilde_j"] += .01
        else:
            rows[0]["L_ref"] += .01
    formal, itt = inputs(tmp_path, monkeypatch, change_itt=change)
    with pytest.raises(ValueError):
        io.load_inputs(formal.path, itt.path)


def test_phase_transport_not_silently_accepted(tmp_path, monkeypatch):
    def change(assignment, index):
        if index == 1:
            assignment["opportunity"]["identity"]["phase"] = "speech_pk"
    formal, itt = inputs(tmp_path, monkeypatch, change_source=change)
    with pytest.raises(ValueError, match="only covers speech"):
        io.load_inputs(formal.path, itt.path)


def test_complete_synthetic_analysis_deterministic_immutable_exclusive_and_predictions(tmp_path, monkeypatch):
    formal, itt = inputs(tmp_path, monkeypatch)
    before = snapshot(formal.path), snapshot(itt.path)
    captures = []
    original = io.source_provenance
    def capture():
        captures.append(True)
        return original()
    monkeypatch.setattr(io, "source_provenance", capture)
    first = io.analyze(formal.path, itt.path, tmp_path / "output-a")
    second = io.analyze(formal.path, itt.path, tmp_path / "output-b")
    assert len(captures) == 2  # exactly once for each analysis, never recollected at publish
    assert first["status"] == "SANITY_PASSED_FOR_REVIEW"
    assert first["manifest_digest"] == second["manifest_digest"]
    assert snapshot(tmp_path / "output-a") == snapshot(tmp_path / "output-b")
    assert before == (snapshot(formal.path), snapshot(itt.path))
    artifact = verify_artifact(tmp_path / "output-a", expected_artifact_type="phase2_terminal_estimator",
                               expected_schema_version=io.VERSION)
    assert set(artifact.manifest["file_table"]) == {
        "primary_itt.json", "fisher_supplement.json", "models/m0.json", "models/m1.json", "models/m2.json",
        "support_gate.json", "diagnostics.json", "report.md"}
    for name, entry in artifact.manifest["file_table"].items():
        data = read_artifact_file(artifact, name)
        assert entry == {"sha256": sha256_bytes(data), "byte_size": len(data)}
    m = artifact.manifest
    assert m["classic_lambda_identified"] is False
    assert m["fisher"]["seed"] == stats.FISHER_SEED and m["fisher"]["repetitions"] == 100000
    assert "time" not in m and "destination" not in m
    all_keys = set()
    def keys(value):
        if isinstance(value, dict):
            all_keys.update(value)
            for child in value.values(): keys(child)
        elif isinstance(value, list):
            for child in value: keys(child)
    keys(m)
    for name in artifact.manifest["file_table"]:
        if name.endswith(".json"): keys(json.loads(read_artifact_file(artifact, name)))
    assert not any(key.startswith("lambda_") for key in all_keys)
    assert not {"router", "Probe", "threshold", "c_PUSH", "c_REDIRECT"} & all_keys
    verified = io.load_estimator(artifact.path, expected_digest=artifact.manifest_digest)
    assert verified.predict("PUSH", [2, 4], .1)["supported"] is True
    assert verified.predict("REDIRECT", [2, 3], .1)["risk"] is None
    assert verified.audit("REDIRECT", [2, 3], .1)["audit_only"] is True
    assert verified.predict("PUSH", [2, 5], .1) == verified.predict("PUSH", [2, 5], .1)
    # Small coefficient drift is numerically sane but no longer artifact-bound.
    intact = deepcopy(verified.models)
    verified.models["M2"]["coefficients"][0] += .01
    assert stats.model_diagnostics(verified.models["M2"])["passed"] is True
    assert verified.predict("PUSH", [2, 5], .1)["status"] == "LINEAGE_MISMATCH"
    assert verified.predict("PUSH", [2, 5], .1, model_name="M0")["risk"] is None
    assert verified.audit("PUSH", [2, 5], .1)["risk"] is None
    verified.models.clear()
    verified.models.update(intact)
    # Invalid M2 must close even an otherwise valid M0 request; no fallback.
    verified.models["M2"]["coefficients"][0] = 10
    assert verified.predict("PUSH", [2, 5], .1, model_name="M0")["risk"] is None
    with pytest.raises(ValueError, match="already exists"):
        io.analyze(formal.path, itt.path, artifact.path)
    with pytest.raises(ValueError, match="manifest digest"):
        io.load_estimator(artifact.path, expected_digest="0" * 64)


def test_output_must_be_disjoint_from_inputs(tmp_path, monkeypatch):
    formal, itt = inputs(tmp_path, monkeypatch)
    for destination in (formal.path / "estimator", itt.path / "estimator", tmp_path):
        with pytest.raises(ValueError, match="disjoint"):
            io.analyze(formal.path, itt.path, destination)


def test_exclusive_publication_rejects_even_identical_race(tmp_path, monkeypatch):
    destination = tmp_path / "racing-output"
    original = io.artifact_io._publish_directory_noreplace
    def race(staging, target):
        # A concurrent publisher creates a byte-identical artifact first.
        import shutil
        shutil.copytree(staging, target)
        return original(staging, target)
    monkeypatch.setattr(io.artifact_io, "_publish_directory_noreplace", race)
    with pytest.raises(FileExistsError):
        io._exclusive_publish(destination, {"artifact_type": "phase2_terminal_estimator", "schema_version": io.VERSION},
                              {"test.json": b"{}"})
    assert destination.exists()
    assert list(tmp_path.glob(".racing-output.staging-*")) == []


@pytest.mark.parametrize("stage", ["write", "promotion"])
def test_publication_failure_cleans_only_own_staging(tmp_path, monkeypatch, stage):
    destination = tmp_path / "failed-output"
    witness = tmp_path / "existing-evidence"
    witness.mkdir()
    (witness / "keep").write_bytes(b"do not touch")
    if stage == "write":
        original = io.artifact_io._write_fsynced
        def fail(path, data):
            original(path, data)
            raise OSError("simulated partial-write failure")
        monkeypatch.setattr(io.artifact_io, "_write_fsynced", fail)
    else:
        def fail(*args):
            raise OSError("simulated promotion failure")
        monkeypatch.setattr(io.artifact_io, "_publish_directory_noreplace", fail)
    with pytest.raises(OSError, match="simulated"):
        io._exclusive_publish(destination, {"artifact_type": "phase2_terminal_estimator", "schema_version": io.VERSION},
                              {"test.json": b"{}"})
    assert not destination.exists()
    assert list(tmp_path.glob(".failed-output.staging-*")) == []
    assert (witness / "keep").read_bytes() == b"do not touch"


def test_corrupt_staged_payload_never_becomes_destination(tmp_path, monkeypatch):
    destination = tmp_path / "corrupt-output"
    original = io.artifact_io._write_fsynced
    def corrupt(path, data):
        original(path, data)
        if path.name == "test.json":
            path.write_bytes(b"corrupt")
    monkeypatch.setattr(io.artifact_io, "_write_fsynced", corrupt)
    with pytest.raises(ValueError):
        io._exclusive_publish(destination, {"artifact_type": "phase2_terminal_estimator", "schema_version": io.VERSION},
                              {"test.json": b"{}"})
    assert not destination.exists()
    assert list(tmp_path.glob(".corrupt-output.staging-*")) == []


def test_source_drift_stops_publication(tmp_path, monkeypatch):
    formal, itt = inputs(tmp_path, monkeypatch)
    real = io._source_files
    calls = []
    def drifting():
        calls.append(True)
        record = real()
        if len(calls) > 1: record[io.PROTOCOL] = "0" * 64
        return record
    monkeypatch.setattr(io, "_source_files", drifting)
    with pytest.raises(ValueError, match="source changed"):
        io.analyze(formal.path, itt.path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_input_drift_stops_publication(tmp_path, monkeypatch):
    formal, itt = inputs(tmp_path, monkeypatch)
    original = stats.fit_models
    def tamper_after_fit(rows):
        models = original(rows)
        path = itt.path / "itt_rows.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
        return models
    monkeypatch.setattr(stats, "fit_models", tamper_after_fit)
    with pytest.raises(ValueError):
        io.analyze(formal.path, itt.path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("pin", ["PROTOCOL_DIGEST", "SUPPORT_REVIEW_DIGEST"])
def test_frozen_document_pin_fails_before_computation(tmp_path, monkeypatch, pin):
    formal, itt = inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(io, pin, "0" * 64)
    def forbidden(*args, **kwargs):
        pytest.fail("invalid protocol must fail before computation")
    monkeypatch.setattr(stats, "primary_itt", forbidden)
    with pytest.raises(ValueError, match="digest mismatch"):
        io.analyze(formal.path, itt.path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_cli_bad_lineage_nonzero_no_output(tmp_path, monkeypatch, capsys):
    formal, itt = inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(io, "ITT_DIGEST", "0" * 64)
    assert cli.main(["--formal", str(formal.path), "--itt", str(itt.path),
                     "--destination", str(tmp_path / "output")]) == 1
    assert "digest mismatch" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()


def test_cli_success_runs_real_independent_entrypoint(tmp_path, monkeypatch, capsys):
    formal, itt = inputs(tmp_path, monkeypatch)
    assert cli.main(["--formal", str(formal.path), "--itt", str(itt.path),
                     "--destination", str(tmp_path / "output")]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["N"] == 120 and summary["status"] == "SANITY_PASSED_FOR_REVIEW"
    assert summary["treatment_counts"] == {"PUSH": 58, "REDIRECT": 62}


@pytest.mark.parametrize("name", ["manifest.json", "models/m2.json", "support_gate.json", "primary_itt.json"])
def test_estimator_bytes_tamper_rejected_before_prediction(tmp_path, monkeypatch, name):
    formal, itt = inputs(tmp_path, monkeypatch)
    before = snapshot(formal.path), snapshot(itt.path)
    summary = io.analyze(formal.path, itt.path, tmp_path / "output")
    path = tmp_path / "output" / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        io.load_estimator(tmp_path / "output", expected_digest=summary["manifest_digest"])
    assert before == (snapshot(formal.path), snapshot(itt.path))


def test_failed_M2_keeps_primary_and_all_models_cli_nonzero(tmp_path, monkeypatch):
    def constant_score(assignment, index):
        assignment["opportunity"]["evidence"]["p_tilde_j"] = .1
    formal, itt = inputs(tmp_path, monkeypatch, change_source=constant_score)
    destination = tmp_path / "failed-model-output"
    assert cli.main(["--formal", str(formal.path), "--itt", str(itt.path), "--destination", str(destination)]) == 2
    artifact = verify_artifact(destination, expected_artifact_type="phase2_terminal_estimator", expected_schema_version=io.VERSION)
    assert artifact.manifest["status"] == "FAIL_CLOSED"
    assert json.loads(read_artifact_file(artifact, "primary_itt.json"))["row_count"] == 120
    m2 = json.loads(read_artifact_file(artifact, "models/m2.json"))
    assert m2["coefficients"] is None and m2["diagnostics"]["passed"] is False
    verified = io.load_estimator(destination, expected_digest=artifact.manifest_digest)
    assert verified.predict("PUSH", [2, 5], .1, model_name="M0")["risk"] is None
