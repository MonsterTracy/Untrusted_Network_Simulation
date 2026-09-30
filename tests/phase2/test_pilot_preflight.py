"""Pilot cannot start on a clean tree until real capabilities are certified."""

from dataclasses import replace
from pathlib import Path
import subprocess

import pytest

from scripts.phase2_intervention_preflight import (
    FrozenArtifactRequirement, _frozen_q_runtime_ready, _smoke_v3_gate,
    assess_pilot_preflight,
    freeze_online_source_provenance, tracked_source_clean,
)
from werewolf.artifact_io import publish_artifact
from werewolf.artifact_io import canonical_json_bytes, canonical_jsonl_bytes, sha256_bytes
from werewolf.phase2_pilot_execution import Phase2PilotExecutionEnvelopeV1
from tests.phase2.test_intervention_risk import values
from werewolf.phase2_outcome import reference_tables_digest


def _git(repo: Path, *args):
    subprocess.run(("git", "-C", str(repo), *args), check=True,
                   capture_output=True, text=True)


def test_preflight_ignores_untracked_but_rejects_staged_and_tracked_dirty(tmp_path,
                                                                         monkeypatch):
    import scripts.phase2_intervention_preflight as preflight
    monkeypatch.setattr(preflight, "SOURCE_FILES", ("tracked.txt",))
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("base")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-qm", "base")
    artifact = publish_artifact(
        tmp_path / "frozen",
        manifest_fields={"artifact_type": "pilot-test", "schema_version": "v1"},
        files={"data.json": b"{}"},
    )
    requirement = FrozenArtifactRequirement(
        artifact.path, "pilot-test", "v1", artifact.manifest_digest)
    assert tracked_source_clean(tmp_path)
    result = assess_pilot_preflight(tmp_path, frozen_artifacts=(requirement,))
    assert result.tracked_clean and result.frozen_artifacts_verified
    assert result.source_provenance == freeze_online_source_provenance(tmp_path)
    frozen = result.source_provenance
    assert frozen["source_sha256"]["tracked.txt"]
    assert not result.final_mapper_lineage_verified
    assert not result.ready
    assert not result.paired_branch_replay_ready
    assert "FULL_PRE_CHECKPOINT_UNAVAILABLE" not in result.blockers
    assert "SMOKE_V3_GATE_NOT_VERIFIED" in result.blockers
    assert "PHASE2_BACKEND_AUDIT_UNBOUND" in result.blockers
    assert "DURABLE_ASSIGNMENT_LEDGER_UNBOUND" in result.blockers
    assert "FINAL_MAPPER_LINEAGE_NOT_VERIFIED" in result.blockers
    assert "SOURCE_COMMIT_PIN_MISMATCH" in result.blockers
    assert "REFERENCE_TABLE_PIN_MISMATCH_OR_UNAVAILABLE" in result.blockers
    head = frozen["commit"]
    reference = values()
    pinned = assess_pilot_preflight(
        tmp_path, frozen_artifacts=(requirement,),
        expected_source_commit=head,
        reference_tables=reference,
        reference_artifact_digest=reference_tables_digest(reference),
        expected_reference_tables_digest=reference_tables_digest(reference))
    assert "SOURCE_COMMIT_PIN_MISMATCH" not in pinned.blockers
    assert "REFERENCE_TABLE_PIN_MISMATCH_OR_UNAVAILABLE" not in pinned.blockers
    assert pinned.source_commit_pin == head
    assert pinned.reference_tables_digest_pin == reference_tables_digest(reference)
    assert "SOURCE_COMMIT_PIN_MISMATCH" in assess_pilot_preflight(
        tmp_path, frozen_artifacts=(requirement,),
        expected_source_commit="0" * 40).blockers
    assert "REFERENCE_TABLE_PIN_MISMATCH_OR_UNAVAILABLE" in assess_pilot_preflight(
        tmp_path, frozen_artifacts=(requirement,),
        reference_tables=reference,
        reference_artifact_digest=reference_tables_digest(reference),
        expected_reference_tables_digest="0" * 64).blockers
    wrong_digest = FrozenArtifactRequirement(artifact.path, "pilot-test", "v1", "0" * 64)
    assert not assess_pilot_preflight(
        tmp_path, frozen_artifacts=(wrong_digest,)).frozen_artifacts_verified
    assert not assess_pilot_preflight(
        tmp_path, frozen_artifacts=()).frozen_artifacts_verified
    tracked.write_text("modified")
    assert not tracked_source_clean(tmp_path)
    assert frozen == result.source_provenance  # pre-LLM object does not drift
    _git(tmp_path, "add", "tracked.txt")
    assert not tracked_source_clean(tmp_path)


def test_q_seal_string_on_duck_typed_predictor_is_not_frozen_runtime():
    from types import SimpleNamespace
    from scripts.qwen3_gameplay_predictor import SEAL_DIGEST
    assert not _frozen_q_runtime_ready(SimpleNamespace(
        seal_digest=SEAL_DIGEST, predict=lambda *_: []))


def test_untracked_pilot_source_is_blocked_without_blocking_unrelated_untracked(tmp_path,
                                                                               monkeypatch):
    import scripts.phase2_intervention_preflight as preflight
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "tracked.py").write_text("pass\n")
    _git(tmp_path, "add", "tracked.py")
    _git(tmp_path, "commit", "-qm", "base")
    (tmp_path / "research.md").write_text("unrelated")
    monkeypatch.setattr(preflight, "SOURCE_FILES", ("tracked.py",))
    assert freeze_online_source_provenance(tmp_path) is not None
    (tmp_path / "pilot.py").write_text("new source")
    monkeypatch.setattr(preflight, "SOURCE_FILES", ("tracked.py", "pilot.py"))
    assert tracked_source_clean(tmp_path)
    assert freeze_online_source_provenance(tmp_path) is None


def test_execution_schema_cannot_publish_an_aborted_outcome():
    record = Phase2PilotExecutionEnvelopeV1(
        "a" * 40, "phase2_checkpoint_v1", "b" * 64,
        "c" * 64, "d" * 64, "e" * 64, 17,
        "PUSH", "ABORTED", "CHECKPOINT_UNAVAILABLE")
    assert record.digest() == record.digest()
    assert record.to_record()["canonical_event_digest"] is None
    with pytest.raises(ValueError, match="cannot publish an outcome"):
        replace(record, terminal_consequence_digest="f" * 64)
    with pytest.raises(ValueError, match="requires verified speech"):
        replace(record, execution_status="COMPLETED", abort_reason=None)


def test_online_smoke_preflight_requires_real_passed_gate_and_mapper_lineage(tmp_path,
                                                                              monkeypatch):
    import scripts.run_phase2_language_smoke as smoke
    from werewolf.phase2_language import LANGUAGE_VERSION
    from werewolf.phase2_actions import CONTRACT_VERSION
    mapper_digest = "a" * 64
    cases = [{"game_id": f"g{i}", "boundary_id": f"b{i}"} for i in range(58)]
    selection_digest = sha256_bytes(canonical_json_bytes(cases))
    monkeypatch.setattr(smoke, "EXPECTED_CASE_SELECTION_DIGEST", selection_digest)
    manifest = {
        "artifact_type": "phase2_language_execution_smoke",
        "schema_version": smoke.VERSION, "study_name": smoke.NAME,
        "language_semantic_version": LANGUAGE_VERSION,
        "action_contract_version": CONTRACT_VERSION,
        "final_mapper_manifest_digest": mapper_digest,
        "case_selection": {"digest": selection_digest,
                           "count": 58},
    }
    payloads = {"selected_cases.json": canonical_json_bytes(cases),
                "case_executions.jsonl": canonical_jsonl_bytes([{}] * 58)}
    failed = publish_artifact(tmp_path / "failed", manifest_fields=manifest,
        files={**payloads, "metrics.json": canonical_json_bytes({
            "overall": {"case_count": 58},
            "gate": {"passed": False, "checks": {"valid": False}}})})
    assert not _smoke_v3_gate(failed.path, failed.manifest_digest, mapper_digest)
    passed = publish_artifact(tmp_path / "passed", manifest_fields=manifest,
        files={**payloads, "metrics.json": canonical_json_bytes({
            "overall": {"case_count": 58},
            "gate": {"passed": True, "checks": {"valid": True}}})})
    assert _smoke_v3_gate(passed.path, passed.manifest_digest, mapper_digest)
    assert not _smoke_v3_gate(passed.path, "0" * 64, mapper_digest)
    assert not _smoke_v3_gate(passed.path, passed.manifest_digest, "b" * 64)
