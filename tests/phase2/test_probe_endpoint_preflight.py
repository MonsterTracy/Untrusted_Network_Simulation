"""Preflight binds validated durable evidence separately from campaign status."""

from dataclasses import replace
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests.phase2.test_probe_policy_resume import (
    SOURCE_COMMIT, completed_probe_trace, crash_prefix,
)
from tests.phase2.test_probe_policy_runner import runtime
from werewolf.canonical_collection.call_audit import AuditedBackend, CanonicalCallAudit
from werewolf.phase2_online_preflight import assess_pilot_preflight


class PreflightRecorder:
    """Canonical commit capability only; optional simulator modules are absent."""

    def commit_verified_speech(self, *args, **kwargs):
        raise AssertionError("read-only preflight must not commit gameplay")


def assess_bound_ledger(runtime, monkeypatch, tmp_path, f, ledger, *,
                        publication_only=False, online_plan=None, head=SOURCE_COMMIT,
                        expected_ledger_path=None):
    """Stub unrelated capability gates, retaining real preflight and ledger checks."""
    import werewolf.phase2_mapper_runtime as mapper_runtime
    import werewolf.phase2_online_preflight as preflight

    monkeypatch.setitem(sys.modules, "werewolf.phase2_online_runner", runtime)
    monkeypatch.setitem(sys.modules, "werewolf.canonical_collection.runtime",
                        SimpleNamespace(CanonicalGameRecorder=PreflightRecorder))
    monkeypatch.setattr(preflight, "_git", lambda *args:
                        subprocess.CompletedProcess(args, 0, head + "\n", ""))
    monkeypatch.setattr(preflight, "tracked_source_clean", lambda *args: True)
    monkeypatch.setattr(preflight, "freeze_online_source_provenance",
                        lambda *args: {"commit": head})
    monkeypatch.setattr(preflight, "_verify_frozen_artifacts", lambda *args: True)
    monkeypatch.setattr(preflight, "_smoke_v3_gate", lambda *args: True)
    monkeypatch.setattr(preflight, "_frozen_q_runtime_ready", lambda *args: True)
    mapper = f.pilot.mapper
    mapper.model_digest = "f" * 64
    monkeypatch.setattr(mapper_runtime, "load_runtime_mapper", lambda *args, **kwargs: mapper)

    audit = CanonicalCallAudit(plan=SimpleNamespace(backend_identity="fixture"),
                               configured_call_limit=10)
    backend = AuditedBackend(f.pilot.backend, audit)
    recorder = PreflightRecorder()
    env = SimpleNamespace(_stage_verified_speech=lambda *args, **kwargs: None,
                          speech_perceiver=SimpleNamespace(backend=backend))
    return assess_pilot_preflight(tmp_path, frozen_artifacts=(),
        expected_source_commit=head,
        expected_reference_tables_digest=f.pilot.reference_artifact_digest,
        mapper_artifact=tmp_path / "fixture-mapper",
        mapper_manifest_digest=mapper.artifact_digest, mapper_runtime=mapper,
        smoke_v3_artifact=tmp_path / "fixture-smoke",
        smoke_v3_manifest_digest="d" * 64,
        online_plan=online_plan or ledger.plan,
        predictor=f.pilot.predictor, backend=backend, call_audit=audit,
        ledger=ledger, expected_ledger_path=expected_ledger_path or ledger.path,
        env=env, recorder=recorder,
        reference_tables=f.pilot.reference_tables,
        reference_artifact_digest=f.pilot.reference_artifact_digest,
        destination=tmp_path / "unpublished-destination",
        publication_only=publication_only)


@pytest.mark.parametrize("target,publication_only", [(2, False), (1, True)])
def test_real_preflight_accepts_bound_endpoint_tail_for_resume_or_publication(
        runtime, monkeypatch, tmp_path, target, publication_only):
    f = completed_probe_trace(runtime, monkeypatch, tmp_path / "complete",
                              target_assignment_count=target)
    ledger, original = crash_prefix(f, tmp_path, "CONSEQUENCE")
    assert ledger.mark_interrupted_on_resume() == ("game-1",)
    assert ledger.path.read_bytes().startswith(original)
    frozen = ledger.path.read_bytes()
    report = assess_bound_ledger(runtime, monkeypatch, tmp_path, f, ledger,
                                 publication_only=publication_only)
    assert report.assignment_ledger_bound
    assert report.ready and report.blockers == ()
    assert ledger.path.read_bytes() == frozen


@pytest.mark.parametrize("kind,publication_only", [
    ("ASSIGNMENT", False), ("EXECUTION", False), ("GAME_RESULT", False),
    ("EXECUTION", True), ("CONSEQUENCE", True),
])
def test_campaign_status_blocks_gameplay_independently_of_valid_durable_binding(
        runtime, monkeypatch, tmp_path, kind, publication_only):
    f = completed_probe_trace(runtime, monkeypatch, tmp_path / "complete",
                              target_assignment_count=1 if kind == "GAME_RESULT" else 2)
    ledger, _ = crash_prefix(f, tmp_path, kind)
    ledger.mark_interrupted_on_resume()
    expected_status = ("READY_TO_SEAL" if kind == "GAME_RESULT" else
                       "RUNNING" if kind == "CONSEQUENCE" else "INCOMPLETE")
    assert ledger.status() == expected_status
    frozen = ledger.path.read_bytes()
    report = assess_bound_ledger(runtime, monkeypatch, tmp_path, f, ledger,
                                 publication_only=publication_only)
    assert report.assignment_ledger_bound
    assert report.blockers == ("CAMPAIGN_NOT_RESUMABLE",)
    assert not report.ready
    assert ledger.path.read_bytes() == frozen


@pytest.mark.parametrize("mismatch", ["plan", "source", "path", "hash_chain"])
def test_endpoint_tail_does_not_relax_plan_source_or_journal_chain_binding(
        runtime, monkeypatch, tmp_path, mismatch):
    f = completed_probe_trace(runtime, monkeypatch, tmp_path / "complete",
                              target_assignment_count=2)
    ledger, _ = crash_prefix(f, tmp_path, "CONSEQUENCE")
    ledger.mark_interrupted_on_resume()
    online_plan, head, expected_path = ledger.plan, SOURCE_COMMIT, ledger.path
    if mismatch == "plan":
        online_plan = replace(ledger.plan, assignment_seed=ledger.plan.assignment_seed + 1)
    elif mismatch == "source":
        head = "b" * 40
    elif mismatch == "path":
        expected_path = tmp_path / "wrong-ledger.jsonl"
    else:
        raw = ledger.path.read_bytes()
        ledger.path.write_bytes(raw.replace(b'"reason":"CONSEQUENCE_WITHOUT_GAME_RESULT"',
                                           b'"reason":"CORRUPTED_ENDPOINT_TAIL"'))
        assert ledger.path.read_bytes() != raw
    frozen = ledger.path.read_bytes()
    report = assess_bound_ledger(runtime, monkeypatch, tmp_path, f, ledger,
                                 online_plan=online_plan, head=head,
                                 expected_ledger_path=expected_path)
    assert not report.assignment_ledger_bound
    assert "DURABLE_ASSIGNMENT_LEDGER_UNBOUND" in report.blockers
    assert not report.ready
    assert ledger.path.read_bytes() == frozen
