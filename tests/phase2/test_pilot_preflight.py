"""Pilot cannot start on a clean tree until real capabilities are certified."""

from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from werewolf.phase2_online_preflight import (
    FrozenArtifactRequirement, _frozen_q_runtime_ready, _smoke_v3_gate,
    assess_pilot_preflight,
    freeze_online_source_provenance, tracked_source_clean,
)
from werewolf.artifact_io import publish_artifact
from werewolf.artifact_io import canonical_json_bytes, canonical_jsonl_bytes, sha256_bytes
from tests.phase2.test_outcome_and_execution import values
from werewolf.phase2_outcome import reference_tables_digest


def _git(repo: Path, *args):
    subprocess.run(("git", "-C", str(repo), *args), check=True,
                   capture_output=True, text=True)


def test_preflight_ignores_untracked_but_rejects_staged_and_tracked_dirty(tmp_path,
                                                                         monkeypatch):
    import werewolf.phase2_online_preflight as preflight
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
    import werewolf.phase2_online_preflight as preflight
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


@pytest.fixture
def preflight_artifacts(tmp_path, monkeypatch):
    from tests.phase2.test_mapper_final import (
        oof_provenance, source_record, synthetic_study,
    )
    from werewolf import phase2_mapper_final as final, phase2_mapper_runtime as runtime
    from werewolf import phase2_online_preflight as preflight
    from werewolf.phase2_actions import CONTRACT_VERSION
    from werewolf.phase2_language import LANGUAGE_VERSION
    from scripts import run_phase2_language_smoke as smoke

    study, examples = synthetic_study(tmp_path, monkeypatch)
    monkeypatch.setattr(final, 'STUDY_DIGEST', study.manifest_digest)
    monkeypatch.setattr(runtime, 'STUDY_DIGEST', study.manifest_digest)
    fitted = final.mapper.fit_full_development(examples, final.SELECTED_SPEC, oof_provenance())
    mapper = final.publish_full_mapper(tmp_path/'mapper', fitted=fitted, rows=examples,
        study=study, source=source_record(), candidate_digest=final.candidate_identity_digest(examples))
    cases = [{'game_id': f'g{i}', 'boundary_id': f'b{i}'} for i in range(58)]
    selection_digest = sha256_bytes(canonical_json_bytes(cases))
    monkeypatch.setattr(smoke, 'EXPECTED_CASE_SELECTION_DIGEST', selection_digest)
    smoke_artifact = publish_artifact(tmp_path/'smoke', manifest_fields={
        'artifact_type': 'phase2_language_execution_smoke', 'schema_version': smoke.VERSION,
        'study_name': smoke.NAME, 'language_semantic_version': LANGUAGE_VERSION,
        'action_contract_version': CONTRACT_VERSION,
        'final_mapper_manifest_digest': mapper.manifest_digest,
        'case_selection': {'digest': selection_digest, 'count': 58},
    }, files={'selected_cases.json': canonical_json_bytes(cases),
        'case_executions.jsonl': canonical_jsonl_bytes([{}]*58),
        'metrics.json': canonical_json_bytes({'overall': {'case_count': 58},
            'gate': {'passed': True, 'checks': {'valid': True}}})})
    loaded = runtime.load_runtime_mapper(mapper.path, expected_manifest_digest=mapper.manifest_digest)
    monkeypatch.setattr(preflight, '_git', lambda *args:
        subprocess.CompletedProcess(args, 0, 'a'*40+'\n', ''))
    monkeypatch.setattr(preflight, 'tracked_source_clean', lambda *args: True)
    monkeypatch.setattr(preflight, 'freeze_online_source_provenance',
        lambda *args: {'commit': 'a'*40})

    def assess(mapper_artifact=mapper, smoke=smoke_artifact, requirements=None, **overrides):
        if requirements is None:
            requirements = tuple(FrozenArtifactRequirement(a.path, a.manifest['artifact_type'],
                a.manifest['schema_version'], a.manifest_digest) for a in (mapper_artifact, smoke))
        arguments = dict(frozen_artifacts=requirements,
            mapper_artifact=mapper_artifact.path, mapper_manifest_digest=mapper_artifact.manifest_digest,
            mapper_runtime=loaded, smoke_v3_artifact=smoke.path,
            smoke_v3_manifest_digest=smoke.manifest_digest)
        arguments.update(overrides)
        return assess_pilot_preflight(tmp_path, **arguments)

    return SimpleNamespace(mapper=mapper, smoke=smoke_artifact, runtime=loaded, assess=assess)


def test_preflight_reads_each_artifact_and_payload_once(preflight_artifacts, monkeypatch):
    fixture = preflight_artifacts
    reads = {}
    original = Path.read_bytes

    def read(path):
        reads[path] = reads.get(path, 0)+1
        return original(path)

    monkeypatch.setattr(Path, 'read_bytes', read)
    result = fixture.assess()
    assert result.frozen_artifacts_verified
    assert result.final_mapper_lineage_verified and result.smoke_v3_gate_passed
    for artifact in (fixture.mapper, fixture.smoke):
        for name in ('manifest.json', *artifact.manifest['file_table']):
            assert reads[artifact.path/name] == 1


@pytest.mark.parametrize('which', ('mapper', 'smoke'))
def test_preflight_detects_payload_changes_between_calls(preflight_artifacts, which):
    fixture = preflight_artifacts
    first = fixture.assess()
    assert first.final_mapper_lineage_verified and first.smoke_v3_gate_passed
    artifact = getattr(fixture, which)
    payload = artifact.path/('model.json' if which == 'mapper' else 'metrics.json')
    payload.write_bytes(b'corrupted')
    second = fixture.assess()
    assert not second.frozen_artifacts_verified
    assert not (second.final_mapper_lineage_verified if which == 'mapper'
                else second.smoke_v3_gate_passed)


@pytest.mark.parametrize('identity', ('path', 'artifact_type', 'schema_version', 'manifest_digest'))
def test_preflight_never_reuses_a_mismatched_requirement(preflight_artifacts, identity):
    fixture = preflight_artifacts
    fields = {'path': fixture.mapper.path,
        'artifact_type': fixture.mapper.manifest['artifact_type'],
        'schema_version': fixture.mapper.manifest['schema_version'],
        'manifest_digest': fixture.mapper.manifest_digest}
    if identity == 'path':
        fields[identity] = fixture.smoke.path
    elif identity == 'manifest_digest':
        fields[identity] = '0'*64
    else:
        fields[identity] = 'wrong'
    result = fixture.assess(requirements=(FrozenArtifactRequirement(**fields),))
    assert not result.frozen_artifacts_verified


@pytest.mark.parametrize('which', ('mapper', 'smoke'))
def test_preflight_retains_mapper_lineage_and_smoke_gate(preflight_artifacts, tmp_path, which):
    fixture = preflight_artifacts
    original = getattr(fixture, which)
    manifest = {k: v for k, v in original.manifest.items() if k not in ('file_table', 'manifest_digest')}
    if which == 'mapper':
        manifest['selected_development_study'] = {'manifest_digest': '0'*64}
    else:
        manifest['case_selection'] = {**manifest['case_selection'], 'count': 57}
    changed = publish_artifact(tmp_path/'invalid-semantic', manifest_fields=manifest,
        files={name: (original.path/name).read_bytes() for name in original.manifest['file_table']})
    # 让运行对象的摘要也匹配错误 Artifact，拒绝必须来自专用 lineage/gate 检查。
    arguments = ({'mapper_artifact': changed, 'mapper_runtime': SimpleNamespace(
        artifact_digest=changed.manifest_digest, model_digest=fixture.runtime.model_digest)}
        if which == 'mapper' else {'smoke': changed})
    result = fixture.assess(**arguments)
    assert result.frozen_artifacts_verified
    assert not (result.final_mapper_lineage_verified if which == 'mapper' else result.smoke_v3_gate_passed)


@pytest.mark.parametrize('which', ('mapper', 'smoke'))
def test_preflight_rejects_changed_specialized_digest(preflight_artifacts, which):
    keyword = 'mapper_manifest_digest' if which == 'mapper' else 'smoke_v3_manifest_digest'
    result = preflight_artifacts.assess(**{keyword: '0'*64})
    assert result.frozen_artifacts_verified
    assert not (result.final_mapper_lineage_verified if which == 'mapper' else result.smoke_v3_gate_passed)
