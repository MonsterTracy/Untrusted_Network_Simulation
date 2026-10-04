"""Read-only pilot capability check; never starts gameplay or a model call."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import json
import os
import subprocess

from werewolf.artifact_io import (
    canonical_json_bytes, read_artifact_file, sha256_bytes, verify_artifact,
)
from werewolf.phase2_intervention import CHECKPOINT_CLOSURE_VERSION
from werewolf.phase2_language import LANGUAGE_VERSION
from werewolf.phase2_online_plan import (
    FROZEN, Phase2OnlineProbePilotPlanV1, Phase2OnlineTerminalPilotPlanV1,
)
from werewolf.phase2_treatment import VERSION as TREATMENT_VERSION


PREFLIGHT_VERSION = "phase2_intervention_pilot_preflight_v1"
SOURCE_FILES = (
    "run_random.py", "scripts/run_phase2_online_intervention_pilot.py",
    "scripts/phase2_intervention_preflight.py", "scripts/phase2_integration_seam.py",
    "scripts/phase2_language_realization.py", "werewolf/phase2_online_plan.py",
    "werewolf/phase2_online_records.py", "werewolf/phase2_backend_audit.py",
    "werewolf/phase2_online_ledger.py",
    "werewolf/phase2_online_runner.py", "werewolf/phase2_online_preflight.py",
    "werewolf/phase2_online_server.py", "werewolf/phase2_execution.py",
    "scripts/__init__.py",
    "werewolf/phase2_decision_opportunity.py", "werewolf/phase2_treatment.py",
    "werewolf/phase2_language.py", "werewolf/phase2_canonical_commit.py",
    "werewolf/phase2_outcome.py", "werewolf/phase2_pilot_dataset.py",
    "werewolf/canonical_collection/runtime.py", "werewolf/envs/werewolf_text_env_v0.py",
    "werewolf/speech/verified_commit.py",
)


@dataclass(frozen=True)
class FrozenArtifactRequirement:
    path: Path
    artifact_type: str
    schema_version: str
    manifest_digest: str

    def __post_init__(self):
        if (not isinstance(self.path, Path)
                or any(not isinstance(value, str) or not value for value in (
                    self.artifact_type, self.schema_version, self.manifest_digest))
                or len(self.manifest_digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in self.manifest_digest)):
            raise ValueError("frozen artifact requires type, schema, and manifest digest")


@dataclass(frozen=True)
class PilotPreflightV1:
    source_head: str | None
    source_provenance: dict | None
    smoke_v3_manifest_digest: str | None
    online_plan_digest: str | None
    destination: str | None
    runtime_token: tuple[int, ...] | None
    tracked_clean: bool
    frozen_artifacts_verified: bool
    final_mapper_lineage_verified: bool
    frozen_q_runtime_available: bool
    smoke_v3_gate_passed: bool
    online_plan_frozen: bool
    treatment_randomization_valid: bool
    outcome_reference_available: bool
    destination_absent: bool
    checkpoint_executable: bool
    canonical_commit_bound: bool
    phase2_backend_audit_bound: bool
    assignment_ledger_bound: bool
    branch_executor_executable: bool
    blockers: tuple[str, ...]
    source_commit_pin: str | None = None
    reference_tables_digest_pin: str | None = None
    publication_only: bool = False

    @property
    def ready(self) -> bool:
        return self.online_randomized_pilot_ready

    @property
    def online_randomized_pilot_ready(self) -> bool:
        return not self.blockers

    @property
    def paired_branch_replay_ready(self) -> bool:
        return False

    def to_record(self) -> dict:
        return {"schema_version": PREFLIGHT_VERSION,
                "source_head": self.source_head,
                "source_commit_pin": self.source_commit_pin,
                "reference_tables_digest_pin": self.reference_tables_digest_pin,
                "publication_only": self.publication_only,
                "source_provenance": self.source_provenance,
                "smoke_v3_manifest_digest": self.smoke_v3_manifest_digest,
                "online_plan_digest": self.online_plan_digest,
                "destination": self.destination,
                "tracked_clean": self.tracked_clean,
                "frozen_artifacts_verified": self.frozen_artifacts_verified,
                "final_mapper_lineage_verified": self.final_mapper_lineage_verified,
                "frozen_q_runtime_available": self.frozen_q_runtime_available,
                "smoke_v3_gate_passed": self.smoke_v3_gate_passed,
                "online_plan_frozen": self.online_plan_frozen,
                "treatment_randomization_valid": self.treatment_randomization_valid,
                "outcome_reference_available": self.outcome_reference_available,
                "destination_absent": self.destination_absent,
                "checkpoint_schema": CHECKPOINT_CLOSURE_VERSION,
                "checkpoint_executable": self.checkpoint_executable,
                "canonical_commit_bound": self.canonical_commit_bound,
                "phase2_backend_audit_bound": self.phase2_backend_audit_bound,
                "assignment_ledger_bound": self.assignment_ledger_bound,
                "branch_executor_executable": self.branch_executor_executable,
                "language_version": LANGUAGE_VERSION,
                "treatment_schema": TREATMENT_VERSION,
                "online_randomized_pilot_ready": self.online_randomized_pilot_ready,
                "paired_branch_replay_ready": self.paired_branch_replay_ready,
                "ready": self.ready, "blockers": list(self.blockers)}


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(("git", "-C", str(repo), *args),
                          capture_output=True, text=True, check=False)


def tracked_source_clean(repo: Path) -> bool:
    """Ignore untracked files, but reject worktree and index changes."""
    return (_git(repo, "diff", "--quiet", "--exit-code").returncode == 0
            and _git(repo, "diff", "--cached", "--quiet", "--exit-code").returncode == 0)


def freeze_online_source_provenance(repo: Path) -> dict | None:
    """Freeze audited source once before the first pilot language call."""
    if not tracked_source_clean(repo):
        return None
    head = _git(repo, "rev-parse", "HEAD")
    branch = _git(repo, "branch", "--show-current")
    if head.returncode or branch.returncode:
        return None
    digests = {}
    for name in SOURCE_FILES:
        if (_git(repo, "ls-files", "--error-unmatch", name).returncode != 0
                or not (repo / name).is_file()):
            return None
        digests[name] = sha256_bytes((repo / name).read_bytes())
    return {"commit": head.stdout.strip(), "branch": branch.stdout.strip(),
            "tracked_worktree_clean": True, "staged_tracked_changes": False,
            "source_sha256": digests}


def _verify_frozen_artifacts(requirements: tuple[FrozenArtifactRequirement, ...]) -> bool:
    if not requirements or any(not isinstance(item, FrozenArtifactRequirement)
                               for item in requirements):
        return False
    try:
        for item in requirements:
            artifact = verify_artifact(
                item.path, expected_artifact_type=item.artifact_type,
                expected_schema_version=item.schema_version)
            if artifact.manifest_digest != item.manifest_digest:
                return False
    except (OSError, TypeError, ValueError, KeyError, RuntimeError):
        return False
    return True


def _frozen_q_runtime_ready(predictor) -> bool:
    """Only the client that completed the pinned worker handshake is admissible."""
    try:
        from scripts.qwen3_gameplay_predictor import (
            FIT_DIGEST, SEAL_DIGEST, Qwen3GameplayPredictorClient,
        )
        return (type(predictor) is Qwen3GameplayPredictorClient
                and predictor.fit_digest == FIT_DIGEST
                and predictor.seal_digest == SEAL_DIGEST
                and isinstance(predictor._process, subprocess.Popen)
                and predictor._process.poll() is None)
    except (ImportError, AttributeError, OSError):
        return False


def _smoke_v3_gate(path: Path | None, digest: str | None,
                   mapper_digest: str | None) -> bool:
    if path is None or digest is None:
        return False
    try:
        from scripts.run_phase2_language_smoke import (
            EXPECTED_CASE_SELECTION_DIGEST, NAME, VERSION,
        )
        from werewolf.phase2_actions import CONTRACT_VERSION
        artifact = verify_artifact(path,
            expected_artifact_type="phase2_language_execution_smoke",
            expected_schema_version=VERSION)
        selected = artifact.manifest["case_selection"]
        selected_cases = json.loads(read_artifact_file(artifact, "selected_cases.json"))
        executions = read_artifact_file(artifact, "case_executions.jsonl").splitlines()
        metrics = json.loads(read_artifact_file(artifact, "metrics.json"))
        gate = metrics["gate"]
        return (artifact.manifest_digest == digest
                and artifact.manifest["study_name"] == NAME
                and artifact.manifest["language_semantic_version"] == LANGUAGE_VERSION
                and artifact.manifest["action_contract_version"] == CONTRACT_VERSION
                and artifact.manifest["final_mapper_manifest_digest"] == mapper_digest
                and selected["digest"] == EXPECTED_CASE_SELECTION_DIGEST
                and selected["count"] == 58
                and len(selected_cases) == len(executions) == 58
                and len({(row["game_id"], row["boundary_id"])
                         for row in selected_cases}) == 58
                and sha256_bytes(canonical_json_bytes(selected_cases)) == selected["digest"]
                and metrics["overall"]["case_count"] == 58
                and gate["passed"] is True
                and bool(gate["checks"])
                and all(value is True for value in gate["checks"].values()))
    except (OSError, TypeError, ValueError, KeyError, RuntimeError, ImportError):
        return False


def assess_pilot_preflight(repo: Path, *, frozen_artifacts: tuple[FrozenArtifactRequirement, ...],
                           expected_source_commit: str | None = None,
                           expected_reference_tables_digest: str | None = None,
                           mapper_artifact: Path | None = None,
                           mapper_manifest_digest: str | None = None,
                           mapper_runtime=None,
                           smoke_v3_artifact: Path | None = None,
                           smoke_v3_manifest_digest: str | None = None,
                           online_plan: Phase2OnlineTerminalPilotPlanV1 | Phase2OnlineProbePilotPlanV1 | None = None,
                           predictor=None, backend=None, call_audit=None,
                           ledger=None,
                           env=None, recorder=None, reference_tables=None,
                           reference_artifact_digest: str | None = None,
                           destination: Path | None = None,
                           publication_only: bool = False,
                           ) -> PilotPreflightV1:
    head_result = _git(repo, "rev-parse", "HEAD")
    head = head_result.stdout.strip() if head_result.returncode == 0 else None
    clean = head is not None and tracked_source_clean(repo)
    frozen_source = freeze_online_source_provenance(repo) if clean else None
    if frozen_source is not None and frozen_source["commit"] != head:
        frozen_source = None
        clean = False
    artifacts = _verify_frozen_artifacts(frozen_artifacts)
    mapper_verified = False
    if mapper_artifact is not None and mapper_manifest_digest is not None:
        try:
            from werewolf.phase2_mapper_runtime import load_runtime_mapper
            loaded_mapper = load_runtime_mapper(mapper_artifact,
                expected_manifest_digest=mapper_manifest_digest)
            mapper_verified = (mapper_runtime is not None
                               and mapper_runtime.artifact_digest == loaded_mapper.artifact_digest
                               and mapper_runtime.model_digest == loaded_mapper.model_digest)
        except (OSError, TypeError, ValueError, KeyError, RuntimeError):
            mapper_verified = False
    smoke_passed = _smoke_v3_gate(smoke_v3_artifact, smoke_v3_manifest_digest,
                                 mapper_manifest_digest)
    plan_types = (Phase2OnlineTerminalPilotPlanV1, Phase2OnlineProbePilotPlanV1)
    plan_frozen = (isinstance(online_plan, plan_types)
                   and online_plan.selection_status == FROZEN)
    randomization_valid = isinstance(online_plan, plan_types)
    q_ready = _frozen_q_runtime_ready(predictor)
    try:
        from werewolf.canonical_collection.call_audit import AuditedBackend
        from werewolf.canonical_collection.runtime import CanonicalGameRecorder
        from werewolf.canonical_collection.call_audit import CanonicalCallAudit
        backend_bound = (isinstance(backend, AuditedBackend)
                         and isinstance(call_audit, CanonicalCallAudit)
                         and backend._audit is call_audit)
        commit_bound = (isinstance(recorder, CanonicalGameRecorder)
                        and callable(getattr(recorder, "commit_verified_speech", None))
                        and callable(getattr(env, "_stage_verified_speech", None))
                        and getattr(getattr(getattr(env, "speech_perceiver", None),
                                            "backend", None), "_audit", None)
                        is call_audit)
    except ImportError:
        backend_bound = commit_bound = False
    from werewolf.phase2_offline import ReferenceTables
    from werewolf.phase2_outcome import reference_tables_digest
    try:
        reference_ready = (isinstance(reference_tables, ReferenceTables)
                           and isinstance(reference_artifact_digest, str)
                           and reference_tables_digest(reference_tables) ==
                               reference_artifact_digest
                           and reference_artifact_digest ==
                               expected_reference_tables_digest)
    except (OSError, TypeError, ValueError, KeyError):
        reference_ready = False
    destination_absent = destination is not None and not os.path.lexists(destination)
    from werewolf.phase2_online_ledger import OnlinePilotAssignmentLedgerV1
    try:
        ledger_bound = (isinstance(ledger, OnlinePilotAssignmentLedgerV1)
                        and isinstance(online_plan, plan_types)
                        and ledger.plan.digest() == online_plan.digest()
                        and ledger.source_commit == head
                        and ledger.path != destination
                        and (ledger.status() == "RUNNING" if not publication_only else
                             ledger.status() == "READY_TO_SEAL" and ledger.sealable()))
    except (OSError, TypeError, ValueError, KeyError):
        ledger_bound = False
    blockers = []
    try:
        from werewolf.phase2_online_runner import require_online_gameplay_hook
        require_online_gameplay_hook()
    except (ImportError, TypeError, ValueError):
        blockers.append("PRODUCTION_GAMEPLAY_HOOK_UNAVAILABLE")
    if not clean:
        blockers.append("TRACKED_SOURCE_OR_INDEX_DIRTY")
    if frozen_source is None:
        blockers.append("PILOT_SOURCE_UNTRACKED_OR_NOT_FROZEN")
    if (not isinstance(expected_source_commit, str)
            or len(expected_source_commit) != 40
            or head != expected_source_commit):
        blockers.append("SOURCE_COMMIT_PIN_MISMATCH")
    if not artifacts:
        blockers.append("FROZEN_ARTIFACTS_NOT_VERIFIED")
    if not mapper_verified:
        blockers.append("FINAL_MAPPER_LINEAGE_NOT_VERIFIED")
    if not q_ready:
        blockers.append("FROZEN_Q_RUNTIME_UNAVAILABLE")
    if not smoke_passed:
        blockers.append("SMOKE_V3_GATE_NOT_VERIFIED")
    if not plan_frozen:
        blockers.append("FROZEN_ONLINE_PILOT_PLAN_MISSING_OR_INVALID")
    if not randomization_valid:
        blockers.append("TREATMENT_RANDOMIZATION_INVALID")
    if not reference_ready:
        blockers.append("REFERENCE_TABLE_PIN_MISMATCH_OR_UNAVAILABLE")
    if not destination_absent:
        blockers.append("PILOT_ARTIFACT_DESTINATION_NOT_ABSENT")
    if not commit_bound:
        blockers.append("PHASE2_CANONICAL_COMMIT_BINDING_UNAVAILABLE")
    if not backend_bound:
        blockers.append("PHASE2_BACKEND_AUDIT_UNBOUND")
    if not ledger_bound:
        blockers.append("DURABLE_ASSIGNMENT_LEDGER_UNBOUND")
    # Replay is an optional paired-counterfactual capability, outside these blockers.
    runtime_token = ((id(env), id(recorder), id(call_audit), id(backend),
                      id(predictor), id(mapper_runtime), id(reference_tables), id(ledger))
                     if backend_bound and commit_bound else None)
    return PilotPreflightV1(head, frozen_source,
                            smoke_v3_manifest_digest if smoke_passed else None,
                            online_plan.digest() if randomization_valid else None,
                            str(destination) if destination is not None else None,
                            runtime_token, clean, artifacts, mapper_verified,
                            q_ready, smoke_passed, plan_frozen,
                            randomization_valid, reference_ready,
                            destination_absent, False, commit_bound,
                            backend_bound, ledger_bound, False, tuple(blockers),
                            expected_source_commit,
                            expected_reference_tables_digest, publication_only)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--frozen-artifact", action="append", nargs=4,
                        metavar=("PATH", "TYPE", "SCHEMA", "MANIFEST_DIGEST"),
                        default=[])
    parser.add_argument("--mapper-artifact", type=Path)
    parser.add_argument("--mapper-manifest-digest")
    args = parser.parse_args()
    import json
    result = assess_pilot_preflight(args.repo,
        frozen_artifacts=tuple(FrozenArtifactRequirement(Path(path), kind, schema, digest)
                               for path, kind, schema, digest in args.frozen_artifact),
        mapper_artifact=args.mapper_artifact,
        mapper_manifest_digest=args.mapper_manifest_digest)
    print(json.dumps(result.to_record(), ensure_ascii=False, sort_keys=True))
    return 0 if result.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
