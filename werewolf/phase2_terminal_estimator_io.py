"""Sealed inputs and exclusive publication for Terminal Estimator Protocol V1.

This independent analysis adapter never follows historical work/evidence paths,
reconstructs outcomes, or invokes gameplay. Input digest pins have no CLI override.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

import numpy as np

from scripts import phase2_terminal_itt_analysis as itt_source
from werewolf.artifact_io import canonical_json_bytes, canonical_jsonl_bytes
from werewolf.artifact_io import read_artifact_file, sha256_bytes, verify_artifact
from werewolf.artifact_io import canonical as artifact_io
from werewolf import phase2_terminal_estimator as stats


NAME = "paper-phase2-terminal-estimator-v1"
VERSION = "phase2_terminal_estimator_v1"
GATE_VERSION = "phase2_terminal_estimator_support_gate_v1"
ITT_DIGEST = "58a8611a0c9f5605f6cbdf5a7a9ea9f0c74fc69e637760e507f073e298cfaf8c"
ITT_ROWS_SHA256 = "7ebb3b4848712ea40142ef6a1bebe1ef9ebbf1eb6e1ff1c0513d673cdfb86779"
PROTOCOL = "docs/research/phase2-terminal-estimator-protocol-v1.md"
SUPPORT_REVIEW = "docs/research/phase2-terminal-estimator-support-review-v1.md"
PROTOCOL_DIGEST = "d6dd3dc34b49f5ea6479a1e13e024008b5c0ea6539e6be3e540b9a6bfb3cfde0"
SUPPORT_REVIEW_DIGEST = "e82d789352cb14b0f368597bf9a844da8d8b3d373325ff494baa817d3088ea8d"
ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "werewolf/phase2_terminal_estimator.py",
    "werewolf/phase2_terminal_estimator_io.py",
    "scripts/phase2_terminal_estimator.py",
    "scripts/phase2_terminal_itt_analysis.py",
    "werewolf/artifact_io/canonical.py",
    PROTOCOL, SUPPORT_REVIEW,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(artifact, name):
    data = read_artifact_file(artifact, name)
    value = itt_source.strict_json(data)
    require(canonical_json_bytes(value) == data, f"noncanonical {name}")
    return value


@dataclass(frozen=True)
class VerifiedInputs:
    formal: artifact_io.VerifiedArtifact
    itt: artifact_io.VerifiedArtifact
    rows: tuple


def load_inputs(formal, itt):
    """Verify the two explicitly supplied sealed artifacts, including exact joins."""
    formal_artifact, tables = itt_source.load_formal(formal)
    indexed, executions, consequences = itt_source.exact_joins(*tables)
    itt_artifact = verify_artifact(itt, expected_artifact_type="phase2_terminal_itt_analysis",
                                  expected_schema_version=itt_source.VERSION)
    require(itt_artifact.manifest_digest == ITT_DIGEST, "ITT manifest digest mismatch")
    m, fm = itt_artifact.manifest, formal_artifact.manifest
    require(m.get("study_name") == itt_source.NAME and m.get("analysis_only") is True
            and m.get("primary_population") == "all_120_randomized_assignments"
            and m.get("source_formal_manifest_digest") == formal_artifact.manifest_digest
            and m.get("source_formal_commit") == itt_source.FORMAL_SOURCE
            and m.get("reference_table_digest") == itt_source.REFERENCE_DIGEST
            and m.get("row_count") == 120 and m.get("recovery_count") == 1
            and m.get("recovery_rule_version") == itt_source.RECOVERY_RULE
            and m.get("fitted_lambda") is None, "ITT lineage/population mismatch")
    provenance = fm["server_run_provenance"]
    require(m.get("source_run_inputs_digest") == provenance["inputs_digest"]
            and m.get("canonical_game_plan_digest") == provenance["game_plan_digest"],
            "ITT formal run/plan binding mismatch")
    data = read_artifact_file(itt_artifact, "itt_rows.jsonl")
    require(sha256_bytes(data) == ITT_ROWS_SHA256, "ITT row payload digest mismatch")
    rows = tuple(itt_source.strict_json(line) for line in data.splitlines())
    require(canonical_jsonl_bytes(rows) == data, "ITT rows must be canonical JSONL")
    validated = stats.validate_rows(rows)
    require(all(a["opportunity"]["identity"]["phase"] == "speech" for a in indexed.values()),
            "formal estimator only covers speech")
    require(all(e.get("theta_audit_label") is None for e in executions.values())
            and all(c.get("theta_audit_label") is None for c in consequences.values()),
            "frozen theta audit labels must remain null")
    audit = _json(itt_artifact, "recovery_audit.json")
    missing_id = next(k for k in indexed if k not in consequences)
    execution = executions[missing_id]
    require(audit.get("recovery_rule_version") == itt_source.RECOVERY_RULE
            and audit.get("game_id") == itt_source.FAILED_GAME
            and audit.get("assignment_id") == missing_id
            and audit.get("source_formal_manifest_digest") == formal_artifact.manifest_digest
            and audit.get("reference_table_digest") == itt_source.REFERENCE_DIGEST
            and execution["success"] is False
            and execution["failure_reason"] == "PRIVATE_FACT_CLAIM"
            and execution["canonical_commit_success"] is False,
            "sealed recovery/execution binding mismatch")
    # Verify only sealed audit values. These historical paths are provenance text,
    # never filesystem input to this estimator and never a new recovery operation.
    proofs = audit["canonical_partial_evidence"]
    require(proofs["execution"]["file_sha256"] == itt_source.EXECUTION_SHA
            and proofs["game-result"]["file_sha256"] == itt_source.GAME_RESULT_SHA,
            "sealed recovery evidence pins mismatch")
    expected = itt_source.itt_rows(indexed, executions, consequences, audit["recovered_outcome"])
    expected_by_id = {r["assignment_id"]: r for r in expected}
    require({r["assignment_id"] for r in validated} == set(indexed)
            and all(canonical_json_bytes(r) == canonical_json_bytes(expected_by_id[r["assignment_id"]])
                    for r in validated),
            "ITT rows differ from frozen assignment/execution/consequence joins")
    return VerifiedInputs(formal_artifact, itt_artifact, validated)


def _git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args]).decode("utf-8").strip()


def _source_files():
    return {name: sha256_bytes((ROOT / name).read_bytes()) for name in SOURCE_FILES}


def source_provenance():
    """Capture once before computation; no clock or output path enters the digest."""
    files = _source_files()
    require(files[PROTOCOL] == PROTOCOL_DIGEST, "frozen estimator protocol digest mismatch")
    require(files[SUPPORT_REVIEW] == SUPPORT_REVIEW_DIGEST, "frozen support review digest mismatch")
    return {"commit": _git("rev-parse", "HEAD"), "branch": _git("branch", "--show-current"),
            "files": files,
            "tracked_diff_digest": sha256_bytes(_git("diff", "HEAD", "--binary").encode()),
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "system": platform.system(), "machine": platform.machine()}}


def support_contract():
    return {"schema_version": GATE_VERSION, "score_semantics": "mapper score",
            "states": [{"S_pre": list(s), "inclusive_score_interval": list(interval),
                        "status": "SUPPORTED"} for s, interval in stats.SUPPORTED_INTERVALS.items()],
            "audit_only_states": [[2, 3]], "unsupported_states": [[1, 4]],
            "unknown_state_status": "UNSUPPORTED", "audit_only_status": "AUDIT_ONLY",
            "invalid_input_status": "INVALID_INPUT", "lineage_failure_status": "LINEAGE_MISMATCH",
            "model_failure_status": "MODEL_SANITY_FAILED", "inclusive_endpoints": True,
            "requires_verified_lineage": True, "requires_M2_sanity_pass": True,
            "risk_mean_tolerance": 1e-10, "clip_predictions": False,
            "conditional_support_is_not_precision_certification": True}


def _exclusive_publish(destination, manifest_fields, files):
    """Use canonical envelope/durable primitives without its idempotent reuse branch."""
    destination = Path(destination).absolute()
    require(not os.path.lexists(destination), "estimator destination already exists")
    artifact_io._atomic_noreplace_rename()  # Fail before writing on unsupported hosts.
    manifest, payloads = artifact_io._build_manifest(manifest_fields, files, "manifest.json")
    artifact_io.ensure_durable_directory(destination.parent)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    published = False
    try:
        for name, data in sorted(payloads.items()):
            artifact_io._write_fsynced(staging / name, data)
        artifact_io._write_fsynced(staging / "manifest.json", canonical_json_bytes(manifest))
        artifact_io._fsync_tree(staging)
        # Validate bytes before the destination becomes visible, not only after
        # promotion: a corrupted staging tree must never resemble a completed run.
        verify_artifact(staging, expected_artifact_type="phase2_terminal_estimator",
                        expected_schema_version=VERSION)
        # This is atomic and raises if a competitor creates even identical output.
        artifact_io._publish_directory_noreplace(staging, destination)
        published = True
        artifact_io._fsync_directory(destination.parent)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return verify_artifact(destination, expected_artifact_type="phase2_terminal_estimator",
                           expected_schema_version=VERSION)


def _report(primary, models, fisher, status):
    arms = primary["arm_means"]
    return ("# Terminal Estimator V1\n\n"
            "Primary population: all 120 randomized assignments, including language failure.\n\n"
            f"PUSH mean={arms['PUSH']!r}; REDIRECT mean={arms['REDIRECT']!r}.\n\n"
            f"Raw difference={primary['difference_in_means']!r}; "
            f"Neyman SE={primary['standard_error']!r}; normal 95% CI={primary['interval']!r}.\n\n"
            "The Neyman interval uses the conditional 58-of-120 randomization reference, "
            "independent Bernoulli-style assignment interpretation and no carryover. "
            "It is a normal approximation, not a finite-sample exact interval.\n\n"
            f"Fisher supplement: two-sided sharp-null Monte Carlo, B={fisher['repetitions']}, "
            f"seed={fisher['seed']}; result recorded separately in fisher_supplement.json.\n\n"
            + "\n".join(f"{name}: k={len(record['columns'])}, "
                         f"sanity={record['diagnostics']['passed']}, "
                         f"reasons={record['diagnostics']['reasons']!r}.\n"
                         for name, record in models.items())
            + f"\nOverall status: {status}.\n\n"
            "All three unweighted OLS models use all rows, fixed categorical coding and HC1; "
            "there is no model selection or fallback. Conditional estimates are reduced-form "
            "working means; p_tilde is a score. HC1 pointwise mean intervals are diagnostic, "
            "unclipped and not individual outcome intervals. Singleton-state uncertainty is "
            "not recovered by finite HC1. The frozen gate checks state and score hull, not "
            "local density or precision. No classic lambda is identified, no action costs "
            "are added, and this artifact does not authorize deployment or action selection.\n")


def analyze(formal, itt, destination):
    """Future one-shot analysis entrypoint; callers must explicitly supply both inputs."""
    destination = Path(destination).absolute()
    for protected in (Path(formal).resolve(), Path(itt).resolve()):
        target = destination.resolve()
        require(target != protected and not target.is_relative_to(protected)
                and not protected.is_relative_to(target), "analysis output must be disjoint from frozen inputs")
    require(not os.path.lexists(destination), "estimator destination already exists")
    provenance = source_provenance()
    inputs = load_inputs(formal, itt)
    primary = stats.primary_itt(inputs.rows)
    fisher = stats.fisher_sharp_null(inputs.rows)
    require(fisher["seed"] == stats.FISHER_SEED and fisher["repetitions"] == stats.FISHER_REPETITIONS,
            "Fisher supplement configuration differs from frozen protocol")
    models = stats.fit_models(inputs.rows)
    status = "SANITY_PASSED_FOR_REVIEW" if models["M2"]["diagnostics"]["passed"] else "FAIL_CLOSED"
    diagnostics = {"status": status, "operational_deployment_authorized": False,
                   "models": {name: model["diagnostics"] for name, model in models.items()}}
    files = {"primary_itt.json": canonical_json_bytes(primary),
             "fisher_supplement.json": canonical_json_bytes(fisher),
             "support_gate.json": canonical_json_bytes(support_contract()),
             "diagnostics.json": canonical_json_bytes(diagnostics),
             "report.md": _report(primary, models, fisher, status).encode()}
    files.update({f"models/{name.lower()}.json": canonical_json_bytes(record)
                  for name, record in models.items()})
    # Source is captured once; checks detect drift rather than regenerating provenance.
    require(_source_files() == provenance["files"] and _git("rev-parse", "HEAD") == provenance["commit"]
            and _git("branch", "--show-current") == provenance["branch"]
            and sha256_bytes(_git("diff", "HEAD", "--binary").encode()) == provenance["tracked_diff_digest"],
            "analysis source changed during computation")
    for artifact, artifact_type, version in (
            (inputs.formal, "phase2_online_terminal_pilot", "phase2_online_terminal_pilot_v1"),
            (inputs.itt, "phase2_terminal_itt_analysis", itt_source.VERSION)):
        again = verify_artifact(artifact.path, expected_artifact_type=artifact_type, expected_schema_version=version)
        require(again.manifest_digest == artifact.manifest_digest, "input artifact changed during analysis")
    manifest = {"artifact_type": "phase2_terminal_estimator", "schema_version": VERSION, "study_name": NAME,
                "analysis_only": True, "source_provenance": provenance,
                "formal_manifest_digest": inputs.formal.manifest_digest,
                "itt_manifest_digest": inputs.itt.manifest_digest,
                "itt_rows_sha256": ITT_ROWS_SHA256, "reference_table_digest": itt_source.REFERENCE_DIGEST,
                "input_row_count": 120, "treatment_counts": {"PUSH": 58, "REDIRECT": 62},
                "primary_population": "all_120_randomized_assignments",
                "estimator_protocol_source_digest": PROTOCOL_DIGEST,
                "support_review_source_digest": SUPPORT_REVIEW_DIGEST,
                "model_specs": {"M0": "L ~ A", "M1": "L ~ A + S_pre",
                                "M2": "L ~ A + S_pre + p_tilde + A:p_tilde"},
                "coding_convention": {"PUSH": 1, "REDIRECT": 0, "state_reference": [2, 5],
                                      "columns": {n: r["columns"] for n, r in models.items()}},
                "covariance_convention": "HC1", "support_gate_version": GATE_VERSION,
                "fisher": {"method": "sharp_null_two_sided_plus_one_Monte_Carlo", "PRNG": "PCG64",
                           "seed": stats.FISHER_SEED, "repetitions": stats.FISHER_REPETITIONS},
                "status": status, "classic_lambda_identified": False}
    published = _exclusive_publish(destination, manifest, files)
    return {"destination": str(published.path), "manifest_digest": published.manifest_digest,
            "status": status, "N": 120, "treatment_counts": manifest["treatment_counts"]}


@dataclass(frozen=True)
class VerifiedEstimator:
    artifact: artifact_io.VerifiedArtifact
    models: dict
    model_digests: tuple

    def _model_lineage_valid(self):
        try:
            return (set(self.models) == {name for name, _ in self.model_digests}
                    and all(sha256_bytes(canonical_json_bytes(self.models[name])) == digest
                            for name, digest in self.model_digests))
        except (TypeError, ValueError, KeyError):
            return False

    @staticmethod
    def _lineage_failure():
        return {"status": "LINEAGE_MISMATCH", "supported": False, "risk": None,
                "variance": None, "standard_error": None, "interval": None}

    def predict(self, action, S_pre, p_tilde, *, model_name="M2"):
        if not self._model_lineage_valid():
            return self._lineage_failure()
        require(model_name in self.models, "unknown estimator model")
        return stats.predict_risk(self.models[model_name], action, S_pre, p_tilde,
                                  lineage_verified=True,
                                  m2_diagnostics_passed=(self.models["M2"]["diagnostics"]["passed"] is True
                                                        and stats.model_diagnostics(self.models["M2"])["passed"]))

    def audit(self, action, S_pre, p_tilde, *, model_name="M2"):
        if not self._model_lineage_valid():
            return {**self._lineage_failure(), "audit_only": True}
        require(model_name in self.models, "unknown estimator model")
        return stats.audit_prediction(self.models[model_name], action, S_pre, p_tilde)


def load_estimator(path, *, expected_digest):
    """Require an independently supplied estimator digest before enabling predictions."""
    artifact = verify_artifact(path, expected_artifact_type="phase2_terminal_estimator", expected_schema_version=VERSION)
    m = artifact.manifest
    require(artifact.manifest_digest == expected_digest, "estimator manifest digest mismatch")
    require(m.get("formal_manifest_digest") == itt_source.FORMAL_DIGEST and m.get("itt_manifest_digest") == ITT_DIGEST
            and m.get("itt_rows_sha256") == ITT_ROWS_SHA256
            and m.get("reference_table_digest") == itt_source.REFERENCE_DIGEST
            and m.get("estimator_protocol_source_digest") == PROTOCOL_DIGEST
            and m.get("support_review_source_digest") == SUPPORT_REVIEW_DIGEST
            and m.get("input_row_count") == 120 and m.get("treatment_counts") == {"PUSH": 58, "REDIRECT": 62}
            and m.get("covariance_convention") == "HC1" and m.get("support_gate_version") == GATE_VERSION
            and m.get("classic_lambda_identified") is False
            and m.get("source_provenance", {}).get("files") == _source_files(), "estimator lineage mismatch")
    require(_json(artifact, "support_gate.json") == support_contract(), "estimator support contract mismatch")
    models = {name: _json(artifact, f"models/{name.lower()}.json") for name in stats.MODEL_COLUMNS}
    for name, model in models.items():
        require(model["model_name"] == name and model["columns"] == list(stats.MODEL_COLUMNS[name]),
                "estimator model coding mismatch")
    diagnostics = _json(artifact, "diagnostics.json")
    require(diagnostics["models"] == {n: r["diagnostics"] for n, r in models.items()}
            and diagnostics["status"] == m["status"]
            and m["status"] == ("SANITY_PASSED_FOR_REVIEW" if models["M2"]["diagnostics"]["passed"]
                                else "FAIL_CLOSED"), "estimator diagnostic binding mismatch")
    return VerifiedEstimator(artifact, models, tuple(
        (name, artifact.manifest["file_table"][f"models/{name.lower()}.json"]["sha256"])
        for name in stats.MODEL_COLUMNS))
