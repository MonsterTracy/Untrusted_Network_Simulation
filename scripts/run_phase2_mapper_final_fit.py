"""Fit and publish the one selected full-development M3 mapper from sealed OOF Q."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.cli import _artifact_path, _storage_root
from werewolf import phase2_offline as offline
from werewolf.phase2_mapper import OOFProvenance, fit_full_development
from werewolf import phase2_mapper_final as final
from scripts import run_phase2_mapper_study as development


OUTPUT_RELATIVE = Path("paper-studies/mapper-final") / final.FINAL_NAME
FINAL_SOURCE_FILES = (
    "scripts/run_phase2_mapper_final_fit.py",
    "werewolf/phase2_mapper_final.py",
    "werewolf/phase2_mapper_runtime.py",
    "docs/research/phase2-mapper-final-runbook.md",
)


def attest_final_source() -> dict:
    source = development.attest_source()  # Clean tracked HEAD and frozen implementation.
    root = Path(__file__).resolve().parents[1]
    development._git(root, "ls-files", "--error-unmatch", "--", *FINAL_SOURCE_FILES)
    for relative in FINAL_SOURCE_FILES:
        content = (root / relative).read_bytes()
        if development._git(root, "show", f"HEAD:{relative}") != content:
            raise final.FinalMapperError(f"final mapper source differs from HEAD: {relative}")
        source["source_sha256"][relative] = sha256_bytes(content)
    return source


def preflight(storage_profile: Path | str):
    source = attest_final_source()
    root = _storage_root(storage_profile)
    destination = _artifact_path(root, OUTPUT_RELATIVE)
    if os.path.lexists(destination):
        raise final.FinalMapperError(f"final mapper destination already exists: {destination}")
    study = final.verify_selected_study(final.STUDY_PATH)
    return destination, source, study


def run_final_fit(storage_profile: Path | str):
    destination, source, study = preflight(storage_profile)
    evaluation = development.EVALUATION
    oof = OOFProvenance.from_evaluation_root(evaluation)
    if study.manifest["inputs"]["oof"] != oof.record():
        raise final.FinalMapperError("current sealed OOF evaluation differs from selected study")
    layer = offline.build_development_layer(development.PUBLICATION, evaluation)
    development.validate_population(layer, oof)  # Six counts, including 5412 positives, before fit.
    candidate_digest = final.validate_study_candidate_population(study, layer.candidates)
    fitted = fit_full_development(layer.candidates, final.SELECTED_SPEC, oof)
    if fitted.manifest["oof_q"] != oof.record():
        raise final.FinalMapperError("full fitted mapper lost sealed OOF Q lineage")
    return final.publish_full_mapper(destination, fitted=fitted, rows=layer.candidates,
                                     study=study, source=source,
                                     candidate_digest=candidate_digest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-profile", type=Path, required=True,
                        help="storage profile JSON, normally configs/server.json")
    parser.add_argument("--preflight", action="store_true",
                        help="verify committed source, selected study, and unused output; do not fit")
    args = parser.parse_args(argv)
    if args.preflight:
        destination, source, study = preflight(args.storage_profile)
        print(canonical_json_bytes({"destination": str(destination),
                                    "source_commit": source["commit"],
                                    "selected_study_digest": study.manifest_digest}).decode())
        return 0
    artifact = run_final_fit(args.storage_profile)
    print(canonical_json_bytes({"artifact_path": str(artifact.path),
                                "manifest_digest": artifact.manifest_digest,
                                "model_digest": artifact.manifest["model"]["model_digest"]}).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
