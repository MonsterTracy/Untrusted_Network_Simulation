"""Read-only seal interface; provenance is supplied by the reviewed upstream chain."""

from werewolf.artifact_io import verify_artifact, sha256_bytes
from werewolf.tom_llm.common import digest, require, require_digest, strict_json
from werewolf.tom_llm.selection import validate_selection

BINDING_VERSION = "classic7_llm_selection_binding_v1"


def validate_selection_source(selection, folds, *, expected, provenance, prepared_manifest):
    """No label/prediction inputs; compare the exact B1 payload before publication."""
    require(set(provenance) == {"publication_digest", "fold_manifest_digest", "prepared_manifest_digest",
                               "pre_reference_selection_evidence_digest", "reconciliation_evidence_digest"},
            "selection provenance schema mismatch")
    for value in provenance.values():
        require_digest(value)
    require(type(prepared_manifest) is dict and prepared_manifest["artifact_type"] == "backbone_tom_study",
            "frozen prepared manifest required")
    body = {k:v for k,v in prepared_manifest.items() if k != "manifest_digest"}
    require(digest(body) == prepared_manifest["manifest_digest"] == provenance["prepared_manifest_digest"],
            "prepared manifest identity drift")
    require(digest(prepared_manifest["folds"]) == digest(folds)
            and all(prepared_manifest["protocol_inputs"][k] == provenance[k]
                    for k in ("publication_digest", "fold_manifest_digest")), "prepared source/folds drift")
    validate_selection(selection, folds)
    require(set(expected) == {"parent_folds_digest", "selection_digest", "game_ids_digest", "selection_payload_sha256"},
            "selection expectation schema mismatch")
    from werewolf.artifact_io import canonical_json_bytes
    actual = {"parent_folds_digest": digest(folds), "selection_digest": selection["selection_digest"],
              "game_ids_digest": digest(selection["game_ids"]),
              "selection_payload_sha256": sha256_bytes(canonical_json_bytes(selection))}
    require(actual == expected, "reviewed selection/source identity drift")
    return {"schema_version": BINDING_VERSION, **provenance, **actual,
            "rule": selection["rule"], "scope": selection["scope"]}


def open_selection_binding(root, folds, *, expected, provenance, prepared_manifest):
    """Read an existing selection artifact; never publish or replace it here."""
    artifact = verify_artifact(root, expected_artifact_type="llm_game_selection",
                               expected_schema_version="classic7_llm_game_selection_v1")
    require(set(artifact.manifest["file_table"]) == {"selection.json"}, "selection inventory drift")
    raw = (artifact.path / "selection.json").read_bytes()
    selection = strict_json(raw)
    bound = validate_selection_source(selection, folds, expected=expected, provenance=provenance, prepared_manifest=prepared_manifest)
    require(sha256_bytes(raw) == bound["selection_payload_sha256"]
            and artifact.manifest["scope"] == selection["scope"]
            and artifact.manifest["selection_digest"] == bound["selection_digest"]
            and artifact.manifest["parent_folds_digest"] == bound["parent_folds_digest"], "selection seal binding drift")
    return {**bound, "selection_artifact_manifest_digest": artifact.manifest_digest}
