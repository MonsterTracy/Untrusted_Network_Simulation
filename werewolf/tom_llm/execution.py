"""Explicit v2 engineering execution; v1 artifacts are never auto-upgraded."""

from dataclasses import dataclass
from pathlib import Path

from werewolf.artifact_io import publish_artifact, verify_artifact, canonical_jsonl_bytes
from werewolf.tom_llm.common import digest, identity, require, strict_json
from werewolf.tom_llm.policy import policy_binding
from werewolf.tom_llm.server_fixed import ServerFixedService, wire_binding, WIRE_VERSION
from werewolf.tom_llm.study import validate_request, context_preflight
from werewolf.tom_llm.development_source import development_source_identity as source_identity

EXECUTION_VERSION = "classic7_llm_mainline_mock_execution_v1"
STUDY_VERSION = "classic7_llm_mainline_engineering_study_v2"


@dataclass(frozen=True)
class ServerFixedExecution:
    service: ServerFixedService

    def identity(self):
        require(type(self.service) is ServerFixedService, "explicit service identity required")
        return {"schema_version": EXECUTION_VERSION, "engineering_only": True,
                "wire": wire_binding(), "service": self.service.identity(),
                "policy": policy_binding(), "source": source_identity()}


def prepare_mock_study_v2(destination, requests, *, execution, tokenizer_identity, count_tokens):
    require(type(execution) is ServerFixedExecution, "explicit v2 execution required")
    require(type(tokenizer_identity) is str and tokenizer_identity.startswith("synthetic:"), "mock tokenizer required")
    require(requests and len({identity(r) for r in requests}) == len(requests), "duplicate or empty public requests")
    bound = execution.identity()
    return publish_artifact(Path(destination) / "contract", manifest_fields={
        "artifact_type": "llm_engineering_study", "schema_version": STUDY_VERSION,
        "engineering_only": True, "wire_version": WIRE_VERSION,
        "execution": bound, "execution_digest": digest(bound),
        "tokenizer_identity": tokenizer_identity, "preflight": context_preflight(requests, count_tokens),
        "row_identity_digest": digest([identity(r) for r in requests]), "request_count": len(requests)},
        files={"requests.jsonl": canonical_jsonl_bytes(requests)})


def validate_execution(contract, execution):
    require(type(execution) is ServerFixedExecution, "explicit v2 execution required")
    bound = execution.identity()
    require(contract.manifest["wire_version"] == WIRE_VERSION and contract.manifest["engineering_only"] is True
            and contract.manifest["execution"] == bound
            and contract.manifest["execution_digest"] == digest(bound), "execution/service/config/source drift")


def open_study_v2(root, *, execution):
    artifact = verify_artifact(Path(root) / "contract", expected_artifact_type="llm_engineering_study",
                               expected_schema_version=STUDY_VERSION)
    validate_execution(artifact, execution)
    rows = [strict_json(line) for line in (artifact.path / "requests.jsonl").read_bytes().splitlines()]
    for row in rows:
        validate_request(row)
    m = artifact.manifest
    require(len(rows) == m["request_count"] and len({identity(r) for r in rows}) == len(rows)
            and digest([identity(r) for r in rows]) == m["row_identity_digest"], "study request coverage mismatch")
    return artifact, rows
