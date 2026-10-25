"""Durable mock-only request lifecycle; no network factory or model loader."""

import fcntl
import math
import time
from pathlib import Path
from dataclasses import dataclass

from werewolf.artifact_io import publish_artifact, verify_artifact, canonical_json_bytes, canonical_jsonl_bytes
from werewolf.tom_llm.common import digest, identity, require, strict_json
from werewolf.tom_llm.output import parse_output, validate_prediction
from werewolf.tom_llm.study import open_study, wire_request

ATTEMPT_VERSION = "classic7_llm_mainline_mock_attempt_v1"
RESULT_VERSION = "classic7_llm_mainline_mock_result_v1"
SEAL_VERSION = "classic7_llm_mainline_mock_seal_v1"


class TransportFailure(Exception):
    """Recoverable transport failure simulated by the local mock."""


class InfrastructureFailure(Exception):
    """Service/OOM failure pauses execution rather than fabricating predictions."""


@dataclass(frozen=True)
class Reply:
    content: str | None
    finish_reason: str
    usage: dict


class MockBackend:
    def __init__(self, respond):
        self.respond = respond
        self.calls = []

    def chat(self, request):
        self.calls.append(request)
        reply = self.respond(request)
        require(type(reply) is Reply, "mock backend must return Reply")
        return reply


def _open(root, execution):
    if execution is None:
        return open_study(root)
    from werewolf.tom_llm.execution import open_study_v2
    return open_study_v2(root, execution=execution)


def _version(v1, contract):
    from werewolf.tom_llm.execution import STUDY_VERSION
    if contract.manifest["schema_version"] == STUDY_VERSION:
        return v1.removesuffix("v1") + "v2"
    from werewolf.tom_llm.study import VERSION
    require(contract.manifest["schema_version"] == VERSION, "unsupported study version")
    return v1


def _wire(contract, request):
    if _version(RESULT_VERSION, contract) != RESULT_VERSION:
        from werewolf.tom_llm.server_fixed import wire_request_v2
        return wire_request_v2(request)
    return wire_request(request)


def _check_execution(contract, execution):
    if execution is not None:
        from werewolf.tom_llm.execution import validate_execution
        validate_execution(contract, execution)


def _execution_fields(contract):
    if _version(RESULT_VERSION, contract) == RESULT_VERSION:
        return {}
    return {"execution_digest": contract.manifest["execution_digest"],
            "wire_version": contract.manifest["wire_version"],
            "service_digest": contract.manifest["execution"]["service"]["manifest_digest"]}


def _attempt(path, fields, payload=None, *, contract):
    return publish_artifact(path, manifest_fields={"artifact_type": "llm_mock_attempt",
        "schema_version": _version(ATTEMPT_VERSION, contract), **_execution_fields(contract), **fields},
        files={} if payload is None else {"reply.json": canonical_json_bytes(payload)})


def _terminal(root, contract, request, attempts):
    latest = attempts[-1]
    require(latest.manifest["outcome"] == "response", "request lacks a terminal response")
    reply = strict_json((latest.path / "reply.json").read_bytes())
    prediction = parse_output(reply["content"], request["observer"], finish_reason=reply["finish_reason"]).to_record()
    result = {"game_id": request["game_id"], "boundary_id": request["boundary_id"],
        "prefix_digest": request["prefix_digest"], "plan_digest": request["plan_digest"],
        "observer": request["observer"], "fold": request["fold"], "canonical_shift": 0,
        "temporal_condition": "explicit_day_phase", "prediction": prediction,
        "prompt_digest": request["prompt_digest"], "request_digest": digest(_wire(contract, request)),
        "study_digest": contract.manifest_digest, "attempt_digests": [a.manifest_digest for a in attempts],
        "usage": reply["usage"],
        "dispatch_elapsed_seconds": [a.manifest["elapsed_seconds"] for a in attempts], **_execution_fields(contract)}
    return publish_artifact(root / "result", manifest_fields={"artifact_type": "llm_mock_result",
        "schema_version": _version(RESULT_VERSION, contract), **_execution_fields(contract), "study_digest": contract.manifest_digest,
        "row_identity": list(identity(request)), "attempt_digests": result["attempt_digests"]},
        files={"result.json": canonical_json_bytes(result)})


def run_mock_study(root, backend, *, after_response=None, execution=None):
    """Exact two-dispatch cap, persistent responses, and no semantic repair."""
    require(type(backend) is MockBackend, "phase-two execution accepts only local MockBackend")
    root = Path(root)
    contract, requests = _open(root, execution)
    with (root / "inference.lock").open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (root / "seal").exists(), "sealed study closes inference")
        for request in requests:
            _check_execution(contract, execution)
            lineage = root / "requests" / digest(identity(request))
            if (lineage / "result").exists():
                verify_result(lineage / "result", contract, request)
                continue
            wire = _wire(contract, request)
            attempts = []
            for number in (1, 2):
                start_path, finish_path = lineage / f"dispatch-{number}", lineage / f"attempt-{number}"
                if finish_path.exists():
                    attempt = verify_artifact(finish_path, expected_artifact_type="llm_mock_attempt",
                                              expected_schema_version=_version(ATTEMPT_VERSION, contract))
                    require(attempt.manifest["study_digest"] == contract.manifest_digest
                            and attempt.manifest["request_digest"] == digest(wire)
                            and attempt.manifest["number"] == number, "attempt identity drift")
                else:
                    if start_path.exists():
                        raise InfrastructureFailure("interrupted in-flight request; no automatic regeneration")
                    _check_execution(contract, execution)
                    _attempt(start_path, {"study_digest": contract.manifest_digest, "request_digest": digest(wire),
                                         "number": number, "outcome": "dispatched"}, contract=contract)
                    started = time.perf_counter()
                    try:
                        reply = backend.chat(wire)
                    except TransportFailure:
                        attempt = _attempt(finish_path, {"study_digest": contract.manifest_digest,
                            "request_digest": digest(wire), "number": number, "outcome": "transport_failure",
                            "elapsed_seconds": time.perf_counter() - started}, contract=contract)
                    else:
                        elapsed = time.perf_counter() - started
                        _check_execution(contract, execution)
                        attempt = _attempt(finish_path, {"study_digest": contract.manifest_digest,
                            "request_digest": digest(wire), "number": number, "outcome": "response",
                            "elapsed_seconds": elapsed},
                            {"content": reply.content, "finish_reason": reply.finish_reason, "usage": reply.usage}, contract=contract)
                        if after_response is not None:
                            after_response()
                attempts.append(attempt)
                if attempt.manifest["outcome"] == "response":
                    break
                require(attempt.manifest["outcome"] == "transport_failure", "invalid attempt outcome")
            if attempts[-1].manifest["outcome"] != "response":
                raise InfrastructureFailure("transport retry budget exhausted; no uniform fabrication")
            _terminal(lineage, contract, request, attempts)


def verify_result(path, contract, request):
    artifact = verify_artifact(path, expected_artifact_type="llm_mock_result", expected_schema_version=_version(RESULT_VERSION, contract))
    m = artifact.manifest
    for key, value in _execution_fields(contract).items():
        require(m.get(key) == value, "result execution identity drift")
    require(m["study_digest"] == contract.manifest_digest and m["row_identity"] == list(identity(request))
            and set(m["file_table"]) == {"result.json"}, "result study/row binding mismatch")
    row = strict_json((artifact.path / "result.json").read_bytes())
    require(all(row.get(k) == v for k, v in _execution_fields(contract).items()), "row execution identity drift")
    require(row["study_digest"] == contract.manifest_digest and identity(row) == identity(request)
            and row["fold"] == request["fold"] and row["plan_digest"] == request["plan_digest"]
            and row["canonical_shift"] == 0 and row["temporal_condition"] == "explicit_day_phase"
            and row["prompt_digest"] == request["prompt_digest"] and row["request_digest"] == digest(_wire(contract, request))
            and row["attempt_digests"] == m["attempt_digests"], "result provenance mismatch")
    require(1 <= len(m["attempt_digests"]) <= 2, "invalid terminal attempt count")
    for number, expected in enumerate(m["attempt_digests"], 1):
        attempt = verify_artifact(artifact.path.parent / f"attempt-{number}",
                                 expected_artifact_type="llm_mock_attempt", expected_schema_version=_version(ATTEMPT_VERSION, contract))
        dispatch = verify_artifact(artifact.path.parent / f"dispatch-{number}",
                                  expected_artifact_type="llm_mock_attempt", expected_schema_version=_version(ATTEMPT_VERSION, contract))
        for item in (attempt, dispatch):
            require(all(item.manifest.get(k) == v for k, v in _execution_fields(contract).items()),
                    "attempt execution identity drift")
            require(item.manifest["number"] == number and item.manifest["study_digest"] == contract.manifest_digest
                    and item.manifest["request_digest"] == row["request_digest"], "attempt provenance mismatch")
        require(attempt.manifest_digest == expected and dispatch.manifest["outcome"] == "dispatched",
                "attempt digest/dispatch mismatch")
        duration = attempt.manifest["elapsed_seconds"]
        require(type(duration) in (int, float) and math.isfinite(duration) and duration >= 0,
                "invalid attempt elapsed time")
        require(len(row["dispatch_elapsed_seconds"]) == len(m["attempt_digests"])
                and row["dispatch_elapsed_seconds"][number - 1] == duration, "attempt timing provenance mismatch")
        last = number == len(m["attempt_digests"])
        require(attempt.manifest["outcome"] == ("response" if last else "transport_failure"), "attempt chain mismatch")
        if last:
            reply = strict_json((attempt.path / "reply.json").read_bytes())
            require(row["prediction"] == parse_output(reply["content"], request["observer"],
                    finish_reason=reply["finish_reason"]).to_record() and row["usage"] == reply["usage"],
                    "terminal prediction differs from durable response")
    require(row["prediction"]["observer"] == row["observer"], "prediction observer mismatch")
    validate_prediction(row["prediction"])
    return artifact, row


def seal_mock_study(root, *, execution=None):
    root = Path(root)
    contract, requests = _open(root, execution)
    with (root / "inference.lock").open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows, results = [], []
        require((root / "requests").is_dir()
                and {p.name for p in (root / "requests").iterdir()} == {digest(identity(r)) for r in requests},
                "terminal request inventory mismatch")
        for request in requests:
            artifact, row = verify_result(root / "requests" / digest(identity(request)) / "result", contract, request)
            rows.append(row)
            results.append(artifact.manifest_digest)
        return publish_artifact(root / "seal", manifest_fields={"artifact_type": "llm_mock_seal",
            "schema_version": _version(SEAL_VERSION, contract), **_execution_fields(contract), "engineering_only": True, "study_digest": contract.manifest_digest,
            "row_identity_digest": digest([identity(r) for r in rows]), "request_count": len(rows),
            "terminal_digests": results}, files={"predictions.jsonl": canonical_jsonl_bytes(rows)})


def open_mock_predictions(root, *, execution=None):
    contract, requests = _open(root, execution)
    seal = verify_artifact(Path(root) / "seal", expected_artifact_type="llm_mock_seal", expected_schema_version=_version(SEAL_VERSION, contract))
    require(all(seal.manifest.get(k) == v for k, v in _execution_fields(contract).items()), "seal execution identity drift")
    require(seal.manifest["study_digest"] == contract.manifest_digest and seal.manifest["engineering_only"] is True,
            "prediction seal binding mismatch")
    rows = [strict_json(line) for line in (seal.path / "predictions.jsonl").read_bytes().splitlines()]
    require(len(rows) == seal.manifest["request_count"] == len(requests)
            and [identity(r) for r in rows] == [identity(r) for r in requests]
            and digest([identity(r) for r in rows]) == seal.manifest["row_identity_digest"], "prediction seal coverage mismatch")
    terminals = []
    for request, row in zip(requests, rows, strict=True):
        artifact, terminal = verify_result(Path(root) / "requests" / digest(identity(request)) / "result", contract, request)
        require(row == terminal, "sealed prediction differs from terminal")
        terminals.append(artifact.manifest_digest)
    require(terminals == seal.manifest["terminal_digests"], "prediction seal terminal binding mismatch")
    return seal, rows
