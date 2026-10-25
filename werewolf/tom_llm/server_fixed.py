"""Versioned server-fixed wire and local evidence checks; no service/model factory."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from werewolf.artifact_io import verify_artifact, sha256_bytes
from werewolf.backends.openai_compatible import _is_loopback_base_url
from werewolf.tom_llm.common import digest, require, require_digest, strict_json
from werewolf.tom_llm.policy import POLICY, ASSET_DIGESTS, policy_binding
from werewolf.tom_llm.prompt import asset_bytes
from werewolf.tom_llm.study import wire_request as wire_request_v1
from werewolf.tom_llm.transport import InjectedChatTransport

WIRE_VERSION = "classic7_llm_wire_server_fixed_v2"
SERVICE_VERSION = "classic7_llm_mainline_mock_service_v1"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/paper/tom-llm-wire-server-fixed-v2.json"
WIRE_CONFIG = {
    "schema_version": WIRE_VERSION,
    "template_origin": "server_fixed",
    "template_source": "werewolf/tom_llm/assets/chat_template.jinja",
    "template_sha256": ASSET_DIGESTS["chat_template.jinja"],
    "tokenizer_asset_set_digest": "2a79647fc03fd742128d932e571e644d73d350076d88388e3baf49e5aa3fca91",
    "server": {
        "model": POLICY["model"], "revision": POLICY["model_revision"],
        "tokenizer_revision": POLICY["model_revision"], "dtype": POLICY["dtype"], "quantization": None,
        "served_model_name": POLICY["model"], "max_model_len": 8192,
        "chat_template": "chat_template.jinja", "chat_template_content_format": "string",
        "trust_request_chat_template": False, "generation_config": "auto",
        "generation_config_sha256": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
        "add_generation_prompt": True, "continue_final_message": False, "add_special_tokens": False,
        "reasoning_parser": None,
    },
}


def wire_binding():
    policy_binding()
    config = strict_json(CONFIG_PATH.read_bytes())
    require(digest(config) == digest(WIRE_CONFIG), "server-fixed wire configuration drift")
    return {"version": WIRE_VERSION, "config": config, "config_digest": digest(config)}


def wire_request_v2(request):
    """Only the approved request-template member is removed from frozen v1."""
    wire_binding()
    wire = wire_request_v1(request)
    del wire["extra_body"]["chat_template"]
    return wire


def open_mock_service(root, *, expected_digest):
    """Verify bytes and pinned origin, not physical server state. Mock evidence only."""
    require_digest(expected_digest)
    artifact = verify_artifact(root, expected_artifact_type="llm_mock_serving_identity",
                               expected_schema_version=SERVICE_VERSION)
    m = artifact.manifest
    require(m["engineering_only"] is True and artifact.manifest_digest == expected_digest,
            "service identity drift")
    require(set(m["file_table"]) == {"service.json", "startup.json", "chat_template.jinja"},
            "service evidence inventory mismatch")
    config = wire_binding()["config"]
    require(digest(strict_json((artifact.path / "startup.json").read_bytes())) == digest(config["server"]),
            "server startup configuration drift")
    template = (artifact.path / "chat_template.jinja").read_bytes()
    require(template == asset_bytes("chat_template.jinja")
            and sha256_bytes(template) == config["template_sha256"], "server template bytes drift")
    service = strict_json((artifact.path / "service.json").read_bytes())
    require(set(service) == {"service_instance_id", "endpoint", "runtime", "weights_manifest_sha256",
                             "template_origin", "template_source", "wire_config_digest", "tokenizer_asset_set_digest"},
            "service identity schema mismatch")
    require(type(service["service_instance_id"]) is str and service["service_instance_id"].startswith("synthetic:")
            and _is_loopback_base_url(service["endpoint"]), "only synthetic loopback service evidence accepted")
    require(service["template_origin"] == "server_fixed" and service["template_source"] == config["template_source"]
            and service["wire_config_digest"] == digest(config)
            and service["tokenizer_asset_set_digest"] == config["tokenizer_asset_set_digest"],
            "service origin/config/assets drift")
    require_digest(service["weights_manifest_sha256"])
    require(type(service["runtime"]) is dict and set(service["runtime"]) == {"vllm", "python", "torch"}
            and all(type(v) is str and v for v in service["runtime"].values()), "service runtime identity missing")
    return artifact, service


@dataclass(frozen=True)
class ServerFixedService:
    """External expected digest is mandatory; re-read evidence before each dispatch."""
    root: Path
    expected_digest: str

    def identity(self):
        artifact, service = open_mock_service(self.root, expected_digest=self.expected_digest)
        return {"schema_version": SERVICE_VERSION, "manifest_digest": artifact.manifest_digest,
                "template_origin": "server_fixed", "template_sha256": WIRE_CONFIG["template_sha256"],
                "service": service}


class InjectedServerFixedTransport(InjectedChatTransport):
    """Injected client seam for mock qualification; no client or formal runner is created."""

    def __init__(self, client, *, service):
        require(type(service) is ServerFixedService, "explicit versioned service binding required")
        bound = service.identity()
        super().__init__(client, attestation={
            "model": POLICY["model"], "revision": POLICY["model_revision"], "dtype": POLICY["dtype"],
            "tokenizer_revision": POLICY["model_revision"], "chat_template_sha256": WIRE_CONFIG["template_sha256"],
            "weights_manifest_sha256": bound["service"]["weights_manifest_sha256"]})
        self.service = service
        self.bound_identity = deepcopy(bound)
        require(str(client.base_url).rstrip("/") == bound["service"]["endpoint"].rstrip("/"), "service endpoint drift")

    def chat_public(self, request):
        return self.chat(wire_request_v2(request))

    def chat(self, request):
        require(self.service.identity() == self.bound_identity, "service identity drift")
        require(str(self.client.base_url).rstrip("/") == self.bound_identity["service"]["endpoint"].rstrip("/"),
                "service endpoint drift")
        # Reconstructing from public messages/schema prevents a caller from adding hidden parameters.
        expected = validated_wire_surface_v2(request)
        require(digest(request) == digest(expected), "server-fixed request parameter/template drift")
        reply = super().chat(request)
        require(self.service.identity() == self.bound_identity, "service identity drift during dispatch")
        return reply


def validated_wire_surface_v2(request):
    """Validate the transport surface; study validation separately binds messages to a public plan."""
    wire_binding()
    d = POLICY["decoding"]
    schema = request.get("response_format", {}).get("json_schema", {}).get("schema")
    from werewolf.tom_llm.prompt import schema as response_schema
    require(digest(schema) in {digest(response_schema(i)) for i in range(7)}, "response schema drift")
    messages = request.get("messages")
    require(type(messages) is list and len(messages) == 2
            and [m.get("role") for m in messages] == ["system", "user"]
            and all(set(m) == {"role", "content"} and type(m["content"]) is str for m in messages)
            and messages[0]["content"] == asset_bytes("system.txt").decode(), "message format/system drift")
    return {"model": POLICY["model"], "messages": messages, "temperature": d["temperature"],
            "top_p": d["top_p"], "n": 1, "max_tokens": d["max_tokens"], "seed": d["seed"],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "non_self_suspicion", "strict": True, "schema": schema}},
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}, "top_k": -1, "min_p": 0}}
