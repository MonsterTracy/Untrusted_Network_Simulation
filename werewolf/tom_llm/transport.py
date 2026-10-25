"""Injected provider adapter; never creates a client, loads weights or starts a service."""

import openai

from werewolf.backends.openai_compatible import _extract_usage, _is_loopback_base_url
from werewolf.tom_llm.common import require, require_digest
from werewolf.tom_llm.policy import POLICY, ASSET_DIGESTS
from werewolf.tom_llm.inference import Reply, TransportFailure, InfrastructureFailure


class InjectedChatTransport:
    """For later deployment qualification only; phase-two runner rejects this backend."""

    def __init__(self, client, *, attestation):
        require(type(getattr(client, "max_retries", None)) is int and client.max_retries == 0,
                "SDK retries must be explicitly disabled")
        require(_is_loopback_base_url(getattr(client, "base_url", None)), "only declared loopback serving is accepted")
        require(set(attestation) == {"model", "revision", "dtype", "tokenizer_revision", "chat_template_sha256",
                                    "weights_manifest_sha256"}, "provider attestation schema mismatch")
        require(attestation["model"] == POLICY["model"] and attestation["revision"] == POLICY["model_revision"]
                and attestation["tokenizer_revision"] == POLICY["model_revision"]
                and attestation["dtype"] == POLICY["dtype"]
                and attestation["chat_template_sha256"] == ASSET_DIGESTS["chat_template.jinja"],
                "provider model/tokenizer/template identity mismatch")
        require_digest(attestation["weights_manifest_sha256"])
        self.client = client
        self.attestation = dict(attestation)
        self.endpoint = str(client.base_url).rstrip("/")

    def _check_client(self):
        require(type(self.client.max_retries) is int and self.client.max_retries == 0
                and str(self.client.base_url).rstrip("/") == self.endpoint, "provider endpoint or retry drift")

    def chat(self, request):
        # Retry belongs to the outer persistent lifecycle, never the SDK.
        self._check_client()
        require(self.client.max_retries == 0 and request["model"] == self.attestation["model"],
                "provider/request identity or retry drift")
        try:
            response = self.client.chat.completions.create(**request)
        except (openai.APIConnectionError, openai.RateLimitError) as error:
            raise TransportFailure("provider transport failure") from error
        except Exception as error:
            raise InfrastructureFailure("provider service failure; no synthetic fallback") from error
        self._check_client()
        choices = getattr(response, "choices", None)
        if getattr(response, "model", None) != request["model"] or not choices or len(choices) != 1:
            raise InfrastructureFailure("provider response model/choice mismatch")
        choice = choices[0]
        content = None if getattr(choice.message, "refusal", None) else choice.message.content
        usage = _extract_usage(getattr(response, "usage", None)) or {}
        return Reply(content, choice.finish_reason, usage)
