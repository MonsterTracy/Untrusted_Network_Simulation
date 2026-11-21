from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from ipaddress import ip_address
from time import perf_counter
from urllib.parse import urlparse

import openai

from werewolf.backends.base import BackendError, LLMBackend


_TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens")


@dataclass(frozen=True)
class _RequestContext:
    records: list | None = None
    gameplay_only: bool = False
    timeout: float | None = None
    game_seed: int | str = "UNKNOWN"
    step: int | str = "UNKNOWN"
    seat: int | str = "UNKNOWN"
    method: str = "UNKNOWN"
    phase: str = "UNKNOWN"
    kind: str = "UNKNOWN"
    attempt: int | str = "UNKNOWN"
    strategy: str | None = None


_request_context = ContextVar("openai_request_context", default=None)


@contextmanager
def model_request_context(records=None, *, gameplay_only=False, timeout=None,
                          game_seed="UNKNOWN", step="UNKNOWN", seat="UNKNOWN",
                          method="UNKNOWN", phase="UNKNOWN"):
    """为本次操作设置可选记录和客户端策略，退出后恢复调用方上下文。"""
    token = _request_context.set(_RequestContext(
        records=records, gameplay_only=gameplay_only, timeout=timeout,
        game_seed=game_seed, step=step, seat=seat, method=method, phase=phase,
    ))
    try:
        yield
    finally:
        _request_context.reset(token)


@contextmanager
def model_request_details(**details):
    """复用调用方明确给出的生成阶段、尝试序号和所选策略。"""
    scope = _request_context.get()
    if scope is None or scope.records is None:
        yield
        return
    first_record = len(scope.records)
    token = _request_context.set(replace(scope, **details))
    try:
        yield
    except Exception as error:
        if "attempt" in details and len(scope.records) == first_record + 1:
            record = scope.records[first_record]
            if record["status"] == "response_received":
                record["validation_error_type"] = type(error).__name__
        raise
    finally:
        _request_context.reset(token)


def in_gameplay_request() -> bool:
    scope = _request_context.get()
    return scope is not None and scope.gameplay_only


def _is_loopback_base_url(base_url) -> bool:
    if base_url is None:
        return False
    try:
        hostname = urlparse(str(base_url)).hostname
    except (TypeError, ValueError):
        return False
    if hostname is None:
        return False

    normalized = hostname.lower()
    if normalized == "localhost":
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


class OpenAICompatibleBackend(LLMBackend):
    def __init__(
        self,
        api_key=None,
        base_url=None,
        default_model=None,
        client=None,
        max_retries=None,
        supports_json_schema=False,
    ):
        if not isinstance(
            supports_json_schema,
            bool,
        ):
            raise TypeError(
                "supports_json_schema must be boolean"
            )
        if client is None:
            if not api_key:
                raise BackendError(
                    "api_key is required when an OpenAI-compatible client is not injected."
                )
            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            if max_retries is not None:
                client_kwargs["max_retries"] = max_retries
            if _is_loopback_base_url(base_url):
                client_kwargs["http_client"] = openai.DefaultHttpx2Client(
                    trust_env=False,
                )
            client = openai.OpenAI(**client_kwargs)

        self.client = client
        self.base_url = base_url
        self.chat_completions_endpoint = (
            f"{base_url.rstrip('/')}/chat/completions"
            if base_url
            else None
        )
        self.default_model = default_model
        self.supports_json_schema = (
            supports_json_schema
        )

    def chat(
        self,
        messages,
        model=None,
        temperature=0.7,
        max_tokens=None,
        response_format=None,
        **kwargs,
    ) -> str:
        content, _usage = self.chat_with_metadata(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            **kwargs,
        )
        return content

    def chat_with_metadata(
        self,
        messages,
        model=None,
        temperature=0.7,
        max_tokens=None,
        response_format=None,
        **kwargs,
    ) -> tuple[str, dict[str, int | str | None] | None]:
        """Return text plus provider-reported token usage when available."""

        resolved_model = model or self.default_model
        if not resolved_model:
            raise BackendError("A model is required for an LLM chat request.")

        request = dict(kwargs)
        request["model"] = resolved_model
        request["messages"] = messages
        if temperature is not None:
            request["temperature"] = temperature
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if response_format is not None:
            request["response_format"] = response_format

        scope = _request_context.get()
        record = None
        try:
            client = self.client
            if scope is not None and scope.gameplay_only:
                client = client.with_options(max_retries=0)
            if scope is not None and scope.timeout is not None:
                request["timeout"] = scope.timeout

            create = client.chat.completions.create
            if scope is not None and scope.records is not None:
                record = {
                    "request_index": len(scope.records) + 1,
                    "game_seed": scope.game_seed, "step": scope.step, "seat": scope.seat,
                    "method": scope.method, "phase": scope.phase, "kind": scope.kind,
                    "attempt": scope.attempt, "strategy": scope.strategy, "model": resolved_model,
                    "dispatched": True, "status": "dispatched", "response_received": "UNKNOWN",
                    "error_type": None, "finish_reason": "UNKNOWN", "elapsed_seconds": "UNKNOWN",
                    "sdk_retries": 0 if scope.gameplay_only or getattr(client, "max_retries", None) == 0 else "UNKNOWN",
                    **dict.fromkeys(_TOKEN_FIELDS, "UNKNOWN"),
                }
                scope.records.append(record)
                started = perf_counter()

            response = create(**request)
            if record is not None:
                record["elapsed_seconds"] = perf_counter() - started
                record["response_received"] = True
                metadata = _extract_usage(getattr(response, "usage", None))
                for field in _TOKEN_FIELDS:
                    value = (metadata or {}).get(field)
                    if value is not None:
                        record[field] = value

            choice = response.choices[0]
            content = choice.message.content
            if record is not None:
                finish_reason = getattr(choice, "finish_reason", None)
                if isinstance(finish_reason, str):
                    record["finish_reason"] = finish_reason
            if not isinstance(content, str):
                raise BackendError(
                    "OpenAI-compatible chat response content must be text."
                )
            if record is None:
                metadata = _extract_usage(getattr(response, "usage", None))
                finish_reason = getattr(choice, "finish_reason", None)
            if isinstance(finish_reason, str):
                metadata = dict(metadata or {})
                metadata["finish_reason"] = finish_reason
            if record is not None:
                record["status"] = "response_received"
            return content, metadata
        except (Exception, KeyboardInterrupt) as exc:
            if record is not None:
                record["status"] = (
                    "invalid_response" if record["response_received"] is True else "request_error"
                )
                if isinstance(exc, openai.APIStatusError):
                    record["response_received"] = True
                record["error_type"] = type(exc).__name__
                if record["elapsed_seconds"] == "UNKNOWN":
                    record["elapsed_seconds"] = perf_counter() - started
            if isinstance(exc, (BackendError, KeyboardInterrupt)):
                raise
            raise BackendError(
                "OpenAI-compatible chat request failed."
            ) from exc


def _extract_usage(usage):
    """Normalize OpenAI and Responses-style token names without guessing."""

    if usage is None:
        return None

    def read(*names):
        for name in names:
            if isinstance(usage, dict):
                value = usage.get(name)
            else:
                value = getattr(usage, name, None)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        return None

    return {
        "input_tokens": read("prompt_tokens", "input_tokens"),
        "output_tokens": read("completion_tokens", "output_tokens"),
        "total_tokens": read("total_tokens"),
    }
