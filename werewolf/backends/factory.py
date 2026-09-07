import os
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

from werewolf.backends.base import BackendError, LLMBackend
from werewolf.backends.openai_compatible import OpenAICompatibleBackend
from werewolf.runtime_config import normalize_runtime_config


def _non_empty(value):
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def is_local_unauthenticated_backend(config: Mapping) -> bool:
    """Return whether one backend is an explicit loopback endpoint."""

    if _non_empty(config.get("api_key_env")) is not None:
        return False

    base_url = _non_empty(config.get("base_url"))
    if base_url is None:
        return False

    parsed = urlparse(base_url)
    return (
        parsed.scheme in {"http", "https"}
        and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    )


def load_named_backends(
    config: Mapping,
    env_file: str | Path | None = ".env",
    *,
    max_retries: int | None = None,
) -> dict[str, LLMBackend]:
    normalized = normalize_runtime_config(config)
    if env_file is not None:
        load_dotenv(dotenv_path=env_file, override=False)

    backends = {}
    for name, backend_config in normalized["backends"].items():
        backend_type = backend_config["type"]
        if backend_type != "openai_compatible":
            raise ValueError(
                f"unsupported backend type: {backend_type}"
            )

        if is_local_unauthenticated_backend(backend_config):
            api_key = "local-mlx"
        else:
            api_key_env = (
                _non_empty(backend_config.get("api_key_env"))
                or "OPENAI_API_KEY"
            )
            api_key = _non_empty(os.environ.get(api_key_env))
            if api_key is None:
                raise ValueError(
                    f"API key environment variable {api_key_env} "
                    f"is required for backend '{name}'"
                )

        backend_kwargs = {
            "api_key": api_key,
            "base_url": backend_config.get("base_url"),
            "default_model": backend_config.get("default_model"),
            "supports_json_schema": backend_config[
                "supports_json_schema"
            ],
        }
        if max_retries is not None:
            if (
                isinstance(max_retries, bool)
                or not isinstance(max_retries, int)
                or max_retries < 0
            ):
                raise ValueError("max_retries must be a non-negative integer")
            backend_kwargs["max_retries"] = max_retries
        backends[name] = OpenAICompatibleBackend(**backend_kwargs)
    return backends


def resolve_backend(name, backends):
    if name in backends:
        return backends[name]

    available_names = ", ".join(
        sorted(str(backend_name) for backend_name in backends)
    )
    if not available_names:
        available_names = "(none)"
    raise ValueError(
        f"unknown backend '{name}'. "
        f"Available backends: {available_names}"
    )
