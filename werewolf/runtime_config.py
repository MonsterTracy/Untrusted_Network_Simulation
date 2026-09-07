"""One named-backend runtime configuration, with no historical schema conversion."""

from collections.abc import Mapping
from copy import deepcopy


def _as_mapping(value, field):
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _required_string(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required")
    return value


def normalize_backend_config(config):
    config = _as_mapping(config, "backend config")
    if set(config) - {"type", "base_url", "api_key_env", "default_model", "supports_json_schema"}:
        raise ValueError("unsupported backend fields")
    if config.get("type") != "openai_compatible":
        raise ValueError("unsupported backend type")
    supports = config.get("supports_json_schema", False)
    if type(supports) is not bool:
        raise ValueError("backend supports_json_schema must be boolean")
    return {"type": "openai_compatible", "base_url": config.get("base_url"),
            "api_key_env": config.get("api_key_env"), "default_model": config.get("default_model"),
            "supports_json_schema": supports}


def normalize_parser_config(config, backends):
    config = _as_mapping(config, "parser config")
    if set(config) - {"backend", "model", "model_params"}:
        raise ValueError("unsupported parser fields")
    backend = _required_string(config.get("backend"), "parser.backend")
    if backend not in backends:
        raise ValueError(f"parser backend does not exist: {backend}")
    return {"backend": backend, "model": _required_string(config.get("model"), "parser.model"),
            "model_params": deepcopy(dict(_as_mapping(config.get("model_params", {}), "parser.model_params")))}


def normalize_agent_profile(profile, backends):
    profile = _as_mapping(profile, "agent profile")
    if set(profile) - {"profile_name", "agent_type", "backend", "model", "model_params", "sample_ratio"}:
        raise ValueError("unsupported agent profile fields")
    name = _required_string(profile.get("profile_name"), "agent profile profile_name")
    agent_type = _required_string(profile.get("agent_type"), f"agent profile {name}.agent_type")
    if agent_type != "gpt":
        raise ValueError("the canonical playing agent type is gpt")
    backend = _required_string(profile.get("backend"), f"agent profile {name}.backend")
    if backend not in backends:
        raise ValueError(f"agent profile backend does not exist: {backend}")
    params = deepcopy(dict(_as_mapping(profile.get("model_params", {}), "agent profile model_params")))
    if set(params) - {"temperature", "gameplay_max_tokens", "gameplay_prompt_profile"}:
        raise ValueError("unsupported agent model parameters")
    if params.get("gameplay_prompt_profile", "strict_classic7") != "strict_classic7":
        raise ValueError("only strict_classic7 gameplay is supported")
    params["gameplay_prompt_profile"] = "strict_classic7"
    if "gameplay_max_tokens" in params and (type(params["gameplay_max_tokens"]) is not int or params["gameplay_max_tokens"] <= 0):
        raise ValueError("agent profile model_params.gameplay_max_tokens must be a positive integer")
    return {"profile_name": name, "agent_type": agent_type, "backend": backend,
            "model": _required_string(profile.get("model"), f"agent profile {name}.model"),
            "model_params": params, "sample_ratio": profile.get("sample_ratio", 1.0)}


def normalize_runtime_config(config):
    config = _as_mapping(config, "runtime config")
    if set(config) - {"backends", "parser", "agent_config", "env_config"}:
        raise ValueError("unsupported runtime fields; named backends are required")
    raw = _as_mapping(config.get("backends"), "backends")
    if not raw:
        raise ValueError("backends must not be empty")
    backends = {_required_string(k, "backend name"): normalize_backend_config(v) for k, v in raw.items()}
    agents = _as_mapping(config.get("agent_config"), "agent_config")
    if set(agents) - {"all_candidates", "must_include", "allow_cross_team_profiles"}:
        raise ValueError("unsupported agent_config fields")
    raw_candidates = agents.get("all_candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("agent_config.all_candidates must be a nonempty list")
    candidates = [normalize_agent_profile(p, backends) for p in raw_candidates]
    names = [p["profile_name"] for p in candidates]
    if len(set(names)) != len(names):
        raise ValueError("duplicate profile_name")
    required = agents.get("must_include", [])
    if not isinstance(required, list) or any(name not in names for name in required):
        raise ValueError("must_include profile does not exist")
    cross_team = agents.get("allow_cross_team_profiles", False)
    if type(cross_team) is not bool:
        raise ValueError("allow_cross_team_profiles must be boolean")
    return {"backends": backends, "parser": normalize_parser_config(config.get("parser"), backends),
            "agent_config": {"all_candidates": candidates, "must_include": deepcopy(required),
                             "allow_cross_team_profiles": cross_team},
            "env_config": deepcopy(dict(_as_mapping(config.get("env_config"), "env_config")))}
