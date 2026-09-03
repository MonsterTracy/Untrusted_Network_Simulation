"""Run one seven-player Werewolf rollout with optional belief self-reports."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import random
import time
from copy import deepcopy
from typing import Any

import yaml

from werewolf.agents import agent_registry
from werewolf.backends import (
    create_backend,
    load_named_backends,
    resolve_backend,
)
from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
from werewolf.models import SpeechPerceiver
from werewolf.runtime_config import normalize_runtime_config


PUBLIC_SPEECH_RUNTIME_PHASES = frozenset({"speech", "speech_pk"})


def _act_with_optional_pre_speech_belief(
    acting_agent,
    observation,
    pre_speech_belief,
):
    if pre_speech_belief is None:
        return acting_agent.act(observation)
    belief_aware_act = getattr(
        acting_agent,
        "act_with_pre_speech_belief",
        None,
    )
    if not callable(belief_aware_act):
        raise TypeError(
            "speech agent must support immutable PRE-belief cognition handoff"
        )
    return belief_aware_act(
        observation,
        pre_speech_belief=pre_speech_belief,
    )


def eval(
    env,
    agent_list,
    roles_,
    canonical_recorder=None,
    call_audit=None,
):
    """Run one game, optionally through the sole Canonical Collection recorder.

    The environment needs the hidden role assignment to simulate the
    game. The recorder owns PRE construction and evidence; no trained ToM
    output is accepted by this runtime interface.
    """

    for agent in agent_list:
        agent.reset()

    done = False
    obs = env.reset(
        roles=roles_,
    )
    if canonical_recorder is not None:
        canonical_recorder.start(
            env,
            roles=roles_,
        )
    step_idx = 0
    info = None

    while not done:
        current_act_idx = obs[
            "current_act_idx"
        ]
        trigger = getattr(env, "phase", None)
        pre_speech_belief = None
        if canonical_recorder is not None:
            try:
                pre_speech_belief = canonical_recorder.before_agent_act(
                    env,
                    step_idx=step_idx,
                    acting_player_id=current_act_idx,
                    delivered_observation=obs,
                    speech_kind=(
                        trigger
                        if trigger in PUBLIC_SPEECH_RUNTIME_PHASES
                        else None
                    ),
                )
            except Exception as exc:
                raise canonical_recorder.failure_from_exception(exc) from exc

        audit_context = (
            call_audit.action_context(
                acting_player_id=current_act_idx,
                boundary_id=(
                    None
                    if pre_speech_belief is None
                    else pre_speech_belief.boundary_id
                ),
                is_public_speech=(trigger in PUBLIC_SPEECH_RUNTIME_PHASES),
            )
            if call_audit is not None
            else nullcontext()
        )
        try:
            with audit_context:
                action = _act_with_optional_pre_speech_belief(
                    agent_list[current_act_idx - 1],
                    obs,
                    pre_speech_belief,
                )
        except Exception as exc:
            if canonical_recorder is not None:
                raise canonical_recorder.failure_from_exception(exc) from exc
            raise

        if canonical_recorder is not None:
            canonical_recorder.after_agent_act(action)

        is_speech = trigger in PUBLIC_SPEECH_RUNTIME_PHASES
        env_audit_context = (
            call_audit.speech_perception_context(
                event_id=f"event-{len(env.public_events):06d}",
                boundary_id=pre_speech_belief.boundary_id,
                speaker_id=current_act_idx,
            )
            if call_audit is not None and is_speech
            else nullcontext()
        )
        try:
            with env_audit_context:
                obs, _, done, info = env.step(
                    action
                )
        except Exception as exc:
            if canonical_recorder is not None:
                raise canonical_recorder.failure_from_exception(exc) from exc
            raise

        if canonical_recorder is not None:
            canonical_recorder.after_env_step(
                env,
                observation_after=obs,
                terminal_after=done,
            )

        step_idx += 1

    if not isinstance(info, dict):
        raise RuntimeError(
            "finished game has no result information"
        )

    if info.get("Werewolf") == 1:
        if canonical_recorder is not None:
            canonical_recorder.finish(
                env,
                winner="Werewolf",
            )
        return "Werewolf win"

    if info.get("Werewolf") == -1:
        if canonical_recorder is not None:
            canonical_recorder.finish(
                env,
                winner="Villager",
            )
        return "Villager win"

    raise RuntimeError(
        "finished game has no recognized winner"
    )


def _weighted_profile_choice(profiles):
    if not profiles:
        raise ValueError(
            "no eligible agent profiles"
        )

    weights = [
        profile["sample_ratio"]
        for profile in profiles
    ]

    if (
        any(
            weight < 0
            for weight in weights
        )
        or not any(
            weight > 0
            for weight in weights
        )
    ):
        raise ValueError(
            "eligible agent profiles must have "
            "a positive sample_ratio"
        )

    return random.choices(
        profiles,
        weights=weights,
        k=1,
    )[0]


def assign_agents(
    candidate_profiles,
    env_config,
    log_save_path,
    assigined_roles,
    must_include,
    backends,
    allow_cross_team_profiles=False,
):
    """Assign configured agent profiles to the fixed role list.

    By default, a profile name remains exclusive to one faction. This
    preserves the random-competition behavior used to compare agent
    profiles across Werewolf and village teams.

    Collection-only configurations may explicitly set
    ``allow_cross_team_profiles=True`` so one API agent profile can
    control all seven players without duplicating equivalent profiles.
    """

    if not isinstance(
        allow_cross_team_profiles,
        bool,
    ):
        raise TypeError(
            "allow_cross_team_profiles must be boolean"
        )

    werewolf_team = {
        "Werewolf",
    }

    village_team = {
        "Villager",
        "Seer",
        "Witch",
        "Guard",
    }

    env_param = {
        "n_player": env_config[
            "n_player"
        ],
        "n_role": env_config[
            "n_role"
        ],
    }

    all_agent_profiles = {}
    profile_backend_ids = {}
    role2agent_list = []

    village_profiles = set()
    werewolf_profiles = set()

    forced_profile = None

    if must_include:
        required_profiles = [
            profile
            for profile in candidate_profiles
            if profile["profile_name"]
            in must_include
        ]

        forced_profile = (
            _weighted_profile_choice(
                required_profiles
            )
        )

    for index, role in enumerate(
        assigined_roles
    ):
        if (
            index == 0
            and forced_profile is not None
        ):
            profile = forced_profile
        else:
            if (
                role not in werewolf_team
                and role not in village_team
            ):
                raise ValueError(
                    f"unsupported role: {role}"
                )

            if allow_cross_team_profiles:
                eligible_profiles = list(
                    candidate_profiles
                )
            else:
                if role in werewolf_team:
                    opposite_profiles = (
                        village_profiles
                    )
                else:
                    opposite_profiles = (
                        werewolf_profiles
                    )

                eligible_profiles = [
                    profile
                    for profile
                    in candidate_profiles
                    if profile["profile_name"]
                    not in opposite_profiles
                ]

            if not eligible_profiles:
                raise ValueError(
                    "no eligible agent profiles "
                    f"for role: {role}"
                )

            profile = (
                _weighted_profile_choice(
                    eligible_profiles
                )
            )

        profile_name = profile[
            "profile_name"
        ]
        profile_backend_ids[profile_name] = profile["backend"]

        if role in werewolf_team:
            werewolf_profiles.add(
                profile_name
            )
        else:
            village_profiles.add(
                profile_name
            )

        if (
            profile_name
            not in all_agent_profiles
        ):
            model_params = dict(
                profile["model_params"]
            )

            all_agent_profiles[
                profile_name
            ] = agent_registry.build(
                profile["agent_type"],
                backend=resolve_backend(
                    profile["backend"],
                    backends,
                ),
                model_name=profile[
                    "model"
                ],
                **model_params,
            )

        role2agent_list.append(
            profile_name
        )

    agent_list = []

    for index, _role in enumerate(
        assigined_roles
    ):
        log_file = (
            os.path.join(
                log_save_path,
                f"Player_{index + 1}.jsonl",
            )
            if log_save_path is not None
            else None
        )

        profile_name = (
            role2agent_list[index]
        )

        agent_type, agent_param = (
            all_agent_profiles[
                profile_name
            ]
        )

        agent = (
            agent_registry.build_agent(
                agent_type,
                index,
                agent_param,
                env_param,
                log_file,
            )
        )
        agent.backend_id = profile_backend_ids[profile_name]

        agent_list.append(
            agent
        )

    return (
        role2agent_list,
        agent_list,
    )


def _resolve_backend_map(
    normalized,
    *,
    backend=None,
    backend_settings=None,
    backends=None,
):
    """Resolve injected, legacy, or configured backend instances."""

    if backends is not None:
        return dict(
            backends
        )

    backend_names = list(
        normalized["backends"]
    )

    if backend is not None:
        if len(backend_names) != 1:
            raise ValueError(
                "a single injected backend "
                "requires exactly one "
                "configured backend"
            )

        return {
            backend_names[0]: backend,
        }

    if backend_settings is not None:
        if len(backend_names) != 1:
            raise ValueError(
                "legacy backend_settings "
                "requires exactly one "
                "configured backend"
            )

        return {
            backend_names[0]: (
                create_backend(
                    backend_settings
                )
            ),
        }

    return load_named_backends(
        normalized,
        env_file=".env",
    )


def build_runtime(
    parsed_yaml,
    log_save_path,
    backend=None,
    backend_settings=None,
    roles=None,
    random_seed=None,
    backends=None,
):
    """Build the game environment and action agents."""

    config_for_normalization = (
        deepcopy(parsed_yaml)
    )

    if (
        backend_settings is not None
        and "backend"
        in config_for_normalization
        and "backends"
        not in config_for_normalization
    ):
        legacy_backend = (
            config_for_normalization[
                "backend"
            ]
        )

        for field in (
            "default_model",
            "agent_model",
            "parser_model",
        ):
            value = getattr(
                backend_settings,
                field,
                None,
            )

            if value is not None:
                legacy_backend.setdefault(
                    field,
                    value,
                )

    normalized = (
        normalize_runtime_config(
            config_for_normalization
        )
    )

    agent_config = normalized[
        "agent_config"
    ]

    all_candidate_agents = (
        agent_config[
            "all_candidates"
        ]
    )

    env_config = dict(
        normalized["env_config"]
    )

    backend_map = (
        _resolve_backend_map(
            normalized,
            backend=backend,
            backend_settings=(
                backend_settings
            ),
            backends=backends,
        )
    )

    parser_config = normalized[
        "parser"
    ]

    speech_perceiver = (
        SpeechPerceiver(
            backend=resolve_backend(
                parser_config["backend"],
                backend_map,
            ),
            model_name=parser_config[
                "model"
            ],
        )
    )

    env_config[
        "log_save_path"
    ] = log_save_path

    env = WerewolfTextEnvV0(
        **env_config,
        speech_perceiver=(
            speech_perceiver
        ),
        random_seed=random_seed,
    )

    if env_config.get(
        "n_hunter",
        0,
    ) != 0:
        raise ValueError(
            "classic-7 ToM environment "
            "does not support Hunter."
        )

    if random_seed is not None:
        random.seed(
            random_seed
        )

    if roles is None:
        roles = (
            ["Werewolf"]
            * env_config["n_werewolf"]
            + ["Villager"]
            * env_config["n_villager"]
            + ["Seer"]
            * env_config["n_seer"]
            + ["Witch"]
            * env_config["n_witch"]
            + ["Guard"]
            * env_config["n_guard"]
        )

        random.shuffle(
            roles
        )
    else:
        roles = list(
            roles
        )

    must_include = agent_config.get(
        "must_include",
        [],
    )

    allow_cross_team_profiles = (
        agent_config.get(
            "allow_cross_team_profiles",
            False,
        )
    )

    if not isinstance(
        allow_cross_team_profiles,
        bool,
    ):
        raise TypeError(
            "agent_config.allow_cross_team_profiles "
            "must be boolean"
        )

    role2agent_list, agent_list = (
        assign_agents(
            all_candidate_agents,
            env_config,
            log_save_path,
            roles,
            must_include=(
                must_include
            ),
            backends=backend_map,
            allow_cross_team_profiles=(
                allow_cross_team_profiles
            ),
        )
    )

    return (
        env,
        agent_list,
        roles,
        role2agent_list,
    )


def _write_role_assignment(
    *,
    log_save_path,
    roles,
    role2agent_list,
) -> None:
    """Write game-audit roles outside the subjective ToM JSONL."""

    records = [
        {
            "id": index + 1,
            "role": role,
            "model": (
                role2agent_list[index]
            ),
        }
        for index, role in enumerate(
            roles
        )
    ]

    output_file = os.path.join(
        log_save_path,
        "roles_model_assignment.json",
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            records,
            json_file,
            ensure_ascii=False,
            indent=4,
        )


def main_cli(args):
    """CLI implementation."""

    if args.log_save_path is None:
        run_name = time.strftime(
            "%Y%m%d_%H%M%S"
        )

        args.log_save_path = os.path.join(
            "logs",
            run_name,
        )

    os.makedirs(
        args.log_save_path,
        exist_ok=True,
    )

    with open(
        args.config,
        "r",
        encoding="utf-8",
    ) as config_file:
        parsed_yaml = yaml.safe_load(
            config_file
        )

    if not isinstance(
        parsed_yaml,
        dict,
    ):
        raise ValueError(
            "runtime config must be a mapping"
        )

    config_save_path = os.path.join(
        args.log_save_path,
        "config.yaml",
    )

    with open(
        config_save_path,
        "w",
        encoding="utf-8",
    ) as config_file:
        yaml.dump(
            parsed_yaml,
            config_file,
            allow_unicode=True,
            sort_keys=False,
        )

    normalized = (
        normalize_runtime_config(
            deepcopy(parsed_yaml)
        )
    )

    backend_map = load_named_backends(
        normalized,
        env_file=".env",
    )

    (
        env,
        agent_list,
        roles,
        role2agent_list,
    ) = build_runtime(
        parsed_yaml,
        log_save_path=(
            args.log_save_path
        ),
        random_seed=getattr(
            args,
            "random_seed",
            None,
        ),
        backends=backend_map,
    )

    print(
        "New rollout: ",
        roles,
    )
    print()

    for role, profile_name in zip(
        roles,
        role2agent_list,
    ):
        print(
            role,
            "\t",
            profile_name,
        )

    if len(roles) != len(
        role2agent_list
    ):
        raise RuntimeError(
            "roles and role2agent_list "
            "must have equal lengths"
        )

    _write_role_assignment(
        log_save_path=(
            args.log_save_path
        ),
        roles=roles,
        role2agent_list=(
            role2agent_list
        ),
    )

    begin = time.time()
    result = eval(
        env,
        agent_list,
        roles,
    )

    print(
        time.time() - begin,
        result,
    )

    return result


def build_arg_parser() -> (
    argparse.ArgumentParser
):
    """Build the command-line parser."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help=(
            "path to the game runtime config"
        ),
    )

    parser.add_argument(
        "--log_save_path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
    )

    return parser


if __name__ == "__main__":
    arguments = (
        build_arg_parser().parse_args()
    )

    main_cli(
        arguments
    )
