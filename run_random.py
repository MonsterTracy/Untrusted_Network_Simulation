"""Classic7 runtime assembly and canonical game execution (no standalone rollout CLI)."""

from __future__ import annotations

from contextlib import nullcontext
import os
import random
from copy import deepcopy
from typing import Any


from werewolf.agents.gpt_agent import GPTAgent
from werewolf.backends import (
    load_named_backends,
    resolve_backend,
)
from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
from werewolf.speech.speech_perceiver import SpeechPerceiver
from werewolf.runtime_config import normalize_runtime_config


PUBLIC_SPEECH_RUNTIME_PHASES = frozenset({"speech", "speech_pk"})


def _act_at_boundary(
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


def eval(env, agent_list, roles_, *, canonical_recorder, call_audit):
    """Execute one game exclusively through canonical evidence construction."""
    if canonical_recorder is None or call_audit is None:
        raise TypeError("canonical recorder and call audit are required")
    for agent in agent_list:
        agent.reset()
    observation = env.reset(roles=roles_)
    canonical_recorder.start(env, roles=roles_)
    done, step = False, 0
    while not done:
        actor = observation["current_act_idx"]
        is_speech = env.phase in PUBLIC_SPEECH_RUNTIME_PHASES
        try:
            handoff = canonical_recorder.before_agent_act(
                env, step_idx=step, acting_player_id=actor, delivered_observation=observation,
                speech_kind=env.phase if is_speech else None)
            if is_speech and handoff is None:
                raise TypeError("public speech requires the collected Speaker PRE Belief Handoff")
            with call_audit.action_context(acting_player_id=actor,
                    boundary_id=handoff.boundary_id if handoff else None, is_public_speech=is_speech):
                action = _act_at_boundary(agent_list[actor - 1], observation, handoff)
            canonical_recorder.after_agent_act(action)
            context = call_audit.speech_perception_context(
                event_id=f"event-{len(env.public_events):06d}",
                boundary_id=handoff.boundary_id, speaker_id=actor) if is_speech else nullcontext()
            with context:
                observation, _, done, info = env.step(action)
            canonical_recorder.after_env_step(env, observation_after=observation, terminal_after=done)
        except Exception as error:
            raise canonical_recorder.failure_from_exception(error) from error
        step += 1
    if not isinstance(info, dict) or info.get("Werewolf") not in (1, -1):
        raise RuntimeError("finished game has no recognized winner")
    winner = "Werewolf" if info["Werewolf"] == 1 else "Villager"
    canonical_recorder.finish(env, winner=winner)
    return f"{winner} win"


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

            if profile["agent_type"] != "gpt":
                raise ValueError("the canonical playing agent type is gpt")
            all_agent_profiles[profile_name] = {
                "backend": resolve_backend(profile["backend"], backends),
                "model_name": profile["model"], **model_params}

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

        agent = GPTAgent(log_file=log_file, **all_agent_profiles[profile_name])
        agent.backend_id = profile_backend_ids[profile_name]

        agent_list.append(
            agent
        )

    return (
        role2agent_list,
        agent_list,
    )


def build_runtime(
    parsed_yaml,
    log_save_path,
    roles=None,
    random_seed=None,
    backends=None,
):
    """Build the game environment and action agents."""

    normalized = normalize_runtime_config(parsed_yaml)

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

    backend_map = dict(backends) if backends is not None else load_named_backends(normalized, env_file=".env")

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
