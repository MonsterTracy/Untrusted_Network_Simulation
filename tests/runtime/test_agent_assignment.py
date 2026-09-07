"""Tests for random-runtime agent profile assignment."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import run_random


SHARED_PROFILE = {
    "profile_name": "shared_deepseek",
    "agent_type": "gpt",
    "backend": "deepseek",
    "model": "test-model",
    "model_params": {
        "temperature": 0.0,
    },
    "sample_ratio": 1.0,
}


FULL_ROLES = [
    "Werewolf",
    "Villager",
    "Seer",
    "Werewolf",
    "Witch",
    "Villager",
    "Villager",
]


ENV_CONFIG = {
    "n_player": 7,
    "n_role": 4,
    "n_werewolf": 2,
    "n_villager": 3,
    "n_seer": 1,
    "n_witch": 1,
    "n_guard": 0,
    "n_hunter": 0,
}


def patch_playing_agent(monkeypatch):
    class FakeAgent:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    monkeypatch.setattr(run_random, "GPTAgent", FakeAgent)


def test_shared_profile_can_control_both_teams(
    monkeypatch,
):
    patch_playing_agent(
        monkeypatch
    )

    (
        role_to_profile,
        agents,
    ) = run_random.assign_agents(
        candidate_profiles=[
            SHARED_PROFILE,
        ],
        env_config=ENV_CONFIG,
        log_save_path=None,
        assigined_roles=FULL_ROLES,
        must_include=[],
        backends={
            "deepseek": object(),
        },
        allow_cross_team_profiles=True,
    )

    assert role_to_profile == [
        "shared_deepseek",
    ] * 7

    assert len(agents) == 7
    assert {agent.backend_id for agent in agents} == {"deepseek"}

    assert {agent.model_name for agent in agents} == {"test-model"}


def test_default_mode_preserves_team_exclusivity(
    monkeypatch,
):
    patch_playing_agent(
        monkeypatch
    )

    with pytest.raises(
        ValueError,
        match=(
            "no eligible agent profiles "
            "for role: Villager"
        ),
    ):
        run_random.assign_agents(
            candidate_profiles=[
                SHARED_PROFILE,
            ],
            env_config=ENV_CONFIG,
            log_save_path=None,
            assigined_roles=[
                "Werewolf",
                "Villager",
            ],
            must_include=[],
            backends={
                "deepseek": object(),
            },
        )


def test_invalid_cross_team_setting_is_rejected(
    monkeypatch,
):
    patch_playing_agent(
        monkeypatch
    )

    with pytest.raises(
        TypeError,
        match=(
            "allow_cross_team_profiles "
            "must be boolean"
        ),
    ):
        run_random.assign_agents(
            candidate_profiles=[
                SHARED_PROFILE,
            ],
            env_config=ENV_CONFIG,
            log_save_path=None,
            assigined_roles=[
                "Werewolf",
            ],
            must_include=[],
            backends={
                "deepseek": object(),
            },
            allow_cross_team_profiles=1,
        )
