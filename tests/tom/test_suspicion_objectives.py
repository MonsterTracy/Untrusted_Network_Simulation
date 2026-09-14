import pytest
import torch

from scripts.suspicion_objectives import wolf_suspicion_mass

P = tuple(f"player{i}" for i in range(1, 8))


def uniform():
    b = torch.full((7, 7), 1 / 6, dtype=torch.float64)
    b.fill_diagonal_(0)
    return b


def score(b, **kwargs):
    args = dict(alive_players=P, alive_wolves=(P[0], P[4]), self_player=P[0])
    args.update(kwargs)
    return wolf_suspicion_mass(b, **args)


def test_all_alive_conditional_equals_raw_and_diagnostics_sum():
    b = uniform()
    before = b.clone()
    result = score(b)
    assert result.alive_conditional_team_mass == pytest.approx(result.raw_team_mass)
    assert result.raw_team_mass == pytest.approx(1 / 3)
    assert result.raw_self_mass == pytest.approx(1 / 6)
    assert result.alive_conditional_self_mass == pytest.approx(result.raw_self_mass)
    assert sum(v for _, v in result.per_alive_wolf_raw_mass) == pytest.approx(result.raw_team_mass)
    assert sum(v for _, v in result.per_alive_wolf_conditional_mass) == pytest.approx(result.alive_conditional_team_mass)
    assert torch.equal(b, before)


def test_dead_mass_does_not_artificially_reduce_conditional_score():
    b = uniform()
    shifted = b.clone()
    alive = P[:-1]
    for i in range(6):
        shifted[i, :6] *= .01
        shifted[i, 6] = 1 - shifted[i, :6].sum()
    before = shifted.clone()
    first, second = score(b, alive_players=alive), score(shifted, alive_players=alive)
    assert second.raw_team_mass < first.raw_team_mass / 10
    assert second.alive_conditional_team_mass == pytest.approx(first.alive_conditional_team_mass)
    assert second.alive_conditional_self_mass == pytest.approx(first.alive_conditional_self_mass)
    assert torch.equal(shifted, before)


def test_denominator_excludes_observer_and_empty_sets_fail():
    b = uniform()
    result = score(b, alive_players=(P[0], P[1]), alive_wolves=(P[0],))
    assert result.alive_conditional_team_mass == pytest.approx(1)
    for args, message in [(dict(alive_wolves=()), "empty W_t"),
                          (dict(alive_players=(P[0], P[4])), "empty O_t")]:
        with pytest.raises(ValueError, match=message):
            score(b, **args)
    b[1] = 0
    b[1, 6] = 1
    with pytest.raises(ValueError, match="zero alive"):
        score(b, alive_players=(P[0], P[1]), alive_wolves=(P[0],))


@pytest.mark.parametrize("mutation", ["self", "nan", "negative", "not_normalized", "shape"])
def test_invalid_probability_contract(mutation):
    b = uniform()
    if mutation == "self":
        b[0, 0], b[0, 1] = b[0, 1].item(), 0
    elif mutation == "nan":
        b[0, 1] = float("nan")
    elif mutation == "negative":
        b[0, 1] = -1
    elif mutation == "not_normalized":
        b *= 2
    else:
        b = b[:1]
    with pytest.raises(ValueError):
        score(b)


def test_normalize_each_observer_before_averaging():
    b = uniform()
    # A={P1,P2,P3}, W={P1}. Z_2=.2 with conditional wolf mass=.5;
    # Z_3=.8 with conditional wolf mass=.25. Mean must be .375, not .3.
    b[1] = torch.tensor([.1, 0, .1, .8, 0, 0, 0], dtype=b.dtype)
    b[2] = torch.tensor([.2, .6, 0, .2, 0, 0, 0], dtype=b.dtype)
    r = score(b, alive_players=P[:3], alive_wolves=(P[0],))
    assert r.alive_conditional_team_mass == pytest.approx(.375)
    assert r.raw_team_mass == pytest.approx(.15)


@pytest.mark.parametrize("kwargs", [dict(alive_wolves=(P[0], P[0])),
    dict(alive_players=P[:-1], alive_wolves=(P[0], P[6])),
    dict(self_player=P[1]), dict(alive_players=("player8",))])
def test_invalid_population_inputs(kwargs):
    with pytest.raises(ValueError):
        score(uniform(), **kwargs)
