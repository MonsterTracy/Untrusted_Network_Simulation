"""Small contract fixtures; large sealed inputs are opt-in only."""

from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import os

import pytest

from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.phase2_offline import (
    Phase2DataError, PostDayState, ResolvedOutcome,
    UnsupportedStateError, _Opportunity, build_candidate_rows,
    build_development_layer, build_value_tables, candidate_opportunities,
    classify_outcome, phase_context, post_day_states, r2_input,
    theta_ac, validate_q_row, _resolved_exiles,
)


P = PLAYER_IDS
W = frozenset(P[:2])


def event(kind, index, phase, day=1, *, speaker=None, votes=(), affected=()):
    return NS(event_type=kind, event_index=index,
              temporal_state=NS(phase=NS(value=phase), day=day),
              speaker=speaker, votes=tuple(NS(voter=v, target=t) for v, t in votes),
              affected_players=tuple(affected))


def ballot(targets):
    return tuple(zip(P, targets))


def pre(events, alive=P, actor=P[0]):
    return NS(public_temporal_state=events[-1].temporal_state,
              public_event_history=NS(events=tuple(events)),
              alive_observer_ids=tuple(alive), current_speaker=actor,
              boundary_id="boundary-1", prefix_digest="digest-1")


def qrow(observer, weights):
    row = [0.0] * 7
    for target, mass in weights.items():
        row[P.index(target)] = mass
    return validate_q_row(observer, row)


def test_candidate_population_identity_and_label_runtime_panel():
    events = [event("phase_change", 0, "discussion"),
              event("turn_start", 1, "discussion", speaker=P[0])]
    reports = [NS(observation_id=f"obs-{i}", game_id="g", boundary_id="boundary-1",
                  prefix_digest="digest-1", observer_id=i, status=NS(value="success"),
                  label_observed=True, observer_alive=True, suspicion_support=(P[2],))
               for i in P[2:]]
    prefix = pre(events)
    prefix.belief_observation_links = tuple(NS(observation_id=r.observation_id,
                                               observer_id=r.observer_id) for r in reports)
    game = NS(game_id="g", belief_observations=tuple(reports),
              authoritative_pre_prefixes=(prefix,))
    roles = {p: "Werewolf" if p in W else "Villager" for p in P}
    found = candidate_opportunities((game,), {"g": roles}, {"g": 0})
    assert len(found) == 1
    assert found[0].candidates == P[2:]
    assert set(found[0].supports) == set(P[2:])
    broken = reports[0]
    broken.label_observed = False
    with pytest.raises(Phase2DataError, match="O_label"):
        candidate_opportunities((game,), {"g": roles}, {"g": 0})


def test_pk_tie_set_and_public_position():
    votes = ballot((P[2], P[3], P[3], P[2], None, None, None))
    events = [event("phase_change", 0, "vote"),
              event("vote_result", 1, "vote", votes=votes),
              event("exile_result", 2, "vote"),
              event("phase_change", 3, "pk_discussion"),
              event("turn_start", 4, "pk_discussion", speaker=P[3])]
    assert phase_context(pre(events, actor=P[3])) == ("speech_pk", (P[2], P[3]), 1)
    events.extend((event("public_speech", 5, "pk_discussion", speaker=P[3]),
                   event("turn_start", 6, "pk_discussion", speaker=P[2])))
    assert phase_context(pre(events, actor=P[2]))[2] == 0


def test_no_candidate_pk_pre_is_recorded_and_skipped():
    votes = ballot((P[1], P[0], P[0], P[1], None, None, None))
    events = [event("phase_change", 0, "vote"),
              event("vote_result", 1, "vote", votes=votes),
              event("exile_result", 2, "vote"),
              event("phase_change", 3, "pk_discussion"),
              event("turn_start", 4, "pk_discussion", speaker=P[0])]
    prefix = pre(events)
    game = NS(game_id="g", belief_observations=(),
              authoritative_pre_prefixes=(prefix,))
    roles = {p: "Werewolf" if p in W else "Villager" for p in P}
    diagnostics = Counter()
    assert candidate_opportunities((game,), {"g": roles}, {"g": 0},
                                   diagnostics=diagnostics) == ()
    assert diagnostics["skipped_no_candidate"] == 1


def test_theta_tie_positive_and_target_specific_denominators():
    supports = {i: frozenset() for i in P[2:]}
    supports[P[4]] = frozenset((P[2], P[3]))
    assert theta_ac(P, W, P, supports, P[2])
    assert theta_ac(P, W, P, supports, P[3])
    assert not theta_ac(P, W, P, supports, P[4])
    with pytest.raises(Phase2DataError, match="empty"):
        theta_ac(P[:3], W, P[:3], {P[2]: frozenset()}, P[2])


def test_theta_uses_normalized_breadth_when_raw_k_favors_wolf():
    supports = {i: frozenset((P[0],) if i == P[2] else (P[0], P[2]))
                for i in P[2:]}
    k_candidate = sum(P[2] in supports[i] for i in P[3:])
    k_wolf = sum(P[0] in supports[i] for i in P[2:])
    assert (k_candidate, k_wolf) == (4, 5)
    assert k_candidate / 4 == k_wolf / 5 == 1
    assert theta_ac(P, W, P, supports, P[2])


def test_q_contract():
    assert qrow(P[0], {P[1]: 1.0})[0] == 0
    with pytest.raises(Phase2DataError, match="simplex"):
        validate_q_row(P[0], [0.1] + [0.15] * 6)
    with pytest.raises(Phase2DataError, match="simplex"):
        validate_q_row(P[0], [0] + [0.1] * 6)


def test_r2_raw_mass_competitor_and_population_std():
    alive = P[:4]
    q = {
        P[2]: qrow(P[2], {P[0]: .5, P[1]: .3, P[3]: .2}),
        P[3]: qrow(P[3], {P[0]: .4, P[1]: .3, P[2]: .1, P[4]: .2}),
    }
    z = r2_input(alive, W, alive, P[2], q, phase="speech", remaining=3)
    assert z.mu == pytest.approx(.1)
    assert z.delta == pytest.approx(.1 - .45)  # Acting wolf is a competitor.
    assert z.sigma == 0  # A single report is not consensus.
    assert z.audience_size == 1
    assert z.competition_size == 4


def test_entropy_single_target_and_zero_competition_mass():
    alive = P[:4]
    q = {P[2]: qrow(P[2], {P[0]: .4, P[1]: .4, P[3]: .2}),
         P[3]: qrow(P[3], {P[0]: .4, P[1]: .4, P[2]: .2})}
    z = r2_input(alive, W, (P[2], P[3]), P[2], q,
                 phase="speech_pk", remaining=1)
    assert z.h == 0  # Observer P3 compares only P2.
    assert z.mu == pytest.approx(.2)
    zero = {P[2]: qrow(P[2], {P[0]: 1.0}),
            P[3]: qrow(P[3], {P[0]: 1.0})}
    with pytest.raises(Phase2DataError, match="zero Q mass"):
        r2_input(alive, W, (P[2], P[3]), P[2], zero,
                 phase="speech_pk", remaining=1)


def test_entropy_conditional_renormalization_and_zero_log_zero():
    alive = P[:5]
    competition = P[2:5]
    q = {P[2]: qrow(P[2], {P[3]: .5, P[4]: .5}),
         P[3]: qrow(P[3], {P[2]: .1, P[4]: .3, P[5]: .6}),
         P[4]: qrow(P[4], {P[0]: .8, P[2]: .2})}
    z = r2_input(alive, W, competition, P[2], q,
                 phase="speech_pk", remaining=2)
    assert z.mu == pytest.approx(.15)  # Raw .1/.2 masses, not conditioned masses.
    assert z.h == pytest.approx(.4056390622295664)
    assert 0 <= z.h <= 1


def test_normal_and_pk_day_resolution():
    ordinary_votes = ballot((P[2], P[2], None, P[2], P[2], P[2], P[2]))
    ordinary = [event("vote_result", 0, "vote", votes=ordinary_votes),
                event("exile_result", 1, "vote", affected=(P[2],))]
    assert _resolved_exiles(ordinary)[1] is ordinary[1]
    tie = [event("vote_result", 0, "vote",
                 votes=ballot((P[2], P[3], P[3], P[2], None, None, None))),
           event("exile_result", 1, "vote"),
           event("vote_result", 2, "pk_vote", votes=ordinary_votes),
           event("exile_result", 3, "pk_vote", affected=(P[2],))]
    assert _resolved_exiles(tie)[1] is tie[3]
    tie[2] = event("vote_result", 2, "pk_vote",
                   votes=ballot((P[2], P[3], P[3], P[2], None, None, None)))
    tie[3] = event("exile_result", 3, "pk_vote")
    assert _resolved_exiles(tie)[1].affected_players == ()
    tie[2] = event("vote_result", 2, "pk_vote", votes=ballot((None,) * 7))
    assert _resolved_exiles(tie)[1].affected_players == ()


def test_post_day_state_applies_final_exile_once():
    votes = ((P[0], P[2]), (P[1], P[3]), (P[2], P[3]),
             (P[3], P[2]), (P[6], None))
    pk_votes = ((P[0], P[2]), (P[1], P[2]), (P[2], None),
                (P[3], P[2]), (P[6], P[2]))
    events = [event("phase_change", 0, "night"),
              event("death_announcement", 1, "night", affected=(P[4], P[5])),
              event("phase_change", 2, "discussion"),
              event("phase_change", 3, "vote"),
              event("vote_result", 4, "vote", votes=votes),
              event("exile_result", 5, "vote"),
              event("phase_change", 6, "pk_discussion"),
              event("phase_change", 7, "pk_vote"),
              event("vote_result", 8, "pk_vote", votes=pk_votes),
              event("exile_result", 9, "pk_vote", affected=(P[2],))]
    roles = {p: "Werewolf" if p in W else "Villager" for p in P}
    game = NS(game_id="g", public_event_stream=NS(events=tuple(events)))
    states = post_day_states((game,), {"g": roles}, {"g": 2})
    assert len(states) == 1
    assert states[0].s0 == (2, 2)
    assert states[0].terminal and states[0].final_wolf_win
    assert states[0].exiled_player == P[2]
    assert states[0].final_exile_event_index == 9


def test_candidate_row_identity_and_evaluation_only_outcome():
    alive = P[:5]
    opportunity = _Opportunity("g", "b", "d", P[0], "speech", 1,
        alive, W, alive, P[2:5], 4,
        {i: frozenset(set(P[2:5]) - {i}) for i in P[2:5]}, 0)
    q = {("g", "b", P[2]): qrow(P[2], {P[0]: .5, P[3]: .5}),
         ("g", "b", P[3]): qrow(P[3], {P[0]: .5, P[2]: .5}),
         ("g", "b", P[4]): qrow(P[4], {P[0]: .5, P[2]: .5})}
    resolved = (PostDayState("g", 1, 2, 2, True, True, P[2], 9, 0),)
    rows = build_candidate_rows((opportunity,), q, resolved)
    assert len(rows) == 3
    assert {(r.game_id, r.boundary_id, r.acting_wolf, r.phase, r.candidate_j)
            for r in rows} == {("g", "b", P[0], "speech", P[2]),
                             ("g", "b", P[0], "speech", P[3]),
                             ("g", "b", P[0], "speech", P[4])}
    assert all(r.theta_ac for r in rows)
    assert all(r.s_pre == (2, 3) for r in rows)
    assert all(not hasattr(r.z, "s_pre") for r in rows)
    assert rows[0].resolved_outcome == ResolvedOutcome.TARGET_J_EXILED
    assert rows[1].resolved_outcome == ResolvedOutcome.OTHER_NONWOLF_EXILED
    assert not hasattr(rows[0].z, "resolved_outcome")
    with pytest.raises(Phase2DataError, match="S_pre/Y disagrees"):
        build_candidate_rows((opportunity,), q,
                             (replace(resolved[0], n_nw=3),))


@pytest.mark.parametrize("exile,expected", [
    (P[2], ResolvedOutcome.TARGET_J_EXILED),
    (P[3], ResolvedOutcome.OTHER_NONWOLF_EXILED),
    (P[0], ResolvedOutcome.ACTING_WOLF_EXILED),
    (P[1], ResolvedOutcome.TEAMMATE_WOLF_EXILED),
    (None, ResolvedOutcome.NO_EXILE),
])
def test_all_five_outcome_classes(exile, expected):
    assert classify_outcome(exile, acting_wolf=P[0], candidate_j=P[2], wolves=W) == expected


def state(game, fold, key, win, day=1):
    w, nw = key
    return PostDayState(game, day, w, nw, win, w == 0 or w >= nw, None, 2, fold)


def test_terminal_crossfit_full_and_unsupported_value():
    assignments = {f"g{i}": i for i in range(5)}
    visits = (state("g0", 0, (2, 4), True),
              state("g1", 1, (2, 4), False),
              state("g2", 2, (2, 4), True),
              state("g3", 3, (2, 4), False),
              state("g4", 4, (2, 4), True))
    tables = build_value_tables(visits, assignments)
    cf = tables.development("g0")
    assert cf.lookup((2, 4)).value == .5
    assert cf.lookup((2, 4)).state_visit_count == 4
    assert cf.lookup((2, 4)).unique_game_count == 4
    assert cf.lookup((2, 4)).training_game_count == 4
    assert tables.v_ref_pub_full.lookup((2, 4)).value == .6
    assert tables.v_ref_pub_full.lookup((2, 4)).training_game_count == 5
    assert tables.development_loss("g0", (2, 4), ResolvedOutcome.NO_EXILE).loss == .5
    assert tables.deployment_loss((2, 4), ResolvedOutcome.NO_EXILE).loss == .4
    assert cf.lookup((0, 4)).value == 0
    assert cf.lookup((2, 2)).value == 1
    assert cf.lookup((2, 2)).source == "rule"
    with pytest.raises(UnsupportedStateError):
        cf.lookup((1, 4))


def test_crossfit_excludes_all_games_and_days_in_held_out_fold():
    assignments = {"g0": 0, "g0b": 0, "g1": 1, "g2": 2}
    visits = (state("g0", 0, (2, 4), True, day=1),
              state("g0", 0, (2, 4), True, day=2),
              state("g0b", 0, (2, 4), False),
              state("g1", 1, (2, 4), True),
              state("g2", 2, (2, 4), False))
    tables = build_value_tables(visits, assignments)
    cf = tables.development("g0").lookup((2, 4))
    assert tables.development("g0b").lookup((2, 4)) == cf
    assert (cf.excluded_fold, cf.training_game_count,
            cf.state_visit_count, cf.unique_game_count, cf.wins, cf.value) == (
                0, 2, 2, 2, 1, .5)
    full = tables.v_ref_pub_full.lookup((2, 4))
    assert (full.state_visit_count, full.unique_game_count, full.wins, full.value) == (5, 4, 3, .6)


def test_reference_table_provenance_rejects_full_wrong_fold_and_entry():
    assignments = {f"g{i}": i for i in range(5)}
    visits = tuple(state(f"g{i}", i, (2, 4), i % 2 == 0) for i in range(5))
    tables = build_value_tables(visits, assignments)
    target = "g0"
    outcome = ResolvedOutcome.NO_EXILE
    assert tables.development_loss(target, (2, 4), outcome).value.excluded_fold == 0
    wrong_full = replace(tables, v_ref_pub_cf={**tables.v_ref_pub_cf,
                                               0: tables.v_ref_pub_full})
    with pytest.raises(Phase2DataError, match="fold provenance"):
        wrong_full.development_loss(target, (2, 4), outcome)
    wrong_fold = replace(tables, v_ref_pub_cf={**tables.v_ref_pub_cf,
                                               0: tables.v_ref_pub_cf[1]})
    with pytest.raises(Phase2DataError, match="fold provenance"):
        wrong_fold.development_loss(target, (2, 4), outcome)
    table = tables.v_ref_pub_cf[0]
    wrong_entry = replace(table, cells={**table.cells,
        (2, 4): replace(table.cells[(2, 4)], excluded_fold=1)})
    with pytest.raises(Phase2DataError, match="entry provenance"):
        replace(tables, v_ref_pub_cf={**tables.v_ref_pub_cf,
                                      0: wrong_entry}).development_loss(target, (2, 4), outcome)
    with pytest.raises(Phase2DataError, match="full V_ref_pub table provenance"):
        replace(tables, v_ref_pub_full=table).deployment_loss((2, 4), outcome)


def test_reference_loss_three_team_transitions():
    visits = (state("g0", 0, (2, 4), True), state("g1", 1, (2, 4), False),
              state("g2", 2, (1, 5), False), state("g3", 3, (1, 5), True),
              state("g4", 4, (2, 5), True), state("g5", 0, (2, 5), False))
    tables = build_value_tables(visits, {**{f"g{i}": i for i in range(5)},
                                         "g5": 0})
    losses = {y: tables.deployment_loss((2, 5), y)
              for y in ResolvedOutcome}
    assert losses[ResolvedOutcome.TARGET_J_EXILED].s_post == (2, 4)
    assert losses[ResolvedOutcome.TARGET_J_EXILED].s_pre == (2, 5)
    assert losses[ResolvedOutcome.OTHER_NONWOLF_EXILED].loss == losses[ResolvedOutcome.TARGET_J_EXILED].loss
    assert losses[ResolvedOutcome.ACTING_WOLF_EXILED].s_post == (1, 5)
    assert losses[ResolvedOutcome.TEAMMATE_WOLF_EXILED].s_post == (1, 5)
    assert losses[ResolvedOutcome.NO_EXILE].s_post == (2, 5)


@pytest.mark.skipif(not (os.environ.get("PHASE2_PUBLICATION_ROOT") and
                         os.environ.get("PHASE2_OOF_ROOT")),
                    reason="sealed development inputs are not local test fixtures")
def test_sealed_development_population_regression():
    layer = build_development_layer(Path(os.environ["PHASE2_PUBLICATION_ROOT"]),
                                    Path(os.environ["PHASE2_OOF_ROOT"]))
    assert (layer.pre_count, len(layer.candidates), len(layer.post_days)) == (5761, 21638, 3317)
    assert layer.skipped_no_candidate_pk_pre_count == 74
    assert len({s.game_id for s in layer.post_days}) == 1500
    assert Counter(row.phase for row in layer.candidates)["speech_pk"] > 0
