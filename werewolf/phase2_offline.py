"""Offline Phase-2 candidate evidence and publication-policy reference values.

This module reads sealed development evidence. It does not call a predictor,
train a mapper, or implement an intervention policy.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path

from werewolf.artifact_io import verify_artifact
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.development_publication import open_publication, open_role_sidecar


PUBLICATION_DIGEST = "396ea0fa3f3f03dd1dfce8b25c0162ca489b992321d0c9e6f772eb2b0c4a2acd"
ROLE_SIDECAR_DIGEST = "276b6ee182bfb07639ae3ab99a5c8037b52ffb99b37f5c60f7f26a229178a2da"
FOLD_MANIFEST_DIGEST = "754fe0e517aab9e9237f0cb5097bde0daf8a43a09518b9691cfca67724539e69"
OOF_SEAL_DIGEST = "66bd044bd072a8673d4761ad043d0af3a208ae55b809ebffded2228b0371f925"
EXPECTED_GAMES = 1500
EXPECTED_PRE = 5761
EXPECTED_CANDIDATES = 21638
EXPECTED_POST_DAYS = 3317
EXPECTED_SKIPPED_NO_CANDIDATE_PK_PRE = 74


class Phase2DataError(ValueError):
    """An offline record violates the frozen Phase-2 data contract."""


class UnsupportedStateError(Phase2DataError):
    """A nonterminal state has no reference observations."""


class ResolvedOutcome(str, Enum):
    TARGET_J_EXILED = "target_j_exiled"
    OTHER_NONWOLF_EXILED = "other_nonwolf_exiled"
    ACTING_WOLF_EXILED = "acting_wolf_exiled"
    TEAMMATE_WOLF_EXILED = "teammate_wolf_exiled"
    NO_EXILE = "no_exile"


@dataclass(frozen=True)
class R2Input:
    mu: float
    delta: float
    sigma: float
    h: float
    phase: str
    audience_size: int
    competition_size: int
    remaining_speakers_before_vote: int


@dataclass(frozen=True)
class CandidateRow:
    game_id: str
    boundary_id: str
    acting_wolf: str
    phase: str
    candidate_j: str
    fold: int
    prefix_digest: str
    s_pre: tuple[int, int]  # Loss/evaluation metadata, never part of z.
    theta_ac: bool
    z: R2Input
    resolved_outcome: ResolvedOutcome  # Evaluation-only; never part of z.


@dataclass(frozen=True)
class PostDayState:
    game_id: str
    resolved_day: int
    n_w: int
    n_nw: int
    final_wolf_win: bool
    terminal: bool
    exiled_player: str | None
    final_exile_event_index: int
    fold: int

    @property
    def s0(self) -> tuple[int, int]:
        return self.n_w, self.n_nw

    @property
    def s_post(self) -> tuple[int, int]:
        return self.s0


@dataclass(frozen=True)
class ValueEstimate:
    value: float
    source: str  # rule or empirical
    excluded_fold: int | None
    training_game_count: int
    state_visit_count: int | None
    unique_game_count: int | None
    wins: int | None
    empirical_value: float | None


@dataclass(frozen=True)
class ValueTable:
    excluded_fold: int | None
    training_game_count: int
    cells: dict[tuple[int, int], ValueEstimate]

    def lookup(self, state: tuple[int, int]) -> ValueEstimate:
        n_w, n_nw = _counts(state)
        if n_w == 0 or n_w >= n_nw:
            return ValueEstimate(float(n_w > 0), "rule", self.excluded_fold,
                                 self.training_game_count, None, None, None, None)
        try:
            return self.cells[(n_w, n_nw)]
        except KeyError as error:
            raise UnsupportedStateError(f"unobserved nonterminal S0 {(n_w, n_nw)}") from error


@dataclass(frozen=True)
class ReferenceTables:
    fold_of_game: dict[str, int]
    v_ref_pub_cf: dict[int, ValueTable]
    v_ref_pub_full: ValueTable

    def development(self, game_id: str) -> ValueTable:
        try:
            fold = self.fold_of_game[game_id]
            table = self.v_ref_pub_cf[fold]
        except KeyError as error:
            raise Phase2DataError("unknown development game") from error
        if (table.excluded_fold != fold
                or table.training_game_count != sum(f != fold for f in self.fold_of_game.values())):
            raise Phase2DataError("development V_ref_pub fold provenance mismatch")
        return table

    def development_loss(self, game_id: str, s_pre: tuple[int, int],
                         outcome: ResolvedOutcome) -> ReferenceLoss:
        fold = self.fold_of_game.get(game_id)
        table = self.development(game_id)
        result = _reference_outcome_loss(s_pre, outcome, table)
        if (result.value.excluded_fold != fold
                or result.value.training_game_count != table.training_game_count):
            raise Phase2DataError("development V_ref_pub entry provenance mismatch")
        return result

    def deployment_loss(self, s_pre: tuple[int, int],
                        outcome: ResolvedOutcome) -> ReferenceLoss:
        table = self.v_ref_pub_full
        if (table.excluded_fold is not None
                or table.training_game_count != len(self.fold_of_game)):
            raise Phase2DataError("full V_ref_pub table provenance mismatch")
        result = _reference_outcome_loss(s_pre, outcome, table)
        if (result.value.excluded_fold is not None
                or result.value.training_game_count != table.training_game_count):
            raise Phase2DataError("full V_ref_pub entry provenance mismatch")
        return result


@dataclass(frozen=True)
class ReferenceLoss:
    outcome: ResolvedOutcome
    s_pre: tuple[int, int]
    s_post: tuple[int, int]
    value: ValueEstimate
    loss: float


@dataclass(frozen=True)
class OfflinePhase2:
    candidates: tuple[CandidateRow, ...]
    post_days: tuple[PostDayState, ...]
    values: ReferenceTables
    pre_count: int
    skipped_no_candidate_pk_pre_count: int


@dataclass(frozen=True)
class _Opportunity:
    game_id: str
    boundary_id: str
    prefix_digest: str
    acting_wolf: str
    phase: str
    day: int
    alive: tuple[str, ...]
    wolves: frozenset[str]
    competition: tuple[str, ...]
    candidates: tuple[str, ...]
    remaining: int
    supports: dict[str, frozenset[str]]  # Label-only; not passed to R2.
    fold: int


def _counts(state: tuple[int, int]) -> tuple[int, int]:
    if (not isinstance(state, tuple) or len(state) != 2
            or any(type(n) is not int or n < 0 for n in state)
            or state[0] > 2 or state[1] > 5):
        raise Phase2DataError("invalid Classic7 S0 counts")
    return state


def fold_assignment(fold_manifest) -> dict[str, int]:
    if fold_manifest.fold_count != 5 or len(fold_manifest.folds) != 5:
        raise Phase2DataError("expected five development folds")
    result = {}
    for fold in fold_manifest.folds:
        for game in fold.game_ids:
            if game in result:
                raise Phase2DataError("game appears in multiple folds")
            result[game] = fold.fold_index
    return result


def _vote_winner(ballot) -> tuple[str, ...]:
    counts = Counter(v.target for v in ballot.votes if v.target is not None)
    if not counts:
        return ()
    top = max(counts.values())
    return tuple(p for p in PLAYER_IDS if counts[p] == top)


def phase_context(pre) -> tuple[str, tuple[str, ...], int]:
    """Use only the current PRE to reconstruct C and remaining queue turns."""
    state = pre.public_temporal_state
    if state.phase.value not in ("discussion", "pk_discussion"):
        raise Phase2DataError("not a speech PRE")
    phase = "speech" if state.phase.value == "discussion" else "speech_pk"
    alive = tuple(pre.alive_observer_ids)
    events = pre.public_event_history.events
    starts = [i for i, e in enumerate(events) if e.event_type == "phase_change"]
    if not starts:
        raise Phase2DataError("missing public phase change")
    start = starts[-1]
    current = events[start + 1:]
    if any(e.event_type not in ("turn_start", "public_speech") for e in current):
        raise Phase2DataError("unexpected public event inside speech phase")
    observed = tuple(e.speaker for e in current if e.event_type == "turn_start")
    if not observed or observed[-1] != pre.current_speaker:
        raise Phase2DataError("current speaker differs from public queue")
    if phase == "speech":
        competition = alive
        phase_speaker_members = alive
    else:
        if len(starts) < 2:
            raise Phase2DataError("PK lacks preceding public vote")
        previous = events[starts[-2]:start]
        if (tuple(e.event_type for e in previous) !=
                ("phase_change", "vote_result", "exile_result")
                or any(e.temporal_state.day != state.day or
                       e.temporal_state.phase.value != "vote" for e in previous)
                or previous[2].affected_players):
            raise Phase2DataError("PK lacks a same-day normal-vote tie")
        ballot = previous[1]
        if {v.voter for v in ballot.votes} != set(alive):
            raise Phase2DataError("PK ballot does not cover alive voters")
        if any(v.target is not None and
               (v.target not in alive or v.target == v.voter) for v in ballot.votes):
            raise Phase2DataError("PK ballot contains illegal target")
        competition = _vote_winner(ballot)
        if len(competition) < 2:
            raise Phase2DataError("PK ballot has no positive maximum tie")
        phase_speaker_members = competition  # Current PK rule: tied targets speak.
    if observed[0] not in phase_speaker_members:
        raise Phase2DataError("first speaker is outside phase speaker queue")
    start_seat = phase_speaker_members.index(observed[0])
    phase_speaker_queue = (phase_speaker_members[start_seat:]
                           + phase_speaker_members[:start_seat])
    if observed != phase_speaker_queue[:len(observed)]:
        raise Phase2DataError("public turns violate cyclic queue order")
    return phase, competition, len(phase_speaker_queue) - len(observed)


def theta_ac(alive: tuple[str, ...], wolves: frozenset[str],
             competition: tuple[str, ...], supports: dict[str, frozenset[str]],
             candidate: str) -> bool:
    nonwolves = set(alive) - wolves
    if set(supports) != nonwolves or candidate not in nonwolves or candidate not in competition:
        raise Phase2DataError("label panel/candidate contract mismatch")
    breadth = {}
    for target in competition:
        observers = nonwolves - {target}
        if not observers:
            raise Phase2DataError(f"empty label observer panel for {target}")
        count = sum(target in supports[i] for i in observers)
        breadth[target] = count / len(observers)
    return breadth[candidate] > 0 and all(
        breadth[candidate] >= breadth[k] for k in competition if k != candidate)


def validate_q_row(observer: str, row) -> tuple[float, ...]:
    if observer not in PLAYER_IDS or not isinstance(row, (list, tuple)) or len(row) != 7:
        raise Phase2DataError("Q must contain seven absolute-seat probabilities")
    if any(type(x) not in (float, int) or not math.isfinite(x) or x < 0 for x in row):
        raise Phase2DataError("Q contains invalid probability")
    values = tuple(float(x) for x in row)
    if values[PLAYER_IDS.index(observer)] != 0.0 or abs(math.fsum(values) - 1.0) > 1e-6:
        raise Phase2DataError("Q violates frozen non-self simplex")
    return values


def r2_input(alive: tuple[str, ...], wolves: frozenset[str],
             competition: tuple[str, ...], candidate: str,
             q_by_observer: dict[str, tuple[float, ...]], *, phase: str,
             remaining: int) -> R2Input:
    """Compute runtime-compatible evidence; no canonical reports enter here."""
    nonwolves = set(alive) - wolves
    if (candidate not in nonwolves or candidate not in competition or len(competition) < 2
            or not set(competition) <= set(alive) or set(q_by_observer) != nonwolves
            or phase not in ("speech", "speech_pk")
            or type(remaining) is not int or not 0 <= remaining < len(PLAYER_IDS)):
        raise Phase2DataError("invalid R2 population or context")
    q = {i: validate_q_row(i, row) for i, row in q_by_observer.items()}
    panel = {k: tuple(i for i in PLAYER_IDS if i in nonwolves and i != k)
             for k in competition}
    if any(not observers for observers in panel.values()):
        raise Phase2DataError("empty R2 observer panel")
    mu = {k: math.fsum(q[i][PLAYER_IDS.index(k)] for i in observers) / len(observers)
          for k, observers in panel.items()}
    values = [q[i][PLAYER_IDS.index(candidate)] for i in panel[candidate]]
    center = mu[candidate]
    sigma = math.sqrt(math.fsum((value - center) ** 2 for value in values) / len(values))
    entropy = []
    for observer in panel[candidate]:
        targets = tuple(k for k in competition if k != observer)
        if not targets:
            raise Phase2DataError("empty entropy competition mask")
        masses = [q[observer][PLAYER_IDS.index(k)] for k in targets]
        total = math.fsum(masses)
        if total <= 0:
            raise Phase2DataError("zero Q mass on current competition set")
        if len(targets) == 1:
            entropy.append(0.0)
            continue
        conditional = [mass / total for mass in masses]
        value = -math.fsum(p * math.log(p) for p in conditional if p > 0) / math.log(len(targets))
        if not -1e-12 <= value <= 1 + 1e-12:
            raise Phase2DataError("normalized entropy outside [0,1]")
        entropy.append(min(1.0, max(0.0, value)))
    return R2Input(center, center - max(mu[k] for k in competition if k != candidate),
                   sigma, math.fsum(entropy) / len(entropy), phase,
                   len(panel[candidate]), len(competition), remaining)


def _resolved_exiles(events) -> dict[int, object]:
    """Return exactly the final vote/PK exile event for every completed day."""
    by_day = defaultdict(lambda: {"vote": [], "pk_vote": [], "exile": []})
    for event in events:
        if event.event_type == "vote_result":
            if event.temporal_state.phase.value not in ("vote", "pk_vote"):
                raise Phase2DataError("vote result outside vote phase")
            by_day[event.temporal_state.day][event.temporal_state.phase.value].append(event)
        elif event.event_type == "exile_result":
            by_day[event.temporal_state.day]["exile"].append(event)
    result = {}
    for day, phases in by_day.items():
        normal, pk, exiles = phases["vote"], phases["pk_vote"], phases["exile"]
        if len(normal) != 1 or len(pk) > 1 or len(exiles) != 1 + len(pk):
            raise Phase2DataError("incomplete or duplicated day resolution")
        normal_winners = _vote_winner(normal[0])
        if len(normal_winners) > 1:
            if (len(pk) != 1 or len(exiles) != 2
                    or exiles[0].temporal_state.phase.value != "vote"
                    or exiles[0].affected_players
                    or exiles[1].temporal_state.phase.value != "pk_vote"):
                raise Phase2DataError("ordinary tie lacks valid PK resolution")
            final_winners = _vote_winner(pk[0])
            expected = final_winners[0] if len(final_winners) == 1 else None
            final = exiles[1]
        else:
            if pk or exiles[0].temporal_state.phase.value != "vote":
                raise Phase2DataError("unexpected PK after resolved ordinary vote")
            expected = normal_winners[0] if normal_winners else None
            final = exiles[0]
        if tuple(final.affected_players) != (() if expected is None else (expected,)):
            raise Phase2DataError("final exile disagrees with final ballot")
        result[day] = final
    return result


def post_day_states(games, role_maps: dict[str, dict[str, str]],
                    fold_of_game: dict[str, int]) -> tuple[PostDayState, ...]:
    states = []
    for game in games:
        roles = role_maps[game.game_id]
        if set(roles) != set(PLAYER_IDS) or sum(r == "Werewolf" for r in roles.values()) != 2:
            raise Phase2DataError("invalid Classic7 role assignment")
        alive = set(PLAYER_IDS)
        events = game.public_event_stream.events
        final_by_day = _resolved_exiles(events)
        seen_days = set()
        local = []
        for index, event in enumerate(events):
            if event.event_index != index:
                raise Phase2DataError("noncanonical public event index")
            if event.event_type in ("death_announcement", "exile_result"):
                removed = set(event.affected_players)
                if len(removed) != len(event.affected_players) or not removed <= alive:
                    raise Phase2DataError("repeated or invalid public elimination")
                alive -= removed
            if final_by_day.get(event.temporal_state.day) is event:
                day = event.temporal_state.day
                if day in seen_days or len(event.affected_players) > 1:
                    raise Phase2DataError("day has multiple or invalid final resolutions")
                seen_days.add(day)
                n_w = sum(roles[p] == "Werewolf" for p in alive)
                n_nw = len(alive) - n_w
                terminal = n_w == 0 or n_w >= n_nw
                if terminal:
                    if index != len(events) - 1:
                        raise Phase2DataError("terminal day has later public events")
                else:
                    if index + 1 >= len(events):
                        raise Phase2DataError("nonterminal day lacks continuation")
                    next_event = events[index + 1]
                    if (next_event.event_type != "phase_change"
                            or next_event.temporal_state.phase.value != "night"
                            or next_event.temporal_state.day != day):
                        raise Phase2DataError("nonterminal day must enter same-day night")
                local.append(PostDayState(game.game_id, day, n_w, n_nw, False,
                                          terminal, event.affected_players[0] if event.affected_players else None,
                                          index, fold_of_game[game.game_id]))
        if set(final_by_day) != seen_days or not local:
            raise Phase2DataError("game has missing completed day")
        last_w = sum(roles[p] == "Werewolf" for p in alive)
        last_nw = len(alive) - last_w
        if last_w > 0 and last_w < last_nw:
            raise Phase2DataError("publication game has no terminal winner")
        if events[-1].event_type not in ("exile_result", "death_announcement"):
            raise Phase2DataError("game ends at unexpected public event")
        winner = last_w > 0
        for item in local:
            if item.terminal and (item.n_w > 0) != winner:
                raise Phase2DataError("terminal day contradicts final game winner")
            states.append(PostDayState(item.game_id, item.resolved_day, item.n_w,
                                       item.n_nw, winner, item.terminal, item.exiled_player,
                                       item.final_exile_event_index, item.fold))
    return tuple(states)


def classify_outcome(exiled_player: str | None, *, acting_wolf: str,
                     candidate_j: str, wolves: frozenset[str]) -> ResolvedOutcome:
    if acting_wolf not in wolves or candidate_j in wolves or candidate_j not in PLAYER_IDS:
        raise Phase2DataError("invalid wolf/candidate roles for outcome")
    if exiled_player is None:
        return ResolvedOutcome.NO_EXILE
    if exiled_player == candidate_j:
        return ResolvedOutcome.TARGET_J_EXILED
    if exiled_player == acting_wolf:
        return ResolvedOutcome.ACTING_WOLF_EXILED
    if exiled_player in wolves:
        return ResolvedOutcome.TEAMMATE_WOLF_EXILED
    if exiled_player in PLAYER_IDS:
        return ResolvedOutcome.OTHER_NONWOLF_EXILED
    raise Phase2DataError("exiled player is not a canonical seat")


def _transition(s_pre: tuple[int, int], outcome: ResolvedOutcome) -> tuple[int, int]:
    """Apply the resolved daytime exile once to a pre-outcome count state."""
    n_w, n_nw = _counts(s_pre)
    if n_w == 0 or n_w >= n_nw:
        raise Phase2DataError("resolved outcome requires nonterminal S_pre")
    if not isinstance(outcome, ResolvedOutcome):
        raise Phase2DataError("unknown resolved outcome")
    if outcome in (ResolvedOutcome.TARGET_J_EXILED, ResolvedOutcome.OTHER_NONWOLF_EXILED):
        s_post = (n_w, n_nw - 1)
    elif outcome in (ResolvedOutcome.ACTING_WOLF_EXILED, ResolvedOutcome.TEAMMATE_WOLF_EXILED):
        s_post = (n_w - 1, n_nw)
    else:
        s_post = (n_w, n_nw)
    return _counts(s_post)


def candidate_opportunities(games, role_maps: dict[str, dict[str, str]],
                            fold_of_game: dict[str, int], *,
                            diagnostics: Counter | None = None) -> tuple[_Opportunity, ...]:
    opportunities = []
    for game in games:
        roles = role_maps[game.game_id]
        wolves = frozenset(p for p, role in roles.items() if role == "Werewolf")
        reports = {o.observation_id: o for o in game.belief_observations}
        if len(reports) != len(game.belief_observations):
            raise Phase2DataError("duplicate canonical belief observation")
        for pre in game.authoritative_pre_prefixes:
            if pre.public_temporal_state.phase.value not in ("discussion", "pk_discussion"):
                continue
            actor = pre.current_speaker
            alive = tuple(pre.alive_observer_ids)
            if actor not in alive or actor not in wolves:
                continue
            phase, competition, remaining = phase_context(pre)
            candidates = tuple(p for p in competition if p not in wolves)
            if not candidates:
                if phase != "speech_pk":
                    raise Phase2DataError("ordinary speech PRE has no non-wolf candidate")
                if diagnostics is not None:
                    diagnostics["skipped_no_candidate"] += 1
                continue
            supports = {}
            for link in pre.belief_observation_links:
                report = reports.get(link.observation_id)
                if (report is None or report.game_id != game.game_id
                        or report.boundary_id != pre.boundary_id
                        or report.prefix_digest != pre.prefix_digest
                        or report.observer_id != link.observer_id):
                    raise Phase2DataError("broken PRE belief observation link")
                i = report.observer_id
                if (i in alive and i not in wolves and report.status.value == "success"
                        and report.label_observed is True and report.observer_alive is True):
                    supports[i] = frozenset(report.suspicion_support)
            if set(supports) != set(alive) - wolves:
                raise Phase2DataError("O_label differs from runtime-legal O_R2")
            opportunities.append(_Opportunity(game.game_id, pre.boundary_id,
                pre.prefix_digest, actor, phase, pre.public_temporal_state.day,
                alive, wolves, competition, candidates, remaining, supports,
                fold_of_game[game.game_id]))
    return tuple(opportunities)


def load_oof_probabilities(publication, evaluation_root: Path | str,
                           opportunities: tuple[_Opportunity, ...]) -> dict[tuple[str, str, str], tuple[float, ...]]:
    """Read only sealed held-out Q probabilities needed by eligible PREs."""
    root = Path(evaluation_root)
    contract = verify_artifact(root / "contract",
        expected_artifact_type="backbone_oof_evaluation_contract",
        expected_schema_version="classic7_backbone_oof_evaluation_v1")
    seal = verify_artifact(root / "seal",
        expected_artifact_type="backbone_oof_evaluation_report",
        expected_schema_version="classic7_backbone_oof_report_v1")
    if (seal.manifest_digest != OOF_SEAL_DIGEST
            or seal.manifest["evaluation_digest"] != contract.manifest_digest):
        raise Phase2DataError("Qwen3 OOF evaluation seal mismatch")
    descriptors = {(row["architecture"], row["fold"]): row
                   for row in seal.manifest["predictions"]}
    if len(descriptors) != len(seal.manifest["predictions"]):
        raise Phase2DataError("duplicate OOF prediction descriptor")
    fold_of_game = fold_assignment(publication.public_view.fold_manifest)
    needed = {}
    for opportunity in opportunities:
        for observer in set(opportunity.alive) - opportunity.wolves:
            key = (opportunity.game_id, opportunity.boundary_id, observer)
            expected = (opportunity.prefix_digest, opportunity.fold)
            if key in needed and needed[key] != expected:
                raise Phase2DataError("inconsistent PRE identity")
            needed[key] = expected
    found = {}
    for fold in range(5):
        descriptor = descriptors.get(("qwen3", fold))
        if descriptor is None:
            raise Phase2DataError("sealed Qwen3 fold descriptor missing")
        artifact = verify_artifact(root / "folds" / "qwen3" / str(fold),
            expected_artifact_type="backbone_oof_fold_predictions",
            expected_schema_version="classic7_backbone_oof_predictions_v1")
        manifest = artifact.manifest
        if (artifact.manifest_digest != descriptor["prediction_digest"]
                or manifest["evaluation_digest"] != contract.manifest_digest
                or manifest["architecture"] != "qwen3" or manifest["fold"] != fold
                or manifest["temporal_condition"] != "explicit_day_phase"
                or manifest["training_seal_digest"] != seal.manifest["training_seal_digest"]):
            raise Phase2DataError("Qwen3 OOF fold lineage mismatch")
        count = 0
        with (artifact.path / "predictions.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                count += 1
                game_id, boundary_id = row["game_id"], row["boundary_id"]
                observer_index = row["observer"]
                if (type(observer_index) is not int or not 0 <= observer_index < 7
                        or fold_of_game.get(game_id) != fold
                        or row["temporal_condition"] != "explicit_day_phase"
                        or row["checkpoint_digest"] != manifest["checkpoint_digest"]):
                    raise Phase2DataError("OOF prediction row lineage mismatch")
                observer = PLAYER_IDS[observer_index]
                key = (game_id, boundary_id, observer)
                if key not in needed:
                    continue
                if (key in found or needed[key] != (row["prefix_digest"], fold)):
                    raise Phase2DataError("OOF PRE row duplicate or prefix mismatch")
                found[key] = validate_q_row(observer, row["probability"])
        if count != manifest["row_count"]:
            raise Phase2DataError("OOF prediction row count mismatch")
    if set(found) != set(needed):
        raise Phase2DataError(f"OOF Q coverage mismatch: {len(found)}/{len(needed)}")
    return found


def build_candidate_rows(opportunities: tuple[_Opportunity, ...],
                         probabilities: dict[tuple[str, str, str], tuple[float, ...]],
                         post_days: tuple[PostDayState, ...]) -> tuple[CandidateRow, ...]:
    final = {(s.game_id, s.resolved_day): s for s in post_days}
    if len(final) != len(post_days):
        raise Phase2DataError("duplicate post-day state")
    rows = []
    identities = set()
    for opportunity in opportunities:
        state = final.get((opportunity.game_id, opportunity.day))
        if state is None:
            raise Phase2DataError("candidate PRE has no resolved day")
        n_w_pre = len(set(opportunity.alive) & opportunity.wolves)
        s_pre = _counts((n_w_pre, len(opportunity.alive) - n_w_pre))
        observers = set(opportunity.alive) - opportunity.wolves
        q = {i: probabilities[(opportunity.game_id, opportunity.boundary_id, i)]
             for i in observers}
        for candidate in opportunity.candidates:
            identity = (opportunity.game_id, opportunity.boundary_id,
                        opportunity.acting_wolf, opportunity.phase, candidate)
            if identity in identities:
                raise Phase2DataError("duplicate candidate identity")
            identities.add(identity)
            outcome = classify_outcome(state.exiled_player,
                acting_wolf=opportunity.acting_wolf, candidate_j=candidate,
                wolves=opportunity.wolves)
            if _transition(s_pre, outcome) != state.s_post:
                raise Phase2DataError("candidate S_pre/Y disagrees with post-day S0")
            rows.append(CandidateRow(opportunity.game_id, opportunity.boundary_id,
                opportunity.acting_wolf, opportunity.phase, candidate, opportunity.fold,
                opportunity.prefix_digest, s_pre,
                theta_ac(opportunity.alive, opportunity.wolves, opportunity.competition,
                         opportunity.supports, candidate),
                r2_input(opportunity.alive, opportunity.wolves, opportunity.competition,
                         candidate, q, phase=opportunity.phase, remaining=opportunity.remaining),
                outcome))
    return tuple(rows)


def build_value_tables(states: tuple[PostDayState, ...],
                       fold_of_game: dict[str, int]) -> ReferenceTables:
    if not states or len({(s.game_id, s.resolved_day) for s in states}) != len(states):
        raise Phase2DataError("empty or duplicated post-day state population")
    if any(s.fold != fold_of_game.get(s.game_id) for s in states):
        raise Phase2DataError("post-day fold assignment mismatch")

    def fit(excluded_fold: int | None) -> ValueTable:
        training_games = {g for g, f in fold_of_game.items() if f != excluded_fold}
        grouped = defaultdict(list)
        for state in states:
            _counts(state.s0)
            if state.game_id in training_games and not state.terminal:
                if state.n_w == 0 or state.n_w >= state.n_nw:
                    raise Phase2DataError("nonterminal post-day state is terminal by rule")
                grouped[state.s0].append(state)
        cells = {}
        for key, visits in grouped.items():
            wins = sum(state.final_wolf_win for state in visits)
            empirical = wins / len(visits)
            cells[key] = ValueEstimate(empirical, "empirical", excluded_fold,
                len(training_games), len(visits), len({s.game_id for s in visits}),
                wins, empirical)
        return ValueTable(excluded_fold, len(training_games), cells)

    result = ReferenceTables(fold_of_game, {fold: fit(fold) for fold in range(5)}, fit(None))
    for state in states:
        result.development(state.game_id).lookup(state.s0)
    return result


def _reference_outcome_loss(s_pre: tuple[int, int], outcome: ResolvedOutcome,
                            table: ValueTable) -> ReferenceLoss:
    s_post = _transition(s_pre, outcome)
    estimate = table.lookup(s_post)
    return ReferenceLoss(outcome, s_pre, s_post, estimate, 1.0 - estimate.value)


def build_development_layer(publication_path: Path | str,
                            evaluation_root: Path | str) -> OfflinePhase2:
    """Verify fixed evidence and build the in-memory Phase-2 offline layer."""
    publication = open_publication(publication_path)
    if (publication.manifest_digest != PUBLICATION_DIGEST
            or publication.public_view.fold_manifest.manifest_digest != FOLD_MANIFEST_DIGEST
            or publication.manifest["game_count"] != EXPECTED_GAMES):
        raise Phase2DataError("development publication provenance mismatch")
    sidecar = open_role_sidecar(publication)
    if sidecar.sidecar_digest != ROLE_SIDECAR_DIGEST:
        raise Phase2DataError("development role sidecar provenance mismatch")
    games = publication.public_view.games
    if len(games) != EXPECTED_GAMES or len(sidecar.games) != EXPECTED_GAMES:
        raise Phase2DataError("development game count mismatch")
    fold_of_game = fold_assignment(publication.public_view.fold_manifest)
    if set(fold_of_game) != {g.game_id for g in games}:
        raise Phase2DataError("fold manifest does not cover development games")
    role_maps = {g.game_id: dict(g.role_assignment) for g in sidecar.games}
    if set(role_maps) != set(fold_of_game):
        raise Phase2DataError("role sidecar does not cover development games")
    diagnostics = Counter()
    opportunities = candidate_opportunities(games, role_maps, fold_of_game,
                                            diagnostics=diagnostics)
    if len(opportunities) != EXPECTED_PRE:
        raise Phase2DataError(f"candidate PRE population mismatch: {len(opportunities)}")
    if diagnostics["skipped_no_candidate"] != EXPECTED_SKIPPED_NO_CANDIDATE_PK_PRE:
        raise Phase2DataError("no-candidate PK PRE population mismatch: "
                              f"{diagnostics['skipped_no_candidate']}")
    if sum(len(o.candidates) for o in opportunities) != EXPECTED_CANDIDATES:
        raise Phase2DataError("candidate-row population mismatch")
    states = post_day_states(games, role_maps, fold_of_game)
    if len(states) != EXPECTED_POST_DAYS:
        raise Phase2DataError(f"post-day state population mismatch: {len(states)}")
    probabilities = load_oof_probabilities(publication, evaluation_root, opportunities)
    rows = build_candidate_rows(opportunities, probabilities, states)
    if len(rows) != EXPECTED_CANDIDATES:
        raise Phase2DataError("built candidate-row population mismatch")
    values = build_value_tables(states, fold_of_game)
    return OfflinePhase2(rows, states, values, len(opportunities),
                         diagnostics["skipped_no_candidate"])
