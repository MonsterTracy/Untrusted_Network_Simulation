"""Read-only availability audit on current-PRE public queue and legal wolf team."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from werewolf import phase2_offline as offline
from werewolf.development_publication import open_publication, open_role_sidecar
from werewolf.phase2_actions import ActionContextV1, context_from_pre, probe_continuation


DEFAULT_PUBLICATION = Path(
    "/data/yuxiao/Untrusted_Network_Simulation/publications/"
    "paper-development-qwen35-9b-1500-v1")


def development_contexts(publication_path: Path | str) -> tuple[ActionContextV1, ...]:
    """Open frozen inputs; inspect current PREs only, never later game events."""
    publication = open_publication(publication_path)
    sidecar = open_role_sidecar(publication)
    if (publication.manifest_digest != offline.PUBLICATION_DIGEST
            or sidecar.sidecar_digest != offline.ROLE_SIDECAR_DIGEST
            or len(publication.public_view.games) != offline.EXPECTED_GAMES):
        raise ValueError("availability inputs differ from frozen development publication")
    roles = {game.game_id: dict(game.role_assignment) for game in sidecar.games}
    contexts = []
    for game in publication.public_view.games:
        wolves = frozenset(p for p, role in roles[game.game_id].items()
                             if role == "Werewolf")
        for pre in game.authoritative_pre_prefixes:
            if pre.public_temporal_state.phase.value not in ("discussion", "pk_discussion"):
                continue
            if pre.current_speaker not in wolves or pre.current_speaker not in pre.alive_observer_ids:
                continue
            context = context_from_pre(pre, wolves)
            if context.legal_targets:
                contexts.append(context)
            elif context.phase != "speech_pk":
                raise ValueError("ordinary wolf PRE lacks legal target")
    return tuple(contexts)


def analyze_contexts(contexts: tuple[ActionContextV1, ...]) -> dict:
    """Availability is structural: no Q, speech response, vote, or outcome read."""
    if not isinstance(contexts, tuple) or any(not isinstance(c, ActionContextV1) for c in contexts):
        raise TypeError("contexts must be a tuple of ActionContextV1")
    seen = set()
    by_phase = {phase: {"candidate_pre": 0, "candidate_rows": 0,
                        "push_available": 0, "redirect_available": 0,
                        "probe_available": 0, "probe_eligible_pre": 0,
                        "action_sets": Counter()} for phase in ("speech", "speech_pk")}
    gaps = Counter()
    continuation_relation = Counter()
    continuation_by_phase = {"speech": Counter(), "speech_pk": Counter()}
    for context in contexts:
        key = (context.game_id, context.boundary_id, context.acting_wolf, context.phase)
        if key in seen:
            raise ValueError("duplicate candidate PRE")
        seen.add(key)
        phase_counts = by_phase[context.phase]
        phase_counts["candidate_pre"] += 1
        has_probe = False
        for j in context.legal_targets:
            phase_counts["candidate_rows"] += 1
            phase_counts["push_available"] += 1
            actions = ["P"]
            can_redirect = len(context.legal_targets) > 1
            if can_redirect:
                phase_counts["redirect_available"] += 1
            continuation = probe_continuation(context, j)
            if continuation is not None:
                phase_counts["probe_available"] += 1
                has_probe = True
                actions.append("B")
                actor, _window = continuation
                relation = "teammate" if actor != context.acting_wolf else "same_wolf"
                continuation_relation[relation] += 1
                continuation_by_phase[context.phase][relation] += 1
                queue = context.public_speaker_queue
                first = queue.index(context.acting_wolf)
                middle = queue.index(j)
                last = queue.index(actor)
                gaps[f"{context.phase}:before_j={middle-first}:after_j={last-middle}"] += 1
            if can_redirect:
                actions.append("N")
            phase_counts["action_sets"]["{" + ",".join(actions) + "}"] += 1
        if has_probe:
            phase_counts["probe_eligible_pre"] += 1
    for phase_counts in by_phase.values():
        phase_counts["probe_eligibility_rate"] = (
            phase_counts["probe_available"] / phase_counts["candidate_rows"]
            if phase_counts["candidate_rows"] else None)
        phase_counts["probe_pre_rate"] = (
            phase_counts["probe_eligible_pre"] / phase_counts["candidate_pre"]
            if phase_counts["candidate_pre"] else None)
        phase_counts["action_sets"] = dict(sorted(phase_counts["action_sets"].items()))
    overall = {key: sum(by_phase[phase][key] for phase in by_phase)
               for key in ("candidate_pre", "candidate_rows", "push_available",
                           "redirect_available", "probe_available", "probe_eligible_pre")}
    overall["probe_eligibility_rate"] = (
        overall["probe_available"] / overall["candidate_rows"]
        if overall["candidate_rows"] else None)
    overall["probe_pre_rate"] = (
        overall["probe_eligible_pre"] / overall["candidate_pre"]
        if overall["candidate_pre"] else None)
    action_sets = Counter()
    for counts in by_phase.values():
        action_sets.update(counts["action_sets"])
    overall["action_sets"] = dict(sorted(action_sets.items()))
    return {"schema_version": "phase2_action_availability_v1",
            "population": overall, "by_phase": by_phase,
            "probe_continuation_relation": dict(sorted(continuation_relation.items())),
            "probe_continuation_relation_by_phase": {
                phase: dict(sorted(counts.items())) for phase, counts in continuation_by_phase.items()},
            "probe_candidate_position_gaps": dict(sorted(gaps.items())),
            "inputs_used": ["current_PRE_alive", "known_wolf_team", "phase",
                            "current_PRE_public_queue", "candidate_j"],
            "future_outcome_used": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", type=Path, default=DEFAULT_PUBLICATION)
    args = parser.parse_args(argv)
    contexts = development_contexts(args.publication)
    result = analyze_contexts(contexts)
    if (result["population"]["candidate_pre"] != offline.EXPECTED_PRE
            or result["population"]["candidate_rows"] != offline.EXPECTED_CANDIDATES):
        raise ValueError("availability population differs from frozen Phase-2 candidates")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
