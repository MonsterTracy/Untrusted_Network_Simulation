"""CPU-only contextual DTRS demonstration using built-in synthetic inputs only.

No input paths, remote calls, model training service, gameplay or artifact writes.
The tiny in-memory D2 fit is a synthetic fixture, not a research experiment.
"""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from werewolf.canonical_collection import construct_authoritative_pre_prefix, freeze_public_event_history
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.observer_dtrs.contextual_policy import propose_speech
from werewolf.observer_dtrs.speech_runtime import select_observer
from werewolf.observer_dtrs.d2_protocol import frozen_protocol
from werewolf.observer_dtrs.decision import TiePolicy
from werewolf.observer_dtrs.estimation import TrainingRow, fit_model
from werewolf.observer_dtrs.estimation_offline import implementation_digest
from werewolf.observer_dtrs.features import ObserverFeatures
from werewolf.observer_dtrs.probabilities import ProbabilityRow


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    # A valid first Day1 speech PRE. No raw report, hidden role or sidecar exists.
    events = [
        {"event_id": "e0", "event_index": 0, "event_type": "phase_change", "day": 0, "phase": "night"},
        {"event_id": "e1", "event_index": 1, "event_type": "death_announcement", "dead_players": []},
        {"event_id": "e2", "event_index": 2, "event_type": "phase_change", "day": 1, "phase": "discussion"},
        {"event_id": "e3", "event_index": 3, "event_type": "turn_start", "speaker": "player1"},
    ]
    pre = construct_authoritative_pre_prefix(
        game_id="synthetic-contextual-demo", boundary_id="synthetic-pre", step_index=3,
        report_trigger_id="synthetic-trigger", current_speaker="player1", alive_observer_ids=PLAYER_IDS,
        public_event_history=freeze_public_event_history(events), v1_annotations=(),
        belief_observation_ids_by_observer={p: f"synthetic-link-{p}" for p in PLAYER_IDS})
    protocol = frozen_protocol()
    rows = []
    for j in range(30):
        r = (.1, .4, .8)[j // 10]
        rows.append(TrainingRow(f"synthetic-{j}", f"synthetic-training-{j}",
                    ObserverFeatures(r/2, r/2, r, 0, "discussion", 2), int(j >= 15)))
    source = {"scope": "controlled_fixture", "protocol_digest": protocol.digest,
        "dataset_digest": None, "split_manifest_digest": None, "tom_fold": None,
        "outer_fold": None, "inner_fold": None, "game_sizes": {r.game_id: 1 for r in rows},
        "origins": {"fixture": "builtin_synthetic_only"}, "implementation_digest": implementation_digest()}
    model = fit_model(rows, protocol=protocol, model_id="OBS-EQ1", bins=3, smoothing=1, source=source)
    examples = []
    for r in (.1, .4, .8):
        q = ProbabilityRow(pre.game_id, pre.boundary_id, pre.prefix_digest, "player3",
                           (r/2, r/2, 0, 1-r, 0, 0, 0))
        result = propose_speech(pre, known_wolf_ids=("player1", "player2"), observer_id="player3",
                               prediction=q, model=model, protocol=protocol,
                               tie_policy=TiePolicy.TERMINAL_NEG_FIRST)
        examples.append(result.audit_record())
    print(json.dumps({"scope": "synthetic_only_not_formal_or_gameplay", "examples": examples,
                      "observer_selection": asdict(select_observer(pre, ("player1", "player2")))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
