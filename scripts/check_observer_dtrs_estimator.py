"""Controlled D2 estimator acceptance; never reads a formal dataset or fits CV."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from werewolf.observer_dtrs.d2_protocol import frozen_protocol
from werewolf.observer_dtrs.estimation import TrainingRow, fit_model
from werewolf.observer_dtrs.estimation_offline import implementation_digest, publish_model, read_model
from werewolf.observer_dtrs.features import ObserverFeatures
from werewolf.observer_dtrs.scoring import score_predictions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True,
                        help="New controlled model artifact; immutable, no overwrite")
    args = parser.parse_args(argv)
    protocol = frozen_protocol()
    positive = ObserverFeatures(.2, 0, .2, None, "discussion", 1)
    negative = ObserverFeatures(.1, .1, .2, 0, "pk_discussion", 2)
    rows = (TrainingRow("controlled-a", "controlled-game-a", positive, 1),
            TrainingRow("controlled-b", "controlled-game-b", negative, 0))
    source = {"scope": "controlled_fixture", "protocol_digest": protocol.digest,
        "dataset_digest": None, "split_manifest_digest": None,
        "tom_fold": None, "outer_fold": None, "inner_fold": None,
        "game_sizes": dict(Counter(r.game_id for r in rows)),
        "origins": {"fixture": "two_games_root_half_wolf_three_quarters_state_seven_eighths"},
        "implementation_digest": implementation_digest()}
    m0 = fit_model(rows, protocol=protocol, model_id="OBS-EQ0", bins=1, smoothing=1, source=source)
    m1 = fit_model(rows, protocol=protocol, model_id="OBS-EQ1", bins=1, smoothing=1, source=source)
    predictions = {r.sample_id: m0.predict(r.features) for r in rows}
    expected = {"controlled-a": .875, "controlled-b": .125}
    if predictions != expected or predictions != {r.sample_id: m1.predict(r.features) for r in rows}:
        raise ValueError("controlled exact probability acceptance failed")
    artifact = publish_model(args.destination, m0, protocol=protocol)
    loaded = read_model(artifact.path, expected_manifest_digest=artifact.manifest_digest, protocol=protocol)
    if loaded.payload != m0.payload:
        raise ValueError("controlled artifact round-trip failed")
    print(json.dumps({"scope": "controlled_fixture_not_performance_validation", "protocol_digest": protocol.digest,
        "model_path": str(artifact.path), "manifest_digest": artifact.manifest_digest,
        "predictions": predictions, "scores": score_predictions(rows, predictions)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
