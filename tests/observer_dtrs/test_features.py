from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tests.observer_dtrs.conftest import observation_for, prefix_for
from werewolf.canonical_collection.public_history import PLAYER_IDS
from werewolf.observer_dtrs.context import contexts_for_pre
from werewolf.observer_dtrs.features import observer_features
from werewolf.observer_dtrs.labels import label_from_observation
from werewolf.observer_dtrs.probabilities import ProbabilityMatrix, ProbabilityRow


def qrow(c, values=(0.1, 0.2, 0, 0.1, 0.1, 0.2, 0.3)):
    return ProbabilityRow(c.game_id, c.boundary_id, c.prefix_digest, c.observer_id, values)


@pytest.mark.parametrize("phase", ("discussion", "pk_discussion"))
@pytest.mark.parametrize("dead", ((), ("player2", "player7")))
def test_features_keep_dead_mass_and_canonical_phase(phase, dead):
    c = contexts_for_pre(prefix_for(phase=phase, dead=dead), ("player1", "player2"))[0]
    f = observer_features(c, qrow(c))
    assert f.s == 0.1 and f.phase == phase
    assert f.m == (0 if dead else 0.2)
    assert f.r == pytest.approx(0.1 if dead else 0.3)
    assert f.dW == (None if dead else 0.1)
    assert f.nW == (1 if dead else 2)
    # Dead player2/player7 retain 0.2/0.3; no surviving-column rescale.
    assert qrow(c).values == (0.1, 0.2, 0, 0.1, 0.1, 0.2, 0.3)


def test_arbitrary_wolf_seats_and_acting_wolf_swap():
    c = contexts_for_pre(prefix_for(actor="player5"), ("player3", "player5"))[0]
    q = qrow(c, (0, 0.1, 0.2, 0.1, 0.4, 0.1, 0.1))
    f = observer_features(c, q)
    assert (f.s, f.m, f.dW) == (0.4, 0.2, 0.2)
    c2 = contexts_for_pre(prefix_for(actor="player3"), ("player3", "player5"))[0]
    f2 = observer_features(c2, qrow(c2, q.values))
    assert (f2.s, f2.m) == (0.2, 0.4)
    assert f2.r == f.r and f2.dW == f.dW


def test_matrix_and_oof_row_share_exact_feature_path():
    c = contexts_for_pre(prefix_for(dead=("player2",)), ("player1", "player2"))[0]
    values = [[0 if i == j else 1 / 6 for j in range(7)] for i in range(7)]
    values[2] = list(qrow(c).values)
    matrix = ProbabilityMatrix(c.game_id, c.boundary_id, c.prefix_digest, values)
    assert observer_features(c, matrix) == observer_features(c, qrow(c))
    values[2][0] = 0.9
    assert matrix.values[2][0] == 0.1  # Immutable snapshot of caller buffers.


@pytest.mark.parametrize("values", (
    (0.1,) * 6, (0.1,) * 7, (-1, -2, 0, -3, -4, -5, -6),
    (0.1, 0.2, 0.01, 0.1, 0.1, 0.2, 0.29),
    (float("nan"), 0.2, 0, 0.1, 0.1, 0.2, 0.3),
    (float("inf"), 0.2, 0, 0.1, 0.1, 0.2, 0.3),
    (True, 0, 0, 0, 0, 0, 0), ("0.1", 0.2, 0, 0.1, 0.1, 0.2, 0.3),
    (1.1, -0.1, 0, 0, 0, 0, 0),
))
def test_invalid_probability_or_log_q_is_rejected(values):
    c = contexts_for_pre(prefix_for(), ("player1", "player2"))[0]
    with pytest.raises(ValueError):
        qrow(c, values)


@pytest.mark.parametrize("field,value", (("game_id", "other"), ("boundary_id", "other"),
    ("prefix_digest", "f" * 64), ("observer_id", "player1")))
def test_identity_mismatch_is_rejected(field, value):
    c = contexts_for_pre(prefix_for(), ("player1", "player2"))[0]
    # Use a valid numerical row for the different observer too.
    data = {"game_id": c.game_id, "boundary_id": c.boundary_id,
            "prefix_digest": c.prefix_digest, "observer_id": c.observer_id,
            "values": qrow(c).values}
    data[field] = value
    if field == "observer_id":
        data["values"] = (0, 0.2, 0.1, 0.1, 0.1, 0.2, 0.3)
    with pytest.raises(ValueError, match="identity mismatch"):
        observer_features(c, ProbabilityRow(**data))


def test_axes_and_incomplete_matrices_are_rejected():
    c = contexts_for_pre(prefix_for(), ("player1", "player2"))[0]
    with pytest.raises(ValueError, match="absolute"):
        replace(qrow(c), seat_ids=tuple(reversed(PLAYER_IDS)))
    with pytest.raises(ValueError, match="seven absolute"):
        ProbabilityMatrix(c.game_id, c.boundary_id, c.prefix_digest, (qrow(c).values,))
    values = [[0 if i == j else 1 / 6 for j in range(7)] for i in range(7)]
    values[6][6] = 0.1  # Even an unused/dead observer row must be valid.
    with pytest.raises(ValueError):
        ProbabilityMatrix(c.game_id, c.boundary_id, c.prefix_digest, values)


def test_supervision_cannot_enter_or_change_conditions():
    c = contexts_for_pre(prefix_for(), ("player1", "player2"))[0]
    yes = label_from_observation(c, observation_for(c, ("player1",)))
    no = label_from_observation(c, observation_for(c, ("player4",)))
    assert (yes.event_label_y, no.event_label_y) == (1, 0)
    for wrong in (yes, no, {"context": c, "supervision": yes}):
        with pytest.raises(TypeError):
            observer_features(wrong, qrow(c))
        with pytest.raises(TypeError):
            observer_features(c, wrong)
    with pytest.raises(TypeError):
        observer_features(c, qrow(c), supervision=yes)


def test_legal_feature_import_closure_excludes_restricted_and_model_modules():
    code = """import json,sys
from werewolf.observer_dtrs.features import observer_features
print(json.dumps(sorted(sys.modules)))
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], check=True,
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2])
    modules = json.loads(result.stdout)
    assert not any(m.startswith(("werewolf.observer_dtrs.offline", "werewolf.observer_dtrs.labels",
        "werewolf.observer_dtrs.oof", "werewolf.development_publication", "werewolf.phase2",
        "werewolf.tom", "torch", "sklearn")) for m in modules)
