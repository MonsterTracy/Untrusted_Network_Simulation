"""Synthetic terminal-estimator rows; never a substitute for formal evidence."""

import math


def synthetic_rows():
    """Return deterministic full-rank 120-row evidence, with 58/62 arm counts.

    State counts mirror the frozen support table solely to exercise the contract.
    Every score, outcome and identity is generated here and is explicitly synthetic.
    """
    intervals = {
        (2, 4): (0.0077301424406220195, 0.6566524289627977),
        (2, 5): (0.00521034560968036, 0.5252706354667532),
        (2, 3): (0.05, 0.35),
        (1, 4): (0.15, 0.15),
    }
    groups = (
        ("PUSH", (2, 5), 40), ("PUSH", (2, 4), 15), ("PUSH", (2, 3), 3),
        ("REDIRECT", (2, 5), 38), ("REDIRECT", (2, 4), 21),
        ("REDIRECT", (2, 3), 2), ("REDIRECT", (1, 4), 1),
    )
    rows = []
    for action, state, count in groups:
        low, high = intervals[state]
        for local_index in range(count):
            index = len(rows)
            score = low if count == 1 else low + (high - low) * local_index / (count - 1)
            # Preserve the inclusive endpoint bytes used by the support gate.
            if local_index == count - 1:
                score = high
            push = action == "PUSH"
            state_term = {(2, 5): 0.0, (2, 4): .05, (2, 3): .01, (1, 4): .03}[state]
            loss = .2 + .08 * push + state_term + .11 * score + .03 * push * score
            loss += .01 * math.sin(index * 1.2)
            rows.append({
                "assignment_id": f"synthetic-assignment-{index:03d}",
                "game_id": f"synthetic-game-{index:03d}",
                "assigned_action": action,
                "assignment_probability": .5,
                "S_pre": list(state),
                "p_tilde_j": score,
                "L_ref": loss,
                "V_ref": 1 - loss,
                "phase": "speech",
                "execution_success": index != 0,
                "execution_failure_reason": "SYNTHETIC_FAILURE" if index == 0 else None,
                "outcome_source": "synthetic_test_fixture",
                "candidate_j": "player4",
                "acting_wolf": "player2",
            })
    return rows
