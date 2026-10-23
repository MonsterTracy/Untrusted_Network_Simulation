"""Spawn-picklable artificial provider; no game runtime or supervision imports."""
import os
import random
import sys
import time

from werewolf.observer_dtrs.probabilities import ProbabilityMatrix
from werewolf.observer_dtrs.shadow import QPrediction
from werewolf.structured_history import plan_structured_history


class ProcessTom:
    def __init__(self, identity, behavior='ok', marker=None):
        self.identity, self.behavior, self.marker = identity, behavior, marker

    def predict(self, prefix, *, deadline_monotonic):
        forbidden = {'run_random', 'werewolf.observer_dtrs.dataset_offline',
                     'werewolf.observer_dtrs.training_audit', 'werewolf.observer_dtrs.decision_offline',
                     'werewolf.envs.werewolf_text_env_v0', 'werewolf.agents.gpt_agent'}
        assert not forbidden.intersection(sys.modules)
        # Deliberately consume both global RNGs in the CHILD. Parent states must
        # still be identical in the hook on/off comparison.
        random.random()
        import numpy as np
        np.random.random()
        if self.marker is not None:
            with open(self.marker, 'a') as stream:
                stream.write(str(os.getpid())+'\n')
        if self.behavior == 'crash':
            os._exit(37)
        if self.behavior == 'hang':
            while True:
                time.sleep(.1)
        if self.behavior == 'late':
            time.sleep(3.)
        if self.behavior == 'error':
            raise ValueError('artificial provider failure; never a BND')
        values = [[0. if i == j else 1/6 for j in range(7)] for i in range(7)]
        values[2] = [.1, .2, 0., .1, .1, .2, .3]
        return QPrediction(ProbabilityMatrix(prefix.game_id, prefix.boundary_id, prefix.prefix_digest, values),
                           self.identity, plan_structured_history(prefix).plan_digest)


def wrong_response(spec, root, request, token, pipe):
    pipe.send({'status': 'SUCCEEDED', 'request_digest': request.digest,
               'token': 'stale-token', 'result_digest': '0'*64})
    pipe.close()
