"""Trusted capture boundary: delivered actor view -> legal immutable request only."""
from werewolf.observer_dtrs.online_context import build_shadow_request, seat
from werewolf.observer_dtrs.shadow_worker import IsolatedShadowWorker


class ShadowPREHook:
    """Explicit opt-in; no callback/result path into an agent or environment.

    A trusted owner creates/closes the worker outside gameplay. Storage errors
    disable Shadow, expose audit_error, and never escape into canonical execution.
    """

    def __init__(self, worker, *, shadow_run_id, attempt_id):
        if (type(worker) is not IsolatedShadowWorker
                or any(type(s) is not str or not s.strip() for s in (shadow_run_id, attempt_id))):
            raise TypeError('isolated worker and explicit run/attempt identities required')
        self.worker = worker
        self.shadow_run_id, self.attempt_id = shadow_run_id, attempt_id
        self.capture_errors = 0

    def capture(self, prefix, acting_observation, *, canonical_attempt_id):
        pre = {'game_id': prefix.game_id, 'boundary_id': prefix.boundary_id,
               'prefix_digest': prefix.prefix_digest, 'step_index': prefix.step_index,
               'actor': prefix.current_speaker, 'shadow_run_id': self.shadow_run_id,
               'attempt_id': canonical_attempt_id, 'configured_attempt_id': self.attempt_id}
        try:
            if canonical_attempt_id != self.attempt_id:
                raise ValueError('Shadow/canonical attempt mismatch')
            # Verify public binding before declaring even a nonwolf skip. Unknown
            # or mismatched views are capture errors, not evidence of ineligibility.
            o = acting_observation
            state = prefix.public_temporal_state
            public = o['authoritative_public_state']
            runtime = {'discussion': 'speech', 'pk_discussion': 'speech_pk'}[state.phase.value]
            alive = tuple(seat(p) for p in public['alive_players'])
            if (o['identity'] not in {'Werewolf', 'Villager', 'Seer', 'Witch'}
                    or seat(o['current_act_idx']) != prefix.current_speaker
                    or seat(o['observer_id']) != prefix.current_speaker
                    or prefix.current_speaker not in alive or len(set(alive)) != len(alive)
                    or set(alive) != set(prefix.alive_observer_ids)
                    or o['phase'] != f'{state.day}_day_{runtime}'
                    or public['phase'] != runtime or public['day_or_night'] != 'day'
                    or type(public['day']) is not int or public['day'] != state.day):
                raise ValueError('delivered actor/public PRE mismatch')
            if acting_observation.get('identity') != 'Werewolf':
                return self.worker.capture(pre, status='SKIPPED_NONWOLF')
            request = build_shadow_request(prefix, acting_observation,
                shadow_run_id=self.shadow_run_id, attempt_id=self.attempt_id)
        except Exception:
            self.capture_errors += 1
            return self.worker.capture(pre, status='CAPTURE_INVALID')
        return self.worker.capture(pre, request=request)

    def capture_error(self, *, boundary_id=None, prefix_digest=None, actor=None):
        """Fail-open gameplay; record accessor failure without retaining its exception."""
        self.capture_errors += 1
        try:
            self.worker.capture({'shadow_run_id': self.shadow_run_id, 'attempt_id': self.attempt_id,
                'boundary_id': boundary_id, 'prefix_digest': prefix_digest, 'actor': actor},
                status='CAPTURE_ERROR')
        except Exception:
            self.worker.audit_error = True
