"""Default-off PRE hook: deterministic fake gameplay is byte-for-byte unchanged."""
from copy import deepcopy
from dataclasses import asdict, replace
import inspect
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

from run_random import eval as run_game
from tests.agents.test_agent_backend import MetadataBackend, _belief_for, _day_cognition
from tests.canonical_collection.test_runtime_game_evidence import ROLES, _runtime as canonical_runtime
from tests.canonical_collection.test_game_bundle import _plan
from tests.observer_dtrs.conftest import prefix_for
from tests.observer_dtrs.test_shadow import deployed
from tests.observer_dtrs.test_shadow import request as artificial_request
from tests.observer_dtrs.test_shadow_worker import worker
from tests.tom.test_gameplay_ablation import _agent, _wolf_observation, _OneActionEnv, _Audit
from werewolf.artifact_io import canonical_json_bytes
from werewolf.canonical_collection.pre import construct_speaker_pre_belief_handoff
from werewolf.canonical_collection.runtime import CanonicalGameRecorder
from werewolf.agents.prompt_template_v0 import derive_belief_constraints
from werewolf.observer_dtrs.shadow_hook import ShadowPREHook


class Recorder:
    """Only recorder owns its pending slot; exercise the real public accessor."""
    public_pre_for_handoff = CanonicalGameRecorder.public_pre_for_handoff

    def __init__(self, prefix):
        self.prefix = prefix
        self.claim = SimpleNamespace(attempt_id='controlled-attempt')
        self.handoff = construct_speaker_pre_belief_handoff(prefix,
            observation_id=prefix.speaker_pre_belief_observation_id,
            observation_digest='a'*64, observer_id=prefix.current_speaker,
            observation_status='success', suspicion_support=())
        self.report_count = 0
        self.actions = []

    def start(self, *args, **kw):
        self._pending = None

    def before_agent_act(self, *args, speech_kind, **kw):
        if speech_kind is None:
            return None
        self.report_count += len(self.prefix.alive_observer_ids)
        self._pending = SimpleNamespace(prefix=self.prefix, handoff=self.handoff, raw_action=None)
        return self.handoff

    def after_agent_act(self, action):
        self.actions.append(deepcopy(action))
        if self._pending is not None:
            self._pending.raw_action = action

    def after_env_step(self, *args, **kw):
        pass

    def finish(self, *args, **kw):
        pass

    def failure_from_exception(self, error):
        return error


class ObservedAgent:
    """Capture exactly what the real GPTAgent is given, without modifying it."""
    def __init__(self, agent):
        self.agent, self.delivered = agent, []

    def reset(self):
        self.agent.reset()

    def act(self, observation):
        self.delivered.append((deepcopy(observation), None))
        return self.agent.act(observation)

    def act_with_pre_speech_belief(self, observation, *, pre_speech_belief):
        self.delivered.append((deepcopy(observation), pre_speech_belief.to_record()))
        return self.agent.act_with_pre_speech_belief(observation, pre_speech_belief=pre_speech_belief)


class SpeechVoteEnv(_OneActionEnv):
    """Two deterministic game steps: public speech then vote; no LLM/service."""
    def __init__(self, observation, vote):
        super().__init__(observation)
        self.vote = vote

    def step(self, action):
        self.actions.append(action)
        self.public_events.append({'fixture_submitted_action': deepcopy(action)})
        if len(self.actions) == 1:
            self.phase = 'vote'
            return deepcopy(self.vote), None, False, {}
        return deepcopy(self.vote), None, True, {'Werewolf': -1}


def observable_bytes(value):
    # Log is the runtime's value wrapper, not a pointer or mutable agent state.
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
        default=lambda obj: vars(obj), separators=(',', ':')).encode()


def evidence_bytes(value):
    """Lossless encoding of every dataclass field, including FrozenJSON bytes."""
    def encode(v):
        if isinstance(v, bytes):
            return {'__exact_bytes_hex__': v.hex()}
        if isinstance(v, dict):
            return {k: encode(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [encode(x) for x in v]
        return v
    return canonical_json_bytes(encode(asdict(value)))


@pytest.mark.parametrize('phase', ['discussion', 'pk_discussion'])
@pytest.mark.parametrize('dead', [(), ('player2', 'player7')])
@pytest.mark.parametrize('failure', ['ok', 'hang', 'crash', 'corrupt', 'backpressure'])
def test_fake_backend_on_off_exact_observation_handoff_prompt_speech_vote_rng(
        tmp_path, deployed, phase, dead, failure):
    prefix = prefix_for(phase=phase, dead=dead)
    runtime_phase = 'speech_pk' if phase == 'pk_discussion' else 'speech'
    o = _wolf_observation(runtime_phase, alive=[int(p[-1]) for p in prefix.alive_observer_ids])
    o['observer_id'] = 1  # current real get_observation_for contract; old prompt fixture omits it
    o['authoritative_public_state']['phase'] = runtime_phase
    o['authoritative_public_state']['suggestible_exile_targets'] = [
        int(p[-1]) for p in prefix.alive_observer_ids if p != 'player1']
    if phase == 'pk_discussion':
        o['game_log'][-1].content['speech_queue'] = [int(p[-1]) for p in prefix.alive_observer_ids[:2]]
    vote = _wolf_observation('vote');vote['observer_id'] = 1
    vote['authoritative_public_state']['alive_players'] = [int(p[-1]) for p in prefix.alive_observer_ids]
    roles = derive_belief_constraints(vote)[1]
    results = []
    for enabled in (False, True):
        random.seed(834);np.random.seed(441)
        w, hook = None, None
        if enabled:
            behavior = 'hang' if failure == 'backpressure' else ('ok' if failure == 'corrupt' else failure)
            w = worker(tmp_path/'worker', deployed, behavior,
                       timeout=1.5, max_pending=1)
            hook = ShadowPREHook(w, shadow_run_id='hook-fixture', attempt_id='controlled-attempt')
            if failure == 'corrupt':
                # Existing audit corruption: capture is disabled, baseline must still run.
                (w.root/'receipts/contract/record.json').write_bytes(b'bad log')
            elif failure == 'backpressure':
                assert w.capture({}, request=artificial_request(attempt='preexisting')) == 'SUBMITTED'
        backend = MetadataBackend([_day_cognition(o), _belief_for(roles), '{"target":0}'])
        agent = ObservedAgent(_agent(backend))
        env = SpeechVoteEnv(o, vote)
        env.public_events = prefix.public_event_history.to_records()
        recorder = Recorder(prefix)
        winner = run_game(env, [agent]*7, (), canonical_recorder=recorder,
                          call_audit=_Audit(), observer_shadow=hook)
        if w is not None:
            # Waits only in the trusted test owner after the baseline game returned.
            w.close()
        results.append((observable_bytes({'delivered': agent.delivered,
            'backend_calls': backend.calls, 'actions': env.actions, 'vote': env.actions[-1],
            'winner': winner, 'events': env.public_events, 'reports': recorder.report_count,
            'canonical_actions': recorder.actions}), random.getstate(), np.random.get_state()))
        if w is not None and failure != 'corrupt':
            s = w.inspect()
            if failure == 'backpressure':
                assert s['capture_attempts'] == s['eligible_captures'] == 2
                assert s['capture_statuses'] == {'SUBMITTED': 1, 'SKIPPED_BACKPRESSURE': 1}
                assert s['submitted'] == s['completed'] == 1
            else:
                assert s['capture_attempts'] == s['eligible_captures'] == s['submitted'] == s['completed'] == 1
            expected = {'ok': 'SUCCEEDED', 'hang': 'FAILED_TIMEOUT', 'crash': 'FAILED_CRASH',
                        'backpressure': 'FAILED_TIMEOUT'}[failure]
            assert s['outcome_statuses'] == {expected: 1}
        elif w is not None:
            assert w.audit_error
    assert results[0][0] == results[1][0]  # includes actual cognition/realization prompt bytes
    assert results[0][1] == results[1][1]
    a, b = results[0][2], results[1][2]
    assert a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]


@pytest.mark.parametrize('seed,eligible', [(101, False), (2, True)])
def test_real_canonical_recorder_fake_backend_evidence_and_budget_unchanged(tmp_path, deployed, seed, eligible):
    """Existing deterministic one-speech environment fixture; no real game/LLM."""
    results = []
    for enabled in (False, True):
        random.seed(17);np.random.seed(37)
        _, claim, env, agents, audit, recorder, backend = canonical_runtime(plan=_plan(ordered_seed_pool=(seed,)))
        w = worker(tmp_path/'canonical-worker', deployed) if enabled else None
        hook = ShadowPREHook(w, shadow_run_id='canonical-controlled', attempt_id=claim.attempt_id) if w else None
        winner = run_game(env, agents, ROLES, canonical_recorder=recorder,
                          call_audit=audit, observer_shadow=hook)
        if w:
            w.close()
            s = w.inspect()
            assert s['capture_attempts'] == len(recorder.prefixes) == 1
            assert s['eligible_captures'] == int(eligible)
            assert s['capture_statuses'] == ({'SUBMITTED': 1} if eligible else {'SKIPPED_NONWOLF': 1})
            if eligible:
                assert s['outcome_statuses'] == {'SUCCEEDED': 1}
        # All original evidence, including call IDs, report digests and private replay,
        # must be identical. Do not filter nondeterministic fields to make this pass.
        results.append((evidence_bytes(recorder.complete_evidence()),
                        deepcopy(backend.requests), winner, random.getstate(), np.random.get_state()))
    assert results[0][:4] == results[1][:4]
    assert np.array_equal(results[0][4][1], results[1][4][1])
    assert results[0][4][2:] == results[1][4][2:]


def test_accessor_rejects_wrong_handoff_consumed_or_missing_pre():
    recorder = Recorder(prefix_for())
    recorder.start()
    with pytest.raises(ValueError):
        recorder.public_pre_for_handoff(recorder.handoff)
    recorder.before_agent_act(speech_kind='speech')
    assert recorder.public_pre_for_handoff(recorder.handoff) is recorder.prefix
    with pytest.raises(ValueError):
        recorder.public_pre_for_handoff(replace(recorder.handoff, boundary_id='wrong'))
    recorder.after_agent_act(('speech', 'unchanged'))
    with pytest.raises(ValueError):
        recorder.public_pre_for_handoff(recorder.handoff)


def test_default_is_off_and_new_hook_never_reads_private_pending():
    assert inspect.signature(run_game).parameters['observer_shadow'].default is None
    from werewolf.observer_dtrs import shadow_hook
    text = inspect.getsource(shadow_hook)
    assert '_pending' not in text and 'env.step' not in text and 'agent.act' not in text


def test_accessor_error_does_not_abort_baseline(tmp_path, deployed):
    p = prefix_for();o = _wolf_observation()
    o['observer_id'] = 1
    recorder = Recorder(p)
    def broken(_):
        raise ValueError('artificial accessor mismatch')
    recorder.public_pre_for_handoff = broken
    w = worker(tmp_path/'worker', deployed)
    hook = ShadowPREHook(w, shadow_run_id='s', attempt_id=recorder.claim.attempt_id)
    b = MetadataBackend([_day_cognition(o)])
    result = run_game(_OneActionEnv(o), [_agent(b)]*7, (), canonical_recorder=recorder,
                      call_audit=_Audit(), observer_shadow=hook)
    w.close()
    assert result == 'Villager win' and len(b.calls) == 2
    assert w.inspect()['capture_statuses'] == {'CAPTURE_ERROR': 1}
    assert w.ledger.events[0]['pre'] == {'shadow_run_id': 's', 'attempt_id': recorder.claim.attempt_id,
        'boundary_id': p.boundary_id, 'prefix_digest': p.prefix_digest, 'actor': p.current_speaker}
