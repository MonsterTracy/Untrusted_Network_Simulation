"""正式三支发言的原生运行、证据绑定与失败边界。"""

from dataclasses import asdict, replace
import json
import re
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import collect_games, evaluate_observer_dtrs_action_selection as study
from tests.canonical_collection.test_game_bundle import _plan
from tests.canonical_collection.test_runtime_game_evidence import ROLES, _runtime
from tests.observer_dtrs.test_speech_candidates import P, candidate, frozen_q
from tests.observer_dtrs.test_speech_consequences import fixture_game, model, predictor
from tests.observer_dtrs.test_speech_runtime import runtime as pending_runtime
from tests.runtime.test_runtime_config import new_config
from tests.tom.test_verified_speech_commit import snapshot
from werewolf.agents.gpt_agent import GPTAgent
from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection import BackendCallPurpose, BackendCallStatus, validate_canonical_game_evidence
from werewolf.canonical_collection.collector import CanonicalAttemptFailure
from werewolf.canonical_collection.failure_evidence import CanonicalFailureStage
from werewolf.canonical_collection.call_audit import AuditedBackend
from werewolf.canonical_collection.game_bundle import publish_canonical_game_bundle
from werewolf.canonical_collection.production_runtime import _RecordedRuntime, classic7_full_game_replay_executor
from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
from werewolf.observer_dtrs.report_state_decision import FrozenSharedRule, Thresholds
from werewolf.observer_dtrs.speech_consequences import build_consequence_plan
from werewolf.observer_dtrs.speech_runtime import SpeechDecisionUnavailable, ThreeWaySpeechPolicy
from werewolf.speech.commit_workflow import commit_public_speech


RULE = FrozenSharedRule(Thresholds.shared(.2, .75), 'rule', 'probability')
HOLD = '  先核对原话，再讨论。\n  '
PUBLIC = '我质疑3号的公开证据。'
PREDICTED = '我想回应玩家对我的指认。'


class CandidateBackend:
    supports_json_schema = True

    def __init__(self, *, undetermined=False):
        self.requests = []
        self.parses = []
        self.undetermined = undetermined

    def chat_with_metadata(self, *, messages, response_format, **kwargs):
        schema = response_format['json_schema']['name']
        context = json.loads(messages[-1]['content'].rsplit('\n\n', 1)[1])
        self.requests.append((schema, context))
        if schema == 'speech_candidates':
            focal = context['wolf_suspicion_mass']['observer_ids'][0]
            rows = [candidate('INFORM', HOLD), candidate('ADVOCATE', PUBLIC),
                    candidate('ADVOCATE', PREDICTED, premises=({'observer_id': focal, 'event': 'speaker_hit',
                                                               'direction': 'POS'},))]
            result = {'topic': '公开证据如何影响本轮判断', 'candidates': rows}
        else:
            assert schema == 'speech_candidate_bases'
            result = {'candidates': [dict(index=c['index'], required_premises=c['required_premises'],
                public_evidence_refs=c['public_evidence_refs'], reason='合成核验',
                status='undetermined' if self.undetermined else 'identified') for c in context['candidates']]}
        return json.dumps(result), {'finish_reason': 'stop'}

    def chat(self, *, messages, **kwargs):
        prompt = messages[-1]['content']
        speaker = re.search(r'当前发言者：(player[1-7])', prompt).group(1)
        self.parses.append(prompt)
        if HOLD in prompt:
            return 'NONE' if 'SpeechActionValidationError' in prompt else '格式错误'
        if PUBLIC in prompt:
            return f'{speaker} | oppose | player3\n{speaker} | vote_intent | player3'
        if PREDICTED in prompt:
            return f'{speaker} | oppose | player4'
        return 'NONE'


def build_policy(predictor, model, audit, agents, *, delta, undetermined=False, missing=False):
    backend = CandidateBackend(undetermined=undetermined)
    wrapped = AuditedBackend(backend, audit)
    adapter = GPTAgent(backend=wrapped, model_name=audit.plan.model_identity)

    def generate(*args, **kwargs):
        return adapter.generate_speech_candidates(*args, **kwargs)

    def verify(*args, **kwargs):
        return adapter.verify_speech_candidate_bases(*args, **kwargs)

    for agent in agents[:2]:
        agent.generate_speech_candidates = generate
        agent.verify_speech_candidate_bases = verify

    def predict(pre):
        wolf = P.index(pre.current_speaker)
        mass = .25 if pre.public_temporal_state.day == 1 else .9
        q = [[(0. if i == j else mass if j == wolf else (1 - mass) / 5) for j in range(7)]
             if i != wolf else list(frozen_q(pre).values[i]) for i in range(7)]
        return SimpleNamespace(game_id=pre.game_id, boundary_id=pre.boundary_id, prefix_digest=pre.prefix_digest, probabilities=q)

    predictor.predict = Mock(side_effect=predict)

    def information(batch, team, q):
        pre = batch.pre
        live_wolves = tuple(seat for seat in team if seat in pre.alive_observer_ids)
        observers = tuple(seat for seat in pre.alive_observer_ids if seat not in team)
        return dict(game_id=q.game_id, boundary_id=q.boundary_id, prefix_digest=q.prefix_digest,
            step_index=pre.step_index, acting_wolf=pre.current_speaker, alive_wolf_ids=live_wolves, head_statuses={},
            candidates=[dict(index=i, text_sha256=sha256_bytes(c.text.encode()), observers={seat: {
                'predicted_redecision_opportunity': .8, 'predicted_target_stability': .8,
                'predicted_cognitive_gain': None if missing else (.7, .4, .9)[i],
            } for seat in observers}) for i, c in enumerate(batch.candidates)])

    records = []
    policy = ThreeWaySpeechPolicy(agents=agents, predictor=predictor, model=model, rule=RULE,
        cognitive=SimpleNamespace(predict_candidates=information), delta=delta, record_evidence=records.append)
    return policy, wrapped, records


def test_native_full_game_reuses_only_real_selection_and_updates_all_wolf_pres(tmp_path, model, predictor):
    plan, claim, setup, agents, audit, recorder, _ = _runtime(plan=_plan(ordered_seed_pool=(1,)))
    env = WerewolfTextEnvV0(random_seed=claim.seed, log_save_path=None, speech_perceiver=setup.speech_perceiver)
    policy, wrapped, rows = build_policy(predictor, model, audit, agents, delta='1/4')
    env.speech_perceiver.backend = wrapped
    native_calls = []
    for agent in agents:
        def native_speech(observation, *, pre_speech_belief, original=agent.act_with_pre_speech_belief):
            assert observation['identity'] != 'Werewolf'
            assert 'current_q' not in observation and 'permitted_premises' not in observation
            native_calls.append('non_wolf_speech')
            return original(observation, pre_speech_belief=pre_speech_belief)

        def native_action(observation, original=agent.act):
            phase = observation['phase']
            seat = observation['current_act_idx']
            native_calls.append(phase)
            if phase.endswith(('skill_wolf', 'vote_pk')):
                target = 0
            elif phase == '1_day_vote':
                target = (2, 1, 1, 1, 2, 2, 0)[seat - 1]
            elif phase.endswith('_day_vote'):
                exile = 1 if phase.startswith('2_') else 2
                target = exile if seat != exile else 0
            else:
                return original(observation)
            return next(action for action in observation['valid_action'] if action[1] == target)

        agent.act_with_pre_speech_belief = native_speech
        agent.act = native_action

    replay = classic7_full_game_replay_executor(new_config())
    product = _RecordedRuntime(env, agents, list(ROLES), recorder, audit, replay, online_pilot=policy).run()
    bundle = publish_canonical_game_bundle(tmp_path / recorder.game_id, plan=plan, claim=claim,
                                          evidence=product.evidence, replay_executor=replay)
    validate_canonical_game_evidence(product.evidence)
    assert bundle.manifest['canonical_eligibility'] is True
    wolf_pres = [pre for pre in recorder.prefixes if pre.current_speaker in P[:2]]
    assert [call.args[0] for call in predictor.predict.call_args_list] == wolf_pres
    assert len(rows) == len(wolf_pres) == 7
    assert [kind for kind, _ in wrapped._backend.requests] == ['speech_candidates', 'speech_candidate_bases'] * len(rows)
    assert native_calls.count('non_wolf_speech') == len(recorder.prefixes) - len(rows)
    assert any('skill_wolf' in call for call in native_calls) and any('vote_pk' in call for call in native_calls)
    annotations = {item.event_id: item for item in env.speech_annotations}
    public = {item['event_id']: item for item in env.public_events}
    posts = set()
    for pre, record in zip(wolf_pres, rows, strict=True):
        assert record['status'] == 'committed' and record['delta_exact'] == '1/4'
        decision = record['decision']
        assert 'arms' not in decision
        if pre.public_temporal_state.day == 1:
            assert decision['selection']['premise_not_permitted_indices'] == [2]
            assert decision['selection']['selected_index'] == 0
            assert decision['selection']['used_delta'] > 0
        else:
            assert decision['selection']['premise_not_permitted_indices'] == []
            assert decision['selection']['selected_index'] == 2
        selected = decision['candidates'][decision['selection']['selected_index']]
        event_id = record['commit']['event_id']
        assert public[event_id]['raw_text'] == selected['text']
        assert [action.to_record() for action in annotations[event_id].actions] == selected['v1_parse']['normalized_actions']
        post = record['post_speech']
        assert post['reused_selected_prediction'] and post['q'] == selected['social_prediction']['q']
        posts.add((pre.public_temporal_state.phase.value, public[post['boundary_event_id']]['event_type']))
        calls = [call for call in audit.records if call.boundary_id == pre.boundary_id]
        official = [call for call in calls if call.purpose is BackendCallPurpose.SPEECH_PERCEPTION]
        assert [call.call_id for call in official] == [item.call_id for item in annotations[event_id].attempts]
        assert [call.private_payload.to_value()['response'] for call in official] == [item.raw_response for item in annotations[event_id].attempts]
        assert len(official) == (2 if selected['text'] == HOLD else 1)
        assert official[0].status is (BackendCallStatus.ERROR if selected['text'] == HOLD else BackendCallStatus.SUCCESS)
        private_parses = [call for call in calls if call.purpose is BackendCallPurpose.RUNTIME and '-candidate-' in call.operation_id
                          and call.private_payload.to_value()['method'] == 'chat']
        assert len(private_parses) + len(official) == 4
    assert posts == {('discussion', 'turn_start'), ('discussion', 'phase_change'),
                     ('pk_discussion', 'turn_start'), ('pk_discussion', 'phase_change')}


def test_zero_delta_uses_strict_social_best_then_commits_same_multiple_v1_actions(model, predictor):
    env, recorder, _ = pending_runtime(native=True)
    policy, wrapped, rows = build_policy(predictor, model, recorder.call_audit, [SimpleNamespace() for _ in range(7)], delta=0)
    env.speech_perceiver.backend = wrapped
    policy.handle_pre(env=env, recorder=recorder, call_audit=recorder.call_audit,
                      observation=env.get_observation(), handoff=recorder._pending.handoff)
    decision = rows[0]['decision']
    assert decision['selection']['selected_index'] == 1 and decision['selection']['used_delta'] == 0
    assert recorder.submitted_gameplay_actions[-1].action_payload.to_value() == PUBLIC
    assert [row.to_record() for row in env.speech_annotations[-1].actions] == decision['candidates'][1]['v1_parse']['normalized_actions']
    assert len(env.speech_annotations[-1].actions) == 2
    assert sum(PUBLIC in prompt for prompt in wrapped._backend.parses) == 1


def test_different_real_structure_infers_post_again_instead_of_reusing_hypothesis(predictor):
    pre, _, game = fixture_game('pk_discussion', last=True)
    unrelated = build_consequence_plan(pre, '这一候选没有 V1 动作。', ())
    forecast = predictor.forecast(unrelated)
    observed = predictor.observe_after_speech(game.public_event_stream, game.speech_annotations_v1,
        {'model_input_digest': unrelated.model_input_digest, 'q': forecast.matrix.values})
    assert predictor._model.calls == 2 and not observed['reused_selected_prediction']
    assert observed['public_history_digest'] == game.public_event_stream.digest
    assert observed['boundary_event_id'] == game.public_event_stream.events[-1].event_id
    assert list(forecast.matrix.values[0]) != observed['q'][0]


@pytest.mark.parametrize(('waiting', 'missing', 'status'), [(True, False, 'no_certified_feasible_candidate'),
                                                           (False, True, 'info_value_missing_in_band')])
def test_uncertain_or_missing_selection_preserves_pending_without_legacy_action(model, predictor, waiting, missing, status):
    env, recorder, _ = pending_runtime(native=True)
    policy, wrapped, rows = build_policy(predictor, model, recorder.call_audit, [SimpleNamespace() for _ in range(7)],
                                         delta=0, undetermined=waiting, missing=missing)
    env.speech_perceiver.backend = wrapped
    before = snapshot(env, recorder)
    with pytest.raises(SpeechDecisionUnavailable, match=status) as captured:
        policy.handle_pre(env=env, recorder=recorder, call_audit=recorder.call_audit,
                          observation=env.get_observation(), handoff=recorder._pending.handoff)
    assert captured.value.status == rows[0]['decision']['status'] == status and rows[0]['status'] == 'failed'
    assert snapshot(env, recorder) == before and 'commit' not in rows[0]
    assert all(call.purpose is not BackendCallPurpose.SPEECH_PERCEPTION for call in recorder.call_audit.records
               if call.boundary_id == rows[0]['boundary_id'])


def test_wrong_current_q_stops_before_any_candidates_or_commit(model, predictor):
    env, recorder, _ = pending_runtime(native=True)
    policy, wrapped, rows = build_policy(predictor, model, recorder.call_audit, [SimpleNamespace() for _ in range(7)], delta=0)
    pre = recorder._pending.prefix
    q = replace(frozen_q(pre), prefix_digest='f' * 64)
    predictor.predict.side_effect = None
    predictor.predict.return_value = SimpleNamespace(game_id=q.game_id, boundary_id=q.boundary_id,
                                                      prefix_digest=q.prefix_digest, probabilities=q.values)
    with pytest.raises(ValueError, match='PRE'):
        policy.handle_pre(env=env, recorder=recorder, call_audit=recorder.call_audit,
                          observation=env.get_observation(), handoff=recorder._pending.handoff)
    assert not wrapped._backend.requests and rows[0]['stage'] == 'cognition' and not recorder.submitted_gameplay_actions


def test_staged_speech_cannot_claim_another_candidate_call_or_mismatched_response():
    env, recorder, backend = pending_runtime(native=True)
    backend.perception_response = 'NONE'
    audit = recorder.call_audit
    pre, handoff = recorder._pending.prefix, recorder._pending.handoff
    speech = '明确指认3号。'
    with audit.candidate_perception_context(boundary_id=pre.boundary_id, speaker_id=1, candidate_index=0, speech=speech):
        perception = env.speech_perceiver.parse_with_audit(speaker=1, speech=speech, day=env.day, phase=env.phase)
    scored = dict(index=1, text=speech, text_sha256=sha256_bytes(speech.encode()), v1_parse=asdict(perception))
    before = snapshot(env, recorder)
    with pytest.raises(ValueError, match='唯一的真实解析'):
        commit_public_speech(speech, pre=pre, env=env, recorder=recorder, handoff=handoff, scored_candidate=scored)
    scored['index'] = 0
    scored['v1_parse']['generation_attempts'][0]['raw_response'] = '别的回答'
    with pytest.raises(ValueError, match='响应与暂存 V1'):
        commit_public_speech(speech, pre=pre, env=env, recorder=recorder, handoff=handoff, scored_candidate=scored)
    assert snapshot(env, recorder) == before
    assert audit.records[-1].purpose is BackendCallPurpose.RUNTIME


def test_online_cli_passes_exact_delta_to_single_official_method(monkeypatch):
    campaign = Mock(return_value=0)
    monkeypatch.setattr(collect_games, 'run_campaign', campaign)
    monkeypatch.setattr(study, '_frozen_publication', Mock(return_value=object()))

    def loaded(args, _publication):
        return {'rule': RULE, 'delta': args.delta}, {'capsule': 'pinned', 'delta_exact': str(args.delta)}

    monkeypatch.setattr(study, '_decision_models', loaded)
    args = ['online', '--campaign', 'campaign.json', '--capsule', 'capsule', '--capsule-digest', 'pinned']
    with pytest.raises(SystemExit) as missing:
        study.main(args)
    assert missing.value.code == 2
    campaign.assert_not_called()
    assert study.main([*args, '--delta', '.125']) == 0
    factory, sources = campaign.call_args.kwargs['online_policy']
    assert str(factory.keywords['delta']) == sources['delta_exact'] == '1/8'
    assert factory.func is ThreeWaySpeechPolicy and sources['method'] == 'tom_3wd'


def test_fixed_campaign_keeps_all_attempts_and_failure_evidence(tmp_path, monkeypatch):
    from werewolf import backends
    from werewolf.canonical_collection import production_runtime
    from werewolf.canonical_collection.attempt_ledger import validate_attempt_ledger

    storage, campaign = tmp_path / 'storage.json', tmp_path / 'campaign.json'
    storage.write_text(json.dumps({'artifact_root': str(tmp_path)}))
    campaign.write_text(json.dumps({'collection_id': 'observer-dtrs', 'target_games': 2,
                                    'seed_pool_size': 2, 'call_limit': 40}))
    provenance = {name: 'fixture' for name in ('served_model_name', 'serve_config_sha256', 'model_manifest_sha256', 'hf_revision')}
    provenance['runtime_config_sha256'] = sha256_bytes(canonical_json_bytes(collect_games.cli._runtime(collect_games.RUNTIME)))
    monkeypatch.setattr(collect_games, 'inspect_inputs', lambda: (provenance, 'mock'))
    monkeypatch.setattr(collect_games, 'clean_head', lambda: '2' * 40)
    monkeypatch.setattr(collect_games, 'live_preflight', Mock())
    monkeypatch.setattr(backends, 'load_named_backends', lambda *a, **k: {})

    class FailedRuntime:
        agents = ()
        recorder = SimpleNamespace(runtime_configuration=new_config())

        def __init__(self, plan, claim):
            self.game_id = f'{plan.collection_id}-game-{claim.ordinal:06d}-seed-{claim.seed}'

        def run(self):
            self.online_pilot({'step_index': 8, 'status': 'no_certified_feasible_candidate'})
            raise CanonicalAttemptFailure(game_id=self.game_id, stage=CanonicalFailureStage.RUNTIME,
                error_category='SpeechDecisionUnavailable', error_message='no_certified_feasible_candidate', retry_exhausted=False)

    monkeypatch.setattr(production_runtime.Classic7RuntimeFactory, '__call__',
                        lambda self, *, plan, claim: FailedRuntime(plan, claim))
    method = {'method': 'tom_3wd', 'delta_exact': '0'}
    result = collect_games.run_campaign(campaign, storage_profile=storage,
        online_policy=(lambda *, agents, record_evidence: record_evidence, method))
    dest = tmp_path / 'canonical' / 'observer-dtrs'
    plan = collect_games.load_plan(dest / 'collection_plan.json')
    state = validate_attempt_ledger(dest / 'attempt_ledger', plan)
    assert result == 1 and len(state.claims) == len(state.terminals) == 2
    assert {t.outcome.value for t in state.terminals} == {'canonical_failure'}
    assert not (dest / 'games').exists()
    assert json.loads((dest / 'speech-method.json').read_bytes()) == {'plan_digest': plan.plan_digest, **method}
    for claim in state.claims:
        record = dest / 'restricted' / 'speech-evidence' / claim.attempt_id / '000008.json'
        assert json.loads(record.read_bytes())['status'] == 'no_certified_feasible_candidate'
