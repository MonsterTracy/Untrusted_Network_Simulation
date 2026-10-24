"""Fixed 30-scene / three-condition synthetic verified-speech CPU comparison.

Repository/dev checkout only: reuses the existing synthetic runtime test fixture.
No input-data option, online predictor, formal estimator, LLM/network or GPU call.
Reported reference risks are plug-in cognitive risks, not measured game utility.
"""
import argparse
from collections import Counter
from dataclasses import asdict
from fractions import Fraction as F
import csv
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.observer_dtrs.test_speech_runtime import POLICY, runtime, source
from werewolf.agents.gpt_agent import GPTAgent
from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.canonical_collection.call_audit import audited_backends
from werewolf.observer_dtrs.context import contexts_for_pre
from werewolf.observer_dtrs.contextual_policy import contextual_loss
from werewolf.observer_dtrs.d2_protocol import frozen_protocol
from werewolf.observer_dtrs.decision import Region, conditional_risks, decide, thresholds
from werewolf.observer_dtrs.estimation import TrainingRow, fit_model
from werewolf.observer_dtrs.estimation_offline import implementation_digest
from werewolf.observer_dtrs.features import ObserverFeatures
from werewolf.observer_dtrs.speech_runtime import Comparison, execute_pending_speech, select_observer


VERSION = 'synthetic_paired_speech_comparison_v1'
PHASES = ('discussion', 'pk_discussion')
EVIDENCE = {'self_accused': 'player1', 'team_accused': 'player2', 'no_accusation': None}
R_VALUES = (.1, .3, .5, .7, .9)
P_VALUES = (F(1, 8), F(1, 4), F(1, 2), F(3, 4), F(7, 8))
CAVEAT = ('Under the same p and static D3 reference loss, static_d3 is the argmin by '
          'definition. This comparison cannot establish a benefit of contextual costs. '
          'These are estimated cognitive risks on an equally weighted synthetic grid, '
          'not observed losses, language effectiveness, information gain or win rates.')


def scenarios():
    return [dict(scene_id=f'{phase}-{evidence}-q{i}', phase=phase, evidence=evidence,
                 target=target, q_r=r, expected_p_exact=str(P_VALUES[i]))
            for phase in PHASES for evidence, target in EVIDENCE.items()
            for i, r in enumerate(R_VALUES)]


def synthetic_model():
    """One fixed real D2 fit; artificial labels exist ONLY in this fixture fit.

    12 distinct one-row games/cell, smoothing=4, parent prior=1/2. Positive
    counts 0,2,6,10,12 give exactly 1/8,1/4,1/2,3/4,7/8 after shrinkage.
    No fixture Y or whole-role table is passed to execute_pending_speech.
    """
    protocol = frozen_protocol()
    rows = []
    for phase in PHASES:
        for i, (r, positives) in enumerate(zip(R_VALUES, (0, 2, 6, 10, 12))):
            for j in range(12):
                game = f'synthetic-paired-training-{phase}-{i}-{j}'
                rows.append(TrainingRow(game, game, ObserverFeatures(r/2, r/2, r, 0, phase, 2),
                                        int(j < positives)))
    provenance = dict(scope='controlled_fixture', protocol_digest=protocol.digest,
        dataset_digest=None, split_manifest_digest=None, tom_fold=None, outer_fold=None,
        inner_fold=None, game_sizes={row.game_id: 1 for row in rows},
        origins={'fixture': VERSION}, implementation_digest=implementation_digest())
    model = fit_model(rows, protocol=protocol, model_id='OBS-EQ1', bins=5, smoothing=4,
                      source=provenance)
    for phase in PHASES:
        for r, p in zip(R_VALUES, P_VALUES):
            assert F(model.predict(ObserverFeatures(r/2, r/2, r, 0, phase, 2))) == p
    return protocol, model


class ControlledActor:
    """Read only the existing public intent block; deterministic grammar, not LLM."""
    def __init__(self):
        self.calls = []
        self.speeches = []

    def chat_with_metadata(self, *, messages, **_kwargs):
        prompt = messages[-1]['content']
        self.calls.append(prompt)
        intent = prompt.split('本轮必须完整表达、且只能表达以下冻结意图：\n', 1)[1].split('写作约束：', 1)[0].strip()
        if intent.startswith('- 指认为非狼人：'):
            target = re.search(r'player[1-7]', intent).group()
            text = f'我认为{target}不是狼人。'
        elif intent.startswith('- 公开质疑：'):
            target = re.search(r'player[1-7]', intent).group()
            text = f'我质疑{target}此前的判断。'
        elif intent.startswith('- 不作明确表态：'):
            text = '我暂时不作明确表态。'
        else:
            raise ValueError('controlled generator received unsupported intent')
        self.speeches.append(text)
        return text, {'finish_reason': 'stop'}


class ControlledParser:
    """Independent grammar reads raw speech only, never the selected plan or p."""
    def __init__(self):
        self.calls = []
        self.responses = []

    def chat(self, *, messages, **_kwargs):
        prompt = messages[-1]['content']
        self.calls.append(prompt)
        speech_line = prompt.split('待解析的玩家发言：\n', 1)[1]
        speaker, text = speech_line.split(': ', 1)
        clear = re.fullmatch(r'我认为(player[1-7])不是狼人。', text)
        oppose = re.fullmatch(r'我质疑(player[1-7])此前的判断。', text)
        if clear:
            action, target = 'point_as_non_werewolf', clear[1]
        elif oppose:
            action, target = 'oppose', oppose[1]
        elif text == '我暂时不作明确表态。':
            action, target = 'no_commitment', 'NONE'
        else:
            raise ValueError('speech outside the controlled parser grammar')
        response = f'{speaker} | {action} | {target}'
        self.responses.append(response)
        return response


def run_one(scene, condition, protocol, model):
    # Test-only fixture builds legal public PRE/handoff. Runtime sees only its
    # current actor's filtered observation, not the fixture's synthetic roles.
    env, recorder, _ = runtime(scene['phase'], scene['target'])
    pre = recorder._pending.prefix
    prediction = source(pre, scene['q_r'])
    team = ('player1', 'player2')  # Synthetic setup/audit only, not an extra runtime input.
    selection = select_observer(pre, team)
    evidence = [dict(event=e.to_record(), annotation=next(
        (a.to_record() for a in pre.v1_annotations if a.event_id == e.event_id), None))
        for e in pre.public_event_history.events if e.event_type == 'public_speech']
    actor_backend, parser_backend = ControlledActor(), ControlledParser()
    backends = audited_backends({'actor': actor_backend, 'parser': parser_backend}, recorder.call_audit)
    actor = GPTAgent(backend=backends['actor'], model_name='controlled-grammar-only')
    env.speech_perceiver.backend = backends['parser']  # Same real perceiver and protocol.
    before = len(recorder.submitted_gameplay_actions)
    rng_before = env._rng.getstate()
    row = dict(scene=scene, condition=condition.value, status='failed', audit=None,
        pairing=dict(prefix_digest=pre.prefix_digest,
            q_digest=sha256_bytes(canonical_json_bytes(asdict(prediction.matrix))),
            selection=asdict(selection), model_digest=model.digest, policy=POLICY.to_record(),
            env_rng_digest=sha256_bytes(canonical_json_bytes(rng_before))),
        public_evidence=evidence, parse_consistent=False, committed=False)
    try:
        execution = execute_pending_speech(env=env, recorder=recorder, actor=actor,
            source=prediction, model=model, protocol=protocol, condition=condition, policy=POLICY)
        row['audit'] = execution.audit
        annotation = env.speech_annotations[-1]
        row['parse'] = annotation.to_record()
        row['parse_consistent'] = annotation.is_successful and len(actor_backend.calls) == len(parser_backend.calls) == 1
        row['committed'] = recorder._pending is None and len(recorder.submitted_gameplay_actions) == before+1
        assert F(execution.audit['decision']['probability_exact']) == F(scene['expected_p_exact'])
        assert env._rng.getstate() == rng_before
        assert row['parse_consistent'] and row['committed']
        p = F(execution.audit['decision']['probability_exact'])
        risks = conditional_risks(p, POLICY.static_loss)
        selected_risk = risks[Region(execution.audit['decision']['region'])]
        row['reference_risks_exact'] = {a.value: str(v) for a, v in risks.items()}
        row['reference_selected_risk_exact'] = str(selected_risk)
        row['reference_excess_exact'] = str(selected_risk-min(v for _, v in risks.items()))
        row['status'] = 'ok'
    except Exception as error:
        # Preserve every planned row. Never silently drop a failed scene/condition.
        row['error'] = {'type': type(error).__name__, 'message': str(error)}
    row['speech'] = actor_backend.speeches
    row['parser_responses'] = parser_backend.responses
    return row


def tie_diagnostics():
    """Exact rational oracle is separate from float-D2 / verified execution."""
    output = []
    for phase in PHASES:
        env, recorder, _ = runtime(phase)
        context = contexts_for_pre(recorder._pending.prefix, ('player1', 'player2'))[0]
        losses = {'static': POLICY.static_loss,
                  'contextual': contextual_loss(context, tie_policy=POLICY.static_loss.tie_policy)}
        for name, loss in losses.items():
            for boundary in ('beta', 'alpha'):
                value = getattr(thresholds(loss), boundary)
                for side, delta in (('below', -F(1, 1024)), ('tie', F(0)), ('above', F(1, 1024))):
                    output.append(dict(scope='RATIONAL_D3_ONLY_NOT_D2_OR_GAMEPLAY', phase=phase,
                        loss_name=name, loss=loss.to_record(), boundary=boundary,
                        boundary_exact=str(value), side=side, decision=decide(value+delta, loss).to_record()))
    return output


def summarize(rows):
    """Fixed denominators; incomplete runs have no risk mean or paired estimate."""
    expected = {(s['scene_id'], c.value) for s in scenarios() for c in Comparison}
    actual = [(r['scene']['scene_id'], r['condition']) for r in rows]
    complete = len(actual) == len(expected) and set(actual) == expected
    groups = {s['scene_id']: [r for r in rows if r['scene']['scene_id'] == s['scene_id']] for s in scenarios()}
    pairing_ok = complete and all(len({json.dumps(r['pairing'], sort_keys=True) for r in group}) == 1
                                   for group in groups.values())
    valid = pairing_ok and all(r['status'] == 'ok' for r in rows)
    conditions = {}
    for condition in Comparison:
        subset = [r for r in rows if r['condition'] == condition.value]
        ok = [r for r in subset if r['status'] == 'ok']
        conditions[condition.value] = dict(planned=30, recorded=len(subset), passed=len(ok),
            region_counts=dict(Counter(r['audit']['decision']['region'] for r in ok)),
            intent_counts=dict(Counter(r['audit']['public_payload']['action'] for r in ok)),
            legal_commits=sum(r['committed'] for r in subset),
            legal_commit_rate=sum(r['committed'] for r in subset)/30,
            parse_consistency_rate=sum(r['parse_consistent'] for r in subset)/30,
            mean_reference_risk_exact=str(sum((F(r['reference_selected_risk_exact']) for r in ok), F())/30) if valid else None)
    contrasts = []
    if valid:
        for left, right in ((Comparison.STATIC_D3, Comparison.CONTEXTUAL_LOSS),
                            (Comparison.CONTEXTUAL_LOSS, Comparison.CONTEXTUAL_SPEECH),
                            (Comparison.STATIC_D3, Comparison.CONTEXTUAL_SPEECH)):
            pairs = [(next(r for r in group if r['condition'] == left.value),
                      next(r for r in group if r['condition'] == right.value)) for group in groups.values()]
            transitions = Counter(a['audit']['decision']['region']+'->'+b['audit']['decision']['region'] for a, b in pairs)
            contrasts.append(dict(left=left.value, right=right.value, paired_scenes=30,
                region_flips=sum(a['audit']['decision']['region'] != b['audit']['decision']['region'] for a, b in pairs),
                intent_changes=sum(a['audit']['public_payload'] != b['audit']['public_payload'] for a, b in pairs),
                region_transitions=dict(transitions),
                mean_reference_risk_difference_exact=str(sum((F(b['reference_selected_risk_exact'])-F(a['reference_selected_risk_exact']) for a, b in pairs), F())/30)))
    return dict(version=VERSION, scope='SYNTHETIC_CONTROLLED_ONLY', status='PASS' if valid else 'FAIL',
        complete_grid=complete, pairing_ok=pairing_ok, reference_caveat=CAVEAT,
        conditions=conditions, contrasts=contrasts)


def write_csv(path, rows):
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output):
    output = Path(output)
    if not output.is_absolute() or output != output.resolve() or not output.parent.is_dir():
        raise ValueError('require canonical absolute output path with an existing parent')
    if any((p/'.git').exists() for p in (output, *output.parents)):
        raise ValueError('output must be outside a Git checkout')
    output.mkdir()  # Existing output always rejected; never overwrite inputs/results.
    protocol, model = synthetic_model()
    root = Path(__file__).resolve().parents[1]
    paths = ['scripts/compare_observer_dtrs_synthetic.py', 'tests/observer_dtrs/test_speech_runtime.py',
        'werewolf/observer_dtrs/speech_runtime.py', 'werewolf/observer_dtrs/contextual_policy.py',
        'scripts/speech_realization.py', 'scripts/verified_speech_commit.py',
        'tests/canonical_collection/test_runtime_game_evidence.py']
    plan = dict(version=VERSION, scope='SYNTHETIC_CONTROLLED_ONLY', scenarios=scenarios(),
        conditions=[c.value for c in Comparison], policy=POLICY.to_record(), model_digest=model.digest,
        protocol_digest=protocol.digest, reference_loss=POLICY.static_loss.to_record(),
        weighting='each of the 30 predeclared scenes has equal weight; no outcome filtering',
        comparison_caveat=CAVEAT,
        limitations=['two live wolves only', 'scripted generator/parser, single-speech test fixture',
                    'no causal/game/language-effect claim', 'PK alpha=3/7 exact tie only in rational D3 diagnostics; float D2 brackets it'],
        source=dict(base_head=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
                    files={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}))
    def save(name, value):
        (output/name).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+'\n')
    save('plan.json', plan)  # Written before any condition executes; defines the full grid/config/costs.
    save('execution.json', dict(python=platform.python_version(), platform=platform.platform(),
                               output_path=str(output), path_role='local_execution_provenance_only'))
    save('synthetic-model.json', model.record)
    rows = []
    with (output/'rows.jsonl').open('x') as log:
        for scene in plan['scenarios']:
            for condition in Comparison:
                row = run_one(scene, condition, protocol, model)
                rows.append(row)
                log.write(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n')
                log.flush()
    summary = summarize(rows)
    summary['plan_digest'] = sha256_bytes(canonical_json_bytes(plan))
    save('summary.json', summary)
    save('rational-ties.json', tie_diagnostics())
    flat = []
    for row in rows:
        audit = row['audit'] or {}
        decision = audit.get('decision', {})
        flat.append(dict(scene_id=row['scene']['scene_id'], phase=row['scene']['phase'], evidence=row['scene']['evidence'],
            condition=row['condition'], p=decision.get('probability_exact'), costs=json.dumps(audit.get('loss', {}).get('exact_costs', {}), sort_keys=True),
            thresholds=json.dumps(decision.get('thresholds', {}).get('exact_values', {}), sort_keys=True),
            region=decision.get('region'), intent=json.dumps(audit.get('public_payload', {}), sort_keys=True),
            observer=row['pairing']['selection']['observer_id'], public_event=row['pairing']['selection']['event_id'],
            reference_risk=row.get('reference_selected_risk_exact'), parsed=json.dumps(row.get('parse', {}), ensure_ascii=False, sort_keys=True),
            committed=row['committed'], status=row['status']))
    write_csv(output/'results.csv', flat)
    table = ['| condition | NEG | BND | POS | reference risk | legal commits | parse consistency |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, counts in summary['conditions'].items():
        regions = counts['region_counts']
        table.append(f"| {name} | {regions.get('NEG', 0)} | {regions.get('BND', 0)} | {regions.get('POS', 0)} | {counts['mean_reference_risk_exact']} | {counts['legal_commits']}/30 | {counts['parse_consistency_rate']} |")
    (output/'RESULTS.md').write_text('\n'.join(table)+'\n\n'+CAVEAT+'\n\n'+
        'Differences are descriptive fixture checks. See summary.json for paired flips and intent changes; rows.jsonl for public evidence and verified annotations.\n')
    save('sha256.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='new canonical absolute directory outside all Git checkouts')
    args = parser.parse_args(argv)
    result = run(args.output)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
