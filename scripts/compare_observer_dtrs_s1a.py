"""Artificial, paired G/pi comparison. No external inputs or model services.

The fixture initializes legal PREs; only pure selection is evaluated here.
Cognitive-risk argmin and its explicit D3 region are equivalent controls.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

from scripts.compare_observer_dtrs_synthetic import scenarios, synthetic_model
from scripts.speech_planning import generate_candidates
from tests.observer_dtrs.test_speech_runtime import EVIDENCE, POLICY, runtime, source
from werewolf.artifact_io import canonical_json_bytes, sha256_bytes
from werewolf.observer_dtrs.context import contexts_for_pre
from werewolf.observer_dtrs.contextual_policy import contextual_loss, estimate_event
from werewolf.observer_dtrs.decision import decide
from werewolf.observer_dtrs.speech_evidence import (
    ObserverCognition, SelectionMode, process_cognition, select_observer, select_speech,
)

VERSION = 'observer_dtrs_s1a_paired_selection_v1'
COSTS = ('static', 'contextual')
ARMS = tuple((cost, mode) for cost in COSTS for mode in SelectionMode)
CAVEAT = ('Artificial selection-only controls; no true Y, measured speech payoff, '
          'calibration or game effect. Continuous cognitive argmin equals its D3 '
          'region with the same exact risks and tie order. No incremental benefit '
          'is identified by that equivalent representation comparison.')


def compare_scene(scene, protocol, model):
    env, recorder, _ = runtime(scene['phase'], scene['target'])
    pre = recorder._pending.prefix
    selection = select_observer(pre, ('player1', 'player2'))
    context = next(c for c in contexts_for_pre(pre, ('player1', 'player2'))
                   if c.observer_id == selection.observer_id)
    prediction, _ = source(pre, scene['q_r']).probabilities(pre)
    features, estimate, _ = estimate_event(context, prediction, model=model, protocol=protocol,
                                          loss=POLICY.static_loss)
    candidates = generate_candidates(pre)
    pairing = dict(prefix_digest=pre.prefix_digest,
        q_digest=sha256_bytes(canonical_json_bytes(asdict(prediction))),
        features=features.to_record(), probability=estimate['probability'],
        model_digest=model.digest, selected_observer=selection.observer_id,
        strategy=EVIDENCE.to_record(), candidates=[p.public_payload() for p in candidates],
        inference_budget={'tom_predictions': 1, 'event_probability_predictions': 1,
                          'language_calls': 0, 'commits': 0})
    rows = []
    for cost, mode in ARMS:
        loss = POLICY.static_loss if cost == 'static' else contextual_loss(
            context, tie_policy=POLICY.static_loss.tie_policy)
        cognition = ObserverCognition(context, features, model.digest, estimate['level'],
                                      loss, decide(estimate['probability'], loss))
        evidence = process_cognition(pre, panel=(cognition,))
        chosen = select_speech(pre, evidence, candidates=candidates, strategy=EVIDENCE, mode=mode)
        rows.append(dict(scene_id=scene['scene_id'], cost=cost, mode=mode.value,
            pairing=pairing, cognitive_processing=evidence.to_record(), selection=chosen.to_record()))
    return rows


def summarize(rows):
    expected = {(s['scene_id'], c, m.value) for s in scenarios() for c, m in ARMS}
    keys = [(r['scene_id'], r['cost'], r['mode']) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('incomplete or duplicate scene/arm coverage')
    groups = {s['scene_id']: [r for r in rows if r['scene_id'] == s['scene_id']] for s in scenarios()}
    if any(len({sha256_bytes(canonical_json_bytes(r['pairing'])) for r in group}) != 1
           for group in groups.values()):
        raise ValueError('unpaired PRE/Q/p/model/candidate/strategy/budget')
    contrasts = []
    pairs = [(('static', SelectionMode.CONTINUOUS_RISK), ('static', SelectionMode.EXPLICIT_THREE_WAY)),
             (('contextual', SelectionMode.CONTINUOUS_RISK), ('contextual', SelectionMode.EXPLICIT_THREE_WAY)),
             (('static', SelectionMode.CONTINUOUS_RISK), ('contextual', SelectionMode.CONTINUOUS_RISK)),
             (('static', SelectionMode.EXPLICIT_THREE_WAY), ('contextual', SelectionMode.EXPLICIT_THREE_WAY))]
    for left, right in pairs:
        matches = [(next(r for r in group if (r['cost'], r['mode']) == (left[0], left[1].value)),
                    next(r for r in group if (r['cost'], r['mode']) == (right[0], right[1].value)))
                   for group in groups.values()]
        contrasts.append(dict(left=[left[0], left[1].value], right=[right[0], right[1].value],
            paired_scenes=len(matches),
            region_flips=sum(a['selection']['cognitive_branch'] != b['selection']['cognitive_branch'] for a, b in matches),
            intent_changes=sum(a['selection']['selected_plan'] != b['selection']['selected_plan'] for a, b in matches)))
    return dict(version=VERSION, scope='ARTIFICIAL_SELECTION_ONLY', status='PASS',
        scenes=30, rows=len(rows), complete_grid=True, pairing_ok=True, caveat=CAVEAT,
        arms=[dict(cost=c, mode=m.value,
            regions=dict(Counter(r['selection']['cognitive_branch'] for r in rows if (r['cost'], r['mode']) == (c, m.value))),
            intents=dict(Counter(r['selection']['selected_plan']['action'] for r in rows if (r['cost'], r['mode']) == (c, m.value))))
              for c, m in ARMS], contrasts=contrasts)


def run(output):
    output = Path(output)
    if (not output.is_absolute() or output != output.resolve() or not output.parent.is_dir()
            or any((p / '.git').exists() for p in (output, *output.parents))):
        raise ValueError('new canonical absolute output outside Git with an existing parent required')
    output.mkdir()  # Exclusive, existing result directories are never overwritten.
    protocol, model = synthetic_model()
    root = Path(__file__).resolve().parents[1]
    paths = ['scripts/compare_observer_dtrs_s1a.py', 'scripts/compare_observer_dtrs_synthetic.py',
             'scripts/speech_planning.py', 'werewolf/observer_dtrs/speech_evidence.py',
             'werewolf/observer_dtrs/speech_runtime.py', 'werewolf/observer_dtrs/contextual_policy.py',
             'tests/observer_dtrs/test_speech_runtime.py']
    plan = dict(version=VERSION, scenarios=scenarios(), arms=[[c, m.value] for c, m in ARMS],
        strategy=EVIDENCE.to_record(), static_loss=POLICY.static_loss.to_record(),
        contextual_rule='existing contextual_loss, unchanged', model_digest=model.digest,
        protocol_digest=protocol.digest, scope='ARTIFICIAL_SELECTION_ONLY', caveat=CAVEAT,
        source=dict(base_head=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
                    files={p: sha256_bytes((root/p).read_bytes()) for p in paths}),
        budget='one shared Q and p per scene; four pure selections; no language/commit calls',
        limitations=['two living wolves in the paired grid; single-wolf unit tests separate',
                     'selected-only panel preserves V2 inference budget', 'not a frozen experiment'])
    (output/'plan.json').write_bytes(canonical_json_bytes(plan))  # Full grid before comparison.
    rows = []
    with (output/'rows.jsonl').open('x') as log:
        for scene in plan['scenarios']:
            for row in compare_scene(scene, protocol, model):
                rows.append(row)
                log.write(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n')
                log.flush()
    summary = summarize(rows)
    (output/'summary.json').write_bytes(canonical_json_bytes(summary))
    files = {p.name: dict(bytes=p.stat().st_size, sha256=sha256_bytes(p.read_bytes())) for p in output.iterdir()}
    (output/'manifest.json').write_bytes(canonical_json_bytes(dict(version=VERSION, files=files)))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    print(json.dumps(run(parser.parse_args().output), ensure_ascii=False, sort_keys=True))
