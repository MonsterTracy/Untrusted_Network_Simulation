# Phase 3A: existing Actor realization and exact semantic verification

Baseline: 8886af9505036b55662d54ae9a108b09121e8c71.

## Existing path and adapter

GPTAgent._act normally calls _generate_day_cognition, compiles its selection
into DiscussionAct values, then invokes _generate_public_speech. The latter
retains _retry_validated_generation, the existing realization prompt and local
public-speech quality validation. Its prompt sees only speaker/day/phase,
PublicClaim raw public speeches, and frozen DiscussionAct intent.

scripts/speech_realization.py invokes that same realization method directly,
without cognition or any changes under werewolf. Transport is exactly the
existing SpeechPlan.public_payload(): action and target. It intentionally does
not import speech_planning, which imports the sealed consumer. No IPC is built.

The six mappings are ACCUSE_WOLF -> point_as_werewolf(target), CLEAR ->
point_as_non_werewolf(target), SUPPORT -> support(target), OPPOSE ->
oppose(target), SELF_DEFEND -> point_as_non_werewolf(speaker), NO_COMMITMENT ->
no_commitment(None). Each yields exactly one canonical V1SpeechAction and one
existing DiscussionAct (whose target uses integer seats).

PublicRealizationContext contains only canonical speaker, day, public phase and
PublicClaim tuple. The caller must supply authentic public claims. No full
private observation is accepted. Same public payload/context and Actor settings
produce identical initial Actor requests regardless of selector origin. No
score, Delta, wolf membership, ranking or seal metadata enters the interface.

## Perception, verification and failures

realize_parse_verify calls parse_with_audit exactly once after Actor realization.
The deterministic verifier checks parse_status=ok, absent errors, and exact list
equality to the singleton expected action. NONE/empty is not NO_COMMITMENT.
Wrong targets/actions, additional actions, role/skill/vote claims all fail.
Nothing is dropped, repaired or remapped. There is no semantic retry, candidate
switch, alternate selector or fallback.

The successful result carries the exact same SpeechParseAuditResult object with
speech and expected action. SemanticVerificationError retains that same audit on
failure. The verifier never parses or mutates it. Existing Actor quality retries
remain; existing parser-internal bounded retries remain. One perception API call
does not imply one parser backend generation on parse failure.

## Compatibility and Phase 3B blockers

Original Agent methods, templates, env commit and run_random.py are unchanged.
No sealed runtime validation is bypassed. The new module imports neither planner
nor sealed consumer. It performs no canonical artifact construction or commit.

Phase 3B must bind selected payload/public context to the authentic opportunity,
check eligibility, and route the retained perception into the real commit path
without re-parsing. This module alone cannot prevent a future caller from
incorrectly parsing again. The audit object's nested lists are mutable in the
existing API; future commit binding must guard against intervening mutation.
No result here is canonical evidence or an authorization to commit. Frozen ToM
and gameplay runtime separation/transport remains future work. Parser agreement
is a semantic acceptance criterion, not proof of natural-language truth.

## Tests and server commands

NOT RUN. Scripted backends use real Actor realization and real SpeechPerceiver.
Tests cover six mappings, deterministic intent, identical requests versus both
selection origins and direct Original realization, private-field rejection,
exact/mismatched/extra semantics, NONE, parser failure, same audit identity,
no mutation, one parse API invocation, no semantic regeneration, and fresh-process
absence of planner/sealed-consumer imports.

Run from the server repository root:

```sh
python -m pytest -q tests/tom/test_speech_realization.py
python -m pytest -q tests/tom/test_speech_planning.py
python -m pytest -q tests/tom/test_counterfactual_consumer.py tests/tom/test_suspicion_objectives.py
python -m pytest -q tests/agents/test_agent_backend.py
python -m pytest -q
```
