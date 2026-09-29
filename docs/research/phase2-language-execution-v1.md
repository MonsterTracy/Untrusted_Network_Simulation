# Phase-2 Language Realization and Execution Verification V1

This is an independent sidecar after frozen Action Contract V1. It does not change canonical V1 speech annotation, the offline layer, mapper, completed NoToM/+ToM gameplay, or actual vote generation. It has no environment commit path.

## Flow and trust boundary

`Phase2SemanticPlanV1 -> Phase2LanguageActorV1 -> public text -> Phase2SemanticPerceiverV1 -> verify_language_execution`.

The plan is the actor's source of truth. The actor is told to express only its frozen public strategy. The independent perceiver's API accepts only text and `PublicLanguageContextV1`; it has no plan argument. Its public context contains current phase, speaker, alive players, public competition targets, and public speeches already present in the current PRE. `public_language_context_from_pre` validates the source PRE and its identity against the action context. It does not provide wolf-team knowledge, role truth, hidden ballots, later speech, or future outcomes.

`Phase2SpeechSemanticV1` stores **plural** commitment targets, rejected targets, vote-intent targets, and information requests so conflicting or double commitments remain visible. It also records explicit abstention and claimed private facts. `action_identity` is derived from these fields, not supplied as a label by the perceiver. The JSON transport is strict and versioned; malformed outputs fail closed.

## Executed public semantics

- **Push(j):** exactly one perceived main commitment and vote intent, both j; no rejection, request, abstention, or private-fact claim. No wolf-role assertion is required.
- **Redirect(j,k):** exactly one perceived rejection j, commitment k, and vote intent k; no request or abstention. Rejection means “not this round's main exile target,” not “j is good.” k is already selected by frozen Action Contract V1.
- **Probe(j):** exactly one perceived `CURRENT_SUSPICION_BASIS` request addressed to and about j; no exile commitment, vote intent, rejection, or abstention. Empty vote intent alone does not count as Probe.

`verify_language_execution` first checks the frozen structured plan against the legal context, then compares independent perception. It returns a specific reason code for a mismatch. Language validity says only that the **current request was expressed**; it does not say j answered, a new observation occurred, reconsideration happened, or information was gained.

## Bounded retry and audit

`realize_verify_action` makes at most two attempts: generate, perceive, verify, then optionally regenerate once using the **same plan and public context** plus only the previous failure reason. It never changes the action or target and does not silently fall back. On exhaustion it returns no speech for a future commit adapter. A transport exception outside the contract remains an exception; malformed semantic perception and empty/truncated model outputs fail closed.

`Phase2LanguageAuditV1` embeds the existing structured action audit and adds separate `structured_execution_valid` and `language_execution_valid` fields, requested-plan digest, every attempt's generated text/digest and perceived semantics/digest, and a final invalid reason. Later response, observation, reconsideration, and information-gain fields remain null. The audit is private and only serializes in memory; this phase publishes no formal artifact.

## Future integration seam

A later isolated `ToM+TWD` runner may, at an authentic wolf speech PRE, compute frozen Q and runtime `p_tilde`, select a plan via a future router, call `realize_verify_action`, and publish only its verified speech through a separately designed commit adapter. The existing `run_random.py` default loop and later actual vote recomputation remain unchanged. No such router, integration, gameplay experiment, or real LLM call is part of this V1.
