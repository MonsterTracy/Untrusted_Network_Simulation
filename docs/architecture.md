# Phase-1 architecture

The frozen authority is CONTEXT.md, ADRs 0001–0022 and the approved specification.

| Boundary | Authoritative owner | Allowed output |
| --- | --- | --- |
| Game execution | env + gameplay agents + named backends | Public events and restricted replay evidence |
| Canonical collection | canonical_collection | PRE prefixes, observations, handoff, V1, bundles, attempt ledger |
| Public semantic tokens | structured_history | Pure ordered token plan, states, count, digest |
| Development publication | development_publication | Verified public records, deterministic five folds, restricted Role Sidecar |
| Tensorization / targets | tom.dataset | Full public tensors, q, observed/alive audit metadata |
| Role-based eligibility | tom.population | Boolean Primary masks only |
| All-Alive eligibility | tom.population | Public alive masks, no sidecar read |
| Protocol preparation | tom.experiment + protocol + state + temporal | Immutable schedules, masks, canonical states and temporal bytes |
| Primary training | tom.training | Fixed-budget terminal states and recovery chain |
| Prediction / scoring | tom.evaluation + scoring + reporting | One held-out prediction artifact, two named reports, paired game summaries |

Publications validate Collection's records without reconstructing PRE histories,
reparsing speech or tensorizing. The one pure token planner is shared for
Publication count-only statistics and Dataset numeric materialization.

Role Sidecar is physically restricted and provenance-bound. Its role semantics
terminate in the Primary selector. The Dataset does not select an estimand.
Eligibility and label status control supervision, never representation.

All five folds and both temporal conditions train before the checkpoint-set
manifest seals all ten terminal digests. Only then are held-out records opened.
Implicit and Explicit share trainable graph, canonical initial state, game order,
rotations, batches, RNG and optimizer protocol. Their only intended difference
is a deterministic temporal code versus zero at the same injection point.

Runtime internal role-action phases are not public temporal states. The five
public phases are night, discussion, vote, pk_discussion and pk_vote.

Final Fit is a separate all-development artifact lifecycle, specified in
[ADR 0022](adr/0022-seal-paired-final-models-before-independent-evaluation.md).
It reuses the same public tensorizer, model, optimizer loop and scoring functions.
`final_experiment` owns pre-evaluation derived capacity and preparation;
`final_training` owns paired terminal recovery/sealing; `final_publication`
owns the independent data projection; `final_evaluation` owns seal-gated public
inference, immutable consumption and paired reports. No fake fold or changed
OOF schema is introduced. See [the final lifecycle](final-lifecycle.md).
