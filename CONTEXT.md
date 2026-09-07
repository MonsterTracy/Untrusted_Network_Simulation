# Classic7 Observer-Conditioned ToM

This context defines the scientific language of the end-to-end Classic7 Theory of Mind research system. The repository owns trusted data generation and belief modeling, while gameplay control research remains outside its scope.

## Research System

**Classic7**:
A seven-player, multi-day, text-only Werewolf setting with two Werewolves, three Villagers, one Seer, and one Witch.
_Avoid_: ONUW setting, five-player setting

**Game Runtime**:
The canonical scientific data-generation infrastructure that executes Classic7 games from which trusted research observations are collected. It is not a gameplay-policy research program.
_Avoid_: Policy engine, control system

**Canonical Collection**:
The trusted research-data lineage that records causally valid Classic7 public history, PRE belief observations, semantic speech perception, and their provenance.
_Avoid_: Gameplay experiment, rollout benchmark

**Canonical Trajectory**:
A provenance-bound Classic7 game record that satisfies the Canonical Collection contract.
_Avoid_: Training sample, policy rollout

**Canonical Game Bundle**:
The immutable per-game evidence unit stating what happened, what was publicly visible, which Belief Observations occurred at PRE Boundaries, and whether the record is trustworthy. It is not a model-ready dataset or experiment split.
_Avoid_: Training dataset, fold, tensor bundle

**Private Replay Evidence**:
The private portion of a Canonical Game Bundle retained only for deterministic replay, canonical validation, and provenance-bound Role Sidecar derivation. It is never copied into published Belief Observation records or exposed to Dataset, tensorization, or model forward.
_Avoid_: Private model context, development record, hard-knowledge feature

**Canonical Eligibility**:
The fail-closed determination that a Canonical Game Bundle satisfies the trusted collection contract and may be considered by a Development Publication.
_Avoid_: Evaluation population, supervision eligibility

**Complete-Observation Canonicality**:
The canonical eligibility rule requiring a successful Belief Observation from every alive observer at every PRE Boundary. Any exhausted observer failure makes the game ineligible for Development Publication while preserving its failure evidence for selection-bias audits.
_Avoid_: Partial-row canonicality, missing-rate threshold, silent row dropping

**Canonical Failure Evidence**:
The immutable audit record of an exhausted collection failure, including its planned game identity, causal location, observer when applicable, error category, and retry evidence. It is not a fallback trajectory or an eligible Canonical Game Bundle.
_Avoid_: Pilot bundle, fallback game, replacement seed

**Canonical Collection Attempt Ledger**:
The append-only record that durably claims each planned seed before its first gameplay or backend call and closes the attempt as canonical success, normal failure, or interrupted failure. Once claimed, a seed is permanently consumed for that Collection Plan identity; resume continues only with the next unclaimed seed.
_Avoid_: Retry queue, replaceable seed list, operator-selected rerun

**Development Publication**:
An immutable research lineage that selects and verifies a declared set of canonically eligible games, publishes their public-history and Belief Observation records, game-level folds, Role Sidecar, and provenance manifest. It does not create belief targets, tensors, populations, models, or evaluation results.
_Avoid_: Collection directory, training run, arbitrary file selection

**Development Game Set**:
The complete set of canonically eligible successes produced by one predeclared collection plan when it reaches its target success count. It cannot be manually filtered or retrospectively sampled.
_Avoid_: Hand-picked game subset, train split, test split

**Development Fold Manifest**:
The deterministic, versioned assignment of every game in a Development Game Set to exactly one of five game-level OOF folds, bound to game identities and digests.
_Avoid_: Snapshot split, configurable fold count, sealed split

**Outer Selection Blindness**:
The OOF rule that games assigned to an outer held-out fold cannot influence epoch choice, early stopping, learning-rate adaptation, hyperparameters, configuration, model selection, or retry decisions. Their Primary and All-Alive evaluations begin only after the fold checkpoint identity and digest are frozen.
_Avoid_: Validation-fold OOF, best-on-held-out checkpoint, post-evaluation rerun

**Fixed-Budget Training Protocol**:
The predeclared fold-training contract that uses all outer-training games for an immutable number of epochs or optimizer updates and emits the terminal checkpoint. The full budget and protocol are frozen before any outer evaluation and are identical between corresponding Implicit and Explicit temporal lineages.
_Avoid_: Best checkpoint, early stopping, inner validation, post-result budget change

**Game-Balanced Belief Cross-Entropy**:
The Primary training objective that first averages belief-distribution cross-entropy over each training game's Primary effective-supervision rows, then gives every game in the minibatch equal weight. A training game with no such rows makes the operation fail closed.
_Avoid_: Pooled observer-row loss, all-alive training loss, silently skipped game

**Balanced Cyclic Seat Augmentation**:
The required training-only augmentation that gives every outer-training game balanced exposure to all seven cyclic player-ID shifts over its formal fixed budget. Its deterministic schedule is versioned, stable-identity based, and paired across temporal conditions; it never derives eligibility from role labels.
_Avoid_: Shuffle-coupled rotation, Dataset-index rotation, partial seat cycle

**Canonical Seat Evaluation**:
The evaluation rule that Primary and All-Alive predictions use the original player identifiers at cyclic shift zero. Test-time rotation ensembles are not part of the formal evaluation.
_Avoid_: Rotation-averaged evaluation, random evaluation shift

**Final Evaluation Publication**:
A possible future, separately preregistered scientific protocol over an independent game set. It is not part of the current development protocol and cannot be created by carving a test subset from a Development Publication.
_Avoid_: Current test split, sealed compatibility path

**ToM Mainline**:
The formal research lineage covering dataset publication, observer-conditioned belief modeling, training, and development out-of-fold evaluation. It never controls gameplay.
_Avoid_: Gameplay agent, policy model

## Scientific Observations

**PRE Boundary**:
The causal instant after the observer-visible Speaker Turn Boundary for the current speaker and before that speaker produces the corresponding public speech. The terminal `turn_start(current_speaker)` is part of the model-visible input; the subsequent public speech and all V1 Speech Actions derived from it are not.
_Avoid_: Post-speech snapshot, pre-turn snapshot, stripped turn boundary

**Speaker Turn Boundary**:
The authoritative public `turn_start(current_speaker)` event that terminates a canonical PRE input prefix and identifies the player about to speak. It is an already occurred public fact, not part of the speaker's forthcoming speech content.
_Avoid_: Audit-only delimiter, upcoming-speech feature, post-speech event

**Authoritative PRE Prefix**:
The single immutable Structured Public History prefix constructed by Canonical Collection after the Speaker Turn Boundary and before the current public speech. Development Publication, Dataset, and replay may validate or consume it without loss, but may not crop, complete, reorder, repair, or reinterpret it.
_Avoid_: Dataset-derived prefix, publication reconstruction, repaired PRE history

**Full-Prefix Tensorization**:
The Dataset rule that every token in an Authoritative PRE Prefix is represented in model input. The experiment's frozen `max_seq_len` is only a capacity validation bound and never a truncation, windowing, reset, compression, or token-dropping policy.
_Avoid_: Recent-history window, left truncation, daily tensor reset

**Maximum Structured Token Count**:
The largest token count of any Authoritative PRE Prefix in a Development Publication. A formal experiment must freeze a `max_seq_len` at least this large before training and fail closed on any violation.
_Avoid_: Observed truncated length, dynamic context extension, advisory maximum

**PRE Validation**:
A defensive check that an Authoritative PRE Prefix has the expected identity, ordering, cutoff, digest, speaker boundary, and speech exclusion without transforming the prefix. Multiple boundary-specific validations may exist without becoming competing constructors.
_Avoid_: PRE reconstruction, semantic slicing, fallback repair

**Structured Public History**:
The cumulative, cross-day chronological sequence of publicly observable Classic7 events and V1 semantic speech actions available before a PRE Boundary.
_Avoid_: Daily history, private observation history

**Belief Observation**:
An observer-specific PRE report of which players the observer suspects are Werewolves, together with whether the report was successfully observed.
_Avoid_: Role label, ground-truth belief

**Realized PRE Belief**:
The actual successful Belief Observation produced from a playing observer's legally available cognition at a PRE Boundary. It may reflect Seer or Witch private state even though those private facts are never model inputs.
_Avoid_: Public-only counterfactual belief, ToM prediction, role label

**Speaker PRE Belief**:
The current speaker's successful Belief Observation at a PRE Boundary, frozen as the sole wolf-belief cognition for the immediately following public speech generation.
_Avoid_: ToM prediction, second speech belief, post-speech belief

**Speaker PRE Belief Handoff**:
The canonical data-generation rule that passes the immutable Speaker PRE Belief back to the same playing agent for its next speech cognition. It is not a trained-ToM feedback or gameplay-policy mechanism.
_Avoid_: ToM control, policy feedback, belief resampling

**Successful Empty Belief Observation**:
A successful Belief Observation whose suspicion set is empty; it remains observed and denotes zero mass on self and equal mass on each of the other six players.
_Avoid_: Missing belief, failed belief

**Observer-Conditioned Belief**:
One observer's probability distribution over all seven players as Werewolf suspects at a PRE Boundary, inferred only from Structured Public History and the observer identity.
_Avoid_: Global belief, role-conditioned belief

**Non-Self Suspicion Simplex**:
The fixed output and target support for observer `i`: a seven-position player axis with exact zero at `i` and the other six positions summing to one. Successful non-empty support excludes self and is uniform over its members; Successful Empty Belief Observation is uniform over all six non-self players.
_Avoid_: Seven-way self-role belief, configurable diagonal mask, self-suspicion support

**Role Truth**:
The actual Classic7 role assignment, admissible only for selecting supervision rows and never as an input to the ToM model.
_Avoid_: Model role feature, private conditioning

**V1 Speech Action**:
The canonical semantic interpretation of one explicit public-speech proposition as a subject, action, and object triple.
_Avoid_: V2 auto-candidate, generator intent

**V1 Speech Perception**:
The sole authoritative collection-time transformation from one raw public speech into its frozen V1 Speech Actions and audited perception status. Development Publication may validate it and Dataset may consume it, but neither may reparse, repair, replace, or fall back.
_Avoid_: Publication-time parsing, Dataset parser, shadow replacement

**Successful No-Action Perception**:
A successful V1 Speech Perception whose parser explicitly reports that the public speech contains no extractable V1 Speech Action. It has `status=no_action` and an empty action list and does not impair Canonical Eligibility.
_Avoid_: Parser failure, missing annotation, empty fallback

**Speech Perception Failure**:
A V1 Speech Perception whose bounded attempts end in `status=error`. It preserves complete parser and backend failure evidence and makes the game canonically ineligible rather than being represented as an empty action list.
_Avoid_: No-action speech, repaired annotation, accepted missing action

## Evaluation

**Non-Wolf Alive Population**:
The alive observer rows whose Role Truth is not Werewolf at the corresponding PRE Boundary. It is the fixed Primary population, not a claim that every realized target is strictly recoverable from public information.
_Avoid_: Role diagnostic population, non-wolf diagnostic

**Primary Development OOF**:
The formal development out-of-fold evaluation of public-only ToM predictions against Realized PRE Beliefs in the Non-Wolf Alive Population. Its targets may contain latent Seer or Witch private-state effects.
_Avoid_: Formal OOF, diagnostic OOF, non-wolf diagnostic

**Game-Macro Estimand**:
The unique headline aggregation for Primary Development OOF and All-Alive Identifiability Stress: compute a score within each held-out game from that game's effective-supervision rows, then average the per-game scores with every game weighted equally. OOF folds produce predictions but are not statistical weighting units.
_Avoid_: Fold-macro result, observer-row-weighted primary, mixed-weight estimate

**Belief KL Headline Score**:
The unique headline scoring rule: calculate `D_KL(q || p)` for every effective-supervision row, average those divergences within each held-out game, then apply the Game-Macro Estimand. Lower is better.
_Avoid_: Cross-entropy headline, normalized-gap headline, support-accuracy primary

**Uniform Non-Self Reference**:
The required non-model reference assigning zero probability to the observer and equal probability to the other six players. It is scored on exactly the same effective-supervision rows and with the same game-macro aggregation as the model, but it does not define the headline score.
_Avoid_: Training prior, normalized-gap primary, unmatched baseline population

**Game-Cluster Bootstrap**:
The uncertainty procedure that resamples held-out games with replacement and recomputes the exact Game-Macro Estimand in every replicate. Its interval may accompany only the matching game-macro point estimate.
_Avoid_: Row bootstrap, fold bootstrap, game-macro interval on a row-weighted estimate

**Observer-Row-Weighted Diagnostic**:
A secondary result that weights every effective-supervision observer row equally across games. It must be named as a diagnostic and cannot serve as a second primary result or use a Game-Cluster Bootstrap interval.
_Avoid_: Observer-weighted primary, unlabeled pooled metric

**All-Alive Identifiability Stress**:
A population-expansion evaluation that applies Primary-trained fold checkpoints to every alive observer, incrementally adding Werewolves with deterministic teammate knowledge to a Primary population that may already contain Seer or Witch latent-private variation. All-alive rows never participate in training or model selection, and the result is not a primary model-quality benchmark.
_Avoid_: All-alive training, all-alive primary OOF, second primary benchmark, diagnostic OOF

**Identifiability Stress Penalty**:
The paired game-macro difference between All-Alive and Primary per-game mean belief KL under the same temporal condition and exact fold checkpoint: `K_all - K_primary`. Positive means greater error after population expansion; negative values remain valid. Its interval comes from paired game-cluster bootstrap.
_Avoid_: Stress ratio, clipped penalty, difference of independent intervals

**Role Sidecar**:
A provenance-bound truth artifact containing complete role assignments for development games, used only to define an observer population. It is never part of public history, a training sample, a model-visible batch, or model state.
_Avoid_: Role feature, role metadata, private model input

**Population Selector**:
The sole semantic consumer and propagation endpoint of Role Truth, converting a Role Sidecar and public alive state into observer eligibility without exposing role identities downstream.
_Avoid_: Scope switch, role filter in Dataset

**Observer Eligibility Mask**:
Supervision-control metadata stating whether each observer row belongs to an operation's fixed population. It controls loss, evaluation inclusion, and provenance audits, never prediction or representation computation.
_Avoid_: Model mask, attention mask, role mask

**Canonical ToM Sample**:
The unique public-only learning record derived from one canonical PRE history and its observer-specific Belief Observation. It contains model-visible public tensors, the current belief target, observation status, public alive state, and causal audit metadata, but no evaluation-population or role semantics.
_Avoid_: Population-filtered sample, private-conditioned sample

**Observer Alive State**:
The publicly observable fact that an observer is alive at a PRE Boundary. It describes sample state and does not by itself define an evaluation population.
_Avoid_: Observer eligibility, supervision scope

**Effective Supervision Mask**:
The operation-owned inclusion mask formed from an Observer Eligibility Mask and successful Belief Observation status. It is consumed only by loss, metrics, and supervision audits.
_Avoid_: Model feature, Dataset scope, attention mask

**Temporal Evaluation Matrix**:
The controlled two-by-two reporting design crossing Implicit Temporal and Explicit Day/Phase with Primary Development OOF and All-Alive Identifiability Stress. It contains two Primary training lineages and four evaluation cells; each stress cell reuses the exact checkpoint of its corresponding Primary temporal condition.
_Avoid_: Four-training-lineage matrix, all-alive retraining

**Primary Temporal Information Effect**:
The paired game-macro reduction in Primary mean belief KL from adding direct deterministic day/phase information: `K_primary_implicit - K_primary_explicit`. Positive means Explicit Day/Phase performs better; negative values remain valid. Its interval comes from paired game-cluster bootstrap.
_Avoid_: All-alive temporal primary, independent-interval subtraction, interaction headline

**Fold-Local Baseline**:
A non-model reference estimated exclusively from the training games of one OOF fold and evaluated on that fold's held-out games. It is reporting context, never a model input or selection signal.
_Avoid_: Global prior, full-development prior, checkpoint feature

**Worst-Case Example**:
A report-derived Canonical ToM Sample selected to expose large current-semantics model errors for scientific inspection. It is not a separate annotation source or experiment lineage.
_Avoid_: V2 comparison case, relabeled sample

**Memorization Sanity**:
A test-suite check that the canonical model and training path can overfit a deliberately tiny synthetic or development-safe fixture. It is not a standalone scientific run or report lineage.
_Avoid_: Memorization benchmark, production runner

## Methodological Lineage

**Methodological Reference Boundary**:
The rule that ONUW supplies the high-level progression from natural-language social interaction through structured chronological actions to Transformer belief distributions, but does not specify Classic7 game rules, schemas, features, dimensions, supervision timing, model configuration, interfaces, or parity behavior.
_Avoid_: ONUW compatibility contract, reference implementation parity

## Reconstruction Scope

**Phase-1 Reconstruction**:
The contract-complete reconstruction of the current scientific mainline through an end-to-end executable validation on controlled small-scale data. It excludes formal large-scale collection, paper-level model results, unrelated cleanup, aesthetic module consolidation, speculative abstractions, and performance or distributed optimization.
_Avoid_: Repository-wide rewrite, production-scale experiment, performance milestone

## Temporal Representations

**Implicit Temporal**:
The temporal configuration that represents time through chronological token order, positional representation, and the unchanged authoritative public-event sequence, including Phase Transition events, without direct day-index or phase-category features.
_Avoid_: No-time baseline, phase-transition-free history

**Explicit Day/Phase**:
The controlled temporal configuration that uses the same canonical events, PRE boundaries, targets, folds, and model capacity as Implicit Temporal while additionally representing each structured token's day and phase directly.
_Avoid_: Expanded-history configuration, alternate Dataset schema

**Public Temporal State**:
The publicly knowable coarse day and game phase established by the most recent authoritative Phase Transition. It never names or exposes a role-specific Private Action Scheduler Phase, and a Phase Transition enters and itself carries its newly established state.
_Avoid_: Private scheduler phase, inferred future phase, daily-reset state

**Stateful Day/Phase Propagation**:
The strictly causal assignment of the current Public Temporal State to every structured token until the next Phase Transition establishes a new state.
_Avoid_: Marker-only temporal feature, future backfill

**Night0**:
The opening night before Day1 discussion in the formal Classic7 game definition. Its private actions are followed by a public night result before the game enters Day1.
_Avoid_: Pre-game warmup, optional opening rule

**Private Action Scheduler Phase**:
An internal runtime phase used to sequence role-specific private actions during a night. It is not public temporal state and is never represented in Structured Public History.
_Avoid_: Public phase, model phase

**Public Phase**:
One of the five observer-visible temporal states `night`, `discussion`, `vote`, `pk_discussion`, or `pk_vote`. Public result events inherit the phase that produced them rather than creating separate result phases.
_Avoid_: Role skill phase, death-result phase, exile-result phase

**Parameter-Parity Contract**:
The temporal-ablation rule that Implicit Temporal and Explicit Day/Phase have the identical trainable parameter graph, initialization, optimizer, model dimensions, training protocol, Dataset, folds, and event sequence. Explicit temporal information is supplied only through deterministic, versioned, non-trainable codes, while the implicit code is zero.
_Avoid_: Equal declared parameter count, gated unused embeddings, compensated architecture

**Paired Initialization Contract**:
The per-outer-fold temporal-ablation rule that Implicit and Explicit start from the exact same canonical trainable tensor bytes and share optimizer initialization, game and batch order, augmentation schedule, RNG/dropout schedule, and optimizer-step budget. Both manifests bind one `paired_initial_state_digest`.
_Avoid_: Same-seed assumption, independent initialization, matched distribution only

**Deterministic Temporal Code**:
The versioned, non-trainable 256-dimensional representation added to each explicit-temporal structured token, with equal-energy 128-dimensional day and phase subspaces and combined RMS `0.02`. The day subspace is sinusoidal, the phase subspace is non-ordinal and pairwise orthogonal, and the corresponding implicit-temporal code is zero.
_Avoid_: Learned temporal embedding, trainable temporal projection, ordinal phase code

**Canonical Day-Code Table**:
The immutable, provenance-bound table of canonical sinusoidal day-code bytes materialized for all non-negative day indices present in one development experiment. It is an Explicit Day/Phase input artifact, not a learned model parameter or Dataset semantic field.
_Avoid_: Day vocabulary, dynamic day cache, checkpoint parameter
