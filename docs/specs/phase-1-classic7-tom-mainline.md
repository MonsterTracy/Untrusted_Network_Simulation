# Phase-1 Classic7 ToM Mainline Implementation Specification

Status: ready for implementation review

Authority: `CONTEXT.md` and ADR 0001–0021

Scope: Phase-1 contract-complete execution on controlled small-scale data

This specification does not reopen the frozen scientific decisions. It chooses
only the physical artifacts, module seams, validation mechanics, and execution
order needed to implement them.

The labels below distinguish the source of each requirement:

- **Scientific invariant** — fixed by `CONTEXT.md` or ADR 0001–0021; an
  implementation may enforce it but may not make it configurable.
- **Implementation choice** — selected here to realize a frozen invariant.
- **Runtime/config value** — required to be declared and frozen for one
  experiment, but not a permanent scientific constant.

### Frozen-decision traceability

| ADR | Implemented by this specification |
|---|---|
| 0001 repository boundary | Sections 1, 2, 4, and Out of scope |
| 0002 Primary versus All-Alive | Sections 2, 3.7, 6.6, and 6.7 |
| 0003 role-truth termination | Sections 2.2, 3.5, and 3.7 |
| 0004 population-blind Dataset | Sections 3.6 and 4.2 |
| 0005 temporal ablation/code artifacts | Sections 3.8, 3.9, and 6.5 |
| 0006 Night0/public phases | Sections 3.4 and 7.2 |
| 0007 Bundle versus Publication | Sections 3.4 and 3.5 |
| 0008 Speaker PRE Belief Handoff | Sections 3.4 and 7.2 |
| 0009 complete-observation canonicality | Sections 3.3–3.5 |
| 0010 Private Replay Evidence | Sections 2.2, 3.4, and 3.5 |
| 0011 plan-closed games/five folds | Sections 3.3 and 3.5 |
| 0012 Methodological Reference Boundary | Sections 1 and Out of scope |
| 0013 terminal speaker turn in PRE | Sections 3.4, 3.6, and 6.1 |
| 0014 sole PRE constructor | Sections 2, 3.4, and 4 |
| 0015 sole V1 perception | Sections 2 and 3.4 |
| 0016 game-macro scoring/training loss | Sections 6.4 and 6.7 |
| 0017 selection-blind fixed-budget OOF | Sections 3.10 and 6.5–6.6 |
| 0018 full prefix/no truncation | Sections 3.5, 3.6, and 3.10 |
| 0019 balanced cyclic rotation | Sections 6.2 and 6.3 |
| 0020 durable attempt claim | Section 3.3 |
| 0021 Non-Self Suspicion Simplex | Sections 3.6 and 6.1 |

## 1. Problem statement and current-to-target architecture map

The repository currently contains useful implementations of Classic7 runtime,
strict-PRE collection, V1 speech perception, structured event features, an
observer-relative Qwen2 predictor, metrics, and replay validation. Those live
pieces are interleaved with superseded experiment lineages: pilot collection,
old empty-target semantics, generic supervision scopes, private-conditioned
features, V2 and shadow annotation paths, GPT-2 compatibility, best-checkpoint
outer-fold selection, final-fit/sealed evaluation, and historical split
materializers.

The target is not another abstraction layer around these paths. It is one
executable scientific path whose modules have one semantic owner each.

### Current executable shape

```text
Game Runtime
  ├─ canonical collection
  ├─ pilot/fallback collection
  └─ several raw trajectory/sample projections
          ├─ V1 / legacy-empty Dataset
          ├─ V1 current-empty Dataset
          ├─ V2 sidecars and ablations
          ├─ public-only / private-conditioned models
          ├─ Qwen2 / GPT-2 backbones
          ├─ generic supervision scopes
          └─ development OOF / diagnostic OOF / final fit / sealed eval
```

### Target Phase-1 shape

```text
Game Runtime
    ↓
Canonical Collection
    ↓
Canonical Game Bundles + Canonical Collection Attempt Ledger
    ↓
Development Publication
    ├─ Public Development Records
    ├─ Development Fold Manifest
    └─ Role Sidecar ──→ Population Selector ──→ Eligibility Mask
    ↓
Canonical ToM Dataset
    ↓
Primary fold training (Implicit / Explicit paired lineages)
    ↓
fixed-budget terminal checkpoints
    ↓
held-out predictions at canonical shift 0
    ├─ Primary Development OOF
    └─ All-Alive Identifiability Stress
    ↓
game-macro and paired game-cluster-bootstrap reports
```

The target contains no compatibility seam back to the old paths. Historical
artifacts remain the responsibility of the old repository.

ONUW supplies only the principle-level natural-language → structured social
action → chronological Transformer belief-modeling pattern. It supplies no
Phase-1 schema, timing, player-count, model-size, modality, metric, or parity
requirement.

## 2. Solution: authoritative Phase-1 dataflow

### 2.1 End-to-end authority chain

| Stage | Authoritative input | Authoritative output | Sole semantic authority |
|---|---|---|---|
| Game Runtime | Collection-owned seed, runtime config, gameplay agents | Runtime transitions, public events, submitted actions, private replay state | Classic7 state transition and public observation generation |
| Canonical Collection | Runtime evidence, belief/perception backends, immutable Collection Plan | Attempt Ledger records and Canonical Game Bundles | PRE prefix construction, Belief Observation, Speaker PRE Belief Handoff, V1 Speech Perception, canonical eligibility |
| Development Publication | One completed Collection identity | One immutable Development Publication | Plan-closed game set, public/private stripping, five game-level folds, Role Sidecar derivation, publication statistics |
| Dataset | Verified public view of one Development Publication | Canonical public tensors, targets, observation/alive metadata | One structured tensorization and one belief-target conversion |
| Population operation | Publication public records; Role Sidecar only for Primary | Observer Eligibility Mask | Scientific supervision/evaluation row membership |
| Experiment preparation | Publication, model/training config, temporal config values | Immutable Experiment Manifest and prerequisite artifacts | Capacity preflight, paired initial states, paired training schedules, temporal artifact binding |
| Training | Primary masks, Dataset output, frozen schedule/state/protocol | One terminal checkpoint per fold and temporal condition | Primary-only fixed-budget optimization |
| Fold evaluation | Frozen terminal checkpoint, held-out Dataset output, named operation mask | Held-out predictions and operation-specific fold scores | Pure forward evaluation at canonical shift 0 |
| OOF reporting | All fold prediction/score artifacts | Four operation reports and paired headline contrasts | Game-macro aggregation and game-cluster bootstrap |

### 2.1.1 Per-stage gates

| Stage | Invariants and fail-closed gate | Provenance identity/digests | Allowed / forbidden flow |
|---|---|---|---|
| Game Runtime | Night0 and Classic7 rules; observer-visible coarse phases only; invalid action or exhausted generation cannot silently fall back | seed, runtime/agent/backend config, source revision | May send public and private evidence to Collection; cannot receive trained ToM output or implement policy research |
| Canonical Collection | durable claim first; strict PRE; complete alive-observer reports; sole V1 perception; one speaker handoff; errors make game ineligible | Collection Plan, claim/terminal, call/retry/parser versions, bundle and child digests | May use private observer cognition to collect realized reports and replay; may not create targets, tensors, folds, or model features |
| Development Publication | exact ledger-defined success set; replay and complete PRE validation; five whole-game folds; public/private stripping | Collection and Bundle digests, game-set digest, publication/fold/sidecar digests | May read Private Replay Evidence only for validation and Role Sidecar creation; may not parse speech, alter prefixes, convert targets, select populations, or train |
| Dataset | exact frozen prefix; one V1 source; one target semantics; full history; no population knowledge | Publication/game/prefix/planner versions and digests | May consume public records and emit public tensors/targets; cannot open Role Sidecar or private evidence |
| Population operation | Primary and All-Alive are named, fixed operations; masks are supervision-only | publication, row-identity, selector, sidecar digest only for Primary | Primary selector alone interprets roles; only booleans cross its output seam; All-Alive cannot open the sidecar |
| Experiment preparation | capacity bound; exact two temporal conditions; parameter parity; complete seven-shift schedules; manifest sealed before training | publication/fold, temporal, masks, bootstrap, schedules, initial states, protocol digest | May inspect declared publication statistics and materialize prerequisites; cannot inspect outcomes or introduce an extra condition |
| Training | Primary rows only; exact per-minibatch game-balanced CE; outer-training records only; fixed complete cycles; deterministic step-boundary recovery; terminal checkpoint only | experiment, training partition/schedule, paired initial state, recovery/RNG/protocol, terminal checkpoint-set digests | Receives only training-Primary partition; cannot open held-out records/masks, All-Alive rows, or role semantics; recovery closes permanently when checkpoint set is sealed |
| Fold evaluation | all ten checkpoints frozen first; canonical shift 0; one prediction artifact; same checkpoint for both named operations | checkpoint set, held-out rows/prefixes, prediction and named report digests | May open held-out records only after checkpoint-set seal; cannot train, select, rerun, rotate-ensemble, or change model state |
| OOF reporting | all games held out exactly once; per-game KL then equal game mean; paired game resampling | experiment, all fold reports, bootstrap plan, aggregate report digests | May reduce immutable predictions/reports; cannot weight by fold, select checkpoints, or feed results upstream |

### 2.2 Allowed information flow

```text
private runtime state
    └─→ Private Replay Evidence
            ├─→ bundle replay/canonical validation
            └─→ Development Publication's Role Sidecar derivation
                    └─→ Population Selector
                            └─→ boolean Observer Eligibility Mask

public runtime state
    └─→ Authoritative PRE Prefix + V1 actions + Belief Observations
            └─→ public Development Records
                    └─→ Dataset public tensors and targets
                            └─→ model.forward(public tensors only)
```

### 2.3 Forbidden information flow

The implementation must make each of the following unrepresentable at the
public interfaces, not merely discouraged:

- Role Sidecar, role names, private checks, teammate knowledge, or hard-knowledge
  masks entering Dataset construction or `model.forward`.
- Dataset, Publication, or replay constructing, cropping, completing,
  reordering, or repairing an Authoritative PRE Prefix.
- Dataset or Publication reparsing raw speech or replacing a V1 annotation.
- All-Alive rows entering training, early stopping, model selection, retry, or
  experiment selection.
- Held-out content influencing fold membership or checkpoint model tensor bytes,
  including through provenance digests used as numerical training controls.
  Full artifact manifests/digests may change to record different provenance;
  that change must not itself change model tensors.
- The trained ToM predictor entering gameplay generation or control.
- A generic population, target-semantics, annotation-version, backbone, or
  private-conditioning switch reintroducing a deleted lineage.

### 2.4 Phase-1 stop condition

Phase-1 is complete when production interfaces execute this entire chain on a
controlled deterministic small-scale collection and every artifact passes its
normal validator. Model quality thresholds, large-scale collection, final
evaluation, performance tuning, and repository-wide beautification are not
acceptance conditions.

## 3. Artifact and schema contracts

### 3.1 Common artifact envelope

**Implementation choice.** Every immutable artifact directory contains one
manifest (`manifest.json` or the domain-specific manifest name shown below)
with:

- `artifact_type` and one exact `schema_version`;
- its semantic identity fields and parent artifact identities/digests;
- an exact file table of every non-manifest relative path, byte size, and
  SHA-256;
- implementation/provenance versions required to interpret the artifact;
- a `manifest_digest` computed over the canonical manifest object with that
  field omitted.

Canonical JSON is UTF-8, sorted by key, compactly encoded, rejects NaN and
infinity, and does not normalize or rewrite string values such as raw speech.
JSONL uses the same object encoding per line and one `\n` terminator. Raw file
SHA-256 is over the bytes on disk. Artifact validators reject missing, extra,
renamed, or digest-mismatched files and unsupported schema versions. There is
no version fallback.

Immutable directories are built in a sibling temporary directory, every file
and nested directory is flushed and `fsync`ed, and the completed tree is
published with an operating-system atomic no-replace directory primitive such
as `renameat2(RENAME_NOREPLACE)` or `renamex_np(RENAME_EXCL)`. The destination
parent is then `fsync`ed before success is returned. A platform without a
verified no-replace primitive fails preflight; a check-then-ordinary-rename is
not an accepted substitute because it can race and overwrite. Existing
immutable artifacts are verified and reused by identity; they are never
overwritten or repaired.

Timestamps, host details, and logs are provenance but not semantic identity.
Changing a semantic input, schema, code version, collection plan, publication,
fold assignment, temporal artifact, model graph, training schedule, or fixed
protocol creates a new identity.

### 3.2 Physical artifact layout

**Implementation choice.** The Phase-1 artifact root has four explicit
lineages. Training and Dataset code receive one verified artifact handle, not
an arbitrary directory to search.

```text
collections/<collection-id>/
  manifest.json                         # written only when collection is sealed
  collection_plan.json
  attempt_ledger/
    .staging/                            # non-authoritative publication source
    <ordinal>-<seed>.claim.json
    <ordinal>-<seed>.terminal.json
  attempts/<attempt-id>/
    partial_evidence/
    failure_evidence.json              # when applicable
  games/<game-id>/
    manifest.json
    public/
      public_event_stream.jsonl
      authoritative_pre_prefixes.jsonl
      belief_observations.jsonl
      speech_annotations_v1.jsonl
    audit/
      call_budget_summary.json
      parser_summary.json
    private/
      submitted_gameplay_actions.jsonl
      backend_call_evidence.jsonl
      private_replay_evidence.json
  collection_summary.json

publications/<publication-id>/
  manifest.json
  public/
    games/<game-id>.json
    development_fold_manifest.json
  restricted/
    role_sidecar.json

experiments/<experiment-id>/
  experiment_manifest.json
  temporal/
    phase_codebook.manifest.json
    phase_codebook.bin
    canonical_day_code_table.manifest.json
    canonical_day_code_table.bin
  bootstrap/
    game_cluster_bootstrap_plan.manifest.json
    game_cluster_bootstrap_indices.bin
  folds/<fold-id>/
    population/
      training_primary/<game-id>.json
      held_out_primary/<game-id>.json
      held_out_all_alive/<game-id>.json
    training_schedule.jsonl
    initial_state/
      manifest.json
      tensors.bin

runs/<experiment-id>/
  checkpoint_set_manifest.json             # all ten checkpoints, before evaluation
  <temporal-condition>/<fold-id>/
    training_manifest.json
    training_log.jsonl
    recovery/<optimizer-step>/
      manifest.json
      model_tensors.bin
      optimizer_tensors.bin
      rng_state.bin
    terminal_checkpoint/
      manifest.json
      tensors.bin
    held_out_predictions.jsonl
    primary_development_oof/
      fold_report.json
    all_alive_identifiability_stress/
      fold_report.json
    run_provenance.json
```

The `restricted/` name is an information-boundary marker, not a claim of OS
access control. The enforcement is the typed loader boundary: Dataset has no
method that opens `restricted/`, and only Population Selector accepts a Role
Sidecar handle.

### 3.3 Canonical Collection Plan and Attempt Ledger

The Collection Plan freezes:

- ordered unique seed pool and target canonical-success count;
- runtime and agent identities;
- backend, model, parser, prompt, retry, and call-budget identities;
- public-event, PRE-prefix, Belief Observation, V1 annotation, and bundle schema
  versions;
- source revision and environment provenance required for replay.

It contains one production collection semantics; there is no mode field.

**Implementation choice: crash-atomic durable publication.** The Attempt Ledger
is an append-only directory of immutable event files. Claim and terminal records
use one `publish_ledger_record` interface owned by the Ledger module; callers
cannot open or mutate final ledger paths directly.

For each record, the implementation performs this exact sequence on one
filesystem:

1. create a uniquely named staging file under `attempt_ledger/.staging/` with
   `O_CREAT|O_EXCL`;
2. write the complete canonical bytes, flush them, and `fsync` the staging file;
3. atomically publish the staged inode at the final record path using a
   same-filesystem hard link, whose create-if-absent semantics fail with
   `EEXIST` rather than replacing an existing path;
4. `fsync` the ledger directory; only this completed directory sync makes the
   final record durable and allows the caller to proceed;
5. unlink the staging name and `fsync` the staging directory as cleanup.

The staging file is never authoritative. A crash before step 4 leaves either
no final record or a final record that must be validated after restart; no
gameplay/backend call is permitted because publication had not returned. A
leftover staging file is ignored and safely removed on recovery. A final record
is always complete because the inode was fully written and synced before the
atomic link. If the final path already exists, publication fails closed even
when the bytes appear equal; the Ledger never overwrites, rewrites, or
idempotently replaces a claim or terminal record. Filesystems that cannot prove
same-filesystem hard-link atomicity and directory-`fsync` durability fail
collection preflight.

Before any gameplay or backend call, the collector durably publishes the
seed's claim using this sequence. The claim binds plan digest, ordinal, seed,
attempt ID, runtime provenance, and claim timestamp. A crash before a final
claim survives recovery consumes nothing; any valid final claim permanently
consumes the seed.

A terminal record uses the same publication sequence, after every referenced
Bundle/failure artifact has itself been durably published and verified. Its
outcome is exactly one of:

- `canonical_success`, binding the Canonical Game Bundle digest;
- `canonical_failure`, binding complete failure evidence and any partial bundle;
- `interrupted_failure`, binding all recoverable partial evidence.

On resume, the Ledger module first removes non-authoritative staging files,
then scans final records in plan order, rejects duplicate or out-of-order
claims, and verifies every referenced digest. It durably publishes an
`interrupted_failure` terminal for every valid claim without a terminal record
before allowing another seed claim. Resume never invokes the same seed again.
Reaching the target stops new claims; unclaimed remaining seeds remain
unconsumed.

Fail closed on plan mismatch, backend/parser/retry provenance mismatch,
duplicate seed, ledger gap or overwrite, a terminal record without a claim,
invalid prior bundle/failure digest, an attempted claim after the target was
already reached, an invalid final record, unsupported filesystem durability,
or exhausted seed pool before target success.

### 3.4 Canonical Game Bundle

Each completed attempt may produce one immutable bundle. Its manifest binds
the Collection Plan, claim, attempt, seed, game identity, source revision,
runtime configuration, every child file digest, replay result, and
`canonical_eligibility`.

#### Authoritative PRE Prefix record

**Implementation choice.** Each PRE Boundary is stored as one complete record
in `authoritative_pre_prefixes.jsonl`, rather than as a cutoff into a mutable or
shared event stream. The record contains:

- boundary, game, step, current-speaker, and report-trigger identities;
- public `(day, phase)` and ordered alive-observer IDs at the boundary;
- the complete cumulative public event sequence ending with the matching
  `turn_start(current_speaker)`;
- all successful V1 annotations for public speeches already present in that
  sequence, and no annotation for the upcoming speech;
- event, annotation, and PRE-prefix digests;
- maximum public day present;
- the exact linked Belief Observation IDs;
- the unique Speaker PRE Belief Handoff observation ID.

This deliberate duplication makes the Collection-produced prefix the value
consumed downstream. Publication copies it unchanged after validation; Dataset
tensorizes it unchanged. Neither stage derives it by slicing the whole-game
event stream.

The prefix validator proves:

- continuous event ordering beginning with the initial
  `phase_change(day=0, phase=night)`;
- the frozen five-value Public Phase ontology and causal propagation, including
  that each `phase_change` token carries the new state it enters;
- the terminal matching `turn_start` is present;
- the next public speech and every action derived from it are absent;
- one-to-one V1 annotation coverage for earlier public speeches;
- all stored digests recompute exactly;
- all linked observations share this boundary and prefix digest.

#### Belief Observation record

The raw schema represents `success` and explicit failure statuses. Success
stores a canonical, duplicate-free, seat-ordered suspicion support that excludes
the observer; an empty list is a successful observation. Failure stores no
support, `label_observed=false`, categorized error and bounded-attempt evidence.
Failure evidence identifies seed, game/attempt, PRE Boundary, observer,
day/Public Phase, failure stage/category, and every exhausted retry attempt.
An eligible bundle must have exactly one success for every alive observer at
every PRE Boundary. No missing or failed row can be hidden by omission.

The current speaker's successful row is referenced by the prefix and by the
immediately following submitted speech cognition. A second belief generation
call for that PRE-to-speech transition is a canonical failure.

#### V1 Speech Annotation record

Each record binds one public-speech event by event ID, speaker, raw-text digest,
and public-event-stream digest. It stores the canonical ordered
`(subject, action, object)` list and all parser/prompt/model/backend attempt
provenance. Final status is `ok`, `no_action`, or `error`; `no_action` has an
empty action list and is successful. An `error` makes the bundle ineligible.

#### Private Replay Evidence

Private Replay Evidence is physically isolated under `private/`. It may contain
role assignment, submitted gameplay actions, and observer-private runtime
evidence needed for deterministic replay and canonical validation. Authoritative
public outcomes of those actions remain in the public event stream. Private
submitted actions may not be referenced from a public record
except by an opaque file digest in the bundle manifest. It is never copied to
Development Publication public records.

The `audit/` partition contains only non-sensitive counts, statuses, budget
usage, parser coverage, and opaque call digests. Full backend prompts/responses
or call evidence that can reveal private state stay under `private/`.

An eligible bundle fails validation if replay diverges, PRE coverage is
incomplete, a belief or V1 perception failed, self-suspicion survived bounded
validation, any fallback action occurred, the Speaker PRE Belief Handoff is not
unique, a private scheduler phase appears publicly, or any child digest fails.

### 3.5 Development Publication

Development Publication accepts exactly one verified Collection identity. It
does not accept a caller-supplied game list. It derives the Development Game Set
from the ordered Attempt Ledger and requires it to equal every
`canonical_success` needed to first reach the plan target, with no omissions,
extra successes, duplicates, or retrospective filters.

Because formal OOF has five non-empty held-out folds, a Development Publication
requires at least five eligible games. This is a structural precondition of the
fixed five-fold protocol, not a hard-coded development-game count.

`public/games/` contains one immutable record per game. Per-game files let a
fold-training reader open only its outer-training games; it never has to scan a
monolithic file containing held-out targets. Each game record contains its
immutable game and bundle identity, its complete Authoritative PRE
Prefix records, linked successful Belief Observations, public alive state, and
public audit metadata. It contains no roles, checks, teammate knowledge,
hard-knowledge masks, private observations, or private replay path.

Publication validation recomputes and records:

- game, boundary, observation, event, V1 annotation, and prefix counts;
- complete-observation coverage;
- `max_observed_day`;
- `max_structured_token_count` over all Authoritative PRE Prefixes;
- the exact game-ID and bundle-digest sequence;
- successful/failure attempt counts and a complete-case exclusion summary.

The token maximum is computed by the single deterministic Structured Token
Planner introduced with the frozen public-event/PRE contract, before the
Publication module. Its interface maps an already-frozen prefix to one
immutable semantic token plan containing ordered token descriptors, propagated
public temporal state, count, and plan digest. Publication reads only the count,
digest, and planner version; it never materializes token IDs or tensors.
Dataset later consumes that exact planner interface and alone maps the plan to
numeric tensors. Publication therefore does not depend on the later Dataset
commit, and this requirement does not create a second tokenization
transformation.

#### Development Fold Manifest

**Implementation choice.** Fold assignment is deterministic and balanced:

1. For every publication game, compute SHA-256 over canonical JSON UTF-8 bytes
   of `[fold_assignment_version, game_id]`, with assignment version
   `classic7_sha256_game_identity_5fold_v2`. Game IDs must be stable identities
   fixed independently of generated public, belief, role, and private content.
2. Sort ascending by that digest, then by game ID as a total-order tie breaker.
3. Assign sorted item `n` to fold `n mod 5`.

`development_game_set_digest` remains computed over the full ledger-ordered
`(game_id, bundle_digest)` pairs for provenance and validation only. Neither it
nor a bundle/public-content digest controls ranking or membership. The fixed five
folds differ in game count by at most one. The manifest records
the algorithm version, ordered ranking digest, and exact held-out game
IDs/bundle digests per fold. Fold count and fold assignment are not CLI values.
Any game overlap, omission, duplicate, changed digest, or boundary split fails
closed.

With the eligible game-ID set fixed, changing any game's content leaves fold
membership unchanged, while the full fold/publication provenance changes.
Canonical eligibility and plan closure are still validated before assignment.

#### Role Sidecar

Development Publication reads verified Private Replay Evidence exactly once to
write a separate Role Sidecar containing the complete role assignment for each
published game. It binds publication, game, and bundle digests. It contains no
private observation history or hard-knowledge timeline.

The public publication manifest records the Role Sidecar artifact digest but
does not embed the role map. The Role Sidecar validator checks complete
seven-seat assignments and Classic7 role multiplicities. No public loader
returns its contents.

### 3.6 Canonical ToM Dataset

The Dataset constructor accepts a verified public Development Publication
handle and a fixed set of game IDs supplied by a fold manifest. Its public
signature has no population, scope, role, sidecar, private-conditioning,
belief-source, target-semantics, annotation-version, or truncation argument.

For every Authoritative PRE Prefix it emits:

- one complete structured token sequence and attention/padding mask;
- event/action IDs and public player references;
- observer identity and observer-relative player relations;
- per-token propagated day and Public Phase indices;
- the seven-by-seven belief target, `label_observed`, and public
  `observer_alive` metadata;
- boundary/game identity, prefix digest, token count, and causal audit metadata.

The single target conversion is:

- non-empty support: uniform mass over support;
- successful empty: observer diagonal zero, all other six entries `1/6`;
- failed/missing raw status: all-zero placeholder and `label_observed=false`.

Published development records are complete for alive observers, so
`observer_alive=true AND label_observed=false` during normal Dataset
construction is a publication-validation failure. Dead-observer dense rows may
use the zero placeholder with `label_observed=false`; they are never effective
supervision. The raw failure representation remains testable without becoming
a training path.

The token builder validates `token_count <= max_seq_len`, pads only after the
complete sequence, and never truncates. `max_seq_len` is supplied only through
the verified Experiment Manifest. The same builder and tensors serve both
temporal conditions.

The structured-token planner assigns the new state to a `phase_change` token
itself, propagates it forward until the next transition, and assigns one shared
state to a `public_speech` boundary and all of its V1 action tokens. Missing
initial state and any future-derived backfill fail closed.

### 3.7 Population artifacts

There are two named operations and no generic population interface:

- `PrimaryPopulationSelector` accepts public records and the matching Role
  Sidecar, evaluates `observer_alive AND role != Werewolf`, and writes a boolean
  Observer Eligibility Mask. The role comparison exists only inside this deep
  module. Its output binds publication, selector version, sidecar digest, and
  exact row identities, but contains no role value.
- `AllAliveEligibility` accepts only public records and writes
  `observer_alive`. Its manifest has no Role Sidecar field or digest.

Training and evaluation combine a named eligibility artifact with
`label_observed` to create the effective-supervision rows. If masks accompany a
dense batch, they live in the orchestration wrapper and are not accepted by
`model.forward`.

Primary targets remain the playing non-wolf observers' realized cognition and
may contain effects of lawful Seer or Witch private state. The selector excludes
Werewolf observers but does not claim that retained targets are strictly
public-identifiable. All-Alive is the incremental information-mismatch and
population-expansion stress from adding alive Werewolf rows; it is not a second
training estimand.

Rotation permutes an already-created Primary mask with the same seat
permutation as the public sample. It never reopens Role Sidecar or recomputes
role logic.

For each fold, masks are physically partitioned into training Primary,
held-out Primary, and held-out All-Alive files. The trainer receives only the
training-Primary handle; the two held-out handles are not opened until the
terminal checkpoint has been frozen. No training All-Alive artifact exists.

### 3.8 Deterministic temporal artifacts

The phase codebook and Canonical Day-Code Table implement ADR 0005 byte for
byte. The fixed phase order is `night`, `discussion`, `vote`, `pk_discussion`,
`pk_vote`, mapped to rows `r=1..5`; coordinate `c=0..127` is
`0.02 * (-1)^popcount(r & c)`. The canonical matrix is `[5,128]`, row-major,
C-contiguous little-endian float32. Its 2,560 canonical raw bytes have SHA-256
`27e974cdecc7927b04ccd2f5f2e4d13351a2edf382c88f12854633dd0be77490`;
generation and validation tests use this value as a fixed known-answer test.

For day and `k=0..63`, the exact rows are
`D(day,2k)=sin(day/10000^(2k/128))` and
`D(day,2k+1)=cos(day/10000^(2k/128))`, followed by the one frozen global
amplitude factor `0.02 * sqrt(2)`, which gives RMS `0.02`; per-row
normalization is forbidden. The
day table is generated once for rows `0..max_observed_day` and stored as
row-major C-contiguous little-endian float32. Day and phase halves have equal
energy and the concatenated 256-vector has RMS `0.02`. Generation records
platform/library provenance; subsequent execution only loads and verifies the
bytes.

Both temporal lineages bind and verify the same temporal artifacts. A
`TemporalCodeProvider` has two named constructors:

- Implicit Temporal returns an all-zero 256-vector at the common injection
  point;
- Explicit Day/Phase concatenates the verified table rows for each token.

Both use the same model code and trainable graph. Unknown phase, negative or
out-of-range day, wrong vocabulary/order/shape/dtype/layout, formula-version
mismatch, or digest mismatch fails closed. There is no runtime recomputation,
extension, clip, bucket, learned parameter, projection, gate, or scale.

The phase manifest's `temporal_code_version` binds phase vocabulary/order, row
mapping, parity formula, dimension, amplitude, dtype, endianness, layout, and
canonical byte digest. The day manifest separately binds that version, formula
identity, `max_observed_day`, shape, canonical byte digest, and generation
provenance. Temporal bytes are verified inputs, not model-state tensors,
trainable parameters, or optimizer state.

### 3.9 Canonical model-state artifacts

**Implementation choice.** PyTorch pickle is not used as the canonical digest
representation. A model-state artifact stores tensors in lexicographic state
name order in one raw binary payload. The manifest records every name, shape,
canonical dtype, C-order byte offset, byte length, trainable flag, and per-tensor
SHA-256. Canonical trainable parameters are little-endian float32; unsupported
trainable dtypes fail preparation. The whole payload and manifest are digested.

For each outer fold, experiment preparation instantiates the one Qwen2 model
graph once from the declared initialization seed and materializes one initial
state. Both temporal lineages strictly load this same artifact. Their manifests
record the same `paired_initial_state_digest`; exact key, shape, dtype, and byte
comparison is required before training.

The terminal checkpoint uses the same verified tensor container and contains
only the fixed-budget terminal model state plus a manifest binding experiment,
fold, temporal condition, training schedule, initial state, optimizer protocol,
terminal step, and training log. Evaluation never loads optimizer state.

Recovery artifacts are not selectable checkpoints. At a predeclared
optimizer-step cadence they preserve the exact model tensors, optimizer tensor
and scalar state, scheduler state, next schedule cursor, Python/NumPy/Torch
CPU/device RNG states, and training-log prefix digest. They use immutable,
atomic no-replace directory publication. Only the training module can load
them, and only through the deterministic recovery rule in section 6.5.

### 3.10 Experiment Manifest

Experiment preparation must finish and atomically publish one immutable
manifest before any fold training or evaluation. It binds:

- Development Publication and Fold Manifest identities/digests;
- both fixed temporal conditions and temporal artifact digests;
- Qwen2 model graph, hidden size, layers, heads, output contract, and
  `max_seq_len`;
- the assertion `max_seq_len >= max_structured_token_count`;
- per-fold initial-state and paired training-schedule digests;
- Primary and All-Alive mask digests;
- optimizer, fixed scheduler if any, batch construction, rotation-cycle count,
  derived optimizer-step budget, RNG/dropout schedule, determinism settings,
  `deterministic_step_boundary_resume_v1`, and its recovery-checkpoint cadence;
- evaluation metric versions, bootstrap-plan digest, bootstrap replicate count,
  interval method/level, and reporting schema versions;
- runtime, dependency, device/backend, and source provenance.

**Runtime/config values** include the actual `max_seq_len`, optimizer settings,
batch size, complete rotation-cycle count, initialization/schedule seeds,
recovery-checkpoint cadence, bootstrap replicate count, and interval level.
They must be explicit—there are
no hidden defaults in a formal experiment—and identical across the paired
temporal lineages where required. They are not permanent scientific constants.

Numerical training settings, including capacity and seeds, must be declared
without consulting held-out content. Preparation validates capacity against
publication statistics; it does not select or enlarge capacity from those
statistics. Experiment schema `classic7_experiment_v2` retains the full
`protocol_inputs` and `protocol_digest` as provenance identities. They bind the
publication, folds, model, complete configuration and evaluation versions, but
are not inputs to initialization seeds, training RNG, game order or seat shifts.
An artifact identity may therefore differ while its model tensor bytes remain
identical. No compatibility reader for the former control semantics is provided.

## 4. Module and API seams

The target exposes a small external interface and keeps validation and artifact
mechanics inside deep modules.

### 4.1 Module topology

```text
Game Runtime
  existing Classic7 environment, agents, backend adapters

Canonical Collection
  public event ontology and temporal state
  PRE boundary and Speaker PRE Belief Handoff
  V1 speech perception
  attempt ledger
  Canonical Game Bundle writer/validator
  deterministic replay validator

Structured Public History
  one pure Structured Token Planner over frozen PRE prefixes

Development Publication
  collection reader
  public-record publisher/validator
  deterministic five-fold assignment
  Role Sidecar writer/validator

ToM Mainline
  canonical Dataset and rotation primitive
  named population operations
  deterministic temporal artifact provider
  public-only observer-relative Qwen2 model
  experiment preparation and model-state artifacts
  Primary fold trainer
  fold inference and named evaluators
  game-macro/paired bootstrap reporter
```

### 4.2 Public module interfaces

The implementation should expose these conceptual interfaces; exact Python
type names may follow repository conventions.

- `collect(CollectionPlan, RuntimeFactory, BackendSet, destination) -> CollectionSummary`
- `open_verified_bundle(path) -> VerifiedCanonicalGameBundle`
- `plan_structured_history(AuthoritativePREPrefix) -> StructuredTokenPlan`
- `publish_development(VerifiedCollection, destination) -> VerifiedDevelopmentPublication`
- `open_publication(path) -> VerifiedDevelopmentPublication`
- `build_dataset(PublicationPublicView, game_ids, ExperimentCapacity) -> CanonicalToMDataset`
- `select_primary_population(PublicationHandle) -> ObserverEligibilityArtifact`
- `build_all_alive_eligibility(PublicationPublicView) -> ObserverEligibilityArtifact`
- `prepare_experiment(PublicationHandle, ExperimentConfig, destination) -> VerifiedExperiment`
- `train_primary_fold(VerifiedExperiment, fold, temporal_condition) -> TerminalCheckpoint`
- `predict_held_out_fold(VerifiedExperiment, TerminalCheckpoint) -> PredictionArtifact`
- `evaluate_primary(...) -> PrimaryFoldReport`
- `evaluate_all_alive(...) -> AllAliveFoldReport`
- `aggregate_development_oof(VerifiedExperiment, all_fold_reports) -> OOFReportSet`

There is intentionally no generic `evaluate(scope=...)`,
`Dataset(target_semantics=...)`, model registry, parser registry, or backbone
registry. The two temporal conditions may be an enum because both are frozen
members of one controlled ablation; no third value is accepted.

### 4.3 Formal CLI seam

The same CLI also owns `capacity-check`, a synthetic-only engineering operation
outside the scientific lifecycle. It accepts only explicit synthetic sizes and
device, reads/writes no scientific artifacts, and reports no scientific scores.
Its scratch optimizer settings and output cannot select or modify a formal
protocol. It executes the existing graph and loss without changing their math;
OOM fails immediately without resizing or fallback.

One `uns` command owns a small set of subcommands:

1. `collect` — execute one immutable Collection Plan or resume its exact ledger.
2. `publish-development` — validate the completed collection and atomically
   create one Development Publication.
3. `prepare-experiment` — materialize masks, temporal artifacts, bootstrap
   plan, schedules, paired initial states, and the immutable Experiment Manifest.
4. `run-development-oof` — train the two paired Primary lineages, freeze
   all ten terminal checkpoints, seal the checkpoint-set manifest, then execute
   both named evaluations and aggregate reports.
5. `validate-artifact` — dispatch by the artifact's declared type only; it does
   not repair or upgrade artifacts.

Deployment paths are supplied by an explicit storage profile (`artifact_root`),
separate from ExperimentConfig. `werewolf.cli` owns the sole `uns` console entry.
Named experiment destinations retain immutable inputs and digest-addressed
mutable runs in one experiment-specific namespace; see
[server execution](../server-execution.md). No latest-artifact lookup is allowed.
Existing collection ledgers and OOF runs require explicit `--resume`; this flag
only authorizes the existing recovery policy, never an alternate checkpoint.

During its training phase, the OOF command may continue only through
`deterministic_step_boundary_resume_v1`. During its evaluation phase, it may
reuse verified immutable checkpoints/predictions/reports and continue missing
pure evaluation work. It may not reinterpret an incomplete training run as a
fresh attempt, rerun a failed lineage, reopen training after the checkpoint set
is sealed, replace a checkpoint after any held-out access, or change the
Experiment Manifest. A replacement training attempt requires a new experiment
identity declared without using outer-evaluation results.

### 4.4 Highest end-to-end test seam

The highest acceptance seam invokes the same four production interfaces, not a
test-only pilot mode:

```text
deterministic RuntimeFactory and backend adapters
→ production collect
→ production publish-development
→ production prepare-experiment
→ production run-development-oof
→ validate four reports and paired contrasts
```

The controlled fixture supplies at least five canonically eligible games so all
five outer folds are non-empty, one complete seven-shift training cycle, short
but complete histories, and a fixed small call budget. The test uses the actual
public-event builder, Dataset, Qwen2 graph, loss, checkpoint, inference, and
reporting modules. Deterministic adapters replace only external model calls.

## 5. Current-module disposition and deletion plan

Tests do not confer scientific reachability. A current module is kept only if
the frozen mainline still consumes its capability.

### 5.1 KEEP

| Current module/capability | Reason and required boundary |
|---|---|
| Classic7 environment and core role/game rules | Required Game Runtime; retain Night0 and private scheduler internally |
| Gameplay agents and backend protocol | Required only to generate trajectories; no policy optimization or ToM feedback |
| V1 speech perceiver and its bounded-attempt audit | Sole frozen semantic perception path after narrowing statuses/provenance |
| Private belief perceiver used for real observer cognition | Required for Belief Observation and Speaker PRE Belief Handoff, not model conditioning |
| Structured action ontology and player normalization | Current live V1 and tensor semantics |
| Observer-relative Qwen2 core, hidden 256/layers 4/heads 8 direction | No evidence supports changing baseline capacity or relative representation |
| Low-level metric primitives and Uniform Non-Self Reference | Reused under new game-macro orchestration |
| Deterministic replay and backend call-budget audit capabilities | Required canonical validation/provenance |
| `CONTEXT.md` and ADR 0001–0021 | Authoritative contract sources |

### 5.2 MODIFY

| Current module/capability | Required modification |
|---|---|
| `run_random` and Game Runtime integration | Remove deterministic legal fallback and pilot behavior; emit coarse public phases and use one Speaker PRE Belief Handoff |
| `trajectory` | Split manifest-bound public, audit, and Private Replay Evidence partitions; retain replay evidence without leaking it |
| `public_events` | Replace private scheduler phase names, include initial Night0 state, retain terminal `turn_start`, propagate day/phase causally, remove the R0 prefix-stripping function |
| `speech_annotations` and V1 perceiver | Make `no_action` explicit success, bind full attempt provenance, make exhausted `error` ineligible |
| Belief snapshot collection | Collect every alive observer once, reject self-suspicion, write explicit linked observations and handoff identity |
| Qwen2 belief backbone | Remove GPT-2/private/learned-temporal branches, structurally mask the diagonal, inject deterministic/zero temporal code at one point |
| Losses and metrics | Narrow to game-balanced CE and named current diagnostics; remove scope/private/legacy parameters |
| Baselines | Keep Uniform Non-Self; permit empirical priors only when fit solely on the matching outer-training games and report them as secondary |
| Replay/audit and worst-case reporting | Consume current bundle/report schemas only; no legacy target comparison |
| Package/config documentation | Describe the frozen mainline and formal CLI only |

### 5.3 REPLACE

| Current module/capability | Replacement |
|---|---|
| `TWDToMSampleCollector`, raw snapshot projection, and batch summary ownership | Canonical Collection deep module plus Bundle/Attempt Ledger artifacts |
| `samples` public snapshot schema | Authoritative PRE Prefix and Belief Observation records with no downstream reconstruction |
| `materialize_canonical_belief_dataset` | Development Publication; no train/validation/test split |
| `materialize_development_folds` | Publication-owned deterministic five-fold manifest |
| standalone `materialize_role_sidecar` | Publication-owned restricted Role Sidecar writer |
| `dataset`, `dense_dataset`, and `action_features` orchestration | One population-blind Canonical ToM Dataset and one rotation primitive |
| `supervision` generic scopes | Named Primary Population Selector and All-Alive Eligibility operation |
| current `checkpoint` pickle contract | Canonical tensor-state artifact and fixed terminal checkpoint |
| current `train`, `eval`, and `run_development_oof` | Experiment preparation, fixed Primary trainer, pure fold inference, named evaluators, and OOF reporter |
| standalone dataset/canonical audit scripts | Validators owned by Bundle, Publication, Dataset, and Experiment modules |
| script-per-operation CLI surface | One `uns` command with the five subcommands above |

Replacement means the old interface and tests are deleted in the same commit
that makes the replacement reachable. It does not mean wrapping the old path.

### 5.4 DELETE

Delete these code paths, configs, docs, and tests without an archive or
compatibility namespace:

- pilot collection mode, gameplay fallback action, missing-PRE fallback speech,
  and pilot-only configs/branches;
- V2 annotation sidecars, auto-candidate logic, repeatability/ablation runners,
  and V2 target sources;
- shadow parser configs, audit runner, and shadow replacement hooks;
- private-conditioned Dataset tensors, model embeddings, train/eval/checkpoint
  fields, and private leakage compatibility tests;
- GPT-2 block backbone and checkpoint compatibility;
- legacy empty-unobserved conversion and every target-semantics selector;
- generic `all_alive`, `non_wolf_alive`, `villager_alive`, speaker, diagnostic,
  and formal supervision-scope switches;
- old non-wolf diagnostic runner and terminology;
- historical split schemas including 48/6/6, 54/6, train/validation/test
  materialization, and hard-coded game counts;
- best-checkpoint, early-stopping, outer-validation selection, and `best.pt`;
- final-fit, sealed-evaluation, sealed-marker, and historical provenance code,
  configs, docs, and tests;
- memorization-sanity runner lineage; retain only its relevant assertions as
  ordinary test-suite acceptance tests;
- server deployment and sealed-evaluation documents that describe superseded
  research protocols;
- old worst-case report comparisons across legacy/V2 target semantics;
- aliases, loaders, adapters, migrations, and checkpoint upgraders whose only
  purpose would be reading artifacts owned by the historical repository.

## 6. Training and evaluation protocol implementation

### 6.1 Model forward contract

`model.forward` accepts only padded complete public-token tensors, attention
mask, public player references/observer-relative relations, observer identity,
and per-token public day/phase indices used by the configured temporal provider.
It does not accept target, label status, alive state, eligibility, population,
role, private knowledge, fold, or checkpoint-selection metadata.

The output has a seven-player target axis for every observer. The observer
diagonal logit is masked to negative infinity before `log_softmax`, giving exact
zero diagonal probability and a six-category non-self normalization. A batch
with non-finite off-diagonal output, non-zero diagonal, or non-unit non-self sum
fails validation.

The terminal `turn_start(current_speaker)` token is the only
upcoming-speaker representation. It encodes `current_speaker` through the same
ordinary subject/source player-reference field and the same observer-relative
player-reference encoding used by other structured tokens. The event-type ID
distinguishes `turn_start`; no dedicated upcoming-speaker embedding, relative-
speaker channel, standalone speaker feature, query conditioning, or second
speaker ID is allowed in `model.forward`. A `speaker_id` retained for audit or
row identity stays outside model-visible tensors.

### 6.2 Balanced seven-shift schedule

**Implementation choice.** The unit of formal augmentation is a complete
rotation cycle. In one cycle, every outer-training game appears exactly once at
each shift `0..6`. For game `g` and cycle `c`, one starting offset is fixed for
the entire cycle:

```text
starting_offset(g,c) =
    SHA256(schedule_version || schedule_seed || fold || c || game_id)
    mod 7
```

The SHA-256 digest is interpreted as one unsigned big-endian integer before the
modulo operation, and every concatenated field uses the manifest's canonical
length-delimited byte encoding. The starting-offset input contains no round.
Round `r=0..6` uses
`shift(g,c,r) = (starting_offset(g,c) + r) mod 7`. Addition by a fixed offset is
a bijection on `Z_7`, therefore
`{shift(g,c,r) | r=0..6} = {0,1,2,3,4,5,6}` for every game and cycle. The
per-round game order is independently derived from
`SHA256(schedule_order_version || schedule_seed || fold || c || r || game_id)`;
it may depend on round because it affects batching order, not shift coverage.

`schedule_seed` is the explicit frozen integer configuration value in
`0..2**63-1`; it is not an artifact/content digest. Schedule and order versions
are `classic7_balanced_seven_shift_v2` and `classic7_round_game_order_v2`.
Schedule records bind this seed, fold, cycle count, batch size, versions and
the resulting training-game sequence. Validation reconstructs the schedule
from those controls and rejects inconsistent records even with valid hashes.
The full experiment protocol digest is still computed over the declared
publication, model, training, evaluation, and seed inputs before derived
schedules and states are added, but serves provenance/validation only.
Initialization and training RNG seeds remain derived from their respective
explicit configuration seeds and fold identity. Publication, role-sidecar,
held-out-label and private-evidence digests never control numerical training.

The schedule writer materializes the exact ordered `(game_id, shift)` sequence
and batch boundaries before either temporal lineage trains. It does not use
Dataset indices or generic shuffling. Batches are formed within each rotation
round, so one batch cannot contain the same game twice. It neither drops nor
pads games; a final smaller batch in a round is allowed and is recorded. The
batch loss remains the equal mean
of the distinct per-game losses present in that batch. The same schedule bytes
are consumed by Implicit and Explicit.

The fixed budget is declared as a positive integer number of complete rotation
cycles. With no gradient accumulation in Phase-1, the derived budget is
`rotation_cycles * 7 * ceil(outer_training_game_count / game_batch_size)`
optimizer steps; each game therefore receives each shift exactly
`rotation_cycles` times. Both the declared cycles and derived step count are
stored in the manifest. Training rejects a
step-only budget that cuts a cycle, a schedule that omits or repeats a
game/shift within a cycle, a batch containing duplicate game identities, or a
terminal step that differs from the manifest. Evaluation always uses shift 0
and never invokes the schedule.

### 6.3 Cyclic rotation primitive

One pure rotation function permutes all seven-seat public references, V1
subjects/objects, speakers, observers, target rows and columns, public alive
state, and already-derived eligibility. It leaves chronology, event/action
types, day/phase, game/boundary identity, and fold unchanged. It has no Role
Sidecar input.

Required metamorphic assertion:

```text
rotate(sample, precomputed_eligibility, shift)
== rotate(sample, shift) + permute(precomputed_eligibility, shift)
```

Applying shift `s` followed by `7-s`, or applying all seven shifts and mapping
back, recovers the canonical sample exactly.

### 6.4 Game-balanced Primary loss

For each distinct game in a training batch:

1. combine its precomputed Primary eligibility with `label_observed`;
2. compute row cross-entropy only on those rows;
3. average rows within that game;
4. average the resulting game scalars equally across games in the batch.

For optimizer step `s`, let `B_s` be its distinct games and `R_g` the Primary
effective-supervision rows for game `g`. The exact loss is:

```text
L_s = (1 / |B_s|) * sum_{g in B_s}
          [(1 / |R_g|) * sum_{r in R_g} CE(r)]
```

Row cross-entropy is computed directly from the model's non-self log
probabilities as `-sum_j q_j log p_j`; zero target entries contribute zero and
no epsilon, clipping, or diagonal renormalization is applied.

“Game-balanced” therefore means equal game weight within each optimizer step.
When a rotation round ends in a partial minibatch, each game in that minibatch
has step coefficient `1/|B_s|`, which is larger than `1/game_batch_size`; the
specification does not claim equal cumulative SGD coefficient across games over
the complete schedule. For audit, the schedule records each game's cumulative
coefficient `sum_{s:g in B_s} 1/|B_s|`. Balanced Cyclic Seat Augmentation claims
equal shift-exposure counts, not equality of those cumulative coefficients.
Implicit and Explicit consume identical batch boundaries, so any partial-batch
weighting is exactly paired. An experiment may choose a batch size that divides
every outer-training fold size to eliminate partial minibatches, but that is a
declared Runtime/config value, not a different loss or fallback.

All-Alive masks are not loaded by the trainer. A game with zero effective
Primary rows, a false observed label in a published game, a target outside the
Non-Self Suspicion Simplex, or a role-bearing batch field fails closed. Loss
normalization is tested with deliberately unequal boundary/alive-row counts.

### 6.5 Fixed-budget paired training

For each outer fold:

1. Verify the Experiment Manifest, fold games, primary masks, temporal
   artifacts, training schedule, and canonical initial state.
2. Load the identical initial tensor bytes for Implicit and Explicit.
3. Initialize optimizer, deterministic game/batch/rotation order, and
   RNG/dropout streams from the same recorded schedule. Each lineage starts in
   a fresh process, reseeds Python, NumPy, Torch CPU, and the declared device
   backend after initial-state loading, and records the initial RNG-state
   digests; data order itself consumes no RNG.
4. Train only on the four outer-training fold partitions for exactly the
   manifest's complete cycles and optimizer steps.
5. Write one immutable terminal checkpoint; never write `best.pt`.
6. Freeze checkpoint digest before opening held-out targets for evaluation.

Schedulers, if used, must be a fully specified function of optimizer step and
may not read training or held-out metrics. Training curves are audit artifacts
only.

#### Formal process-crash recovery

**Implementation choice.** Formal training has exactly one predeclared crash
semantics: `deterministic_step_boundary_resume_v1`. There is no choice between
restart, resume, retry, or rerun at failure time.

- A recovery point is written only after an optimizer update and any fixed
  scheduler update have completed, with no partial gradient accumulation, and
  before the next scheduled minibatch begins.
- On process start, the trainer first verifies that no held-out prediction or
  evaluation artifact exists. It then loads the highest contiguous, fully
  verified recovery point for that exact experiment/fold/temporal lineage. If
  none exists, the canonical paired initial state is step zero.
- The trainer restores model, optimizer, scheduler, next schedule cursor, and
  all RNG states exactly. It may not choose an earlier recovery point, restart
  from initialization when a later valid point exists, change batch order, or
  alter the fixed terminal-step budget.
- Work after the latest durable recovery point may be re-executed after a
  crash, but only under the identical recorded software/backend/determinism
  environment. Because schedule and RNG state are restored, this is continuation
  of the predeclared lineage, not a new model-selection attempt.
- Formal training makes no remote/backend calls and enables the declared
  deterministic-algorithm mode; an operation that cannot satisfy it fails
  rather than silently becoming a nondeterministic restart path.
- A corrupt/non-contiguous recovery chain, environment mismatch, changed
  manifest, or failure to restore exact state marks the lineage failed closed.
  It cannot be restarted under the same experiment identity.

All five folds for both temporal conditions—ten terminal checkpoints—must be
successfully frozen and bound into one immutable `checkpoint_set_manifest`
before any outer held-out record is opened. After that manifest exists, every
training and recovery interface is permanently closed for the experiment;
only pure held-out inference/reporting may resume. Therefore no Primary or
All-Alive evaluation result can influence a training restart, rerun, retry, or
replacement checkpoint. A failed experiment may be replaced only by a new
fully declared Experiment Manifest chosen without consulting held-out results
from the failed experiment; evaluation-dependent reruns are forbidden.

### 6.6 Held-out prediction and named evaluations

Fold inference loads one terminal checkpoint and generates one immutable
prediction record for every Dataset observer row on the held-out games at
canonical shift 0. It records model probability, target, public alive and
observation status, row/boundary/game identity, checkpoint digest, prefix
digest, and temporal condition. It does not select rows by role.

Primary evaluation applies the precomputed Primary mask. All-Alive evaluation
applies the public-only All-Alive mask to the same prediction artifact. The two
fold reports have different artifact types and directories but must cite the
same checkpoint and prediction digests. All-Alive cannot trigger another
forward path with a changed model, let alone training.

The Primary report declares `population_identity=non_wolf_alive`, Role Sidecar
digest, and Population Selector version without a role map. The All-Alive
report declares `population_identity=all_alive` and has no direct Role Sidecar
field or loader dependency. It still cites the Primary-trained checkpoint as
required provenance; that transitive training lineage does not make Role
Sidecar an All-Alive evaluation input.

### 6.7 OOF aggregation and bootstrap

The reporter first verifies that each publication game is held out exactly
once per temporal condition and that all expected row identities are present.
It merges held-out predictions across folds before statistical aggregation;
fold ID is provenance, never a weight.

For each operation and game it computes the arithmetic mean of row-level
`KL(q || p)`, implemented as `sum_{j:q_j>0} q_j(log q_j-log p_j)` from the same
non-self log probabilities, without epsilon or clipping. A game with zero
effective rows fails closed. The headline point estimate is the equal mean of those game
scores. Uniform Non-Self uses exactly the same effective rows and aggregation.
Cross-entropy, TV, absolute error, support metrics, normalized reducible-gap,
and observer-row-weighted results are named secondary diagnostics. A
row-weighted result never borrows or is displayed with the game-macro interval.

**Implementation choice: bootstrap plan.** Experiment preparation orders game
IDs canonically and materializes a two-dimensional integer index table with one
row per bootstrap replicate and one draw per game. Indices are generated by a
versioned SHA-256 counter stream with rejection sampling into `[0, game_count)`;
this avoids Python/NumPy RNG-version ambiguity. The canonical table is
little-endian int32, C-contiguous, digested, and shared by all paired reports.
Replicate count, seed material, confidence level, and percentile interval
method are explicit Runtime/config values in the Experiment Manifest.

Every replicate recomputes the same point estimand after resampling game IDs
with replacement. Paired quantities use the same index row for all members:

- Identifiability Stress Penalty: per-game `KL_all - KL_primary`, then equal
  game mean;
- Primary Temporal Information Effect: per-game
  `KL_primary_implicit - KL_primary_explicit`, then equal game mean.

Negative values are retained. Ratios, clipping, independent resamples,
subtraction of separate confidence intervals, fold bootstrap, and row
bootstrap are validation failures. All-Alive temporal contrast may be emitted
as a clearly secondary diagnostic; no interaction headline is created.

## User stories and acceptance behaviors

1. As a collector operator, I can start from an immutable ordered seed plan and
   prove that no backend call occurred before a durable claim.
2. As a collector operator, I can resume after a crash and see the claimed seed
   closed as `interrupted_failure`, never attempted again.
3. As a data auditor, I can locate complete failure evidence for every excluded
   game and characterize complete-case selection by boundary, observer, day,
   phase, and failure category.
4. As a scientific user, I can inspect a PRE record and see the terminal
   speaker turn boundary but no current-speaker speech or derived action.
5. As a gameplay agent, I receive exactly the speaker's already-collected PRE
   belief for the next speech cognition, never a second sample or a ToM model
   prediction.
6. As a publication builder, I cannot select a preferred subset of successful
   games; the ledger and target define the complete Development Game Set.
7. As a publication consumer, I can validate every public record without
   gaining access to private replay evidence.
8. As Population Selector, I am the only code that interprets actual roles and
   my output contains booleans rather than role semantics.
9. As All-Alive evaluation, I can execute with the Role Sidecar physically
   unavailable.
10. As Dataset, I receive an already-frozen complete PRE prefix and have no API
    with which to crop, reconstruct, parse speech, choose labels, or choose a
    population.
11. As a model, I receive only public tensors and always return a normalized
    non-self distribution with exact zero diagonal.
12. As an experiment operator, I cannot start training until complete-history
    capacity, temporal artifacts, masks, schedules, initial states, and protocol
    are frozen in one Experiment Manifest.
13. As a temporal-ablation reviewer, I can compare the two trainable graphs and
    initial states tensor by tensor and verify that only the deterministic
    temporal input differs.
14. As a training reviewer, I can prove every game saw every cyclic shift the
    same number of times and that paired lineages consumed the same schedule.
15. As a training reviewer, I can verify that unequal numbers of rows per game
    do not change equal game weighting within a minibatch, and I can audit the
    exact larger coefficient assigned inside every partial minibatch without a
    false claim of globally equal SGD coefficients.
16. As an OOF reviewer, I can prove held-out games were never opened for any
    training, checkpoint, scheduler, retry, or model-selection decision.
17. As an OOF reviewer, I can prove Primary and All-Alive fold reports cite the
    same terminal checkpoint and prediction digest.
18. As a reporting user, I receive game-macro KL as the only headline model
    score and a Uniform Non-Self Reference on identical rows.
19. As a reporting user, I receive paired, game-resampled uncertainty for the
    stress penalty and temporal information effect rather than differences of
    unrelated intervals.
20. As a maintainer, I cannot invoke V2, shadow, private-conditioned, GPT-2,
    pilot, generic-scope, best-checkpoint, final-fit, or sealed paths because
    their code and tests no longer exist.
21. As an operator recovering a crashed training process, I have one automatic
    step-boundary resume rule and no failure-time choice of checkpoint, restart,
    retry, or changed budget.
22. As an OOF reviewer, I can prove all ten terminal checkpoints were sealed
    before any held-out record was opened, so evaluation cannot influence a
    later training rerun.

## 7. Testing decisions and validation matrix

### 7.1 Test levels

- **Contract unit tests** target pure schema, digest, phase propagation, target,
  rotation, loss, metric, and bootstrap functions.
- **Module interface tests** write and reopen complete immutable artifacts
  through the same public interface used by production.
- **Boundary-negative tests** inject one violation at a time and require a
  fail-closed error before downstream work begins.
- **Integration tests** connect real modules with deterministic external
  adapters.
- **One end-to-end acceptance test** crosses the full production seam described
  in section 4.4.

### 7.2 Validation matrix

| Boundary | Positive acceptance | Required negative cases |
|---|---|---|
| Collection Plan / Ledger | staged-file `fsync`, atomic no-replace hard-link publication, directory `fsync`, ordered claims, target stop, exact resume | injected crash at every publication step, partial/final corruption, leftover staging file, pre-existing final path, pre-claim call, duplicate/out-of-order claim, plan drift, orphan terminal, interrupted rerun, post-target claim, unsupported filesystem durability |
| Runtime public phase | Night0 initial state; five coarse phases; causal inheritance | private skill phase, missing initial state, future backfill, unknown phase |
| PRE Prefix | full cumulative history ends in matching `turn_start`; current speech absent | R0 deletion, current-speech/action leakage, reconstruction from cutoff, digest/order mismatch |
| Belief Observation | all alive observers exactly once; empty success preserved | missing/duplicate row, self-suspicion, fallback, second speaker estimate, exhausted failure marked eligible |
| V1 Perception | one bound `ok`/`no_action` record per prior speech | reparse, partial parse, `error` eligible, raw-text/event digest mismatch, V2/shadow source |
| Bundle | all file/digest/replay/provenance checks pass | private data in public files, fallback count, replay divergence, unlisted/extra files |
| Publication | plan-closed game set, complete coverage, correct maxima | caller subset, hard-coded count, old split, role leakage, missing/extra success |
| Structured Token Planner | one prefix-to-plan interface; deterministic order/state/count/digest | alternate planner, future-state backfill, Publication importing Dataset, count disagreement |
| Five folds | deterministic, whole-game, balanced, exhaustive | fold-count config, overlap, omitted game, changed game digest, boundary split |
| Role Sidecar / selector | complete Classic7 roles; boolean Primary mask | Dataset opens sidecar, roles in mask/batch, All-Alive sidecar dependency, eligibility recomputed after rotation |
| Dataset | one target semantics; full tokens; public tensors only | legacy empty, target/source switch, truncation, private tensor, population argument, terminal turn removal |
| Rotation | round-independent starting offset; formal `Z_7` coverage; inverse and mask metamorphism | round included in offset, duplicate/missing shift, role input, identity based on Dataset index, evaluation rotation |
| Temporal artifacts | exact formula/bytes/digest; equal parameter graph | learned temporal tensor, runtime trig recompute, phase RNG/QR, clip/extend, missing/unknown state |
| Initial state | paired digest and tensor-byte equality | key/shape/dtype mismatch, temporal-only parameter, pickle-only identity |
| Experiment preflight | all parent artifacts and capacity bound verified | incomplete manifest, `max_seq_len` overflow, hidden default, paired schedule/protocol mismatch |
| Model | public-only signature; exact non-self simplex; ordinary terminal-turn player reference is sole upcoming-speaker encoding | dedicated speaker embedding/channel/feature, eligibility/label/role input, non-zero diagonal, non-finite output, GPT-2/private branch |
| Training | exact per-step game-balanced CE including partial-batch coefficients; fixed cycle/step terminal state; deterministic recovery; all-ten checkpoint seal | false global-weighting claim, zero-row game, All-Alive row, best checkpoint, early stop, held-out access, partial rotation cycle, manual earlier restart, corrupt recovery chain, post-seal training |
| Evaluation | checkpoint-set seal first; shift 0; same checkpoint/predictions for named reports | evaluation before all checkpoints, TTA, All-Alive retraining, role inside model, report/checkpoint mismatch |
| Aggregation | merged OOF, per-game KL then game mean | fold mean, row headline, mismatched rows for baseline, mixed estimand/CI |
| Paired bootstrap | game resampling and exact paired differences | row/fold resample, unpaired indices, ratio, clipping, independent-CI subtraction |
| Full E2E | five non-empty folds, two checkpoints lineages, four report cells | any legacy CLI/config required, any sealed artifact accessed |

### 7.3 Provenance assertions in tests

Every artifact test must mutate one parent digest, one file byte, one schema
version, and one forbidden field to prove validation fails. The E2E test must
assert the complete digest chain:

```text
Collection Plan
→ Attempt claim/terminal
→ Canonical Game Bundle
→ Development Publication + Fold Manifest + Role Sidecar
→ Experiment Manifest + temporal/mask/schedule/initial-state artifacts
→ recovery chain → all-ten terminal checkpoint set
→ held-out predictions
→ named fold reports
→ aggregate and paired reports
```

## 8. Dependency-ordered implementation work packages

This section is an implementation responsibility map, not a set of mandatory
Git commit boundaries. Work-package numbering expresses dependency order,
ownership, required replacements/deletions, and tests. Each package retains its
goal and completion criteria, but need not be implemented, tested, or committed
in isolation from the consumers required for a coherent cutover.

Sections 1–7 and 9 remain authoritative and unchanged: target architecture,
artifact contracts, information boundaries, testing requirements, and deletion
requirements are not relaxed by this execution strategy. References elsewhere
to a planned commit identify the corresponding work package, not a mandatory
Git boundary.

Implementation may span multiple work packages to change related producers,
consumers, tests, CLI entry points, and exports together. Follow the real
dependency graph and preserve one coherent end-to-end cutover; do not force
unmigrated consumers to depend on modules deleted solely to finish an earlier
package. Actual Git commits should be cohesive and reviewable, but need not
correspond one-to-one with packages or number fifteen in total.

Do not introduce a compatibility layer, adapter, fallback, migration path, or
duplicate semantic implementation to make an intermediate package pass alone.
Existing superseded implementation may remain only while it still has a real
consumer awaiting migration. Delete it immediately when its last real consumer
is migrated, together with obsolete tests, imports, exports, and entry points.
For the replace-and-delete execution rule in Section 5.3, replacement
reachability means this coherent consumer cutover, not the first availability
of a producer scaffold. Do not postpone already-safe deletion to Work Package
15, and do not retain obsolete interfaces merely to keep historical tests
passing.

Final acceptance is determined by the target Phase-1 dataflow, Section 7
validation matrix, forbidden information flows, Section 5 DELETE inventory,
and full repository test suite, not by completing a predefined number of Git
commits. Required package tests and completion criteria remain obligations;
cross-package execution changes their scheduling, not their substance.

Work Packages 1–5 were completed as the original Commits 1–5. Their history is
not rewritten or replayed. The scaffold and cutover notes below record that
completed sequence, including the first production cutover in Commit 5; they
do not impose new Git boundaries. Work Packages 6–15 retain the original
remaining implementation responsibilities under the execution rules above.

### Work Package 1 — Canonical artifact I/O and validation scaffold

- **Goal:** add canonical JSON/JSONL, digest, immutable-directory publication,
  and raw tensor-container primitives.
- **Modules:** shared artifact envelope and validator utilities.
- **Tests:** byte-stable serialization, NaN rejection, file-table mismatch,
  atomic no-overwrite behavior, tensor ordering/dtype/digest corruption.
- **Completion:** artifact fixtures round-trip and every single-byte mutation
  fails validation.
- **Dependency note:** prerequisite scaffold; it changes no scientific path.

### Work Package 2 — Public-history, PRE, V1, and token-plan scaffold

- **Goal:** implement the frozen coarse public-event/PRE/V1 interfaces and the
  one pure Structured Token Planner required by later Collection, Publication,
  and Dataset modules, without yet changing the executable collector.
- **Modules:** public-event ontology/validator, Authoritative PRE Prefix
  constructor, V1 annotation contract, Speaker PRE Belief Handoff value,
  Structured Token Planner.
- **Delete:** none; this prerequisite module is exercised only through its new
  interface until Commit 5.
- **Tests:** Night0 timeline, all five phases, causal inheritance, strict PRE
  leakage cases, `no_action`, V1 provenance, single speaker handoff, token-plan
  order/state/count/digest and count-only consumption.
- **Completion:** deterministic public-history/PRE fixtures yield valid prefixes
  and token plans through the new interfaces; no future Dataset module is
  needed to compute a structured-token count.
- **Dependency note:** prerequisite scaffold; Commit 5 makes it authoritative
  and simultaneously deletes R0/alternate constructors.

### Work Package 3 — Crash-atomic Attempt Ledger scaffold

- **Goal:** implement Collection Plan validation plus the deep Attempt Ledger
  module, without claiming that collection has already cut over.
- **Modules:** Collection Plan, `publish_ledger_record`, claim/terminal schemas,
  ledger recovery/validation.
- **Delete:** none; this is a prerequisite scaffold exercised only through its
  new interface.
- **Tests:** injected crashes after staging create/write/file-`fsync`/hard-link/
  directory-`fsync`, leftover staging cleanup, pre-existing final path,
  duplicate/out-of-order/orphan records, interruption closure, provenance
  drift, target stopping, and seed exhaustion.
- **Completion:** at every injected crash point, recovery observes either no
  authoritative record or one complete immutable record; published paths are
  never overwritten, and a claimed seed can only reach one terminal outcome.
- **Dependency note:** prerequisite scaffold; production runtime integration is
  intentionally deferred to Commit 4.

### Work Package 4 — Canonical Game Bundle and replay scaffold

- **Goal:** implement immutable Bundle public/audit/private partitions and the
  replay validator without claiming that collection has already cut over.
- **Modules:** trajectory evidence types, Bundle writer/reader, deterministic
  replay, canonical failure evidence/summary.
- **Replace/Delete:** none; this prerequisite module accepts deterministic
  evidence fixtures through its interface.
- **Tests:** complete eligible Bundle, belief/perception/gameplay failure
  evidence, private-field leakage, replay divergence, child digest corruption,
  atomic no-replace publication, and failure-artifact round trip.
- **Completion:** fixtures can produce only a verified Bundle or explicit
  failure artifact, and the Bundle validator is the sole read interface.
- **Dependency note:** prerequisite scaffold; it has no Collection or Dataset
  import and needs no future commit to prove its artifact contract.

### Work Package 5 — Production-only Canonical Collection cutover

- **Goal:** integrate the Commit-2 public-history interfaces, Commit-3 Ledger,
  and Commit-4 Bundle into the sole plan-closed executable collector.
- **Modules:** Game Runtime public-event emission, collection orchestrator,
  belief/perception adapters, call-budget/failure evidence, Speaker PRE Belief
  Handoff, Ledger and Bundle interfaces.
- **Replace/Delete:** raw `TWDToMSampleCollector` output ownership, old batch
  summary/projection contracts, R0 terminal-turn stripping, private scheduler
  phase exposure, alternate PRE constructors, pilot mode/configs, gameplay
  fallback, missing-PRE fallback speech, and replacement/rerun branches.
- **Tests:** deterministic game-to-Bundle integration, complete and failed
  observations/perception/gameplay, backend call forbidden before durable
  claim, Bundle durable before terminal record, interrupted attempt never
  rerun, target stopping, no second speaker belief, and absence of old imports.
- **Completion:** the sole reachable collector produces a verified Bundle or
  explicit failure evidence, publishes exactly one terminal record, never calls
  a backend before claim durability, and exposes no old/fallback path.

### Work Package 6 — Development Publication, folds, and Role Sidecar

- **Goal:** publish the ledger-defined game set, public records, deterministic
  five folds, statistics, and restricted Role Sidecar.
- **Modules:** publication writer/reader, Structured Token Planner from Work
  Package 2 in count-only use, fold assigner, sidecar writer/validator.
- **Delete:** canonical train/validation/test materializer, standalone fold and
  sidecar materializers, historical split schemas/counts.
- **Tests:** plan closure, no cherry-pick, fold determinism/balance/isolation,
  max day/token statistics matched to the Work-Package-2 planner, sidecar
  multiplicities, private stripping, and a dependency test proving no Dataset
  import is required.
- **Completion:** a publication can be built only from the exact completed
  collection and exposes separate public and restricted handles.

### Work Package 7 — One population-blind Dataset and rotation primitive

- **Goal:** implement full-prefix tensorization, one target conversion, public
  alive/status metadata, and pure cyclic rotation.
- **Modules:** Dataset, numeric tensor materializer consuming the Work-Package-2
  Structured Token Plan, target converter, collator, rotation primitive.
- **Delete:** V2/legacy target sources, private masks, scope/role Dataset
  arguments, silent truncation, duplicate dense/point tensorizers.
- **Tests:** empty/non-empty/failure targets, terminal turn inclusion, no
  current-speech leakage, full-history capacity, rotation inverse/metamorphism.
- **Completion:** Dataset has one constructor contract and no import of Role
  Sidecar, roles, V2, or private knowledge.

### Work Package 8 — Named population operations

- **Goal:** terminate role truth at Primary selection and create public-only
  All-Alive eligibility.
- **Modules:** Primary Population Selector, All-Alive Eligibility, eligibility
  artifacts and orchestration wrapper.
- **Delete:** generic supervision scopes and old diagnostic/formal naming.
- **Tests:** exact masks, sidecar is sole role consumer, All-Alive works with
  sidecar unavailable, rotation only permutes booleans.
- **Completion:** downstream row identity artifacts contain no role semantics.

### Work Package 9 — Deterministic temporal artifacts and public-only Qwen2 model

- **Goal:** implement Parameter-Parity Contract, exact temporal bytes, one
  injection point, and Non-Self Suspicion Simplex output.
- **Modules:** temporal artifact writer/provider, Qwen2 model, observer-relative
  feature path.
- **Delete:** GPT-2 backbone, private conditioning, learned day/phase parameters,
  model/backbone registry and compatibility checkpoint branches.
- **Tests:** known Walsh rows/digest, day-table metadata/digest, fail-closed
  unknown/out-of-range states, parameter graph equality, zero-vs-explicit
  injection, exact diagonal exclusion.
- **Completion:** the only model graph is public-only Qwen2 and the temporal
  condition changes no trainable key or shape.

### Work Package 10 — Experiment preparation and paired schedules/states

- **Goal:** freeze capacity, masks, temporal inputs, bootstrap plan, balanced
  schedules, paired initial state, deterministic recovery cadence, and full
  fixed protocol before training.
- **Modules:** Experiment Manifest, schedule writer, canonical state writer,
  bootstrap-plan writer, preflight validator.
- **Tests:** round-independent starting offsets and exact seven-shift coverage,
  stable batch schedule, paired bytes, recovery-policy/cadence binding, hidden
  default rejection, token-capacity failure, artifact-parent mismatch.
- **Completion:** a verified Experiment handle is sufficient for training and
  contains no mutable or undecided protocol field.

### Work Package 11 — Fixed-budget Primary fold training

- **Goal:** implement game-balanced CE and exact terminal-checkpoint training on
  outer-training games only, including its sole deterministic crash recovery.
- **Modules:** trainer, loss, training log, recovery-state writer/loader,
  terminal checkpoint writer, checkpoint-set sealer.
- **Delete:** best checkpoint, early stopping, inner/outer validation selection,
  generic scope/private/V2 training paths.
- **Tests:** unequal row/game counts and exact partial-batch coefficients,
  zero-row fail, All-Alive access spy, schedule/step mismatch, crashes between
  recovery publication steps and between optimizer/recovery steps, exact
  optimizer/RNG/cursor restore, rejection of earlier-point/manual restart,
  held-out access spy, terminal checkpoint identity, and training closure after
  checkpoint-set seal.
- **Completion:** paired fold runs start from the same digest, consume identical
  schedules/protocols, recover only through the predeclared rule, emit one
  terminal checkpoint each, and seal all ten checkpoint digests before any
  held-out record can be opened.

### Work Package 12 — Pure fold inference and two named evaluations

- **Goal:** freeze held-out predictions once and derive Primary and All-Alive
  fold reports from the same checkpoint/prediction bytes.
- **Modules:** inference, Primary evaluator, All-Alive evaluator, per-game metric
  primitives, Uniform Non-Self Reference.
- **Delete:** generic eval, non-wolf diagnostic runner, private/V2 evaluation,
  test-time rotation.
- **Tests:** selection-blind call graph/spies, canonical shift 0, mask row sets,
  required checkpoint-set seal, same checkpoint digest, no All-Alive training
  entry, and permanent rejection of training/recovery after held-out access.
- **Completion:** every fold/temporal checkpoint has exactly two distinct named
  reports sharing one prediction artifact.

### Work Package 13 — OOF aggregation and paired reporting

- **Goal:** implement game-macro KL, bootstrap intervals, stress penalty, and
  Primary temporal information effect.
- **Modules:** OOF merger, game score reducer, bootstrap evaluator, aggregate
  report schemas, current worst-case export.
- **Delete:** row-weighted headline, fold-weighted aggregation, legacy/V2
  worst-case comparisons, final-fit/sealed reporting.
- **Tests:** hand-computed unequal-game fixtures, exact paired resamples,
  negative differences, Uniform reference row identity, report naming.
- **Completion:** two training lineages yield four cell reports and exactly the
  frozen two paired headline contrasts.

### Work Package 14 — Formal CLI cutover and full end-to-end acceptance

- **Goal:** expose the five formal subcommands and prove the controlled
  small-scale production path.
- **Modules:** root CLI composition, deterministic test adapters, E2E fixture.
- **Delete:** superseded script-per-lineage entry points and configs.
- **Tests:** CLI argument snapshots exclude forbidden switches; full five-fold,
  two-lineage, four-report E2E digest chain.
- **Completion:** Phase-1 can be reproduced from Collection Plan to aggregate
  reports without importing an old runner.

### Work Package 15 — Legacy deletion and documentation closure

- **Goal:** remove remaining unreachable historical lineage and make docs match
  the executable mainline.
- **Modules:** package exports, configs, README and current architecture/
  collection/ToM docs.
- **Delete:** V2, shadow, private, GPT-2, pilot, generic-scope, final/sealed,
  server deployment, old split, compatibility code/tests/docs not already
  removed; memorization runner becomes test assertions only.
- **Tests:** import/config/CLI inventory assertions and full suite; repository
  search asserts forbidden public options and obsolete module names are absent.
- **Completion:** every remaining non-test module is reachable from the frozen
  mainline or has a documented runtime/provenance justification.

## 9. Out of scope, risks, and unresolved implementation-only questions

### Out of scope

- MCTS, candidate action/speech scoring, policy optimization, reinforcement
  learning, or ToM-controlled gameplay;
- multimodal inputs, face/tone, or a raw-text ToM backbone;
- V2/shadow perception, private-conditioned modeling, GPT-2 compatibility, or
  historical reproduction;
- final/sealed evaluation or a Final Evaluation Publication;
- all-alive training, a third population, counterfactual public cognition, or a
  new temporal condition;
- hyperparameter search, adaptive budgets, early stopping, distributed or
  performance optimization;
- large-scale collection or paper-level scientific conclusions;
- generic framework extraction, cross-repository schemas, compatibility
  adapters, or unrelated module/file-count cleanup.

### Risks and mitigations

1. **Full-prefix duplication increases artifact size.** This is accepted in
   Phase-1 because it preserves single construction authority and makes no-loss
   consumption auditable. Content-addressed deduplication is deferred until
   measured size warrants it.
2. **Crash durability differs by filesystem.** The ledger and immutable writer
   must test file/directory `fsync`, same-filesystem atomic no-replace hard-link,
   and immutable-directory publication assumptions on supported local
   filesystems and record filesystem/platform provenance. Unsupported semantics
   fail preflight.
3. **Day-code bytes may differ when newly generated elsewhere.** Existing
   experiments always reuse the stored table; generation provenance and raw
   digest are authoritative, so mismatch creates a new artifact rather than an
   implicit equivalent.
4. **Fixed-budget training may not be cross-hardware byte deterministic.** The
   contract requires paired initial bytes and schedules, records backend and
   determinism settings, and does not claim identical final bytes across
   hardware. Crash recovery therefore requires the exact declared runtime/
   backend environment; otherwise the lineage fails instead of silently
   restarting elsewhere.
5. **Complete-case collection may induce selection bias.** Phase-1 preserves
   exhaustive failure evidence and exclusion summaries; it does not weaken
   complete-observation canonicality.
6. **A small five-game acceptance fixture is not a performance test.** It proves
   contract reachability only and cannot be reported as scientific evidence.

### Unresolved implementation-only questions

None block Phase-1 specification. The following are deliberately Runtime/config
values that an experiment author must declare before `prepare-experiment`, not
questions for code to answer implicitly:

- actual ordered seed pool and target canonical-success count;
- `max_seq_len` after publication statistics are known;
- optimizer and fixed step-based scheduler settings;
- batch size and complete rotation-cycle count;
- deterministic recovery-checkpoint cadence;
- initialization, schedule, and bootstrap seed material;
- bootstrap replicate count and percentile interval level;
- production backend/model endpoints and bounded retry/call budgets.

Changing one of these values creates a new Collection Plan or Experiment
Manifest as appropriate. It does not create a new scientific operation, schema
branch, or compatibility path.
