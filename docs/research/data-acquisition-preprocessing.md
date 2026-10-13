# Data Acquisition & Preprocessing

This branch maintains Canonical acquisition and derived data artifacts for
Classic7. Collection size is an explicit campaign parameter. It does not change
game rules, prompts, subjective belief supervision, V1 scientific semantics,
frozen ToM models or Canonical success qualification. Model training adaptation
and multi-model final Test evaluation are separate research work.

Branch ownership and history are maintained in [the branch registry](../BRANCHES.md).
This document owns the stable technical contracts. Actual deployment paths,
approved commands, complete production seed arrays and measured resource costs
belong to private campaign records.

## Architecture and lifecycle

```text
approved clean source revision + explicit campaign/storage/runtime inputs
  → optional preregistered Split Protocol
  → immutable Collection Plan
  → Canonical collector + Attempt Ledger + original game Bundles
  → complete Collection validation
  → native terminal audit
  → fixed Split Manifest, when preregistered
  → Development Publication V2 + Independent Test identity
  → source-side controlled release receipt + archived parent artifacts
```

| Owner | Responsibility |
|---|---|
| `scripts/collect_games.py` | Operator preflight, complete seed derivation, Plan freezing, explicit resume and storage selection |
| `canonical_collection` | Durable claims/terminals, call budget, PRE, public history, belief and replay evidence |
| `development_publication.py` | Verified Collection, V1 full-success publication, shared public codecs, roles and five-fold algorithm |
| `canonical_split.py` | Preregistered rules, stable ranking, Plan binding, full membership recomputation |
| `canonical_terminal_audit.py` | Independent real-engine terminal, winner, roles, legal actions and public-history audit |
| `development_publication_v2.py` | Dev-only projection and role sidecar with verified Split lineage |
| `canonical_release.py` | Audit, Split, Dev validation and immutable release receipt |
| `inspect_canonical_collection_readonly.py` | Stdlib-only metadata snapshot using explicit paths and candidate parameters |

Preprocessing creates derived artifacts. It does not rewrite Bundles, relabel
reports, remove failures, select games by outcomes or replace invalid source
evidence. Original collection data remain intact.

## Configuration and different collection sizes

The campaign has exactly four fields:

| Field | Constraint |
|---|---|
| `collection_id` | Safe, independently allocated campaign identity |
| `target_games` | Positive integer success target |
| `seed_pool_size` | Positive integer, at least the target; full pool frozen before first claim |
| `call_limit` | Positive integer dispatch-call cap, bound in the Plan |

The optional Split configuration has exactly `collection_id`,
`development_count`, `test_count` and `split_seed`. Counts cover the whole success
target; Development is at least five games and Test at least one. Assignment
uses the existing versioned stable identity/hash rule and never labels, roles,
winners or model performance. Bundle digests bind evidence without selecting
membership.

Five Development folds retain the V1 assignment algorithm. Counts divisible by
five produce equal folds; otherwise fold sizes differ by at most one. The
examples use 10 total, 8 Dev and 2 Test, demonstrating unequal fold sizes.
Larger counts use the same contracts without a special protocol version.

Non-sensitive templates:

- `configs/operator/canonical_collection.example.json`
- `configs/operator/canonical_split.example.json`
- `configs/operator/acquisition_storage.example.json`

Create reviewed external activity files with a new ID and absolute Artifact
root. The examples are not a frozen production Plan. The existing operator
runtime/deployment profiles remain its implementation defaults; deployment must
review and freeze those profiles and model/environment identities. This
migration adds explicit Artifact storage selection, not a new model-service
deployment system.

### Collection without a Development/Test split

Existing collection already supports a Plan with no Split Protocol binding.
Omit both Split options for that path. Canonical collection, Ledger, original
Bundle validation and standalone native terminal audit remain available.
V1 `publish-development` still publishes all successes from a complete unbound
collection and requires enough games for five folds.

A split-bound Plan requires both the Protocol path and its trusted digest on
new runs and resumes. It cannot use V1 publication or the old Final Test path.
V2 requires both partitions; zero-Test or post-hoc split variants are not added.
Decide whether to bind a Split before freezing the Plan, not after acquisition.

## Preparing and resuming an activity

Before production, approve the source SHA, model/HF identity, Prompt/schema,
runtime/service configuration, target, call cap, full seed pool, exclusions,
storage paths and any Split Protocol. Check all historical planned pools, not
only previously successful seeds. Qualification uses a separate identity and
does not become part of a formal success target.

The commands below are templates, not authorization to execute production.
They require an installed checkout with the approved clean source and matching
runtime; the new dirty migration worktree cannot qualify as that source.

```sh
python -m werewolf.cli --storage-profile /absolute/path/to/storage.json \
  prepare-canonical-split-protocol --config /absolute/path/to/split.json \
  --destination split-protocol-id

python -m scripts.collect_games --campaign /absolute/path/to/campaign.json \
  --storage-profile /absolute/path/to/storage.json \
  --split-protocol split-protocol-id --split-protocol-digest TRUSTED_PROTOCOL_SHA256
```

Resume the same operator command with `--resume`. All frozen identities,
parameters and paths must match. An already durable claim consumes its seed;
interruption uses the existing Ledger recovery rules. No reroll, pool expansion,
silent restart or replacement destination is permitted. Collection owns
Canonical failure eligibility; this branch does not redefine it.

`uns collect` remains the low-level consumer of the explicit frozen Plan.
Neither operator nor low-level collection can silently omit a bound Protocol.

## Validation, terminal audit and release

Complete Collection verification requires the exact success target, closed
Ledger, verified claims/terminals, failed-attempt evidence and every successful
Bundle. PRE, observations, effective belief reports, public history and private
replay retain their existing definitions.

The independent native audit runs no backend. It verifies legal role counts and
configuration, actual engine terminal/winner, legal actor/action order, no actions
after termination and equality of replayed and recorded public history. Its
schema is `classic7_native_terminal_audit_v1`. The old replay identity is unchanged.

**A terminal error blocks release; it does not reclassify a Canonical success or
authorize deleting/replacing that game.** Stop, retain evidence and review any
scientific protocol change separately.

The production release entry point is:

```sh
python -m werewolf.cli --storage-profile /absolute/path/to/storage.json \
  release-canonical-development-v2 --collection collection-id \
  --runtime-config /absolute/path/to/runtime.yaml \
  --split-protocol split-protocol-id --split-protocol-digest TRUSTED_PROTOCOL_SHA256 \
  --destination release-id
```

It validates all native terminals before creating the Split/Dev outputs, reopens
parents, verifies whole-game coverage/folds, compares Development public records
and role assignments with their authoritative source Bundles, and finally writes
a receipt. Source-side receipt verification repeats these comparisons.
The receipt binds Plan, Collection, Protocol, audit, Split, Dev and Test identity.
Schema: `classic7_canonical_collection_release_v1`.

For independent checks, `audit-canonical-terminals` accepts a complete collection.
Low-level `publish-canonical-split` and `publish-development-v2` remain available,
but are not substitutes for the controlled production release gate.
`validate-artifact` requires the trusted Protocol digest for a Protocol, the
trusted Split digest for a Split or Development V2, and the trusted receipt
digest for a release. Standalone terminal audit validation additionally requires
`--collection` and `--runtime-config`. Receipt validation uses its bound source
references and embedded runtime configuration, so it needs neither extra flag.
Use absolute artifact paths for validation; source parents must remain available
in the recorded Artifact-root layout.

```sh
python -m werewolf.cli --storage-profile /absolute/path/to/storage.json \
  validate-artifact /absolute/path/to/research-artifacts/releases/release-id \
  --expected-digest TRUSTED_RECEIPT_SHA256
```

An identical immutable publication request validates/reuses its result. Changed
inputs or an incompatible existing destination reject. On a partial release,
retain intermediate artifacts and retry with the exact original inputs; never
overwrite conflicting evidence or report success without a complete receipt.

## Data access boundaries

Full Collection, Plan/Ledger, full Split, per-game audit outcomes, failure partials,
backend traces, service logs and backups remain in source/archive custody.
Source-side verification legitimately checks every game before release.

Dev projection filters membership before public projection or role extraction;
it does not read Test Bundles to construct Development records. Dev public
readers reject Test IDs and unknown inventories and do not open restricted role
bytes. Only Dev roles enter its restricted sidecar. Five-fold partitions contain
only Dev games. The full Split and source archive are not training inputs.

Current policy is `same_account_process_isolation_v1`: operator procedure and
audit records on one account. It is not OS-enforced blindness. A privileged or
same-account owner can read source data, and known seeds may reveal reconstructible
role information. Do not claim inaccessible Test labels. Production seed arrays
and actual activity files stay outside GitHub.

Independent Test has a verifiable identity, not a newly implemented evaluation
consumer. Existing model seals remain enforced; the old Final Publication rejects
split-bound Test data. New ToM V2 readers/training adapters are not migrated.

V1 data-schema compatibility does not grant a new source revision permission to
evaluate historical sealed models. Backbone Evaluation checks inference source
bytes against its frozen training revision, including `attempt_ledger.py` and
`development_publication.py`. The data-branch changes to these files therefore
trigger that existing rejection. Historical sealed evaluation must use its
approved source checkout; this branch adds no source allowlist or seal bypass.
The branch-wide CI test policy for these historical byte-equality checks needs
separate review before claiming a passing complete research suite.

## Read-only inspection and monitoring

The metadata inspector requires explicit `--repo` and `--data-root`. Optional
reference collection checks require `--collection` plus trusted `--plan-digest`.
Candidate files use repeated `--campaign`, with explicit repeated `--history-root`;
`--excluded-seeds` supplies a reviewed JSON array of exclusions. Additional output
roots, runtime/serve files, model revision file, environment directories and
service port have explicit options and no personal defaults.

It prints JSON, uses no SSH/HTTP, imports no project/Torch/vLLM and runs no GPU
initialization, generation, recovery or publication. Historical pool summaries
are metadata checks, not full Canonical or weight qualification. Supplied roots
must cover the relevant archives; a local metadata result does not prove a remote
server was inspected. Users perform all server operations manually.

## Manual server deployment

All server operations are performed by the researcher, with separate approval
for inspection, deployment, service startup, qualification and production. Local
test acceptance does not establish server readiness or authorize collection.

1. After the reviewed commit is pushed, create a separate clean server checkout
   at its approved full SHA. Verify `git rev-parse HEAD` and an empty
   `git status --porcelain --untracked-files=all`. Retain existing research
   checkouts, collections and frozen models.
2. Install this checkout in the selected environment; verify that `werewolf`
   resolves to this checkout. Keep actual campaign, storage, model and service
   records outside Git. Review the inherited runtime/deployment profiles and
   their explicit local paths before using the operator.
3. Manually inspect historical seed pools, capacity, disk and identities. Approve
   service startup and a separately identified qualification campaign; retain
   qualification evidence without adding it to the production target.
4. After qualification, approve the formal four-field campaign and complete seed
   pool, freeze any Split Protocol before the Collection Plan, and use the
   operator commands above. Resume only the same clean SHA and frozen inputs.
5. After the complete Collection validates, run the controlled Release command
   and independently verify its trusted receipt digest. Archive all parents in
   the recorded layout and rehash the archive; distribute only the Development
   projection through approved development workflows.

The approved commit is the execution identity. Do not pull into a running
campaign checkout, edit its source or profiles, or resume from a moving branch.
Deployment commands and measured resource budgets belong to the approved private
activity record, not these generic templates.

Monitor attempt/success throughput, failure categories, open claims, call usage,
Ledger validation time, GPU/process state, disk bytes/inodes and remaining margin.
Freeze activity-specific stopping thresholds before execution. Systemic failures,
terminal/provenance errors, disk/fsync/OOM problems or severe sustained degradation
stop acquisition; preserve all evidence and the original resume contract.

## Archiving and local acceptance

Seal the original collection and all parent artifacts, complete failure Ledger,
approved source SHA, model/HF and weight identity, actual service arguments,
runtime/environment versions, full pool, parameters, timestamps, resource/call
statistics and trusted digests. Separate source/Test-bearing archives from Dev
distribution. Keep existing collections and frozen models intact. Independently
rehash a backup and reopen its artifact bindings; a Git commit is not a data backup.

Local migration acceptance uses synthetic fixtures and CPU native engine replay.
It covers V1/V2 compatibility, Plan binding, tampering, deterministic disjoint
coverage, unequal folds, Dev/Test information boundaries, immutable recovery,
terminal release gates and a child process that rejects all ToM/Torch/Transformers
imports while publishing and validating V2. No real LLM, collection or training
is executed. Both original dirty worktrees and the verified private backup are
checked separately before and after migration.
