# Branch Registry

This is the initial registry draft on `data/acquisition-preprocessing`. The
version on `twd/mainline` is to become the authoritative registry after a
separately reviewed documentation integration. This draft does not change the
dirty research worktree or authorize any merge, commit, push, rename or deletion.

## Branch responsibilities

Inventory snapshot: 2026-10-13. Commit IDs below are inventory snapshots, not
instructions to resolve a moving branch for an experiment. Every frozen run uses
the complete source SHA recorded in its Plan or research manifest.

| Existing branch | Responsibility | Snapshot and history | Maintenance |
|---|---|---|---|
| `main` | Historical GitHub default and Phase-1 compatibility baseline | `33d580c`; ancestor of the research mainline | Preserve old source identities; do not automatically advance |
| `twd/mainline` | Core research mainline | `d8397ce`; cumulative ToM and Phase-2 engineering | Receive separately reviewed research and shared infrastructure changes |
| `data/acquisition-preprocessing` | **Data Acquisition & Preprocessing** | Local branch created from full SHA `d8397ce56c65d6d46b24a6b53f0b71431c04267b`; initial migration uncommitted | Maintain collection, evidence validation, audit, Split, Publication and release; independent of campaign size |
| `research/main-experiments` | Independent Experiment A | `13ca742`; two independent commits above common base `57fd2d7` | Retain separately; development paused; no integration in this migration |
| `tom/backbone-train` | Backbone preparation and training engineering milestone | `ebf98b7`; included in research-mainline ancestry | Retain milestone identity |
| `tom/backbone-eval` | Backbone OOF, contrasts and Qwen3 Final Fit milestone | `f199a9e`; included in research-mainline ancestry | Retain milestone identity |
| `ablation/tom-gameplay` | Paired ToM gameplay protocol, collection and analysis milestone | `be6bf36`; included in research-mainline ancestry | Retain milestone identity |
| `feature/canonical-split-publication-v2` | Original V2 development and preservation lineage | Local-only at `d8397ce`; mixed data and ToM adaptations remain uncommitted | Keep its worktree and all original work; selectively copy approved data changes |

At this snapshot, the six historical published branches have remote counterparts.
The V2 and data branches are local-only. There are no additional named planned
branches approved for creation. `feature/canonical-5800-collection-v2` does not
exist in the audited inventory. Future planned branches must be listed as planned
until creation is verified; remote publication is a separate status.

## Historical relationships

The committed ancestor chain is:

```text
main → tom/backbone-train → tom/backbone-eval → ablation/tom-gameplay
     → twd/mainline
```

Experiment A diverged at `57fd2d7` with commits `39f4ea2` and `13ca742`.
The V2 development branch and this data branch start at `d8397ce`; the data
migration does not include the research worktree's uncommitted gameplay changes.
Ancestor inclusion, patch equivalence and research completion are different
claims. A branch with no unique commits can still contain unique worktree files
or identify a frozen experiment milestone.

## Identities and storage

| Object | Meaning |
|---|---|
| Branch | Moving reference to code history; insufficient as a frozen run identity |
| Git commit | Exact source revision; use the complete SHA for provenance |
| Worktree | A checkout that may contain staged, unstaged or untracked work |
| Collection ID | Identity of one acquisition campaign; distinct from branch identity |
| Collection Plan | Immutable contract binding source, runtime, model, full seed pool and acquisition controls |
| Split Protocol / Manifest | Preregistered assignment rule and verified whole-game membership for a completed collection |
| Publication / release receipt | Versioned, digest-bound data projection and validation evidence |
| Data, models and logs | External research artifacts with their own provenance and custody |

GitHub stores source, non-sensitive configuration examples and documentation.
Raw games, roles, belief records, full production seed pools, weights, checkpoints,
actual deployment configuration and server logs belong to external campaign
storage and private backups. A public-game projection is not permission to
publish research records on GitHub.

## Maintenance rules

1. Infrastructure changes start on the data branch; model training, evaluation
   adaptation and decision research retain their research ownership.
2. Review changes to shared modules before integrating them between long-lived
   branches. Do not independently maintain incompatible schema meanings.
3. A running campaign continues from its frozen clean revision even when a
   branch advances. Never replace its Plan, seed pool or source identity.
4. Preserve dirty worktrees before any future cleanup. Commit counts alone do
   not justify deleting a branch or worktree.
5. Keep Experiment A and historical ToM branches during this migration.
6. Update this registry for branch lifecycle decisions. Runtime procedures,
   parameters and verification belong to the module document and campaign
   archive, not this registry.

Technical documentation: [Data Acquisition & Preprocessing](research/data-acquisition-preprocessing.md).
