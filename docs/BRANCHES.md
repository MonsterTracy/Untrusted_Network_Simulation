# Branch Registry

Effective registry on `twd/mainline`, updated 2026-10-24. This supersedes the
2026-10-13 Data-branch draft. The functional integration baseline before this
documentation update is `29fba64f64b3ec310d3f12ecd4247ff46879ec54`.
The recorded Mainline tip below is this historical snapshot; future commits may
advance it.

`twd/mainline` is the single ongoing research development and integration line
for ToM, Baselines, data, three-way decisions and speech research. `main` remains
the GitHub stable-release/default branch; update it only after complete project
acceptance and separate release authorization. Other refs remain pending research
sources or frozen history, not parallel long-term development lines.

## Integrated research sources

| Module | Preserved source identity | Included scope and limits |
|---|---|---|
| Data Acquisition & Preprocessing | `99f559c5f274f79fbaf4e2056b40c96c84441531` | Approved collection, audit, split, publication and release infrastructure; not all uncommitted V2 work |
| Observer-DTRS D1–D3 | `7b51f276b04b386ec1aa4ae44454c1d809a10897` | PRE/data contracts, conditional event estimation, losses, risks, thresholds and POS/BND/NEG decisions |
| Three-round contextual speech prototype | `2f5043f80096cf18521901580e95f050454b7664` | Eight opt-in source/test/document files; synthetic paired comparison and controlled verified speech, not a frozen experiment |
| D4 engineering | `2f4f7c7ece3610e0d69d6db00af9c72c8c0591da`, merged at `24a87eb8d77335392154875c71c5d1e10184ced2` | Capsule, Resident Service, Broker, Shadow Hook and CPU qualification infrastructure; W1 drafts excluded |
| Experiment A | Execution `39f4ea25906808bf1ed28b05fdfb63865cc3ffb3`; Analysis `13ca742d7d98e41d5f68e5cb98ad5901f017115c`; merged at `29fba64f64b3ec310d3f12ecd4247ff46879ec54` | Paired temporal/Q-representation development paths and mapper contrasts; original execution/analysis identities remain frozen |

Source inclusion, supported runtime admission and formal scientific validation
are separate gates. Synthetic CPU checks do not establish real-model service
qualification, Final/CAL/TEST admission, language effectiveness or win-rate gains.
A new Mainline revision cannot impersonate an older experiment's source SHA.

## Registered branches

Tips are inventory snapshots, not instructions to run an experiment from a moving
branch. Publication below follows the last verified GitHub Push receipt on
2026-10-24 (nine remote branches), not a new server or deployment check.

| Branch | Recorded tip | Responsibility / Mainline relationship | Retention / publication |
|---|---|---|---|
| `twd/mainline` | `29fba64f64b3ec310d3f12ecd4247ff46879ec54` | Current research development and integration, in W3 | Active; published |
| `main` | `33d580cdf0700f6e38ae5b14648fcfdaa9be2b9a` | Stable release; ancestor of research Mainline | Preserve until separate release; published |
| `data/acquisition-preprocessing` | `99f559c5f274f79fbaf4e2056b40c96c84441531` | Approved Data source fully inherited | Retain provenance ref; published |
| `three-way` | `2f4f7c7ece3610e0d69d6db00af9c72c8c0591da` | Committed D1–D4 source inherited; W1 drafts are not included | Retain source/drafts; published |
| `research/main-experiments` | `13ca742d7d98e41d5f68e5cb98ad5901f017115c` | Both Experiment A commits inherited; frozen study source remains distinct | Retain execution/analysis lineage; published |
| `tom/backbone-train` | `ebf98b7e3249f78d3f1fa1fd46c20affe0486913` | Preparation, training and recovery milestone already in ancestry | Retain frozen source; published |
| `tom/backbone-eval` | `f199a9e6c64168a412de91ff8ca04d0c795c57e2` | OOF, contrasts and Qwen3 Final Fit lifecycle milestone already in ancestry | Retain frozen predictor source; published |
| `ablation/tom-gameplay` | `be6bf36b43727ca8ac2ec0e66c505aa65c66df42` | Paired ToM gameplay protocol/collection/analysis source already inherited | Retain experiment provenance; published |
| `codex/tom-analysis-tools` | `610790ff3eed880ec56ed00f491753ea85b5b9a9` | Published visualization tools not integrated; reviewed 17-PRE input unavailable, integration deferred | Retain tools and private W4 cases; published ref |
| `feature/canonical-split-publication-v2` | `d8397ce56c65d6d46b24a6b53f0b71431c04267b` | Ref is an ancestor; W2 has uncommitted Data/ToM adaptations and private experiment identities | Retain all work; local ref only |
| `codex/tom-llm-reasoner` | `d8397ce56c65d6d46b24a6b53f0b71431c04267b` | Ref is an ancestor; W5 LLM Reasoner implementation remains uncommitted and unintegrated | Retain Candidate/source gates; local ref only |
| `feature/observer-dtrs-d4-shadow` | `7b51f276b04b386ec1aa4ae44454c1d809a10897` | Ref is an ancestor; W7 retains original uncommitted D4 bytes and provenance | Read-only source copy; local ref only |

Ancestor inclusion does not include an uncommitted overlay. In particular, merging
the W2/W5 ref again would not preserve its working files. Integration of
`codex/tom-analysis-tools` is postponed, not a prerequisite for Mainline development.
This is distinct from Experiment A's retained Analysis Source. `tom-baseline` and
`experiment/tom-baselines` are not registered/existing branches; no new Baseline
branch is planned. Future approved Baseline work goes to `twd/mainline`.

## Seven retained Mac worktrees

All seven share `~/Desktop/VscodeProjects/Untrusted_Network_Simulation/.git`.
Here `~` denotes the original Mac user's home directory for these recorded paths.
Paths and pending-file counts below are the 2026-10-24 snapshot before this
three-document update; counts exclude ignored payloads.

| ID | Path | Branch / current responsibility | Keep state |
|---|---|---|---|
| W1 | `~/Desktop/VscodeProjects/Untrusted_Network_Simulation` | `three-way`; common-store primary checkout, two uncommitted CI/fixture drafts | Preserve original bytes; not the current development checkout |
| W2 | `~/.codex/worktrees/canonical-split-publication-v2/Untrusted_Network_Simulation` | `feature/canonical-split-publication-v2`; 18 modified + 22 untracked Data/ToM files | Preserve; private Test seeds/campaign/split identities not approved for public release |
| W3 | `~/.codex/worktrees/data-acquisition-preprocessing/Untrusted_Network_Simulation` | `twd/mainline`; active development/integration checkout, clean at the source snapshot | Continue here; directory name no longer denotes branch ownership |
| W4 | `~/.codex/worktrees/tom-analysis-tools/Untrusted_Network_Simulation` | `codex/tom-analysis-tools`; deferred tools plus three untracked real-role case files | Preserve; those cases must not enter public Git history |
| W5 | `~/.codex/worktrees/tom-llm-reasoner/Untrusted_Network_Simulation` | `codex/tom-llm-reasoner`; 3 modified + 54 untracked files, plus ignored Candidate material | Preserve d839 source + overlay and frozen Candidate v2; engineering admission precedes source migration |
| W6 | `~/Desktop/VscodeProjects/research-worktree` | `research/main-experiments`; clean frozen Experiment A source checkout | Retain paired execution/analysis provenance and artifact dependencies |
| W7 | `~/Desktop/VscodeProjects/Untrusted_Network_Simulation_D4_Shadow` | `feature/observer-dtrs-d4-shadow`; 2 modified + 49 untracked original D4 files | Read-only source copy; no duplicate development or cleanup |

Keep every worktree for now. Branch retention, worktree archival/removal, directory
migration, remote-ref deletion and server deployment retirement are different
operations. Each needs separate authorization and recovery/dependency checks.
W2–W5 are Codex-managed: do not manually move them. W1 also owns the common Git
store. An ancestor ref or successful source commit is not proof that uncommitted
or ignored payloads are recoverable; existing preservation limits still apply.

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

## Source, artifacts and maintenance

- New research work defaults to W3 `twd/mainline`, with one agreed writer at a time.
  Existing pending tasks retain their source until separately approved; do not
  create branches or worktrees just to rename a research responsibility.
- Frozen runs use full SHA plus their original manifest, config, seed, digest and
  seal contracts. A branch advance or source integration neither upgrades a
  deployed model nor rebinds an existing artifact. See [server version roles](server-execution.md#source-versions-and-server-responsibilities).
- GitHub holds approved source, non-sensitive examples and documentation. Raw
  games, real roles, belief records, production seed pools, weights, checkpoints,
  deployment credentials/configuration and server logs remain external/private.
  W2 Test identities and W4 real-role cases remain withheld; W5 Candidate material
  must not be uploaded as source.
- Preserve shared stash, reflog-only/unreachable objects and existing preservation
  records; no branch/worktree cleanup is authorized by this registry. Stable
  `main` publication remains a separate project-level acceptance decision.

Technical entry points: [Data](research/data-acquisition-preprocessing.md),
[controlled speech prototype](research/observer-dtrs-contextual-speech-prototype.md),
[ToM contract](tom_contract.md).
