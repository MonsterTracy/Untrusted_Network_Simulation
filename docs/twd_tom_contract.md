# Current ToM contract

`werewolf.tom.dataset.CanonicalToMDataset` consumes only a verified publication
public view, a publication game partition and an explicit capacity bound.
The unique Structured Token Planner supplies chronological descriptors.
No truncation, window, daily reset, compression or token dropping is allowed.

For a successful nonempty suspicion support, q is uniform on that non-self
support. Successful empty support is observed and uniform on all six non-self
players. Failure/missing observations have zero placeholders and observed=false;
canonical published alive rows cannot have that status.

The random-init Qwen2 graph has hidden size 256, four layers and eight heads.
It uses public structured embeddings, observer queries and local observer-relative
player references. Terminal turn_start's ordinary player reference is the only
upcoming-speaker representation. Self logits are excluded before log_softmax.

Implicit and Explicit use the same graph and initial-state bytes. Explicit adds
the frozen 128-dimensional day and 128-dimensional phase code, with combined
RMS 0.02; Implicit adds zero. Both load and verify the same canonical artifacts.
No temporal parameter is learned or serialized into learned checkpoint state.

Training is Primary-only: eligible AND observed. Per-game mean cross-entropy
is averaged equally over the distinct games present in each batch. Smaller
batches use their actual game count; exact cumulative game coefficients are
audited, not claimed to be equal globally. Full seven-shift cycles ensure
balanced seat exposure. Evaluation uses shift zero only.

Five-fold OOF uses predeclared fixed budgets, terminal checkpoints, paired
initialization and deterministic step-boundary recovery. Outer held-out data
cannot affect training or recovery decisions. All-Alive never trains.

Experiment provenance binds observed implementation source bytes as well as
declared revision, software/runtime and determinism settings. Training, recovery
and evaluation reject runtime/source drift. Before checkpoint-set publication,
the sealer validates each terminal's complete recovery chain and training/RNG/log
provenance; evaluation does not load optimizer or recovery state.

Full-record preflight validation is permitted during preparation or after a
verified all-ten seal. Explicit artifact validation during unsealed training is
rejected before opening held-out games or their eligibility records.

Headline score is per-game mean KL, then game-macro mean. Uniform non-self is
reported on identical rows. Paired game-cluster bootstrap estimates population
expansion penalties (All-Alive minus Primary) and the Primary temporal effect
(Implicit minus Explicit). Negative differences remain negative. Row-weighted,
cross-entropy, TV, absolute-error, support and nonzero-gap normalized-improvement
diagnostics are secondary, with no row-weighted headline CI.
