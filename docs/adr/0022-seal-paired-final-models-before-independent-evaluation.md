# Seal paired final models before independent evaluation

Development OOF remains the five-fold protocol qualification and development
analysis path. A separate Final Experiment trains both temporal conditions on
all games of a Development Publication. It has no fold, held-out partition,
validation selection, early stop, or best checkpoint. Both conditions share
initial tensor bytes, RNG initialization, game/seat schedule and the immutable
temporal table. They use the existing Primary selector, Dataset, model graph,
AdamW/constant-LR optimizer, game-balanced loss and seven-shift cycle budget.
The terminal step count is `7 * rotation_cycles * ceil(N / game_batch_size)`.

Final preparation derives day capacity as `floor(max_seq_len/4)` under
`classic7_public_event_history_v1`, `classic7_authoritative_pre_prefix_v1` and
`classic7_structured_token_planner_v1`, with full-prefix/no-truncation input.
The derivation has its own version and is saved with the frozen sequence
capacity. For L=1024, the one immutable table covers days 0..256 inclusive.
Neither publication statistics nor a manually supplied maximum day determines
this capacity. Both temporal conditions validate the same range; implicit
continues to return zero temporal codes. The current runtime's tighter length
theorem is a sanity check only and is never a model capacity control.

Final numerical controls depend only on frozen configuration and development
training identities. Publication, role-sidecar, actual clean Git HEAD and
implementation/runtime identities remain audit bindings; their digests are
not schedule/RNG seeds. The existing formal protocol's declared source revision
is preserved, while the actual clean preparation HEAD is recorded separately.
Training and recovery require that same clean HEAD and runtime identity.

The two terminal checkpoints must pass their complete recovery-chain audit and
share initial RNG identity before the Final Model Seal can be written. A seal
permanently closes this experiment's training and recovery. Recovery is explicit,
restores model/optimizer/RNG/cursor/log state, and never changes the budget.
Ordinary exceptions mark the lineage failed; abrupt process interruption allows
only the existing deterministic step-boundary policy.

Final Evaluation Publication is a separate plan-closed canonical collection
projection, without folds, bound to an already completed model seal. Its public
record codec and restricted role assignment validation are shared with
Development Publication. The scientific CLI verifies the seal before opening
the final source collection. The evaluation entry point verifies the seal
before opening even the final publication manifest. The final collection must
have disjoint collection, game, bundle and attempted-seed identities from
Development. This audit cannot prove the absence of unrecorded human leakage.

Before loading final payloads, evaluation writes an immutable consumption record
binding the experiment, model seal, publication and scoring version. Another
publication cannot replace it. An explicit resume may only finish or validate
that identical consumption; it cannot select models or data. Every PRE must
pass both sequence and derived day capacity before the first prediction.
Overflow fails the entire evaluation. There is no clipping, subset continuation,
truncation, extension or alternate table. Failed evaluation remains failed.

Final reporting reuses the same row diagnostics, game-macro aggregation and
paired game bootstrap as OOF. Primary and All-Alive use the same predictions;
implicit minus explicit Primary KL uses the same final games and bootstrap
indices. Bootstrap draws are generated only after sealing, from the already
frozen seed and final game identities, and are evaluation artifacts only.
A report is published only after complete prediction and scoring succeed.

This is scientific governance, not an access-control system. Deliberately copying
or republishing previously examined data under new identities can evade it and
is not a legitimate independent evaluation. Existing OOF artifacts, manifests,
source declarations and mathematical contracts are unchanged.
