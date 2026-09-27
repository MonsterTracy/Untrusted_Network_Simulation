# Wolf ToM gameplay ablation protocol

## Research question and arms

We compare **Wolf+ToM** with **Wolf-NoToM** on wolf-side win rate. Both arms use the same game rules, gameplay code, Qwen3.5-9B backend, runtime configuration, call limit, seed, and initial role assignment. The only assigned difference is whether an eligible wolf receives frozen Qwen3 ToM information during day cognition immediately before public `speech` or `speech_pk`. Wolf-NoToM follows the original speech cognition path without a placeholder or synthetic ToM value. This is a two-arm ablation; no third decision arm is included.

## Paired design and treatment boundary

For every candidate seed, assign both arms to separate runtime instances under the same production configuration. Verify that their initial roles match. **One seed and its two games form one paired experimental unit**; the arms are not independent samples. Keep the gameplay source revision and backend configuration fixed across the formal run.

At each eligible boundary, use the canonical Public PRE frozen by `recorder.before_agent_act()`. Wolf+ToM calls the sealed predictor only for a living wolf at `speech` or `speech_pk` and inserts its private context only into that wolf's day-cognition request. Observer eligibility is determined from the wolf's legal observation and public alive seats, without environment role truth or restricted label sidecars. The predicted matrix describes each observer's `suspected_werewolves` self-report distribution over target seats; its entries are **not** ground-truth role probabilities.

ToM has no direct input path to speech realization, `vote`, `vote_pk`, or night actions, and is not written to persistent agent state or the public transcript. Speech changed by treatment may subsequently alter the public game trajectory, including votes and night decisions; those are treatment downstream effects. A completed Wolf+ToM game with zero eligible treatment boundaries remains in the assigned-arm analysis; treatment count is diagnostic, not an inclusion criterion.

## Frozen predictor

Use `werewolf.tom.qwen3_final.SealedQwen3Predictor` through the existing gameplay predictor client. Its frozen identity is fit `87c2485acf48443034c59430d79f098a13151677852988bf691599ca2d2f8248`, terminal `b977ec24d6292ad329e664c931a1aae39dc4738b4b33685b19d92fd4fb1921af`, model `fdf858b5e21a1951da9ed2c2b9e988f8a1f1edcf99d2bb84f54faf055dc7e3e9`, and seal `fcbb81ee43050142ebdf01c0aee576156221d725598ff5ee2c02055049f72718`. The predictor runs from its separate clean source checkout at `f199a9e6c64168a412de91ff8ca04d0c795c57e2`. Do not retrain, select another checkpoint, relax fit/seal/source verification, or expose restricted role truth or labels to gameplay.

## Outcomes and paired analysis

The primary endpoint is binary wolf-side victory in a completed game with valid canonical evidence. For valid seed pair $i$, let $Y_i^+$ and $Y_i^0$ indicate wolf victory under Wolf+ToM and Wolf-NoToM. Define $n_{01}$ as the number of pairs with a NoToM loss and a +ToM win, and $n_{10}$ as the number with a NoToM win and a +ToM loss. Report each arm's win rate and the primary effect

\[
\widehat{\Delta}=\frac{1}{N}\sum_{i=1}^{N}(Y_i^+-Y_i^0)
=\widehat{p}_{+}-\widehat{p}_{0}
=\frac{n_{01}-n_{10}}{N}.
\]

Construct a 95% confidence interval by bootstrapping **whole seed pairs**. Test the paired binary outcome with a two-sided McNemar-type test based on the discordant counts $n_{01}$ and $n_{10}$. Do not treat the two arms as independent Bernoulli samples or use an independent-proportions test.

## Sample size and stopping rule

The formal experiment identity is `paper-tom-gameplay-ablation-v1`. Before formal play, derive and freeze the ordered pool of **1000 candidate seeds** with the project's existing `scripts.collect_games.derive_seed_pool("paper-tom-gameplay-ablation-v1", 1000)`. Candidate ordinals are zero-based positions in this pool. The pool size is not the statistical sample size.

The formal target is **400 completed, valid seed pairs**: 400 Wolf-NoToM games plus 400 Wolf+ToM games, or **800 completed gameplay runs**. This is a design with 400 paired experimental units, not two independent samples of 400 Bernoulli observations. For the paired binary wolf-win endpoint and effect $\Delta=P(\text{wolf win}\mid\text{+ToM})-P(\text{wolf win}\mid\text{NoToM})$, planning uses two-sided $\alpha=0.05$, target power $0.80$, minimum detectable absolute win-rate difference $0.10$, and discordant-pair rate $q=0.50$. A paired/McNemar-style approximation requires approximately 391 seed pairs; the target is rounded upward to 400.

Process candidate seeds strictly in pool order and stop after the first 400 pairs in which both arms are valid. After a technical failure, advance to the next prederived seed. Never select, skip, replace, or stop based on wins, effect size, or interim statistics. Fix analysis details before collecting formal outcomes. If the pool is exhausted before 400 valid pairs, report the shortfall; do not derive ad hoc replacements.

Balance execution order deterministically by candidate pair ordinal: even ordinals run **Wolf-NoToM then Wolf+ToM**; odd ordinals run **Wolf+ToM then Wolf-NoToM**. Order controls execution only, not the paired analysis: both arms always use that candidate's same seed and matching initial roles. Regardless of order, the runner must not create or hold a `Qwen3GameplayPredictorClient` during Wolf-NoToM; create it before Wolf+ToM, close it after that game, and do not retain it across pairs.

## Technical failures, diagnostics, and smoke

Backend, predictor, call-budget, and canonical-evidence failures are technical failures, not wins or losses. A seed pair enters the primary analysis only when **both arms** finish with valid evidence. If either arm fails technically, mark the entire pair as failed and exclude both games from the paired outcome denominator. Record the candidate ordinal, seed, affected arm, and failure stage/type; an already completed other arm cannot enter the primary analysis alone. The failed pair does not count toward the 400 completed pairs. Never combine games from different seeds into a pair. Technical failures can make the number of attempted games exceed 800.

Report attempted candidate pairs, completed valid pairs, technical failed pairs, and failure reasons with counts. Record game length or terminal day, backend call count, speech-boundary count, and treatment count as auxiliary diagnostics, not primary endpoints. The completed paired gameplay smoke verifies execution and treatment containment only. Its games and outcomes are excluded from the 400 formal seed pairs.
