# Wolf ToM gameplay ablation protocol

## Research question and arms

We compare **Wolf+ToM** with **Wolf-NoToM** on wolf-side win rate. Both arms use the same game rules, gameplay code, Qwen3.5-9B backend, runtime configuration, call limit, seed, and initial role assignment. The only assigned difference is whether an eligible wolf receives frozen Qwen3 ToM information during day cognition immediately before public `speech` or `speech_pk`. Wolf-NoToM follows the original speech cognition path without a placeholder or synthetic ToM value. This is a two-arm ablation; no third decision arm is included.

## Paired design and treatment boundary

For each candidate seed, attempt the arms in the frozen execution order, subject to the pair-level fail-fast rule below. **A pair enters the primary analysis only if both Wolf-NoToM and Wolf+ToM complete successfully with valid evidence under that same seed.** Run the two arms in separate runtime instances under the same production configuration and verify matching initial roles. One completed seed pair is one experimental unit; the arms are not independent samples. Keep the gameplay source revision and backend configuration fixed across the formal run.

At each eligible boundary, use the canonical Public PRE frozen by `recorder.before_agent_act()`. Wolf+ToM calls the sealed predictor only for a living wolf at `speech` or `speech_pk` and inserts its private context only into that wolf's day-cognition request. Observer eligibility is determined from the wolf's legal observation and public alive seats, without environment role truth or restricted label sidecars. The predicted matrix describes each observer's `suspected_werewolves` self-report distribution over target seats; its entries are **not** ground-truth role probabilities.

ToM has no direct input path to speech realization, `vote`, `vote_pk`, or night actions, and is not written to persistent agent state or the public transcript. Speech changed by treatment may subsequently alter the public game trajectory, including votes and night decisions; those are treatment downstream effects. A completed Wolf+ToM game with zero eligible treatment boundaries remains in the assigned-arm analysis; treatment count is diagnostic, not an inclusion criterion.

## Frozen predictor

Use `werewolf.tom.qwen3_final.SealedQwen3Predictor` through the existing gameplay predictor client. Its frozen identity is fit `87c2485acf48443034c59430d79f098a13151677852988bf691599ca2d2f8248`, terminal `b977ec24d6292ad329e664c931a1aae39dc4738b4b33685b19d92fd4fb1921af`, model `fdf858b5e21a1951da9ed2c2b9e988f8a1f1edcf99d2bb84f54faf055dc7e3e9`, and seal `fcbb81ee43050142ebdf01c0aee576156221d725598ff5ee2c02055049f72718`. The predictor runs from its separate clean source checkout at `f199a9e6c64168a412de91ff8ca04d0c795c57e2`. Do not retrain, select another checkpoint, relax fit/seal/source verification, or expose restricted role truth or labels to gameplay.

## Outcomes and paired analysis

The primary endpoint is binary wolf-side victory in a completed game with valid canonical evidence. For valid seed pair $i$, let $Y_i^+$ and $Y_i^0$ indicate wolf victory under Wolf+ToM and Wolf-NoToM. Define $n_{01}$ as the number of pairs with a NoToM loss and a +ToM win, and $n_{10}$ as the number with a NoToM win and a +ToM loss. Report both arm-specific wolf win rates, $n_{01}$, $n_{10}$, and the primary effect

\[
\widehat{\Delta}=\frac{1}{N}\sum_{i=1}^{N}(Y_i^+-Y_i^0)
=\widehat{p}_{+}-\widehat{p}_{0}
=\frac{n_{01}-n_{10}}{N}.
\]

Construct a 95% confidence interval by bootstrapping **whole seed pairs**. Test the paired binary outcome with a two-sided McNemar-type test based on the discordant counts $n_{01}$ and $n_{10}$. Do not treat the two arms as independent Bernoulli samples or use an independent-proportions test. A non-significant result does not establish that ToM is ineffective; it means this limited gameplay sample did not yield clear statistical evidence of an effect.

## Sample size and stopping rule

The formal experiment identity is `paper-tom-gameplay-ablation-v1`. Before formal play, derive and freeze the ordered pool of **100 candidate seeds** with the project's existing `scripts.collect_games.derive_seed_pool("paper-tom-gameplay-ablation-v1", 100)`. Candidate ordinals are zero-based positions in this pool. The pool size is not the statistical sample size.

This is a **fixed computational-budget design** with a formal target of **$N=40$ completed, valid seed pairs**: 40 Wolf-NoToM games plus 40 Wolf+ToM games, or **80 valid gameplay runs**. Both arms must be valid for a seed to count. These are 40 paired experimental units, not two independent Bernoulli samples. Observed server gameplay time is about 5 minutes per game, so 80 games are expected to take about 400 minutes (6 hours 40 minutes). The approximately 8-hour collection budget leaves about 1 hour 20 minutes for long games, predictor startup/shutdown, service variation, and a small number of technical failures. This is a planning estimate, not a time-based stopping rule.

The sample size is determined by compute budget, **not** by a claim of 80% power to detect a 10 percentage-point win-rate difference. For sensitivity context only, with planning discordant-pair rate $q\approx0.50$, two-sided $\alpha=0.05$, and target power $0.80$, a paired/McNemar-style approximation gives a detectable absolute effect of about $0.31$ at $N=40$. The experiment is therefore mainly informative about sizable gameplay utility effects and has limited power for small effects.

Process candidate seeds strictly in pool order and stop after the first 40 pairs in which both arms are valid. After a technical failure, advance to the next prederived seed. Never select, skip, replace, or stop based on wins, effect size, or interim statistics; there is no effect-dependent early stopping. Fix analysis details before collecting formal outcomes. If the pool is exhausted before 40 valid pairs, report the shortfall; do not derive ad hoc replacements.

Balance execution order deterministically by candidate pair ordinal: even ordinals run **Wolf-NoToM then Wolf+ToM**; odd ordinals run **Wolf+ToM then Wolf-NoToM**. Order controls execution only, not the paired analysis: whenever both arms run, they use that candidate's same seed and matching initial roles. Regardless of order, the runner must not create or hold a `Qwen3GameplayPredictorClient` during Wolf-NoToM; create it before Wolf+ToM, close it after that game, and do not retain it across pairs.

## Technical failures, diagnostics, and smoke

Backend, predictor, call-budget, and canonical-evidence failures are technical failures, not wins or losses. If the first arm has a confirmed `canonical_failure`, `interrupted_failure`, or another protocol-defined terminal technical failure, immediately mark the candidate pair invalid. Do not run its second arm or retry that seed; advance to the next prederived candidate. This pair-level fail-fast rule avoids spending gameplay budget on a candidate that cannot enter the primary analysis.

If the first arm succeeds and the second fails, the pair is invalid. Retain the first arm's successful artifact for audit, exclude it from the primary analysis, and advance to the next candidate. If the first arm has a durable success terminal and the second has not been claimed when the process exits, resume the same candidate at the second arm without rerunning the first. An open claim left by a terminated process follows canonical collection recovery: close it as `interrupted_failure`, mark the pair invalid, and rerun neither that arm nor that seed. If both arms have durable success terminals but the pair record is missing, resume by revalidating both artifacts and publishing the pair record without rerunning gameplay.

Record each invalid candidate's ordinal, seed, affected arm, terminal outcome, and failure stage/type; record interruption as such without inferring an unobserved technical cause. Failed pairs do not count toward 40 completed pairs. Continue through later prederived candidates until 40 valid pairs are obtained; never combine games from different seeds. Technical failures can make the number of attempted games exceed 80. Because fail-fast leaves the second arm unattempted after a first-arm failure, per-arm technical failure rates must not be interpreted as a complete, symmetric arm-level comparison.

Report attempted candidate pairs, completed valid pairs, invalid pairs, and failure reasons with counts, identifying `interrupted_failure` separately from confirmed technical failures. Record game length or terminal day, backend call count, speech-boundary count, and treatment count as auxiliary diagnostics, not primary endpoints. The completed paired gameplay smoke verifies execution and treatment containment only. Its games and outcomes are excluded from the 40 formal seed pairs.
