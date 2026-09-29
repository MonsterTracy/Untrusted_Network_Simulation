# Phase-2 mapper development protocol

The target is the frozen binary `Theta_AC`; the estimand is `P(Theta_AC=1 | Z)` in the development population. The mapper's raw probability is `p_tilde`, not an observer-level Q mass or a deployment-calibrated `p_hat`. The only input is frozen `Z=(mu,delta,sigma,h,phase,audience_size,competition_size,remaining_speakers_before_vote)`. `S_pre`, post-day S0, resolved Y, vote/exile, reference values/losses, canonical reports, and OOF `q`/`label_observed` are excluded from the design matrix.

## Prespecified model family

| Spec | Context | ToM evidence | Form |
| --- | --- | --- | --- |
| M0 Context Logistic | `phase` and three numeric X fields | none | linear logistic |
| M1 R1 Logistic | same | `mu` | linear logistic |
| M2 R2 Linear Logistic | same | `mu,delta,sigma,h` | linear logistic |
| M3 R2 Additive Logistic | same | same four, each expanded separately | linear logistic over one-dimensional hinge bases |

`phase` has the fixed encoding `speech=0`, `speech_pk=1`; no empirical category frequencies are fitted. Every numeric column is standardized with its training games' population mean and standard deviation (zero standard deviation uses scale 1). M3 uses the standardized evidence's training-fold 1/3 and 2/3 quantiles as two knots per evidence coordinate, producing that coordinate's linear term and two positive-part hinge terms. Context stays linear. There are no cross-feature products or interactions. This is a fixed low-capacity piecewise-linear additive basis, not a spline search. All four specs use unweighted binary negative log likelihood with a separately recorded weak L2 penalty, fixed `C=100`, `lbfgs`, `max_iter=1000`, and `tol=1e-8`. The weak fixed penalty stabilizes correlated hinge columns without making regularization a search variable. There is no class weighting, resampling, or hyperparameter selection in this protocol.

## Splits and provenance

The five existing publication folds partition *whole games*. A mapper fold fits its means, scales, knots, and coefficients only on the other games. Every candidate PRE and candidate from one game remains together. Rows must originate from that game's verified held-out Qwen3 OOF fold. The harness records candidate identity and prefix, mapper fold, game OOF fold, Qwen3 fold prediction digest, fixed evaluation seal, and model digest. A fit manifest records its spec, feature-contract version, training game IDs/digest, training fold membership, OOF provenance, offline-publication/fold/role digests and offline source digest, preprocessing values, likelihood, regularization, and coefficients. No formal model artifact is published by this harness. A separate full-development fit *interface* requires the complete 1,500-game/21,638-row OOF population; it is not invoked in this stage.

This is downstream mapper development CV, **not** a final independent test or end-to-end nested independent validation: the five upstream ToM OOF models were themselves trained using games from the other held-out folds. A future independent `D_cal` must contain games outside ToM/mapper development and use the final frozen Qwen3. No deployment calibration parameter `omega` is fitted or sealed here.

## Evaluation boundary

The **primary model-selection metric** is candidate-pooled binary log loss. Candidate-pooled Brier score is a consistency gate. Game-macro log loss and Brier score compute a metric within each game and then average games equally; game-macro log loss is a promotion gate, while game-macro Brier is reported for robustness. Pooled AUROC, AUPRC, calibration intercept/slope, and fixed equal-width reliability bins are descriptive only. Accuracy, F1, and fixed-threshold metrics do not enter model selection. No classification threshold is chosen.

Calibration intercept/slope come from an unpenalized diagnostic logistic regression of `Theta_AC` on the mapper's saved finite raw score `z`. They are not used to reject a model, and these coefficients are not a deployment calibrator. Prediction probabilities remain unchanged. For log loss alone, exact probability 0 or 1 is numerically clipped to `[1e-15, 1-1e-15]` inside metric evaluation; the saved prediction is never rewritten. Diagnostics with one label class, constant score, or complete separation return unavailable intercept/slope.

## Prespecified paired comparison and selection

Every model must have **exactly the same five-part candidate identities** `(game_id, boundary_id, acting_wolf, phase, candidate_j)`, labels, fold assignment, and upstream provenance. Mismatch fails explicitly. The fixed complexity path is `M0 < M1 < M2 < M3`. For complex model `b` and simpler model `a`, `Delta_LL(b,a) = LL_pool(b) - LL_pool(a)`, so a negative difference favors the complex model. The three primary adjacent comparisons are `M1-M0`, `M2-M1`, and `M3-M2`. `M2-M0` and `M3-M0` are secondary rescue comparisons. The same candidate population is used for every comparison.

The default development interval is a **paired, fold-stratified game-cluster bootstrap** with 100,000 replicates, seed `20260929`, and a two-sided percentile 95% confidence interval (linear empirical quantiles at 0.025 and 0.975). Each replicate samples, with replacement, the original number of held-out *games within each of the five mapper folds*. A selected game contributes all its candidate rows. If drawn multiple times, its entire row panel receives that multiplicity in pooled metrics. All four models receive the identical game multiplicities in a replicate. Game-macro metrics first compute each game's metric once, then average these game-level values weighted by the sampled game multiplicities. Candidate rows are never sampled independently. Intervals for pooled Brier and both game-macro differences are retained as robustness diagnostics; only the pooled log-loss interval enters the promotion gate.

Starting from M0, an adjacent upgrade passes only if **all three** conditions hold: the upper endpoint of the bootstrap 95% CI for its `Delta_LL` is strictly below zero; its candidate-pooled Brier point estimate is no greater than the simpler model's; and its game-macro log-loss point estimate is no greater than the simpler model's. The first failed step stops sequential promotion and retains the selected simpler model. That upgrade is marked `not development-supported / inconclusive`; no AUROC, accuracy, calibration diagnostic, or subgroup result can override this rule. All adjacent differences are still reported even after a stop.

If sequential promotion stops, the two prespecified rescue comparisons are still reported. A rescue model satisfying the **same three gates against M0** is marked `development-supported alternative`, but the harness does not make it the primary selection. A research report must explain the interpretation: an intermediate representation was not supported on its own, while a joint representation may contain additional information. The rescue finding does not retroactively change the adjacent path.

Bootstrap replicates reuse the already cross-fitted mapper predictions; **no mapper is refitted inside a replicate**. The resulting interval describes evaluation uncertainty conditional on those fitted cross-validation models, not uncertainty over the complete training procedure. Small replicate counts are allowed only for synthetic tests. The formal development analysis must use the frozen defaults and must not tune M0-M3, their features, or the M3 basis in response to results.

Phase (`speech`, `speech_pk`), audience size 1/2/3/4, and remaining-speaker subgroups may be reported as diagnostics. They cannot alter automatic model selection or create a phase-specific mapper.

## Separation of development, calibration, and final testing

This protocol governs **development model comparison** on five-fold cross-fitted candidate predictions. The selected specification may later be fit on the full development population, but this comparison itself does not make a deployment model or fitted calibrator. A future independent `D_cal`, outside ToM and mapper development, is reserved for estimating deployment calibration `C_omega`. A future independent final test is reserved for locked evaluation after the model and calibration choices are frozen. Neither set is used by this harness.

The current five-fold mapper CV is **not strict end-to-end nested independent validation**: upstream ToM OOF models trained across other development folds. Accordingly, its bootstrap intervals and selection results are development evidence, not final-test performance claims.
