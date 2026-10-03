# Terminal Estimator Support Review V1

日期：2026-10-03（Asia/Shanghai）。状态：只读 support/identification review；没有拟合任何候选模型或 lambda，没有构造 router、启动 Probe、重新采集或调用 LLM。

【理论建议】当前数据支持完整 ITT marginal risk 的估计，以及低容量 conditional risk 的后续预注册设计；它没有证明全状态、全 score 范围的稳定条件风险估计能力。最充足的局部联合支持位于 `S_pre=(2,5)` 的低 score 区域。`(1,4)` 缺 PUSH，`(2,3)` 极稀疏；高 score 尾部以及 0/1 endpoints 不应作为已经支持的条件风险或经典 lambda。

本报告中的边界来自 assignment 前 covariates 和计数，未依据 L_ref 表现、拟合优度或臂间结果差异选模型或区域。报告提出的支持域和计数分级是下一轮冻结时的建议，尚未成为 runtime gate。全部 120 assignments 始终保留为 primary ITT population。

## 1. Git、数据来源与只读边界

【服务器/数据事实】开始时：

```text
git status --short
?? docs/research/phase2-wolf-pt3wd-review-2026-09-13.md
git rev-parse HEAD
ea40dcc0f59f532611ed4f45606f789d25ac1ad9
git branch --show-current
twd/mainline
```

这是本次 review 的本地 HEAD；formal execution 的 source pin 是 `05c217c98dab74e919e525f76c55ca3a1e12867e`，两者用途不同。

【服务器/数据事实】通过服务器 `werewolf.artifact_io.verify_artifact` 完整核验两个 artifact 的 canonical manifest、file table、bytes/digests/tree。之后读取以下表，在本地 `/tmp` 计算；没有执行发布脚本，也没有实例化 assignment ledger 或重新计算随机化 draw。

| 输入 | 值 |
| --- | --- |
| ITT path | /data/yuxiao/Untrusted_Network_Simulation/paper-studies/paper-phase2-terminal-itt-analysis-v1 |
| ITT manifest digest | 58a8611a0c9f5605f6cbdf5a7a9ea9f0c74fc69e637760e507f073e298cfaf8c |
| formal path | /data/yuxiao/Untrusted_Network_Simulation/paper-studies/paper-phase2-online-terminal-pilot-v1 |
| formal manifest digest | ec1cc8730aefe8fce57ed83a3c84ca9231ae645ea6c9047cd588f8022d2223c2 |
| reference table digest | 1bc2fa51ea771e90389cb511bc0c61a442d862c872a121d61037aca6308f7c2c |
| ITT analysis source SHA256 | 6717f9ca40b7992a97779fb3712927952685e5d75f501801c172b97d0cb58647 |
| source run_inputs digest | b154506d5ec403c0714eac9070525a8cc33fd514ff65f7c58673d88b5a3d113e |
| canonical game-plan digest | add3fe53054d9b89b6c6570b585bce13070f010dcf2e28a40a4671fcd713c64b |


| 读取的文件 | SHA256 |
| --- | --- |
| itt/itt_rows.jsonl | 7ebb3b4848712ea40142ef6a1bebe1ef9ebbf1eb6e1ff1c0513d673cdfb86779 |
| itt/support.json | 690f27ad04d24c23426f104a9d8d69a548e40a7befd70f7a3485e2327ed10c20 |
| itt/recovery_audit.json | bb90c6e5841f1ca04106ae12985f76510d2d6c3e8981c39e7edb999403e5562a |
| formal/assignments.jsonl | 8544fb65e628a6876d1b1bc1c2d580e28bfd1c4e7b15c1345200be4296d4339a |
| formal/executions.jsonl | 079d5b87e50e98edfa2d60652abb3d223c64b5bdcabf7db06100dcb554e52368 |


【服务器/数据事实】120 个唯一 assignment IDs、120 个唯一 game IDs；逐行 SHA 联结 formal raw assignment，`S_pre`、`p_tilde_j`、arm、acting wolf、candidate j、概率与原始 assignment 一致。120 行全部 `speech`；candidate selection 全部为冻结的 `first_pn_eligible_pre_uniform_candidate_v1`，每行两臂均合法、assignment probability 均为 0.5。当前样本是每个 game 第一个 eligible PRE 的一个均匀选中 candidate，不代表所有 speech、所有 candidate 或 speech_pk 的分布。

【服务器/数据事实】119 个 L_ref 来自既有 formal consequences；1 个来自已冻结的 canonical trajectory recovery。唯一失败仍是失败，仍在 PUSH 58 行中。本轮没有再次恢复或改写结果。`execution_success` 只用于验证 ITT population 未被筛选，未用于 predictor、支持域或模型选择。formal executions 的 `theta_audit_label` 全部 120 行为 null；ITT rows 未含 Theta。null 没有被当作 0。

【描述性统计】PUSH N=58；REDIRECT N=62；raw L_ref means 分别为 0.3118433375828878 / 0.23960506311607913；差为 0.07223827446680867。这些已冻结数字只做数值一致性核验，不作为选模型依据。

## 2. 目标与统计 convention

【统计语义推导】`G=(S_pre,p_tilde_j)`，`S_pre=(n_w,n_nw)` 是 assignment 前的存活狼/非狼计数；`p_tilde_j` 直接读取冻结 assignment 的 score。没有加入无变化的 phase、post-treatment execution_success、实际 vote 或后续 outcome 作 predictor。

令 `L(a)` 为设置 assignment 为 a 后、执行冻结 language/repair/verifier/failure handling 和 baseline continuation 所产生的 reference loss。目标为

\[
m_a(g)=E[L(a)\mid G=g]=E[L_{ref}\mid A=a,G=g].
\]

右侧识别要求设计 exchangeability、positivity、一致的 treatment 版本和无跨 game 干扰。它对应 assigned-action 策略（包含失败后 baseline），不能解释为“每次都成功实现动作”的效果。仅用两个 G 变量仍是在它们之外的 PRE 情景混合中取平均；随机化不保证它们足以把风险运输到其他机会选择规则、execution 版本或未来分布。[Hernán–Robins，What If，Ch.2–3](https://miguelhernan.org/s/hernanrobins_WhatIf_2jan25.pdf)

【统计语义推导】设计 probability=0.5 说明随机化设计中的 overlap；观察到的空格是当前有限样本支持不足，不证明总体结构性 positivity 为零。连续 score 在每个点上几乎不重复，不能以“精确相同 score 必须有双臂”为要求，也不能以 min/max 区间相交替代局部支持。[Petersen et al., 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC4107929/)

【描述性统计】下列所有 count 是精确整数，连续统计打印 Python float round-trip 数值。分位数用 linear/type-7：排序后 `t=(n-1)q`，在 floor(t)/ceil(t) 之间线性插值。SD 为 sample SD（ddof=1）；n=0 的统计均 NA；n=1 的 SD 为 NA。分位数插值值不是新的观测 loss。

【理论建议】固定诊断 bins：deciles `[0,.1), …, [.9,1]`；粗 bins `[0,.1), [.1,.3), [.3,.5), [.5,1]`。最后一格含 1，其余左闭右开。它们没有按 outcome 调整，也不是用于重新标定概率或拟合离散模型。共同包络 inclusive min/max；没有把包络外值 clip 入包络。粗格分级：双臂各≥10 / 双臂均非空但至少一臂<10 / 单臂 / 两臂空。≥10 只是透明的描述性筛查，不是识别、功效或精度保证。

## 3. Randomized arm × S_pre support

【描述性统计】

| S_pre | 总 N | PUSH | REDIRECT | 实际 PUSH fraction |
| --- | --- | --- | --- | --- |
| (1, 4) | 1 | 0 | 1 | 0.0 |
| (2, 3) | 5 | 3 | 2 | 0.6 |
| (2, 4) | 36 | 15 | 21 | 0.4166666666666667 |
| (2, 5) | 78 | 40 | 38 | 0.5128205128205128 |
| overall | 120 | 58 | 62 | 0.48333333333333334 |


【统计解释】`(2,4)` 与 `(2,5)` 共 114/120（95%），两臂合计 55/59；四个状态都观察到 REDIRECT，仅三个状态观察到 PUSH。`(1,4)` 的 PUSH 条件风险只能借助其他状态的模型假设；`(2,3)` 仅 5 行，不能把“有双臂”解读为稳定支持。

## 4. p_tilde quantiles、ranges 与 arm × score bins

【描述性统计】全体随机臂 score 分布：

| arm | N | mean | sample SD | distinct scores |
| --- | --- | --- | --- | --- |
| PUSH | 58 | 0.12695228674193595 | 0.14964743862353752 | 57 |
| REDIRECT | 62 | 0.14962919702828653 | 0.17863068212774164 | 62 |


| quantile | PUSH | REDIRECT |
| --- | --- | --- |
| 0 | 0.002997389598876192 | 0.00521034560968036 |
| 0.05 | 0.007182357177759325 | 0.0064079105688707675 |
| 0.1 | 0.013529818359788188 | 0.037124942264578624 |
| 0.25 | 0.05365343657442622 | 0.057093343938050205 |
| 0.5 | 0.07115297292143535 | 0.08213572772040978 |
| 0.75 | 0.11900529656246102 | 0.12861189546983978 |
| 0.9 | 0.3340542577953514 | 0.3635281485443082 |
| 0.95 | 0.4894790595002396 | 0.49648373416650177 |
| 1 | 0.6566524289627977 | 0.8522179847673137 |


【描述性统计】按 state × arm 的 score quantiles（min/max 在下一节完整列出）：

| S_pre | arm | N | q05 | q25 | q50 | q75 | q95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (1, 4) | PUSH | 0 | NA | NA | NA | NA | NA |
| (1, 4) | REDIRECT | 1 | 0.10805805250245967 | 0.10805805250245967 | 0.10805805250245967 | 0.10805805250245967 | 0.10805805250245967 |
| (2, 3) | PUSH | 3 | 0.1419031689333514 | 0.1992605576261855 | 0.27095729349222814 | 0.3298623705215191 | 0.37698643214495187 |
| (2, 3) | REDIRECT | 2 | 0.1657635561071365 | 0.1784874013306424 | 0.19439220786002476 | 0.21029701438940712 | 0.223020859612913 |
| (2, 4) | PUSH | 15 | 0.008084544873670873 | 0.09096629351148344 | 0.11716189738957261 | 0.21180743702845145 | 0.6201006749085892 |
| (2, 4) | REDIRECT | 21 | 0.009085246245134617 | 0.09692799210729416 | 0.11842062804531202 | 0.25855156895209397 | 0.6495943144086811 |
| (2, 5) | PUSH | 40 | 0.004074636290933328 | 0.052537937462643 | 0.061394804946994994 | 0.08002189931053048 | 0.3192336054569242 |
| (2, 5) | REDIRECT | 38 | 0.005563105741091356 | 0.05394088901701573 | 0.0635083773371921 | 0.0828571987808654 | 0.4406910573246278 |


| score bin | PUSH | REDIRECT | total |
| --- | --- | --- | --- |
| [0, 0.1) | 39 | 39 | 78 |
| [0.1, 0.2) | 10 | 10 | 20 |
| [0.2, 0.3) | 2 | 4 | 6 |
| [0.3, 0.4) | 2 | 3 | 5 |
| [0.4, 0.5) | 2 | 3 | 5 |
| [0.5, 0.6) | 1 | 0 | 1 |
| [0.6, 0.7) | 2 | 1 | 3 |
| [0.7, 0.8) | 0 | 1 | 1 |
| [0.8, 0.9) | 0 | 1 | 1 |
| [0.9, 1] | 0 | 0 | 0 |


【统计解释】78/120（65%）在 `[0,.1)`；104/120（86.6667%）小于 .3。`.3+` 仅 PUSH 7 / REDIRECT 9，`.5+` 仅 3/3，`.7+` 仅 0/2。`[.5,.6)` 没有 REDIRECT，`[.7,.8)` 与 `[.8,.9)` 没有 PUSH，`[.9,1]` 双臂全空。当前 117 个不同 score；p=0 与 p=1 的观测数均为 0。

【描述性统计】仅用 score 计算：`.3+` 的 7 个 PUSH 提供该臂中心化 score 平方和的 77.7998%；9 个 REDIRECT 提供 79.3719%。这是 score 变异的尾部集中度，不是拟合模型的 leverage、效果或显著性结果。

【统计解释】M2 的 score slope 很可能主要由少量尾部点约束；低 score 行很多并不能证明高 score 的 arm-specific slope 稳定。

## 5. S_pre × arm × p_tilde 的精确联合支持

【描述性统计】粗 bin counts：

| S_pre | score bin | PUSH | REDIRECT | 计数分级 |
| --- | --- | --- | --- | --- |
| (1, 4) | [0, 0.1) | 0 | 0 | 两臂空 |
| (1, 4) | [0.1, 0.3) | 0 | 1 | 单臂 |
| (1, 4) | [0.3, 0.5) | 0 | 0 | 两臂空 |
| (1, 4) | [0.5, 1] | 0 | 0 | 两臂空 |
| (2, 3) | [0, 0.1) | 0 | 0 | 两臂空 |
| (2, 3) | [0.1, 0.3) | 2 | 2 | 稀疏双臂 |
| (2, 3) | [0.3, 0.5) | 1 | 0 | 单臂 |
| (2, 3) | [0.5, 1] | 0 | 0 | 两臂空 |
| (2, 4) | [0, 0.1) | 5 | 7 | 稀疏双臂 |
| (2, 4) | [0.1, 0.3) | 7 | 9 | 稀疏双臂 |
| (2, 4) | [0.3, 0.5) | 1 | 3 | 稀疏双臂 |
| (2, 4) | [0.5, 1] | 2 | 2 | 稀疏双臂 |
| (2, 5) | [0, 0.1) | 34 | 32 | 双臂各≥10 |
| (2, 5) | [0.1, 0.3) | 3 | 2 | 稀疏双臂 |
| (2, 5) | [0.3, 0.5) | 2 | 3 | 稀疏双臂 |
| (2, 5) | [0.5, 1] | 1 | 1 | 稀疏双臂 |


【描述性统计】16 个 state/coarse-bin 格：1 格双臂各≥10，包含 66 行；8 格稀疏双臂，包含 52 行；2 格单臂，包含 2 行；5 格双臂空。唯一双臂各≥10 的格为 `(2,5) × [0,.1)`，PUSH 34 / REDIRECT 32。宽 bin 中非空并不能消除 bin 内的细分空隙。

【描述性统计】decile 细分，单元格为 PUSH/REDIRECT：

| score bin | (1, 4) | (2, 3) | (2, 4) | (2, 5) |
| --- | --- | --- | --- | --- |
| [0, 0.1) | 0/0 | 0/0 | 5/7 | 34/32 |
| [0.1, 0.2) | 0/1 | 1/1 | 6/8 | 3/0 |
| [0.2, 0.3) | 0/0 | 1/1 | 1/1 | 0/2 |
| [0.3, 0.4) | 0/0 | 1/0 | 0/2 | 1/1 |
| [0.4, 0.5) | 0/0 | 0/0 | 1/1 | 1/2 |
| [0.5, 0.6) | 0/0 | 0/0 | 0/0 | 1/0 |
| [0.6, 0.7) | 0/0 | 0/0 | 2/1 | 0/0 |
| [0.7, 0.8) | 0/0 | 0/0 | 0/0 | 0/1 |
| [0.8, 0.9) | 0/0 | 0/0 | 0/1 | 0/0 |
| [0.9, 1] | 0/0 | 0/0 | 0/0 | 0/0 |


【统计解释】例如 `(2,5)` 的 `[.1,.2)` 为 3/0、`[.2,.3)` 为 0/2；粗 bin `[.1,.3)` 的 3/2 不等于各个局部区间都有双臂。对 `(2,4)`，`[.3,.4)` 为 0/2；更高区间也零散。稀疏区间的预测即使落在总体 min/max 内，仍依赖平滑/形状和跨区间借力。

## 6. Overlap 与 extrapolation diagnostics

【描述性统计】完整 min/max 与共同包络：

| 范围 | PUSH min/max | REDIRECT min/max | 共同包络 | 包络内 P/R | 包络外 P/R |
| --- | --- | --- | --- | --- | --- |
| overall | [0.002997389598876192, 0.6566524289627977] | [0.00521034560968036, 0.8522179847673137] | [0.00521034560968036, 0.6566524289627977] | 55/60 | 3/2 |
| (1, 4) | 无 | [0.10805805250245967, 0.10805805250245967] | 无 | 0/0 | 0/1 |
| (2, 3) | [0.12756382176014286, 0.3887674475508101] | [0.16258259480126003, 0.2262018209187895] | [0.16258259480126003, 0.2262018209187895] | 0/2 | 3/0 |
| (2, 4) | [0.0077301424406220195, 0.6566524289627977] | [0.00626699816485688, 0.8522179847673137] | [0.0077301424406220195, 0.6566524289627977] | 15/19 | 0/2 |
| (2, 5) | [0.002997389598876192, 0.5252706354667532] | [0.00521034560968036, 0.7792881171669388] | [0.00521034560968036, 0.5252706354667532] | 37/37 | 3/1 |


【统计解释】pooled 包络内 115/120，按各自 state 包络则只有 110/120（PUSH55/REDIRECT55）；后者仍包含极稀疏 `(2,3)` 的 2 条 REDIRECT，不能标为可靠局部覆盖。`(2,3)` 的 PUSH scores 为 `0.12756382176014286, 0.27095729349222814, 0.3887674475508101`；REDIRECT 为 `0.16258259480126003, 0.2262018209187895`。共同包络内 PUSH=0 / REDIRECT=2，说明 hull 相交只是几何筛查。

【描述性统计】以下是原始 covariate 分布比较，没有 propensity model、显著性检验或 outcome：

| diagnostic | value |
| --- | --- |
| score_SMD_PUSH_minus_REDIRECT | -0.13762136547971623 |
| score_ECDF_max_gap | 0.1173526140155729 |
| decile_OVL | 0.9098998887652946 |
| state_OVL | 0.9037819799777531 |
| joint_state_coarse_OVL | 0.8876529477196884 |


【描述性统计】SMD 为 `(mean_P−mean_R)/sqrt((var_P+var_R)/2)`，两方 variance 均 sample variance。ECDF gap 为全部观察 score 阈值上的 `max|F_P−F_R|`。OVL 为固定格的 `sum min(n_P/58,n_R/62)`；state/coarse joint OVL 对全部 16 格计算。OVL 在 0–1，受 bin 定义影响，不能当作局部覆盖保证。

【统计解释】这些 marginal/joint 摘要总体接近，但 state 缺臂和尾部空格仍存在。p_tilde 是 outcome-state score，不是 assignment propensity；随机化 propensity 已知为 .5，不需要拟合它，也不能拿 p_tilde 代替它。

【描述性统计】同一 state 内到 opposite-arm 最近 score 的绝对距离，以及同臂相邻 score 最大空隙：

| S_pre | arm | NN median | NN q90 | NN max | largest within-arm gap |
| --- | --- | --- | --- | --- | --- |
| (1, 4) | PUSH | NA | NA | NA | NA |
| (1, 4) | REDIRECT | NA | NA | NA | NA |
| (2, 3) | PUSH | 0.04475547257343865 | 0.13900359582030425 | 0.16256562663202062 | 0.14339347173208528 |
| (2, 3) | REDIRECT | 0.03988712280727791 | 0.0437818026202065 | 0.04475547257343865 | 0.06361922611752946 |
| (2, 4) | PUSH | 0.0017911010704925456 | 0.01854340452651708 | 0.04515867695189546 | 0.16650750783002044 |
| (2, 4) | REDIRECT | 0.005547638396632068 | 0.07391211983995771 | 0.19556555580451596 | 0.2026236703586326 |
| (2, 5) | PUSH | 0.0011279031833482957 | 0.010786027967499856 | 0.0623022233659509 | 0.17255715113540643 |
| (2, 5) | REDIRECT | 0.001070734817999984 | 0.017065125629908594 | 0.25401748170018557 | 0.2802073445810288 |


【统计解释】nearest distance 只是局部邻近诊断；一个邻居不能形成充分的 panel。未用距离自动选择带宽、邻居数或训练域。特别 `(2,5)` REDIRECT 尾部距最近 PUSH 可达 .254，pooled range 很容易掩盖这种条件外推。

## 7. Outcome / L_ref support

【描述性统计】按随机臂 outcome counts：

| Y | PUSH | REDIRECT | total |
| --- | --- | --- | --- |
| target_j_exiled | 15 | 10 | 25 |
| other_nonwolf_exiled | 18 | 36 | 54 |
| acting_wolf_exiled | 20 | 11 | 31 |
| teammate_wolf_exiled | 4 | 2 | 6 |
| no_exile | 1 | 3 | 4 |


【描述性统计】state × arm 的 outcome counts，单元格为 PUSH/REDIRECT：

| S_pre | target_j_exiled | other_nonwolf_exiled | acting_wolf_exiled | teammate_wolf_exiled | no_exile |
| --- | --- | --- | --- | --- | --- |
| (1, 4) | 0/0 | 0/1 | 0/0 | 0/0 | 0/0 |
| (2, 3) | 0/0 | 2/1 | 1/1 | 0/0 | 0/0 |
| (2, 4) | 6/3 | 1/13 | 6/3 | 1/2 | 1/0 |
| (2, 5) | 9/7 | 15/21 | 13/7 | 3/0 | 0/3 |


【描述性统计】L_ref 由冻结 reference table 映射，精确离散支持如下；同一 state 的未列值在本批为 0 条：

| S_pre | L_ref（原始 float） | PUSH | REDIRECT |
| --- | --- | --- | --- |
| (1, 4) | 0.4222222222222223 | 0 | 1 |
| (2, 3) | 0.0 | 2 | 1 |
| (2, 3) | 0.4222222222222223 | 1 | 1 |
| (2, 4) | 0.03873239436619713 | 7 | 16 |
| (2, 4) | 0.235 | 1 | 0 |
| (2, 4) | 0.4256055363321799 | 7 | 5 |
| (2, 5) | 0.235 | 24 | 28 |
| (2, 5) | 0.3157894736842105 | 0 | 3 |
| (2, 5) | 0.5337078651685394 | 16 | 7 |


【描述性统计】L_ref summary，NA 不是 0：

| S_pre / population | arm | N | min | q25 | median | q75 | max | mean | sample SD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | PUSH | 58 | 0.0 | 0.235 | 0.235 | 0.5337078651685394 | 0.5337078651685394 | 0.3118433375828878 | 0.17544931639968178 |
| overall | REDIRECT | 62 | 0.0 | 0.03873239436619713 | 0.235 | 0.3157894736842105 | 0.5337078651685394 | 0.23960506311607913 | 0.16023384398793833 |
| (1, 4) | PUSH | 0 | NA | NA | NA | NA | NA | NA | NA |
| (1, 4) | REDIRECT | 1 | 0.4222222222222223 | 0.4222222222222223 | 0.4222222222222223 | 0.4222222222222223 | 0.4222222222222223 | 0.4222222222222223 | NA |
| (2, 3) | PUSH | 3 | 0.0 | 0.0 | 0.0 | 0.21111111111111114 | 0.4222222222222223 | 0.14074074074074075 | 0.24377011365784204 |
| (2, 3) | REDIRECT | 2 | 0.0 | 0.10555555555555557 | 0.21111111111111114 | 0.3166666666666667 | 0.4222222222222223 | 0.21111111111111114 | 0.29855619650098675 |
| (2, 4) | PUSH | 15 | 0.03873239436619713 | 0.03873239436619713 | 0.235 | 0.4256055363321799 | 0.4256055363321799 | 0.23235770099257594 | 0.19343795209524975 |
| (2, 4) | REDIRECT | 21 | 0.03873239436619713 | 0.03873239436619713 | 0.03873239436619713 | 0.03873239436619713 | 0.4256055363321799 | 0.13084504721524065 | 0.16884528165757265 |
| (2, 5) | PUSH | 40 | 0.235 | 0.235 | 0.235 | 0.5337078651685394 | 0.5337078651685394 | 0.35448314606741577 | 0.1482006031035554 |
| (2, 5) | REDIRECT | 38 | 0.235 | 0.235 | 0.235 | 0.2955921052631579 | 0.5337078651685394 | 0.29640324940085283 | 0.1163512732092119 |


【统计解释】overall L_ref 只有 7 个不同值，实际范围 `[0,0.5337078651685394]`。不同 Y 可能映射同一 reference loss；例如在 `(2,5)`，两种非狼 exile 都映射 .235。对 bounded/discrete surrogate loss 的 mean 建模不要求把 L 当 binary outcome，不能由小参数数目推断 normal/homoskedastic errors；loss 分布随 state 改变，也使 state-invariant classical lambda 更需要理论论证。

【理论建议】本轮 outcome 表只检查响应变量是否完整、离散支持与类别稀疏情况。没有根据这些均值、类别表现或差值挑模型、knot、域或阈值。`L_ref` 是冻结 publication-policy reference surrogate；不是本 intervention 下实际 final win 的直接 risk。

## 8. Recommended admissible modeling domain 与 fail-closed 建议

【理论建议】需要区分三个用途，而非把一张 support mask 静默作用于原始 primary population：

1. **完整 marginal ITT**：始终 120 行、58/62，包含唯一失败。M0 可作为该 marginal estimand 的 reference。它不证明 `m_a(G)` 对 G 恒定。
2. **conditional working-model 的后续设计域**：优先讨论 `(2,4)` / `(2,5)` 两个 dominant states（114行，55/59）的极低容量模型；state-only M1 可以跨该 state 的 score 分布做平均，但不能据此宣称 pointwise score 风险已被验证。`(2,3)` 另列 sparse；`(1,4)` 另列缺臂。若未来限制训练或报告 population，必须明确这是新的 conditional/secondary estimand，保留原始 120-row primary ITT。
3. **未来可靠风险比较的候选局部域**：最有数据支持的域为

\[
D_{dense}=\{S=(2,5),\quad 0.00521034560968036\le p_{tilde}<0.1\}.
\]

该域同时要求落在 state 内双臂 hull，含 PUSH31 / REDIRECT32（63行）；原 coarse 格34/32中的3个 PUSH 太靠近0，低于 REDIRECT min，未因低score而偷填支持。这个域仍需要平滑、精度评估和冻结邻域标准，不能仅凭≥10放行部署。

【理论建议】更宽但明确稀疏、仅适合 exploratory 模型审查的低 score 包络为

\[
D_{explore}=\{S=(2,4),\ 0.0077301424406220195\le p_{tilde}<0.3\}
\cup\{S=(2,5),\ 0.00521034560968036\le p_{tilde}<0.3\}.
\]

共95行，PUSH46/REDIRECT49；两个 state 分别12/15与34/34。它包含 decile 单臂空隙，不能作为稳定 pointwise interpolation 认证。两域都由 pre-treatment count/range 决定，没有删除 ITT artifact 中任何行，也未成为现有 runtime 行为。

【理论建议】未来若需要给出两臂风险并作比较，应预先冻结 unsupported/fail-closed 规则，至少涵盖：

| 区域/输入 | 支持状态 | 建议 |
| --- | --- | --- |
| speech_pk、不同机会/candidate选择规则、不同execution版本 | 未采样/不可自动运输 | 本批模型对它们 unsupported；需要独立论证 |
| 未观察的 S_pre | state 外推 | unsupported |
| S_pre=(1,4) | PUSH=0；REDIRECT仅1 | 双臂条件风险比较 fail closed；不把单臂点推广为可用state |
| S_pre=(2,3) | 3/2；共同score包络内0/2 | 本批稳定局部条件比较 unsupported；可保留audit/sparse描述 |
| state 内共同 min/max 外 | 至少一臂需要 range 外推 | 双臂比较 fail closed |
| 共同包络内、细 bin 单臂/两臂空 | 需要跨空隙平滑或借力 | 标为 sparse/model-dependent；未经冻结支持规则不宣称稳定 |
| 高 score（尤其 .3+/.5+） | 尾部极少；各state局部缺臂 | 本批不能认证稳定 arm-specific slope；保守列为unsupported comparison |
| p=0 或 p=1 | 两端点均无观测且在所有有效包络外 | endpoint extrapolation；不能解释为经验lambda |
| p 非有限、超出[0,1]、S缺失、lineage不匹配 | 数据契约违规 | fail closed；不要clip/填0或改reference |


【统计解释】共同 hull 内是几何上的 interpolation 范围；在稀疏双臂/空隙中仍只有 model-dependent interpolation。共同 hull 外、缺臂 state 或未见 state 是条件比较的 extrapolation。这里的 fail closed 指未来不给出“数据已支持的两臂条件风险比较”；本轮没有指定替代动作、构造 router 或改变游戏。

【理论建议】≥10 的诊断 screen、dense/explore 域均须由研究者在下一轮确定用途；不允许未来依据 L_ref 拟合好坏移动边界。任何 restricted-domain conclusion 不能替换全120行 ITT headline。

## 9. M0 / M1 / M2：只审查、不拟合

【统计语义推导】采用 identity-link conditional mean working model，`I_A=1{A=PUSH}`；S 先按 categorical state 编码。参数规模只数 mean coefficients，不含 residual variance 或 uncertainty tuning。

\[
M0:\quad \alpha+\tau I_A;
\]
\[
M1:\quad \alpha+\tau I_A+\sum_{s\ne s_0}\gamma_s1\{S=s\};
\]
\[
M2:\quad \alpha+\tau I_A+\sum_{s\ne s_0}\gamma_s1\{S=s\}
+\beta p_{tilde}+\eta I_Ap_{tilde}.
\]

M1 的 arm difference 对所有 S 固定；M2 的 arm difference 为 `tau+eta*p_tilde`，对所有 S 共享，两个 arm 的 score slopes 分别为 beta / beta+eta。它们没有 action×S interaction。若真实风险不符合形状，拟合系数仅是依所冻结目标/权重定义的 working projection，不能自动视为真实 `m_a(G)`。

【描述性统计】只用 design covariates、未读取 L_ref 去估计任何 coefficient，计算 full column rank（Gaussian elimination tolerance 1e-10）：

| 设计域 | N | categorical K | M0 columns/rank | M1 columns/rank | M2 columns/rank |
| --- | --- | --- | --- | --- | --- |
| all4 | 120 | 4 | 2/2 | 5/5 | 7/7 |
| dual3 | 119 | 3 | 2/2 | 4/4 | 6/6 |
| dominant2 | 114 | 2 | 2/2 | 3/3 | 5/5 |


【统计解释】full-rank 只表明设计矩阵代数上不奇异；全域 M1/M2 的 `(1,4)` state effect 由唯一 REDIRECT 提供，预测该 state 的 PUSH 仍靠共同 arm effect/slope 外推。总 n/参数的比值不能证明各区域支持充分。

【统计语义推导】若 S 用两个 numeric counts，名义 M0/M1/M2 为2/4/6；全域列可独立，但狼数=1的变化仅来自唯一 `(1,4)`，不能靠“少1参数”获得稳健 wolf-count effect。去掉 `(1,4)` 后 n_w 恒为2，与截距共线；numeric M1/M2 有效 rank为3/5，并额外假设 nonwolf-count 主效应线性。因此建议把 categorical 编码先列为主设计审查 convention，不把 numeric 降维当作无假设的简化。

【理论建议】

- **M0**：定义和两臂 marginal support 明确，可作为 primary marginal reference；本轮不作效果显著性或精度结论。
- **M1**：可以进入低容量、dominant-state conditional mean 的预注册讨论。其共同 treatment difference 是强工作假设，稀疏/缺臂 state 不能藉此变成经验支持。
- **M2**：代数规模很小、设计可满秩，但 score slope 尾部约束来自7/9条，joint coverage存在空隙。可以作为明确受限域的候选设计，不足以批准 global score-dependent decision risk、0/1端点或经典lambda。
- **一维 hinge**：本轮没有支持需要增加knot的证据。若未来仍讨论，单个固定knot的 `(p-c)+` 加1个mean参数，若再加 arm×hinge则加2；需要冻结knot及其两侧各arm支持。当前尾部不足，不建议实现或自动寻knot。

只做预注册设计审查，没有拟合系数、训练/交叉验证、bootstrap模型、拟合性能、最佳模型或基于outcome的选择。下一轮还需冻结估计目标、loss/link、适用域、uncertainty/精度要求；bounded L_ref 不应通过未预注册的事后clip掩盖外推。

## 10. Classical lambda 的识别结论

【统计语义推导】对真实 PRE label Theta，令

\[
\pi(g)=P(\Theta=1\mid G=g),\quad
\mu_{a\theta}(g)=E[L(a)\mid\Theta=\theta,G=g].
\]

迭代期望给出

\[
m_a(g)=\mu_{a1}(g)\pi(g)+\mu_{a0}(g)(1-\pi(g)).
\]

要解释为 `lambda_a+*p + lambda_a-*(1-p)`，至少需：

1. p 等于相关信息集下的真实 posterior，当前不能把 p_tilde 自动当 externally calibrated p_hat。若只保证 `P(Theta=1|p_tilde)=p_tilde` 的边际校准，而G仍含S，还不足以推出 `P(Theta=1|S,p_tilde)=p_tilde`。校准的定义约束条件发生率，不是输出落在0–1或模型已冻结即可成立。[Guo et al., 2017, Sec.2](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf)
2. 真实类内风险 `mu_aθ(g)` 在适用域内恒为对应 lambda，或明确改为state-dependent lambda；当前不同 S 的loss映射和剩余情景差异不支持默认恒定。
3. 相同冻结 treatment/execution版本、label定义与测量时点；需要运输稳定性才能用于另一分布。
4. 有能约束两个真实状态分量的数据/外部信息，并避免把窄score范围的直线外推解释成类内真实风险。

【服务器/数据事实】两臂均未观察p=0/1；高score不足；全部120个theta_audit_label为null，不能进行 `A×Theta` 直接分组。没有静默补算Theta，没有用Q或后续vote替代真标签，也没有把null当false。

【统计语义推导】未来若另一个获授权的独立canonical PRE audit获得完整、assignment前、outcome-free真实Theta，且 `A×Theta` 四格具有支持，随机化可识别当前population的 `E[L(a)|Theta=theta]`。这些值对 `G|Theta` 作混合，不自动等于每个g的类内风险，也不自动构成上述posterior mixture。缺少0/1 score并不在逻辑上阻止这种真标签分组对象；当前直接阻断是标签全missing及类内支持尚未知。

【统计语义推导】若日后仅拟合 `alpha_a+beta_a*p_tilde`，把 `lambda_a-=alpha_a`、`lambda_a+=alpha_a+beta_a` 只是直线的0/1端点重参数化；未经上述假设，它们是外推工作值，可能超出合法loss范围，不能称为已识别的经典lambda。

【理论建议】本批可以把 p_tilde 作为固定pre-treatment score协变量讨论 `m_a(S,p_tilde)`；不能冻结classical lambda的状态损失解释。本轮不估计lambda。

## 11. 下一步仅理论冻结

【理论建议】下一轮先冻结一个受限、低容量terminal mean estimation protocol，而非policy：

- 明确 target 是 assigned-action ITT 的 reference loss，execution failure保留；full120 marginal报告与restricted conditional域并列。
- 明确S categorical编码、score只作covariate、phase不入主输入；不加入execution_success。
- 冻结support mask的用途、局部计数/邻域标准和稀疏区间的unsupported处理，不能只用full-rank或pooled range。
- 在拟合前确定M0/M1/M2的角色与复杂度、适用域和uncertainty/精度判据；不根据本轮outcome表选“最佳模型”。
- 如经典lambda是必须保留的理论对象，单独决定是否授权PRE truth-label audit及校准/类内risk稳定性审查；不能以当前null字段启动直接分组。

【理论建议】结论是：**可推进低容量estimator的预注册设计，不能据此批准全域terminal risk deployment或classical lambda。** 对M1可讨论dominant-state风险平均；对M2必须明确稀疏score区间的模型依赖与外推边界。当前数据没有给出global conditional risk精度保证。

## 12. 验证与未改动范围

【服务器/数据事实】根计算与独立复算的120-row精确joins、state counts、decile/coarse counts、state min/max/common hull/区间外counts完全一致；type-7 quantiles仅有≤1e-15的float运算顺序差异。arm means复现冻结数值；各分层counts及outcome/loss支持表回加到120。

【服务器/数据事实】仅新增本研究Markdown报告；计算脚本、输入副本和机器可读统计均在本地/tmp。没有修改Phase-2 runtime/core、formal/ITT artifact、reference tables、mapper或Q；没有新建服务器artifact，没有调用LLM、gameplay、randomization函数或发布函数；没有commit/push。既有未跟踪研究文档SHA256仍为`88a30495f24f491872abf12fad63cda01d2e2ee3eeb65d03e33aeb6d544da38d`。

【服务器/数据事实】输入读取时的完整服务器核验成功。报告完成后尝试再次完整核验，两次SSH连接均在执行前被关闭（`Connection closed ... port 3001`），因此没有声称最终服务器复核通过。上述统计基于首次完整核验、逐文件SHA校验并由独立计算复核的冻结输入快照；本轮远端命令均只读。

【描述性统计】临时计算/报告脚本compile通过；`git diff --check`通过，新报告的no-index whitespace check无诊断。独立审阅确认全部数值一致、未发现实质理论错误；修复生成Markdown时的LaTeX字符串转义后，控制字符检查与关键公式literal检查通过。没有修改production code，因此未重新运行runtime测试。
