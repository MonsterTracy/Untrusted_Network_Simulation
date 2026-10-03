# Terminal Estimator Protocol V1

状态：FROZEN SPECIFICATION；本轮只冻结文档，未拟合或计算新的推断结果。本协议在 Formal Pilot-T、120-row ITT analysis 与 Support Review V1 完成后、terminal estimator 拟合前冻结；不宣称它是 gameplay 采集前的预注册。

本文的“必须”“不得”和数值 convention 是未来拟合的约束。通过模型 sanity checks 只获得进入下一阶段审阅的资格，不等于授权部署、构造 router、启动 Probe 或选择动作。

## 1. 来源、population 与冻结范围

【源码/数据事实】本轮开始时：

    git status --short
    ?? docs/research/phase2-wolf-pt3wd-review-2026-09-13.md
    git rev-parse HEAD
    5a6f2a657d63bb09791e052cfc51de47a2557556
    git branch --show-current
    twd/mainline

【数据事实】本协议依据已冻结的 [Terminal Estimator Support Review V1](phase2-terminal-estimator-support-review-v1.md)；该文档 SHA256 为 e82d789352cb14b0f368597bf9a844da8d8b3d373325ff494baa817d3088ea8d。本轮未重新读取服务器 artifact、拟合模型或读取新的模型结果。

| 冻结输入 | 标识 |
| --- | --- |
| ITT artifact | /data/yuxiao/Untrusted_Network_Simulation/paper-studies/paper-phase2-terminal-itt-analysis-v1 |
| ITT manifest digest | 58a8611a0c9f5605f6cbdf5a7a9ea9f0c74fc69e637760e507f073e298cfaf8c |
| ITT rows SHA256 | 7ebb3b4848712ea40142ef6a1bebe1ef9ebbf1eb6e1ff1c0513d673cdfb86779 |
| Formal artifact | /data/yuxiao/Untrusted_Network_Simulation/paper-studies/paper-phase2-online-terminal-pilot-v1 |
| Formal manifest digest | ec1cc8730aefe8fce57ed83a3c84ca9231ae645ea6c9047cd588f8022d2223c2 |
| Formal execution source commit | 05c217c98dab74e919e525f76c55ca3a1e12867e |
| Frozen reference table digest | 1bc2fa51ea771e90389cb511bc0c61a442d862c872a121d61037aca6308f7c2c |
| Primary population | 全部 120 randomized assignments，PUSH=58，REDIRECT=62 |
| Observation unit | 一个唯一 assignment / game；每 game 最多一个 assignment |
| Outcome | 冻结 ITT row 中的 L_ref |
| Outcome completeness | 119 formal consequences + 1 已冻结 canonical recovery；120 行均有 L_ref |
| Phase | 全部 speech；不是变化的 predictor |
| Theta audit | theta_audit_label 全部 120 行为 null |

【协议冻结】未来执行前必须核验 manifest、完整 file table、120 个唯一 assignment IDs 与 game IDs，以及原始 assignment 的 arm、S_pre、p_tilde、概率联结。若缺失、重复、digest/lineage 不符或 outcome 非有限，停止整个分析，不删除 row、填值、重新恢复结果或另选 population。

不得改变 formal/ITT artifact、assignment labels、frozen seed、reference loss mapping、Q、mapper、Action/Language Contract、gameplay 或 runtime。稀疏 states 和唯一 language failure 都继续属于完整 randomized dataset。

## 2. Primary causal estimand 与唯一 point estimator

【协议冻结】Primary population 永远是全部 120 randomized assignments。定义

\[
\tau_{\mathrm{ITT}}
=E[L_{\mathrm{ref}}\mid assignment=PUSH]
-E[L_{\mathrm{ref}}\mid assignment=REDIRECT].
\]

不按 execution_success、S_pre、p_tilde、后续 vote、exile 或 model fit status 过滤 primary ITT。支持 gate 只限制 conditional operational prediction，不能过滤 primary estimator 或其 uncertainty 的输入。

【协议冻结】唯一 primary point estimator 是未经模型调整的 difference in means：

\[
\overline L_a=\frac{1}{n_a}\sum_{u:A_u=a}L_u,\qquad
\widehat\tau_{\mathrm{ITT}}=\overline L_P-\overline L_N,
\quad n_P=58,\ n_N=62.
\]

P 表示 PUSH，N 表示 REDIRECT。报告两臂 N、arm means、raw difference in means 和下节的 95% uncertainty；不以 M0/M1/M2 coefficient、标准化预测均值、模型选择或调整后的差值替代 primary point estimate。

【数据事实】已有冻结的数值一致性目标：

| 数量 | 冻结值 |
| --- | --- |
| PUSH mean L_ref | 0.3118433375828878 |
| REDIRECT mean L_ref | 0.23960506311607913 |
| PUSH minus REDIRECT | 0.07223827446680867 |

未来重算允许绝对差不超过 1e-12 的浮点舍入差异；不允许调整数据以匹配这些数字。本轮不计算新的 SE、CI 或 p-value。

【统计语义】这是 assigned-action 策略的 ITT：冻结 language realization、repair、verification、失败处理与 baseline continuation 都是 treatment version 的组成部分。保留失败 outcome 不等于只估计“成功语言动作”的效果。L_ref 是冻结 reference surrogate，不是实际 final-game win 的直接 risk。

## 3. Primary uncertainty：Neyman 95% 正态近似

【协议冻结】两臂 sample variance 使用 n_a−1：

\[
s_a^2=\frac{1}{n_a-1}\sum_{u:A_u=a}(L_u-\overline L_a)^2.
\]

Primary randomization/Neyman variance 与区间固定为

\[
\widehat V_N=\frac{s_P^2}{58}+\frac{s_N^2}{62},\qquad
\widehat{SE}_N=\sqrt{\widehat V_N},
\]
\[
CI^{Neyman}_{95\%}
=\widehat\tau_{\mathrm{ITT}}
\ \pm\ 1.959963984540054\,\widehat{SE}_N.
\]

明确标为“Neyman randomization-based normal-approximation 95% interval”。不使用 pooled equal-variance t formula，不因模型结果切换 variance estimator，不以 conditional OLS covariance 替代它，也不把 CI 截断到某个看似合理范围。

【统计语义】在固定潜在结果、给定 arm counts 的完全随机分配参考下，真实方差为

\[
V(\widehat\tau)=
\frac{S_P^2}{58}+\frac{S_N^2}{62}
-\frac{S_\tau^2}{120}.
\]

不可观察的 treatment-effect heterogeneity 项不被估计或扣除。Neyman variance 的期望保守不意味着此正态区间具有有限样本精确 95% 覆盖率。Fisher sharp null 与平均效应为零的 weak null 也不相同。[Ding, A paradox from randomization-based causal inference](https://arxiv.org/abs/1402.0142)

【协议冻结】随机化 reference 假设必须在未来报告中明示：冻结的每-unit 0.5 assignment 采用独立、均匀的 Bernoulli-style pseudo-randomization 解释；单位在 assignment 前确定，跨 game 无 carryover/干扰，停止规则只依 assignment 总数，不依 arm/outcome，也没有根据 outcome 选择 seed。源代码的冻结 hash 是可复现实现，不能单凭记录概率为 .5 就声称已经证明数学上的独立随机性。

条件于观察到的 n_P=58，等概率独立 Bernoulli reference 给出均匀的 58-of-120 分配参考集。这里条件化的是 randomization reference，不是删除 row，也不声称原 gameplay design 预先强制58/62。若未来 design audit 与这些前提冲突，停止 uncertainty 执行并重新审阅理论，不能静默套用均匀 permutation。[Branson–Bind, Randomization-based Inference for Bernoulli-Trial Experiments, Sec.3](https://arxiv.org/pdf/1707.04136)

## 4. Fisher sharp-null supplement

【协议冻结】Fisher test 可作为单独标记的补充；执行与否不能由 primary CI、模型表现或试算 p-value 决定。未来拟合运行若实现本 supplement，必须按下列固定规则报告；若不实现，明确标为“未计算”，不得把它作为通过 M2 的条件。

Sharp null：

\[
H_0^{sharp}:L_u(PUSH)=L_u(REDIRECT)\quad
\text{for every one of the 120 units}.
\]

这是冻结 assigned-action 策略的 unit-level sharp null，强于平均 ITT 为0。Supplement 不替代 primary point estimate/Neyman CI，不被称为 weak-null 的精确检验。

| Supplement 规则 | 冻结 convention |
| --- | --- |
| Population | 完整120行 |
| Statistic | 两侧 absolute raw difference in means |
| Reference set | 所有恰有58 PUSH / 62 REDIRECT的 allocations，在上节设计前提下均匀 |
| Permutation constraints | 不按 S_pre 分层、不平衡 score、不按 success 或 outcome筛选 |
| Monte Carlo draws | B=100000；每次调用同一Generator的permutation(120)，前58个索引为PUSH，其余为REDIRECT；各次allocation独立抽取，允许不同次重复 |
| Row order | assignment_id 的 ASCII lexicographic 顺序 |
| PRNG | 显式 NumPy Generator(PCG64(seed))；未来记录并锁定NumPy/运行环境版本 |
| Analysis seed | 5027505451237408553，仅用于此 supplement |
| Seed derivation | low63(first8_big_endian(SHA256(UTF8("phase2-terminal-estimator-protocol-v1:fisher-sharp-null")))) |
| Tie rule | absolute statistic ≥ observed absolute statistic；浮点比较容差1e-12，近于阈值者计入tail |
| p-value | (b+1)/(B+1)，b为tail count；报告B、b与p |

【协议冻结】完整枚举不作为本V1实现流程；使用上述预定 Monte Carlo procedure，不根据精度或显著性自动增加 B，不反复换 seed。此 analysis seed 不替代、不修改历史 assignment_seed，也不得调用 assignment/gameplay 函数生成新 treatment 或新 outcome。

Plus-one 避免零 p-value，包含ties；该数值是 Monte Carlo supplement，不宣称是完全枚举的精确值。设计有效性仍依赖上节reference假设。[Phipson–Smyth, Permutation P-values Should Never Be Zero](https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf)

## 5. Conditional terminal risk 的对象与输入

【协议冻结】Conditional target：

\[
m_a(S_{\mathrm{pre}},\widetilde p)
=E[L_{\mathrm{ref}}\mid do(A=a),S_{\mathrm{pre}},\widetilde p].
\]

G 只含 assignment 前的 S_pre=(n_w,n_nw) 与冻结 assignment 中的 p_tilde_j。p_tilde 始终称 score；不得称 calibrated posterior 或当前 Pilot-T 的 externally calibrated p_hat，也不是 assignment propensity。

phase 全部 speech，不进入主 design matrix。不得加入 execution_success、retry/commit结果、实际vote、exile、S_plus、Y、Theta、future information 或 action costs 作主 predictor。

【统计语义】随机化、一致性、正概率assignment与无跨game干扰支持条件因果均值的识别；它们不保证某条低容量直线是真实 risk function。M1/M2 是 reduced-form mean working approximations；misspecification 时系数是完整拟合population与固定 squared-error objective 下的projection，不能自动解释为真实条件函数。

## 6. M0 / M1 / M2：唯一候选集合与拟合 convention

【协议冻结】未来拟合三个模型都使用完整120 rows、每row等权、unregularized ordinary least squares、identity-link mean。保留四个 categorical S_pre 水平；不得合并、删除稀疏 state，不将S改成两个numeric counts以减少参数。无sample weights、class balancing、feature selection、regularization或训练/验证拆分来选“最佳模型”。

编码固定：

- I_A=1 表示 PUSH，0 表示 REDIRECT。
- S的reference level固定为(2,5)；三个dummy按(1,4)、(2,3)、(2,4)顺序。
- p_tilde使用原始score；不二值化、裁剪、再校准或变换。
- M0列顺序为截距、I_A。
- M1追加三个state dummy。
- M2再追加p_tilde、I_A × p_tilde。

\[
M0:\ \alpha+\tau I_A,
\]
\[
M1:\ \alpha+\tau I_A
+\gamma_{14}1\{S=(1,4)\}
+\gamma_{23}1\{S=(2,3)\}
+\gamma_{24}1\{S=(2,4)\},
\]
\[
M2:\ M1+\beta\widetilde p+\eta I_A\widetilde p.
\]

| 模型 | Mean parameters | 固定角色 |
| --- | --- | --- |
| M0: L ~ A | 2 | marginal arm-risk baseline；核对其arm预测与raw means一致；不取代primary的独立raw计算 |
| M1: L ~ A + S_pre | 5 | state-adjusted additive reduced-form诊断；强制共同arm difference |
| M2: L ~ A + S_pre + p_tilde + A:p_tilde | 7 | 本阶段最大复杂度；仅容许arm-specific linear score slope，state共享该形状 |

三个模型都输出诊断，不按test metric、效果大小、p-value、AIC/BIC、CV表现或哪个动作loss更低来选“最好模型”。M2通过sanity不是模型比较胜出，也不是授权使用模型选择动作。

【协议冻结】不得增加quadratic、高阶polynomial、hinge、spline、tree/forest、boosting、neural network、A:S、多重state interaction或其他高阶interaction。任一失败都不触发自动升级复杂度、变更link或事后clip。

## 7. Conditional domain 与 support gate

【协议冻结】state support如下。训练行保留与operational资格是两个不同概念：

| S_pre | PUSH / REDIRECT | State status | Conditional operational prediction |
| --- | --- | --- | --- |
| (2,4) | 15 / 21 | SUPPORTED | 还必须通过score interval与model sanity |
| (2,5) | 40 / 38 | SUPPORTED | 还必须通过score interval与model sanity |
| (2,3) | 3 / 2 | AUDIT_ONLY / UNSUPPORTED_FOR_OPERATIONAL_PREDICTION | 不允许 |
| (1,4) | 0 / 1 | UNSUPPORTED | 不允许 |
| 其他未见state | 0 / 0（本批未见） | UNSUPPORTED | 不允许 |

(2,3)的5行无法支持稳定条件比较；(1,4)缺PUSH且REDIRECT仅1行。Regression可能对它们输出诊断数字，但数字不得作为可用operational risk。

【协议冻结】对每个supported state s，冻结

\[
\mathcal I_s=
[\max(\min_{P,s}\widetilde p,\min_{N,s}\widetilde p),
\ \min(\max_{P,s}\widetilde p,\max_{N,s}\widetilde p)].
\]

当前完整120-row assignment-pre score support给出：

| Supported S_pre | Inclusive score admissible interval |
| --- | --- |
| (2,4) | [0.0077301424406220195, 0.6566524289627977] |
| (2,5) | [0.00521034560968036, 0.5252706354667532] |

任一interval外：UNSUPPORTED_EXTRAPOLATION。即使某臂有观测或regression能输出数字，也不放行两臂conditional operational prediction。边界左闭右闭，按上述冻结float64值比较；不加epsilon扩张区间、夹入边界或根据新的结果重估边界。

【协议冻结】未来gate依次检查：

1. 输入/data/model/protocol lineage与冻结规则一致，仍属于此formal speech、opportunity selection和treatment execution版本的适用范围；G非缺失，p_tilde有限且在[0,1]。
2. S_pre恰为(2,4)或(2,5)；(2,3)返回audit-only status，(1,4)/未知state返回unsupported。
3. score在该state的冻结inclusive interval内，否则返回UNSUPPORTED_EXTRAPOLATION。
4. 拟合和model sanity通过；M2未通过时不得以M0/M1自动替补推进。

任何不通过都不给出可用operational risk；可以保留明确标注的raw audit/diagnostic prediction，不能被自动降级为可用输出。M0/M1也不得因较简单而绕过同一conditional gate。完整primary ITT及三模型拟合输入都不受gate过滤。

【统计语义】此V1 gate冻结的是state与双臂range交集这一必要支持条件；区间内仍有稀疏尾部和细bins缺臂，不等于精度/校准认证。Support Review中的bins、≥10 screen、dense/explore建议只作audit，本V1不增加这些operational阈值，也不将高score在区间内一律拒绝。原Support Review不改写。

## 8. Conditional uncertainty 与 M2 sanity eligibility

【协议冻结】未来固定使用HC1 sandwich covariance作三模型的诊断uncertainty convention，k包括截距：

\[
\widehat C_{HC1}
=\frac{120}{120-k}(X^\top X)^{-1}
X^\top\operatorname{diag}(\widehat e_u^2)X
(X^\top X)^{-1},\quad k\in\{2,5,7\}.
\]

对通过gate的g，mean prediction SE为

\[
SE_a(g)=\sqrt{x_a(g)^\top\widehat C_{HC1}x_a(g)}.
\]

未来可报告固定95% normal pointwise diagnostic interval：
prediction ± 1.959963984540054 × SE。
这是conditional model-mean的渐近诊断区间，不是个体outcome prediction interval、simultaneous band，也不替代primary Neyman CI。区间超出[0,1]按raw数值报告，不clip；下表risk-range sanity针对mean prediction，而非强制区间也在[0,1]。

【统计语义】singleton (1,4) 的hat diagonal数学上为1、residual为0，HC2/HC3中的(1−h)分母未定义。不得删row、clamp分母或静默更换covariance。HC1的可计算公式固定见 [statsmodels HC1官方文档](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.HC1_se.html)；leverage分母定义见 [HC3官方文档](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.HC3_se.html)。HC1不能恢复singleton未知方差；finite或zero SE不代表该state风险确定。其operational prediction与operational uncertainty均禁止。

【协议冻结】M2只有下列项目全部PASS，才有资格进入下一阶段；报告M0/M1同类诊断但不得择优替补：

| Sanity item | 预定判据 |
| --- | --- |
| Design full rank | float64，SVD rank=7；rank tolerance=max(n,k) × machine epsilon × largest singular value；报告singular values/condition number |
| Finite coefficients | 7个系数全部finite；不得ridge、删列、合并state来修复 |
| Finite covariance/uncertainty | HC1所有entry finite；数值上symmetric和positive semidefinite；支持域各arm的prediction variance/SE finite |
| Finite mean predictions | 两supported states、两arms、各interval所有score上的mean finite |
| 非明显非法risk | supported intervals内两arm的raw mean必须在[0,1]内，最多容许1e-10浮点边界误差；超过即FAIL，不clip救回 |
| Reproducible arm predictions | 相同input order、design、coefficients和锁定环境，重新加载/复现得到的两arm预测绝对差≤1e-10；无LLM参与 |
| Gate完全生效 | 两state边界inclusive；下一float的区间外值拒绝；(2,3)、(1,4)、未知state、缺失/非有限score和lineage mismatch都拒绝；拟合失败时拒绝 |

M2对每个固定state/arm是score的affine函数，因此区间mean的finite/range检查必须覆盖两端点；由端点可解析保证整段mean，不用随机grid代替。Prediction variance是score的二次函数，其全段finite/nonnegative诊断检查端点和区间内stationary point（若存在）；不只检查训练row。

【协议冻结】covariance的symmetry tolerance为1e-10 × max(1,max_abs_entry)，PSD tolerance为1e-10 × max(1,max_abs_eigenvalue)。仅为eigenvalue数值检查，使用(C+Cᵀ)/2，不以此覆盖保存的raw HC1 covariance。允许数值零eigenvalue，不要求positive definite；singleton可导致covariance奇异，不能因此删row。

Projected variance的唯一数值容差为

\[
\epsilon_v(g)=10^{-10}\max\{1,\|x_a(g)\|_2^2
\max_j|\lambda_j((\widehat C_{HC1}+\widehat C_{HC1}^\top)/2)|\}.
\]

这里lambda_j只表示矩阵eigenvalue，不是action-loss lambda。若v(g)<−epsilon_v(g)则FAIL；若−epsilon_v(g)≤v(g)<0，只能明确标为numerical roundoff后按0取sqrt，raw variance照录，不得修正covariance或掩盖实质负variance。risk-range容差仅用于浮点判定，raw prediction照录，不能据此扩张support interval。

必须输出各model的k、rank、condition number、leverage摘要、所有sanity状态和失败原因。有限covariance、full rank、120>7与gate PASS均不构成有限样本coverage、低方差或risk准确性的证明。

【协议冻结】数据契约失败或任一M2 sanity失败：FAIL_CLOSED，保留primary与诊断记录，停止下一阶段推进并重新审阅理论。不自动升级模型，不调整support边界，不重新采集、不删除稀疏rows、不通过更改verifier/runtime补救。本轮未执行这些checks或宣称M2已PASS。

## 9. Lambda interpretation 仍未识别

【协议冻结】M2的拟合对象是reduced-form risk approximation
m_a(S_pre,p_tilde)，不是

\[
R_a(p)=\lambda_{a+}p+\lambda_{a-}(1-p).
\]

Regression intercept、arm coefficient、score slope及其线性组合使用alpha/tau/beta/eta命名，不称lambda_a−或lambda_a+。

【统计语义】对真实PRE Theta，恒等分解为

\[
m_a(g)=\mu_{a1}(g)\pi(g)+\mu_{a0}(g)(1-\pi(g)),
\]
\[
\pi(g)=P(\Theta=1\mid G=g),\qquad
\mu_{a\theta}(g)=E[L(a)\mid\Theta=\theta,G=g].
\]

经典lambda interpretation至少另需：

1. 在相关信息集/目标population上有效的calibrated posterior p；score的边际校准不自动等于给定S的posterior。
2. theta-conditioned loss的结构假设，例如适用域内类内risk恒定为lambda，或另行明确定义state-dependent对象。
3. endpoint/relevant support或足以识别相关分量的合法数据；当前p=0/1无观测，尾部稀疏，不能由直线端点外推替代此要求。
4. 对真实PRE标签、四格arm×Theta支持、treatment version与运输假设的独立可识别性审查。

【数据事实】theta_audit_label全部120行为null。本protocol不补算Theta，不作Theta分组估计，不把null当false，不用Q/后续vote代替Theta。单凭线性重参数化alpha_a和alpha_a+beta_a不获得经典lambda；即使未来真标签分组可识别population类均值，也不自动证明上述g-specific mixture。

【协议冻结】本阶段不拟合lambda、不产出lambda artifact，不使用lambda阈值构造router或启动Probe。

## 10. Action cost

【协议冻结】Terminal objective只使用已冻结的L_ref；暂不另设c_PUSH、c_REDIRECT或其他action penalty。Language execution failure已通过完整ITT consequence进入既有gameplay outcome，本协议不再人为惩罚失败或删除失败。

未来若提出独立于gameplay loss的明确成本，必须先定义测量对象、单位、取值及新objective，并另行获准修改理论protocol；不得看过拟合/动作效果后事后添加penalty。本V1没有额外成本项。

## 11. 下一步拟合阶段的唯一允许流程

【协议冻结】本轮无拟合授权。下一轮获得拟合授权后，只允许以下顺序：

1. 固定本protocol文本digest、分析source/environment/package versions；只读核验冻结formal/ITT/reference lineage、完整120-row population、row order与PRE covariates，保存独立分析记录而不改原artifact。
2. 对全120行先独立计算arm means、raw difference、预定Neyman SE/95% interval；如果实现Fisher supplement，严格按第4节一次预定运行，明确sharp-null与Monte Carlo属性。
3. 对完整120行按固定编码、等权OLS分别拟合M0/M1/M2；不过滤success/state/score，不搜索模型、特征、hyperparameters或score变换。复现运行仅验证一致性，不能用来择优。
4. 固定HC1诊断uncertainty，输出三模型完整结果、设计/稀疏/leverage诊断与sanity记录；应用冻结state×score support gate区分audit numbers和可用域。Unsupported输出不得提升为operational风险。
5. M2所有sanity PASS仅获得下一阶段理论审阅资格；任何FAIL则fail closed。完整primary ITT不因model结果改变，也不从M0/M1中事后挑替代“最佳模型”。
6. 报告primary与conditional对象、uncertainty局限、support状态和lambda未识别；不选择动作、不实现router、不运行Probe、不改变gameplay、不新增数据、不修改formal/ITT artifacts。

本protocol没有授予模型部署或重跑Pilot-T的权限。未来若需要变更规则，必须先形成新的显式protocol revision，记录变更理由和已知结果，不能覆盖V1或通过内部fallback静默改变方法。

## 12. 本轮完成边界

【数据事实】本轮只新增此Markdown协议；已有Support Review和未跟踪研究文档保持原字节。没有训练/拟合M0/M1/M2、计算新的CI/p-value、读取新模型结果、拟合lambda、调用真实LLM、运行gameplay或Probe、修改runtime/core/formal/ITT artifacts，亦没有commit/push。
