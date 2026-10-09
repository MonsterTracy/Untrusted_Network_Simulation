# Phase-2 Three-Way Decision：方法契约 V1

> **历史方案，非当前执行契约。** 本文记录早期 Pilot-T 设计，正文保留供研究演进追溯。
> 第 4–6 节的 `Phase2TerminalLossTableV1`、`Phase2ProbeValueModel` 和三支风险比较核已不在当前源码中；λ、Probe kernel/value 与经典阈值未获得识别，不能作为运行前置条件。
> 当前入口见 [工程状态](phase2-current-status.md)；Terminal 以 [Estimator Protocol V1](phase2-terminal-estimator-protocol-v1.md) 为准，Probe 以 [Probe Policy Protocol V1](phase2-probe-policy-protocol-v1.md) 为准，本地 Router 以 [Router-v1 contract](phase2-router-v1-contract.md) 为准。
> 下文的未启动状态、恢复规则、配对分支占位接口和未来依赖顺序均属于当时方案，不覆盖后续冻结协议及当前 production source。

## 1. 决策对象与信息时点

研究对象是当前存活狼人 (w) 在公开发言 PRE 边界对一个合法候选 (j) 的战略承诺 (x=(s_t,j))，而不是对整局胜负的直接分类。对象身份为 `(game_id, boundary_id, prefix_digest, acting_wolf, phase, candidate_j)`。所有在线输入必须在这次发言前可得；后续真实投票、放逐、被问者回答、隐藏报告有效性只属于执行或离线评价。当前实现的 `Phase2DecisionOpportunityV1` 存储 PRE 合法状态与证据，不存储未来结果或 θ 真值。

## 2. 证据后验与竞争社会状态

冻结 ToM 输出 $Q\in\mathbb R^{7\times7}$。$Q_{i,k}$ 是 observer (i) 对 target (k) 的相对嫌疑质量；对角线为零，每行在六个 non-self seats 上归一化，dead target 不从原始 support 中剔除。它不是狼身份、放逐、动作成功或胜率概率。合法目标集合为 $J_t^{speech}(w)=A_t\setminus KnownWolfTeam_w$，PK 阶段为 $J_t^{PK}(w)=J_t^{speech}(w)\cap T_t^{PK}$；竞争集合分别为 $C_t^{speech}=A_t$、$C_t^{PK}=T_t^{PK}$。$J$ 与 $C$ 不能混同，当前排除 bussing。

开发标签 θ：

\[
\Theta^{AC}_{t,j}=\mathbf1\{B_{t,j}>0,\ B_{t,j}\ge\max_{k\in C_t\setminus\{j\}}B_{t,k}\},
\]

其中 (B) 是 non-wolf observer 的 canonical suspicion-support normalized breadth。真实标签仅离线可见。在线以冻结 full-development `M3_r2_additive_logistic` 的 $\tilde p_{t,j}$ 估计 $P(\Theta^{AC}_{t,j}=1\mid Z_{t,j})$，不是行动后果。$Z=(\mu,\delta,\sigma,h,phase,|O|,|C|,r)$ 严格沿用冻结 R2 特征契约；机会构造器只调用既有 `RuntimeMapper.infer(..., audit=True)`，并记录 final Qwen3 predictor artifact digest、Q 矩阵与 R2 特征摘要。调用方必须对 final predictor digest 做上游验证；单靠记录一个哈希不能证明 Q 的来源。后续后果模型不得把这个 $\tilde p$ 重新解释为行动成功率。

## 3. 三支动作与合法几何

`Push` (P(j)) 主动推动放逐 (j)。`Redirect` $N(j\to k)$ 拒绝把 (j) 作为本轮主要目标，并主动转向 (k)；拒绝不等于保护、沉默或宣称 (j) 是好人。$k=\rho_t(j)=\arg\max_{u\in J_t(w)\setminus\{j\}}\tilde p_{t,u}$，精确并列按 canonical seat order。语言模型不选择 (k)。`Probe` (B(j,r)) 向 (j) 公开询问其当前主要怀疑或投票判断及已公开依据，request type 固定为 `CURRENT_SUSPICION_BASIS`。

Push 对合法 (j) 始终可选。Redirect 需要另一合法目标。Probe 需要公开队列满足 `pos(w) < pos(j) < pos(w_next)`；continuation actor 是 (j) 后第一位 later wolf teammate。观察窗是当前 Probe 发言后的公开 speech events，截止该队友的同阶段 PRE 之前，不含 vote、PK/exile 结算或私密信息。因此可行动作集合可以是 `{P}`、`{P,N}`、`{P,B}` 或 `{P,B,N}`，不能强迫每个 PRE 三臂。`Phase2TreatmentV1` 对每臂冻结 plan、请求、目标、分配源、propensity 与 randomization key；自然历史的 `vote_intent` 从未被视为 `do(P)` 或 `do(N)`。

## 4. 后续价值、结果分类与终止后果

发言前计数状态 $S_{pre}=(n_W,n_{NW})$。白天真实结算 $Y$ 的互斥类别为 (y_j)（候选 (j) 被放逐）、(y_{NW})（其他非狼）、(y_{Wa})（acting wolf）、(y_{Wm})（狼队友）、(y_{empty})（无人放逐）。转移 $S_+=\tau(S_{pre},Y)$ 与参考损失

\[
L_{ref}(Y;S_{pre})=1-V_{ref}^{pub}(\tau(S_{pre},Y))
\]

完全复用 `phase2_offline.ReferenceTables` 的 development cross-fit 或 full-development API，不另建价值表。`Phase2DayOutcomeV1` 对非当前存活玩家的放逐 fail closed。参考 $V_{ref}^{pub}$ 是冻结 public-policy 续局胜率基准，不是新三支政策的胜率估计。

终止臂 $a\in\{P,N\}$ 的待估条件损失为

\[
\lambda_{a,\theta}(g)=c_a(g)+\sum_y P(Y=y\mid do(a),\Theta=\theta,g)L_{ref}(y;g).
\]

(g) 是预先指定的合法 context；(c_a)、条件结果分布及 λ 尚无 pilot 支持，不能自行赋值。只有取得受控 intervention 后才可选择 consequence estimator。纯风险核按同一 $\tilde p$ 计算

\[
R_P(p)=\lambda_{P,+}p+\lambda_{P,-}(1-p),\qquad
R_N(p)=\lambda_{N,+}p+\lambda_{N,-}(1-p).
\]

`Phase2TerminalLossTableV1` 只是带 context/provenance 的数值容器。当前 production loader 必然 fail closed；synthetic 单测用的数字不表示项目 λ。经典固定六损失时可由同一 Bayes-risk 下包络审计 $\alpha,\beta,\gamma$，但真实 Probe 是序贯干预，静态阈值仅为代数诊断，不是 production router。

## 5. Probe 的序贯价值

Probe 不是一个随意设定的常数或 (1-p)、entropy(Q) 等启发式风险。其研究目标是

\[
R_B(x;q)=\kappa_q(x)+\mathbb E_{O\sim P(O\mid do(B),x,q)}
 [V_{t+1}(T_q(x,O))].
\]

这里 $O$ 仅含冻结 observation window 的公开 speech；response opportunity、实际公开观察和后续再决策到达必须分开。`Phase2ProbeSequenceRecordV1` 记录 T0 原 PRE、T1 request execution、T2 公开观察窗、T3 队友 continuation PRE 与更新 Q/mapper panel、T4 可选日终结果。`information_gain` 当前恒为 null；Δp 的绝对值不能擅自命名为 information gain。`Phase2ProbeValueModel` 显式分开 observation kernel、transition 与 continuation value；生产求值在 pilot/kernel/value 模型未验证前直接拒绝，synthetic exact-expectation 只用于单测。

## 6. 三支比较与语言执行的角色

当且仅当一个机会的全部合法臂风险有来源、有限、context 兼容时，数学比较核返回 `argmin`；精确并列顺序固定为 Push、Redirect、Probe。这个核明确是 non-production：当前既无 fitted λ，也无 Probe observation/continuation model，故不能形成最终 `ThreeWayRouter`。LLM 的职责只是在上游动作、(j)、(k)、请求冻结后生成公开发言。独立 perceiver 只看发言与可信公开 context，不看 requested plan；verifier 对照冻结 plan，最多 repair 一次。验证失败即无 commit-ready treatment 文本，不自动切换动作或目标；Online Pilot-T 若让当前游戏按 baseline 发言继续，必须把原 assignment 标为 noncompliance，不能把 baseline 发言称为被分配的 treatment。actual vote 仍由原 gameplay 后续逻辑决定。

## 7. Online Pilot-T 的随机化识别

自然历史的 vote intent 不等于干预。Pilot-T V1 每局只考察第一个 P/N 双合法的存活狼 `speech`/`speech_pk` PRE。记该 PRE 上 P/N 双合法的候选池为 $J_{PN}$，则先均匀抽取 $j$，$P(j\mid PRE)=1/|J_{PN}|$，再用独立域分隔的随机键按 $\pi(P)=\pi(N)=0.5$ 分配终止动作。分配只用 PRE 身份、冻结 plan 和 seed，先于语言生成、真实投票、θ 标签与结果。一个 game 至多分配一次；此后恢复原 baseline gameplay。这样的在线随机化直接实施单路径 `do(a)`，无需同一 PRE 的中途 checkpoint/replay。

`Phase2OnlineTerminalPilotPlanV1` 冻结此候选与动作双重随机化规则。候选池、被选 (j)、$1/|J_{PN}|$、两个随机键及 P/N propensity 分开记录。qualification 固定 10 次 assignment，正式 Pilot-T 固定 120 次 assignment；失败执行同样计数，显式 `max_games_attempted` 只作安全上限。assignment 在任何语言调用前写入并 `fsync` append-only 账本；重启后不能凭缺失的游戏 checkpoint 重演旧 game，旧 assignment 被保留并标为 interrupted。三张表分别保留所有分配、执行/不依从、以及成功执行后真实日终 (Y,L_ref)；qualification 显式排除未来 λ 数据集。本 V1 estimand population 是“每局首个 P/N 双合法狼 PRE 上均匀抽样候选后的干预总体”，不是所有 Phase-2 opportunity 的总体；向更广运行时总体的泛化留待 support review。本轮不选择 ITT/per-protocol estimator，不拟合 λ。

完整 PRE checkpoint/restore 与同源配对 branch replay 保留为**可选的 paired-counterfactual enhancement**，可用于未来方差降低或机制诊断。当前 replay API 继续 `executable=false` 并 fail closed；这不阻断 online randomized Pilot-T。公开 PRE JSON 仍不能冒充完整可恢复状态。

## 8. 未来运行链与当前边界

Pilot-T 的 opt-in 链为：自然 wolf speech PRE → frozen Q → frozen RuntimeMapper $\tilde p$ panel → 候选 (j) 选择 → `Phase2DecisionOpportunityV1` → 随机分配 P/N → verified language → 现有 canonical public speech commit → 原 baseline gameplay → 原有真实 vote → 日终 (Y,L_ref)。`run_random.eval` 只有显式传入 online pilot hook 才走此路；默认路径和 actual vote 语义保持不变。Phase-2 sidecar 与 canonical V1 parser annotation 分开，调用审计在 canonical `RUNTIME` 类别留下后端证据，不冒充 V1 `SPEECH_PERCEPTION` 尝试。

证据依赖顺序是 Smoke-V3 → Online Pilot-T → support review → terminal consequence estimator → λ(P/N) → Probe Pilot → Probe value model → production ThreeWayRouter → gameplay integration → paired win-rate experiment。Probe 的 T0–T3 接口保留，但不参与 Pilot-T 随机化。最终游戏胜负只作描述性审计，不能代替日终 $Y$ 与 $L_{ref}$。

本契约的关键原则是：**evidence determines posterior; consequence determines loss**。未获得受控分支数据前，本文只规定表示、合法性、记录与可验证数学运算，不声称三支策略优于现有 gameplay，也不从 smoke language gate 推断胜率。
