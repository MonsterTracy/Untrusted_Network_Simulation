# Phase-S2-P0：上下文认知风险与候选发言评议

状态：**CONTROLLED RESEARCH PROTOTYPE / NOT A FROZEN STRATEGY EXPERIMENT**。
基线：`twd/mainline@9e0a4b523b7dd0cace9111d743f0e264bd48a498`，加五项未提交 S1a；本轮保留其字节。
新代码不被 `run_random`、`speech_runtime`、Shadow 或采集入口调用，也不导入 env/agent/recorder/监督 reader。
复用 D3 数学、S1a G 和原候选；现有 Actor/Parser/Verified Commit 完全不改。

## 1. 原上下文成本：核查结果

原 `contextual_loss(context, tie_policy=...)` 保留：
`c_FP=1`，`c_FN=2**(1[PK]+1[nW=1])`；普通讨论 `k=1/4`、PK `k=2/5`，`b0=k*c_FP`、`b1=k*c_FN`。

| phase | nW | c_FN | beta | alpha |
|---|---:|---:|---:|---:|
| discussion | 2 | 1 | 1/4 | 3/4 |
| discussion | 1 | 2 | 1/7 | 3/5 |
| pk_discussion | 2 | 2 | 1/4 | 3/7 |
| pk_discussion | 1 | 4 | 1/7 | 3/11 |

各状态 `b0/c_FP+b1/c_FN=2k<1`，非退化，`0<beta<gamma<alpha<1`。
所有风险、阈值和并列由原 `decision.py` 计算；无概率舍入、裁剪或阈值近似。
这是依当前合法 phase/活狼数的规范性成本假设，不是从真实发言收益拟合的函数；倍数与 k 本轮未改变。
普通讨论双狼成本只是本轮静态 scenario 22 的正比例缩放；PK 改变相对成本，不能把自身风险数值跨成本直接当共同收益比较。

## 2. 认知超额风险

沿原事件 Y=1[有效嫌疑支持命中至少一名活狼]，不是角色概率、点名端点或行动成功：

\[
\Delta_i(d)=R_i(d)-\min_{z\in\{P,B,N\}}R_i(z),\quad
\bar\Delta_i(d)=\Delta_i(d)/(c_{FP}+c_{FN}).
\]

`judgment_risk(p, loss, judgment)` 仅接受 p、原 LossSpec 和认知 Region，调用原 `decide`，输出全部原风险、最优集合、阈值、强制判断风险、Δ 和归一化 Δ 的精确有理数记录。
`LossSpec` 已保证分母为正。合法成本整体乘 λ>0 时，Δ 乘 λ，归一化 Δ、argmin、依据门控和相同行动评分下的选择均不变。
它是强制认知判断的额外代价；没有真实行动后果概率，不是策略实际损失、游戏 regret、reward 或胜率。

## 3. 预绑定依据接口

```text
原 generate_candidates(PRE)
  → bind_bases(PRE, complete_candidates, explicit CandidateBasis tuple)
  → S1a G（合法 context/features/p/原 D3 decision）
  → appraise_candidates(PRE, G, BoundBases, mode, independent values)
  → 只返回既有 SpeechPlan/依据 ID/私有审计，或明确无结果
```

`CandidateBasis` 是已声明的论证类型，不是自然语言模型的事后自由解释：

| kind | 所依赖的认知判断 | 含义 |
|---|---|---|
| PUBLIC_ONLY | 无 | 仅使用已冻结公共 PRE；可显式引用实际 V1 evidence |
| ASSUME_HIT | POS | 需要采信观察者有效支持命中活狼的命题 |
| ASSUME_MISS | NEG | 需要采信该命题不成立 |
| KEEP_BOTH | BND | 保留两种可能，不断言事件真假 |

认知依据明确指向一个存活非狼 observer；没有该 G 行、或该 seat 不在合法论域时拒绝该依据，不补 BND。
PUBLIC_ONLY 不能携带 observer 私有认知依赖；无额外 V1 引用时只表示调用者显式声明的 PRE 公共理由，不自动添加任何事实断言。
`PublicReference(event_id,annotation_digest)` 核对实际成功 V1 注释。它证明引用存在/未变，**不证明理由在语义上足以支持某句话，更不证明发言有效**。
不接受任意自然语言“理由”字符串来伪装真实标签。依据 ID 只是稳定标识；kind、对象及引用才是可审计模式。

绑定接口不接收 p/Q/风险或选中动作；固定 PRE、完整候选、全部依据及 digest 后才评议。受控脚本在 Q/p 计算和候选评议前把全部目录/人工评分写入 `bindings-before-appraisal.jsonl`。
内存不可变类型本身不能证明外部调用者的时间诚实；正式实验仍须运行前封存配置/依据目录，不能把一次事后构造的合法对象当预登记证据。

同一 SpeechPlan 可绑定多个不同依据。任一依据可采信则该计划可评议；未声明依据的计划仍在完整合法集合中，但标记 NO_DECLARED_BASIS。依据增加不叠加行动分数。
重复 basis ID、错 PRE、错/重复/非规范公共引用、非完整候选或伪造 G 全部报错。

## 4. 门控：明确的规范性假设

- `RECORD_ONLY`：保留所有具备合法认知行/公共引用的已声明依据，记录风险但不按判断门控。
- `THREE_WAY`：认知依据必须匹配原 `decide` 经显式 TiePolicy 选中的判断。PUBLIC_ONLY 不受认知门控。
- `CONTINUOUS_EQUIVALENCE`：直接对同一三项原风险按同一 tie order 取 argmin，使用同一依据规则。与显式三支完全等价，**不是独立优化算法或增量收益基线**。
- `FORCED_BINARY`：复用原 `forced_binary`，只在 P/N 间选判断；认知 KEEP_BOTH 依据被排除，公共理由仍可用。这是强制二支消融，不修改 D3 或行动集合。

本版不新引入风险容忍阈值、惩罚系数或归一化风险—行动效用混合权重。
“匹配已选认知判断”是可检验的**依据可采信规则**，不是从行动效果推导出的最优策略。
精确并列时，Δ=0 的另一判断可能仅因既有 tie order 被拒绝；审计明确记录 ZERO_EXCESS_ALTERNATIVE_REJECTED_BY_EXPLICIT_TIE。故不能把它描述成仅凭 Δ>0 才拒绝的规则。
强制二支可能选到相对完整三支最小值有正 Δ 的判断；该 Δ 仍相对完整 P/B/N 最小风险报告，不偷偷改参考基准。

KEEP_BOTH 是认知暂缓的理由类型，不是 Probe 或“必须试探”。任意现有合法动作都能由不同类别依据承载；没有 POS→SELF_DEFEND 等内置行动映射。

## 5. 独立行动值及确定性选择

`artificial_action_values(PRE,candidates,scores,fixture_id)` 只绑定公共 PRE、完整候选和显式有限偏好分数，不接收 G、p、Y 或风险。
当前仅接受类型明确的 `ArtificialActionValues`，scope 固定为 ARTIFICIAL_PREFERENCE_NOT_MEASURED_SPEECH_EFFECT；**没有正式行动效果来源**，也不对真实发言价值排序。
必须为每个合法候选提供一项分数；缺失、NaN/Inf/bool、来源 PRE/候选摘要不符即拒绝。
人工工厂不能独立证明调用者从未用标签造分数；本轮受控脚本的固定 `score=N-canonical_index` 与源码/运行前 plan 证明其生成规则，不冒称正式价值模型 admission。

流程：完整候选 → 引用/论域/依据检查 → Δ 记录和显式门控 → 在有可采信依据的计划中按独立分数最大化 → 原 canonical candidate 顺序破平局。
同计划的多个可采信依据按 basis_id 字典序选择展示 ID，仅作确定性审计，不改变该计划分数。
不对分数加减认知风险，不使用预测 Q 变化作为真实收益，不拟合候选价值。

无可采信依据返回 NO_ADMISSIBLE_BASIS；有可评议计划但无行动值返回 ACTION_VALUE_UNAVAILABLE。二者 `selected_plan=None`，不会自动生成 NO_COMMITMENT、均匀 Q 或替代计划，也不代表向游戏提交“跳过行动”。
选中时为 SELECTED_ARTIFICIAL_ONLY。输出不含 env.step、投票、提交接口或 verified envelope；本轮未把新评议器连接 Actor。

## 6. 固定人工对照

复用 S1a 的 30 个场景：普通/PK × self/team/无公开指控 × 5 个合成 Q 的 r；两狼存活。单狼/死亡人群另由单元测试覆盖，不将 30 场景外推为完整游戏效果。
共享同一 PRE、Q、D2 p、公共选择观察者、完整候选、预绑定目录和行动评分；每场景一次合成 Q、一次 D2 predict，行动表一次填满。模型是原小型 artificial D2 fixture，不加载/训练 ToM，所有评议模式无额外模型/语言调用或提交。

四个固定目录：全 PUBLIC_ONLY；按候选序轮流 HIT/MISS/BOTH；最高偏好计划有三类依据、其他公开依据；全部 KEEP_BOTH。
每目录 2 成本 ×（S1a 参考 + 记录风险 + 三支门控 + 连续等价 + 二支）=10 单元；总计 30×4×10=1200。
S1a 沿用其原 EVIDENCE 映射，虽然共享已准备好的评分表，**它不使用新目录或分数选择**；它只作历史策略参考，不能将 S1a→P0 的差异全归因于门控。严格门控消融是 P0 RECORD_ONLY→THREE_WAY。
候选枚举/分数查询预算固定，无树搜索和重试。每个模式包含的确定性验证运算不同，不能声称精确 CPU 时钟一致。

完整逐单元/逐候选/逐依据记录、40 组汇总及 52 项配对对照由脚本输出；没有只筛选发生变化的场景。
“完整 SpeechPlan 变化”包含目标变化；脚本另外报告动作类型变化和仅目标变化。人工分数偏好前排候选，选出 ACCUSE_WOLF 是夹具构造结果，不是策略推荐。

## 7. 数学和兼容验收

单元测试覆盖：原四状态成本及所有 tie、阈值/非退化；Δ 手算/整体正缩放；全部 45 成本×3 tie 的阈值端点与连续等价；退化成本；三种认知域、同动作不同依据、公共理由不变、全拒绝/价值缺失；原候选和引用校验；评分反向变化与风险不变；单/双狼、PK、死亡玩家；监督反事实、无读取和 RNG。
复用 S1a、D1 特征/标签、D2 估计器、D3 decision、Actor/Parser/verified commit 与 canonical 相关回归。新模块不调用这些受控执行入口；旧闭环通过只能说明兼容，不能证明新评议策略有效。
准确的测试通过/失败/跳过数、运行命令、完整对照数值、案例及源码指纹记录在本轮项目外验收包。

## 8. 未解决问题

依据模式是否应采纳 selected-judgment、全部最优集合或有证明的风险容忍规则，需要未来独立研究，本轮只实现前者，记录 exact-tie 敏感性。
自然语言理由的充分性和真实行动价值没有资格证据；人工分数不解决 G1/G2、CAL 或在线迁移。公共依据过宽可能绕开认知门控，三类依据齐备时门控可能只换理由不换计划——这正是必须报告的反例，不是 bug 或必须修掉的无差异。
完整面板融合、真实价值评估、正式冻结与真实生产接入均未实现。当前只用明确受控 G，不放宽原 ToM/OOF/final/CAL 的来源检查。
本轮完成后停止，不进入正式 G1/G2、真实发言效果、MCTS/CFR/CPT/后悔理论或胜率研究。
