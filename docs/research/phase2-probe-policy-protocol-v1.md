# Phase 2：PROBE→REDIRECT 整体策略协议 v1

本协议比较两条随机分配的完整策略：立即尝试 REDIRECT，或先尝试 PROBE、观察公共发言后由指定狼队友尝试 REDIRECT。研究对象包含冻结 Language V1.2 验证失败后的基线分支。
状态：研究定义冻结用于本地开发；正式样本参数及规范执行计划仍须在执行前冻结。

## 1. 入组与随机化

令当前 PRE 的完整合法非狼候选集合为 `J_t`，定义：

\[
E_t=\{j\in J_t:\operatorname{probe\_continuation}(t,j)\text{ 存在，且 }|J_t|\ge2\}.
\]

- 每局在基线运行中寻找首个 `E_t` 非空的 PRE；阶段依 ActionContract 包含 `speech` 和 `speech_pk`。
- 从 `E_t` 均匀选择 `j`；单元素集合也合法。两臂共用这一初始规则，`j` 选定后冻结。
- 每局独立以 `0.5/0.5` 分配完整策略 `π_R` 或 `π_B`。候选选择与策略分配使用分离的随机域。
- 在任何 Phase 2 后端调用前持久化初始上下文、候选池、选择结果、随机键、概率和策略分配；每局仅分配一次。
- 不设旧 M2 状态/分数支持门、不设额外阶段截断；不用未来发言、最终结局或答案内容筛选初始机会。
- Q/R2/M3 的完整面板是运行质量前提，不是分数入组条件；构建失败不得改选 `j` 或跳过已分配游戏。

动作定义及候选约束见 [ActionContract](phase2-action-contract-v1.md) 与 [phase2_actions.py](../../werewolf/phase2_actions.py)；既有单局分配保护见 [phase2_online_runner.py](../../werewolf/phase2_online_runner.py)，真实 PRE 注入点见 [run_random.py](../../run_random.py)。

## 2. 两条完整策略

`T0` 为初始 PRE，`T1` 为当前发言的 LANGUAGE 验证及规范提交，`T3` 为 PROBE 合同指定的后续狼队友 PRE。
REDIRECT 在所处 PRE 的完整当前面板上计算：

\[
k_0=\rho(H_0,j),\quad k_1=\rho(H_3,j),\quad
\rho(H,j)=\arg\max_{k\in J(H)\setminus\{j\}}\widetilde p_H(k).
\]

采用冻结选择器及规范座位顺序处理并列；不固定 `k_1=k_0`。T0 与 T3 分别使用对应可信狼视角、当前 PRE 的冻结 Q/R2/M3 流水线。

**`π_R`：立即尝试 REDIRECT。** 在 T0 构建面板并选择 `k_0`，尝试 `REDIRECT(j,k_0)`，随后回到基线。
若最终 LANGUAGE 判为无效，按既有规则执行当前基线发言，余下过程均为基线。

**`π_B`：尝试 PROBE，再尝试指定队友 REDIRECT。**

1. T1 尝试冻结的 `PROBE(j)`，询问 `j` 当前主要怀疑/投票对象及公共依据，不要求承诺。
2. 若最终 LANGUAGE 判为无效，不提交无效文本；取消 T3，执行当前基线发言，余下过程均为基线，不再追加受控干预。
3. 仅在 LANGUAGE 有效且通过规范路径成功提交后启用 T3。
4. 观察合同冻结的窗口：当前狼发言之后、`j` 之后首位狼队友发言之前的全部公共发言；窗口内依基线继续。
5. 在指定队友的 T3 PRE，以原始固定 `j` 和实际当前完整面板选择 `k_1`，尝试 `REDIRECT(j,k_1)`，随后回到基线。
6. T3 的最终 LANGUAGE 无效时，仅执行该 PRE 的既有基线分支，之后不再追加受控干预。

每个计划沿用冻结的最多一次 repair；无效生成文本只保留为审计，不进入公开历史。恢复基线意味着在同一 PRE 执行真实基线发言，不跳过发言回合。
speaker/phase 来自各阶段可信公共 PRE；perceiver 仍不接收 requested plan，计划只用于 actor realization 与最终 verifier comparison。合法狼队仅用于既有合法集合构造，不向公开语言接口注入 private role facts。

T3 不以 `j` 是否回答、答案是否有信息或分数变化触发；不使用旧 M2、`κ`、kernel 或 router。当前 REDIRECT 选择器本身已允许随面板变化调整目标。
T3 继续原始单局分配，不新分配或再次随机化；其阶段计划/摘要须在 T3 后端调用前绑定原始分配及实际 PRE 并持久化。

LANGUAGE 本身只处理当前语句的生成与验证；成功触发和失败取消 T3 是本协议新增的调度规则，不能声称既有 PROBE 运行时已实现。
现有最终无效→基线原则与单次分配停止规则分别见 [pilot runbook](phase2-intervention-pilot-runbook.md)、[phase2_online_runner.py](../../werewolf/phase2_online_runner.py)；验证及规范事件核对见 [phase2_language_realization.py](../../scripts/phase2_language_realization.py)、[phase2_verified_speech.py](../../werewolf/phase2_verified_speech.py)。

## 3. 合同异常与证据

同一公共发言阶段按队列推进，正常窗口不发生死亡或合法候选集合变化；见 [环境发言分支及队列推进](../../werewolf/envs/werewolf_text_env_v0.py)。
本协议的最终 LANGUAGE 无效仅指结构合法的计划在冻结生成/验证路径中被明确判 invalid；结构、PRE 身份及运维异常不属于这一分支。

- 若 T3 上下文或合法性意外不符合合同、指定队友/PRE 不存在、实际 Q 证据缺失、面板构建失败或规范提交异常，立即停止并保存证据。
- 这些异常不得套用 LANGUAGE 无效→基线分支；不得替换 `j`、替换队友、改到后续阶段、重试干预或重新分配。
- 保留原始分配、事件与提交状态；提交不确定时不能宣称 PROBE 成功，也不能继续 T3。
- 仅当真实规范结局已经产生且可审计时恢复该结局记录；否则标记结局不完整，不插补、不删除、不重跑替代该游戏。

## 4. 主效应与分析

在上述初始入组及均匀选 `j` 的总体上，主目标为：

\[
\tau_{\mathrm{ITT}}=\mathbb E\!\left[L_{\mathrm{ref}}^{\mathrm{assigned}\ \pi_B}-L_{\mathrm{ref}}^{\mathrm{assigned}\ \pi_R}\mid\text{初始入组与选择规则}\right].
\]

负值有利于 `π_B`。这是实际尝试策略的总效应，包含 LANGUAGE 无效分支；不是保证完美执行的效应，也不是纯信息效应。
公共提问对其他玩家的影响、等待期间发言、执行发言的狼的变化和目标更新均属于该效应。每局一次分配，跨局不进行自适应学习。

所有初始分配均保留在主 ITT 分母中，不按 T1 成功、T3 执行、答案质量或分数变化过滤。
两臂沿用同一初始日结局口径：完成该日最终投票/PK 决议后，相对于 T0 的 `S_pre`、原始 `j` 和合法狼队，复用 [既有 outcome bridge](../../werewolf/phase2_outcome.py) 与冻结的 `V_ref_pub_full` 完整表计算 `L_ref=1-V_ref_pub_full(S_plus)`；不能改用 T3 作为新的结局起点。实际选票由原 gameplay 计算，发言的 vote intent 不直接设置选票。
普通投票平票产生的中间 `exile=[]` 不是该日最终结局。

报告各臂数量、均值与原始均值差 `τ̂=mean(B)-mean(R)`；使用 Neyman 标准误
`sqrt(s_B²/n_B+s_R²/n_R)` 及正态近似 95% 区间 `τ̂ ± 1.959963984540054 × SE`，样本方差使用 `ddof=1`；无需结局模型。任一臂少于两个结局时不报告该区间，不能自行设方差为零。
T1 验证、规范提交、T3 执行/取消及目标变化仅作次要过程记录，不用于主分析筛选。
任一已分配游戏缺少真实最终结局时，停止完整 ITT 分析声明，明确报告不完整状态。

## 5. 执行前冻结与资格检查

正式执行前另行冻结总分配样本量、种子列表/生成规则、游戏池、运行上限、规范执行计划，以及 Q/R2/M3、LANGUAGE、基线和参考表的具体版本及内容摘要。
保留现有 source provenance、tracked-clean、模型/后端及 canonical runtime guards，不因本协议放松核验。
不得直接沿用旧 `120/240` 作为新实验参数；不以观察到的效果提前成功停止。
资格检查与正式实验分开记录，不将资格样本并入正式效应分析。

本次只读可行性核对使用本地缓存 `/tmp/phase2-terminal-support-input-v1.json` 中的正式历史分配：

- `assignments.jsonl` 内容 SHA-256：`8544fb65e628a6876d1b1bc1c2d580e28bfd1c4e7b15c1345200be4296d4339a`。
- 正式 manifest digest：`ec1cc8730aefe8fce57ed83a3c84ca9231ae645ea6c9047cd588f8022d2223c2`。
- 120 个历史首个 P/N speech PRE、553 个候选行中，88 个 PRE 存在合法 `E_t`，合计 194 个候选；历史已选 `j` 有 43/120 合法。
- `|E_t|` 直方图为 `{0:32, 1:29, 2:25, 3:23, 4:9, 5:2}`；其中 29 个单候选 PRE 仍纳入新规则。

这些是旧首个 P/N PRE 的机会描述，不能当作新规则扫描首个 `E_t` PRE 的未来入组率，也不证明 T3 执行率；候选行不是独立游戏。
较大开发机会集的用户交接记录给出普通 speech 1358 PRE/2759 候选行；PK 另有 8 PRE/8 行，其 REDIRECT 交集尚未核对。因此交接范围为 1358–1366 PRE/2759–2767 行，本次未重新验证原始开发发布数据。

源代码核对基准：`twd/mainline`，HEAD `56588c1d075c5b97511d84ceb4ed3c68215d8cc0`。
