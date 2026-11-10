# 全项目科研架构与模块导航

本文连接现有科研模块、数据流和文档，不新增开发规范、执行协议或实验门禁。状态依据截至 2026-11-10 的本地审阅、既有归档与用户确认，不是服务器实时状态。各实验仍以其原始 Source、冻结合同和封存证据为准。

## 1. 项目目标与科研问题

项目围绕七人、多日、文本狼人杀（Classic7），从可信游戏数据研究观察者认知，再研究如何合法消费认知预测开展决策、语言行动和策略评价。默认 Canonical Collection 是数据生成路径；下游策略研究通过显式 opt-in 接入，不自动改变默认玩法。

需要区分以下对象：

- **真实狼人身份**是模拟器中的隐藏真值，不是 ToM 要直接预测的监督目标。
- **观察者主观怀疑**是该观察者在 PRE 时实际报告的怀疑集合，可能反映其合法私有认知。
- **ToM 怀疑质量 Q**只从公共结构化历史和观察者身份预测非自身席位上的相对怀疑质量，不是狼人身份后验或胜率。
- **社会状态与条件事件**分别由 Phase-2 的 θ 和 Observer-DTRS 的事件合同定义；二者及其估计量不能互换，也不能直接用 Q 代替。
- **行动风险与序贯决策**依赖明确的损失、行动和后果证据；**最终游戏胜率**是完整策略与游戏运行的结果，不能由认知指标或语言合规率直接推出。

## 2. 整体科研流程

下图中 `→` 表示数据或 Artifact 依赖，`⇢` 表示研究流程或执行次序；均不表示 Python 模块的直接导入关系。

```text
Canonical Collection → Publication / Fold
  ├─ 公共 PRE、观察报告与训练人口 → Observer-ToM
  │    ├─ OOF Q → 离线评价 / Baseline Reference / DTRS 与 Mapper 研究
  │    └─ Final Fit → 封存预测器 → 在线 PRE 的 Q → Mapper score panel
  └─ 同一公共 PRE → TomBaseline V1 / V2 → 与冻结 OOF Reference 配对比较

决策与策略研究 ⇢ 预定行动 ⇢ 语言实现与验证 ⇢ Canonical Commit
  ⇢ 原 Gameplay / Vote ⇢ 日终后果或完整 GAME_RESULT ⇢ 对应实验评价
```

Publication 中的观察报告在下游用于训练监督、离线标签或评价，不进入 Baseline Prompt。TomBaseline 研究通用模型的 ToM 任务能力；Observer-DTRS / Phase-2 研究认知证据如何进入决策与策略；语言执行层实现并验证预定行动，不能用语言成功代替行动效果。

## 3. 模块职责与输入输出

文档入口保留各自的协议、草案、归档或实现说明身份；被总览引用不代表获得新的科研资格。

| 模块 | 科研职责 | 主要源码位置 | 核心输入 | 核心输出 | 当前科研状态 | 合同／证据入口 |
|---|---|---|---|---|---|---|
| 游戏环境和 Canonical Collection | 执行 Classic7，记录因果合法的公共历史和采集来源 | [envs](../werewolf/envs/)、[canonical_collection](../werewolf/canonical_collection/) | Collection Plan、Agent、Backend、游戏规则 | PRE、观察报告、Bundle、Ledger、失败证据 | 可信数据基础设施已实现；具体采集完成以原产物为准 | [采集合同](collection_contract.md)、[数据生命周期](research/data-acquisition-preprocessing.md) |
| Development Publication | 验证并发布 Development 人口与整局划分，隔离受限角色信息 | [Publication V1](../werewolf/development_publication.py)、[V2](../werewolf/development_publication_v2.py)、[Split](../werewolf/canonical_split.py) | 已验证 Collection、预定划分规则 | 公共 Publication、Fold、受限 Role Sidecar | V1/V2 合同并存；不是两个任意互换的数据入口 | [数据生命周期](research/data-acquisition-preprocessing.md) |
| Observer-ToM | 建模 observer-specific 非自身怀疑质量 | [werewolf/tom](../werewolf/tom/) | 公共 PRE、观察者、合法训练监督与人口掩码 | OOF Q、Checkpoint、封存 Final Fit 预测器、认知评价 | OOF、Final Fit、独立评价生命周期已实现，各自保留来源 | [ToM 合同](tom_contract.md)、[Qwen3 Final Fit](qwen3-final-fit.md) |
| TomBaseline | 在共同公共观察任务上比较通用模型与专用 ToM | [werewolf/tom_llm](../werewolf/tom_llm/) | 公共 Task、Selection / Cohort、模型、冻结 Reference | 模型预测、Run / Seal、配对分析 | 三模型 V2 GPU 推理及单模型分析获用户确认完成；联合分析待确认 | [现行操作说明](operations/tom-llm-unified.md)；历史实验仍用原 Source 合同 |
| Observer-DTRS | 条件事件估计、显式损失与静态三支风险比较，以及 D4 工程证据链 | [werewolf/observer_dtrs](../werewolf/observer_dtrs/) | PRE、Q、离线事件标签、显式 LossSpec | 条件概率、风险、POS/BND/NEG、部署与审计证据 | D1–D4 分阶段实现和归档，不等于完整在线最优策略 | [D2 结果归档](research/observer-dtrs-v1-d2-results.md)、[D3 受控实现](research/observer-dtrs-v1-d3b.md)、[D4 工程](research/observer-dtrs-v1-d4c2b2-resident-service.md) |
| Phase-2 Mapper | 从 Q 和合法竞争状态构造 R2，估计社会状态 score | [phase2_offline.py](../werewolf/phase2_offline.py)、[phase2_mapper_runtime.py](../werewolf/phase2_mapper_runtime.py) | 离线 OOF Q / 标签；运行时 Final Q 与合法状态 | Mapper、p_tilde score panel | 离线 Study、Final Fit 与运行时路径已实现；不授予行动效果解释 | [离线数据合同](research/phase2-offline-data-contract.md)、[当前导航](research/phase2-current-status.md) |
| Terminal / Probe / Router | 研究终止行动后果、固定序贯策略及游戏级策略比较 | [phase2_online_runner.py](../werewolf/phase2_online_runner.py)、[phase2_gameplay.py](../werewolf/phase2_gameplay.py) | 冻结预测器、Mapper、合法机会、实验 Plan 与 Ledger | Assignment、日终后果、ITT、完整游戏结果 | Terminal、Probe 有交接完成记录；Router 实现与正式游戏实验状态分开 | [Terminal 协议](research/phase2-terminal-estimator-protocol-v1.md)、[Probe 协议](research/phase2-probe-policy-protocol-v1.md)、[Router 合同](research/phase2-router-v1-contract.md) |
| Language / Speech | 采集期 V1 语义感知；下游预定行动的语言实现、独立解析和验证 | [speech](../werewolf/speech/)、[phase2_verified_speech.py](../werewolf/phase2_verified_speech.py) | 合法观察、原始公开发言或冻结行动 Plan | V1 注释或经验证的行动发言、审计 | 两种职责均已实现，合同独立；语言层不替代实际投票 | [采集合同](collection_contract.md)、[Action 合同](research/phase2-action-contract-v1.md) |
| Planning Research | 探索候选规划、情境损失和认知依据如何影响策略意图 | [planning_research.py](../werewolf/observer_dtrs/planning_research.py)、[speech_planning.py](../scripts/speech_planning.py) | 合法 PRE / 视角、Q、显式研究假设 | 候选、受控评分、策略意图和原型比较 | 研究进行中；历史消融与原型不等于当前 Router | [动态规划研究](research/observer-dtrs-dynamic-planning-phase1.md)、[情境发言原型](research/observer-dtrs-contextual-speech-prototype.md) |
| Evaluation / Reporting | 按各任务固定人口和终点进行评分、统计及可视化 | [tom/reporting.py](../werewolf/tom/reporting.py)、[baseline_analysis.py](../werewolf/tom_llm/baseline_analysis.py)、[统计脚本](../scripts/) | 对应任务的封存预测、标签、Assignment 或游戏结果 | 认知指标、配对统计、ITT、游戏结果报告 | 实现存在不代表每份正式报告已生成 | [ToM 合同](tom_contract.md)、[Baseline 操作](operations/tom-llm-unified.md)、[Phase-2 导航](research/phase2-current-status.md) |

## 4. 各研究阶段的当前状态

**代码已经实现**只说明功能存在；**正式实验已完成**需要用户确认、正式 Seal 或实验报告等明确来源；**研究仍在进行**表示方法或实验尚在完善；**历史研究**保留原始方法与执行记录，不代表现行操作协议。

- **TomBaseline**：用户确认 Qwen3、Yi、Mistral 三模型 V2 GPU 推理及单模型分析已完成，冻结执行 Source 为 `bacc35135fffa527f60b532ee12e3ca2859398fc`。本轮未复核其 Seal。三模型联合配对分析是否完成，待 Baseline 对话提供最新证据。当前开发主线的轻量执行合同不替换这一历史身份。
- **Observer-ToM**：OOF、Final Fit 与独立 Final Evaluation 是不同生命周期。已有实现与历史模型来源不自动证明当前 Checkout 或服务器实例通过真实模型验证。
- **Observer-DTRS**：D1 提供数据与特征，D2 有条件概率验证结果归档，D3 提供静态决策与受控评价，D4 提供 Capsule / Broker / Resident / Shadow 工程。D2 归档明示其来源核验范围；D3/D4 阶段文档中的受控验收或准备状态不能升级为完整正式实验完成。
- **Terminal / Probe**：[当前导航](research/phase2-current-status.md)记录用户交接报告已完成 Terminal estimator fit 和 Formal Probe ITT；总览沿用该证据层级，未重新核验正式产物。**Router-v1** 的 hook 与游戏级实验基础设施已实现，正式游戏实验完成仍待相应证据；当前不能统称为完整的最优三支策略规划器。
- **历史方案与原型**：早期三支方法、旧 Baseline 执行／资格流程和规划消融保留追溯价值。草案、synthetic / CPU 验收和代码集成均不能单独证明真实模型效果或胜率提升。

## 5. 重要信息与方法边界

1. **公共 PRE、合法私有视角、Gold 分开。** ToM 与 Baseline 的公共输入不包含角色真值、监督或未来事件。游戏角色可使用其实际合法收到的私有观察，但不能从 Role Sidecar 或环境完整角色表补造策略视角。模拟器执行规则所需真值不等于 Agent 可见信息。
2. **OOF 与 Final Fit 分开。** OOF 用于 Development held-out 评价及相应离线消费者；Final Fit 预测器用于其合同允许的合法 PRE 推断。二者不替换来源，也不把在线策略结果当作 OOF 认知评价。
3. **V1 概率与 V2 集合分开。** Baseline V1 输出非自身概率分布，V2 直接输出怀疑集合并以集合端点为主。原始空集与全六席集合不能从相同的均匀质量反推；解析、失败处理和补充概率评价分别服从原协议。
4. **Q、事件概率与 Mapper score 分开。** Q 是 observer-specific 怀疑质量；Observer-DTRS 概率针对其定义的事件；Phase-2 p_tilde 是社会状态 score，没有外部 calibration，不能解释为行动成功概率。
5. **静态风险与序贯规划分开。** 显式 LossSpec 下的风险比较不证明已识别真实行动损失。Terminal reduced-form risk 不是经典六损失；固定 PROBE→REDIRECT regime 与 Immediate Redirect 的比较也不是全局价值最优规划。
6. **语言执行与政策效果分开。** Actor 实现预定行动，Parser / Verifier 验证表达；合规发言不保证投票改变或胜率提高。语言失败及不依从按所属实验保留，不改成成功执行。
7. **日终后果与完整 GAME_RESULT 分开。** Terminal / Probe 的日终终点不能替代整局策略评价；相同 seed 不保证相同 PRE，现有 Router 合同不提供严格 paired counterfactual replay。
8. **冻结 Source 与开发 HEAD 分开。** 旧实验继续绑定原始 Source、协议、路径和摘要。后续入口、重构或总览不重签旧 Artifact，也不保证当前分析入口直接兼容旧 Run 格式。

## 6. 权威文档导航

- **科学定义与 Phase-1 范围**：[CONTEXT](../CONTEXT.md)、[Phase-1 架构](architecture.md)、[Phase-1 模块职责](repository_structure.md)。后两者不是完整的后续模块清单；CONTEXT 中关于 gameplay 不属范围的表述按 Phase-1 语境理解，不用来否定后续 opt-in 研究。
- **数据与认知建模**：[采集合同](collection_contract.md)、[ToM 合同](tom_contract.md)、[Qwen3 Final Fit](qwen3-final-fit.md)。后续 Backbone 和 paired 研究仍遵守各自冻结合同，不能把某一图或训练参数推广到全部模型。
- **当前 Baseline 操作**：[统一操作说明](operations/tom-llm-unified.md)；该说明区分当前执行合同与历史实验，不授予旧 Run 新身份。
- **Phase-2 当前实现与策略**：[当前导航](research/phase2-current-status.md)、[Router-v1 合同](research/phase2-router-v1-contract.md)。Terminal、Probe 及各模块的专用协议见模块表。
- **开发与历史来源**：[项目协作入口](../AGENTS.md)、[开发规范](development-guidelines.md)、[分支历史快照](BRANCHES.md)。本总览不替代这些约束。

历史文档继续保留原有实验、方法、失败和证据范围；总览不覆盖或改写它们。阅读状态时区分文档的快照日期、用户交接与实际封存证据；缺少最新证据的事项保留待确认，不据旧草案推定当前结论。
