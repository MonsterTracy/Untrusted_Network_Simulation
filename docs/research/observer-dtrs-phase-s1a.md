# Phase-S1a：认知证据 G 与候选发言选择 π

状态：**受控研究接口，未冻结正式策略实验**。基线 `twd/mainline`：
`6249485c56cc3a5107a0b0de2252de3435421b45`。
本阶段不改变 D1–D3、45 个成本场景、原有 V2 映射假设或正式来源准入。

## 接口与调用顺序

```text
现有 pending PRE + acting wolf 的合法 observation
  → 既有公共证据观察者选择、ToM source admission
  → 原 observer_features / D2 predict_detail / D3 decide
  → G: process_cognition(PRE, panel=tuple[ObserverCognition])
  → π: select_speech(PRE, CognitiveEvidence, candidates, strategy, mode)
  → SpeechPlan.public_payload()
  → 原 Actor / Parser / semantic verification / Verified Commit
```

实现：`werewolf/observer_dtrs/speech_evidence.py`。G、π 无环境、agent、recorder、
文件 reader 或语言调用句柄；不接受 Y、真实报告、sidecar、未来事件、训练支持或旧 Mapper。
调用者仍必须使用经验证来源生成 panel。G 中的 model digest 是绑定/审计引用，
**不是部署模型 admission 或密码学 provenance 证明**；不替代原 Q/估计器的读取验证。

`ObserverCognition` 只含 `ObserverContext / ObserverFeatures / model_digest /
estimator_level / LossSpec / Decision`。检查上下文属性、特征数值和原 D3 风险/概率/成本绑定。
G 重新核对 PRE、合法观察者、canonical 席位顺序、重复与跨 PRE/模型/成本错配；
返回不可变 `CognitiveEvidence`：已处理各观察者、资格域、公开选取依据和 panel 覆盖类型。
公开依据为 V1 event ID、annotation digest、指控方向与优先级，均可追溯到冻结 PRE。

G 的认知处理依据为：POS 可考虑“有效嫌疑集合命中活狼”的命题，BND 保留两种命题可能，
NEG 不采用该命题作为前提。它们不等于事实真相、对方身份或任何发言动作。
G 不决定 SpeechPlan，也不依据 Q/p/风险对观察者排序或取平均。

完整 panel 可处理全体合法非狼观察者；V2 runtime 继续只估计公共规则已选出的一个观察者，
明确记录 `EXPLICIT_SUBSET`，**不冒称全体面板**。子集必须包含同一公共规则选中的观察者；
缺失即报错，不因另一个观察者 p 更高而替换。全体面板的融合/聚合规则尚未设计。

π 要求完整、原顺序的 `generate_candidates(PRE)` tuple 和显式 `SpeechStrategy`。
拒绝删减、重排、追加候选或修改 G 的公开依据。记录完整候选及摘要、G 摘要、模式、
策略配置、认知分支、选中的原 SpeechPlan 和理由。不存在新的行动枚举或语言框架。
旧 `speech_runtime` 的 `ObserverSelection / SpeechStrategy / select_observer / strategy_plan`
导入与调用接口继续兼容。

## 模式和原型行为

`EXPLICIT_THREE_WAY` 使用原 D3 region；`CONTINUOUS_RISK` 使用同一已计算三项原始风险，
按同一显式 TiePolicy 取精确 argmin。两者在 π 选择依据上区分，但**数学上等价**。
G 的完整性校验仍核对原 D3 decision，不允许伪造 region。没有重写阈值公式、裁剪概率、
自创风险加权或把认知风险当成候选发言收益。

π 的 region/risk 分支仅查显式策略配置；相同 POS 可以配置自辩、质疑或不承诺。
缺少指向狼队的明确公开指控时仍保留 V2 的 NO_COMMITMENT gate。
这是既有、可检验的发言映射假设，不是三域的固有行动含义。
对其他观察者的 p 变化不改变既有公共观察者选择。

runtime 默认模式为显式三支，原三个 `Comparison` 条件不增加、不改名。
仅新增 `selection_mode` 关键字参数及私有审计 `cognitive_processing / candidate_selection`；
审计 schema 版本更新为 `observer_dtrs_evidence_speech_research_v3_s1a`。
Actor 仍只收到原 `action/target` 公开意图，G、p、风险、成本、模型身份不进入 prompt。
原 sealed source、受控估计器 scope、当前游戏不在训练集等 guard 原样保留。
未修改 `contextual_policy.py`、`scripts/speech_planning.py` 或原 verified commit 链。

## 受控对照与已执行结果

入口（在含 CPU/dev 依赖的本地主线 checkout 中运行；输出目录必须不存在且位于 Git 外）：

```bash
TMPDIR=/private/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B -m scripts.compare_observer_dtrs_s1a \
  --output /private/tmp/observer-s1a-new-run
```

复用原受控 30 场景：普通讨论/PK × self/team/无公开指控 × 五个固定 r。
合成 D2 训练夹具一次生成、与运行游戏隔离；p 为 `1/8,1/4,1/2,3/4,7/8`，并非直接取 r。
新四组共用同一 PRE、Q、p、完整候选、公开选中观察者与 EVIDENCE 策略；
每场景共享一次 Q、一次 p 估计，四次纯选择，无语言调用和动作提交。
这四组是**纯选择比较**，不是额外 120 次游戏闭环；闭环回归另使用原 90 行样例。

| 成本 | 表示 | NEG/BND/POS | NO_COMMITMENT/SELF_DEFEND/OPPOSE |
|---|---|---|---|
| 原 D3 scenario 22 静态成本 | 显式三支 | 12/6/12 | 18/4/8 |
| 同上 | 连续认知风险 | 12/6/12 | 18/4/8 |
| 既有上下文成本 | 显式三支 | 12/3/15 | 18/5/7 |
| 同上 | 连续认知风险 | 12/3/15 | 18/5/7 |

30×4=120 行完整配对，成本变化各导致 3/30 换域、1/30 换意图；
相同成本下连续风险与显式三支均为 0/30 换域、0/30 换意图。
无差异是等价公式的预期结果，不能解释为“三支无用”或“三支产生额外收益”。
不同成本下的自身风险也不可直接当共同游戏效用比较。

旧三条件的 90 行在拆分前后逐字段一致（排除审计版本和两个新增字段）；
原 79 项测试通过。新测试覆盖全体/子集、单狼/双狼、死亡玩家、PK 平票资格、
错误 PRE/模型/成本/候选、监督反事实、无文件读取、RNG、全部 45 成本场景的
精确端点与三种 TiePolicy，以及原 Actor/Parser/一次性提交。
独立脚本发布 plan/rows/summary/manifest；未完成或重复网格直接失败，已有输出拒绝覆盖。
最终针对性与相关回归共 **295 项通过**（含本轮新增 23 项，15.62 秒）；
执行环境为 Mac 本地 `anaconda3/envs/3wd` 的 CPU Python。
完整命令、JUnit、源码差异与最终保护校验另存本轮本地验收包。

## 边界和待解决问题

连续认知 argmin 不是独立的“无三支优化算法”；本阶段没有可识别的三支表示增量效果。
如以后要比较连续风险如何影响语言细节或候选效用，需要另订可证伪协议，不能在此临时造公式。
成本和意图映射仍为研究假设，未依据真实标签或 D3 最优场景选择；没有胜率、语言有效性、
信息增益、真实校准或 CAL/TEST 证据。没有启动真实模型或新游戏采集。

runtime 仍仅允许 controlled_fixture 概率估计器；正式 OOF 不能直接迁至 final ToM。
完整面板的真实推理预算、CAL 概率包来源及策略效用验证不在本轮完成。
纯接口的“无读取”是受控调用证据，不是 OS 权限沙箱或正式部署资格。
本阶段保持未提交，须研究者审阅后再决定是否冻结或进入后续实验。
