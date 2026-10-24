# 上下文损失与发言衔接：最小研究原型 v1

状态：**RESEARCH PROTOTYPE / NOT A FROZEN EXPERIMENT**。
开发基础：Mainline `6ac38ec15108b12c97f2c3a6eb731bdb82430bdf`。
D1–D3 来源前缀：`7b51f276b04b386ec1aa4ae44454c1d809a10897`。
原 D2/D3 配置、45 个静态损失情景、来源 guard、artifact/schema 均保持不变。
本模块不接入默认游戏、采集、训练、OOF、Router 或正式评估。

## 事件、输入与估计

沿用 D1 的事件：`Y_j = 1[观察者 j 的 effective suspicion support 与存活狼队相交]`。
这不是“j 是狼人”，不是胜率、投票成功率，也不是 ToM 对任何席位的原始怀疑概率。
观察者必须存活且不在合法狼队中；发言者为存活狼人。

`propose_speech` 只接收：当前 authoritative PRE、发言狼人合法知道的两个狼队席位、
明确选择的一个观察者、与 game/PRE/prefix/observer 对齐的 ProbabilityRow/Matrix、
既有 FittedModel + FrozenProtocol，以及显式 TiePolicy。
狼队来自该狼人的合法观察；调用者不能从 sidecar、环境全角色表或 Test 标签构造它。
入口不读取真实数据、隐藏角色、报告标签或未来事件，不拟合模型、不调用 ToM 推理。
当前游戏若出现在估计器训练 game_sizes 中即拒绝。

路径复用 `contexts_for_pre -> observer_features -> validate_model -> FittedModel.predict_detail`。
只有该 D2 估计器输出的 **事件估计概率**进入 `decide`；原始 Q 仅进入合法条件特征。
现有 `s/m/r/dW/phase/nW` 定义不变。Q 数值和身份合法不等于 ToM 来源/模型 seal 已验收；
本接口也不提供正式模型文件的 ownership、分区或 backbone admission。正式接入须复用原
artifact/source 校验入口，不能拿此内存原型冒充已冻结实验。概率校准/识别误差仍须实证检查。

## 上下文损失（新规范性假设）

令 `u=1[phase=pk_discussion]`、`v=1[只剩一名存活狼人]`，均在当前 PRE 合法可知。
取 `c_fp=1`、`c_fn=2^(u+v)`，普通讨论 `k=1/4`，PK `k=2/5`。
令 `b0=k*c_fp`、`b1=k*c_fn`。所有成本均为任意认知效用单位，不声称是游戏收益。
漏判成本指把实际 Y=1 的观察者判为 NEG 的认知代价；误报成本指 Y=0 却判 POS。
BND 的成本是一阶保留判断的规范性成本，不是未来信息增益或动态规划的续期价值。

| 认知域 | Y=1 | Y=0 |
|---|---:|---:|
| POS：判定存在该事件 | 0 | c_fp |
| BND：保留判断 | b1 | b0 |
| NEG：判定该事件不存在 | c_fn | 0 |

复用原 `conditional_risks/thresholds/decide`，不重写公式：

- `R_POS=(1-p)*c_fp`
- `R_BND=p*b1+(1-p)*b0`
- `R_NEG=p*c_fn`
- `alpha=(c_fp-b0)/(c_fp-b0+b1)`
- `beta=b0/(b0+c_fn-b1)`
- `gamma=c_fp/(c_fp+c_fn)`

因为 `b0/c_fp+b1/c_fn=2*k<1`，四种状态均非退化，`0<beta<gamma<alpha<1`。
阈值是风险交点；最终始终用原 `decide` 精确风险最小化和显式 tie order，而非阈值近似。
若 p 是真实条件事件概率，这些是相应**单步认知损失**的 Bayes 风险；使用估计 p 时，
能保证的是所提供风险的 argmin，不能据此宣称真实 Bayes 最优、序贯最优或游戏策略最优。

| phase | 活狼数 | c_fn | beta | alpha |
|---|---:|---:|---:|---:|
| discussion | 2 | 1 | 1/4 | 3/4 |
| discussion | 1 | 2 | 1/7 | 3/5 |
| pk_discussion | 2 | 2 | 1/4 | 3/7 |
| pk_discussion | 1 | 4 | 1/7 | 3/11 |

“动态”在此仅指随当前可见状态重新计算的 memoryless 上下文规则。没有另训动态损失模型，
没有用标签选择成本，没有按累计 reward 或历史评估结果自适应参数。倍数 2、k 的取值及其
研究意义均是**暂定假设**，需之后另定协议、对照和评价；本轮不把它们写入冻结 D3。
规则只使用 phase 和活狼数，不使用 day、turn/step、游戏 ID、source/artifact digest 或路径。

## 三域到发言：独立策略假设

认知域不直接等于攻击、投票、Probe 或对目标角色的判断。
原型复用 `SpeechPlan`、`generate_candidates` 和既有 Actor/parser 传输：

| 认知域 | 原型意图 | 规则与限制 |
|---|---|---|
| POS | SELF_DEFEND | 宣称自己非狼的策略性发言；是允许的欺骗性公开陈述，不是模型证实的真相 |
| BND | OPPOSE(j) | 针对 j 当日已有公开发言的依据提出质疑；若无合法当日发言则 NO_COMMITMENT，认知域仍为 BND |
| NEG | NO_COMMITMENT | 不额外承诺身份/查验/技能/投票；不把低事件概率解释为 j 是好人或不具威胁 |

E 只表示怀疑支持命中某一存活狼，无法区分怀疑哪只狼。统一使用自我辩护只是一个可检验的
策略提议，并非已证明正确的响应。对 BND 的公开质疑也可能适得其反，不能当作信息增益。
未增加“请求后必须回应”的规则，不保证 j 有后续发言机会，不调用正式 Phase-2 PROBE。

`generate_candidates` 从已验证公共历史检查循环发言顺序及真实 PK 平票；仅存活玩家发言，
PK 发言者必须在平票候选中。已经达到狼人数不少于非狼人数的终局被拒绝。
PK 中可讨论当日曾发言的非 PK 玩家，但不强迫其回话。三种意图均不产生任何选票、夜间行动、
放逐、技能结果或环境状态变更；没有绕开普通/PK 目标限制的投票指令。

`ContextualSpeechProposal.audit_record()` 含合法私有狼队知识、特征、事件概率、损失及模型摘要，
**只能留在研究私有审计中**。给 Actor 的 `public_payload()` 仅为 action/target 两字段。
`public_context` 仅含当前发言者/天数/阶段及 PRE 中当日真实公开发言；不含狼队、概率或阈值。
公开原话本身是未可信的玩家陈述，不升级为环境事实。

## 合成 CPU 闭环与下一步

新测试使用既有 PublicWitness、合成训练行和真实 D2 `fit_model`（小型内存统计拟合），验证：
独立精确风险 oracle、四状态阈值、tie、同 p 跨 phase 换域、Q/估计隔离、身份/训练游戏拒绝、
PK 与末位发言、无依据时保留、输入不变性及禁止隐藏/未来字段。
语言闭环调用真实 `GPTAgent._generate_public_speech`、SpeechPerceiver 和 `verify_semantics`，
仅 backend 用脚本化自然语言及解析结果。它验证调用/语义运输和拒绝行为，**不是 LLM 效果测试**。

```python
proposal = propose_speech(pre, known_wolf_ids=legal_team, observer_id=selected_observer,
                          prediction=validated_q, model=validated_d2_model,
                          protocol=frozen_d2_protocol, tie_policy=explicit_policy)
result = realize_parse_verify(proposal.public_payload(), proposal.public_context,
                              actor=actor, perceiver=perceiver)
```

下一轮可在独立、明确 opt-in 的研究入口衔接真实 PRE：沿用已验收 ToM 源码/模型加载与分区守卫、
从发言狼人自己的合法 observation 获取队友、明确单观察者选择/多观察者聚合规则，然后给现有
`realize_parse_verify` 注入真实 Actor/perceiver。必须保留独立解析和语义不匹配拒绝。
若需真正提交游戏发言，应另行审阅 `scripts/verified_speech_commit.py` 的 pending PRE、
perceiver 身份、verified envelope、一次性消费及失败原子性；当前 proposal/result 不是提交许可，
不要直接 `env.step`，也不要修改默认 canonical 数据采集或旧实验路径。

待实证：估计概率在新上下文的校准、上下文成本优于静态成本与否、三域到意图的映射是否有效、
真实语言实现的语义一致率、公开信息变化、游戏收益与潜在反作用。

仓库提供可直接执行的合成演示入口（stdout 输出三域 private audit，全部身份均为内置合成值）：

```bash
python -B -m scripts.check_observer_dtrs_contextual_policy
```

该命令只用 30 个合成训练行进行既有 D2 内存统计拟合，再遍历三种合成 Q，验证
NEG/BND/POS 到现有意图运输。它不接受输入路径、不写产物、不运行网络或游戏。
入口没有正式输入选项，因此不会意外消费真实数据；不要将输出作为正式研究结果。

## 第二轮：证据选择与实际 verified speech 接通

新增 `speech_runtime.py` 为显式 opt-in 调用点，版本
`observer_dtrs_evidence_speech_research_v2`。第一轮的 `propose_speech` 和合成演示继续保留，
不作为第二轮在线准入入口。两者现在共用 `estimate_event`，只维护一处 Q 特征 -> D2 事件
估计 -> 原 D3 `decide` 的计算。第一轮 V1 固定映射不参与第二轮运行时策略选择。

### 观察者与合法信息

`execute_pending_speech` 只读取 `env.get_observation()` 当前发言者的过滤视图：必须为当前
狼人自己的 identity/observer/current_actor、相同日阶段和公开存活集合，且恰有一条合法
`werewolf_team_info`。不读取 `env.roles`、其他玩家的私有 observation、Role Sidecar 或 Y。
`contexts_for_pre` 继续限定为存活非狼观察者，拒绝非狼人行动者及终局。

`select_observer` 不接收 Q/p。只看当前 PRE、当日每位候选观察者**最后一次**公开发言的
既有 V1 annotation，按以下优先级选择，组内取最新事件，最后用 canonical seat 打破平手：

1. 明确 `point_as_werewolf(当前行动狼人)`。
2. 明确 `point_as_werewolf(其他存活狼人)`。
3. 其他成功解释的当日公开发言。
4. 有当日原话、但 V1 不成功或缺失。
5. 无当日发言：首个 canonical seat，标记 `CANONICAL_SEAT_NO_EVIDENCE`。

同一发言对同一席位同时指狼与非狼时，不把这个席位作为明确指狼方向。较新的清白表态不会
被该玩家较早的指狼发言覆盖；不复活过去天数的指控，不从原话重新解析/猜测意图。
选择保留 event_id、annotation_digest、reason、relation。公开指控也不等于真实认知 Y。
只对被选中的那个观察者调用条件估计，不平均 Q 或事件概率，不按预测概率排序挑人。

### 可配置策略假设

`SpeechStrategy` 显式配置 positive_self / positive_team / boundary / negative 四个分支。
允许的操作限于已有 SELF_DEFEND / OPPOSE / NO_COMMITMENT；SELF_DEFEND 仅允许配置在
POS 且有明确指向当前行动狼人的分支。其他分支不接受它，不新增指狼/清白/技能/投票命题。
OPPOSE 指向所选观察者的公开说法，仍须满足现有 `generate_candidates` 资格。
没有明确指向存活狼队的公开指控时，无论 POS/BND/NEG 都输出 NO_COMMITMENT。
这是可解释的策略 abstention，不把未知/缺失输入填成任意角色判断。

例如可以声明：POS-self -> 自辩、POS-team -> 质疑该观察者、BND -> 保留或质疑（显式选项）、
NEG -> 保留。事件 Y 命中任一存活狼，因此仅凭 p 或 POS **不能**推断当前行动狼被怀疑。
上述意图映射仍是研究假设；公开证据只能支持其输入合法性，不证明策略有效。

### 预测来源与尚未开放的正式边界

运行时拒绝裸 ProbabilityMatrix、任意 duck-typed predictor 或只有一个数值合法 Q 的输入。
只接受两种明确来源：

- `SyntheticPrediction`：矩阵身份绑定当前 PRE，审计固定 `SYNTHETIC_TEST_ONLY`。
- `SealedQwen3Source`：必须是原 `SealedQwen3Predictor` 实例，显式 fit/seal SHA-256 pins；
  原 `_source_gate`、`verify_seal`、`log_probabilities` 保持原样，先验证 source/runtime 与
  完整 seal，再使用原 PRE/容量/simplex 校验；不绕过、mock 或接受另一套摘要。

当前 W3 与历史 Qwen3 source 不同且有未提交开发，因此**不能据此宣称原冻结模型已在线接通**。
既有原 verifier 还会检查恢复链和原运行环境；本轮不加载真实 checkpoint，不执行完整恢复或
任何 GPU 路径。缺少合格来源时抛错，不把合成 Q 冒充真实模型输出，也不自动转入合成模式。
真实冻结 source 与当前研究调用端的边界合同仍需独立验收。

D2 运行时目前只允许 `controlled_fixture` 内存模型。仅有形式合法模型 digest，不能证明
formal OOF 估计器适用于新的 all-development final-Q 分布；正式模型的 online transfer
需另审来源及兼容合同，本轮拒绝它，不打开真实监督数据。第一轮通用内存验证并未被弱化。
每条私有审计保存 Q matrix digest、模型 digest/level、PRE 身份和明确来源 scope。

### 三种比较条件

`ResearchPolicy` 显式提供一个注册 D3 静态 LossSpec，以及 reference/contextual 两份策略配置。
静态成本必须属于既有 45 项注册情景，tie policy 为 D3 已有 primary 或 sensitivity 定义。
不新增或修改冻结 D3 情景、实验 schema 或成本选择规则。

| 条件 | 认知损失 | 发言意图映射 |
|---|---|---|
| static_d3 | 显式注册静态成本 | 同一 reference_strategy |
| contextual_loss | 第一轮上下文成本 | 同一 reference_strategy |
| contextual_loss_and_speech | 同上 | 显式 contextual_strategy |

三组共用观察者选择、同一个 Q 和 D2 估计器。前两组只改变损失规则，后两组只改变策略配置。
条件名、完整两份策略/静态成本、实际使用损失和策略、p/三域结果、证据和 payload 均进入私有审计。
reference 是研究层的公共传输对照，不是追溯声称冻结 D3 本来就执行了发言。

比较成本策略时不能直接用各自成本下的数值风险声称提升；应另定共同参考损失或共同任务结果。
配置和输入相同的情形允许产生相同意图，不保证三组每次输出都不同，也不强迫造出效果。
当前条件是软件控制入口，尚不是已冻结、已分配或已执行的正式实验。

### 实际执行路径与验证

```text
已存在的 recorder pending PRE / handoff
 -> 当前 acting wolf 的合法 observation
 -> select_observer（公共 V1 证据）
 -> 显式预测来源验证
 -> estimate_event（Q 条件特征 -> D2 -> 原 D3 风险最小化）
 -> strategy_plan（配置 + 指控方向 + 合法候选）
 -> 原 realize_and_commit
 -> 原 GPTAgent -> SpeechPerceiver -> verify_semantics
 -> bind_verified_speech -> recorder.commit_verified_speech
 -> env.step_verified_speech
```

没有新增语言框架、环境 step 替代或隐式失败 fallback。只提交 action/target，私有选择、模型、
概率、阈值和配置不发给 Actor，也不进入 canonical public events。来源与 pending PRE 错配在
语言调用前拒绝；语义失败沿用既有原子性；成功后 pending 被消费，同一机会不能重复提交。

新测试从合成真实环境和 recorder 获取 PRE，使用实际 Actor/Parser 类、真实 call audit 和
verified 提交路径。backend 仅返回受控自然语言及语义响应；不启动真实 LLM、GPU、服务器
或完整游戏实验。分别覆盖 self/team 指控、normal/PK、没有证据、拒绝来源、三组条件、
非平均估计、语义失败不变性与单次消费。第一轮 41 项测试继续保留。

下一轮最小实验先采用同一组合成 PRE/估计器，交叉 normal/PK、指控方向与 p 跨阈值状态，
对三组条件成对运行，记录共同参考风险、域/意图变化、合法率、解析一致率及提交原子性。
真实语言遵从、信息增益和胜率仍未验证；真实冻结模型接通须先解决上文来源与 D2 转移准入，
不能因为这个 CPU 闭环通过就进入正式研究或默认 canonical 采集。

## 第三轮：固定合成场景的三条件配对比较

入口：`python -B -m scripts.compare_observer_dtrs_synthetic --output /private/tmp/uns-paired-speech-new-run`。
必须使用含项目 CPU/dev 依赖的仓库 checkout；输出为一个**此前不存在**、canonical absolute、
位于 Git checkout 外的目录。脚本不接受外部 PRE/Q/标签/模型/语言服务输入，不是正式实验 runner。
它复用第二轮 `tests/observer_dtrs/test_speech_runtime.py` 的真实环境、PRE、handoff 和 recorder
夹具（合成角色只用于环境初始化），不另造游戏执行器。该 one-speech fixture 不用于发布完整
canonical game，不把 step 的终止标记当作真实对局胜负。测试夹具本轮只增加 target=None 的
中性发言分支，既有故障、来源拒绝和阶段验证保持不变。

### 运行前声明的计划

完整 Cartesian grid：2 阶段（普通讨论、真实平票后 PK）×3 公开证据方向（当前行动狼被指控、
队友被指控、无明确指控）×5 个合成 Q 的 r 值（.1/.3/.5/.7/.9），共 **30 场景×3 条件=90 行**。
固定为两只存活狼；一狼情形仍由既有单元测试覆盖，本轮配对结果不外推至一狼或完整对局。
普通讨论选取有证据的 player7，PK 选取 player3；选择来自真实 PRE 中已有 V1 annotation，
不从 r/p 决定，不在观察者间平均。没有指向狼队的明确公开指控时三条件均保留 NO_COMMITMENT。

D2 模型只构造一次，固定 OBS-EQ1 / 5 bins / smoothing=4：每阶段每 bin 12 个不同的一行
合成训练 game；正例数分别为 0/2/6/10/12，父概率为 1/2。真实既有 D2 拟合和验证接口输出
p=1/8、1/4、1/2、3/4、7/8。训练 game 与运行 game 分离，Y 只存在于合成拟合夹具中，
不进入运行时或语言提示。p=.5 与 r=.5 恰好相同不意味着使用原始 Q 代替条件事件估计。
这些人为概率不具备真实数据校准或因果识别证据。

三条件使用第二轮同一 ResearchPolicy：

| 条件 | 成本 | 意图映射 |
|---|---|---|
| static_d3 | 原 D3 scenario 22：c_fp=c_fn=1/2，b0=b1=1/8 | reference：POS-self 自辩；其余保留 |
| contextual_loss | 第二轮 contextual_loss；tie order 不变 | 同一 reference |
| contextual_loss_and_speech | 同一 contextual_loss | evidence：POS-self 自辩、POS-team 质疑、BND 质疑、NEG 保留 |

所有情况下均经过原 strategy_plan 的公开证据与资格 gate。NEG-first terminal tie order 固定。
运行前写出 `plan.json`（全部场景、配置、共同成本、源文件摘要、model/protocol digest），再执行
全部条件；每场景对齐 PRE/prefix、Q、所选观察者及事件、模型、完整 policy 和环境 RNG digest。
有失败则保留行、固定分母，整轮 FAIL，不用成功子集计算平均风险或配对差异。

### 原路径与记录

`synthetic PRE/Q -> execute_pending_speech -> select_observer -> observer_features -> D2 ->
原 D3 decide -> strategy_plan -> realize_and_commit -> GPTAgent -> SpeechPerceiver ->
verify_semantics -> bind_verified_speech -> recorder.commit_verified_speech`。

生成 backend 只把已有公开意图块变为三种固定中文表达；独立解析 backend 只读原话，并按一个
明确的小型语法返回三元组，不读取计划、条件、p 或预期结果。它们仍经过真实 Actor、parser、
call audit、语义一致性校验及一次性 verified commit。100% 只验证受控语法下的接口，不证明真实
LLM 生成/理解能力。后台不联网，无 GPU，没有把生成结果作为下一个场景的输入。

`rows.jsonl` 保存每条件 private audit：p、成本、阈值、风险、认知域、意图、公共原话/annotation、
选择依据、解析结果、commit 状态与来源摘要。`results.csv` 为 90 行展开表；`summary.json` 记录
域/意图分布、完整性、配对 flips、转移矩阵和合法提交/解析一致率；`synthetic-model.json` 保存
合成拟合载荷。`execution.json` 的绝对路径只用于本机执行审计，不参与科学 plan identity。
所有文件有 SHA-256 清单，已有输出目录拒绝覆盖。

p=1/4、3/4 是实际 D2 浮点输出可精确表示的阈值 tie。PK contextual alpha=3/7 无法由浮点 p
精确等于，因此执行记录只覆盖其两侧，**不 snap/round 概率**。`rational-ties.json` 另用原
`decide(Fraction)` 检查各阈值 below/tie/above（±1/1024），固定标记
`RATIONAL_D3_ONLY_NOT_D2_OR_GAMEPLAY`。这不是额外三条件场景，不计入90次提交或成功率。

### 结果及解释边界

在预先共同声明的静态 scenario 22 成本下，统计每场景所选域的
`conditional_risks(p, reference_loss)[region]`。这只是 plug-in 单步**认知风险**，不包含发言成本、
后续投票、信息增益或胜负效用。给定同一 p，static_d3 由定义最小化该成本；其优势不可被用来
证明或反驳动态策略的真实游戏收益。也不跨条件比较使用不同成本量纲的自身最小风险。

| 条件 | NEG/BND/POS | NO_COMMITMENT/SELF_DEFEND/OPPOSE | 共同参考平均风险 | 合法提交/解析一致 |
|---|---|---|---:|---|
| static_d3 | 12/6/12 | 26/4/0 | 1/10 | 30/30；100% |
| contextual_loss | 12/3/15 | 25/5/0 | 9/80 | 30/30；100% |
| contextual_loss_and_speech | 12/3/15 | 18/5/7 | 9/80 | 30/30；100% |

static→context：3/30 换域（均 PK p=.5 的 BND→POS），1/30 意图变化，平均参考风险差 +1/80。
context→context+speech：0/30 换域，7/30 意图变化，参考风险差0。
static→context+speech：3/30 换域、8/30 意图变化。普通讨论成本仅正比例缩放，不改变域；
无指控三条件均保留。没有为制造差异强制改变任何同条件结果，没有筛除不利或相同场景。

上述结果支持“接口能隔离成本和意图映射两个因素，并稳定执行”的工程假设；不支持“动态成本
更好”“质疑带来信息增益”“胜率提高”。下一步若继续研究，应先独立审阅 formal D2 在线迁移与
真实 frozen ToM admission，再在同一合法 PRE 中让真实 Actor/parser 检验发言语义与拒绝行为。
随后才可能通过预先定义的实际对局指标验证成本假设、映射假设及潜在负作用；当前不启动该实验。
