# Observer-DTRS v1 D4a：在线接口与 Shadow 推理设计

日期：2026-10-17。状态：**DESIGN_REVIEW_READY_NOT_IMPLEMENTED_NOT_FROZEN**。
审阅源码：`three-way` / `e513fd8b69404d07d69cd308023b673554f5bbf0`。
本轮仅新增本报告和来源核对 JSON，没有修改生产代码、训练概率模型、创建游戏或访问服务器。

## 1. 结论与证据范围

**合法输入已经具备；纯特征/概率/六损失计算可以组合；正式在线部署链尚未形成。**
阻碍集中在三处：运行循环没有无反馈 Shadow 接口；D2 没有已封存的在线部署模型选择契约；
已有游戏侧 final ToM 与 D2 OOF ToM 的来源不同，不能靠同为 Qwen3、相同七列形状证明兼容。

| 已检查事实 | 源码或本地材料 | 含义 |
| --- | --- | --- |
| canonical PRE 先冻结，再执行玩家行动 | `run_random.py:90`；`canonical_collection/runtime.py:228` | 可在真实发言前截取不可变输入 |
| PRE 的 step 是终端公共事件索引 | `canonical_collection/runtime.py:262` | 不使用游戏循环的 `step` 作 canonical 身份 |
| 玩家观测只给自己的身份和按 viewer 过滤的日志 | `envs/werewolf_text_env_v0.py:863`、`:963` | 狼队身份可从合法队友消息取得，无需角色 sidecar |
| ToM-only 已有合法观察者枚举和公开 PRE 推理 | `scripts/tom_gameplay_ablation.py:9`、`:53` | 可以借鉴身份核对，现有文字注入 hook 会影响发言，不能当 Shadow |
| 七列 Q → 观察者特征已独立于监督 | `observer_dtrs/features.py:29` | 复用原始质量，不裁剪、不按存活列重归一化 |
| EQ 模型已有只读 `predict` | `observer_dtrs/estimation.py:142` | 算术入口存在，不等于新游戏部署契约存在 |
| D2 外层包内含四模型及预测 | `nested_validation.py:145`；本地 25×2 个 outer_predictions | 200 个已拟合模型，不是一个唯一 final 概率估计器 |
| 纯三支只收概率与 LossSpec | `observer_dtrs/decision.py:184` | 可直接复用，不接收 Y、训练支持、报告或角色真值 |

文中的相对路径均相对于仓库根；行号是本轮审阅基点的位置。完整摘要见配套
`observer-dtrs-v1-d4a-source-review.json`。

用户已确认 D3 正式 `d3-v1-run1` 完成并通过带外部摘要的完整性验证。
本地 `Documents/Untrusted_Network_Simulation-results/d3-v1-run1-summary/` 目前只有
`d3-model-contrasts.csv`、`d3-model-risks.csv`，没有完整 D3 contract/complete/manifest。
因此本轮以用户提供的正式完成状态为研究背景，不声称独立复验 D3 全封存；没有分析 CSV 来挑成本或模型。
D4a 的接口判断不依赖成本场景的输赢。D2/D1c 的完整本地材料可用，本轮抽查其已封存模型结构与来源。

## 2. 发言前真正可用的信息与插入时序

实际调用顺序：

```text
env 当前 actor 的合法 observation
  → recorder.before_agent_act
      → 冻结公共历史（末事件包含当前 speaker 的 turn_start）
      → 构造 AuthoritativePREPrefix
      → 现有 belief_collector 收集报告
      → 返回 speaker 自己的 PRE-belief handoff
  → [未来 opt-in Shadow：只捕获 PRE 与当前狼人合法队友信息]
  → 原有 agent.act_with_pre_speech_belief
  → recorder.after_agent_act → env.step → recorder.after_env_step
```

本轮不添加方括号中的 hook。建议它放在 `run_random.eval` 获得 handoff 之后，
任何 `online_pilot.handle_pre`、constrained speech、ToM 文字注入和 baseline agent action 之前。
**Shadow 必须是另一个观察型 hook，不能用具有提交发言能力的 online_pilot 代替。**
第一版只资格验收 original/baseline 路径；与现有干预路径共用时，捕获位置虽相同，但需要另行验收。

主循环当前通过 `recorder._pending.prefix` 使用 PRE。未来最小集成可增加只返回已冻结 PRE 的公开只读访问器，
由控制层提取后传入 Shadow；不要把 recorder 本身交给 Shadow，也不要让它访问 `_pending` 或 observations。
访问器必须返回同一 PRE，不重建、截断、补齐或重排；D4b 核心可先接受显式 PRE，无需马上改循环。

### 输入白名单

- ToM：只接收 `AuthoritativePREPrefix`，模型 tensorization 为 `tensorize_public_pre`。
  PRE 含报告**身份链接**，不含报告嫌疑集合；这些链接也不成为角色/标签特征。
- 控制层从当前狼人的 delivered observation 核对自己的 `observer_id/current_act_idx`、Werewolf 身份、
  一条且仅一条 `werewolf_team_info`、两个不重复合法座位及包含 actor。
  与 PRE 当前 speaker、alive 集合、day 和 phase 交叉核对。
- DTRS 上下文：公共 PRE + 该狼人合法知道的两个狼队座位；不接收其他玩家 observation。
  不调用其他玩家的 `get_observation_for` 来取其私有知识，不读 `env.roles`、sidecar 或 reports。
- `W_t = known_wolf_ids ∩ alive_player_ids`；仅当前 actor∈W_t 时枚举全部 `A_t\\W_t`。
  PK 也枚举全部存活非狼观察者，不采用 Phase2 的候选目标、redirect、suggestible-exile 筛选。
- 座位固定 `player1...player7`；runtime `speech/speech_pk` 对应 canonical `discussion/pk_discussion`。
  observation.phase 是带日次的字符串，应核对 `authoritative_public_state.phase`，不要直接把它当 canonical phase。

当前 canonical 运行会在此位置之前收集其他人的报告，这是原有数据生成机制，不代表报告可进入 Shadow。
本设计不改变采集次数、报告失败处理、speaker 自己的 handoff 或原有游戏资格规则。
仅拿到角色观察中的原始私有日志还不够：必须复用 canonical PRE，不能把整份 observation 交给 ToM。

## 3. ToM 与概率模型的来源问题

### 3.1 可复用的冻结 ToM

`Qwen3GameplayPredictorClient.predict(prefix)` 已有请求号、prefix 摘要、fit/seal 响应核对；
worker 要求固定、干净的 predictor checkout，并核验 model/terminal。它返回 **7×7 概率质量**。
worker 内部对明确的 `log_probabilities` 取 exp；Shadow 收到 probabilities 后不得再次 exp。

当前客户端固定来源为：

- predictor source：`f199a9e6c64168a412de91ff8ca04d0c795c57e2`；
- final fit：`87c2485acf48443034c59430d79f098a13151677852988bf691599ca2d2f8248`；
- final seal：`fcbb81ee43050142ebdf01c0aee576156221d725598ff5ee2c02055049f72718`。

`qwen3_final.prepare` 的 partition 明确为 `all_development`，与 OOF 五个 checkpoint 不同。
`SealedQwen3Predictor` 只读已封存参数，不需要重训；完整 prefix 受 max_seq_len/day capacity 约束，
现有推理 tensorization 长度为 1024；`validate_final_pre` 超界时报错，不能静默裁切或日次回绕。
worker/权重在 Mac 的可加载性、GPU资源和服务器活跃任务本轮未验证；本轮不启动客户端。
现有 600 秒客户端等待上限不适合直接阻塞 Shadow 游戏路径，需隔离 worker 和队列。

### 3.2 D2 实际已有与尚缺的部分

`FittedModel.predict(features)`、`baselines.predict_many(model, features)` 可在数值上预测任意合法新特征。
但 D2 的部署范围仍是固定来源折下的嵌套验证：每个 source fold 300 局，外层训练240局、测试60局；
2种权重×4模型×25块，共200个外层模型。`selection` 固定该块参数，`outer_predictions.record.json`
把模型与逐样本预测放在同一个包里。不能把整包（含历史样本或间接可连监督字段）传给在线模块。

本轮抽查 block `0-0/game_equal`：EQ1 参数 bins=5、lambda=.25，训练240局，tom_fold=0、outer_fold=0；
模型摘要 `bc1565bcd366dcfd9d61c04efa922666051a344b2e6a0ef891f5b3b06b72793f`。
这只是**来源例证**，不是推荐选择该块部署。其对应 ToM terminal 是
`41a242399c45924f3d7f1f320f3176681731da11a660db9add5f755b6ed10ca1`，而非上面的 final seal。
model.source.origins 同时列出五折，实际来源折须以 source.tom_fold 选择对应项，不能把列表第一项当所有模型来源。

没有发现 Observer-DTRS 的线上模型注册、唯一 final 选参/拟合发布、ToM→p 兼容声明或 Shadow 调用路径。
`estimation_offline.read_model` 是离线文件模块，传递导入包含监督 reader；
`FittedModel(payload)` 本身也不是来源验收入口，须先经结构和摘要验证。
未来应有独立的合法部署 reader，且不导入 dataset_offline/training_audit/nested_validation。

### 3.3 备选路径及推荐

| 路径 | 优点 | 局限/条件 |
| --- | --- | --- |
| 固定已有 D2 某个 outer 模型 + **相同折**冻结 ToM | 不拟合任何新模型，来源最容易追溯 | 必须预先无标签地指定折/块，或完整登记集成规则；现有 final-only worker 不支持该来源，需新的只读 fold provider；权重文件可用性尚未查验 |
| final ToM + 固定 OOF 训练的概率估计器 | 工程接入现有 final worker 最省事 | 属 OOF→final 的分布迁移，不能直接称为已经验证的事件概率；只能显式标注未验证迁移 Shadow，不能默认偷偷放行来源不匹配 |
| final ToM + 独立 calibration/development 游戏上固定 p | 条件属性直接来自部署 ToM，训练/评价对象最清楚 | 需要未来单独冻结校准协议与 CAL/TEST 划分；只训练小型 p，不重训 ToM；本轮不执行 |

**推荐分两步**：D4b 先实现人工 provider/已封存模型的只读提取与 Shadow 核心，使用明确 scope；
进入可信的新游戏概率评价时，优先采用第三条，先冻结 final ToM，再在单独 CAL 游戏上估计与选参 p，
封存部署包后才接触全新的 TEST 游戏。它不要求现在生成 CAL 游戏。
第一条可作不重拟合的参考路径，但与既有 final-ToM gameplay 基线不是同一处理，不能混在同一对照里。

如果不希望新增校准数据，第二条可作为**事先声明的迁移假设**进入纯 Shadow；
应固定旧模型/全体集成规则后用未来独立数据检验迁移，承认 p 校准未知。
不把拒绝此未验证组合解释为算法不可计算，也不把形状正确解释为来源已经验证。
禁止按 D3 风险、胜负或新游戏标签挑“最佳 outer 模型”；25个外层表不能拼成依赖测试游戏/报告的线上 lookup。
若以后用全部历史 OOF 重拟合，属于新 D4 部署协议：不可直接放宽 D2 的240/160局、同来源折约束，
也不可用 final ToM 对其已训练的1500局重算 r 再冒充 OOF。跨来源汇总/选参及迁移身份必须另行登记。

## 4. 小而明确的 Shadow 接口（设计，非已实现接口）

```python
# 在可信控制层做一次白名单提取；Shadow 核心永远不持有 env/agent/recorder。
request = build_shadow_request(prefix, acting_observation,
                               shadow_run_id=..., attempt_id=...)
result = engine.infer(request)  # 内部使用已注入且已验证的 ToM 和概率模型
sink.append(result)            # 独立私有审计；返回值不送入 agent 或 env.step
```

| 接口/对象 | 唯一输入 | 输出及不变量 |
| --- | --- | --- |
| ShadowRequest | 原始不可变公共 PRE、合法 team 白名单、run/attempt 身份 | actor、alive/phase/day、canonical step/boundary/prefix 绑定；无报告、Y、role sidecar |
| VerifiedQ | predictor 返回的概率、请求身份、冻结模型证明 | 经 ProbabilityMatrix 检验的7×7 Q；来源证明与数值对象分开保存 |
| `contexts_for_pre` + `observer_features` | PRE、合法 team；context + VerifiedQ.matrix | 每名存活非狼一个 context；s/m/r/dW/phase/nW；死亡目标质量保留 |
| FrozenEventEstimator.predict | ObserverFeatures | 事件概率 p、模型摘要和来源；不接受上下文报告、Y 或测试分区句柄 |
| `decide(p, loss)` | p + 显式 LossSpec/TiePolicy | 三项精确风险、全部最优行动、选中区域、risk_gap、阈值/距离；不接收训练支持 |
| ShadowResult | 上述合法计算及模型证明 | 绑定所有观察者、场景、并列策略和状态的不可变私有记录；没有 gameplay action 字段 |

Q 的自列必须为0、每行六名非自身质量和约1，绝对座位不能按存活数压缩。
单狼 m=0、dW=None；双狼 r=s+m，dW=|s−m|。EQ1 实际只用 r、phase、nW，s/m/dW保留作合法诊断。
数字和身份错误应拒绝，无静默 clipping、重归一化、均匀 Q、旧 Theta_AC 或其他模型替代。

来源证明至少绑定 ToM source/fit/seal/terminal/checkpoint/temporal/tokenizer-or-token-plan 版本、
prefix/token-plan 摘要、绝对座位、Q摘要、事件模型部署manifest及payload摘要、训练/校准游戏集合摘要、
参数/分箱/权重与选择规则、D0标签定义版本、D3损失配置摘要和新 D4 执行源码 OID。
在线包可引用离线来源摘要，但不得携带角色真值、逐条训练Y或历史测试行。
训练节点计数/支持若保留，存独立审计载荷；预测/decide 接口不接收它们。

`ProbabilityInput` 目前位于 `decision_offline.py`，不为在线使用而移动或修改它。
在线结果使用独立的合法概率记录；只有离线评价 adapter 把它转换为 ProbabilityInput，再单独打开 targets。
原D3完整运行器是受限离线模块，不能直接成为在线 engine。

### 身份与时间联结

沿用 `dataset.sample_id(context)` 的算法：摘要包含 dataset schema、game、boundary、acting_wolf、observer、prefix。
外层日志键为 `(shadow_run_id, attempt_id, sample_id, model_digest, scenario_id, tie_policy)`；
样本身份不混入标签/区域，部署模型更新不冒充同一结果。
在捕获时记录 terminal turn_start/event index、capture 时间；推理完成时间另存。
异步结果可以在原发言之后完成，但只能标记为对应 PRE 的 Shadow 估计，不能回写为“当时已用于决策”。
同一冻结 ToM 身份对一份 PRE 最多算一次 Q，再供各观察者与全部成本复用；
若另行登记多来源折集成，各fold须使用自己的Q，不能把同一Q冒充不同checkpoint输出。
重试幂等按完整身份核对，不能把上一PRE缓存给当前PRE。

### 损失与支持

继承冻结D3的全部45个非退化矩阵及主 TERMINAL_NEG_FIRST、DEFER_FIRST 精确并列敏感性。
Shadow 可完整记录45×2结果，不选唯一主成本、不把全部场景塞入玩家 prompt。
TERMINAL_POS_FIRST 仅保留受控退化诊断，不升为新主策略。
支持/回退是概率表来源诊断；低支持不是 BND。概率/成本扰动仍为外生敏感性，不是置信区间。
没有 p 时记录推理失败/缺失，不能生成 BND 来填空。

## 5. Shadow 如何保持原游戏路径

建议先做“捕获不可变请求 → 独立有界队列/worker → 私有 sink”的旁路。
worker 不共享 agent、环境、游戏 RNG 或带计费的 gameplay backend；不调用聊天/报告接口，不新增采集报告。
如果原路径已有 ToM，只有在来源与请求完全相同且不会改原调用/重试语义时才可只读复用；否则独立计数。

默认关闭。开启时只记录，不改 agent prompt/messages/memory、action、vote、public_events、V1 annotations、
canonical recorder evidence、采集配置/配额、winner 或 speech handoff；不返回“已提交一步”的控制信号。
不借用在线 Pilot 的一次干预配额、候选目标筛选或真实发言提交机制。

启动前验证来源/配置失败：拒绝开启 Shadow，不改游戏配置来迁就来源。
运行期间 worker超时、非法Q、来源错配、队列满和 sink I/O 异常：原游戏继续，Shadow独立标记
FAILED/SKIPPED_BACKPRESSURE 等技术状态，保留尝试与覆盖分母，不能把异常默认为有效分类。
影子审计损坏会使 Shadow 验收失败，但不能将游戏胜负重编码或静默丢掉对应游戏。
如磁盘已无法记录失败，使用控制层独立错误计数/健康状态，不能宣称审计完整。

异步运行仍会消耗资源并可能影响真实 backend 的服务时序；这不是能够承诺“真实生成文本逐字相同”的机制。
受控 fake backend 可验请求/行动/状态逐字一致；生产只能在预先设定资源限额下审计额外延迟和服务影响。
首版可先只捕获请求并在本地离线 replay Shadow，待隔离和预算验收后再接实时 worker；不碰现有 vLLM 服务。

## 6. 新游戏隔离与未来对照

### 6.1 先固定系统，再打开独立评价

历史 DEV（D1–D3）已被反复用于开发，不能重新标成 D4 独立测试。
如果走新 CAL 路径，事先冻结：CAL/TEST 的整局 game/seed pools、ToM身份、p算法/候选网格、
选参准则/并列规则、权重、完整训练数据资格、缺失失败处理和停止规则。
只在 CAL 内按完整游戏做选参/校准；封存最终p包后才运行/打开TEST，TEST标签不能用于重分箱、再校准或挑成本。
历史RNG来源限制仍报告；未来未接触的新游戏可以评价这条固定部署系统，但不能把D2/D3解释为独立证据，
也不能据一次固定系统评价声称重复上游训练/模型选择程序的不确定性已被覆盖。

新评价报告应在受控离线进程中显式联结 PRE 报告与角色 sidecar，复用 D1 标签函数：
成功空仍是六名非自身；死亡狼不计入 W_t；失败/缺失无Y，不填0。
完整 publication 违反报告契约仍报错；未来非完整 Shadow 运行的标签缺失率单列，不能由在线支持筛选掉难例。
原始线上输出须在监督开放前封存；监督验证与线上推理进程隔离，最好以文件权限限制在线进程读取受限目录。
新游戏与历史/CAL 排除身份、seed池重用及整局重放重复；常见的相同开局公共PRE不等于整局泄漏，不能据此任意删样本。

### 6.2 ToM-only、ToM+p、ToM+p+DTRS 检验的是什么

`p=f(Q,合法context)`，`region=d(p,loss,tie)`。固定模型/成本时三域是p的确定变换，
**它不增加关于Y的原始信息**；可能增加的是表示、提示和下游利用的便利性。
Shadow 不给 agent 任何新增信息，故它只能检验合法性/覆盖/概率与认知风险，不能检验胜率增益。

未来独立干预协议可比较：

| Arm | 给同一狼人的额外认知材料 | 主要解释 |
| --- | --- | --- |
| T：ToM-only | 同一版本、同一座位格式的合法 Q | 固定ToM输入基线 |
| P：ToM+p | T + 每名观察者事件概率p | 学得事件概率表示的增量 |
| D：ToM+p+DTRS | P + 在预先指定成本/策略下的区域标签 | 显式三域表示的增量 |

三个arm保持 ToM、p部署版本、观察者集合、语言模型、generation预算和baseline策略一致。
P/D 必须具有相同成本说明与并列规则可见性；否则D比P多的不只是三域表示。
若D额外给三项风险/阈值/支持解释，必须匹配对照信息或明确它是复合处理，不能归因于区域标签本身。
训练支持不作为主要prompt特征，三域不直接翻译成攻击/Probe/忽略。
不能按D3哪项成本效果好选择D的单矩阵；未来要么外部规范性理由预先指定，要么登记成本因素/代表性场景完整比较。

对照随机化/配对单位用整局游戏或同初始seed的完整三臂区组，顺序平衡；不是观察者行。
同seed不保证服务采样轨迹相同，各arm后续PRE会因发言反馈而分叉，不能逐行强行匹配不同轨迹。
技术失败、零合格PRE游戏、缺失报告和资源中止规则预先规定；报告全部分配和失败，不按胜负/区域比例筛局。
真正的增量效应需要后续独立随机化/配对策略实验，而非从D3风险差推断游戏效益。本轮不制定胜率结论或执行该实验。

## 7. 最小模块清单与实施顺序

以下是建议新增路径，不是本轮已实现内容；D0–D3 文件不为迁就部署而改动。

| 阶段/建议文件 | 最小职责 |
| --- | --- |
| D4b `observer_dtrs/online_context.py` | 白名单提取/身份校验、ShadowRequest；组合原 contexts_for_pre/sample_id |
| D4b `observer_dtrs/online_model.py` | 合法只读部署包reader、ToM配对检查、合法p推理；不导入受限reader |
| D4b `observer_dtrs/shadow.py` | 注入 provider，组合 Q→features→p→decide；完整45场景、明确技术失败；不持有env |
| D4b `observer_dtrs/shadow_artifacts.py` | 私有日志、幂等/完整性/覆盖和恢复，复用artifact_io；与canonical目录分离 |
| D4b `tests/observer_dtrs/test_shadow.py` | 人工矩阵/冻结模型fixture与反事实、错配、队列/写入异常验收 |
| D4c `observer_dtrs/deployment_offline.py`、独立部署配置/发布脚本 | 在另行批准协议下，离线抽取既有模型或CAL拟合并封存；不复用整份outer_predictions在线读取 |
| D4c 主循环小型opt-in hook/只读PRE访问器及受控测试 | 经D4b验收后单独审阅，只送不可变请求，不改变旧默认路径；再作获授权的资源受限Shadow试运行 |
| D4c `observer_dtrs/shadow_evaluation_offline.py` | 打开封存Shadow和受限标签做新游戏概率/风险评价；不被线上导入 |

在线部署包至少分 manifest、合法 inference payload、单独训练支持 audit。
抽取D2模型时离线先通过既有外层审计；仅复制参数/节点概率/边界或系数与来源摘要，
不复制训练targets、测试predictions或角色侧文件；编译前后推理在每个分箱边界、端点、回退状态和随机人工特征上等价。
新CAL/refit的模型必须使用新D4 schema与独立数据身份，不伪装成240局 D2 formal_training_partition。
旧 `validate_model`/D2 source guard 的固定人口约束保持不变。

**先后顺序**：D4a 本设计 → D4b 无游戏改动的纯Shadow/合法reader及人工验收 →
D4c 明确模型来源路径、独立冻结部署包与CAL/TEST协议 → 小型hook受控非干预验收 →
用户另行授权后资源检查/新游戏Shadow → 独立离线评价。策略干预与胜率研究后续单独登记。

## 8. 受控验收条件与本轮实际验证

D4b应覆盖：

1. 普通/PK，所有存活非狼、单狼/双狼、任意狼座位、死亡目标列、自列和完整7×7映射；
   runtime loop step故意与canonical step不同仍正确绑定。
2. 仅改变其他人的真实报告、Y、sidecar、成功空标志和训练支持，合法输入相同则p/区域逐字不变；
   上述对象作为infer/decide参数直接拒绝。改变模型训练数据不要求概率不变，勿误用反事实测试。
3. 用“触碰即抛错”的env.roles/其他玩家观测/受限reader替身证明Shadow无法访问；检查在线导入闭包。
4. 非法Q/log-Q、座位错位、跨PRE/游戏/attempt重试、伪造摘要/模型/ToM兼容声明均拒绝；不默认fallback模型。
5. 已验收模型→部署包的全边界推理等价，LR连续标准化复用训练mean/std；拒绝重新分箱/线上拟合。
6. 默认关闭与开启Shadow的fake-backend请求、行动、投票、历史、handoff、agent状态、call budget、
   canonical evidence以及Python/NumPy RNG完全一致；Shadow仅新增自己的私有记录。
7. worker失败、超时、写入失败、队列满、重复请求、异步过期回复、恢复/损坏不会改变baseline；
   覆盖与失败分母完整。超容量PRE报错，不裁切；没有p不生成BND。
8. 若未来生产执行源码有新增，使用真实新OID及独立D4配置验收原guard，不以e513或历史D2 OID替代新源码。
   原D3协议和封存不重写，不为测试修改/绕过生产guard。

本轮实际命令（既有纯计算/人工产物单元测试，没有启动ToM权重、LLM服务或新游戏）：

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/name_yuxiao/anaconda3/envs/3wd/bin/python -B -m pytest -q \
  tests/observer_dtrs/test_features.py tests/observer_dtrs/test_estimation.py \
  tests/observer_dtrs/test_decision.py -p no:cacheprovider --maxfail=2
```

结果：**90 passed in 6.53s**。这是可复用模块的验收，不是尚未编写的Shadow集成测试通过。
本地逐个验证25×2个外层包的manifest/record内容完整性、模型数、来源折和240局训练声明；
没有重拟合/重选参，也没有重算任何正式D3区域或损失。完整D3生产运行完整性本轮未独立重验。

## 9. 进入下一步的条件

**可以开始 D4b Shadow 核心与人工 provider 的代码实现；暂不能宣称真实在线链已经验收。**
不需要先挑一个最优损失矩阵。真正阻挡正式新游戏执行的研究选择只有概率来源路线：
已有同折ToM+outer表、明确未验证迁移，或推荐的final ToM+独立CAL。
CAL预算、最终p选参细节和TEST停止规则须在收集/查看相应评价结果前冻结，而不是在本轮臆定。
后续成本对玩家的呈现方式与干预评估需另行协议，不阻挡不反馈的Shadow核心开发。

研究限制继续保留：历史ToM RNG依赖全量publication摘要，历史开发集已用于后验研究；
固定预测Bootstrap不包含上游重训与下游重新选参；三域是认知判定，尚不能推出游戏动作或胜率改善。
