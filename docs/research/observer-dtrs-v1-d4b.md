# Observer-DTRS v1 D4b：独立 Shadow 核心与受控验收

日期：2026-10-17。状态：实现供阶段审阅，**未提交、未接游戏主循环、未启用正式部署**。
基点 `three-way` / `e513fd8b69404d07d69cd308023b673554f5bbf0`。
实施依据为 `observer-dtrs-v1-d4a-shadow-design.md`；D0目标事件、D2估计公式和D3全部45个成本不变。
实际命令、测试结果、指纹、人工产物位置与保护检查见配套 `observer-dtrs-v1-d4b-acceptance.json`。

## 1. 新增模块与接口

| 文件（相对仓库根） | 职责 |
| --- | --- |
| `werewolf/observer_dtrs/online_context.py` | `build_shadow_request`、不可变 `ShadowRequest`，只读当前狼人的合法观测白名单 |
| `werewolf/observer_dtrs/online_model.py` | `TomIdentity`、`FrozenEventEstimator`、`read_deployment`，独立且不导入受限reader |
| `werewolf/observer_dtrs/deployment_offline.py` | 人工模型发布、经审计外层模型的只读编译导出；在线模块不导入它 |
| `werewolf/observer_dtrs/shadow.py` | `QPrediction`、`ShadowEngine`、不可变 `ShadowResult`、有界 `ShadowQueue` |
| `werewolf/observer_dtrs/shadow_artifacts.py` | 独立私有目录、互斥写入、不可变claim/result/failure、恢复与清单校验 |
| `tests/observer_dtrs/test_shadow.py` | PRE/座位/特征、45场景、异常、幂等、日志、导入闭包、反事实与非干预 |
| `tests/observer_dtrs/test_shadow_deployment.py` | 四类推理模型等价、摘要/来源绑定、损坏包与原外层审计fixture导出 |
| `scripts/check_observer_dtrs_shadow.py` | 独立人工验收与带外部摘要的inspect；没有真实游戏/服务命令 |

没有修改 `run_random`、recorder、agent、prompt、Pilot、冻结ToM、D0–D3文件或既有科研产物。
没有新游戏、真实ToM权重、服务器访问、commit或push。小型拟合仅发生在明确人工的测试fixture中。

## 2. 上下文白名单和观察者身份

```python
request = build_shadow_request(prefix, acting_observation,
    shadow_run_id="controlled-shadow", attempt_id="attempt-1")
```

只读取 acting observation 的 identity、observer_id、current_act_idx、公共day/phase/alive、
一条 `werewolf_team_info`。`Werewolf`自身身份、actor与speaker一致、actor存活、两个不同合法队友座位、
公开存活集合、day与runtime/canonical phase均核对。其他字段不保存、不遍历其内容。
其他玩家观测、真实报告、sidecar、Y、agent/env句柄不是ShadowRequest的字段。

调用者仍是合法观测的信任起点；本模块不能从任意Python字典证明其确由环境交付。
本轮不扩大数据权限，D4c控制层需保证传入当前actor已过滤的观测。

复用 `contexts_for_pre` 枚举全部存活非狼，包括普通与PK；不做Phase2候选筛选。
sample_id复用D1已有算法；run/attempt身份另外绑定，避免把不同尝试混为同一调用。
canonical step复制PRE的公共事件索引，不接收游戏循环step参数。
捕获时间为日志元数据，不参与幂等摘要；相同PRE重新创建请求不会因墙钟不同而再次调用ToM。

## 3. 部署包和来源验证

使用既有 `artifact_io.publish_artifact/verify_artifact` 发布、读取，不另造可覆盖的模型文件体系。
包内只有 `inference.json` 和 manifest，带调用者显式提供的 manifest SHA-256锚。
manifest与载荷字节在不可变模型值对象内再次绑定，不能换payload却继续使用旧manifest身份。

合法载荷只含：scope、model_id、weighting、TomIdentity、原模型/审计/选择规则摘要及推理参数。
EQ0/EQ1/EF只保存root、wolf、phase_nW、叶节点概率和原边界；LR只保存训练mean/std、系数或单类别常数。
没有原训练行、测试行、角色真值、Y、原始嫌疑报告、game_sizes、正类频数或训练支持记录。

TomIdentity严格区分 `controlled_fixture`、`d2_oof`、`final_tom`，绑定source OID、fit/execution、
seal、terminal、weights/checkpoint、fold、容量、时间与token-plan版本。相同形状/权重摘要的一部分相同不等于兼容。
model、reader显式expected_tom、engine注入的provider身份和每次Q响应身份必须一致。

**D4b只启用controlled_fixture和verified_d2_fixture包**；不启用final概率部署，也不提供静默OOF→final迁移开关。
人工包必须配人工ToM身份；真实D2只读fixture必须配同一OOF来源折的身份。元数据绑定不是权重计算重验。
实际ToM provider目前是注入接口，本轮只用人工实现；真实provider的权重/源码/容量证明留到D4c。

### D2如何安全封装

离线先用既有D3c-2流程得到 `OuterAudit`；导出需要外部钉住的完整audit摘要和目标model摘要，
以及预先指定的model/weight/source-fold/outer-fold和选择规则摘要。导出不查D3结果、不选模型、不拟合。
只允许有独立D1c、外层模型、原选参和保存推理核验标志的audit，核对其外层selection/predictions/evaluation引用。
真实D2从 `model.source.tom_fold` 选择对应OOF来源，不能默认取origins列表第一项。

`export_verified_outer_fixture` 将旧模型投影为白名单推理载荷，原受限证据不拷出。
原训练支持留在旧审计载荷，只有摘要引用进入新包。ToM max_seq_len须由调用者在后续部署资格验收中与实际冻结ToM容量核对；
本轮不根据D2节点表猜测权重容量。人工outer audit导出仍标为controlled_fixture，不冒充真实D2。

本轮正向导出验收使用旧完整审计函数产生的**人工**outer包，覆盖四模型。
没有给任何真实外层模型发布部署包，也没有在本轮重新打开真实监督去形成新audit。
因此不宣称某个真实OOF模型已被选定或已通过游戏侧final ToM的资格验收。

## 4. Shadow计算

```python
model = read_deployment(package_path,
    expected_manifest_digest=trusted_digest, expected_tom=trusted_tom_identity)
engine = ShadowEngine(tom_provider=provider, probability_provider=model,
    expected_deployment_digest=trusted_digest, journal_root=new_private_directory)
result = engine.infer(request)
verified = engine.inspect([request],
    expected_result_digests={request.digest: result.digest})
```

provider只接收公共PRE和 `deadline_monotonic`，返回 `QPrediction(matrix, tom, plan_digest)`。
engine验证完整PRE容量、响应身份、token-plan及 `ProbabilityMatrix` 7×7约束；Q必须是质量，不能是log-Q。
通过原 `observer_features` 保留死亡目标质量；单狼m=0/dW=None，座位恒为player1...player7。

只读模型按原边界/最近已观测父节点推理，不重新分箱或学习；LR调用原 `predict_many` 的数值路径，
不是重写优化器或标准化。使用原 `decide` 和冻结D3配置，输出每名观察者全部45场景×两种策略：
主 TERMINAL_NEG_FIRST、精确并列敏感性DEFER_FIRST。没有选择唯一主矩阵，也没有新阈值公式。

日志保存PRE/observer/sample身份、ToM和模型来源、Q/plan摘要、合法features、p、三风险、全部最优认知行动、
三域、risk gap与阈值信息。`minimizing_actions`是原D3认知行动集合；没有可提交游戏的 `action` 字段。
日志包含合法狼队私有信息，必须作为私有科研sidecar存放，不能公开注入游戏或提交Git。

低支持、缺失概率、技术错误不会自动成为BND。在线模型没有训练支持输入；监督和支持不进入 `decide`。
纯计算仍可由原D3测试独立验证；engine的日志副作用仅在独立目录，不是游戏状态副作用。

## 5. 技术失败、幂等与恢复

- 一个新请求先持久化不可变claim，再调用一次ToM，再封存结果。全请求受独占writer锁保护，重复写者明确拒绝。
- 同请求成功后重试/重启：验证来源、Q、features、概率与全部决策并复用结果，不再次调用ToM。
- 失败：持久化技术code/stage和空rows，不存任意异常消息、原始provider载荷或伪造分类；同请求失败后不自动重试。
- claim存在而没有result/failure：只读inspect报告pending；显式恢复时封存 `INTERRUPTED_NO_RESULT`，不重复可能已执行的ToM调用。
  这是at-most-once恢复，优先保留证据，不承诺所有中断请求都获得预测。新的尝试需显式新身份，不覆盖旧记录。
- 错误Q、错PRE/来源/plan、超时、日志缺失/截断/错配均报错，不能清理、重写或用均匀Q/其他模型补齐。
- 结果写入失败不会返回成功；若磁盘连失败记录都不能写，错误向调用worker传播，不能声称审计完整。
- `inspect`接收显式完整请求集合，核对清单、来源和逐行语义。外部 `expected_result_digests` 验证调用者先前保留的结果摘要。
  仅有内部自洽，无法识别Q和全部派生字段一起重算的整体替换；需要外部锚，不能把无锚inspect宣传为真实性证明。

### 超时与队列的本轮边界

同步core传递deadline、接受provider主动TimeoutError，并拒绝超过deadline才返回的结果。
**它不能强制终止一个卡死的任意Python/外部调用。** 不能把这个检查宣传为硬超时或直接放进游戏线程。
`ShadowQueue.submit`有界、非阻塞且不调用provider；`work_one`由独立人工worker pump驱动。
本轮不启动后台进程/真实服务。D4c实时接入前须有进程隔离、硬超时/健康状态和资源预算资格验收。

队列满返回SKIPPED_BACKPRESSURE；重复排队返回DUPLICATE。submission receipt和错误状态由调用者保存；
当前队列本身是内存接口，**不是**耐久的完整游戏覆盖ledger。D4c需将提交/跳过receipt单独可靠封存，不能只报告成功样本。

## 6. 受控验收内容

测试覆盖普通/PK、单狼/双狼、任意狼座位、死亡观察者、死亡目标质量、完整绝对座位、canonical身份；
全部45×2判断逐项等于原 `decide`。四模型编译前后在边界、nextafter、端点、两种phase/nW和固定人工随机特征上数值完全一致。

“触碰即抛错”的报告/sidecar/env/agent对象证明白名单不读取这些对象；相同合法PRE下改变监督不改变请求和结果。
受控观测、独立假状态、Python/NumPy RNG保持不变。新进程导入闭包不包含监督reader、训练审计、ToM/torch、
env、agent、run_random或Phase2。它证明接口隔离，不冒充尚未编写的真实游戏hook请求逐字一致验收。

还覆盖外部模型/ToM摘要、OOF与final拒绝、payload偷换、重算摘要后的损坏、日志发布失败、
claim中断、重复请求、缓存、只读inspect、外部锚与有界队列。测试数据均是人工，不能解释为概率校准或策略效用证据。

实际针对性/相关回归命令及结果以acceptance JSON为准：

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/name_yuxiao/anaconda3/envs/3wd/bin/python -B -m pytest -q \
  tests/observer_dtrs/test_shadow.py tests/observer_dtrs/test_shadow_deployment.py \
  -p no:cacheprovider --maxfail=3

PYTHONDONTWRITEBYTECODE=1 /Users/name_yuxiao/anaconda3/envs/3wd/bin/python -B -m pytest -q \
  tests/observer_dtrs tests/canonical_collection tests/development_publication \
  tests/tom/test_dataset.py tests/tom/test_boundaries.py \
  -k 'not off_diagonal_readout_can_memorize_controlled_public_fixture' \
  -p no:cacheprovider --maxfail=3
```

独立入口只接受 `controlled` 和 `inspect`，不接受run/dataset/服务参数；controlled输出目录必须全新。
它直接构造公开人工PRE和两条人工训练记录，不依赖tests包，不调用LLM或加载ToM。
本轮CLI人工封存为4个PRE、18条观察者、1620次认知判定；首次4次人工ToM调用，重启0次，带摘要inspect为0次。
重复controlled不会覆盖已有run，恢复语义由core的不可变journal及inspect验收。

```bash
python -B scripts/check_observer_dtrs_shadow.py controlled --destination /private/tmp/NEW_ARTIFICIAL_SHADOW_RUN
python -B scripts/check_observer_dtrs_shadow.py inspect --destination /private/tmp/NEW_ARTIFICIAL_SHADOW_RUN \
  --expected-summary-digest ACTUAL_CONTROLLED_SUMMARY_SHA256
```

回归曾发现原 `test_all_production_modules_belong_to_current_cli_import_closure` 报告新模块不可达。
原规则同时认可真实独立可执行工具，因此以上可复现人工入口解决了可达性，无需修改冻结CLI、测试或游戏hook。
它是有实际验收功能的工具，不是仅为图检查添加的空导入；其受控流水线及锚定inspect另有自动测试。

## 7. D4c准备判断

具备进入**最小PRE hook的设计/集成审阅**条件：Shadow核心有独立接口和受控证据，不需要修改原策略或放宽数据权限。
不等于已具备在主游戏线程启动真实ToM/正式Shadow的条件。

正式新游戏前还需冻结：

1. 实际ToM provider的source/fit或execution/seal/terminal/weights/fold、容量、时间编码与运行环境。
2. 概率包选择/抽取或新CAL拟合协议、训练/CAL游戏集合、权重、参数/选参规则、模型摘要，以及明确的ToM配对。
   不能按D3风险挑最优块；不能默许OOF→final。优先保持D4a推荐的final ToM+独立CAL，再封存后打开TEST。
3. 全新CAL/TEST整局seed/game pools、监督开放时点、缺失/技术失败和停止规则；历史开发集不重新标为独立测试。
4. D4执行源码新OID、协议/日志身份、进程隔离/硬超时、耐久submission receipts、输出权限与资源预算。
   旧生产guard保持不变；本轮新增生产源码未提交，guard拒绝正式执行是预期行为，不绕过。

45个成本和显式并列策略保留，不为试运行挑单矩阵。三域仍是认知状态，没有固定发言动作、Router或胜率结论。
固定历史ToM、开发集后验研究、历史RNG publication依赖和固定预测Bootstrap的研究限制继续保留。
