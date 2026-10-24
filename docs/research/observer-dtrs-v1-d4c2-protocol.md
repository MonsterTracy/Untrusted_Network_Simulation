# Observer-DTRS v1 D4c-2：final ToM、CAL/TEST 与部署冻结草案

状态：**PROPOSED_NOT_FROZEN_NOT_EXECUTABLE**。本轮只增加文档、机器可读提案和来源审计记录。
基点 `three-way / e513fd8b69404d07d69cd308023b673554f5bbf0`；D4a–D4c-1 未提交成果保留。
配套：`configs/observer_dtrs/d4-cal-test-proposed-v1.json`、`observer-dtrs-v1-d4c2-source-audit.json`。
没有服务器连接、真实模型启动、新游戏、ToM重训、正式拟合、部署、commit或push。

## 1. 结论和证据等级

推荐路线是 **冻结 final ToM → 独立 CAL → 固定只读概率包 → 全新 TEST**。
本地可以核对源码及既有元数据；**现在不能冻结为可执行协议**。
尚缺研究者参数确认、真实 final 权重/时间表的带外证据，以及真实 provider 的隔离、取消和容量资格。

| 等级 | 本轮确认范围 | 不代表什么 |
| --- | --- | --- |
| 已核实源码事实 | 当前6个ToM核心文件与本地Git中的f199历史blob逐字节相同；客户端期望身份、图、容量、调用和访问路径可审阅 | 服务器 checkout、环境或权重已验证 |
| 已核实本地元数据 | D1c manifest摘要及覆盖元数据、D2 contract/选定外层包内五折来源；D4既有验收记录 | 本轮加载了最终ToM或复演了上游训练 |
| 已知历史结果 | D2报告的EQ1 Brier=0.0958662878、LR-r=0.0904103094；D3已完成的用户确认 | final ToM上的新游戏校准、胜率或唯一优胜成本 |
| 推荐设计 | 本文CAL划分、候选比较、TEST双轨评价与资源资格 | 已获得研究者确认或已经执行 |
| 待补证据 | 实际fit/seal/terminal/权重/temporal字节、服务器隔离和取消测试、完整历史排除池 | 可由相同名称、矩阵形状或源码常量代替 |

### 1.1 final ToM 的期望身份

以下来自当前 `scripts/qwen3_gameplay_predictor.py:12–16` 的常量，是**预期锚点**，并非本轮读取权重后获得的校验结果：

| 项目 | 完整身份 |
| --- | --- |
| frozen predictor checkout | `f199a9e6c64168a412de91ff8ca04d0c795c57e2` |
| final fit | `87c2485acf48443034c59430d79f098a13151677852988bf691599ca2d2f8248` |
| seal | `fcbb81ee43050142ebdf01c0aee576156221d725598ff5ee2c02055049f72718` |
| terminal | `b977ec24d6292ad329e664c931a1aae39dc4738b4b33685b19d92fd4fb1921af` |
| model container descriptor digest | `fdf858b5e21a1951da9ed2c2b9e988f8a1f1edcf99d2bb84f54faf055dc7e3e9` |

`qwen3_final.py:340–359` 中最后一项为 `SHA256(canonical_json(last.manifest.containers.model))`，
**不是** `model.bin` 裸文件SHA-256。部署证明须分别保存descriptor摘要、recovery manifest摘要和payload文件SHA，不能混用。
本地项目和已知下载结果目录中未找到 `qwen3_final_manifest.json`，因此实际fit.protocol_inputs、model.bin、时间表字节摘要仍待提供。
这不声称全Mac所有目录都没有该文件。

final是全部1500局DEV上的独立训练生命周期，31500步、唯一terminal，不是某个OOF checkpoint。
`INITIAL_FOLD=0` 只是未训练初始化张量来源，不表示final是fold-0模型。
当前bootstrap helper文件不在f199提交中；它在调用方源码中启动，再从固定checkout导入ToM。
后续需同时固定**调用方helper源码摘要/新D4 OID**与**冻结predictor源码身份**，不能把二者合成一个OID。
实际训练source字段还须从fit manifest核验；不能只把客户端期望checkout当成已读过的训练来源。

### 1.2 图、时间与输入容量

这是 `BackboneToM('qwen3')` 的结构化连续输入模型：hidden=256、4层、8头、FP32/eager，
序列参数3148288 + shell参数282631 = **3430919参数**。没有LLM tokenizer、词表嵌入或语言生成head。
生成玩家发言/报告的Qwen3.5-9B和vLLM是另一个模型/执行路径。

公共PRE使用 `classic7_structured_token_planner_v1` 和固定事件/座位/action编码。
`explicit_day_phase` 使用保存的时间表字节，day表shape=257×128、phase表shape=5×128；推理不重算三角函数。
phase code bytes的源码固定SHA为 `27e974cdecc7927b04ccd2f5f2e4d13351a2edf382c88f12854633dd0be77490`；
day表及其parent/protocol摘要仍需真实artifact验证。
最大序列1024个**结构化token**，推导day容量256；两个限制都检查。
并非任意256天游戏都能装下1024 token。超容量拒绝，不截断、不回绕、不缩减到存活玩家列。

`SealedQwen3Predictor.log_probabilities` 返回7×7 log-Q；旧进程worker执行一次exp后返回概率。
新provider必须返回 `QPrediction(ProbabilityMatrix(...), exact_TomIdentity, plan_digest)`，不得再次exp。
自列0、其余六列质量和约1，绝对座位player1...player7，死亡目标质量保留。

### 1.3 与D2 OOF的差别

D2是每个ToM模型不使用其held-out 300局的五来源折特征；p再在该300局内嵌套验证。
其训练source为 `ebf98b7e3249f78d3f1fa1fd46c20affe0486913`，evaluation source为
`46b0cb24421a2f98e063d67aedaedb8e31875403`；五个terminal/checkpoint身份列于配套审计JSON。
它们不同于final的all-development fit/seal/terminal。相同图、时间语义或7×7形状不证明概率映射可迁移。
不得把旧fold=0填给final；final的ToM fold为null。

历史OOF训练RNG含全量publication绑定摘要的限制继续保留。
本地final源码的训练RNG另以其DEV game_ids等字段派生，不能把OOF RNG公式误说成final公式；
两者都不应在D4引入新的CAL/TEST标签或publication摘要来生成随机控制。

## 2. 当前真实客户端不能直接投入Shadow

| 阻碍 | 源码证据 | 正式接入要求 |
| --- | --- | --- |
| 接口不同 | 旧 `predict(prefix)` 返回裸matrix；D4要求deadline参数、完整TomIdentity和plan_digest | 新独立adapter验证外部锚并包装，不伪造尚未验收的身份 |
| 启动无上界 | client:118 `_read(timeout_seconds=None)` | 冷启动、ready、query、cancel、close分别有界 |
| 600秒不是硬期限 | client:125–141；select之后readline可等残缺帧；write/flush也无硬截止 | 有界帧长、累计字节、读写截止与独立进程监督 |
| 退出无上界 | client:154–157 terminate后wait，无kill升级/有界确认 | 对自己拥有的进程树/作业实施取消并确认释放；不杀共享服务 |
| 不是vLLM ToM RPC | 旧client通过本地Popen+stdin/stdout启动PyTorch worker | 不能把游戏LLM的vLLM取消机制当ToM取消证明 |
| 启动读取受限历史材料 | `open_fit:188–237` 打开DEV publication/role sidecar/primary.json | 在线进程不可因此获得监督读取权限 |
| 冷启动会复核完整恢复链 | `verify_seal→verify_terminal→_chain(complete=True)`，32个recovery点及optimizer/RNG恢复校验 | 不应为每个PRE重复完整审计/重载模型；离线核验与驻留推理解耦 |
| 杀wrapper不等于杀GPU子任务 | D4c-1只保证自己spawn的子进程结束 | 真实服务须有request/lease、cancel ack、作业终止及资源回收证据 |
| 现有scope主动拒绝 | `online_model.validate_payload`、`IsolatedShadowWorker.__init__`只资格验收fixture | 新D4正式schema/guard须独立实现与验收；不放宽fixture为final的别名 |

旧客户端源码保留，不在本轮修补。它可以作为数值与来源参照，尚不能证明满足正式Shadow的权限和生命周期约束。
旧 `_serve` 还导入run_random；D4c-1人工provider的“子进程无env/agent/监督模块”证明不能直接移植给它。

### 推荐的后续实现

1. **受限离线ToM部署导出**：在完整核验原fit、study、temporal、32点恢复链和seal后，
   生成新的只读推理capsule，引用原来源，复制验证后的model tensor payload、temporal bytes、图和公共编码规范。
   不复制primary.json、DEV报告、角色侧文件、optimizer、RNG或训练loss。
2. **最小公开输入loader + 独占驻留推理进程**：只挂载capsule和新PRE，不挂载publication/CAL/TEST targets。
   原graph/tensorizer的传递导入也包含dataset/publication模块，须明确审计并建立可证明的最小合法导入路径；
   不能只改文件权限却宣称启动接口已无受限依赖。新loader与原封存模型的数值等价需另行验证，原ToM源码保持不变。
3. **可杀的请求客户端**：D4c-1 worker可继续隔离每次请求，但只持有不可变endpoint/config，不能pickle活的Popen/CUDA句柄。
   驻留进程由独立owner管理、一次只服务一个请求；租约/截止后先取消并确认idle，无法取消则只重启自己拥有的进程组，
   确认旧作业已结束后才恢复服务。终止客户端不算取消成功；没有服务端证据则本次资格失败。
4. **只算一次Q**：多个p模型/权重共享同一已验证PRE-Q。不能分别调用多个现有engine而重复加载或计算ToM。
   结果仍分别绑定概率包、模型ID、权重、场景和TiePolicy，不共享错身份缓存。

这些是待实现模块，不是本轮已发布的capsule/provider。新loader、服务owner和请求adapter均须有独立新D4源码身份。
当前 `TomIdentity` 不单独绑定新capsule、loader源码、runtime和实际day表字节；正式D4证明需在其外增加不可变deployment身份，
并同时绑定概率包、服务握手和Q响应。原模型来源OID保留在reference lineage中，新loader绝不能冒称运行于f199。
已有 `Untrusted_Network_Simulation_qwen3_predictor_f199` 等服务器目录及vLLM/采集任务保持原状，后续也不得默认重用其活跃进程。

## 3. CAL：生成样本、合法特征与受限拟合

### 3.1 避免“先有p才能采CAL”的循环

推荐先用固定的原baseline生成CAL canonical游戏，不给Agent注入ToM、p或区域。
完成publication后，用冻结final ToM对**合法冻结公共PRE**做受控离线物化；只输入PRE，绝不输入报告/角色/未来历史。
按D1的 `(g, PRE, acting_wolf, observer)` 和D0事件构造标签，按D1b原函数生成features：

\[
s=Q_{i,w},\quad m=\sum_{k\in W_t\setminus\{w\}}Q_{i,k},\quad r=s+m,
\quad d^W=|Q_{i,w_1}-Q_{i,w_2}|.
\]

仅存活狼人发言PRE；全部存活非狼观察者，包括PK中的非候选观察者。
单狼m=0、dW=None；保留phase/nW。Q不是log-Q，不重归一化存活列。
成功空报告仍为六名非自身玩家的有效集合；失败/缺失没有有效Y；仅嫌疑死亡狼人则Y=0。
完整publication缺报告或身份不匹配时拒绝构造，不静默丢弃。

CAL的Q/技术覆盖须先封存，再向拟合分析者开放CAL标签。
原canonical完整性验证者可在受限侧读取报告检查契约，但不能把其内容输给ToM或在线读者。
CAL物化是as-of-PRE的离线计算，不冒称已通过实时服务资格。

### 3.2 复用范围和新schema

复用 `publish_development/open_publication/open_role_sidecar` 的来源机制；它支持至少5局的plan-closed publication，
并非只能1500局。新ID必须表明CAL/TEST实验角色。
publisher附带的五折元数据只作容器完整性，不是新的ToM来源折，也不用于CAL选参。

复用 `build_observer_samples`、`contexts_for_pre`、`sample_id`、`label_from_observation`、`observer_features` 的数学与身份检查。
**不能**直接把final Q塞进D1c的OOF来源schema；需新D4数据envelope，独立绑定final来源、CAL/TEST split与PRE-Q。
CAL/TEST `public/games` 中仍可能含真实BeliefObservation；“public”目录名不等于线上可读。
只导出白名单PRE/Q/feature合法视图，线上不挂载整个publication。

同时，旧 `estimation._validate_source` 的formal scope固定D2 dataset、5个ToM折和240/160局。
不得把真实CAL标成controlled_fixture、伪造OOF折、放宽旧guard或修改冻结D2协议。
后续新增D4校准拟合envelope/validator，复用经等价测试的分箱、支持、收缩、Logistic纯数值运算。

### 3.3 推荐的候选和CAL内选择（全部待确认）

推荐固定比较EQ0、EQ1、LR-r；**EQ1为结构化主机制，EQ0主对照，LR-r预测精度对照**。
D2中LR-r损失更低是保留它的理由，不是final上的部署校准证明。
不依据D3最有利成本或TEST表现选模型，不指定唯一损失矩阵。
另一个可选方案是预登记“CAL-CV Brier选唯一模型族”，但会改变研究问题，本文不默认采用。
EF可留作后续另行登记的消融；D3原四模型结果完整保留。

推荐CAL内整局5折；final ToM已在DEV封存，CAL从未进入其训练，因此不再套用D2的25个同来源外层块。
先按预分配ordinal/seed身份哈希分折，每局一次验证；不按标签分层，不用bundle/publication/样本/预测摘要生成数值随机控制。
训练分箱、标准化、模型拟合只用该折训练游戏；验证Y只评分和选参。

推荐沿用已审阅的候选网格和数值规则，但作为**新D4协议参数**登记：

| 模型 | 条件/网格 | 并列规则 |
| --- | --- | --- |
| EQ0 | phase+nW；lambda∈{0.25,1,4} | Brier最小+1e−8内取较大lambda |
| EQ1 | bin(r)+phase+nW；B∈{1,3,5}，同一lambda网格 | 容差内先较少B，再较大lambda |
| LR-r | 原6系数design；tau∈{.001,.01,.1,1} | 容差内较大tau |

推荐训练权重游戏等权：完整合法样本数为n_g，u_gj=1/n_g。
这里的训练游戏数G只计至少一条合法样本的训练游戏；零样本局另记覆盖，不能进入1/n_g运算。
选参目标是各CAL验证游戏Brier的合并等权平均，不能简单等权平均大小不同的fold均值。
分箱仍采用训练内加权逆CDF、不拆相同r、10个不同游戏和Kish游戏质量支持5的确定合并。
收缩保留根 `(S+0.5)/(G+1)` 及global→nW→phase×nW→bin层级，未见类回退最近已观察父节点。
EQ1单箱同lambda必须等于EQ0；单类别保留/收缩，空训练集报错，不能用测试标签决定回退。
LR只用训练mean/std，零方差z=0，单类别训练用原平滑常数，优化失败不换模型。

CAL选定各模型参数后，使用全部CAL训练游戏重学边界/标准化并拟合一次，封存全部候选的最终包。
推荐游戏等权为主；记录等权若保留为D2式敏感性，应独立拟合/选参，而非把主模型重新加权冒称另一套训练。
是否保留第二套拟合是待确认资源选择；只重新加权评分时须明确标签为“评分测度敏感性”。
所有训练支持、fold评分、原始Y和来源审计留在受限载荷；预测只收ObserverFeatures。

## 4. CAL/TEST 数据池、标签开放和停止规则

### 4.1 预分配与排除

1. 先固定独立QUAL、CAL、TEST collection IDs和整个候选seed池，不根据运行结果找seed。
   可复用 `derive_seed_pool` 的既有SHA256(collection_id, ordinal)规则；当前仅设计，不生成正式seed。
2. 校验三池两两不交，并排除DEV全部**计划池**、历史采集/ToM消融/Experiment A/Phase2/Pilot/qualification等全部计划池，
   不只排除成功或实际执行过的seed。现有本地D1c可确认1500个成功DEV游戏，但不足以证明所有历史候选池无重用。
3. 计划ordinal→seed→CAL fold提前登记；canonical失败后不改后续fold、不重抽、不按类别或winner替换。
   相同普通开局不是泄漏证据，不能据此选择性删样本；完整重放/复制bundle按预登记来源规则拒绝。
4. 固定Agent、发言/报告/解析prompt、语言后端及参数、角色规则、采集预算；CAL和TEST同一生成机制。
   Shadow不反馈。未来ToM-only/ToM+p/ToM+p+DTRS的策略干预需另行协议，不能默认视为相同分布。

### 4.2 数量建议，不是功效保证

| 备选 | CAL成功canonical局 | TEST成功canonical局 | 定位 |
| --- | --- | --- | --- |
| 资源较紧 | 300 | 300 | 初步独立评价，稀疏子组限制较强 |
| **推荐预算起点** | **500** | **500** | CAL五折每折约400训练局；TEST保留500个整局聚类单位 |
| 更重TEST | 500 | 1000 | 提高固定模型下测试精度，仍不保证所有PK子组充分 |

这些数字尚未确认，不写入可执行计划。建议候选池上限暂按成功目标2倍讨论，也未冻结。
历史DEV约15.12条观察者/局、3.89个狼人PRE/局，只是工程规划参考，不能保证新数据yield。
原PK单狼组在1500局仅31条负记录，故500局不能承诺该子组精确校准或显著性。
预算最终应由资源资格和预期精度目标决定；整局配对差的规划关系为h≈1.96·sd(局差)/sqrt(G)，
新模型/分布下sd未知，不能用22685行当独立样本量，也不把DEV区间当新实验功效保证。

### 4.3 停止、技术失败和零样本游戏

推荐兼容既有collection：固定canonical成功目标N和候选池K，依原顺序执行，达到N个canonical成功即停；
耗尽K/预算或触发安全条件时停止，封存全部attempt ledger，标为未达到计划，不能自动扩池。
canonical成功标准保持原报告完整性契约；**Shadow成功不是补足canonical目标的条件**。
同局Shadow排队满、超时、Q无效、日志坏不能换一局，也不能让baseline换动作。
全程禁止按风险/胜负/区域比例/正负样本数延长或提前结束；协议变更只适用于新的版本/新池。
安全条件至少包括来源/摘要错配、取消不确认、资源预算耗尽、审计损坏；具体数值阈值待QUAL后确认。
此处先停用Shadow，并在游戏边界停止接纳新局；当前baseline继续原动作和原失败规则，不能由Shadow异常改写或强制结束游戏。

所有计划、尝试、canonical失败、完整游戏、无狼人PRE、合格PRE、观察者样本、接纳/跳过/迟到/失败/成功、
标签可用与概率可用均分别计数。原报告失败在Hook之前停止的PRE由canonical failure evidence补充总体，不能从Shadow日志推断不存在。
零合法样本游戏仍保留在游戏/覆盖台账；其Brier没有行均值，不伪造为0。
概率风险估计定义于G+（至少一条合法样本的完整游戏），覆盖分母同时报告全部分配/尝试游戏，明确条件性。
CAL拟合要求所纳入游戏的整个合法论域Q与标签完整；不能把漏Q后的剩余行重新定义成n_g。

### 4.4 推荐TEST双轨及标签开放

**推荐但待确认**：把实时工程表现和统计评价分开。

- live轨：真实PRE捕获/队列/截止日志，不补写成功、不更改FAILED/迟到状态；报告全部覆盖、延迟和资源。
- replay轨：协议事先登记，在TEST结束后，仅用冻结公共PRE、同一冻结ToM和概率包做完整只读重放；
  不读取TEST Y，不重新拟合。单独来源/目录/时间戳，核对与可用live预测的一致性。
  用于固定管线在全G+上的概率与三支风险评价；不能称为当时已完成的在线推理。

如不采用replay，只能对live概率可用子集评分，必须明确技术失败可能与难度相关，不能当作全体新游戏风险。
两轨都未能完整覆盖时，不通过删除未预测样本发布完整模型比较；可封存不完整状态与明确的描述性子集结果。

标签开放顺序：

`协议/排除池冻结 → CAL游戏 → CAL公共Q封存 → CAL受限拟合/选参 → 全部最终概率包冻结 →
TEST游戏/受限报告封存 → live及预登记replay预测与技术覆盖封存 → 外部摘要锚确认 → TEST标签开放 → 固定评分封存`。

原baseline的speaker自报告handoff照常存在；隔离的是Shadow和分析者的额外监督访问，不修改原Agent认知。
TEST任何target、成功空标志、winner、支持频数都不参与provider、模型选择、分箱、取消/重试或技术接纳。

## 5. 部署包和来源接口（待实现）

建议新artifact type：`observer_dtrs_final_cal_deployment`，独立v1 schema；不沿用fixture scope。

| 层 | 合法输入/载荷 | 受限且不进入在线包 |
| --- | --- | --- |
| ToM capsule | model payload/descriptor SHA、图、temporal字节、公共编码、完整训练来源引用 | DEV角色/报告/primary、optimizer/RNG/loss |
| CAL dataset | sample_id、run/attempt、game/PRE/actor/observer、Q来源、features合法视图 | Y、原/有效嫌疑集合、角色与成功空审计 |
| fit audit | CAL来源、split、训练集、网格、评分、选参、参数与推理等价证明 | 原始训练行/Y/节点支持，全部受限 |
| online p package | family/deployment ID、权重、exact final TomIdentity、特征/标签定义版本摘要、CAL集合/split/fit证据摘要、推理参数 | 样本Y、真实报告、训练计数/支持、TEST预测或targets |
| run contract | 新执行源码OID/字节快照、所有部署摘要、45损失配置与显式tie、队列/超时/覆盖规则 | 测试标签不得成为数值控制 |

EQ只导出节点概率和原边界；LR只导出mean/std/coefficients或常数。
建议模型族沿用OBS-EQ0/EQ1/LR-r，另用独立 `deployment_id=d4-final-cal-...` 表明新来源，禁止与D2 outer model digest混称。
离线编译前后在边界、nextafter、单狼/双狼、两阶段、端点、未见类和随机人工特征上逐项等价。
线上reader不导入CAL targets/训练审计，也不靠目录名假设安全；支持诊断另读独立审计，绝不进入decide。

## 6. 评价与统计（推荐规则，非已冻结结果）

游戏等权Brier为主，LL、校准bias/预设概率箱及记录等权另列；评分LL只在计算时用epsilon=1e−12，保留原p和端点计数。
推荐主对比EQ1−EQ0，关键次要LR-r−EQ1，不能相减两个区间获得直接配对区间。
三支沿用D3完整45成本、主TERMINAL_NEG_FIRST及DEFER_FIRST敏感性，TERMINAL_POS_FIRST仍保留退化诊断。
不设唯一主矩阵；报告实际六损失、强制二支/固定行动对照、覆盖和错误断言。
成本96条邻接边、EQ1概率扰动半径0/.01/.025/.05和原支持分带保持原语义；扰动不是误差置信界，低支持不是BND。

推荐TEST完整游戏配对Bootstrap 2000次；同一抽样用于所有模型、权重和成本，记录等权重算分子/分母。
一个固定final来源不再按旧ToM折/外层块分层；若预登记采集batch，按该batch分层，否则整池整局抽样。
不重用D2的1500局索引，不按Y或事后区域建层；新的Bootstrap/CAL split固定seed均待确认。
区间条件于固定ToM、CAL模型选择/拟合与预测；不包含完整上游重训或CAL重新选参。
多成本是规范性敏感性分析，不把45个点区间当45个独立确证性检验。
子组phase×nW、成功空报告、支持/回退主要作描述性点值，稀疏组不删、不强行宣称充分。

TEST真正独立时，可评价固定ToM+p+DTRS在相同生成机制的新游戏泛化；
不能把旧DEV重新命名为TEST，也不能推出新的Agent/策略/胜率泛化。
固定历史ToM、DEV后验设计和历史RNG来源限制继续披露；新TEST不进入上游训练或随机控制。

## 7. 资源资格和工程预算

目前只有源码/人工证据，不是服务器配置验收。`configs/server.json` 指向 `/data/yuxiao/Untrusted_Network_Simulation`；
Conda预期 `/data/yuxiao/envs/untrusted-network-simulation` 的实际软件/设备状态未核验。
现有vLLM配置gpu-memory-utilization=.90，不代表剩余GPU空间或并行运行已安全。

| 项目 | 本地可算的规划值 | 必须另行验收 |
| --- | --- | --- |
| ToM参数载荷 | FP32约13723676字节≈13.09MiB；时间表约0.128MiB | 不是RSS/VRAM；框架、1024长度attention和服务开销未测 |
| Q请求数量 | DEV均值3.89狼人PRE/局；500+500局约3890次，只作估算 | 新轨迹长度、排队、max请求大小、峰值到达率 |
| p拟合 | 5折×(3+9+4)候选+3全CAL重拟合=83个逻辑fit/权重；两权重166 | 新D4合法来源实现与CPU时间；不是ToM训练 |
| 全决策日志 | 人工D4c-1约62KB/观察者/模型/权重，含45×2判断；1000局Shadow×历史yield×3模型约2.62GiB/权重；推荐TEST500约1.31GiB/权重 | 实际分数字符长度、replay副本、canonical/LLM日志、manifest开销和安全余量 |

CAL仅需Q/features及受限拟合证据，不需要生成三域日志；上面的1000局是日志规模的换算单位，不是把500 CAL也算作TEST。
两种权重、live/replay双份封存均会另增体积。人工概率的Fraction字符串通常更短，这些数字不是实际产物体积上界。

初始讨论值可以是一个独占驻留ToM进程、query并发1和有界pending=2，**不是已冻结资格参数**。
CPU部署也只能作为新capsule运行时的待验收候选，不能改旧fit.device/关闭_source_gate来获得CPU版本。
真实资源检查由用户另行执行，至少证明：

- 固定环境/设备、冷/热启动、最大合法PRE、p50/p95/p99和最坏时延、CPU/RSS/VRAM、Q和日志大小。
- query并发/排队容量、ready/write/query/cancel/close各期限、字节上限、崩溃和残缺帧、晚响应租约。
- 取消后不再计算或占用GPU；无法证实时服务保持unhealthy，不接纳后续请求、不碰共享vLLM。
- 目录0700、模型只读、在线用户无publication/targets权限、每attempt日志、fsync延迟和外部锚封存。
- 与现有采集/vLLM有明确资源隔离；不宣称真实采样文本逐字相同。磁盘失效时不声称失败也已耐久。

所有硬期限、CPU线程数、GPU/内存/磁盘预算、失败突发阈值均保持null，待独立QUAL证据和研究者批准。

## 8. 两阶段冻结检查表与最小实施顺序

### F1：采集/选参协议冻结（CAL之前）

- [ ] 研究者确认预算、候选/权重与选择口径、TEST双轨口径；全部数值控制种子和候选池上限已固定。
- [ ] 原始final fit/study/seal/terminal/recovery/temporal证据由用户提供并与预期锚匹配，权重descriptor和文件SHA分清。
- [ ] 新ToM inference capsule/loader、D4来源型CAL fitter/reader/部署publisher、服务owner/adapter实现并受控验收。
- [ ] 用户手动完成独立QUAL；取消、容量、访问隔离及现有任务安全证据通过。
- [ ] 全部历史计划排除清单到齐，QUAL/CAL/TEST池无重用；原采集器的停止契约保持兼容。
- [ ] 新D4生产源码已审阅、固定新OID、guard通过；协议摘要和合法来源包被带外固定。

F1不要求尚未拟合的CAL模型已有摘要；其合法生成规则必须冻结。当前上述多项未完成，不能改状态为FROZEN。

### F2：概率部署冻结（CAL之后、TEST之前）

- [ ] CAL完整封存、Q与标签身份交叉核对、预登记5折覆盖/全部候选执行完整；没有TEST参与。
- [ ] 每个模型/权重按既定规则选参、全CAL重拟合，部署白名单和数值等价证明通过。
- [ ] 概率包与确切final ToM配对；部署manifest、合法payload、受限fit/support audit分别固定。
- [ ] TEST运行contract引用F1、全部最终包、源码与资源资格；标签尚未向分析者开放。

F2之后运行TEST；预测和技术覆盖封存后才开放Y并固定评价。不允许TEST结束再“冻结”模型。

**后续最小实现顺序**：用户补齐final元数据证据 → 审阅新只读ToM capsule/生命周期与D4 CAL来源schema →
人工数值/权限/取消验收 → 用户独立QUAL → F1 → CAL → F2 → TEST → 标签开放与评价。
本轮没有实现这些生产模块，没有给现有fixture放行final来源，也没有修改D0–D3或D4c-1非干预边界。
