# Observer-DTRS D3d-1：正式执行协议候选与可恢复运行器

日期：2026-10-15。状态：**CONTROLLED_ACCEPTANCE_PASSED_PROTOCOL_READY_FOR_REVIEW_NOT_FROZEN**。
Mac `three-way`，HEAD 仍为 `3ff578f97b9d4c3620a15a67357ddabd848d10fc`。
本轮没有正式 D3 三域实验、真实六损失比较、模型重训、服务器访问或 Git 提交/推送。
D0–D2 冻结数据、模型和结果以及 D3b/D3c 已有文件保持不变。

## 1. 文件与协议身份

新增：

- `configs/observer_dtrs/d3-execution-v1.json`：完整机器可读候选执行协议。
- `werewolf/observer_dtrs/d3_protocol.py`：协议验证、45场景、成本邻接与正式源码 guard 扩展。
- `werewolf/observer_dtrs/d3_inputs_offline.py`：复用 D3c-2 的真实输入组装、独立监督/支持容器与原 Bootstrap 字节。
- `werewolf/observer_dtrs/d3_runner.py`：评价、检查点、完整封存、并列/成本诊断与 seal 检查。
- `scripts/run_observer_dtrs_decisions.py`：controlled / inspect / run 三种明确入口。
- `tests/observer_dtrs/test_d3_runner.py`：35项受控验收。
- 本文和 `observer-dtrs-v1-d3d1-acceptance.json`：研究协议、实际验收及文件指纹。

协议版本 `observer_dtrs_d3_execution_protocol_v1`，ID `observer-dtrs-d3-v1`。
Canonical JSON SHA-256：
`5050755cdaa3746acf0ac917fda07e2c7cc6245ba418840203e4e27e80fe054e`。
**当前状态 READY_FOR_REVIEW_NOT_FROZEN，尚不是允许正式运行的冻结协议。**
未来明确审阅后再将状态改为 FROZEN_FOR_EXECUTION、形成新摘要并固定源码 OID；本轮没有自动执行这一步。

## 2. 决策对象、损失与并列

目标事件仍为 `Y=1[有效嫌疑集合与存活狼人集合相交非空]`；概率仍来自已封存 D2 外层预测。
纯函数 `decide(p, LossSpec)` 不接收 Y、原始行、空报告标记、训练支持或角色真值。
三域仍为认知命题接受/暂缓/拒绝，不解释为任何发言或攻击动作。

六损失保持 P=(0,c_FP)、B=(b1,b0)、N=(c_FN,0)，风险为：

- R_P=c_FP(1-p)
- R_B=b1*p+b0*(1-p)
- R_N=c_FN*p

45场景完全复用原 candidate_losses 的有理数成本与顺序，C00–C44：
`rho∈{1/4,1/2,1,2,4}`、`kappa∈{1/5,1/2,4/5}`、`h∈{1/4,1/2,3/4}`；
`c_FP=rho/(1+rho)`、`c_FN=1/(1+rho)`、`b1=kappa*(1-h)*c_FN`、`b0=kappa*h*c_FP`。
因此 `c_FP+c_FN=1`，且 `b0/c_FP+b1/c_FN=kappa<1`，全部严格非退化。
没有新增或删除成本，没有根据标签、风险或区域比例选主矩阵。

阈值仍为 alpha=(c_FP-b0)/(c_FP-b0+b1)、beta=b0/(b0+c_FN-b1)，beta<alpha。
直接 argmin 是规范依据；所有精确并列使用输入的精确有理值，无浮点吸附容差。

- 主策略 TERMINAL_NEG_FIRST：p≤beta 为 NEG，p≥alpha 为 POS，中间 BND，符合 D0 非退化边界。
- DEFER_FIRST：在精确 beta/alpha 处选择 BND，作为固定成本/概率下的精确并列敏感性。
- TERMINAL_POS_FIRST：对本45非退化成本的选域与主策略一致；两终端同时最优的区别只在退化诊断中出现。

主执行表为45×2策略×4模型×2权重=720单元，策略身份与成本共同进入 loss digest。
强制二支另行固定 NEG 优先，不能随三支策略变化。
弱序、零暂缓和三者并列等退化矩阵只在12个**无标签人工诊断矩阵/策略组合**中计算
p∈{0,1/2,1}，与45正式成本表隔离，不应用于真实样本。

## 3. 统计对象与主要指标

主要模型 OBS-EQ1，对照 OBS-EQ0；OBS-LR-r 和 OBS-EF 为预定补充。
游戏等权是主分析；record_equal 使用原 D2 对应权重的预测管线及记录等权评价。
它不是单纯给同一组模型预测换权重，不将两者差异解释为纯粹测度效应。

完整游戏 g 有 n_g 条合格样本。主测度每条质量 1/n_g：
`R_game = (1/G) sum_g [sum_i loss(a_gi,Y_gi)/n_g]`。
记录等权为 `R_record=sum_rows loss/N`。对子组先固定原完整 n_g，再取子组质量之比；
不能把子组内每局重新归一化为等权。所有原游戏保留，单类别不删除，零分母比率为 null。

每个单元报告：

- 实际六损失、模型自身预测最小风险（两者不可混同）。
- POS/BND/NEG 的记录数、涉及游戏数、质量、比例、事件率及全人口损失贡献。
- 终端决定覆盖、联合 FP/FN、按 Y 条件的错误率、POS/NEG 错误率、选择性错误率。
- BND 损失贡献，及三个区域贡献之和等于总实际风险的检查。
- 三个固定行动分别报告，强制二支单列；报告三支减各对照的风险差，不从测试 Y 选择最好常数。
- 主配对差 EQ1−EQ0；LR-r−EQ0、EF−EQ0 补充报告。负值仅说明在该成本下较低风险。

各模型比较均使用相同游戏、成本、策略与权重，保留按游戏的损失分子/质量分母。
不跨规范性成本平均风险形成一个未经论证的总排名，不将分域比例本身作为模型有效性证据。

## 4. Bootstrap 与多场景解释

复用 D2 complete 的**原始** `bootstrap_indices.int64le`，而不是重新选种子或重新抽样。
来源 fold×外层块共25层，每次在每层抽相同数量的完整游戏，所有模型、成本和指标共用2000次索引。
来源/形状/分层和原字节均核验；D3 bootstrap 产物另存同样字节及原游戏顺序。

对每模型总实际风险及成对模型风险差报告95%百分位区间，quantile method=linear。
record_equal 在每次抽样后用重复游戏的损失分子与记录分母重新取比，不能平均单局风险冒充记录等权。
至少95%重复具有有限估计才给区间；零质量/未定义不填0。
覆盖、错误率、子组、并列/成本翻域及训练支持诊断在此版本仅给描述性点统计，不臆造其置信区间。

45场景完整保留，全部区间是规范性敏感性研究的逐点区间，不是45次独立确证检验。
不由区间是否跨0删场景或选赢家，不在本版本生成显著性计数、p值或未登记的主矩阵。
Bootstrap 条件于历史 ToM 产物、D2已选模型和固定预测，不包含重新训练、重新选参的方差。
训练游戏重叠和历史来源依赖不因配对 Bootstrap 消失，区间只是近似的游戏构成稳定性。

## 5. 预先登记的诊断汇总

### 训练支持

各 case 保存普通/PK×单/双狼、成功空/非空、五个ToM折、25外层块、使用层级和支持带的统计。
Kish 带建议 `[0,5),[5,10),[10,20),[20,50),[50,∞)`；游戏数带
`[0,10),[10,20),[20,50),[50,∞)`，均左闭右开，不从测试数据学习边界。
层级为 global/nW/phase_nW/r_bin，Logistic 单列 NOT_APPLICABLE；回退 yes/no 与空子组保留。
这些是**待审阅的描述性分组值，不是可靠性检验阈值**，不排除样本、不改域、不产生 BND。

### 概率扰动

建议对主策略下 OBS-EQ1、两种权重、全部45场景报告外生半径
`delta∈{0,1/100,1/40,1/20}`。以 `[p-delta,p+delta]∩[0,1]` 枚举区域稳定性，
按全人口与既定子组质量报告不稳定比例和平均情景最大后悔值，总体另报告最大值。
这里假定未知 eta 位于给定外生区间；其后悔值不是用单条 Y 估计的真实 Bayes regret，
也不是概率误差置信界。其他模型/策略明确标为本项不适用，不能临时按结果补选半径。
精确最优并列、风险间隔0、外生扰动不稳定和低训练支持是四种不同诊断。

### 成本与并列敏感性

保留原5×3×3网格，沿每一轴的相邻场景组成96条无向边；主 EQ1/主并列策略下
每种权重完整报告96条边的9种区域转移质量和翻域比例，共192个产物。
它只比较已登记规范性场景，不优化成本，也不引入新矩阵。
另对45×2权重×4模型生成360份 DEFER−NEG 精确并列诊断，记录翻域与实际风险差；
必须保持原始风险与完整最优集合相同，唯一最优行动不得因并列策略改变。
TERMINAL_POS_FIRST 保留在人工退化诊断中，不增加真实45场景执行。

## 6. 工程入口与产物

```text
controlled_inputs() -> D3Inputs(scope=controlled_fixture)
prepare_formal(D2_run, D1c_dataset, protocol=...) -> 只读核验后的 D3Inputs
execute_controlled(destination, inputs=..., protocol=...)
execute_formal(destination, run_path=..., dataset_path=..., protocol=...,
               expected_protocol_digest=..., expected_revision=..., artifact_root=...)
inspect_result(destination) -> 字节/清单 seal 检查，明确不是输入语义回放
```

正式入口先调用原 D2 production source guard，再补验 D3 CLI/config 已跟踪且无修改，
然后要求已冻结协议和显式摘要；这两关通过前不读真实输入、不创建输出、不执行评价。
概率来源仍是 D2 OID `3ff...`，未来 D3 执行源码使用新的已提交 OID，两者分别记录。
controlled 入口只接受明确人工 scope，拒绝正式来源路径参数；没有跳过生产 guard 的正式执行开关。

所有产物复用原 publish_unit/read_unit 与不可变 artifact envelope；D3 record 有独立版本。

```text
contract/                 完整输入/协议/源码身份
source/                   执行源码与运行环境快照
restricted_support/       单独受限支持表及监督摘要；不传给 decide
bootstrap/                原D2索引字节、游戏顺序与分层
cases/POLICY/Cxx/W/M/      720份逐样本纯决策、整体/分组统计、逐游戏风险、风险区间
comparisons/POLICY/Cxx/W/  180份配对模型差与区间
tie_sensitivity/Cxx/W/M/  360份精确并列诊断
cost_edges/W/xx-yy/        192份成本邻接诊断
diagnostics/              人工退化矩阵，无真实监督
complete/                 只有全部登记单元验证通过后发布
failures/                 追加式异常证据，不删除历史
```

逐样本决策保存 SID/game、域、全部最优行动、三风险和间隔的精确值、强制二支及扰动诊断；
不复制 Y、成功空标记或支持字段。概率仍由固定来源绑定，不以总体均值/校准箱重造。
聚合层显式另开监督和支持。所有损失、模型、权重、抽样、来源与研究范围都进入 contract/单元身份。

根锁防止并行写入。每个 case 完成后单独发布；恢复时验证身份和字节，重新计算固定 p 的
决策/评分验证检查点语义，**不会重新拟合，也不会跳过不一致单元或自动修复原文件**。
这会重放已完成评价，恢复并非零计算量。非完整前缀无 complete；失败保留原件及错误阶段。
若连 contract/source 都未验证通过则直接停止，不写入冒称可信来源的失败记录。
独立 inspect 检查1458个登记产物的封存引用；完整输入语义回放使用相同源输入再次执行。

正式吞吐量和空间尚未实测：720个case共约1633万条决策记录，不能用人工小例吞吐量声称
正式耗时；Exact Fraction、扰动和语义回放有CPU/空间成本。无需GPU。下一轮正式执行前
应单独核对 CPU/RAM/磁盘预算，不启动拟合或以部分结果更改协议。

## 7. 实际验收

最终针对性35 passed，27.57s；相关回归499 passed、1 deselected，105.98s。
本地 Python 3.12.13；排除项沿用 ToM memorization 测试，不声称全仓或服务器测试通过。
新Python文件另通过3.10语法解析，这不等于服务器运行验收。

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/name_yuxiao/anaconda3/envs/3wd/bin/python -B -m pytest -q \
  tests/observer_dtrs/test_d3_runner.py -p no:cacheprovider --maxfail=2

PYTHONDONTWRITEBYTECODE=1 /Users/name_yuxiao/anaconda3/envs/3wd/bin/python -B -m pytest -q \
  tests/observer_dtrs tests/canonical_collection tests/development_publication \
  tests/tom/test_dataset.py tests/tom/test_boundaries.py \
  -k 'not off_diagonal_readout_can_memorize_controlled_public_fixture' \
  -p no:cacheprovider --maxfail=3

PYTHONDONTWRITEBYTECODE=1 /Users/name_yuxiao/anaconda3/envs/3wd/bin/python -B \
  scripts/run_observer_dtrs_decisions.py controlled \
  --destination /private/tmp/observer-dtrs-d3d1-controlled-20261015
```

人工fixture：26局/27条，25层中一层含两局、局长不同，用原固定抽样规则生成2000次人工索引，
不声称人工索引就是1500局D2索引。完整720/180/360/192单元及受限支持/抽样封存通过。
受控完整 manifest：`979f4d239a5d03c7fa74f7bbaad5baac9907b92d5463b65172952600535296aa`。
同源重复执行产生同摘要；中断、缺失、截断、重算摘要后的统计/身份冲突和跨层索引均有拒绝测试。
人工结果与原 D3b evaluator 比较，覆盖不同权重、子组、零类别、标签/支持反事实、实际损失与风险界。

保留开发过程的区别：首轮一个反事实测试在对称成本/人工平衡样本上总风险恰好相同，
改为只翻转一个终端样本标签，以验证不变决策与变化损失，未修改生产决策语义。
首次CLI用 `/tmp` 被既有 durable artifact 的符号链接检查拒绝；改用实路径 `/private/tmp` 后通过，
没有放宽已有 artifact 规则或清理/修复输入。

### 真实输入只读组装

实际从已下载 D1c/D2 重新执行独立标签、来源、外层与支持校验，并组装不可变输入。
在该进程中将纯 decide 及运行器的 decide 都替换成报错函数；整个过程未分类。
1500局/22685条/25块/8映射均通过，未发布真实区域、风险或模型排名。
原D2 Bootstrap字节 SHA-256：
`33e94ab2b2875ff7f47de212062a628de613de37e4b8f3ff0b887eb479f0a3da`。
验收 JSON 保存新合法/监督/支持映射摘要和原来源锚。

这仍不包括最初 canonical 原始轨迹全链重建、2400个内层候选拟合回放、上游重训或新独立测试。
历史训练 RNG 的全量 publication 摘要依赖和架构选择独立性限制继续保留。

## 8. 冻结前需审阅的选择

推荐采用本协议的主模型/对照、游戏等权、主 TERMINAL_NEG_FIRST、DEFER 精确并列敏感性、
45场景完整报告与原2000次索引复用。这些与本轮给定研究方向一致。
新增数值/范围建议仍需确认：

1. Kish/游戏数描述性支持带，以及概率半径0/.01/.025/.05。
2. 概率和成本扰动限定主EQ1/主策略；成本敏感性完整报告网格邻接边；覆盖/子组仅点统计。
3. 维持逐点风险区间和后验规范性研究解释，不指定唯一主矩阵或作全局确证性显著声明。

输入已满足拟议正式评价的来源要求，工程具备进入协议审阅与源码固定准备的条件。
**本轮尚不能直接正式运行**：协议未冻结、新生产源码未提交，原 guard 的拒绝已实际验证。
审阅后先固定协议及新的D3源码版本，另行授权正式执行；不使用旧D2 OID替新D3代码背书。
