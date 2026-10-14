# Observer-DTRS v1 D2b-1：冻结协议、整局划分与估计器受控验收

日期：2026-10-14。阶段结果：**协议已冻结、split 已发布并重建验证、EQ 估计器受控验收通过**。
没有运行正式全量嵌套 CV、模型性能比较、正式校准/置信区间或新 ToM 训练。
目标事件、成功空报告、七座位 Q 与合法属性语义保持 D0–D1c 定义。

## 1. 冻结身份与相对 D2a 的补足

执行配置：`configs/observer_dtrs/d2-execution-v1.json`。

- schema：`observer_dtrs_d2_execution_protocol_v1`；状态：`FROZEN_NOT_RUN`。
- canonical protocol SHA-256：`c76940b446ffaa3dc9e873620fe775a3b64db8544d0c5190a49778a2e48a6c20`。
- parent proposal canonical SHA-256：`d393bca91c584a17f5d7a84c804b51d0d012ad4f3651cd5bf3b1e23e60c7f53c`。
- 原 D2a 六个文件保留为历史草案和实际诊断，不把草案悄悄改成已执行结果。

相对草案：冻结版本/状态/父摘要；明确支持统计、根先验、浮点边界、候选并列规则、
单箱等价、Log Loss 数值处理、配对要求与模型访问边界；将连续 r Logistic 登记为
次要基线。它的数学设计/惩罚网格已登记，优化器实现及验收留给 D2b-2 正式比较前。
OBS-EF 仍作为经验频率消融登记，不是旧 Phase2 模型；本轮没有执行该消融。

一处显式补足：记录等权敏感性分析用其自身的记录等权内层 Brier 选参；主方案仍用
游戏等权 Brier。两者使用相同游戏划分/网格/并列规则，结果分别标记。
这补足了草案未单列的聚合口径，没有改变主目标、主网格或监督语义。

## 2. 统计定义及隔离

令训练游戏为 T，每局完整合法样本数 n_g。主权重 u_gj=1/n_g；记录等权敏感性用
u_gj=|T|/N_T，使总质量均为 |T|，λ 始终以 game-equivalent mass 为单位。
不在每个粒内重新按游戏归一化，不做类别重采样。

对粒 C：

\[
E_C=\sum u_{gj}1_C,\quad S_C=\sum u_{gj}Y_{gj}1_C,\quad
 a_{gC}=\sum_j u_{gj}1_C,\quad H_C=E_C^2/\sum_g a_{gC}^2.
\]

零支持时 H_C=0，distinct games 为 a_gC>0 的游戏数。行数、游戏数、正/负类游戏数、
E、S、游戏质量平方和及 H 均保存。H 是支持诊断，不称独立伯努利有效样本量。

根节点：(S+.5)/(E+1)，正负各 .5 单位伪质量。其他节点：

\[
p_C=(S_C+\lambda p_{parent})/(E_C+\lambda),\quad
\lambda\in\{.25,1,4\}.
\]

路径：global → nW → phase×nW → r-bin。这是加权层级收缩规则，不称精确层级
贝叶斯后验。空的整个训练集报错；未见 phase×nW 回退已观察 nW，未见 nW 回退根。
合法 phase 之外的运行时别名报错。单类别粒保留并收缩，不按标签拆、删或合并。

EQ0 使用 phase×nW。EQ1 请求箱数 {1,3,5}，当前训练集各状态内按加权左逆经验 CDF
分箱；相同 r 不拆、重复切点去重、最大端点切点删除。箱约定 (-∞,c1]、(c1,c2]、…。
测试值超训练范围只进入端点箱。继承 D1 的 Q 求和容差：r 可至 1+1e-6，不裁剪或
重归一化；s/m 各自仍须在 [0,1]，并校验 r=s+m、单狼 m=0/dW=None。

叶门槛：distinct games≥10 且 H≥5（容差 1e-12）。只依据训练 X/权重，从最左失败
叶开始，与质量较小邻居合并，平局选左；直到支持满足或只剩一叶。
**EQ1 请求一箱或合并至一箱时，直接使用 b=phase×nW 父层；同 λ 下与 EQ0 预测完全
一致，不额外收缩。** 状态身份保留，不把回退到相同概率的两个状态称同一等价类。

EQ0 的 3 个 λ 与 EQ1 的 9 个 (B,λ) 候选使用完全相同内层游戏、权重和评分。
EQ0 网格正是 EQ1 的 B=1 子集。内层最小分数+1e-8 范围内，先少箱、后大 λ；
候选完整覆盖且内层验证游戏相同才允许选参。选后在外层训练集重新学习全部边界。

主二分类 Brier 为 (p-y)^2；自然 Log Loss 仅评分时夹到 [1e-12,1-1e-12]，记录
触发行和错误端点数。原概率用于 Brier/校准，模型输出不裁剪。平滑概率必须有限且
严格位于 (0,1)。先逐局平均再等权平均；记录等权敏感性单列。

预测接口仅接受 ObserverFeatures；TrainingRow.y 是显式离线标签。报告集合、
成功空报告标记、训练 q target、Theta_AC 均不进入预测条件。成功空报告保留既定
有效集合和 Y，仅在将来结果中作审计子组，不能作为在线分流标记。

## 3. 整局划分及真实发布

服务器已发布：

```text
/data/yuxiao/Untrusted_Network_Simulation/observer-dtrs/protocols/observer-dtrs-d2-execution-v1
protocol manifest: f6d12824a5b74b45faf00fe3efea3b8f21db9a78ad06c158c8dc1a355f2fc5eb
implementation snapshot: 02bfea7f6f90b171d1b7bb0ecc3720dfff584a1808103d6e0ea45ccc4911ab31

/data/yuxiao/Untrusted_Network_Simulation/observer-dtrs/splits/paper-development-qwen35-9b-1500-v1-d2-splits-v1
split manifest: 32e30246cbadec8240f47291422a791b4419482e1386b4d8b570344a2e45acc5
outer assignment: 99c50a8125559022823f6084c8f456601f93ed9c0620f6b0ceef7cf0b17a80f4
```

原数据集仍为 `paper-development-qwen35-9b-1500-v1-observer-dtrs-v1`，摘要仍是
`d25537be18da93ddc2f49be695f876db2edef5f31904fea0999a292f491abfa3`。

每个 ToM 来源折 300 局，各自外层 5×(240/60)，内层 3×(160/80)。外层共 25 块，
内层共 75 块；1500 个唯一游戏各进外层测试一次，测试样本总计 22685。
每个内层只含其外层训练游戏；其三个验证块覆盖外层训练游戏一次。
全部 ToM 来源折保持分离，不把不同 f_k 的映射或 bin ID 混用。

排序规则保持 D2a：canonical JSON `[domain,20261014,tom_fold,parent_outer_or_null,game_id]`
的 SHA-256 排序后 rank mod K。dataset digest 只作身份校验，不参与数值控制；不按
标签/报告状态分层、不搜索 seed。split 发布仅读 source manifest 元数据，不读监督。
读入会从来源 manifest 和固定协议重新计算全部分配并逐字节比对。

使用已有 publish_artifact/verify_artifact 的 canonical JSON、SHA-256、file table、
原子不覆盖发布与同内容复用。protocol artifact 另存十个实现/配置/验收源文件的
文本快照，明确“尚未提交的源文件”，不把 Git 基点冒充包含新代码的 commit。
source snapshot 仅用于审计，读取器不执行 artifact 中存储的源码。

首次 SSH 连接曾在获得执行结果前关闭；先只读确认两个目标仍不存在，再重试发布。
发布后在独立 SSH 会话再次完整验证 protocol、split、D1 数据和覆盖。服务器采用
封存 D1 实现加 stdin 内存加载的本地新源码，nice=10、idle I/O、Python -B；仅新增
上述两个 artifact，不修改 checkout、predictor、vLLM 或采集服务。

实际结果及逐块测试样本数保存在 `observer-dtrs-v1-d2b1-release.json`。

## 4. 实现接口

- `d2_protocol.py`：固定配置 digest、FrozenProtocol、协议发布/读取、实现快照。
- `splits.py`：game_partition/build_splits/partition_games、split 发布/读取/完整重建。
- `estimation.py`：TrainingRow、CellStats、train_edges、fit_model、FittedModel、validate_model。
- `estimation_offline.py`：fit_partition，仅开启 split 指定完整训练游戏的合法分区及其
  supervision；校验 ToM fold。publish_model/read_model 沿用现有 artifact 机制。
- `scoring.py`：逐游戏及记录评分、配对差值、完整候选网格选参。
- `scripts/manage_observer_dtrs_d2.py`：publish/verify；没有正式拟合入口。
- `scripts/check_observer_dtrs_estimator.py`：内置两局人工样例，独立验收和模型往返读取。

```python
from werewolf.observer_dtrs.d2_protocol import frozen_protocol
from werewolf.observer_dtrs.estimation_offline import fit_partition

# D2b-2 才调用正式分区拟合；本轮未对真实数据调用。
model = fit_partition(dataset_path, split_path=split_path,
    expected_split_digest=split_digest, protocol=frozen_protocol(),
    tom_fold=0, outer_fold=0, inner_fold=0,
    model_id="OBS-EQ1", bins=3, smoothing=1)
p = model.predict(legal_features)  # 不接收监督或报告
```

模型 artifact 保存协议、数据、split、来源 fold/外层/内层、完整训练游戏行数、
上游 origins、实现摘要、训练记录指纹、参数、边界、各层支持与概率。受控 fixture
必须显式标记，不能冒充正式 dataset/fold。读取验证统计、父子质量守恒、概率重算、
门槛与 frozen 参数。它不冒充对原训练 Y 的重新重建；完整重放仍需原训练分区。

## 5. 人工样例和自动化验收

两局样例：一局单狼 discussion 为正，一局双狼 PK 为负，每局一条。
根 p=.5；对应 nW 层分别 .75/.25；b 层分别 **.875/.125**。EQ1 单箱与 EQ0 相同。
两条 Brier 均为 .015625。这只是公式验收，不是泛化性能。

另验证：九条正例的一局加一条负例的一局，记录基率 .9、游戏基率 .5；根节点按
游戏权重仍为 .5，按归一化记录权重为 2.3/3。30 局、r∈{.1,.4,.8} 各十局时切点
恰为 [.1,.4]；等于切点归左；40 局五请求箱经支持合并得到 16/24 两叶。

自动测试覆盖：标签/验证属性反事实隔离、完整训练人口、X-only 合并、Kish 门槛、
单箱等价、零/单类与未见 nW/phase 回退、非法属性和非有限概率、游戏/记录权重、
配对候选选参、严格来源身份、真实 artifact 往返/冲突/损坏/重哈希篡改及 CLI 入口。

最终相关回归：**313 passed, 1 deselected in 51.15s**。

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/name_yuxiao/anaconda3/envs/3wd/bin/python -B -m pytest -q \
  tests/observer_dtrs tests/canonical_collection tests/development_publication \
  tests/tom/test_dataset.py tests/tom/test_boundaries.py \
  -k 'not off_diagonal_readout_can_memorize_controlled_public_fixture' \
  -p no:cacheprovider --maxfail=3
```

排除项仍为旧 ToM 小模型训练测试，不声称全仓库测试通过。
先前一次回归发现三个新模块没有可执行入口（311 passed/1 failed）；补上真实受控
验收脚本后，针对性 48 项通过，随后上述回归全部通过，没有修改旧测试来绕过约束。
服务器 Python 3.10.21 另执行两局纯内存数学验收，没有发布模型或读取正式样本拟合。

## 6. 不确定性、历史来源限制与 D2b-2 门槛

bootstrap 固定以完整游戏为单位，在 (ToM fold,outer fold) 25 块内各抽 60 局，
配对索引同时用于两模型/指标。PCG64、seed=20261014、2000 次、95% percentile
linear quantile；按 replicate→(k,j) 字典序→game_id 排序抽样，保存索引摘要。
记录等权敏感性也抽完整游戏，按抽中游戏的重复行数重新计算比率，不逐行抽样。
区间只描述**固定已拟合预测下病例组成的近似稳定性**，不包含共享 ToM、重叠 CV
训练及选参的完整方差；不能称完整端到端算法风险的严格置信保证。
bootstrap/校准聚合器本轮仅冻结规则，实际实现与验收留给 D2b-2，不生成区间。

历史 ToM training_rng 包含全量 publication 所绑定 preparation 摘要的限制继续
明确保留：fold 内方案不能消除这条哈希→数值 seed 依赖，也未证明架构选择独立。
机械依赖不证明存在乐观偏差，本轮没有量化影响。当前仅为固定历史产物条件下的
下游开发验证，不声称完全独立的端到端泛化。

**具备进入 D2b-2 开发的条件；尚不能跳过运行器验收立即启动正式全量实验。**
D2b-2 需完成成对嵌套执行/恢复、内层选参和外层重拟合的集成验收；实现已登记
Logistic 及频率消融、校准/分组报告/配对 bootstrap；核对来源、候选与覆盖后，
才能运行已冻结网格。不得据正式外层分数改边界、网格、失败处理或监督语义。

## 7. 推荐提交边界（本轮未提交）

1. **D1a–D1c：21 文件**。九个原 observer_dtrs 模块、七个原测试/fixture、两个 D1
   脚本、三份 D1 文档。建议 `feat(observer-dtrs): publish audited offline dataset pipeline`。
   原九模块及发布脚本十文件已与正式数据源码快照逐项 SHA-256 比对一致。
2. **D2a：6 文件**。proposal config、协议/方法/诊断三文档、新诊断脚本及其测试。
   建议 `docs(observer-dtrs): specify conditional probability validation protocol`。
3. **D2b-1：12 文件**。五模块、冻结配置、两个脚本、两个测试、本报告与发布记录。
   建议 `feat(observer-dtrs): freeze splits and implement weighted probability estimator`。

完整路径分组见 release JSON 的 stage_file_groups。不要把整个 observer_dtrs、tests 或
scripts 目录一起暂存到 D1，以免混入 D2。D2 新文件使“当前全源码快照”自然改变，
但不会回写或替换 D1 已封存快照。当前分支仍为 three-way，HEAD 仍为 d8397ce56c65d6d46b24a6b53f0b71431c04267b；
未暂存、commit、push、merge/rebase，也未恢复旧 stash。
