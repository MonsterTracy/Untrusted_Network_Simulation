# Observer-DTRS v1 D2b-2：嵌套运行器、次要基线与统计分析验收

日期：2026-10-14。本轮只做实现、受控验收和服务器只读/内存环境检查。
**未对正式 1500 局数据执行任何候选拟合或正式嵌套实验，未发布模型比较结论。**
D2 冻结 JSON、已发布 protocol/split、D1c 数据和 D0 事件语义均保持不变。

## 1. 实现文件与入口

新增五个生产模块：

- `werewolf/observer_dtrs/validation_data.py`：合法 PredictionInput 与显式离线 targets 分开读取；
  FormalData 绑定已验证数据及 split；ControlledData 仅用于明确标记的人工样例。
- `werewolf/observer_dtrs/baselines.py`：OBS-LR-r、OBS-EF、合法批预测及分箱合并审计。
- `werewolf/observer_dtrs/validation_artifacts.py`：不可变 run contract、原子 checkpoint、
  来源/运行时快照、排他锁和失败记录。
- `werewolf/observer_dtrs/nested_validation.py`：25 块成对嵌套执行、候选/外层恢复、
  全覆盖 seal，以及只读跨引用/人口/评分/统计重建。
- `werewolf/observer_dtrs/statistical_analysis.py`：总体/子组评分、校准、配对差值、
  完整游戏分层 bootstrap。没有 ToM 重训或三支区域逻辑。

新增 CLI：`scripts/run_observer_dtrs_validation.py`：

```bash
# 受控 fixture，75 个明确人工游戏、150 条样本；不接触正式数据
PYTHONDONTWRITEBYTECODE=1 python -B scripts/run_observer_dtrs_validation.py controlled \
  --destination /absolute/path/to/new-controlled-run

# 只读检查：核对 checkpoint 引用、候选/人口；完整结果重算评分、summary 和全部抽样索引
python -B scripts/run_observer_dtrs_validation.py inspect --destination /absolute/path/to/run
```

新增测试：`tests/observer_dtrs/test_secondary_analysis.py`、`test_nested_validation.py`。
另新增本文和 `observer-dtrs-v1-d2b2-acceptance.json`；此前三个阶段共 39 个文件没有修改。

## 2. 隔离、配对与完整候选

正式来源只允许 frozen protocol 所钉住的数据集，以及调用者提供摘要并重建验证的
整局 split；每个来源 ToM fold 内独立运行 5 个外层、每个外层 3 个内层。
运行器没有“换 seed”“删游戏”“跳过失败模型”或扩展网格的选项。

每个内层分别开启完整训练游戏及验证游戏。只有训练游戏的合法属性和显式 Y 进入
拟合；合法预测接口不接收 targets、空报告标记、真实支持集合或 Theta_AC。
训练/验证/外层测试游戏交集、ToM fold、完整行数均核对。

每个权重方案、每个外层的内层候选固定为：

| 模型 | 候选 | 3 个内层总拟合数 |
| --- | --- | ---: |
| OBS-EQ0 | B=1，λ∈{.25,1,4} | 9 |
| OBS-EQ1 | B∈{1,3,5}，λ∈{.25,1,4} | 27 |
| OBS-LR-r | τ∈{.001,.01,.1,1} | 12 |
| OBS-EF | 无独立网格，绑定选定 EQ1 | 0 |

所有候选验证相同的完整游戏。保存每候选/每内层的模型、逐样本预测和逐游戏分数，
汇总三个内层游戏后才选择。EQ0/EQ1 沿用 D2b-1 的冻结选择函数；LR 使用同一
Brier 测度及 1e-8 并列容差，并列选更大 τ。
游戏等权为主；记录等权敏感性使用同一划分/网格，按记录测度拟合、选参和报告。

选参后在完整外层训练游戏重新拟合（包括标准化、分箱、合并和层级统计）。
**两种权重、所有模型的外层预测均封存以后，才首次开启该块测试 targets 评分。**
测试失败不会引发新超参数、重抽划分或降级模型。PK 单类别组保留。
训练分箱记录初始边界、确定性合并轨迹、实际边界/支持，预测记录粒层级/回退层级。

## 3. Logistic 和经验频率

OBS-LR-r 使用六列：全局截距、三个 phase×nW 虚拟变量（reference=discussion×1）、
两个按 nW 区分的标准化 r 斜率。均值/方差只从当前训练游戏的指定权重计算；
零方差 z=0。目标严格为：

\[
L(\beta)=|T|^{-1}\sum u_{gj}\operatorname{logloss}(Y_{gj},\sigma(X\beta))
 +\frac\tau2\sum_{q\ne intercept}\beta_q^2.
\]

确定性零初始化、float64 damped Newton；记录求解器设置、迭代、梯度和目标。
不惩罚全局截距。max_iter=200，梯度无穷范数≤1e-9；Armijo 线搜索最多 50 次。
接近机器精度、预期下降≤32eps 时，只有目标变化在舍入容差内且梯度严格改善才
接纳步长，最终仍必须满足原梯度阈值。单类别训练返回 (S+.5)/(E+1) 根常数。
不收敛或非有限参数立即失败，不更换 τ 或模型。
本地缺少 SciPy，采用 NumPy 实现；没有新增 SciPy/scikit-learn 训练依赖。

OBS-EF 按 D2a 的“同一训练粒结构”实现：使用选定 EQ1 的训练分箱和权重，各观察
粒输出未平滑 S/E。未见粒向最近观察父层回退，根也用经验 S/E；保留真实 0/1。
它记录 parent_EQ1_digest，不独立调参，不反向影响 EQ1 选择。它是收缩消融，不是
另一个经过独立最佳化的分类器。这里明确落实既有消融定义，没有增加参数范围。

## 4. 评分、校准和 bootstrap

Brier 为 (p-y)^2，自然 Log Loss 仅评分时夹到 [1e-12,1−1e-12]；原始预测不裁剪。
记录 clipped_rows、wrong_endpoint_rows，尤其不以有限评分掩盖 EF 的错误端点。
逐游戏评分及模型间同游戏差值均保存；主要对比 EQ1−EQ0，次要对比明确标记。

输出总体、phase×nW、成功空/非空审计组，以及每 ToM fold 的分数。子组使用主
测度的条件比率，游戏权重分母保持整局 n_g，不改为子组内 n_gH。
成功空报告仅是离线审计字段，不进入任何拟合/预测特征；继续保留既定 Y。

校准使用固定 [0,.1),…,[.9,1]，包含 0 和 1 的正确端点。输出均值预测、正例率、
行数、游戏数、质量、整体 bias 和描述性 ECE；空箱为 null。不做后验再校准。
支持≥20 游戏才考虑校准置信带，重采样有效比例<95% 的点不给区间。

bootstrap 严格为 PCG64/seed=20261014/2000 次，replicate-major，其内 (ToM fold,
outer fold) 字典序，各块 game_id 排序；正式每块有放回抽 60 个完整游戏。
同一整局索引用于全部模型、权重、指标和子组；记录等权按重复游戏记录数重算比率。
保存 int64 little-endian 索引矩阵（正式形状 2000×1500）、SHA-256、game order
摘要、NumPy 版本。95% percentile，quantile method=linear。
ControlledData 按其每块 3 局抽样，明确为小样例，不能冒充正式 60 局块。

区间解释保持冻结定义：固定预测下病例组成变化的近似稳定性，不包含共享 ToM、
重叠 CV 训练、选择的完整方差；不称端到端算法风险的严格置信保证。
**历史 ToM preparation/publication hash 影响训练 RNG，以及架构选择独立性未证实的
限制，强制保存在 contract 和统计 summary；不得从后续结果中省略。**

## 5. 产物、恢复与异常

沿用 publish_artifact/verify_artifact、canonical JSON、file-table/SHA-256 和原子
不覆盖发布，不另外定义不兼容的可变结果文件。整套 run 为受限离线研究产物，
含监督审计和拟合统计，不能作为绕过 D1 gate 的在线合法输入接口。

```text
RUN/
  runner.lock
  contract/                 record.json + sources.json + protocol.json
  blocks/K-J/
    game_equal/            # record_equal 同结构
      inner/H/0...15/      模型、验证预测、评分、分箱审计
      selection/           完整候选逐游戏 Brier、选择、候选摘要
      outer_predictions/   所有四模型/逐样本合法预测，先封存
      evaluation/          之后读取测试监督生成的评分与监督摘要
    complete/              成对块 seal、逐样本审计、子产物引用
  failures/000000.../      不可变异常/阶段记录，保留历史失败
  complete/                仅全部25块通过后发布
    record.json            统计summary和全部块引用
    bootstrap_indices.int64le
```

contract 钉住协议、dataset/split、原始来源、游戏/样本人口、代码文本/HEAD、Python/
NumPy/BLAS/线程环境；数值控制不使用这些来源哈希。正式入口要求显式提交 OID 与
当前 HEAD 一致、全部生产源码已跟踪且无改动。本轮脏工作区只允许 controlled。

排他 flock 防止同一 run 并发写入。每个已完成候选可复用；resume 重新核对原始
分区身份、模型 schema/参数、预测及评分，一处不一致即停，不覆盖/自动修复。
选择与外层产物也必须逐字节一致才能复用。中断只留下可验证 checkpoint 和失败
记录，单块返回 PARTIAL_NOT_FULL_EXPERIMENT；完整 seal 要求 25 块、每局一次、
每条样本一次及全部四模型/两权重覆盖。残留 staging 目录不会当已完成结果，不自动清理。

inspect 仅读取，不执行 artifact 中保存的源码；核对候选/选择/外层引用和完整人口，
重建外层评分、summary、全部 bootstrap 字节。它不冒充从训练 Y 重新拟合模型；
resume 会额外回源检查原始分区。应使用 inspect 验证完整性，不只检查 complete 路径存在。

## 6. 实际验收

相关回归：**324 passed, 1 deselected in 64.51s**。

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/name_yuxiao/anaconda3/envs/3wd/bin/python -B -m pytest -q \
  tests/observer_dtrs tests/canonical_collection tests/development_publication \
  tests/tom/test_dataset.py tests/tom/test_boundaries.py \
  -k 'not off_diagonal_readout_can_memorize_controlled_public_fixture' \
  -p no:cacheprovider --maxfail=3
```

排除项仍为旧 ToM 小模型训练测试，不声称全仓库测试通过。
新增 11 项验收覆盖：Logistic 解析梯度/Hessian 与有限差分、六列设计/零方差/
单类别/顺序确定性；EF 端点与合并轨迹；bootstrap 分层/配对/确定性；校准端点、
空报告组、游戏权重；测试监督开启时序/反事实隔离；中断恢复/不重拟合；损坏及
重哈希重复预测拒绝；正式未提交源码 guard；完整 25/75/2400 候选覆盖及只读统计重建。

一个控制案例暴露了 Newton 在约1.32e-9梯度处的目标舍入停滞。已修复求解器舍入
处理，保留最终1e-9梯度验收与全部冻结研究参数；失败 checkpoint 没有被跳过。

独立 CLI 等价受控运行实际构造 75 局/150 样本、2400 个内层候选 checkpoint：
构建约4.78秒、只读重建约1.25秒、重复执行约1.67秒，完整 seal 摘要一致。
实际路径和摘要见 `observer-dtrs-v1-d2b2-acceptance.json`。它是临时本地工程产物，
不是正式科研数据或模型成绩。

服务器只读核对：Python3.10.21/NumPy2.2.6，三个原 artifact manifest 摘要未变；
主 checkout 尚无新运行器。vLLM PID3030144 等进程仍在运行，本轮未操作服务。
另在服务器 nice10/idleIO、stdin 内存源码下，仅对12个人工游戏拟合四个τ，并对
人工固定预测执行2000次 bootstrap（约0.36秒）；不读取真实样本拟合、不写服务器文件。
本轮没有部署、切换或清理任何服务器工作树。

## 7. 正式命令、负担与启动前提

**先由用户确认提交，再在不被现有任务使用的执行 checkout 部署同一提交。**
不能直接在尚缺运行器、可能承担任务的服务器主目录切分支。下面命令是下一轮
执行模板，本轮未运行；SOURCE_OID 应替换为确认后的实际提交，不能用旧 d8397ce。

```bash
# 在已安全部署并确认版本的执行源码 checkout 中运行；不是要求切换现有任务目录。
D2_SOURCE_OID='<confirmed-full-commit-OID>'
D2_RUN_ROOT='/data/yuxiao/Untrusted_Network_Simulation/observer-dtrs/validation-runs/d2-v1-run1'
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  ionice -c 3 nice -n 10 /data/yuxiao/envs/untrusted-network-simulation/bin/python -B \
  scripts/run_observer_dtrs_validation.py run \
  --destination "$D2_RUN_ROOT" \
  --dataset /data/yuxiao/Untrusted_Network_Simulation/observer-dtrs/datasets/paper-development-qwen35-9b-1500-v1-observer-dtrs-v1 \
  --protocol-artifact /data/yuxiao/Untrusted_Network_Simulation/observer-dtrs/protocols/observer-dtrs-d2-execution-v1 \
  --expected-protocol-manifest-digest f6d12824a5b74b45faf00fe3efea3b8f21db9a78ad06c158c8dc1a355f2fc5eb \
  --split /data/yuxiao/Untrusted_Network_Simulation/observer-dtrs/splits/paper-development-qwen35-9b-1500-v1-d2-splits-v1 \
  --expected-split-manifest-digest 32e30246cbadec8240f47291422a791b4419482e1386b4d8b570344a2e45acc5 \
  --source-revision "$D2_SOURCE_OID" --storage-profile configs/server.json
```

恢复时重复相同命令，源码/数据/参数/运行时不一致会拒绝复用。加 `--block K J`
可作明确标记的有限块工程执行，但不会生成全量完成 seal。本轮没有对真实数据
使用此选项。完成后 `inspect --destination "$D2_RUN_ROOT"` 只读验证。

正式负担：两权重共 2400 个内层拟合，150 个外层重拟合（EQ0/EQ1/LR），另50个
EF派生；约2,903,680条内层验证预测、181,480条外层模型预测；bootstrap索引约24MB。
算法只用CPU，不调用ToM推理或训练，不需要GPU。建议单作业/BLAS单线程。
工程预估预留约1–2GB产物空间及约2GB内存，运行时间可能为数十分钟，受实际
行数、字符串、文件系统fsync及服务器负载影响；这是预算，未以真实全量计时验证。
受控小样例不能作为正式运行时或性能外推证明。

结论：工程基础和受控集成验收已具备下一步正式实验条件；当前仍须先完成用户
确认的源码提交/服务器安全部署。正式1500局统计表现、真实执行耗时及该提交
在服务器上的整条运行器集成尚未验证，不把本轮人工成绩当研究结论。

## 8. 提交边界

此前约定三组保持原样并通过当前组合回归：D1a–D1c 21文件、D2a 6文件、D2b-1
12文件，适合依次提交。已逐项核对D1/D2a保存指纹，并核对D2b-1封存的十文件
源码快照摘要；与正式数据和发布身份一致。该结论不是声称已checkout三个拟议
commit分别运行测试——目前没有创建这些commit。

本轮新增十文件另成D2b-2阶段，建议：
`feat(observer-dtrs): add resumable paired nested validation and analysis`。
完整分组/指纹见 acceptance JSON。不要整目录暂存把后续阶段混入前一组。
当前仍为three-way、HEAD d8397ce56c65d6d46b24a6b53f0b71431c04267b；
没有git add/commit/push/merge/rebase，也没有操作旧stash或其他worktree。
