# Observer-DTRS v1 — D1c 完整离线数据集与来源审计

本阶段将 D1a 的样本/监督与 D1b 的 Q/连续属性整合为不可变研究 artifact。
沿用原目标事件、成功空报告语义、绝对七座位概率质量及 canonical PRE 身份；
不确定分箱、等价类、平滑、概率模型或新训练/验证划分。它们属于 D2。

## 实现与接口

- `werewolf/observer_dtrs/dataset.py`：合法分区、完整游戏读取、Q/属性重算与样本人口检查。
- `werewolf/observer_dtrs/dataset_offline.py`：完整构造、原子发布、显式监督读取、覆盖统计、
  元数据证明核验和回源重建。
- `scripts/manage_observer_dtrs_dataset.py`：`publish / verify / audit-sources` 三个入口。
- `tests/observer_dtrs/test_dataset.py`：发布/读取/重建、损坏、权限分区及跨游戏来源回归。
- D1b `oof.py` 新增 `load_qwen3_oof_bundle`：返回完整五折来源和覆盖，包括没有请求
  样本的 fold。原 `load_qwen3_oof` 返回接口与数值语义保持兼容。此项扩展避免 D1c
  根据“有样本的折”错误推断全部来源；没有修改 D1a 标签或旧模块。

```python
from werewolf.observer_dtrs.dataset import open_dataset
from werewolf.observer_dtrs.dataset_offline import (
    open_supervision, verify_dataset, audit_dataset_sources,
)

view = open_dataset(dataset_path, expected_digest=dataset_digest, game_ids=(game_id,))
view.rows  # LegalDatasetRow(sample_id, context, probability, features)
offline = open_supervision(view)  # 显式受控离线操作，仅打开同一游戏分区
verify_dataset(dataset_path, expected_digest=dataset_digest)  # 全部文件与内部审计
audit_dataset_sources(dataset_path, publication_path=publication_path,
    evaluation_path=oof_root, expected_seal_digest=oof_seal,
    expected_digest=dataset_digest)  # 原始来源完整重建，逐字节比较
```

`open_dataset` 不导入或读取 supervision、role sidecar、ToM 模型、Phase2 或 OOF 文件。
它只对所选完整游戏读取合法 payload；没有选择其他游戏监督的隐藏通道。
`open_supervision` 只接受已打开的 view，读取范围与该 view 相同。
这些是项目内的接口/数据分区约束，不代替服务器文件系统访问控制。

## Artifact 契约

artifact type 为 `observer_dtrs_offline_dataset`，schema 为
`observer_dtrs_offline_dataset_v1`。使用现有 `publish_artifact`、canonical JSON/JSONL、
manifest/file-table/SHA-256、原子不覆盖发布、相同内容复用和冲突拒绝机制。

```text
manifest.json
legal/games/<game_id>.jsonl
restricted/supervision/<game_id>.jsonl
restricted/provenance/<game_id>.jsonl
restricted/audit.json
restricted/upstream_manifests.json
restricted/implementation_sources.json
```

全部 publication 游戏均出现在 manifest，零样本游戏保留空 JSONL 与覆盖项。
sample_id 是数据集版本及 `(game, PRE, acting_wolf, observer, prefix_digest)` 的
canonical SHA-256，不依赖监督、Q 或标签分布。每个 PRE 必须按座位顺序完整枚举
存活非狼观察者，canonical step 顺序不变；不套用 Phase2 竞争目标过滤。

合法行保存 context、完整七列预测概率及原始连续 `s/m/r/dW/phase/nW`。
读取时重新校验 Q 并计算属性，不接受相互不一致的预存特征。
单狼保留 `m=0`、`dW=null`，不删除死亡列或重新归一化。
Q 是预测的规范化嫌疑分配质量；`r` 不是狼人身份概率，也不是目标事件的条件概率。

监督行保存 D1a 的原始支持、有效支持、Y、report ID/digest/attempt。
正式数据集仅允许成功且有效的监督；成功空报告展开为六名非自身玩家，失败、缺失
不得形成正式标签。完整角色映射不复制进新数据集。

来源行通过 sample_id 严格联结合法行与监督，并分别保存 D1a SampleSource 和
D1b OOFSource。每行的 publication/sidecar/fold/game-bundle/game-record/模型来源
必须与全局元数据匹配。原 ToM fold 仅记录来源，不解释为 D2 已选择的划分。

manifest 保存以下信息：

- context/sample/label/feature/OOF/audit 各版本及明确的访问策略。
- publication、collection/game-set/sidecar/fold 摘要与五折 Qwen3 预测完整来源。
- 按原 publication 顺序的游戏、fold、样本数、狼人 PRE 数及样本 ID 序列摘要。
- 原 canonical PRE、完整 OOF、最终样本的覆盖计数。
- 源码基点、实际源码快照摘要、Python/pydantic 版本及输入路径定位信息。

受限元数据证明保存 publication/fold、evaluation/training/preparation、五折
prediction/terminal manifests；不复制模型权重、优化器、训练 target、真实完整
sidecar 或 bootstrap draws。既有 canonical fold 生成/验证函数用于重新核对划分。

当前代码尚未提交，所以除了 Git 基点，还保存实际 `werewolf/**/*.py`、发布脚本与
`setup.py` 文本快照。快照只作为审计 payload，读取器不会执行其中源码。
构造前后源码必须一致；基点不冒充包含本轮修改的 commit。

## 三层核验及其限度

1. `open_dataset`：manifest/inventory、所选合法文件大小/摘要/格式、身份、Q、
   连续属性、PRE 观察者人口与游戏覆盖。**不宣称已验证受限文件内容。**
2. `verify_dataset`：打开全部分区，验证所有 payload、监督有效集合/Y、来源联结、
   元数据证明、源码快照和全部覆盖统计。无需原始输入文件。
3. `audit_dataset_sources`：重新调用 D1a/D1b，从原 canonical publication、受控
   sidecar 和完整五折 OOF 重建，比较所有数据/审计/来源文件的完整字节。

前两层不能单独证明“重新编造但内部一致的 Q/标签”真的来自原始文件。测试专门
构造了数值合法且重算特征的反事实 Q：内部校验通过，回源重建必须失败。已知
dataset manifest 摘要应在使用时钉住；回源重建负责原始语义和完整论域证明。
完整 OOF 文件的校验范围仍是 D1b：不重新运行冻结模型或复核历史权重数值计算。

## 覆盖与类别审计

`restricted/audit.json` 保存总体、全部游戏（含零样本）、discussion/PK、nW、
fold、空/非空报告及 phase×nW 的分组统计。每组包含样本数、游戏数、狼人 PRE 数、
正类、负类、成功空报告和成功非空报告数。没有新建或选择实验评估子集。

审计计数针对观察者样本；`wolf_pre_count` 对 `(game_id,boundary_id)` 去重，
`game_count` 对有样本游戏去重。publication 总游戏数另记在 source coverage。
同一 PRE 的不同观察者可能分别提交空/非空报告，因此这两组的 PRE 数或游戏数
可以重叠；应按 sample_count 检查观察者样本分组的可加性。
这些是数据覆盖描述，不是统计模型性能、校准或论文研究结论。

## 运行方式

```bash
python -B -m scripts.manage_observer_dtrs_dataset publish \
  --publication /absolute/canonical/publication \
  --oof-root /absolute/sealed/evaluation \
  --expected-oof-seal-digest FULL_TRUSTED_OOF_SEAL_SHA256 \
  --destination /configured/artifact_root/observer-dtrs/datasets/DATASET_ID \
  --dataset-id DATASET_ID --base-revision GIT_BASE_40_HEX \
  --storage-profile /absolute/configs/server.json

python -B -m scripts.manage_observer_dtrs_dataset verify \
  --dataset /absolute/dataset --expected-dataset-digest FULL_DATASET_SHA256

python -B -m scripts.manage_observer_dtrs_dataset audit-sources \
  --dataset /absolute/dataset --expected-dataset-digest FULL_DATASET_SHA256 \
  --publication /absolute/canonical/publication --oof-root /absolute/sealed/evaluation \
  --expected-oof-seal-digest FULL_TRUSTED_OOF_SEAL_SHA256
```

storage profile 沿用 `artifact_root`；发布目录不得处于输入 publication/OOF 之内，
也不得成为其祖先。目的地已有不同 artifact 时拒绝覆盖。所有正式大文件留在
服务器 `/data/yuxiao/Untrusted_Network_Simulation` 下，不进入 Git。

## D2 尚需确定

首先确定以完整游戏为单位的外层训练/验证边界，以及原 ToM OOF 训练集与新 DTRS
验证游戏的关系；每行自身 OOF 不自动证明整个两阶段评估没有跨层信息泄漏。
再确定使用哪些连续属性/phase/nW 分层、训练集内拟合分箱与等价类、稀疏/未见类、
平滑与样本权重、模型选择及评价聚合方式。保留最终评估边界，避免用全体数据类别
统计调节模型后再宣称独立验证。

本轮没有改变目标事件，没有估计条件概率或使用六损失，也没有向正式游戏路径
接入数据集读取器或发言决策。

## 实际验收

本地分支 `three-way`，基点 `d8397ce56c65d6d46b24a6b53f0b71431c04267b`，
工作成果保持未提交/未暂存，未恢复原 stash。未改动旧 Phase2、冻结 ToM 或发言策略。

来源复核与 D1b 一致：publication 摘要
`396ea0fa3f3f03dd1dfce8b25c0162ca489b992321d0c9e6f772eb2b0c4a2acd`，
sidecar `276b6ee182bfb07639ae3ab99a5c8037b52ffb99b37f5c60f7f26a229178a2da`，
fold `754fe0e517aab9e9237f0cb5097bde0daf8a43a09518b9691cfca67724539e69`，
OOF seal `66bd044bd072a8673d4761ad043d0af3a208ae55b809ebffded2228b0371f925`。

复核服务器配置 `artifact_root=/data/yuxiao/Untrusted_Network_Simulation`，
Conda Python 3.10.21。构造前数据盘空闲约 3.25 TB。原采集进程 PID 3039215 与
vLLM PID 3030144 正在运行；新任务仅使用 CPU，nice=10、idle I/O 优先级，不调用
模型服务，不修改任何服务器 Git worktree 或冻结 predictor 目录。

本轮源码与合成测试已封存为独立 artifact（不是 Git worktree）：

`/data/yuxiao/Untrusted_Network_Simulation/observer-dtrs/implementations/56ca7fc2c64f829ffc0d846c706a1be48c33eade4c3fba1f772700c97701750d`

源码 artifact 摘要为
`48c306b979cadfa46b097acb4be4c174783a370c9557a466fbe71a7aff23e6d4`。
数据集使用的 107 个实际实现文件及基点快照摘要为
`d32ec7ea93f01d19f79cee2f0e840d06ef44281c97819e2592bc4f2421125879`。

本地实际执行：

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/name_yuxiao/anaconda3/envs/3wd/bin/python -B -m pytest -q \
  tests/observer_dtrs tests/canonical_collection tests/development_publication \
  tests/tom/test_dataset.py tests/tom/test_boundaries.py \
  -k 'not off_diagonal_readout_can_memorize_controlled_public_fixture' \
  -p no:cacheprovider --maxfail=3
```

结果 **274 passed, 1 deselected in 50.51s**。排除的是既有 ToM 小模型记忆训练测试，
不宣称全仓库测试通过。D1c 新增 26 项测试。

服务器使用封存目录、Conda 解释器及独立测试目录执行：

```bash
python -B -m pytest -q tests/observer_dtrs -p no:cacheprovider \
  --basetemp /data/yuxiao/Untrusted_Network_Simulation/observer-dtrs/test-runs/d1c-zut339r1/pytest \
  --maxfail=2
```

结果 **126 passed in 509.60s**；没有安装或改动依赖。该 basetemp 是本轮独立目录，
复跑时应选择新的临时目录，避免 pytest 清理已有测试记录。
18 个相关 Python 文件的 AST/行尾空白检查通过，CLI `--help` 正常。

正式数据集已通过原子发布和发布入口的完整回读：

`/data/yuxiao/Untrusted_Network_Simulation/observer-dtrs/datasets/paper-development-qwen35-9b-1500-v1-observer-dtrs-v1`

manifest SHA-256：
`d25537be18da93ddc2f49be695f876db2edef5f31904fea0999a292f491abfa3`。

覆盖为 1500 局、19,157 canonical PRE、134,099 行完整 OOF、5,835 狼人 PRE、
22,685 观察者样本。正式输入路径和 OOF 各折摘要沿用 D1b 记录。

独立新进程的完整读取验证通过。数据集共 4503 个 payload 文件，合计
87,843,290 字节，包含 1500 局的逐游戏审计；零样本游戏为 0。
服务器的实际实现快照摘要与本地 `d32ec7…25879` 完全一致。

| 样本分组 | 样本数 | 正标签 | 负标签 | 成功空报告 |
| --- | ---: | ---: | ---: | ---: |
| 总体 | 22685 | 19296 | 3389 | 5700 |
| discussion | 21136 | 17832 | 3304 | 5692 |
| pk_discussion | 1549 | 1464 | 85 | 8 |
| nW=1 | 4173 | 3352 | 821 | 94 |
| nW=2 | 18512 | 15944 | 2568 | 5606 |
| 成功空报告 | 5700 | 5700 | 0 | 5700 |
| 成功非空报告 | 16985 | 13596 | 3389 | 0 |
| discussion / nW=1 | 3768 | 2978 | 790 | 93 |
| discussion / nW=2 | 17368 | 14854 | 2514 | 5599 |
| pk_discussion / nW=1 | 405 | 374 | 31 | 1 |
| pk_discussion / nW=2 | 1144 | 1090 | 54 | 7 |
| fold 0 | 4542 | 3851 | 691 | 1125 |
| fold 1 | 4502 | 3813 | 689 | 1077 |
| fold 2 | 4603 | 3901 | 702 | 1057 |
| fold 3 | 4434 | 3778 | 656 | 1229 |
| fold 4 | 4604 | 3953 | 651 | 1212 |

不同分组维度分别统计。成功空报告全正来自目标事件定义，不代表 ToM 预测性能。
原始 1500 游戏的每局样本数、狼人 PRE 数、正负标签与空报告数均保存于数据集
`restricted/audit.json` 的 `by_game`，没有在 Git 中复制逐行数据或完整研究产物。

完整原始来源重建通过：`source_reconstruction=exact_match`。所有合法行、监督行、
来源行、上游元数据证明和覆盖统计均从原始输入重新生成并逐文件比较，未出现重复、
遗漏、身份错配或 fold 错配。发布、独立进程回读及回源构造合计约 634.75 秒。

独立审计凭证：

`/data/yuxiao/Untrusted_Network_Simulation/observer-dtrs/audits/d25537be18da93ddc2f49be695f876db2edef5f31904fea0999a292f491abfa3`

审计 artifact 摘要：
`cd37749441ab7e43fee4f1140166cca36131ea49b260406f64238babe9b3b8d0`。
完整结果在其 `audit.json`；该凭证保存在数据集之外，没有向已发布数据集追加文件。
构造后的原采集/vLLM 进程仍存活，原输入、服务器 worktree 和 predictor checkout
均未被修改或清理。

实际回源入口（在上述封存源码目录中运行）：

```bash
/data/yuxiao/envs/untrusted-network-simulation/bin/python -B \
  -m scripts.manage_observer_dtrs_dataset audit-sources \
  --dataset /data/yuxiao/Untrusted_Network_Simulation/observer-dtrs/datasets/paper-development-qwen35-9b-1500-v1-observer-dtrs-v1 \
  --expected-dataset-digest d25537be18da93ddc2f49be695f876db2edef5f31904fea0999a292f491abfa3 \
  --publication /data/yuxiao/Untrusted_Network_Simulation/publications/paper-development-qwen35-9b-1500-v1 \
  --oof-root /data/yuxiao/Untrusted_Network_Simulation/paper-studies/evaluations/0b8c1d7220aae6fcf577038f8d930c97d6453b7f45616ed0a19c817e7a80b956 \
  --expected-oof-seal-digest 66bd044bd072a8673d4761ad043d0af3a208ae55b809ebffded2228b0371f925
```

本轮已闭合 D1a–D1c 的可复现离线数据链：监督/来源审计与合法特征接口分开，
未发现 D0 语义冲突或监督字段进入特征。尚未重新运行历史 ToM 权重计算、训练
恢复链或全仓库测试；也未进行 DTRS 建模、校准、泛化性能或在线行为验证。
建议进入 D2 的协议设计，先解决上述外层游戏划分及跨层 OOF 隔离，再选择统计模型。
