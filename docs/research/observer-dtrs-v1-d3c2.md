# Observer-DTRS D3c-2：独立 D1c 对照与外层训练支持审计

日期：2026-10-15。状态：**IMPLEMENTED_AND_REAL_OUTER_PACKAGE_AUDITED_NOT_FORMAL_D3_RUN**。
分支 `three-way`，HEAD 保持 `3ff578f97b9d4c3620a15a67357ddabd848d10fc`。
本轮只在 Mac 开发和只读核验，无服务器访问、真实拟合、正式三域/六损失比较、提交或推送。
D0–D3b 的数学定义、45 个成本候选和全部显式并列策略未改变。

## 1. 新增文件及接口

- `werewolf/observer_dtrs/training_audit_offline.py`：显式受限 D1c 对照、外层三类产物校验、训练节点解析。
- `scripts/inspect_observer_dtrs_training_audit.py`：只读 CLI；没有成本、分类或拟合选项；输出只可新建，不能写入两个输入目录。
- `tests/observer_dtrs/test_training_audit.py`：34 个受控测试，不读取真实科研数据。
- 本文及 `observer-dtrs-v1-d3c2-acceptance.json`：最终验收与来源证据。

```python
view = read_formal_d2(run_path, anchors=explicit_anchors)
link = crosscheck_d1c(view, dataset_path, expected_dataset_digest=explicit_dataset_digest)
targets = open_d1c_targets(view, link)  # 显式打开；兼容原 evaluate_batch
outer = verify_outer_artifacts(run_path, view, link)
model = outer.model(model_id='OBS-EQ1', weighting='game_equal', tom_fold=k, outer_fold=j)
support = resolve_training_support(sample_origin, legal_dataset_row, model)
# 原纯决策器仍只接收 p 和 LossSpec；支持、监督与原始 row 均不传入。
```

`D1cLink.legal_rows` 仅含原有合法 LegalDatasetRow（context、Q、features、sample_id）；
监督封存在私有 canonical bytes 中，通过 `open_d1c_targets` 单独返回 SealedTargets。
内部字典访问返回副本，合法行和模型 bytes 使用不可变容器；这延续工程访问边界，
不将 Python 私有属性解释为安全沙盒。没有把原始监督行传给纯决策器。
`OuterAudit` / `VerifiedOuterModel` / 支持记录属于单独的受限训练审计层，不能当作在线属性。

## 2. 独立 D1c 对照

先复用 `verify_dataset` 完整核验文件、来源胶囊、实现快照和覆盖审计，再明确调用
`open_dataset → open_supervision`。按 sample_id 联结，不按两份产物的行位置配对。

对每条记录核对完整游戏人口、规范身份、phase、nW、publication/OOF 来源折、Y 和成功空标记。
正式 reader 重算 sample_id，从而将 D2 的哈希身份绑定至 D1c 的
`game_id / boundary_id / prefix_digest / acting_wolf / observer_id`。
D1c 的 canonical `step_index` 保留并由 reader 验证；D2 不另存该字段，不能声称独立
重建了它或把它换成 runtime 步数。拒绝重复 canonical 身份、缺失、跨游戏、错折及标签冲突。

本层独立比较了两个已封存产物，并重新验证 D1c 有效集合与标签语义（包含成功空报告）；
**没有调用 `audit_dataset_sources` 从最初 canonical 轨迹、role sidecar 和历史 OOF 原件重建 D1c**。

## 3. 外层模型与预测证据

全部 25 块 × 2 权重，分别打开 `selection / outer_predictions / evaluation`，每类 50 个产物。
复用 D2 artifact envelope、摘要和 canonical JSON 验证，逐级对照 complete 的固定引用。

- selection：完整固定网格、候选整局 Brier 表、48 个候选摘要引用、对应来源折/权重/训练游戏，
  使用原 `_select` 核验保存分数所确定的参数及并列规则。未打开内层候选产物，未证明那些分数由重新拟合得到。
- outer model：精确训练/测试范围、D1c 特征/训练标签指纹、源码快照摘要、dataset/split/protocol、
  模型 ID、权重和已选参数；EQ 复用 validate_model 核验节点概率、质量守恒、Kish 与分箱结构。
  EF 必须精确等于保存 EQ1 的原节点/权重/分箱所定义的经验频率消融。
  Logistic 检查 schema、参数、来源、训练指纹、solver 与收敛证据。
- predictions：各模型样本集完整且唯一；通过 SID 查找对应合法特征，使用保存模型执行纯推理，
  校验 probability 和 level；与 complete 的 p 比较是精确比较。没有改写、裁剪或替换概率。
- evaluation：test target digest、prediction digest、四模型逐游戏人口及原 D2 评分均复核。
  这只验证历史评分一致性，不输出新模型排名，更不计算 D3 六损失。

保存模型推理/评分的跨平台数值复核采用绝对容差 1e-12；这个容差仅用于审计比较，
不改变任何决策、阈值、选参或成本，并不应用于 complete 与 outer 保存概率的精确对照。
本轮未重新学习 Logistic 标准化参数、求解系数或重放任何内层拟合。

## 4. 训练信息粒支持

只从已验证外层模型读取保存的 edges、节点和 stats，按原 `bisect_left(edges, r)` 定位。
精确落在切点时取左粒，与原推理一致。路径为已有的 root → nW → phase/nW → r_bin，
未见状态按原模型回到 nW 或 root。只有一个有效分箱时使用 state 节点，不将其冒称未见状态回退。

返回 sample/game、模型/权重/折、模型和产物摘要、完整路径、实际层级、bin_index、原 edges、
请求/实际分箱数、是否状态缺失回退，以及原节点 training_stats：
`rows / games / positive_games / negative_games / mass / positive_mass / squared_game_mass / kish_games`。
正例质量等训练标签派生字段仍是受限审计信息。

若粒 E 的训练游戏质量为 u(E,g)，Kish 为
`(sum_g u(E,g))² / sum_g u(E,g)²`。它是游戏聚类质量的有效支持，
不是独立伯努利样本量、概率区间或真实预测可靠性的保证。
输出取自保存节点；没有重新分箱、合并、拟合或从测试标签计数。
训练指纹核对会读取训练 Y，但该步骤与支持解析函数分离。

OBS-LR-r 返回 `NOT_APPLICABLE`，不伪造等价类支持。支持不参与 decide，
不改变区域，也不将低支持解释为 BND。本轮没有实际三域分配或风险比较。

## 5. 真实材料和实际结果

D2：`/Users/name_yuxiao/Documents/Untrusted_Network_Simulation-results/d2-v1-run1-d3c/`

D1c：`/Users/name_yuxiao/Documents/Untrusted_Network_Simulation-results/d1c-observer-dtrs-v1/`

| 核验项 | 结果 |
| --- | --- |
| D1c manifest | `d25537be18da93ddc2f49be695f876db2edef5f31904fea0999a292f491abfa3` |
| D2 complete | `870e75455572f4ce51dc3a0fc6d6ac050f658363d21ad130ba6616480bc831d1` |
| D2 split 引用 | `32e30246cbadec8240f47291422a791b4419482e1386b4d8b570344a2e45acc5` |
| 生产源码 | `3ff578f97b9d4c3620a15a67357ddabd848d10fc`，模型 implementation_digest 与封存快照一致 |
| 独立 D1c 标签/成功空/身份/折 | 1500 局、22685 条全部一致 |
| 外层审计 | 25 块、150 个外层产物、200 份保存模型全部通过 |
| 概率对照 | 4 模型 × 2 权重，各22685条，共181480个绑定全部通过 |
| 保存模型推理、原 D2 评分 | 全部通过；未发表新评分或排名 |
| 训练支持路径 | EQ0 使用 state；EQ1/EF 使用原 r_bin；LR 全部不适用 |

上述最后一行只描述支持定位，不是三域比例、训练支持的可靠性结论或模型优劣。
每个外层证据摘要、逐模型支持记录摘要及层级计数均保存在 acceptance JSON。

仍未验证：原始 canonical 全链重建、2400 个内层候选产物/拟合回放、full verify_result、
上游重训及新独立测试。独立 D1c 对照和外层审计不能消除历史训练 RNG 的全量 publication
摘要依赖；后续仍是固定历史 ToM 产物条件下的后验开发分析，bootstrap 不包含上游重训或重新选参。

## 6. 验收与复现

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/name_yuxiao/anaconda3/envs/3wd/bin/python -B -m pytest -q \
  tests/observer_dtrs/test_training_audit.py tests/observer_dtrs/test_sealed_d2.py \
  -p no:cacheprovider --maxfail=2

PYTHONDONTWRITEBYTECODE=1 /Users/name_yuxiao/anaconda3/envs/3wd/bin/python -B -m pytest -q \
  tests/observer_dtrs tests/canonical_collection tests/development_publication \
  tests/tom/test_dataset.py tests/tom/test_boundaries.py \
  -k 'not off_diagonal_readout_can_memorize_controlled_public_fixture' \
  -p no:cacheprovider --maxfail=3
```

最终针对性 **65 passed，13.83s**（新34项 + D3c-1原31项）；相关回归
**464 passed，1 deselected，132.18s**。排除项沿用此前 ToM memorization 测试，
不声称全仓测试或服务器环境通过；本地为 Python 3.12.13。
另对3个新Python文件做了 Python 3.10 语法解析，不将其当作服务器运行验收。

fixture 使用人工数据创建真实 artifact envelope；只在人工数据上生成保存模型。
验收时将 fit、train_edges、decide 替换为报错函数，确认 reader 不调用它们。
覆盖成功空/非空、标签改变、错折、重复、缺失、来源重算摘要后冲突、错模型/权重/分箱、
切点归属、root/nW 回退、监督/支持隔离、预测反序的 SID 联结和原件不修复。

真实只读命令（省略 --output 则仅输出审计 JSON）：

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/name_yuxiao/anaconda3/envs/3wd/bin/python -B \
  scripts/inspect_observer_dtrs_training_audit.py \
  --run /Users/name_yuxiao/Documents/Untrusted_Network_Simulation-results/d2-v1-run1-d3c \
  --dataset /Users/name_yuxiao/Documents/Untrusted_Network_Simulation-results/d1c-observer-dtrs-v1 \
  --expected-complete-digest 870e75455572f4ce51dc3a0fc6d6ac050f658363d21ad130ba6616480bc831d1 \
  --expected-source-revision 3ff578f97b9d4c3620a15a67357ddabd848d10fc \
  --expected-dataset-digest d25537be18da93ddc2f49be695f876db2edef5f31904fea0999a292f491abfa3 \
  --expected-split-digest 32e30246cbadec8240f47291422a791b4419482e1386b4d8b570344a2e45acc5
```

最终代码已重跑真实核验。5276个既有文件（415个仓库文件及4861个输入产物文件）
SHA-256 未变，原 stash 完整 OID 未变。仅新增本轮5文件，暂存区为空，无 commit/push。
原正式生产 guard 对当前未提交生产源码仍按设计拒绝；本次只读审计没有绕过该 guard 启动正式运行器。

## 7. 正式 D3 配置还需明确

输入审计已达到本阶段目标，可以准备冻结执行配置；仍须明确：

1. 精确并列策略的主报告/敏感性安排（保留全部显式策略，不自动指定默认）。
2. 45个规范性成本场景的报告方式；没有外部依据时不从真实结果挑选唯一主矩阵。
3. 训练支持分层/扰动分析的预设报告规则，不用标签或三域比例反向设阈值，不以支持改变决策。
4. 如要求 canonical 原始重建或完整内层回放，应另外补齐材料；当前正式报告必须准确声明验证层级。
5. 正式执行前另行固定 D3 源码提交与配置摘要；本轮没有提交授权，也没有正式执行。
