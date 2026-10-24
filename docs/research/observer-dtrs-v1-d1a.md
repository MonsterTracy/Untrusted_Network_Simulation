# Observer-DTRS v1 — D1a 离线观察者样本与标签

本实现采用已经确认的 D0 事件定义。它不替换 Phase2 的 Theta_AC Mapper、
Router、ToM 或发言策略。版本为 `observer_dtrs_offline_sample_v1`，标签版本为
`observer_dtrs_effective_support_live_wolf_v1`。

## 论域与标签

每个样本身份为 `(game_id, boundary_id, acting_wolf, observer_id)`，另保存
`prefix_digest` 及来源摘要。只在当前发言者属于存活狼队时，枚举所有存活非狼
观察者。普通讨论与 PK 讨论使用 canonical 的 `discussion/pk_discussion`，
不继承旧 Phase2 的 competition/candidate 过滤。即使 PK 竞争目标全是狼人，
仍保留其他存活非狼观察者。

`known_wolf_ids` 是完整两狼队伍，`alive_wolf_ids` 为其与 PRE 存活集合的交集。
`step_index` 原样复制 canonical PRE 字段；它不是运行循环计数或重新编号的样本行号。
座位始终使用 `player1` 至 `player7`，不做整数或 P1 格式的隐式转换。

成功非空报告保留原始支持集合；成功空报告的有效集合为六名非自身玩家，包含
可能死亡的目标。标签 Y 是有效集合是否命中至少一名**存活**狼人。只怀疑死亡
狼人而未命中存活狼人时 Y=0；当前论域内成功空报告必为 Y=1。

纯标签函数对失败/缺失报告返回 `label_observed=False`、`event_label_y=None`、
`effective_suspicion_support=None`。失败报告的原始支持为空元组，缺失报告的
原始支持为 None，两者均不同于成功空报告。

正式 publication 入口先完整验证所选游戏，失败、缺失、重复或身份错配的报告
必须报错，包括最后不产生观察者样本的非狼发言 PRE；不得静默过滤后输出样本。

## 接口与信息隔离

```python
from werewolf.observer_dtrs.offline import build_observer_samples

samples = build_observer_samples(publication_path)
# 可选 game_ids=(...) 是显式的完整游戏读取分区，不是新建训练/验证划分。
for sample in samples:
    sample.context       # 仅合法上下文
    sample.supervision   # 仅离线：原始报告、有效集合、Y、报告身份及摘要
    sample.source        # publication / sidecar / bundle / record / fold 来源
    sample.to_record()   # 显式分开的三部分，属于离线监督记录
```

`context.py` 的 `contexts_for_pre(prefix, known_wolf_ids)` 只需要 canonical PRE
和合法狼队信息。`ObserverContext` 不含报告、标签、其他玩家角色分配或
publication 读取句柄。包初始化不导入离线模块。未来在线属性接口应只接收
这种合法上下文及经过验证的预测 Q，不能接收完整 `ObserverSample`、
`PublishedGame` 或 Role Sidecar。本阶段尚未实现该属性接口或任何 Q 输入。

离线入口复用 `open_publication`、`open_role_sidecar`，不接受另行传入、可能混配
的角色文件。既有读取器负责 canonical schema/digest、PRE—报告绑定、游戏覆盖、
完整角色座位/数量及 publication/sidecar 来源验证。纯标签函数复用 canonical
`_observation_from_record` 解码器检查报告摘要和状态，不改写原始报告。

每条样本记录 publication ID/digest、collection/game-set/sidecar/fold 摘要、
游戏 bundle/record 摘要、原 publication fold，以及 observation ID/digest、
attempt ID。原 fold 只作为来源记录，不代表本模块已完成新事件的评估划分。
结果顺序为 publication 游戏顺序、canonical PRE 顺序、观察者座位顺序。
没有合法狼人 PRE 的已验证游戏可以贡献零行。

## Role Sidecar 的受控研究用途

这项用途由 Observer-DTRS D0/D1a 研究契约明确授权：**仅在离线构造器内**读取
已绑定 publication 的受限角色信息，以确定狼队、样本资格和监督事件。完整
七人角色映射不输出到样本；合法两狼队信息与离线监督分别保存。

这补充了 [ADR 0003](../adr/0003-terminate-role-truth-at-population-selection.md)
在冻结 Phase1 ToM 训练路径中的角色终止约束：ToM Dataset、模型、loss、metric
和 Population Selector 保持原样。本例外不许可角色真值、其他观察者真实报告、
训练 target 或 Theta_AC 进入冻结 ToM、未来在线属性或语言生成。

## 只读检查入口

```bash
python -m scripts.inspect_observer_dtrs_samples --publication /absolute/publication
python -m scripts.inspect_observer_dtrs_samples --publication /absolute/publication \
  --game-id exact-game-id
```

命令仅计算并打印覆盖及正类计数，不写文件、不加载模型、不公开逐行报告。
`games_with_samples` 只统计有样本游戏，不能解释为 publication 总游戏数。

## 测试与范围

```bash
python -m pytest -q tests/observer_dtrs
```

测试使用手工可核对的轨迹和确定性**合成** replay，通过实际 canonical bundle、
ledger、publication writer/reader 进行往返。五局 fixture 是满足现有五折 publication
格式的最小受控集合，不是新游戏仿真实验或统计证据。覆盖普通/PK、两狼/单狼、
死亡观察者、成功空/非空、失败/缺失、重算摘要后仍存在的来源错配及只读入口。

D1a 返回内存样本及版本化 record。尚未产出正式 DTRS 数据集 artifact，未读取
服务器研究数据，未实现 OOF Q 适配、条件属性、分箱、概率估计、六损失或在线
三支判定，也没有调整既有 canonical publication 的资格规则。

## 本地验收记录（2026-10-14）

在 `three-way`、基点 `d8397ce56c65d6d46b24a6b53f0b71431c04267b` 上执行，
解释器为 `/Users/name_yuxiao/anaconda3/envs/3wd/bin/python`（Python 3.12.13，
pytest 9.1.1），未安装或修改依赖。

实现前 publication/PRE/ToM dataset 基线为 24 passed。D1a 新增 54 个测试案例，
全部包含在以下实际执行的回归检查中：

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/name_yuxiao/anaconda3/envs/3wd/bin/python -B -m pytest -q \
  tests/observer_dtrs tests/canonical_collection tests/development_publication \
  tests/tom/test_dataset.py tests/tom/test_boundaries.py \
  -k 'not off_diagonal_readout_can_memorize_controlled_public_fixture' \
  -p no:cacheprovider --maxfail=3
```

结果：**202 passed, 1 deselected in 40.49s**。未运行该 ToM 小模型记忆训练测试；
这不是整个仓库的全量测试结果。新文件的 Python AST 与行尾空白检查通过，
只读命令的 `--help` 正常。没有读取服务器 publication、执行模型推理或采集。
