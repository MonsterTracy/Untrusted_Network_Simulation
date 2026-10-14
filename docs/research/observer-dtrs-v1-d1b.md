# Observer-DTRS v1 — D1b Q 来源与条件属性

本阶段延续 [D1a 样本与标签契约](observer-dtrs-v1-d1a.md)，不修改目标事件、
成功空报告的六名非自身玩家语义、正式 publication 的报告完整性要求。
不复用旧 Phase2 候选目标、Primary 掩码或 Theta_AC 概率。

## D1a 接口复核

`ObserverSample.context / supervision / source` 已分离；合法上下文仅有 public PRE
身份、公开时间/存活状态、发言者和狼队合法已知身份。D1b 无需修改 D1a 的样本或
标签实现。纯标签函数的报告验证仍由 canonical 解码器承担，正式离线入口仍先经
publication / role sidecar 验证。包初始化不自动导入离线模块。

受控 Role Sidecar 研究用途继续限于 D1a 离线构造；D1b 特征函数不读取 sidecar、
publication、真实报告或训练 target。OOF 读取器属于离线来源层，不是在线接口。

## 合法 Q 与特征接口

```python
from werewolf.observer_dtrs.features import observer_features
from werewolf.observer_dtrs.probabilities import ProbabilityRow, ProbabilityMatrix

# 两种输入均绑定 game_id / boundary_id / prefix_digest。
row = ProbabilityRow(game_id, boundary_id, prefix_digest, observer_id, seven_probabilities)
features = observer_features(context, row)
matrix = ProbabilityMatrix(game_id, boundary_id, prefix_digest, seven_by_seven_probabilities)
assert observer_features(context, matrix) == observer_features(context, matrix.row(context.observer_id))
```

绝对行列轴均为 `player1 ... player7`。值对象要求七列有限、非负、至多 1，
自身列严格为零，行和与 1 的绝对误差不超过 `1e-6`（不使用相对误差）。矩阵要求
全部七行合法，包括死亡观察者行；不接受只含存活座位的压缩矩阵或其他座位顺序。
调用者列表被复制为不可变元组。不会自动指数化、裁剪、补行或重归一化。

`ProbabilityRow/Matrix` 保证数值与身份约束，**不自行证明模型来源**。离线行必须
经下述 OOF 适配器取得；未来在线 provider 必须负责冻结模型及合法 PRE 的来源保证。
本阶段提供在线矩阵兼容的值接口，不接入游戏运行或加载在线模型。

`observer_features(context, prediction)` 只接受 `ObserverContext` 与上述 Q 值对象，
不接受 `ObserverSample`、supervision 或普通字典。身份必须完全一致。输出版本为
`observer_dtrs_conditions_v1`：

| 字段 | 定义 |
| --- | --- |
| `s` | 当前发言狼人列的预测质量 |
| `m` | 其余存活狼人列质量之和；单狼时为 0 |
| `r` | `s + m` |
| `dW` | 双狼列质量之差的绝对值；单狼时为 `None` |
| `phase` | canonical `discussion / pk_discussion` |
| `nW` | 存活狼人数 1 或 2 |

死亡目标列始终保留。例：观察者 player3 的行是
`[.1,.2,0,.1,.1,.2,.3]`，狼队为 player1/player2、当前发言者为 player1。
双狼时 `(s,m,r,dW)=(.1,.2,.3,.1)`；player2 死亡后为 `(.1,0,.1,None)`，
不能把死亡狼的 .2 或其他死亡玩家的质量分配给存活玩家。

## 历史 Qwen3 OOF 来源契约

```python
from werewolf.observer_dtrs.oof import load_qwen3_oof

predictions = load_qwen3_oof(publication_path, evaluation_root,
    (s.context for s in samples), expected_seal_digest=trusted_evaluation_seal_digest)
features = [observer_features(p.context, p.probability) for p in predictions]
```

仅适配实际历史格式 `classic7_backbone_oof_predictions_v1`、架构 `qwen3`、
时间条件 `explicit_day_phase`。调用者必须显式指定已知评价 seal 的完整 SHA-256，
不能在适配器内自动“信任当前目录”。从该锚点核验：

1. evaluation seal → evaluation contract → training execution / seal → preparation。
2. preparation 中 publication、game-set、fold manifest 的摘要与实际 publication 一致。
3. 五折 held-out 游戏与 canonical fold 完全一致，training 游戏恰为其补集，无重复或泄漏。
4. 每折 Qwen3 terminal 的来源、fold、时间条件及末端 checkpoint 身份，与 sealed
   prediction descriptor、prediction manifest、每条预测行一致。
5. 每个 prediction 文件完整大小/SHA-256、canonical JSONL、行数与全部 PRE×7 覆盖。
   game / boundary / prefix / structured-token-plan / observer 均核对；重复或缺失报错。
6. 请求上下文的 canonical step、发言者、day、phase、存活集合与实际 PRE 一致。

准备快照的文件清单及 manifest 身份通过既有 `open_artifact_envelope` 检查。
**不读取或重哈希权重/优化器 payload，不重演训练和预测，不复核历史 bootstrap
draws 或训练恢复链。** 本模块信任显式钉住的历史 seal 所证明的计算来源，验证当前
预测文件是否确为该 seal 声明的产物；这不等于再次证明权重计算正确。

每行历史记录同时包含：

- `probability`：唯一允许读入特征的七列预测质量。
- `q`：历史训练 target，不读取其值、不保留到返回对象。
- `non_self_log_probability`：不作为输入，也不用于自动指数化。
- `label_observed / observer_alive`：不用于本模块的样本筛选。

这些历史字段必须存在以识别 v1 行格式，但其值不参与新特征或来源人口统计。
新论域由 D1a 上下文定义。旧 Primary identity 摘要只与 seal 做来源一致性比较，
不据此筛选。即使只请求部分上下文，读取器仍核验完整 publication 和五折预测覆盖。
无效来源不会回退到 target、真实报告、均匀 Q 或其他模型。

返回 `OOFObserverPrediction(context, probability, source)`；source 单独保存 publication、
fold、evaluation/preparation/training seal、源码版本、terminal/checkpoint、预测
manifest 与 payload SHA-256。返回顺序与请求上下文一致，不保留 target 或真实报告。
`OOFSource` 版本为 `observer_dtrs_qwen3_oof_source_v1`。

## 只读入口

```bash
python -m scripts.inspect_observer_dtrs_samples --publication /absolute/publication \
  --oof-root /absolute/evaluation \
  --expected-oof-seal-digest FULL_TRUSTED_SHA256
```

新增两项 OOF 参数必须同时给出；原 D1a 用法保持有效。命令仅输出覆盖统计，不写入
样本、预测或新数据集。`--game-id` 仍是完整游戏读取分区，不是模型训练划分。

## 测试与后续范围

新增 `test_features.py` 和 `test_oof.py`。OOF fixture 通过真实 artifact writer/reader
制作明确标注的合成来源包，使用合成 checkpoint 摘要，不代表真实模型训练。
测试既覆盖数值和座位映射，也覆盖重算所有摘要后依然存在的 fold 泄漏、模型/PRE
错配、重复/缺失、未封存篡改、外部锚点错误。独立子进程检查合法特征 import
closure；改变真实监督及旧 target/log/mask 后，相同 context/Q 的特征必须不变。

进入 D1c 前仍需确定新事件的属性配置、分箱与等价类协议、训练/验证游戏边界及
零频/稀疏类处理。旧 ToM OOF fold 只是预测来源，不能自动证明后续分箱或概率估计
没有验证泄漏。尚未实现分箱、条件概率估计、六损失、三域、校准或发言策略，未发布
正式 Observer-DTRS 数据集。

## 验收记录

本地基点为 `three-way@d8397ce56c65d6d46b24a6b53f0b71431c04267b`，
解释器 `/Users/name_yuxiao/anaconda3/envs/3wd/bin/python`，未安装或修改依赖。

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/name_yuxiao/anaconda3/envs/3wd/bin/python -B -m pytest -q \
  tests/observer_dtrs tests/canonical_collection tests/development_publication \
  tests/tom/test_dataset.py tests/tom/test_boundaries.py \
  -k 'not off_diagonal_readout_can_memorize_controlled_public_fixture' \
  -p no:cacheprovider --maxfail=3
```

实际结果：**248 passed, 1 deselected in 44.99s**。其中 Observer-DTRS 共 100 项，
D1b 新增 46 项；排除的是既有 ToM 小模型记忆训练测试，不宣称全仓库测试通过。

另对最终 D1b 特征/来源测试执行：

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/name_yuxiao/anaconda3/envs/3wd/bin/python -B -m pytest -q \
  tests/observer_dtrs/test_features.py tests/observer_dtrs/test_oof.py \
  -p no:cacheprovider --maxfail=3
```

结果：**46 passed in 3.96s**。14 个 Observer-DTRS/检查入口 Python 文件的 AST 和
行尾空白检查通过；只读 CLI `--help` 正常。

## 正式服务器产物只读核验（2026-10-14）

实际通过 SSH `tencent.aerber.cn:3001` 访问服务器，使用既有 Conda 环境的
Python `-B`，将本轮模块经 stdin 载入内存运行核验，`nice -n 10` 降低 CPU
优先级。没有向服务器部署或写入文件，没有打开模型权重、调用 vLLM、执行新预测
或干预采集进程。完整样本与预测核验约 206.44 秒。

publication 路径：
`/data/yuxiao/Untrusted_Network_Simulation/publications/paper-development-qwen35-9b-1500-v1`

OOF evaluation 路径：
`/data/yuxiao/Untrusted_Network_Simulation/paper-studies/evaluations/0b8c1d7220aae6fcf577038f8d930c97d6453b7f45616ed0a19c817e7a80b956`

| 来源 | 实际核验摘要 |
| --- | --- |
| publication | `396ea0fa3f3f03dd1dfce8b25c0162ca489b992321d0c9e6f772eb2b0c4a2acd` |
| role sidecar（仅 D1a 离线入口读取） | `276b6ee182bfb07639ae3ab99a5c8037b52ffb99b37f5c60f7f26a229178a2da` |
| canonical fold manifest | `754fe0e517aab9e9237f0cb5097bde0daf8a43a09518b9691cfca67724539e69` |
| evaluation seal（显式锚点） | `66bd044bd072a8673d4761ad043d0af3a208ae55b809ebffded2228b0371f925` |
| evaluation contract | `eee9e7d04383ca93b23ac6de8b4629faede61cc99305e43f4989f41407ba73ef` |
| training execution | `0b8c1d7220aae6fcf577038f8d930c97d6453b7f45616ed0a19c817e7a80b956` |
| training seal | `10f37a27dc951ab89021653dc433e510f844350c82f71b0d7711dfff957a8f86` |
| preparation | `f4ff88373da25caf21e8bc608fd049eba8cc378e886fb28a0ef0646d1a21907a` |

五折预测 payload 的大小/SHA-256、canonical 行内容、数值约束及全部身份/覆盖
校验均通过。publication 共 **1500 局 / 19,157 PRE**；全体七观察者预测共
**134,099 行**，各折分别为 `[26831,26411,26971,26236,27650]`。

新论域生成 **22,685 条样本/特征**，来自全部 1500 局的 **5,835 个狼人 PRE**，
无缺失预测、重复或来源错配：

| 划分 | 特征条数 |
| --- | ---: |
| discussion | 21,136 |
| pk_discussion | 1,549 |
| 双狼存活 | 18,512 |
| 单狼存活 | 4,173 |
| fold 0 / 1 / 2 / 3 / 4 | 4542 / 4502 / 4603 / 4434 / 4604 |

实际 `s` 范围 `[0.0064607346430420876, 0.943675696849823]`，
`m` 范围 `[0, 0.9266564846038818]`，
`r` 范围 `[0.007331902626901865, 0.9649299308657646]`。
这些只是来源和计算验收统计，不是新事件的模型性能或 DTRS 研究结论。

本轮没有保存/发布逐行正式数据集，也没有重新验证历史权重的数值计算或训练
恢复链。该限制与上述来源契约一致，不影响已经通过的预测文件来源与完整性验证。
