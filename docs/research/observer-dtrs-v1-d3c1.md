# Observer-DTRS D3c-1：封存外层概率与监督的只读适配

日期：2026-10-15。状态：**IMPLEMENTED_SEALED_MINIMAL_PACKAGE_NOT_FORMAL_D3_RUN**。
本地分支 three-way，HEAD 仍为 `3ff578f97b9d4c3620a15a67357ddabd848d10fc`。
本轮无 SSH/SCP/服务器访问，无训练/拟合，无正式 45 场景决策或模型优劣统计；
没有改变任何已冻结产物、并列策略、损失候选或既有 D3a/D3b 文件。

## 1. 新增实现与输入契约

- `werewolf/observer_dtrs/sealed_d2_offline.py`：最小包 reader、不可变合法视图、
  单独监督访问器、模型/权重 batch 绑定和显式离线评价桥。
- `scripts/inspect_observer_dtrs_sealed_d2.py`：只读身份/覆盖审计；不提供分类或成本选择命令。
- `tests/observer_dtrs/test_sealed_d2.py`：明确 synthetic 的完整 25 块 fixture 及拒绝案例。
- `observer-dtrs-v1-d3c1-acceptance.json`：真实材料的来源/覆盖审计与验收记录。
- 本文：校验层级、材料差异和 D3c-2 接口约定。

```python
anchors = SourceAnchors(complete_digest, source_revision, dataset_digest, split_digest)
view = read_formal_d2(local_run_root, anchors=anchors)
batch = view.batch('OBS-EQ1', 'game_equal')
targets = open_sealed_targets(view)  # 显式受限离线访问
# 以下只在受控验收中调用；本轮真实材料未执行：
result = evaluate_batch(batch, targets, loss=explicit_loss,
                        binary_tie_policy=explicit_binary_policy)
```

四个信任锚均必填。FrozenProtocol 仍用原机制验证冻结 payload，没有依赖当前
未提交 D3 代码假造新的生产 OID。`read_controlled_d2` 是单独的人工包入口，
强制 scope=controlled_fixture；formal reader 拒绝该 scope，不存在放宽真实来源的开关。

## 2. 实际校验路径

1. 使用既有 `verify_artifact`/`read_artifact_file`、canonical JSON、D2 版本和摘要
   机制核对 complete/contract/25 个块的清单、字节大小及 SHA-256。缺失或冲突
   直接报错，不生成 partial view，不修复、不补写输入文件。
2. complete 的身份摘要、contract_digest 和 25 个块摘要必须逐级一致；contract
   的 run_identity 必须等于其记录摘要。protocol/dataset/split/source OID 均与
   调用者信任锚匹配；来源限制及研究范围不得被删除。
3. 对 sources.json 核对 snapshot 摘要与指定生产 OID；文本仅校验，不 import/exec。
   本轮另外对真实 117 个生产文件与历史 Git 对象清单/内容作独立只读比对，全部一致。
   这是生产来源核对，不是将当前 D3 未提交代码绕过 source guard 执行正式拟合。
4. 正式包复用 `build_splits` 从封存 source_games 重建所有整局内外划分，与 contract
   精确比较；复用 `partition_games` 验证每块训练/测试游戏、来源折和全局一次覆盖。
   source_games 的游戏大小/折/样本列表摘要必须与 contract 对应项一致。
5. 每个块要求原始 row 的精确字段集合，两个权重和四模型都完整，p 为合法概率。
   检查 sample_id 唯一、完整游戏人数、phase/nW、Y 类型、成功空时 Y=1，
   并核对封存的逐游戏样本 ID 列表摘要。**顺序只用于这一来源完整性指纹**；
   batch/targets/评价一律按 sample_id 关联，反序 batch 的受控结果完全一致。
6. summary 的人口、各组记录数/游戏数/正例数与封存 rows 一致；从实际 test_games
   重建游戏顺序和固定 bootstrap 索引，校验完整 bytes。未重新计算真实三域，
   也未重放任何真实模型的条件风险比较或内层选择。

模型验证的准确含义是：验证封存行上的四个模型槽位、权重和外层产物摘要引用，
并由 complete 信任锚固定这些 bytes。尚未读取拟合模型自身；在未取得外层产物前，
不能独立重算模型概率，也不能从值判断某两个模型的槽位是否历史上被错误标记。
不夸大最小包能提供的证据。

## 3. 合法输入与监督隔离

`SealedRun.samples` 只含不可变 LegalSample：sample/game/phase/nW、ToM/outer fold、
块摘要及八个概率；没有 Y 或成功空字段。`batch.rows` 是 D3b 原有
`ProbabilityInput(sample_id,game_id,probability)`，不扩展纯 `decide` 的参数。
每个 batch 带模型/权重、生产来源、相应外层预测摘要、概率映射摘要及样本来源摘要。
重新解释模型名、权重、概率、来源元数据时均核对绑定，避免混用另一组数值。

`open_sealed_targets(view)` 是显式离线 opt-in，返回单独 SealedTargets；其标签/
空报告字典每次复制，不暴露可修改内部状态。batch 与 targets 的 run identity、
封存监督摘要和 sample/game 映射必须匹配。评价输出保留 loss digest、完整 source、
逐样本合法来源以及原 D3b 统计；原 row、Y、空报告标记、角色和支持记录不进入纯决策。
JSON bytes/冻结 dataclass 防止意外别名修改；它们是工程访问边界，不是 Python 安全沙盒。

本层不引入任何默认 LossSpec 或 TiePolicy。TERMINAL_POS_FIRST、TERMINAL_NEG_FIRST、
DEFER_FIRST 以及 45 候选均保持原样。读取/审计不选择损失，CLI 也没有该选项。

## 4. 本地真实材料与校验结果

实际读取目录：
`/Users/name_yuxiao/Documents/Untrusted_Network_Simulation-results/d2-v1-run1-d3c/`。
本轮发现该目录已有完整最小包；与此前只有三个 complete 文件的
`.../d2-v1-run1/` 是不同目录，不移动或覆盖旧下载。

| 来源 | 本地已验证身份 |
| --- | --- |
| complete | `870e75455572f4ce51dc3a0fc6d6ac050f658363d21ad130ba6616480bc831d1` |
| contract | `abbd3206f85ff12303639f18a501531257570baaa8d2c3aabdc6b0046addf754` |
| D2 生产源码 | `3ff578f97b9d4c3620a15a67357ddabd848d10fc`，117 文件与 Git blob 完全一致 |
| 源快照 | `e56bee9f9ec8e422c367766340ec78a2d87616486d23a6c89e9232f2af83c990` |
| dataset 引用锚 | `d25537be18da93ddc2f49be695f876db2edef5f31904fea0999a292f491abfa3` |
| split 引用锚 | `32e30246cbadec8240f47291422a791b4419482e1386b4d8b570344a2e45acc5` |
| 冻结 protocol payload | `c76940b446ffaa3dc9e873620fe775a3b64db8544d0c5190a49778a2e48a6c20` |

27 个已验证 artifact（complete、contract、25 块），1500 局、22685 个唯一观察者
样本；4 模型×2 权重均各有 22685 行，共 181480 个来源绑定概率值。25 个外层
测试块、75 个内层划分的**划分记录**与冻结规则一致；没有读取 2400 个候选拟合产物。
完整 CLI 审计和八组概率摘要见 acceptance JSON；本报告不列任何真实三域比例、
六损失实际风险或模型优劣结论。

真实只读命令：

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/name_yuxiao/anaconda3/envs/3wd/bin/python -B \
  scripts/inspect_observer_dtrs_sealed_d2.py \
  --run /Users/name_yuxiao/Documents/Untrusted_Network_Simulation-results/d2-v1-run1-d3c \
  --expected-complete-digest 870e75455572f4ce51dc3a0fc6d6ac050f658363d21ad130ba6616480bc831d1 \
  --expected-source-revision 3ff578f97b9d4c3620a15a67357ddabd848d10fc \
  --expected-dataset-digest d25537be18da93ddc2f49be695f876db2edef5f31904fea0999a292f491abfa3 \
  --expected-split-digest 32e30246cbadec8240f47291422a791b4419482e1386b4d8b570344a2e45acc5
```

省略 --output 可重复只读审计；保存新的本地 JSON 时使用 --output，拒绝覆盖文件。

### 证据层级

- 已验证：最小包 bytes、来源链/生产源快照、完整游戏/样本人口、封存模型槽位/
  权重映射、合法标量 p 和封存 Y、固定索引与划分记录。
- 未验证：独立 D1c 标签与 canonical 上下文、外层模型内容/节点支持、内层选择
  回放，以及 full `verify_result`。四个布尔证据字段均保持 False。
- dataset/split 摘要是 contract 中与预期一致的引用；split payload 已从来源元数据
  重建，但本轮没有把独立 dataset/split 原 artifact 文件再次下载或打开。
- source_snapshot 与 Git 的独立比对已做；未重训历史 ToM，也不消除其全量
  publication 摘要影响训练 RNG、架构选择独立性未确证等限制。
- Sealed Y 验证是身份、完整性、类型和既定空报告语义检查，不是从 canonical
  原始报告重新证明每条标签；成功空真值和原始支持的独立复核留给 D1c 对照。

## 5. 受控验收

使用 75 局/150 行的明确 ControlledData，直接发布人工概率的最小 artifact；
未拟合 ToM、EQ 或 Logistic。仅对受控概率执行 D3b 评价桥，验证策略和 loss digest
被保留、反序概率行按 ID 联结，以及决策记录不含 Y/成功空信息。
测试还将加载路径的 decide 替换成报错函数，证明加载不调用分类。

覆盖 25 块/八映射完整性、读入即分类禁止、缺失和截断不修复、重算摘要后的重复/
遗漏/跨游戏/错折/模型/权重/样本/监督冲突、非法概率、来源锚和快照不符、研究限制
被删除、整局划分冲突、batch 重解释、来源和 targets 被修改，以及不可变访问器。
受控 reader 不能用于认证正式 scope，formal reader 也不接受 synthetic scope。

命令：

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/name_yuxiao/anaconda3/envs/3wd/bin/python -B -m pytest -q \
  tests/observer_dtrs/test_sealed_d2.py -p no:cacheprovider --maxfail=3

PYTHONDONTWRITEBYTECODE=1 /Users/name_yuxiao/anaconda3/envs/3wd/bin/python -B -m pytest -q \
  tests/observer_dtrs tests/canonical_collection tests/development_publication \
  tests/tom/test_dataset.py tests/tom/test_boundaries.py \
  -k 'not off_diagonal_readout_can_memorize_controlled_public_fixture' \
  -p no:cacheprovider --maxfail=3
```

最终针对性 **31 passed，1.42s**；最终相关回归 **430 passed，1 deselected，65.85s**。
记录见 acceptance JSON；排除项仍为此前 ToM 小模型训练测试，不声称全仓库通过。
不把对旧版本的回归或 Python 3.10 语法解析冒充新代码在服务器环境运行通过；
本轮只使用本地 Python 3.12.13。410 个既有文件（含此前16个未提交阶段文件）
指纹保持不变，仅新增本轮5个文件，暂存区为空。

## 6. D3c-2 训练支持审计的接口

建议独立实现以下只读步骤；当前不实现、不制造缺失特征：

```text
crosscheck_d1c(view, dataset_path, expected_dataset_digest)
  -> 明确 gated 的合法样本关联、独立标签核对报告

open_outer_model(run_root, view, *, model_id, weighting, tom_fold, outer_fold)
  -> 验证后的模型/外层 prediction/selection/evaluation 引用

resolve_training_support(sample_origin, legal_ObserverFeatures, verified_outer_model)
  -> model/artifact digest、使用路径/层级、bin/回退、训练 rows/games/mass/Kish、适用状态
```

D1c 必须使用原 `open_dataset`→`open_supervision`，按 sample_id 关联 game、PRE、
acting_wolf、observer；当前块只有 sample_id、phase/nW，没有 r 或完整 canonical
上下文，不能从 hash、预测概率或校准箱反推这些字段。现有 reader 要求完整 4503
载荷清单；不能用零散 supervision 绕过 gate。

还需用户手动提供 `blocks/K-J/WEIGHTING/{outer_predictions,selection,evaluation}`
的 manifest/record（各50目录）和完整 D1c。outer prediction 中的 models 可直接
用于支持读取，无需训练；先核对其摘要与已验证块引用一致，再核对训练游戏确实
为对应外层训练集合、模型 ID/权重/参数/来源以及原始概率。

EQ0/EQ1/EF 使用保存的 phase/nW/r 与原模型 edges，按 `bisect_left` 定位训练叶/
回退节点，读取原训练 stats，不能在测试游戏重新计数或学习边界。仅 p+level 不足
以定位唯一叶。支持不进入 `decide`；低支持不自动变成 BND。正例质量和训练类
分布保留受限审计属性，Kish 不是概率区间或独立伯努利样本量。LR 的 r_bin
支持返回 NOT_APPLICABLE，不伪造可比粒支持；可另列其训练人口等全局元数据。

若还要求 full `verify_result`，再取全部内层候选产物；仅有上述外层审计包时仍
不能声称内层过程独立回放。详见既有 d3c-download-plan。

## 7. 是否进入配置冻结

工程上已具备进入正式 D3 配置冻结讨论的输入基础，不再受“没有真实逐样本概率/Y”
阻碍。但本轮没有冻结或运行正式协议。需确认：完整并列序（包括退化 P/N 并列）、
全部45场景的报告方式/无规范依据时不设唯一主损失、采用封存 Y 的证据层级，
以及 D3c-2 独立标签/训练支持审计是否作为正式执行前置项。
若正式指标包含训练支持，须先补材料完成 D3c-2，不能用缺失值伪装已审计。

真实 D3 统计仍是固定历史 ToM 产物条件下的后验开发分析，不是新独立测试；
固定预测 bootstrap 不含完整重训/重新选参不确定性。源码仍未提交，正式生产
guard 应继续拒绝当前新生产源码，不修改或绕过它。未来执行前需另行固定 D3
代码版本，保留历史 D2 生产 OID 和既有全部来源限制。
