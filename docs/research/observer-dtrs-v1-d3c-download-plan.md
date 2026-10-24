# Observer-DTRS D3c：用户手动下载的材料与验证层级

2026-10-15；依据已固定 D2 源码 `3ff578f97b9d4c3620a15a67357ddabd848d10fc` 的
`nested_validation.run_block/execute/verify_result`、`FormalData.targets` 和 D1c reader。
这是源码清单核对，不是本轮服务器目录检查；没有 SSH/SCP、服务器部署或文件修改。

## 1. 目录与已有材料

服务器 D2 run 根路径（以下以 RUN 表示）：
`/data/yuxiao/Untrusted_Network_Simulation/observer-dtrs/validation-runs/d2-v1-run1`。

用户现有 Mac 目录
`/Users/name_yuxiao/Documents/Untrusted_Network_Simulation-results/d2-v1-run1/`
实际保存的是 RUN/**complete 的三个文件**，并非完整 RUN 根树。均已校验，
不要移动、改名或覆盖它们。后续手动取回的完整/紧凑 run 材料应保留下面相对路径，
另存到用户指定的本地数据目录；不要因目录同名误把 complete 当完整 run。
这些是科研产物目录，不是新的源码 clone 或 worktree；不向 Git 添加模型或大索引。

## 2. 最小受限分析包：已经有 p/Y，无需重新拟合

| 相对 RUN 路径 | 文件 | 用途 |
| --- | --- | --- |
| `complete/` | `manifest.json`, `record.json`, `bootstrap_indices.int64le` | 原 complete 锚点、25 块摘要、统计与整局索引；已有文件可在本地保留副本 |
| `contract/` | `manifest.json`, `record.json`, `sources.json`, `protocol.json` | 源码快照/HEAD、源数据锚点、完整游戏大小/折、split、样本摘要和来源限制 |
| `blocks/K-J/complete/` | `manifest.json`, `record.json`，全部25块 | 原始逐样本身份、phase/nW、Y、成功空标记，以及四模型×双权重概率 |

K、J 各为 0…4，取全部笛卡尔积。块 record 的结构明确为：

```text
status = PAIRED_BLOCK_COMPLETE
train_games, test_games, references
rows[]:
  sample_id, game_id, phase, nW, y, successful_empty
  predictions[game_equal|record_equal][OBS-EQ0|OBS-EQ1|OBS-LR-r|OBS-EF]
```

因此在验证块摘要与 contract 后，**仅作固定损失下的逐样本风险/域质量分析不需要
重算概率或重新训练，也不需要原始 role sidecar/ToM 权重**。该包使用的是 D2
封存的离线 Y，尚未独立对照 D1c 标签；未包含拟合模型、节点支持和全部引用产物，
不能声称 full `verify_result` 或所有训练来源已经重建通过。

块 complete 混有受限监督，只供离线读取。D3c 适配必须先拆为合法概率行与显式
targets，再调用决策/评价层，不能把整个 row 传进在线或纯决策接口。

## 3. 推荐 D3c 审计包：保存真实外层概率、模型和标签对应

在上述包之外，全部下载两种权重（game_equal、record_equal）的下列目录。
每个目录必须同时取 `manifest.json` 和 `record.json`，不改内容或重新序列化：

```text
blocks/K-J/WEIGHTING/outer_predictions/
blocks/K-J/WEIGHTING/selection/
blocks/K-J/WEIGHTING/evaluation/
```

每类各 25×2=50 个目录。`outer_predictions/record.json` 含
`models[model_id]`、`predictions[model_id][]`、EQ0/EQ1 的 `bin_audit`。
单条 prediction 是 sample_id/game_id/phase/nW/probability/level；不是完整 7×7 Q，
也不含合法 r 或保证唯一的 bin ID。模型的来源、训练游戏大小、参数及节点统计
可在同一记录中核对。selection 给选择参数及内层候选摘要；evaluation 给逐游戏
评分、test_target_digest 和 predictions_digest，**不保存全部逐样本 Y**。

因此外层预测与 D1c 监督可以直接严格联结做逐样本风险分析：

1. 从 complete 信任锚逐级验证 contract、块 complete 和预测/选择/评价的摘要引用。
2. 每个 outer/weighting/model 只取其原始 probability；核对 model/source、来源折、
   测试游戏和完整样本集，并与块 complete 的 p 逐条对照。
3. 显式用 D1c `open_dataset`→`open_supervision` 打开监督；按 sample_id 联结，
   同时交叉检查 game、canonical PRE、acting_wolf、observer 及来源。严格拒绝
   缺失、重复、跨折或跨模型/权重行；不按 probability 排序做位置配对。
4. 对照块中 y/成功空标记和 target digest，再将 Y 单独交给 `evaluate`；不把任何
   报告/身份字段交给 `decide`。每局分母来自完整合法样本数。

**独立标签/上下文和训练支持审计所需 D1c**：建议用户手动取回完整已封存目录

`/data/yuxiao/Untrusted_Network_Simulation/observer-dtrs/datasets/paper-development-qwen35-9b-1500-v1-observer-dtrs-v1/`

及其既有信任锚 `d25537be18da93ddc2f49be695f876db2edef5f31904fea0999a292f491abfa3`。
它有 4503 个载荷文件和 manifest，含 `legal/games/*.jsonl`、
`restricted/supervision/*.jsonl`、`restricted/provenance/*.jsonl` 及三份受限元信息。
现有 `open_artifact_envelope` 检查完整目录文件清单；即使按游戏选择性读取，也
不能只下载 manifest 加几份 supervision 就期待 reader 通过。不要修改 manifest、
伪造占位文件或绕过 gate；如以后确需小型导出，应另行设计有来源约束的派生产物。

对应外层 EQ 模型加 D1c 合法 r/phase/nW 可恢复每条样本的训练节点/回退路径，
读取该节点的训练支持。**仅 probability+level 不够**：多个叶可能概率相同，
不能由 p 反推唯一训练粒。LR 没有同义的 r_bin 支持，保留 NA。
不必额外取回原始 canonical/完整 role sidecar/ToM 权重，除非另行要求从原始来源
重建 D1c；本轮及下一阶段固定产物的风险评价不需要训练或 GPU。

该推荐包可完成 D3 所需的来源/标签/支持审计，但 selection 仍引用全部内层候选；
若没取内层文件，应明确停在“D3 相关外层路径已验证”，不能调用 full verifier
后将缺失内层误报为正式结果损坏，也不能声称内层选择过程已独立回放。

## 4. 完整 D2 回放包（需要 full verify_result 时）

除前述目录外，再取所有
`blocks/K-J/WEIGHTING/inner/H/C/` 的 manifest.json/record.json：
H=0,1,2；C=0…15，共 25×2×3×16=2400 个候选目录。
这才满足现有 `nested_validation.verify_result(RUN)` 对内层候选、选择、外层
评分、完整 summary 和 bootstrap 字节的跨引用重建要求。无需重新拟合。
如保留执行取证，可另外带 `failures/`；不要删除服务器历史失败记录。
`runner.lock` 不用于离线结果完整性，不需复制为新的执行锁。

完整 artifact 回放不等于重新训练，也不自动证明历史 ToM RNG 或架构选择已独立。
所有层级都继续保留固定历史产物/开发验证/固定预测 bootstrap 的研究限制。

## 5. 下一步最小顺序

先确定用户准备提供的包层级与本地路径，再开发只读 D3c 适配器及其 controlled
伪造/缺失/重复拒绝测试。当前推荐以第3节的外层审计包和完整 D1c 为目标；若只
提供第2节，则先完成受限封存标签的基本风险分析并明确较弱证据范围。
任何真实 45 场景执行都需之后单独确认并固定新源码、完整并列策略与执行配置；
不得根据真实结果选择代价或修改冻结 D2。所有服务器下载均由用户手动完成。
