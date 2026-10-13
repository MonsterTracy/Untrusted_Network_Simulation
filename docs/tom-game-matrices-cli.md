# ToM 游戏检索与中文三线表

独立、只读的离线展示工具：`python -m scripts.show_tom_game_matrices`。
需要已有可视化工具相同的 NumPy 依赖，不需要 Matplotlib 绘图、Qwen3 worker、
GPU、LLM、游戏运行或网络。使用等宽字体，可直接复制 UTF-8 TXT 至 PPT。

```sh
# 单个或多个根目录；参数可重复。目录必须实际可读。
python -m scripts.show_tom_game_matrices \
  --root /path/to/synced/data /path/to/another/data --list

# 单局、两列；四个案例 PRE
python -m scripts.show_tom_game_matrices --root /path/to/synced/data \
  --game 1 --pre-indices 9 10 14 15 --per-row 2 --output-dir /path/to/new/txt-two

# 多局（列表中确有游戏 1、2 才能使用）
python -m scripts.show_tom_game_matrices --root /path/to/data \
  --games 1 2 --per-row 3

# 全部游戏、按真实 ID 选择
python -m scripts.show_tom_game_matrices --root /path/to/data --all
python -m scripts.show_tom_game_matrices --root /path/to/data --game-id EXACT_GAME_ID
```

Mac 可指定同步文件夹；服务器可指定 `/data/...`，工具不会认为 Mac 已挂载该目录。
无选择参数时默认列出清单。所有选择器互斥，不提供分页器或自动改布局。

| 参数 | 含义 |
|---|---|
| `--root PATH [PATH ...]` | 必需，可重复；递归搜索 manifest，不跟随子目录符号链接 |
| `--list` | 列出所有识别到的 ToM game_id、来源与校验状态 |
| `--game N` | 按清单中的一基序号选择一局 |
| `--games N [N ...]` | 选择多局，不允许重复选择 |
| `--game-id ID` | 精确选择真实 game_id |
| `--all` | 展示所有已发现的游戏；任一选中游戏无效或冲突则拒绝展示 |
| `--per-row {1,2,3}` | 每行矩阵数，默认 2；最后不足一行完整保留 |
| `--pre-indices N [N ...]` | 全局 PRE 一基序号，按真实顺序显示；越界/重复拒绝 |
| `--precision N` | 显示小数位数 0–8，默认 3；不改变数据 |
| `--delta-tolerance FLOAT` | 仅用于箭头显示，默认 `1e-8`；有限且非负 |
| `--ambiguous-width {1,2}` | 东亚模糊字符（箭头、点、线）的终端列宽，默认 1 |
| `--terminal-width N` | 显式指定终端列宽；不指定时只在 stdout 为 TTY 时检查实际宽度 |
| `--output-dir PATH` | 新目录，保存 `tom-game-matrices.txt`；禁止覆盖或写入已发现输入产物树 |
| `--help` | 参数帮助 |

默认精度与字符宽度下，单个表宽 56 列、两列布局 116 列、三列布局 176 列。
TTY 不足时明确拒绝并提示改布局；不会截掉 P7。非 TTY（重定向/管道）按完整
所需宽度打印。`--terminal-width` 可用于复现实验和主动检查布局，例如设为 200。
中文固定为双列宽；若终端将箭头等模糊字符按双列显示，使用 `--ambiguous-width 2`。
TXT 不含 ANSI 控制序列，不使用 less/more。PPT 仍需等宽字体并为中文使用匹配字宽。
展示 TXT 记录参数、输入 provenance 和展示/读取脚本 SHA256，便于复现。

## 发现与核验

只计 `artifact_type=tom_one_game_all_pre_matrices`、
`schema_version=tom_one_game_all_pre_export_v1` 的导出身份。
其他模型/格式不冒充此已验证格式。只有 canonical bundle、图表 artifact 或目录名
不会被计作已有 ToM 游戏。清单按 game_id 排序，索引随搜索范围改变，长期引用用
`--game-id`。

通过 game_id 与 source_bundle_manifest_digest 关联 canonical manifest。
复用 `visualize_tom_single_game.load_review` 的矩阵、CSV、冻结模型 pins、
manifest、PRE、座位、存活状态和公共前缀校验。读取器现在允许显式导出/证据路径，
支持标准 bundle 的 `public/` 以及同步公开快照的扁平布局；不依赖导出目录名。
同一路径或同内容副本去重；相同 game_id 的不同 manifest 内容或来源报告 CONFLICT，
禁止静默取一个副本。坏输入报告 INVALID；清单可看错误，矩阵不能正常输出。

`VERIFIED_PUBLIC` 仅代表此读取范围通过校验，**不代表完整 canonical 重放通过**。
所有已提供的 canonical 载荷都会核对摘要；缺少私密文件时明确报告覆盖比例。
也不重新校验模型权重。清单区分已发现身份与实际通过校验的游戏数，坏输入的
PRE/矩阵数量不被当作真实数量。

## 权威信息与显示语义

- 角色只来自与同一 manifest 绑定的 `private/private_replay_evidence.json`。
  使用现有生产 record parser 校验角色数量、内部摘要、game_id、seed、公共历史
  与 runtime configuration 绑定。没有该文件就输出“角色信息未提供”。
- 在线 ToM 与离线矩阵分开：目前支持从已绑定、类型校验的
  `private/backend_call_evidence.jsonl` 查证成功 gameplay 请求中的既有 ToM 注入段落，
  并核对同一真实 PRE 的 speaker 与狼人角色。这里只确认请求收到 ToM 文本，
  不宣称在线模型权重或其预测来源重新验收。响应中出现 ToM 文字、失败调用、
  配置名称、目录名或离线 Q 都不能成为阳性证据。
- 缺少上述证据，或是其他在线接口未被这个工具识别，状态为“无法确定”。
  缺少注入段落不等于证明没有使用 ToM，不能报告假阴性。
- 发言顺序来自实际 `public_speech` 事件，按日和普通/PK 分开；不将未来队列
  当成已完成发言。实际票数/弃票/放逐来自 vote_result 与对应 exile_result，
  不是发言中的投票意图；缺少放逐记录显示“未提供”，空结果显示“无人放逐”。
- 每个 PRE 显示存活集合；死亡观察者行、死亡目标列均加“亡”，全部 7×7
  原值保留。狼人目标列加“狼”，角色信息绝不进入推理、Agent 或 Router。
- 狼人列箭头使用未四舍五入的原值，对比上一**实际** PRE，包括未选来展示的 PRE：
  增加 ↑、减少 ↓、容差内 →、首个 PRE ·。自身结构零不加箭头。
  第一张选中的图若不是 PRE1，仍有真实前一 PRE 比较基准。
- ToM 是观察者条件化的预测嫌疑质量，不是真实狼人身份概率；变化不作因果解释。

退出码：0 成功；1 清单有扫描/校验问题；2 选择、布局、输出安全或展示失败。
已验证的其他游戏仍可单独选择，不能用它们掩盖同 game_id 的坏副本。

## 测试

```sh
UNS_TOM_SINGLE_GAME_REVIEW=/path/to/real/tom-single-game-review \
  python -m pytest tests/tom/test_show_tom_game_matrices.py \
  tests/tom/test_single_game_visualization.py -q
```

真实回归使用已同步的 17-PRE 数据，不将私密资料放入版本库。
多局身份和可选角色/请求证据测试使用明确的合成副本；它们不是新采集的游戏，
不能计入真实实验样本。原始测试输入不写入，所有损坏/重绑定均发生在临时副本。
