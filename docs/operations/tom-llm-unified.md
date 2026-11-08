# TomBaseline：单一集合执行主线

本页描述新的 `classic7_suspicion_set_execution_v2` 执行协议，复用已批准的
`classic7_suspicion_set_research_protocol_v2` 科研定义。它不是历史
`bacc35135fffa527f60b532ee12e3ca2859398fc` 的执行协议，不能为旧 Candidate、
Qualification、Admission、Run 或 Seal 重签身份。

本轮仅完成 Mac CPU/Mock 开发。下面是用户后续手动操作界面，不表示服务器已经更新或 GPU 已验证。
新 Source 须先由用户审阅、提交及部署；不能把未提交开发源码或旧 HEAD 当作新正式 Source。

## 准备一次

复用原模型、官方 Tokenizer 小资产、Conda 环境、原始公开 PRE Cohort 和 Reference。
不下载模型，不发布 Deployment、Suite、Physical Admission 或 Permit。
旧 Cohort 仅作为不可变科学输入读取，不能充当新执行资格。

将 [配置示例](../../configs/paper/tom-llm-operations.example.yaml) 复制到用户批准的配置路径，
填写真实路径、GPU UUID 和封存摘要。三个模型共享同一 Cohort、Publication、Reference 和 GPU UUID。
当前正式入口只支持集合任务；V1 概率推理须使用原历史 Source，历史结果不迁移、不覆盖。

```bash
uns tom-llm --config "$CONFIG" prepare --model qwen3 --run-id "$QWEN_RUN"
```

此命令仅使用 CPU：核对提交 Source、固定 150 局／10,714 条有序请求和 Selection；
核对登记模型版本、分片、stat 及小资产摘要；对全部实际任务执行官方 Tokenizer 文本和 token IDs Parity，
核对每条输入不超过 7,680、输出预留 512、上下文 8,192。无截断、筛行或 Mock 预算替代。
只发布 `output_root/run-id/contract`；Run ID 必须是新名称。
也可省略单独 prepare，直接 run 在目标目录不存在时准备一次。

旧 V1 预算不能继承。新正式 Run 要求源码库存干净；Mac 尚有 DTRS 修改时不能宣称正式 Source clean。
Tokenizer 缺失或不匹配即阻断，不把 CPU/Mock 视为真实 Parity。

## 单实例连续执行

```bash
uns tom-llm --config "$CONFIG" run --model qwen3 --run-id "$QWEN_RUN" --execute-live
uns tom-llm --config "$CONFIG" status --run-id "$QWEN_RUN"
```

运行命令仅在 Linux 并明确 `--execute-live` 时启动服务。
服务启动前取得本项目共用 GPU 锁，只查询配置选中的 GPU，要求是空闲 RTX 3090。
不清理其他项目进程，不扫描全局端口、进程或 GPU。端口由 ModelSpec 适配固定为 Qwen3 18581、
Yi 18582、Mistral 18583；占用则失败，不接管。
三个模型使用同一张卡，前一模型的 owned 服务退出后，用户再运行下一条：

```bash
uns tom-llm --config "$CONFIG" run --model yi --run-id "$YI_RUN" --execute-live
uns tom-llm --config "$CONFIG" run --model mistral --run-id "$MISTRAL_RUN" --execute-live
```

每个模型只启动一个固定 vLLM 0.27.0 实例。GET /v1/models 就绪后连续执行真实正式请求；
没有独立合成 Qualification 请求或两次加载。模型原生模板、固定解码参数及 xgrammar Wire 投影保持原定义。
科研完整 Schema 仍保留 uniqueItems，Wire 仅删除后端不支持的约束，应用层继续拒绝重复玩家。
输出为非法 JSON、非法／自身玩家、重复玩家时保留无效预测及原因，不静默修复，也不重试生成。

写入范围为 Run 下的 contract、requests.jsonl、execution.lock、runtime/ 和最后的 seal/；
共享可复用缓存限于对应 Baseline 数据根的 runtime-cache/model 与 runtime-ipc/model。
运行环境强制离线、不写 Python 字节码，TMPDIR 指向本 Run。
启动/停止仅作用于所有权记录准确匹配的 PID；检查 UID、可执行文件、argv、cwd、boot ID、
start ticks 和进程组，并通过 pidfd 发送 SIGTERM。无强制全局 Kill。

## 断线、停止和恢复

```bash
uns tom-llm --config "$CONFIG" stop --run-id "$QWEN_RUN" --execute-live
uns tom-llm --config "$CONFIG" run --model qwen3 --run-id "$QWEN_RUN" --resume --execute-live
```

哈希链日志按任务顺序写入并 fsync；第一次创建日志还 fsync 父目录，dispatch 落盘后才能发请求。
稳定 task digest 和 Run digest 绑定每条请求。
状态查询显示完成数、总数、有效数、失败原因、尝试数、累计已完成请求耗时、当前状态和输出目录。

| 状态 | 恢复行为 |
|---|---|
| pending／无 dispatch | 可首次发送 |
| not_executed | 只有连接发送前失败才使用；显式 resume 后可再次尝试，不自动重试 |
| dispatched／unknown | 可能已送达；阻断，不能自动重发或补造预测 |
| received | 原始响应已落盘，只离线解析；不再次发请求 |
| completed | 预测已落盘，跳过，继续下一条 |

所有非 200 HTTP 结果保守视为 unknown。进程中断在 dispatch 后同样阻断。
完整日志校验只在恢复和结束读取一次，不为每行重新打开 Run 或 Cohort。
响应已落盘但信封无法解析时，保留响应并阻断，不补 uniform 基础设施结果。
语义非法模型内容仍按原规则成为无效输出，并进入统一评价分母。

服务若仍是原拥有实例，resume 复用同一实例；否则仅在无未知请求时重启。
停止超时不 SIGKILL，不删除控制文件，用户检查 owned 日志后处理。
启动发生在 Popen 与所有权落盘之间的歧义，由 launch.pending.json 阻断，不自动接管或发信号。
日志尾部残缺或哈希链损坏也阻断，不自动截断、修复或重发。这些故障需要人工证据审查。

## 结束与统一评价

结束核对完整请求数、唯一性、有序覆盖、持久响应与预测的一致性及 Source 后，
发布一个 `seal/`，含全部 predictions.jsonl 与日志摘要，不再逐条生成中间 Artifact。
输出封存摘要由命令返回，用户记录为外部可信分析锚。三个 Run 必须独立根目录和 Run ID。

```bash
uns tom-llm --config "$CONFIG" report \
  --sealed-run "qwen3=$QWEN_ROOT" --sealed-run-digest "qwen3=$QWEN_SEAL" \
  --sealed-run "yi=$YI_ROOT" --sealed-run-digest "yi=$YI_SEAL" \
  --sealed-run "mistral=$MISTRAL_ROOT" --sealed-run-digest "mistral=$MISTRAL_SEAL" \
  --output "$NEW_REPORT"
```

此命令 CPU 读取；写入唯一新报告目录，存在即拒绝覆盖。
先核对三 Run 的 Source、执行协议、Cohort、Reference、Selection 和有序 Task 身份完全一致，
再一次读取共享 Cohort 与 Reference。输出 analysis.json、metrics.csv、paired.csv、
supplementary.csv、performance.csv、report.md。不为简单集合对照伪造模型 Run Seal。

保留 Full Primary、NU Comparable、空集／全六席／非空真子集诊断，
Always Empty、Always All-Six、NU 上固定 OOF 的严格均匀参照投影，
以及补充概率 KL／TV／Top-1。原始 BeliefObservation 是集合 Gold；
非法输出不丢行，缺失预测阻断完整评价。
游戏等权、10,000 次配对 Bootstrap、比较层级和点态区间解释不变；
已观察 150 局上的 V1/V2 方法差异不能解释为表示改变的因果效应。
计时只报告已收到响应的客户端请求耗时，不声称纯 GPU 时间或 ToM 加速比。

## 校验成本与证据边界

| 频率 | 校验 |
|---|---|
| 准备 Run 一次 | Source、完整 Cohort、固定 Reference 身份、模型小资产和分片 stat、全量官方 Tokenizer／预算 |
| 每次服务启动 | 小资产／stat 与 Run 比对、环境版本、选中 GPU 空闲、项目锁、端口、原生模板、服务就绪 |
| 每条请求 | 公共 Task Wire、owned 进程、状态转换、落盘、严格输出解析 |
| status／stop | 小型 Run 合同、日志状态或 owned 进程身份；无权重 SHA、Cohort 或 Source 扫描 |
| 结束／分析 | Source 与完整持久覆盖、Seal；分析时核对三 Run 及原始标签 |

完整大权重 SHA 不再重算；stat 与已登记版本不是新的加密内容完整性证明。
本协议明确依赖此前已核验模型资产的来源，仍拒绝版本、分片、尺寸、小资产或 stat 漂移。
CPU/Mock 测试不证明真实 Tokenizer 资产、GPU 实际加载、服务就绪或物理吞吐。
历史 bacc351 产物保持原绝对路径及摘要，用其原 Source 解释；新入口拒绝将旧 Run 当成新协议。
