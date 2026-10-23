# Observer-DTRS v1 D4c-1：最小 PRE Hook 与非干预验收

本阶段是本地人工验收，未提交、未接真实服务、未生成新研究游戏。
基点为 `three-way / e513fd8b69404d07d69cd308023b673554f5bbf0`。
实际命令、日志、文件指纹与保护检查见 `observer-dtrs-v1-d4c1-acceptance.json`。

## 1. 插入位置与修改边界

执行顺序：

1. 原 `CanonicalGameRecorder.before_agent_act` 冻结公共 PRE，按原规则采集全部存活玩家报告，构造原 speaker handoff。
2. 仅当显式传入 `observer_shadow` 且处于 `speech/speech_pk`，控制层取得匹配的只读 PRE；可信捕获器读取当前 actor 的合法 observation 白名单。
3. 捕获器验证公开身份绑定，再调用原 `build_shadow_request`；worker 只收到不可变的 `ShadowRequest`。
4. 原 online Pilot / constrained speech / Agent cognition、发言、报告、`env.step` 路径继续，完全不读取 Shadow 返回值。

`run_random.eval(..., observer_shadow=None)` 默认关闭；关闭时不导入或实例化 Shadow、不新增日志与线程。
`CanonicalGameRecorder.public_pre_for_handoff(handoff)` 验证同一未消费 handoff、boundary、prefix、actor 后，只返回原不可变公共 PRE。
新 Hook 不读取 `_pending`，不把 handoff、env、agent、recorder 或其他人的 observation 交给 Shadow。
控制层仅额外提取公开 `recorder.claim.attempt_id` 这一不含角色/报告的来源标识，与 Hook 配置严格匹配；错 attempt 记录捕获失败，不推理且不停止原游戏。
已有 wolf_speech_tom 路径中的旧 `_pending` 访问没有顺便重构。

只有上述两个已有控制文件发生修改；D0–D3 估计器、决策数学、协议、ToM、Phase2 和原实验产物不变。

## 2. 新增接口

| 文件 | 职责 |
| --- | --- |
| `werewolf/observer_dtrs/shadow_hook.py` | 可信 PRE 捕获，公开字段与当前 actor 联结，wolf/非wolf/未知资格分流，屏蔽技术异常 |
| `werewolf/observer_dtrs/shadow_worker.py` | `WorkerSpec`、`IsolatedShadowWorker`，有界入队、spawn 子进程、硬超时、独立 supervisor、生命周期与结果验收 |
| `werewolf/observer_dtrs/shadow_receipts.py` | `ReceiptLedger`，独立耐久捕获/终态记录、单写者锁、链式摘要、覆盖统计 |
| `tests/observer_dtrs/shadow_worker_fixtures.py` | 可 spawn 的人工 ToM，模拟卡死、崩溃、迟到和错 token；不导入游戏或监督 |
| `tests/observer_dtrs/test_shadow_worker.py` | 进程、截止、队列、幂等、损坏与中断恢复验收 |
| `tests/observer_dtrs/test_shadow_hook.py` | 真实 GPTAgent 的人工 backend 对照与原 canonical recorder 证据对照 |

所有者的初始化和收尾操作在游戏外完成：

```python
spec = WorkerSpec(artificial_provider, controlled_model, hard_timeout_seconds=5.0)
worker = IsolatedShadowWorker(new_private_root, spec=spec, max_pending=2)
hook = ShadowPREHook(worker, shadow_run_id=run_id, attempt_id=attempt_id)
try:
    winner = run_game(env, agents, roles, canonical_recorder=recorder,
                      call_audit=audit, observer_shadow=hook)
finally:
    worker.close(timeout=30.0)  # owner-side drain; never inside the game thread's PRE hook
audit = worker.inspect(expected_admitted_requests, expected_head=external_receipt_head)
```

这只是人工集成接口说明。没有新增真实游戏 CLI、自动启用配置、真实模型加载或线上部署命令。
Hook 不持有先前 observation；合法狼队信息只从当前 actor 的过滤 observation 提取。
`observer_id` 确由当前 `get_observation_for` 提供；旧 prompt 测试 fixture 缺少此字段，新的受控 fixture 显式补齐了真实字段，生产环境没有增加观测字段。

## 3. 隔离与硬超时

游戏线程不等待队列容量或 ToM；`BoundedSemaphore.acquire(blocking=False)` 限制**运行中 + 等待中**请求数。
一个 supervisor 线程串行管理请求，每个请求使用独立的 multiprocessing **spawn 子进程**。
IPC 只携带已验证模型的不可变合法载荷、人工 provider、公共 PRE/合法 team 请求和执行 token。
子进程调用原 `ShadowEngine.infer`，仍执行原 Q/来源/容量检查、条件属性、概率、45场景×显式并列策略；没有新决策公式。

硬截止包含 spawn、推理和结果封存；不依赖 provider 主动配合 deadline。
截止后先 terminate，最多等待0.25秒，再 kill，最多等待1秒；不能确认子进程结束时停用 Shadow，不再启动新推理。
按 request_digest 和 admission token 验证独立 pipe 响应，截止后的成功和错 token 不接纳。
只有子进程正常退出、返回身份匹配、结果不可变产物与返回摘要一致时，才写 SUCCEEDED。
失败不产生认知行，不转成 BND、均匀 Q 或默认概率。

当前仅允许 controlled_fixture ToM 与 controlled_fixture 概率包；来源不匹配仍由原 engine 拒绝。
没有放宽 OOF/final 配对，没有选择 D2 outer 模型，没有唯一主成本矩阵。
真实服务若由 client 子进程调用，杀 client **不保证取消远端 GPU 作业**；这项容量/取消资格留到 D4c-2，当前没有服务调用。

## 4. 耐久审计、计数和恢复

输出根目录必须仅当前用户可访问。新目录使用0700；已有权限不合格时拒绝，不自动改权限。
独立目录结构：

```text
PRIVATE_ROOT/
  receipts/contract/{manifest.json,record.json}
  receipts/events/00000000/{manifest.json,record.json}
  receipts/events/00000001/...
  receipts/writer.lock
  inference/contract/...
  inference/claims/REQUEST_DIGEST/...
  inference/results/REQUEST_DIGEST/...
  inference/failures/REQUEST_DIGEST/...
  inference/writer.lock
```

复用 `publish_artifact/verify_artifact` 的原子发布、fsync、SHA-256 与禁止覆盖机制。
capture 与 outcome 分开封存，序号连续、previous 链绑定，contract 绑定 engine、模型、队列、超时、spawn、重试规则和受控工作区源码指纹。
访问器异常只从已存在 handoff 白名单提取 boundary/prefix/actor/run/attempt 标识写 CAPTURE_ERROR，不传递 handoff 的报告内容。
capture 保存 PRE 来源及合法 request identity；没有真实报告、Y、sidecar、训练支持或游戏 action。
没有原始异常消息写入日志，以免泄漏任意 provider 载荷。

| 计数 | 口径 |
| --- | --- |
| capture_attempts | 耐久的每次 speech PRE 捕获尝试；重复调用也计数 |
| eligible_captures | 成功构造合法狼人 request 的捕获次数，包含重复、队列满和停用跳过 |
| unique_eligible_PREs | 上述 request_digest 去重，包含 run/attempt/canonical PRE 身份 |
| eligibility_unknown | CAPTURE_INVALID + CAPTURE_ERROR；来源不足，不能算作已验证的非狼 |
| submitted | SUBMITTED；先耐久记录 admission，后内存入队 |
| skipped | SKIPPED_NONWOLF、SKIPPED_BACKPRESSURE、SKIPPED_DISABLED 分列 |
| duplicate | 已接纳 request 再次到达；不重复推理；以前队列满的请求不算已接纳 |
| completed | 每个接纳请求恰好一个耐久技术终态，不等于成功数 |
| success | SUCCEEDED，受返回摘要与结果文件共同绑定 |
| failure | FAILED_TIMEOUT/CRASH/INFERENCE/START/STOPPED、INTERRUPTED_NO_RETRY 分列 |
| pending | submitted − completed，不能纳入成功率分母之外而不报告 |

干净封存应满足：所有捕获分类之和 = capture_attempts；终态分类之和 + pending = submitted。
`inspect(requests=...)` 还可用完整预期接纳请求核对来源，并用原 engine 重算成功结果语义。
只有模型与结果文件摘要的校验不证明实际权重计算；外部 receipt head 及结果摘要锚用于识别整尾截断/整体替换。
正式来源仍需独立冻结，不能把内部自洽宣传为真实性证明。

重启只读取、验证历史，不自动重放已接纳请求；没有确认终态的 admission 封存为 INTERRUPTED_NO_RETRY。
这也适用于 fsync admission 后、尚未入队就中断的情况。
子进程被杀可能留下 inner inference claim；外层技术终态有权威接受状态，inner pending/未确认结果仍保留供审计，不删、不冒充成功。
原 writer 仍活动、日志损坏、错源码/模型/配置时拒绝恢复，不清理修复。

**非阻塞边界**：非阻塞指不等待队列容量或推理。耐久 capture 要同步完成磁盘 fsync，也会短暂取得审计锁；不保证硬实时 I/O 延迟。
当前完整链复核适用于每局/每attempt的短日志；正式部署应使用每attempt独立私有根目录，不能未经容量验证直接复用一个无限增长的全局ledger。
fsync 失败时立即停用 Shadow，暴露 audit_error 和本进程未耐久捕获计数，原游戏继续。
若磁盘完全失效，不能承诺失败记录也已耐久；进程猝死更不能证明尚未封存的捕获没有发生。
因此正式完整覆盖必须与原 canonical PRE 总体/外部锚对照，不能仅凭成功日志或健康布尔值声称100%覆盖。

## 5. 非干预证据与执行限制

人工对照采用真实 GPTAgent + fake backend，执行 speech 和 vote 两个控制步骤。
20组配对覆盖：普通/PK × 单狼/双狼 × 正常/硬超时/崩溃/日志损坏/队列满。
逐字节比较 Agent observation、原 PRE handoff、实际 cognition/realization/vote prompts、返回发言、投票、事件、winner、报告次数与 canonical action。
Python/NumPy 最终 RNG 完全一致；人工 provider 特意在子进程消耗两种 RNG，并断言没有游戏或受限监督模块导入。

另使用原 canonical recorder 和现有确定性 one-speech environment fixture，分别验证首发非狼和首发狼人。
不删字段地比较完整 canonical evidence（含原 FrozenJSON 字节、报告、调用证据、private replay）、backend 请求与预算。
这是人工环境的严格对照，不是新一轮真实研究游戏。

最终实际通过数以 acceptance JSON 为准。额外尝试的两个旧 Phase2 Pilot 测试因本地缺少 scikit-learn 在收集阶段失败，不能报告为通过。
原有一个较重的 ToM memorization 测试沿用此前回归范围排除；没有以它的未执行作为新功能证据。
Hook 的覆盖总体为原 `before_agent_act` 成功返回的 speech PRE。若原报告采集已失败，原游戏仍在 Hook 前按原规则停止；此类 PRE 需与原 canonical failure evidence 对照，不能仅看 Shadow 日志声称全覆盖。

默认关闭路径保持原输出；人工开启路径的技术失败不会改变原游戏动作或采集语义。
真实服务下仍可能有 CPU/磁盘/GPU 资源竞争和延迟，不能预先承诺采样生成文本逐字一致。
原游戏自身的报告、Agent 或 env 错误仍按原 canonical 失败规则处理，Shadow 不吞掉原游戏错误。

## 6. D4c-2 的最小冻结项目

可以进入部署协议和 CAL/TEST 来源冻结，尚不能直接运行正式新游戏 Shadow：

1. 固定 final ToM 的完整 source/fit或execution/seal/terminal/weights、容量、时间编码、token-plan、运行环境和 provider 的实测身份。
2. 预先登记概率部署包的选参/拟合来源与 ToM 配对。优先 final ToM + 独立 CAL；固定包后才开放独立 TEST。不得按 D3 风险选 outer 模型，不暗中允许 OOF→final。
3. 固定整局 CAL/TEST seed/game pool、监督开放时点、失败/排队/容量规则和来源排除集合。
4. 验证真实 ToM 客户端的并发、GPU负载、硬超时后的服务端取消/作业占用、退出与资源释放；冻结队列、截止、磁盘预算和审计锚交接。
5. 审阅并提交新的 D4 源码后，再验收正式 guard 与唯一新OID。目前 D4 未提交，旧 D2/D3 guard 应拒绝，不能绕过。

D0事件、D3的45成本与并列策略、固定历史 ToM、后验开发数据、历史RNG来源与固定预测 Bootstrap 的研究限制保持不变。
没有发言动作映射、胜率结论或真实在线概率校准结论。
