# Observer-DTRS v1 D4c-2b-2：驻留 CPU 服务与有界客户端

状态：本地人工受控验收，未提交、未部署、未取得真实模型 QUAL。
基点：three-way / 7b51f276b04b386ec1aa4ae44454c1d809a10897。

本阶段只新增文件；D4c-2b-1 的 Capsule/Graph/Loader、D4c-1 Hook/Worker、
D0–D3、45 成本与 TiePolicy 均不变。没有真实模型、游戏、vLLM、CAL/TEST 或服务器操作。

## 1. 模块与所有权

| 文件 | 职责 |
| --- | --- |
| werewolf/observer_dtrs/final_tom_wire.py | 版本化 canonical JSON 帧、字节/时间限额、消息来源 envelope |
| werewolf/observer_dtrs/final_tom_service.py | 驻留 CPU 模型、单推理线程、响应控制线程、结束后确认取消 |
| scripts/serve_final_tom_capsule.py | owner 专用 FD 启动入口；没有 TCP listener、游戏或训练入口 |
| werewolf/observer_dtrs/final_tom_client.py | 独占 owner、有界队列、客户端验证、期限、停止/恢复与结果 |
| werewolf/observer_dtrs/final_tom_service_audit.py | 私有不可变 contract/events、链式摘要、覆盖和来源审计 |
| tests/observer_dtrs/resident_service_fixtures.py | 显式 CONTROLLED_FAULT_FIXTURE 子进程；不属于正式服务配置 |
| tests/observer_dtrs/test_final_tom_service.py | 人工权重、PRE、损坏协议、取消与生命周期测试 |

owner 只管理自己 Popen 创建的进程，不接受外部 PID，不向进程组或共享服务发送信号。
子进程使用新 session、独立 Python interpreter、socketpair 的单一继承 FD；
stdin/stdout/stderr 与游戏隔离，不通过 shell、pickle、Agent 或 Hook 启动。
单进程加载一次 Capsule，多个不同 PRE 顺序处理；活动推理并发固定为1。

owner 不导入 torch、Capsule Loader、训练 publication、Shadow 或游戏。
子进程使用原 CapsulePredictor，不导入原 fit/seal、DEV、训练生命周期、角色或报告 reader。
canonical 包 initializer 仍会加载公共/报告 schema 类型，不能将其描述为“完全不含报告类”；
没有访问受限数据文件，也没有把这些 schema 作为推理输入。

## 2. 来源身份与公开输入

ServiceTrust 显式要求四项带外身份：

- Capsule manifest digest 与 scope；
- Loader implementation digest；
- 完整预期 runtime record 的 canonical digest。

cpu_threads / interop_threads 必须与资格 runtime 一致；只在子进程设置线程数，
不改父进程 Python/NumPy/Torch RNG、线程数或 dtype。
本实现仅为 CPU 路径，不能把这次验收解释为 CUDA 内核取消、GPU 显存或真实服务资格。

START → BOOTED → READY，分别绑定：

- IPC v1、owner epoch、独立 boot token；
- 子进程自建 instance ID、实际 Popen PID；
- 服务源码与实际 entry 文件 SHA；
- Capsule 的原始来源、独立 Loader 来源、观察到的 Runtime；
- model_load_count=1、IDLE、qualification=NOT_ESTABLISHED。

服务来源类型为 observer_dtrs_capsule_service_source_v1，不冒称 f199 predictor。
原 f199 只存在于 Capsule 的历史 origin。
普通入口 scope 为 CAPSULE_CPU_QUALIFICATION_ONLY，仍不授权 CAL/TEST；
人工故障入口另标 CONTROLLED_FAULT_FIXTURE，并记录实际入口源码摘要。
改变 owner trust、limits、入口或服务源码后，不得沿用原 contract 启动/恢复。

每个 QUERY 包含 request_id、执行 token、epoch/instance、PRE 原始公共 record、
game/boundary/prefix/plan/token_count 身份、绝对 monotonic deadline、session digest。
同机 socketpair 使用共享 monotonic 时基，不能直接移植为跨机器时钟协议。
服务复用原 canonical PRE parser、roundtrip 与 final capacity validator。

RESULT 包含完整 session、相同 PRE/plan 身份及 7×7 Q。
用已有 ProbabilityMatrix 验证概率质量；不做指数转换、裁剪或重新归一化，
不改变 Capsule 本身的一次 log-Q→Q 转换和死亡目标列语义。
任何失败都没有 Q，不生成 BND，不调用替代模型。

## 3. 有界协议与运行资源

四字节大端长度 + canonical JSON；拒绝重复键、NaN/Inf、非 canonical 编码、未知 envelope、
错 request/token/epoch/instance、错 PRE/plan/source、残缺/超长帧及累计字节超限。
不截断合法 PRE；超过 transport 容量时明确拒绝。

以下是工程开发默认值，**不是冻结的 CAL/TEST 或真实 QUAL 参数**：

| 参数 | 默认 |
| --- | --- |
| cold start / ready | 10s / 15s |
| write / query / service execution | 1s / 10s / 8s |
| cancel / close grace | 1s / 2s |
| terminate / kill wait | 0.5s / 1s |
| incomplete frame age | 1s |
| max frame / public PRE bytes | 4MiB / 2MiB |
| cumulative read / write per service instance | 64MiB / 64MiB |
| active + queued | 2 |
| accepted requests per owner | 1024 |
| active inference | 1 |

read 缓冲不超过单帧+4字节；有生命周期累计计数。输入 JSON 在序列化后检查大小，
因此不能声称从调用者已有的任意大 Python 对象到 wire 的所有临时分配都严格为零。
队列不等待空间；完成/取消后释放公共 PRE bytes，仅保留去重身份与有界结果。
request 数和 event 数也有上限；达到上限拒绝，不自动扩容或自动换实例。

所有网络式等待实际是本地 FD 的 nonblocking send/recv + select，带明确截止。
公开 cancel 的同步等待上限为 cancel_s+2×(terminate_s+kill_s)+0.1s；
close 上限为 close_s+2×(terminate_s+kill_s)，均另受下面的本地 I/O/调度限制。
Popen、OS 调度、JSON/验证 CPU、进程回收和文件 fsync 不是硬实时内核承诺。
审计 I/O 故障会禁用 owner；不能声称磁盘失效时仍已耐久记录失败。
真实 QUAL 要测最坏可接受延迟与资源，不能仅依据本地默认值放行。

## 4. 取消、失败与恢复

主状态：NEW → STARTING → IDLE ↔ RUNNING；取消期间 CANCELING。
发生无法确认的取消、协议/来源错误、崩溃或审计失败则 UNHEALTHY。
tainted 标记独立存在；迟到响应不能把已禁用实例恢复成 IDLE。

- 取消尚未执行的队列项：封存 NOT_DISPATCHED，绝不发送 QUERY，也不取消其他活动请求。
- 活动取消：控制线程记取消意图；仅当推理线程不再存活且 join 完成才发送
  CANCELLED / execution_stopped=true / IDLE。可能是等待原计算自然结束，
  **不是声称 PyTorch 内核被中途抢占**。
- service deadline 到期同样不能在内核仍运行时冒称停止；
  计算结束后返回 SERVICE_TIMEOUT，丢弃 Q。
- client deadline 到期发送 CANCEL；取消期间任何 SUCCEEDED 都拒绝，
  记录 LATE_SUCCESS_REJECTED，等待匹配的停止确认。
- 取消期限内未确认：CANCEL_UNCONFIRMED + 原 trigger；
  立即 taint，terminate 自己的进程，必要时 kill，并 wait/poll 核对退出。
- PID 未退出时 execution_stopped=false；不接纳新请求，recover 也拒绝。
  即使清理成功，超时实例仍不自动复用，需显式 recover 新 epoch/instance。
- 客户端收到超时或无法确认取消不等于 CPU/GPU 已停止，二者分别记录。
  此处 CPU 推理全在该进程线程内，确认退出即不再有该 CPU 推理线程；
  没有声称证明任意后台外部 GPU/远端任务或完整进程树资源释放。

技术状态独立：CLIENT_TIMEOUT、SERVICE_TIMEOUT、CLIENT_CANCELLED、CANCEL_UNCONFIRMED、
SERVICE_CRASH、PROTOCOL_ERROR、WRITE_TIMEOUT、COLD_START_TIMEOUT、READY_TIMEOUT、
CLOSE_TIMEOUT、CLEANUP_UNCONFIRMED、AUDIT_UNAVAILABLE 等。

同 request ID / 同 PRE 只返回原 ticket，记录 DUPLICATE；新 ID 重复 PRE 记录 DUPLICATE_PRE。
同 ID 对应不同 PRE 报错。已接纳或可能进入执行的请求不自动重试，包括失败和跨 epoch。
显式 recover 要求旧 PID 退出、队列/活动项归零、审计完整、来源契约不变；
不恢复原请求，不自动重放。新 owner 不重用旧 journal 目录。

close 停止接纳、取消/排空队列、请求服务关闭并确认退出；
关闭超时只能对 owner 自己的进程执行清理，不能杀共享 vLLM 或采集任务。
保留清理失败状态与 PID 引用，不能仅关闭客户端 socket 就认为计算结束。

## 5. 耐久覆盖与 inspect

只创建全新、外部、0700 owner 目录，拒绝源码/既有 artifact 目录及别名。
目录包含 contract、连续编号 events 和独占 writer.lock。
复用项目 publish_artifact / verify_artifact、fsync、不可覆盖发布与 canonical SHA。

分别记录 EPOCH_START、READY、ADMISSION/拒绝、DISPATCHED、STARTED、OUTCOME、
迟到成功拒绝、START_FAILED、CLOSE。
每个接纳请求至多一个 OUTCOME；未接纳、未发出、发出但未确认 STARTED、执行与终态不能混为一类。
inspect 分别返回 submitted/dispatched/started、终态和拒绝计数、incomplete、execution_unconfirmed；
技术终态已封存不等于进程停止已确认。这里的分母是 owner 请求，整局/PRE 资格分母仍由 D4c-1 capture 台账负责。
不覆盖失败，不修复坏日志，不将不完整记录当成功。

每次 append 验证 contract 和 tail；recover/close/inspect 做完整来源/序列/hash-chain、
PRE、Q、重复终态与来源检查。不是每次 append 扫描整个历史，因此旧记录损坏可能在全量
inspect 才发现。正式封存必须 inspect 并带外固定 head；没有外部 head 不能证明抗整条链重写或尾截断。
日志只有合法 PRE 身份引用、Q、来源、技术状态；不保存原始 PRE 文本、Y、嫌疑报告或角色 sidecar。
它是独立私有日志，不是 canonical recorder 的替代品。

审计损坏/磁盘错误后停止新接纳。若 OUTCOME 写入失败，返回 AUDIT_UNAVAILABLE，
不返回未封存的有效 Q；inspect 如实显示 incomplete，而非伪造“失败也已完整保存”。

## 6. Shadow 接缝

现有 TomIdentity / QPrediction / ShadowEngine 没有修改；当前严格 fixture/OOF/final 检查保持。
ResidentResult 暴露经验证的 ProbabilityMatrix 与完整 session，不能强转成旧 TomIdentity，
不能把原 f199 revision 填入新服务身份来掩盖 Loader 或 Capsule。

建议后续正式概率包绑定稳定的 TomDeploymentIdentity：
capsule digest、原 model raw SHA、公共编码版本、Loader digest、经 QUAL 的 runtime profile。
RPC epoch/instance/token 是一次服务寿命/请求租约，不应使每次重启都变成一个新统计模型。
CAL 包另行发布，严格匹配上述稳定身份；旧 D2 OOF 包不兼容。
本阶段只以显式人工绑定 p 验证 Q→ObserverFeatures→原 decide，
不是 CAL 拟合或正式 Shadow 部署，也不产生真实成本比较。

D4c-1 Hook/Worker 与非阻塞、失败隔离、覆盖统计没有改动；本 owner 需要在游戏外明确创建和关闭。
当前 owner 内含有界客户端，但带线程/socket/Popen，不能直接 pickle 给现有 spawn Worker。
正式接入还需要显式的跨进程客户端/owner broker 适配；本轮人工桥接没有冒称完成这项集成。
完整连接驻留服务和新的 CAL 概率包需要单独审阅，不向原 prompt/handoff/状态传递 Shadow 输出。
人工 state/RNG 检查与旧 D4c-1 fake-backend 回归，不能证明真实语言服务资源竞争时文本逐字一致。

## 7. 本地验收与 D4c-2b-3

实际命令、数量、源码指纹和临时人工证据见配套 acceptance JSON。
最终相关回归 **730 passed，224.13s**，含本阶段全部 **47 项**新增测试；没有跳过或失败。
独立服务测试一轮为47 passed / 85.94s；最终源码又纳入上述完整回归验收。
原/驻留 Q 逐位一致、同 PID 多 PRE、普通/PK及单狼/双狼 context、死亡目标质量、
重复/冲突、队列满/过期/取消、冷/ready/query/write/close期限、进程崩溃、
错 token/epoch/instance/source/PRE/Q、残缺/超长/累计限额、清理失败、恢复新 epoch、
日志损坏与重算摘要后的错配、父/子 RNG、禁止受限导入和目录读取均有受控验收。

仍需用户另行批准并手动完成的真实证据：

1. 实际 final Capsule 编译与外部 digest；本阶段没有读取真实权重。
2. 原 sealed predictor 与新 Loader/驻留服务的实际张量、Temporal、log-Q/Q 数值等价；
   不能用人工等价替代，不能用32节点字节/codec审计替代完整语义恢复。
3. 原 recovery loss/schedule、optimizer/RNG 恢复链的独立状态。
4. 固定源码 OID、服务/Loader/Runtime guard、真实 OS 用户/权限隔离：
   本地 Python import/open 拦截测试不是 OS ACL 安全沙箱。
5. 最大合法 PRE、冷/热启动、CPU/RSS/VRAM、容量、并发/取消、进程树/GPU资源释放、
   延迟与日志磁盘预算；真实 CUDA 或远端服务不能直接继承本 CPU 取消结论。
6. 独立 QUAL/CAL/TEST 来源排除与正式参数冻结、CAL 概率包发布、TEST 标签隔离。

可进入 D4c-2b-3 的资格验证准备和人工脚本审阅；不代表本轮已取得真实资格，
也不授权实际模型加载、服务器操作或 CAL/TEST。研究协议仍为 PROPOSED。
固定历史 ToM、历史 DEV 与全量 publication 摘要影响 RNG 的来源限制继续保留；部署封装不会消除这些限制。
