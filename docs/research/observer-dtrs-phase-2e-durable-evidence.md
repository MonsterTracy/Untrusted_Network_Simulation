# Phase-2E：独立研究持久化与外部读回受控验收

状态：**CONTROLLED_RESEARCH_STORE / REAL_INTEGRATION_NOT_APPROVED / ACTION_EXPERIMENT_NOT_STARTED**。
W3：`twd/mainline@ec008ef50dd10f2c678aa2d7e7439d31e333b80d`。本阶段源码未提交。
仅新增独立模块、测试及本文档，不改Phase-2D状态机及原16项未提交成果。

## 身份与尝试范围

`canonical_collection/production_runtime.py:112`在生产采集中以collection/claim ordinal/seed生成game_id；
`collector.py:115`生成attempt ordinal/seed身份，`collector.py:274`说明中断claim终结而不重跑该seed。
`attempt_ledger.py:1301`分别拒绝重复ordinal和attempt_id；`next_planned_attempt`继续下一个ordinal。
这些是同一collection/plan中的唯一性，不是任意环境重试的全球唯一性。
`runtime.py:249`的boundary是game_id加公共event_index；相同game_id的重新执行可能重用boundary。
Recorder构造器接受外部game_id，并不单独建立不可重用的执行尝试身份。

Phase-2D纯validator按(game_id,boundary_id)去重，日终按(game_id,day)共享。
不能把它直接用于多尝试混合历史。本阶段不修改受保护模块：每个持久化store的不可变contract
绑定唯一ledger_id、game_id、game_attempt_id，以及规范化后的journal/anchor绝对路径。
OPPORTUNITY进入纯validator之前检查game/attempt，重开也验证完整历史。
不同尝试必须使用不同store、ledger身份和锚点；此实现不选择游戏机会、不定义真实重试策略。

这解决**已声明的同一store内**尝试混用。不会证明调用者声明真实：原DAY_WITNESS没有attempt字段，
给它错误尝试的外部见证仍可能语法自洽。真实集成还需attempt-aware的公开结算reader和独立来源验证。
不能仅因日终digest存在就声称已经验证其真实尝试或日终内容。

## 复用与接口

`ActionEvidenceStore.create(root, anchor_root, ledger_id=..., game_id=..., game_attempt_id=...)`
独占创建新的0700目录；文件0600，拒绝符号链接根、非普通文件、硬链接、替换和额外文件。
父目录必须已存在，不自动建立其他实验目录。失败不清理已产生的证据。
`reopen(..., expected_anchor=...)`要求调用方给出可信的64位高水位摘要，连空历史也不允许无锚恢复。
`append(operation_id, kind, payload)`调用原`extend_ledger`，不再实现五类记录状态机。
`inspect()`重放原`validate_ledger`并核验完整journal/anchor。
`records`返回副本；`close()`仅释放自己拥有的fd/锁；支持context manager。

`DurableReceipt`保存operation、原记录摘要、sequence和**当前**高水位摘要，`execution_permission=False`。
重复operation和完全相同内容只返回原记录回执，不追加；同ID异内容拒绝。
纯`extend_ledger`的返回值没有被改成提交许可，durable回执也不是游戏提交许可。

## 所有权与持久化顺序

| 步骤 | 规则 | 中断含义 |
|---|---|---|
| 所有权 | journal和独立anchor目录的生命周期非阻塞flock；原进程/线程绑定 | 竞争writer立即拒绝，不等待、不偷锁、不删除lock文件 |
| 读前核验 | 文件inode、目录inode、模式、contract、完整规范JSON帧及原状态机 | 外部修改或替换导致停止 |
| 记录追加 | 完整JSONL、写满检查、fsync数据fd | 半帧或未确认数据不允许恢复执行 |
| 高水位 | 独立路径append-only JSONL，绑定contract、generation、previous、count、head、byte_count、全部字节SHA；fsync | 记录比锚点多时拒绝；不自动接纳后缀 |
| 返回 | 两次fsync返回后才确认回执 | ACK丢失仍可能已写入；旧head拒绝，不能重新执行动作 |

两类文件每个上限16 MiB，用于小型研究ledger；超限明确失败，不截断。每次追加重放短历史，
不宣称适合大规模日志性能。单次创建和追加不是跨文件原子事务；失败窗口会保留不一致供审查。
append异常将实例标为unhealthy，必须关闭；不自动修复、截断、重试或选择分支。
创建阶段fsync文件及目录，恢复不创建缺失文件、不覆盖已有内容。

**信任边界**：flock只排斥遵守同一协议的本地writer；不隔离同UID恶意进程、NFS/多主机或真实游戏执行器。
独立目录不等于独立信任域。生产抗回滚需要调用方/外部保管者保存不可回滚的最新摘要。
没有可信锚点时不开放可写恢复。日志、anchor和调用方摘要若一起回滚，系统无法识别；测试明确展示这一限制。
ACK丢失后从当前anchor文件自行算出摘要仅可作**人工fixture oracle**，不是独立生产历史证明。
本轮没有建立或假冒正式可信锚点服务；真实集成在此项上未获准入。
进程kill测试证明本机进程级关闭/锁释放，不等于断电、磁盘控制器缓存或所有文件系统的持久性证明。

## 外部读回：只收证据，不再执行

`reconcile_readback(store, operation_id=..., assignment_ref=..., provider=...)`向注入reader提供
不可变`ReadbackQuery`：assignment_ref、reservation_ref、attempt_token及规范化opportunity字节。
没有env、Agent、提交或回退回调。

- provider返回None或超时：维持原未知状态，不补成功/失败结论。
- 有见证：只接受COMMITTED/NOT_COMMITTED，绑定原assignment/reservation；完整PRE/token/停止证据由原validator核验。
- 错attempt/prefix/token、缺停止证据或非法字段：拒绝，磁盘不改变。
- 已解决后的同operation重读回执不再调用provider；不同operation不得改写既定结论。
- NOT_COMMITTED也不发放再次执行许可；reservation不能重抽、重提。日终缺失仍是MISSING，不是无放逐。

人工provider提供的readback_digest只是声明，模块不能独立证明执行器已停止、外部历史真实或provider权限。
本轮无生产provider、无F0/F1执行机制、无fallback、无真实随机化或奖励。

## 原提交核心只读结论

`realize_and_commit`复用原pending PRE、一次Actor/Parser和bind_verified_speech，不重新收集PRE/报告。
`CanonicalGameRecorder.commit_verified_speech`先stage/验证，然后`_mark_speech_annotation`，
接着`env._publish_verified_speech`，最后更新Recorder的pending/actions/raw_actions。
`after_env_step`产生SubmittedGameplayAction及其公共event绑定，并清空pending。
这些结构可以成为未来reader的核验对象，**当前没有**可跨进程查询的durable assignment/attempt_token幂等提交接口。
已有`public_pre_for_handoff`只读pending的身份，不是目标提交的完整读回接口。

| 外部窗口 | 当前能力 | 仍缺证据 |
|---|---|---|
| Actor/Parser失败 | 原严格单动作语义检查；通常尚未调用commit | 超时不证明远端执行终止 |
| stage失败 | 原环境正常验证路径未发布 | 任意异常不能一概视为未提交 |
| env发布后、Recorder更新前 | 两者可能短暂/永久不同步 | 缺跨两者事务及durable readback |
| Recorder更新后、研究日志失败 | 外部可能已经提交 | 不能从日志缺失推断未提交 |
| 研究reservation持久后崩溃 | 原assignment和意图可恢复 | 不知道是否调用/完成外部提交，必须读回 |

本轮没有修改提交核心来掩盖这些缺口。研究store只确认记录，不保证外部环境严格一次提交。

## 验收口径

测试全部使用人工PRE/固定assignment、人工external reader及任务自有临时文件/子进程。
真实多进程竞争、kill和`os._exit`覆盖before-data、半帧、data-fsync后、anchor-fsync后；
另外覆盖有效分叉、有效前缀后缀丢失、旧head、共同回滚限制、锁文件替换、路径别名、权限、重复恢复及日终共享。

首轮新测试34通过/2失败：错误地把同operation的幂等回执读取当成新提交意图。仅修正本轮测试，
改用新operation/token测试拒绝第二意图；保留原失败日志。不改Phase-2D语义。
最终数量、命令、退出码、JUnit、逐case日志/原始人工journal及SHA在项目外验收包中列明，不累计中间通过数。

三层结论：
1. **受控实现**：独立研究写盘/读回及本机故障验收；条件是声明的单尝试、合作锁协议和可信外部锚点。
2. **真实集成**：NOT_APPROVED。缺实际外部reader、执行器停止证明、可信锚点保管及跨env/Recorder协调；没有生产exactly-once证明。
3. **行动实验**：NOT_STARTED / NOT_READY。未冻结G2、F0/F1、目标、随机概率、继续策略，也未生成行动收益监督。
