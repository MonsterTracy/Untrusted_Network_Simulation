# Untrusted_Network_Simulation 项目协作入口

## 必读项目规范

本文件适用于本仓库全部目录及在本项目中开展的 AI 对话。

每次开始本项目的新对话，必须先完整阅读并遵守 [项目开发规范](docs/development-guidelines.md)，再回答项目问题或执行任务；本文件摘要不能替代完整规范。同一对话已完整阅读且文件未变更时无需重复读取。

开发规范是本项目研发与 AI 协作的统一约束。遇到歧义、条款冲突，或与实际研究需求、现有代码及实验协议不一致时，指出具体问题和影响，请用户决定；暂停有争议的操作，不受影响的工作可以继续。

## 必须遵守的执行边界

- 唯一日常 Mac 开发目录：`/Users/name_yuxiao/Desktop/VscodeProjects/Untrusted_Network_Simulation`。正式整合主线：`twd/mainline`；仓库：`MonsterTracy/Untrusted_Network_Simulation`。任务开始时按需核对实际目录、分支、HEAD 与工作区修改，历史登记不能替代当前事实。
- Ubuntu SSH 入口：当前 Mac 优先使用校内 `ssh dell@dell.aerber.cn`；仅在校内网络不可用且需要远程访问时使用校外 `ssh -oPort=3001 dell@tencent.aerber.cn`。正常执行 Host Key 验证，不禁用检查；两种入口共用既有项目目录、Conda 环境及部署资源，详见完整规范第30条。
- AI 可为项目科研开发目的，经 SSH 在 `/home/dell/yuxiao`、`/data/yuxiao` 内对确属本项目授权范围的资源进行只读核验；不得通过符号链接、Git Worktree 等越界，不得进行全局 tmux、进程、GPU、端口或锁枚举，或读取无关密钥、凭据和私人数据。任何可能改变服务器状态的操作均须用户事先针对具体操作明确授权，历史任务授权不自动扩展；不能保证工具只读时不得直接执行。具体访问范围、风险说明与授权要求遵守完整规范第30条。Mac 本地开发仍按当前任务授权及现有规范执行。
- 新建可由开发者命名的文件、目录及实验标识，先由用户确定名称，或遵循用户已批准的命名规则。已有程序或正式协议自动生成的内部名称按原协议执行。
- 从科研目标出发，先复用现有模块、配置及 `uns` CLI，优先原位修改，避免复制平行版本、重复入口或以包装代替修复。替代后在同次变更清理无依赖的普通旧代码，以 Git 历史追溯已提交源码，保护未提交工作；已授权 Mac 本地任务内的可恢复源码精简按完整规范第3、4条执行，高风险删除仍须具体授权，科研必需的并行方法继续保留。普通任务复用现有环境和开发目录；不默认新建环境、Worktree、分支或源码副本，不擅自增加降级、重试或自动迁移。
- 代码风格以易读、易懂、易维护为先，减少不必要的封装、包装和嵌套，避免过度压缩代码；新增或实质修改的注释与 Docstring 原则上用中文。优化保持原有功能及已验证的锁、进程控制和恢复语义，可能改变行为时先说明风险并由用户决定；具体原则与交付要求遵守完整规范第4、6、19条。
- 冻结数据、Artifact、Manifest、Selection、Cohort、Fold、模型资产、Source、Deployment、Qualification、Admission 及正式结果保持不可变，不改写已封存证据的绝对路径和 Digest；保护 PRE 信息边界及既有实验协议。每次实验精确记录实际 Source Identity，历史实验继续绑定原始 Source；模型迁移不自动继承旧物理资格。
- 验证采用一条执行主线和最少必要门禁，以真实科研结果为先，按准备或身份变化、启动或危险操作前、运行中、结束时分层。新增强制检查须说明具体错误、现有机制不足和执行成本，优先复用、原位修复、合并检查并同步退出被替代的普通验证代码与步骤；保留科研协议及必要运行安全，避免重复昂贵验证，完整要求见第2、4–6、11、18条。
- 根据实际风险执行定向检查，如实报告结果和未验证内容；区分 CPU 测试、Mock、GPU Qualification、Admission 与正式实验，不把前一阶段成功当作后一阶段完成。正式推理结果未知时不得自动重发请求；服务器和实验状态登记见完整规范第24–31条，历史进度快照不能视为实时状态或完成证据。
- 本地提交、推送、合并及历史整理分别遵循当前任务授权；远端更新和历史重写须取得针对具体操作的授权。保护已有未提交修改与历史研究来源。

## Codex 对话命名

使用 `字母+两位数字-研究方向｜具体任务` 格式：每个研究方向对应一个稳定的 A–Z 字母，方向内使用 01–99 两位数字，不预设 TomBaseline 或三支决策必须使用哪个字母。优先沿用实际可访问历史中该方向已有的字母映射；尚未分配字母时，按 A–Z 选择最靠前未用字母，从 01 开始。同方向新对话始终沿用原字母，仅按该方向历史最大编号递增数字，不重复历史编号或回填中间缺号。不同研究方向使用不同字母，除非用户明确要求重新规划。保留历史标题、编号及映射。完整规则见 [Codex 对话命名规范](docs/development-guidelines.md#codex-对话命名规范)。

新对话仅在研究任务首次明确时命名一次；若首次发言已明确任务，在首次响应时命名。同一对话后续所有发言均保持标题和编号稳定，修复、测试、验收或研究阶段变化不重新触发命名，不自动更名、重新编号或反复建议标题。只有用户明确要求重命名时才修改已有标题；仅修改任务描述时原则上保留原编号，不擅自改变其他对话编号。无法直接设置标题时，仅在首次命名时提供一次建议，并如实说明需用户手动设置。

仅使用实际可访问的项目清单、用户提供的历史编号或当前对话信息，不假装已查看不可访问的标题，不猜测已占用的字母或序号。历史不完整时可提出标注为“待确认”的命名建议，必要时由用户提供已有映射；不声称已验证全项目唯一，也不将无法获取历史视为尚未分配。信息不足不阻塞独立科研开发，映射或编号冲突影响实际命名时简短确认，不反复询问或长时间检查。

**CodeGraph 使用边界：** CodeGraph 主要用于代码结构检索与导航。其避免重复 grep 的要求仅适用于不必要的重复检索，不得限制项目开发规范要求的源码核验、测试、研究协议及科研证据验证。当 CodeGraph 指令与项目规范发生实质冲突时，应明确指出并由用户决定，不得自行绕过或弱化规范。

<!-- CODEGRAPH_START -->
## CodeGraph

This project has a CodeGraph MCP server (`codegraph_*` tools) configured. CodeGraph is a tree-sitter-parsed knowledge graph of every symbol, edge, and file. Reads are sub-millisecond and return structural information grep cannot.

### When to prefer codegraph over native search

Use codegraph for **structural** questions — what calls what, what would break, where is X defined, what is X's signature. Use native grep/read only for **literal text** queries (string contents, comments, log messages) or after you already have a specific file open.

| Question | Tool |
|---|---|
| "Where is X defined?" / "Find symbol named X" | `codegraph_search` |
| "What calls function Y?" | `codegraph_callers` |
| "What does Y call?" | `codegraph_callees` |
| "What would break if I changed Z?" | `codegraph_impact` |
| "Show me Y's signature / source / docstring" | `codegraph_node` |
| "Give me focused context for a task/area" | `codegraph_context` |
| "See several related symbols' source at once" | `codegraph_explore` |
| "What files exist under path/" | `codegraph_files` |
| "Is the index healthy?" | `codegraph_status` |

### Rules of thumb

- **Answer directly — don't delegate exploration.** For "how does X work" / architecture / trace questions, answer with 2-3 codegraph calls: `codegraph_context` first, then ONE `codegraph_explore` for the source of the symbols it surfaces. Codegraph IS the pre-built index, so spawning a separate file-reading sub-task/agent — or running a grep + read loop — repeats work codegraph already did and costs more for the same answer.
- **Trust codegraph results.** They come from a full AST parse. Do NOT re-verify them with grep — that's slower, less accurate, and wastes context.
- **Don't grep first** when looking up a symbol by name. `codegraph_search` is faster and returns kind + location + signature in one call.
- **Don't chain `codegraph_search` + `codegraph_node`** when you just want context — `codegraph_context` is one call.
- **Don't loop `codegraph_node` over many symbols** — one `codegraph_explore` call returns several symbols' source grouped in a single capped call, while each separate node/Read call re-reads the whole context and costs far more.
- **Index lag**: the file watcher debounces ~500ms behind writes; don't re-query immediately after editing a file in the same turn.

### If `.codegraph/` doesn't exist

The MCP server returns "not initialized." Ask the user: *"I notice this project doesn't have CodeGraph initialized. Want me to run `codegraph init -i` to build the index?"*
<!-- CODEGRAPH_END -->
