# Untrusted_Network_Simulation 项目协作入口

## 必读项目规范

本文件适用于本仓库全部目录及本项目的 AI 对话。新对话先阅读本文件，再按任务需要读取 [项目开发规范](docs/development-guidelines.md) 的相关章节；简单任务无需反复完整通读。

执行前至少阅读适用约束：实现与代码整理见第2–6条，命名见第8条，验证和结论见第11、18、19条，Git 与工作区见第14–16条，服务器与历史资产见第24–31条。涉及冻结研究须同时遵守第6、12条。遇到实质歧义或冲突，说明问题和影响，由用户决定；只暂停有争议的操作。

## 必须遵守的执行边界

- 唯一日常 Mac 源码目录：`/Users/name_yuxiao/Desktop/VscodeProjects/Untrusted_Network_Simulation`；唯一正式、持续维护的 Ubuntu 源码目录：`/home/dell/yuxiao/Untrusted_Network_Simulation`。统一主线 `twd/mainline`，仓库 `MonsterTracy/Untrusted_Network_Simulation`。按任务核对实际 Git 状态；历史 Worktree 仅在真实需要时按原 SHA 临时恢复，遵守命名和服务器授权，不作为常驻开发目录。
- 科研数据根目录：`/data/yuxiao/Untrusted_Network_Simulation`。模型、环境及既有资产按实际职责管理，不因统一源码强制合并或迁移；历史目录退出不等于其 Source 和证据可删除。
- Ubuntu 优先使用校内 `ssh dell@dell.aerber.cn`；校内网络不可用且确需远程访问时使用 `ssh -oPort=3001 dell@tencent.aerber.cn`。正常验证 Host Key，两种入口共用既有资源。
- AI 只读访问限于 `/home/dell/yuxiao`、`/data/yuxiao` 内授权的本项目资源，不通过符号链接或 Worktree 越界，不读取无关密钥、私人数据或其他用户项目，不进行全局 tmux、进程、GPU、端口或锁枚举。工具不能保证只读时不得直接执行。
- 任何服务器状态变更须事前说明具体操作、完整路径或资源、目的、风险及冻结证据影响，并取得明确授权。同一已批准且范围清楚的任务不逐条重批普通子命令；新操作、扩围或高风险删除另行批准。Mac 开发按具体任务授权执行，详见第30条。
- 优先复用与原位修复，同次删除或合并被替代且无依赖的普通旧逻辑；以 Git 历史追溯已提交源码，保护未提交工作。最小实现不默认新建 Runner、CLI、兼容包装、环境、Worktree 或平行源码；科研必需的独立方法及安全机制继续保留。
- 新文件、目录及实验标识由用户命名或遵循已批准规则；名称简短、稳定、语义明确，不擅加序号、版本、日期、SHA、随机或临时后缀。正式协议自动生成及历史冻结名称按原协议执行。
- 可读性优先于代码简短；新增或实质修改的注释、Docstring 原则上用中文。保护既有功能、锁、进程归属及恢复语义，可能改变行为时先说明风险并由用户决定。
- 冻结 Artifact、数据、模型、Manifest、Receipt、Seal、Source Identity、Deployment、Qualification、Admission 及正式结果保持不可变，保护物理绑定，不改写封存路径或 Digest；保护 PRE 信息边界，模型迁移不自动继承资格。
- 仅做与当前风险相关的必要验证，复用仍有效的证据；新增门禁须说明具体风险、现有机制不足和成本。区分代码、推理、真实运行链、科研结果与最终指标改善，未知推理结果不自动重发，不以检查数量代替科研进展。
- Git 提交、推送、合并和历史整理按当前任务授权；保护其他未提交成果，远端更新或历史重写须获具体授权。

## Codex 对话命名

使用 `字母+两位数字-研究方向｜具体任务`：每个方向使用稳定 A–Z 字母，沿用已有映射，未分配时选择最靠前未用字母；同方向新对话按历史最大两位编号递增，不回填缺号。保留历史标题、编号及映射。完整规则见 [第33条](docs/development-guidelines.md#codex-对话命名规范)。

研究任务首次明确时仅命名一次；后续发言、修复、测试及阶段变化保持标题和编号稳定，只有用户明确要求才重命名。无法直接设置时只给一次建议；历史不可完整访问时不猜测，可标注“待确认”，必要时简短确认，不阻碍独立科研任务，不反复命名或检查。

**CodeGraph 使用边界：** 优先用于源码结构导航，不能代替或限制真实源码读取、Git 状态和科研证据核验。工具不可用、未初始化或索引滞后时，可直接使用现有搜索和读取工具继续，不将初始化或索引更新作为普通开发门禁。下方标记区块保留工具原文，项目补充写在区块外；避免重复检索的建议不得阻止必要核验，其他实质冲突仍由用户决定。

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
