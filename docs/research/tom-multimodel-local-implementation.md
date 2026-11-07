# 多模型 ToM Baseline：实现与研究协议

## 已批准的怀疑集合 V2 协议与本地 CPU 实现

本节记录任务 B 本轮的受控实现。用户已批准科研方法和本地 CPU/Mock 实现，未批准 Git 提交、服务器写入、物理 Qualification 或正式 GPU 实验。下文第一阶段记录保留为历史快照；本节不重新评分历史三个 Direct-Probability Run。历史实验继续以各自 Contract/Seal 记录的 Source 为准，不能由当前开发 HEAD 推定。

### 协议身份与数值合同

配置同时选择 `task_version: classic7_llm_suspicion_set_task_v2` 和 `protocol_version: classic7_suspicion_set_research_protocol_v2`。已批准科研协议摘要为 `e135168b1927c1542d7e0c991aaeb5fe9dc8814317b9316071dd1ffc655b71dc`，执行状态仍为 `RESEARCH_APPROVED_CPU_ONLY_FORMAL_EXECUTION_BLOCKED`。该摘要描述方法合同，既不是未来 Git Source，也不是 Run、Prediction Seal、Run Seal 或 Analysis 的身份。

严格均匀参照投影规则为 `classic7_uniform_reference_strict_projection_v1`，摘要为 `d25ce2e3f22dfe99dad3693f1dda5c5a00cc832bc242a91f53ca697becd255f0`。读取器先验证原始 OOF JSONL 等于既有 canonical 序列化、七席概率合法、自身为零，再将同一 canonical 十进制数值解释为精确有理数，执行

\[
\widehat S_o=\{j\ne o:6p_j>\sum_{k\ne o}p_k\}.
\]

等号排除；按 canonical 玩家顺序输出。六席相同值产生空集，正总质量使全六席不可能同时入选。不使用二进制浮点边界修补、Gold、真实集合大小或开发阈值，不改变输入概率及旧 log-probability 评分。它是均匀参照确定性投影，不是已证明的 Set-F1 最优解。

固定 OOF 合同摘要为 `38407ffe97f5c55c59a5340ca8572a20c540d026f98068ea8c496f1edd33185f`，在 `reference.fixed_oof_contract()` 中绑定 qwen3、explicit_day_phase、历史 Reference Seal、Publication、Fold、Bootstrap、五折 Prediction 与 Checkpoint 以及训练/评价 Source。正式来源必须逐项匹配；人工测试 Reference 只登记 `ENGINEERING_ONLY_NOT_FORMAL_OOF`，不能提升为真实 OOF 核验。旧 `ENGINEERING_ONLY >=` 阈值接口只供旧 Mock 兼容，新协议拒绝通过配置覆盖固定规则。

未提供新 `protocol_version` 的 V1 配置与历史早期 V2 Mock 继续验证原协议字典及 Digest，不自动升级。模型注册表、原概率 Prompt 资产、概率评分、平滑、请求状态机、Runner、Bootstrap 和 Artifact 类型均复用原实现。

### 任务、Reference 与评价

LLM 根据指定观察者的合法公共 PRE 预测该观察者原始报告的怀疑集合，不以真实狼人身份为预测目标。输出 `suspected_players` 允许空集、全部六个非自身玩家以及其他非空真子集；死亡目标仍可入选。输入构造不读取原集合标签、私人角色或当前及未来发言。

Gold 在评价阶段由同一封存 Publication 的原始 `BeliefObservation` 恢复，逐行核对 Game、PRE、observer、observation digest、Fold、Reference 和 q。成功空集、成功全六席、真子集与失败/不可恢复状态保持独立。禁止由均匀 q 逆推空集或全六席。

Full Primary 的原始集合预测是主要任务，Full Set-F1 为主要描述性终点。NU Comparable 只包含原集合与 q 正支持一致的非空真子集，专用 Observer-ToM 只在该人口参加集合对照。旧冻结材料已核验的计数是 Full 7,480 行/150 局、NU 5,502 行/150 局、空集 1,895 行、全六席 83 行/33 局、公共请求 10,714 条；这些是既有只读核验来源，本轮未重新读取服务器或计算正式分数。代码从实际 Artifact 计算人口和身份，不用这些计数构造人工“正式”证据。

集合指标沿用既有约定：双方空集时 Precision、Recall、Set-F1、Jaccard、Exact Match 均为 1；仅一方为空时均为 0；普通非空集合按交并和基数计算。非法响应保留 `None` 及失败类型，全部端到端集合指标、Valid Output Rate 记 0，仍在统一分母内。缺失、重复、外来或身份错配的预测使完整分析失败，禁止取成功响应交集。不可观测 Gold 单列覆盖，不伪造空集。

Always Empty 与 Always All-Six 从同一公共行身份生成合法确定性预测，共用原集合评分器。两者参加 Full、NU 及诊断子组，Valid Output Rate 为 1；派生方法只登记父行身份和预测摘要，明确 `model_run_seal: null`，不伪造模型 Run、请求成本、GPU 或资格证据。

合法集合的补充分布在非空集合内等分、集合外为零，空集转六席均匀，再复用原 `(1-epsilon)*p+epsilon/6`、`epsilon=1e-6`。非法输出的补充分布仍沿用原失败均匀策略。旧 KL/TV/Top-1_NU 数学和评分路径保持不变；新集合表示不保证 KL 下降。

### 统计单位、配对与比较层级

所有集合指标先局内平均，再游戏等权。沿用原 10,000 次游戏级配对 Bootstrap、原随机种子、95% percentile-linear 点态区间，差值统一为左减右。Full/NU 使用各自有样本的排序游戏列表；全六席诊断子组只对其有样本的游戏重采样，未命中的游戏登记 `zero_sample_games`，不补零分数。相同人口的所有模型和差值共用同一索引摘要。

Full 三个 LLM 的主要描述性结果外，五种方法共有 10 对：LLM 与两个简单基线的 6 对为简单基线对照，LLM 间 3 对及两个简单基线间 1 对为探索性。NU 加入 Observer-ToM 后为 15 对：三个 LLM 对 Observer-ToM 的 3 对为关键方法对照，四种学习方法对两个简单基线的 8 对为简单基线对照，其余 4 对为探索性。其他子组均为诊断。未实施多重比较校正，各区间只用于描述与探索，不支持未经控制的联合优越性声明。

公共 Valid Output Rate 是全部公开请求上的行级有效比例；Full/NU 中的 Valid Output Rate 与其他指标一样按相应 Gold 人口游戏宏平均。两种分母和统计单位在结果中分别命名，不能相互代替。

### 三个独立 Run 的分析消费

现有 `report/analyze` 增加 `--sealed-run model=path` 和 `--sealed-run-digest model=sha256`，要求恰好 qwen3、yi、mistral 三个独立单模型 Run，显式新输出目录且不设置聚合 Run ID。消费者复用旧 Contract、Run Seal 和 Prediction Seal 读取链，逐项核验相同 Source、执行模式、科研协议、Cohort、Selection、Reference、游戏、完整任务序列与行身份。缺失外部 Seal 锚点、协议混用、Source 不同、重复 Run 或不完整覆盖均拒绝。

只发布原类型 `mm_baseline_analysis`，其中登记三个真实父 Run Seal、逐模型配置和 Prediction Seal。内部分析上下文不发布为新 Run/Seal；两个常数集合和 ToM 投影也不产生替代 Seal。本轮仅在临时人工资料上测试消费者，不自动生成服务器正式分析。

### Schema 准备与资格边界

原 `mm_native_suite` 类型增加内部版本 `classic7_multimodel_qualification_inputs_set_v2`，绑定实际集合 Task、Prompt、Schema、原生模板及 token IDs；旧 V1 和冻结 Qwen 76-call 套件不改写。仍沿用 12 个 PRE、每轮 38 次、两轮 76 次的原调度和证据链，V2 的完成响应需通过原集合解析器。CPU Suite/Tokenizer 输入准备只读取显式提供的官方本地小资产，不安装、下载或加载权重。

无真实资产时报告 BLOCKED/NOT_RUN。Mock Tokenizer 和 Mock 引擎只能证明接口和验证链，不能证明真实 CPU Parity、物理 Qualification 或 Admission。新 V2 物理部署准备及 technical/formal Run 继续明确阻断；未扩大 Qualification 兼容排除集合，也不继承旧 V1 物理资格。

### Source 准备、独立性与后续授权

当前 `twd/mainline@98d8e038c86c35b87490c433acb9d97caf886c2d` 是含未提交修改的开发树，不是新增 V2 正式运行的干净 Source。现有 Source inventory 已覆盖任务 A 五份公共运行源码及 Observer-DTRS 的既有运行源码，不能从最终 Source 中忽略、排除或伪装成旧 HEAD。

后续应由用户分别授权任务 A、Observer-DTRS 与任务 B 的归属提交及整合；不移动、撤销或将其他任务文件混入本次 Baseline 提交。在全部拟运行源码经相应授权进入干净工作树后，记录新的 Git Source、完整 inventory/implementation digest；再绑定新 Cohort、Budget、Deployment、Qualification、Permit 和三个独立 Run ID。科研协议摘要保持独立，不代替这些物理/软件身份。本轮不执行上述提交、部署或发布。

不使用 1,350 局拟合阈值。五折 OOF 的训练信息关系不能仅由 Game ID 不重叠认定完全独立；150 局已经被观察，后续评价属于方法比较，不是恢复为未观察测试，也不能仅由 V1/V2 差异推断输出表示的因果效应。

正式前置项仍包括三模型真实 Tokenizer Parity、全量 V2 Token Budget、新的干净 Source 和 Artifact 身份、新 Schema 的真实物理 Qualification/Admission/Permit、用户指定的新 Run ID，以及服务器写入、Git 操作和 GPU 执行的分别授权。本轮在本地完成 CPU/Mock 回归后停止。

2026年11月7日本地实际主回归为 **718 passed、54 skipped、0 failed**，耗时317.02秒，范围是 `tests/tom_llm` 和任务 A 已有的 `tests/artifact_io/test_canonical_artifacts.py`。其中包含三份独立 Run 的真实回环 HTTP Mock、现有 V1/旧 Mock 读取及评分回归、新严格十进制投影和新 Schema 的人工两轮证据。随后新增的固定 OOF 合同逐项身份测试单独运行，实际 **8 passed、0 skipped、0 failed**；该测试显式使用人工身份适配器，不能替代服务器原始 OOF 的再核验。

主回归的54项跳过分别为37项缺少官方 Tokenizer/Suite 小资产、11项可选 B1 Tokenizer 检查和6项缺少指定 vLLM 0.27 argparse 源材料；未报告真实 Tokenizer Parity PASS。命令使用 `CUDA_VISIBLE_DEVICES=''`、离线模式、`python -B`、关闭自动插件及 pytest cache；所有新实验产物限于测试临时目录。


### 本轮 Source 清单与工作区保护证据

本轮开始为36项 tracked 修改，结束为37项：任务 A 的9项、任务 B 的25项及 Observer-DTRS 的3项，另保留原2份未跟踪规范。任务 B 本轮只编辑17个既有文件，其中16个已经有未提交修改，另外1个是既有 `qualification_evidence.py`；未新建项目文件。其余623个起始文件字节不变。任务 A 九文件、Observer-DTRS 三文件与两份规范均逐文件比较起止 SHA-256 相同；没有覆盖、撤销、暂存或移动其他任务工作。

任务 A 九文件的 `path -> SHA256` 映射按既有 canonical digest 的起止摘要均为 `48ce77c2625aa429656598794bb75fa47e0b8d4c1de7ba4509e5c23087aa0be8`。Git index 起止 SHA-256 均为 `a0632bcc228161ca54beea960c5ccd8648f5a6b533b0f8708e16be19ca59c365`。HEAD 与分支保持 `twd/mainline@98d8e038c86c35b87490c433acb9d97caf886c2d`。

当前开发 Source inventory 共213项，implementation digest 为 `ff99f0911d354482bf27ed51ae97160ac61f19108494f0649b41fe03098680a5`，`source_inventory_clean=false`；此值只记录开发工作字节，不是正式 Source、Qualification 或执行许可。全部 `werewolf/` 运行源码继续纳入原 inventory，包括任务 A 五份公共源码及 DTRS 的既有运行源码，未扩大兼容排除集合。

本轮结束时完整 Git 原始状态如下，路径相对于本仓库根目录：

```text
## twd/mainline...origin/twd/mainline
 M README.md
 M configs/paper/tom-llm-experiment.example.json
 M docs/BRANCHES.md
 M docs/repository_structure.md
 M docs/research/observer-dtrs-phase-2g-native-evidence.md
 M docs/research/tom-multimodel-local-implementation.md
 M tests/artifact_io/test_canonical_artifacts.py
 M tests/observer_dtrs/test_native_action_link.py
 M tests/tom_llm/conftest.py
 M tests/tom_llm/test_baseline_system.py
 M tests/tom_llm/test_multimodel.py
 M tests/tom_llm/test_operations.py
 M tests/tom_llm/test_physical_tools.py
 M tests/tom_llm/test_prompt_output.py
 M tests/tom_llm/test_reference_evaluation.py
 M werewolf/artifact_io/canonical.py
 M werewolf/canonical_collection/attempt_ledger.py
 M werewolf/canonical_collection/failure_evidence.py
 M werewolf/canonical_collection/pre.py
 M werewolf/canonical_collection/trajectory_evidence.py
 M werewolf/observer_dtrs/native_action_link.py
 M werewolf/tom_llm/baseline_analysis.py
 M werewolf/tom_llm/baseline_cli.py
 M werewolf/tom_llm/cli.py
 M werewolf/tom_llm/evaluation.py
 M werewolf/tom_llm/experiment.py
 M werewolf/tom_llm/live_runner.py
 M werewolf/tom_llm/model_spec.py
 M werewolf/tom_llm/multimodel.py
 M werewolf/tom_llm/multimodel_evaluation.py
 M werewolf/tom_llm/operations.py
 M werewolf/tom_llm/output.py
 M werewolf/tom_llm/prompt.py
 M werewolf/tom_llm/qualification_evidence.py
 M werewolf/tom_llm/qualification_inputs.py
 M werewolf/tom_llm/reference.py
 M werewolf/tom_llm/task.py
?? AGENTS.md
?? docs/development-guidelines.md
```

## 第一阶段历史记录

以下记录对应原第一阶段，不表示本轮的分支或实时资格。原阶段是 CPU/Mock 工程框架，不授予物理模型 Admission、GPU Qualification 或正式 Execution。审阅基点为 `twd/mainline@9e0a4b523b7dd0cace9111d743f0e264bd48a498`，当时开发分支为 `codex/tom-multimodel-baseline`。不继承 W5 Candidate 或历史服务资格。

## 合同与最小闭环

1. `task.public_tasks(public_prefixes, fold_of_game)` 复用原 public_requests，只产生公开存活观察者请求。共享记录保留完整 StructuredTokenPlan、观察者、科学行键、原中文 messages 和六席 Schema，去除 Qwen 专属 rendered_prompt/prompt_digest。`validate_task` 严格重建原 renderer，拒绝额外字段、消息注入、缺失/不合法 PRE。`public_tasks` 不读取 Reference 或私人 sidecar。
2. Selection 继续由原 `selection`/`selection_binding` 验证，沿用原1500局、每折30局和既定150局名单；本轮没有重新选择真实游戏。`publish_mock_cohort` 将独立公共请求与严格 Reference/Selection 来源绑定，验证全部 public-alive 行；公共请求文件不包含 q、标签或 Primary。Reference 只在这一独立绑定边界和评价中读取。
3. `ModelSpec(key)` 是封闭三模型注册表，不接受任意名称或浮动 revision。注册表物理 SHA 固定在代码中；模板另做物理 SHA 核验。共享科学消息、Schema、输出 parser 不随模型变化。原生角色包装和特殊 token 允许不同。
4. `prepare_model_mock` → `run_model_mock` → `seal_model_mock` → `compare_models` 构成完整合成闭环。每个模型使用独立目录、合成 service_id、contract/execution/cohort/model-spec 摘要。只接受原 MockBackend 的精确类型，service_id 必须以 synthetic: 开头。所有 artifact 使用新类型和版本，旧 S4/W5 响应不能混用。
5. 所有模型完成同一公共队列后才调用原 compare。先验 seal 锚点、任务顺序、科学行键、Selection、Reference 必须相等，禁止缺行、重复、外来行和交集评分。沿用原 Primary、KL/TV/Top-1_NU、平滑、fallback、game-macro 和 bootstrap；新增 ΔTV 与 ΔTop-1_NU，方向均为 LLM−专用 Qwen3-block Observer-ToM。NU 仅使用有 NU 行的游戏，不补零。

Python 最小调用顺序：

```python
from werewolf.tom_llm.task import public_tasks
from werewolf.tom_llm.model_spec import ModelSpec
from werewolf.tom_llm.multimodel import ModelExecution, prepare_model_mock, run_model_mock, seal_model_mock
from werewolf.tom_llm.multimodel_evaluation import publish_mock_cohort, compare_models

# 先用只含公开 PRE 的 handle 独立构造 tasks；Reference 是另一评价边界。
# publish_mock_cohort(..., selection_root=..., expected=..., provenance=..., prepared_manifest=...)
# 每模型单独 prepare_model_mock(..., execution=ModelExecution(ModelSpec(key), 'synthetic:...'),
#     tokenizer_identity='synthetic:...', count_tokens=synthetic_counter)
# run_model_mock(..., MockBackend(...)); seal_model_mock(...)
# compare_models(..., runs={key: (root, execution), ...}, expected_seal_digests=external_anchors)
```

这里没有真实模型 client、live runner 或正式 Admission publisher。`require_live_admission` 明确拒绝 CPU/Mock 资格。不要通过删除类型、来源或版本检查接入真实模型。

## 模型与原生渲染

| key | 官方 repo / 完整 revision | 原生模板和终止 | 本轮资格 |
|---|---|---|---|
| qwen3 | Qwen/Qwen3-8B / b968826d9c46dd6066d109eabc6255188de91218 | 原 S4 Non-thinking 模板、空 think 块；原 Wire v2，逐字段复用 | CPU/Mock；不继承 W5 |
| yi | 01-ai/Yi-1.5-9B-Chat-16K / 2f8753bed95bd4fede3fd64f8305428620b4b212 | 官方 system 前缀 + ChatML；BOS=1，tokenizer EOS=7，generation EOS=2，显式 extra stop=7 | CPU/Mock，权重未核验 |
| mistral | mistralai/Mistral-7B-Instruct-v0.3 / c170c708c41dac9275d15a8fff4eca08d52bab71 | 官方初始 system + user 合并到单轮 [INST]；BOS=1，EOS=2；BOS 只一次 | CPU/Mock，权重未核验 |

Yi/Mistral 的模板逐字取自指定官方 tokenizer_config.json，置于独立 model_assets/，不修改原 assets/。注册表记录 tokenizer/config/generation_config/special_tokens/vocab 等实际所需文件的 SHA 和字节数。新模型 revision 来自官方仓库快照，最终下载 URL 全部使用完整 SHA；没有下载权重。完整权重、index、许可证、环境和实际服务仍需后续独立核验。

三模型执行意图均为 BF16、无量化、8192 总上下文、max_tokens=512、T=0、top_p=1、top_k=-1、min_p=0、n=1、seed=20261014；Qwen 的 enable_thinking=false 仅传给 Qwen。Yi/Mistral 不添加 Qwen think 块，不新增规则、翻译、few-shot 或历史响应。Yi 的双 EOS 来源差异被显式登记，并不是已验证 vLLM 终止行为。

`check_cpu_parity` 仅加载明确提供的本地小资产，local_files_only=True、trust_remote_code=False；先检查必需文件 SHA、大小和目录清单，拒绝未绑定的额外模板等隐式资产。固定模板文本与官方 apply_chat_template 逐字比较，再比较每个 input_id；Qwen 另检查原文本、原 Wire 和最后7个 Non-thinking token。token_preflight 要求 input+512<=8192，禁止截断。未提供资产时 CLI 只报告 NOT_RUN；资产存在但身份错误则报错，不能当作通过或静默降级。

从本源码 checkout 执行（不下载、不启动服务）：

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -B scripts/check_tom_multimodel_cpu.py \
  --requests /absolute/path/public-tasks.jsonl \
  --qwen3-assets /absolute/path/qwen3-small-assets \
  --yi-assets /absolute/path/yi-small-assets \
  --mistral-assets /absolute/path/mistral-small-assets
```

CLI 输出 CPU_PARITY_PASS / NOT_RUN、模型身份、文本与 input_ids 序列摘要和 token 数；physical_admission=false。本阶段新增资源按源码 checkout 部署；原 setup.py 的冻结 package_data 不变，不宣称现有 wheel 已自动包含新注册表/模板。

## 失败、恢复与来源

原 Mock runner 和旧异常语义原样保留。新模块只提供未来真实 adapter 可调用的保守分类 seam，不进行 HTTP 请求：明确429拒绝、连接建立前失败为 NotExecuted；读取超时、写入错误、连接结果不明为 OutcomeUnknown；其他基础设施失败停止，不能生成均匀 fallback。未来真实 factory 必须单独验证 SDK max_retries=0，禁止借用旧 transport 的宽泛重试分类。

新 Mock 执行保留两次 dispatch 上限和一次语义响应。NotExecuted 的确证事件可重试；已 dispatch 而无耐久 reply 的请求禁止自动重发。回复先落盘后生成结果，中断后复用已封存 reply。格式错误/长度截断只经原 parser 产生语义 fallback，不重新问模型；合法均匀预测仍标 valid。运行开始时检查整个已有请求目录，任何较后行的跨模型污染、不连续事件或未知请求都在首次新调用前拒绝。

Source 身份包含实际 HEAD/tree、原开发源码清单、新运行模块、ModelSpec 注册表、原生模板和 CPU CLI 的 SHA。服务、模型、revision、模板、源文件或 cohort 漂移均是新身份，不能原地恢复旧执行。当前均为 engineering_only；后续 commit 后需要生成新工程身份和新产物，不能复用本轮 dirty-source Mock contract。

## 验收范围与限制

测试覆盖共享公共任务/Schema、三模型原生外壳、真实 offline tokenizer、Qwen 兼容、身份/模板漂移、完整 PRE 溢出、因果隔离、合法均匀与 fallback、模型/服务/revision 混用、未知超时禁止重发、耐久回复恢复、公共行全覆盖、相同 Reference/Selection、非零 ΔTV/NU 以及无 NU 游戏的处理。

合成 Reference fixture 与旧测试一致，仅在测试进程缩小为5个游戏、bootstrap20次；生产选择器仍为1500局、每折30，生产bootstrap仍10000次。合成结果不是研究估计。本轮11种 tokenizer case 不是150/1500局真实上下文预算；后者必须在物理 CPU Admission 中按完全相同请求清单逐模型重测。

论文继续披露预训练/指令微调、中文能力、显式规则和家族关系，比较方法系统而非纯架构。新 wrapper 仅对 Qwen 保留 qwen_family_related_to_generator 标签。多模型联合优越性声明或多重比较校正需事先另行登记，不修改既有单对照区间。

## 下一阶段：Mac → GitHub → Ubuntu

1. **Mac**：审阅本轮差异及测试证据后，才由用户另行授权 commit/push/PR。保留独立分支，W3 和 W5 不变。审阅通过前不发布、不覆盖旧 Candidate。
2. **GitHub**：以批准的精确新 commit 作为 Source，保存新 ModelSpec、模板、测试和来源摘要。提交后重新生成摘要，不能声称本轮未提交执行身份仍有效。
3. **Ubuntu（全部由用户手动）**：在独立新目录取得该 commit。原 Qwen 权重/环境仅可作为物理资产复用并重新核验，不能继承 W5 运行资格。Yi/Mistral 完整资产下载和依赖变更需要单独授权；逐文件 SHA/index、license、磁盘、独立缓存和运行环境均需验收。
4. **CPU Admission**：为三个模型分别建立真实 Model/Tokenizer/Deployment/Service artifact；绑定共同既定 Selection、Reference、公共请求清单，重测全部真实请求 token 分布，禁止按模型取不同样本或截断 PRE。核对本地/服务端实际模板、BOS/EOS 和最终 input_ids。
5. **GPU Qualification**：由用户安排独立窗口、手动启动。每模型独立候选及合成预期，核验实际 vLLM/FlashInfer/JIT、BF16/8192、JSON Schema、有效 SamplingParams（含 generation_config 和 Yi EOS）、显存、吞吐、超时留痕及重启复现。历史 W5 76次套件与资格不改变，新模型方案另行审阅；CPU/Mock 不代替此步骤。
6. **Live Execution（仍待实现/授权）**：新增独立真实 factory/runner，loopback服务/版本/进程/源码身份校验，耐久 request ID 和 accepted/unknown 语义，SDK零隐藏重试；与 Mock 使用独立 artifact domain。全部资格通过后才正式每模型150局，封存完整预测后共同配对统计。
7. **效率**：同GPU、预热/并发/cache条件，区分冷加载/JIT、warm墙钟、TTFT、token、显存和失败成本；同时按观察者/PRE全覆盖/游戏报告。专用ToM单次给七观察者，不用其整PRE延迟直接除LLM单请求延迟。正式计时与质量预测seal分离，不覆盖OOF。

本轮没有服务器操作、GPU、真实模型回复、正式评分或 Git 提交。正式150局及其运行资格仍全部未完成。
