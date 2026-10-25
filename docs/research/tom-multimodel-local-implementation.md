# 多模型 ToM Baseline：第一阶段本地实现

本阶段是 CPU/Mock 工程框架，不授予物理模型 Admission、GPU Qualification 或正式 Execution。审阅基点为 `twd/mainline@9e0a4b523b7dd0cace9111d743f0e264bd48a498`，开发分支为 `codex/tom-multimodel-baseline`。不继承 W5 Candidate 或历史服务资格。

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
