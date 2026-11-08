# TomBaseline 本地主线收敛记录

本次替换重型执行协议，科研 Task/Prompt/标签/计分保持原定义。操作方式见
[统一操作说明](../operations/tom-llm-unified.md)。

## 执行依赖变化

旧：uns → operations/baseline_cli → experiment → Suite/Budget/Deployment →
Round0/1 → Physical Admission/Permit → qualification/live 两种服务 →
多层 transport 与多种 Mock/live Runner → 逐请求 Artifact → Run Seal → analysis。

新：uns → cli/operations → experiment 一次准备 →
service_manager 一个拥有的 vLLM 实例 → live_runner 一个执行函数 →
单一持久日志 → 单一预测 Seal → baseline_analysis 三 Run 比较。

Mock 只注入传输和显式合成 token counter，使用同一执行、日志和结束封存代码，
保留 mode=mock 且不能发布 formal_result=true。

## 删除及保留边界

删除 baseline_cli、execution、inference、transport、server_fixed、
gpu_probe、live_service、candidate_client、qualification_inputs、qualification_evidence、
physical_admission、live_admission、development_source。
删除四个 scripts/tom_multimodel 相关旧入口、Qwen 资格资产、旧实验／资格／Wire 配置，
以及仅用于这些旧流程的测试。未保留兼容包装器或新 fast Runner。

multimodel 只保留共享 Cohort 与一次性 Source；
study 只保留公开 PRE 构造和验证，供冻结 Task 校验消费；
model_spec 保留注册表、原生模板和 Wire；
physical 改为版本、分片、stat 与小文件检查；
deployment 只构造固定启动 profile；service_manager 统一拥有进程；
experiment 只准备和读取新的单模型合同；operations 只编排统一 CLI；
live_runner 执行一次连续预测；baseline_analysis 只消费三个封存结果。
multimodel_evaluation 保留科学输入核对和补充概率统计，不执行模型。

task、prompt、policy、reference 与模型原生模板不改；
output 仅删除未批准且不再消费的 ENGINEERING_ONLY >= 阈值接口，其余解析和固定投影不改。
旧概率解析/平滑保留用于补充概率评价，旧公开请求表示保留用于冻结 Task 构造校验；
两者都不是第二条正式执行路径。通用 Artifact、科学数据和其他研究模块不重构。
全局 werewolf/cli.py 只删除被替代的 check-llm-reasoner-config 分支。

## 版本与历史

新执行身份为 classic7_suspicion_set_execution_v2；
科研定义仍引用 classic7_suspicion_set_research_protocol_v2，
科研历史摘要仅作为来源记录，不能当成新执行协议摘要。
新执行合同及预测／分析采用新 Artifact 类型；旧 bacc351 Qualification、Admission、Run、Seal
不修改、不重签、不自动升级，历史 Git Source 继续解释旧证据。

取消两轮物理资格、合成认证请求、重复加载、Admission、Permit 和逐请求 Artifact。
保留完整有序输入、固定解码、官方 Tokenizer／全量容量、owned 服务、未知结果停止和结束覆盖。
本次性能证据是 CPU/Mock 的调用次数／文件数量断言，不将其伪称真实 GPU 加速比。

## 本轮验收结果

运行包从 36 个 Python 模块／5,765 行减至 23 个／2,575 行，减少 55.3%，未新增源码模块。
最终综合 CPU/Mock 回归 765 passed、0 failed、0 skipped，耗时 121.07 秒。
覆盖 TomBaseline、Artifact I/O、公共 PRE、Publication、公共 CLI、集合／概率统计以及 DTRS 相关接口。
另补充 HTTP 429／500 两项本机传输测试，验证保存具体状态、未知结果停止且不重发。
这些结果不代表真实 Tokenizer Parity、GPU 模型加载或正式推理已经完成。

可复核的成本断言包括：每个模型整段请求只调用一次服务启动及停止；N 个请求只发布合同和预测 Seal，
不发布逐请求 Artifact；请求循环不调用模型核验、Cohort 或 Source 扫描；
status／stop 不哈希权重；1 GiB 稀疏权重夹具禁止读取权重内容；三 Run 分析只打开一次共享 Cohort。
全量 CPU Parity 的文本、token IDs 和 Wire 摘要采用 canonical 数组的流式同值计算，
不将全部 token IDs 常驻内存。

直接服务入口比对依据为
[vLLM 0.27.0 api_server.__main__ 与 run_server](https://github.com/vllm-project/vllm/blob/v0.27.0/vllm/entrypoints/openai/api_server.py#L688)；
本轮没有在该环境中实际启动引擎。实际 GPU、CUDA 缓存和停止耗时须由用户部署后核验。
Source 当前仍未提交；正式运行要求服务器使用用户随后批准提交的干净库存。
已核对本地 HEAD、Git index、DTRS 三文件和两份规范与开始时相同。

## 实际变更清单（相对 bacc351）

以下是本轮源码清单，不包含三项原有 DTRS 修改及两份未跟踪规范。

实际删除文件：

- `configs/paper/tom-llm-experiment-v1.json`
- `configs/paper/tom-llm-experiment.example.json`
- `configs/paper/tom-llm-qualification-recipe-v1.json`
- `configs/paper/tom-llm-wire-server-fixed-v2.json`
- `scripts/analyze_tom_multimodel_baseline.py`
- `scripts/check_tom_multimodel_cpu.py`
- `scripts/run_tom_multimodel_baseline.py`
- `scripts/tom_multimodel_admission.py`
- `tests/tom_llm/test_mainline_source.py`
- `tests/tom_llm/test_server_fixed.py`
- `tests/tom_llm/test_transport.py`
- `werewolf/tom_llm/baseline_cli.py`
- `werewolf/tom_llm/candidate_client.py`
- `werewolf/tom_llm/development_source.py`
- `werewolf/tom_llm/execution.py`
- `werewolf/tom_llm/gpu_probe.py`
- `werewolf/tom_llm/inference.py`
- `werewolf/tom_llm/live_admission.py`
- `werewolf/tom_llm/live_service.py`
- `werewolf/tom_llm/physical_admission.py`
- `werewolf/tom_llm/qualification_assets/qwen3-suite-v1.json`
- `werewolf/tom_llm/qualification_evidence.py`
- `werewolf/tom_llm/qualification_inputs.py`
- `werewolf/tom_llm/server_fixed.py`
- `werewolf/tom_llm/transport.py`

实际合并／修改的现有文件：

- `README.md`
- `configs/paper/tom-llm-operations.example.yaml`
- `docs/operations/tom-llm-unified.md`
- `docs/research/tom-multimodel-local-implementation.md`
- `tests/tom/test_cli.py`
- `tests/tom_llm/conftest.py`
- `tests/tom_llm/test_baseline_system.py`
- `tests/tom_llm/test_multimodel.py`
- `tests/tom_llm/test_operations.py`
- `tests/tom_llm/test_physical_tools.py`
- `tests/tom_llm/test_prompt_output.py`
- `tests/tom_llm/test_reference_evaluation.py`
- `tests/tom_llm/test_selection_inference.py`
- `werewolf/cli.py`
- `werewolf/tom_llm/baseline_analysis.py`
- `werewolf/tom_llm/cli.py`
- `werewolf/tom_llm/deployment.py`
- `werewolf/tom_llm/evaluation.py`
- `werewolf/tom_llm/experiment.py`
- `werewolf/tom_llm/live_runner.py`
- `werewolf/tom_llm/model_spec.py`
- `werewolf/tom_llm/multimodel.py`
- `werewolf/tom_llm/multimodel_evaluation.py`
- `werewolf/tom_llm/operations.py`
- `werewolf/tom_llm/output.py`
- `werewolf/tom_llm/physical.py`
- `werewolf/tom_llm/provider_semantics.py`
- `werewolf/tom_llm/service_manager.py`
- `werewolf/tom_llm/study.py`

旧符号位置清单如下。列出的是不再存在的旧类／函数／方法位置；被合并的能力在新主线内实现，不是保留兼容别名。例如 publish_mock_cohort 合并为输入构造的 publish_cohort，Service 的目录属性改为直接上下文值。

| 原文件 | 删除或被合并的旧符号 |
|---|---|
| baseline_analysis.py | `analyze`, `performance`, `publish_analysis`, `sealed_inputs` |
| baseline_cli.py | `execute`, `keyed`, `main`, `prepare` |
| candidate_client.py | `SampledGpuMemory`, `SampledGpuMemory.__enter__`, `SampledGpuMemory.__exit__`, `SampledGpuMemory.__init__`, `SampledGpuMemory._sample`, `SyntheticCandidateClient`, `SyntheticCandidateClient.__init__`, `SyntheticCandidateClient.check`, `SyntheticCandidateClient.close`, `SyntheticCandidateClient.dispatch`, `collect_physical_round`, `collect_round`, `create_candidate`, `model_stats`, `normalize_reply` |
| cli.py | `check_config`, `register_operations` |
| deployment.py | `_runner_inference_body`, `capture_instance`, `driver_version`, `gpu_snapshot`, `open_deployment`, `prepare_deployment`, `publish_bundle`, `qualification_compatibility`, `recipe`, `validate_cache_paths`, `validate_process_environment`, `validate_source_evidence`, `verify_cpu_binding` |
| development_source.py | `development_source_identity` |
| evaluation.py | `analyze_mock` |
| execution.py | `ServerFixedExecution`, `ServerFixedExecution.identity`, `open_study_v2`, `prepare_mock_study_v2`, `validate_execution` |
| experiment.py | `choose_games`, `live_set_binding`, `source_v3` |
| gpu_probe.py | `JsonlSink`, `JsonlSink.__call__`, `JsonlSink.__init__`, `JsonlSink.close`, `install_probe`, `lifetime`, `main`, `snapshot_sampling`, `start_deployment`, `validate_launch_args` |
| inference.py | `InfrastructureFailure`, `MockBackend`, `MockBackend.__init__`, `MockBackend.chat`, `Reply`, `TransportFailure`, `_attempt`, `_check_execution`, `_execution_fields`, `_open`, `_terminal`, `_version`, `_wire`, `open_mock_predictions`, `run_mock_study`, `seal_mock_study`, `verify_result` |
| live_admission.py | `_cpu_child`, `capture_bound_environment`, `capture_live_instance`, `open_permit`, `publish_permit`, `qualification_runtime`, `qualified_offline`, `service_profile` |
| live_runner.py | `HttpTransport.guard`, `HttpTransport.proof`, `LiveTransport`, `LiveTransport.__init__`, `LiveTransport.guard`, `LiveTransport.proof`, `collect_rows`, `event`, `finalize_received`, `inspect_request`, `make_row`, `owned`, `pending_tasks`, `seal_model`, `session_artifacts`, `utc`, `validate_reply` |
| live_service.py | `_eos`, `install_live_probe`, `main` |
| model_spec.py | `token_preflight` |
| multimodel.py | `ModelExecution`, `ModelExecution.binding`, `event`, `inspect_pending`, `open_model_mock`, `open_model_predictions`, `owned`, `prepare_model_mock`, `result_row`, `run_model_mock`, `seal_model_mock`, `verify_result` |
| multimodel_evaluation.py | `compare_models`, `publish_mock_cohort` |
| operations.py | `_save_refs`, `anchor`, `check_environment_binding`, `check_execution_task`, `gpu_status`, `prepare_inputs`, `prepare_model`, `ref`, `save_configuration`, `service` |
| output.py | `probability_to_set` |
| physical.py | `asset_spec`, `capture_environment`, `file_identity`, `publish_receipt`, `read_receipt`, `regular_root` |
| physical_admission.py | `open_admission`, `open_round`, `publish_admission` |
| provider_semantics.py | `classify_provider_error`, `require_live_admission` |
| qualification_evidence.py | `validate_memory`, `validate_qualification`, `validate_round`, `validate_sampling` |
| qualification_inputs.py | `budget_matrix`, `budget_record`, `build_suite`, `export_cpu_cohort`, `legacy_suite`, `open_budget`, `open_suite`, `publish_budget`, `publish_suite`, `synthetic_tasks`, `validate_suite` |
| server_fixed.py | `InjectedServerFixedTransport`, `InjectedServerFixedTransport.__init__`, `InjectedServerFixedTransport.chat`, `InjectedServerFixedTransport.chat_public`, `ServerFixedService`, `ServerFixedService.identity`, `open_mock_service`, `validated_wire_surface_v2`, `wire_binding`, `wire_request_v2` |
| service_manager.py | `Service.binding`, `Service.capture`, `Service.directory`, `Service.logs`, `Service.pending_launch`, `owned_service` |
| study.py | `context_preflight`, `open_study`, `prepare_mock_study`, `source_identity`, `wire_request` |
| transport.py | `InjectedChatTransport`, `InjectedChatTransport.__init__`, `InjectedChatTransport._check_client`, `InjectedChatTransport.chat` |
