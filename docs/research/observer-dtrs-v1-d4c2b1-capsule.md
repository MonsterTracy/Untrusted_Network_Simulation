# D4c-2b-1：Final ToM 只读 Inference Capsule

日期：2026-10-18。状态：本地人工验收；未冻结、未提交、未部署、未授权 CAL/TEST。
开发基点 `7b51f276b04b386ec1aa4ae44454c1d809a10897`。

本阶段新增独立编译器与加载器，不改变 f199、D0–D3、D4b/D4c-1 或 CAL/TEST 草案。
用户报告真实 Metadata / Chain-files 审计通过、32 个 recovery 字节与 codec 验证通过，
并已独立核对审计文件 SHA。这里没有读取该真实证据包或权重，也没有主动访问服务器。
这些审计不能替代原模型完整语义恢复、数值等价或真实部署 QUAL。

## 接口与边界

| 新文件 | 职责 |
| --- | --- |
| `final_tom_capsule_build.py` | 受控离线编译；验证带外审计锚、重开原来源、白名单导出 |
| `final_tom_capsule.py` | 独立只读 reader、公共 PRE 编码、概率输出、加载器/运行时身份 |
| `final_tom_graph.py` | 原 Qwen3 图的推理投影；不导入训练 dataset/publication |
| `scripts/build_final_tom_capsule.py` | 固定真实来源的编译 CLI；没有改 pin 或 fixture 放行参数 |
| `tests/observer_dtrs/test_final_tom_capsule.py` | 完整人工权重、来源 fixture、损坏与隔离测试 |

模块路径位于 `werewolf/observer_dtrs/`。人工 fixture 通过 Python 私有 `_pins` 明确标记
`CONTROLLED_FIXTURE`；公共 CLI 拒绝其审计报告，不能把人工来源冒称真实 final。
正式来源型包也仅标记 `PINNED_FINAL_INFERENCE_UNQUALIFIED`，不是实验执行许可。

```python
artifact = compile_capsule(original_fit, new_external_directory,
    audit_path=accepted_chain_files_evidence,
    expected_audit_raw_sha256=independently_saved_raw_sha)

predictor = CapsulePredictor(capsule_path,
    expected_manifest_digest=externally_pinned_capsule_digest,
    expected_scope="PINNED_FINAL_INFERENCE_UNQUALIFIED", device="cpu")
prediction = predictor.predict(frozen_public_pre)
# prediction: game_id, boundary_id, prefix_digest, plan_digest, token_count,
#             probabilities (immutable 7×7 tuple), identity
```

编译器需要 CPU 构造模型来检查严格恢复，**不做 forward、优化器创建或训练**。
加载器默认 CPU/FP32/eager；显式 CUDA 是后续 QUAL 候选，不是已验证替代原 fit.device。
API 不接受 observation 字典、环境、agent、角色、报告、Y、训练支持或任意张量化输入。
只允许验证过的 `AuthoritativePREPrefix`；其中原报告身份链接不含报告内容，也不进入 token 编码。

## Capsule 格式

复用 `publish_artifact` / `verify_artifact` 的 canonical JSON、SHA、完整 inventory、
原子不可覆盖发布和 fsync。类型 `observer_dtrs_final_tom_inference_capsule`，schema 同名 `_v1`。
manifest 白名单为 artifact/schema/scope/inference_digest/origin/compiler/file_table/manifest_digest。

载荷**恰好六个文件**：

```text
manifest.json
model.bin
inference.json
temporal/phase_codebook.manifest.json
temporal/phase_codebook.bin
temporal/canonical_day_code_table.manifest.json
temporal/canonical_day_code_table.bin
```

`model.bin` 保留原末端文件字节，不经 pickle、torch.load 或重新编码；inference 中保留
原 Tensor Codec model descriptor，按名称/shape/FP32/trainable/有限性与完整图严格恢复。
Temporal 四文件保留原字节和原 protocol parent；不能把 parent 改成新 capsule digest、
按新环境重算 sin/cos、缩小天数或重建 phase codebook。

inference 白名单：graph、public_contract、model_container、temporal_digest、
temporal_parent_protocol、loader_source。只包含图、公共编码、容量、张量描述符和源码摘要。
origin 仅为源码/fit/protocol/study/publication/initial/terminal/seal/ancestry/权重/时间表/
审计证据摘要及两项 NOT_RUN 状态。没有局 ID 清单、Primary、角色、嫌疑集合、训练标签、
优化器、RNG、训练损失、schedule、支持频数或原训练行。模型权重仍应按部署文件权限管理，
不能把“不含监督载荷”等同于证明模型不存在任何训练信息记忆。

## 来源绑定

1. 调用者带外提供 **审计文件原始 SHA**，编译器验证精确文件字节、canonical JSON、
   内部 evidence_digest、chain-files 模式、全部必需 PASS 与正式/人工 scope。
2. 正式来源 pins 沿用 D4c-2a：f199、fit、terminal、seal、model descriptor，不能 CLI 改写。
3. 不调用 `_fit`（会 hash Primary）、`open_fit`、`verify_seal`。重用审计工具
   `_envelope`、`_chain(chain_files=False)`、`_temporal`；其历史 32 节点 ancestry/terminal/seal
   绑定逻辑不另写一套。只打开 recovery 的 model.bin 与 Temporal 载荷；其余只读 manifest。
   因此编译不是重新验证全部 recovery 文件，已有完整文件审计由外部锚证明其历史身份。
   正式编译还核对报告中的9个历史 reader/config SHA 与当前复用文件字节；人工来源明确跳过这一历史来源证明。
4. 原 fit raw manifest、protocol、研究 parent、source/runtime、last recovery、ancestry、
   raw model SHA 与报告交叉核对；Temporal 摘要、原字节、shape 和 parent 逐项一致。
   编译结束重开来源 manifests；复制的时间字节也核对原 file table。不是文件系统快照，
   正式编译仍要求独占只读稳定来源，不能承诺防御恶意并发目录替换。
5. 对完整解码权重进行新图严格恢复；缺失/额外张量、非有限值、shape/dtype/flag 不符即拒绝。
6. 输出要求已存在的规范父目录、全新目标；拒绝 `..`、symlink 别名、已有目标、原 fit/runs、
   本源码及任何带 `.git`/artifact manifest 标记的祖先。不输出失败证据到原来源，失败只返回错误。

必须区分的身份：

| 身份 | 含义 |
| --- | --- |
| 原 source_revision / implementation_digest | f199 训练来源；不等于本加载器源码 |
| 原 saved_runtime_digest | 原 fit 保存的完整 runtime 的摘要；不冒称当前 CPU 环境相等 |
| model_container_descriptor_sha256 | canonical JSON(model descriptor) 的摘要 |
| model_bin_raw_sha256 | 原始 model.bin 文件字节摘要；不能用上一项代替 |
| manifest_digest | 本次 inference capsule 的完整来源/载荷版本身份 |
| loader_source | 新加载器及复用公共依赖的逐文件 SHA/实现摘要 |
| checkout_revision / all_loader_bytes_match_head | 当前 HEAD 观察值和所列源码的 Git blob 对照；当前新代码未提交，后者为 false |
| actual runtime | 当前 Python/NumPy/Torch/Transformers、设备、线程、确定性设置等观测值 |

loader_source 与编译时 spec 必须逐字节匹配。历史 OID 不能替代未提交的新文件身份。
此阶段 `formal_execution_guard=NOT_ESTABLISHED`、`qualification=NOT_ESTABLISHED`；
没有调用或绕过任何既有 D2/D3 guard，也没有修改原 predictor 的 `_source_gate`。
正式服务前仍需源码固定、部署/运行 contract 与新 guard、限定运行时及 QUAL；开发 reader 本身不授权正式运行。

## 数值语义与传递导入

原 `backbone_model` → `dataset` → `development_publication`，不作为在线 import。
新 `InferenceGraph` 保留相同参数名称、4 层/256 宽 Qwen3 栈、observer shell、原 forward；
AST 测试核对 forward 完全相同。仅构造入口固定 qwen3、公共容量及局部占位初始化 seed=0，
随后**所有**参数严格恢复，初始化值不会参与预测，CPU RNG 在构造前后不变。
无 vocabulary / language head，无 vLLM、远端服务或权重下载调用。

公共编码是原 `tensorize_public_pre` 的最小推理投影，仍调用同一 canonical validator、
结构 token planner 和 `validate_final_pre`；不能导入原 dataset 来取得该函数。
对完整前缀右填充至 1024，不截断、不重排；天容量 256。Temporal 用原 provider，
tensor 恢复用原 `state.restore_model_tensors` 和 Tensor Codec。
`state` 中其他 lazy validation helpers 没有被调用，因而不导入其 dataset/model。

加载/推理的项目 import 闭包为 22 个文件，全部记录在 loader_source 并通过新进程检查。
canonical_collection 的包 initializer 会导入公共/报告**schema 类型**；这不是训练 publication
或受限数据读取能力，未调用采集/报告/写入函数。没有声称“任何报告类也没有导入”。
新进程阻止 publication、dataset、旧模型/训练生命周期、audit/compiler、Shadow、Phase2、env/agent
导入；它仍完成加载与公共 PRE 推理，并拒绝加载模块路径与声明源码不符。另有文件打开 guard 拒绝 Primary/roles/schedule/
optimizer/RNG/loss，及完全拒绝原 lifecycle 访问的独立 loader 测试。

原图输出 log-Q，新预测 API 只做一次 log-Q→Q 指数转换。七行七列为绝对 player1…player7，
自列严格零，非自质量约和为一；死亡目标质量保留，不对存活列归一化。
任何无效 PRE、超容量、损坏/不匹配来源或坏 Q 均异常退出，没有均匀 Q、BND 或替代模型回退。

## 与 Shadow 的下一阶段接缝

当前 `TomIdentity` 只有 family/revision/fit/seal/terminal/weights/fold，无法完整表达
capsule + 新 loader + 原权重 raw SHA + 实际 runtime；`ShadowEngine` 又严格要求匹配的
`TomIdentity`/`FrozenEventEstimator`，当前 worker 仍限人工来源。本阶段**没有放宽**这些检查。
不能把本 prediction 强转成 `final_tom` 并只填 f199，以此隐藏新源码/部署来源。

下一步服务返回应含请求/PRE/plan 身份、Q、capsule digest、新 loader/runtime 身份。
版本化 provider/概率部署 schema 应引用同一完整 capsule 身份；经单独审阅后才把该 Q
包装为 `ProbabilityMatrix`，将完整 identity 连同 `QPrediction` 显式传给扩展后的 Shadow。
现有 D2 OOF TomIdentity/概率包不会匹配，也不允许选择 D2 outer 模型代替 final CAL。
本轮不创建相容性“便捷”适配，不接入游戏 Hook，不变更 45 成本或 TiePolicy。

驻留服务由 D4c-2b-2 独立实现：单 owner、进程寿命、ready/query/cancel/close、限字节/并发/
有界队列/硬期限、迟到响应、终止后不占用 GPU 的证明、日志封存。杀本地 client 不是远端取消证明。
此同步 CPU 核心不承诺这些生命周期语义；后续人工服务验收可以使用当前 Capsule fixture。

## 验收与尚缺证据

本地测试结果和实际指纹见 `observer-dtrs-v1-d4c2b1-acceptance.json`，不混用之前审计的 90/150 项结果。
人工模型 3,430,919 FP32 参数；无训练。普通/PK × 全存活/死亡目标、绝对 player5 speaker，
逐张量/编码/Temporal 相同，原/新 log-Q 与 Q **CPU 逐位相等**；另验 1024 token 容量边界。
反例涵盖外部 SHA、失败/metadata 报告、来源、损坏摘要、重算摘要后的缺张量/shape/dtype/
非有限值/flag、短时间表、extra files、别名/覆盖、无效 PRE 和坏输出。

后续用户手动、另行批准执行的编译模板（本轮没有执行）：

```bash
PYTHONDONTWRITEBYTECODE=1 /data/yuxiao/envs/untrusted-network-simulation/bin/python -B \
  scripts/build_final_tom_capsule.py \
  --fit /ABSOLUTE/ORIGINAL/FINAL_FIT \
  --audit /ABSOLUTE/EXTERNAL/chain-files-evidence.json \
  --audit-sha256 EXTERNALLY_SAVED_RAW_AUDIT_SHA256 \
  --output /data/yuxiao/Untrusted_Network_Simulation/observer-dtrs/tom-capsules/NEW_VERSION
```

输出父目录事先存在且不在原 fit/runs/checkout 内；真实路径和真实审计 SHA 由用户填写。
成功返回0、manifest digest 与 `PUBLISHED_NOT_QUALIFIED`；失败返回2，禁止覆盖/回退。
编译需要可读原来源；之后 reader 只需 capsule，不再需要 DEV 或 recovery 权限。
不在本轮提供自动启动正式新游戏的命令。

仍需：真实审计包/权重编译、外部固定 capsule digest、原 sealed predictor 与新 loader 的真实 PRE
逐张量/Temporal/log-Q/Q 对照、完整语义恢复的独立状态、源码固定、CPU/GPU runtime 资格和服务
资源/取消证明。原始32节点字节/codec审计不能替代 optimizer/RNG 恢复与 loss/schedule 语义重验。
固定历史 ToM/DEV 与历史 RNG 来源限制保留；CAL/TEST 仍是 PROPOSED、不能执行。

**可开始 D4c-2b-2 的人工驻留服务与有界客户端开发；尚不具备真实 ToM 上线或 CAL/TEST 冻结条件。**
