# D4c-2a：final ToM 原始封存来源审计准备

本轮状态：**LOCAL_CONTROLLED_ACCEPTANCE_ONLY / SERVER_EVIDENCE_PENDING**。
CAL/TEST 提案仍是 `PROPOSED_NOT_FROZEN_NOT_EXECUTABLE`，没有冻结或启动 CAL/TEST。
保留 CAL 500 / TEST 500 的预算起点（QUAL 后定）、EQ0/EQ1/LR-r、独立拟合的记录等权敏感性，
以及实时覆盖与独立 PRE 重放评分的双轨边界。

本轮新增独立工具，不改 predictor、Shadow、游戏、D0–D3、既有模型或实验产物。
没有服务器连接、真实 ToM 构造/forward、权重加载到模型、优化器恢复、训练或项目 Git 修改。
测试只在临时 Git fixture 中建立人工提交，不提交当前项目。

## 1. 工具入口和证据层级

入口：`scripts/audit_qwen3_final_provenance.py`；实现：
`werewolf/observer_dtrs/final_tom_audit.py`。
它是来源证据采集工具，**不是推理 capsule、正式生产 guard 或部署资格证书**。

| 检查 | `--mode metadata`（默认） | `--mode chain-files` |
| --- | --- | --- |
| final fit 原始 manifest、外部期望摘要、协议、1500 局及31500步 | 核对 | 核对 |
| fit 所列全部文件的大小及字节 SHA | 核对，包括 Primary 的字节哈希；不解析其标签 | 同左 |
| publication/study、初始化来源、sidecar 身份联结 | 只读 manifest 与精确目录封套，不打开其受限文件 | 同左 |
| terminal/seal 与32个 recovery manifest 的绑定、步数、前驱、完整 ancestry | 核对 | 核对 |
| 末端 model.bin、模型容器描述符和逐张量 codec | 核对，分别保存 raw SHA 与 descriptor SHA | 核对 |
| 所有32个恢复点的 model/optimizer/RNG/loss 文件 SHA | **未检查**，仅核对 manifest/file-table 声明间绑定 | 核对全部字节，并复用 tensor codec；不恢复任何状态 |
| 原始训练 schedule 重建、loss/schedule 逐行语义、Primary 独立重建 | **未执行** | **未执行** |
| 原始模型、优化器/RNG 恢复与 recapture 字节精确等价 | **未执行** | **未执行** |
| 原始完整恢复验证器 `verify_terminal/verify_seal` | **未执行** | **未执行** |
| GPU/确定性完整 runtime 重验、现有运行中 predictor 进程身份 | **未验证** | **未验证** |

`chain-files` 是完整恢复链的**文件/绑定审计**，不能简称为“完整恢复重验通过”。
输出永远包含 `full_semantic_recovery=NOT_RUN`、`inference_qualification=NOT_RUN`。
成功退出只表示相应范围的证据核对成功，不授权启动 ToM、CAL、TEST 或部署。

## 2. 外部锚点与原始路径

以下是从现有 predictor client 读出的**预期值**，本地没有对应原始 final 产物：

| 身份 | 固定预期值 |
| --- | --- |
| predictor checkout | `f199a9e6c64168a412de91ff8ca04d0c795c57e2` |
| final fit manifest digest | `87c2485acf48443034c59430d79f098a13151677852988bf691599ca2d2f8248` |
| terminal manifest digest | `b977ec24d6292ad329e664c931a1aae39dc4738b4b33685b19d92fd4fb1921af` |
| seal manifest digest | `fcbb81ee43050142ebdf01c0aee576156221d725598ff5ee2c02055049f72718` |
| model container descriptor digest | `fdf858b5e21a1951da9ed2c2b9e988f8a1f1edcf99d2bb84f54faf055dc7e3e9` |

CLI 不允许覆盖这些锚点；测试替代锚点只在显式标记 `CONTROLLED_FIXTURE` 的 Python 测试接口中使用。
`model_bin_raw_sha256` 必须由真实文件计算并同时匹配 recovery file-table 与容器 payload 字段；
不能预填上表中的 descriptor SHA。manifest 自身原始文件 SHA、canonical manifest digest 也分别记录。

需要研究者提供/确认的真实路径：

1. **原始 final fit 目录**，包含 `qwen3_final_manifest.json`；本地证据尚不能给出其服务器绝对路径。
   不可用 OOF fit、D2 outer model、复制后重命名的目录代替。
2. fit 中绑定的原始 `study_path` 与 `publication_path`，以及它们的原始文件清单。
   工具不提供换路径覆盖，不会重写 manifest。
3. `fit.parent.parent/runs/<fit-digest>/`；必须包含 `recovery/{1000,...,31000,31500}`、`terminal`、`seal`。
   工具不搜索 latest checkpoint。
4. 现有 clean f199 predictor checkout。先前看到的名称
   `/home/dell/yuxiao/Untrusted_Network_Simulation_qwen3_predictor_f199` 只是待核实候选，
   **并非本轮已确认的真实路径**；也可能位于其他父目录。
5. 审计工具的当前执行源码位置，与 f199 predictor checkout 分开。工具/helper 本身不属于 f199。
6. 新的外部证据文件路径；不能置于 fit/runs 生命周期、任一不可变 artifact 或两个源码目录中。

输入路径必须指向真实目录，不允许 symlink 祖先；原始 fit 路径还必须等于封存的 `destination_path`。
对迁移/复制导致的路径不一致，应停止并讨论历史来源迁移方案，不能修改原始 manifest 来“通过”。
输入被拒绝后，writer 仍按输入的真实目录执行输出保护，不因审计失败而放宽。
能可靠解析的别名输入可以产生**外部**失败证据；不能可靠解析的保护根（例如悬空链接或循环）拒绝写入。

## 3. 复用现有验证器，不重写恢复逻辑

- `artifact_io.open_artifact_envelope`：canonical JSON、manifest digest、精确文件/目录清单、拒绝 symlink。
- `artifact_io.read_artifact_file`：每个选定文件的长度和字节 SHA。
- `artifact_io.tensor_state.decode_tensor_container`：末端模型和可选全部容器的字段、dtype、shape、
  连续 offsets、每张量 SHA 及完整 payload 验证；仅解码为只读 NumPy 数组，不构造 ToM。
- `TemporalCodeProvider.explicit`：复用既有完整时间表验证，含 parent、phase/day metadata、
  原始字节 SHA、dtype、shape、有限值；不重算 day 三角函数。
- `derived_capacity(1024)`：复用完整 PRE/结构化 token 版本及 day=256 的容量推导。

审计工具中的新增历史逻辑只限于已有 manifest 字段的身份联结与 ancestry 比较。
不重写 schedule 派生、训练损失语义、状态恢复、optimizer validation 或 RNG recapture。

`qwen3_final.open_fit` 不只是轻量 manifest reader：它读取 publication、角色 sidecar 和 Primary。
`verify_seal → verify_terminal → _source_gate / _model_optimizer / _chain` 会配置 runtime、构造真实
BackboneToM/AdamW、恢复模型/优化器/RNG，再检查 recapture。其源码已有完整恢复验证能力，
没有必要另写一套；但它不适用于本轮“不启动真实 ToM/不恢复状态”的边界。
**不要把现有 `uns validate-qwen3-final-fit` 当作本轮轻量审计命令运行。**
完整语义恢复重验须在后续单独批准的隔离进程和原始兼容 runtime 中复用既有验证器；
本轮不提供会隐式启动这一流程的 `--full` 开关。

## 4. 源码、执行 checkout 与 runtime 的关系

工具分别记录：

- fit 保存的训练 `source` 和 runtime 摘要。
- 指定 predictor checkout 的 HEAD、clean 状态、全部 `werewolf/**/*.py + run_random.py` 的实现摘要，
  以及磁盘字节与 HEAD blob 的比较。Git 使用 `GIT_OPTIONAL_LOCKS=0`，不刷新 index；
  不能依赖 `assume-unchanged` 隐藏变更。
- 工具正在执行的调用方 checkout 与工具/helper 字节指纹，单独记录，不冒称是 f199。
- 当前 Python/platform/NumPy/Torch/transformers 的已安装版本与保存字段的逐项比较。
  不探测 GPU、不调用 `configure_runtime`，不证明 CUDA、cuDNN、TF32、线程、确定性或完整 runtime 等价。

特别注意，冻结 backbone config 的 `source_revision` 是
`927c4cd7576be04792d5729999bb254b22c27626`；`runtime_provenance` 将这个配置字段写入 runtime。
它与 `fit.protocol_inputs.source.source_revision`（实际训练 checkout）是不同字段。
工具检查前者与 config 相符，后者与指定的 clean f199 checkout 相符，不能强行令二者相等。

重用的 artifact/temporal/capacity/PRE 读取代码及 backbone config，还会与 f199 checkout 的实际
字节核对；这九个文件在当前 Mac 源码与本地 f199 Git blob 中相同。服务器是否相同尚未验证。
当前调用方源码含 D4 未提交成果，因此其 dirty/head-blob 状态如实输出；这不是正式运行源码资格通过。
本工具不调用或绕过任何 D2/D3 formal source guard。

工具不查询运行中的模型服务。即使指定 checkout 通过，也不能据此说现有服务实际从该目录启动；
证据明确保留 `existing_running_predictor_process_identity=NOT_OBSERVED`。

## 5. 服务器手动步骤（本轮未执行）

先决条件：本轮工具尚未提交/推送/部署。后续须先由研究者审阅，再通过既定 GitHub 同步流程使
服务器可读到相同工具源码；本轮不为了部署工具而切换任何在用目录、创建 clone/worktree 或操作服务器。
以下命令假定工具已可用；不能拿旧 checkout 中不存在的工具假装已执行。

### 5.1 只读找路径与检查使用状态

以下均由研究者在服务器手动执行。`rg` 列出文件；如果未列出，不能据此假定不存在或另选最近模型。

```sh
rg --files --hidden --no-ignore /data/yuxiao/Untrusted_Network_Simulation \
  -g qwen3_final_manifest.json
git -C /home/dell/yuxiao/Untrusted_Network_Simulation worktree list --porcelain
ps -eo pid,ppid,lstart,args
```

不要把完整 `ps`、manifest 或路径列表作为公开证据包发布。人工确认没有写入 fit/runs 的训练、
resume、seal 或维护任务；只读 inference 是否竞争磁盘/GPU也须另行确认。
工具不加锁、不停进程；它会复查 manifest 与源码是否在读取中变化，但**不是文件系统快照**，
不能证明没有并发写入或持久不可变。若无法确认静止来源，先不运行或停止采信。

### 5.2 设置实际路径并执行轻量审计

```sh
conda activate /data/yuxiao/envs/untrusted-network-simulation
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export GIT_OPTIONAL_LOCKS=0

UNS_AUDIT_SOURCE=/home/dell/yuxiao/Untrusted_Network_Simulation
UNS_FINAL_FIT='/填写已确认的原始final-fit绝对真实路径'
UNS_FINAL_CHECKOUT='/填写已确认的clean-f199绝对真实路径'
UNS_AUDIT_OUTPUT=/data/yuxiao/Untrusted_Network_Simulation/observer-dtrs/source-audits

# 仅创建独立证据目录，不修改任何被审计 artifact。
mkdir -p "$UNS_AUDIT_OUTPUT"
python -B "$UNS_AUDIT_SOURCE/scripts/audit_qwen3_final_provenance.py" \
  --fit "$UNS_FINAL_FIT" \
  --predictor-checkout "$UNS_FINAL_CHECKOUT" \
  --mode metadata \
  --output "$UNS_AUDIT_OUTPUT/final-tom-metadata-01.json"
echo $?
sha256sum "$UNS_AUDIT_OUTPUT/final-tom-metadata-01.json"
```

不要把工具复制进 f199 checkout。命令不启动 client/模型，不调用训练/重训/恢复，不读 TEST。
`metadata` 也不是“只看几个 JSON”：它会哈希 fit 文件和末端 model.bin，导入既有 temporal reader
所需的 NumPy/Torch 库，但不实例化真实 ToM、分配 CUDA tensor 或读取 optimizer payload。
其内存需求包含 Python/Torch 导入开销与一个模型容器的解码；不是零内存操作。

### 5.3 可选的全链文件审计

在人工确认磁盘 I/O 预算及目录静止后，再用**新的输出名**执行：

```sh
python -B "$UNS_AUDIT_SOURCE/scripts/audit_qwen3_final_provenance.py" \
  --fit "$UNS_FINAL_FIT" \
  --predictor-checkout "$UNS_FINAL_CHECKOUT" \
  --mode chain-files \
  --output "$UNS_AUDIT_OUTPUT/final-tom-chain-files-01.json"
echo $?
sha256sum "$UNS_AUDIT_OUTPUT/final-tom-chain-files-01.json"
```

此模式检查全部32个点，读 model/optimizer/RNG/loss 的原始字节并验证 tensor codec；
逐 artifact/容器处理，不保留整条链，但 I/O 可能为 GB 级并有重复校验读取。
不需要 GPU，也不启动已有 vLLM。未经容量确认不要与正在高负载写盘的采集任务竞争。
失败后不要修复、清理、迁移或覆盖原产物；保留失败 JSON，用新证据输出名重查。

### 5.4 回传的最小材料

回传上述 JSON、命令打印的 `evidence_digest` 与 `evidence_file_raw_sha256`，并通过独立渠道保存
`sha256sum` 输出。不要仅传文件内自带摘要。证据文件权限是0600，独占创建并 fsync，
拒绝覆盖、写入不可变 artifact 或源码目录。输出失败返回3，未声称证据已保存。

## 6. 成功、失败与尚未执行

成功 stdout 为最小对象：`status=PASS_WITH_EXPLICIT_SCOPE_LIMITS`、`evidence_digest`、
`evidence_file_raw_sha256`。退出0只代表上表选定范围。JSON 中保存各阶段状态；未到达的检查为 NOT_RUN。

失败返回2，同时尽量保存脱敏证据：`status=FAILED`、`failure.stage/code/exception_type`。
不回显原始异常文本、路径、manifest 或 tensor 名称。
拒绝输出（文件已存在、父目录不存在、symlink、保护目录或不能确认的保护根等）返回3；命令行/启动环境错误另行失败。

以下任一情形必须停止进入部署资格阶段：

- fit、terminal、seal 或 descriptor 外部锚点不符；实际权重字节不符；恢复点缺失/多出/断链。
- 重新计算摘要后仍有来源、步数、前驱、容器、publication/study、initial 或 temporal parent 错配。
- 时间表 shape、dtype、原始 SHA、day 有限性或1024/256容量不符。
- predictor checkout 非 clean f199、磁盘源码与 HEAD 不同，或 saved implementation/runtime binding 不符。
- 读取中 manifest/源码变化、读取失败或损坏；报告缺少/未执行关键检查。
- `runtime.observed_fields_status=MISMATCH_OR_PACKAGE_MISSING`：字节审计可成功，但环境资格必须停止补证。
- 尚未获得独立受控的完整 runtime、原始语义恢复、真实服务取消/超时/容量和进程来源证明。

不允许用新 fit/最近 checkpoint、OOF/final 互换、CPU device 覆盖或重新封存摘要来补齐。
工具输出 PASS 不是最终 capsule 准入；收到真实证据后，再讨论需要补做哪一级验证。

## 7. 输出范围与可公开性

默认只输出摘要、版本、计数、shape、容量、clean/一致性状态和固定错误码。
不输出角色、标签、嫌疑报告、game/sample IDs、Primary 内容、完整 runtime/manifest、
模型或 optimizer 值、内部绝对路径。训练内容可能被读取做字节哈希，但不解析成新的监督数据。

其中源码 OID、artifact/权重/时间表摘要、容量、步骤数、版本及分层状态通常可作为论文复现证据；
公开前仍需研究者审阅数据/模型发布权限与运行环境版本。摘要不自动赋予原始数据或模型的公开权限。
如需进一步脱敏，应制作另行标记/摘要的共享副本，保留原始0600证据与外部 SHA，不能改写原证据。

## 8. 本轮验收与下一步

人工 fixture 的 fit、publication、1500个游戏 ID、32个点、8字节张量均明确是受控来源；
不是正式训练样本、真实 final 权重或实际服务器验收。测试涵盖错误锚点、容器/裸文件 SHA 混淆、
重算摘要后的绑定/ancestry错误、缺失/多余/损坏文件、时间表/parent/NaN、源码与 runtime 错配、
assume-unchanged、symlink、并发 manifest 变化、输出拒绝/脱敏和输入字节/权限保留。
阶段验收 JSON 单独记录最终实际测试结果和源码指纹。

**本轮结束后等待研究者手动回传真实证据。**
缺少的实际 final fit、raw model SHA、temporal day SHA、服务器 checkout/runtime、全链文件状态，
以及原始语义恢复状态都不能在本地标记通过。此后才决定 inference capsule 开发与 QUAL；
CAL/TEST、正式模型选择及执行冻结仍不授权。

## 9. P1 输出保护修复与五文件隔离复验

修复前的 writer 使用 `Path.absolute()` 得到的未规范化输入保护根。
公开 CLI 在来源检查失败后仍保存失败 JSON；当 predictor 输入是 symlink 别名而输出使用其真实路径时，
词法父目录比较漏拦截，可能在冻结 checkout 中新增未跟踪证据文件。
上一轮仅在人工目录复现，五文件依赖独立性成立，但安全同步被这一 P1 阻止。

本轮修复局限于输出 writer 和其测试：

- 对保护根严格解析已存在祖先，再允许普通的缺失后缀；遇到悬空链接、循环、不可访问祖先或
  无法解析的 `..` 时失败关闭。恢复链位置由**真实 fit** 的父目录推导。
- 输出父目录必须已存在且为规范化路径，拒绝通过 symlink/`..` 间接输出。
- 同时比较规范化路径和目录的 `(st_dev, st_ino)`，覆盖大小写等目录身份别名。
- 成功证据和失败证据使用同一保护；检查全部输出祖先的 artifact manifest 与 `.git` 标记，
  标记即使损坏或是悬空链接也拒绝写入，保护其他普通仓库和 linked worktree。
- 打开父目录时使用 `O_DIRECTORY|O_NOFOLLOW`，并重新检查目录身份；相对于持有的目录 fd
  使用 `O_EXCL|O_NOFOLLOW` 创建0600新文件，仍 fsync 文件和父目录，返回完整文件 SHA。
  这降低路径检查到创建之间的父目录替换风险，**仍不是文件系统快照或并发恶意重命名的隔离机制**。
  原有来源静止及独立证据目录要求继续有效。

没有修改 CLI 参数、生产锚点、0/2/3返回码、D0–D3、artifact_io、predictor、Shadow或控制层。
未增添 pin override 或真实推理入口。测试中的成功 CLI 使用显式 `CONTROLLED_FIXTURE` 锚点注入；
公开 CLI 子进程负例始终使用未经改动的生产锚点，不能把人工成功当作真实 final 验收。

先运行公开 CLI 回归并确认旧实现新增 `forbidden-evidence.json` 导致目录快照失败，
再修改 writer，确认该回归转绿；随后在原始冻结基点
`e513fd8b69404d07d69cd308023b673554f5bbf0` 的433个文件加本阶段五文件中独立复验。
临时 Git fixture 不注册 worktree，不修改原仓库、分支、index或stash。

复验覆盖 canonical/alias/`..` 输入和输出、fit/recovery/temporal/publication/其他源码目录、
成功与失败外部证据、坏manifest、来源不匹配、循环/悬空链接、覆盖/硬链接拒绝和 fsync。
负例同时比较完整文件快照、目录清单和 Git clean/HEAD/源码身份，不能只断言退出码。
大小写目录别名测试只在实际文件系统支持时执行，否则明确跳过。

验收 JSON 的 `history` 原样保留修复前记录及其原始 SHA，并分别记载：
原138项整工作区历史回归、上一轮95项五文件隔离审阅、P1修复前失败与修复后隔离验收。
**此次隔离测试数量以修复后记录为准，不复用138项作为本轮结果。**
服务器原始 fit、raw weights、真实 runtime 和完整语义恢复状态仍待手动证据，不因本轮修复自动升级。

本轮实际结果：五文件隔离的针对性测试 **90 passed in 27.13s**；包含必要基点回归共
**150 passed in 58.82s**，无失败或跳过。两个受控 CLI 模式均返回0，
输出 scope 为 `CONTROLLED_FIXTURE`，完整语义恢复仍为 `NOT_RUN`。
35个公开CLI保护路径负例返回3，所有输入的文件、权限、目录清单与Git状态均保持。
原本clean的人工checkout继续clean；故意注入源码错误的样例保持注入后的原状态，不自动修复。
符号链接/`..` 输入在能可靠确认保护根且输出独立时仍可返回2并保存外部失败证据。
具体日志摘要、受控输出摘要及当前源码指纹见修复后验收JSON；原138项只保留在历史记录中。
