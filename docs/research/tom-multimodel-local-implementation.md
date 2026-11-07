# 多模型 ToM Baseline：实现与研究协议

## V2 xgrammar Wire Schema 兼容修复与失败证据保护

本节对应 Mac 起点 `twd/mainline@f6a546309fdc9284ef24653a363d6dae61f359df` 后的局部修复。用户报告 Ubuntu Qwen3 V2 Qualification Round 0 首次请求 HTTP 500、`EngineGenerateError`、没有正常 input/output 事件，并已形成 `outcome_unknown`。这些是用户提供的现场事实，本轮未 SSH 复核，也不重新分类、重试、补写或覆盖原 Round。下文旧操作手册的 Source 起点和验收数字是历史快照，不能替代本轮及服务器的实际状态。

### 根因证据与边界

原任务 `prompt.set_schema()` 在 `properties.suspected_players` 声明 `uniqueItems:true`；`render_set_prompt()` 又把完整科研 Schema 写入 user 消息。`task.validate_task()` 验证该 Schema 和消息仍与冻结 renderer 精确一致。修复前，`model_spec.wire()` 把同一 Schema 原样放入 OpenAI `response_format.json_schema.schema`；`candidate_client.collect_round()`、`SyntheticCandidateClient.dispatch()` 和 `live_runner.run_model()` 均消费该 Wire，并绑定其 Digest。

官方 vLLM 0.27.0 的对应路径为：

1. [OpenAI Schema 转换，engine/protocol.py:179–209](https://github.com/vllm-project/vllm/blob/v0.27.0/vllm/entrypoints/openai/engine/protocol.py#L179-L209)：将 `response_format.json_schema` 放入 `StructuredOutputsParams.json`。
2. [ChatCompletionRequest:639–726](https://github.com/vllm-project/vllm/blob/v0.27.0/vllm/entrypoints/openai/chat_completion/protocol.py#L639-L726)：提取结构化约束并传入 SamplingParams。
3. [InputProcessor:91–109](https://github.com/vllm-project/vllm/blob/v0.27.0/vllm/v1/engine/input_processor.py#L91-L109) 调用 `params.verify()`；[SamplingParams:923–1014](https://github.com/vllm-project/vllm/blob/v0.27.0/vllm/sampling_params.py#L923-L1014) 在固定 xgrammar 后端下调用其校验，不使用 auto 后端兜底。
4. [backend_xgrammar.py:225–269、272–324](https://github.com/vllm-project/vllm/blob/v0.27.0/vllm/v1/structured_output/backend_xgrammar.py#L225-L324)：数组中出现 `uniqueItems` 即被判不支持，连 `uniqueItems:false` 也不能保留；校验在 grammar 构造前抛出 ValueError。
5. [AsyncLLM:354–402](https://github.com/vllm-project/vllm/blob/v0.27.0/vllm/v1/engine/async_llm.py#L354-L402) 在调用本项目所审计的 `_add_request` 之前处理输入；[同文件:652](https://github.com/vllm-project/vllm/blob/v0.27.0/vllm/v1/engine/async_llm.py#L652) 存在 `EngineGenerateError` 包装路径。因此，缺少探针 input/output 与前置验证失败相容，不能据此将旧未知结果改成“确认未执行”。现场 500 是否还包含其他内层异常，仍须原始异常链或新 Candidate 的实际资格验证判定。

Mac CPU 复现从官方固定 tag 读取源码到内存，只执行无模型依赖的 `has_xgrammar_unsupported_json_features()`；没有导入 vLLM 引擎。该文件实际 SHA256 为 `231f6b9d7dab5e8d68aba486fa5912db99f8bdd3f9d8842ee3e0bb12bdb7cb67`。修复前三模型 Wire 均被该函数拒绝；修复后三模型各 12 条、共 36 条 Wire 全部通过该特征检查，原科研 Schema 仍被原函数判不支持。此结果不是实际 xgrammar grammar 编译或 GPU Qualification PASS。

### 唯一修复规则与科研语义

现有 `model_spec.wire_schema()` 先验证原 Task，再仅对 SET_VERSION 的 Schema 深复制并删除 `/properties/suspected_players/uniqueItems`。原 Task、Prompt 中完整 Schema、原生模板、玩家枚举、自身排除、数组长度、`additionalProperties:false`、Wire 的 `strict:true` 以及应用层解析全部保留。不做递归删关键字、宽泛降级、换后端、去重修复或请求重试。

软件规则身份为 `classic7_suspicion_set_xgrammar_wire_projection_v1`，规则摘要为 `04273ad1a40df4199428baf51dab87a8f1fcee2b5dddd1648436e01caf2d2ea6`。它是现有 V2 的后端 Wire 投影身份，不是新 Task、V3 方法或新的 Admission 类型。

- CPU Parity 结果增加相同 `wire_schema_projection` 和实际 `wire_sequence_digest`；文本/Token IDs 仍针对完整原任务消息核验。
- SET Suite 记录相同投影身份，每个 `case.wire_digest` 绑定投影后的完整 OpenAI 请求；Source 仍要求精确匹配。旧投影前 Suite 不自动升级或兼容放行。
- `gpu_probe`、`qualification_evidence.validate_round()`、`live_service`、`live_runner.validate_reply()` 全部以同一个 `wire_schema(task)` 比较捕获的实际 SamplingParams Schema。比较仍是精确相等，不接受“原 Schema 或投影后 Schema 任意一种”。
- `output.parse_set_output()` 保持不变：合法空集、全六席和真子集语义不变；重复玩家仍是 `duplicate_player`，集合为 `None`，正式评价保留原非法分母；不会去重后改判有效。Qualification 中完整 stop 响应仍须通过原应用层集合校验。

科研协议 `classic7_suspicion_set_research_protocol_v2` 及其摘要 `e135168b1927c1542d7e0c991aaeb5fe9dc8814317b9316071dd1ffc655b71dc` 保持不变。原概率投影规则、Gold、Full/NU、Set-F1、Bootstrap、历史 Direct-Probability 均不改变。必须同时披露：后端生成空间现在允许产生重复成员，唯一性最终由应用层拒绝；有效输出定义相同不意味着模型生成分布、有效率或指标必然相同。不能把本次修复描述为 Wire 字节不变或保证分数提高。

### Chat Template warmup 单独结论

[官方 BaseRenderer.warmup:237–259](https://github.com/vllm-project/vllm/blob/v0.27.0/vllm/renderers/base.py#L237-L259) 使用单条 user `warmup` 消息，异常仅记为 warning。冻结 Qwen 模板第二行直接访问 `messages[1]`；Mac 使用同一模板对该单消息输入复现 `list object has no element 1`。真实任务包含 system＋user，三模型各 12 条真实任务形状的消息均可渲染。本轮不修改模板、模板 SHA 或 Prompt，也不把该非任务预热警告与实际 500 合并为一个根因。

### 必须重新生成与可以复用的证据

|对象|处理与需冻结的字段|
|---|---|
|科研协议、原 Task/Prompt/Gold、Selection/Reference/OOF、模型与环境|继续复用原身份；历史权重、Tokenizer 和 Conda 文件不修改；物理字节检查仍执行|
|Source v2/v3|新 Git Revision、Tree、Implementation Digest、各文件 SHA；当前 Mac dirty Source 不具备正式执行资格|
|真实 CPU Parity|重新运行，保留 Task/Prompt/System、原生 Token IDs 与新 Wire 投影/序列摘要；旧 CPU PASS 不代表新 Wire 已验收|
|V2 Cohort|在新 Source 下重新发布；请求 Task/科学行身份应不变，Manifest Source/Artifact Digest 改变，禁止改写旧 Manifest|
|三模型 SET Suite|重新发布投影身份、12 个 Wire Digest、Suite Digest、Source；Prompt/输入 IDs 应相同但须真实核验|
|完整 Token Budget|按新 Source 和新 Cohort 重测 10,714 条；不复制旧预算换签，不截断/筛行，不推定长度统计必然相同|
|Deployment、两轮 Qualification、Admission|新 Candidate、新 Deployment Digest、两次独立进程生命身份和新 Round Seal；旧失败 Round 0 永久保留，不在原路径重试|
|Run Contract、Permit、Prediction/Run Seal 与 Analysis|新 Run ID、新 Source/Task/Cohort/Budget/Admission 绑定；如已有旧对象则保留，不重签、不覆盖|

V1 Wire 字节与原 76-call 协议保持不变；旧 V1 读取及数值语义按回归验证。投影前的 V2 Suite/Deployment 在当前严格读取链中可能被拒绝，这是 Wire 变化后的预期门禁；旧证据以其原 Source 和通用 Artifact 校验核读，不能给旧内容补新投影字段或重算 Digest 来“修复”。`CONTROL_FILES`、Source 兼容算法、两轮资格、Admission、Permit、显式执行和未知结果停止均未放宽。

### 用户手动部署的安全顺序

1. 在更新服务器 Source 前，用旧 Source/旧配置核对旧 Candidate 的自有服务；如仍运行，由用户按原 Manager 的所有权检查安全停止。保留旧 Candidate 全目录、日志、dispatch/failure 和 `outcome_unknown`。不能用新读取器失败为理由删除旧证据。
2. 用户审阅本次 Mac Diff 后自行 Commit/Push；然后自行核验服务器实际 HEAD、干净 detached 状态、origin 和祖先关系，再更新至准确新 Commit。旧手册 A 中 `094d0033…` 是历史起点，本轮必须使用用户核验的实际旧 Source；Codex 不执行更新。
3. 由用户确定新的 CPU 根、新配置、新 Candidate 名和三个新 Run ID。仅沿用历史科学输入和通过核验的 assets/environment Receipt；不沿用旧 V2 Cohort/Suite/Budget/Deployment/Admission/Permit 指针。
4. 在固定干净新 Source 下，沿用现有 `uns tom-llm prepare --stage tokenizer/cohort/suite/budget/model` 顺序及保存机制。先验三模型真实 Tokenizer Parity 和相同 Wire 投影，再验完整 150 局/10,714 请求及 7,680＋512/8,192 预算。阶段失败即停止。
5. 经用户独立 GPU 授权，在每个新 Candidate 中完成两次不同实例的 Round 0/1，再生成新 Admission。旧 Round 0 不重试；任何未知结果仍停止，不以本次 CPU 诊断作“确认未执行”的豁免。
6. 用户准备三个独立正式 Run Contract/Permit，再明确授权并手动正式推理；冻结各自预测和 Run Seal 后才执行跨 Run 分析。仍复用既有 CLI、Runner、发布与门禁，不新建平行实验框架。

本轮 Mac 验收：新增 60 项参数化用例，涵盖三模型全部 12 个实际 synthetic Wire、不可变 Task/Prompt、合法集合/重复玩家、CPU Parity Wire 身份、Suite Digest/规则篡改拒绝、实际 Sampling Schema 漂移和正式探针/响应读取链。先观察最小缺陷回归失败，再修复；定向 63 passed，完整 `tests/tom_llm` 加 canonical Artifact 回归为 **847 passed、54 skipped、0 failed**。54 项中 48 项缺显式提供的官方 Tokenizer 小资产，6 项缺官方 vLLM argparse 源文件；没有把 Mock 当真实 Parity。官方纯验证函数另行复现原 Schema 的确切 ValueError，未导入真实 grammar 编译器或 GPU。`git diff --check` 通过。本轮未 SSH、Commit 或 Push，未修改服务器、模型、环境及旧 Artifact。

## V2 正式执行最小适配与用户服务器操作手册

本节对应 `094d0033d8839da6d3e9610381acd16652139f73` 之后的 Mac 本地开发。Codex 仅修改本地代码、运行 CPU/Mock；没有执行下列服务器命令。新 Commit 尚须用户审阅、授权提交和推送。以下命令仅供用户在 Ubuntu 的 Bash 中逐阶段执行；每阶段成功后再进入下一阶段。命令不是新的服务器或 GPU 授权。

科研协议 `classic7_suspicion_set_research_protocol_v2` 的原字典及摘要 `e135168b1927c1542d7e0c991aaeb5fe9dc8814317b9316071dd1ffc655b71dc` 保持不变，包括原有 CPU 阶段状态文字。正式执行另绑定 `classic7_suspicion_set_live_execution_v1`：获批 Task、科研协议摘要、同一干净提交的 Source v2/v3、冻结 Cohort 和每模型 Budget Manifest。它是软件合同，不是物理资格，也不替代 Admission、Permit 或 `--execute-live`。

V2 的新 Suite 和 Budget 记录实际 Source。旧 V1 和早期无 Source 字段的 V2 CPU 输入仍可离线核读；后者不得用于新 V2 物理资格。新链条不继承 V1 Suite、Budget、Deployment、Admission 或 Permit；历史权重、官方 Tokenizer 和 Conda 环境可只读复用。`CONTROL_FILES` 及资格兼容算法未扩大。

### A. 固定新提交并核验服务器 Source

前提：新代码已经用户批准并提交、推送到既有 `origin/twd/mainline`；唯一 checkout 仍是下列目录，原 HEAD 为 `094d0033…`，干净且 detached。以下阶段只写该 checkout 的 Git 对象和受控工作树；不更新其他 checkout。若任一检查失败，停止，不 Reset、Clean、Stash、Merge、Rebase 或强制 Checkout。

```bash
set -euo pipefail
PROJECT=/home/dell/yuxiao/Untrusted_Network_Simulation_mainline_6249485c56cc
BASELINE_PY=/data/yuxiao/envs/untrusted-network-simulation-tom-baseline/bin/python
V2_COMMIT='填写用户批准并已推送的新40位Commit'
export PROJECT BASELINE_PY V2_COMMIT
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export GIT_OPTIONAL_LOCKS=0 GIT_NO_LAZY_FETCH=1
[[ "$V2_COMMIT" =~ ^[0-9a-f]{40}$ ]]
cd "$PROJECT"
test "$(pwd -P)" = "$PROJECT"
test "$(git rev-parse --show-toplevel)" = "$PROJECT"
git rev-parse --git-common-dir
test "$(git rev-parse HEAD)" = 094d0033d8839da6d3e9610381acd16652139f73
if git symbolic-ref -q HEAD; then exit 1; fi
test -z "$(git -c core.fsmonitor=false status --porcelain=v1 --untracked-files=all)"
git diff --cached --quiet
ORIGIN=$(git remote get-url origin)
case "$ORIGIN" in
  https://github.com/MonsterTracy/Untrusted_Network_Simulation|https://github.com/MonsterTracy/Untrusted_Network_Simulation.git|git@github.com:MonsterTracy/Untrusted_Network_Simulation.git) ;;
  *) exit 1 ;;
esac
test "$(git ls-remote --heads origin refs/heads/twd/mainline | cut -f1)" = "$V2_COMMIT"
if git config --get-regexp '^filter\..*\.(smudge|process|required)$'; then
  echo '发现外部过滤器；停止，先核验副作用'
  exit 1
fi
test ! -x "$(git rev-parse --git-path hooks/post-checkout)"
git fetch --no-tags origin twd/mainline
test "$(git rev-parse FETCH_HEAD)" = "$V2_COMMIT"
git merge-base --is-ancestor HEAD "$V2_COMMIT"
git checkout --detach "$V2_COMMIT"
test "$(git rev-parse HEAD)" = "$V2_COMMIT"
if git symbolic-ref -q HEAD; then exit 1; fi
test -z "$(git -c core.fsmonitor=false status --porcelain=v1 --untracked-files=all)"
git diff --cached --quiet
"$BASELINE_PY" -B - <<'PY'
import json,os
from werewolf.tom_llm.multimodel import source_identity
from werewolf.tom_llm.experiment import source_v3,protocol,APPROVED_PROTOCOL_VERSION
from werewolf.tom_llm.task import SET_VERSION
from werewolf.tom_llm.deployment import validate_source_evidence
from werewolf.tom_llm.common import digest
assert digest(protocol(SET_VERSION,protocol_version=APPROVED_PROTOCOL_VERSION))=='e135168b1927c1542d7e0c991aaeb5fe9dc8814317b9316071dd1ffc655b71dc'
for source in (source_identity(),source_v3()):
    validate_source_evidence(source)
    assert source['source_revision']==os.environ['V2_COMMIT']
    assert source['source_inventory_clean'] and source['source_status_digest']==digest('')
    print(json.dumps({k:source[k] for k in ('schema_version','source_revision','commit_tree',
        'implementation_digest','source_inventory_clean','source_status_digest')},sort_keys=True))
    print('inventory_count',len(source['file_digests']))
PY
```

成功判据：精确目标 SHA、detached、工作区及 index 干净，Source v2/v3 同 Revision/Tree、库存为 210/213，输出实际摘要。不要用 Mac 含 DTRS 开发修改的摘要替代服务器结果。此后直至三个 Run 封存完成，不再修改 Source；Source 改变就必须在新路径重新准备 V2 输入与资格，不能重签旧证据。

### B. 新配置、真实 CPU Parity、Cohort、Suite 与全量 Budget

前提：A 通过；用户已批准新的目录名、Candidate 名、三个 Run ID 和实际 GPU UUID。以下变量由用户填写，均不是本手册声称存在的 Artifact、服务或设备。三个历史 YAML 只读，必须是已验收的 V1 来源配置。

```bash
V2_ROOT='/data/yuxiao/TomBaseline-data/填写用户批准的新V2目录名'
CANDIDATE_ID='填写用户批准的新Candidate名'
RUN_QWEN3='填写用户批准的Qwen3新RunID'
RUN_YI='填写用户批准的Yi新RunID'
RUN_MISTRAL='填写用户批准的Mistral新RunID'
GPU_UUID='填写用户确认的实际GPU-UUID'
CFG="$V2_ROOT/operations.json"
export V2_ROOT CANDIDATE_ID RUN_QWEN3 RUN_YI RUN_MISTRAL GPU_UUID CFG
"$BASELINE_PY" -B - <<'PY'
import os,re
from pathlib import Path
base=Path('/data/yuxiao/TomBaseline-data');root=Path(os.environ['V2_ROOT'])
assert root.parent==base and not root.exists()
assert not any(p.is_symlink() for p in (root,*root.parents))
for name in ('CANDIDATE_ID','RUN_QWEN3','RUN_YI','RUN_MISTRAL'):
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',os.environ[name])
assert len({os.environ[n] for n in ('RUN_QWEN3','RUN_YI','RUN_MISTRAL')})==3
assert os.environ['GPU_UUID'].startswith('GPU-') and '填写' not in os.environ['GPU_UUID']
root.mkdir()
for name in ('cpu-cache','tmp','logs','results'):(root/name).mkdir()
PY
export CUDA_VISIBLE_DEVICES=''
export HF_HOME="$V2_ROOT/cpu-cache/huggingface"
export HF_HUB_CACHE="$V2_ROOT/cpu-cache/hub"
export HF_MODULES_CACHE="$V2_ROOT/cpu-cache/modules"
export XDG_CACHE_HOME="$V2_ROOT/cpu-cache/xdg"
export TORCH_HOME="$V2_ROOT/cpu-cache/torch"
export TMPDIR="$V2_ROOT/tmp"
uns() { "$BASELINE_PY" -B -m werewolf.cli "$@"; }
"$BASELINE_PY" -B - <<'PY'
import os
from pathlib import Path
from copy import deepcopy
from werewolf.artifact_io import canonical_json_bytes
from werewolf.tom_llm.operations import load_config,anchor
from werewolf.tom_llm.physical import read_receipt
from werewolf.tom_llm.deployment import open_deployment,profile
from werewolf.tom_llm.experiment import APPROVED_PROTOCOL_VERSION
from werewolf.tom_llm.task import SET_VERSION
from werewolf.tom_llm.model_spec import ModelSpec
base=Path('/data/yuxiao/TomBaseline-data');root=Path(os.environ['V2_ROOT'])
names={'qwen3':'Qwen3-8B','yi':'Yi-1.5-9B-Chat-16K','mistral':'Mistral-7B-Instruct-v0.3'}
science=('publication','prepared','evaluation','expected_reference','selection','expected_selection_json',
         'selection_provenance_json','pre_reference_evidence','reconciliation_evidence')
config=None
for key,name in names.items():
    old=load_config(base/(name+'.yaml'))
    shared={field:old[field] for field in science}
    if config is None:
        config={**shared,'output_root':str(root/'experiments'),'task_version':SET_VERSION,
                'protocol_version':APPROVED_PROTOCOL_VERSION,'models':{}}
    else:assert all(config[field]==shared[field] for field in science)
    item=old['models'][key]
    dep=anchor(item['deployment'])
    _,record=open_deployment(dep.path,dep.manifest_digest,current_source=False)
    candidate=base/'multimodel'/key/os.environ['CANDIDATE_ID']
    assert not candidate.exists()
    p=record['profile']
    new={'candidate_root':str(candidate),'cpu_root':str(root/'cpu'/key),
         'model_directory':record['model']['directory'],'environment_prefix':p['environment_prefix'],
         'prior_audit_digest':record['environment']['prior_audit_digest'],
         'tokenizer_assets':str(Path('/data/yuxiao/models/tokenizer-assets')/key),
         'gpu_uuid':os.environ['GPU_UUID']}
    assert new['model_directory']==item['model_directory']
    assert new['environment_prefix']==item['environment_prefix']
    assert new['model_directory'].startswith('/data/yuxiao/models/')
    assert Path(new['tokenizer_assets']).is_dir()
    for field,kind,payload in (('assets','model','model'),('environment','environment','environment')):
        receipt=anchor(item[field])
        assert read_receipt(receipt.path,kind,receipt.manifest_digest)==record[payload]
        new[field]=deepcopy(item[field])
    profile(ModelSpec(key),gpu_uuid=new['gpu_uuid'],candidate_root=candidate,
            model_directory=new['model_directory'],environment_prefix=new['environment_prefix'])
    config['models'][key]=new
path=Path(os.environ['CFG'])
with path.open('xb') as stream:stream.write(canonical_json_bytes(config))
print(path)
PY
for MODEL in qwen3 yi mistral; do
  uns tom-llm --config "$CFG" prepare --stage tokenizer --models "$MODEL" \
    | tee "$V2_ROOT/results/tokenizer-$MODEL.json"
done
uns tom-llm --config "$CFG" prepare --stage cohort --save-config \
  | tee "$V2_ROOT/results/cohort.json"
for MODEL in qwen3 yi mistral; do
  uns tom-llm --config "$CFG" prepare --stage suite --models "$MODEL" --save-config \
    | tee "$V2_ROOT/results/suite-$MODEL.json"
  uns tom-llm --config "$CFG" prepare --stage budget --models "$MODEL" --games 150 --save-config \
    | tee "$V2_ROOT/results/budget-$MODEL.json"
done
"$BASELINE_PY" -B - <<'PY'
import json,os
from werewolf.tom_llm.operations import load_config
from werewolf.tom_llm.qualification_inputs import synthetic_tasks,open_suite,open_budget,budget_matrix
from werewolf.tom_llm.task import SET_VERSION,task_binding
from werewolf.tom_llm.common import digest
from werewolf.tom_llm.multimodel import source_identity,read_cohort
from werewolf.tom_llm.model_spec import ModelSpec
from werewolf.tom_llm.experiment import FORMAL_PUBLIC_ROWS_DIGEST,protocol,APPROVED_PROTOCOL_VERSION
c=load_config(os.environ['CFG']);source=source_identity();reports={}
assert source['source_inventory_clean']
cohort,tasks=read_cohort(c['cohort']['path'],c['cohort']['digest'],current_source=True)
assert len(tasks)==10714 and len(cohort.manifest['selection']['game_ids'])==150
assert cohort.manifest['row_identity_digest']==FORMAL_PUBLIC_ROWS_DIGEST
assert cohort.manifest['reference']==protocol(SET_VERSION,protocol_version=APPROVED_PROTOCOL_VERSION)['fixed_oof']['reference']
for key,item in c['models'].items():
    with open(os.path.join(os.environ['V2_ROOT'],'results','tokenizer-'+key+'.json')) as stream:parity=json.load(stream)[key]['tokenizer']
    assert parity['status']=='CPU_PARITY_PASS' and parity['task']==task_binding(SET_VERSION)
    assert parity['task_sequence_digest']==digest(synthetic_tasks(version=SET_VERSION))
    s=item['suite'];open_suite(s['path'],ModelSpec(key),s['digest'],task_schema_version=SET_VERSION,expected_source=source)
    b=item['budget'];reports[key]=open_budget(b['path'],ModelSpec(key),b['digest'],task_schema_version=SET_VERSION,expected_source=source)
print(json.dumps(budget_matrix(reports),sort_keys=True))
PY
```

成功判据：每模型 12 条真实 V2 原生模板 Token IDs Parity；新集合 Task/Prompt/System 绑定完整；三个 Suite 各含 12 cases/38 calls 的资格日程；每个 Budget 覆盖同一 10,714 行/150 局，最大输入 ≤7,680、输出预留 512、上下文 8,192。逐行 IDs 比较用于 Parity/预算验证；Budget 持久化长度序列和统计，不声称存储全部 Token IDs。缺资产、缺行、超预算立即阻断，不筛行、不截断、不继承 V1 预算。

本阶段写入白名单：新 `$CFG` 及其原子保存临时文件、`$V2_ROOT/{cpu-cache,tmp,results}`、`$V2_ROOT/experiments/cohort`、`$V2_ROOT/cpu/<model>/{suite,budget}`。官方 Tokenizer 加载及 Python 依赖可能创建 HF/XDG 缓存和临时文件，全部定向到上述新根。`-B`/`PYTHONDONTWRITEBYTECODE=1` 禁止源码字节码；离线选项禁止下载。不安装依赖、不修改历史 YAML、权重或 Conda 环境。若旧配置缺少已封存 Receipt 或字段不一致，停止，报告实际缺项，不虚构 Digest 或重写历史。

### C. 同源新 Deployment，沿用模型和环境

前提：B 全部通过，三个新 Candidate 不存在，旧模型/环境与读取的 Receipt 字节一致。

```bash
for MODEL in qwen3 yi mistral; do
  uns tom-llm --config "$CFG" prepare --stage model --models "$MODEL" --save-config \
    | tee "$V2_ROOT/results/deployment-$MODEL.json"
done
```

写入：`/data/yuxiao/TomBaseline-data/multimodel/<model>/$CANDIDATE_ID/deployment` 和新 `$CFG`；使用 B 已生成的 Suite/Budget 与历史 assets/environment Receipt，不重新下载、上传或训练模型。既有严格物理检查仍会流式读取权重 SHA、核验环境版本和关键源码，属于 CPU/I/O，不加载模型张量；不能用 Receipt 存在替代当前物理一致性。成功判据：三个新 Deployment 的 Task/Source 与同源 Cohort/Suite/Budget 一致，`physical_gpu_admission=false`。此时尚不具备 GPU 资格。

### D. 两个不同服务实例的 GPU Qualification 与 Admission

前提：用户已独立批准实际 GPU 操作，设备及端口由现有 Manager 核验为空闲；三个模型串行使用原服务参数。下面整段一次只执行一个模型，分别令 `MODEL=qwen3`、`MODEL=yi`、`MODEL=mistral`，每次核查成功后再继续。不能在一个实例重复两轮。

```bash
MODEL=qwen3
uns tom-llm --config "$CFG" serve start --purpose qualification --model "$MODEL" --execute-live
uns tom-llm --config "$CFG" qualify --model "$MODEL" --round 0 --execute-live \
  | tee "$V2_ROOT/results/qualification-$MODEL-round-0.json"
uns tom-llm --config "$CFG" serve stop --purpose qualification --model "$MODEL" --execute-live
uns tom-llm --config "$CFG" serve start --purpose qualification --model "$MODEL" --execute-live
uns tom-llm --config "$CFG" qualify --model "$MODEL" --round 1 --execute-live \
  | tee "$V2_ROOT/results/qualification-$MODEL-round-1.json"
uns tom-llm --config "$CFG" serve stop --purpose qualification --model "$MODEL" --execute-live
uns tom-llm --config "$CFG" admit --models "$MODEL" --save-config \
  | tee "$V2_ROOT/results/admission-$MODEL.json"
```

写入：各新 Candidate 的 `service-control` 日志/所有权状态、`cache`、`tmp`、`qualification/engine`、`qualification/round-{0,1}`、`qualification/admission`；GPU Manager 还使用历史独立基线根 `/data/yuxiao/tom-llm-reasoner-b2/baseline-gpu.lock` 和本模型父目录 `/data/yuxiao/TomBaseline-data/multimodel/<model>` 内的短 RPC socket。`profile.environment_requirements` 给出精确缓存路径，由 Manager 创建；不写权重或环境。成功判据：每轮 38 calls 完整封存、实际新集合 Schema 合法、采样/原生 Token IDs/数值/显存审计通过；两轮进程生命身份不同；新 Admission 绑定此 Deployment 与两个 Round Seal。V1 Admission 或 CPU/Mock Artifact 均不能通过。任一轮失败或结果未知，保留证据，停止，不重发、不删除 Round、不伪造资格。

### E. 三个独立正式 Run 的 Contract 和 Permit

前提：三模型 D 全部通过，Source 未变。三个 Run 共用一个 Cohort，各自只含一个模型；新 ID 不复用历史三 Run。准备合同是 CPU/服务器写入，Permit 仍会核验真实 Admission、环境、模型及 Budget，不产生推理。

```bash
for MODEL in qwen3 yi mistral; do
  case "$MODEL" in
    qwen3) RUN_ID="$RUN_QWEN3" ;;
    yi) RUN_ID="$RUN_YI" ;;
    mistral) RUN_ID="$RUN_MISTRAL" ;;
  esac
  uns tom-llm --config "$CFG" prepare --stage run --models "$MODEL" \
    --run-id "$RUN_ID" --games 150 --mode formal \
    | tee "$V2_ROOT/results/contract-$MODEL.json"
  uns tom-llm --config "$CFG" prepare --stage permit --models "$MODEL" --run-id "$RUN_ID" \
    | tee "$V2_ROOT/results/permit-$MODEL.json"
done
```

写入：`$V2_ROOT/experiments/<RunID>/contract` 和 `models/<model>/permit/{permit.json,vllm.yaml,chat_template.jinja}`，以及命令结果目录。成功判据：正式 150 局、10,714 请求、固定科学行摘要 `8fd129c892d9657ed429d984c83acc278792637a5e96e06eb9c148f3b98e80c8`；研究协议摘要不变；Source v3 与 Qualification Source v2 来自同一干净提交，共享文件逐字节匹配；Permit 绑定准确的模型、Run、Source、Task、Admission 和同一个 Budget Manifest/路径。合同的执行绑定不授予资格，Permit 缺失仍不得启动。

### F. 串行启动、进度、停止与恢复

前提：用户已明确批准正式推理，E 全部通过。一次只运行一个模型；下列示例先 Qwen3，再显式替换为 Yi/Mistral 及对应 Run ID。

```bash
MODEL=qwen3
RUN_ID="$RUN_QWEN3"
uns tom-llm --config "$CFG" run --models "$MODEL" --run-id "$RUN_ID" \
  --execute-live --manage-services | tee "$V2_ROOT/logs/run-$MODEL.json"
```

该命令前台运行，由原 Manager 管理服务并在正常退出路径停止自身服务。可在第二个 Bash 使用相同变量查看进度（仅用户在已授权独占服务器资源上执行）：

```bash
uns tom-llm --config "$CFG" status --models "$MODEL" --run-id "$RUN_ID"
uns tom-llm --config "$CFG" serve logs --purpose run --model "$MODEL" --run-id "$RUN_ID" --lines 100
uns tom-llm --config "$CFG" serve status --purpose run --model "$MODEL" --run-id "$RUN_ID"
```

`status`/`serve status` 的既有实现也读取 GPU occupancy；不是只读取请求数。若只需请求进度且不做 GPU 枚举，用既有底层 CPU status（不写 Run）：

```bash
export RUN_ID
"$BASELINE_PY" -B - <<'PY'
import json,os
from pathlib import Path
from werewolf.tom_llm.live_runner import status
print(json.dumps(status(Path(os.environ['V2_ROOT'])/'experiments'/os.environ['RUN_ID']),sort_keys=True))
PY
```

人工需要中止时，先在前台控制终端用一次 Ctrl-C 停止控制器，再核对服务状态；若自身服务仍运行，才执行：

```bash
uns tom-llm --config "$CFG" serve stop --purpose run --model "$MODEL" --run-id "$RUN_ID" --execute-live
```

原 Manager 只对已记录的自身进程、匹配生命身份使用 SIGTERM；超时或身份不明时停止，不强杀、不接管其他进程。已有请求可能正在执行，手动停止不保证可恢复。出现 `unknown`、未配对 dispatch、`launch_incomplete`、`pending_launch` 或所有权异常时，禁止自动恢复或手工删除文件；先保全证据，单独审查。仅当已有检查证明未执行或已完整收到响应、没有未知请求、Permit/Source/预算均不变时，才能显式恢复同一 Run：

```bash
uns tom-llm --config "$CFG" run --models "$MODEL" --run-id "$RUN_ID" \
  --execute-live --manage-services --resume | tee "$V2_ROOT/logs/resume-$MODEL.json"
```

写入：Run 内 `execution.lock`、`models/<model>/{service,requests,sessions,seal}`、Run Seal 及控制日志；service 的所有缓存位于该 Run 的 service 根，短 RPC socket 仍位于已绑定模型父目录。有效与非法模型输出都占正式请求位置；未知结果不造 fallback，不补发。成功判据：每模型 completed_requests=10,714、blocked_requests=0、errors 为空、Prediction Seal 和 Run Seal 均通过读取链；不以 HTTP 成功或进度文件存在替代封存验收。

### G. 三个 Run Seal 核验与跨 Run Set-F1 报告

前提：三个 Run 独立完整封存。下面先通过既有读取链获得实际 Seal 摘要并输出；须与运行完成时保存的可信结果摘要相等后，再传给分析 CLI。不能只从目录名或新算一个未经锚定的值宣称已核验。

```bash
"$BASELINE_PY" -B - <<'PY'
import os,json
from pathlib import Path
from werewolf.tom_llm.baseline_analysis import sealed_inputs
from werewolf.tom_llm.common import strict_json
root=Path(os.environ['V2_ROOT'])
for key,var in [('qwen3','RUN_QWEN3'),('yi','RUN_YI'),('mistral','RUN_MISTRAL')]:
    completed=strict_json((root/'logs'/('run-'+key+'.json')).read_bytes())
    assert set(completed)=={'path','digest'}
    _,run,tasks,seal,rows,_=sealed_inputs(root/'experiments'/os.environ[var],completed['digest'])
    assert set(run['models'])=={key} and run['mode']=='formal' and len(tasks)==10714
    assert len(rows[key])==10714 and run['game_count']==150
    print(json.dumps({'model':key,'run':run['run_id'],'run_seal':seal.manifest_digest},sort_keys=True))
PY
QWEN3_SEAL='填写上一步与可信运行记录核对后的实际摘要'
YI_SEAL='填写上一步与可信运行记录核对后的实际摘要'
MISTRAL_SEAL='填写上一步与可信运行记录核对后的实际摘要'
uns tom-llm --config "$CFG" report --output "$V2_ROOT/cross-run-analysis" \
  --sealed-run "qwen3=$V2_ROOT/experiments/$RUN_QWEN3" --sealed-run-digest "qwen3=$QWEN3_SEAL" \
  --sealed-run "yi=$V2_ROOT/experiments/$RUN_YI" --sealed-run-digest "yi=$YI_SEAL" \
  --sealed-run "mistral=$V2_ROOT/experiments/$RUN_MISTRAL" --sealed-run-digest "mistral=$MISTRAL_SEAL" \
  | tee "$V2_ROOT/results/cross-run-analysis.json"
```

若曾恢复运行，`run-<model>.json` 可能是中止的空/部分输出；应改为读取实际成功封存那次命令的可信输出（例如 `resume-<model>.json`），不要覆盖、拼接或伪造旧日志。若返回的是已封存 Run 的 status 而非新 Seal 引用，使用此前成功封存时保存的外部锚点，不能将 status 当 Seal。

分析仅写新 `$V2_ROOT/cross-run-analysis`，其父目录允许短期 `.report-cache-*` 绘图缓存；不产生聚合模型 Run Seal。成功判据：三独立单模型 Seal、相同 Source/Task/协议/Cohort/Reference/Selection/科学行与请求顺序；Full/NU 同分母、游戏等权，Full 10 对、NU 15 对及既定比较层级；固定 OOF 合同通过；实际核对 Full 7,480/150、NU 5,502/150 和 10,000 次配对 Bootstrap。报告给出 Set-F1、其他集合指标及补充概率指标，保留两个确定性派生基线的 `model_run_seal:null`。不改变历史三个概率结果，不把本次已观察 150 局的方法比较表述为独立泛化或因果效应。

全流程计算量：CPU Parity 每模型 12 条；完整 Budget 每模型 10,714 条分词，共 32,142 条；物理 Qualification 每模型两实例共 76 calls，三模型 228 calls；正式预测每模型 10,714 条，共 32,142 条。墙钟时间取决于真实 Tokenizer、磁盘、模型及 GPU；本地 Mock 结果不能估算真实吞吐或授予任何物理资格。

### 本轮 Mac CPU/Mock 验收与保护记录

最终源码下，两组不重叠回归合计 **787 passed、54 skipped、0 failed**：三个入口/物理边界/端到端测试文件为 376 passed、39 skipped；其余 TomBaseline 与任务 A canonical Artifact 测试为 411 passed、15 skipped。此前整套 775 passed 后补齐的双向入口拒绝已在最终两组中覆盖，不能将多轮重复通过数相加。54 个跳过项中 48 项缺少显式提供的官方 Tokenizer 小资产，6 项缺少官方 vLLM 0.27 argparse 源文件；真实 CPU Parity、全量正式 Budget、物理 Qualification、Admission 和 GPU 推理均未在本轮运行。

测试覆盖同源新 V2 输入、150/10,714 合同边界、获批协议、版本双向错配、dirty Source、缺行/多行/重复行、超预算、固定 OOF 锚点、缺失或 Fake 资格、Permit 缺失/错绑、旧 V1/Mock、三个独立 Run 与原始封存读取。物理边界的正例使用明确标记的合成 Source/Tokenizer/资源桩，不构成真实物理资格。运行命令使用 Mac 既有 `3wd` Python、`CUDA_VISIBLE_DEVICES=''`、`-B`、禁用 pytest cache/自动插件及离线模式；loopback/Unix socket 仅限本机 Mock。

本轮修改范围为五个已审计入口、`baseline_analysis.py` 的执行状态/逐 Run 绑定透传、三个现有测试及本手册，共十文件。冻结科研协议函数、`CONTROL_FILES`、资格兼容算法经 AST 比较保持不变；Prompt、标签、投影、评分、Runner、模型注册表和 Admission 类型未修改。手册中 21 条 `uns tom-llm` 命令通过现有 CLI Parser 静态解析；11 段 Bash 与 6 段嵌入 Python 通过语法检查，没有执行服务器命令。

核对本轮起点、已登记的受保护文件哈希和当前 640 个项目文件后，差异仅上述十文件；DTRS 三文件与 `AGENTS.md`、`docs/development-guidelines.md` 五项字节/工作区身份保持不变，任务 A 九文件未改。分支仍为 `twd/mainline`、HEAD 仍为 `094d0033d8839da6d3e9610381acd16652139f73`，暂存区为空，index SHA256 仍为 `5a5e4bf43269b7cbc06a6624e9dbec9a979e5e3eced474d385a492fa4098b013`。未 Commit、Push、SSH 或修改服务器；当前 Mac Source 仍 dirty，不是可执行的正式 Source。下一步等待用户审阅代码并授权明确白名单的提交/推送，服务器更新、CPU 资产核验和 GPU 全部由用户另行手动执行。

## 已批准的怀疑集合 V2 协议与本地 CPU 实现

本节保留此前任务 B 协议与 CPU 实现阶段的历史记录；执行准备以本文件最前的新手册为准。下文不重新评分历史三个 Direct-Probability Run。历史实验继续以各自 Contract/Seal 记录的 Source 为准，不能由当前开发 HEAD 推定。

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
