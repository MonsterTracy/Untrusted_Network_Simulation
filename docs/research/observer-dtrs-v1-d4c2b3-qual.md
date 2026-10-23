# D4c-2b-3：真实 Final ToM QUAL 准备、CPU 对照工具与 spawn broker

状态：**QUAL_PREPARATION_ONLY**。真实权重、来源审计文件未在本地取得，本轮没有真实模型 QUAL。
源码基点 `three-way / 7b51f276b04b386ec1aa4ae44454c1d809a10897`；该 OID **不包含**未提交的 D4 实现。
本报告、机器记录和受审源码快照共同说明工作字节；不升级任何正式执行许可。
CAL/TEST 配置仍为 `PROPOSED_NOT_FROZEN_NOT_EXECUTABLE`。

## 1. 本轮实际范围

保留原有2个tracked修改、39个untracked文件及stash；仅新增本阶段文件。
不改Capsule、Loader、历史图、驻留服务/owner、原Shadow/Worker/Hook、D0–D3及旧Phase2。
测试仅在临时独立目录创建Git fixture提交；不属于本项目提交，也不操作其他worktree。
没有SSH/SCP、模型下载、真实权重加载、新游戏、vLLM、训练、CAL/TEST、commit或push。

检索已知Documents研究结果目录，以及用户指定Downloads和Desktop（排除隐藏目录、开发依赖目录，后者检索至7层）：
未找到真实 `qwen3_final_manifest.json`、`model.bin`、Metadata/Chain-files审计JSON。
其他项目的 `files_metadata.json` 不属于本项目，未作为来源。该结果不是全Mac/服务器不存在的证明。
历史用户确认的服务器审计PASS仍保留为用户证据；本轮不声称重新打开过真实材料。

新增文件：

| 文件 | 职责 |
| --- | --- |
| `werewolf/observer_dtrs/qual_sources.py` | 完整runtime源码inventory、原始字节/权限、不可覆盖快照与本地字节验证 |
| `werewolf/observer_dtrs/final_tom_qualification.py` | 公共PRE白名单、审计证据联结、精确CPU数值检查、驻留探测与失败封存 |
| `werewolf/observer_dtrs/final_tom_broker.py` | 有界socketpair租约，spawn子进程不持有owner/线程/Popen |
| `scripts/qualify_final_tom_capsule.py` | snapshot / verify-source / cpu；正式来源入口拒绝人工PRE和人工Capsule |
| `scripts/export_final_tom_qual_reference.py` | 干净历史checkout中独立导入原图与tensorizer，导出CPU reference；不打开训练生命周期 |
| `tests/observer_dtrs/test_final_tom_qualification.py` | 快照/来源/数值/失败/容量/CLI受控测试 |
| `tests/observer_dtrs/test_final_tom_broker.py` | spawn、断连、挂死取消、epoch、限额和错配测试 |
| `tests/observer_dtrs/qual_broker_fixtures.py` | 不导入torch、Agent或监督reader的人工spawn目标 |
| 本文及 `observer-dtrs-v1-d4c2b3-acceptance.json` | 协议、材料交接、实测及证据分级 |

## 2. 源码身份与路径保护

使用现有canonical JSON与artifact机制发布全新外部目录：

```text
manifest.json
snapshot.json
sources/werewolf/**/*.py
sources/run_random.py
sources/scripts/{build_final_tom_capsule,serve_final_tom_capsule,qualify_final_tom_capsule,export_final_tom_qual_reference}.py
sources/configs/observer_dtrs/*.json
sources/configs/paper/tom-backbone-study-v1.json
```

snapshot记录base HEAD、branch、精确路径、SHA-256、长度、mode及inventory digest。
它是宽于最小导入闭包的runtime源码快照，不含下载研究数据、测试产物、权重、监督、Git配置或stash。
验证既比较逐文件字节，也重新枚举代码人口：新增未审Python文件、遗漏、权限改变、额外载荷均拒绝。
输出要求全新路径、已存在规范父目录；拒绝 `..`、符号链接别名、checkout/artifact/显式保护树。
保守地禁止在输入文件的父目录内输出；输入PRE/审计文件应放在独立input目录，输出另设兄弟目录。
原artifact层提供原子不覆盖发布与fsync。失败目录保留，不清理、不修复，不自动重复推理。

源码快照明确标记 `REVIEWABLE_WORKING_BYTES_NOT_FORMAL_SOURCE_AUTHORIZATION`。
它可以固定未提交源字节，不能当成Git commit、正式部署批准或原D2/D3 production guard替代。
只做执行前后稳定性检查，不声称OS只读快照、跨进程源目录锁或抵抗恶意TOCTOU。
最终实际snapshot路径、manifest及inventory摘要见本阶段acceptance记录。

## 3. 数值对照的精确范围

输入packet必须是canonical JSON，最多32例、总字节≤16MiB，带外固定原始SHA：

```json
{"schema_version":"observer_dtrs_qual_public_pres_v1","scope":"PUBLIC_PRE_QUALIFICATION_ONLY","cases":[{"case_id":"explicit-public-case","pre":{}}]}
```

`pre`须为已有 `AuthoritativePREPrefix.to_record()` 完整合法记录；示例空对象不可执行。
parser roundtrip拒绝额外Y/报告/角色/训练支持字段；共享容量validator拒绝超1024结构token/256 day。
PRE必须由受限侧白名单导出，不能把整个publication或其public目录挂给推理进程。
人工packet标记 `CONTROLLED_FIXTURE`，只允许测试内核使用，公开cpu入口不接收。

比较规则预先固定 **atol=0、rtol=0**，不使用近并列或裁剪：

1. Capsule payload、外部manifest锚、原model raw SHA、descriptor SHA、Temporal parent及Loader身份验证。
2. 重开带外SHA固定的chain-files JSON，复用既有审计packet reader，核对原fit/source/seal/terminal/ancestry及权重/时间表身份。
3. 在另一个Python进程，要求历史checkout HEAD、clean状态、所有生产源码Git blob及implementation digest匹配。
4. 从该checkout独立导入原 `BackboneToM`、`tensorize_public_pre`、Temporal与strict restore，读取同一已验证Capsule权重。
5. 逐张量名称/dtype/shape/raw SHA、Temporal原值、整数输入/attention mask、PRE/plan/token_count完全一致。
6. 比较42个非自身log-Q（自身列的负无穷由原图/Loader约束），全部49个Q、重复PRE确定性。
7. 一个驻留PID顺序处理packet全部PRE；重复request必须返回DUPLICATE，不再次执行；关闭后确认PID退出。
8. 重开Capsule与源码快照，确认源文件和载荷未变化。

**重要限制：**参考类别为 `PINNED_ORIGINAL_GRAPH_CPU_SAME_CAPSULE_NOT_SEALED_LIFECYCLE`。
这是历史原图/编码/权重投影的CPU对照，**不是原 `SealedQwen3Predictor` 构造路径执行**。
没有调用 `open_fit` / `verify_seal`，不复演Primary、训练loss、schedule、optimizer/RNG恢复。
因为两侧从同一Capsule读取权重，不能单靠数值相等证明编译器没有选错原始权重；该证明依赖原审计外部锚与Capsule来源验证。
完整历史恢复和原sealed predictor初始化必须另外授权并保存证据，不能把这项Graph对照改名为完整恢复PASS。

原图reference与Loader分别记录实际CPU/package/thread身份；数值基础runtime字段必须一致，interop线程数另行记录，不冒称原训练runtime相等。
数值失败按tensor/Temporal/编码/log-Q/Q/驻留阶段停止；不增大容差，不重归一化，不回退模型。
输出 `failure/report.json` 包含失败阶段及cleanup；无complete、无伪造成功或BND。
`failed_check` 使用白名单常量区分恢复、Temporal、输入编码、log-Q、Q转换、重复推理及来源/runtime错误，不复制外部异常或监督内容。
完整工具在人工来源下测试，实际f199 checkout/权重/环境尚未运行。

## 4. 驻留probe与broker边界

CPU入口要求带外源码snapshot、Capsule manifest、PRE原始SHA和审计原始SHA。
独立CLI进程限定CPU线程（默认1）及interop=1，实际Runtime写入证据，未改变游戏进程线程/RNG。
Capsule/Loader/Runtime与服务epoch/instance身份由既有owner握手验证；没有放宽旧服务来源规则。
冷启动/查询/取消/关闭及帧限额沿用D4c-2b-2的工程默认，**尚未冻结为真实QUAL预算**。
`query_seconds`是单例提交、返回及直接Loader数值对照的端到端时间，**不是纯服务查询延迟**；不得据此宣称真实服务吞吐量。

broker由游戏外的可信控制层持有已经启动的owner；每条lease为新socketpair、随机token、完整session digest和epoch。
只传递socket和不可变绑定字节到spawn子进程，不序列化owner/Popen/线程，不增加网络listener。
父进程 `process.start()` 后须 `broker.release_parent_copy(lease)`，否则父端残留FD会阻止子进程EOF检测。
客户端读写有界，服务端累计字节/请求/队列限额仍有效；同一lease禁止并发查询。
子进程断连、崩溃、过期或broker关闭，父端调用原owner.cancel；取消未确认仍按原owner停用并终止其独占PID。
旧epoch lease不能在owner恢复后使用；恢复后显式申请新lease；无自动重试和旧成功响应重放。

测试证明spawn客户端无需torch/训练/Agent依赖，可顺序取得真实接口形状的人工Q。
它不转换旧 `TomIdentity`，不生成 `QPrediction` 的伪final身份，也不发布CAL包。
**本阶段没有把broker接入现有 `IsolatedShadowWorker` 或 `ShadowPREHook`。**
旧worker的fixture-only gate保持；正式连接仍需版本化final部署身份、新CAL包及受审capture/receipt适配。
broker不声称完整游戏覆盖台账；资格、提交及跳过分母仍须来自D4c-1耐久capture ledger。
已进入owner的请求沿用owner耐久日志；在broker入队前拒绝的坏帧不冒充owner已提交请求。

日志与I/O仍有OS调度/fsync限制，不是硬实时系统。Python导入/open测试不是OS权限沙箱。
本地只证明自己创建的推理PID退出，不证明完整进程树、GPU内核或远端任务已取消。

## 5. 真实材料交接清单

| 必需材料 | 用途与约束 | 当前状态 |
| --- | --- | --- |
| 原final fit绝对路径及manifest原字节 | 核对fit/protocol/source/study/Temporal parent；路径不能猜测 | BLOCKED_NOT_AVAILABLE_LOCALLY |
| terminal、seal、32 recovery manifests | ancestry/末端step与身份；沿用原目录不移动或重写 | 用户历史PASS，本轮NOT_RUN |
| recovery/31500/model.bin及原container descriptor | 分别计算raw SHA与descriptor SHA；strict tensor恢复 | NOT_RUN |
| Temporal四个原始文件 | 保留phase/day manifest、bin原字节、shape与parent | NOT_RUN |
| Metadata及Chain-files原JSON＋独立保存SHA | 外部信任锚；metadata不能替代chain-files | BLOCKED_NOT_AVAILABLE_LOCALLY |
| 干净f199 checkout及运行时信息 | 固定原图/编码字节；不向其中复制本阶段脚本 | 未实际检查服务器 |
| 完整恢复/原sealed predictor证据 | 区分32点字节/codec审计与真正状态恢复 | NOT_RUN |
| 公共PRE白名单packet及外部SHA | 普通/PK、死亡目标、短/长/1024边界；不含报告和Y | 目前仅人工packet |
| 实际Capsule及外部manifest SHA | 编译产物，不能用人工包替代 | NOT_RUN |
| OS访问约束、资源测量及故障campaign | 独立UID/只读挂载、PID树、RSS、冷/热延迟、取消和日志容量 | NOT_RUN/NOT_ESTABLISHED |

原来源锚（仅预期，等待材料重新核对）：

```text
source   f199a9e6c64168a412de91ff8ca04d0c795c57e2
fit      87c2485acf48443034c59430d79f098a13151677852988bf691599ca2d2f8248
terminal b977ec24d6292ad329e664c931a1aae39dc4738b4b33685b19d92fd4fb1921af
seal     fcbb81ee43050142ebdf01c0aee576156221d725598ff5ee2c02055049f72718
descriptor fdf858b5e21a1951da9ed2c2b9e988f8a1f1edcf99d2bb84f54faf055dc7e3e9
model.bin RAW SHA: UNKNOWN — 不得填入descriptor摘要
```

## 6. 用户手动服务器只读定位（本轮没有执行）

以下只查目录/读JSON/计算SHA，不导入torch，不启动服务、不写科研文件、不碰共享进程。
先在服务器执行，核对输出，**不要根据目录名称挑第一个模型**：

```bash
find /data/yuxiao/Untrusted_Network_Simulation -type f -name qwen3_final_manifest.json -print
find /data/yuxiao/Untrusted_Network_Simulation -type f \( -iname '*chain*files*.json' -o -iname '*metadata*.json' -o -iname '*final*tom*audit*.json' \) -print
```

对上一步实际找到的fit，替换下面唯一的位置参数；审计文件路径可作为后续位置参数追加。
脚本逐一验证canonical manifest摘要、32节点声明链和末端raw文件SHA；不验证codec或完整恢复。
若报错或存在多份不匹配来源，保留输出并停止，不自动修复：

```bash
python -B - '/实际查得的/final/fit/目录' '/实际查得的/metadata.json' '/实际查得的/chain-files.json' <<'PY'
import hashlib,json,sys
from pathlib import Path
def canonical(r): return json.dumps(r,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(r): return hashlib.sha256(canonical(r)).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def manifest(path,expected=None):
    raw=path.read_bytes();r=json.loads(raw)
    assert canonical(r)==raw and digest({k:v for k,v in r.items() if k!='manifest_digest'})==r['manifest_digest']
    if expected: assert r['manifest_digest']==expected
    return r
fit=Path(sys.argv[1]).resolve(strict=True)
m=manifest(fit/'qwen3_final_manifest.json','87c2485acf48443034c59430d79f098a13151677852988bf691599ca2d2f8248')
assert m['protocol_inputs']['source']['source_revision']=='f199a9e6c64168a412de91ff8ca04d0c795c57e2'
runs=fit.parent.parent/'runs'/m['manifest_digest'];previous=None
for step in list(range(1000,31001,1000))+[31500]:
    rec=manifest(runs/'recovery'/str(step)/'manifest.json')
    assert rec['optimizer_step']==step and rec['previous_checkpoint_digest']==previous
    previous=rec['manifest_digest']
t=manifest(runs/'terminal/manifest.json','b977ec24d6292ad329e664c931a1aae39dc4738b4b33685b19d92fd4fb1921af')
s=manifest(runs/'seal/manifest.json','fcbb81ee43050142ebdf01c0aee576156221d725598ff5ee2c02055049f72718')
assert t['last_recovery_digest']==previous and s['terminal_digest']==t['manifest_digest']
model=runs/'recovery/31500/model.bin';raw_sha=sha(model);descriptor=digest(rec['containers']['model'])
assert raw_sha==rec['file_table']['model.bin']['sha256']
assert descriptor==t['model_digest']=='fdf858b5e21a1951da9ed2c2b9e988f8a1f1edcf99d2bb84f54faf055dc7e3e9'
result={'fit_path':str(fit),'fit_manifest_digest':m['manifest_digest'],'fit_manifest_raw_sha256':sha(fit/'qwen3_final_manifest.json'),
        'model_bin_raw_sha256':raw_sha,'model_container_descriptor_sha256':descriptor,'terminal':t['manifest_digest'],'seal':s['manifest_digest'],
        'recovery_count':32,'codec_and_semantic_recovery':'NOT_RUN','audits':[]}
for name in ('phase_codebook.manifest.json','phase_codebook.bin','canonical_day_code_table.manifest.json','canonical_day_code_table.bin'):
    result['temporal/'+name]=sha(fit/'temporal'/name)
for name in sys.argv[2:]:
    path=Path(name).resolve(strict=True);raw=path.read_bytes();a=json.loads(raw)
    assert a['evidence_digest']==digest({k:v for k,v in a.items() if k!='evidence_digest'})
    c=a['checks'];assert c['fit']['manifest_digest']==m['manifest_digest']
    assert c['recovery']['terminal_digest']==t['manifest_digest'] and c['recovery']['seal_digest']==s['manifest_digest']
    assert c['recovery']['model_bin_raw_sha256']==raw_sha and c['recovery']['model_container_descriptor_sha256']==descriptor
    result['audits'].append({'path':str(path),'raw_sha256':sha(path),'evidence_digest':a['evidence_digest'],'mode':a['mode'],'status':a['status']})
print(json.dumps(result,ensure_ascii=False,indent=2))
PY
```

这些输出含路径和来源摘要，不含权重、标签或报告；对外公开前可去掉机器路径。
原审计JSON可能含路径/环境，不应默认公开整个文件；带外保存原SHA后私下交接。
以上辅助脚本只做只读筛查，不替代项目审计reader。完整恢复验证可能按原fit.device分配GPU/模型/优化器，不能作为无资源影响的定位命令执行。

## 7. 后续经单独批准的 CPU QUAL 命令模板（本轮未运行真实来源）

前提：受审源码已经通过认可的同步方式到达执行目录；核对固定source snapshot，不能只检查旧HEAD。
原f199目录保持干净，只用于只读reference，所有输出在全新独立QUAL目录，避开活跃采集/vLLM/predictor。
先确认CPU/RSS/磁盘与独占任务预算，再由用户手动运行；没有自动SSH或部署入口。

```bash
# Mac：保存并带外记录source snapshot摘要，不能把它称为新Git OID
python -B scripts/qualify_final_tom_capsule.py snapshot --output /实际外部审阅目录/d4-qual-source-v1

# 编译：沿用既有工具；外部parent已存在，目标全新，审计SHA为独立保存值
python -B scripts/build_final_tom_capsule.py --fit /实际原fit \
  --audit /实际外部证据目录/chain-files.json --audit-sha256 FULL_RAW_SHA \
  --output /实际部署parent/new-final-capsule

# 执行前核对源字节；跨平台mode不同会明确拒绝，不能静默放宽
python -B scripts/qualify_final_tom_capsule.py verify-source \
  --snapshot /实际source-snapshot --snapshot-digest FULL_SNAPSHOT_MANIFEST

# 原图CPU对照+独立Loader+驻留常规生命周期；不授权CAL/TEST
python -B scripts/qualify_final_tom_capsule.py cpu \
  --capsule /实际只读capsule --capsule-digest FULL_CAPSULE_MANIFEST \
  --snapshot /实际source-snapshot --snapshot-digest FULL_SNAPSHOT_MANIFEST \
  --pres /独立输入目录/public-pres.json --pres-sha256 FULL_RAW_PRE_SHA \
  --audit /独立证据目录/chain-files.json --audit-sha256 FULL_RAW_AUDIT_SHA \
  --reference-checkout /实际干净f199目录 --cpu-threads 1 \
  --output /独立输出parent/new-cpu-qual
```

成功产物为 `complete/{manifest.json,report.json,reference.json}` 与独立 `resident/`日志。
服务head、报告manifest及输入原SHA须另行带外保存。失败保留 `failure/`，不得重用/覆盖原输出。
机器报告即使为 `REAL_MODEL_LOCAL_VALIDATED`，也显式留下原sealed恢复、OS隔离、真实故障/预算和CAL资格缺口。
缺文件/错摘要/错源码/超容量/数值差异/未确认退出都返回非零或failure；没有忽略或修复选项。

## 8. 本轮验收与冻结门槛

最终相关回归 **768 passed，265.65s**，无失败/跳过，包含本阶段新增 **38项**。
运行范围与D4c-2b-2的730项相同并加入新测试，不把它解释为全仓库或服务器测试。
最后一次新增针对性运行37 passed/38.93s；之后增加的生命周期lease quota测试纳入最终768项，最终回归为当前源码证据。

受审源码快照实际位置：`/private/tmp/observer-qual-r1ce3lcs/reviewed-sources-final/`，共152个runtime源码/配置文件。
manifest：`b47e87ff47fc143a608c8fad5da28023e346dc8e01bf70ffda5ccdf7c08c549a`；
inventory：`2ca6569401315518e131c6a73bcc17286a8ef991cce9898224fe1c776eaaa064`。
公开 `verify-source` 已通过，未把base HEAD当完整D4源码身份。
早期 `reviewed-sources/` 快照及768项回归记录保留为runtime/失败证据加固前的历史，不冒称最终字节验收。
这是外部**临时目录**，不是长期归档位置；长期保存时应原样复制完整artifact并再次校验manifest，不仅保存摘要或终端输出。

最终命令、数量、耗时、来源指纹、源码快照及人工完整证据见配套acceptance JSON。
人工案例包括普通讨论、PK、死亡目标、有效1024 token边界及1028 token拒绝；无真实模型或真实游戏。
测试验证0容差能发现1e-7差异，失败先封存阶段、不开启后续resident，不提高容差或生成默认Q。
第一次目标测试2项失败来自新测试把输出放进被保护input父目录、以及用不存在的input检验后续scope；
更正fixture的独立输入/输出前提后通过，没有放宽生产路径保护。历史失败日志保留。

进入正式部署资格审批仍需：真实外部来源材料、实际Capsule、真实数值结果、完整恢复状态、
独立OS权限与资源/故障资格、稳定部署身份、正式CAL schema及Shadow接入审阅、源码固定和研究者批准。
不因本地人工通过而冻结D4配置或启动CAL/TEST；固定历史ToM/DEV与RNG来源限制继续保留。
