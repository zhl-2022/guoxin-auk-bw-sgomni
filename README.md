# AuK on BW + SGLang-Omni 0.1.9 AuK 兼容 fork（session006 / final007 开发交付）

## 1. 简介

本报告记录赛事选项 `Tencent-Hunyuan/AuK` 在超算互联网 BW 单卡上的适配及真实开发检查。模型使用固定 `tencent/AuK` base 发布权重，以 Qwen2.5-Omni-3B Thinker 提取条件，再经 DiT flow matching 与音频 VAE 生成语音；保留 NFE 32、CFG 2、sway −1，未替换为 AuK-Flash。

**真实 pilot004 已完成：三个 HTTP 接口返回 200，生成 24 kHz、单声道、PCM16、三秒 WAV，并完成对应进程和端口的清理。下载到本地的原始 ZIP、内部 manifest、源码、checkpoint 身份、HTTP、WAV、配置和清理记录均已核验。正式精度、正式性能、官方验收及赛事提交仍未完成。**

| 获取对象 | 固定来源 / 交付位置 |
|---|---|
| AuK 权重 | [tencent/AuK](https://huggingface.co/tencent/AuK/tree/790742b71a4430120daf2b2099192abae449eb9f)，revision `790742b71a4430120daf2b2099192abae449eb9f` |
| 条件编码器 | [Qwen/Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B/tree/f75b40e3da2003cdd6e1829b1f420ca70797c34e)，revision `f75b40e3da2003cdd6e1829b1f420ca70797c34e` |
| 框架 / cookbook | [SGLang-Omni AuK cookbook](https://github.com/sgl-project/sglang-omni/blob/f615735fd2a4d42daf80f2b81bd8eea4cf5a9b88/docs/cookbook/auk.md)，上游 0.1.9，commit `f615735fd2a4d42daf80f2b81bd8eea4cf5a9b88` |
| 国内固定下载映射 | `model-acquisition-lock.json`；保留 ModelScope revision、来源 URL 和与固定 HF 内容一致的身份依据 |
| 私有交付仓库 | [zhl-2022/guoxin-auk-bw-sgomni](https://github.com/zhl-2022/guoxin-auk-bw-sgomni)；备份版本以实际远端分支和commit为准，评审仍需单独获得访问权限 |
| 镜像完整 OCI 引用 | 尚未取得；已观察的镜像候选名称不能替代 registry/repository:tag |

本工程保留真实四 stage、SimpleScheduler、pipeline、Coordinator 和 HTTP 服务，身份如实描述为 **AuK 专用 SGLang-Omni 原生 Torch 兼容 fork**。该实现不提供 SRT 模型引擎；其赛事计件身份仍以主办方认定为准。

本标准目录包含下列文件。原始记录保持字节不变；`README.md` 即本报告，历史 standard004 报告另存 `historical/standard004-README.md`。

| 文件 / 目录 | 作用 |
|---|---|
| `env.sh`、`env_check.sh` | 固定模型、框架、BW、TP1、端口和官方材料变量；新运行可采集 `env.log` |
| `run.sh`、`stop.sh`、`smoke.sh` | 准备隔离依赖、启动真实服务、按明确 owner 停止、保存新 HTTP / WAV |
| `verify.py`、`bench.sh` | 正式材料接入入口；缺材料明确拒绝，不产生伪造分数 |
| `candidate/`、`source-binding.json` | 326 个冻结 candidate004 文件，逐文件大小 / SHA 身份不变 |
| `cpu-wheels/`、`cpu-wheel-lock.json` | 12 个锁定离线 CPU wheel；不包含替换 GPU 栈的发行包 |
| `tools/download_auk_models_006.py`、`model-acquisition-lock.json` | 20 个固定模型文件、共 18,748,600,288 字节的获取计划；本包不分发大权重 |
| `evidence/remote-originals.zip`、`evidence/remote-originals/` | 完整原始结果 ZIP 及全部 63 个解压成员，含失败003、成功004和原 manifest |
| `evidence/pilot004-local-audit.json`、`evidence/pilot003-failure-audit.json` | 本地独立成功 / 预期失败审计；失败003本身仍标为 `pilot_passed=false` |
| `licenses/` | 与固定 model-lock 核对的两模型 LICENSE；框架和代码许可另随 `candidate/vendor/` 保留 |
| `verify_bundle.py`、`delivery-manifest.json` | 在开始新运行前校验交付文件的精确成员集合、大小及 SHA |

## 2. 验证环境

平台界面为超算互联网华中一区 A 区、BW ×1。硬件规格平台截图随包保留在 `evidence/platform/`；截图只证明其记录时的界面状态。实际运行的三个模型 stage 配置均为 `device=cuda`，使用镜像原生 DAS Torch 后端。以下软件版本来自成功004的 `native-before.json`，并与 `native-after.json` 逐项一致。

| 组件 / 配置 | 实测版本 / 取值 | 证据 |
|---|---|---|
| Python | 3.10.12 | 原生 import004 探针及实际服务解释器 |
| Torch | `2.10.0+das.opt1.dtk2604.20260325.g6b060a` | 原生包版本、路径、METADATA SHA 前后相同 |
| Transformers | 5.5.0 | 同上；真实小 Thinker CPU forward 与 processor 另有原件 |
| vLLM / vllm-hcu | `0.18.1+das.fa71803.dtk2604` / `0.18.1+das.dtk2604` | 镜像内包；本 AuK pipeline 不通过 vLLM 模型引擎执行 |
| Triton | `3.6.0+gitc73250c4.staging` | 原生包未覆盖；本 profile 关闭 AuK fused Triton 路径 |
| NumPy / Pydantic | 1.25.0 / 2.12.5 | 原生 metadata 前后相同 |
| 精度与并行 | BF16 DiT；FP32 Thinker / VAE 权重；TP1、batch1、并发1 | 固定上游模型代码及实测 resolved config |
| AuK / Thinker 路径 | `/root/private_data/guoxin-candidates/models/auk006` / `/root/private_data/guoxin-candidates/models/qwen25-omni3b006` | 20 文件逐项身份核验 |
| 服务与映射 | `127.0.0.1:30139`；无外部服务映射 | 实际 launch config、HTTP 和服务日志 |
| IPC | `/tmp/a006-a3e59176c8456dbbefd89b2189cb4165/ipc` | 成功004的 resolved config |
| 开发镜像候选名称 | `jupyter-vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423` | 历史部署选择；完整 OCI 引用仍缺 |

额外依赖安装在独立 `--without-pip --system-site-packages` venv 中，继承原生 pip。成功004的解释器为 `/root/private_data/guoxin-auk-port006/venv002/bin/python`。10 个受保护原生包的版本、路径和 METADATA SHA 前后完全相同；该证据不等于对整个镜像文件系统作了快照。

设备执行的证据组合为：冻结源码中的 `/dev/kfd` 和 `torch.cuda.is_available()` 启动门槛、三个模型 stage 的 `cuda` 配置、原始服务日志的 `1442/1442` 权重装载、DAS HIP `cutlassfa_adapter.h` 注意力路径，以及真实音频 POST 完成。本次本地审计没有再次探测远端设备，也不补造 `hy-smi` 或独立利用率采样。WAV 检查器的 `GPU_verified=false` 保持原值，因为它只检查音频字节。

## 3. 适配优化方案

固定上游完整 904 文件曾按 Git tree blob 校验。候选包含 311 个 AuK / 通用框架原文件及 2 个适配文件，共 313 个 vendor 文件；补丁修改或新增 7 文件。AuK 模型目录的 17 文件，包括模型结构、四个 factory、采样常量、请求映射和条件编码器，保持固定上游字节。

| 改动位置 | 原因与行为 | 影响边界 |
|---|---|---|
| `platforms/__init__.py`、`platforms/torch_port006.py`、`platforms/interface.py` | 用原生 Torch 设备和 SHM 能力代替新 SRT DeviceMixin 引导；SRT engine policy 显式拒绝 | 未改 AuK 模型计算；正式误差待评测 |
| `utils/port_process006.py`、`pipeline/stage_workers.py`、`serve/launcher.py` | 小范围移植 loopback URL 与 Linux 父进程退出信号 helper | 保留上游多进程和 HTTP 逻辑 |
| `utils/audio.py` | 去掉 SRT LazyImport 依赖；没有可用 torchaudio 时走上游既有 SoundFile / SciPy fallback | 参考音频分支尚未实际大模型补测 |
| 自有 `serve.py` | 四 stage 同 OS process；关闭 compile、CUDA graph、fused QK norm / RoPE、MPS 与 weight sharing | 保留 base NFE32 / CFG2 / sway−1；正式精度未知 |
| 自有004 IPC路径修订 | 独占 `/tmp/a006-<32位owner ID>`，拒绝既有路径及软链祖先；按上游算法预检107字符预算 | 只改通信目录；修复003的134字符路径失败 |

来源锁、模型锁和兼容补丁为 `candidate/source-lock.json`、`candidate/model-lock.json` 和 `candidate/compatibility.patch`。004相对003只改变5个自有文件；313 vendor、patch、source-lock和 shared supervisor 不变。成功004前后 321 个执行相关源文件 SHA 一致，且分别与冻结004源 ZIP 对应字节一致；标准入口进一步绑定全部326个候选文件。

离线 CPU 依赖为 OmegaConf2.3.0、ANTLR runtime4.9.3、x-transformers1.31.14、SoundFile0.13.1、librosa0.11.0、soxr0.5.0.post1、audioread3.0.1、pooch1.8.2、lazy-loader0.4、scikit-learn1.6.1、threadpoolctl3.6.0、PyAV14.2.0。每个 wheel 的来源、大小和 SHA 均在 lock 中。ANTLR wheel 由固定官方 sdist 构建并保留来源记录。PyAV14.2低于框架通用依赖声明，真实 AuK 导入和文字生成路径已运行通过，不据此推广到其它模型。

标准入口仅用 `--no-index --no-deps --require-virtualenv` 安装锁定 CPU wheel。已有 venv 只读检查，缺包时拒绝覆盖修复。原生 DAS Torch / Transformers / vLLM / Triton 未被通用 GPU 发行物替换。

## 4. 服务启动脚本

真实运行目录为 `/root/private_data/guoxin-auk-port006-004`，run ID 为 `a3e59176c8456dbbefd89b2189cb4165`。外层 runner 的实际命令、解释器和返回码保存在 `evidence/remote-originals/native/auk-pilot-runner002-final.json`；对应调用为：

```bash
cd /root/private_data/guoxin-auk-port006-004
timeout -s TERM -k 80s 2400s \
  /root/private_data/guoxin-auk-port006/venv002/bin/python -B pilot.py
```

该 run 的原始配置为 load timeout900秒、service最大1800秒、stop grace45秒。`pilot.py` 先核验完整20文件checkpoint，记录source / native前照，调用真实服务，保存HTTP和WAV，在 `finally` 中停止本次owner，再保存后照。实际 `start.log`、`stop.log` SHA 已与结果JSON绑定。

shared supervisor 的 SHA256为 `0edc33da22c07a804c93047f918a4a39a58af90997390c49695d8b39a891d85b`。进程组、`/proc` start token、配置、命令和run ID共同绑定；停止必须显式提供本次ID，不按进程名称或端口批量停止其它任务。

新标准目录可先做离线文件检查和模型获取计划，再在已分配BW的原生镜像中执行：

```bash
python -B verify_bundle.py
python -B tools/download_auk_models_006.py plan --lock model-acquisition-lock.json
# 缺少固定模型文件时，在SCNet Linux执行下面的固定文件下载；已有文件仍核对SHA。
python -B tools/download_auk_models_006.py download --lock model-acquisition-lock.json
bash run.sh
bash env_check.sh
bash smoke.sh
# 使用run.sh实际打印的32位ID，不能复用历史run ID。
bash stop.sh --run-id 本次启动打印的32位ID
```

`verify_bundle.py` 校验封存时精确交付成员集合，仅排除clone后生成的 `.git/` 管理目录，须在新运行写入runtime / log之前执行。标准目录的 dry run、source / wheel identity和测试已在本地验证；**本轮真实GPU执行对象是独立candidate004，该新包装入口未在GPU重新运行**。冻结候选与离线依赖身份相同不能替代包装入口的实际复跑。

## 5. Smoke 测试

以下数值均来自下载后复核的实际原件。成功run于 `2026-10-23T16:54:34.885998Z` 结束。主路径为 `evidence/remote-originals/auk004/a3e59176c8456dbbefd89b2189cb4165/`。

| 检查 | 实测结果 |
|---|---|
| `GET /health` | HTTP200；响应原字节、headers和大小 / SHA均保留 |
| `GET /v1/models` | HTTP200；暴露 `tencent-auk-bw-port006` |
| `POST /v1/audio/speech` | HTTP200；输入 `Welcome home.`，描述 `A clear, natural voice.`，seed1234，3秒 |
| WAV | 144,044字节；24,000Hz；单声道PCM16；72,000frames；3.000秒 |
| 数值诊断 | finite_samples=true；peak=0.207122802734375；RMS=0.04004450603667967；非全零 |
| WAV SHA256 | `f021918544801641c2d552626dae4e766a1aea25fc803b180e27aabfe5b2108e` |
| checkpoint身份 | 20文件全部匹配固定revision、大小及内容hash；包括完整保守Qwen混装分片 |
| 源码 / 原生metadata | 执行相关321源文件前后一致，绑定冻结004；10原生包metadata前后一致 |
| owner进程 / 端口 | state属于本run；supervisor和server实际token均null；30139端口关闭；cleanup.verified=true |
| 退出日志中的其它事实 | 应用shutdown完成；Python resource_tracker仍警告1个leaked semaphore |

本地强审计重新读取21个成功run原件，复算HTTP与WAV SHA，重新解析音频，检查配置、模型身份、冻结源码及owner清理记录。完整原ZIP为251,876字节 /63成员，SHA256为 `0b7172198eb446bf875457c3441f089e7715f800e19153aaaf21b002c09e98ba`；其内部manifest的62项大小和SHA全部一致。

失败003也完整保留：run `95a9c6002ec36822475d889bfd3738a3` 在模型加载前因Unix socket路径134>107字符失败。其321执行源码绑定冻结003，14个原始文件保留；`pilot_passed=false`，对应进程 / 端口清理为true。该失败与成功004分别入档。

原生CPU准备证据包括真实四stage / launcher导入4/4通过，136,160参数小Thinker保存 / 重载 / forward、真实processor local_files_only通过；它们是兼容性证据，不冒充大模型GPU结果。此前import003缺PyAV、ensurepip缺失、在线pip超时、004之前socket失败均保留原日志。

WAV结构、非零与时长不证明文本读对、音色合适或达到正式WER。服务日志另有缺Mooncake / NIXL、条件编码器system prompt和API弃用警告；当前配置采用本地通信，实际文字生成完成，但尚未验证参考音频clone、编辑 / 降噪、长音频、多并发或streaming。进程和端口已清理不等于已证明所有IPC资源零残留；1个semaphore警告仍是需复测的问题。

## 6. 精度验证

目前缺少与本题绑定的官方evaluator、自检、固定任务 / 文本 / 参考、ASR解码与规范化、GPU baseline，以及“劣化不超过2%”的具体比较契约。未生成或冒充 `verify_result.json`、WER、CER、音色相似度或正式PASS。

`verify.py` 接入受控外部评测bridge，缺少 `AUK_OFFICIAL_VERIFY_SCRIPT`、`AUK_OFFICIAL_DATA_DIR`、`AUK_OFFICIAL_BASELINE` 时列明缺项并退出2。取得材料后，应保留脚本 / 输入hash、自检、原输出及正式比较契约。当前状态为 `official_accuracy_verified=false`、`official_acceptance=false`、`submission_ready=false`。

```bash
source env.sh
python -B verify.py --dry-run
```

该dry run的缺材料退出码不代表模型正式评测失败，也不代表评测已执行。

## 7. 性能测试

本轮仅有单次三秒、并发1的开发记录，无独立预热后的多样本基准。

| 时间 / 计数 | 原始值 | 含义 |
|---|---:|---|
| 启动步骤wall | 130.63249379536137秒 | 包含入口权重身份校验与启动等待；不是纯模型加载时间 |
| 单次HTTP请求wall | 8.49845259822905秒 | 客户端POST到完整音频接收 |
| `x-engine-time` | 8.479809秒 | 服务端原header；保留独立计时边界 |
| 输出latent计数 | 150 | 3秒 ×50Hz；不当作语言token吞吐 |

草稿和此前口头回传中不同的HTTP耗时 / RMS数值已由原件核对值替代；历史草稿保留，不作为本版数值依据。以上时间不代表正式T04性能、加速比或官方验收结果。

T04音频负载时长、并发、样本数、首音频 / 整段计时、RTF / 吞吐、预热及两轮口径仍缺官方契约。`bench.sh` 要求本题专属 `AUK_OFFICIAL_BENCH_SCRIPT` 和 `AUK_OFFICIAL_BENCH_OUTPUTS_JSON`，缺项时退出2。当前 `official_performance_verified=false`；未套用其它题型的4K /16K文本矩阵。

## 8. 注意事项与交付边界

冻结candidate004源包为972,888字节 /326成员，SHA256 `f087c025de08e7f764118d44f187ff7de78e9ca265539a89d353b18b95c0bdfb`；历史standard004为55,298,361字节 /352成员，SHA256 `b9cbb0f9989b59cc10bfe9b6b9686534e1f8ba8a96baf25219659feb7facb5b7`。两者保持原字节。本版另建final007目录与ZIP，其最终SHA由封包回执提供，避免把ZIP自身hash写进内部报告形成循环。

新交付保留全部候选源码、12个CPU wheel、完整成功 / 失败raw、获取锁、模型许可及精确文件manifest；不递归装入大权重、venv、cache或运行中的目录。原始ZIP中的其它模型准备记录作为原归档组成部分保留，不据此宣称其它赛题已通过。

源码中的上游Apache-2.0、AuK MIT、NVIDIA / alias-free文件声明各自保留。Qwen2.5-Omni-3B模型保留 `Qwen RESEARCH LICENSE AGREEMENT`，未改标为MIT / Apache；新编工具未选择整仓库开源重许可。离线wheel继续携带各自包内许可。

仍需闭合完整OCI镜像获取引用、评审私有仓库访问、官方精度 / 性能材料及正式评测、包装入口GPU复跑、semaphore警告复测和主办方框架身份认定。本报告是已核验的开发交付记录；最终赛事表单由参赛者手动提交，本工程未提交赛事表单。
