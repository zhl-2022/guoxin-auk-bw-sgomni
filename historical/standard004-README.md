# AuK on BW + SGLang-Omni 0.1.9 兼容 fork（session006 / candidate004 开发稿）

## 1. 简介

本目录为 `tencent/AuK` 的独立参赛工程入口，绑定固定 SGLang-Omni commit `f615735fd2a4d42daf80f2b81bd8eea4cf5a9b88` 的 AuK 四阶段与真实 pipeline / HTTP 实现。赛事下拉选项为 `Tencent-Hunyuan/AuK`。AuK 通过 Qwen2.5-Omni-3B Thinker 提取条件，使用 DiT flow matching 和音频 VAE 生成语音；本工程使用 base AuK 的 32 步 / CFG 2 / sway -1，不替换为 AuK-Flash。

当前已经完成来源锁、原生 Torch 兼容适配、小随机网络 CPU 执行、标准入口及离线 CPU 依赖准备。主代理已反馈 SCNet A 的 native Transformers5.5 小 Thinker forward / 真实 processor、四 stage、DiT / Flow、state codec 均通过；完整 launcher 的首次原生导入缺 PyAV，补上已锁 CPU PyAV14.2 后主代理反馈全部导入通过。首次完整 pilot003 在 checkpoint 加载前遇到 Unix socket 路径134字符超限，因此本目录改绑 candidate004 的短运行目录。**实际 BW 大 checkpoint HTTP / WAV 结果仍需与远端原件核对后补入；此开发稿不宣称已经完成正式精度、正式性能或验收。**

来源：

- [AuK 固定权重](https://huggingface.co/tencent/AuK/tree/790742b71a4430120daf2b2099192abae449eb9f)。
- [Qwen 条件编码器固定权重](https://huggingface.co/Qwen/Qwen2.5-Omni-3B/tree/f75b40e3da2003cdd6e1829b1f420ca70797c34e)。
- [AuK 原版代码](https://github.com/Tencent-Hunyuan/AuK/tree/370b4d922157289d33ca19dbd566c741e67146c0)。
- [SGLang-Omni AuK cookbook](https://github.com/sgl-project/sglang-omni/blob/f615735fd2a4d42daf80f2b81bd8eea4cf5a9b88/docs/cookbook/auk.md)。

镜像完整 registry/repository:tag、最终私有仓库地址和评审访问方式尚待本轮交付闭合，不填猜测值。最后的赛事表单由参赛者手动提交。

| 文件 | 作用 | 产出 |
|---|---|---|
| `env.sh` | 固定模型、框架、BW、BF16、TP1、端口和本题独立官方材料变量 | 环境变量 |
| `env_check.sh` | 元数据、实际设备探针与 hy-smi；不把镜像名称当完整引用 | `env.log`，旧原件保存在 `runtime/log-archive/` |
| `requirements-extra.txt`、`cpu-wheel-lock.json`、`cpu-wheels/` | 12 个离线 CPU 依赖及来源 / SHA | 新建隔离 venv 的安装记录 |
| `candidate/`、`source-binding.json` | 冻结 candidate004 的 326 个精确源文件 | `compatibility.patch`、逐文件 SHA |
| `run.sh` / `stop.sh` | 精确 owner supervisor，停止必须提供本次 run ID | candidate 的 runtime 配置、state、原始服务日志 |
| `smoke.sh` | 健康、模型列表、一次三秒语音 HTTP 开发检查 | `smoke.log`、`smoke-evidence/<id>/` 下的原 HTTP / WAV |
| `verify.py` | 接入外部正式 evaluator；无材料明确拒绝 | 有材料后保留外部自检、评测与原输出 |
| `bench.sh` | 接入本 T04 的外部负载与输出契约 | 有材料后保留外部性能原件 |

```bash
bash run.sh                   # 准备新隔离 venv并启动；保存输出中的32位run ID
bash env_check.sh
bash smoke.sh
source env.sh && python verify.py --dry-run
bash bench.sh --dry-run
bash stop.sh --run-id 具体启动输出中的32位ID
```

`run.sh --dry-run`、`smoke.sh --dry-run` 不安装、不开服务、不写结果。当前无官方材料，正式两项的 dry run 返回2并列缺项，不能把这个退出码写成模型失败或评分完成。

## 2. 验证环境

当前候选面向 SCNet 华中一区 A 区、BW64GB 单卡和既有 DTK26.04 / vLLM Jupyter 镜像。此前已观察 native Python3.10.12、Torch `2.10.0+das.opt1.dtk2604.20260325.g6b060a`、Transformers5.5.0；本轮最终报告必须换成当时 `env.log` 和原件中的实际值。

| 项目 | 当前配置 / 状态 |
|---|---|
| AuK checkpoint | `/root/private_data/guoxin-candidates/models/auk006` |
| 条件编码器 | `/root/private_data/guoxin-candidates/models/qwen25-omni3b006` |
| 框架 | 上游0.1.9 / 固定commit的 AuK 专用兼容 fork；不提供 SRT 模型引擎 |
| 设备 / 精度 / 并行 | BW ×1；BF16 DiT / FP32条件编码器与VAE；TP1 |
| 服务 | `127.0.0.1:30139`，无外部服务映射 |
| 隔离环境 | 首次运行仅新建 `runtime/native-venv`，`--without-pip --system-site-packages`；继承 native pip，避免缺 ensurepip |
| 完整镜像引用 | 尚缺；候选名称保存在环境变量，不作为精确 registry 引用 |
| 原生依赖保护 | 安装前后记录 Torch / Transformers / vLLM / Triton 等受保护包的版本、路径及 METADATA SHA |

已存在的 venv 只做只读检查；不自动修复、覆盖或重装。`AUK_PYTHON_BIN` 可指向本轮已验证的隔离解释器。无 `/dev/kfd` 或无真实 Torch设备时拒绝大 checkpoint 加载。

## 3. 适配优化方案

完整的 upstream 904 个文件已与固定 Git tree blob 校验。`candidate/vendor/` 保留 AuK 加通用框架，不导入其他模型实现；`candidate/source-lock.json` 记录每个源文件。AuK 的17个模型目录文件、四个 factory、采样、模型结构、权重与 tokenizer 不改。

`candidate/compatibility.patch` 共7个修改 / 新增文件：

- 原生 Torch 平台引导代替必须依赖新 SRT 的 DeviceMixin 引导；SRT engine policy 显式拒绝。
- 只替换两个小范围 OS / loopback URL helper，保留上游真实多进程与服务逻辑。
- 可选 torchaudio；没有 DAS audio wheel 时走上游既有 SoundFile / SciPy fallback。

关闭 DiT compile、CUDA graph 和 fused QK norm / RoPE，不进入本模型的专属 Triton kernel；这不是改变采样步数、结构或绕过计算。关闭 MPS、weight sharing，四 stage 同一个 OS process，选择 SHM；进程内使用上游 LocalStageDispatcher 和 `tensor_cpu` state codec，不要求 NIXL / Mooncake。

依赖均通过固定官方 PyPI 内容 SHA 下载；ANTLR4.9.3 无官方 wheel，使用精确 sdist 本地构建纯 Python wheel，记录原 sdist SHA、构建工具与产物 SHA，并在 WSL 实际 zipimport。PyAV14.2.0 来自此前固定的 LinuxCP310 CPU轮子；其是否满足本 AuK 实际路径以真实运行为准，不从通用 pyproject 较新版本的要求推断通过。

原生 Torch / Transformers / vLLM / Triton 不作为额外依赖安装。仅用 `--no-index --no-deps --require-virtualenv` 安装12个已锁 CPU wheel，不解析并替换GPU栈。依赖包的原许可、框架 Apache-2.0、AuK MIT、文件内 NVIDIA / alias-free 声明及 Qwen 模型许可各自保留；新编代码保留版权，未增加整仓库重许可。

## 4. 服务启动脚本

`run.sh` 检查冻结 candidate004 全部326文件后准备新的任务 venv，生成32位 run ID，调用真实上游服务。服务实际使用先前已测的同一 shared supervisor，源码 SHA256 `0edc33da22c07a804c93047f918a4a39a58af90997390c49695d8b39a891d85b`。

IPC目录为 `/tmp/a006-<32位run ID>`；拒绝已有目录、软链和软链祖先，最终目录独占创建。采用上游原件的socket路径预算算法；WSL Linux实际10个socket连接往返通过，最长107字符。该WSL结果不作为DAS结果。

默认 load900秒 / service1800秒 / stop grace45秒；最大允许 service3600秒。进程组与 `/proc` start token 绑定，停止还核对配置路径、命令和run ID。标准 `stop.sh` 要求明确传入启动的ID，不按端口或进程名寻找其他任务。

本 profile 为三秒生成、并发1 / batch1、NFE32 / CFG2 / sway−1、单卡 BF16 DiT。实际模型加载先逐文件复核 `candidate/model-lock.json` 中20个必需文件的大小与 SHA256 / Git blob identity。HF模型解析离线，使用真实发布权重，不执行远程 Python。

## 5. 功能测试

CPU准备已通过真实四 stage导入、真实小随机 DiT / Flow、有参考和无参考分支、相同seed逐元素复现、state tensor_cpu往返，以及小 Thinker BF16保存 / 重载 / FP32前向。主代理反馈相同探针在实际 SCNet native5.5上通过；最终应引用其远端原件，而不是把本地CPU结果写成BW结果。

`smoke.sh` 保存 `/health`、`/v1/models`、`/v1/audio/speech` 的原请求、原响应、headers、用时及WAV。仅检查24kHz、单声道、三秒容差、有限 / 非空 / 非全零样本；语义、音质、音色相似度等不由WAV结构推断。

当前实际大 checkpoint HTTP / WAV和最终停止证据仍待补入。开发 smoke 不生成或冒充 `verify_result.json`。

## 6. 精度验证

目前未获得本题正式 evaluator、固定数据与 GPU baseline。`verify.py` 复用原有受控外部评测 bridge：检查必要材料、保留外部脚本与输入哈希、先执行其自检、运行正式评测、保留原结果，结果解析保持 `executed_uninterpreted`，直到正式分题契约完成集成。

未配置 `AUK_OFFICIAL_VERIFY_SCRIPT`、`AUK_OFFICIAL_BASELINE`、`AUK_OFFICIAL_DATA_DIR` 时明确列缺项，既不启动精度评测，也不写伪造分数。正式准确率、容差、通过状态当前均未获得。

## 7. 性能测试

当前三秒 / 并发1为开发 profile，启动与单请求用时只用于记录实际过程，不能当作官方性能。正式T04音频负载、输出契约、计时边界、固定数据和 GPU baseline 尚缺。需要更长音频或其他负载时，应先按官方契约调整并重新实测profile，不把请求静默裁为三秒后当正式结果。

`bench.sh` 要求本题专属 `AUK_OFFICIAL_BENCH_SCRIPT` 和 `AUK_OFFICIAL_BENCH_OUTPUTS_JSON`；不套用T02图像6档矩阵，不继承此前其他题的 `BENCH_MODE` / `IMAGE_RES`。没有材料时只报告缺项。

## 8. 总结与已知问题

当前交付层包括完整源码、精确上游与权重锁、已审兼容补丁、12个离线CPU依赖、真实pilot、模板标准入口、原始结果保存与精确停止机制。冻结candidate004和旧session004/005成果未被修改。

仍待闭合：实际 BW 大 checkpoint HTTP / WAV与设备清理结果；当前环境原件；精确镜像获取引用；本题正式evaluator / 数据 / baseline / 性能口径；最终私有备份与评审访问方式。框架身份按模型×卡型×框架粒度如实说明为AuK专用SG-Omni兼容fork，不把它称为全部SGLang模型栈已适配。最后提交由参赛者手动操作，本工程不发送赛事表单。
