# AuK BW语音模型适配技术报告

本件对应T04语音模型，提交人张汇浏。正式数值比较采用完整中文hardcase400条×三轮；额外固定根入口按授权使用100/400。正式准确率、性能、根入口复现、镜像、评审访问、主办方验收与实际表单提交分别记录；未出现提交成功记录时submitted=false。

## 1. 模型简介

模型为tencent/AuK原base检查点，固定revision `790742b71a4430120daf2b2099192abae449eb9f`；条件编码器Qwen/Qwen2.5-Omni-3B固定revision `f75b40e3da2003cdd6e1829b1f420ca70797c34e`。使用SG-Omni `f615735fd2a4d42daf80f2b81bd8eea4cf5a9b88`的AuK-only原生Torch BW兼容端口006。没有把这一受限子集写成完整SG-Omni所有SRT模型引擎已适配。

原模型与编码器20个保守快照文件、上游许可和字节身份见candidate/model-lock.json与model-acquisition-lock.json。所有实际推理的模型文件已逐个重算SHA或原Git blob身份，不替换其他检查点。最终新增数据盘模型/venv准备和交付封装，candidate的327个源码文件保持根入口与完整评测实际字节。

### 最终提交材料绑定（Session037）

原有全部资料保留；本次整理不重新采样、不改变模型推理或原判分源码；修正顶层验证入口并补真实准备链路验收。实际发布版本和评审获取状态由 `CURRENT_SUBMISSION_STATE037.json` 及外部提交清单记录；尚未完成的门禁不写成通过。

| 项目 | 固定引用 |
| --- | --- |
| 光合代码项目 | [项目入口](https://developer.sourcefind.cn/codes/ghu_30ebea88e/guoxin-auk-bw-sgomni-final-session037) |
| SCNet镜像 | [实际镜像详情](https://www.scnet.cn/ui/console/index.html?#/image-management/detail/2115317990002249729) |
| 固定镜像tag | `dtk2604-auk-base-final-20261028` |
| 已实测前版代码 | `4abf67046a0377392be781ad0cb92c75a1a339fd` |
| 本次材料分支 | `codex/submission-ready037-final`，以外部清单的完整commit为准 |

本件实际为SG-Omni的AuK-only原生Torch兼容路径，未将它写成vLLM/SRT引擎已适配。模型×框架及中文CER/SIM协议的赛事适用性、最终环境准备路径和评审获取方式，仍按外部清单如实记录。


## 2. 环境配置

超算华中一区A区，BW64GB×1；Torch实际设备探针为BW、count1、HIP6.3.26093。发行元数据为Torch2.10.0+das.opt1.dtk2604、TorchVision0.25.0+das、Transformers5.5.0、NumPy1.25.0，运行时torch.__version__显示2.10.0。vLLM0.18.1随基础环境存在，本模型的AuK-only路径不以vLLM/SRT为模型引擎。22个实际包版本和METADATA SHA在publisher032/expected-native-and-venv-packages.json，完整设备输出在env.log。

原生参数：NFE32、CFG2.0、sway=-1、FP32权重与BF16autocast、batch1、单模型进程/单设备。显式输出上限100秒用于保留400条中的22个超30秒样本；最长样本实际约94.70秒，不截短或排除。原默认30秒不适用于本轮数据，修改范围和原因已披露。服务最大10800秒、加载1200秒、停止宽限45秒均保持实际配置，不更改时钟/看门狗或平台限制。

## 3. 依赖、模型和数据准备

DTK变量只在项目Shell子进程设置。服务解释器固定 `/root/private_data/guoxin-auk-port006/venv002/bin/python`，与完整1200、96性能和新根100配置的命令完全一致。run.sh真实启动前调用scripts/prepare_runtime032.py ensure：优先复用数据盘20个固定模型文件和已验证venv，只有缺失时才按不可变官方源、大小和SHA准备同一字节；发现现有文件或环境不同则保留并报错。离线12个CPU依赖wheel只装进专用venv，保护原生Torch/vLLM/Transformers等包；不把权重复制回系统盘。

CPU计划已检查20个下载映射与模型锁完全一致、三类路径/来源拒绝、缺失计划不落盘。最终在A实际保留原环境及首次失败环境后，从缺失的固定venv路径重新创建、离线安装12个CPU依赖wheel并保护原生包；20个模型文件全SHA复用、没有重新完整下载。22包版本指纹检查仅对ANTLR4.9.3已观察到的两种METADATA封装差异做严格兼容：原/新metadata SHA固定，56个实际antlr4运行文件逐个SHA一致，其他包仍精确匹配。详细原件见publisher037/final-preparation-replay；这是真实全新venv创建与后续功能验收，不声称另建Notebook或从新镜像启动了另一实例。

数据源为zhaochenyang20/seed-tts-eval-arrow revision `81d1901582dee1293a537a6d945d084301712c41`，固定中文hardcase400条，来源parquet SHA `5d73be79cf2e84fd0bebf5d47212438a88dd264a591031d092164ba84545f1de`。完整400参考音频与samples.jsonl随publisher031/data提供，样本表SHA `ea61e12ac36e6c1520a7c41ed608ebcbc7471933d47f5964b1e5dde02cc98701`，选择在任何预测前固定。

## 4. 运行与根入口复现

```bash
python3 -B verify_bundle.py --allow-git --arrays
python3 -B scripts/prepare_runtime032.py plan
bash run.sh --run-id YOUR_UNIQUE_32_HEX_RUN_ID
bash env_check.sh
bash smoke.sh
python3 -B verify.py --selfcheck
python3 -B verify.py --plan
python3 -B verify.py --run-id YOUR_UNIQUE_32_HEX_RUN_ID --output /absolute/new/auk-root100
bash bench.sh --plan
bash stop.sh --run-id YOUR_UNIQUE_32_HEX_RUN_ID
```

每次使用独立UUID和新输出目录；只停止自己的确切run-id。bench.sh实际执行同一96请求计划，但本次已有完整性能，不重复运行。verify默认为预先固定100条额外根复现，实时写入请求/WAV/进度；CPU CER评分独立执行，不能把WAV成功当作准确率通过。镜像项目目标路径 `/opt/guoxin-auk032/project`，完整原始生成/SIM另存 `/opt/guoxin-auk032/evaluation-originals`。代码、完整参考数据、CPU依赖wheel和固定评测工具随镜像；AuK与Qwen权重保留数据盘，不违反15GB新增层门禁。

CPU CER在独立评测环境运行，不能安装到原生服务venv。固定Paraformer-zh revision `d7811ee3ac581fbcfdeb37c98c6ba674028433dc`，模型8文件锁/校验准备器在publisher032/metrics/prepare_paraformer.py；关键CPU版本见同目录requirements-cpu.txt。完整400和固定100评分器均保留原发布方SeedTTS规范化函数，源SHA `be5e608e3fa1d99ddeb92853dd9e768348633702b51c6a607ed557dcf7bbb2ea`。示例先用准备器check或download，再用score_auk_root100031.py的--model、--source、--data、--selected-samples、--predictions、--audio-root和独占--output参数指向对应文件；目标文本不传给ASR识别器。


完整可复制的CPU CER准备、400×3评分、额外100条评分、SIM固定权重获取/校验和原scorer三轮评分命令见[评分复现说明](publisher037/SCORING-REPLAY.md)。新增`score_sim_replay037.py`只移除历史控制器的硬编码主机/PID/路径，显式接收目录，保持原scorer、batch1和百分数除100；原固定scorer源码不改。本机CPU已实际执行两个新CLI的plan并核对1200个固定请求身份；没有把计划当作Linux完整评分、GPU或新一次SIM神经推理。历史控制器只作为原件保留，不可直接恢复运行。

T04正文规定TTS使用公示任务指标（含ASR回测），AuK具体附录采用匹配平台Torch部署；本件中文hardcase按发布方中文CER/SIM三轮平均协议记录，并提供中文ASR回测与相同固定scorer。比赛总计件仍写模型×卡型×SGLang/vLLM，本报告明确SG-Omni AuK-only实际后端是原生Torch，不冒称完整SRT或vLLM。最终登记时须与实际模型/框架选项一致；没有取得的官方例外不写成已经认可。

## 5. Smoke及固定版本验证

固定根Git `5446333abde67a6169e258e7a6d07b59a3655b7c`，1251个精确Git文件，从独立ZIP提取后在A于2026-10-28 04:47:37至05:05:10UTC执行。selfcheck、bench-plan、dry-run、launch、environment、smoke、verify、stop八阶段均实际退出0，服务health200，最后自有进程清理和端口关闭。模型20文件在启动前全身份核验，服务config与完整1200逐键比较仅task/UUID不同。Smoke是三秒功能WAV，不作为正式准确率。

额外根入口100条全部完成；全部输入、参考音频、重建请求SHA、原始WAV/长度、1251源码/Git blob和闭合归档所有CRC/载荷SHA独立核验。新100条CPU CER也逐样本从原始ASR转录用作者函数重新计算，零排除。原根完整归档SHA `2d19a0fdebe39f4368991438c9cd3d8db79f5f49b14f476b552e4985e805fa77`，精确源码与响应审计见publisher032。


### 最终缺失环境准备与顶层入口验收（Session037）

16:19至16:21UTC，固定原路径不存在时实际新建venv、离线安装并验证22包，保护原生包，20模型全SHA复用；之后顶层`verify.py`仍误调旧通用入口，报`missing OFFICIAL_VERIFY_SCRIPT`，在任何模型启动前失败。完整错误、原脚本与新环境均保留。本版把模板要求的顶层`verify.py`接到此前已通过的`entry031`固定根验证器，原327核心文件不变，旧verify保存在provenance。

16:35:46至16:38:55UTC，在又一个独立精确源码目录复用刚创建的新venv，依次完成精确源码/旧数组验证、环境全指纹回读、顶层selfcheck、根启动、与完整1200实际配置逐键比对、环境检查、三秒功能WAV和自有stop，八阶段全0。实际配置仅task目录/UUID不同；health/models/speech均HTTP200，WAV24kHz单声道、非全零；A本机另行核对精确自有PID全部退出、七模型端口关闭。原始8日志、实际config/final state、功能音频与新建环境的前一阶段原件见`publisher037/final-preparation-replay/INDEX.json`。没有重复完整精度、96性能或根100生成，不用功能WAV代替精度。

镜像保持原固定ID/tag。使用本次光合源码和附件版本，在独立新目录运行；镜像内部旧project/README仍保留历史时点，不冒充当前037代码。不重复保存镜像，不把权重搬回系统盘。完整缺失模型下载分支只提供固定锁和实际准备代码，未宣称重新下载全部检查点。

## 6. 精度测评

正式比较只采用完整中文400×三轮、种子1234/2345/3456，共1200条。发布方CER5.91%、SIM0.782，采用较严格相对2%容差：CER≤6.0282%、SIM≥0.76636。全部1200正常音频、超时/失败记录、请求、SIM及CER原始转录均保留，不挑轮次。

| 轮次 | 种子 | 完整样本 | CER百分比 | SIM |
| --- | --- | --- | --- | --- |
| 1 | 1234 | 400 | 5.9704227876 | 0.7782538089 |
| 2 | 2345 | 400 | 6.0776758077 | 0.7795175091 |
| 3 | 3456 | 400 | 5.8247703112 | 0.7836355405 |
| 三轮均值 | 全三轮 | 1200总音频 | 5.9576229689 | 0.7804689528 |

均值两指标均达数值比较门槛。第二轮CER高于单个门槛仍保留，遵守发布方三轮平均协议。CPU Paraformer只读实际AuK音频，不接收目标文本；SIM采用同一固定WavLM源文件SHA `18528106c718384e6e096b79031c78fbc90890344d80ba6ae3d4cbadc0d2545e`及两个固定权重，来源锁保留在metrics/SIM-assets-READY.json。完整1200的音频/请求/参考SHA、所有CER公式、SIM对应关系与均值独立复核；没有声称另一次独立SIM神经推理。

**为何减少根复现数据量：**完整400×3和96性能已经完成，为缩短额外验证时间，按用户授权采用预先固定1/4，即100/400。按估计时长四层各25、确定性SHA排序，并覆盖最长样本；清单SHA `d056bd4a0b0402ec7fa98887af0d303cda1d6cda7548e8017d31e58f6cc01a57`在新请求前冻结，非按正确率选样。现有完整1200已满足本次正式数值比较，故不重复全量生成。

额外100条CER为7.5285996930%；原完整第一轮中**同一100条**为7.4844415803%，变化+0.0441581127个百分点（相对+0.589999%）。该时长分层子集不直接与发布方全split基线比较，不写成全量不达标或全量通过。100份波形中51份SHA完全相同，92条ASR原始转录相同、94条逐样本CER相同；不宣称逐字节确定性，没有重试选最好结果。完整1200正式成绩保持独立。

历史英语完整1088、batch1轮WER1.6790354884%高于1.0404%上限，先前三轮均值1.7098984373%也未通过，继续披露；不改写为通过。中文原控制器SIM阶段缺少libgalaxyhip.so.5是库路径只传给模型子进程导致的基础设施错误，保留原进度，只接续评分，全部1200正常生成没有重做。

## 7. 性能参考

使用同一原生Torch AuK服务配置。事先固定完整400中估计输出1至15秒、ID排序的前16条，客户端并发1/2/4各两轮，96测量请求；每组另有1条预热，合计6条，原件全部保留。模型端batch1，较高客户端并发主要形成排队，不能写成4卡或模型batch4。完整WAV响应，首body字节是缓冲输出的响应边界，不是流式语音TTFT，也不冒充SGLang文本六档压测。

| 轮次 | 客户端并发 | 测量请求 | 组耗时秒 | RTFX | 平均RTF | 平均首body字节秒 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 16 | 83.239309 | 2.015154 | 0.515297 | 5.163950 |
| 1 | 2 | 16 | 77.298146 | 2.170039 | 0.975851 | 9.419478 |
| 1 | 4 | 16 | 77.230525 | 2.171939 | 1.884572 | 17.729888 |
| 2 | 1 | 16 | 82.212357 | 2.040326 | 0.511339 | 5.124860 |
| 2 | 2 | 16 | 77.192604 | 2.173006 | 0.972131 | 9.394718 |
| 2 | 4 | 16 | 77.165122 | 2.173780 | 1.925274 | 17.727928 |

全部102份WAV与重建请求SHA、六组统计公式和服务实际config独立核验，原完整归档SHA `804eba6f981598eee09f7fbf7a3dd5d0b4b7f9796fa6e7946fa07b1fe8fd680f`。两轮摘要在bench/round1.jsonl、round2.jsonl，完整原件在publisher032/originals。根入口只检查相同bench计划，不重采样已完成96条。这些是实际配置性能记录，不能替代主办方性能验收。

## 8. 注意事项与交付范围

- 最终固定Git/光合链接、报告/ZIP SHA、镜像ID/tag和真实目标区域可用回执在外部清单闭合；仓库README与八章报告原字节一致。光合项目名称为guoxin-auk-bw-sgomni-final-session037；环境镜像名称guoxin-auk-bw-sgomni032、tag dtk2604-auk-base-final-20261028。
- 模型、代码、环境和数据身份分别校验；最终封装新增数据盘模型/venv准备与结果打包，推理核心327文件、模型权重、采样和评分器字节不变。当前venv复用已实测，缺失时的固定准备计划已验证，未另建新Notebook进行最终环境镜像GPU整套复跑，跨账号评审访问也未独立验证。
- 完整1200原始生成/SIM清单和结果与额外根100全部响应保留；镜像另提供原始生成/SIM，检查点本体不放新系统层。上游版权/许可和历史失败保留，不添加全仓开源许可证或公开私有资产。
- 两台原Notebook按授权持续保留，仅清理自有模型服务；不为表单启动GPU，不充值购买，不修改或绕过平台限制，不干预其他实例。八卡监测已取消。
- 表单关闭时不提交，不改投报名表；只有真实成功页面或提交记录才submitted=true。技术材料齐备和数值比较达标不等同于主办方验收。
