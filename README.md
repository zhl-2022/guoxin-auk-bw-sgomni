# AuK on BW + SGLang-Omni native Torch（Session019 候选）

当前保存中文完整难例的接续代码和已验证的 CPU 评分环境。新一轮 GPU 推理尚未开始，`technical_delivery_complete=false`、`submitted=false`。此前完整英语失败结果继续保留。

## 1. 简介

目标为 [tencent/AuK](https://huggingface.co/tencent/AuK) 基础模型，固定 revision `790742b71a4430120daf2b2099192abae449eb9f`，使用发布方 Qwen2.5-Omni-3B 条件编码器。保留 NFE32、CFG2、sway=-1、FP32 DiT 权重和 BF16 autocast，不切换 AuK-Flash。

本分支准备 SeedTTS test-zh-hard 全部400条、三轮均值验证，范围在生成前固定。光合固定版本、镜像和最终报告等待实际结果。

## 2. 验证环境

目标为 SCNet 华中一区 A 区 BW64GB 单卡，DTK26.04、原生 Torch2.10.0+das。生成解释器为既有 `/root/private_data/guoxin-auk-port006/venv002/bin/python`，使用固定 SGLang-Omni 源码与既有兼容补丁。旧轮环境原件保留；新轮尚未生成独立环境实测。

CER 指标模型为 `funasr/paraformer-zh` revision `d7811ee3ac581fbcfdeb37c98c6ba674028433dc`。CPU 环境：Torch2.10.0、torchaudio2.10.0+cpu、NumPy1.26.4、FunASR1.4.16、jiwer3.0.5、zhconv1.4.3、zhon2.1.1。完整8个文件哈希核对，官方样例已实际识别；基础 Anaconda 的五个保护包元数据前后相同。

## 3. 适配方案

`publisher019/session019-auk-zhhard-queue001.zip` 固定731个输入文件，SHA256 `7531ea92033be065b6a664dc3d5006e619e59aeb0568ff5b4e4edc42bd6df263`。包含全部400个参考音频、数据身份、候选源码和控制器，不含模型权重或认证信息。

保留此前 batch=1 候选，仅给启动器增加显式 `AUK_MAX_SECONDS`：默认30，允许上限100秒。完整难例有22条按原算法估算超过30秒、最长94.6936秒，不能截短或排除。新 `serve.py` SHA256 `a7464a867538ebc9e9061dcd411874f5900fcb2ca529db973d7bc428f4051fe7`；vendor、权重、tokenizer和采样参数不改。

长输入形状尚未通过 GPU 验证。队列预先按估算时长排序，首条为最长样本；若返回音频与原算法预期时长相差超过41毫秒，保存原始返回字节并报告失败。

## 4. 启动与接续

`scripts/submission013/run_auk_zhhard_queue019.py` 为实际源码。A 的 Kimi 性能、固定入口复跑、RoboBrain RefSpatial 队列结束且自有服务完成清理后，才启动本题，不停止其他模型或关闭 Notebook。当前控制器 PID45258，进度 `/root/private_data/guoxin-session019/auk-zhhard001/progress.json`，状态为等待既有 Robo 队列。

seed1234、2345、3456 各生成全部400条，保存1,200个 WAV、请求哈希和时间；停止自身 AuK 服务后计算完整 SIM，保留10800秒自有服务看门狗。新候选尚未封装成最终根入口一键复现；旧根入口为历史版本。

## 5. 功能与评分自检

固定 Paraformer 官方音频已在 CPU 实际识别，见 `publisher019/paraformer-cpu-probe002.json`，不作为 AuK 精度。CER 通过 AST 加载固定 SeedTTS 原始 `process_one` 函数，保持其去标点及中文字符拆分逻辑。识别器不接收目标文本。

`watch_auk_cpu019.py` 仅用授权 SSH 读取完成的远端原件。它等待完整生成和 SIM、控制器退出后下载 ZIP，核对 CRC、全部文件哈希及控制器源码，再自动逐轮计算400条 CPU CER。密码只保存在进程内存。

## 6. 完整精度

旧英语全部1,088条、batch=1、seed1234：WER1.6790354884%，门槛1.0404%未通过；SIM0.7832359353，门槛0.77224通过。全部原始音频和独立复算保留；本分支保存 `publisher019/English1088-failed-accuracy-audit.json`，旧报告见 `docs/README-publisher010-historical.md`。

新增中文难例基线来自 [AuK论文Table5](https://arxiv.org/html/2609.08936v1)：CER5.91%、SIM0.782。按基线相对2%容差，CER不高于6.0282%、SIM不低于0.76636；按论文报告三轮完整均值，不额外 prompt enhancement。当前没有本模型新中文分数。

数据为 `zhaochenyang20/seed-tts-eval-arrow` revision `81d1901582dee1293a537a6d945d084301712c41` 全部400条 hardcase。参考音频映射已全部核对，没有独立证明该公开重打包与原 Google Drive 包逐字节相同。

## 7. 性能

新100秒上限配置尚未执行正式性能验证。CPU样例识别耗时不属于 AuK 生成性能。完整中文精度通过后，再补最终配置对应的 TTS 性能、环境、可复现入口及原件。

## 8. 交付状态与版权

本分支用于接续和私有备份，未最终投递。仍须实际三轮 CER/SIM、对应性能、最终报告、固定光合版本及超算镜像；只有真实成功页面或记录才更新 `submitted=true`。

新编代码保留版权说明，上游源码、checkpoint和条件编码器各自原许可不变，不自行增加全仓许可证。仓库不保存密码、Token或模型权重。
