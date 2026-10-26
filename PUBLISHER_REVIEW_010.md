# AuK on BW + SGLang-Omni：Session010 完整评测

## 1. 简介

完整三轮生成和 WER/SIM 回测已完成，数值未达标，不作为合格投递项目。三轮平均 WER 为 **1.709898437299%**，上限 1.0404%；SIM 为 **0.784561590582**，下限 0.77224。SIM 通过，WER 未通过；官方验收与赛事提交均未发生。

## 2. 验证环境

两张独立 BW 64 GB 卡并行，华中一区 A 区、DTK 26.04、15 CPU / 59 GB 内存。原生 PyTorch fallback，关闭 NVIDIA 专用融合、CUDA Graph 和编译路径。模型 `tencent/AuK` revision `790742b71a4430120daf2b2099192abae449eb9f`；框架 `sgl-project/sglang-omni` revision `f615735fd2a4d42daf80f2b81bd8eea4cf5a9b88`。

## 3. 适配优化方案

完整 voice cloning 支持参考音频、参考文本和目标文本；保留原版 UTF-8 文本比例估算时长、32 Euler steps、CFG 2、sway -1。DiT 参数以 FP32 驻留、BF16 autocast；批量与并发均为 4，最大时长上限 30 秒，实际固定数据均不到 10 秒。

修复 x-transformers 1.31.14 仅接受一维位置序列的问题：二维批次逐行调用未改动的旧实现，再堆叠频率表。CPU / CUDA、batch 1 / 4 的频率表及 BF16 Q/K 结果均逐位相等；保留模型权重和采样语义。实际 8 条参考音频小批量全部通过。

## 4. 服务启动脚本

`publisher010/candidate` 是实际执行的完整适配源码；`tools` 是固定三轮生成、评分和有截止时间的 owned process 控制代码。`adaptation-lock010.json` 记录当前文件哈希和补丁前后哈希。

本轮已执行脚本采用账户持久盘 `/root/private_data/guoxin-session010` 及已有 AuK / Qwen2.5-Omni / Whisper / WavLM 模型路径。它是本次完整验证的保存件，不把这些路径声称为已经封装进独立交付镜像。历史 session007 默认入口保持原样，新配置与原件位于 `publisher010/`。

## 5. Smoke 测试

真实 8 输入并发 pilot 全部成功。正式 3 × 1088 条请求全部成功，返回 24 kHz 单声道 PCM16 WAV，非静音且时长与原版估算一致。三组种子的每一对都有 1088 条不同的音频哈希；没有重复同一轮充数。

## 6. 精度测评

完整 SeedTTS 英文 Arrow 分集实际为 1088 条，采用框架公开推荐的数据副本，revision `81d1901582dee1293a537a6d945d084301712c41`；不截断到 README 中约数 1000。预先固定种子 1234 / 2345 / 3456。

Whisper-large-v3 的原始未过滤逐句平均 WER；参考 revision `06f233fe06e710322aca913c1bc4249a0d71fce1`，FP32、greedy、English/transcribe。SIM 采用固定框架的 WavLM speaker verification 实现及其推荐检查点，模型 SHA、s3prl 0.4.18 来源均保留。SIM 的 GPU forward 未在本地再次执行，本地已核对所有原始相似度、音频身份和汇总。

| 轮次 | 种子 | 样本 | WER | SIM | 聚合 RTF |
| --- | --- | --- | --- | --- | --- |
| 1 | 1234 | 1088 | 1.724520538% | 0.783343209 | 0.752006 |
| 2 | 2345 | 1088 | 1.637868181% | 0.782417044 | 0.701403 |
| 3 | 3456 | 1088 | 1.767306592% | 0.787924518 | 0.742978 |

英文发布基线是 WER 1.02% / SIM 0.788，不使用跨语言平均 WER 2.65 冒充英文基线。本地独立审计全部 3264 个请求、WAV、WER 行和 SIM 行，WER 用独立 NLTK 编辑距离复算，保留高错误率样本，`numeric_comparison_pass=false`。

## 7. 性能参考

表中 RTF 为该轮生成墙钟时间 / 4558.38 秒输出音频，含真实 HTTP 请求和排队。各轮 p50/p95 请求延迟及完整原始记录在审计 JSON 中；不把非流式整段 WAV 的返回时间称为增量音频首包延迟。

## 8. 注意事项

- 这是已完成的工程验证和未达标记录，不能登记为精度验收通过。
- 完整代码、模型来源、评分记录和失败批次诊断保存在 `publisher010/evidence/complete-review.zip`。
- 完整音频原件已保存在本地 `dist/session010-auk-generation003-A-results.zip`、`...-B-results.zip` 与超算持久盘；大文件哈希和范围在归档回执中，未把大型权重上传到代码仓库。
- 本轮 GPU 任务和平台实例均已停止。私有仓库备份不等于赛事表单提交。
- 上游版权与许可证保留，新编代码继续保留版权说明。
