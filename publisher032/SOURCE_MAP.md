# 本轮数据与证据范围

- results/full1200/run1至run3：每轮全400生成请求、CPU CER原始转录、评分器协议与结果。
- results/full1200/sim.jsonl：全1200原始SIM；SIM-controller.py和generation-controller.py为实际执行源码，源字节SHA不变。原SIM环境错误进度保留。
- results/full1200/complete-originals-manifest.json：全生成/SIM原件清单；完整698848643字节ZIP另行保留，SHA4f1d9edbf03a58e88c77479363833e2245e512476df780e0d5e8f8424e8bf0a2。镜像另含全部生成原件供复核，未把699MB单文件推入Git。
- originals/session030-auk-performance-A001.zip：同配置完整96测量+6预热全部原件，SHA804eba6f981598eee09f7fbf7a3dd5d0b4b7f9796fa6e7946fa07b1fe8fd680f。
- results/root100：冻结额外100条的全部请求、WAV、独立CPU CER、原始转录和协议。原完整根归档SHA2d19a0fdebe39f4368991438c9cd3d8db79f5f49b14f476b552e4985e805fa77。
- root031-frozen-project-lock.json：实际GPU根Git5446333abde67a6169e258e7a6d07b59a3655b7c的1251文件锁；最终新增环境准备与材料封装不改candidate327文件。
- metrics：原发布方CER规范化、完整400及固定100 CPU评分器、固定Paraformer8文件锁/准备器、实际SIM源码与固定2权重来源锁。CPU评分在独立环境执行，不能装入或改动服务的原生依赖。
- expected-native-and-venv-packages.json、auk-native-helper-check.json：实际原生/venv22包指纹及20模型文件重算SHA复用通过；未模拟新Notebook，不虚构整模型重新下载或新建venv安装已经实测。
