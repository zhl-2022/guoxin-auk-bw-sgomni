# AuK中文完整三轮的评分复现

本说明只评分已有音频，不重新请求AuK服务。完整正式结果为400条×三轮，种子1234/2345/3456。原完整生成、CER、SIM、失败记录和根入口100条均保留。复核存量数组使用 `python3 -B verify_bundle.py --allow-git --arrays`，不会进行神经网络评分；`plan`同样不代表新的模型实测。

## 1. 固定输入和音频位置

从本次光合固定commit解压的新项目根目录执行。镜像内旧项目目录为历史版本，不覆盖它。以下路径对应已经封装并逐SHA核验的原件布局：

```bash
PROJECT="$(pwd)"
DATA="$PROJECT/publisher031/data"
RESULTS="$PROJECT/publisher032/results/full1200"
AUDIOS="/opt/guoxin-auk032/evaluation-originals/generation"
METRICS="/root/private_data/auk-metrics037"
```

`DATA/samples.jsonl`为固定400条，SHA256为`ea61e12ac36e6c1520a7c41ed608ebcbc7471933d47f5964b1e5dde02cc98701`。`RESULTS/run1`至`run3`的`generation-requests.jsonl`提供每条音频SHA、seed和请求身份；音频分别在`AUDIOS/run1/audio`至`run3/audio`。镜像原件清单为`/opt/guoxin-auk032/evaluation-originals/MANIFEST.json`，1544个载荷含全部1200条正常WAV。单次根入口100条输出目录具有自己的`requests.jsonl`和`audio/`，不能用它替代完整split门槛。

## 2. CER：独立CPU评测环境

使用独立Python3.10或兼容环境，不向AuK服务venv安装CPU Torch。原评分器、固定模型准备器和依赖清单都在`publisher032/metrics/`；它们不会把目标文本传给Paraformer识别器。以下安装是评审需要重做评分时的完整准备命令，材料整理期间没有重复全部1200神经推理：

```bash
python3 -m venv "$METRICS/cer-venv"
CER_PY="$METRICS/cer-venv/bin/python"
"$CER_PY" -m pip install --index-url https://download.pytorch.org/whl/cpu \
  torch==2.10.0 torchaudio==2.10.0
"$CER_PY" -m pip install -r publisher032/metrics/requirements-cpu.txt

python3 -B publisher032/metrics/prepare_paraformer.py plan \
  --destination "$METRICS/paraformer-zh"
# 模型已存在时使用check；首次缺失时用download按固定8文件SHA下载。
python3 -B publisher032/metrics/prepare_paraformer.py download \
  --destination "$METRICS/paraformer-zh"
python3 -B publisher032/metrics/prepare_paraformer.py check \
  --destination "$METRICS/paraformer-zh"

for ROUND in 1 2 3; do
  "$CER_PY" -B publisher032/metrics/score_auk_zh019.py \
    --model "$METRICS/paraformer-zh" \
    --source "$PROJECT/publisher032/metrics/seedtts-run-wer.py" \
    --data "$DATA" \
    --predictions "$RESULTS/run${ROUND}/generation-requests.jsonl" \
    --audio-root "$AUDIOS/run${ROUND}" \
    --output "$METRICS/cer-new-run${ROUND}"
done
```

三个`--output`目录必须原先不存在；失败或partial目录保留，下一次另取新名称。Paraformer固定revision为`d7811ee3ac581fbcfdeb37c98c6ba674028433dc`。三个`cer-result.json`分别为每轮400条均值；三轮正式CER为三个`CER_percent`的算术平均，不挑最好一轮，不排除正常失败。现存结果为5.9704227876%、6.0776758077%、5.8247703112%，均值5.9576229689%；第二轮超单轮线的原件仍保留。

额外100条根复现使用`score_auk_root100031.py`，上述参数外加：

```bash
"$CER_PY" -B publisher032/metrics/score_auk_root100031.py \
  --model "$METRICS/paraformer-zh" \
  --source "$PROJECT/publisher032/metrics/seedtts-run-wer.py" \
  --data "$DATA" \
  --selected-samples "$PROJECT/publisher031/auk-root-quarter100-frozen-samples.jsonl" \
  --predictions /absolute/new/auk-root100/requests.jsonl \
  --audio-root /absolute/new/auk-root100 \
  --output "$METRICS/cer-new-root100"
```

该100条是额外根入口复现，不能直接与完整split发布门槛比较。

## 3. SIM：固定原评分器和两份权重

`speaker_similarity.py`保持上游源SHA `18528106c718384e6e096b79031c78fbc90890344d80ba6ae3d4cbadc0d2545e`。两个不可变权重的源revision、完整大小和SHA由`SIM-assets-READY.json`锁定。默认可以复用此前数据盘的`/root/private_data/guoxin-session010/wavlm-sim`；首次取回时使用独立目录：

```bash
ASSETS="$METRICS/wavlm-sim"
python3 -B publisher032/metrics/prepare_sim_assets037.py plan --destination "$ASSETS"
python3 -B publisher032/metrics/prepare_sim_assets037.py download --destination "$ASSETS"
python3 -B publisher032/metrics/prepare_sim_assets037.py check --destination "$ASSETS"
```

SIM原正式运行使用匹配DTK的AuK专用venv和`s3prl==0.4.18`，batch1、原scorer的`score_batch`百分数除100。重做评分前先停止自己的AuK服务并确认卡上没有另一项计时任务；不停止其他实例的服务。匹配镜像环境中执行：

```bash
source ./env.sh
SIM_PY="$PYTHON_BIN"
"$SIM_PY" -c 'import importlib.metadata as m; assert m.version("s3prl") == "0.4.18"'

"$SIM_PY" -B publisher032/metrics/score_sim_replay037.py plan \
  --audio-root "$AUDIOS" --assets "$ASSETS" --device cuda
"$SIM_PY" -B publisher032/metrics/score_sim_replay037.py check \
  --audio-root "$AUDIOS" --assets "$ASSETS" --device cuda
"$SIM_PY" -B publisher032/metrics/score_sim_replay037.py score \
  --audio-root "$AUDIOS" --assets "$ASSETS" --device cuda \
  --max-seconds 7200 --output "$METRICS/sim-new-full1200"
```

若隔离环境缺少`s3prl`，使用独立metric venv继承匹配镜像的Torch/音频依赖，再安装该固定包；不要替换原生Torch或在活动服务venv内安装：

```bash
python3 -m venv --system-site-packages "$METRICS/sim-venv"
"$METRICS/sim-venv/bin/python" -m pip install --no-deps s3prl==0.4.18
SIM_PY="$METRICS/sim-venv/bin/python"
```

如果任何导入、固定SHA或完整输入检查失败，应保留错误并补齐环境，不使用随机权重或跳过样本。该准备分支未在这次材料整理中重新全链路神经推理；本件正式1200条SIM已经完整真实执行并独立审核。

新CLI显式传入目录，保留原scorer、batch1和缩放方式，拒绝非有限分数、缺失/重复样本、错误音频SHA和形状；输出新目录，逐行保存1200条。它不复用旧`SIM-controller.py`中的历史hostname/PID/路径门禁，不请求生成音频。`plan`仅验证固定表与请求身份、不读取WAV/权重、不导入Torch；`check`只做CPU字节/形状检查；`score`才真实执行神经评分。可选`--device cpu`是独立设备路径，不能把未实测CPU得分或耗时当作原GPU正式结果。

原1200条`sim.jsonl`和三轮CER位于`RESULTS/`；原SIM均值0.7804689528，对照发布方0.782、相对2%下限0.76636。全部原件和旧失败保留。任何新评分均保留其设备、版本、脚本SHA、源音频SHA和独立输出，不覆盖原成绩或声称新增官方认可。
