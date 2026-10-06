#!/usr/bin/env bash
# CosyVoice2 专用 env —— 必须钉 torch==2.3.1（官方 requirements 钉死）
#
# 为什么不复用 ml2026_exp：
#   ml2026_exp 是 torch 2.11.0+cu130，与 CosyVoice2 钉的 2.3.1 冲突。
#   检测器侧（AASIST）继续用 ml2026_exp；克隆侧用本 env。两套并存，互不干扰。
#
# 网络：github/HF 直连不通 -> pip 走清华源；HF 走 hf-mirror
set -x
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

export HF_ENDPOINT=https://hf-mirror.com
CONDA=/home/cuitianxu/.miniconda3/bin/conda
PIP=/home/cuitianxu/.miniconda3/envs/voice/bin/pip
TUNA="-i https://pypi.tuna.tsinghua.edu.cn/simple"

# 0) 清掉之前跑错方向（不钉版本 -> 解析成 cu13 torch 2.14）的半成品
$CONDA env remove -n voice -y 2>/dev/null

# 1) 建 env（官方要求 python 3.10）
$CONDA create -n voice python=3.10 -y \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
  --override-channels || exit 1

# 2) torch 钉死 2.3.1（PyPI 默认轮子即 cu121）
$PIP install $TUNA torch==2.3.1 torchaudio==2.3.1 || exit 2

# 3) 音频 + 科学计算
$PIP install $TUNA soundfile librosa "numpy==1.26.4" scipy pandas scikit-learn \
  matplotlib tqdm pyyaml imageio-ffmpeg || exit 3

# 4) 自检
/home/cuitianxu/.miniconda3/envs/voice/bin/python - <<'EOF'
import torch, torchaudio, soundfile, librosa, numpy
print("torch      ", torch.__version__)
print("torch cuda ", torch.version.cuda)
print("cuda avail ", torch.cuda.is_available(), "n_gpu", torch.cuda.device_count())
print("torchaudio ", torchaudio.__version__)
print("soundfile  ", soundfile.__version__)
print("librosa    ", librosa.__version__)
print("numpy      ", numpy.__version__)
EOF
echo "COSYVOICE_ENV_DONE"
