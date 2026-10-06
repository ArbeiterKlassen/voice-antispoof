#!/usr/bin/env bash
# 建立 voice env（检测器 + 音频处理栈）
# 网络现实：github/HF 直连不通 → pip 用清华源；HF 用 hf-mirror
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

# 1) 建 env（用清华 anaconda 镜像，避免 repo.anaconda.com 慢）
$CONDA create -n voice python=3.10 -y \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
  --override-channels || exit 1

# 2) torch 栈（PyPI 的 torch 自带 CUDA 12.x，驱动 13.0 向后兼容）
$PIP install $TUNA torch torchaudio || exit 2

# 3) 音频 + 科学计算
$PIP install $TUNA soundfile librosa numpy scipy pandas scikit-learn \
  matplotlib tqdm pyyaml || exit 3

# 4) ffmpeg 静态二进制（编解码攻击必需；系统无 ffmpeg）
$PIP install $TUNA imageio-ffmpeg || exit 4

# 5) 自检
/home/cuitianxu/.miniconda3/envs/voice/bin/python - <<'EOF'
import torch, torchaudio, soundfile, librosa, numpy
print("torch      ", torch.__version__)
print("torchaudio ", torchaudio.__version__)
print("cuda avail ", torch.cuda.is_available(), "n_gpu", torch.cuda.device_count())
print("soundfile  ", soundfile.__version__)
print("librosa    ", librosa.__version__)
import imageio_ffmpeg
print("ffmpeg     ", imageio_ffmpeg.get_ffmpeg_exe())
EOF
echo "SETUP_DONE"
