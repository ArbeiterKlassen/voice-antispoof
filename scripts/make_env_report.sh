#!/usr/bin/env bash
# 生成 env_report.txt —— T01 §1 环境检查的产物
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

OUT=$WS/results/env_report.txt
{
echo "==================================================================="
echo " VOICE 项目环境报告 (T01 §1)"
echo " 生成时间: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo " 主机: $(hostname)   用户: $(whoami)"
echo "==================================================================="
echo
echo "--- 1. GPU (nvidia-smi) ---"
nvidia-smi
echo
echo "--- 2. GPU 摘要 (index / 显存 / 利用率) ---"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv
echo
echo "--- 3. 当前占卡进程 ---"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
echo "(pid -> 用户/命令)"
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  echo -n "  pid $p -> "
  ps -o user=,etime=,args= -p "$p" 2>/dev/null | head -c 120
  echo
done
echo
echo "--- 4. 驱动 / CUDA ---"
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | sed 's/^/driver_version: /'
echo "CUDA Version (driver-reported): $(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+' | head -1)"
echo
echo "--- 5. Python / PyTorch (项目选用 env: ml2026_exp) ---"
$PY --version
$PY -c "import torch; print('torch          ', torch.__version__)" 2>&1
$PY -c "import torch; print('torch.version.cuda', torch.version.cuda)" 2>&1
$PY -c "import torch; print('cuda.is_available()', torch.cuda.is_available())" 2>&1
$PY -c "import torch; print('device_count   ', torch.cuda.device_count())" 2>&1
$PY -c "import torch; print('device_name[0] ', torch.cuda.get_device_name(0))" 2>&1
$PY -c "import torch; print('device_capability[0]', torch.cuda.get_device_capability(0))" 2>&1
$PY -c "import torchaudio; print('torchaudio     ', torchaudio.__version__)" 2>&1
$PY -c "import soundfile; print('soundfile      ', soundfile.__version__)" 2>&1
$PY -c "import librosa; print('librosa        ', librosa.__version__)" 2>&1
$PY -c "import numpy; print('numpy          ', numpy.__version__)" 2>&1
echo
echo "--- 6. 磁盘 ---"
df -h
echo
echo "--- 7. 本机 conda env 清点 (哪些有 torch) ---"
/usr/bin/env conda env list 2>/dev/null || ls -1 /home/cuitianxu/.miniconda3/envs
echo
echo "--- 8. 音频工具 ---"
for t in ffmpeg sox soxi lame flac; do
  printf "  %-8s: %s\n" "$t" "$(command -v $t || echo 'NOT FOUND (系统级)')"
done
echo "  imageio-ffmpeg: $($PY -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())' 2>&1 | tail -1)"
echo
echo "--- 9. 网络可达性 (关键：决定所有下载方式) ---"
for u in https://github.com https://huggingface.co https://hf-mirror.com https://pypi.org https://pypi.tuna.tsinghua.edu.cn https://www.openslr.org; do
  printf "  %-38s " "$u"
  timeout 12 curl -sS -o /dev/null -w "HTTP %{http_code}  %{time_total}s\n" "$u" 2>&1 | tail -1
done
echo
echo "--- 10. 结论 ---"
echo "  * 8x RTX 4090, 单卡 49140 MiB (约 48 GB)"
echo "  * github.com / huggingface.co 直连不通 -> 必须用 gh-proxy / hf-mirror"
echo "  * 项目目标卡 = GPU2 (唯一 0 占用)"
echo "==================================================================="
} > "$OUT" 2>&1

echo "已生成: $OUT ($(wc -l < "$OUT") 行)"
