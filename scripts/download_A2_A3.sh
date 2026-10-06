#!/usr/bin/env bash
# 任务 02 A 级数据下载：A2 = LibriSpeech test-clean, A3 = train-clean-100
# 来源 hf-mirror（实测 ~241 KB/s；openslr 仅 15 KB/s，按任务 02 §0 只作后台限速通道）
# 走 parquet（HF openslr/librispeech_asr），比 tar.gz 分片可控、可断点续传
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

BASE="https://hf-mirror.com/datasets/openslr/librispeech_asr/resolve/main"
D=$WS/data/_downloads
mkdir -p "$D"

dl () {  # $1=远端相对路径  $2=本地名
  local url="$BASE/$1" out="$D/$2"
  if [ -s "$out" ]; then echo "[skip] $2 已存在 ($(stat -c %s "$out") B)"; return 0; fi
  echo "[$(date +%T)] 下载 $1"
  curl -L --retry 3 -C - -o "$out" "$url" || { echo "  ⚠️ 失败 $1"; return 1; }
  echo "[$(date +%T)] 完成 $2  $(stat -c %s "$out") B"
}

echo "===== A2: LibriSpeech test-clean ====="
dl "clean/test/0000.parquet" "test-clean.parquet"

echo
echo "===== A3: LibriSpeech train-clean-100 (14 分片, ~6.3 GB) ====="
for i in $(seq -w 0 13); do
  dl "clean/train.100/00$i.parquet" "train-clean-100-$i.parquet" || echo "  (继续下一个)"
done

echo
echo "===== 下载汇总 ====="
ls -la "$D"/*.parquet
echo "A2A3_DOWNLOAD_DONE"
