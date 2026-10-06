#!/usr/bin/env bash
# LibriSpeech dev-clean 获取
# ⚠️ openslr.org 实测仅 15 KB/s（ETA 6 小时）→ 改用 hf-mirror（实测 241 KB/s，快 16×）
# hf-mirror 的 resolve URL 会 302 到 CDN，必须带 -L
set -e
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

D=$WS/data
mkdir -p "$D/_downloads"
cd "$D/_downloads"

URL="https://hf-mirror.com/datasets/openslr/librispeech_asr/resolve/main/clean/validation/0000.parquet"
# clean/validation = LibriSpeech dev-clean（2703 条），341 MB

if [ ! -s dev-clean.parquet ]; then
  echo "[$(date +%T)] 开始下载 dev-clean parquet (341 MB) 来自 hf-mirror..."
  curl -L --retry 3 -C - -o dev-clean.parquet "$URL"
fi
SZ=$(stat -c %s dev-clean.parquet)
echo "[$(date +%T)] 下载结束, 大小 $SZ 字节 (期望 341833540)"
if [ "$SZ" -ne 341833540 ]; then
  echo "⚠️ 大小不符，文件可能不完整，需要断点续传重跑"
  exit 1
fi
echo "DOWNLOAD_DONE"
