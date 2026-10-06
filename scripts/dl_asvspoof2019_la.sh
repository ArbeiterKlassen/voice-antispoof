#!/usr/bin/env bash
# ASVspoof2019 LA（镜像打包版，含 train/dev/eval 三划分 + labels）
# 用途：**与文献可比的基准轨道**（AASIST 论文 0.83% EER 即此集）
# 纪律：镜像非官方发布，报告时标注「非官方镜像」；但划分与协议文件来自该包，不自造
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

OUT=$WS/data/_downloads/asvspoof2019_la
BASE="https://hf-mirror.com/datasets/Bisher/ASVspoof_2019_LA/resolve/main"
mkdir -p $OUT/data
cd $OUT
unset https_proxy http_proxy all_proxy HTTPS_PROXY HTTP_PROXY ALL_PROXY
for f in train validation test; do
  N=$(ls data/${f}-*.parquet 2>/dev/null | wc -l)
  if [ "$N" -gt 0 ] && python3 -c "import pyarrow.parquet as pq; pq.ParquetFile('$(ls data/${f}-*.parquet|head -1)')" 2>/dev/null; then
    echo "[$(date +%T)] $f 已完好，跳过"; continue
  fi
  echo "[$(date +%T)] 下载 $f"
  for i in $(seq 1 40); do
    curl -L -C - --retry 5 --retry-delay 5 --connect-timeout 30 --max-time 3600 \
         -o data/${f}-00000-of-00001.parquet "$BASE/data/${f}-00000-of-00001.parquet" && break
    echo "  第 $i 次中断，已 $(du -m data/${f}*.parquet 2>/dev/null|cut -f1) MB"
    sleep 15
  done
done
echo; echo "=== 完整性检验（逐划分读 parquet）==="
python3 - <<'PY'
import glob, pyarrow.parquet as pq
import os
for f in sorted(glob.glob(os.environ["VOICE_WS"]+"/data/_downloads/asvspoof2019_la/data/*.parquet")):
    try:
        pf=pq.ParquetFile(f)
        n=pf.metadata.num_rows
        print(f"  ✅ {os.path.basename(f):<40} {n:>7} 行  {os.path.getsize(f)/2**20:8.1f} MB")
        print(f"      列: {pf.schema_arrow.names}")
    except Exception as e:
        print(f"  ❌ {os.path.basename(f)}: {e}")
PY
echo "LA_DL_DONE"
