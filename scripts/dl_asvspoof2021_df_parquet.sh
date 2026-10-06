#!/usr/bin/env bash
# ASVspoof2021 DF（SpeechAntiSpoofingBenchmarks 打包版）下载
#
# 【为什么用这一版而不是原始 zip】
#   1. 原版 DF flac 有 ~40% 是 libsndfile/soundfile 读不了的
#      （"flac decoder lost sync"），而本机管线正是用 soundfile；
#      这一版作者已逐条解码重编码，且**声称 PCM 逐位不变**。
#   2. 这一版自带 data/labels.parquet 与 notes 字段（codec/attack_id/vocoder），
#      不必再等官方 keys。
#
# 规模：80 分片 + labels，合计约 32 GiB。按 3-4 MB/s 约 2.5-3 小时。
# 校验：每个分片能打开；行数合计 = 611829；labels 分布 = 22617/589212。
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

OUT=$WS/data/_downloads/asvspoof2021_df
BASE="https://hf-mirror.com/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF/resolve/main"
mkdir -p $OUT/data
cd $OUT
unset https_proxy http_proxy all_proxy HTTPS_PROXY HTTP_PROXY ALL_PROXY

echo "[$(date +%T)] 下载 labels.parquet"
curl -sL --retry 5 --retry-delay 5 -o data/labels.parquet "$BASE/data/labels.parquet"

echo "[$(date +%T)] 下载 80 个分片"
for i in $(seq -w 0 79); do
  f="data/test-000${i}-of-00080.parquet"
  # 已完好则跳过（本脚本可反复跑，断点按分片粒度续）
  if [ -f "$f" ] && python3 -c "
import pyarrow.parquet as pq,sys
try: pq.ParquetFile('$f'); sys.exit(0)
except Exception: sys.exit(1)" 2>/dev/null; then
    echo "  [$i/79] 已完好，跳过"; continue
  fi
  for k in $(seq 1 20); do
    curl -L -C - --retry 3 --retry-delay 5 --connect-timeout 30 --max-time 1800 \
         -o "$f" "$BASE/$f" && break
    echo "  [$i/79] 第 $k 次中断，15s 后续传（已 $(du -m "$f" 2>/dev/null|cut -f1) MB）"
    sleep 15
  done
  printf "  [%s/79] %s  %s MB\n" "$i" "$(basename $f)" "$(du -m "$f" 2>/dev/null|cut -f1)"
done

echo; echo "[$(date +%T)] 完整性检验"
python3 - <<'PY'
import glob, os, sys
import pyarrow.parquet as pq
import pandas as pd
D=os.environ["VOICE_WS"]+"/data/_downloads/asvspoof2021_df/data"
shards=sorted(glob.glob(f"{D}/test-*.parquet"))
rows=0; bad=[]
for p in shards:
    try: rows += pq.ParquetFile(p).metadata.num_rows
    except Exception as e: bad.append((os.path.basename(p), str(e)[:60]))
print(f"  分片 {len(shards)}/80  行数合计 {rows}（期望 611829）")
if bad:
    print("  ❌ 坏分片:"); [print("    ", b) for b in bad]; sys.exit(1)
if len(shards)!=80 or rows!=611829:
    print("  ❌ 不完整"); sys.exit(1)
L=pd.read_parquet(f"{D}/labels.parquet")
vc=dict(L.label.value_counts())
print(f"  labels 分布 {vc}（期望 1:589212 / 0:22617）")
print("  ✅ 完整" if (vc.get(1)==589212 and vc.get(0)==22617) else "  ❌ labels 不符")
PY
echo "DF_PARQUET_DL_DONE"
