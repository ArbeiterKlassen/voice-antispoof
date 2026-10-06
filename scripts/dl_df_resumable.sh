#!/usr/bin/env bash
# DF 全量可续传下载（理论侧 §六：挂后台跑两天即可，不占关键路径）
# 80 分片 × ~400 MB + labels，约 32 GiB。curl -C - 断点续传，失败重试。
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

D=$WS/data/_downloads/df
BASE="https://hf-mirror.com/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF/resolve/main"
cd $D
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
FILES="/tmp/df_filelist.txt"
[ -s "$FILES" ] || { echo "缺 $FILES（先跑清单抓取）"; exit 1; }

for round in 1 2 3 4 5 6; do
  echo "########## 第 $round 轮 ##########"
  n_done=0; n_todo=0
  while read -r f; do
    [ -z "$f" ] && continue
    # 目录不存在则建
    mkdir -p "$(dirname "$f")"
    # 跳过已完成（对比远程大小太贵，用本地存在+非零+能列 parquet 尾判粗略；简单起见只查存在非零）
    if [ -s "$f" ]; then
      n_done=$((n_done+1)); continue
    fi
    n_todo=$((n_todo+1))
    echo "  ↓ $f"
    curl -sL -C - --retry 5 --retry-delay 10 --max-time 7200 \
      -H "User-Agent: $UA" -o "$f" "$BASE/$f"
    if [ -s "$f" ]; then echo "    ✔ $(du -h "$f" | cut -f1)"; else echo "    ✗ 仍未完成"; fi
  done < "$FILES"
  echo "  本轮：已完成 $n_done，待续 $n_todo"
  # 全部非空即认为完（大小校验留给 prepare_asvspoof_df.py）
  missing=$(while read -r f; do [ -z "$f" ] && continue; [ -s "$f" ] || echo x; done < "$FILES" | wc -l)
  echo "  剩余未完成文件: $missing"
  [ "$missing" -eq 0 ] && { echo "DF_DL_ALL_DONE"; exit 0; }
  sleep 300
done
echo "DF_DL_PARTIAL：本轮跑满未拿全，留待人工重跑"
