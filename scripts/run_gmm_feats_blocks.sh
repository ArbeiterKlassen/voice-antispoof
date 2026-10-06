#!/usr/bin/env bash
# GMM feats 后备路径：8 个分块子进程并行（每块 ~3173 条，路径序**连续区间**，读盘各走一段顺序流）。
#
# 用途：Pool 版若再出现「主进程 stime 恒 100 tick/s + 进度停住」的病理，用本脚本单独把
# 特征缓存跑出来（每块自带清单与分片），然后重跑 run_gmm_full.sh（阶段 1 幂等会跳过 feats）。
# 分片命名 train_feats_p<块号>_*.npz，与主脚本的 glob（train_feats_*.npz）兼容。
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

cd "$WS"
CACHE="$WS/data/gmm_cache"
mkdir -p "$CACHE"
LOG="$WS/results/gmm_feats_blocks.log"
: > $LOG
MAN="$WS/data/manifests/asvspoof2019_la_manifest.csv"
N=${GMM_FEATS_BLOCKS:-8}
TOTAL=25380
BLK=$(( (TOTAL + N - 1) / N ))

pids=()
for j in $(seq 0 $((N - 1))); do
  A=$(( j * BLK )); B=$(( A + BLK ))
  [ "$B" -gt "$TOTAL" ] && B=$TOTAL
  [ "$A" -ge "$TOTAL" ] && break
  echo "  块 $j：行 [$A, $B)" | tee -a $LOG
  # 每块：单 worker（父进程零累积负担被控制在 ~72MB：每 500 条 flush 一个 ~24MB 分片）
  nice -n 10 $PY -u code/gmm_lfcc.py feats --manifest "$MAN" \
    --out "$CACHE/train_feats_p${j}.npz" --workers 1 --shard-files 500 --block "$A:$B" \
    >> "$LOG" 2>&1 &
  pids+=($!)
done
echo "启动 ${#pids[@]} 块：${pids[*]}" | tee -a $LOG
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
n=$(ls "$CACHE"/train_feats_p*.npz 2>/dev/null | wc -l)
echo "结束：$n 个分片文件，fail=$fail" | tee -a $LOG
[ $fail -eq 0 ] && [ "$n" -gt 0 ] && echo "FEATS_BLOCKS_DONE" || { echo "FEATS_BLOCKS_FAILED"; exit 2; }
