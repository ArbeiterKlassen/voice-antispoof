#!/usr/bin/env bash
# 第二后端 AASIST-L 跑同一套 LA-P4（理论侧 §四.2）
# ⚠️ 本次不再对输出做 grep 过滤：过滤器会把报错一起吞掉（已踩过一次，静默失败）
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

cd "$WS"/code
MAN=$WS/data/manifests
RES=$WS/results
LOG=$RES/aasistL_full.log
: > $LOG
for spec in "la_clean:la_p4_clean_manifest.csv" "la_p4fix:la_p4fix_manifest.csv" "la_p4r:la_p4r_manifest.csv"; do
  tag=${spec%%:*}; mf=${spec#*:}
  echo "=== AASIST-L @ $tag ===" | tee -a $LOG
  CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=8 $PY -u score_group.py \
    --ckpt aasist/models/weights/AASIST-L.pth --model-config aasist/config/AASIST-L.conf \
    --manifest $MAN/$mf --split test \
    --n-boot 1000 --num-workers 8 --num-threads 8 --device cuda --batch-size 32 \
    --save-scores $RES/scores/AASISTL.$tag.npz --out $RES/scores/AASISTL.$tag.json >> $LOG 2>&1
  echo "    exit=$?" | tee -a $LOG
done
echo "AASISTL_P4_DONE"
