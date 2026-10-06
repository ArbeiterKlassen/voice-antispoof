#!/usr/bin/env bash
# 音高族评估：两后端 × la_p4pitch（不加 grep 过滤，防吞报错）
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

cd "$WS"/code
MAN=$WS/data/manifests
RES=$WS/results
LOG=$RES/pitch_eval.log
: > $LOG

echo "=== AASIST @ la_p4pitch ===" | tee -a $LOG
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=8 $PY -u score_group.py \
  --ckpt aasist/models/weights/AASIST.pth --manifest $MAN/la_p4pitch_manifest.csv --split test \
  --n-boot 1000 --num-workers 12 --num-threads 8 --device cuda --batch-size 32 \
  --by attack_type \
  --save-scores $RES/scores/official_AASIST.la_p4pitch.npz \
  --out $RES/scores/official_AASIST.la_p4pitch.json >> $LOG 2>&1
echo "    exit=$?" | tee -a $LOG

echo "=== AASIST-L @ la_p4pitch ===" | tee -a $LOG
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=8 $PY -u score_group.py \
  --ckpt aasist/models/weights/AASIST-L.pth --model-config aasist/config/AASIST-L.conf \
  --manifest $MAN/la_p4pitch_manifest.csv --split test \
  --n-boot 1000 --num-workers 12 --num-threads 8 --device cuda --batch-size 32 \
  --by attack_type \
  --save-scores $RES/scores/AASISTL.la_p4pitch.npz \
  --out $RES/scores/AASISTL.la_p4pitch.json >> $LOG 2>&1
echo "    exit=$?" | tee -a $LOG
echo "PITCH_EVAL_DONE"
