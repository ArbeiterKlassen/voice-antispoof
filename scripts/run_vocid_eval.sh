#!/usr/bin/env bash
# 恒等臂评估（两后端）+ 音高族位移表素材（音高族分数已有，无需重打）
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

cd "$WS"/code
MAN=$WS/data/manifests
RES=$WS/results
LOG=$RES/vocid_eval.log
: > $LOG

echo "=== AASIST @ la_p4vocid ===" | tee -a $LOG
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 $PY -u score_group.py \
  --ckpt aasist/models/weights/AASIST.pth --manifest $MAN/la_p4vocid_manifest.csv --split test \
  --n-boot 1000 --num-workers 8 --num-threads 8 --device cuda --batch-size 32 \
  --by attack_type \
  --save-scores $RES/scores/official_AASIST.la_p4vocid.npz \
  --out $RES/scores/official_AASIST.la_p4vocid.json >> $LOG 2>&1
echo "    exit=$?" | tee -a $LOG

echo "=== AASIST-L @ la_p4vocid ===" | tee -a $LOG
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 $PY -u score_group.py \
  --ckpt aasist/models/weights/AASIST-L.pth --model-config aasist/config/AASIST-L.conf \
  --manifest $MAN/la_p4vocid_manifest.csv --split test \
  --n-boot 1000 --num-workers 8 --num-threads 8 --device cuda --batch-size 32 \
  --by attack_type \
  --save-scores $RES/scores/AASISTL.la_p4vocid.npz \
  --out $RES/scores/AASISTL.la_p4vocid.json >> $LOG 2>&1
echo "    exit=$?" | tee -a $LOG
echo "VOCID_EVAL_DONE"
