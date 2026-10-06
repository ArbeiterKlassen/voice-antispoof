#!/usr/bin/env bash
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

cd "$WS"/code
MAN=$WS/data/manifests
RES=$WS/results
for tag in clean p4fix p4r; do
  echo "--- durmatch_$tag"
  CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=8 $PY -u score_group.py \
    --ckpt aasist/models/weights/AASIST.pth --manifest $MAN/durmatch_${tag}_manifest.csv --split test \
    --n-boot 1000 --num-workers 12 --num-threads 8 --device cuda --batch-size 32 \
    --save-scores $RES/scores/official_AASIST.dm_$tag.npz \
    --out $RES/scores/official_AASIST.dm_$tag.json 2>&1 | grep -E "EER|AUC|n_eval" | head -3
done
echo "DURMATCH_EVAL_DONE"
