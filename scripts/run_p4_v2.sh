#!/usr/bin/env bash
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

cd "$WS"/code
RES=$WS/results/det_signalproc
MAN=$WS/data/manifests
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

echo "########## [1/2] 匹配版（主）：real=$(($(wc -l < $MAN/p4_manifest_matched.csv)-1-2250)) 条 ##########"
$PY -u eval_detector.py --ckpt $RES/best.pth \
  --manifest $MAN/p4_manifest_matched.csv --split test \
  --by attack_type --n-boot 1000 --frozen-threshold 1.782691 \
  --num-workers 2 --num-threads 8 \
  --out $RES/p4_v2_matched.json

echo; echo "########## [2/2] 全量真实现（对照）##########"
$PY -u eval_detector.py --ckpt $RES/best.pth \
  --manifest $MAN/p4_manifest.csv --split test \
  --by attack_type --n-boot 1000 --frozen-threshold 1.782691 \
  --num-workers 2 --num-threads 8 \
  --out $RES/p4_v2_full.json
echo "P4_V2_ALL_DONE"
