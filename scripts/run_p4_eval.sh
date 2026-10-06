#!/usr/bin/env bash
# 等前一个评估结束，再跑 P4 鲁棒性评估（避免两个 CPU 任务互抢）
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

while pgrep -f "[e]val_detector.py" >/dev/null; do sleep 20; done
echo "[$(date +%T)] 前序评估已结束，启动 P4"
cd "$WS"/code
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
/home/cuitianxu/.miniconda3/envs/ml2026_exp/bin/python -u eval_detector.py \
  --ckpt $WS/results/det_signalproc/best.pth \
  --manifest $WS/data/manifests/p4_manifest.csv \
  --split test --by attack_type --n-boot 1000 --frozen-threshold 1.782691 \
  --num-workers 2 --num-threads 8 \
  --out $WS/results/det_signalproc/p4_robustness_metrics.json
echo "P4_EVAL_DONE"
