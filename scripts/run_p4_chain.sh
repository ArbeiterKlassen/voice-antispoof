#!/usr/bin/env bash
# 等 P1 评估结束 -> 跑 P4 鲁棒性评估 -> 退出（供后台任务托管，跑完自动提醒）
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

cd "$WS"/code
LOGS=$WS/logs
RES=$WS/results/det_signalproc

echo "[$(date +%T)] 等待 P1 评估结束..."
while pgrep -f "[e]val_detector.py" >/dev/null; do sleep 15; done
echo "[$(date +%T)] P1 已结束"

echo "[$(date +%T)] 启动 P4 鲁棒性评估"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
$PY -u eval_detector.py \
  --ckpt $RES/best.pth \
  --manifest $WS/data/manifests/p4_manifest.csv \
  --split test --by attack_type --n-boot 1000 --frozen-threshold 1.782691 \
  --num-workers 2 --num-threads 8 \
  --out $RES/p4_robustness_metrics.json
rc=$?
echo "[$(date +%T)] P4 结束 (rc=$rc)"
echo "ALL_DONE"
exit $rc
