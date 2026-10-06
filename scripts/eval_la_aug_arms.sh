#!/usr/bin/env bash
# LA 增广臂评估（H5/H6）：对每臂 last.pth 打分 dev(全量) + la_p4_clean + la_p4fix
# 主结果用 last.pth（固定预算），best.pth 并列备查。
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

cd "$WS"/code
MAN=$WS/data/manifests
RES=$WS/results
LOG=$RES/la_aug_eval.log
: > $LOG

for ARM in aug ctl; do
  CKPT=$RES/la_arm_$ARM/last.pth
  [ -f "$CKPT" ] || { echo "❌ 缺 $CKPT" | tee -a $LOG; exit 2; }
  for spec in "dev:asvspoof2019_la_manifest.csv:dev" \
              "p4clean:la_p4_clean_manifest.csv:test" \
              "p4fix:la_p4fix_manifest.csv:test"; do
    tag=${spec%%:*}; rest=${spec#*:}; mf=${rest%%:*}; sp=${rest#*:}
    echo "=== [$ARM] $tag ($mf split=$sp) ===" | tee -a $LOG
    CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 $PY -u score_group.py \
      --ckpt $CKPT --manifest $MAN/$mf --split $sp \
      --n-boot 500 --num-workers 8 --num-threads 8 --device cuda --batch-size 32 \
      --by attack_type \
      --save-scores $RES/scores/la_arm_$ARM.$tag.npz \
      --out $RES/scores/la_arm_$ARM.$tag.json >> $LOG 2>&1
    echo "    exit=$?" | tee -a $LOG
  done
done
echo "LA_AUG_EVAL_DONE" | tee -a $LOG
