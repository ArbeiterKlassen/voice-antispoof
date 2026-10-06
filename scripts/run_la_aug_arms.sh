#!/usr/bin/env bash
# LA 增广臂实验（报告 §6.2；判据 H5/H6 已在报告里**预注册**，本脚本只执行）
#
# 两臂（唯一差异 = 训练集是否含增广；seed/epochs/batch/init-weights 全同）：
#   ① AUG：la_aug_train_manifest.csv（25,380 LA-train + 9,000 增广 = 34,380）
#   ② CTL：asvspoof2019_la_train_subdev.csv（25,380，同构造无增广）
# 均从官方 AASIST.pth 微调 6 epoch（--init-weights 默认）；主结果用 **last.pth**（固定预算，
# 见 §3 协议修正：best.pth 是噪声选出来的）。
#
# 用法: bash scripts/run_la_aug_arms.sh   （GPU 3；两臂顺序跑，约 3 h）
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

cd "$WS"/code
MAN=$WS/data/manifests
RES=$WS/results
LOG=$RES/la_aug_arms.log
: > $LOG

run_arm () {   # $1=名字 $2=manifest
  echo "=== [$1] manifest=$(basename $2) ===" | tee -a $LOG
  CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 $PY -u train_detector.py \
    --manifest "$2" --epochs 6 --batch-size 16 --num-workers 8 \
    --out-dir $RES/la_arm_$1 >> $LOG 2>&1
  echo "    exit=$?" | tee -a $LOG
}

run_arm aug $MAN/la_aug_train_manifest.csv
run_arm ctl $MAN/asvspoof2019_la_train_subdev.csv

echo "LA_AUG_ARMS_DONE" | tee -a $LOG
