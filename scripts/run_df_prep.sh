#!/usr/bin/env bash
# DF 预处理分级运行：检查 → 冒烟(200) → 全量(约 4.5 万条)
# 前置：data/_downloads/df/ 已完整（1,223,658 行 = 611,829×2，footer 校验通过）
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

cd "$WS"
LOG=results/df_prep.log
: > $LOG

echo "########## [1/3] --check-only（泄漏检查 + labels 对拍）##########" | tee -a $LOG
OMP_NUM_THREADS=8 $PY -u code/prepare_asvspoof_df.py --check-only >> $LOG 2>&1
rc=$?; echo "    exit=$rc" | tee -a $LOG
[ $rc -ne 0 ] && { echo "❌ check-only 未过，停止（不得绕过泄漏检查）"; exit 2; }

echo "########## [2/3] 冒烟：伪造只抽 200 ##########" | tee -a $LOG
OMP_NUM_THREADS=8 $PY -u code/prepare_asvspoof_df.py --max-spoof 200 >> $LOG 2>&1
rc=$?; echo "    exit=$rc" | tee -a $LOG
[ $rc -ne 0 ] && { echo "❌ 冒烟失败"; exit 3; }
# 冒烟产出独立 manifest，避免与全量混淆
[ -f data/manifests/asvspoof2021_df_manifest.csv ] && \
  mv data/manifests/asvspoof2021_df_manifest.csv data/manifests/asvspoof2021_df_smoke200.csv

echo "########## [3/3] 全量（真实 22617 + 伪造分层 22617）##########" | tee -a $LOG
OMP_NUM_THREADS=8 $PY -u code/prepare_asvspoof_df.py >> $LOG 2>&1
rc=$?; echo "    exit=$rc" | tee -a $LOG
[ $rc -ne 0 ] && { echo "❌ 全量失败"; exit 4; }
echo "DF_PREP_DONE" | tee -a $LOG
