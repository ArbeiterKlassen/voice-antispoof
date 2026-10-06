#!/usr/bin/env bash
# DF 轨三口径评估（判据预注册见 results/df_eval_prereg.md；本脚本只执行）
#
#   ① 定长（诊断列）        score_group.py（np.tile 到 64600，与 LA 轨同管线）
#   ② 全长-整条（主列）     code/score_df.py --mode full
#   ②b 全长-滑窗（稳健列）  code/score_df.py --mode window（64600 窗 / hop 32300 / 均值）
#
# 模型 = LA 轨两后端：官方 AASIST（d_args 内嵌）、AASIST-L（须 --model-config）。
# GPU：环境变量 DF_GPU（默认 3）。⚠️ 启动前先确认该卡空（共享机纪律，一次一张卡）。
# 幂等：已存在的 npz 跳过；日志 results/df_eval.log（不 grep，防静默吞错——踩过一次）。
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

cd "$WS"/code
MAN=$WS/data/manifests
RES=$WS/results/scores
DFMAN=$MAN/asvspoof2021_df_manifest.csv
GPU=${DF_GPU:-3}
LOG=$WS/results/df_eval.log
: > $LOG
mkdir -p "$RES"

for spec in "official_AASIST:aasist/models/weights/AASIST.pth:" \
            "AASISTL:aasist/models/weights/AASIST-L.pth:aasist/config/AASIST-L.conf"; do
  TAG=${spec%%:*}; rest=${spec#*:}; CK=${rest%%:*}; MC=${rest#*:}
  MCARG=(); [ -n "$MC" ] && MCARG=(--model-config "$MC")
  echo "########## $TAG @ DF（GPU $GPU）##########" | tee -a $LOG

  if [ ! -f "$RES/$TAG.df_fix.npz" ]; then
    echo "--- ① 定长（np.tile 64600）---" | tee -a $LOG
    CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 $PY -u score_group.py \
      --ckpt "$CK" "${MCARG[@]}" --manifest "$DFMAN" --split test \
      --n-boot 1000 --num-workers 12 --num-threads 8 --device cuda --batch-size 32 \
      --save-scores "$RES/$TAG.df_fix.npz" --out "$RES/$TAG.df_fix.json" >> $LOG 2>&1
    echo "    exit=$?" | tee -a $LOG
  fi

  if [ ! -f "$RES/$TAG.df_full.npz" ]; then
    echo "--- ② 全长-整条（主列）---" | tee -a $LOG
    CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 $PY -u score_df.py \
      --ckpt "$CK" "${MCARG[@]}" --manifest "$DFMAN" --split test --mode full \
      --num-threads 8 --device cuda --save-scores "$RES/$TAG.df_full.npz" >> $LOG 2>&1
    echo "    exit=$?" | tee -a $LOG
  fi

  if [ ! -f "$RES/$TAG.df_win.npz" ]; then
    echo "--- ②b 全长-滑窗（均值）---" | tee -a $LOG
    CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 $PY -u score_df.py \
      --ckpt "$CK" "${MCARG[@]}" --manifest "$DFMAN" --split test --mode window \
      --num-threads 8 --device cuda --batch-size 32 \
      --save-scores "$RES/$TAG.df_win.npz" >> $LOG 2>&1
    echo "    exit=$?" | tee -a $LOG
  fi
done
echo "DF_EVAL_DONE" | tee -a $LOG
