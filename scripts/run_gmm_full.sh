#!/usr/bin/env bash
# GMM 第三后端全量流水线（预注册见 results/gmm_prereg.md）
# 冒烟已过（200+200 训练 → 干净集 EER 15.4%/AUC 0.933）。
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

# 共享机纪律：线程封顶（feats/score 是 Pool×8，每 worker 限 1 线程；
# train 是单进程 BLAS 重活，在那一行内单独放宽到 6。2026-10-07 实测 8×64 线程超订把 load 打到 525）
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

cd "$WS"
CACHE="$WS/data/gmm_cache"          # data/* 被 .gitignore 挡（含本缓存）✓
mkdir -p "$CACHE"
LOG="$WS/results/gmm_full.log"
: > $LOG

echo "########## [1/5] 训练帧（LA train 25,380，逐条 ≤200 帧；分片流式）##########" | tee -a $LOG
nice -n 10 $PY -u code/gmm_lfcc.py feats \
  --manifest "$WS/data/manifests/asvspoof2019_la_manifest.csv" \
  --out "$CACHE/train_feats.npz" --workers 8 >> $LOG 2>&1
echo "    exit=$?" | tee -a $LOG
ls "$CACHE"/train_feats_*.npz >/dev/null 2>&1 || { echo "❌ feats 失败"; exit 2; }

echo "########## [2/5] GMM(256, diag) 拟合 ##########" | tee -a $LOG
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 nice -n 10 $PY -u code/gmm_lfcc.py train --feats "$CACHE/train_feats_*.npz" \
  --model "$CACHE/gmm256.pkl" --components 256 --n-train-frames 200000 --max-iter 100 >> $LOG 2>&1
echo "    exit=$?" | tee -a $LOG
[ ! -s "$CACHE/gmm256.pkl" ] && { echo "❌ train 失败"; exit 3; }

echo "########## [3/5] 打分：LA eval 全量（已知答案锚）##########" | tee -a $LOG
nice -n 10 $PY -u code/gmm_lfcc.py score \
  --manifest "$WS/data/manifests/asvspoof2019_la_manifest.csv" --split test \
  --model "$CACHE/gmm256.pkl" --out "$WS/results/scores/gmm.la_full.npz" --workers 8 >> $LOG 2>&1
echo "    exit=$?" | tee -a $LOG

echo "########## [4/5] 打分：P4 四套 ##########" | tee -a $LOG
for spec in "p4fix:la_p4fix_manifest.csv" "p4r:la_p4r_manifest.csv" \
            "p4pitch:la_p4pitch_manifest.csv" "p4vocid:la_p4vocid_manifest.csv"; do
  tag=${spec%%:*}; mf=${spec#*:}
  echo "  --- $tag ---" | tee -a $LOG
  nice -n 10 $PY -u code/gmm_lfcc.py score \
    --manifest "$WS/data/manifests/$mf" --split test \
    --model "$CACHE/gmm256.pkl" --out "$WS/results/scores/gmm.$tag.npz" --workers 8 >> $LOG 2>&1
  echo "    exit=$?" | tee -a $LOG
done

echo "########## [5/5] G1 预检（已知答案 + 编解码族判别力）##########" | tee -a $LOG
nice -n 10 $PY - <<'PYEOF' >> $LOG 2>&1
import numpy as np, pandas as pd, sys, os
sys.path.insert(0, os.path.join(os.environ["VOICE_WS"], "code"))
from metrics import evaluate
WS = os.environ["VOICE_WS"]
z = np.load(f"{WS}/results/scores/gmm.la_full.npz", allow_pickle=False)
m = evaluate(z["score"], z["y"])
print(f"[G1-a] LA eval 全量：EER {m['eer_percent']:.3f}%  AUC {m['auc']:.4f}  "
      f"（已知答案目标 8–14%，容带 [6,18]）")
z = np.load(f"{WS}/results/scores/gmm.p4fix.npz", allow_pickle=False)
d = pd.DataFrame({"score": z["score"], "label": z["label"], "attack": z["attack_type"]})
for fam in ["mp3-64k", "aac-128k", "opus-32k"]:
    f = d[(d.label == "fake") & (d.attack == fam)]; r = d[(d.label == "real") & (d.attack == fam)]
    s = np.concatenate([f.score.values, r.score.values])
    y = np.concatenate([np.zeros(len(f), int), np.ones(len(r), int)])
    mm = evaluate(s, y)
    print(f"[G1-b] {fam}: AUC {mm['auc']:.4f}（门槛 ≥0.95）"
          f" {'✅' if mm['auc'] >= 0.95 else '❌'}")
PYEOF
echo "GMM_FULL_DONE" | tee -a $LOG
