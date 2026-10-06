#!/usr/bin/env bash
# LA 轨接力：等 GPU 空出 → 落地 LA → **已知答案对拍** → 停下报数
#
# ⚠️ 刻意在已知答案测试后**停住**，不自动往下训 LA：
#    若官方 AASIST.pth 复现不出论文的 0.83%，说明我们的评估管线有问题，
#    此时再花两小时训 LA 得到的数毫无意义。先看对拍结果，再决定。
#
# 触发条件：臂A20 训练完成（results/det_signalproc20/last.pth 出现）→ GPU4 空出。
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

cd "$WS"/code
MAN=$WS/data/manifests
RES=$WS/results
DL=$WS/data/_downloads/asvspoof2019_la/data
GPU=${1:-4}

echo "########## [1/4] 等 LA 下载完整（三分片可读且行数正确）##########"
for i in $(seq 1 240); do
  ok=$($PY - <<'PY' 2>/dev/null
import pyarrow.parquet as pq, os
exp={"train":25380,"validation":24844,"test":71237}
D=os.environ["VOICE_WS"]+"/data/_downloads/asvspoof2019_la/data"
good=0
for k,v in exp.items():
    p=f"{D}/{k}-00000-of-00001.parquet"
    try:
        if pq.ParquetFile(p).metadata.num_rows==v: good+=1
    except Exception: pass
print(good)
PY
)
  [ "${ok:-0}" = "3" ] && { echo "[$(date +%T)] 三分片完整"; break; }
  echo "[$(date +%T)] 尚未完整（$ok/3），等 30s"
  sleep 30
done
[ "${ok:-0}" = "3" ] || { echo "❌ LA 下载未完成，退出"; exit 1; }

echo; echo "########## [2/4] 落地 LA（纯 CPU/IO，先做，免得等 GPU 时空耗）##########"
$PY -u prepare_asvspoof.py --check-only
$PY -u prepare_asvspoof.py

echo; echo "########## [3/4] 等臂A20 训练结束（GPU4 空出）##########"
for i in $(seq 1 480); do
  [ -f $RES/det_signalproc20/last.pth ] && { echo "[$(date +%T)] A20 已完成"; break; }
  sleep 30
done
[ -f $RES/det_signalproc20/last.pth ] || echo "⚠️ A20 迟迟未完成，仍继续（后续 GPU 任务可能与他人训练共存）"

echo; echo "########## [4/4] 已知答案对拍 + 我们的检测器跨数据集 ##########"
LAMAN=$MAN/asvspoof2019_la_manifest.csv
echo "--- 官方 AASIST.pth（论文 LA eval EER 0.83%）---"
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 $PY -u score_group.py \
  --ckpt aasist/models/weights/AASIST.pth --manifest $LAMAN --split test \
  --n-boot 1000 --num-workers 12 --num-threads 8 --device cuda --batch-size 32 \
  --by attack_type \
  --save-scores $RES/scores/official_AASIST.la.npz --out $RES/scores/official_AASIST.la.json

for spec in "A_best:$RES/det_signalproc/best.pth" "B_best:$RES/det_aug/best.pth" "D_best:$RES/det_augr/best.pth"; do
  n=${spec%%:*}; ck=${spec#*:}
  [ -f "$ck" ] || { echo "  (跳过 $n)"; continue; }
  echo "--- $n（自建轨训练，看跨数据集退化）---"
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 $PY -u score_group.py \
    --ckpt "$ck" --manifest $LAMAN --split test \
    --n-boot 1000 --num-workers 12 --num-threads 8 --device cuda --batch-size 32 \
    --save-scores $RES/scores/$n.la.npz --out $RES/scores/$n.la.json
done

echo; echo "════════ 对拍判读 ════════"
$PY - <<'PYEOF'
import json, os
R=os.environ["VOICE_WS"]+"/results/scores"
o=json.load(open(f"{R}/official_AASIST.la.json"))["overall"]
e=o["eer_percent"]
print(f"  官方 AASIST.pth 在 LA eval：EER {e:.3f}%  (论文 0.83%；差异需用不同评测管线的实现细节解释)")
print(f"    判断：", end="")
print("✅ 管线可信（与论文同量级）" if e < 2.0 else
      ("⚠️ 同量级但偏高（2–5%），需查原因" if e < 5 else
       "❌ 与论文严重不符（>5%），**停下**，先查评估管线（padding/方向/重采样/划分覆盖）"))
for n in ["A_best","B_best","D_best"]:
    p=f"{R}/{n}.la.json"
    if os.path.exists(p):
        d=json.load(open(p))["overall"]
        print(f"  {n:<8} 跨数据集 EER {d['eer_percent']:.3f}%  AUC {d['auc']:.4f}")
PYEOF
echo "LA_CHAIN_DONE"
