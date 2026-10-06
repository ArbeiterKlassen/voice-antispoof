#!/usr/bin/env bash
# 公开基准轨（ASVspoof2019 LA）—— 目的：让「辨伪能力」第一次有可与文献对齐的刻度
#
# 关键一步是**已知答案测试**：
#   官方 AASIST.pth（论文报告 LA eval EER 0.83%）在我们自己的评估管线上跑一遍。
#   复现得上 → 管线可信，之后的数才有意义；复现不上 → 先修管线，不许往下走。
#   注意 AASIST.pth 与 LA eval 玩家共享关系：LA 官方 train/dev/test 说话人不相交，
#   官方权重是在 LA train 上训的，故此为正当的 held-out 评估（非泄漏）。
#
# 用法: bash run_la.sh [--sub N]     # --sub N = LA eval 只取 N 条（快速迭代）
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

cd "$WS"/code
MAN=$WS/data/manifests
RES=$WS/results
GPU=4
SUB=${1:-}

echo "########## [1/3] 落地 LA 数据（先校验，再写盘）##########"
$PY -u prepare_asvspoof.py --check-only
$PY -u prepare_asvspoof.py

echo; echo "########## [2/3] 已知答案对拍：官方 AASIST.pth 在 LA eval 上 #########"
echo "  论文报告：LA eval EER 0.83%（AASIST, Jee-weon Jung et al.）"
M=$MAN/asvspoof2019_la_manifest.csv
if [ -n "$SUB" ]; then
  # 分层抽样：按 (label, attack_type) 分层，保持伪造/真实的攻击构成比例
  M=$MAN/asvspoof2019_la_manifest_sub$SUB.csv
  $PY - "$MAN/asvspoof2019_la_manifest.csv" "$M" "$SUB" <<'PYEOF'
import sys, pandas as pd
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
df = pd.read_csv(src); df = df[df.split == "test"]
frac = n / len(df)
keep = df.groupby(["label", "attack_type"], group_keys=False).apply(
    lambda g: g.sample(n=max(1, int(round(len(g)*frac))), random_state=20261004))
print(f"  子集: {len(keep)}/{len(df)} 条（分层 keep，实际比例与全集一致）")
print(keep.groupby(["label"]).size().to_string())
keep.to_csv(dst, index=False)
PYEOF
fi
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 $PY -u score_group.py \
  --ckpt aasist/models/weights/AASIST.pth --manifest $M --split test \
  --n-boot 1000 --num-workers 12 --num-threads 8 \
  --device cuda --batch-size 32 --by attack_type \
  --save-scores $RES/scores/official_AASIST.la.npz \
  --out $RES/scores/official_AASIST.la.json

echo; echo "########## [3/3] 我们的检测器在同集上（跨数据集，预期差很多）##########"
for n in A_best B_best D_best; do
  ck=$RES/det_signalproc/best.pth
  [ "$n" = "B_best" ] && ck=$RES/det_aug/best.pth
  [ "$n" = "D_best" ] && ck=$RES/det_augr/best.pth
  [ -f "$ck" ] || { echo "  (跳过 $n：$ck 不存在)"; continue; }
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 $PY -u score_group.py \
    --ckpt $ck --manifest $M --split test \
    --n-boot 1000 --num-workers 12 --num-threads 8 \
    --device cuda --batch-size 32 \
    --save-scores $RES/scores/$n.la.npz --out $RES/scores/$n.la.json
done
echo "LA_TRACK_DONE"
