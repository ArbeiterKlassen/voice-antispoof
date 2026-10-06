#!/usr/bin/env bash
# LA 轨评估批处理 —— 等 GPU 空出后一次跑完
#
# 清单（逐项都可跳过已完成者，本脚本可反复跑）：
#   1. 官方 AASIST 在 **LA dev**（干净）→ 取冻结阈值（EER 交点 + 零错误区间）
#   2. 官方 AASIST 在 **LA test 全量**（71237）→ 已知答案对拍（论文 0.83%）
#   3. 官方 AASIST 在 **la_p4_clean / p4fix / p4r** → 后处理鲁棒性（干净基线用同一批 798 源）
#   4. LA 微调模型（la_aasist/best|last）跑上面同样三项
#
# 口径：全部走 score_group.py（已验证与 eval_detector.py 逐位等价）
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

cd "$WS"/code
MAN=$WS/data/manifests
RES=$WS/results/scores
GPU=${1:-4}
mkdir -p $RES

run() {   # run <名称> <ckpt> <manifest> <split> [--by col]
  local name=$1 ck=$2 mf=$3 sp=$4 by=${5:-}
  if [ -f "$RES/$name.json" ]; then echo "  (跳过 $name：已存在)"; return; fi
  echo "  --- $name  <- $(basename $ck)  $(basename $mf)"
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 $PY -u score_group.py \
    --ckpt "$ck" --manifest "$mf" --split "$sp" \
    --n-boot 1000 --num-workers 12 --num-threads 8 --device cuda --batch-size 32 \
    $by --save-scores $RES/$name.npz --out $RES/$name.json 2>&1 | grep -E "EER|AUC|协议. n_eval" | head -3
}

OFF=$WS/code/aasist/models/weights/AASIST.pth

echo "########## [1] 官方 AASIST：LA 三划分 + P4 三套 ##########"
run official_AASIST.la_dev   $OFF $MAN/asvspoof2019_la_manifest.csv dev
run official_AASIST.la       $OFF $MAN/asvspoof2019_la_manifest.csv test
run official_AASIST.la_clean $OFF $MAN/la_p4_clean_manifest.csv test
run official_AASIST.la_p4fix $OFF $MAN/la_p4fix_manifest.csv test "--by attack_type"
run official_AASIST.la_p4r   $OFF $MAN/la_p4r_manifest.csv   test "--by attack_type"

echo; echo "########## [1b] 自建轨三臂跨数据集（只在 LA test 全量上跑一次）##########"
for spec in "A_best:$WS/results/det_signalproc/best.pth" \
            "B_best:$WS/results/det_aug/best.pth" \
            "D_best:$WS/results/det_augr/best.pth"; do
  n=${spec%%:*}; ck=${spec#*:}
  [ -f "$ck" ] || { echo "  (跳过 $n)"; continue; }
  run $n.la $ck $MAN/asvspoof2019_la_manifest.csv test
done

echo; echo "########## [2] LA 微调模型（若已训练）##########"
for t in best last; do
  ck=$WS/results/la_aasist/$t.pth
  [ -f "$ck" ] || { echo "  (无 $ck，跳过)"; continue; }
  run la_ours_$t.la       $ck $MAN/asvspoof2019_la_manifest.csv test
  run la_ours_$t.la_clean $ck $MAN/la_p4_clean_manifest.csv test
  run la_ours_$t.la_p4fix $ck $MAN/la_p4fix_manifest.csv test "--by attack_type"
  run la_ours_$t.la_p4r   $ck $MAN/la_p4r_manifest.csv   test "--by attack_type"
done

echo; echo "########## [3] 汇总 ##########"
$PY - <<'PYEOF'
import json, glob, os, sys
sys.path.insert(0,os.environ["VOICE_WS"]+"/code")
R=os.environ["VOICE_WS"]+"/results/scores"
print(f"  {'结果':<34}{'EER%':>9}{'AUC':>9}")
for f in sorted(glob.glob(f"{R}/*.la*.json")):
    try:
        d=json.load(open(f)); o=d.get("overall",{})
        if "eer_percent" in o:
            print(f"  {os.path.basename(f)[:-5]:<34}{o['eer_percent']:>9.3f}{o.get('auc',float('nan')):>9.4f}")
    except Exception: pass
PYEOF
echo "LA_EVALS_DONE"
