#!/usr/bin/env bash
# ASVspoof2019 LA 训练 + 官方 test 评估（公开基准轨）
#
# 【为什么 dev 要抽子集】
#   官方 dev 有 24484 条。train_detector.py 每个 epoch 全评一遍 dev，
#   GPU 上约 8 min/epoch，比训练本身还贵，20 epoch 要多花近 3 小时。
#   → 训练期监控用**说话人分层子集**；**最终报数一律用官方 test 全集**。
#
# 【与文献对齐的前提】
#   官方 AASIST.pth 已在 LA train 上训过（论文 LA eval EER 0.83%）。
#   本脚本以它为初始化做微调，故 baseline 就是「0.83%」这个数，
#   我们自己的增广/改动只有相对它的差值才有意义。
#   注意 LA eval 的攻击系统是 A07–A19，**训练集只有 A01–A19 中的 A01–A06**，
#   所以 LA eval 天然就是「跨生成器」测试（这也是它成为标准的原因）。
#
# 用法: bash train_la.sh [epochs] [gpu]
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

cd "$WS"/code
MAN=$WS/data/manifests
RES=$WS/results
EP=${1:-10}
GPU=${2:-4}
FULL=$MAN/asvspoof2019_la_manifest.csv

echo "########## [1/4] 造训练用 manifest（全 train + 说话人分层 dev 子集）##########"
$PY - "$FULL" "$MAN/asvspoof2019_la_train_subdev.csv" <<'PYEOF'
import sys, pandas as pd
src, dst = sys.argv[1], sys.argv[2]
df = pd.read_csv(src)
tr = df[df.split == "train"]
dv = df[df.split == "dev"]
# dev 子集：按 (label, speaker) 分层，目标 ~3000 条，保持真实/伪造与说话人构成
target = 3000
frac = target / len(dv)
sub = (dv.groupby("speaker_id", group_keys=False)
         .apply(lambda g: g.sample(n=max(2, int(round(len(g) * frac))), random_state=20261004)))
keep = pd.concat([tr, sub], ignore_index=True)
print(f"  train 全部 {len(tr)} 条 + dev 子集 {len(sub)}/{len(dv)} 条 （说话人 {sub.speaker_id.nunique()}/{dv.speaker_id.nunique()}）")
for s in ["train", "dev"]:
    x = keep[keep.split == s]
    print(f"  {s:<6} real {int((x.label=='real').sum()):>6} / fake {int((x.label=='fake').sum()):>6}")
    print(f"         攻击编号 {sorted(set(x.attack_type))[:8]}{'...' if x.attack_type.nunique()>8 else ''}")
keep.to_csv(dst, index=False)
print(f"-> {dst} ({len(keep)} 条)")
PYEOF

echo; echo "########## [2/4] 训练（微调官方 AASIST 权重，$EP epochs @GPU$GPU）##########"
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 $PY -u train_detector.py \
  --manifest $MAN/asvspoof2019_la_train_subdev.csv --epochs $EP \
  --batch-size 16 --num-workers 6 --out-dir $RES/la_aasist

echo; echo "########## [3/4] 官方 test 全集评估（71237 条，GPU）##########"
for tag in best last; do
  ck=$RES/la_aasist/$tag.pth
  [ -f "$ck" ] || { echo "  (无 $ck，跳过)"; continue; }
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 $PY -u score_group.py \
    --ckpt $ck --manifest $FULL --split test \
    --n-boot 1000 --num-workers 12 --num-threads 8 --device cuda --batch-size 32 \
    --by attack_type \
    --save-scores $RES/scores/la_ours_$tag.test.npz \
    --out $RES/scores/la_ours_$tag.test.json
done

echo; echo "########## [4/4] 与我们自建轨的对照（同 ckpt 在自建 P1 上）##########"
$PY - "$RES/scores" <<'PYEOF'
import json, glob, os, sys
print(f"  {'结果文件':<34} {'EER%':>8} {'AUC':>8}")
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    try:
        d = json.load(open(f))
        o = d.get("overall", {})
        if "eer_percent" in o:
            print(f"  {os.path.basename(f):<34} {o['eer_percent']:>8.3f} {o.get('auc',float('nan')):>8.4f}")
    except Exception:
        pass
PYEOF
echo "TRAIN_LA_DONE"
