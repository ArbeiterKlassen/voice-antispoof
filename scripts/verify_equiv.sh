#!/usr/bin/env bash
# score_group.py 的**已知答案对拍**（memory: known-answer-test-rule）
#
# 背景：eval_detector.py 在 --by 分组时每组重打分（14400 次前向/次评估，四臂要两小时）。
#      score_group.py 改为「打一次分 + 按索引分组」。快路径必须先用已提交的结果验等价，
#      不能"我觉得等价"。
#
# 对拍对象：results/det_signalproc/p4_correct.json （臂A best.pth 的 P4 结果，已落盘）
# 产物同时留作正式结果，省一次评估。
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

cd "$WS"/code
MAN=$WS/data/manifests
RES=$WS/results
FRZ=1.782691
mkdir -p $RES/scores

echo "########## [0/2] 冻结阈值来源自检（必须来自 P1 dev，不是凭空写死）##########"
$PY - <<'PYEOF'
import json
p=os.environ["VOICE_WS"]+"/results/det_signalproc/p1_dev_metrics.json"
d=json.load(open(p))
def find(o, keys=("eer_threshold","threshold")):
    if isinstance(o,dict):
        for k,v in o.items():
            if k in keys and isinstance(v,(int,float)): yield k,v
            yield from find(v, keys)
hit=dict(find(d))
print("p1_dev_metrics.json 里的阈值字段:", hit)
print("链中写死的 FRZ = 1.782691")
ok=any(abs(v-1.782691)<1e-5 for v in hit.values())
print("一致性:", "✅ 相符（确实来自 P1 dev）" if ok else "⚠️ 未对上，需人工核对来源")
PYEOF

echo; echo "########## [1/2] 快路径跑臂A best.pth 的 P4（同时作为正式结果落盘）##########"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 $PY -u score_group.py \
  --ckpt $RES/det_signalproc/best.pth \
  --manifest $MAN/p4_correct_manifest.csv --split test \
  --by attack_type --frozen-threshold $FRZ --n-boot 1000 \
  --num-workers 3 --num-threads 8 --batch-size 16 \
  --save-scores $RES/scores/A_best.p4.npz \
  --out $RES/scores/A_best.p4.json

echo; echo "########## [2/2] 与 eval_detector.py 已落盘结果逐组对拍 ##########"
$PY - <<'PYEOF'
import json
REF=os.environ["VOICE_WS"]+"/results/det_signalproc/p4_correct.json"   # 老路径产出
NEW=os.environ["VOICE_WS"]+"/results/scores/A_best.p4.json"            # 快路径产出
a=json.load(open(REF)); b=json.load(open(NEW))
def val(m,k):
    v=m.get(k)
    return float(v) if isinstance(v,(int,float)) else None
rows=[]
for k in ("eer_percent","auc","accuracy","eer_threshold"):
    rows.append(("overall",k,val(a["overall"],k),val(b["overall"],k)))
ga,gb=a.get("groups",{}),b.get("groups",{})
assert set(ga)==set(gb), f"组名不一致: 只在A {set(ga)-set(gb)} / 只在B {set(gb)-set(ga)}"
for g in ga:
    for k in ("eer_percent","auc","accuracy"):
        rows.append((g,k,val(ga[g],k),val(gb[g],k)))
bad=[]
for name,k,x,y in rows:
    if x is None or y is None: bad.append((name,k,x,y,"缺值")); continue
    if abs(x-y)>1e-9: bad.append((name,k,x,y,f"差 {abs(x-y):.3g}"))
print(f"对拍项数 {len(rows)}（overall 4 项 + {len(ga)} 组 × 3 项）")
if bad:
    print("❌ 不等价：")
    for r in bad[:20]: print("   ", r)
else:
    print("✅ 全部逐位相同 —— 快路径与 eval_detector.py 数值等价")
# 冻结阈值列也要对
fa=a["overall"].get("at_frozen_threshold"); fb=b["overall"].get("at_frozen_threshold")
if fa and fb:
    d=max(abs(fa[k]-fb[k]) for k in fa if isinstance(fa[k],(int,float)))
    print(f"   冻结阈值列最大差 {d:.3g}  —— {'✅' if d<1e-9 else '❌'}")
print("EQUIV_CHECK_DONE")
PYEOF
