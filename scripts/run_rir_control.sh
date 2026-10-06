#!/usr/bin/env bash
# RIR 控制实验全链路：生成 → 合 manifest → 评估 A_best → 与固定版对比
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

cd "$WS"/code
MAN=$WS/data/manifests
RES=$WS/results
$PY -u make_rir_control.py --src $MAN/_p4_real_src.csv --out $MAN/rirctl_real.csv \
  --label real --num-threads 8
$PY -u make_rir_control.py --src $MAN/_p4_src_stratified.csv --out $MAN/rirctl_fake.csv \
  --label fake --num-threads 8
$PY - <<'PYEOF'
import pandas as pd
MAN=os.environ["VOICE_WS"]+"/data/manifests"
a=pd.read_csv(f"{MAN}/rirctl_real.csv"); b=pd.read_csv(f"{MAN}/rirctl_fake.csv")
d=pd.concat([a,b],ignore_index=True)
assert d.utt_id.is_unique and not d.utt_id.duplicated().any()
print(f"  控制集 {len(d)} 条: real {int((d.label=='real').sum())} / fake {int((d.label=='fake').sum())}")
print(f"  utt_id 与固定版是否重名: {len(set(d.utt_id) & set(pd.read_csv(f'{MAN}/p4_correct_manifest.csv').utt_id))} 个")
d.to_csv(f"{MAN}/p4_rirctl_manifest.csv",index=False)
PYEOF
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 $PY -u score_group.py \
  --ckpt $RES/det_signalproc/best.pth --manifest $MAN/p4_rirctl_manifest.csv --split test \
  --n-boot 1000 --num-workers 3 --num-threads 8 --batch-size 16 \
  --save-scores $RES/scores/A_best.rirctl.npz --out $RES/scores/A_best.rirctl.json 2>&1 | tail -3
echo "RIRCTL_DONE"
