#!/usr/bin/env bash
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

cd "$WS"/code
MAN=$WS/data/manifests
RES=$WS/results/det_signalproc
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
ATK=mp3-128k,mp3-64k,aac-128k,opus-32k,g711-8k,resample-8k,awgn-snr5,awgn-snr10,awgn-snr20,bandpass-300-3400,reverb-rt60-0.2,reverb-rt60-0.4,reverb-rt60-0.8,speed0.9,speed1.1

echo "########## [1/3] 对真实语音施加同样的攻击 ##########"
$PY -u make_attacks.py --src $MAN/_p4_real_src.csv --limit 54 --splits test \
  --num-threads 8 --label real --source real_attack --attacks $ATK \
  --out-manifest $MAN/real_attacks_manifest.csv

echo; echo "########## [2/3] 建正确 P4 manifest（攻守双方都过攻击）##########"
$PY - <<'PYEOF'
import pandas as pd
MAN=os.environ["VOICE_WS"]+"/data/manifests"
ra=pd.read_csv(f"{MAN}/real_attacks_manifest.csv")     # 真实+攻击 (label=real)
fa=pd.read_csv(f"{MAN}/fake_attacks_manifest.csv")     # 伪造+攻击 (label=fake)
for c in ["text_source","prompt_utt_id","sha256"]:
    for d in (ra,fa):
        if c not in d.columns: d[c]="na"
cols=["utt_id","audio_path","label","source","clone_model","attack_type","attack_params",
      "text","text_source","prompt_utt_id","speaker_id","sample_rate","duration","split","sha256"]
df=pd.concat([ra[cols],fa[cols]],ignore_index=True)
assert df.utt_id.is_unique, "utt_id 重复"
print(f"攻后真实 {len(ra)} + 攻后伪造 {len(fa)} = {len(df)}")
print(df.groupby(["label","attack_type"]).size().unstack(fill_value=0).to_string())
df.to_csv(f"{MAN}/p4_correct_manifest.csv",index=False)
print("-> p4_correct_manifest.csv")
PYEOF

echo; echo "########## [3/3] 评估：攻后真实 vs 攻后伪造 ##########"
$PY -u eval_detector.py --ckpt $RES/best.pth \
  --manifest $MAN/p4_correct_manifest.csv --split test \
  --by attack_type --n-boot 1000 --frozen-threshold 1.782691 \
  --num-workers 2 --num-threads 8 \
  --out $RES/p4_correct.json
echo "P4_CORRECT_DONE"
