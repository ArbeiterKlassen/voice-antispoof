#!/usr/bin/env bash
# 相位声码器恒等臂（vocoder-id）生成 —— 空对照，预注册见 results/vocoder_id_control_prereg.md
# n_steps=0：librosa 不短路（已实测逐点差 >0），走完整 time_stretch(rate=1)+resample
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

cd "$WS"/code
MAN=$WS/data/manifests

echo "########## [1/3] 真实 200 ##########"
nice -n 10 $PY -u make_attacks.py --src $MAN/_la_p4_real_src.csv --limit 200 --splits test \
  --num-threads 4 --label real --source la_p4vocid --attacks vocoder-id \
  --out-manifest $MAN/la_p4vocid_real_manifest.csv || exit 2

echo; echo "########## [2/3] 伪造 598 ##########"
nice -n 10 $PY -u make_attacks.py --src $MAN/_la_p4_fake_src.csv --limit 600 --splits test \
  --num-threads 4 --label fake --source la_p4vocid --attacks vocoder-id \
  --out-manifest $MAN/la_p4vocid_fake_manifest.csv || exit 2

echo; echo "########## [3/3] 合并 ##########"
$PY - <<PYEOF
import pandas as pd
MAN="$MAN"
COLS=["utt_id","audio_path","label","source","clone_model","attack_type","attack_params",
      "text","text_source","prompt_utt_id","speaker_id","sample_rate","duration","split","sha256"]
def norm(d):
    d=d.copy()
    for c in COLS:
        if c not in d.columns: d[c]="na"
    return d[COLS]
a=norm(pd.read_csv(f"{MAN}/la_p4vocid_real_manifest.csv"))
b=norm(pd.read_csv(f"{MAN}/la_p4vocid_fake_manifest.csv"))
d=pd.concat([a,b],ignore_index=True)
assert d.utt_id.is_unique
assert len(a)==200 and len(b)==598, f"数量不对 real={len(a)} fake={len(b)}"
d.to_csv(f"{MAN}/la_p4vocid_manifest.csv", index=False)
print(f"la_p4vocid_manifest.csv: {len(d)} 条")
PYEOF
echo "LA_P4VOCID_DONE"
