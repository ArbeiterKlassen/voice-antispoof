#!/usr/bin/env bash
# LA-P4PITCH：补音高族 ±2/±4 半音（理论侧 §四.4）
# 与 speed 族机理互补：变速保音高、变音高保时长 ⇒ 分开「时长类线索」与「频谱包络线索」。
# 生成链复用 make_attacks.py（固定参数版），源 = LA-P4 同一批 200 真实 + 598 伪造。
# ⚠️ 这是**预注册后**新增的族（p4fix 已分析完毕），报告里须标注为 post-hoc 追加。
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

cd "$WS"/code
MAN=$WS/data/manifests
ATK=pitch-down2,pitch-up2,pitch-down4,pitch-up4

echo "########## [1/3] 真实 200 × 4 族 ##########"
$PY -u make_attacks.py --src $MAN/_la_p4_real_src.csv --limit 200 --splits test \
  --num-threads 8 --label real --source la_p4pitch --attacks $ATK \
  --out-manifest $MAN/la_p4pitch_real_manifest.csv || exit 2

echo; echo "########## [2/3] 伪造 598 × 4 族 ##########"
$PY -u make_attacks.py --src $MAN/_la_p4_fake_src.csv --limit 600 --splits test \
  --num-threads 8 --label fake --source la_p4pitch --attacks $ATK \
  --out-manifest $MAN/la_p4pitch_fake_manifest.csv || exit 2

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
a=norm(pd.read_csv(f"{MAN}/la_p4pitch_real_manifest.csv"))
b=norm(pd.read_csv(f"{MAN}/la_p4pitch_fake_manifest.csv"))
d=pd.concat([a,b],ignore_index=True)
assert d.utt_id.is_unique, "utt_id 撞车"
# 独立守卫：每族 real/fake 都得齐（防某一族静默全灭）
g=d.groupby(["attack_type","label"]).size().unstack(fill_value=0)
print(g)
assert (g["real"]==200).all() and (g["fake"]==598).all(), "有族数量不对"
d.to_csv(f"{MAN}/la_p4pitch_manifest.csv", index=False)
print(f"la_p4pitch_manifest.csv: {len(d)} 条")
PYEOF
echo "LA_P4PITCH_DONE"
