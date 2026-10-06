#!/usr/bin/env bash
# 生成"攻击增广"训练集 —— 针对 P4 测出的两个失效模式
#
# 【为什么要这样设计】
# P4 实测暴露两个失效（2026-10-04）：
#   A. 判别力退化：EER 0% -> 8.7~22.7%
#   B. 标定崩溃：58.5% 的攻击后真实语音被冻结阈值误判为伪造
#
# 只对伪造做攻击增广 → 只能修 A，修不了 B（模型仍会学"攻击伪影=>伪造"）。
# 正确的增广必须**两类都过攻击**：
#   - 攻击后的伪造 标 fake  -> 教模型"伪影不因攻击消失"
#   - 攻击后的真实 标 real  -> 教模型"攻击伪影 != 合成伪影"   ← 修 B 的关键
#
# 输出: data/manifests/aug_attacked_real_manifest.csv (label=real)
#       data/manifests/aug_attacked_fake_manifest.csv (label=fake)
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

cd "$WS"/code
MAN=$WS/data/manifests
ATK=mp3-64k,aac-128k,opus-32k,g711-8k,awgn-snr5,awgn-snr10,bandpass-300-3400,reverb-rt60-0.4,speed0.9

# 源：只在 train 划分采样，保持 speaker-disjoint
$PY - <<'PYEOF'
import pandas as pd
MAN=os.environ["VOICE_WS"]+"/data/manifests"
full=pd.read_csv(f"{MAN}/data_manifest_15col.csv")
tr=full[full.split=="train"]
real=tr[tr.label=="real"].sample(n=min(500,(tr.label=="real").sum()),random_state=20261004)
fake=tr[tr.label=="fake"].sample(n=min(500,(tr.label=="fake").sum()),random_state=20261004)
real.to_csv(f"{MAN}/_aug_real_src.csv",index=False)
fake.to_csv(f"{MAN}/_aug_fake_src.csv",index=False)
print(f"增广源: real {len(real)} 条, fake {len(fake)} 条 (均来自 train 划分)")
print(f"说话人 real {real.speaker_id.nunique()} / fake {fake.speaker_id.nunique()}")
PYEOF

echo "########## [1/2] 攻击后的真实语音 (label=real) ##########"
$PY -u make_attacks.py --src $MAN/_aug_real_src.csv --limit 500 --splits train \
  --num-threads 8 --label real --source aug_real --attacks $ATK \
  --out-manifest $MAN/aug_attacked_real_manifest.csv

echo; echo "########## [2/2] 攻击后的伪造语音 (label=fake) ##########"
$PY -u make_attacks.py --src $MAN/_aug_fake_src.csv --limit 500 --splits train \
  --num-threads 8 --label fake --source aug_fake --attacks $ATK \
  --out-manifest $MAN/aug_attacked_fake_manifest.csv

echo "AUG_TRAIN_DONE"
