#!/usr/bin/env bash
# LA-P4：把「后处理鲁棒性」实验搬到公开基准上（这是本项目主张唯一能有文献刻度的落点）
#
# 【为什么值得做】
#   自建轨的 P4 结论（判别力 vs 标定、分数整体下移）本身是好的，
#   但挂在"自生成 signalproc 伪造"上无法与任何人比较。
#   搬到 LA 后：干净基线 = 官方 AASIST 的 0.83%，后处理后的退化就可比了。
#
# 【设计】
#   源：LA **test** 划分里按说话人分层抽 200 真实 + 600 伪造（13 个攻击系统按比例）
#   攻击：与自建轨 P4 相同的族（固定参数版 + 逐文件随机参数版各一套）
#   → 每套 (200+600)×6 族 ≈ 4800 条；两套 ≈ 9600 条。
#   真实与伪造**都过同样的后处理**（这是 P4 首版最大的坑，不再犯）。
#
# 【口径声明（写进报告）】
#   · 这是 LA eval 的**子集**，不是全量 71237 → 干净基线须用同一子集另测一次
#   · 后处理族 6 类（codec/g711/awgn/reverb/speed/bandpass），
#     与自建轨 P4-fixed 的 15 种不完全对应，跨轨对比时须说明
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

cd "$WS"/code
MAN=$WS/data/manifests
LA=$MAN/asvspoof2019_la_manifest.csv
N_REAL=${1:-200}
N_FAKE=${2:-600}

echo "########## [1/3] 抽 LA test 子集（说话人分层、攻击系统按比例）##########"
$PY - <<PYEOF
import pandas as pd
MAN="$MAN"
df = pd.read_csv("$LA")
te = df[df.split == "test"]
real = te[te.label == "real"]
fake = te[te.label == "fake"]
# 真实：按说话人分层抽
r = (real.groupby("speaker_id", group_keys=False)
         .apply(lambda g: g.sample(n=max(1, round(len(g)*$N_REAL/len(real))), random_state=20261004))
         .head($N_REAL))
# 伪造：按攻击系统分层抽（保持 A07–A19 比例）
f = (fake.groupby("attack_type", group_keys=False)
         .apply(lambda g: g.sample(n=max(1, round(len(g)*$N_FAKE/len(fake))), random_state=20261004))
         .head($N_FAKE))
print(f"  真实 {len(r)} 条（{r.speaker_id.nunique()} 位说话人）")
print(f"  伪造 {len(f)} 条，攻击系统分布: {dict(f.attack_type.value_counts())}")
r.to_csv(f"{MAN}/_la_p4_real_src.csv", index=False)
f.to_csv(f"{MAN}/_la_p4_fake_src.csv", index=False)
PYEOF

echo; echo "########## [2/3] 固定参数版（沿用 P4-fixed 的族与参数）##########"
ATK=mp3-64k,aac-128k,opus-32k,g711-8k,awgn-snr5,awgn-snr10,bandpass-300-3400,reverb-rt60-0.4,speed0.9
$PY -u make_attacks.py --src $MAN/_la_p4_real_src.csv --limit 200 --splits test \
  --num-threads 8 --label real --source la_p4fix --attacks $ATK \
  --out-manifest $MAN/la_p4fix_real_manifest.csv
$PY -u make_attacks.py --src $MAN/_la_p4_fake_src.csv --limit 600 --splits test \
  --num-threads 8 --label fake --source la_p4fix --attacks $ATK \
  --out-manifest $MAN/la_p4fix_fake_manifest.csv

echo; echo "########## [3/3] 随机参数版（逐文件随机族与参数）##########"
# per-source = 9，与固定版同量级；前缀/落盘目录都独立，避免与自建轨撞名
OMP_NUM_THREADS=8 $PY -u make_aug_random.py \
  --real-src $MAN/_la_p4_real_src.csv --fake-src $MAN/_la_p4_fake_src.csv \
  --per-source 9 --split-in test --split-out test \
  --out-root $WS/data/attacks_la --utt-prefix lap4r \
  --out-real $MAN/la_p4r_real_manifest.csv --out-fake $MAN/la_p4r_fake_manifest.csv

echo; echo "########## 合并成两个评测 manifest ##########"
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
for tag,(rf,ff) in {"p4fix":("la_p4fix_real_manifest.csv","la_p4fix_fake_manifest.csv"),
                    "p4r":  ("la_p4r_real_manifest.csv","la_p4r_fake_manifest.csv")}.items():
    a=norm(pd.read_csv(f"{MAN}/{rf}")); b=norm(pd.read_csv(f"{MAN}/{ff}"))
    d=pd.concat([a,b],ignore_index=True)
    assert d.utt_id.is_unique
    d.to_csv(f"{MAN}/la_{tag}_manifest.csv", index=False)
    print(f"  la_{tag}_manifest.csv: {len(d)} 条  real {int((d.label=='real').sum())} / fake {int((d.label=='fake').sum())}")
    print(f"     族分布: {dict(d.attack_type.value_counts())}")
PYEOF
echo "LA_P4_DONE"
