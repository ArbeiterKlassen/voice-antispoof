#!/usr/bin/env bash
# LA 增广训练集：对 LA **train** 划分（不是 test！）抽源并施加后处理，两类都过同等攻击
#
# 【要检验什么】
#   自建轨上「攻击增广」（臂B/D）修好了噪声类的判别力，却**没修标定崩溃**。
#   机制解释：增广打的是判别力，而标定崩溃是分数整体下移 —— 不是同一个问题。
#   在 LA 上重做这个检验：若结论复现，则「增广不能修标定」就不是平凡伪造的产物。
#
# 【源】LA train 划分抽 1500 真实 + 1500 伪造（说话人分层），每源 3 次随机参数攻击
#   → 9,000 条增广（4,500 真实 / 4,500 伪造），与 25,380 条干净训练样本合并
#   ⚠️ 只从 train 抽：从 test 抽就是泄漏（LA 官方划分说话人不相交，必须保持）
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

cd "$WS"/code
MAN=$WS/data/manifests

echo "########## [1/3] 从 LA train 抽源（说话人分层）##########"
$PY - <<'PYEOF'
import pandas as pd
MAN=os.environ["VOICE_WS"]+"/data/manifests"
df=pd.read_csv(f"{MAN}/asvspoof2019_la_manifest.csv")
tr=df[df.split=="train"]
r=tr[tr.label=="real"]; f=tr[tr.label=="fake"]
rs=(r.groupby("speaker_id",group_keys=False)
      .apply(lambda g: g.sample(n=max(1,round(len(g)*1500/len(r))),random_state=20261004)).head(1500))
fs=(f.groupby("attack_type",group_keys=False)
      .apply(lambda g: g.sample(n=max(1,round(len(g)*1500/len(f))),random_state=20261004)).head(1500))
assert set(rs.speaker_id) <= set(tr.speaker_id) and set(fs.speaker_id) <= set(tr.speaker_id), "源必须都来自 train"
rs.to_csv(f"{MAN}/_la_aug_real_src.csv",index=False)
fs.to_csv(f"{MAN}/_la_aug_fake_src.csv",index=False)
print(f"  真实源 {len(rs)}（{rs.speaker_id.nunique()} 说话人）/ 伪造源 {len(fs)}（{f.attack_type.nunique()} 系统）")
print(f"  全部来自 train 划分 ✅")
PYEOF

echo; echo "########## [2/3] 生成随机参数攻击（两类都过同等攻击）##########"
OMP_NUM_THREADS=8 $PY -u make_aug_random.py \
  --real-src $MAN/_la_aug_real_src.csv --fake-src $MAN/_la_aug_fake_src.csv \
  --per-source 3 --split-in train --split-out train \
  --out-root $WS/data/attacks_la_aug --utt-prefix laaug \
  --out-real $MAN/la_aug_real_manifest.csv --out-fake $MAN/la_aug_fake_manifest.csv

echo; echo "########## [3/3] 合并训练 manifest ##########"
$PY - <<'PYEOF'
import pandas as pd
MAN=os.environ["VOICE_WS"]+"/data/manifests"
COLS=["utt_id","audio_path","label","source","clone_model","attack_type","attack_params",
      "text","text_source","prompt_utt_id","speaker_id","sample_rate","duration","split","sha256"]
def norm(d):
    d=d.copy()
    for c in COLS:
        if c not in d.columns: d[c]="na"
    return d[COLS]
full=norm(pd.read_csv(f"{MAN}/asvspoof2019_la_manifest.csv"))
base=full[full.split.isin(["train","dev"])]
tr=base[base.split=="train"]
# dev 用说话人分层子集（与 train_la.sh 同一构造），保证三臂 dev 监控口径一致
dv=base[base.split=="dev"]
frac=3000/len(dv)
sub=(dv.groupby("speaker_id",group_keys=False)
       .apply(lambda g: g.sample(n=max(2,round(len(g)*frac)),random_state=20261004)))
aug=pd.concat([norm(pd.read_csv(f"{MAN}/la_aug_real_manifest.csv")),
               norm(pd.read_csv(f"{MAN}/la_aug_fake_manifest.csv"))],ignore_index=True)
aug["split"]="train"
d=pd.concat([tr,sub,aug],ignore_index=True)
assert d.utt_id.is_unique, "utt_id 重复"
x=d.groupby("speaker_id")["split"].nunique(); assert (x<=1).all(), "speaker 跨 split"
for s in ["train","dev"]:
    y=d[d.split==s]
    print(f"  {s:<6} real {int((y.label=='real').sum()):>6} / fake {int((y.label=='fake').sum()):>6}  说话人 {y.speaker_id.nunique()}")
d.to_csv(f"{MAN}/la_aug_train_manifest.csv",index=False)
print(f"-> la_aug_train_manifest.csv ({len(d)} 条)")
PYEOF
echo "LA_AUG_DONE"
