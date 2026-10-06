#!/usr/bin/env bash
# P4-random：P4-fixed 的对照评测集 —— 同一批源、同样规模，只把「固定参数」换成「逐文件随机」
#
# 【为什么要造它】
# 查出 make_attacks.py 的一个 bug：att_awgn/att_reverb 的 seed 默认 0，
# 而 ATTACKS 表里**没传 seed** → 全项目所有文件的噪声是同一段、
# 所有文件的 RIR 是**同一个滤波器**。
# 于是现有的 "P4 混响鲁棒性" 其实只是「对某一个特定 RIR 的鲁棒性」，
# 而且训练增广与评估用的是**同一个 RIR** —— 臂B 在 reverb 上的恶化
# 可能正是与这个共享 RIR 的交互，而不是混响本身。
#
# 【设计】源完全相同（_p4_real_src 54 条 test 真实 / _p4_src_stratified 150 条 test 伪造），
# 每条源 15 次抽取，参数在区间内随机：
#   rt60~U(0.10,1.20)  snr~U(0,25)  speed~U(0.85,1.15)  码率∈{32,48,64,96,128}
# → 与 P4-fixed 逐项可比：差的就是「固定 vs 随机」这一个变量。
# 注：族集合为 6 族（codec/g711/awgn/reverb/speed/bandpass），
#     不含 P4-fixed 里的 resample-8k（该族在随机脚本里未实现），报告中须写明。
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

cd "$WS"/code
MAN=$WS/data/manifests
RES=$WS/results

echo "########## [0/3] 源文件核对（必须与 P4-fixed 是同一批）##########"
$PY - <<'PYEOF'
import pandas as pd
MAN=os.environ["VOICE_WS"]+"/data/manifests"
r = pd.read_csv(f"{MAN}/_p4_real_src.csv")
f = pd.read_csv(f"{MAN}/_p4_src_stratified.csv")
ref = pd.read_csv(f"{MAN}/p4_correct_manifest.csv")
print(f"  real 源 {len(r)} 条（P4-fixed 产出 810 = 54×15 → {'✅同源' if len(r)==54 else '❌'}）")
print(f"  fake 源 {len(f)} 条（P4-fixed 产出 2250 = 150×15 → {'✅同源' if len(f)==150 else '❌'}）")
print(f"  fake 源的族构成: {dict(f.attack_type.value_counts())}")
print(f"  列: real={list(r.columns)[:8]}...")
assert len(r)==54 and len(f)==150, "源条数与 P4-fixed 不一致，停止"
PYEOF

echo; echo "########## [1/3] 生成随机参数攻击 ##########"
OMP_NUM_THREADS=8 $PY -u make_aug_random.py \
  --real-src $MAN/_p4_real_src.csv --fake-src $MAN/_p4_src_stratified.csv \
  --per-source 15 --split-in test --split-out test \
  --out-root $WS/data/attacks_p4rand --utt-prefix p4r \
  --out-real $MAN/p4r_attacked_real_manifest.csv \
  --out-fake $MAN/p4r_attacked_fake_manifest.csv

echo; echo "########## [2/3] 合并为 p4_random_manifest.csv（与 p4_correct 同结构）##########"
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
ra=norm(pd.read_csv(f"{MAN}/p4r_attacked_real_manifest.csv"))
fa=norm(pd.read_csv(f"{MAN}/p4r_attacked_fake_manifest.csv"))
df=pd.concat([ra,fa],ignore_index=True)
assert df.utt_id.is_unique, "utt_id 重复"
print(f"  攻后真实 {len(ra)} + 攻后伪造 {len(fa)} = {len(df)}  （P4-fixed 为 810+2250=3060）")
print(df.groupby(["label","attack_type"]).size().unstack(fill_value=0).to_string())
df.to_csv(f"{MAN}/p4_random_manifest.csv",index=False)
print("-> p4_random_manifest.csv")
PYEOF

echo; echo "########## [3/3] 待评估（各臂 e4 结果出来后跑）##########"
cat <<'EOS'
  评估命令（每臂一次，CPU 约 11 min；GPU 更快）:
    python score_group.py --ckpt <CKPT> --manifest data/manifests/p4_random_manifest.csv \
      --split test --by attack_type --num-workers 3 --num-threads 8 \
      --save-scores results/scores/<ARM>.p4r.npz --out results/scores/<ARM>.p4r.json
EOS
echo "P4RAND_DONE"
