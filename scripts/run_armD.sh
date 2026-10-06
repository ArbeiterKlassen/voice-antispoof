#!/usr/bin/env bash
# 臂D 全链路：参数随机化增广 → 训练 → 与臂A/臂B 同协议评估
#
# 【本次要回答的问题】
#   臂B（500源 × 9个固定条件）在噪声类上大幅改善（awgn-snr5 22.00→10.00），
#   但混响/变速**训过反而变差**（reverb-rt0.4 5.33→27.33，且不在增广集里的
#   reverb-0.2/0.8 也一起变差 4.00→13.33 / 21.33→29.33）。
#   假设A：模型记住了「具体参数点」→ 参数随机化可修。
#   假设B：攻击增广本身对某些族有害（如破坏伪造线索）→ 参数随机化修不了。
#   臂D 是这两个假设的判别实验：族相同、仅把固定参数换成连续区间采样。
#
# 【协议对齐（血泪点）】
#   臂A：traindev_manifest.csv  --epochs 15 --batch-size 16
#   臂B：aug_train_manifest.csv --epochs 20 --batch-size 16   ← 增广 4500/类(500源×9条件)
#   → 臂D 必须 --epochs 20 --batch-size 16，且增广**条数同为 4500/类**，
#     否则「随机化有效」不可归因（量差也是变量）。
#   → 另补臂A20（无增广、20ep）：消掉「臂A只有15ep」这个预算混淆。
#
# 【固定预算原则】
#   dev EER 早已饱和到 0.000%（臂A ep5 / 臂B ep4），best.pth 的选择实为噪声驱动。
#   故主结果用 **last.pth（固定20epoch）**，best.pth 一并报出，两者之差本身是结论。
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

cd "$WS"/code
MAN=$WS/data/manifests
RES=$WS/results
GPU=4                      # GPU6 已被他人占用(10GB/99%)，用回本项目的 GPU4
FRZ=1.782691               # P1 dev 冻结阈值（verify_equiv.sh 已自检来源）
mkdir -p $RES/scores

echo "########## [1/7] 生成参数随机化增广（per-source 9 → 4500/类，对齐臂B）##########"
OMP_NUM_THREADS=8 $PY -u make_aug_random.py --per-source 9 --num-threads 8

echo; echo "########## [2/7] 校验：参数是否真随机 + 条数是否对齐 ##########"
$PY - <<'PYEOF'
import pandas as pd, re, sys
MAN=os.environ["VOICE_WS"]+"/data/manifests"
ok = True
for tag in ["real", "fake"]:
    df = pd.read_csv(f"{MAN}/augr_attacked_{tag}_manifest.csv")
    print(f"\n--- augr_attacked_{tag}: {len(df)} 条（臂B 同族为 4500）---")
    print("    族分布:", dict(df.attack_type.value_counts()))
    if not (4300 <= len(df) <= 4500):
        print(f"    ❌ 条数偏离臂B太多（{len(df)} vs 4500），增广量本身变成变量"); ok = False
    for fam, pat, lo, hi in [("awgn", r"snr([\d.]+)", 0, 25),
                             ("reverb", r"rt60-([\d.]+)", 0.10, 1.20),
                             ("speed", r"speed([\d.]+)", 0.85, 1.15)]:
        s = df[df.attack_type == fam].attack_params
        if len(s) == 0:
            print(f"    ❌ 族 {fam} 缺失"); ok = False; continue
        v = [float(re.search(pat, x).group(1)) for x in s if re.search(pat, x)]
        uniq = len(set(round(x, 2) for x in v))
        good = uniq > 20 and lo <= min(v) and max(v) <= hi + 1e-6
        print(f"    {fam:<8} n={len(v):<5} 取值 {min(v):.2f}~{max(v):.2f} 不同值 {uniq:<4}"
              f" {'✅' if good else '❌ 仍是固定点/越界'}")
        ok &= good
    # 码率/类型是否真的在变
    c = df[df.attack_type == "codec"].attack_params
    if len(c): print(f"    codec    n={len(c):<5} 不同参数值 {c.nunique():<4} "
                     f"{'✅' if c.nunique() > 10 else '❌'}")
print(f"\n参数随机化+配额判定: {'✅ 通过' if ok else '❌ 未通过 —— 停止，先修生成'}")
sys.exit(0 if ok else 1)
PYEOF
[ $? -ne 0 ] && { echo "生成校验未通过，中止链路"; exit 1; }

echo; echo "########## [3/7] 建训练集 augr_train_manifest.csv ##########"
$PY - <<'PYEOF'
import pandas as pd
MAN=os.environ["VOICE_WS"]+"/data/manifests"
COLS=["utt_id","audio_path","label","source","clone_model","attack_type","attack_params",
      "text","text_source","prompt_utt_id","speaker_id","sample_rate","duration","split","sha256"]
def norm(d):
    d = d.copy()
    for c in COLS:
        if c not in d.columns: d[c] = "na"
    return d[COLS]
base = norm(pd.read_csv(f"{MAN}/traindev_manifest.csv"))          # 与臂A/臂B 同一基底
aug  = pd.concat([norm(pd.read_csv(f"{MAN}/augr_attacked_real_manifest.csv")),
                  norm(pd.read_csv(f"{MAN}/augr_attacked_fake_manifest.csv"))], ignore_index=True)
aug["split"] = "train"                                            # 增广只进 train
df = pd.concat([base, aug], ignore_index=True)
assert df.utt_id.is_unique, "utt_id 重复"
x = df.groupby("speaker_id")["split"].nunique(); assert (x <= 1).all(), "speaker 跨 split"
for s in ["train", "dev"]:
    sub = df[df.split == s]
    print(f"  {s:<6} real {int((sub.label=='real').sum()):>5} / fake {int((sub.label=='fake').sum()):>5}")
# 与臂B逐项对照（漏掉这行就会出现「量差被当成方法差」）
ref = pd.read_csv(f"{MAN}/aug_train_manifest.csv")
r = ref[ref.split=="train"]
print(f"  臂B 对照: train real {int((r.label=='real').sum())} / fake {int((r.label=='fake').sum())}")
df.to_csv(f"{MAN}/augr_train_manifest.csv", index=False)
print(f"-> augr_train_manifest.csv ({len(df)} 条)")
PYEOF

echo; echo "########## [4/7] 臂D：随机化增广 20ep @GPU$GPU ##########"
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 $PY -u train_detector.py \
  --manifest $MAN/augr_train_manifest.csv --epochs 20 --batch-size 16 --num-workers 6 \
  --out-dir $RES/det_augr

echo; echo "########## [5/7] 臂A20：无增广同协议 20ep（消预算混淆）##########"
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 $PY -u train_detector.py \
  --manifest $MAN/traindev_manifest.csv --epochs 20 --batch-size 16 --num-workers 6 \
  --out-dir $RES/det_signalproc20

echo; echo "########## [6/7] 评估（P1 域内 + P4 逐攻击），每臂 best 与 last 都评 ##########"
eval_one() {
  local name=$1 ckpt=$2
  echo "--- eval $name <- $ckpt"
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 $PY -u score_group.py --ckpt "$ckpt" \
    --manifest $MAN/p4_correct_manifest.csv --split test --by attack_type \
    --frozen-threshold $FRZ --n-boot 1000 --num-workers 3 --num-threads 8 --batch-size 16 \
    --save-scores $RES/scores/$name.p4.npz --out $RES/scores/$name.p4.json >/dev/null 2>&1 \
    && echo "    P4 ok" || echo "    ❌ P4 失败"
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 $PY -u score_group.py --ckpt "$ckpt" \
    --manifest $MAN/data_manifest_15col.csv --split test \
    --frozen-threshold $FRZ --n-boot 1000 --num-workers 3 --num-threads 8 --batch-size 16 \
    --save-scores $RES/scores/$name.p1.npz --out $RES/scores/$name.p1.json >/dev/null 2>&1 \
    && echo "    P1 ok" || echo "    ❌ P1 失败"
}
# 两路并行（每路 8 线程，合计 16，共享机上限内）
( eval_one A_last    $RES/det_signalproc/last.pth ) &
( eval_one A20_best  $RES/det_signalproc20/best.pth ) & wait
( eval_one A20_last  $RES/det_signalproc20/last.pth ) &
( eval_one B_best    $RES/det_aug/best.pth ) & wait
( eval_one B_last    $RES/det_aug/last.pth ) &
( eval_one D_best    $RES/det_augr/best.pth ) & wait
eval_one D_last $RES/det_augr/last.pth
# A_best 已由 verify_equiv.sh 产出（同一 ckpt，且做过对拍）

echo; echo "########## [7/7] 汇总 ##########"
$PY $RES/../scripts/summarize_arms.py --detail
echo "ARM_D_CHAIN_DONE"
