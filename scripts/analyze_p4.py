#!/usr/bin/env python3
"""
P4 后分析 —— 只用已缓存的分数 npz，不再前向。

解决的三个方法论问题：
  1. **口径混杂**：既有 P4 口径是「某族伪造 vs 全部真实（15 族混合）」。
     真实池里混着 15 种攻击，而伪造只有 1 种 —— 这个差本身会偏。
     本脚本**同时**给「混合真实池」与「同族真实池」两列，两者之差本身就是结论。
  2. **单次实现的伪鲁棒**：make_attacks.py 的 seed 恒为 0 →
     全项目共用一个 RIR / 一段噪声。故比对 P4-fixed 与 P4-random（逐文件随机参数）
     的差距，衡量「之前的鲁棒性有多少是特定实现的巧合」。
  3. **冻结阈值下的误报**：判别力（EER）与标定（FPR@frz）是两个独立维度，分开报。

用法:
  python analyze_p4.py --scores results/scores --arms A_last B_last D_last
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, "code"))
from metrics import evaluate                                       # noqa: E402
import os

BONAFIDE = 1


def load_npz(p):
    z = np.load(p, allow_pickle=False)
    return pd.DataFrame({"utt_id": z["utt_id"], "score": z["score"], "y": z["y"],
                         "speaker": z["speaker_id"], "attack": z["attack_type"],
                         "label": z["label"]})


def _eer(f, r):
    """f/r = fake/real 的 DataFrame。score 越大越像 bonafide(real)，y: real=1"""
    s = np.concatenate([f.score.values, r.score.values])
    y = np.concatenate([np.zeros(len(f), dtype=int), np.ones(len(r), dtype=int)])
    return evaluate(s, y)["eer_percent"]


def eer_mixed(df, fam):
    """某族伪造 vs 全部真实（既有口径）"""
    f = df[(df.label == "fake") & (df.attack == fam)]
    r = df[df.label == "real"]
    if len(f) == 0 or len(r) == 0:
        return np.nan, len(f), len(r)
    return _eer(f, r), len(f), len(r)


def eer_matched(df, fam):
    """某族伪造 vs 同族真实（本脚本新增口径）"""
    f = df[(df.label == "fake") & (df.attack == fam)]
    r = df[(df.label == "real") & (df.attack == fam)]
    if len(f) == 0 or len(r) == 0:
        return np.nan, len(f), len(r)
    return _eer(f, r), len(f), len(r)


def fpr_frozen(df, fam, frz):
    r = df[(df.label == "real") & (df.attack == fam)]
    if len(r) == 0:
        return np.nan
    return float((r.score.values < frz).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=f"{WS}/results/scores")
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--frozen-threshold", type=float, default=1.782691)
    ap.add_argument("--mid-threshold", type=float, default=1.616711,
                    help="P1 dev 上使 EER=0 的阈值区间的中点（最大间隔选择）。"
                         "⚠️ 区间定义 = (max_fake, min_real] = (1.4504, 1.7830]，宽仅 0.33；"
                         "现行冻结 1.782691 落在**最激进**端点，故 FPR@frozen 是**上界**。"
                         "（曾把区间算成 [1.783, 6.601] 并由此得出非法中点 4.1920，已更正）")
    args = ap.parse_args()

    D = args.scores
    arms = args.arms or sorted({f[:-len(".p4.npz")] for f in os.listdir(D) if f.endswith(".p4.npz")})
    print(f"臂: {arms}")

    # ---- 表1：P4-fixed vs P4-random（逐族，混合真实池口径）----
    for tag, suf in [("P4-fixed（单一实现）", ".p4.npz"), ("P4-random（逐文件随机）", ".p4r.npz")]:
        have = [a for a in arms if os.path.exists(f"{D}/{a}{suf}")]
        if not have:
            print(f"\n[{tag}] 无数据（{suf} 缺失）"); continue
        data = {a: load_npz(f"{D}/{a}{suf}") for a in have}
        fams = sorted(set(data[have[0]].attack))
        fams = [f for f in fams if f != "none"]
        print("\n" + "=" * 100)
        print(f"{tag} —— 逐族 EER%（伪造=该族 / 真实=全部真实）")
        print("=" * 100)
        print(f"  {'族':<20} " + "".join(f"{a:>12}" for a in have) + f"{'族内真实数':>12}")
        for f in fams:
            line = f"  {f:<20} "
            n_r = 0
            for a in have:
                v, nf, nr = eer_mixed(data[a], f)
                n_r = nr
                line += f"{v:>12.3f}"
            print(line + f"{n_r:>12}")
        # 并池（所有伪造 vs 所有真实）
        line = f"  {'【并池】':<18} "
        for a in have:
            d = data[a]
            line += f"{evaluate(d.score.values, d.y.values)['eer_percent']:>12.3f}"
        print(line)

        # ---- 表2：同族真实池口径 ----
        print(f"\n  {'同族真实池口径 EER%':<20} " + "".join(f"{a:>12}" for a in have))
        for f in fams:
            line = f"  {f:<20} "
            for a in have:
                v, nf, nr = eer_matched(data[a], f)
                line += f"{v:>12.3f}"
            print(line)

        # ---- 表3：冻结阈值下的同族误报率（两列：端点=下界 / 中点=最大间隔选择）----
        print(f"\n  {'真实误报率（同族）':<20} " + "".join(f"{a:>14}" for a in have))
        print(f"  {'':<20} " + "".join(f"{'frz / mid':>14}" for a in have))
        for f in fams:
            line = f"  {f:<20} "
            for a in have:
                v1 = fpr_frozen(data[a], f, args.frozen_threshold)
                v2 = fpr_frozen(data[a], f, args.mid_threshold)
                line += f"{v1:>6.3f}/{v2:<7.3f}"
            print(line)
        print(f"  ↑ frz={args.frozen_threshold:.6f}（EER=0 区间**最激进**端点 ⇒ FPR 上界）/ "
              f"mid={args.mid_threshold:.6f}（区间中点，最大间隔）")
        print("    合法区间 = (max_fake, min_real]，宽仅 0.33；两列之差反映阈值选择的影响，实测很小")

    # ---- 表4：固定 vs 随机 的差值（核心问题：伪鲁棒有多少）----
    if all(os.path.exists(f"{D}/{a}.p4.npz") and os.path.exists(f"{D}/{a}.p4r.npz") for a in arms):
        print("\n" + "=" * 100)
        print("固定 vs 随机 的 EER 差（Δ = 随机 − 固定；正值 = 固定版高估了鲁棒性）")
        print("=" * 100)
        for a in arms:
            d1, d2 = load_npz(f"{D}/{a}.p4.npz"), load_npz(f"{D}/{a}.p4r.npz")
            fams = sorted(set(d1.attack) & set(d2.attack)); fams = [f for f in fams if f != "none"]
            print(f"\n  [{a}]")
            for f in fams:
                v1, _, _ = eer_mixed(d1, f); v2, _, _ = eer_mixed(d2, f)
                print(f"    {f:<20} 固定 {v1:>7.3f}  随机 {v2:>7.3f}  Δ {v2-v1:>+8.3f}")
    print("\nANALYZE_P4_DONE")


if __name__ == "__main__":
    main()
