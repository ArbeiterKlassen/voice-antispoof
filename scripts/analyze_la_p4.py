#!/usr/bin/env python3
"""
LA-P4 分析 —— 检验预注册判据 H1/H2/H3/H4（见 results/la_p4_prereg.md）

只用已缓存分数 npz，不再前向。

口径（两种都给，差值本身是结论）：
  · mixed   ：该族伪造 vs **全部**真实（真实池含 9 种后处理）
  · matched ：该族伪造 vs **同族**真实（控制后处理这个变量）——鲁棒性该看这个
FPR 在 **LA dev 冻结阈值** 下报（LA dev 无零错误带，阈值唯一：−1.093850）。
"""
import argparse, os, sys
import numpy as np
import pandas as pd

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, "code"))
from metrics import evaluate                                        # noqa: E402
import os
R = f"{WS}/results/scores"


def load(p):
    z = np.load(p, allow_pickle=False)
    return pd.DataFrame({"score": z["score"], "y": z["y"], "label": z["label"],
                         "attack": z["attack_type"], "spk": z["speaker_id"]})


def eer(f, r):
    s = np.concatenate([f.score.values, r.score.values])
    y = np.concatenate([np.zeros(len(f), int), np.ones(len(r), int)])
    return evaluate(s, y)["eer_percent"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=-1.093850,
                    help="LA dev 冻结阈值（EER 交点，唯一确定）")
    ap.add_argument("--tag", default="official_AASIST")
    args = ap.parse_args()
    T = args.threshold

    sets = {k: load(f"{R}/{args.tag}.la_{k}.npz")
            for k in ["clean", "p4fix", "p4r"] if os.path.exists(f"{R}/{args.tag}.la_{k}.npz")}
    if not sets:
        print("没有 LA 结果 npz"); return

    clean = sets.get("clean")
    if clean is not None:
        c_eer = eer(clean[clean.label == "fake"], clean[clean.label == "real"])
        print(f"【干净基线】798 条子集 EER = {c_eer:.3f}%")

    for tag in ["p4fix", "p4r"]:
        if tag not in sets:
            continue
        P = sets[tag]
        r_all = P[P.label == "real"]
        f_all = P[P.label == "fake"]
        pooled = eer(f_all, r_all)
        c_eer = eer(clean[clean.label == "fake"], clean[clean.label == "real"]) if clean is not None else float("nan")
        print("\n" + "=" * 96)
        print(f"【{tag}】并池 EER = {pooled:.3f}%（干净基线 {c_eer:.3f}%）"
              f"   Δ = {pooled - c_eer:+.2f} pp   "
              f"{'✅ H1 成立' if pooled - c_eer >= 5 else '❌ H1 不成立'}")
        print("=" * 96)
        fams = sorted(set(f_all.attack))
        print(f"  {'族':<20}{'n_fake':>7}{'EER(mixed)%':>13}{'EER(matched)%':>15}"
              f"{'真实分中位':>12}{'FPR@冻结':>10}{'干净中位':>10}")
        cm = np.median(clean[clean.label == "real"].score.values) if clean is not None else np.nan
        med_shift = []
        fprs = []
        for fam in fams:
            f = f_all[f_all.attack == fam]
            r = r_all[r_all.attack == fam]
            if len(f) == 0 or len(r) == 0:
                continue
            em = eer(f, r_all)          # mixed
            ec = eer(f, r)              # matched
            med = float(np.median(r.score.values))
            fpr = float((r.score.values < T).mean())
            med_shift.append(med - cm); fprs.append(fpr)
            print(f"  {fam:<20}{len(f):>7}{em:>13.3f}{ec:>15.3f}{med:>12.3f}{fpr:>10.3f}{cm:>10.3f}")
        ms = np.array(med_shift); fp = np.array(fprs)
        print(f"\n  分数中位下移：均值 {ms.mean():+.2f}（最负 {ms.min():+.2f}）；"
              f"FPR@冻结：宏平均 {fp.mean():.3f}，最大 {fp.max():.3f}")
        print(f"  H2（标定崩溃）：中位下移 ≥2 且 FPR ≥0.5 的族数 = "
              f"{int(((ms <= -2) & (fp >= 0.5)).sum())}/{len(ms)} → "
              f"{'✅ 成立' if ((ms<=-2)&(fp>=0.5)).sum() >= 3 else '❌ 不成立'}")
        codec_like = [i for i, f in enumerate(fams) if f in
                      ("mp3-64k", "aac-128k", "opus-32k", "codec")]
        if codec_like:
            print(f"  H4（族间差异）：编解码族中位下移 "
                  f"{np.array([ms[i] for i in codec_like]).mean():+.2f} "
                  f"vs 其余 {np.array([ms[i] for i in range(len(ms)) if i not in codec_like]).mean():+.2f} "
                  f"{'✅ 成立' if abs(np.array([ms[i] for i in codec_like]).mean()) < 2 else '❌'}")
    print("\nANALYZE_LA_P4_DONE")


if __name__ == "__main__":
    main()
