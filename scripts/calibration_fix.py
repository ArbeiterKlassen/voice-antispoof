#!/usr/bin/env python3
"""
同信道标定实验 —— 「P4 失效主要是标定问题」这一主张的可复算实现。

## 机制（见 memory: voice-score-shift-not-loss）
后处理把 bonafide 分数整体下移（reverb-0.4 中位 −3.88，干净 dev 真实区间 +1.78~+6.60），
于是固定阈值塌陷；但类间排序基本不变（该集最优阈值下 EER 仅 5.33%）。
→ 若允许「按信道标定」，失效应能大幅缓解。

## 协议（关键：不泄漏）
1. 说话人**分半**：一半做校准、另一半评估。校准与评估说话人无交集。
2. 干净参考分布：用该臂在 **P1 test** 上的真实分数中位（独立于 P4）。
3. 校准规则只有一步：`thr_cal = thr_clean + (该校准组同信道真实分数中位 − 干净真实分数中位)`
   —— 即「知道音频过了哪种后处理链」时的最小标定动作。
4. 评估：在**另一半说话人**上算 FPR，并按说话人做 bootstrap（配对比较）。

## 诚实边界（脚本会打印）
- 测试说话人少时 CI 很粗 → 只报方向与量级，不给精确值。
- 该规则假设部署方知道信道类型；不知道时需另做（不在本脚本范围）。
- 噪声很重的条件（awgn-snr5/10）存在**真实重叠**，标定救不回，须与纯标定问题分开说。

用法:
  python calibration_fix.py --arm A_best \
     --clean results/scores/A_best.p1.npz \
     --p4 results/scores/A_best.p4.npz \
     --threshold 1.782691
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import os

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, "code"))


def load(p):
    z = np.load(p, allow_pickle=False)
    return pd.DataFrame({"score": z["score"], "y": z["y"], "label": z["label"],
                         "attack": z["attack_type"], "spk": z["speaker_id"]})


def macro_fpr(P, cal_spk, test_spk, thr, mu_clean):
    """返回 (校正前宏平均FPR, 校正后宏平均FPR, 逐条件明细)"""
    before, after, rows = [], [], []
    for fam in sorted(P.attack.unique()):
        if fam == "none":
            continue
        r = P[(P.label == "real") & (P.attack == fam)]
        rc, rt = r[r.spk.isin(cal_spk)], r[r.spk.isin(test_spk)]
        if len(rc) == 0 or len(rt) == 0:
            continue
        shift = np.median(rc.score.values) - mu_clean
        b = float((rt.score.values < thr).mean())
        a = float((rt.score.values < thr + shift).mean())
        before.append(b); after.append(a)
        rows.append((fam, len(rt), b, a, shift))
    return float(np.mean(before)), float(np.mean(after)), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--clean", required=True, help="该臂 P1 test 的 scores npz（干净真实分布）")
    ap.add_argument("--p4", required=True, help="该臂 P4 的 scores npz（攻击后）")
    ap.add_argument("--threshold", type=float, default=1.782691)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261004)
    args = ap.parse_args()

    C = load(args.clean)
    P = load(args.p4)
    clean_real = C[C.label == "real"].score.values
    mu_clean = float(np.median(clean_real))
    thr = args.threshold

    spks = sorted(P.spk.unique())
    if len(spks) < 2:
        print("说话人不足，无法分半"); return
    half = len(spks) // 2
    CAL, TEST = spks[:half], spks[half:]
    print(f"臂 {args.arm}")
    print(f"  干净参考({os.path.basename(args.clean)} 真实,n={len(clean_real)}) 中位 = {mu_clean:.3f}；基线阈值 {thr:.6f}")
    print(f"  校准说话人 {CAL} / 评估说话人 {TEST}")
    b, a, rows = macro_fpr(P, CAL, TEST, thr, mu_clean)
    print(f"\n  {'条件':<20}{'n_test真实':>11}{'校正前FPR':>11}{'校正后FPR':>11}{'平移量':>9}")
    for fam, n, x, y, s in rows:
        print(f"  {fam:<20}{n:>11}{x:>11.3f}{y:>11.3f}{s:>9.2f}")
    print(f"  {'【宏平均】':<19}{'':>11}{b:>11.3f}{a:>11.3f}")

    rng = np.random.RandomState(args.seed)
    bs = []
    for _ in range(args.n_boot):
        samp = list(rng.choice(TEST, size=len(TEST), replace=True))
        bs.append(macro_fpr(P, CAL, samp, thr, mu_clean)[:2])
    bs = np.array(bs)
    lo, hi = np.percentile(bs[:, 0], [2.5, 97.5])
    lo2, hi2 = np.percentile(bs[:, 1], [2.5, 97.5])
    d = bs[:, 0] - bs[:, 1]
    print(f"\n  bootstrap（{args.n_boot} 次，按评估说话人重采样；仅 {len(TEST)} 位 → CI 很粗）")
    print(f"    校正前 {bs[:,0].mean():.3f} [95% {lo:.3f}, {hi:.3f}]")
    print(f"    校正后 {bs[:,1].mean():.3f} [95% {lo2:.3f}, {hi2:.3f}]")
    print(f"    配对差 {d.mean():.3f} [95% {np.percentile(d,2.5):.3f}, {np.percentile(d,97.5):.3f}]"
          f"  为正比例 {(d>0).mean():.3f}")
    print("\n  边界：只给方向与量级；awgn 重噪声类属真实重叠，标定救不回，须分开陈述。")
    print("CALIB_DONE")


if __name__ == "__main__":
    main()
