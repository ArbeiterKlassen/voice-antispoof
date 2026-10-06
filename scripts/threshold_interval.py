#!/usr/bin/env python3
"""
从某个干净 dev 集的分数算「使 EER=0 的阈值区间」与 EER 交点阈值。

为什么需要它：本项目用「冻结阈值」跨集报误报率，但 dev 上 EER 常为 0，
此时使 EER=0 的阈值是一段**区间**，`compute_eer` 的 argmin 只会取到其中一个端点，
于是 FPR@frozen 会系统性偏乐观。报告必须给区间（见 memory: voice-score-shift-not-loss）。

用法:
  python threshold_interval.py --npz results/scores/official_AASIST.la_dev.npz
"""
import argparse, os, sys
import numpy as np

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, "code"))
from metrics import compute_eer                                    # noqa: E402
import os

BONAFIDE = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=False)
    s, y = z["score"], z["y"]
    b, sp = s[y == BONAFIDE], s[y != BONAFIDE]
    eer, thr = compute_eer(b, sp)
    tag = args.label or os.path.basename(args.npz)
    print(f"[{tag}]  真实 {len(b)} / 伪造 {len(sp)}")
    print(f"  真实分数范围 [{b.min():.4f}, {b.max():.4f}]")
    print(f"  伪造分数范围 [{sp.min():.4f}, {sp.max():.4f}]")
    print(f"  compute_eer 给的阈值 = {thr:.6f}  (EER {eer*100:.4f}%)")
    # EER=0 的充要条件：**所有伪造 < thr 且 所有真实 >= thr**
    #   ⇒ thr > max(fake) 且 thr <= min(real)
    # ⚠️ 曾把区间写成 [min_real, max_real]，方向反了（会导致"最保守/最激进"判反）
    lo, hi = sp.max(), b.min()          # 正确区间 = (lo, hi]
    if hi > lo:
        pos = (thr - lo) / (hi - lo) * 100
        print(f"  使 EER=0 的阈值区间 = ({lo:.6f}, {hi:.6f}]  宽 {hi-lo:.4f}")
        print(f"  → compute_eer 落在区间内 {pos:.1f}% 处"
              f"（0%=最保守=误报最低, 100%=最激进=误报最高）")
        if eer == 0:
            print(f"  → EER=0 故阈值欠定：报告须给区间；"
                  f"frozen={thr:.6f} / mid={(lo+hi)/2:.6f}")
        else:
            print(f"  → dev 上 EER={eer*100:.4f}%≠0，说明类间有重叠，"
                  f"EER 交点阈值唯一确定，**无欠定问题**（区间只是理论上的零错误带）")
    else:
        print(f"  不存在零错误阈值带（真实最低分 {hi:.4f} ≤ 伪造最高分 {lo:.4f}）"
              f"→ 阈值由 EER 交点唯一确定，无欠定问题")
    print("THRESHOLD_INTERVAL_DONE")


if __name__ == "__main__":
    main()
