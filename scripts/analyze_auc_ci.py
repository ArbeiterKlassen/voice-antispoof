#!/usr/bin/env python3
"""
§2.5（及 G3）复算：逐族同族 AUC + **说话人分层 bootstrap 95% CI**。

口径（与报告表一致）：
- matched = 该族伪造 vs 同族真实
- CI = metrics.bootstrap_ci_auc（按说话人有放回重采样，n_boot=2000，seed=20260918）
- 表内数值与复算值的差异为蒙特卡洛抖动（实测 ≤0.006），定性判据不受影响。

用法：python scripts/analyze_auc_ci.py [--tags official_AASIST AASISTL gmm]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, "code"))
from metrics import bootstrap_ci_auc                                    # noqa: E402

R = f"{WS}/results/scores"
SETS = {"p4fix": ["mp3-64k", "aac-128k", "opus-32k", "g711-8k", "bandpass-300-3400",
                  "awgn-snr5", "awgn-snr10", "reverb-rt60-0.4", "speed0.9"],
        "p4r": ["codec", "g711", "bandpass", "awgn", "reverb", "speed"]}


def load(p):
    z = np.load(p, allow_pickle=False)
    return pd.DataFrame({"score": z["score"], "label": z["label"].astype(str),
                         "attack": z["attack_type"].astype(str), "spk": z["speaker_id"].astype(str)})


def cell(d, fam):
    f = d[(d.label == "fake") & (d.attack == fam)]
    r = d[(d.label == "real") & (d.attack == fam)]
    if len(f) == 0 or len(r) == 0:
        return None
    s = np.concatenate([f.score.values, r.score.values])
    y = np.concatenate([np.zeros(len(f), int), np.ones(len(r), int)])
    ci = bootstrap_ci_auc(s, y, np.concatenate([f.spk.values, r.spk.values]), n_boot=2000)
    return ci["auc_point"], ci["ci_lo"], ci["ci_hi"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+",
                    default=["official_AASIST", "AASISTL", "gmm"])
    args = ap.parse_args()
    for tag in args.tags:
        for st, fams in SETS.items():
            p = f"{R}/{tag}.la_{st}.npz"
            if not os.path.exists(p):
                continue
            d = load(p)
            print(f"\n【{tag} @ la_{st}】同族 AUC [说话人分层 95% CI]")
            for fam in fams:
                c = cell(d, fam)
                if c is None:
                    continue
                a, lo, hi = c
                flag = "  ← CI 上界 < 0.5" if hi < 0.5 else ("  ← CI 跨界" if lo < 0.5 < hi else "")
                print(f"  {fam:<20}{a:>8.4f}  [{lo:.3f}, {hi:.3f}]{flag}")
    print("\nANALYZE_AUC_CI_DONE")


if __name__ == "__main__":
    main()
