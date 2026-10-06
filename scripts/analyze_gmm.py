#!/usr/bin/env python3
"""
GMM 第三后端分析：G1（判别力门槛）/ G2（反转判定）/ G3（三后端并表）
判据见 results/gmm_prereg.md（预注册在先）。
"""
import os
import sys

import numpy as np
import pandas as pd

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, "code"))
from metrics import evaluate, bootstrap_ci_auc                          # noqa: E402

R = f"{WS}/results/scores"
FAMS = ["mp3-64k", "aac-128k", "opus-32k", "g711-8k", "bandpass-300-3400",
        "awgn-snr5", "awgn-snr10", "reverb-rt60-0.4", "speed0.9"]
INV_FAMS = ["reverb-rt60-0.4", "speed0.9"]


def load(p):
    z = np.load(p, allow_pickle=False)
    return pd.DataFrame({"score": z["score"], "label": z["label"].astype(str),
                         "attack": z["attack_type"].astype(str), "spk": z["speaker_id"].astype(str)})


def matched(d, fam):
    f = d[(d.label == "fake") & (d.attack == fam)]
    r = d[(d.label == "real") & (d.attack == fam)]
    s = np.concatenate([f.score.values, r.score.values])
    y = np.concatenate([np.zeros(len(f), int), np.ones(len(r), int)])
    m = evaluate(s, y)
    # G2 判据的「CI 上界」是 **AUC 的 CI**（audit: 曾用 bootstrap_ci_eer 串口径）
    ci = bootstrap_ci_auc(s, y, np.concatenate([f.spk.values, r.spk.values]), n_boot=2000)
    return m, ci


def main():
    # ---------- G1 ----------
    z = np.load(f"{R}/gmm.la_full.npz", allow_pickle=False)
    ev = evaluate(z["score"], z["y"])
    print(f"[G1-a] LA eval 全量 71237：EER = {ev['eer_percent']:.3f}%  AUC = {ev['auc']:.4f}")
    ok_eval = ev["eer_percent"] <= 18.0
    print(f"        已知答案带 [6%,18%]：{'✅ 在带内' if 6.0 <= ev['eer_percent'] <= 18.0 else ('⚠️ 低于 6%（异常好，需查泄漏）' if ev['eer_percent'] < 6 else '❌ 超 18%，实现有误')}")

    pf = load(f"{R}/gmm.p4fix.npz")
    print(f"\n[G1-b] 编解码族同族 AUC（门槛 ≥0.95，证明判别力在）：")
    g1_ok = ok_eval
    for fam in ["mp3-64k", "aac-128k", "opus-32k"]:
        m, _ = matched(pf, fam)
        good = m["auc"] >= 0.95
        g1_ok = g1_ok and good
        print(f"        {fam:<12} AUC {m['auc']:.4f}  {'✅' if good else '❌'}")
    print(f"  ⇒ G1 {'✅ 过' if g1_ok else '❌ 不过'}（不过则不得用于任何反转结论）")

    # ---------- G2 ----------
    print(f"\n[G2] 反转判定（仅 G1 过后有效；门槛：AUC<0.5 且 CI 上界<0.5）：")
    g2 = {}
    for st in ["p4fix", "p4r"]:
        try:
            d = load(f"{R}/gmm.{'p4fix' if st == 'p4fix' else 'p4r'}.npz")
        except FileNotFoundError:
            continue
        fams = INV_FAMS if st == "p4fix" else ["reverb", "speed"]
        for fam in fams:
            m, ci = matched(d, fam)
            hi = ci.get("ci_hi", np.nan)
            verdict = ("反转稳健" if (m["auc"] < 0.5 and hi < 0.5) else
                       "方向一致、不稳健" if m["auc"] < 0.5 else "正常")
            g2[f"{st}:{fam}"] = (m["auc"], ci.get("ci_lo", np.nan), hi, verdict)
            print(f"        {st:<6}{fam:<20} AUC {m['auc']:.4f}  "
                  f"CI [{ci.get('ci_lo', float('nan')):.3f}, {hi:.3f}]  {verdict}")

    # 音高/vocid
    for tag, fams in [("p4pitch", ["pitch-down2", "pitch-down4", "pitch-up2", "pitch-up4"]),
                      ("p4vocid", ["vocoder-id"])]:
        p = f"{R}/gmm.{tag}.npz"
        if not os.path.exists(p):
            continue
        d = load(p)
        for fam in fams:
            m, ci = matched(d, fam)
            print(f"        {tag:<8}{fam:<14} AUC {m['auc']:.4f}  "
                  f"CI [{ci.get('ci_lo', float('nan')):.3f}, {ci.get('ci_hi', float('nan')):.3f}]")

    # ---------- G3 ----------
    print(f"\n[G3] 三后端并表（同族 AUC；官 AASIST / AASIST-L / LFCC-GMM）：")
    print(f"        {'族':<20}{'AASIST':>9}{'AASISTL':>9}{'GMM':>9}")
    for fam in FAMS:
        row = []
        for tag in ["official_AASIST", "AASISTL"]:
            p = f"{R}/{tag}.la_p4fix.npz"
            d = load(p)
            row.append(matched(d, fam)[0]["auc"])
        row.append(matched(pf, fam)[0]["auc"])
        print(f"        {fam:<20}{row[0]:>9.4f}{row[1]:>9.4f}{row[2]:>9.4f}")
    print("ANALYZE_GMM_DONE")


if __name__ == "__main__":
    main()
