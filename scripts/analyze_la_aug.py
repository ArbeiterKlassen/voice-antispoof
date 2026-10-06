#!/usr/bin/env python3
"""
LA 增广臂 H5/H6 分析（判据见报告 §6.2，**预注册在先**）。

H5：增广**改善** awgn 族同族 EER（预期 11→<8）、**恶化** reverb/speed（预期 72→>72）、
    干净 EER 恶化 <1pp。
H6：增广**改善** reverb/speed 的 FPR@（各自臂的 dev 冻结阈值）（预期 1.00→<0.6）
    但**不改**其同族 EER。

口径：同族（该族伪造 vs 同族真实）；阈值 = 各臂**自己**的 LA dev EER 交点阈值。
两臂唯一差异 = 训练集是否含增广（seed/epochs/batch/init 全同）。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, "code"))
from metrics import evaluate                                       # noqa: E402

R = f"{WS}/results/scores"
FAMS_DECISIVE = ["awgn-snr5", "awgn-snr10", "reverb-rt60-0.4", "speed0.9"]
FAMS_CONTROL = ["mp3-64k", "aac-128k", "opus-32k", "g711-8k"]


def load(p):
    z = np.load(p, allow_pickle=False)
    return pd.DataFrame({"score": z["score"], "label": z["label"],
                         "attack": z["attack_type"], "spk": z["speaker_id"]})


def eer_auc(f, r):
    s = np.concatenate([f.score.values, r.score.values])
    y = np.concatenate([np.zeros(len(f), int), np.ones(len(r), int)])
    m = evaluate(s, y)
    return m["eer_percent"], m["auc"]


def main():
    out = {}
    for arm in ["ctl", "aug"]:
        dev = load(f"{R}/la_arm_{arm}.dev.npz")
        s = np.concatenate([dev[dev.label == "fake"].score.values,
                            dev[dev.label == "real"].score.values])
        y = np.concatenate([np.zeros((dev.label == "fake").sum(), int),
                            np.ones((dev.label == "real").sum(), int)])
        m = evaluate(s, y)
        out[arm] = {"dev": m, "thr": m["eer_threshold"]}
        print(f"【臂 {arm.upper()}】dev 全量 EER = {m['eer_percent']:.3f}%  "
              f"（冻结阈值 {m['eer_threshold']:.6f}）  AUC {m['auc']:.5f}")

    cl = {a: load(f"{R}/la_arm_{a}.p4clean.npz") for a in ["ctl", "aug"]}
    pf = {a: load(f"{R}/la_arm_{a}.p4fix.npz") for a in ["ctl", "aug"]}

    print(f"\n{'族':<20}{'CTL EER':>9}{'AUG EER':>9}{'Δ':>8}   "
          f"{'CTL FPR':>9}{'AUG FPR':>9}{'Δ':>8}")
    rows = {}
    for fam in FAMS_DECISIVE + FAMS_CONTROL:
        line = f"{fam:<20}"
        rec = {}
        for metric in ["eer", "fpr"]:
            vals = []
            for arm in ["ctl", "aug"]:
                f = pf[arm][(pf[arm].label == "fake") & (pf[arm].attack == fam)]
                r = pf[arm][(pf[arm].label == "real") & (pf[arm].attack == fam)]
                if metric == "eer":
                    v = eer_auc(f, r)[0]
                else:
                    v = float((r.score.values < out[arm]["thr"]).mean()) * 100
                vals.append(v)
            rec[metric] = vals
        rows[fam] = rec
        print(f"{fam:<20}{rec['eer'][0]:>9.2f}{rec['eer'][1]:>9.2f}"
              f"{rec['eer'][1]-rec['eer'][0]:>+8.2f}   "
              f"{rec['fpr'][0]:>9.1f}{rec['fpr'][1]:>9.1f}"
              f"{rec['fpr'][1]-rec['fpr'][0]:>+8.1f}")

    # 干净基线
    ce = []
    for arm in ["ctl", "aug"]:
        c = cl[arm]
        ce.append(eer_auc(c[c.label == "fake"], c[c.label == "real"])[0])
    print(f"\n{'干净(798)':<20}{ce[0]:>9.2f}{ce[1]:>9.2f}{ce[1]-ce[0]:>+8.2f}")

    # ⚠️ 参照列：两臂微调后 dev EER 可能都≈0 ⇒ 各自 dev 阈值可能贴近边界，单看它会失真。
    # 补一列**公共阈值**（官方 AASIST 的 LA dev 阈值 −1.093850）下的 FPR，两臂可直接对读。
    T_COMMON = -1.093850
    print(f"\n参照列：FPR@{T_COMMON}（公共阈值，脱离各自 dev 退化的影响）")
    print(f"  {'族':<20}{'CTL':>9}{'AUG':>9}{'Δ':>8}")
    for fam in FAMS_DECISIVE:
        vals = []
        for arm in ["ctl", "aug"]:
            r = pf[arm][(pf[arm].label == "real") & (pf[arm].attack == fam)]
            vals.append(float((r.score.values < T_COMMON).mean()) * 100)
        print(f"  {fam:<20}{vals[0]:>9.1f}{vals[1]:>9.1f}{vals[1]-vals[0]:>+8.1f}")

    print("\n═══ 预注册判据核对 ═══")
    a5, a10 = rows["awgn-snr5"]["eer"], rows["awgn-snr10"]["eer"]
    rv, sp = rows["reverb-rt60-0.4"], rows["speed0.9"]
    h5a = a5[1] < 8 and a10[1] < 8
    h5b = rv["eer"][1] > rv["eer"][0] and sp["eer"][1] > sp["eer"][0]
    h5c = (ce[1] - ce[0]) < 1.0
    h6a = rv["fpr"][1] < 60 and sp["fpr"][1] < 60
    h6b = abs(rv["eer"][1] - rv["eer"][0]) < 5 and abs(sp["eer"][1] - sp["eer"][0]) < 5
    print(f"  H5a 噪声族改善到 <8: awgn5 {a5[0]:.1f}→{a5[1]:.1f}, awgn10 {a10[0]:.1f}→{a10[1]:.1f} "
          f"=> {'✅' if h5a else '❌'}")
    print(f"  H5b 混响/变速恶化: reverb {rv['eer'][0]:.1f}→{rv['eer'][1]:.1f}, "
          f"speed {sp['eer'][0]:.1f}→{sp['eer'][1]:.1f} => {'✅' if h5b else '❌'}")
    print(f"  H5c 干净恶化 <1pp: {ce[0]:.2f}→{ce[1]:.2f} (Δ{ce[1]-ce[0]:+.2f}) => {'✅' if h5c else '❌'}")
    print(f"  H6a FPR@各自dev阈值 <60%: reverb {rv['fpr'][0]:.1f}→{rv['fpr'][1]:.1f}, "
          f"speed {sp['fpr'][0]:.1f}→{sp['fpr'][1]:.1f} => {'✅' if h6a else '❌'}")
    print(f"  H6b 同族 EER 不变(±5): reverb Δ{rv['eer'][1]-rv['eer'][0]:+.1f}, "
          f"speed Δ{sp['eer'][1]-sp['eer'][0]:+.1f} => {'✅' if h6b else '❌'}")
    print("ANALYZE_LA_AUG_DONE")


if __name__ == "__main__":
    main()
