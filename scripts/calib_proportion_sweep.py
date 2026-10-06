#!/usr/bin/env python3
"""
实验 D：批内真实比例的敏感性（理论侧裁定 2.1）——预注册见 results/p4_calib_prereg_round2.md

无监督中位数对齐：shift = median(批) − median(同组成干净参考)；thr_cal = T_LA + shift。
批次 n=200，真实比例 p ∈ {50%, 30%, 10%}，每 (族,p) 抽 200 次。
指标：评估半边该族**全部真实**在 thr_cal 下的 FPR；宏平均 = 逐族取 200 次中位后跨 9 族平均。

判据（预注册）：
  H7a p=50% 宏平均 FPR ≤ 0.10    H7b p=30% ≤ 0.10    H7c p=10% > 0.10
  H7d 单调 FPR(50) ≤ FPR(30) ≤ FPR(10)（中位）
附报：假设错配（假定 50%/30%，实际 10%）。
"""
import os
import sys

import numpy as np

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, "code"))

R = f"{WS}/results/scores"
T_LA = -1.093850            # LA dev 冻结阈值（analzye_la_p4 同款）
N_BATCH = 200
N_DRAW = 200
SEED = 20261005


def load(p):
    z = np.load(p, allow_pickle=False)
    d = {"score": z["score"], "label": z["label"].astype(str),
         "attack": z["attack_type"].astype(str), "spk": z["speaker_id"].astype(str)}
    return d


def draw_batch(rng, r, f, p, n=N_BATCH):
    n_r = int(round(p * n)); n_f = n - n_r
    ir = rng.choice(len(r), n_r, replace=False)
    iff = rng.choice(len(f), n_f, replace=False)
    return np.concatenate([r[ir], f[iff]])


def main():
    P = load(f"{R}/official_AASIST.la_p4fix.npz")
    C = load(f"{R}/official_AASIST.la_clean.npz")

    spks = sorted(set(P["spk"]))
    half = len(spks) // 2
    TEST = set(spks[half:])
    te = np.isin(P["spk"], list(TEST))

    c_real = C["score"][C["label"] == "real"]
    c_fake = C["score"][C["label"] == "fake"]
    fams = sorted(set(P["attack"][P["label"] == "fake"]))
    print(f"评估说话人 {len(TEST)}；干净参考 real {len(c_real)} fake {len(c_fake)}；"
          f"族 {len(fams)}；每格 {N_DRAW} 抽")

    # 逐族评估半边真实（固定，不被抽掉）
    eval_real = {f: P["score"][te & (P["label"] == "real") & (P["attack"] == f)] for f in fams}
    eval_fake = {f: P["score"][te & (P["label"] == "fake") & (P["attack"] == f)] for f in fams}

    # 未校正基线（同评估半边）
    base = np.mean([float((eval_real[f] < T_LA).mean()) for f in fams])
    print(f"\n未校正宏平均 FPR（本评估半边）= {base:.3f}")

    results = {}
    for p in (0.5, 0.3, 0.1):
        rng = np.random.RandomState(SEED)
        per_fam = {}
        for f in fams:
            # 批次（评估半边，后处理信道 = 该族）
            b_r = P["score"][te & (P["label"] == "real") & (P["attack"] == f)]
            b_f = P["score"][te & (P["label"] == "fake") & (P["attack"] == f)]
            fprs = []
            for _ in range(N_DRAW):
                batch = draw_batch(rng, b_r, b_f, p)
                ref = draw_batch(rng, c_real, c_fake, p)
                shift = np.median(batch) - np.median(ref)
                fprs.append(float((eval_real[f] < T_LA + shift).mean()))
            per_fam[f] = float(np.median(fprs))
        macro = float(np.mean(list(per_fam.values())))
        results[p] = (macro, per_fam)
        print(f"\n  p={p:.0%}  宏平均 FPR = {macro:.3f}")
        for f in fams:
            mark = "←反转族" if f in ("reverb-rt60-0.4", "speed0.9") else ""
            print(f"      {f:<20}{per_fam[f]:>8.3f} {mark}")

    print("\n预注册判据核对：")
    m50, m30, m10 = results[0.5][0], results[0.3][0], results[0.1][0]
    print(f"  H7a p=50%: {m50:.3f} ≤ 0.10 → {'✅' if m50 <= 0.10 else '❌'}")
    print(f"  H7b p=30%: {m30:.3f} ≤ 0.10 → {'✅' if m30 <= 0.10 else '❌'}")
    print(f"  H7c p=10%: {m10:.3f} > 0.10 → {'✅' if m10 > 0.10 else '❌'}")
    print(f"  H7d 单调: {m50:.3f} ≤ {m30:.3f} ≤ {m10:.3f} → "
          f"{'✅' if m50 <= m30 <= m10 else '❌'}")

    # 附报：假设错配（参考按假定 p 建，批次实际 10%）
    print("\n附报（假设错配）：批次实际 p=10%，参考按假定比例建")
    for p_assumed in (0.5, 0.3):
        rng = np.random.RandomState(SEED + 1)
        per_fam = {}
        for f in fams:
            b_r = P["score"][te & (P["label"] == "real") & (P["attack"] == f)]
            b_f = P["score"][te & (P["label"] == "fake") & (P["attack"] == f)]
            fprs = []
            for _ in range(N_DRAW):
                batch = draw_batch(rng, b_r, b_f, 0.1)
                ref = draw_batch(rng, c_real, c_fake, p_assumed)
                shift = np.median(batch) - np.median(ref)
                fprs.append(float((eval_real[f] < T_LA + shift).mean()))
            per_fam[f] = float(np.median(fprs))
        print(f"  假定 {p_assumed:.0%} 实际 10%: 宏平均 FPR = "
              f"{np.mean(list(per_fam.values())):.3f}")
    print("PROPORTION_SWEEP_DONE")


if __name__ == "__main__":
    main()
