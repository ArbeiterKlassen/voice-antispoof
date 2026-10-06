#!/usr/bin/env python3
"""
实验 C：签号翻转规则（理论侧裁定 2.3）——预注册见 results/p4_calib_prereg_round2.md

规则：每族用**校准半边**（说话人分半）的 m_r=median(真实)、m_f=median(伪造)；
      m_r < m_f ⇒ 判为反转 ⇒ 评估半边该族分数翻号 s -> -s。
部署假设：一个批次 = 一个信道（无需条件分类器路由）。

判据（预注册，先写死）：
  R1 传输性：触发集合 == 评估半边 AUC<0.5 的族集合（四张表各自核，允许 0 处不符）
  R2 上限：触发族翻号后 AUC ≤ 0.80（翻号修符号不修能力）
  R3 误触发：AUC>0.99 的编解码族触发数 = 0
"""
import os
import sys

import numpy as np
import pandas as pd

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, "code"))
from metrics import compute_auc                                    # noqa: E402

R = f"{WS}/results/scores"
CODEC_LIKE = ("mp3-64k", "aac-128k", "opus-32k", "codec", "g711-8k", "g711", "bandpass-300-3400")


def load(p):
    z = np.load(p, allow_pickle=False)
    return pd.DataFrame({"score": z["score"], "label": z["label"],
                         "attack": z["attack_type"], "spk": z["speaker_id"]})


def main():
    all_ok = {"R1": True, "R2": True, "R3": True}
    for tag, name in [("official_AASIST", "AASIST"), ("AASISTL", "AASIST-L")]:
        for st in ["p4fix", "p4r"]:
            p = f"{R}/{tag}.la_{st}.npz"
            if not os.path.exists(p):
                print(f"  ⚠️ 缺 {p}，跳过"); continue
            P = load(p)
            spks = sorted(P.spk.unique())
            half = len(spks) // 2
            CAL, TEST = set(spks[:half]), set(spks[half:])
            print("=" * 100)
            print(f"【{name} @ {st}】校准说话人 {len(CAL)} / 评估说话人 {len(TEST)}"
                  f"（首/中/尾 {spks[0]}/{spks[half]}/{spks[-1]}）")
            print(f"  {'族':<18}{'m_r':>9}{'m_f':>9}{'触发':>6}{'AUC前':>9}{'AUC后':>9}"
                  f"{'评估AUC<0.5':>13}{'传输':>6}")
            for fam in sorted(P[P.label == "fake"].attack.unique()):
                fc = P[(P.label == "fake") & (P.attack == fam) & (P.spk.isin(CAL))]
                rc = P[(P.label == "real") & (P.attack == fam) & (P.spk.isin(CAL))]
                fe = P[(P.label == "fake") & (P.attack == fam) & (P.spk.isin(TEST))]
                re_ = P[(P.label == "real") & (P.attack == fam) & (P.spk.isin(TEST))]
                if min(len(fc), len(rc), len(fe), len(re_)) == 0:
                    continue
                m_r = float(np.median(rc.score.values))
                m_f = float(np.median(fc.score.values))
                fire = m_r < m_f
                a_before = compute_auc(re_.score.values, fe.score.values)
                a_after = compute_auc(-re_.score.values, -fe.score.values)
                inv_eval = a_before < 0.5
                match = (fire == inv_eval)
                if not match:
                    all_ok["R1"] = False
                if fire and a_after > 0.80:
                    all_ok["R2"] = False
                if (not inv_eval) and fire and fam in CODEC_LIKE:
                    all_ok["R3"] = False
                print(f"  {fam:<18}{m_r:>9.3f}{m_f:>9.3f}{'✓' if fire else '·':>6}"
                      f"{a_before:>9.4f}{a_after:>9.4f}{'是' if inv_eval else '否':>13}"
                      f"{'✅' if match else '❌':>6}")
            # 汇总该表
            n_fire = 0; n_inv = 0
            for fam in sorted(P[P.label == "fake"].attack.unique()):
                fc = P[(P.label == "fake") & (P.attack == fam) & (P.spk.isin(CAL))]
                rc = P[(P.label == "real") & (P.attack == fam) & (P.spk.isin(CAL))]
                fe = P[(P.label == "fake") & (P.attack == fam) & (P.spk.isin(TEST))]
                re_ = P[(P.label == "real") & (P.attack == fam) & (P.spk.isin(TEST))]
                if min(len(fc), len(rc), len(fe), len(re_)) == 0:
                    continue
                if float(np.median(rc.score.values)) < float(np.median(fc.score.values)):
                    n_fire += 1
                if compute_auc(re_.score.values, fe.score.values) < 0.5:
                    n_inv += 1
            print(f"  ── 触发 {n_fire} 族 / 评估反转 {n_inv} 族 "
                  f"{'✅ 一致' if n_fire == n_inv else '❌ 不一致'}")
    print("\n预注册判据核对：")
    for k, v in all_ok.items():
        print(f"  {k}: {'✅ 成立' if v else '❌ 不成立'}")
    print("SIGN_FLIP_DONE")


if __name__ == "__main__":
    main()
