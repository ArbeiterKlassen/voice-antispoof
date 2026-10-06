#!/usr/bin/env python3
"""
DF 轨分析：D1–D3 核对（预注册见 results/df_eval_prereg.md）

D1：主列 ② 与诊断列 ① 的 EER 差 ≥ 1.0pp ⇒ 定长口径对 DF 有实质偏差
D2：② 与 ②b（滑窗均值）EER 差 < 0.5pp ⇒ 整条前向对聚合方式不敏感
D3：无外部锚 ⇒ 表注写"无外部锚，仅作轨内比较"
反例条款：② 若 EER > 50%（反转迹象），先查超长文件处理，且 ②b 一致才允许写"DF 上亦反转"。

口径纪律：DF 块只报 EER/AUC（阈值无关）；CI 用说话人分层 bootstrap（EER）。
"""
import os
import sys

import numpy as np

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, "code"))
from metrics import evaluate, bootstrap_ci_eer                          # noqa: E402

R = f"{WS}/results/scores"
TAGS = ["official_AASIST", "AASISTL"]


def load(p):
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in ("score", "y", "speaker_id", "utt_id")}, z


def stat(npz):
    s, y, sp = npz["score"], npz["y"], npz["speaker_id"].astype(str)
    m = evaluate(s, y)
    ci = bootstrap_ci_eer(s, y, sp, n_boot=1000)
    return m, ci


def main():
    print(f"{'模型':<16}{'①定长 EER%':>12}{'②全长 EER%':>12}{'②b滑窗 EER%':>13}"
          f"{'D1:②-①(pp)':>12}{'D2:|②-②b|(pp)':>15}{'② AUC':>9}{'② CI(EER)':>18}")
    d1_fail_all, d2_fail_all = [], []
    for tag in TAGS:
        ps = {k: f"{R}/{tag}.df_{k}.npz" for k in ("fix", "full", "win")}
        if not all(os.path.exists(p) for p in ps.values()):
            print(f"  ⚠️ {tag}: 缺 {[k for k, p in ps.items() if not os.path.exists(p)]}，跳过")
            continue
        m1 = stat(np.load(ps["fix"], allow_pickle=False))[0]
        m2, ci2 = stat(np.load(ps["full"], allow_pickle=False))
        mb = stat(np.load(ps["win"], allow_pickle=False))[0]
        d1 = m2["eer_percent"] - m1["eer_percent"]
        d2 = abs(m2["eer_percent"] - mb["eer_percent"])
        if d1 < 1.0:
            d1_fail_all.append(tag)
        if d2 >= 0.5:
            d2_fail_all.append(tag)
        ci_s = f"[{ci2['ci_lo_percent']:.2f},{ci2['ci_hi_percent']:.2f}]"
        print(f"{tag:<16}{m1['eer_percent']:>12.3f}{m2['eer_percent']:>12.3f}"
              f"{mb['eer_percent']:>13.3f}{d1:>+12.3f}{d2:>15.3f}{m2['auc']:>9.4f}{ci_s:>18}")
        if m2["eer_percent"] > 50.0:
            print(f"  ⚠️ 反例条款：{tag} ② EER>50% —— 先查超长文件处理；"
                  f"②b {'一致（可写' if d2 < 0.5 else '不一致（不可写'}「DF 上亦反转」)")
    print(f"\nD1（②-① ≥ 1.0pp）: {'✅ 成立' if not d1_fail_all else '❌ 不成立（' + ','.join(d1_fail_all) + '）'}")
    print(f"D2（|②-②b| < 0.5pp）: {'✅ 成立' if not d2_fail_all else '❌ 不成立（' + ','.join(d2_fail_all) + '）'}")
    print("D3: 无外部锚 ⇒ 表注必须写「无外部锚，仅作轨内比较」（预注册原文，勿省）")
    print("ANALYZE_DF_DONE")


if __name__ == "__main__":
    main()
