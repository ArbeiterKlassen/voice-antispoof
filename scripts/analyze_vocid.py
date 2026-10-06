#!/usr/bin/env python3
"""
音高族空对照分析（判据见 results/vocoder_id_control_prereg.md，预注册在先）：
- V1/V2/V2'：恒等臂的真实分位移动 + 同族 AUC ⇒ 判定相位声码器是否中性
- V3：双侧位移表（真实位移/伪造位移/AUC），音高四族 + 恒等臂 + 时长匹配对照，与 §2.2 同格式
"""
import os
import sys

import numpy as np
import pandas as pd

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, "code"))
from metrics import evaluate                                       # noqa: E402

R = f"{WS}/results/scores"


def load(p):
    z = np.load(p, allow_pickle=False)
    return pd.DataFrame({"score": z["score"], "label": z["label"],
                         "attack": z["attack_type"]})


def auc_eer(f, r):
    s = np.concatenate([f.score.values, r.score.values])
    y = np.concatenate([np.zeros(len(f), int), np.ones(len(r), int)])
    m = evaluate(s, y)
    return m["auc"], m["eer_percent"]


def main():
    clean = load(f"{R}/official_AASIST.la_clean.npz")
    c_real_med = float(np.median(clean[clean.label == "real"].score.values))
    print(f"干净基线（798 子集）真实分中位 = {c_real_med:.3f}")

    rows = []
    for tag, name in [("official_AASIST", "AASIST"), ("AASISTL", "AASIST-L")]:
        cl = load(f"{R}/{tag}.la_clean.npz")
        cr = float(np.median(cl[cl.label == "real"].score.values))
        for fam, npz in [("vocoder-id", f"{tag}.la_p4vocid"),
                         ("pitch-down2", f"{tag}.la_p4pitch"),
                         ("pitch-down4", f"{tag}.la_p4pitch"),
                         ("pitch-up2", f"{tag}.la_p4pitch"),
                         ("pitch-up4", f"{tag}.la_p4pitch")]:
            p = f"{R}/{npz}.npz"
            if not os.path.exists(p):
                continue
            d = load(p)
            f = d[(d.label == "fake") & (d.attack == fam)]
            r = d[(d.label == "real") & (d.attack == fam)]
            if len(f) == 0 or len(r) == 0:
                continue
            a, e = auc_eer(f, r)
            rows.append({"后端": name, "族": fam, "n_f": len(f), "n_r": len(r),
                         "AUC": round(a, 4), "EER%": round(e, 2),
                         "真实位移": round(float(np.median(r.score.values)) - cr, 2),
                         "伪造位移": round(float(np.median(f.score.values)) - float(
                             np.median(cl[cl.label == "fake"].score.values)), 2)})
        # 时长匹配对照（AASIST 才有 dm_ 文件）
        for fam, tag2 in [("reverb-rt60-0.4", "dm_p4fix"), ("speed0.9", "dm_p4fix")]:
            p = f"{R}/{tag}.{tag2}.npz"
            if not os.path.exists(p):
                continue
            d = load(p)
            f = d[(d.label == "fake") & (d.attack == fam)]
            r = d[(d.label == "real") & (d.attack == fam)]
            if len(f) == 0 or len(r) == 0:
                continue
            a, e = auc_eer(f, r)
            rows.append({"后端": name, "族": f"{fam}（时长匹配）", "n_f": len(f), "n_r": len(r),
                         "AUC": round(a, 4), "EER%": round(e, 2), "真实位移": np.nan, "伪造位移": np.nan})

    df = pd.DataFrame(rows)
    print("\n" + "=" * 96)
    print("V3 双侧位移表（与 §2.2 同格式；位移相对各自后端干净基线中位）")
    print("=" * 96)
    print(df.to_string(index=False))

    # V1/V2/V2' 判定（对官方 AASIST）
    v = df[(df.后端 == "AASIST") & (df.族 == "vocoder-id")]
    if len(v):
        shift = float(v.真实位移.iloc[0]); a = float(v.AUC.iloc[0])
        print(f"\n═══ 判据核对（V1/V2/V2'，官方 AASIST、恒等臂）═══")
        print(f"  恒等臂真实分中位下移 = {shift:+.2f}   同族 AUC = {a:.4f}")
        if shift <= -2 or a < 0.5:
            print("  ⇒ **V1 成立：相位声码器本身即可造成失真** ⇒ 音高族反转降级为「声码器伪影混淆」，")
            print("     §2.6 须标注撤回/降级。")
        elif abs(shift) < 1 and a > 0.9:
            print("  ⇒ **V2 成立：相位声码器中性** ⇒ 音高族反转保留为音高效应证据。")
        else:
            print("  ⇒ **V2'：部分混淆** ⇒ 音高结论降级为「方向性证据、量级不可全归音高」。")
    else:
        print("\n（缺 vocoder-id 结果，无法判定）")
    print("ANALYZE_VOCID_DONE")


if __name__ == "__main__":
    main()
