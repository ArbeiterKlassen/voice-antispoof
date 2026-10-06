#!/usr/bin/env python3
"""
在 ASVspoof2019 LA eval 上算 min-tDCF —— 用**官方实现**（vendored 在 code/aasist/evaluation.py，
源自 asvspoof.org 的 tDCF_python_v2.zip），不自造公式。

## 已知答案
AASIST 官方 README:57 写明：评估 AASIST 应得到 `EER: 0.83%, min t-DCF: 0.0275`。
所以我们既有 EER 的锚（0.830%），也有 min-tDCF 的锚（0.0275）——这是第二道对拍。

## 输入
- CM 分：我们管线产出的 LA eval 全量分数（`results/scores/official_AASIST.la.npz`）
- ASV 分：镜像下载的官方 ASV 打分文件（三列：cm_label asv_label score；官方码取 [:,1] 与 [:,2]）
  ⚠️ 该文件被拆成**按行对齐**的两个文件（trl 带 utt_id，scores 不带）；已校验 0 行不符。
  官方实现**不按 utt 做 join**，而是分别取 target/nontarget/spoof 的分数分布，故三列文件可直用。

## ⚠️ 不改官方文件
官方码写于 `np.float` 尚存之时，NumPy≥1.24 已删除该别名。
这里用**运行时 shim**（`np.float = float`）而不是编辑官方文件 —— 保持其逐字节可验证。

用法: python run_tdcf_la.py
"""
import os
import sys

import numpy as np

# --- 运行时 shim：官方码用 np.float（NumPy>=1.24 已删除）---
for _alias, _t in (("float", float), ("int", int), ("bool", bool), ("object", object)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _t)

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, "code", "aasist"))
from evaluation import calculate_tDCF_EER                        # noqa: E402

ASV = f"{WS}/data/_downloads/asvspoof2019_la/asv/asv_eval_gi_scores.txt"


def write_cm_scores(npz_path, out_path):
    """把我们的分数写成官方要的 4 列：utt_id, attack_type|none, bonafide|spoof, score"""
    z = np.load(npz_path, allow_pickle=False)
    utt, y, atk, sc = z["utt_id"], z["y"], z["attack_type"], z["score"]
    n = 0
    with open(out_path, "w") as f:
        for u, yy, a, s in zip(utt, y, atk, sc):
            key = "bonafide" if int(yy) == 1 else "spoof"
            src = a if key == "spoof" else "none"      # 官方码只用 col1 去查 A07..A19
            f.write(f"{u} {src} {key} {float(s):.6f}\n")
            n += 1
    print(f"  CM 分写出 {n} 行 -> {out_path}")
    return n


def main():
    npz = f"{WS}/results/scores/official_AASIST.la.npz"
    cm = "/tmp/cm_scores_la.txt"
    out = f"{WS}/results/scores/official_AASIST.la_tdcf.txt"
    print("[1/3] 写 CM 分文件")
    write_cm_scores(npz, cm)
    print("[2/3] 校验两份文件的覆盖")
    asv = np.genfromtxt(ASV, dtype=str)
    print(f"  ASV 行 {len(asv)}  列 {asv.shape[1]}  "
          f"asv_label 分布 {dict(zip(*np.unique(asv[:, 1], return_counts=True)))}")
    print("[3/3] 调官方 calculate_tDCF_EER")
    eer, tdcf = calculate_tDCF_EER(cm, ASV, out, printout=True)
    print(f"\n  === 结果 ===")
    print(f"  EER      = {eer:.4f}%      （官方锚 0.83%）")
    print(f"  min-tDCF = {tdcf:.6f}      （官方锚 0.0275）")
    print(f"  差       = {tdcf - 0.0275:+.6f}")
    print("TDCF_LA_DONE")


if __name__ == "__main__":
    main()
