#!/usr/bin/env python3
"""
LA-P4 时长匹配对照 —— 关掉「源级时长不平衡」这个口子（理论侧 §四.3）。

## 问题
LA-P4 的源是 200 真实（按时长中位 3.30s）+ 598 伪造（2.72s）分层抽的。
**只用时长**就能得到 AUC **0.6369**（干净集同样 0.6369 ⇒ 与攻击无关，是抽样造成的源级不平衡）。
自建轨当时这一项是 0.5004，所以 LA 轨这里是个新引入的缺陷，必须关掉。

## 做法
不改攻击、不改模型：只**重选源**，让真实与伪造的时长分布匹配（分位点匹配），
再把三套 manifest（clean/p4fix/p4r）限制到匹配后的源上，重评一次。
⇒ 与主结果的差 = 时长混淆的影响量。

## 口径
- 配对规则：按真实时长分位点，对每个真实挑一个时长最接近的伪造（无放回）⇒ 1:1 匹配
- 匹配后仍报「同族口径」（该族伪造 vs 同族真实）

用法: python make_la_p4_durmatched.py [--n 200]
"""
import argparse, os
import numpy as np
import pandas as pd
import os

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAN = f"{WS}/data/manifests"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="匹配后每类的条数")
    args = ap.parse_args()

    clean = pd.read_csv(f"{MAN}/la_p4_clean_manifest.csv")
    r = clean[clean.label == "real"].copy()
    f = clean[clean.label == "fake"].copy()
    print(f"  原始：真实 {len(r)}（中位 {r.duration.median():.2f}s）"
          f" / 伪造 {len(f)}（中位 {f.duration.median():.2f}s）")

    # 贪心 1:1 时长匹配（按真实时长排序，逐个挑最近的未用伪造）
    r = r.sort_values("duration").reset_index(drop=True)
    pool = f.sort_values("duration").reset_index(drop=True)
    used = np.zeros(len(pool), dtype=bool)
    pv = pool.duration.values
    picks = []
    for t in r.duration.values[:args.n]:
        d = np.abs(pv - t)
        d[used] = np.inf
        j = int(np.argmin(d))
        used[j] = True
        picks.append(j)
    fm = pool.iloc[picks]
    rm = r.iloc[:args.n]
    print(f"  匹配后：真实 {len(rm)}（中位 {rm.duration.median():.2f}s）"
          f" / 伪造 {len(fm)}（中位 {fm.duration.median():.2f}s）")

    from metrics import compute_auc  # noqa: E402
    a = compute_auc(rm.duration.values, fm.duration.values)
    a2 = compute_auc(-rm.duration.values, -fm.duration.values)
    print(f"  匹配后「只用时长」的 AUC = {max(a, a2):.4f}"
          f"  {'✅ 已关掉（≈0.5）' if abs(max(a,a2)-0.5) < 0.06 else '⚠️ 仍有残余'}")

    src_r = set(rm.utt_id); src_f = set(fm.utt_id)
    print(f"\n  三套 manifest 限制到匹配源后重评（攻击文件按源文件名对应，无需重生成）:")
    for tag, mf in [("durmatch_clean", "la_p4_clean_manifest.csv"),
                    ("durmatch_p4fix", "la_p4fix_manifest.csv"),
                    ("durmatch_p4r", "la_p4r_manifest.csv")]:
        m = pd.read_csv(f"{MAN}/{mf}")
        if "p4" in tag:
            # 从 audio_path 的 basename 反推源 utt_id。
            # ⚠️ 两个生成脚本命名规则不同，必须分开处理（曾因此抽出 0 行）：
            #    make_attacks.py（p4fix）    -> "<源>.wav"        （无后缀）
            #    make_aug_random.py（p4r）   -> "<源>_<j>.wav"    （有 _j 后缀）
            m = m.copy()
            base = m.audio_path.str.split("/").str[-1].str.replace(r"\.wav$", "", regex=True)
            m["_src"] = base.str.rsplit("_", n=1).str[0] if tag.endswith("p4r") else base
            keep = m[((m.label == "real") & m._src.isin(src_r)) |
                     ((m.label == "fake") & m._src.isin(src_f))]
            keep = keep.drop(columns=["_src"])
        else:
            keep = m[m.utt_id.isin(src_r | src_f)]
        out = f"{MAN}/{tag}_manifest.csv"
        keep.to_csv(out, index=False)
        print(f"    {tag:<18} {len(keep):>5} 条（real {int((keep.label=='real').sum())} / "
              f"fake {int((keep.label=='fake').sum())}）-> {os.path.basename(out)}")
    print("DURMATCH_DONE")


if __name__ == "__main__":
    main()
