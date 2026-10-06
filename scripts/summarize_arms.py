#!/usr/bin/env python3
"""
多臂汇总表 —— 从 results/scores/*.{p1,p4}.json + npz 生成，不重算口径。

报数纪律（沿用本项目的既有约定）：
  · P4 逐攻击 EER 给 **攻击类型等权平均**（macro）与 **并池 EER**（micro）两列。
    两者差本身是结论：macro 高 micro 低 = 退化集中在少数攻击上。
  · 同时给 **冻结阈值下的真实误报率**（标定维度），与 EER（判别力维度）分开看。
  · ckpt 必须带 epoch 与选择方式（best=dev 选优 / last=固定预算），
    因为 dev EER 早已饱和到 0，"选优"实为噪声驱动，不可比。

用法:
  python summarize_arms.py                     # 汇总 results/scores/
  python summarize_arms.py --detail            # 附逐攻击 EER 矩阵
"""
import argparse, json, os, sys
import numpy as np

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, "code"))
from metrics import evaluate                                    # noqa: E402
import os


def load_json(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=f"{WS}/results/scores")
    ap.add_argument("--detail", action="store_true")
    args = ap.parse_args()

    names = sorted({f[:-len(".p4.json")] for f in os.listdir(args.dir) if f.endswith(".p4.json")})
    if not names:
        print(f"{args.dir} 下没有 *.p4.json"); return

    rows = []
    per_attack = {}
    for n in names:
        p4 = load_json(f"{args.dir}/{n}.p4.json")
        p1 = load_json(f"{args.dir}/{n}.p1.json")
        if p4 is None:
            continue
        g = p4.get("groups", {})
        eers = [v["eer_percent"] for v in g.values() if "eer_percent" in v]
        macro = float(np.mean(eers)) if eers else float("nan")
        pooled = float("nan")
        npz = f"{args.dir}/{n}.p4.npz"
        if os.path.exists(npz):
            z = np.load(npz, allow_pickle=False)
            if len(np.unique(z["y"])) == 2:
                pooled = evaluate(z["score"], z["y"])["eer_percent"]
        frz = p4["overall"].get("at_frozen_threshold", {}).get("fpr_real_misjudged_fake", float("nan"))
        rows.append({
            "arm": n, "epoch": p4.get("ckpt_epoch", "?"),
            "p1": (p1["overall"]["eer_percent"] if p1 else float("nan")),
            "p4_macro": macro, "p4_pooled": pooled,
            "p4_worst": (max(eers) if eers else float("nan")),
            "fpr_frz": frz,
        })
        per_attack[n] = {k: v["eer_percent"] for k, v in g.items()}

    rows.sort(key=lambda r: (r["p1"] if r["p1"] == r["p1"] else 9e9, r["p4_macro"]))
    print("\n" + "=" * 92)
    print("多臂汇总（P1=域内 test split；P4=攻击后两类都过同等攻击；frz 阈值 = P1 dev 冻结值）")
    print("=" * 92)
    print(f"  {'臂':<12} {'ep':>3} {'P1域内%':>9} {'P4等权%':>9} {'P4并池%':>9} {'P4最差%':>9} {'误报率@frz':>11}")
    for r in rows:
        print(f"  {r['arm']:<12} {str(r['epoch']):>3} {r['p1']:>9.3f} {r['p4_macro']:>9.3f} "
              f"{r['p4_pooled']:>9.3f} {r['p4_worst']:>9.3f} {r['fpr_frz']:>11.4f}")
    print()
    print("  读法：判别力看 P4等权/并池；标定看 误报率@frz。二者是两个独立的失效模式，不许合成一个数。")

    if args.detail:
        allatk = sorted({a for d in per_attack.values() for a in d})
        print("\n" + "=" * 92)
        print("逐攻击 EER%")
        print("=" * 92)
        print(f"  {'攻击':<20}" + "".join(f"{n:>12}" for n in names))
        for a in allatk:
            line = f"  {a:<20}"
            for n in names:
                v = per_attack[n].get(a)
                line += f"{v:>12.3f}" if v is not None else f"{'-':>12}"
            print(line)

    rep = os.path.join(args.dir, "summary.json")
    json.dump({"rows": rows, "per_attack": per_attack}, open(rep, "w"), ensure_ascii=False, indent=2)
    print(f"\n已写入 {rep}")


if __name__ == "__main__":
    main()
