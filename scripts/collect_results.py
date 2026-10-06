#!/usr/bin/env python3
"""
结果总看板 —— 把所有已落盘的评估 json 汇总成一张表，供最终报告取数。

只读 json/npz，不重算。分组：
  A. 公开基准轨（LA）      —— 可与文献对照
  B. 自建轨（LibriSpeech+signalproc 伪造）
  C. 对照实验（P4-fixed vs P4-random，衡量固定 seed 缺陷的影响）
"""
import glob, json, os, sys
import numpy as np

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = f"{WS}/results/scores"
sys.path.insert(0, os.path.join(WS, "code"))
from metrics import evaluate                                       # noqa: E402
import os


def j(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def pooled(p):
    try:
        z = np.load(p, allow_pickle=False)
        if len(np.unique(z["y"])) < 2:
            return float("nan")
        return evaluate(z["score"], z["y"])["eer_percent"]
    except Exception:
        return float("nan")


def macro(p):
    d = j(p)
    if not d:
        return float("nan")
    g = d.get("groups", {})
    v = [x["eer_percent"] for x in g.values() if "eer_percent" in x]
    return float(np.mean(v)) if v else float("nan")


GROUPS = {
    "A. 公开基准轨 ASVspoof2019 LA": [
        ("official_AASIST.la_dev", "官方AASIST @ LA dev（域内 A01–A06）"),
        ("official_AASIST.la", "官方AASIST @ LA eval **全量71237**（论文 0.830%）"),
        ("official_AASIST.la_sub12k", "官方AASIST @ LA eval 子集12k（先跑的判词）"),
        ("official_AASIST.la_clean", "官方AASIST @ LA-P4 干净子集798"),
        ("official_AASIST.la_p4fix", "官方AASIST @ LA-P4 固定参数"),
        ("official_AASIST.la_p4r", "官方AASIST @ LA-P4 随机参数"),
    ],
    "B. 自建轨（LibriSpeech dev-clean + signalproc 伪造）": [
        ("A_best.p1", "臂A best @ P1 test（域内）"),
        ("A_last.p1", "臂A last @ P1 test（域内）"),
        ("A_best.p4", "臂A best @ P4（固定参数）"),
        ("A_last.p4", "臂A last @ P4"),
        ("A20_best.p1", "臂A20 best @ P1"), ("A20_last.p1", "臂A20 last @ P1"),
        ("A20_best.p4", "臂A20 best @ P4"), ("A20_last.p4", "臂A20 last @ P4"),
        ("B_best.p1", "臂B best @ P1"), ("B_last.p1", "臂B last @ P1"),
        ("B_best.p4", "臂B best @ P4"), ("B_last.p4", "臂B last @ P4"),
        ("D_best.p1", "臂D best @ P1"), ("D_last.p1", "臂D last @ P1"),
        ("D_best.p4", "臂D best @ P4"), ("D_last.p4", "臂D last @ P4"),
        ("A_best.la", "臂A best 跨数据集 @ LA eval 全量"),
        ("B_best.la", "臂B best 跨数据集 @ LA eval 全量"),
        ("D_best.la", "臂D best 跨数据集 @ LA eval 全量"),
    ],
    "C. 固定 vs 随机参数（衡量 make_attacks seed 缺陷）": [
        ("A_best.p4r", "臂A best @ P4-random"),
        ("A_last.p4r", "臂A last @ P4-random"),
        ("B_best.p4r", "臂B best @ P4-random"),
        ("B_last.p4r", "臂B last @ P4-random"),
        ("D_best.p4r", "臂D best @ P4-random"),
        ("D_last.p4r", "臂D last @ P4-random"),
    ],
}


def main():
    for gname, items in GROUPS.items():
        print("\n" + "=" * 84)
        print(gname)
        print("=" * 84)
        print(f"  {'产物':<34}{'EER%':>9}{'AUC':>9}{'等权%':>9}{'n_eval':>9}")
        for key, label in items:
            pj, pz = f"{R}/{key}.json", f"{R}/{key}.npz"
            d = j(pj)
            if d is None:
                print(f"  {label:<34}{'— 未跑':>9}")
                continue
            o = d.get("overall", {})
            print(f"  {label:<34}{o.get('eer_percent', float('nan')):>9.3f}"
                  f"{o.get('auc', float('nan')):>9.4f}{macro(pj):>9.3f}"
                  f"{o.get('real_count', 0) + o.get('fake_count', 0):>9}")
    print("\nCOLLECT_DONE")


if __name__ == "__main__":
    main()
