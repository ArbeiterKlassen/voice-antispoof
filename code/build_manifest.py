#!/usr/bin/env python3
"""
合并 real + 各类 fake 的 manifest，产出统一 all_manifest.csv

设计要点：
  * **split 继承自源真实语音**（fake 由其源 real 派生）→ 自动保持 speaker-disjoint
  * 支持按条件筛选，便于做**留一条件泛化实验**（P2 cross-model 的最小可行版）：
      - 训练用 melvoc+pitch，测试用 formant（生成器/攻击类型从未见过）
  * 输出前做三项健全性校验，不通过直接失败

用法:
  python build_manifest.py --out data/manifests/all_manifest.csv
  python build_manifest.py --out .../p2_holdout_formant.csv --exclude-attack formant
"""
import argparse, glob, os, sys
import pandas as pd
import os

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAN = f"{WS}/data/manifests"
COLS = ["audio_path", "label", "source", "clone_model", "attack_type",
        "text", "speaker_id", "duration", "sample_rate", "split", "utt_id"]


def load(p):
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    for c in COLS:
        if c not in df.columns:
            df[c] = "none"
    return df[COLS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{MAN}/all_manifest.csv")
    ap.add_argument("--real", default=f"{MAN}/real_manifest.csv")
    ap.add_argument("--fake-glob", default=f"{MAN}/fake_*_manifest.csv")
    ap.add_argument("--exclude-attack", default="", help="逗号分隔，排除这些 attack_type（做留一泛化）")
    ap.add_argument("--exclude-clone", default="", help="逗号分隔，排除这些 clone_model")
    args = ap.parse_args()

    parts = []
    r = load(args.real)
    if r is not None:
        parts.append(r)
        print(f"[载入] real        {len(r):>5} 条  {args.real}")
    for p in sorted(glob.glob(args.fake_glob)):
        f = load(p)
        if f is not None:
            parts.append(f)
            print(f"[载入] fake        {len(f):>5} 条  {os.path.basename(p)}")

    if not parts:
        sys.exit("❌ 没有载入任何 manifest")

    df = pd.concat(parts, ignore_index=True)

    # ---- 排除 ----
    for col, spec in [("attack_type", args.exclude_attack), ("clone_model", args.exclude_clone)]:
        vals = [v.strip() for v in spec.split(",") if v.strip()]
        if vals:
            before = len(df)
            df = df[~df[col].isin(vals)]
            print(f"[排除] {col} in {vals}: {before} -> {len(df)}")

    df = df.reset_index(drop=True)

    # ---- 健全性校验 ----
    print("\n===== 校验 =====")
    assert df["label"].isin(["real", "fake"]).all(), "label 只能 real/fake"
    assert df["audio_path"].is_unique, "❌ 有重复 audio_path"
    # 文件存在性（抽样 200 条，避免全量 stat 太慢）
    samp = df.sample(min(200, len(df)), random_state=0)
    miss = [p for p in samp.audio_path if not os.path.exists(
        p if os.path.isabs(p) else os.path.join(WS, p))]
    assert not miss, f"❌ 抽样 {len(samp)} 条中有 {len(miss)} 条文件不存在，例: {miss[:3]}"
    print(f"✅ 文件存在性（抽样 {len(samp)}）")

    # speaker-disjoint：同一 speaker 不得跨 split
    x = df.groupby("speaker_id")["split"].nunique()
    bad = x[x > 1]
    assert len(bad) == 0, f"❌ speaker 跨 split: {bad.to_dict()}"
    print(f"✅ speaker-disjoint（{df.speaker_id.nunique()} 个说话人，无跨 split）")

    # 每个 split 都要 real 和 fake 都有，否则训不了
    for s in sorted(df["split"].unique()):
        sub = df[df["split"] == s]
        n_r, n_f = int((sub.label == "real").sum()), int((sub.label == "fake").sum())
        assert n_r > 0 and n_f > 0, f"❌ split={s} 缺 real 或 fake (real={n_r}, fake={n_f})"
        print(f"  split {s:<6} real {n_r:>5} / fake {n_f:>5}  说话人 {sub.speaker_id.nunique()}")

    # ---- 落盘 ----
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n合计 {len(df)} 条 -> {args.out}")
    print("来源分布:")
    print(df.groupby(["source", "split"]).size().unstack(fill_value=0).to_string())
    print("BUILD_MANIFEST_DONE")


if __name__ == "__main__":
    main()
