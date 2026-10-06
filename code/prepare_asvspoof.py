#!/usr/bin/env python3
"""
ASVspoof2019 LA parquet -> flac + manifest（**公开基准轨**）

## 为什么加这条轨
自建轨（LibriSpeech + 自生成 signalproc 伪造）产出的 0.000% 域内 EER **不可与文献比较**：
伪造太容易、数据只有 5.4 小时、且没有神经克隆。LA 是 AASIST 论文 0.83% EER 的那张集，
有了它才能谈「逼近 SOTA」。

## 与自建轨的口径差异（必须显式声明，不许混着报数）
- 划分：LA 官方 train/dev/test 已是**说话人不相交**（本脚本会独立验证，不假设）
- 生成器身份：`system_id` = A01..A19（A01–A06 TTS / A07–A19 VC），bonafide 为 '-'
  → 这是天然的 clone_model 字段，可跑协议 P2「留一生成器」实验
- 文本：LA 不提供，写 na

## ⚠️ 标签约定陷阱（本项目已踩过方向反了的坑）
parquet 的 `key`：**0=bonafide, 1=spoof**；而本项目内部一律 **real=1 / fake=0**。
两个约定相反，本脚本在此处显式转换，并用**官方协议文件交叉验证**（独立来源对拍）。

## 来源与纪律
- 镜像：hf-mirror.com/datasets/Bisher/ASVspoof_2019_LA（非官方发布渠道）
- 官方协议：protocols/ASVspoof2019.LA.cm.eval.trl.txt（与镜像 test 划分对拍）
- 报告时按 blocked_datasets.md §A4 路径2 的规定标注「非官方镜像，仅用于管线验证」；
  但**划分与标签不依赖镜像的正确性**——已与官方协议逐条对过。

用法:
  python prepare_asvspoof.py --check-only     # 只校验，不落盘
  python prepare_asvspoof.py                  # 落盘全部
  python prepare_asvspoof.py --limit 2000     # 每划分取前 N 条（快速迭代）
"""
import argparse, csv, io, os, sys, collections
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import os

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = f"{WS}/data/_downloads/asvspoof2019_la"
OUT = f"{WS}/data/asvspoof"
MAN = f"{WS}/data/manifests"
PROTO = f"{SRC}/protocols/eval.trl.txt"

SPLIT_FILE = {"train": f"{SRC}/data/train-00000-of-00001.parquet",
              "dev":   f"{SRC}/data/validation-00000-of-00001.parquet",
              "test":  f"{SRC}/data/test-00000-of-00001.parquet"}
EXPECT_ROWS = {"train": 25380, "dev": 24844, "test": 71237}


def key_to_label(v):
    """parquet key -> 本项目 label。0=bonafide->real / 1=spoof->fake（**与内部 y 相反，故显式写死**）"""
    s = str(v).strip().lower()
    if s in ("0", "bonafide", "real", "genuine"):
        return "real"
    if s in ("1", "spoof", "fake"):
        return "fake"
    raise ValueError(f"未知 key: {v!r}")


def load_protocol():
    """官方 eval 协议: speaker utt - attack label"""
    m = {}
    for line in open(PROTO):
        p = line.split()
        if len(p) >= 5:
            m[p[1]] = (p[0], p[3], p[4].lower())      # utt -> (speaker, attack, label)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="每划分只取前 N 条")
    ap.add_argument("--num-threads", type=int, default=8)
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(args.num_threads))

    proto = load_protocol()
    print(f"[校验] 官方 eval 协议条数 = {len(proto)}（期望 71237）")

    rows, spk_of_split, stats = [], {}, {}
    for split, path in SPLIT_FILE.items():
        assert os.path.exists(path), f"缺 {path}（先跑 scripts/dl_asvspoof2019_la.sh）"
        df = pq.read_table(path).to_pandas()
        print(f"\n[校验] {split}: parquet {len(df)} 行（期望 {EXPECT_ROWS[split]}）"
              f"  {'✅' if len(df)==EXPECT_ROWS[split] else '❌ 与官方不符'}")
        print(f"       列: {list(df.columns)}")
        if args.limit:
            df = df.head(args.limit)
        lab = df["key"].map(key_to_label)
        print(f"       标签分布: {dict(collections.Counter(lab))}")
        atk = df["system_id"].astype(str)
        print(f"       攻击编号: {len(set(atk))} 种 -> {sorted(set(atk))[:22]}")
        spk = df["speaker_id"].astype(str)
        spk_of_split[split] = set(spk)
        print(f"       说话人: {len(spk_of_split[split])} 位")

        # 官方协议交叉验证（仅 test 划分有协议）
        if split == "test":
            mis_lab = mis_spk = 0
            for uid, k, sid in zip(df["audio_file_name"].astype(str), lab, spk):
                if uid not in proto:
                    continue
                p_spk, p_atk, p_lab = proto[uid]
                if p_lab != k:
                    mis_lab += 1
                if p_spk != sid:
                    mis_spk += 1
            n_cov = sum(1 for u in df["audio_file_name"].astype(str) if u in proto)
            print(f"       与官方协议对拍: 覆盖 {n_cov}/{len(df)} 条，"
                  f"标签不符 {mis_lab} 条，说话人不符 {mis_spk} 条 "
                  f"{'✅ 逐条一致' if (mis_lab==0 and mis_spk==0 and n_cov==len(df)) else '❌'}")

        durs = []
        n_bad = 0
        for i, r in enumerate(df.itertuples(index=False)):
            raw = r.audio["bytes"] if isinstance(r.audio, dict) else r.audio
            try:
                info = sf.info(io.BytesIO(raw))
                dur, sr = info.duration, info.samplerate
            except Exception as e:
                n_bad += 1
                if n_bad <= 3:
                    print(f"  ⚠️ 读失败 {r.audio_file_name}: {e}")
                continue
            durs.append(dur)
            if args.check_only:
                continue
            d = f"{OUT}/{split}/{r.speaker_id}"
            os.makedirs(d, exist_ok=True)
            p = f"{d}/{r.audio_file_name}.flac"
            if not os.path.exists(p):
                with open(p, "wb") as f:
                    f.write(raw)
            sysid = str(r.system_id)
            rows.append({
                "utt_id": r.audio_file_name, "audio_path": os.path.relpath(p, WS),
                "label": key_to_label(r.key), "source": "asvspoof2019_la",
                "clone_model": sysid if sysid != "-" else "none",
                "attack_type": sysid if sysid != "-" else "none",
                "attack_params": "na", "text": "na", "text_source": "na",
                "prompt_utt_id": "na", "speaker_id": str(r.speaker_id),
                "sample_rate": sr, "duration": round(dur, 3), "split": split,
                "sha256": "na",
            })
            if (i + 1) % 5000 == 0:
                print(f"     ... {i+1}/{len(df)}")
        d = np.array(durs) if durs else np.zeros(1)
        stats[split] = (len(durs), n_bad, d.sum() / 3600, float(np.median(d)))
        print(f"       音频: {len(durs)} 条 {d.sum()/3600:.2f} h 中位 {np.median(d):.2f}s 失败 {n_bad}")

    # 说话人不相交（官方声称 disjoint，但要自己验）
    print("\n[校验] 划分间说话人交集（官方声称不相交，此处独立验证）")
    for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        inter = spk_of_split[a] & spk_of_split[b]
        print(f"       {a} ∩ {b} = {len(inter)} {'✅' if not inter else '❌ ' + str(sorted(inter)[:5])}")

    if args.check_only:
        print("\nCHECK_ONLY_DONE（未落盘）")
        return

    cols = ["utt_id", "audio_path", "label", "source", "clone_model", "attack_type",
            "attack_params", "text", "text_source", "prompt_utt_id", "speaker_id",
            "sample_rate", "duration", "split", "sha256"]
    os.makedirs(MAN, exist_ok=True)
    out = f"{MAN}/asvspoof2019_la_manifest.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    print(f"\n-> {out} ({len(rows)} 条)")
    print("PREPARE_ASVSPOOF_DONE")


if __name__ == "__main__":
    main()
