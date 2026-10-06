#!/usr/bin/env python3
"""
ASVspoof2021 DF parquet -> flac + manifest（**跨数据集/压缩域鲁棒性轨**）

## 与 LA 轨的关系
DF 只有 eval 划分（611829 trials，真实 22617 / 伪造 589212，极度不平衡），
所以它**不能用来训练**，只能做「在别的数据上训练 → 跨到 DF」的评估。
它比 LA 更接近真实场景：**带编解码压缩**（codec 字段），且攻击系统更多。

## 🔴 必须先验证的泄漏风险
DF 的真实语音说话人 id 形如 `LA_0023`，与 ASVspoof2019 LA 同一命名体系。
若 DF 的真实样本与 **LA train/dev** 说话人重叠，则在 LA 上训练后评 DF 就是**说话人泄漏**，
结果全部作废。本脚本**强制检查**该交集，不通过即拒写。
（若与 LA **test** 重叠则无害——test 本来就是 held-out。）

## 抽样设计（611829 条全评不现实）
- **真实全留**（22617 条）：EER 的方差主要由少数类决定，不能抽稀。
- 伪造按 (codec, attack_id) 分层抽 1 倍真实数（约 22617），保持压缩/攻击构成。
→ 约 4.5 万条，GPU 上评估约 20-30 min。

## 标签约定
parquet `label`：0=bonafide / 1=spoof（与本项目 real=1 相反，此处显式转换）。
另用同仓的 `data/labels.parquet` 做**独立第二来源**对拍。

用法:
  python prepare_asvspoof_df.py --check-only        # 只读索引+校验，不写盘
  python prepare_asvspoof_df.py --max-spoof 22617
"""
import argparse, csv, glob, io, json, os, sys, collections
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import os

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = f"{WS}/data/_downloads/df"   # 2026-10-05 改指完整副本（旧 asvspoof2021_df/ 是限流期残缺件，已改名）
OUT = f"{WS}/data/asvspoof_df"
MAN = f"{WS}/data/manifests"
LA_MAN = f"{MAN}/asvspoof2019_la_manifest.csv"


def lab_of(v):
    s = str(v).strip().lower()
    if s in ("0", "bonafide", "real"):
        return "real"
    if s in ("1", "spoof", "fake"):
        return "fake"
    raise ValueError(f"未知 label: {v!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--max-spoof", type=int, default=22617)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--num-threads", type=int, default=8)
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(args.num_threads))

    shards = sorted(glob.glob(f"{SRC}/data/test-*.parquet"))
    assert shards, f"没有分片：{SRC}/data/test-*.parquet（先跑下载）"
    print(f"[1/4] 读 {len(shards)} 个分片的元数据列（不碰音频）")
    idx = []
    for i, p in enumerate(shards):
        t = pq.read_table(p, columns=["path", "label", "notes"])
        d = t.to_pandas()
        for path, lb, nt in zip(d["path"], d["label"], d["notes"]):
            n = json.loads(nt) if isinstance(nt, str) else (nt or {})
            idx.append((path, lab_of(lb), n.get("utterance_id"), n.get("speaker_id"),
                        n.get("codec"), n.get("attack_id"), n.get("vocoder")))
        if (i + 1) % 20 == 0:
            print(f"    ... {i+1}/{len(shards)} 分片，累计 {len(idx)} 条")
    import pandas as pd
    df = pd.DataFrame(idx, columns=["path", "label", "utt_id", "speaker_id",
                                    "codec", "attack_id", "vocoder"])
    print(f"    合计 {len(df)} 条  真实 {int((df.label=='real').sum())} / 伪造 {int((df.label=='fake').sum())}")
    print(f"    codec 分布: {dict(df.codec.value_counts())}")
    print(f"    攻击系统 {df.attack_id.nunique()} 种")

    # ---- 独立第二来源：labels.parquet ----
    lp = f"{SRC}/data/labels.parquet"
    if os.path.exists(lp):
        L = pd.read_parquet(lp)
        m = dict(zip(L.utterance_id, L.label))
        mism = sum(1 for u, l in zip(df.utt_id, df.label)
                   if u in m and lab_of(m[u]) != l)
        cov = sum(1 for u in df.utt_id if u in m)
        print(f"[2/4] 与同仓 labels.parquet 对拍: 覆盖 {cov}/{len(df)}，标签不符 {mism} 条 "
              f"{'✅' if (mism==0 and cov==len(df)) else '❌'}")

    # ---- 🔴 泄漏检查：DF 真实说话人 vs LA 说话人 ----
    print("[3/4] 🔴 泄漏检查：DF 真实说话人 × LA 各划分说话人")
    if not os.path.exists(LA_MAN):
        print("    ⚠️ 缺 LA manifest，无法检查（先跑 prepare_asvspoof.py）")
    else:
        la = pd.read_csv(LA_MAN, usecols=["speaker_id", "split"])
        la_spk = {s: set(g.speaker_id) for s, g in la.groupby("split")}
        df_real_spk = set(df[df.label == "real"].speaker_id.dropna())
        for s in ["train", "dev", "test"]:
            inter = df_real_spk & la_spk.get(s, set())
            bad = (s in ("train", "dev") and inter)
            print(f"    DF真实 ∩ LA-{s:<5} = {len(inter):>4} 个说话人 "
                  f"{'❌ 泄漏！不允许在 LA 上训练后评 DF' if bad else ('（test 重叠无害）' if s=='test' else '✅')}")
        if df_real_spk & (la_spk.get("train", set()) | la_spk.get("dev", set())):
            print("    ❌ 检出泄漏，拒绝写盘"); sys.exit(1)

    # ---- 抽样 ----
    real = df[df.label == "real"]
    spoof = df[df.label == "fake"]
    rng = np.random.RandomState(args.seed)
    n_sp = min(args.max_spoof, len(spoof))
    # 按 (codec, attack_id) 分层
    spoof = spoof.copy(); spoof["_k"] = spoof.codec.astype(str) + "|" + spoof.attack_id.astype(str)
    frac = n_sp / len(spoof)
    sub = (spoof.groupby("_k", group_keys=False)
                .apply(lambda g: g.sample(n=max(1, int(round(len(g) * frac))), random_state=args.seed)))
    sel = pd.concat([real, sub.drop(columns=["_k"])], ignore_index=True)
    print(f"[4/4] 抽样：真实 {len(real)}（全留）+ 伪造 {len(sub)}（分层抽） = {len(sel)} 条")
    print(f"     抽样后 codec 构成: {dict(sel.codec.value_counts())}")
    if args.check_only:
        print("CHECK_ONLY_DONE（未写盘）"); return

    # 按 utt_id 索引音频：逐分片流式读，只取被选中的行
    want = set(sel.utt_id)
    rows, n_bad = [], 0
    for i, p in enumerate(shards):
        t = pq.read_table(p, columns=["path", "label", "audio"])
        d = t.to_pandas()
        for path, lb, au in zip(d["path"], d["label"], d["audio"]):
            uid = os.path.splitext(str(path))[0]
            if uid not in want:
                continue
            raw = au["bytes"] if isinstance(au, dict) else au
            try:
                info = sf.info(io.BytesIO(raw))
                dur, sr = info.duration, info.samplerate
            except Exception as e:
                n_bad += 1
                if n_bad <= 5: print(f"  ⚠️ 读失败 {uid}: {str(e)[:70]}")
                continue
            lab = lab_of(lb)
            dpath = f"{OUT}/{lab}"
            os.makedirs(dpath, exist_ok=True)
            fp = f"{dpath}/{uid}.flac"
            if not os.path.exists(fp):
                with open(fp, "wb") as f:
                    f.write(raw)
            meta = sel[sel.utt_id == uid].iloc[0]
            rows.append({
                "utt_id": uid, "audio_path": os.path.relpath(fp, WS), "label": lab,
                "source": "asvspoof2021_df", "clone_model": str(meta.attack_id),
                "attack_type": str(meta.attack_id),
                "attack_params": f"codec={meta.codec};vocoder={meta.vocoder}",
                "text": "na", "text_source": "na", "prompt_utt_id": "na",
                "speaker_id": str(meta.speaker_id), "sample_rate": sr,
                "duration": round(dur, 3), "split": "test", "sha256": "na",
            })
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{len(shards)} 分片，已落 {len(rows)} 条（失败 {n_bad}）")

    cols = ["utt_id", "audio_path", "label", "source", "clone_model", "attack_type",
            "attack_params", "text", "text_source", "prompt_utt_id", "speaker_id",
            "sample_rate", "duration", "split", "sha256"]
    out = f"{MAN}/asvspoof2021_df_manifest.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    c = collections.Counter(r["label"] for r in rows)
    print(f"\n-> {out}  ({len(rows)} 条，real {c['real']} / fake {c['fake']}，失败 {n_bad})")
    print("PREPARE_DF_DONE")


if __name__ == "__main__":
    main()
