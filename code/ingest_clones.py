#!/usr/bin/env python3
"""
克隆器产出 -> 检测器 manifest 的适配器（两项目合并的接口）

上游：code/voiceclone/ （Lab_VoiceClone，VALL-E + Token-DiffWave）
下游：code/ 下的检测链（AASIST + metrics.py 的评估协议）

## 为什么要这个适配器
两个项目各有一套目录与命名约定，且有三处**必须转换**：
  1. **采样率**：克隆器输出 24 kHz；AASIST 要求 16 kHz
  2. **划分归属**：克隆件必须继承**参考音频说话人**的 split，否则 speaker-disjoint 被破坏
  3. **跨句核验**：任务 02 §3.2 规定 1 要求克隆目标文本 ≠ 参考音频自身文本，
     且结果要写进 manifest 的 `prompt_utt_id` 列可核查

## 输入元数据格式（克隆侧产出，JSON Lines，每行一条）
  {"wav": "...", "prompt_wav": "...", "prompt_utt_id": "ls-test-00012",
   "prompt_text": "...", "target_text": "...", "clone_model": "valle_diffwave",
   "speaker_id": 2078}
前五项必需；`clone_model` 缺省 "valle_diffwave"；`speaker_id` 缺省从参考音频查表。

用法:
  python ingest_clones.py --meta clones.jsonl --out-manifest data/manifests/fake_clone_manifest.csv
  python ingest_clones.py --meta clones.jsonl --out ... --dry-run
"""
import argparse, csv, hashlib, json, os, sys
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import os

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR_TARGET = 16000                      # AASIST 要求
OUT_AUDIO = f"{WS}/data/fake_clone"    # 统一落盘位置
REAL_MAN = f"{WS}/data/manifests/data_manifest_15col.csv"


def sha256_of(p, bs=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(bs), b""):
            h.update(c)
    return h.hexdigest()


def match_rms(y, ref, peak_limit=0.99):
    """响度对齐 —— 任务 02 §8.3 强制检查项 1；理由见 make_signalproc_fake.py"""
    ry = np.sqrt(np.mean(y ** 2)) + 1e-12
    rr = np.sqrt(np.mean(ref ** 2)) + 1e-12
    y = y * (rr / ry)
    pk = np.abs(y).max()
    if pk > peak_limit:
        y = y * (peak_limit / pk)
    return y


def norm_text(s):
    """文本归一化后比较（大小写/标点/空白差异不算跨句失败）"""
    import re
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="克隆侧元数据 JSONL")
    ap.add_argument("--real-manifest", default=REAL_MAN,
                    help="真实语音 manifest，用于查参考音频的说话人与 split")
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--out-audio", default=OUT_AUDIO)
    ap.add_argument("--dry-run", action="store_true", help="只校验不落盘")
    args = ap.parse_args()

    real = pd.read_csv(args.real_manifest)
    by_utt = {r.utt_id: r for r in real.itertuples(index=False)}
    # 参考音频路径 -> 真实行 的备用索引
    by_path = {os.path.basename(str(r.audio_path)): r for r in real.itertuples(index=False)}

    rows, problems = [], []
    n_cross_fail = n_dup = n_nofind = 0

    with open(args.meta) as f:
        metas = [json.loads(l) for l in f if l.strip()]
    print(f"[协议] 克隆元数据 {len(metas)} 条")

    for i, m in enumerate(metas):
        for k in ("wav", "prompt_wav", "prompt_utt_id", "prompt_text", "target_text"):
            if k not in m:
                problems.append(f"缺字段 {k}: {m.get('wav','?')}")
                continue

        # ---- 跨句核验（任务 02 §3.2 规定 1）----
        if norm_text(m["prompt_text"]) == norm_text(m["target_text"]):
            n_cross_fail += 1
            problems.append(f"❌ 跨句失败（目标文本 == 参考文本）: {m['prompt_utt_id']}")
            continue

        # ---- 找参考音频对应的真实条目（决定 speaker 与 split）----
        ref = by_utt.get(m["prompt_utt_id"])
        if ref is None:
            ref = by_path.get(os.path.basename(str(m["prompt_wav"])))
        if ref is None:
            n_nofind += 1
            problems.append(f"❌ 参考音频查不到对应真实条目: {m['prompt_utt_id']}")
            continue
        spk, split = int(ref.speaker_id), ref.split

        # ---- 音频：重采样 24k->16k + 响度对齐 ----
        src = os.path.join(WS, m["wav"]) if not os.path.isabs(m["wav"]) else m["wav"]
        if not os.path.exists(src):
            problems.append(f"❌ 克隆音频不存在: {src}")
            continue
        y, fs = sf.read(src, dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        if fs != SR_TARGET:
            y = librosa.resample(y, orig_sr=fs, target_sr=SR_TARGET)

        rp = os.path.join(WS, ref.audio_path) if not os.path.isabs(ref.audio_path) else ref.audio_path
        x, _ = sf.read(rp, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        y = match_rms(y, x)                      # 强制检查项 1

        model = m.get("clone_model", "valle_diffwave")
        utt = f"clone-{split}-{i:05d}"
        out = f"{args.out_audio}/{model}/{m['prompt_utt_id']}_{i:05d}.wav"

        if not args.dry_run:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            sf.write(out, np.clip(y, -1, 1).astype(np.float32), SR_TARGET)

        rows.append({
            "utt_id": utt,
            "audio_path": os.path.relpath(out, WS),
            "label": "fake", "source": "clone", "clone_model": model,
            "attack_type": "none", "attack_params": "na",
            "text": m["target_text"], "text_source": "gt",
            "prompt_utt_id": m["prompt_utt_id"],     # ← 跨句核验的证据字段
            "speaker_id": spk, "sample_rate": SR_TARGET,
            "duration": round(len(y) / SR_TARGET, 3), "split": split,
            "sha256": "" if args.dry_run else sha256_of(out),
        })

    # ---- 汇总 ----
    print(f"\n===== 结果 =====")
    print(f"  可用        : {len(rows)}")
    print(f"  跨句失败    : {n_cross_fail}   ← 任务 02 §3.2 规定 1，必须为 0")
    print(f"  参考查不到  : {n_nofind}")
    if problems:
        print(f"  其他问题 {len(problems)} 条，前 5:")
        for p in problems[:5]:
            print(f"    {p}")
    if rows:
        import collections
        c = collections.Counter(r["split"] for r in rows)
        print(f"  按 split    : {dict(c)}")

    # 任务 02 §8.3：三条强制检查中任何一条不通过，该批不得进入训练集
    if n_cross_fail > 0:
        print(f"\n❌ 该批被拒：{n_cross_fail} 条跨句失败（目标文本 == 参考文本）。")
        print("   按任务 02 §3.2 规定 1 与 §8.3，此批不得进入训练集，请先修生成参数再重跑。")
        return 2
    if n_nofind > 0:
        print(f"\n❌ 该批被拒：{n_nofind} 条参考音频查不到对应真实条目，无法确定说话人与 split。")
        return 2

    if args.dry_run:
        print("\n[dry-run] 未落盘")
        return 0
    if not rows:
        sys.exit("❌ 无可用条目")

    cols = ["utt_id", "audio_path", "label", "source", "clone_model", "attack_type",
            "attack_params", "text", "text_source", "prompt_utt_id", "speaker_id",
            "sample_rate", "duration", "split", "sha256"]
    os.makedirs(os.path.dirname(args.out_manifest), exist_ok=True)
    with open(args.out_manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"\n-> {args.out_manifest}  ({len(rows)} 行)")
    print("INGEST_CLONES_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
