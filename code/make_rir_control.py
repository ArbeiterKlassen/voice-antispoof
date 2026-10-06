#!/usr/bin/env python3
"""
RIR 控制实验 —— 隔离 `make_attacks.py` 固定 seed 缺陷的影响。

## 为什么需要它
`make_attacks.py` 的 `ATTACKS` 表不给 `att_reverb` 传 seed → 默认 0 → **全部文件共用一个 RIR**。
但直接把「固定版 reverb-rt60-0.4（单一 RIR）」与「随机版 reverb（rt60∈[0.10,1.20]、逐文件 RIR）」
相减是**混淆的**：差里混了「RIR 实现」与「参数区间变宽」两个因素。

## 本实验
只换 RIR：**rt60 固定 0.4**（与固定版同一条件），但**每个文件独立 RIR**（seed = base + 行号）。
→ 与固定版逐文件可比，差就是纯粹的「单一实现 vs 多样实现」。

其余与 make_attacks 同口径：同一 `att_reverb` 实现、同一 `match_rms` 响度对齐、同一时长守卫。
"""
import argparse, csv, os, sys

_NT = "8"
if "--num-threads" in sys.argv:
    try:
        _NT = sys.argv[sys.argv.index("--num-threads") + 1]
    except IndexError:
        pass
# 线程上限必须在 import numpy/librosa 之前设（本文件顶部，勿下移）
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _NT

import numpy as np                                             # noqa: E402
import pandas as pd                                            # noqa: E402
import soundfile as sf                                         # noqa: E402

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, "code"))
from make_attacks import att_reverb, match_rms, _decode, SR     # noqa: E402
import os

COLS = ["utt_id", "audio_path", "label", "source", "clone_model", "attack_type",
        "attack_params", "text", "text_source", "prompt_utt_id", "speaker_id",
        "sample_rate", "duration", "split", "sha256"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-root", default=f"{WS}/data/attacks_rirctl")
    ap.add_argument("--label", required=True, choices=["real", "fake"])
    ap.add_argument("--rt60", type=float, default=0.4)
    ap.add_argument("--seed-base", type=int, default=70000)
    ap.add_argument("--split-in", default="test")
    # 线程上限在文件顶部（import 之前）已设；此处仅为接收该参数不报错
    ap.add_argument("--num-threads", type=int, default=8)
    args = ap.parse_args()

    df = pd.read_csv(args.src)
    if "split" in df.columns:
        df = df[df.split == args.split_in]
    df = df.reset_index(drop=True)
    print(f"[{args.label}] 源 {len(df)} 条；rt60 固定 {args.rt60}，RIR 逐文件随机（seed={args.seed_base}+i）")
    rows, nfail = [], 0
    for i, r in enumerate(df.itertuples(index=False)):
        p = r.audio_path if os.path.isabs(r.audio_path) else os.path.join(WS, r.audio_path)
        try:
            x = _decode(p)
        except Exception as e:
            print(f"  ⚠️ 读失败 {r.utt_id}: {e}"); nfail += 1; continue
        y = att_reverb(x, SR, rt60=args.rt60, seed=args.seed_base + i)   # ← 只这一处不同
        if not np.isfinite(y).all() or abs(len(y) - len(x)) > max(0.02 * len(x), SR // 100):
            nfail += 1; continue
        y = np.clip(match_rms(y, x), -1, 1).astype(np.float32)
        o = f"{args.out_root}/{r.utt_id}.wav"
        os.makedirs(os.path.dirname(o), exist_ok=True)
        sf.write(o, y, SR)
        rows.append({"utt_id": r.utt_id, "audio_path": os.path.relpath(o, WS),
                     "label": args.label, "source": "rir_control", "clone_model": "none",
                     "attack_type": f"reverb-rt60-{args.rt60}-randRIR",
                     "attack_params": f"rt60-{args.rt60}-seed{args.seed_base + i}",
                     "text": getattr(r, "text", "na"), "text_source": "na", "prompt_utt_id": "na",
                     "speaker_id": r.speaker_id, "sample_rate": SR,
                     "duration": round(len(y) / SR, 3), "split": args.split_in, "sha256": "na"})
    pd.DataFrame(rows)[COLS].to_csv(args.out, index=False)
    print(f"  -> {args.out} ({len(rows)} 条，失败 {nfail})")
    print("RIR_CONTROL_DONE")


if __name__ == "__main__":
    main()
