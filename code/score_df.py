#!/usr/bin/env python3
"""
DF 轨专用打分（预注册见 results/df_eval_prereg.md）。

① 定长口径**不在这里重复实现**——直接用现有 score_group.py（np.tile 到 64600）。
本脚本只做另外两种：
  --mode full    ②全长-整条：每文件按原生长度整条前向（逐条、不截不拼；变长不拼批）
  --mode window  ②b全长-滑窗：64600 窗 / hop 32300，窗分取**均值**；
                 <64600 的文件按 ① 的 np.tile 单窗；
                 末端有未覆盖剩余时，**补一窗锚定在文件末端**（保证全长被覆盖，n_windows 可查）

输出 npz：utt_id / score / y / speaker_id / label / n_windows
表注纪律：DF 块只报 EER/AUC 等阈值无关指标（见预注册"聚合规则的阈值声明"）。
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, "code", "aasist"))
sys.path.insert(0, os.path.join(WS, "code"))

from score_group import build_model                                     # noqa: E402
from detector_dataset import load_manifest, load_wav_16k, pad, NB_SAMP  # noqa: E402
from train_detector import D_ARGS, FREQ_AUG                             # noqa: E402

HOP = 32300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--mode", choices=["full", "window"], required=True)
    ap.add_argument("--model-config", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-threads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="window 模式的窗批量；full 模式恒逐条（变长不拼批）")
    ap.add_argument("--allow-partial-load", action="store_true")
    ap.add_argument("--save-scores", required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.num_threads)
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available())
                          else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    d_args = ck.get("d_args", D_ARGS)
    if args.model_config:
        import json as _json
        cfg = _json.load(open(args.model_config))
        d_args = cfg.get("model_config", cfg)
        print(f"[协议] model_config = {args.model_config}")
    model = build_model(d_args).to(device)
    miss, unexp = model.load_state_dict(ck.get("model_state_dict", ck), strict=False)
    print(f"[协议] ckpt={args.ckpt} load_state missing={len(miss)} unexpected={len(unexp)}")
    if (miss or unexp) and not args.allow_partial_load:
        raise SystemExit(f"❌ 权重与模型不匹配（missing={len(miss)} unexpected={len(unexp)}）")
    model.eval()

    df = load_manifest(args.manifest)
    df = df[df["split"] == args.split].reset_index(drop=True)
    assert len(df) > 0, f"split={args.split} 为空"
    print(f"[协议] mode={args.mode} manifest={args.manifest} n_eval={len(df)} "
          f"(real {int((df.y==1).sum())} / fake {int((df.y==0).sum())})", flush=True)

    def fwd(xb):                                    # xb: (B, T) float32 ndarray
        with torch.no_grad():
            _, out = model(torch.from_numpy(xb).to(device), Freq_aug=FREQ_AUG)
        return out[:, 1].float().cpu().numpy()      # 索引 1 = bonafide

    S = np.zeros(len(df), dtype=np.float64)
    NW = np.zeros(len(df), dtype=np.int64)
    t0 = time.time()

    if args.mode == "full":
        for i, rel in enumerate(df.audio_path.tolist()):
            x = load_wav_16k(rel if os.path.isabs(rel) else os.path.join(WS, rel))
            S[i] = float(fwd(x[None, :])[0]); NW[i] = 1
            if (i + 1) % 500 == 0:
                print(f"  ... {i+1}/{len(df)}  ({time.time()-t0:.0f}s)", flush=True)
    else:
        win_sum = np.zeros(len(df))
        buf = []

        def flush():
            if not buf:
                return
            sc = fwd(np.stack([b[0] for b in buf]))
            for b, s in zip(buf, sc):
                win_sum[b[1]] += s; NW[b[1]] += 1
            buf.clear()

        for i, rel in enumerate(df.audio_path.tolist()):
            x = load_wav_16k(rel if os.path.isabs(rel) else os.path.join(WS, rel))
            if len(x) <= NB_SAMP:
                wins = [pad(x)]                       # 与 ① 同口径（tile 单窗）
            else:
                wins = [x[s:s + NB_SAMP] for s in range(0, len(x) - NB_SAMP + 1, HOP)]
                if (len(x) - NB_SAMP) % HOP != 0:
                    wins.append(x[-NB_SAMP:])         # 末端锚定补窗
            for w in wins:
                buf.append((np.ascontiguousarray(w), i))
            if len(buf) >= args.batch_size:
                flush()
            if (i + 1) % 500 == 0:
                print(f"  ... {i+1}/{len(df)}  ({time.time()-t0:.0f}s)", flush=True)
        flush()
        S = win_sum / np.maximum(NW, 1)

    os.makedirs(os.path.dirname(args.save_scores), exist_ok=True)
    np.savez(args.save_scores, utt_id=df["utt_id"].values.astype(str), score=S,
             y=df.y.values, speaker_id=df["speaker_id"].values.astype(str),
             label=df["label"].values.astype(str), n_windows=NW)
    print(f"分数已存 {args.save_scores}（mode={args.mode}；n_windows 中位 {int(np.median(NW))}，"
          f"max {int(NW.max())}；{time.time()-t0:.0f}s）", flush=True)
    print("SCORE_DF_DONE")


if __name__ == "__main__":
    main()
