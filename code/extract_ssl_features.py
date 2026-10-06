#!/usr/bin/env python3
"""
SSL 前端特征提取（冻结 WavLM-large -> 缓存到磁盘）

【为什么缓存】
WavLM 前端**冻结**，特征不随训练变化 -> 只需算一次。
否则每个 epoch 都要过 315M 参数的模型，训练不可行。
缓存后训练只读 npy，速度提升一个量级。

【形状】
  waveform (64600,)  ->  WavLM-large  ->  last_hidden_state (201, 1024)   [fp16 存盘]
  说明: 64600 样本 = 4.04s @16k；WavLM 的 conv 下采样使 T' ≈ 201

【口径】
用 WavLM 自带的 WavLMModel 前向（含其内部的 feature normalization），
不自己手写归一化 —— 与 HuggingFace 预训练时的处理一致。

用法:
  python extract_ssl_features.py --manifest data/manifests/aug_train_manifest.csv \
      --split-col split --out-dir data/ssl_cache/wavlm-large
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import os

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = f"{WS}/models/wavlm-large"
SR = 16000
NB_SAMP = 64600


def load_wav16k(path):
    x, fs = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if fs != SR:
        import librosa
        x = librosa.resample(x, orig_sr=fs, target_sr=SR)
    return np.ascontiguousarray(np.asarray(x, dtype=np.float32))


def pad_or_crop(x, n=NB_SAMP):
    """定长：短则右侧重复填充，长则取前 n —— 与检测器管线一致（保证可比）"""
    if len(x) >= n:
        return x[:n]
    return np.tile(x, int(n / len(x)) + 1)[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", default=f"{WS}/data/ssl_cache/wavlm-large")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num-threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.num_threads)
    os.makedirs(args.out_dir, exist_ok=True)

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[协议] device     = {dev}")
    print(f"[协议] model      = {MODEL_DIR}")
    print(f"[协议] manifest   = {args.manifest}")
    print(f"[协议] out_dir    = {args.out_dir}")

    from transformers import WavLMModel
    model = WavLMModel.from_pretrained(MODEL_DIR).to(dev).eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[协议] WavLM 参数量 = {n_par:,}（全部冻结）")
    for p in model.parameters():
        p.requires_grad_(False)

    df = pd.read_csv(args.manifest)
    if args.limit:
        df = df.head(args.limit)
    print(f"[协议] n_utts     = {len(df)}")

    todo = []
    for r in df.itertuples(index=False):
        p = r.audio_path if os.path.isabs(r.audio_path) else os.path.join(WS, r.audio_path)
        out = os.path.join(args.out_dir, f"{r.utt_id}.npy")
        if not os.path.exists(out):
            todo.append((r.utt_id, p, out))
    print(f"[协议] 待提取     = {len(todo)}  (已完成 {len(df)-len(todo)}，断点续跑)")

    t0 = time.time()
    n_done, n_fail = len(df) - len(todo), 0
    for i in range(0, len(todo), args.batch_size):
        batch = todo[i:i + args.batch_size]
        xs = []
        for uid, p, out in batch:
            try:
                xs.append(pad_or_crop(load_wav16k(p)))
            except Exception as e:
                print(f"  ⚠️ 读失败 {uid}: {e}"); xs.append(np.zeros(NB_SAMP, np.float32)); n_fail += 1
        wav = torch.from_numpy(np.stack(xs)).to(dev)
        with torch.no_grad():
            h = model(wav).last_hidden_state            # (B, T', 1024)
        h = h.half().cpu().numpy()
        for j, (uid, p, out) in enumerate(batch):
            np.save(out, h[j])
            n_done += 1
        if (i // args.batch_size + 1) % 20 == 0:
            el = time.time() - t0
            done = i + len(batch)
            eta = el / max(done, 1) * (len(todo) - done)
            print(f"  ... {done}/{len(todo)}  {done/max(el,1e-9):.1f} utt/s  ETA {eta/60:.1f} min")

    print(f"\n完成 {n_done} 条 (失败 {n_fail})，耗时 {(time.time()-t0)/60:.1f} min")
    # 抽查一个产出
    k = os.path.join(args.out_dir, f"{df.utt_id.iloc[0]}.npy")
    if os.path.exists(k):
        a = np.load(k)
        print(f"抽查 {os.path.basename(k)}: shape={a.shape} dtype={a.dtype} "
              f"finite={np.isfinite(a).all()}")
    print("EXTRACT_SSL_DONE")


if __name__ == "__main__":
    main()
