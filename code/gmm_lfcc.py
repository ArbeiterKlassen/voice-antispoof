#!/usr/bin/env python3
"""
LFCC-GMM 第三后端（预注册见 results/gmm_prereg.md）

三个子命令：
  feats  —— 抽训练帧（LA train，逐条 ≤200 帧）存 npy
  train  —— 拟合两条 GMM（真实/伪造），各 256 混合对角协方差（官方 512 全协方差的降档，已声明）
  score  —— 对给定 manifest 流式打分（不落特征），输出 npz(score/y/speaker/attack_type/label)

特征口径（写死）：LFCC 20 维（25ms/10ms, 16kHz）+ Δ + ΔΔ（标准回归窗 2）⇒ 60 维/帧；
逐条 CMVN；能量 VAD（≥ 中位 −30 dB，帧数 <20 则整条保留）。
分数 = mean_t [ LLK_real(f_t) − LLK_fake(f_t) ]。
"""
import argparse
import os
import sys
import time

import numpy as np

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 16000
N_CEPS = 20
F_LEN, F_SHIFT = 400, 160      # 25ms / 10ms @16k
MAX_FRAMES_PER_UTT = 200


def load_audio(rel):
    # ⚠️ librosa 惰性导入：它会拖进 numba（数百 MB 冷读），而 LA/DF 全是 16k——
    # 8 worker × 冷导入在共享机械盘上是分钟级开销（2026-10-07 实测 stime 打满的原因之一）
    import soundfile as sf
    p = rel if os.path.isabs(rel) else os.path.join(WS, rel)
    x, fs = sf.read(p, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if fs != SR:
        import librosa                      # 只在真需要重采样时才付这个冷导入
        x = librosa.resample(x, orig_sr=fs, target_sr=SR)
    return x


def frames_energy(x):
    n = 1 + max(0, (len(x) - F_LEN)) // F_SHIFT
    if n <= 0:
        return None
    idx = np.arange(F_LEN)[None, :] + F_SHIFT * np.arange(n)[:, None]
    fr = x[idx]
    return 10 * np.log10((fr ** 2).mean(axis=1) + 1e-12)


def delta(feat, win=2):
    """标准回归差分（窗 2）"""
    n = len(feat)
    denom = 2 * sum(i * i for i in range(1, win + 1))
    pad = np.pad(feat, ((win, win), (0, 0)), mode="edge")
    d = np.zeros_like(feat)
    for t in range(n):
        s = np.zeros(feat.shape[1], dtype=np.float32)
        for i in range(1, win + 1):
            s += i * (pad[t + win + i] - pad[t + win - i])
        d[t] = s / denom
    return d


# ---- LFCC（手写；torchaudio≥2.1 已删 kaldi.lfcc）----
from scipy.fftpack import dct as _dct          # noqa: E402
_NFFT = 512
_N_FILT = 20                                   # 线性三角滤波组个数
_HZ_LO, _HZ_HI = 0.0, SR / 2                   # 线性频率范围
_fb_cache = None


def _lin_fb():
    """线性间隔三角滤波组 (n_fft//2+1, N_FILT)"""
    freqs = np.fft.rfftfreq(_NFFT, 1.0 / SR)
    edges = np.linspace(_HZ_LO, _HZ_HI, _N_FILT + 2)
    fb = np.zeros((len(freqs), _N_FILT), dtype=np.float32)
    for i in range(_N_FILT):
        lo, ce, hi = edges[i], edges[i + 1], edges[i + 2]
        fb[:, i] = np.clip(np.minimum((freqs - lo) / (ce - lo + 1e-9),
                                      (hi - freqs) / (hi - ce + 1e-9)), 0, None)
    return fb


def lfcc_manual(x):
    """(n_frames, 20)：STFT→线性滤波组→log→DCT-II 取 c0..c19（Kaldi 默认含 c0）。
    窗=Hamming；帧 25ms/10ms，帧起点 0 与 frames_energy 对齐。
    ⚠️ 曾有 bug：取 c[1:21] 在 20 滤波组下只剩 19 列（DCT 输出长度=滤波组数）——改为 c0..c19。"""
    global _fb_cache
    if _fb_cache is None:
        _fb_cache = _lin_fb()
    n = 1 + max(0, (len(x) - F_LEN)) // F_SHIFT
    if n <= 0:
        return None
    idx = np.arange(F_LEN)[None, :] + F_SHIFT * np.arange(n)[:, None]
    fr = x[idx] * np.hamming(F_LEN)[None, :]
    sp = np.abs(np.fft.rfft(fr, n=_NFFT)) ** 2          # (n, 257)
    e = sp @ _fb_cache + 1e-10                           # (n, 20)
    c = _dct(np.log(e), type=2, axis=1, norm="ortho")    # (n, 20)
    return c[:, :N_CEPS].astype(np.float32)              # c0..c19（20 维）


def feat_one(rel):
    """返回 (n_frames, 60) float32；失败返回 None"""
    try:
        x = load_audio(rel)
        e = frames_energy(x)
        lf = lfcc_manual(x)
        if e is None or lf is None:
            return None
        n = min(len(lf), len(e))
        lf, e = lf[:n], e[:n]
        gate = e >= (np.median(e) - 30.0)
        if gate.sum() >= 20:
            lf = lf[gate]
        # CMVN（逐条）
        mu, sd = lf.mean(0, keepdims=True), lf.std(0, keepdims=True) + 1e-8
        lf = (lf - mu) / sd
        f = np.concatenate([lf, delta(lf), delta(delta(lf))], axis=1).astype(np.float32)
        return f
    except Exception as ex:
        print(f"  ⚠️ feat 失败 {rel}: {str(ex)[:80]}", flush=True)
        return None


def _feat_worker(rel):
    return feat_one(rel)


def cmd_feats(args):
    """分片流式：每 `--shard-files` 条 flush 一个 npz（单次分配 ≤ ~155MB）。

    ⚠️ 为什么分片：本机（251GB 无 swap、60 人共享）对**单次 ≥~1GB 匿名分配**触发内核自旋
    （utime 冻结 + stime ~100 tick/s + minflt 不涨）——25380 条一次性 concatenate→1.22GB
    已实测卡死；历史处方「拆大分配」每次有效。单分片峰值 3200×200×60×4B ≈ 154MB，落在安全区。
    """
    import pandas as pd
    from multiprocessing import Pool
    df = pd.read_csv(args.manifest)
    df = df[df.split == "train"].copy()
    df["y"] = (df.label == "real").astype(int)   # manifest 列名是 label，本项目约定 real=1
    # 按路径排序：① manifest 原序按标签排（首片会全 real），② 全局打散会把 /data1 机械盘
    #   变成随机寻道（实测速率崩到 ~15 条/s）。路径序实测类别混合 9.8%≈全局（分片仍均衡），
    #   且读盘顺序化（同说话人目录聚簇）。cmd_train 的配额按分片类别计数，不依赖均衡。
    df = df.sort_values("audio_path").reset_index(drop=True)
    if args.block:                               # "START:END" 只处理该行区间（并行分块后备路径）
        a, b = (int(x) for x in args.block.split(":"))
        df = df.iloc[a:b].reset_index(drop=True)
    print(f"[feats] {len(df)} 条 train（逐条 ≤{MAX_FRAMES_PER_UTT} 帧；"
          f"每 {args.shard_files} 条一个分片）", flush=True)
    prefix = args.out[:-4] if args.out.endswith(".npz") else args.out
    buf_f, buf_y, shards, t0 = [], [], [], time.time()

    def flush():
        if not buf_f:
            return
        p = f"{prefix}_{len(shards):03d}.npz"
        # ⚠️ Y 必须是**帧级**（与 F 行数一致）：训练端按帧做布尔索引 F[Y==lab]。
        # 曾经错存成文件级（3200 个文件标签 vs 60 万行帧），被 F[sel] 当成行号静默取前缀
        # → 只用了十几条文件的帧训练（G1-a EER 44.5% 事故）。断言防回归。
        F_sh = np.concatenate(buf_f)
        Y_sh = np.repeat(np.asarray(buf_y, dtype=np.int64), [len(f) for f in buf_f])
        assert len(F_sh) == len(Y_sh), f"分片 F/Y 长度不等：{len(F_sh)} vs {len(Y_sh)}"
        np.savez(p, F=F_sh, Y=Y_sh)
        shards.append(p)
        nr_files = int(np.sum(buf_y))
        print(f"  [shard] {os.path.basename(p)}：{len(buf_f)} 条 / {len(F_sh)} 帧"
              f"（real 文件 {nr_files} / fake {len(buf_y)-nr_files}）", flush=True)
        buf_f.clear(); buf_y.clear()

    with Pool(args.workers) as pool:
        for i, (f, y) in enumerate(zip(pool.imap(_feat_worker, df.audio_path.tolist(), chunksize=16),
                                       df.y.tolist())):
            if f is None:
                continue
            if len(f) > MAX_FRAMES_PER_UTT:
                sel = np.linspace(0, len(f) - 1, MAX_FRAMES_PER_UTT).astype(int)
                f = f[sel]
            buf_f.append(f); buf_y.append(y)
            if len(buf_f) >= args.shard_files:
                flush()
            if (i + 1) % 2000 == 0:
                print(f"  ... {i+1}/{len(df)}  ({time.time()-t0:.0f}s)", flush=True)
    flush()
    with open(f"{prefix}_shards.txt", "w") as fh:
        fh.write("\n".join(shards) + "\n")
    print(f"[feats] 完成：{len(shards)} 分片 -> {prefix}_NNN.npz（清单 {prefix}_shards.txt）",
          flush=True)


def cmd_train(args):
    """分片感知：`--feats` 可以是 glob（如 train_feats_*.npz），逐片按配额子采样。

    配额 = n_train_frames × 该片该类帧数 / 全体该类帧数（末片兜底，总量恰为 n_train_frames）；
    峰值内存 ≈ 单分片 F（~155MB）+ 子采样累积（≤48MB）。
    """
    import glob as _glob
    import pickle
    from sklearn.mixture import GaussianMixture
    paths = sorted(_glob.glob(args.feats)) if _glob.has_magic(args.feats) else [args.feats]
    if not paths:
        sys.exit(f"[train] 无特征分片匹配：{args.feats}")
    cnt = []
    tot = {1: 0, 0: 0}
    for p in paths:
        z = np.load(p); F, Y = z["F"], z["Y"]
        assert len(F) == len(Y), (f"分片 {p} 的 F/Y 长度不等（{len(F)} vs {len(Y)}）——"
                                  f"Y 必须是帧级标签；文件级 Y 会被 F[sel] 静默当行号用")
        nr, nf = int((Y == 1).sum()), int((Y == 0).sum())
        cnt.append((p, nr, nf)); tot[1] += nr; tot[0] += nf
    print(f"[train] {len(paths)} 分片，帧计 real {tot[1]} / fake {tot[0]}；"
          f"每类子采样 {args.n_train_frames}", flush=True)
    rng = np.random.RandomState(args.seed)
    out = {}
    for name, lab in [("real", 1), ("fake", 0)]:
        parts, got = [], 0
        for i, (p, nr, nf) in enumerate(cnt):
            z = np.load(p); F, Y = z["F"], z["Y"]
            idx = np.where(Y == lab)[0]
            if i < len(cnt) - 1:
                q = min(len(idx), int(round(args.n_train_frames * len(idx) / max(tot[lab], 1))))
            else:
                q = min(len(idx), args.n_train_frames - got)   # 末片兜底
            sel = rng.choice(idx, q, replace=False) if q < len(idx) else idx
            parts.append(F[sel]); got += len(sel)
            del z, F, Y
        X = np.concatenate(parts) if len(parts) > 1 else parts[0]
        print(f"[train] {name}: 子采样 {len(X)} 帧 → GMM({args.components}, diag)", flush=True)
        g = GaussianMixture(n_components=args.components, covariance_type="diag",
                            max_iter=args.max_iter, tol=1e-4, random_state=args.seed,
                            verbose=1, verbose_interval=10)
        g.fit(X)
        out[name] = g
    with open(args.model, "wb") as fh:
        pickle.dump(out, fh)
    print(f"[train] 完成 -> {args.model}", flush=True)


_G = None          # 每个 worker 进程各持一份（initializer 载入；局部函数不可 pickle，见下）


def _init_worker(model_path):
    global _G
    import pickle
    with open(model_path, "rb") as fh:
        _G = pickle.load(fh)


def _score_worker(rel):
    f = feat_one(rel)
    if f is None:
        return np.nan
    return float(np.mean(_G["real"].score_samples(f) - _G["fake"].score_samples(f)))


def cmd_score(args):
    import pandas as pd
    from multiprocessing import Pool
    df = pd.read_csv(args.manifest)
    if args.split:
        df = df[df.split == args.split]
    df = df.reset_index(drop=True).copy()
    df["y"] = (df.label == "real").astype(int)
    print(f"[score] {len(df)} 条 @ {os.path.basename(args.manifest)}", flush=True)

    t0 = time.time()
    with Pool(args.workers, initializer=_init_worker, initargs=(args.model,)) as pool:
        scores = list(pool.imap(_score_worker, df.audio_path.tolist(), chunksize=16))
    S = np.array(scores)
    bad = int(np.isnan(S).sum())
    if bad:
        print(f"  ⚠️ {bad} 条特征失败，以该列中位置换（并记录）", flush=True)
        S[np.isnan(S)] = np.nanmedian(S)
    # ⚠️ `.values.astype(str)`（先 values 后 astype）产出定宽 <U 数组；
    # 反过来 `.astype(str).values` 是 object 数组，allow_pickle=False 读不了（与既有 npz 口径不符）
    np.savez(args.out, score=S, y=df.y.values, speaker_id=df.speaker_id.values.astype(str),
             attack_type=df.attack_type.values.astype(str), label=df.label.values.astype(str),
             utt_id=df.utt_id.astype(str).values)
    print(f"[score] {len(df)} 条完成（失败 {bad}），{time.time()-t0:.0f}s -> {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("feats"); p1.add_argument("--manifest", required=True)
    p1.add_argument("--out", required=True); p1.add_argument("--workers", type=int, default=8)
    p1.add_argument("--shard-files", type=int, default=3200,
                    help="每条分片条数（峰值内存 ≈ shard-files×200×60×4B；3200≈154MB）")
    p1.add_argument("--seed", type=int, default=20261006, help="输入打散种子（分片比例稳定用）")
    p1.add_argument("--block", default="", help='只处理路径排序后的行区间 "START:END"（并行分块用）')
    p1.set_defaults(fn=cmd_feats)

    p2 = sub.add_parser("train"); p2.add_argument("--feats", required=True)
    p2.add_argument("--model", required=True); p2.add_argument("--components", type=int, default=256)
    p2.add_argument("--max-iter", type=int, default=100)
    p2.add_argument("--n-train-frames", type=int, default=200000)
    p2.add_argument("--seed", type=int, default=20261006)
    p2.set_defaults(fn=cmd_train)

    p3 = sub.add_parser("score"); p3.add_argument("--manifest", required=True)
    p3.add_argument("--model", required=True); p3.add_argument("--out", required=True)
    p3.add_argument("--split", default=""); p3.add_argument("--workers", type=int, default=8)
    p3.set_defaults(fn=cmd_score)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
