#!/usr/bin/env python3
"""
参数随机化攻击增广 —— 修「过拟合到固定增广条件」

## 为什么改
第一版增广用 **9 个固定条件**（rt60 只有 0.4、snr 只有 5/10、speed 只有 0.9）。
实测结果（2026-10-04，臂A vs 臂B）：
    噪声类大幅改善   awgn-snr5 22.00 -> 10.00
    **混响/变速反而恶化  reverb-0.4 5.33 -> 27.33**
而且这些条件**都在增广集里**——训过的反而变差。

判据：模型记住了「rt60=0.4 这几个具体点」，没学到「混响不改变真伪」这个一般性质。
训练曲线旁证：train EER -> 0.68%，dev EER 反升到 4.88%（典型过拟合）。

## 改法
**每个文件独立采样参数**（连续区间），而不是固定点：
    混响  rt60   ~ U(0.10, 1.20)
    噪声  snr    ~ U(0, 25) dB
    变速  factor ~ U(0.85, 1.15)
    码率  mp3/aac/opus ~ {32,48,64,96,128} kbps 随机
    带通  低频 ~ U(200,400)、高频 ~ U(3000,4000)
每源采 K 个攻击（族随机 + 参数随机）。数据量与固定版相当，但参数覆盖连续区间。

## 红线
与 make_attacks.py 相同：**响度必须对齐到源**（否则学"响=伪造"）。
"""
import argparse, csv, os, random, sys, tempfile, subprocess

# 🔴 共享机线程上限：**必须在 import numpy/librosa 之前设置**
#    （OpenMP/BLAS 在线库初始化时读取；之后设无效 —— 我在 make_attacks.py 里已犯过一次）
_NT = "8"
if "--num-threads" in sys.argv:
    try:
        _NT = sys.argv[sys.argv.index("--num-threads") + 1]
    except IndexError:
        pass
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = _NT

import numpy as np
import soundfile as sf
import librosa
import os

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 16000
OUT = f"{WS}/data/attacks_rand"
FFMPEG = None


def _ffmpeg():
    global FFMPEG
    if FFMPEG is None:
        import imageio_ffmpeg
        FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    return FFMPEG


def match_rms(y, ref, peak_limit=0.99):
    ry = np.sqrt(np.mean(y ** 2)) + 1e-12
    rr = np.sqrt(np.mean(ref ** 2)) + 1e-12
    y = y * (rr / ry)
    pk = np.abs(y).max()
    if pk > peak_limit:
        y = y * (peak_limit / pk)
    return y


def _decode(path, sr=SR):
    x, fs = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if fs != sr:
        x = librosa.resample(x, orig_sr=fs, target_sr=sr)
    return x


def _roundtrip(x, sr, args, suffix):
    ff = _ffmpeg()
    with tempfile.TemporaryDirectory() as td:
        a = os.path.join(td, "in.wav"); enc = os.path.join(td, f"e{suffix}"); dec = os.path.join(td, "dec.wav")
        sf.write(a, x, sr)
        r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", a] + args + [enc], capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"编码失败: {r.stderr.decode()[:120]}")
        r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", enc, "-ar", str(sr), "-ac", "1", dec],
                           capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"解码失败: {r.stderr.decode()[:120]}")
        y, fs = sf.read(dec, dtype="float32", always_2d=True)
        y = y.mean(axis=1)
    if fs != sr:
        y = librosa.resample(y, orig_sr=fs, target_sr=sr)
    return y


# ---------------------------------------------------------------- 随机攻击族
def a_codec(x, sr, rng):
    k = rng.choice([32, 48, 64, 96, 128])
    kind = rng.choice(["mp3", "aac", "opus"])
    if kind == "mp3":
        return _roundtrip(x, sr, ["-b:a", f"{k}k", "-ar", str(sr)], ".mp3"), f"mp3-{k}k"
    if kind == "aac":
        return _roundtrip(x, sr, ["-c:a", "aac", "-b:a", f"{k}k", "-ar", str(sr)], ".m4a"), f"aac-{k}k"
    return _roundtrip(x, sr, ["-c:a", "libopus", "-b:a", f"{k}k", "-ar", str(sr)], ".opus"), f"opus-{k}k"


def a_g711(x, sr, rng):
    return _roundtrip(x, sr, ["-c:a", "pcm_mulaw", "-ar", "8000"], ".wav"), "g711-8k"


def a_awgn(x, sr, rng):
    snr = rng.uniform(0.0, 25.0)
    r = np.random.RandomState(rng.randint(0, 2**31 - 1))
    p = np.mean(x ** 2) / (10 ** (snr / 10))
    return x + r.randn(len(x)) * np.sqrt(p), f"awgn-snr{snr:.1f}"


def a_reverb(x, sr, rng):
    rt60 = rng.uniform(0.10, 1.20)
    r = np.random.RandomState(rng.randint(0, 2**31 - 1))
    n = max(int(rt60 * sr), 8)
    h = r.randn(n) * np.exp(-6.9 * np.arange(n) / n)
    h[0] = 1.0
    h = h / np.sqrt(np.sum(h ** 2))
    return np.convolve(x, h)[:len(x)], f"reverb-rt60-{rt60:.2f}"


def a_speed(x, sr, rng):
    f = rng.uniform(0.85, 1.15)
    return librosa.effects.time_stretch(x, rate=f), f"speed{f:.3f}"


def a_bandpass(x, sr, rng):
    from scipy.signal import butter, filtfilt
    lo = rng.uniform(200, 400); hi = rng.uniform(3000, 4000)
    nyq = sr / 2.0
    b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x).astype(np.float32), f"bandpass{lo:.0f}-{hi:.0f}"


FAMILIES = {
    "codec":    a_codec,      # 每次随机选 mp3/aac/opus 与码率
    "g711":     a_g711,
    "awgn":     a_awgn,       # snr 连续 U(0,25)
    "reverb":   a_reverb,     # rt60 连续 U(0.10,1.20)
    "speed":    a_speed,      # factor 连续 U(0.85,1.15)
    "bandpass": a_bandpass,   # 带通区间连续
}
# 族的采样权重（awgn 权重高：第一版实测它在噪声类上收益最大）
WEIGHTS = {"codec": 0.20, "g711": 0.08, "awgn": 0.30, "reverb": 0.22, "speed": 0.10, "bandpass": 0.10}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-src", default=f"{WS}/data/manifests/_aug_real_src.csv")
    ap.add_argument("--fake-src", default=f"{WS}/data/manifests/_aug_fake_src.csv")
    ap.add_argument("--per-source", type=int, default=6, help="每个源采几个攻击")
    ap.add_argument("--split-in", default="train", help="只取源里此 split 的行")
    ap.add_argument("--split-out", default="train", help="产出 manifest 里写的 split")
    ap.add_argument("--out-root", default=f"{WS}/data/attacks_rand",
                    help="攻击音频落盘根目录（评估集与训练增广分开，便于审计）")
    ap.add_argument("--utt-prefix", default="augr",
                    help="产出 utt_id 前缀。⚠️ 不同用途必须用不同前缀，"
                         "否则两个 manifest 的行 id 会撞名（源音频 id 不同、行 id 却相同）")
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--num-threads", type=int, default=8)
    ap.add_argument("--out-real", default=f"{WS}/data/manifests/augr_attacked_real_manifest.csv")
    ap.add_argument("--out-fake", default=f"{WS}/data/manifests/augr_attacked_fake_manifest.csv")
    args = ap.parse_args()

    # 线程上限已在文件顶部（import 之前）设置 —— 此处设置无效，勿再加

    global OUT
    OUT = args.out_root

    import pandas as pd
    fams = list(WEIGHTS); ws = np.array([WEIGHTS[f] for f in fams]); ws = ws / ws.sum()
    cols = ["audio_path", "label", "source", "clone_model", "attack_type", "attack_params",
            "text", "speaker_id", "duration", "sample_rate", "split", "utt_id"]

    for tag, src, out, lab in [("real", args.real_src, args.out_real, "real"),
                               ("fake", args.fake_src, args.out_fake, "fake")]:
        df = pd.read_csv(src)
        if "split" in df.columns:
            df = df[df.split == args.split_in]
        df = df.reset_index(drop=True)
        print(f"\n===== {tag}: {len(df)} 源 × {args.per_source} 攻击 =====")
        rng = random.Random(args.seed + (1 if tag == "real" else 2))
        np_rng = np.random.RandomState(args.seed + (1 if tag == "real" else 2))
        rows, nfail = [], 0
        for i, r in enumerate(df.itertuples(index=False)):
            p = os.path.join(WS, r.audio_path) if not os.path.isabs(r.audio_path) else r.audio_path
            try:
                x = _decode(p)
            except Exception as e:
                print(f"  ⚠️ 读失败 {r.utt_id}: {e}"); nfail += 1; continue
            for j in range(args.per_source):
                fam = rng.choices(fams, weights=ws)[0]
                try:
                    y, pstr = FAMILIES[fam](x, SR, np_rng)
                except Exception as e:
                    nfail += 1; continue
                if not np.isfinite(y).all() or len(y) < SR // 4:
                    nfail += 1; continue
                if fam != "speed" and abs(len(y) - len(x)) > max(0.02 * len(x), SR // 100, 4096):
                    nfail += 1; continue          # 时长守卫
                y = np.clip(match_rms(y, x), -1, 1).astype(np.float32)   # 红线：响度对齐
                o = f"{OUT}/{fam}/{r.utt_id}_{j}.wav"
                os.makedirs(os.path.dirname(o), exist_ok=True)
                sf.write(o, y, SR)
                rows.append({
                    "audio_path": os.path.relpath(o, WS), "label": lab,
                    "source": f"augr_{tag}", "clone_model": "none",
                    "attack_type": fam,                # 族名（参数在 attack_params 里）
                    "attack_params": pstr, "text": r.text, "speaker_id": r.speaker_id,
                    "duration": round(len(y) / SR, 3), "sample_rate": SR,
                    "split": args.split_out,
                    "utt_id": f"{args.utt_prefix}-{tag}-{i:05d}-{j}",
                })
            if (i + 1) % 100 == 0:
                print(f"  ... {i+1}/{len(df)}  (累计 {len(rows)} 条, 失败 {nfail})")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
        import collections
        c = collections.Counter(x["attack_type"] for x in rows)
        print(f"  -> {out}  ({len(rows)} 条, 失败 {nfail})")
        print(f"     族分布: {dict(c)}")

    print("\nMAKE_AUG_RANDOM_DONE")


if __name__ == "__main__":
    main()
