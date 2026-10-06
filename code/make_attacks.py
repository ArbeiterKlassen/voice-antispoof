#!/usr/bin/env python3
"""
P4 鲁棒性攻击链 —— 对伪造语音施加后处理，检验检测器是否被打崩

任务 02 §3 P4：训练同 P1，测试用 P2 伪造再做编解码、噪声、混响、变速。
任务 02 §3.2 规定 3：每行必须写清编解码格式与码率、信噪比、混响 RT60、变速比例，
确实没有的字段写 na，不允许留空。

⚠️ 与伪造生成同样的红线：**攻击后必须重新做响度对齐**。
   编解码/加噪/滤波都会改变 RMS，不对齐就会引入"响度混淆"，
   检测器可能靠响度而非后处理伪影分辨 —— 结论失效。

攻击清单（与《实验方案》§5.1 对应）：
  编解码  mp3-128k / mp3-64k / aac-128k / opus-32k / 电话窄带(g711 8k)
  噪声    awgn snr 5 / 10 / 20 dB
  信道    reverb rt60 0.2 / 0.4 / 0.8 s ；带通 300-3400 Hz
  变速    speed 0.9 / 1.1
  采样率  8k 往返 / 16k->8k->16k

用法:
  python make_attacks.py --src data/manifests/fake_signalproc_manifest.csv \
      --limit 200 --attacks mp3-128k,awgn-snr10,reverb-rt60-0.4,speed0.9
"""
import argparse, csv, io, os, subprocess, sys, tempfile

# 🔴 共享机线程上限：**必须在 import numpy/librosa 之前设置**。
#    OpenMP/BLAS 在线库初始化时读取这些环境变量，之后再设无效
#    （第一版把设置写在 argparse 之后，实测完全不起作用）。
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
OUT = f"{WS}/data/attacks"

# ffmpeg：系统无 ffmpeg，用 imageio-ffmpeg 自带的静态二进制
def _ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()

FFMPEG = None


def match_rms(y, ref, peak_limit=0.99):
    """响度对齐（与伪造生成同一红线，理由见 make_signalproc_fake.py::match_rms）"""
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


def _roundtrip(x, sr, codec_args, suffix):
    """
    经 ffmpeg 编解码往返。

    ⚠️ **必须用 ffmpeg 解码回 WAV 再读**，不能直接 sf.read 编码产物：
       soundfile(libsndfile) 不支持 M4A(AAC) 与 OPUS 容器（实测其支持列表为
       WAV/FLAC/OGG/MP3/AIFF/... 但没有 M4A/OPUS）。
       第一版直接 sf.read(out.m4a) 报 `Format not recognised`，
       导致 aac/opus 两类攻击**静默全失败**。
    """
    global FFMPEG
    if FFMPEG is None:
        FFMPEG = _ffmpeg()
    with tempfile.TemporaryDirectory() as td:
        a = os.path.join(td, "in.wav")
        enc = os.path.join(td, f"enc{suffix}")
        dec = os.path.join(td, "dec.wav")
        sf.write(a, x, sr)
        r = subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", a] + codec_args + [enc],
                           capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"编码失败: {r.stderr.decode()[:160]}")
        # 解码回 WAV（统一 16-bit PCM，避免不同容器差异）
        r = subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", enc,
                            "-ar", str(sr), "-ac", "1", dec], capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"解码失败: {r.stderr.decode()[:160]}")
        y, fs = sf.read(dec, dtype="float32", always_2d=True)
        y = y.mean(axis=1)
    if fs != sr:
        y = librosa.resample(y, orig_sr=fs, target_sr=sr)
    return y


# ---------------------------------------------------------------- 攻击实现
def att_mp3(x, sr, kbps=128):
    return _roundtrip(x, sr, ["-b:a", f"{kbps}k", "-ar", str(sr)], ".mp3")

def att_aac(x, sr, kbps=128):
    return _roundtrip(x, sr, ["-c:a", "aac", "-b:a", f"{kbps}k", "-ar", str(sr)], ".m4a")

def att_opus(x, sr, kbps=32):
    return _roundtrip(x, sr, ["-c:a", "libopus", "-b:a", f"{kbps}k", "-ar", str(sr)], ".opus")

def att_g711(x, sr):
    """
    电话窄带：G.711 mu-law 编解码，采样率 8 kHz 往返。
    ⚠️ _roundtrip 内部已把 ffmpeg 输出重采样回 sr，此处**不得**再重采样一次 ——
       多采一次会让时长翻倍（实测 9.81s -> 19.62s），在定长裁剪管线里静默错位。
    """
    return _roundtrip(x, sr, ["-c:a", "pcm_mulaw", "-ar", "8000"], ".wav")

def att_resample8k(x, sr):
    y = librosa.resample(x, orig_sr=sr, target_sr=8000)
    return librosa.resample(y, orig_sr=8000, target_sr=sr)

def att_awgn(x, sr, snr_db=10, seed=0):
    rng = np.random.RandomState(seed)
    p_sig = np.mean(x ** 2)
    p_noise = p_sig / (10 ** (snr_db / 10))
    return x + rng.randn(len(x)) * np.sqrt(p_noise)

def att_bandpass(x, sr, lo=300, hi=3400):
    """电话带通 300–3400 Hz（巴特沃斯带通，零相位 filtfilt 避免群延迟）"""
    from scipy.signal import butter, filtfilt
    nyq = sr / 2.0
    b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x).astype(np.float32)

def att_reverb(x, sr, rt60=0.4, seed=0):
    """合成 RIR 卷积：指数衰减噪声"""
    rng = np.random.RandomState(seed)
    n = int(rt60 * sr)
    h = rng.randn(n) * np.exp(-6.9 * np.arange(n) / n)      # -60 dB @ rt60
    h[0] = 1.0
    h = h / np.sqrt(np.sum(h ** 2))
    y = np.convolve(x, h)[:len(x)]
    return y

def att_speed(x, sr, factor=0.9):
    return librosa.effects.time_stretch(x, rate=factor)

def att_pitch(x, sr, semitones=0):
    """
    音高平移（相位声码器）。**保时长**，与 speed 互补——这是理论侧 §四.4 要的机理对照：
    变速保音高、变音高保时长 ⇒ 两者合起来把「时长类线索」与「频谱包络线索」分开。
    librosa.effects.pitch_shift 内部以 length=len(y) 收尾，输出样本数与输入**逐点相同**，
    故时长守卫按常规（非 speed）路径走即可，无需放宽容差。
    无随机数 ⇒ 天然无「共享 seed」类 bug。
    """
    return librosa.effects.pitch_shift(x.astype(np.float32), sr=sr, n_steps=semitones)


# 编解码类攻击：时长守卫需给固定的 priming/padding 容差（见守卫处注释）
CODEC_ATTACKS = {"mp3-128k", "mp3-64k", "aac-128k", "opus-32k", "g711-8k"}

ATTACKS = {
    "mp3-128k":          (att_mp3,      {"kbps": 128}, "mp3-128k"),
    "mp3-64k":           (att_mp3,      {"kbps": 64},  "mp3-64k"),
    "aac-128k":          (att_aac,      {"kbps": 128}, "aac-128k"),
    "opus-32k":          (att_opus,     {"kbps": 32},  "opus-32k"),
    "g711-8k":           (att_g711,     {},            "g711-8k"),
    "resample-8k":       (att_resample8k, {},          "8k-roundtrip"),
    "awgn-snr5":         (att_awgn,     {"snr_db": 5},  "snr5"),
    "awgn-snr10":        (att_awgn,     {"snr_db": 10}, "snr10"),
    "awgn-snr20":        (att_awgn,     {"snr_db": 20}, "snr20"),
    "bandpass-300-3400": (att_bandpass, {},            "bandpass300-3400"),
    "reverb-rt60-0.2":   (att_reverb,   {"rt60": 0.2}, "rt60-0.2"),
    "reverb-rt60-0.4":   (att_reverb,   {"rt60": 0.4}, "rt60-0.4"),
    "reverb-rt60-0.8":   (att_reverb,   {"rt60": 0.8}, "rt60-0.8"),
    "speed0.9":          (att_speed,    {"factor": 0.9}, "speed0.9"),
    "speed1.1":          (att_speed,    {"factor": 1.1}, "speed1.1"),
    # 音高族（理论侧 §四.4 补；机理与 speed 互补）
    "pitch-down2":       (att_pitch,    {"semitones": -2}, "pitch-down2"),
    "pitch-up2":         (att_pitch,    {"semitones": 2},  "pitch-up2"),
    "pitch-down4":       (att_pitch,    {"semitones": -4}, "pitch-down4"),
    "pitch-up4":         (att_pitch,    {"semitones": 4},  "pitch-up4"),
    # 相位声码器**恒等臂**（空对照，理论侧 #4661 §三）：n_steps=0 ⇒ time_stretch(rate=1)+identity resample
    "vocoder-id":        (att_pitch,    {"semitones": 0},  "vocoder-id"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=f"{WS}/data/manifests/fake_signalproc_manifest.csv")
    ap.add_argument("--limit", type=int, default=150, help="源伪造取多少条")
    ap.add_argument("--attacks", default="mp3-128k,awgn-snr10,reverb-rt60-0.4,speed0.9")
    ap.add_argument("--splits", default="test", help="对哪些 split 施加攻击")
    ap.add_argument("--out-manifest", default=f"{WS}/data/manifests/fake_attacks_manifest.csv")
    ap.add_argument("--num-threads", default=8, type=int, help="共享机线程上限")
    ap.add_argument("--label", default="fake", choices=["fake", "real"],
                    help="产出标签。P4 正确设计要对 real 也施加攻击（真实+后处理），故需可标为 real。默认 fake（对伪造施加攻击）")
    ap.add_argument("--source", default="attack", help="产出 source 字段")
    args = ap.parse_args()

    # 线程上限已在文件顶部（import 之前）设置 —— 此处设置无效，勿再加
    names = [a.strip() for a in args.attacks.split(",") if a.strip()]
    for n in names:
        assert n in ATTACKS, f"未知攻击 {n}; 可选 {list(ATTACKS)}"

    import pandas as pd
    df = pd.read_csv(args.src)
    want = [s.strip() for s in args.splits.split(",")]
    df = df[df.split.isin(want)].head(args.limit).reset_index(drop=True)
    print(f"[协议] 源 {len(df)} 条 (split={want}) × {len(names)} 种攻击 = {len(df)*len(names)} 条")

    rows, n_fail = [], 0
    for i, r in enumerate(df.itertuples(index=False)):
        try:
            x = _decode(os.path.join(WS, r.audio_path))
        except Exception as e:
            print(f"  ⚠️ 读失败 {r.utt_id}: {e}"); n_fail += 1; continue
        for nm in names:
            fn, kw, pstr = ATTACKS[nm]
            try:
                y = fn(x, SR, **kw)
            except Exception as e:
                print(f"  ⚠️ {nm} 失败 {r.utt_id}: {str(e)[:80]}"); n_fail += 1; continue
            if not np.isfinite(y).all() or len(y) < SR // 4:
                print(f"  ⚠️ {nm} 产出异常 {r.utt_id} (len={len(y)})"); n_fail += 1; continue
            # 时长守卫：除变速类外，攻击不应改变时长（g711 曾因重复重采样翻倍）
            # ⚠️ 编解码类要给**固定的** priming/padding 容差：
            #    AAC 会在末尾补 768 样本（实测互相关 lag=0，内容对齐，属编码器行为）。
            #    对短文件，768 样本会超过 2% 的相对容差而被误判为异常
            #    （实测因此丢过 11 条 aac）。
            if not nm.startswith("speed"):
                tol = max(0.02 * len(x), SR // 100)
                if nm in CODEC_ATTACKS:
                    tol = max(tol, 4096)          # ~256 ms，覆盖常见编码器 priming/padding
                if abs(len(y) - len(x)) > tol:
                    print(f"  ⚠️ {nm} {r.utt_id} 时长异常 {len(x)/SR:.2f}s -> {len(y)/SR:.2f}s，已跳过")
                    n_fail += 1; continue
            y = match_rms(y, x)                      # ← 响度对齐，同伪造生成红线
            y = np.clip(y, -1, 1).astype(np.float32)
            o = f"{OUT}/{nm}/{r.utt_id}.wav"
            os.makedirs(os.path.dirname(o), exist_ok=True)
            sf.write(o, y, SR)
            rows.append({
                "audio_path": os.path.relpath(o, WS), "label": args.label,
                "source": args.source, "clone_model": "none", "attack_type": nm,
                "attack_params": pstr, "text": r.text, "speaker_id": r.speaker_id,
                "duration": round(len(y) / SR, 3), "sample_rate": SR,
                "split": r.split, "utt_id": f"{r.utt_id}_{nm}",
            })
        if (i + 1) % 25 == 0:
            print(f"  ... {i+1}/{len(df)}")

    if not rows:
        sys.exit("❌ 无产出")
    cols = ["audio_path", "label", "source", "clone_model", "attack_type", "attack_params",
            "text", "speaker_id", "duration", "sample_rate", "split", "utt_id"]
    with open(args.out_manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    import collections
    c = collections.Counter(x["attack_type"] for x in rows)
    print(f"\n生成 {len(rows)} 条攻击变体 (失败 {n_fail})")
    dead = []
    for n in names:
        ok_n = c[n]
        flag = "✅" if ok_n > 0 else "❌ 整类失败"
        if ok_n == 0:
            dead.append(n)
        print(f"  {n:<22} {ok_n:>5}  {flag}")
    # 整类静默失败守卫（教训：aac/opus 曾因 soundfile 不支持容器而全灭）
    if dead:
        print(f"\n❌ 以下攻击类型**一条都没生成**，属于整类失败，不得当作『该攻击无效』：")
        print(f"   {dead}")
        print(f"   先修生成链再重跑。")
        return 2
    print(f"manifest: {args.out_manifest}")
    print("MAKE_ATTACKS_DONE")
    return 0


if __name__ == "__main__":
    main()
