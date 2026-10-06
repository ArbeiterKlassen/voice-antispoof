#!/usr/bin/env python3
"""
第三类伪造：信号处理伪装的伪造语音
（对应 voice-theory《实验方案》§2.2 的第 3 类：传统 TTS 或信号处理伪装的伪造语音）

这不是占位数据 —— 它是方案里本来就要的一类伪造，与 CosyVoice2/F5-TTS 的神经克隆互补：
  神经克隆的伪影在声学细节；信号处理伪装的伪影在相位/频谱包络。
  检测器若只学会认神经克隆，这一类就会暴露它。

三类处理（各生成一批，attack_type 分别记录）：
  1. melvoc  : mel 谱 -> Griffin-Lim 重建（相位被破坏，经典声码器伪影）
  2. pitch   : 音高搬移（共振峰不动 → 音色不自然）
  3. formant : 频谱包络搬移（模拟"假声/换音色"伪装）

用法:
  python make_signalproc_fake.py --limit 400 --kinds melvoc,pitch
"""
import argparse, csv, os, sys
import numpy as np
import soundfile as sf
import librosa
import os

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 16000
OUT = f"{WS}/data/fake_signalproc"


def fake_melvoc(x, sr=SR, rng=None):
    """mel 谱 -> Griffin-Lim 重建：相位信息被丢弃，产生声码器式伪影"""
    n_fft, hop = 512, 128
    mel = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=n_fft, hop_length=hop,
                                         n_mels=80, fmin=0, fmax=sr // 2)
    y = librosa.feature.inverse.mel_to_audio(mel, sr=sr, n_fft=n_fft, hop_length=hop,
                                             n_iter=32, power=1.0)
    return y


def fake_pitch(x, sr=SR, n_steps=3.0):
    """音高搬移（共振峰随包络一起搬，听感不自然）"""
    return librosa.effects.pitch_shift(y=x, sr=sr, n_steps=n_steps)


def fake_formant(x, sr=SR):
    """频谱包络搬移：STFT 幅度沿频率轴拉伸 + 相位保留"""
    n_fft, hop = 1024, 256
    D = librosa.stft(x, n_fft=n_fft, hop_length=hop)
    mag, phase = np.abs(D), np.angle(D)
    n_bins = mag.shape[0]
    warped = np.empty_like(mag)
    src = np.linspace(0, n_bins - 1, n_bins) * 0.88   # 包络压缩 -> 音色变化
    idx = np.clip(src, 0, n_bins - 1)
    lo = np.floor(idx).astype(int); hi = np.clip(lo + 1, 0, n_bins - 1)
    w = (idx - lo)[:, None]
    warped = mag[lo] * (1 - w) + mag[hi] * w
    y = librosa.istft(warped * np.exp(1j * phase), hop_length=hop, length=len(x))
    return y


def match_rms(y, ref, peak_limit=0.99):
    """
    把 y 的响度对齐到参考音频 ref 的 RMS。

    ⚠️ 为什么必须做（实测教训）：
        melvoc 重建后 RMS 0.303 / 峰值顶到 1.000，而源真实语音 RMS 仅 0.048。
        → 不作对齐的话，检测器只要学"响度大 = 伪造"就能刷到很高分，
          这个**响度混淆**会让全部结论失效（学到的是响度不是合成伪影）。
        CosyVoice2 等神经克隆同样要做这一步。
    """
    ry = np.sqrt(np.mean(y ** 2)) + 1e-12
    rr = np.sqrt(np.mean(ref ** 2)) + 1e-12
    y = y * (rr / ry)
    pk = np.abs(y).max()
    if pk > peak_limit:                 # 防削波：整体降幅
        y = y * (peak_limit / pk)
    return y


KINDS = {"melvoc": fake_melvoc, "pitch": fake_pitch, "formant": fake_formant}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=f"{WS}/data/manifests/real_manifest.csv")
    ap.add_argument("--limit", type=int, default=400, help="总共处理多少条真实语音")
    ap.add_argument("--kinds", default="melvoc,pitch")
    ap.add_argument("--splits", default="train,dev",
                    help="只处理这些 split（逗号分隔）。test 也要有 fake 才能做 P1")
    ap.add_argument("--out-manifest", default=f"{WS}/data/manifests/fake_signalproc_manifest.csv")
    args = ap.parse_args()

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    for k in kinds:
        assert k in KINDS, f"未知 kind {k}, 可选 {list(KINDS)}"

    import pandas as pd
    df = pd.read_csv(args.manifest)
    want = [s.strip() for s in args.splits.split(",") if s.strip()]
    df = df[df["split"].isin(want)].head(args.limit).reset_index(drop=True)
    print(f"[协议] 处理 split={want}")
    print(f"[协议] 源真实语音 {len(df)} 条, 生成 {kinds}")

    os.makedirs(OUT, exist_ok=True)
    rows = []
    for i, r in enumerate(df.itertuples(index=False)):
        p = os.path.join(WS, r.audio_path)
        try:
            x, fs = sf.read(p, dtype="float32", always_2d=True)
            x = x.mean(axis=1)
            if fs != SR:
                x = librosa.resample(x, orig_sr=fs, target_sr=SR)
        except Exception as e:
            print(f"  ⚠️ 读失败 {p}: {e}"); continue
        for k in kinds:
            try:
                y = KINDS[k](x)
            except Exception as e:
                print(f"  ⚠️ {k} 处理失败 {r.utt_id}: {e}"); continue
            if not np.isfinite(y).all():
                print(f"  ⚠️ {k} 产出含 NaN/Inf: {r.utt_id}"); continue
            y = match_rms(y, x)                      # 响度对齐，消除响度混淆
            n_clip = int((np.abs(y) >= 0.999).sum())
            if n_clip > 0:
                print(f"  ⚠️ {k} {r.utt_id} 仍有 {n_clip} 点削波")
            y = np.clip(y, -1.0, 1.0).astype(np.float32)
            o = f"{OUT}/{k}/{r.utt_id}.wav"
            os.makedirs(os.path.dirname(o), exist_ok=True)
            sf.write(o, y, SR)
            rows.append({
                "audio_path": os.path.relpath(o, WS), "label": "fake",
                "source": "signalproc", "clone_model": "none", "attack_type": k,
                "text": r.text, "speaker_id": r.speaker_id,
                "duration": round(len(y) / SR, 3), "sample_rate": SR,
                "split": r.split, "utt_id": f"{r.utt_id}_{k}",
            })
        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{len(df)}")

    cols = ["audio_path", "label", "source", "clone_model", "attack_type",
            "text", "speaker_id", "duration", "sample_rate", "split", "utt_id"]
    with open(args.out_manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    import collections
    c = collections.Counter((x["attack_type"], x["split"]) for x in rows)
    print(f"\n生成 {len(rows)} 条信号处理伪造")
    for k in kinds:
        print(f"  {k}: train {c[(k,'train')]} / dev {c[(k,'dev')]}")
    print(f"manifest: {args.out_manifest}")
    print("MAKE_SIGNALPROC_DONE")


if __name__ == "__main__":
    main()
