#!/usr/bin/env python3
"""
辨伪数据集 —— 读我们的 manifest CSV，产出 16 kHz 定长波形

⚠️ 方向约定（与 metrics.py 一致，已对 clovaai/aasist 源码核实）：
    label = 1 → real (bonafide)
    label = 0 → fake (spoof)
    模型 logits 的 **索引 1 = bonafide**（data_utils.py:22 + main.py:307）

⚠️ 定长：AASIST 配置 nb_samp=64600（4.04 s @ 16 kHz）。
    训练随机裁（pad_random），评估固定裁（pad 前部）——与官方 data_utils 行为一致。

用法:
    ds = ManifestDataset(rows, split="train")
    wav, label = ds[0]      # wav: (64600,) float32, label: int
"""
import os
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_SAMP = 64600          # AASIST 标准输入长度
SR = 16000

LABEL_MAP = {"real": 1, "fake": 0}   # 1=bonafide, 0=spoof


def load_manifest(path):
    df = pd.read_csv(path)
    if "label" in df.columns:
        df["y"] = df["label"].map(LABEL_MAP)
        assert df["y"].notna().all(), f"未知 label 值: {df['label'].unique()}"
        df["y"] = df["y"].astype(int)
    return df


def load_wav_16k(path, sr=SR):
    """读音频 -> 单声道 16 kHz float32"""
    x, fs = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)                       # 多声道取均值
    if fs != sr:
        # 简单线性重采样（librosa 更准，但这里避免额外依赖差异）
        import librosa
        x = librosa.resample(x, orig_sr=fs, target_sr=sr)
    return np.ascontiguousarray(x, dtype=np.float32)


def pad(x, max_len=NB_SAMP):
    """定长：短则右侧重复填充，长则取前 max_len（评估用，确定性）"""
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, num_repeats)[:max_len]


def pad_random(x, max_len=NB_SAMP):
    """随机裁（训练用，与官方 data_utils.pad_random 一致）"""
    x_len = x.shape[0]
    if x_len >= max_len:
        stt = np.random.randint(0, x_len - max_len + 1)
        return x[stt: stt + max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, num_repeats)[:max_len]


class ManifestDataset(Dataset):
    def __init__(self, df, train=False, root=WS):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.root = root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        p = r["audio_path"]
        if not os.path.isabs(p):
            p = os.path.join(self.root, p)
        x = load_wav_16k(p)
        x = pad_random(x) if self.train else pad(x)
        return torch.from_numpy(x), int(r["y"])
