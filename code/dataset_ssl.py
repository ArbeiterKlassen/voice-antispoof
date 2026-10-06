#!/usr/bin/env python3
"""SSL 特征数据集：读缓存的 WavLM npy（不碰音频、不做 pad/crop —— 前端已固化）"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL_MAP = {"real": 1, "fake": 0}          # 1=bonafide（与 metrics.py 一致）


class SSLFeatureDataset(Dataset):
    def __init__(self, df, cache_dir, root=WS):
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.root = root
        # 只保留缓存存在的条目（缺失的显式报出来，不静默丢）
        have, miss = [], []
        for i, r in self.df.iterrows():
            p = os.path.join(cache_dir, f"{r['utt_id']}.npy")
            (have if os.path.exists(p) else miss).append(i)
        if miss:
            print(f"  ⚠️ {len(miss)} 条无缓存，已排除；例: {list(self.df.utt_id.iloc[miss[:3]])}")
        self.df = self.df.iloc[have].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        h = np.load(os.path.join(self.cache_dir, f"{r['utt_id']}.npy")).astype(np.float32)
        return torch.from_numpy(h), int(r["y"])
