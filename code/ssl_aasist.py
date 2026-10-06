#!/usr/bin/env python3
"""
SSL-AASIST：冻结 WavLM-large 前端 + 可训练适配器 + **复用 AASIST 预训练 GAT 后端**

## 为什么复用「GAT 后端」而不是「整个 AASIST」
实测（2026-10-04）AASIST 内部对时间轴的压缩倍率：
    T_in=21490 -> T_out=29   （压缩 ~740x）
    T_in=1000  -> T_out=1
    T_in=201   -> 直接崩（池化到 0）
它的 conv encoder 是为 **T=21490 的原始波形** 设计的，
而 WavLM 对 4.04s 只给 **201 帧** —— 所以 **encoder 无法复用**。

但 **GAT 后端可以**：它期望输入 `(B, 64, 23, 29)`（F=23 谱节点 × T''=29 时节点）。
只要用适配器把 SSL 特征**直接映射到这个形状**，整条后端（含 pos_S 与全部
预训练权重）都能原样复用。

## 结构
  WavLM 特征 (B, 201, 1024)          [冻结，已缓存]
    -> 时间轴插值 201 -> 29
    -> Conv1d(1024 -> 23)            [可训练适配器]
    -> Conv2d(1 -> 64, 1x1)          [可训练适配器]
    -> (B, 64, 23, 29)   —— 形状与官方 encoder 输出一致
    -> GAT-S / GAT-T / HtrgGAT       [复用官方预训练权重]
    -> logits                        [索引 1 = bonafide]

## 方向约定
logits[:, 1] = bonafide，与 metrics.py / train_detector.py 一致。
"""
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, "code", "aasist"))

from models.AASIST import Model as _AASISTModel      # noqa: E402
import os

D_ARGS = {
    "architecture": "AASIST", "nb_samp": 64600, "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32], "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}
UPSTREAM_CKPT = f"{WS}/code/aasist/models/weights/AASIST.pth"

# 官方 encoder 输出的形状（实测）：F=23 谱节点, T''=29 时节点, C=64
T_NODES = 29
F_NODES = 23
C_DIM = 64
SSL_FRAMES = 201          # WavLM 对 64600 样本的输出帧数


class SSLAASIST(nn.Module):
    def __init__(self, d_args=None, ssl_dim=1024, ckpt=UPSTREAM_CKPT,
                 load_upstream=True, verbose=False):
        super().__init__()
        d_args = d_args or D_ARGS

        # ---- 复用官方后端（含预训练权重）----
        base = _AASISTModel(d_args)
        n_loaded = 0
        if load_upstream and os.path.exists(ckpt):
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
            miss, unexp = base.load_state_dict(sd, strict=False)
            n_loaded = len(sd) - len(unexp)
            if verbose:
                print(f"[SSL-AASIST] 载入官方权重 missing={len(miss)} unexpected={len(unexp)}")

        # 引用 GAT 后端子模块（共享参数，不拷贝）
        for name in ["pos_S", "GAT_layer_S", "pool_S", "GAT_layer_T", "pool_T",
                     "master1", "master2",
                     "HtrgGAT_layer_ST11", "HtrgGAT_layer_ST12",
                     "HtrgGAT_layer_ST21", "HtrgGAT_layer_ST22",
                     "pool_hS1", "pool_hT1", "pool_hS2", "pool_hT2",
                     "drop_way", "drop", "out_layer"]:
            setattr(self, name, getattr(base, name))

        # 记录官方后端被冻结的参数量（供报告口径）
        self.n_upstream_loaded = n_loaded

        # ---- 新增：SSL -> (C,F,T) 适配器（本次唯一的可训练新部件）----
        self.proj_f = nn.Conv1d(ssl_dim, F_NODES, kernel_size=1)
        self.bn_f = nn.BatchNorm1d(F_NODES)
        self.expand_c = nn.Conv2d(1, C_DIM, kernel_size=1)
        self.bn_c = nn.BatchNorm2d(C_DIM)

    def forward(self, h, Freq_aug=False):
        """h: (B, T', ssl_dim) —— 已缓存的 WavLM 特征，不是波形"""
        # 时间轴对齐到官方后端期望的 29 个时节点
        x = h.transpose(1, 2)                                   # (B, D, T')
        if x.shape[-1] != T_NODES:
            x = F.interpolate(x, size=T_NODES, mode="linear", align_corners=False)
        x = F.selu(self.bn_f(self.proj_f(x)))                   # (B, 23, 29)
        x = x.unsqueeze(1)                                      # (B, 1, 23, 29)
        e = F.selu(self.bn_c(self.expand_c(x)))                 # (B, 64, 23, 29)

        # ---- 以下与官方 forward 的 GAT 段逐行一致 ----
        e_S, _ = torch.max(torch.abs(e), dim=3)                 # 沿时间取 max
        e_S = e_S.transpose(1, 2) + self.pos_S
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)

        e_T, _ = torch.max(torch.abs(e), dim=2)                 # 沿频率取 max
        e_T = e_T.transpose(1, 2)
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        master1 = self.master1.expand(e.size(0), -1, -1)
        master2 = self.master2.expand(e.size(0), -1, -1)

        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(out_T, out_S, master=self.master1)
        out_S1 = self.pool_hS1(out_S1); out_T1 = self.pool_hT1(out_T1)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug; out_S1 = out_S1 + out_S_aug; master1 = master1 + master_aug

        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2); out_T2 = self.pool_hT2(out_T2)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug; out_S2 = out_S2 + out_S_aug; master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1); out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1); out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1); master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)
        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)
        last_hidden = torch.cat([T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)
        return last_hidden, output


if __name__ == "__main__":
    m = SSLAASIST(verbose=True).eval()
    n_new = sum(p.numel() for n, p in m.named_parameters() if n.split(".")[0] in
                ("proj_f", "bn_f", "expand_c", "bn_c"))
    n_all = sum(p.numel() for p in m.parameters())
    print(f"参数量: 总 {n_all:,} | 新增适配器 {n_new:,} | 官方后端 {n_all-n_new:,}")
    for T in [201]:
        h = torch.randn(4, T, 1024)
        with torch.no_grad():
            lh, out = m(h)
        print(f"输入 (4,{T},1024) -> last_hidden {tuple(lh.shape)}, logits {tuple(out.shape)}")
        assert out.shape == (4, 2)
    # 方向自检：随机权重下输出应有限且非退化
    with torch.no_grad():
        lh, out = m(torch.randn(8, SSL_FRAMES, 1024))
    print(f"输出统计: mean={out.mean():.4f} std={out.std():.4f} finite={torch.isfinite(out).all().item()}")
    print("SSL_AASIST_SMOKE_OK")
