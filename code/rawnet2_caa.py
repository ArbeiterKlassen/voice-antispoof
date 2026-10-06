#!/usr/bin/env python3
"""
CAA RawNet2（第三方权重 caa-speech-detection-asvspoof2019/rawnet2）的**重建实现**。

## 为什么要重建
该仓库只发了权重、没有源码（`src.models.rawnet2.model.RawNet2Model` 未公开）。
本实现按 ① state_dict 的逐键形状 ② 模型卡描述（Tak et al. ICASSP 2021 结构 + FMS）
③ 从存储缓冲值反推出的 sinc 参数化 重建。**已知答案**：
- 缓冲值反推：low_hz_ = 0..7937.5Hz 线性等距(Δ62.5)、band_hz_ ≡ 62.5、n_axis = arange(64)·π/sr、
  window = hamming(129)[:64]（前半）
- `front_bn` 的 running stats 是**训练时的真实统计**→ 可用它判定 sinc 组装变体对不对（见 scripts 校验）
- 最终锚：模型卡自报 **eval EER 15.09%**（全量 71237）

## 键名对齐（必须与 best.pt 完全一致）
sinc.{low_hz_,band_hz_,n_axis,window} / front_bn / blocks.N.{bn1,conv1,bn2,conv2,[shortcut],fms.fc}
/ pre_gru_bn / gru / fc / classifier / class_weights

用法（评估）: score_group.py --model-config <json 含 model_config> （架构 "RawNet2CAA"）
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 16000
KERNEL = 129          # 模型卡: sinc_filter_length=129
N_FILT = 128


class SincConv(nn.Module):
    """sinc 滤波器组（固定，不可训练）。半核 64 点 + 中心 + 镜像 = 129。

    组装变体（variant）：
      "A": 位置配对 —— 半核第 j 点配 n_axis[j]（t=j），组装 [flip(half), center, half]
      "E": 反序配对 —— 半核第 j 点配 n_axis[j]，但视为偏移 (K2-j)（外缘点配小窗），
           组装 [half, center, flip(half)]
    中心值 = 2*(f_high-f_low)/sr（理想带通 t=0 值）。
    """
    def __init__(self, out_channels=N_FILT, kernel_size=KERNEL, sample_rate=SR, variant="A"):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel 必须为奇数"
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.variant = variant
        self.unit_gain = False     # True ⇒ 除以 2·bandwidth（官方 SincConv_fast 的归一）
        K2 = (kernel_size - 1) // 2                # 64
        # 参数（不可训练）——名字必须与权重键一致
        self.low_hz_ = nn.Parameter(torch.zeros(out_channels, 1), requires_grad=False)
        self.band_hz_ = nn.Parameter(torch.zeros(out_channels, 1), requires_grad=False)
        self.register_buffer("n_axis", (torch.arange(0, K2, dtype=torch.float32) * np.pi / sample_rate).view(1, K2))
        self.register_buffer("window", torch.hamming_window(kernel_size, periodic=False)[:K2].clone())

    def _filters(self):
        low = self.low_hz_.abs()                    # (C,1) Hz
        high = low + self.band_hz_.abs()            # (C,1) Hz
        n = self.n_axis                             # (1,K2) 弧度/Hz 轴: n_j = j*pi/sr
        # 理想带通半核: [sin(f_h n) - sin(f_l n)] / n ；n=0 处取极限 2(f_h-f_l)
        # ★ phase 不乘 2：由 front_bn 存储统计反推（log-var 相关 0.34→0.89），
        #   等价于陷波间距为半采样（n_j = j*pi/sr ⇒ 相位 = pi*f*j/sr）
        hh = torch.sin(high * n) - torch.sin(low * n)           # (C,K2)
        denom = n.clone(); denom[0, 0] = 1.0        # 占位，中心单算
        half = hh / denom                           # (C,K2)
        half[:, 0] = 2.0 * (high - low).squeeze(1)  # t=0 极限（未除 sr；同族实现的未归一约定）
        half = half * self.window                   # (C,K2) 窗
        center = (2.0 * (high - low)).view(-1, 1)   # (C,1)
        if self.unit_gain:
            band = high - low
            if (band <= 0).any():
                raise RuntimeError("band=0（未载权重？）——unit_gain 会除零")
            half = half / (2.0 * band)
            center = center / (2.0 * band)
        if self.variant == "A":
            kern = torch.cat([torch.flip(half, dims=[1]), center, half], dim=1)
        elif self.variant == "E":
            kern = torch.cat([half, center, torch.flip(half, dims=[1])], dim=1)
        else:
            raise ValueError(self.variant)
        return kern.view(self.out_channels, 1, self.kernel_size)

    def forward(self, x):                            # x: (B, T) or (B,1,T)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        pad = (self.kernel_size - 1) // 2
        return F.conv1d(x, self._filters(), stride=1, padding=pad, bias=None)


class FMS(nn.Module):
    """filter-wise feature map scaling（Tak et al.）: y = x*s + s, s = sigmoid(fc(gap(x)))"""
    def __init__(self, dim):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):                            # x: (B, C, T)
        y = self.pool(x).squeeze(-1)                 # (B, C)
        y = torch.sigmoid(self.fc(y)).unsqueeze(-1)  # (B, C, 1)
        return x * y + y


class ResBlock(nn.Module):
    """pre-activation: bn1→LReLU(0.3)→conv1→bn2→LReLU(0.3)→conv2 →(+shortcut)→fms→maxpool"""
    def __init__(self, in_ch, out_ch, use_fms=True):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.fms = FMS(out_ch) if use_fms else None
        self.pool = nn.MaxPool1d(3)

    def forward(self, x):
        h = F.leaky_relu(self.bn1(x), 0.3)
        h = self.conv1(h)
        h = F.leaky_relu(self.bn2(h), 0.3)
        h = self.conv2(h)
        s = self.shortcut(x) if self.shortcut is not None else x
        h = h + s
        if self.fms is not None:
            h = self.fms(h)
        return self.pool(h)


class Model(nn.Module):
    """与训练管线接口一致：forward(x, Freq_aug=None) -> (last_hidden, output)"""
    def __init__(self, d_args):
        super().__init__()
        cfg = dict(d_args)
        nf = cfg.get("sinc_filters", N_FILT)
        k = cfg.get("sinc_filter_length", KERNEL)
        self.target_samples = cfg.get("target_samples", 64000)
        self.input_scale = cfg.get("input_scale", 1.0)   # 前端输入缩放（匹配其训练时的绝对尺度）
        variant = cfg.get("sinc_variant", "A")
        c1 = cfg.get("first_block_channels", 128)
        c2 = cfg.get("second_block_channels", 512)
        n1 = cfg.get("num_first_blocks", 2)
        n2 = cfg.get("num_second_blocks", 4)
        gh = cfg.get("gru_hidden", 1024)
        ed = cfg.get("embedding_dim", 1024)
        self.sinc = SincConv(nf, k, SR, variant=variant)
        self.front_bn = nn.BatchNorm1d(nf)
        blocks = []
        in_ch = nf
        for i in range(n1 + n2):
            out_ch = c1 if i < n1 else c2
            blocks.append(ResBlock(in_ch, out_ch))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.pre_gru_bn = nn.BatchNorm1d(in_ch)
        self.gru = nn.GRU(input_size=in_ch, hidden_size=gh, num_layers=1, batch_first=True)
        self.fc = nn.Linear(gh, ed)
        self.classifier = nn.Linear(ed, 2)
        self.register_buffer("class_weights", torch.ones(2))

    def forward(self, x, Freq_aug=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.shape[-1] > self.target_samples:
            x = x[..., :self.target_samples]
        elif x.shape[-1] < self.target_samples:
            x = F.pad(x, (0, self.target_samples - x.shape[-1]))
        h = self.sinc(x * self.input_scale)
        h = self.front_bn(h)
        h = F.leaky_relu(h, 0.3)
        for b in self.blocks:
            h = b(h)
        h = self.pre_gru_bn(h)
        h = F.leaky_relu(h, 0.3)
        h = h.permute(0, 2, 1)                # (B, T, C)
        self.gru.flatten_parameters()
        out, _ = self.gru(h)
        last_hidden = self.fc(out[:, -1, :])
        return last_hidden, self.classifier(last_hidden)
