# [VOICE] 实验任务 01 阶段性回传 —— 环境与辨伪基线

实验侧 AI · 2026-09-18 · 工作区 `/data1/cuitianxu/voice_ws`
对应任务书：《任务_实验01_环境与基线.md》

---

## 0. 一句话状态

辨伪基线**已跑通并出数**（P1 + 留一泛化 P2）；克隆基线（CosyVoice2）**环境仍在安装**，未完成。
本报告先把可复算的部分交出去，克隆部分按 §7 的 ETA 补。

---

## 1. 交付物清单（对应任务书 §5）

| # | 任务书要求 | 状态 | 本次上传文件名 |
|---|---|---|---|
| 1 | `env_report.txt` | ✅ | `env_report.txt` |
| 2 | `data_manifest.csv` | ✅ | `real_manifest.csv` / `all_manifest.csv` |
| 3 | `clone_samples/` ≥3 Wav | ⏳ **未完成** | 见 §7 |
| 4 | `detector_metrics.json` | ✅ | `detector_metrics.json` |
| 5 | `train.log` | ✅ | `train_log_signalproc.txt` |

---

## 2. 三条硬约束（已在 #2380 报过，此处存档）

1. **`github.com` / `huggingface.co` 直连不通。**
   GitHub 走 `https://gh-proxy.com/<url>`（实测可 clone）；HF 走 `export HF_ENDPOINT=https://hf-mirror.com`。
   另：**openslr.org 实测仅 15 KB/s**（338 MB 需 6 小时），已改走 hf-mirror（241 KB/s，2 分钟）。
2. **现有 conda env 无一具备音频栈**（torchaudio/soundfile/librosa 全缺），系统无 ffmpeg/sox。
3. **只有 GPU2 可用**（8 卡中 7 卡被他人占用）。检测器 AASIST 仅 297K 参数，单卡足够。

### 任务书事实更正
> §1 写「单卡显存是否约 **24 GB**」→ **实测 49140 MiB ≈ 48 GB**。

---

## 3. 环境实测（`env_report.txt` 摘要）

- 8×RTX 4090，驱动 580.105.08，CUDA 13.0，8 卡均可见
- 项目选用 env `ml2026_exp`：Python 3.10.20 / **torch 2.11.0+cu130** / torchaudio 2.11.0 / soundfile 0.14.0 / librosa 0.11.0
- 磁盘：`/data1` 15T 用 66%，余 4.8T
- 目标卡 GPU2：训练峰值约 11.2 GB

---

## 4. 数据构造

| 项 | 数量 | 说明 |
|---|---|---|
| 真实语音 | **2703 条** | LibriSpeech dev-clean，40 说话人，总 5.39 h，中位 5.92 s |
| 信号处理伪造 | **2733 条** | 500 源 × 3 类（train/dev） + 411 源 × 3 类（test） |
| 合计 | **5436 条** | |

**划分（按说话人，非按条随机）**：

| split | 说话人 | real | fake |
|---|---|---|---|
| train | 28 | 1879 | 1254 |
| dev | 6 | 413 | 246 |
| test | 6 | 411 | 1233 |

✅ **speaker-disjoint 已用断言校验**：同一说话人不跨 split。
（按条随机划分会让同一说话人同时出现在 train 和 test，检测器可能靠"认说话人"而非"认伪造"过关。）

### 伪造类别说明
本轮伪造是**信号处理类**（对应方案 §2.2 第 3 类），非神经克隆：
- `melvoc`：mel 谱 → Griffin-Lim 重建（相位被破坏，声码器式伪影）
- `pitch`：音高搬移
- `formant`：频谱包络搬移

**⚠️ 生成本轮伪造时发现并修正了一个会毁掉实验的混淆**：
初版 `melvoc` 的 RMS 是真实语音的 **6.3 倍**、峰值削波到 1.000。
若不对齐，检测器只要学「**响 = 伪造**」就能刷满，结论全部失效。
已统一把伪造的响度对齐到源真实语音的 RMS（复检三类均精确 1.00x、零削波）。

---

## 5. 辨伪基线结果

### 5.1 方法与口径

- **模型**：`clovaai/aasist` 官方架构（297,866 参数）+ 官方预训练权重
  （md5 `40ebc8a542c4effce39770c22396ff84`，实测 `missing=0 unexpected=0` 完整载入）
- **训练**：15 epoch，bs=16，Adam lr=1e-4，cosine，加权 CE，16 kHz / 4.04 s 定长
- **分数方向**：`logits[1] = bonafide`（已对 `data_utils.py:22` + `main.py:307` 核实）
- **EER 定义**：FAR=FRR 交点；**自实现**，理由见 5.3
- **未报 min-tDCF**：需 ASVspoof2019 官方 ASV 打分文件，自建数据没有，**不编造**

### 5.2 结果

**P1 in-domain（test 1644 条：411 real + 1233 fake）**

| 组 | n_fake | EER | AUC | acc |
|---|---|---|---|---|
| 全部 | 1233 | **0.000%** | 1.0000 | 1.0000 |
| melvoc | 411 | 0.000% | 1.0000 | 1.0000 |
| pitch | 411 | 0.000% | 1.0000 | 1.0000 |
| formant | 411 | 0.000% | 1.0000 | 1.0000 |

**P2 留一攻击类型（训练时排除 formant，测试看 formant）**

| 组 | 训练时见过 | EER | AUC | acc |
|---|---|---|---|---|
| melvoc | ✅ | 0.243% | 1.0000 | 0.9976 |
| pitch | ✅ | 0.243% | 1.0000 | 0.9976 |
| **formant** | ❌ **从未见过** | **0.730%** | 0.9995 | 0.9927 |

### 5.3 官方 `evaluation.py` 为何不用（已核实，两处必崩）

1. `evaluation.py:34` **无条件** `np.genfromtxt(asv_score_file)` —— 该文件只随 ASVspoof2019
   官方数据集发布，自建数据必 `FileNotFoundError`
2. `evaluation.py:37,44` `astype(np.float)` —— **`np.float` 在 NumPy ≥1.24 已被移除**，
   本机 numpy 2.2.5 直接 `AttributeError`

→ 因此 EER/AUC **自实现**（`code/metrics.py`），并**先过 7 项合成已知答案自检**：
完全可分→EER=0；方向反转→EER=1；随机分数→EER≈0.5；偏移增大→EER 单调降。
数值也对得上理论：偏移 0.5 时 AUC=0.6402 vs 理论 Φ(0.5/√2)=0.638。

### 5.4 排除平凡混淆（关键对照）

P1 的 0.000% 太完美，**我没有直接当卖点**，先做了平凡特征对照——用**单个平凡特征**能否自己分开 real/fake：

| 特征 | real 均值 | fake 均值 | 单特征 AUC |
|---|---|---|---|
| **RMS（响度）** | 0.05603 | 0.05517 | **0.5130** |
| 时长 | 7.048 | 7.047 | 0.5004 |
| 峰值 | 0.540 | 0.633 | 0.3538 |
| 谱质心 | 1911.5 | 1830.6 | 0.5269 |
| 谱平坦度 | 0.0411 | 0.0298 | 0.6781 |
| 谱带宽 | 1558.1 | 1458.0 | 0.5784 |

→ **无任何单一平凡特征具判别力**（RMS 的 0.5130 等于随机，证明响度对齐确实生效）。
检测器的高分不是靠平凡特征刷出来的。

---

## 6. 本报告的诚实边界（**请连同数字一起读**）

1. **P1 的 0.000% 不应作为"检测器很强"的证据。**
   本轮伪造是**信号处理类**，伪影粗糙；AASIST 官方权重本就在 ASVspoof2019 上预训练过，
   认出它们不难。**AUC 恰好 1.0000 通常意味着任务简单，而非模型强。**

2. **P2 的 0.730% 只说明"跨信号处理子类可泛化"，不等于"跨到神经克隆可泛化"。**
   formant 与 melvoc/pitch 同属信号处理家族，共享频谱/相位伪影特征。
   真正的外推缺口在**神经克隆**（CosyVoice2/F5-TTS），那是量级不同的难度。

3. 因此：**P1 + P2 必须成对读**；而**决定性数字要等 CosyVoice2 那批伪造**。
   这也正是 #2380 提的预注册协议（P1 in-domain 与 P2 cross-model 成对汇报）的用意。

---

## 7. 未完成项与 ETA

| 项 | 状态 | 阻塞原因 |
|---|---|---|
| CosyVoice2 环境 | 🔄 安装中 | torch 2.3.1 钉版，其依赖链需拉 cu12 系列 nvidia 包（cudnn 731 MB 等），带宽受限 |
| CosyVoice2 权重 | ⏳ 未开始 | 4.5 GB（hf-mirror 实测约 1.8 MB/s，可省 782 MB） |
| `clone_samples/` ≥3 Wav | ⏳ 未开始 | 依赖上两项 |

**ETA：克隆部分 24 h 内**（受带宽制约，非算法问题）。届时补交 `clone_samples/` 与
**基于神经克隆的 P2 数字**——那才是本项目真正的主结论。

---

## 8. 复算方式

```bash
# 环境报告
bash voice_ws/scripts/make_env_report.sh

# 指标自检（应先通过再信数字）
python voice_ws/code/metrics.py --selftest

# P1 复算
python voice_ws/code/eval_detector.py \
  --ckpt voice_ws/results/det_signalproc/best.pth --split test --by attack_type

# P2 留一复算
python voice_ws/code/eval_detector.py \
  --ckpt voice_ws/results/det_holdout_formant/best.pth --split test --by attack_type
```

数据清单字段：`audio_path,label,source,clone_model,attack_type,text,speaker_id,duration,sample_rate,split,utt_id`
（任务书要求的 8 列全含，另加 `sample_rate/split/utt_id` 便于复算）
