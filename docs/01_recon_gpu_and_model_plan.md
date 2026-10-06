# 侦察报告 + 训练模型构建方案

实验侧 AI（occworld-exp）· 2026-09-18 · 工作区 `/data1/cuitianxu/voice_ws`

---

## 一、工作区（已建立）

```
/data1/cuitianxu/voice_ws/
├── code/  data/{real,fake_*,attacks,manifests}/  models/
├── results/  logs/  scripts/  docs/  README.md
```
命名沿用本机惯例（`<name>_ws`）。**未沿用 voice-theory 的 `../数据 / ../模型 / ../结果` 中文相对路径**——
中文路径在多工具链下易出问题，改为 ASCII，映射关系记在 README。

---

## 二、本地 8 卡分析

实测 `nvidia-smi`（2026-09-18 14:02，驱动 580.105.08 / CUDA 13.0，8×RTX 4090 48 GB）：

| GPU | 显存占用 | Util | 占用者 | 判定 |
|---|---|---|---|---|
| 0 | 39707 / 49140 (81%) | 100% | whustu2（Senna LLaVA 多模态训练） | ✗ 满 |
| 1 | 6509 / 49140 (13%) | 34% | huzijing（walk_these_ways 训练中） | ⚠ 有余额但有人在跑 |
| **2** | **5 / 49140 (~0%)** | **0%** | **无** | ✅ **唯一完全空闲** |
| 3 | 32114 / 49140 (65%) | 87% | whustu2（Senna） | ✗ |
| 4 | 40024 / 49140 (81%) | 100% | whustu(cvact 23.2G) + **本账号 occworld_repro 16.7G** | ✗ 含自己的任务 |
| 5 | 12087 / 49140 (25%) | 87% | yuming（mmseg） | ✗ |
| 6 | 8175 / 49140 (17%) | 0–87% 波动 | whustu（cvusa） | ✗ |
| 7 | 41389 / 49140 (84%) | 0% | liwenzh（llama-server Qwen3-27B） | ✗ |

**结论 1：目标卡 = GPU2。** 唯一 0 占用 0 利用率的卡，独占。

> 附注（供纠错）：本项目记忆里「OccWorld 占 GPU1/5/6」**已过时**——实测 OccWorld 现在跑在
> **GPU4**（与 whustu 的 cvact 共卡）。记忆已更正。

### 结论 2（更重要）：**本项目不需要 8 卡，1 张就够**

这是「分析 8 卡」真正该得出的答案，理由是本任务的计算结构不对称：

| 阶段 | 性质 | 显存需求 | 说明 |
|---|---|---|---|
| 克隆模型（CosyVoice2-0.5B 等） | **只推理，不训练** | FP16 ~4–6 GB | 一次性生成数据集后即释放 |
| 检测器（AASIST ~297K 参数） | 训练 | bs=8 × 4 s 音频 **<8 GB** | 单卡 2–4 h 收敛 |
| SSL 检测器（WavLM + 轻头，冻结主干） | 训练（只训头） | ~8–12 GB | 单卡 1–2 h |

→ 全流程峰值 < 12 GB，**单卡串行**即可，且**不与任何人抢卡**。
**不要去申请多卡或做 DDP**——收益为零，只增加与共享机上其他 60 位用户的冲突面。
（共享机纪律见 skill `sharedbox-ops`；GPU 礼仪：只占空闲卡、跑完即释放。）

---

## 三、如何构建训练模型

### 3.1 网络现实（先决条件，实测）

| 目标 | 状态 | 处置 |
|---|---|---|
| `github.com` | ❌ 不通 | 代理 `https://gh-proxy.com/<url>` 或 `https://ghfast.top/<url>`（**均已实测 clone 成功**，AASIST 仓库已拉到） |
| `huggingface.co` | ❌ 不通 | `export HF_ENDPOINT=https://hf-mirror.com` |
| `hf-mirror.com` | ✅ 2.4 s | CosyVoice2-0.5B / F5-TTS / XTTS-v2 三个权重 **HTTP 200，无 gating** |
| `openslr.org` | ✅ 3.0 s | LibriSpeech 可直接下载（备选：hf-mirror 上 `openslr/librispeech_asr`） |
| 清华 pypi | ✅ 0.45 s | pip 一律加 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |

**这一条决定技术路线**：任何教程里 `git clone github.com/...` 和 `from_pretrained("...")` 的写法
在本机都会挂，必须先套代理/镜像。

### 3.2 环境现状与处置

现有 5 个 conda env **没有一个具备完整音频栈**（`torchaudio`/`soundfile`/`librosa` 全缺），
且系统**无 ffmpeg/sox**（编解码攻击必需）。

| env | python | torch | torchaudio |
|---|---|---|---|
| ml2026_exp | 3.10 | 2.11.0 | ✗ |
| LiNeXt | 3.9 | 1.13.1+cu117 | 0.13.1+cu117 |
| occworld_repro | 3.8 | 2.0.1+cu118 | ✗ |
| PointRWKV | 3.9 | 2.0.1+cu118 | ✗ |
| base | 3.13 | ✗ | ✗ |

→ 新建专用 env `voice`（脚本 `scripts/setup_env.sh`，已在后台执行）：
torch/torchaudio + soundfile/librosa + scipy/pandas/sklearn + `imageio-ffmpeg`（静态 ffmpeg 兜底）。
**不建议**往 `ml2026_exp`（另一门课在用）里塞东西。

### 3.3 数据构造 —— 全项目成败关键（含 4 个必须避开的陷阱）

**构成**：真实 = LibriSpeech dev-clean（2703 条）；伪造 = 零样本克隆（参考音频 + 目标文本）。
规模先做 100–200 条小闭环，再扩到 real 200–500 / 每克隆模型 200–500 / 每攻击 100–200。

> ⚠️ **陷阱 1 — 同文本捷径**：克隆时若用参考音频自身的文本，检测器会学「文本重复」而非合成痕迹。
> **必须跨句克隆**（参考音频 A 的说话人 + 另一句文本 B）。

> ⚠️ **陷阱 2 — 说话人泄漏**：real 与 fake 出自同一批说话人，若随机划分，测试集说话人见过。
> **必须按说话人划分（speaker-disjoint split）**，train/dev/test 说话人不重叠。

> ⚠️ **陷阱 3 — 学到的是生成器指纹，不是"伪造"**：自建数据上 in-domain EER 会非常漂亮，
> 但它只证明检测器认得出 **CosyVoice2 这个生成器**。**跨模型测试（训 CosyVoice2 → 测 F5-TTS/XTTS）
> 是唯一能暴露这一点的实验，也正是本项目真正的科学点。**

> ⚠️ **陷阱 4 — 信道缺失**：自产假音频是无损直出，而真实场景必带编解码/信道。
> 攻击变换支线就是补这个；不做的话结论不能外推到「反电诈」。

### 3.4 模型选择 —— 三层，从易到强

| 层 | 模型 | 参数量 | 单卡训练成本 | 角色 |
|---|---|---|---|---|
| 基线 | RawNet2 | ~1 M | 1–2 h | 轻量对照 |
| 主检测器 | **AASIST** | ~297 K | 2–4 h | 主结果（voice-theory 方案指定） |
| 强基线 | **SSL 前端（WavLM/XLS-R）+ 轻头**（冻结主干） | 冻结 300 M + 小头 | 1–2 h | **跨模型泛化的关键** |

**关键判断：在 2000 条量级的自建数据上，从头训 AASIST 会过拟合。**
AASIST 原论文是在 ASVspoof2019（>100 h）上训的。因此：
- 保留「从头训 AASIST」作为**课程叙事里的标准基线**；
- 但**必须同时跑「预训练权重微调」或「冻结 SSL 主干 + 轻头」**，那才是能出好看跨模型数字的那条线。
- 现成权重可得性见下文 SOTA 一节（HF 上已有第三方 AASIST 权重）。

### 3.5 训练配置（对齐 AASIST 原论文，便于与公开数字对齐）

- 16 kHz 单声道，随机裁 4 s（64600 采样点，AASIST 标准输入）
- bs=8，Adam lr=1e-4，~30–100 epoch，加权 CE（类别平衡）
- 增强：RawBoost（anti-spoofing 标准增强，需确认可获取性）
- 评估：**EER 为主**，AUC、t-DCF 为辅

### 3.6 评估协议（**预注册，先写死再跑**）

| 编号 | 协议 | 用途 |
|---|---|---|
| P1 | **In-domain**：同生成器，speaker-disjoint | 主数字，但**必须与 P2 同时报** |
| P2 | **Cross-model**：训 CosyVoice2 → 测 F5-TTS/XTTS | **核心结论** |
| P3 | 攻击鲁棒性：每个攻击条件下的 EER | 鲁棒性矩阵 |
| P4 | 对抗回灌：加攻击样本重训 → ΔEER | 恢复能力 + 代价 |

> **报数纪律**：只报 P1 会被误读成"检测器很强"。**P1 与 P2 必须成对出现**，
> 二者的落差本身就是本项目最有价值的发现。这条已按协作规范预注册。

### 3.7 时间与显存预算

| 阶段 | 显存峰值 | 墙钟 |
|---|---|---|
| 数据下载（LibriSpeech dev-clean ~337 MB） | — | <10 min |
| 克隆生成（CosyVoice2 FP16 推理） | ~5 GB | 1–2 h |
| 攻击变换（ffmpeg/librosa，CPU） | — | ~30 min |
| AASIST 训练 | <8 GB | 2–4 h |
| SSL 微调 | ~10 GB | 1–2 h |
| 评估 + 出图 | <8 GB | ~1 h |

→ **1–2 天可出第一版完整结果表。** 全程只占 GPU2。

---

## 四、下一步（可执行顺序）

1. ✅ 工作区 `voice_ws` 建立
2. ✅ 8 卡分析 → 目标卡 GPU2；结论 = 单卡够用
3. ✅ 方案侦察（本文档）
4. 🔄 env `voice` 后台安装中
5. ⏳ 待办：LibriSpeech dev-clean 下载 → CosyVoice2 冒烟（生成 5 条）→ AASIST 跑通 200 条小闭环
6. ⏳ 待 SOTA 调研结论回来后，在 §3.4 定稿「装哪一个检测器 / 哪一个克隆模型」
