# 两项目合并方案

实验侧 AI · 2026-10-03

---

## 1. 合并了什么

| 项目 | 角色 | 位置 |
|---|---|---|
| **检测侧**（本项目） | 防守方：AASIST 检测器 + 评估协议 + 攻击链 | `/data1/cuitianxu/voice_ws/code/` |
| **克隆侧**（同学） | 攻击方：VALL-E + Token-DiffWave 语音克隆 | `/data1/cuitianxu/voice_ws/code/voiceclone/` |

**Provenance（合并来源，可追溯）**

- 克隆侧仓库：`afterRain-2005/Lab_VoiceClone`，commit `891367f287deb4df19c88b452084098d8bfd808b`（2026-10-03 "first commit"）
- 其上游：**OpenMMLab Amphion**，main 分支，`26f6883110181f1dbfe95c70a7c7dbaf4de5f42a`（2026-10-02 获取）
- 新增 recipe：`egs/tts/VALLE_DiffWaveClone/`
- 获取方式：`ghfast.top` 代理（⚠️ 见 §5 网络变更）

## 2. 克隆侧的技术构成

```
3–10 s 参考音频
  ├─ EnCodec 8 层 RVQ (1024 码本) ─────┐
  └─ ECAPA-TDNN speaker embedding ────┼─> Speaker-conditioned VALL-E
文本 ──> G2P / text token ─────────────┘        │
                                    AR 第 1 层 + NAR 其余 7 层
                                                │
                              8 层 token + speaker embedding
                                                │
                                  Token-conditioned DiffWave
                                                │
                                     DDPM 逐步去噪 -> 24 kHz 波形
```

5 阶段流水线：`0 元数据 → 1 EnCodec/ECAPA 特征 → 2 AR → 3 NAR → 4 DiffWave → 5 推理`
（`egs/tts/VALLE_DiffWaveClone/run.sh` 已封装）

## 3. 接口契约（合并的核心）

新增 `code/ingest_clones.py`：**克隆器产出 → 检测器 manifest** 的适配器。

它处理三处**必须转换**的地方：

| # | 问题 | 处理 |
|---|---|---|
| 1 | **采样率**：克隆器输出 24 kHz，AASIST 要求 16 kHz | 重采样 |
| 2 | **划分归属**：克隆件必须继承**参考音频说话人**的 split | 按 `prompt_utt_id` 查真实 manifest 反填 speaker 与 split，保住 speaker-disjoint |
| 3 | **跨句核验**：任务 02 §3.2 规定 1 | 文本归一化后比对；失败即**整批拒绝**（退出码 2），并写入 `prompt_utt_id` 列作可核查证据 |

附带执行任务 02 §8.3 的**强制检查项 1**（幅度对齐到参考音频 RMS，含削波防护）。

**输入契约**（克隆侧需产出 JSONL，每行一条）：
```json
{"wav": "...", "prompt_wav": "...", "prompt_utt_id": "librispeech-test-00173",
 "prompt_text": "...", "target_text": "...", "clone_model": "valle_diffwave"}
```

**已过合成已知答案自检**：3 条用例（正常 / 跨句失败 / 参考查不到），判定全对且坏批次被正确拒绝。

## 4. ⚠️ 当前阻塞（两个，都不是我能自己解决的）

### 阻塞 1：仓库内**没有任何训练好的权重**

全仓 `pretrained/*` 只有 README.md 占位文件；`find -size +5M` 与 `*.pt/*.pth/*.ckpt` 全仓搜索均无模型权重。

**后果**：克隆器的 5 阶段里，阶段 2/3/4 都需要训练产出的 ckpt（AR / NAR / DiffWave）。
没有 ckpt 就只能从零训练 —— 而见阻塞 2。

**需要**：问同学是否有已训练好的 ckpt 可以给（哪怕只是能跑通推理的最小版本）。

### 阻塞 2：**8 张卡全部占满**

2026-10-03 实测：8 张 4090 的显存占用全部在 48.3–48.5 GB / 49.1 GB 之间（≥98%），
占卡进程里 **9 个是我自己的 OccWorld 任务**。

另外用户 10-03 明确要求**服务器压力大期间暂停新 launch**。

**后果**：即使有 ckpt，现阶段也不该起新的训练/推理任务。
**但注意**：检测侧的评估、指标实现、数据整理都是 CPU 任务，**不受此限，可以继续推**。

## 5. ⚠️ 网络变更（2026-10-03 实测，与前两周不同）

| 目标 | 09-18 状态 | **10-03 状态** |
|---|---|---|
| `gh-proxy.com` | ✅ 可用 | ❌ **连接被拒**（HTTP 000，0.001s） |
| `ghfast.top` | ✅ 可用 | ✅ **仍可用**（本次 clone 成功） |
| `gitclone.com` | ✅ | ✅ 通 |
| `github.com` | ❌ | ❌ 仍不通 |
| `hf-mirror.com` | ✅ | ✅ 仍通 |
| `huggingface.co` | ❌ | ❌ 仍不通 |

**新坑**：本机 shell 里有 `https_proxy=http://127.0.0.1:9999`（LiNeXt_ws 的 CONNECT 隧道），
它的**转发目标会串**（`ah_send.py` 头注释记录：经它连 agenthub 会拿到 `*.unionpayintl.com` 证书）。
**git / curl 都会继承它并失败**（`Proxy CONNECT aborted`）。

→ 所有 git/curl 操作必须显式绕开：`env -u https_proxy -u http_proxy ... ` 或 curl `--noproxy '*'`，
Python 用 `urllib.request.build_opener(ProxyHandler({}))`。

## 6. 合并路线图

| 阶段 | 内容 | 状态 |
|---|---|---|
| M1 | 代码级合并 + provenance 记录 | ✅ **本次完成** |
| M2 | 接口契约 + 适配器 + 自检 | ✅ **本次完成** |
| M3 | 拿到 ckpt（或确认需自训） | ⛔ **待同学/用户** |
| M4 | 跑通克隆推理，产出第一批克隆件 | ⛔ 依赖 M3 + GPU |
| M5 | 经 `ingest_clones.py` 汇入 manifest，跑**协议 P1** | ⛔ 依赖 M4 |
| M6 | **P2 跨模型**（VALL-E vs 我现有的信号处理类）+ ΔEER | ⛔ |
| M7 | **对抗闭环**：检测器 → 攻击链 → 回灌再训练（原项目目标） | ⛔ |

## 7. 现阶段可做（不受 GPU 限制）

1. ✅ 合并与接口（本次）
2. 实现 voice-theory §8.1 要求的 **pAUC@FPR≤1% / pAUC@FPR≤0.1%**（纯 CPU）
3. 用现有信号处理类伪造 + 攻击链，把 **P4 鲁棒性基线**跑出来（CPU/单卡轻量）
4. 整理克隆侧训练所需数据（LibriTTS 等）的前置准备
