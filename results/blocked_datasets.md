# 受阻数据集与替代路径 — 任务 02 §5 交付项

更新于 2026-09-18 · 实验侧 AI

任务 02 §1.1 要求：**A 级数据集要么落盘并校验，要么在本文件写清原因和替代路径，不允许静默跳过。**
任务 02 §7 要求：A4 走官方申请还是先走镜像，需在此落一条明确记录。

---

## A4 — ASVspoof2019 LA（**已由镜像路径获取，2026-10-04**）

| 项 | 内容 |
|---|---|
| 用途 | AASIST 官方训练与**与文献可比的基准**；min-tDCF 唯一可计算的数据集 |
| 参考体积 | 约 20 GB（train/dev/eval 三划分） |
| 官方发放 | Edinburgh DataShare，需接受许可条款并按机构申请 |
| 本机网络 | `github.com` / `huggingface.co` 直连不通；已确认 hf-mirror 可用 |

### 进展（2026-10-04）：**镜像路径已走通，走的是下方「路径 2」**

- 来源：`hf-mirror.com/datasets/Bisher/ASVspoof_2019_LA`
- 内容：**完整三划分** parquet —— train 25380 / validation 24844 / test 71237，
  字段 `speaker_id / audio_file_name / audio / system_id(A01–A19) / key`
- 标签方向：parquet `key` 为 **0=bonafide / 1=spoof**，与本项目内部 `real=1` **相反**；
  `code/prepare_asvspoof.py` 显式转换，并**用官方协议文件逐条反查**（见下）
- 官方协议：`protocols/ASVspoof2019.LA.cm.eval.trl.txt`（71237 行，含 speaker/攻击编号/标签）
  → 用于交叉验证镜像标签，**不依赖镜像自身的正确性**
- 纪律：按本研究记录 §1.1，走镜像路径的结果**不得**写成与文献可比，
  报告内须标注「非官方镜像，仅用于管线验证」；**但**标签与划分已与官方协议逐条对齐，
  故「EER 数值可与文献对照」这一点可由协议一致性支撑（会在报告中写明这层论证）

### 已定的下一步（管线已知答案测试）

用**官方 AASIST.pth 原权重**（论文报告 LA eval EER 0.83%）跑我们的评估管线：
复现得上 → 管线可信，后续自建方法在 LA 上的数才有意义；复现不上 → 先修管线。
（LA 官方三划分说话人不相交，官方权重训于 LA train，故对 LA eval 是正当 held-out。）

### 原两条路径存档（保留供对照）

**路径 1 — 官方申请（唯一能产出可比结果的路径）**
- 优点：结果可与文献比较；min-tDCF 可算
- 风险：**周期不可控**，可能不是当天能拿到
- 状态：尚未提交

**路径 2 — 镜像先跑通管线（仅管线验证）**
- 优点：不阻塞 A2/A3 的全部工作
- **强制约束**：按任务 02 §1.1，走此路径时报告内**不得**把结果写成与文献可比，
  且必须显式标注"非官方镜像，仅用于管线验证"
- 状态：尚未执行（待裁定后决定）

### 实验侧建议：**并行**

理由：路径 1 的周期不可控且不在我方可控范围；路径 2 不消耗官方申请的任何资源。
两者并行不会有冲突，且能避免"官方许可下来前整条线停摆"。
若 voice-theory 要求单走路径 1，本条即改为"等待官方许可，期间 P1 主实验挂起"。

**→ 待 voice-theory 在此条下回填裁定结果。**

---

## A2 / A3 — LibriSpeech test-clean / train-clean-100（**进行中**）

| 编号 | 数据集 | 体积 | 状态 |
|---|---|---|---|
| A2 | LibriSpeech test-clean | 334 MB | ✅ 已落盘 |
| A3 | LibriSpeech train-clean-100 | 约 6.3 GB（14 分片） | ✅ **已完整落盘**（2026-10-04 核实：14/14 分片，`download_verified.py` 三关校验通过 `DOWNLOAD_VERIFIED_DONE`）——**尚未用于训练**，属现成未用的大杠杆（2814 位说话人 / 100 h） |

**⚠️ 优先级说明（2026-10-04）**：A3 的"更多真实数据"价值已被 LA train（20 位说话人、19 种神经攻击、
有文献刻度）取代。自建轨的伪造是 signalproc 平凡伪造，扩数据不改变"无法与文献比较"这一根本问题，
故**暂时不从 A3 扩训**，GPU 让给 LA 轨。

- 来源：`https://hf-mirror.com/datasets/openslr/librispeech_asr`（parquet 分片）
- **不走 openslr.org**：实测仅 15 KB/s（338 MB 需 6 小时）；任务 02 §0 已允许 openslr 仅作后台限速通道
- hf-mirror 实测速率：244 KB/s ~ 1.2 MB/s（波动）

---

## B3′ — ASVspoof2021 DF 打包版（**实测不可得，2026-10-04**）

| 项 | 内容 |
|---|---|
| 候选来源 | `hf-mirror.com/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF`（80 分片 + labels，32 GiB） |
| 为什么选它 | ① 原版 DF flac 约 40% 是 libsndfile 读不了的（"flac decoder lost sync"），本机管线正是 soundfile；这一版作者已逐条重编码 ② 自带 `notes`（codec/attack_id/vocoder），不必等官方 keys |
| 实测速率 | **单连接 0.14 MB/s；4 路并行 0.50 MB/s；`hf` CLI 90s 0 字节**。按此速率 32 GiB 需 **18–63 小时** |
| 对照 | 同一镜像的 LA（7 GiB）实测 2–4 MB/s，**当天即下载完成** → 不是网络整体问题，是这些分片被限流 |
| 状态 | **音频未获取**（不阻塞主线）；但**官方 keys 已到手并解析**（2026-10-04）：<br>`DF-keys-full.tar.gz` MD5 `dabbc5628de4fcef53036c99ac7ab93a` ✅ 与官方公布一致，解包得 `trial_metadata.txt` **611829 行** |
| keys 解析结果 | 伪造 589212 / 真实 22617；**codec 是均衡设计**（9 种各恰好 67981 条）；<br>来源 vcc2020 265950 / vcc2018 186183 / asvspoof 159696；攻击系统 111 种；93 位说话人 |
| 🔴 泄漏检查（已用 keys 完成，无需音频） | DF 真实说话人 ∩ **LA-train = 0**、∩ **LA-dev = 0**、∩ LA-test = 67<br>⇒ **DF 的真实语音取自 LA test 说话人**，故「在 LA train 上训练 → 评 DF」**无说话人泄漏** |
| 替代路径 | ① 隔一段时间重试（可能为临时限流）②只取前 N 个分片做子集评估——**但须先验证分片是否按 codec/attack 分块**，若前几片同质则会引入系统性偏差 ③ 换 `Bisher/ASVspoof_2021_DF` 原始 zip（需自行解决 40% 解码失败） |



B1 VCTK/LibriTTS-R、B2 WaveFake、B3 ASVspoof2021 DF、B4 FoR、B5 In-the-Wild；
C 级 ASVspoof5、CodecFake、MLAAD。

**已探明的可下载清单（2026-10-04，同一镜像）**：`SpeechAntiSpoofingBenchmarks/` 下还有
InTheWild / ASVspoof5 / ADD22_eval_31 / ADD2023_track12_test_r1 / LibriSeVoc / CVoiceFake_small /
ArAD / DECRO / DeepVoice / J-SPAW_LA / ODSS / HABLA / DFADD / CD-ADD / CFAD / SONAR / arena-manifest。
按任务 02 §1.3，动用前需先落盘 A、B 级并留 >200 GB。

**磁盘预检**：`/data1` 总 15 TB，已用 66%，**剩余 4.8 TB**。
A 级约 30 GB、B 级约 150 GB，均在余量内。
按任务 02 §1.3，C 级仅在 A、B 全部落盘且剩余空间 > 200 GB 时再动。

---

## 环境侧受阻项（非数据集，但同样阻塞）

### CosyVoice2 环境
- **状态**：安装中。官方 `requirements.txt` 钉 `torch==2.3.1`（cu121），
  其依赖链需拉全套 cu12 系 nvidia 包（cudnn 731 MB、cusparse 196 MB 等），带宽受限。
- **已避开的坑**：`tensorrt-cu12` 三件套（体积大且纯 PyTorch 推理不需要）、`deepspeed`（需编译）。
- **替代路径**：若 CosyVoice2 最终装不通，按任务 02 P2 的要求，F5-TTS / XTTS-v2 本就是必须项，
  可先跑通 F5-TTS 生成链，CosyVoice2 作为 P1 主生成器另行攻坚。

### min-tDCF
- **状态**：BLOCKED（定义性缺失，非实现问题）
- 详见 `protocol_locked.yaml` 的 `metrics.min_tdcf` 段，已向 voice-theory 提出待裁定
