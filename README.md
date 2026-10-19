# voice-antispoof

语音辨伪与语音克隆对抗项目。检测侧以 AASIST 与 AASIST-L 为主后端，在 ASVspoof2019 LA 与
ASVspoof2021 DF 两条公开轨道上评估，生成侧覆盖 CosyVoice2、F5-TTS、XTTS-v2 的零样本克隆，
并在检测器上施加编解码、加噪、混响、变速、音高等后处理攻击。

项目由理论侧与实验侧两个 AI 协同推进。分工、判据纪律与汇报口径见
[docs/协作约定.md](docs/协作约定.md)。

## 目录结构

| 路径 | 内容 | 是否入库 |
|------|------|----------|
| `code/` | 模型定义、数据加载、训练与评估脚本 | 入库 |
| `scripts/` | 可复算的入口脚本 | 入库 |
| `results/` | 指标 json、分数 npz、汇总表 | 入库，大权重除外 |
| `docs/` | 报告、预注册、任务书、协作约定 | 入库 |
| `data/` | 数据集与 manifest | 数据不入库，见 `data/README.md` |
| `models/` | 预训练权重与检查点 | 权重不入库，见 `models/README.md` |
| `logs/` | 运行日志 | 不入库 |

## 数据与权重政策

数据集与模型权重一律不进版本库。LA 全量约 20 GB，DF 全量约 32 GB，AASIST 权重约 2 MB，
任何一条都不适合放进 git 历史。仓库只保留 manifest、下载脚本与校验和。

manifest 里记录路径、哈希与来源，脚本按 manifest 复现数据准备过程。

## 已知答案锚

管线可信度的判据是三组可核对的公开数值，复算入口见下一节。

| 锚 | 我们的值 | 官方参照 | 差 |
|----|----------|----------|-----|
| AASIST，LA eval 全量 EER | 0.8297% | 0.83% | −0.0003 个百分点 |
| AASIST，LA eval min-tDCF | 0.027514 | 0.0275 | +0.000014 |
| AASIST-L，LA eval 全量 EER | 0.991% | 0.99% | +0.001 个百分点 |

三个锚都过，才认为数据读取、定长填充、打分方向与指标计算这条链路是忠实的。

## 复算入口

命令在实验侧工作区根目录执行。

```bash
# 管线已知答案
bash scripts/run_la_evals.sh 4

# LA-P4 分析，含逐族两口径与误报率
python scripts/analyze_la_p4.py

# 同信道标定
python scripts/calibration_fix.py --arm official_AASIST.p4fix \
   --clean results/scores/official_AASIST.la_clean.npz \
   --p4 results/scores/official_AASIST.la_p4fix.npz --threshold -1.093850

# 三臂汇总与全结果看板
python scripts/summarize_arms.py --detail
python scripts/collect_results.py

# 第二后端与音高族
bash scripts/run_pitch_eval.sh

# 标定第二轮
python scripts/calibration_round2.py

# 生成链客观检查
python scripts/gen_chain_checks.py --selftest
```

## 结果口径

同一批数据必须并列报同族口径与混合口径。同族口径回答该信道下还能不能辨，混合口径回答未知信道
部署时的整体表现，两者结论方向相反，只报一个会误导。

误报率与判别力分开报。同信道标定修误报，不修不可辨，两类失效需要不同手段。

## 引用与上游项目

本项目建立在以下公开项目之上，特此注明来源。生成侧（语音合成与克隆）的代码与权重**不入库**，
按上游地址自行获取；数据与权重政策见 `data/README.md`、`models/README.md`。

**生成侧（语音合成与语音克隆）**

| 项目 | 用途 | 上游 |
|------|------|------|
| **Lab_VoiceClone** | **语音合成侧的原始项目**：本实验所用语音克隆仓库（内容基于 Amphion） | <https://github.com/afterRain-2005/Lab_VoiceClone> |
| Amphion | 语音生成工具箱（Lab_VoiceClone 的上游） | <https://github.com/open-mmlab/Amphion> |
| MaskGCT | 零样本 TTS 克隆模型（含于 Amphion） | <https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct> |
| CosyVoice 2 | 零样本克隆模型（生成侧跨模型对照） | <https://github.com/FunAudioLLM/CosyVoice> |
| F5-TTS | 零样本克隆模型（生成侧跨模型对照） | <https://github.com/SWivid/F5-TTS> |
| XTTS-v2（Coqui TTS） | 零样本克隆模型（生成侧跨模型对照） | <https://github.com/coqui-ai/TTS> |

**检测侧**

| 项目 | 用途 | 上游 |
|------|------|------|
| AASIST | 检测主后端；官方实现代码内嵌于 `code/aasist/`（来源 commit 见 `docs/实验侧环境.md`） | <https://github.com/clovaai/aasist> |
| ASVspoof 2019/2021 | 评测数据集与官方基线协议（第三后端 LFCC-GMM 按官方基线口径实现） | <https://www.asvspoof.org/> |

---

## 完整攻防栈（分支 `voice-arena`）

主干只保留辨伪的实验工作区。分支 `voice-arena` 把综合实验的攻防闭环作为一个可运行的
完整工程收进同一个仓库，新增 `deploy/` 与 `app/` 两块。

| 路径 | 内容 | 规模 |
|------|------|------|
| `deploy/antispoof/` | 辨伪推理服务：`detect_api.py`（FastAPI，`POST /detect`）、启动与自检脚本、四条样例音频 | 1.8 MB / 29 文件 |
| `deploy/synthesis/` | 合成推理服务：GPT-SoVITS v4 精简部署（含两处补丁）、安装与启动脚本、参考音 | 16 MB / 280 文件 |
| `app/` | 演示网页 voice-arena：Node 服务端 + 原生前端，把两侧串起来并提供一键攻防回合 | 0.1 MB / 17 文件 |

### 链路

```text
文本 ──► GPT-SoVITS（:9880）──► 克隆语音 ──► AASIST（:9890）──► 真伪分数与判定
                                                      │
                                                      └──► 网页留档、回放、对抗回合记录
```

两个模型服务都只监听 `127.0.0.1`，对外只暴露网页端口 `8790`。

### 起服务

```bat
:: 1) 两个模型服务（共用 conda 环境 gptsovits）
deploy\antispoof\start_detect_api.bat
deploy\synthesis\start_api.bat

:: 2) 网页
cd app
copy config.example.json config.json   :: 按本机路径改 modelDirs，按需填 access.token
node server\index.mjs                 :: 或 start-voice-arena.bat
```

合成侧的环境安装、Windows 上的五个坑与实测耗时，见 `deploy/synthesis/README_DEPLOY.md`
与 `app/README.md` 的「本机部署记录」。

### 权重与数据

沿用本仓库政策，权重与数据集不入库。

| 文件 | 落位 | 获取方式 |
|------|------|----------|
| `AASIST.pth`、`AASIST-L.pth`（1.7 MB） | `deploy/antispoof/code/aasist/models/weights/` | 随辨伪部署包分发 |
| `s1v3.ckpt`、`S2Gv4.pth`、`vocoder.pth`、BERT/HuBERT、`lid.176.bin`（1.9 GB） | `deploy/synthesis/GPT-SoVITS/GPT_SoVITS/pretrained_models/` | 随合成部署包分发 |
| `g2pW.onnx`（606 MB，中文注音） | `deploy/synthesis/GPT-SoVITS/GPT_SoVITS/text/G2PWModel/` | 随合成部署包分发 |
| ASVspoof 2019 LA / 2021 DF | `data/` | 见 `data/README.md` 的 manifest |

分支里保留了部署包自带的五条演示音频（`samples/` 四条 + `voices/ref_user1.wav`，共 2.3 MB），
其余 `*.wav` 仍按 `.gitignore` 排除。为了让分支可直接运行，这两处是从部署包强制加入的，
不属于数据集的范畴。

### 公网访问

站点通过独立的 Cloudflare Tunnel 暴露，与 agenthub 那条互不影响。接入步骤、令牌门与
免费版限制（单请求约 100 秒超时）见 `app/README.md` 的「公网暴露」一节，
示例配置在 `app/docs/cloudflare-tunnel.example.yml`。
