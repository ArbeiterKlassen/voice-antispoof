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
| Amphion | 语音生成工具箱，本项目克隆侧的工作副本来源 | <https://github.com/open-mmlab/Amphion> |
| MaskGCT | 零样本 TTS 克隆模型（含于 Amphion） | <https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct> |
| CosyVoice 2 | 零样本克隆模型（生成侧跨模型对照） | <https://github.com/FunAudioLLM/CosyVoice> |
| F5-TTS | 零样本克隆模型（生成侧跨模型对照） | <https://github.com/SWivid/F5-TTS> |
| XTTS-v2（Coqui TTS） | 零样本克隆模型（生成侧跨模型对照） | <https://github.com/coqui-ai/TTS> |

**检测侧**

| 项目 | 用途 | 上游 |
|------|------|------|
| AASIST | 检测主后端；官方实现代码内嵌于 `code/aasist/`（来源 commit 见 `docs/实验侧环境.md`） | <https://github.com/clovaai/aasist> |
| ASVspoof 2019/2021 | 评测数据集与官方基线协议（第三后端 LFCC-GMM 按官方基线口径实现） | <https://www.asvspoof.org/> |
