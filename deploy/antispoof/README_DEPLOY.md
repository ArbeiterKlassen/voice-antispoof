# AASIST 语音辨伪推理服务 —— 部署包说明（Windows）

与 GPT-SoVITS 合成服务配套："攻"侧生成克隆语音，"防"侧判断真伪。**本包是"防"侧**：
上传一段音频 → 输出判真/判假与分数。

## 目录结构

```
antispoof_deploy_20261016/
├── code/                 # 官方 AASIST/AASIST-L 模型定义 + 权重（仅 1.7MB）+ 评估代码
│   └── aasist/{models,config,models/weights/AASIST.pth,AASIST-L.pth}
├── samples/              # real.wav / fake.wav（ASVspoof2019 LA 样例）
│   ├── clone.wav         # GPT-SoVITS 生成的克隆语音（"攻防闭环"演示用）
│   ├── ref_real.wav      # 同一说话人的真声（clone 的 prompt 音）
│   └── PROVENANCE.txt    # 各样例来源与实测分数（含口径）
├── detect_api.py         # FastAPI 服务（POST /detect）
├── start_detect_api.bat  # Windows 启动
├── test_detect.py        # 自检（三条样例各打一发）
└── README_DEPLOY.md      # 本文档
```

## 环境

**与合成包共用同一个 conda 环境 `gptsovits`**（librosa/soundfile/torch/fastapi 全都在），
无需新装任何东西。单独部署本包时要求：python 3.10 + torch + soundfile + librosa + fastapi + uvicorn；
**CPU 也能跑**（模型仅 297k 参数），只是每条约慢一个量级。

## 启动

```powershell
start_detect_api.bat
# 等价：conda run -n gptsovits python detect_api.py --port 9890
```
默认端口 **9890**，仅监听 127.0.0.1。模型 1.7MB，10–20 秒就绪。

## API 契约

```
POST http://127.0.0.1:9890/detect
Content-Type: multipart/form-data；字段名 file（支持 wav/flac）
```
响应 JSON：
```json
{
  "aasist_score": 2.31, "aasist_verdict": "real",
  "aasist_l_score": 1.87, "aasist_l_verdict": null,
  "threshold_aasist": -1.09385,
  "note": "bonafide logit；≥阈值判真。阈值=LA dev EER 交点（冻结）…"
}
```
另有 `GET /health`。

## 口径与已知边界（演示时必须如实说明）

1. 分数 = 真实（bonafide）logit，越大越"真"；AASIST 判真阈值 **−1.093850**（LA dev EER 交点，冻结口径）。
2. AASIST-L 仅报分数（其阈值未在本包标定）。
3. 输入按 **4.04s 定长**处理（短则重复填充、长则取头部），与主报告评估口径一致。
4. ⚠️ **强后处理会破坏标定**：混响/变速/音高级别的后处理会让分数整体下移、误报升高——
   这是本课题的核心发现之一（"证据不对称/标定崩溃"）。演示时按此口径说明：干净与编解码条件可靠，
   强后处理条件下需另行标定。
5. 双后端已知答案锚：LA eval 全量 AASIST 0.8297%（官方 0.83%）、AASIST-L 0.991%（官方 0.99%），
   详见主报告 `results/FINAL_REPORT_20261004.md`（voice_ws 仓库）。

## 实测行为（2026-10-16，服务口径头部 4.04s，详见 samples/PROVENANCE.txt）

| 样例 | AASIST 分数 | 判词 |
|---|---|---|
| real.wav（LA 真人） | +5.29 | real ✅ |
| fake.wav（LA 2019 伪造） | −2.82 | fake ✅ |
| ref_real.wav（真声，clone 的 prompt） | +2.09 | real |
| **clone.wav（GPT-SoVITS v4 克隆）** | **−0.79** | **real（逃逸）** |

⚠️ 关键事实：**同一说话人**的真声（+2.09）与克隆（−0.79）在服务口径下分得开，
**但克隆侧越过了"判真"阈值** → 2024 代零样本克隆对"阈值冻结在 2019 LA-dev 分布"的检测器
存在泛化落差（clone 全曲滑窗 −1.96…+1.59，仅部分段被抓）。这与主报告里 DF full-length
EER 16.6%、LA-P4 后处理标定崩塌是同一条结论线：**检测器老旧 + 标定脆弱**。
演示时如实讲这个落差（这正是课题主题），**不要挑窗口凑"抓到"的结论**。

## 自检

```powershell
python test_detect.py
```
- 硬判据：`real.wav→real`、`fake.wav→fake`（返回码依据）
- 信息项：`clone.wav`、`ref_real.wav` 打印分数（已知结果，如实展示，不参与返回码）

## 与合成服务串联（演示流程建议，按实测口径）

1. 网站输入一句文本 → 调合成服务 `:9880/tts` → 得到克隆语音（"攻"）
2. 把克隆语音喂给辨伪服务 `:9890/detect` → 实测**判 real（攻击成功）**（"防"的落差）——
   接着把 LA 2019 伪造 `fake.wav` 也打一发 → 判 fake，说明检测器在训练分布内是有效的
3. 对照播放 `samples/ref_real.wav`（同说话人真声）与 `clone.wav`（克隆）：两段人耳难分，
   检测器也常分不开——这就是"攻防"的现实起点；再引到主报告里的 DF/后处理衰减结果
