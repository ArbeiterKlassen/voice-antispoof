# GPT-SoVITS v4 零样本语音克隆 —— 部署包说明（Windows / NVIDIA 4070）

本包是**推理专用**精简部署：拿一段 3–10 秒参考音，输入任意文本，输出克隆该音色的语音。
开发侧已在 Linux 4090 上完整验证（合成正确性用 ASR 回译逐字核对通过），本包把全部踩过的
坑**前置修掉**，Windows 上按下面步骤走即可。

## 目录结构

```
gptsovits_deploy_20261016/
├── GPT-SoVITS/                 # 官方代码（已打 2 处补丁，见"已知坑"1）
│   └── GPT_SoVITS/
│       ├── configs/tts_infer.yaml        # custom 段已配置为 v4
│       ├── text/G2PWModel/               # 中文注音（ONNX 版，必需）
│       └── pretrained_models/            # 精简 v4 权重（~2GB）
│           ├── s1v3.ckpt                 # GPT 模型
│           ├── gsv-v4-pretrained/        # s2Gv4.pth + vocoder.pth
│           ├── chinese-roberta-wwm-ext-large/   # 文本 BERT（safetensors）
│           ├── chinese-hubert-base/             # 语音 HuBERT（safetensors）
│           └── fast_langdetect/lid.176.bin      # 语种检测
├── voices/ref_user1.wav(.txt)  # 参考音 + 逐字文本（换音色改这里）
├── nltk_data/                  # 文本前端资源（20MB，安装脚本会自动装入环境）
├── out/                        # 合成输出目录（空）
├── install_windows.ps1         # 一键安装（见下）
├── start_api.bat               # 启动服务 + 自动预热
├── test_acceptance.py          # 自检（两条请求 + 时长/字节校验）
└── README_DEPLOY.md            # 本文档
```

## 安装（Windows）

前置：已装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)（或 Anaconda）；
NVIDIA 驱动正常（4070 无需另装 CUDA Toolkit——torch 自带运行时）。

```powershell
# 在解压后的 gptsovits_deploy_20261016 目录里，开 PowerShell：
powershell -ExecutionPolicy Bypass -File .\install_windows.ps1
```

脚本做的事（等价的命令即注释在脚本里，可手动逐条跑）：
1. `conda create -n gptsovits python=3.10 -y`
2. `pip install torch==2.5.1 torchaudio==2.5.1`（清华源；PyPI 默认 wheel 即 cu124 构建）
3. `conda install -c conda-forge opencc -y`（**先装它**，避免 requirements 里 `--no-binary=opencc` 在 Windows 上源码编译失败）
4. `pip install -r requirements.txt`（清华源）
5. `pip install resampy`（requirements 漏项，实测必需）
6. 检查 ffmpeg（没有则提示：`winget install Gyan.FFmpeg`；纯 wav 流程可暂缺）

## 启动

双击 `start_api.bat`（或手动）：等服务加载完模型（~30–60 秒，日志现"Uvicorn running"），
脚本会**自动打一发预热请求**——不要跳过预热（首跑含 CUDA 图捕获，费时且输出不稳）。

服务地址：`http://127.0.0.1:9880`；日志：`api_v2.log`。

## API 契约

`POST http://127.0.0.1:9880/tts`，请求 JSON：

| 字段 | 必填 | 说明 |
|---|---|---|
| `text` | ✅ | 要合成的文本（中文）|
| `text_lang` | ✅ | 固定 `"zh"` |
| `ref_audio_path` | ✅ | 参考音**绝对路径**（`voices/ref_user1.wav`）|
| `prompt_text` | ✅ | 参考音的**逐字文本**（`ref_user1.txt` 内容）|
| `prompt_lang` | ✅ | 固定 `"zh"` |
| `streaming_mode` | — | `false`（本包按整段返回）|
| `media_type` | — | `"wav"` |
| `text_split_method` | — | `"cut5"`（长文本按标点切分，推荐）|
| `parallel_infer` / `use_cuda_graph` | — | **不要传 true**（已在代码里默认 false，见"已知坑"1）|

响应：`audio/wav` 字节流（直接存文件即可）。

```bash
# 最小示例（Windows 自带 curl.exe）
curl -s -X POST http://127.0.0.1:9880/tts -H "Content-Type: application/json" -d "{\"text\":\"你好，这是一条测试。\",\"text_lang\":\"zh\",\"ref_audio_path\":\"D:\\gptsovits_deploy_20261016\\voices\\ref_user1.wav\",\"prompt_text\":\"一大早沁凉的风就暗暗催送时间拨弄湖水的声音。\",\"prompt_lang\":\"zh\",\"streaming_mode\":false,\"media_type\":\"wav\",\"text_split_method\":\"cut5\"}" -o out\test.wav
```

## 换音色

把新的参考音（3–10 秒、干净、单人）放进 `voices/`，附**逐字文本**，请求里改
`ref_audio_path` + `prompt_text` 两个字段即可，无需重启、无需重训。

## 已知坑（开发侧实测记录，全部已修复）

1. **`parallel_infer`/`use_cuda_graph` 默认 true 会静默丢音频段**（多分句输入丢句、
   单句偶发乱码）。本包已把 `api_v2.py` 两处默认值改为 false 并注释——请勿覆盖成 true。
2. **新版 transformers 拒绝用 torch<2.6 加载 `.bin` 权重**（CVE-2025-32434 门槛）。
   本包权重已转为 `model.safetensors`——**不要删**，也不需要 `.bin`。
3. `resampy` 不在官方 requirements 里，缺了会报 `No module named 'resampy'`。
4. 中文注音必须用 **ONNX 版 G2PWModel**（在 `GPT_SoVITS/text/G2PWModel/`，
   与本包不同布局的 transformers 版 `g2pw-chinese` 不可用）。
5. Windows 上 `pip install opencc` 会走源码编译（requirements 的 `--no-binary=opencc`），
   极易失败——用 `conda install -c conda-forge opencc` 绕过。
6. 首次请求慢（模型加载+预热）属正常；之后每条请求秒级。
7. 服务默认只监听 127.0.0.1（本机）。如需局域网/公网访问再按你们的方案改 `-a 0.0.0.0` + 反代/隧道。

## 性能参考（Linux 4090 实测；4070 Laptop 预计 ×1.5–2，仍为秒级）

| 输入 | 合成耗时 | 音频时长 |
|---|---|---|
| 16 字单句 | 1.5 s | 4.0 s |
| 30 字双句 | 3.4 s | 11.2 s |

RTF ≈ 0.3–0.4（实时率）。

## 自检

```bash
python test_acceptance.py
```
发两条请求（单句/双句）存到 `out/`，打印 HTTP 码、字节数、音频时长；
全部通过 + 能人工试听即部署成功。（可选进阶：装 funasr 可做 ASR 回译核对，见脚本注释。）
