# voice-arena · 语音攻防演示平台

数字信号处理综合实验「语音克隆与辨伪对抗」的演示站点。把两侧模型放进同一个页面，
一侧生成克隆语音，另一侧判断真伪，并把每一轮的结果与音频留档。

| 角色 | 模型 | 服务 | 默认端口 |
|------|------|------|----------|
| 攻 | GPT-SoVITS 零样本克隆 | 推理 API | 9880 |
| 防 | AASIST / AASIST-L | `detect_api.py` | 9890 |
| 站点 | 本仓库 | `server/index.mjs` | 8790 |

## 启动

```text
1) start-services.bat     先起两个模型服务（合成 + 辨伪）
2) start-voice-arena.bat  再起本站点，浏览器打开 http://127.0.0.1:8790
```

不需要 `npm install`，也没有构建步骤：服务端只用 Node 内置模块，前端是原生 HTML/CSS/JS。
这样做的原因是现场演示最怕环境出问题，少一层依赖就少一个失败点。

## 页面功能

- **攻击侧**：填文本、选参考说话人音频、调语速与采样参数，调用合成服务得到克隆语音。
  参考音频可以直接用辨伪包里的 `ref_real.wav`，也可以上传自己的。
- **防御侧**：拖入任意音频或点样例的「送检」，得到 AASIST 与 AASIST-L 两条分数、
  判定结论，以及与阈值的余量；页面上用刻度条把分数和阈值画在同一根轴上。
- **完整回合**：「合成并送检」一次跑完攻防闭环，直接给「攻击成功 / 被拦截」的结论。
- **样例库**：辨伪包自带的四条音频，标注组包时的实测分数，方便核对判定口径。
- **回合记录**：每次合成、送检、对抗都落一条记录，可回放音频、清空或导出。

## 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/state` | 两个上游服务的在线状态、阈值、模型目录 |
| GET | `/api/samples` | 辨伪包样例列表与实测分数 |
| POST | `/api/tts` | 只做合成，body 为 `{text, ref, params}` |
| POST | `/api/detect` | 送检，支持裸字节流、multipart 或 `?media=sample:xxx.wav` |
| POST | `/api/attack` | 合成后立刻送检，返回整轮记录 |
| POST | `/api/refs` | 上传参考说话人音频，返回可被合成服务读取的 id |
| GET | `/api/runs` / DELETE | 回合记录 |
| GET | `/media/...` | 播放或下载生成音频与样例音频 |

## 口径

页脚与「口径说明」卡片里的内容直接取自辨伪部署包，演示时照此说明，不挑窗口凑结论。

- 分数是 bonafide logit，越大越「真」；阈值 −1.093850 是 ASVspoof2019 LA dev 的 EER 交点，已冻结。
- 输入按 4.04 秒定长处理（短则重复填充，长则取头部），与主报告评估口径一致。
- AASIST-L 只报分数，其阈值未在本包标定。
- 混响、变速、音高等强后处理会让分数整体下移，该阈值随之失效，这是课题的核心发现之一。
- 跨代零样本克隆（2024 代 GPT-SoVITS）对冻结在 2019 分布的检测器存在泛化落差，
  同说话人的真声与克隆在服务口径下分得开，但克隆侧会越过判真阈值，即「攻击成功」。

## 目录

```text
voice-arena/
├── server/           服务端：静态托管 + 上游转发 + 记录
│   ├── index.mjs     路由与启动
│   ├── upstream.mjs  两个模型服务的客户端
│   ├── multipart.mjs multipart 构造与解析
│   ├── store.mjs     记录与音频落地
│   └── config.mjs    配置与模型目录探测
├── web/              前端，原生实现，无构建
├── data/             运行期数据：media 音频、refs 参考音、runs.json
└── config.json       端口、阈值、模型目录
```

## 本机部署记录（2026-10-16）

按两个部署包的 README 走，Windows 上有五处需要绕，记下来免得重装时再踩。

1. **`install_windows.ps1` 用 Windows PowerShell 5.1 跑会解析失败。** 脚本是无 BOM 的
   UTF-8，中文被当成 GBK 读，报 `Missing expression after ','`。改用 PowerShell 7
   （`pwsh -ExecutionPolicy Bypass -File .\install_windows.ps1`）即可。
2. **`conda install -c conda-forge opencc` 报 `PackagesNotFoundError`。** 而 requirements
   首行是 `--no-binary=opencc`，走 pip 会强制源码编译。改用 uv 装时它直接取了预编译
   wheel，`opencc 1.4.2` 正常。所以安装时把 `--no-binary` 与 `opencc` 那两行摘掉，
   单独 `uv pip install opencc`。
3. **pip 串行下载在这台机器上只有约 240 kB/s**，requirements 那一批估算要四十分钟以上。
   换 uv 后六分半装完（并发下载，USTC 源）。注意 uv 默认并发过高会被镜像限流（429），
   用 `UV_CONCURRENT_DOWNLOADS=4` 压一压。
4. **PyPI 给的 torch 是 CPU 版**（`2.5.1+cpu`，`cuda False`），README 里“PyPI 默认即 cu124”
   对 Windows 不成立。CUDA 轮子要从镜像取，本机用的是
   `https://mirror.sjtu.edu.cn/pytorch-wheels/cu124/`，装 `torch==2.5.1+cu124` 与
   `torchaudio==2.5.1+cu124`，之后 `torch.cuda.is_available()` 为 True。
5. **`onnxruntime-gpu` 的平台标记没被 uv 匹配到**，要手工补 `onnxruntime`。G2PW 注音走 CPU
   完全够用，不必装 GPU 版。

另外 `ffmpeg` 本机未安装，纯 wav 链路不受影响；`nltk_data` 需要手工复制进环境前缀
`sys.prefix\nltk_data`（安装脚本里那一步）。

## 公网暴露（Cloudflare Tunnel）

站点通过一条**独立**隧道暴露到 `https://voice.soyorin.work`，与 agenthub 用的那条
（`agenthub.soyorin.work`）互不影响，停一条不会牵连另一条。

| 项目 | 值 |
|------|-----|
| 隧道名 | `voice-arena` |
| 隧道 ID | `<你的隧道 ID>`（`cloudflared tunnel create` 的输出） |
| 凭据 | `%USERPROFILE%\.cloudflared\<隧道 ID>.json` |
| 公网入口 | `https://voice.soyorin.work` → `http://127.0.0.1:8790` |
| 启动 | `start-tunnel.bat`（前台窗口，Ctrl+C 停） |

隧道只暴露 8790 这一个端口，合成（9880）与辨伪（9890）两个模型服务仍然只在
`127.0.0.1` 上监听，公网碰不到。

### 访问令牌

`config.json` 的 `access.token` 是空的就完全不限制；填了令牌之后，
**非本机 Host** 的请求必须带上它，三种方式任选：

1. 网址后加 `?token=<令牌>`，顺手下发 30 天的 cookie，之后直接开网址即可；
2. 请求头 `X-Voice-Arena-Token: <令牌>`，适合脚本调用；
3. cookie `va_token=<令牌>`。

本机访问（`127.0.0.1` / `localhost`）一律放行，所以调试时不用拼参数。
想彻底关掉这道门，把 `access.token` 置空并重启站点即可。

### 免费版限制与注意事项

- 单请求超时约 100 秒。一次合成一两句话没问题（实测 13 秒出 1 MB 音频）；
  长文本会被 `text_split_method: cut5` 切成多段，累计时间可能顶到上限，演示时建议短句。
- 单次上传上限 100 MB，本项目的音频都在 1 MB 量级，用不到。
- 公网可达后建议在 Cloudflare 侧再加一层 **Zero Trust Access**（给主机名加策略，
  只放行指定邮箱或 GitHub 账号），比共享令牌更稳，也不怕链接外传。
- 隧道跑的是本机 GPU 推理，别把令牌贴在公开场合；真要公开演示，先加 Access。

### 与 agenthub 的关系

两边都放在 `soyorin.work` 下，各自一条隧道、各自一份 `data/cloudflared/config.yml`，
配置文件都在 `.gitignore` 里，不进仓库。要换子域名，改隧道配置里的 `hostname`，
再跑一次 `cloudflared tunnel route dns voice-arena <新域名>` 即可。
