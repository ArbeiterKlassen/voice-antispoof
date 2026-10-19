# GPT-SoVITS v4 推理环境安装（Windows / Miniconda）
# 用法：powershell -ExecutionPolicy Bypass -File .\install_windows.ps1
# 说明：逐条命令都可手动执行（注释里给了等价命令）。开发侧在 Linux 上等价验证过全流程。

$ErrorActionPreference = "Stop"
$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
$REPO = Join-Path $HERE "GPT-SoVITS"
$PIP_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"

Write-Host "=== [1/6] 检查 conda ===" -ForegroundColor Cyan
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "找不到 conda：请先安装 Miniconda 后再运行本脚本。" -ForegroundColor Red
    Write-Host "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
}

Write-Host "=== [2/6] 创建环境 gptsovits (python 3.10) ===" -ForegroundColor Cyan
# conda create -n gptsovits python=3.10 -y
conda create -n gptsovits python=3.10 -y
if ($LASTEXITCODE -ne 0) { Write-Host "conda create 失败（若环境已存在可忽略后继续）" -ForegroundColor Yellow }

Write-Host "=== [3/6] torch 2.5.1 + torchaudio（清华源；PyPI 默认 wheel 即 cu124，4070 直接可用）===" -ForegroundColor Cyan
# conda run -n gptsovits pip install torch==2.5.1 torchaudio==2.5.1 -i $PIP_MIRROR
conda run -n gptsovits pip install torch==2.5.1 torchaudio==2.5.1 -i $PIP_MIRROR
if ($LASTEXITCODE -ne 0) { Write-Host "torch 安装失败" -ForegroundColor Red; exit 1 }

Write-Host "=== [4/6] opencc（必须先装：绕开 requirements 的 --no-binary=opencc 源码编译）===" -ForegroundColor Cyan
# conda install -n gptsovits -c conda-forge opencc -y
conda install -n gptsovits -c conda-forge opencc -y

Write-Host "=== [5/6] requirements.txt + resampy ===" -ForegroundColor Cyan
# conda run -n gptsovits pip install -r GPT-SoVITS\requirements.txt -i $PIP_MIRROR
conda run -n gptsovits pip install -r (Join-Path $REPO "requirements.txt") -i $PIP_MIRROR
if ($LASTEXITCODE -ne 0) {
    Write-Host "requirements 安装有失败项——常见原因见 README_DEPLOY.md『已知坑』。" -ForegroundColor Yellow
}
# resampy：requirements 漏项，必须补（否则合成时报 No module named 'resampy'）
conda run -n gptsovits pip install resampy -i $PIP_MIRROR

# nltk_data 随包自带（20MB，文本前端用），复制进环境前缀，避免联网下载
$NLTK_SRC = Join-Path $HERE "nltk_data"
if (Test-Path $NLTK_SRC) {
    conda run -n gptsovits python -c "import sys, shutil, os; d=os.path.join(sys.prefix,'nltk_data'); shutil.copytree(r'$NLTK_SRC', d, dirs_exist_ok=True); print('nltk_data ->', d)"
} else {
    Write-Host "包内未见 nltk_data（可忽略，首次运行会自动尝试下载）" -ForegroundColor Yellow
}

Write-Host "=== [6/6] ffmpeg 检查（非 wav 输入/输出才需要）===" -ForegroundColor Cyan
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "未找到 ffmpeg。纯 wav 流程可暂缺；建议： winget install Gyan.FFmpeg" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== 安装完成，自检 ===" -ForegroundColor Green
conda run -n gptsovits python -c "import torch, torchaudio, resampy; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"
Write-Host "下一步：双击 start_api.bat 启动服务（会自动预热），再运行 python test_acceptance.py 自检。" -ForegroundColor Green
