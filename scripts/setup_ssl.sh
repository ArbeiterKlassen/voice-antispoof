#!/usr/bin/env bash
# SSL 前端准备：装 transformers + 下 WavLM-large（反欺骗 SOTA 的标准前端）
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

export HF_ENDPOINT=https://hf-mirror.com
PIP=/home/cuitianxu/.miniconda3/envs/ml2026_exp/bin/pip
TUNA="-i https://pypi.tuna.tsinghua.edu.cn/simple"

echo "[1/3] 装 transformers ..."
$PIP install $TUNA transformers || exit 1

echo "[2/3] 下 WavLM-large (1.2 GB, 走 hf-mirror) ..."
$PY - <<'PYEOF'
import os
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")
from huggingface_hub import snapshot_download
p=snapshot_download("microsoft/wavlm-large",
                    local_dir=os.environ["VOICE_WS"]+"/models/wavlm-large",
                    allow_patterns=["*.bin","*.json","*.txt"])
print("下载完成 ->", p)
import glob,os
for f in sorted(glob.glob(p+"/*")):
    if os.path.isfile(f): print(f"  {os.path.basename(f):<28} {os.path.getsize(f)/1048576:.1f} MB")
PYEOF

echo "[3/3] 冒烟：加载并跑一条音频"
$PY - <<'PYEOF'
import os
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")
import torch, numpy as np
from transformers import WavLMModel
m=WavLMModel.from_pretrained(os.environ["VOICE_WS"]+"/models/wavlm-large")
m.eval()
n=sum(p.numel() for p in m.parameters())
print(f"  WavLM-large 参数量: {n:,}")
x=torch.randn(1,64600)
with torch.no_grad(): o=m(x).last_hidden_state
print(f"  输入 (1,64600) -> 输出 {tuple(o.shape)}  (768 维特征)")
print("SSL_SETUP_DONE")
PYEOF
