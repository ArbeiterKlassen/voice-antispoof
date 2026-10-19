#!/usr/bin/env python3
"""
AASIST 语音辨伪推理服务（与 GPT-SoVITS 合成服务配套的"防"侧）。

POST /detect  multipart 上传音频（wav/flac，任意时长）→ JSON：
  {
    "aasist_score": 2.31, "aasist_verdict": "real",
    "aasist_l_score": 1.87, "aasist_l_verdict": "real",
    "threshold_aasist": -1.09385,
    "note": "分数为 bonafide logit；≥阈值判真。阈值=LA dev EER 交点（本包口径）。"
  }

口径（与主报告一致）：
- 定长 64800... 实为 NB_SAMP=64600（4.04s @16k）：短则 np.tile 填充、长则取前 4.04s（评估用确定性口径）。
- 打分：model(x, Freq_aug=False)，取 logits[:, 1]（bonafide）。
- AASIST 判真阈值 = −1.093850（LA dev EER 交点，冻结）。AASIST-L 仅报分数（其阈值未在本包标定）。
- ⚠️ 已知边界（详见 README_DEPLOY.md）：该阈值在**干净/编解码**条件下可靠；对混响/变速/音高等
  强后处理语音，标定会失效（分数整体下移，误报升高）——这是本课题的核心发现之一，演示时按口径说明。

用法：python detect_api.py [--port 9890] [--host 127.0.0.1]
"""
import argparse
import io
import os
import sys

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "code", "aasist"))
sys.path.insert(0, os.path.join(HERE, "code"))

from models.AASIST import Model as AASIST          # noqa: E402
from detector_dataset import load_wav_16k, pad, NB_SAMP  # noqa: E402

THR_AASIST = -1.093850          # LA dev EER 交点（冻结，与主报告一致）
WEIGHTS = {
    "aasist": os.path.join(HERE, "code", "aasist", "models", "weights", "AASIST.pth"),
    "aasist_l": os.path.join(HERE, "code", "aasist", "models", "weights", "AASIST-L.pth"),
}
CONF_L = os.path.join(HERE, "code", "aasist", "config", "AASIST-L.conf")

# 官方 AASIST 默认结构参数（官 ckpt 是裸 state_dict、无内嵌 d_args；
# 与评估管线 code/train_detector.py 的 D_ARGS 逐字一致）
D_ARGS_AASIST = {
    "architecture": "AASIST",
    "nb_samp": 64600, "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}


def build_model(ckpt, model_config=None):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    d_args = ck.get("d_args") or D_ARGS_AASIST
    if model_config:
        import json as _json
        cfg = _json.load(open(model_config))
        d_args = cfg.get("model_config", cfg)
    model = AASIST(d_args)
    miss, unexp = model.load_state_dict(ck.get("model_state_dict", ck), strict=False)
    assert not miss and not unexp, f"权重不匹配 missing={len(miss)} unexpected={len(unexp)}"
    model.eval()
    return model


@torch.no_grad()
def score(model, x16k):
    dev = next(model.parameters()).device
    x = pad(x16k)[None, :]
    _, out = model(torch.from_numpy(x).to(dev), Freq_aug=False)
    return float(out[0, 1])


app = FastAPI(title="antispoof detect api")
MODELS = {}


@app.on_event("startup")
def _load():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODELS["aasist"] = build_model(WEIGHTS["aasist"]).to(device)
    MODELS["aasist_l"] = build_model(WEIGHTS["aasist_l"], CONF_L).to(device)
    MODELS["_device"] = device
    print(f"[detect] 两个后端已加载 device={device}", flush=True)


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    data = await file.read()
    try:
        x, fs = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
    except Exception as e:
        return JSONResponse({"error": f"音频读取失败（支持 wav/flac）: {e}"}, status_code=400)
    if fs != 16000:
        import librosa
        x = librosa.resample(x, orig_sr=fs, target_sr=16000)
    d = MODELS["_device"]
    s1 = score(MODELS["aasist"].to(d), x.astype(np.float32))
    s2 = score(MODELS["aasist_l"].to(d), x.astype(np.float32))
    return {
        "aasist_score": round(s1, 6),
        "aasist_verdict": "real" if s1 >= THR_AASIST else "fake",
        "aasist_l_score": round(s2, 6),
        "aasist_l_verdict": None,   # AASIST-L 阈值未在本包标定，仅报分
        "threshold_aasist": THR_AASIST,
        "note": "bonafide logit；越大越'真'。阈值=LA dev EER 交点（冻结）。"
                "强后处理（混响/变速/音高）下该阈值会失效——见 README 口径说明。",
    }


@app.get("/health")
def health():
    return {"status": "ok", "device": MODELS.get("_device")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9890)
    args = ap.parse_args()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
