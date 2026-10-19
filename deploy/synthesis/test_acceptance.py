#!/usr/bin/env python3
"""
部署自检：向本机 GPT-SoVITS API 发两条请求（单句 / 双句），校验 HTTP 码、字节数、音频时长。
用法：python test_acceptance.py            （需服务已启动并预热）
      python test_acceptance.py --asr      （可选：额外做 ASR 回译核对，需 pip install funasr）
"""
import argparse
import json
import os
import sys
import urllib.request
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
API = "http://127.0.0.1:9880/tts"
REF = os.path.join(HERE, "voices", "ref_user1.wav")
PROMPT_TEXT = open(os.path.join(HERE, "voices", "ref_user1.txt"), encoding="utf-8").read().strip()
OUT = os.path.join(HERE, "out")

CASES = [
    ("acc_short.wav", "今天天气很好我们一起去湖边散步吧"),
    ("acc_multi.wav", "大家好，我是语音克隆演示。今天天气很好，我们一起去湖边散步吧。"),
]


def synth(text, outp):
    body = json.dumps({
        "text": text, "text_lang": "zh",
        "ref_audio_path": REF, "prompt_text": PROMPT_TEXT, "prompt_lang": "zh",
        "streaming_mode": False, "media_type": "wav", "text_split_method": "cut5",
    }).encode()
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
        code = r.status
    with open(outp, "wb") as f:
        f.write(data)
    dur = None
    try:
        with wave.open(outp) as w:
            dur = w.getnframes() / w.getframerate()
    except Exception:
        pass
    return code, len(data), dur


def asr_check(path, want):
    from funasr import AutoModel
    m = AutoModel(model="paraformer-zh", disable_update=True, device="cpu")
    got = m.generate(input=path, batch_size_s=60)[0]["text"].replace(" ", "")
    return got, (got == want.replace(" ", "").replace("，", "").replace("。", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr", action="store_true", help="额外做 ASR 回译核对（需 funasr）")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ok_all = True
    for name, text in CASES:
        p = os.path.join(OUT, name)
        try:
            code, nbytes, dur = synth(text, p)
        except Exception as e:
            print(f"❌ {name}: 请求失败 {e}（服务起了吗？start_api.bat）")
            ok_all = False
            continue
        good = (code == 200 and nbytes > 10000 and dur and dur > 1.0)
        ok_all &= good
        print(f"{'✅' if good else '❌'} {name}: HTTP {code} | {nbytes} bytes | 时长 "
              f"{dur:.2f}s | 文本: {text[:18]}...")
        if args.asr:
            try:
                got, match = asr_check(p, text)
                print(f"   ASR 回译: {'✅ 全对' if match else '⚠️ 不一致'} -> {got}")
                ok_all &= match
            except Exception as e:
                print(f"   ASR 跳过（{e}）")
    print("全部通过 ✓" if ok_all else "存在失败项（对照 README_DEPLOY.md『已知坑』）")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
