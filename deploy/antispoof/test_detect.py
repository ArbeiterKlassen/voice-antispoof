#!/usr/bin/env python3
"""
辨伪服务自检：用包内样例各打一次 /detect，打印分数与判词。

硬判据（两条，返回码据此）：
  real.wav（LA 真人）→ real
  fake.wav（LA 2019 伪造）→ fake
信息项（不参与返回码）：
  clone.wav（GPT-SoVITS v4 克隆，同一说话人）→ 实测**判 real**（逃逸）。
  这是如实展示的已知结果，不是本包故障：AASIST 阈值冻结在 ASVspoof2019 LA-dev 分布上，
  对 2024 代零样本克隆泛化不足（窗口间分数 −1.96…+1.59 波动，仅部分段被抓）。
  演示时按"攻击成功/泛化落差"如实讲——这也是本课题结论的一部分（见主报告 DF EER 16.6% 等）。
用法：python test_detect.py   （需先启动 detect_api；默认本机 9890）
"""
import json
import os
import sys
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
API = os.environ.get("DETECT_URL", "http://127.0.0.1:9890/detect")
HARD = [("samples/real.wav", "real"), ("samples/fake.wav", "fake")]
INFO = [("samples/clone.wav",),
        ("samples/ref_real.wav",)]


def post(path):
    boundary = uuid.uuid4().hex
    with open(path, "rb") as f:
        data = f.read()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"{os.path.basename(path)}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
            ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(API, data=body, headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def run_case(rel, expect=None):
    p = os.path.join(HERE, rel)
    if not os.path.exists(p):
        print(f"⚠️ 缺样例 {rel}")
        return None
    try:
        return post(p)
    except Exception as e:
        print(f"❌ {rel}: 请求失败 {e}（服务起了吗？start_detect_api.bat）")
        return "ERR"


def main():
    ok = True
    print("—— 硬判据（返回码依据）——")
    for rel, expect in HARD:
        d = run_case(rel)
        if d is None:
            ok = False
            continue
        if d == "ERR":
            ok = False
            continue
        got = d.get("aasist_verdict")
        mark = "✅" if got == expect else "❌"
        ok &= mark == "✅"
        print(f"{mark} {rel}: AASIST {d['aasist_score']:+.3f} → {got} "
              f"| AASIST-L {d['aasist_l_score']:+.3f}（期望 {expect}）")
    print("—— 信息项（已知结果，如实展示；不参与返回码）——")
    for (rel,) in INFO:
        d = run_case(rel)
        if not isinstance(d, dict):
            continue
        print(f"ℹ️ {rel}: AASIST {d['aasist_score']:+.3f} → {d['aasist_verdict']} "
              f"| AASIST-L {d['aasist_l_score']:+.3f}")
    print("── 参考事实 ──")
    print("  clone.wav=同一说话人 GPT-SoVITS 克隆，实测判 real（逃逸，段间波动大）")
    print("  ref_real.wav=同一说话人真声，实测判 real —— 两者同判 real 即'泛化落差'现场证据")
    print("自检通过 ✓" if ok else "存在不符项——检查服务日志与音频格式（wav/flac）")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
