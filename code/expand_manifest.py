#!/usr/bin/env python3
"""
把现有 manifest 扩列到任务 02 §4 要求的 15 列。

字段: utt_id,audio_path,label,source,clone_model,attack_type,attack_params,
      text,text_source,prompt_utt_id,speaker_id,sample_rate,duration,split,sha256

新增列的口径（任务 02 §4）：
  utt_id        全局唯一，用「数据集-划分-序号」
  attack_params 如 mp3-128k / snr15 / rt60-0.4 / speed0.95；无攻击写 na
  text_source   gt（人工标注）或 asr（识别结果）
  prompt_utt_id 克隆所用参考语句编号；真实语音写 na
  sha256        音频文件哈希

用法: python expand_manifest.py --in <manifest.csv> --out <manifest_15col.csv>
"""
import argparse, csv, hashlib, os, sys
import pandas as pd
import os

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = ["utt_id", "audio_path", "label", "source", "clone_model", "attack_type",
        "attack_params", "text", "text_source", "prompt_utt_id", "speaker_id",
        "sample_rate", "duration", "split", "sha256"]

# 攻击参数表：与我们实际生成时用的参数一致（如实记录，不填表）
ATTACK_PARAMS = {
    "melvoc":  "griffinlim-nfft512-hop128-nmel80-iter32",
    "pitch":   "pitch+3st",
    "formant": "formant-env-compress-0.88",
    "none":    "na",
}
DATASET_TAG = {"LibriSpeech": "ls", "signalproc": "sp"}


def sha256_of(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-sha", action="store_true", help="跳过 sha256（调试试用）")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    print(f"[载入] {len(df)} 条 <- {args.inp}")

    rows = []
    for i, r in enumerate(df.itertuples(index=False)):
        src = r.source
        tag = DATASET_TAG.get(src, "xx")
        sp = r.split
        setid = "librispeech" if src == "LibriSpeech" else "signalproc"
        rows.append({
            "utt_id": f"{setid}-{sp}-{i:05d}",
            "audio_path": r.audio_path,
            "label": r.label,
            "source": src,
            "clone_model": r.clone_model,
            "attack_type": r.attack_type,
            "attack_params": ATTACK_PARAMS.get(r.attack_type, "na"),
            "text": r.text,
            "text_source": "gt",           # LibriSpeech 文本是人工标注
            "prompt_utt_id": "na",         # 本轮伪造由信号处理变换得到，非克隆
            "speaker_id": r.speaker_id,
            "sample_rate": r.sample_rate,
            "duration": r.duration,
            "split": sp,
            "sha256": "" if args.skip_sha else sha256_of(
                os.path.join(WS, r.audio_path) if not os.path.isabs(r.audio_path) else r.audio_path),
        })
        if (i + 1) % 1000 == 0:
            print(f"  ... {i+1}/{len(df)}")

    out = pd.DataFrame(rows, columns=COLS)

    # 校验
    assert out.utt_id.is_unique, "❌ utt_id 不唯一"
    assert out.attack_params.notna().all() and (out.attack_params != "").all(), "❌ attack_params 有空值（要求写 na）"
    assert out.text_source.isin(["gt", "asr"]).all(), "❌ text_source 只能是 gt/asr"
    assert out.prompt_utt_id.notna().all(), "❌ prompt_utt_id 有空值"
    if not args.skip_sha:
        assert (out.sha256.str.len() == 64).all(), "❌ sha256 长度异常"
    # 无攻击时 attack_params 必须恰为 na
    bad = out[(out.attack_type == "none") & (out.attack_params != "na")]
    assert len(bad) == 0, f"❌ {len(bad)} 行 attack_type=none 但 attack_params 不是 na"
    print("✅ 列与取值校验通过")

    out.to_csv(args.out, index=False)
    print(f"-> {args.out}  ({len(out)} 行 × {len(out.columns)} 列)")
    print("字段:", ",".join(out.columns))
    print("EXPAND_MANIFEST_DONE")


if __name__ == "__main__":
    main()
