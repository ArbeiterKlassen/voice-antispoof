#!/usr/bin/env python3
"""
LibriSpeech dev-clean parquet -> 真实语音 flac + manifest

输入: data/_downloads/dev-clean.parquet  (HF openslr/librispeech_asr 的 clean/validation)
输出:
  data/real/<speaker_id>/<id>.flac
  data/manifests/real_manifest.csv       所有真实语音
  data/manifests/speaker_split.json      说话人划分（train/dev/test 不重叠）

⚠️ 关键设计（见 docs/01 陷阱 2）：
  按【说话人】划分，不按utterance随机划分。
  随机划分会让同一说话人同时出现在 train 和 test，
  检测器可能靠"认说话人"而非"认伪造"过关。

用法: python prepare_real.py [--limit N]
"""
import argparse, io, json, os, sys, csv
import pyarrow.parquet as pq
import soundfile as sf
import numpy as np
import os

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET = f"{WS}/data/_downloads/dev-clean.parquet"
OUT_REAL = f"{WS}/data/real"
OUT_MAN = f"{WS}/data/manifests"
SEED = 20260918
RATIOS = (0.70, 0.15, 0.15)   # train / dev / test  （按说话人）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 条（0=全部）")
    args = ap.parse_args()

    os.makedirs(OUT_REAL, exist_ok=True)
    os.makedirs(OUT_MAN, exist_ok=True)

    t = pq.read_table(PARQUET, columns=["audio", "text", "speaker_id", "id"])
    df = t.to_pandas()
    if args.limit:
        df = df.head(args.limit)
    print(f"[协议] parquet 条数 = {len(df)}")

    # ---- 1) 说话人划分（先定划分，再落盘，保证可复现） ----
    speakers = sorted(df["speaker_id"].unique().tolist())
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(speakers))
    n_tr = int(round(len(speakers) * RATIOS[0]))
    n_dv = int(round(len(speakers) * RATIOS[1]))
    spk_train = sorted(np.array(speakers)[perm[:n_tr]].tolist())
    spk_dev = sorted(np.array(speakers)[perm[n_tr:n_tr + n_dv]].tolist())
    spk_test = sorted(np.array(speakers)[perm[n_tr + n_dv:]].tolist())
    split_of = {s: "train" for s in spk_train}
    split_of.update({s: "dev" for s in spk_dev})
    split_of.update({s: "test" for s in spk_test})

    split_meta = {
        "seed": SEED, "ratios": RATIOS, "unit": "speaker",
        "n_speakers": len(speakers),
        "train_speakers": spk_train, "dev_speakers": spk_dev, "test_speakers": spk_test,
        "note": "speaker-disjoint：三个集合的说话人无交集",
    }
    with open(f"{OUT_MAN}/speaker_split.json", "w") as f:
        json.dump(split_meta, f, ensure_ascii=False, indent=2)

    # ---- 2) 落盘 flac + 收 duration ----
    rows, n_bad = [], 0
    for i, r in enumerate(df.itertuples(index=False)):
        spk = int(r.speaker_id)
        uid = r.id
        raw = r.audio["bytes"]
        try:
            info = sf.info(io.BytesIO(raw))
            dur, sr = info.duration, info.samplerate
        except Exception as e:
            n_bad += 1
            print(f"  ⚠️ 读取失败 {uid}: {e}")
            continue
        d = f"{OUT_REAL}/{spk}"
        os.makedirs(d, exist_ok=True)
        p = f"{d}/{uid}.flac"
        if not os.path.exists(p):
            with open(p, "wb") as f:
                f.write(raw)
        rows.append({
            "audio_path": os.path.relpath(p, WS),
            "label": "real", "source": "LibriSpeech", "clone_model": "none",
            "attack_type": "none", "text": r.text, "speaker_id": spk,
            "duration": round(dur, 3), "sample_rate": sr,
            "split": split_of[spk], "utt_id": uid,
        })
        if (i + 1) % 500 == 0:
            print(f"  ... {i+1}/{len(df)}")

    cols = ["audio_path", "label", "source", "clone_model", "attack_type",
            "text", "speaker_id", "duration", "sample_rate", "split", "utt_id"]
    with open(f"{OUT_MAN}/real_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # ---- 3) 汇报（带协议字段，便于复算） ----
    import collections
    c = collections.Counter(x["split"] for x in rows)
    durs = np.array([x["duration"] for x in rows])
    print("\n===== 汇总 =====")
    print(f"落盘真实语音 : {len(rows)} 条 (失败 {n_bad})")
    print(f"说话人总数   : {len(speakers)}  -> train {len(spk_train)} / dev {len(spk_dev)} / test {len(spk_test)}")
    print(f"按条数       : train {c['train']} / dev {c['dev']} / test {c['test']}")
    print(f"时长         : 总 {durs.sum()/3600:.2f} h, 中位 {np.median(durs):.2f}s, "
          f"min {durs.min():.2f}s, max {durs.max():.2f}s")
    # 划分健全性：speaker 无交集
    assert not (set(spk_train) & set(spk_dev)), "train/dev 说话人有交集!"
    assert not (set(spk_train) & set(spk_test)), "train/test 说话人有交集!"
    assert not (set(spk_dev) & set(spk_test)), "dev/test 说话人有交集!"
    print("✅ speaker-disjoint 校验通过（三集合说话人无交集）")
    print(f"manifest: {OUT_MAN}/real_manifest.csv")
    print(f"split   : {OUT_MAN}/speaker_split.json")
    print("PREPARE_REAL_DONE")


if __name__ == "__main__":
    main()
