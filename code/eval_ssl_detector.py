#!/usr/bin/env python3
"""
SSL-AASIST 评估 —— 与 eval_detector.py **同口径**（同 metrics、同协议字段），
便于和波形 AASIST 逐项对比。

用法:
  CUDA_VISIBLE_DEVICES=6 python eval_ssl_detector.py \
     --ckpt results/ssl_base/best.pth --split test --by attack_type
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, "code", "aasist"))
sys.path.insert(0, os.path.join(WS, "code"))

from detector_dataset import load_manifest              # noqa: E402
from dataset_ssl import SSLFeatureDataset                # noqa: E402
from ssl_aasist import SSLAASIST                         # noqa: E402
from metrics import evaluate, bootstrap_ci_eer, evaluate_at_threshold  # noqa: E402
import os


@torch.no_grad()
def score_all(model, dl, device):
    model.eval()
    sc, ys = [], []
    for h, y in dl:
        h = h.to(device, non_blocking=True)
        _, out = model(h)
        sc.append(out[:, 1].float().cpu().numpy())      # 索引 1 = bonafide
        ys.append(y.numpy())
    return np.concatenate(sc), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default=f"{WS}/data/manifests/data_manifest_15col.csv")
    ap.add_argument("--cache-dir", default=f"{WS}/data/ssl_cache/wavlm-large")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--by", default="")
    ap.add_argument("--frozen-threshold", default=None, type=float)
    ap.add_argument("--n-boot", default=1000, type=int)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = SSLAASIST().to(dev)
    miss, unexp = model.load_state_dict(ck.get("model_state_dict", ck), strict=False)
    print(f"[协议] ckpt       = {args.ckpt} (epoch {ck.get('epoch','?')}, arch={ck.get('arch','?')})")
    print(f"[协议] load_state missing={len(miss)} unexpected={len(unexp)}")
    print(f"[协议] device     = {dev}")
    print(f"[协议] manifest   = {args.manifest}")
    print(f"[协议] HELDOUT    = split={args.split}")

    df = load_manifest(args.manifest)
    df = df[df["split"] == args.split].reset_index(drop=True)
    ds = SSLFeatureDataset(df, args.cache_dir)
    df = ds.df                                    # 只保留有缓存的
    assert len(df) > 0, "无可用缓存"
    print(f"[协议] n_eval     = {len(df)}  (real {int((df.y==1).sum())} / fake {int((df.y==0).sum())})")

    frz = float(args.frozen_threshold) if args.frozen_threshold else None

    def ev(sub):
        d = SSLFeatureDataset(sub, args.cache_dir)
        dl = DataLoader(d, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
        s, y = score_all(model, dl, dev)
        m = evaluate(s, y)
        m["ci_eer"] = bootstrap_ci_eer(s, y, d.df["speaker_id"].values, n_boot=args.n_boot)
        if frz is not None:
            m["at_frozen_threshold"] = evaluate_at_threshold(s, y, frz)
        return m

    overall = ev(df)
    ci = overall["ci_eer"]
    print(f"\n===== SSL-AASIST ({args.split}) =====")
    if "ci_lo_percent" in ci:
        print(f"  EER {overall['eer_percent']:.3f}%  [95% CI {ci['ci_lo_percent']:.3f}%, "
              f"{ci['ci_hi_percent']:.3f}%] ({ci.get('n_speakers','?')} speakers)")
    else:
        # bootstrap_ci_eer 在单类别/单说话人时返回降级字典（无 ci_* 字段）
        print(f"  EER {overall['eer_percent']:.3f}%  [CI 不可算: {ci.get('note','单类别或缺说话人')}]")
    print(f"  AUC {overall['auc']:.4f} | pAUC@1% {overall['pauc_fpr1pct']:.4f} | "
          f"real {overall['real_count']} fake {overall['fake_count']}")
    if frz is not None:
        ft = overall["at_frozen_threshold"]
        print(f"  冻结阈值 {frz:.6f} -> 真实误报率 {ft['fpr_real_misjudged_fake']:.4f}")

    out = {"protocol": "ssl", "split": args.split, "ckpt": args.ckpt,
           "overall": overall, "frozen_threshold": frz, "groups": {}}

    if args.by and args.by in df.columns:
        reals = df[df.y == 1]; fakes = df[df.y == 0]
        print(f"\n===== 按 {args.by} 分组（每类伪造 vs 全部真实 {len(reals)}）=====")
        print(f"  {'组':<20} {'n_fake':>7} {'EER%':>8} {'95% CI':>18} {'AUC':>8}")
        for v, sub_f in fakes.groupby(args.by):
            m = ev(pd.concat([reals, sub_f], ignore_index=True))
            out["groups"][str(v)] = m
            c = m["ci_eer"]
            ci_s = (f"[{c['ci_lo_percent']:.3f},{c['ci_hi_percent']:.3f}]"
                    if "ci_lo_percent" in c else "n/a")
            print(f"  {str(v):<20} {m['fake_count']:>7} {m['eer_percent']:>8.3f} "
                  f"{ci_s:>18} {m['auc']:>8.4f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n结果已写入 {args.out}")
    print("EVAL_SSL_DONE")


if __name__ == "__main__":
    main()
