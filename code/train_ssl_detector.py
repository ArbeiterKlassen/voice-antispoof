#!/usr/bin/env python3
"""
SSL-AASIST 训练（冻结 WavLM 特征 + 可训练适配器 + 复用官方 GAT 后端）

与 train_detector.py 同口径：同一套 metrics、同样的协议字段、同样的报数纪律，
便于和 AASIST 基线逐项对比（改进必须可归因，不能"感觉变好了"）。

GPU: 用 GPU6（2026-10-04 实测空闲）
用法:
  CUDA_VISIBLE_DEVICES=6 python train_ssl_detector.py \
     --manifest data/manifests/aug_train_manifest.csv --epochs 30 --out-dir results/ssl_run1
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, "code", "aasist"))
sys.path.insert(0, os.path.join(WS, "code"))

from detector_dataset import load_manifest              # noqa: E402
from dataset_ssl import SSLFeatureDataset                # noqa: E402
from ssl_aasist import SSLAASIST                         # noqa: E402
from metrics import evaluate                             # noqa: E402
import os


class Tee:
    def __init__(self, *fhs): self.fhs = fhs
    def write(self, s):
        for f in self.fhs: f.write(s); f.flush()
    def flush(self):
        for f in self.fhs: f.flush()


def run_epoch(model, loader, crit, opt, device, train=True):
    model.train() if train else model.eval()
    tot, n = 0.0, 0
    scores, ys = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for h, y in loader:
            h = h.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(-1).long()
            _, out = model(h)
            loss = crit(out, y)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            tot += loss.item() * y.size(0); n += y.size(0)
            scores.append(out[:, 1].detach().float().cpu().numpy())   # 索引1=bonafide
            ys.append(y.detach().cpu().numpy())
    m = evaluate(np.concatenate(scores), np.concatenate(ys))
    m["loss"] = round(tot / max(n, 1), 6)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=f"{WS}/data/manifests/data_manifest_15col.csv")
    ap.add_argument("--cache-dir", default=f"{WS}/data/ssl_cache/wavlm-large")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=20261004)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logf = open(f"{args.out_dir}/train.log", "w")
    sys.stdout = Tee(sys.__stdout__, logf)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print(f"[协议] manifest    = {args.manifest}")
    print(f"[协议] cache_dir   = {args.cache_dir}")
    print(f"[协议] out_dir     = {args.out_dir}")
    print(f"[协议] device      = {dev}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    print(f"[协议] epochs/bs/lr= {args.epochs}/{args.batch_size}/{args.lr}")
    print(f"[协议] seed        = {args.seed}")
    print("=" * 72)

    df = load_manifest(args.manifest)
    tr = df[df["split"] == "train"]; dv = df[df["split"] == "dev"]
    assert len(tr) and len(dv), "train/dev 不能为空"
    assert tr.y.nunique() == 2 and dv.y.nunique() == 2, "train/dev 都必须含 real+fake"
    print(f"[协议] n_fit(train) = {len(tr)}  (real {int((tr.y==1).sum())} / fake {int((tr.y==0).sum())})")
    print(f"[协议] n_eval(dev)  = {len(dv)}  (real {int((dv.y==1).sum())} / fake {int((dv.y==0).sum())})")

    ds_tr = SSLFeatureDataset(tr, args.cache_dir)
    ds_dv = SSLFeatureDataset(dv, args.cache_dir)
    print(f"[协议] 有缓存可用   = train {len(ds_tr)} / dev {len(ds_dv)}")
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, drop_last=True)
    dl_dv = DataLoader(ds_dv, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers)

    model = SSLAASIST(verbose=True).to(dev)
    n_new = sum(p.numel() for n, p in model.named_parameters()
                if n.split(".")[0] in ("proj_f", "bn_f", "expand_c", "bn_c"))
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[协议] 参数量      = 总 {n_all:,} (新增适配器 {n_new:,} / 官方后端 {n_all-n_new:,})")
    print(f"[协议] 前端        = WavLM-large 冻结（特征已缓存，训练不经过前端）")

    cnt = tr.y.value_counts().to_dict()
    w = torch.tensor([len(tr) / (2.0 * cnt[0]), len(tr) / (2.0 * cnt[1])], dtype=torch.float32).to(dev)
    print(f"[协议] class_weight= fake {w[0]:.3f} / real {w[1]:.3f}")
    crit = nn.CrossEntropyLoss(weight=w)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    best = {"eer": 1.0, "epoch": -1}
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        mtr = run_epoch(model, dl_tr, crit, opt, dev, train=True)
        mdv = run_epoch(model, dl_dv, crit, opt, dev, train=False)
        sched.step()
        print(f"[ep {ep:3d}/{args.epochs}] train loss {mtr['loss']:.4f} eer {mtr['eer_percent']:.3f}% | "
              f"dev loss {mdv['loss']:.4f} EER {mdv['eer_percent']:.3f}% AUC {mdv['auc']:.4f} "
              f"pAUC@1% {mdv['pauc_fpr1pct']:.4f} | {(time.time()-t0)/60:.1f} min")
        if mdv["eer"] < best["eer"]:
            best = {"eer": mdv["eer"], "epoch": ep, **mdv}
            torch.save({"model_state_dict": model.state_dict(), "epoch": ep,
                        "dev_metrics": mdv, "arch": "ssl_aasist"}, f"{args.out_dir}/best.pth")
            print(f"    ↑ 新最优，已存 best.pth (dev EER {mdv['eer_percent']:.3f}%)")

    torch.save({"model_state_dict": model.state_dict(), "epoch": args.epochs,
                "arch": "ssl_aasist"}, f"{args.out_dir}/last.pth")
    print("\n" + "=" * 72)
    print(f"训练结束。最优 dev EER = {best['eer']*100:.3f}% @ epoch {best['epoch']}")
    print(f"总耗时 {(time.time()-t0)/60:.1f} min")
    with open(f"{args.out_dir}/best_dev_metrics.json", "w") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)
    print("TRAIN_SSL_DONE")


if __name__ == "__main__":
    main()
