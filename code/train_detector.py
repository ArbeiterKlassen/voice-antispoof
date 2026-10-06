#!/usr/bin/env python3
"""
AASIST 辨伪训练 / 评估

复用 clovaai/aasist 的**官方模型架构**（models/AASIST.py）与官方预训练权重，
但训练循环、数据加载、指标计算全部自写，原因见 metrics.py 头注释
（官方 evaluation.py 有两处必崩：缺 ASVspoof2019 ASV 打分文件 + np.float 已移除）。

⚠️ 分数方向：logits 索引 1 = bonafide（越大越像真实）。
   与 labels 1=real/0=fake 配套。已过 metrics.py --selftest 验证。

GPU: 本机只有 GPU2 空闲 -> 运行前 export CUDA_VISIBLE_DEVICES=2

用法:
  python train_detector.py --manifest data/manifests/all_manifest.csv \
      --epochs 30 --batch-size 8 --out-dir results/det_run1
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, "code", "aasist"))
sys.path.insert(0, os.path.join(WS, "code"))

from models.AASIST import Model as AASIST            # noqa: E402
from detector_dataset import ManifestDataset, load_manifest, NB_SAMP  # noqa: E402
from metrics import evaluate                         # noqa: E402
import os

# 官方 config/AASIST.conf 的 model_config，逐项照抄
# ⚠️ 官方类名是 Model（不是 AASIST）；main.py:216 用 getattr(module, "Model")(model_config)
# 官方 main.py:48-49 —— 配置里没有 freq_aug 时默认 "False"
FREQ_AUG = False

D_ARGS = {
    "architecture": "AASIST",
    "nb_samp": NB_SAMP, "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}


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
        for x, y in loader:
            x = x.to(device, non_blocking=True)        # (B, T) —— forward 内部自己 unsqueeze(1)
            y = y.to(device, non_blocking=True).view(-1).long()
            _, out = model(x, Freq_aug=FREQ_AUG)       # 官方返回 (last_hidden, output)
            loss = crit(out, y)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            tot += loss.item() * y.size(0); n += y.size(0)
            # 索引 1 = bonafide
            scores.append(out[:, 1].detach().float().cpu().numpy())
            ys.append(y.detach().cpu().numpy())
    scores = np.concatenate(scores); ys = np.concatenate(ys)
    m = evaluate(scores, ys)
    m["loss"] = round(tot / max(n, 1), 6)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=f"{WS}/data/manifests/all_manifest.csv")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out-dir", default=f"{WS}/results/det_run1")
    ap.add_argument("--init-weights", default=f"{WS}/code/aasist/models/weights/AASIST.pth",
                    help="官方预训练权重；传 '' 则从头训")
    ap.add_argument("--seed", type=int, default=20260918)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logf = open(f"{args.out_dir}/train.log", "w")
    sys.stdout = Tee(sys.__stdout__, logf)

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("=" * 70)
    print(f"[协议] manifest      = {args.manifest}")
    print(f"[协议] out_dir       = {args.out_dir}")
    print(f"[协议] init_weights  = {args.init_weights or '(从头训)'}")
    print(f"[协议] seed          = {args.seed}")
    print(f"[协议] torch         = {torch.__version__}")
    print(f"[协议] cuda_available= {torch.cuda.is_available()}")
    print("=" * 70)

    df = load_manifest(args.manifest)
    tr = df[df["split"] == "train"]
    dv = df[df["split"] == "dev"]
    print(f"[协议] n_fit(train)  = {len(tr)}  (real {int((tr.y==1).sum())} / fake {int((tr.y==0).sum())})")
    print(f"[协议] n_eval(dev)   = {len(dv)}  (real {int((dv.y==1).sum())} / fake {int((dv.y==0).sum())})")
    assert len(tr) > 0 and len(dv) > 0, "train/dev 至少各需非空"
    assert tr.y.nunique() == 2 and dv.y.nunique() == 2, "train/dev 都必须同时含 real 和 fake"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[协议] device        = {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

    ds_tr = ManifestDataset(tr, train=True)
    ds_dv = ManifestDataset(dv, train=False)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dl_dv = DataLoader(ds_dv, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)

    model = AASIST(D_ARGS).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[协议] model_params  = {n_par:,}")

    if args.init_weights and os.path.exists(args.init_weights):
        ck = torch.load(args.init_weights, map_location="cpu", weights_only=False)
        sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[协议] 载入权重: missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) > 3:
            print(f"  ⚠️ missing 较多，前5: {missing[:5]}")

    # 类别加权 CE（训练集类别不平衡时用）
    cnt = tr.y.value_counts().to_dict()
    w = torch.tensor([len(tr) / (2.0 * cnt[0]), len(tr) / (2.0 * cnt[1])], dtype=torch.float32).to(device)
    print(f"[协议] class_weight  = fake {w[0]:.3f} / real {w[1]:.3f}")
    crit = nn.CrossEntropyLoss(weight=w)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=5e-6)

    best = {"eer": 1.0, "epoch": -1}
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        mtr = run_epoch(model, dl_tr, crit, opt, device, train=True)
        mdv = run_epoch(model, dl_dv, crit, opt, device, train=False)
        sched.step()
        print(f"[ep {ep:3d}/{args.epochs}] "
              f"train loss {mtr['loss']:.4f} eer {mtr['eer_percent']:.3f}% | "
              f"dev loss {mdv['loss']:.4f} EER {mdv['eer_percent']:.3f}% "
              f"AUC {mdv['auc']:.4f} acc {mdv['accuracy']:.4f} | "
              f"{(time.time()-t0)/60:.1f} min")
        if mdv["eer"] < best["eer"]:
            best = {"eer": mdv["eer"], "epoch": ep, **mdv}
            torch.save({"model_state_dict": model.state_dict(), "d_args": D_ARGS,
                        "epoch": ep, "dev_metrics": mdv}, f"{args.out_dir}/best.pth")
            print(f"    ↑ 新最优，已保存 best.pth (dev EER {mdv['eer_percent']:.3f}%)")

    torch.save({"model_state_dict": model.state_dict(), "d_args": D_ARGS,
                "epoch": args.epochs}, f"{args.out_dir}/last.pth")

    print("\n" + "=" * 70)
    print(f"训练结束。最优 dev EER = {best['eer']*100:.3f}% @ epoch {best['epoch']}")
    print(f"总耗时 {(time.time()-t0)/60:.1f} min")
    with open(f"{args.out_dir}/best_dev_metrics.json", "w") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)
    print(f"产物: {args.out_dir}/best.pth, train.log, best_dev_metrics.json")
    print("TRAIN_DETECTOR_DONE")


if __name__ == "__main__":
    main()
