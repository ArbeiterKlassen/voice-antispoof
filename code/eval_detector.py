#!/usr/bin/env python3
"""
评估已训练的 AASIST 检测器 —— 支持 P1/P2/P3 预注册协议

预注册协议（见 docs/01）：
  P1 in-domain   ：同生成器（signalproc 三类），不同说话人（test split）
  P2 cross-model ：训练未见过的生成器 —— 用 --only-clone / --exclude-clone 切
  P3 攻击条件    ：按 attack_type 分组报 EER

⚠️ 报数纪律：P1 与 P2 必须成对出现。只报 P1 会被读成"检测器很强"。

用法:
  # P1: test split 全部
  python eval_detector.py --ckpt results/det_signalproc/best.pth --split test

  # P3: 按攻击类型分组
  python eval_detector.py --ckpt ... --split test --by attack_type

  # P2: 留一生成器（例：训练只有 melvoc+pitch，测试看 formant）
  python eval_detector.py --ckpt ... --split test --by attack_type
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, "code", "aasist"))
sys.path.insert(0, os.path.join(WS, "code"))

from models.AASIST import Model as AASIST            # noqa: E402
from detector_dataset import ManifestDataset, load_manifest  # noqa: E402
from metrics import evaluate, bootstrap_ci_eer, evaluate_at_threshold  # noqa: E402
from train_detector import D_ARGS, FREQ_AUG          # noqa: E402
import os


@torch.no_grad()
def score_all(model, dl, device):
    model.eval()
    sc, ys = [], []
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        _, out = model(x, Freq_aug=FREQ_AUG)
        sc.append(out[:, 1].float().cpu().numpy())   # 索引 1 = bonafide
        ys.append(y.numpy())
    return np.concatenate(sc), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default=f"{WS}/data/manifests/all_manifest.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--by", default="", help="按此列分组报 EER（如 attack_type / clone_model）")
    ap.add_argument("--frozen-threshold", default=None, type=float,
                    help="任务02 §3.1 规则1：冻结阈值，跨集沿用。不传则只报该集最优阈值")
    ap.add_argument("--n-boot", default=1000, type=int, help="bootstrap 重采样次数")
    ap.add_argument("--num-threads", default=8, type=int,
                    help="CPU 线程上限。共享机红线：默认为 8。\n"
                         "不设会吃满全部核（实测单进程 58 核 / 本账号 72 核，机器 load 63）")
    ap.add_argument("--out", default="", help="结果 json 路径")
    args = ap.parse_args()

    # 共享机 BLAS 线程上限（memory: shared-box-blas-thread-cap）
    torch.set_num_threads(args.num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    d_args = ck.get("d_args", D_ARGS)
    model = AASIST(d_args).to(device)
    sd = ck.get("model_state_dict", ck)
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[协议] ckpt        = {args.ckpt} (epoch {ck.get('epoch','?')})")
    print(f"[协议] load_state  missing={len(miss)} unexpected={len(unexp)}")
    print(f"[协议] device      = {device}")
    print(f"[协议] manifest    = {args.manifest}")
    print(f"[协议] HELDOUT     = split={args.split}")

    df = load_manifest(args.manifest)
    df = df[df["split"] == args.split].reset_index(drop=True)
    assert len(df) > 0, f"split={args.split} 为空"
    print(f"[协议] n_eval      = {len(df)}  (real {int((df.y==1).sum())} / fake {int((df.y==0).sum())})")

    frz = None
    if args.frozen_threshold:
        frz = float(args.frozen_threshold)

    def ev(sub):
        """返回 (指标字典, scores, labels, speakers) —— 后三者供 CI 用"""
        ds = ManifestDataset(sub, train=False)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
        s, y = score_all(model, dl, device)
        m = evaluate(s, y)
        # 任务 02 §3.1 规则 3：按说话人分层 bootstrap
        m["ci_eer"] = bootstrap_ci_eer(s, y, sub["speaker_id"].values,
                                       n_boot=args.n_boot)
        # 任务 02 §3.1 规则 1：冻结阈值列（跨集沿用，不重取）
        if frz is not None:
            m["at_frozen_threshold"] = evaluate_at_threshold(s, y, frz)
        return m, s, y, sub["speaker_id"].values

    overall, _s, _y, _sp = ev(df)
    ci = overall["ci_eer"]
    print(f"\n===== P1 in-domain ({args.split}) =====")
    print(f"  EER {overall['eer_percent']:.3f}%  "
          f"[95% CI {ci['ci_lo_percent']:.3f}%, {ci['ci_hi_percent']:.3f}%] "
          f"(说话人分层, n_boot={ci['n_boot']}, {ci['n_speakers']} speakers)")
    print(f"  AUC {overall['auc']:.4f} | acc {overall['accuracy']:.4f} | "
          f"real {overall['real_count']} fake {overall['fake_count']}")
    print(f"  该集最优阈值 = {overall['eer_threshold']:.6f}")
    if frz is not None:
        ft = overall["at_frozen_threshold"]
        print(f"  冻结阈值     = {frz:.6f}  ->  真实误报率(FPR) {ft['fpr_real_misjudged_fake']:.4f} | "
              f"伪造漏报率(FNR) {ft['fnr_fake_misjudged_real']:.4f} | acc {ft['accuracy']:.4f}")
        print(f"  ↑ 两列之差本身是结论（任务 02 §3.1 规则 1）")

    out = {"protocol": "P1", "split": args.split, "ckpt": args.ckpt,
           "ckpt_epoch": ck.get("epoch"), "overall": overall,
           "frozen_threshold": frz, "groups": {}}

    # 分组（P2/P3）
    # ⚠️ 不能直接 groupby(col)：real 的该列值是 'none'，与 fake 的取值不同组，
    #    会导致每组只有单一类别。正确做法 = 每类 FAKE 分别 vs 全部 REAL。
    if args.by and args.by in df.columns:
        reals = df[df.y == 1]
        fakes = df[df.y == 0]
        print(f"\n===== 按 {args.by} 分组（每类伪造 vs 全部真实 {len(reals)} 条）=====")
        print(f"  {'组':<20} {'n_fake':>7} {'EER%':>8} {'95% CI':>18} {'AUC':>8} {'acc':>8}")
        for v, sub_f in fakes.groupby(args.by):
            if len(sub_f) == 0:
                continue
            m, _, _, _ = ev(pd.concat([reals, sub_f], ignore_index=True))
            out["groups"][str(v)] = m
            c = m["ci_eer"]
            ci_s = f"[{c['ci_lo_percent']:.3f},{c['ci_hi_percent']:.3f}]"
            print(f"  {str(v):<20} {m['fake_count']:>7} {m['eer_percent']:>8.3f} "
                  f"{ci_s:>18} {m['auc']:>8.4f} {m['accuracy']:>8.4f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n结果已写入 {args.out}")
    print("EVAL_DETECTOR_DONE")


if __name__ == "__main__":
    main()
