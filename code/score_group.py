#!/usr/bin/env python3
"""
打分一次 + 分组评估 —— 与 eval_detector.py **数值等价**的快路径。

## 为什么重写
eval_detector.py 在 --by 分组时对**每一组**重新建 DataLoader 重新打分
（15 组 => 全部真实被重复前向 15 次，单次 P4 评估 14400 次前向）。
模型在 eval 模式下无随机性（无 dropout / 无 pad_random），
同一文件的分数与分组无关 —— 所以打一次分再按索引分组，结果逐位相同。

## 等价性怎么保证（不是"我觉得等价"）
1. 完全相同的数据管线：ManifestDataset(sub, train=False)
2. 完全相同的指标函数：metrics.evaluate / bootstrap_ci_eer / evaluate_at_threshold
3. **完全相同的行顺序**：分组的子集按 pd.concat([reals, sub_f]) 的顺序拼数组，
   使 bootstrap 的输入序列与 eval_detector.py 一致
4. 落盘前用臂A已提交的 p4_correct.json 做已知答案对拍（见 scripts/verify_equiv.sh）

## 额外产出
--save-scores：把 (utt_id, score, y, speaker_id) 存 npz。
后续要做的「两臂分数相关性 / 集成」不必再前向一遍。

用法:
  CUDA_VISIBLE_DEVICES=4 python score_group.py --ckpt X.pth \
     --manifest data/manifests/p4_correct_manifest.csv --split test \
     --by attack_type --frozen-threshold 1.782691 --out results/x/p4.json
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, "code", "aasist"))
sys.path.insert(0, os.path.join(WS, "code"))

from models.AASIST import Model as AASIST                      # noqa: E402
from detector_dataset import ManifestDataset, load_manifest     # noqa: E402
from metrics import evaluate, bootstrap_ci_eer, evaluate_at_threshold  # noqa: E402
from train_detector import D_ARGS, FREQ_AUG                     # noqa: E402
import os


def build_model(d_args):
    """按 d_args['architecture'] 分派 —— 官方 main.py:214 同法
    （import_module("models.<arch>") 再 .Model(d_args)）。
    目前支持 AASIST / AASIST-L / RawNet2Spoof，三者的 forward 都是 (last_hidden, output)。"""
    arch = d_args.get("architecture", "AASIST")
    if arch == "AASIST":
        return AASIST(d_args)
    import importlib
    return importlib.import_module(f"models.{arch}").Model(d_args)


@torch.no_grad()
def score_all(model, dl, device):
    model.eval()
    sc, ys = [], []
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        _, out = model(x, Freq_aug=FREQ_AUG)
        sc.append(out[:, 1].float().cpu().numpy())      # 索引 1 = bonafide
        ys.append(y.numpy())
    return np.concatenate(sc), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--by", default="")
    ap.add_argument("--frozen-threshold", default=None, type=float)
    ap.add_argument("--n-boot", default=1000, type=int)
    ap.add_argument("--num-threads", default=8, type=int)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model-config", default="",
                    help="提供 d_args 的 json/conf 路径（取其中的 model_config 段）。"
                         "⚠️ 换了架构必须传：官方权重是**裸 state_dict**、不含 d_args，"
                         "不传就会用默认 AASIST 的 d_args 建模，形状不符的键被 strict=False 静默跳过")
    ap.add_argument("--allow-partial-load", action="store_true",
                    help="允许 missing/unexpected 非零。默认**严格**：非零即报错退出，防静默错配")
    ap.add_argument("--save-scores", default="", help="存 npz: utt_id/score/y/speaker")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.set_num_threads(args.num_threads)
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available())
                          else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    d_args = ck.get("d_args", D_ARGS)
    if args.model_config:                       # 换架构时必须显式给（官方权重不含 d_args）
        import json as _json
        cfg = _json.load(open(args.model_config))
        d_args = cfg.get("model_config", cfg)
        print(f"[协议] model_config = {args.model_config}（架构 {d_args.get('architecture')}，"
              f"gat_dims {d_args.get('gat_dims')}）")
    model = build_model(d_args).to(device)
    miss, unexp = model.load_state_dict(ck.get("model_state_dict", ck), strict=False)
    print(f"[协议] ckpt       = {args.ckpt} (epoch {ck.get('epoch','?')})")
    print(f"[协议] load_state missing={len(miss)} unexpected={len(unexp)}")
    if (miss or unexp) and not args.allow_partial_load:
        # 防呆：形状不符的键会被 strict=False 静默跳过 → 结果是垃圾但看起来跑通了
        raise SystemExit(f"❌ 权重与模型不匹配（missing={len(miss)} unexpected={len(unexp)}）。"
                         f"若确认要部分加载，加 --allow-partial-load。"
                         f"\n   missing 例: {list(miss)[:5]}\n   unexpected 例: {list(unexp)[:5]}")
    print(f"[协议] device     = {device}")
    print(f"[协议] manifest   = {args.manifest}  split={args.split}")

    df = load_manifest(args.manifest)
    df = df[df["split"] == args.split].reset_index(drop=True)
    assert len(df) > 0, f"split={args.split} 为空"
    print(f"[协议] n_eval     = {len(df)}  (real {int((df.y==1).sum())} / fake {int((df.y==0).sum())})")

    frz = float(args.frozen_threshold) if args.frozen_threshold else None

    # ---------- 只打一次分 ----------
    ds = ManifestDataset(df, train=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)
    S, Y = score_all(model, dl, device)
    assert len(S) == len(df), f"打分条数 {len(S)} != manifest {len(df)}"
    assert np.array_equal(Y, df["y"].values), "打分顺序与 manifest 不一致（会导致静默错位）"
    SPK = df["speaker_id"].values

    def ev(idx):
        """idx: 行号数组（保持调用方给的顺序）"""
        s, y, sp = S[idx], Y[idx], SPK[idx]
        m = evaluate(s, y)
        m["ci_eer"] = bootstrap_ci_eer(s, y, sp, n_boot=args.n_boot)
        if frz is not None:
            m["at_frozen_threshold"] = evaluate_at_threshold(s, y, frz)
        return m

    all_idx = np.arange(len(df))
    overall = ev(all_idx)
    ci = overall["ci_eer"]
    print(f"\n===== 域内（{args.split}）=====")
    print(f"  EER {overall['eer_percent']:.3f}%  "
          f"[95% CI {ci['ci_lo_percent']:.3f}%, {ci['ci_hi_percent']:.3f}%] "
          f"({ci.get('n_speakers','?')} speakers)")
    print(f"  AUC {overall['auc']:.4f} | acc {overall['accuracy']:.4f} | "
          f"real {overall['real_count']} fake {overall['fake_count']}")
    if frz is not None:
        ft = overall["at_frozen_threshold"]
        print(f"  冻结阈值 {frz:.6f} -> 真实误报率 {ft['fpr_real_misjudged_fake']:.4f} | "
              f"伪造漏报率 {ft['fnr_fake_misjudged_real']:.4f}")

    out = {"protocol": "P1", "split": args.split, "ckpt": args.ckpt,
           "ckpt_epoch": ck.get("epoch"), "overall": overall,
           "frozen_threshold": frz, "groups": {}}

    if args.by and args.by in df.columns:
        # ⚠️ 与 eval_detector.py 同序：pd.concat([reals, sub_f]) => 真实在前
        real_idx = np.flatnonzero(df.y.values == 1)
        fake_pos = np.flatnonzero(df.y.values == 0)
        print(f"\n===== 按 {args.by} 分组（每类伪造 vs 全部真实 {len(real_idx)} 条）=====")
        print(f"  {'组':<20} {'n_fake':>7} {'EER%':>8} {'95% CI':>18} {'AUC':>8}")
        for v, grp in df.iloc[fake_pos].groupby(args.by):
            idx = np.concatenate([real_idx, np.array(grp.index)])
            m = ev(idx)
            out["groups"][str(v)] = m
            c = m["ci_eer"]
            ci_s = (f"[{c['ci_lo_percent']:.3f},{c['ci_hi_percent']:.3f}]"
                    if "ci_lo_percent" in c else "n/a")
            print(f"  {str(v):<20} {m['fake_count']:>7} {m['eer_percent']:>8.3f} "
                  f"{ci_s:>18} {m['auc']:>8.4f}")

    if args.save_scores:
        os.makedirs(os.path.dirname(args.save_scores), exist_ok=True)
        np.savez(args.save_scores, utt_id=df["utt_id"].values.astype(str), score=S,
                 y=Y, speaker_id=SPK.astype(str), attack_type=df["attack_type"].values.astype(str),
                 label=df["label"].values.astype(str))
        print(f"\n分数已存 {args.save_scores}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"结果已写入 {args.out}")
    print("SCORE_GROUP_DONE")


if __name__ == "__main__":
    main()
