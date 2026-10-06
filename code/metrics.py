#!/usr/bin/env python3
"""
辨伪指标：EER / AUC / 固定阈值 Accuracy

⚠️ 分数方向约定（已对源码核实，勿凭 README 猜）：
    score 越大 = 越像 bonafide(真实)
    依据: clovaai/aasist data_utils.py:22  d_meta[key] = 1 if label=="bonafide" else 0
          clovaai/aasist main.py:307     batch_score = batch_out[:, 1]
    → 索引 1 = bonafide。方向搞反 EER 会从 ~1% 变成 ~99%。

⚠️ 为什么不用仓库自带的 evaluation.py（已核实两处必崩）：
    1) evaluation.py:34  np.genfromtxt(asv_score_file) 无条件执行，
       该文件只随 ASVspoof2019 官方数据集发布，自建数据必 FileNotFoundError
    2) evaluation.py:37,44  astype(np.float) —— np.float 在 NumPy>=1.24 已移除
    因此这里自实现，只保留 EER/AUC，不报 min-tDCF（没有 ASV 分数算不出）

自检: python metrics.py --selftest
"""
import numpy as np


def compute_det_curve(bonafide_scores, spoof_scores):
    """返回 (fpr, fnr, thresholds)。沿用 clovaai/aasist evaluation.py 的算法。"""
    scores = np.hstack((bonafide_scores, spoof_scores))
    labels = np.hstack((np.ones(bonafide_scores.shape[0]),
                        np.zeros(spoof_scores.shape[0])))
    n_scores = scores.shape[0]
    n_bonafide = bonafide_scores.shape[0]
    n_spoof = spoof_scores.shape[0]
    desc_score_indices = np.argsort(scores, kind="mergesort")[::-1]
    scores = scores[desc_score_indices]
    labels = labels[desc_score_indices]

    # 相邻不同分数之间取阈值
    distinct_value_indices = np.where(np.diff(scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, n_scores - 1]

    tps = np.cumsum(labels)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    # 末端补 (1,1)
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = scores[threshold_idxs]
    thresholds = np.r_[thresholds[0] + 1, thresholds]

    fpr = fps / n_spoof if n_spoof > 0 else np.zeros_like(fps)
    fnr = 1 - tps / n_bonafide if n_bonafide > 0 else np.zeros_like(tps)
    return np.array(fpr), np.array(fnr), np.array(thresholds)


def compute_eer(bonafide_scores, spoof_scores):
    """EER + 对应阈值。返回 (eer, threshold)。eer 为比例，如 0.0083 = 0.83%"""
    fpr, fnr, thresholds = compute_det_curve(bonafide_scores, spoof_scores)
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = fpr[idx]
    thr = thresholds[idx]
    return float(eer), float(thr)


def compute_auc(bonafide_scores, spoof_scores):
    """ROC-AUC（Mann-Whitney U 形式，处理并列值）"""
    b = np.asarray(bonafide_scores, dtype=float)
    s = np.asarray(spoof_scores, dtype=float)
    n_b, n_s = len(b), len(s)
    if n_b == 0 or n_s == 0:
        return float("nan")
    alls = np.concatenate([b, s])
    order = alls.argsort()
    ranks = np.empty(len(alls), dtype=float)
    ranks[order] = np.arange(1, len(alls) + 1, dtype=float)
    # 并列取平均秩
    _, inv, cnt = np.unique(alls, return_inverse=True, return_counts=True)
    for k in np.where(cnt > 1)[0]:
        m = inv == k
        ranks[m] = ranks[m].mean()
    r_b = ranks[:n_b].sum()
    auc = (r_b - n_b * (n_b + 1) / 2.0) / (n_b * n_s)
    return float(auc)


def compute_accuracy_at_thr(bonafide_scores, spoof_scores, thr):
    """固定阈值下的准确率：score >= thr 判为 bonafide"""
    b = np.asarray(bonafide_scores); s = np.asarray(spoof_scores)
    correct = (b >= thr).sum() + (s < thr).sum()
    n = len(b) + len(s)
    return float(correct / n) if n else float("nan")


def evaluate(scores, labels, label_bonafide=1):
    """
    scores: 越大越像真实;  labels: 1=real/bonafide, 0=fake/spoof
    返回 dict
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    b = scores[labels == label_bonafide]
    s = scores[labels != label_bonafide]
    eer, thr = compute_eer(b, s)
    auc = compute_auc(b, s)
    acc = compute_accuracy_at_thr(b, s, thr)
    return {
        "eer": round(eer, 6),
        "eer_percent": round(eer * 100, 4),
        "auc": round(auc, 6),
        "accuracy": round(acc, 6),
        "eer_threshold": round(thr, 6),
        # 任务 02 §8.1：自建数据上用 pAUC 代替 min-tDCF（自建数据无 ASV 打分）
        "pauc_fpr1pct": round(compute_pauc(b, s, 0.01), 6),
        "pauc_fpr0p1pct": round(compute_pauc(b, s, 0.001), 6),
        "real_count": int(len(b)),
        "fake_count": int(len(s)),
    }


def compute_pauc(bonafide_scores, spoof_scores, max_fpr=0.01):
    """
    部分 AUC：ROC 曲线上 FPR ≤ max_fpr 那段的面积，**归一化到 [0,1]**
    （除以理想面积 max_fpr，故随机分类器给 0.5，完美分类器给 1.0）

    任务 02 §8.1 要求：自建数据上没有 ASV 打分算不了 min-tDCF，
    改用 pAUC@FPR≤1% 与 pAUC@FPR≤0.1% 作为等价信息。

    实现：沿 DET 曲线取点，在 [0, max_fpr] 上对 TPR 做梯形积分，
    并在边界处线性插值（不加插值会因采样点不足而系统性低估）。

    ⚠️ 归一化必须用 **McClish 校正**（sklearn 亦如此）：
        0.5 * (1 + (pauc - min_area) / (max_area - min_area))
        min_area = max_fpr²/2  （随机分类器的面积）
        max_area = max_fpr     （完美分类器的面积）
      这样随机 -> 0.5、完美 -> 1.0。
      **不能**简单除以 max_fpr —— 那样随机分类器会得到 max_fpr/2
      （本机自检实测：FPR≤1% 时给 0.0059 而非 0.5，直接把实现错误暴露出来）。
    """
    b = np.asarray(bonafide_scores, dtype=float)
    s = np.asarray(spoof_scores, dtype=float)
    n_b, n_s = len(b), len(s)
    if n_b == 0 or n_s == 0:
        return float("nan")
    fpr, fnr, _ = compute_det_curve(b, s)
    tpr = 1.0 - fnr
    order = np.argsort(fpr)
    fpr, tpr = fpr[order], tpr[order]

    # 只保留 FPR <= max_fpr 的段，并补上边界插值点
    keep = fpr <= max_fpr
    if not keep.any():
        return 0.0
    f, t = fpr[keep], tpr[keep]
    if f[-1] < max_fpr:
        # 线性插值到 max_fpr（首点前 f=0 处 TPR 取 t[0] 的基线）
        j = np.searchsorted(fpr, max_fpr)
        f0, f1 = fpr[j - 1], fpr[j] if j < len(fpr) else max_fpr
        t0, t1 = tpr[j - 1], tpr[j] if j < len(tpr) else tpr[-1]
        t_at = t0 + (t1 - t0) * (max_fpr - f0) / max(f1 - f0, 1e-12)
        f = np.r_[f, max_fpr]
        t = np.r_[t, t_at]
    # 起点补 (0, 该处 TPR)
    if f[0] > 0:
        f = np.r_[0.0, f]
        t = np.r_[t[0], t]
    area = float(np.trapezoid(t, f))
    # McClish 校正
    min_area = 0.5 * max_fpr ** 2
    max_area = max_fpr
    if max_area <= min_area:
        return float("nan")
    return 0.5 * (1.0 + (area - min_area) / (max_area - min_area))


def evaluate_at_threshold(scores, labels, thr, label_bonafide=1):
    """
    在**冻结阈值**下的指标（任务 02 §3.1 规则 1）。
    score >= thr 判为 bonafide(真实)。

    返回 fpr = 真实被判成伪造的比例（误报率）、fnr = 伪造被判成真实的比例（漏报率）。
    """
    scores = np.asarray(scores, dtype=float); labels = np.asarray(labels)
    is_real = labels == label_bonafide
    pred_real = scores >= thr
    n_real, n_fake = int(is_real.sum()), int((~is_real).sum())
    fp = int((pred_real & ~is_real).sum())   # fake 被判 real
    fn = int((~pred_real & is_real).sum())   # real 被判 fake
    return {
        "threshold": float(thr),
        "fpr_real_misjudged_fake": round(fn / n_real, 6) if n_real else None,
        "fnr_fake_misjudged_real": round(fp / n_fake, 6) if n_fake else None,
        "accuracy": round((n_real + n_fake - fp - fn) / max(n_real + n_fake, 1), 6),
        "real_count": n_real, "fake_count": n_fake,
    }


def bootstrap_ci_eer(scores, labels, speaker_ids, n_boot=1000, alpha=0.05,
                     seed=20260918, label_bonafide=1):
    """
    按**说话人分层**的 bootstrap 95% 置信区间（任务 02 §3.1 规则 3）。

    ⚠️ 为什么按说话人而不是按语句重采样：
       同一说话人的语句高度相关。按语句重采样等于把相关样本当独立样本，
       会**把置信区间压得虚窄**，给出虚假的精度。

    做法：对说话人集合有放回重采样，被抽中的说话人其全部语句一并进入该次重采样。
    """
    scores = np.asarray(scores, dtype=float); labels = np.asarray(labels)
    spk = np.asarray(speaker_ids)
    uniq = np.unique(spk)
    if len(uniq) < 2:
        return {"note": f"说话人只有 {len(uniq)} 个，无法做分层 bootstrap",
                "eer_point": None, "ci_lo": None, "ci_hi": None}

    idx_by_spk = {s: np.where(spk == s)[0] for s in uniq}
    rng = np.random.RandomState(seed)
    eers = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_spk[s] for s in pick])
        sc, lb = scores[idx], labels[idx]
        b, s_ = sc[lb == label_bonafide], sc[lb != label_bonafide]
        if len(b) == 0 or len(s_) == 0:
            continue                       # 该次重采样缺一类，跳过
        eers.append(compute_eer(b, s_)[0])
    if not eers:
        return {"note": "所有重采样都缺类别", "eer_point": None, "ci_lo": None, "ci_hi": None}
    eers = np.array(eers)
    pt = compute_eer(scores[labels == label_bonafide],
                     scores[labels != label_bonafide])[0]
    return {
        "eer_point": round(float(pt), 6),
        "eer_point_percent": round(float(pt) * 100, 4),
        "ci_lo": round(float(np.percentile(eers, 100 * alpha / 2)), 6),
        "ci_hi": round(float(np.percentile(eers, 100 * (1 - alpha / 2))), 6),
        "ci_lo_percent": round(float(np.percentile(eers, 100 * alpha / 2)) * 100, 4),
        "ci_hi_percent": round(float(np.percentile(eers, 100 * (1 - alpha / 2))) * 100, 4),
        "n_boot": int(len(eers)), "n_speakers": int(len(uniq)),
        "stratified_by": "speaker",
    }


def bootstrap_ci_auc(scores, labels, speaker_ids, n_boot=2000, alpha=0.05,
                     seed=20260918, label_bonafide=1):
    """
    说话人分层 bootstrap 95% 置信区间（**AUC 版**，骨架与 bootstrap_ci_eer 全同）。

    ⚠️ AUC 判据（如「CI 上界 < 0.5」）必须用本函数；拿 EER 的 CI 当 AUC 的会串口径。
    """
    scores = np.asarray(scores, dtype=float); labels = np.asarray(labels)
    spk = np.asarray(speaker_ids)
    uniq = np.unique(spk)
    if len(uniq) < 2:
        return {"note": f"说话人只有 {len(uniq)} 个，无法做分层 bootstrap",
                "auc_point": None, "ci_lo": None, "ci_hi": None}

    idx_by_spk = {s: np.where(spk == s)[0] for s in uniq}
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_spk[s] for s in pick])
        sc, lb = scores[idx], labels[idx]
        b, s_ = sc[lb == label_bonafide], sc[lb != label_bonafide]
        if len(b) == 0 or len(s_) == 0:
            continue                       # 该次重采样缺一类，跳过
        aucs.append(compute_auc(b, s_))
    if not aucs:
        return {"note": "所有重采样都缺类别", "auc_point": None, "ci_lo": None, "ci_hi": None}
    aucs = np.array(aucs)
    pt = compute_auc(scores[labels == label_bonafide],
                     scores[labels != label_bonafide])
    return {
        "auc_point": round(float(pt), 6),
        "ci_lo": round(float(np.percentile(aucs, 100 * alpha / 2)), 6),
        "ci_hi": round(float(np.percentile(aucs, 100 * (1 - alpha / 2))), 6),
        "n_boot": int(len(aucs)), "n_speakers": int(len(uniq)),
        "stratified_by": "speaker",
    }


# ----------------------------------------------------------------------
def _selftest():
    """合成已知答案自检 —— 未通过不得用于真实数据"""
    rng = np.random.RandomState(0)
    ok = True

    def check(name, got, want, tol):
        nonlocal ok
        good = abs(got - want) <= tol
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name}: got {got:.6f} want {want:.6f} (tol {tol})")

    print("=== metrics.py 自检 ===")

    # 1) 完全可分 -> EER=0, AUC=1
    b = np.array([0.9, 0.8, 0.85, 0.95]); s = np.array([0.1, 0.2, 0.05, 0.3])
    eer, _ = compute_eer(b, s)
    check("完全可分 EER", eer, 0.0, 1e-9)
    check("完全可分 AUC", compute_auc(b, s), 1.0, 1e-9)

    # 2) 完全反分（方向搞反）-> EER=1
    eer2, _ = compute_eer(s, b)
    check("方向反转 EER", eer2, 1.0, 1e-9)

    # 3) 手工可算的例子
    #    bonafide=[3,2], spoof=[1,0]，阈值扫过去
    #    最常见定义下 EER 为 fpr 与 fnr 的交点
    eer3, thr3 = compute_eer(np.array([3.0, 2.0]), np.array([1.0, 0.0]))
    print(f"  [info] 例3 EER={eer3:.4f} thr={thr3:.4f}")
    assert 0.0 <= eer3 <= 0.5, "例3 EER 应在 [0,0.5]"
    ok &= (0.0 <= eer3 <= 0.5)

    # 4) 大量随机分数 -> EER 接近 0.5
    n = 20000
    b4 = rng.randn(n); s4 = rng.randn(n)
    eer4, _ = compute_eer(b4, s4)
    check("随机分数 EER≈0.5", eer4, 0.5, 0.02)
    check("随机分数 AUC≈0.5", compute_auc(b4, s4), 0.5, 0.02)

    # 5) 已知偏移 -> EER 应显著 < 0.5，且 AUC 明显 > 0.5
    b5 = rng.randn(n) + 0.5; s5 = rng.randn(n)
    eer5, _ = compute_eer(b5, s5)
    auc5 = compute_auc(b5, s5)
    print(f"  [info] 偏移0.5: EER={eer5:.4f} AUC={auc5:.4f}")
    assert eer5 < 0.45 and auc5 > 0.55, "偏移应改善 EER/AUC"
    ok &= (eer5 < 0.45 and auc5 > 0.55)

    # 6) 方向一致性：把 bonafide 整体抬高，EER 必须单调不增
    e_lo, _ = compute_eer(rng.randn(n) + 0.1, rng.randn(n))
    rng2 = np.random.RandomState(1)
    e_hi, _ = compute_eer(rng2.randn(n) + 1.0, rng2.randn(n))
    print(f"  [info] 偏移0.1 EER={e_lo:.4f} vs 偏移1.0 EER={e_hi:.4f}")
    assert e_hi <= e_lo, "偏移越大 EER 应越小（方向约定）"
    ok &= (e_hi <= e_lo)

    # 7) evaluate() 字典字段
    r = evaluate(np.r_[b5, s5], np.r_[np.ones(n), np.zeros(n)])
    for k in ["eer", "auc", "accuracy", "real_count", "fake_count"]:
        assert k in r, f"缺字段 {k}"
    assert r["real_count"] == n and r["fake_count"] == n
    print(f"  [info] evaluate() = {r}")

    # 8) evaluate_at_threshold：已知答案
    #    scores: real=[2,2], fake=[0,0]，阈值取 1 -> 全对
    t = evaluate_at_threshold(np.array([2., 2., 0., 0.]), np.array([1, 1, 0, 0]), 1.0)
    check("阈值1.0 fpr", t["fpr_real_misjudged_fake"], 0.0, 1e-9)
    check("阈值1.0 fnr", t["fnr_fake_misjudged_real"], 0.0, 1e-9)
    check("阈值1.0 acc", t["accuracy"], 1.0, 1e-9)
    #    阈值取 3 -> 全部判 fake：real 全误报(fpr=1)，fake 全对(fnr=0)，acc=0.5
    t2 = evaluate_at_threshold(np.array([2., 2., 0., 0.]), np.array([1, 1, 0, 0]), 3.0)
    check("阈值3.0 fpr", t2["fpr_real_misjudged_fake"], 1.0, 1e-9)
    check("阈值3.0 fnr", t2["fnr_fake_misjudged_real"], 0.0, 1e-9)
    check("阈值3.0 acc", t2["accuracy"], 0.5, 1e-9)

    # 9) bootstrap CI —— 核心性质：**说话人分层必须比语句分层更宽**
    #    ⚠️ 构造要点：说话人随机效应必须落在「real 与 fake 的**间隔**」上，
    #       而不是两类共有的平移 —— 共有平移会在 real-vs-fake 比较里抵消，
    #       造不出说话人级相关（第一版就是这么写错的，比值仅 1.07x）。
    rng3 = np.random.RandomState(7)
    n_spk, per = 20, 50
    sc, lb, sp = [], [], []
    for k in range(n_spk):
        gap = rng3.uniform(0.0, 2.5)          # 该说话人的可分辨程度，因人而异
        for _ in range(per):
            sc.append(rng3.randn() * 0.5)      # real
            lb.append(1); sp.append(k)
        for _ in range(per):
            sc.append(rng3.randn() * 0.5 - gap)  # fake
            lb.append(0); sp.append(k)
    sc, lb, sp = np.array(sc), np.array(lb), np.array(sp)
    ci_spk = bootstrap_ci_eer(sc, lb, sp, n_boot=300, seed=1)
    # 语句分层（错误做法）——用每条语句当独立单元
    ci_utt = bootstrap_ci_eer(sc, lb, np.arange(len(sc)), n_boot=300, seed=1)
    w_spk = ci_spk["ci_hi"] - ci_spk["ci_lo"]
    w_utt = ci_utt["ci_hi"] - ci_utt["ci_lo"]
    print(f"  [info] CI 宽度: 说话人分层 {w_spk:.4f} vs 语句分层 {w_utt:.4f} "
          f"(比值 {w_spk/max(w_utt,1e-9):.2f}x)")
    good = w_spk > w_utt
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] 说话人分层 CI 更宽（按语句重采样会虚假收窄）")
    # 点估计应落在自有 CI 内
    inside = ci_spk["ci_lo"] <= ci_spk["eer_point"] <= ci_spk["ci_hi"]
    ok &= inside
    print(f"  [{'PASS' if inside else 'FAIL'}] 点估计落在自身 95% CI 内 "
          f"({ci_spk['ci_lo']:.4f} <= {ci_spk['eer_point']:.4f} <= {ci_spk['ci_hi']:.4f})")
    # 说话人太少时应优雅降级
    ci1 = bootstrap_ci_eer(np.array([1., 0.]), np.array([1, 0]), np.array([1, 1]))
    nz = ci1.get("eer_point") is None
    ok &= nz
    print(f"  [{'PASS' if nz else 'FAIL'}] 单说话人时优雅降级（不抛异常）")

    # 10) pAUC（任务 02 §8.1）—— 归一化后随机应≈0.5、完美应=1.0
    print("=== pAUC 自检 ===")
    bp = np.array([0.9, 0.8, 0.85, 0.95]); sp = np.array([0.1, 0.2, 0.05, 0.3])
    check("完美分离 pAUC@1%", compute_pauc(bp, sp, 0.01), 1.0, 1e-6)
    check("完美分离 pAUC@0.1%", compute_pauc(bp, sp, 0.001), 1.0, 1e-6)
    # 随机：归一化后应≈0.5
    n2 = 40000
    rb = rng.randn(n2); rs = rng.randn(n2)
    p1 = compute_pauc(rb, rs, 0.01); p01 = compute_pauc(rb, rs, 0.001)
    print(f"  [info] 随机 pAUC@1%={p1:.4f}  pAUC@0.1%={p01:.4f}")
    check("随机 pAUC@1%≈0.5", p1, 0.5, 0.03)
    check("随机 pAUC@0.1%≈0.5", p01, 0.5, 0.05)
    # 方向反转 -> 应显著低于 0.5
    pr = compute_pauc(rs, rb, 0.01)
    print(f"  [info] 方向反转 pAUC@1%={pr:.4f}")
    good = pr < 0.5
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] 方向反转 pAUC < 0.5")
    # 可分性越强 pAUC 应单调不减
    rng4 = np.random.RandomState(3)
    a1 = compute_pauc(rng4.randn(n2) + 0.2, rng4.randn(n2), 0.01)
    rng5 = np.random.RandomState(3)
    a2 = compute_pauc(rng5.randn(n2) + 1.2, rng5.randn(n2), 0.01)
    print(f"  [info] 偏移0.2 pAUC={a1:.4f} vs 偏移1.2 pAUC={a2:.4f}")
    good = a2 >= a1
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] 可分性增强 pAUC 单调不减")

    print("=== 自检" + ("通过 ✅" if ok else "失败 ❌") + " ===")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print(__doc__)
