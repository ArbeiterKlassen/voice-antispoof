#!/usr/bin/env python3
"""
说话人划分自检（任务 02 §3.2 规定 2 / §5 交付项 split_check.txt）

一条命令跑出全部数字：
  python split_check.py > results/split_check.txt

验收口径（任务 02 §5）：输出必须与 manifest 一致，不得手工填写。
"""
import sys
from itertools import combinations
import pandas as pd
import os

WS = os.environ.get("VOICE_WS") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAN = f"{WS}/data/manifests/data_manifest_15col.csv"


def main(path=MAN):
    df = pd.read_csv(path)
    print("=" * 68)
    print(" 说话人划分自检 (split_check)")
    print(f" manifest: {path}")
    print("=" * 68)

    print(f"\n[1] 总览")
    print(f"  总条数    : {len(df)}")
    print(f"  utt_id 唯一: {df.utt_id.is_unique}")
    print(f"  说话人总数 : {df.speaker_id.nunique()}")

    print(f"\n[2] 各 split 的说话人集合")
    sets = {}
    for s in ["train", "dev", "test"]:
        sub = df[df.split == s]
        spk = {int(v) for v in sub.speaker_id.unique()}
        sets[s] = spk
        n_r = int((sub.label == "real").sum()); n_f = int((sub.label == "fake").sum())
        print(f"  {s:<6} 条数 {len(sub):>5} (real {n_r:>5} / fake {n_f:>5})  "
              f"说话人 {len(spk):>3} 个")
        print(f"         说话人列表: {sorted(spk)}")

    print(f"\n[3] 交集大小（验收要求：必须为 0）")
    allzero = True
    for a, b in combinations(["train", "dev", "test"], 2):
        inter = sets[a] & sets[b]
        flag = "✅" if len(inter) == 0 else "❌"
        if len(inter) != 0:
            allzero = False
        print(f"  {flag} |{a} ∩ {b}| = {len(inter)}"
              + (f"   交集: {sorted(inter)}" if inter else ""))

    print(f"\n[4] 判定")
    print(f"  三集合两两交集均为 0: {allzero}")
    print(f"  {'✅ PASS —— speaker-disjoint 成立' if allzero else '❌ FAIL —— 存在说话人泄漏'}")

    # 附加一致性检查（同一 manifest 内不得有重复路径/哈希）
    print(f"\n[5] 附加一致性")
    print(f"  audio_path 唯一: {df.audio_path.is_unique}")
    if "sha256" in df.columns:
        dup = df[df.sha256.duplicated(keep=False)]
        print(f"  sha256 唯一    : {df.sha256.is_unique}  (重复 {len(dup)} 行)")
        if len(dup):
            print(f"    ⚠️ 重复哈希样例: {dup.sha256.iloc[0][:16]}... x{len(dup)}")
    # 每个 split 必须 real/fake 俱全
    print(f"\n[6] 每个 split 的类别完备性")
    for s in ["train", "dev", "test"]:
        sub = df[df.split == s]
        ok = sub.label.nunique() == 2
        print(f"  {'✅' if ok else '❌'} {s}: real={int((sub.label=='real').sum())} "
              f"fake={int((sub.label=='fake').sum())}")

    print("\n" + "=" * 68)
    print("SPLIT_CHECK_DONE")
    return 0 if allzero else 1


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else MAN
    sys.exit(main(p))
