#!/usr/bin/env python3
"""
A 级数据下载（带完整性校验）

⚠️ 教训：第一版 bash 脚本只跑了 curl，**没校验**，结果 test-clean 与两个
   train-clean-100 分片下载中断后留下残缺文件，脚本却直接跳到下一个，
   直到我手动用 pyarrow 打开才发现（`Parquet magic bytes not found in footer`）。
   任务 02 §2 明确要求「记录字节数与文件数，与发布页说明比对」——本脚本补上。

校验三关，缺一不算完成：
  1. 服务器 Content-Length 与实际字节数一致
  2. 文件能被 pyarrow 正常打开并读出 row count
  3. sha256 落盘（供 dataset_audit.csv）

用法: python download_verified.py [--only A2|A3]
"""
import argparse, hashlib, json, os, subprocess, sys, time
import urllib.request
import pyarrow.parquet as pq

BASE = "https://hf-mirror.com/datasets/openslr/librispeech_asr/resolve/main"
D = os.environ["VOICE_WS"]+"/data/_downloads"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")

TARGETS = {
    "A2": [("clean/test/0000.parquet", "test-clean.parquet")],
    "A3": [(f"clean/train.100/{i:04d}.parquet", f"train-clean-100-{i:02d}.parquet")
           for i in range(14)],
}


def remote_size(url):
    r = urllib.request.Request(url, method="HEAD")
    r.add_header("User-Agent", UA)
    with urllib.request.urlopen(r, timeout=60) as resp:
        return int(resp.headers.get("Content-Length", -1))


def verify(path):
    """返回 (ok, info)。ok 需同时满足：能打开 + row_count>0"""
    try:
        pf = pq.ParquetFile(path)
        n = pf.metadata.num_rows
        return (n > 0), f"{n} 行"
    except Exception as e:
        return False, str(e)[:70]


def sha256(path, bs=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(bs), b""):
            h.update(c)
    return h.hexdigest()


def download(rel, name, max_try=4):
    url = f"{BASE}/{rel}"
    out = os.path.join(D, name)
    try:
        want = remote_size(url)
    except Exception as e:
        print(f"  ⚠️ HEAD 失败: {e}; 仍尝试下载")
        want = -1

    for attempt in range(1, max_try + 1):
        have = os.path.getsize(out) if os.path.exists(out) else 0
        if want > 0 and have == want:
            ok, info = verify(out)
            if ok:
                print(f"  [skip] {name} 已完整 ({have} B, {info})")
                return {"name": name, "bytes": have, "rows": info, "status": "ok"}
            print(f"  ⚠️ {name} 字节数对但打不开({info}) -> 删除重下")
            os.remove(out)

        print(f"  尝试 {attempt}/{max_try}: {rel}  (本地 {have} B / 远端 {want if want>0 else '?'} B)")
        # 用 curl 断点续传；失败不抛出
        r = subprocess.run(["curl", "-L", "-C", "-", "--retry", "2",
                            "--connect-timeout", "30", "--max-time", "3600",
                            "-o", out, url],
                           capture_output=True)
        got = os.path.getsize(out) if os.path.exists(out) else 0

        if want > 0 and got != want:
            print(f"     ✗ 字节不符: {got} != {want}  (curl rc={r.returncode}) -> 重试")
            # 文件比远端大说明坏了；否则保留以便续传
            if got > want:
                os.remove(out)
            time.sleep(2)
            continue
        ok, info = verify(out)
        if ok:
            print(f"     ✓ 校验通过: {got} B, {info}")
            return {"name": name, "bytes": got, "rows": info, "status": "ok"}
        print(f"     ✗ 打不开: {info} -> 删除重下")
        if os.path.exists(out):
            os.remove(out)
        time.sleep(2)

    return {"name": name, "bytes": os.path.getsize(out) if os.path.exists(out) else 0,
            "rows": "FAILED", "status": "failed"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="A2 或 A3")
    args = ap.parse_args()
    os.makedirs(D, exist_ok=True)

    keys = [args.only] if args.only in TARGETS else list(TARGETS)
    results = []
    for k in keys:
        print(f"\n===== {k} =====")
        for rel, name in TARGETS[k]:
            results.append(download(rel, name))

    print("\n===== 汇总 =====")
    okn = sum(1 for r in results if r["status"] == "ok")
    for r in results:
        print(f"  [{'OK ' if r['status']=='ok' else 'FAIL'}] {r['name']:<32} "
              f"{r['bytes']:>12} B  {r['rows']}")
    print(f"\n成功 {okn}/{len(results)}")

    # 落 sha256（供 dataset_audit.csv）
    if okn == len(results):
        print("计算 sha256 ...")
        for r in results:
            r["sha256"] = sha256(os.path.join(D, r["name"]))
        with open(os.environ["VOICE_WS"]+"/results/_download_hashes.json", "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("-> results/_download_hashes.json")

    print("DOWNLOAD_VERIFIED_DONE" if okn == len(results) else "DOWNLOAD_VERIFIED_INCOMPLETE")
    return 0 if okn == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
