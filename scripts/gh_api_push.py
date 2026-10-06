#!/usr/bin/env python3
"""
把本地产物经 **GitHub REST API** 推送（git 协议在本机被封：直连超时、代理 CONNECT 被断；
api.github.com 可达）。单提交推送：blobs → tree → commit(parent=远端 main) → 更新 ref。

2026-10-07 加固：超时 120s + 网络层/5xx 自动重试 2 次 + 进度每 25 个 blob（本机网络病态时易超时）。

凭据读取顺序：$GH_TOKEN → ~/.gh_pat_voice_antispoof（一行纯 token）
用法：python3 gh_api_push.py [--dry-run] [--repo ArbeiterKlassen/voice-antispoof]
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO = "ArbeiterKlassen/voice-antispoof"
API = "https://api.github.com"


def token():
    t = os.environ.get("GH_TOKEN", "").strip()
    if not t:
        p = os.path.expanduser("~/.gh_pat_voice_antispoof")
        if os.path.exists(p):
            t = open(p).read().strip()
    return t


def req(method, url, tok, body=None, ok=(200, 201, 409), retries=2):
    data = json.dumps(body).encode() if body is not None else None
    last = None
    for a in range(retries + 1):
        r = urllib.request.Request(url, data=data, method=method, headers={
            "Authorization": f"Bearer {tok}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "voice-ws-api-push",
            "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=120) as resp:
                return resp.status, json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as e:
            if e.code >= 500 and a < retries:          # 5xx 可重试
                last = f"HTTP {e.code}"
                time.sleep(2 * (a + 1)); continue
            return e.code, {"error": e.read().decode()[:300]}
        except Exception as e:                          # 网络层（超时/断连）可重试
            last = f"{type(e).__name__}: {e}"
            if a < retries:
                print(f"    [retry {a+1}/{retries}] {method} {url.split('/')[-1]} :: {last}", flush=True)
                time.sleep(2 * (a + 1)); continue
            return 0, {"error": last}
    return 0, {"error": last}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--message", default="导入语音辨伪工作区\n\nCo-Authored-By: Claude Code <noreply@anthropic.com>")
    args = ap.parse_args()

    tok = token()
    if not tok:
        sys.exit("❌ 缺 token：放到 ~/.gh_pat_voice_antispoof（或设 $GH_TOKEN）")
    W = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
    # -z：非 ASCII 文件名默认会被 quote 成 "\345\215..." 转义串（core.quotepath）
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=W).decode().split("\0")
    files = [f for f in files if f]
    print(f"仓库 {args.repo}；本地文件 {len(files)}", flush=True)

    st, me = req("GET", f"{API}/repos/{args.repo}", tok)
    if st != 200:
        sys.exit(f"❌ 读仓库失败 {st}: {me}")
    perms = me.get("permissions", {})
    print(f"  permissions: {perms}  （需 push=true）")
    if not perms.get("push") and not args.dry_run:
        sys.exit("❌ token 无写权限（需 Contents: Read and write）")

    st, ref = req("GET", f"{API}/repos/{args.repo}/git/ref/heads/{me.get('default_branch','main')}", tok)
    if st != 200:
        sys.exit(f"❌ 读 ref 失败 {st}: {ref}")
    parent = ref["object"]["sha"]
    print(f"  远端 main = {parent[:12]}（本提交将以其为父）")

    if args.dry_run:
        print("  [dry-run] 到此为止，未写任何东西")
        return

    # 逐文件 blob：**内容寻址复用**——先用本地 git hash-object 算 blob sha，GET 查存在性
    # （读请求不计内容创建限速），存在就用，不存在才 POST。
    # ⚠️ 为什么：GitHub 对 content-generating 请求有 500 次/小时的二级限速，超限报
    #    「401 Bad credentials」（极具误导性）；今晚三次全量推送共 558 个 blob POST 撞限。
    tree, t0, n_bytes = [], time.time(), 0
    n_up = n_re = 0
    for i, f in enumerate(files):
        p = os.path.join(W, f)
        raw = open(p, "rb").read()
        n_bytes += len(raw)
        local_sha = subprocess.check_output(["git", "hash-object", p], cwd=W).decode().strip()
        st, r = req("GET", f"{API}/repos/{args.repo}/git/blobs/{local_sha}", tok)
        if st == 200:
            sha = local_sha; n_re += 1
        else:
            st2, r2 = req("POST", f"{API}/repos/{args.repo}/git/blobs", tok,
                          {"content": base64.b64encode(raw).decode(), "encoding": "base64"})
            if st2 not in (200, 201):
                sys.exit(f"❌ blob 失败 {f}: {st2} {r2}")
            sha = r2["sha"]; n_up += 1
        mode = "100755" if os.access(p, os.X_OK) else "100644"
        tree.append({"path": f, "mode": mode, "type": "blob", "sha": sha})
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(files)}（复用 {n_re} / 新传 {n_up}；"
                  f"{n_bytes/1048576:.1f} MB，{time.time()-t0:.0f}s）", flush=True)
    print(f"  blob 汇总：复用 {n_re}，新传 {n_up}")

    st, r = req("POST", f"{API}/repos/{args.repo}/git/trees", tok, {"tree": tree})
    if st not in (200, 201):
        sys.exit(f"❌ tree 失败: {st} {r}")
    tree_sha = r["sha"]
    print(f"  tree = {tree_sha[:12]}")

    st, r = req("POST", f"{API}/repos/{args.repo}/git/commits", tok,
                {"message": args.message, "tree": tree_sha, "parents": [parent]})
    if st not in (200, 201):
        sys.exit(f"❌ commit 失败: {st} {r}")
    commit_sha = r["sha"]
    print(f"  commit = {commit_sha[:12]}")

    st, r = req("PATCH", f"{API}/repos/{args.repo}/git/refs/heads/{me.get('default_branch','main')}",
                tok, {"sha": commit_sha, "force": False})
    if st not in (200, 201):
        sys.exit(f"❌ 更新 ref 失败: {st} {r}")
    print(f"✅ 推送完成：main → {commit_sha[:12]}（{len(files)} 文件，{n_bytes/1048576:.1f} MB）")
    print(f"   远端历史：{parent[:12]}（骨架）→ {commit_sha[:12]}")
    print(f"   ⚠️ 本地 HEAD 与远端 SHA 不同（本地历史无父，内容相同）：git 传输恢复后可 fetch/rebase 对齐。")


if __name__ == "__main__":
    main()
