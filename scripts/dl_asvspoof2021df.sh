#!/usr/bin/env bash
# 下载 ASVspoof2021 DF（镜像版，仅供本机管线/预训练使用，**不得**写成与文献可比）
# 断点续传 + 重试；完成后做 zip 完整性检验
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

OUT=$WS/data/_downloads/asvspoof2021_df
URL="https://hf-mirror.com/datasets/Bisher/ASVspoof_2021_DF/resolve/main/ASVspoof_DF_2021.zip"
mkdir -p $OUT
cd $OUT
unset https_proxy http_proxy all_proxy HTTPS_PROXY HTTP_PROXY ALL_PROXY
echo "[$(date +%T)] 开始 / 续传"
for i in $(seq 1 60); do
  curl -L -C - --retry 5 --retry-delay 5 --connect-timeout 30 --max-time 21600 \
       -o ASVspoof_DF_2021.zip "$URL" && break
  echo "[$(date +%T)] 第 $i 次中断，20s 后续传（已下 $(du -m ASVspoof_DF_2021.zip 2>/dev/null | cut -f1) MB）"
  sleep 20
done
echo "[$(date +%T)] 下载结束，大小 $(du -m ASVspoof_DF_2021.zip | cut -f1) MB"
echo "--- 完整性检验（zipfile）---"
python3 - <<'PY'
import zipfile, sys
p=os.environ["VOICE_WS"]+"/data/_downloads/asvspoof2021_df/ASVspoof_DF_2021.zip"
try:
    z=zipfile.ZipFile(p)
    bad=z.testzip()
    names=z.namelist()
    print(f"条目数 {len(names)}；testzip -> {bad}")
    for n in names[:15]: print("   ", n)
    print("✅ zip 完整" if bad is None else f"❌ 首个坏条目 {bad}")
except Exception as e:
    print("❌ 打开失败:", e); sys.exit(1)
PY
echo "DL_DF2021_DONE"
