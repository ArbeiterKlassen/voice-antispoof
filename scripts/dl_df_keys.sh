#!/usr/bin/env bash
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

cd "$WS"/data/_downloads/df_keys
unset https_proxy http_proxy all_proxy HTTPS_PROXY HTTP_PROXY ALL_PROXY
URL="https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz"
echo "[$(date +%T)] 续传 keys"
for i in $(seq 1 40); do
  curl -L -C - --retry 5 --retry-delay 5 --connect-timeout 30 --max-time 600 \
       -o DF-keys-full.tar.gz "$URL" && break
  echo "[$(date +%T)] 第 $i 次中断，已 $(du -k DF-keys-full.tar.gz | cut -f1) KB"
  sleep 10
done
echo "[$(date +%T)] 大小 $(du -k DF-keys-full.tar.gz | cut -f1) KB"
echo "MD5 = $(md5sum DF-keys-full.tar.gz | cut -d' ' -f1)"
echo "期望 = dabbc5628de4fcef53036c99ac7ab93a"
tar xzf DF-keys-full.tar.gz 2>/dev/null && echo "解包 ok" || echo "❌ 解包失败"
ls -la keys/DF/CM/ 2>/dev/null
wc -l keys/DF/CM/trial_metadata.txt 2>/dev/null
echo "DF_KEYS_DONE"
