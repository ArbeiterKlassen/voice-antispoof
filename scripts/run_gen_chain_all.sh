#!/usr/bin/env bash
# 生成链检查：其余族 + 正对照（判据已冻结在 results/gen_chain_thresholds_v1.md）
# 逐族跑：伪造批(180) + 真实批(120)；正对照族见冻结尾表的期望表。
set -u
VOICE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VOICE_ROOT/scripts/env.local.sh" ] && . "$VOICE_ROOT/scripts/env.local.sh"
export VOICE_WS="${VOICE_WS:-$VOICE_ROOT}"
WS="$VOICE_WS"
PY="${VOICE_PY:-python3}"

cd "$WS"
MAN=data/manifests
OUT=results
export OMP_NUM_THREADS=4

# ⚠️ 逐族都是同一份 p4fix manifest —— 脚本按 --max-files 取前 N 对，会只落在 mp3 族！
# 所以这里按族过滤生成临时 manifest（每种族 180 假 + 120 真的该族行）
for fam in g711-8k bandpass-300-3400 speed0.9 reverb-rt60-0.4 awgn-snr5 aac-128k opus-32k; do
  $PY - "$fam" <<'PYEOF'
import sys, pandas as pd
fam = sys.argv[1]
d = pd.read_csv("data/manifests/la_p4fix_manifest.csv")
sub = d[d.attack_type == fam]
sub.to_csv(f"/tmp/gc_fam_{fam}.csv", index=False)
print(f"    {fam}: {len(sub)} 行")
PYEOF
  echo "########## $fam ##########"
  nice -n 10 $PY -u scripts/gen_chain_checks.py \
    --src-manifest $MAN/_la_p4_fake_src.csv --out-manifest /tmp/gc_fam_$fam.csv \
    --max-files 180 --out-csv $OUT/gc_${fam}_fake.csv 2>&1 | tail -3
  nice -n 10 $PY -u scripts/gen_chain_checks.py \
    --src-manifest $MAN/_la_p4_real_src.csv --out-manifest /tmp/gc_fam_$fam.csv \
    --max-files 120 --out-csv $OUT/gc_${fam}_real.csv 2>&1 | tail -3
done

# 音高族 + 恒等臂（源同上；正对照 C：f0 必须等于设计值）
for spec in "pitch-down2:la_p4pitch_manifest.csv" "pitch-up4:la_p4pitch_manifest.csv" \
            "vocoder-id:la_p4vocid_manifest.csv"; do
  fam=${spec%%:*}; mf=${spec#*:}
  $PY - "$fam" "$mf" <<'PYEOF'
import sys, pandas as pd
fam, mf = sys.argv[1], sys.argv[2]
d = pd.read_csv(f"data/manifests/{mf}")
d[d.attack_type == fam].to_csv(f"/tmp/gc_fam_{fam}.csv", index=False)
PYEOF
  echo "########## $fam ##########"
  nice -n 10 $PY -u scripts/gen_chain_checks.py \
    --src-manifest $MAN/_la_p4_fake_src.csv --out-manifest /tmp/gc_fam_$fam.csv \
    --max-files 180 --out-csv $OUT/gc_${fam}_fake.csv 2>&1 | tail -3
  nice -n 10 $PY -u scripts/gen_chain_checks.py \
    --src-manifest $MAN/_la_p4_real_src.csv --out-manifest /tmp/gc_fam_$fam.csv \
    --max-files 120 --out-csv $OUT/gc_${fam}_real.csv 2>&1 | tail -3
done
echo "GEN_CHAIN_ALL_DONE"
