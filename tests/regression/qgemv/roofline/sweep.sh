#!/usr/bin/env bash
# Roofline sweep for qgemv: every (format, engine) pair across DRAM bandwidths.
# Run from a configured build directory, inside the container:
#   ../tests/regression/qgemv/roofline/sweep.sh [out.csv]
# Knobs (env): CORES (default 8), BW ("banks:data_rate" list), M, K.
# SimX ignores VX_CFG_PLATFORM_MEMORY_PEAK_BW; the DRAM model's bandwidth is
# NUM_BANKS x 16 B (one HBM2 channel) x DATA_RATE / 400 MHz bytes per core cycle.
# L2 is on so the cores share the activations, and CP_LLC_CLIENT is off: with the command
# processor on the L2, the host's weight upload passes through the timed DRAM model and
# roughly doubles the traffic charged to the kernel.
set -u
source ../tests/regression/qgemv/configs.sh
OUT=${1:-roofline.csv}
CORES=${CORES:-8}
BW=${BW:-"2:800 1:160 1:80 1:40 1:20 1:10"}
M=${M:-1024}
K=${K:-2048}
CASES=(
  "q8_0 scalar" "q4_0 scalar" "iq4_nl scalar" "mxfp4 scalar" "q4_k scalar" "q3_k scalar"
  "q8_0 tcu" "q4_0 tcu" "iq4_nl tcu" "mxfp4 tcu"
  "q4_0 dcmp" "iq4_nl dcmp" "mxfp4 dcmp"
)

echo "cores,banks,data_rate,peak_Bpc,type,engine,M,K,instrs,cycles,reqs,status" > "$OUT"
for cores in $CORES; do
  for bd in $BW; do
    banks=${bd%:*}; dr=${bd#*:}
    cfg="$QDEC_DCMP -DVX_CFG_NUM_CORES=$cores -DVX_CFG_L2_ENABLE -DVX_CFG_CP_LLC_CLIENT=0 -DVX_CFG_PLATFORM_MEMORY_NUM_BANKS=$banks -DVX_CFG_PLATFORM_MEMORY_DATA_RATE=$dr"
    make -s -C tests/regression/qgemv clean >/dev/null
    CONFIGS="$cfg -DPERF_ENABLE" make -s -C tests/regression/qgemv >/dev/null 2>&1 || { echo "build failed: $cfg" >&2; exit 1; }
    peak=$(python3 -c "print($banks*16*$dr/400)")
    for c in "${CASES[@]}"; do
      set -- $c; t=$1; e=$2
      out=$(CONFIGS="$cfg" ./ci/blackbox.sh --driver=simx --app=qgemv --perf=7 \
            --args="--type $t --engine $e -m $M -k $K" 2>&1)
      st=PASSED; grep -q '^PASSED' <<<"$out" || st=FAILED
      # multi-core runs print per-core counters first, then the total
      ins=$(grep -oP 'instrs=\K[0-9]+' <<<"$out" | tail -1)
      cyc=$(grep -oP 'cycles=\K[0-9]+' <<<"$out" | tail -1)
      req=$(grep -oP 'memory: reqs=\K[0-9]+' <<<"$out" | tail -1)
      echo "$cores,$banks,$dr,$peak,$t,$e,$M,$K,$ins,$cyc,$req,$st" | tee -a "$OUT"
    done
  done
done
