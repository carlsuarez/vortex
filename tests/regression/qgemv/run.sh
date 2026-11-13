#!/usr/bin/env bash
# Clean, build and run qgemv with one CONFIGS preset from configs.sh.
#   run.sh PRESET [--perf=CLASS] ARGS...   one run, e.g.
#                                          run.sh QDEC_WMMA --type q8_0 --engine scalar -m 256 -k 1024
#   run.sh smoke                           small shapes for every format and engine built so far,
#                                          under QDEC_DCMP; dcmp must match tcu bit for bit
# It works from vortex/build in the container, and re-runs itself there through docker/vx
# when started on the host.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ -f /.dockerenv ] || exec "$HERE/../../../docker/vx" "$HERE/run.sh" "$@"
source "$HERE/configs.sh"
cd "${BUILD:-$HERE/../../../build}"

SMOKE=(
    "--type q8_0 --engine scalar -m 64 -k 256"
    "--type q8_0 --engine scalar -m 37 -k 288 --seed 2"
    "--type q8_0 --engine scalar -m 288 -k 768 --seed 3"
    "--type q8_0 --engine scalar -m 1 -k 32 --seed 4"
    "--type q4_0 --engine scalar -m 64 -k 256"
    "--type q4_0 --engine scalar -m 37 -k 288 --seed 2"
    "--type q4_0 --engine scalar -m 288 -k 768 --seed 3"
    "--type q4_0 --engine scalar -m 1 -k 32 --seed 4"
    "--type iq4_nl --engine scalar -m 64 -k 256"
    "--type iq4_nl --engine scalar -m 37 -k 288 --seed 2"
    "--type iq4_nl --engine scalar -m 288 -k 768 --seed 3"
    "--type iq4_nl --engine scalar -m 1 -k 32 --seed 4"
    "--type mxfp4 --engine scalar -m 64 -k 256"
    "--type mxfp4 --engine scalar -m 37 -k 288 --seed 2"
    "--type mxfp4 --engine scalar -m 288 -k 768 --seed 3"
    "--type mxfp4 --engine scalar -m 1 -k 32 --seed 4"
    # K-quants: rows are whole 256-weight blocks
    "--type q4_k --engine scalar -m 64 -k 256"
    "--type q4_k --engine scalar -m 37 -k 512 --seed 2"
    "--type q4_k --engine scalar -m 288 -k 768 --seed 3"
    "--type q4_k --engine scalar -m 1 -k 256 --seed 4"
    "--type q3_k --engine scalar -m 64 -k 256"
    "--type q3_k --engine scalar -m 37 -k 512 --seed 2"
    "--type q3_k --engine scalar -m 288 -k 768 --seed 3"
    "--type q3_k --engine scalar -m 1 -k 256 --seed 4"
    # TCU engines: one warp per 8 rows, so M=37 and M=1 exercise a partial tile
    "--type q8_0 --engine tcu -m 64 -k 256"
    "--type q8_0 --engine tcu -m 37 -k 288 --seed 2"
)
# tcu and dcmp must agree bit for bit (same yhash) on every shape here
for t in q4_0 iq4_nl mxfp4; do
    for e in tcu dcmp; do
        SMOKE+=("--type $t --engine $e -m 64 -k 256"
                "--type $t --engine $e -m 37 -k 288 --seed 2"
                "--type $t --engine $e -m 1 -k 32 --seed 4")
    done
done

build() {  # CONFIGS
    make -s -C tests/regression/qgemv clean
    CONFIGS="$1" make -s -C tests/regression/qgemv
}

if [ "${1:-}" = smoke ]; then
    build "$QDEC_DCMP"
    fails=0
    declare -A tcu_hash
    for args in "${SMOKE[@]}"; do
        status=FAIL
        if out=$(CONFIGS="$QDEC_DCMP" ./ci/blackbox.sh --driver=simx --app=qgemv --args="$args" 2>&1) &&
           grep -q '^PASSED!' <<< "$out"; then
            status=PASS
        else
            fails=$((fails + 1))
        fi
        hash=$(grep -oP 'yhash=\K\w+' <<< "$out" || true)
        shape=${args/--engine tcu/}
        shape=${shape/--engine dcmp/}
        if [[ $args == *"--engine tcu"* ]]; then
            tcu_hash[$shape]=$hash
        elif [[ $status == PASS && $args == *"--engine dcmp"* && "${tcu_hash[$shape]:-}" != "$hash" ]]; then
            status="FAIL(dcmp!=tcu)"
            fails=$((fails + 1))
        fi
        echo "$status $(grep -E '^QGEMV_RESULT|^PERF: instrs' <<< "$out" | tr '\n' ' ')"
    done
    echo "smoke: $((${#SMOKE[@]} - fails))/${#SMOKE[@]} passed"
    exit $((fails > 0))
fi

preset=${1:?usage: run.sh PRESET [--perf=CLASS] ARGS... | run.sh smoke}
cfg=${!preset:-}
[ -n "$cfg" ] || { echo "run.sh: unknown preset '$preset' (see configs.sh)" >&2; exit 2; }
shift

# The app and the simx driver must be built with identical CONFIGS. With --perf,
# blackbox.sh appends -DPERF_ENABLE to the driver's, so the app gets the same here.
perf=()
app_cfg=$cfg
if [[ ${1:-} == --perf=* ]]; then
    perf=("$1")
    app_cfg="$cfg -DPERF_ENABLE"
    shift
fi
build "$app_cfg"
CONFIGS="$cfg" ./ci/blackbox.sh --driver=simx --app=qgemv "${perf[@]}" --args="$*"
