#!/usr/bin/env bash
# Clean, build and run qgemv with one CONFIGS preset from configs.sh.
#   run.sh PRESET [--perf=CLASS] ARGS...   one run, e.g.
#                                          run.sh QDEC_WMMA --type q8_0 --engine scalar -m 256 -k 1024
#   run.sh smoke                           small shapes for every format and engine built so far
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
)

build() {  # CONFIGS
    make -s -C tests/regression/qgemv clean
    CONFIGS="$1" make -s -C tests/regression/qgemv
}

if [ "${1:-}" = smoke ]; then
    build "$QDEC_WMMA"
    fails=0
    for args in "${SMOKE[@]}"; do
        status=FAIL
        if out=$(CONFIGS="$QDEC_WMMA" ./ci/blackbox.sh --driver=simx --app=qgemv --args="$args" 2>&1) &&
           grep -q '^PASSED!' <<< "$out"; then
            status=PASS
        else
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
