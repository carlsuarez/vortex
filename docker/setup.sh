#!/usr/bin/env bash
# One-time setup of the Vortex Docker environment; safe to re-run.
#   1. builds the vortex-deca image (docker/Dockerfile)
#   2. creates the vortex-deca container, mounting the folder that holds this repo (and
#      llama.cpp/ beside it) and the toolchain dir at their host paths. It holds no state,
#      so it is recreated whenever the image or either path changes.
#   3. configures build/ (first run only), installs the prebuilt toolchain if it is
#      missing, then runs make -s and make install
#   4. runs vecadd on simx as a smoke test
# The toolchain dir defaults to ~/tools; override with TOOLDIR=/path docker/setup.sh
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORKSPACE=$(dirname "$REPO")
TOOLDIR=${TOOLDIR:-$HOME/tools}
NAME=vortex-deca
IMAGE=vortex-deca

docker build -t "$IMAGE" \
    --build-arg USERNAME="$(id -un)" --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" \
    --build-arg HOME_DIR="$HOME" "$REPO/docker"

# Create these on the host first; Docker would create a missing mount point as root.
mkdir -p "$REPO/build" "$TOOLDIR"

config="image=$(docker image inspect -f '{{.Id}}' "$IMAGE") workspace=$WORKSPACE tools=$TOOLDIR"
if [ "$(docker container inspect -f '{{index .Config.Labels "setup"}}' "$NAME" 2>/dev/null)" != "$config" ]; then
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    docker run -d --init --name "$NAME" --hostname "$NAME" --label setup="$config" \
        -v "$WORKSPACE:$WORKSPACE" -v "$TOOLDIR:$TOOLDIR" "$IMAGE" >/dev/null
fi

vx() { "$REPO/docker/vx" "$@"; }

# configure reloads its settings from build/config.mk, so later re-runs need no flags
[ -f "$REPO/build/config.mk" ] || vx ../configure --xlen=32 --tooldir="$TOOLDIR"

if [ ! -x "$TOOLDIR/riscv32-gnu-toolchain/bin/riscv32-unknown-elf-gcc" ] ||
   [ ! -x "$TOOLDIR/llvm-vortex/bin/clang" ] || [ ! -x "$TOOLDIR/verilator/bin/verilator" ]; then
    vx ./ci/toolchain_install.sh
fi

vx make -s -j"$(nproc)"
vx make -s install
vx ./ci/blackbox.sh --driver=simx --app=vecadd
