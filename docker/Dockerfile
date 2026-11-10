# Vortex build environment for this repo: Ubuntu 22.04, the OS Vortex's CI runs on.
# It holds system packages only. docker/setup.sh mounts the folder holding the repo
# (and llama.cpp/ beside it) and the prebuilt toolchain at their host paths, and runs
# the container as the host user, so paths match on both sides and files stay yours.
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# Lines 1-2: baseline from ci/install_dependencies.sh. Line 3: extras docs/install_vortex.md
# lists for 22.04. Line 4: Tcl for the toolchain's yosys and OpenSTA, PyYAML and pytest for
# ci/testcases, numpy and matplotlib for roofline plots. Line 5: llama.cpp (pkg-config finds
# vortex-runtime; older commits need libcurl, newer ones OpenSSL for HTTPS). Line 6:
# ci/toolchain_install.sh, the shell, and bear for docker/compdb.sh.
RUN apt-get update && apt-get install -y \
        build-essential valgrind libstdc++6 binutils python3 uuid-dev ccache cmake libffi8 \
        libpng-dev libboost-serialization-dev opencl-headers ocl-icd-opencl-dev \
        zlib1g-dev libtinfo-dev libncurses-dev libhwloc-dev \
        libtcl8.6 python3-yaml python3-pytest python3-numpy python3-matplotlib \
        pkg-config libcurl4-openssl-dev libssl-dev \
        ca-certificates git wget curl bzip2 xz-utils perl sudo less vim bc bear \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=vortex
ARG UID=1000
ARG GID=1000
ARG HOME_DIR=/home/vortex
RUN (getent group "$GID" >/dev/null || groupadd -g "$GID" "$USERNAME") \
    && useradd -m -u "$UID" -g "$GID" -d "$HOME_DIR" -s /bin/bash "$USERNAME" \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/$USERNAME"

USER $USERNAME
ENV LANG=C.UTF-8
WORKDIR $HOME_DIR
CMD ["sleep", "infinity"]
