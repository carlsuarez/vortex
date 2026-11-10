#!/usr/bin/env bash
# Write build/compile_commands.json for clangd on the host. bear records the real
# compiler calls in the container, and paths are identical on both sides.
#   docker/compdb.sh            clean-rebuild the Vortex tree and record all of it
#                               (several minutes; third_party isn't rebuilt)
#   docker/compdb.sh DIR...     clean-rebuild only these build dirs, e.g.
#                               tests/regression/qgemv, replacing their old entries
# CONFIGS passes through, so kernels can be recorded with a preset's flags. make runs
# with -k, so a file that doesn't compile yet is still recorded along with the rest.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD=$REPO/build
DB=$BUILD/compile_commands.json
vx() { "$REPO/docker/vx" "$@"; }

if [ $# -eq 0 ]; then
    vx bash -c 'cd "$1" && make -s clean && bear --output "$2" -- make -s -k -j"$(nproc)" && make -s install' _ "$BUILD" "$DB"
    exit
fi

for d in "$@"; do
    vx bash -c '
        make -s -C "$1" clean
        bear --output /tmp/compdb.json -- make -s -k -C "$1"
        # Keep every other entry; replace the ones recorded from this directory.
        python3 - "$2" /tmp/compdb.json "$1" <<"EOF"
import json, os, sys
db, new, d = sys.argv[1], sys.argv[2], sys.argv[3].rstrip("/") + "/"
new = json.load(open(new))
if not new:
    sys.exit("compdb.sh: nothing was compiled in " + d + "; kept the old entries")
old = json.load(open(db)) if os.path.exists(db) else []
keep = [e for e in old if not (e["directory"] + "/").startswith(d)]
json.dump(keep + new, open(db, "w"), indent=2)
EOF
    ' _ "$BUILD/${d%/}" "$DB"
done
