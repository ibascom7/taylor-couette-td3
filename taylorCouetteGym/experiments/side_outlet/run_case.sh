#!/usr/bin/env bash
# Reproduce the Yuhe side-outlet flow-through run (no RL) end-to-end.
#
#   ./run_case.sh /path/to/extracted/case   [/path/to/workdir]
#
# The first arg is the unzipped taylor_couette_reactor_short case (the dir that
# holds 0/ constant/ system/). It is copied to WORKDIR, meshed, solved to
# endTime (200 s), then post-processed into plots + a ParaView frame movie.
# Needs: OpenFOAM v2506 in PATH (pimpleFoam, blockMesh), pvbatch, python3+matplotlib.
set -euo pipefail

SRC="${1:?usage: run_case.sh CASE_SRC [WORKDIR]}"
WORK="${2:-$(pwd)/run}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo ">> copying case $SRC -> $WORK"
rm -rf "$WORK"; mkdir -p "$WORK"
cp -r "$SRC"/0 "$SRC"/constant "$SRC"/system "$WORK"/
cd "$WORK"

echo ">> blockMesh"
blockMesh > log.blockMesh 2>&1
echo ">> checkMesh"; checkMesh > log.checkMesh 2>&1 || true

echo ">> pimpleFoam (this is the ~20 min step; high-omega bursts set the cost)"
pimpleFoam > log.run 2>&1
grep -E "ExecutionTime" log.run | tail -1

echo ">> plots (omega / energy / conversion)"
python3 "$HERE/plot_results.py" "$WORK" "$HERE"

echo ">> ParaView frames + montage"
touch case.foam
pvbatch "$HERE/visualize.py" "$WORK" "$WORK/frames" > log.pvbatch 2>&1
python3 "$HERE/montage.py" "$WORK/frames" "$HERE/montage.png" 4 40 100 200

echo ">> done. See $HERE for MEETING_BRIEF.md, RESULTS.md, *.png"
