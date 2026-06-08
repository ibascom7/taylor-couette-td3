#!/bin/bash
# Pack captured ParaView frame trees into committable tarballs.
#
# The raw results/<algo>/<tag>/frames/ trees are hundreds of small OpenFOAM
# files (gitignored). This tars each into a sibling frames.tgz that git DOES
# track, so you can push the visuals to GitHub from your Carya session and
# pull + extract them on your laptop -- no scp / second SSO login.
#
# Run on Carya from the taylorCouetteGym/ dir, after a run finishes:
#   ./pack_frames.sh                 # pack every <algo>/<tag>/frames found
#   ./pack_frames.sh td3/const_base  # pack just one run dir
#
# Then:
#   git add results/**/frames.tgz && git commit -m "frames" && git push
# On your laptop:
#   git pull
#   tar xzf results/td3/const_base/frames.tgz -C results/td3/const_base
#   # open results/td3/const_base/frames/ep0001_*/ep0001_*.foam in ParaView

set -euo pipefail
cd "$(dirname "$0")"

# Directories to scan: explicit args (results/<arg>/frames) or auto-discover.
if [ "$#" -gt 0 ]; then
    FRAME_DIRS=()
    for a in "$@"; do FRAME_DIRS+=("results/$a/frames"); done
else
    mapfile -t FRAME_DIRS < <(find results -type d -name frames 2>/dev/null | sort)
fi

if [ "${#FRAME_DIRS[@]}" -eq 0 ]; then
    echo "No frames/ directories found under results/. Run a capture first."
    exit 0
fi

for fdir in "${FRAME_DIRS[@]}"; do
    if [ ! -d "$fdir" ]; then
        echo "[skip] $fdir (not found)"
        continue
    fi
    run_dir=$(dirname "$fdir")
    tgz="$run_dir/frames.tgz"
    # Store paths relative to run_dir so extraction recreates frames/ in place.
    tar czf "$tgz" -C "$run_dir" frames
    echo "[ok]   $tgz  ($(du -h "$tgz" | cut -f1), $(find "$fdir" -mindepth 1 -maxdepth 1 -type d | wc -l) episodes)"
done

echo
echo "Commit with:  git add results/**/frames.tgz && git commit -m 'paraview frames' && git push"
