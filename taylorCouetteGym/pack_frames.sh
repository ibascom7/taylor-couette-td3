#!/bin/bash
# Pack captured ParaView frame trees into committable tarballs.
#
# The raw results/<algo>/<tag>/frames/ trees are hundreds of small OpenFOAM
# files (gitignored). This tars them into sibling tarball(s) that git DOES
# track, so you can push the visuals to GitHub from your Carya session and
# pull + extract them on your laptop -- no scp / second SSO login.
#
# Run on Carya from the taylorCouetteGym/ dir, after a run finishes:
#   ./pack_frames.sh                 # pack every <algo>/<tag>/frames found
#   ./pack_frames.sh td3/const_base  # pack just one run dir
#   ./pack_frames.sh --split td3/full3d_test_s0   # ONE tarball PER episode
#
# --split mode writes results/<run>/frames_<ep>.tgz per episode instead of a
# single frames.tgz. Use it for the full-3D case: one combined archive of
# 9216-cell episodes exceeds GitHub's 100 MB per-file limit (~43 MB/episode
# split, vs ~128 MB combined), so the combined push gets rejected. --split also
# removes any stale combined frames.tgz in that run dir so you don't commit it.
#
# Then:
#   git add results/**/frames*.tgz && git commit -m "frames" && git push
# On your laptop:
#   git pull
#   # combined:   tar xzf results/td3/const_base/frames.tgz -C results/td3/const_base
#   # split:      for f in results/td3/full3d_test_s0/frames_ep*.tgz; do \
#   #                 tar xzf "$f" -C results/td3/full3d_test_s0; done
#   # open results/<run>/frames/ep0001/ep0001.foam in ParaView

set -euo pipefail
cd "$(dirname "$0")"

# --split: one tarball per episode (keeps each file under GitHub's 100 MB cap).
SPLIT=0
if [ "${1:-}" = "--split" ]; then
    SPLIT=1
    shift
fi

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

    if [ "$SPLIT" -eq 1 ]; then
        # One archive per episode subdir. Drop any stale combined archive so it
        # can't be committed by accident (it's likely the >100 MB one).
        rm -f "$run_dir/frames.tgz"
        shopt -s nullglob
        eps=("$fdir"/*/)
        shopt -u nullglob
        if [ "${#eps[@]}" -eq 0 ]; then
            echo "[skip] $fdir (no episode subdirs)"
            continue
        fi
        for ep in "${eps[@]}"; do
            name=$(basename "$ep")
            tgz="$run_dir/frames_${name}.tgz"
            # Store paths relative to run_dir so extraction recreates frames/<ep> in place.
            tar czf "$tgz" -C "$run_dir" "frames/$name"
            echo "[ok]   $tgz  ($(du -h "$tgz" | cut -f1))"
        done
    else
        tgz="$run_dir/frames.tgz"
        # Store paths relative to run_dir so extraction recreates frames/ in place.
        tar czf "$tgz" -C "$run_dir" frames
        echo "[ok]   $tgz  ($(du -h "$tgz" | cut -f1), $(find "$fdir" -mindepth 1 -maxdepth 1 -type d | wc -l) episodes)"
    fi
done

echo
echo "Commit with:  git add results/**/frames*.tgz && git commit -m 'paraview frames' && git push"
