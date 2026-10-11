#!/bin/bash
# =====================================================================
# Carya: MECHANISM experiments 1-3 + e5 (experiments/mechanism/PLAN.md).
#
# What sets the best stop length? 44 runs on the short reactor at maxCo 0.9,
# each 130 s from a state warmed at constant 300 rpm AT ITS OWN VISCOSITY:
#   e1  stop-length curve, nominal viscosity (13 stops + 2 replicates)
#   e2  same curve at 2x (13) and 0.7x (10) viscosity, D fixed at 1e-8
#   e3  stop level 30/60/100/150 rpm at a 0.5 s stop (4)
#   e5  vortex-state tests: pulse->constant 'hold' and single-stop 'kick' (2)
# One node, 44 threaded workers: base compile (~1 min) -> 3 warm-ups in
# parallel (60 s each, ~2-3 h; 2x viscosity is slowest) -> all 44 runs at once
# (~4-5 h nominal, ~6-8 h for the 2x arm, which starts first). Expect ~10-12 h.
# The cases run on node-local $TMPDIR; each finished run's outputs
# (mech.dat traces, run.json, log tail, final time dir) are copied to
# experiments/mechanism/results/main/runs/<tag>/ and marked COMPLETE.
#
# RESUME after a walltime kill: resubmit. Runs marked COMPLETE are only
# re-parsed; the rest restart from scratch (the node-local work dir is gone).
#
# Submit FROM taylorCouetteGym/ (paths are $SLURM_SUBMIT_DIR-relative):
#     cd /project/.../taylor-couette-td3/taylorCouetteGym
#     mkdir -p experiments/mechanism/logs      # logs/ is gitignored; slurm needs it to exist
#     sbatch experiments/mechanism/run_carya_mechanism.slurm
# Monitor: tail -f experiments/mechanism/logs/mechanism_*.out
# Replot anywhere: python3 experiments/mechanism/mechanism_sweep.py --analyze-only
# =====================================================================

#SBATCH -J mechanism
#SBATCH -o experiments/mechanism/logs/%x_%j.out
#SBATCH -e experiments/mechanism/logs/%x_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48                     # 44 workers + main thread + headroom
#SBATCH --mem=32G
#SBATCH -t 24:0:0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bascomisaiah@gmail.com

set -euo pipefail

OPENFOAM_MODULE="OpenFOAM/v2506-foss-2025b"
set +u; module purge; module load "$OPENFOAM_MODULE"; source "$FOAM_BASH"; set -u
unset PYTHONPATH PYTHONHOME
PYTHON=/project/mwang/ibascom/envs/tc/bin/python

cd "$SLURM_SUBMIT_DIR"
mkdir -p experiments/mechanism/logs
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
WORK_ROOT="${TMPDIR:-/tmp/$USER}/mechanism_work"

echo "=== MECHANISM: host=$(hostname) cores=${SLURM_CPUS_PER_TASK:-?} work=$WORK_ROOT ==="
"$PYTHON" experiments/mechanism/mechanism_sweep.py \
    --work_root "$WORK_ROOT" \
    --results_dir experiments/mechanism/results/main \
    --resume
echo "=== mechanism done -> experiments/mechanism/results/main ==="
