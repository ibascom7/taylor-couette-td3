#!/bin/bash
# =====================================================================
# Carya: MESH CHECK of the Taylor-cell mechanism (experiments/mechanism_mesh).
#
# Repeats the mechanism experiment's key nominal-viscosity runs on a coarser
# (f0.71, ~1.7k cells) and a finer (f1.41, ~6.6k cells) mesh: constant 300,
# stops 0.2/0.3/0.4/0.5/0.6/0.7/1.2 s, and the pulse->constant hold run
# (18 runs). The production mesh (f1.00) rows are reused from
# experiments/mechanism/results/main, so pull that first (it is in git).
#
# One node, 18 threaded workers: build both meshes + compile (~2 min) ->
# 2 warm-ups in parallel (60 s; f1.41 ~4-5 h) -> all 18 runs at once.
# In the local smoke test f1.41 cost ~4.7x the production mesh per simulated
# second (finer cells AND a smaller Courant-limited time step), so expect
# ~15-20 h in total; the f0.71 runs finish in ~2 h. Walltime 48 h for margin
# (a walltime kill restarts unfinished runs from scratch).
#
# Cases run on node-local $TMPDIR; finished runs are copied to
# experiments/mechanism_mesh/results/main/runs/<tag>/ and marked COMPLETE.
# RESUME after a walltime kill: resubmit (COMPLETE runs are only re-parsed).
#
# Submit FROM taylorCouetteGym/:
#     cd /project/.../taylor-couette-td3/taylorCouetteGym
#     mkdir -p experiments/mechanism_mesh/logs     # logs/ is gitignored; slurm needs it
#     sbatch experiments/mechanism_mesh/run_carya_mesh_check.slurm
# Monitor: tail -f experiments/mechanism_mesh/logs/mesh_check_*.out
# Replot anywhere: python3 experiments/mechanism_mesh/mesh_check.py --analyze-only
# =====================================================================

#SBATCH -J mesh_check
#SBATCH -o experiments/mechanism_mesh/logs/%x_%j.out
#SBATCH -e experiments/mechanism_mesh/logs/%x_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 20                     # 18 workers + main thread + headroom
#SBATCH --mem=24G
#SBATCH -t 48:0:0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bascomisaiah@gmail.com

set -euo pipefail

OPENFOAM_MODULE="OpenFOAM/v2506-foss-2025b"
set +u; module purge; module load "$OPENFOAM_MODULE"; source "$FOAM_BASH"; set -u
unset PYTHONPATH PYTHONHOME
PYTHON=/project/mwang/ibascom/envs/tc/bin/python

cd "$SLURM_SUBMIT_DIR"
mkdir -p experiments/mechanism_mesh/logs
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
WORK_ROOT="${TMPDIR:-/tmp/$USER}/mesh_check_work"

echo "=== MESH CHECK: host=$(hostname) cores=${SLURM_CPUS_PER_TASK:-?} work=$WORK_ROOT ==="
"$PYTHON" experiments/mechanism_mesh/mesh_check.py \
    --work_root "$WORK_ROOT" \
    --results_dir experiments/mechanism_mesh/results/main \
    --resume
echo "=== mesh check done -> experiments/mechanism_mesh/results/main ==="
