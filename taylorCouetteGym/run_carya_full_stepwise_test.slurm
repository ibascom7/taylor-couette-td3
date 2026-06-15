#!/bin/bash
# =====================================================================
# Carya: TEST run of the STEPWISE (mixing) TD3 experiment on the FULL 3D
# annulus (full_tc_mixing_case), WITH ParaView frame capture.
#
# Single TD3 run that exercises the whole pipeline end-to-end and saves the
# OpenFOAM time directories of three episodes for offline ParaView videos:
#   - episode 1     (initial / untrained behavior)
#   - episode 20    (mid-training, where the policy starts to settle)
#   - episode final (converged behavior at the end of this run)
# This mirrors the capture pattern from the old constant-omega sweeps, now
# wired into the stepwise env too.
#
# Captured frames land in:  results/td3/<tag>/frames/ep0001, ep0020, ep0025
# Pack them for download with pack_frames.sh, then open the .foam files in
# ParaView (color by C for the dye, or U magnitude for the vortices).
#
# SIZING: a stepwise episode runs max_steps_per_ep+1 = 61 simulated seconds.
# MAX_TIMESTEPS=1525 -> exactly 25 completed episodes, so 'final' = episode 25
# (>20, so all three capture targets are distinct and DO complete/truncate).
#
# COST: ~30 s wall per simulated second on the coarse full mesh, so ~30 s per
# env step -> ~1525 steps is ~13 h plus a one-time ~7 min warmup. -t is set
# well above that. To shrink the test, lower MAX_STEPS_PER_EP and/or the
# episode count (keep MAX_TIMESTEPS a multiple of MAX_STEPS_PER_EP+1, and keep
# the middle capture target below the final episode).
#
# Submit:   sbatch run_carya_full_stepwise_test.slurm
# Monitor:  squeue -u $USER
# =====================================================================

#SBATCH -J tc_td3_full3d_test
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH --mem=16G
#SBATCH -t 24:0:0                 # ~13 h expected; generous ceiling
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bascomisaiah@gmail.com
# #SBATCH -A <your_allocation>
# #SBATCH -p <partition>

set -euo pipefail

# ---- run identity --------------------------------------------------
ALGO=td3
SEED=0
TAG=full3d_test_s${SEED}

# ---- geometry / normalizer for the full 3D case -------------------
R_IN=38.0
R_OUT=40.35
E_MAX_PER_STEP=0.0660901786861
# Warmup spin-up (cached once as 0.warmed). Dye reaches the bottom outlet by
# ~3 s on this geometry; 15 s gives a well-developed dye field before training.
WARMUP_DURATION=15

# ---- training hyperparameters --------------------------------------
MAX_STEPS_PER_EP=60      # -> 61 env steps per episode (see SIZING note above)
MAX_TIMESTEPS=1525       # -> 25 completed episodes ('final' = episode 25)
START_TIMESTEPS=120
SAVE_EVERY=200
CAPTURE_EPISODES=1,20,final

# =====================================================================
# OpenFOAM environment -- IDENTICAL to run_carya.slurm.
# ---------------------------------------------------------------------
OPENFOAM_MODULE="OpenFOAM/v2506-foss-2025b"
set +u
module purge
module load "$OPENFOAM_MODULE"
source "$FOAM_BASH"
set -u
unset PYTHONPATH PYTHONHOME

PYTHON=/project/mwang/ibascom/envs/tc/bin/python   # <-- match run_carya.slurm

# ---- per-task isolated case copy -----------------------------------
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
MASTER_CASE="taylor_couette_mixing/cases/full_tc_mixing_case"
RUN_ROOT="${SLURM_TMPDIR:-$SLURM_SUBMIT_DIR/_runs}/${ALGO}_${TAG}"
RUN_CASE="$RUN_ROOT/full_tc_mixing_case"
rm -rf "$RUN_ROOT"; mkdir -p "$RUN_ROOT"
cp -r "$MASTER_CASE" "$RUN_CASE"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "=== full-3D stepwise TEST: algo=$ALGO seed=$SEED tag=$TAG ==="
echo "    case copy: $RUN_CASE  (Rin=$R_IN Rout=$R_OUT mm)"
echo "    capture episodes: $CAPTURE_EPISODES  -> results/$ALGO/$TAG/frames/"
echo "    host: $(hostname)  cores: ${SLURM_CPUS_PER_TASK:-?}"

# ---- run -----------------------------------------------------------
"$PYTHON" train.py \
    --algo "$ALGO" \
    --env stepwise \
    --seed "$SEED" \
    --tag "$TAG" \
    --case_path "$RUN_CASE" \
    --r_in "$R_IN" \
    --r_out "$R_OUT" \
    --e_max_per_step "$E_MAX_PER_STEP" \
    --warmup_duration "$WARMUP_DURATION" \
    --max_steps_per_ep "$MAX_STEPS_PER_EP" \
    --max_timesteps "$MAX_TIMESTEPS" \
    --start_timesteps "$START_TIMESTEPS" \
    --save_every "$SAVE_EVERY" \
    --capture_episodes "$CAPTURE_EPISODES"

echo "=== full-3D stepwise TEST done ==="
