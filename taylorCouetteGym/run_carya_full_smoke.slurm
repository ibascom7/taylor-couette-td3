#!/bin/bash
# =====================================================================
# Carya: SMOKE TEST of the full 3D stepwise pipeline (full_tc_mixing_case).
#
# This is the QUICK validation run -- NOT a training run. It exists to answer,
# in ~10-15 minutes, two questions before you commit to a multi-hour job:
#   1. Does the whole pipeline work on Carya? (mesh, warmup, env step loop,
#      TD3 policy.train(), checkpoint save, AND frame capture.)
#   2. What is the REAL per-step wall time on Carya? (train.py prints dt=__s
#      every step -- use it to size run_carya_full_stepwise.slurm etc.)
#
# It deliberately runs only 2 tiny episodes (3 steps each -> truncates at the
# 4th step) with a SHORT 5 s warmup, so the numbers here are for validation
# only -- do NOT read any science off it. capture_episodes=1,final exercises
# the snapshot path on both episodes.
#
# Expect ~13 simulated seconds total (5 s warmup + 8 steps); at the wedge-scaled
# ~50 s/sim-s that's ~11 min. The first run also COMPILES the coded
# functionObjects (one-time, adds a couple minutes).
#
# Submit:   sbatch run_carya_full_smoke.slurm
# Then:     grep -o 'dt=[0-9.]*s' logs/tc_td3_full3d_smoke_*.out   # real timing
# =====================================================================

#SBATCH -J tc_td3_full3d_smoke
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH --mem=16G
#SBATCH -t 2:0:0                  # generous ceiling; real run is ~15 min
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bascomisaiah@gmail.com
# #SBATCH -A <your_allocation>
# #SBATCH -p <partition>

set -euo pipefail

# ---- run identity --------------------------------------------------
ALGO=td3
SEED=0
TAG=full3d_smoke_s${SEED}

# ---- geometry / normalizer for the full 3D case -------------------
R_IN=38.0
R_OUT=40.35
E_MAX_PER_STEP=0.0660901786861

# ---- tiny validation budget (NOT science) -------------------------
WARMUP_DURATION=5        # short: just enough to exercise the warmup path
MAX_STEPS_PER_EP=3       # -> 4 env steps per episode
MAX_TIMESTEPS=8          # -> 2 completed episodes ('final' = episode 2)
START_TIMESTEPS=1        # so policy.train() runs from step 2
SAVE_EVERY=2             # so a periodic checkpoint save also fires
CAPTURE_EPISODES=1,final

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
export PYTHONUNBUFFERED=1   # flush train.py prints live so progress/dt show up in the .out log

echo "=== full-3D SMOKE: algo=$ALGO seed=$SEED tag=$TAG ==="
echo "    case copy: $RUN_CASE  (Rin=$R_IN Rout=$R_OUT mm)"
echo "    NOTE: validation only -- watch the dt=__s values for real timing."
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

echo "=== full-3D SMOKE done -- pipeline OK; use dt=__s above to size real runs ==="
