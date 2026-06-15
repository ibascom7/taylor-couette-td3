#!/bin/bash
# =====================================================================
# Carya: STEPWISE (mixing) TD3 on the FULL 3D annulus (full_tc_mixing_case).
#
# Same experiment as run_carya.slurm / run_carya_sweep_short.slurm, but on the
# full 360-degree Taylor-Couette geometry (Andras Horvath reference geometry,
# Rin=38 mm, Rout=40.35 mm, H=20 mm) instead of the 2D wedge. The agent picks a
# delta_omega every simulated second; reward = -(mixing index) - (energy).
#
# The ONLY differences vs the wedge scripts are the three geometry/normalizer
# flags passed to train.py:
#     --case_path <full case copy>   --r_in 38.0   --r_out 40.35
#     --e_max_per_step 0.0660901786861   (calibrated: 1 s @ 52.4 rad/s, see
#                                          _smoke_full_case.py / memory)
#
# NOTE ON COST: the coarse full mesh is ~9216 cells vs the wedge's ~800, so a
# 1 s pimpleFoam step is roughly an order of magnitude slower (~30 s/step here
# vs ~5 s on the wedge). MAX_TIMESTEPS and -t are sized accordingly; tune them.
#
# Submit:   sbatch run_carya_full_stepwise.slurm
# Monitor:  squeue -u $USER
# =====================================================================

#SBATCH -J tc_td3_full3d
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH --mem=16G
#SBATCH -t 48:0:0                 # generous; ~30 s/step on the coarse full mesh
#SBATCH --array=0-2               # 3 seeds (see mapping below)
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --mail-user=bascomisaiah@gmail.com
# #SBATCH -A <your_allocation>
# #SBATCH -p <partition>

set -euo pipefail

# ---- experiment grid: array index -> seed --------------------------
ALGO=td3
SEEDS=(0 1 2)
i=${SLURM_ARRAY_TASK_ID:-0}
SEED=${SEEDS[$i]}

# ---- geometry / normalizer for the full 3D case -------------------
R_IN=38.0
R_OUT=40.35
E_MAX_PER_STEP=0.0660901786861
# Warmup spin-up (cached once as 0.warmed). Dye reaches the bottom outlet by
# ~3 s on this geometry (Taylor-vortex axial transport >> mean throughflow);
# 15 s gives a well-developed dye field before training starts.
WARMUP_DURATION=15

# ---- training hyperparameters --------------------------------------
MAX_TIMESTEPS=2000     # ~17 h at 30 s/step; raise once you know your step cost
START_TIMESTEPS=120
SAVE_EVERY=200
MAX_STEPS_PER_EP=60

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
RUN_ROOT="${SLURM_TMPDIR:-$SLURM_SUBMIT_DIR/_runs}/${ALGO}_full3d_s${SEED}"
RUN_CASE="$RUN_ROOT/full_tc_mixing_case"
rm -rf "$RUN_ROOT"; mkdir -p "$RUN_ROOT"
cp -r "$MASTER_CASE" "$RUN_CASE"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "=== full-3D stepwise task $i: algo=$ALGO seed=$SEED ==="
echo "    case copy: $RUN_CASE  (Rin=$R_IN Rout=$R_OUT mm)"
echo "    host: $(hostname)  cores: ${SLURM_CPUS_PER_TASK:-?}"

# ---- run -----------------------------------------------------------
"$PYTHON" train.py \
    --algo "$ALGO" \
    --env stepwise \
    --seed "$SEED" \
    --tag "full3d_s${SEED}" \
    --case_path "$RUN_CASE" \
    --r_in "$R_IN" \
    --r_out "$R_OUT" \
    --e_max_per_step "$E_MAX_PER_STEP" \
    --warmup_duration "$WARMUP_DURATION" \
    --max_steps_per_ep "$MAX_STEPS_PER_EP" \
    --max_timesteps "$MAX_TIMESTEPS" \
    --start_timesteps "$START_TIMESTEPS" \
    --save_every "$SAVE_EVERY"

echo "=== full-3D stepwise task $i done ==="
