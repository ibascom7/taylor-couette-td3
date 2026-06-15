#!/bin/bash
# =====================================================================
# Carya: CONSTANT-OMEGA TD3 on the FULL 3D annulus (full_tc_mixing_case).
#
# Same experiment as run_carya_sweep_constant.slurm (the agent picks ONE
# absolute omega per episode and is scored on final mixing vs total energy --
# a 1-D black-box optimizer), but on the full 360-degree geometry instead of
# the 2D wedge. Each env.step() is a WHOLE episode: hard-reset + one pimpleFoam
# run of --episode_duration seconds.
#
# The ONLY differences vs the wedge script are the geometry/normalizer flags:
#     --case_path <full case copy>   --r_in 38.0   --r_out 40.35
#     --e_max_per_step 0.0660901786861   (calibrated: 1 s @ 52.4 rad/s)
#
# COST: ~30 s per simulated second on the coarse full mesh, so a 60 s episode
# is ~30 min of wall time. -t is sized for that; tune MAX_TIMESTEPS to taste.
#
# Submit:   sbatch run_carya_full_constant.slurm
# Monitor:  squeue -u $USER        Plot: python plot_sweep.py
# =====================================================================

#SBATCH -J tc_td3_full3d_const
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH --mem=16G
#SBATCH -t 48:0:0
#SBATCH --array=0-2               # 3 configs (see mapping below)
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --mail-user=bascomisaiah@gmail.com
# #SBATCH -A <your_allocation>
# #SBATCH -p <partition>

set -euo pipefail

# ---- sweep grid: array index -> (tag, hyperparameters) -------------
#   index 0  base      -- control (expl 0.1, episode_duration 60 s)
#   index 1  expl0.3   -- more exploration over the omega axis
#   index 2  dur120    -- give each constant omega twice as long to mix
TAGS=(   full3d_const_base  full3d_const_expl0.3  full3d_const_dur120 )
EXPL=(   0.1                0.3                   0.1                 )
EP_DUR=( 60                 60                    120                 )

i=${SLURM_ARRAY_TASK_ID:-0}
TAG=${TAGS[$i]}
EXPL_NOISE=${EXPL[$i]}
EPISODE_DURATION=${EP_DUR[$i]}

# ---- geometry / normalizer for the full 3D case -------------------
R_IN=38.0
R_OUT=40.35
E_MAX_PER_STEP=0.0660901786861
# Warmup spin-up (cached once as 0.warmed). Dye reaches the bottom outlet by
# ~3 s on this geometry (Taylor-vortex axial transport >> mean throughflow);
# 15 s gives a well-developed dye field before training starts.
WARMUP_DURATION=15

# ---- training hyperparameters (shared) -----------------------------
ALGO=td3
SEED=0
DISCOUNT=0.0            # bandit: no bootstrap across independent trials
BATCH_SIZE=256
START_TIMESTEPS=20
SAVE_EVERY=20
MAX_TIMESTEPS=120
CAPTURE_EPISODES=1,5,final

# =====================================================================
# OpenFOAM environment -- IDENTICAL to run_carya_sweep_constant.slurm.
# ---------------------------------------------------------------------
OPENFOAM_MODULE="OpenFOAM/v2506-foss-2025b"
set +u
module purge
module load "$OPENFOAM_MODULE"
source "$FOAM_BASH"
set -u
unset PYTHONPATH PYTHONHOME

PYTHON=/project/mwang/ibascom/envs/tc/bin/python   # <-- match run_carya.slurm

# ---- per-task isolated case copy (keyed by tag) --------------------
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
MASTER_CASE="taylor_couette_mixing/cases/full_tc_mixing_case"
RUN_ROOT="${SLURM_TMPDIR:-$SLURM_SUBMIT_DIR/_runs}/${ALGO}_${TAG}"
RUN_CASE="$RUN_ROOT/full_tc_mixing_case"
rm -rf "$RUN_ROOT"; mkdir -p "$RUN_ROOT"
cp -r "$MASTER_CASE" "$RUN_CASE"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "=== full-3D constant task $i: tag=$TAG expl=$EXPL_NOISE dur=${EPISODE_DURATION}s ==="
echo "    case copy: $RUN_CASE  (Rin=$R_IN Rout=$R_OUT mm)"
echo "    host: $(hostname)  cores: ${SLURM_CPUS_PER_TASK:-?}"

# ---- run -----------------------------------------------------------
"$PYTHON" train.py \
    --algo "$ALGO" \
    --env constant \
    --seed "$SEED" \
    --tag "$TAG" \
    --case_path "$RUN_CASE" \
    --r_in "$R_IN" \
    --r_out "$R_OUT" \
    --e_max_per_step "$E_MAX_PER_STEP" \
    --warmup_duration "$WARMUP_DURATION" \
    --episode_duration "$EPISODE_DURATION" \
    --capture_episodes "$CAPTURE_EPISODES" \
    --max_timesteps "$MAX_TIMESTEPS" \
    --start_timesteps "$START_TIMESTEPS" \
    --expl_noise "$EXPL_NOISE" \
    --batch_size "$BATCH_SIZE" \
    --discount "$DISCOUNT" \
    --save_every "$SAVE_EVERY"

echo "=== full-3D constant task $i ($TAG) done ==="
