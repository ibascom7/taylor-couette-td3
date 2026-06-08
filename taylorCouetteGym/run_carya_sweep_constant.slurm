#!/bin/bash
# =====================================================================
# Carya: CONSTANT-OMEGA sweep -- the second system, where TD3 picks ONE
# absolute omega per episode and is scored on how well that single
# constant angular velocity mixes (final I) vs the energy it burns.
#
# Each env.step() here is a WHOLE episode: it hard-resets the case and
# runs the chosen omega for --episode_duration seconds in one pimpleFoam
# call. So one "timestep" costs ~one stepwise episode of wall time, and
# MAX_TIMESTEPS is therefore ~60x smaller than the stepwise sweep.
#
# Because every trial hard-resets, reward depends only on the chosen
# omega -> this is a 1-D black-box optimizer. discount=0 makes it a clean
# one-shot omega search (no bootstrap across independent trials).
#
# Setup/deps/case-isolation are IDENTICAL to run_carya_sweep_short.slurm.
#
# Submit:   sbatch run_carya_sweep_constant.slurm
# Monitor:  squeue -u $USER        Plot: python plot_sweep.py
# =====================================================================

#SBATCH -J tc_td3_const           # 'const' = constant-omega system
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH --mem=16G
#SBATCH -t 40:0:0                 # generous; constant episodes run a full sim each
                                  # (dur120 is ~2x dur60), -t is just the ceiling
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
TAGS=(   const_base  const_expl0.3  const_dur120 )
EXPL=(   0.1         0.3            0.1          )
EP_DUR=( 60          60             120          )

i=${SLURM_ARRAY_TASK_ID:-0}
TAG=${TAGS[$i]}
EXPL_NOISE=${EXPL[$i]}
EPISODE_DURATION=${EP_DUR[$i]}

# ---- training hyperparameters (shared) -----------------------------
ALGO=td3
SEED=0
DISCOUNT=0.0            # bandit: no bootstrap across independent trials
BATCH_SIZE=256
START_TIMESTEPS=20      # random omega exploration before the policy takes over
SAVE_EVERY=20           # checkpoints + .npy logs re-written this often
MAX_TIMESTEPS=120       # 120 episodes (one constant-omega trial each)
CAPTURE_EPISODES=1,5,final   # save time dirs of these episodes for ParaView video

# =====================================================================
# OpenFOAM environment -- IDENTICAL to run_carya_sweep_short.slurm.
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
MASTER_CASE="taylor_couette_mixing/cases/tc_mixing_case"
RUN_ROOT="${SLURM_TMPDIR:-$SLURM_SUBMIT_DIR/_runs}/${ALGO}_${TAG}"
RUN_CASE="$RUN_ROOT/tc_mixing_case"
rm -rf "$RUN_ROOT"; mkdir -p "$RUN_ROOT"
cp -r "$MASTER_CASE" "$RUN_CASE"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "=== constant sweep task $i: tag=$TAG expl=$EXPL_NOISE dur=${EPISODE_DURATION}s steps=$MAX_TIMESTEPS ==="
echo "    case copy: $RUN_CASE"
echo "    host: $(hostname)  cores: ${SLURM_CPUS_PER_TASK:-?}"

# ---- run -----------------------------------------------------------
"$PYTHON" train.py \
    --algo "$ALGO" \
    --env constant \
    --seed "$SEED" \
    --tag "$TAG" \
    --case_path "$RUN_CASE" \
    --episode_duration "$EPISODE_DURATION" \
    --capture_episodes "$CAPTURE_EPISODES" \
    --max_timesteps "$MAX_TIMESTEPS" \
    --start_timesteps "$START_TIMESTEPS" \
    --expl_noise "$EXPL_NOISE" \
    --batch_size "$BATCH_SIZE" \
    --discount "$DISCOUNT" \
    --save_every "$SAVE_EVERY"

echo "=== constant sweep task $i ($TAG) done ==="
