#!/bin/bash
# =====================================================================
# Carya: SHORT TD3 sweep -- a guaranteed-to-finish preview of
# run_carya_sweep.slurm, for when you need complete curves by a deadline.
#
# 3 configs (fits 3 concurrent slots so NONE queue behind the others) x
# 5000 steps (~7.4 h/config at ~5.3 s/step) -> all done in well under a
# normal workday. Run the full 4-config/12000-step run_carya_sweep.slurm
# afterwards for the real convergence test (same tags -> it OVERWRITES these
# dirs, so save/scp sweep_progress.png BEFORE launching the long run).
#
# Setup/deps/case-isolation are IDENTICAL to run_carya.slurm.
#
# Submit:   sbatch run_carya_sweep_short.slurm
# Monitor:  squeue -u $USER        Plot: python plot_sweep.py
# =====================================================================

#SBATCH -J tc_td3_sweepq          # 'q' = quick; distinct from the full sweep
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH --mem=16G
#SBATCH -t 16:0:0                 # generous; the ~5000-step run finishes far sooner
#SBATCH --array=0-2               # 3 TD3 configs (see mapping below)
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --mail-user=bascomisaiah@gmail.com
# #SBATCH -A <your_allocation>
# #SBATCH -p <partition>

set -euo pipefail

# ---- sweep grid: array index -> (tag, hyperparameters) -------------
#   index 0  base      -- control (expl 0.1, batch 256, episode len 60)
#   index 1  expl0.3   -- more exploration noise
#   index 2  ep120     -- longer horizon per episode (mixing has more time)
# (batch512 is omitted here to keep 3 concurrent slots; it's in the full sweep.)
TAGS=(   base   expl0.3  ep120 )
EXPL=(   0.1    0.3      0.1   )
BATCH=(  256    256      256   )
EP_LEN=( 60     60       120   )

i=${SLURM_ARRAY_TASK_ID:-0}
TAG=${TAGS[$i]}
EXPL_NOISE=${EXPL[$i]}
BATCH_SIZE=${BATCH[$i]}
MAX_STEPS_PER_EP=${EP_LEN[$i]}

# ---- training hyperparameters (shared) -----------------------------
ALGO=td3
SEED=0                  # fixed across configs -> differences are due to the knob
START_TIMESTEPS=1000    # same as the full sweep, so this is a clean truncated preview
SAVE_EVERY=500          # checkpoints + .npy logs re-written this often
MAX_TIMESTEPS=5000      # SHORT: ~83 episodes at ep_len 60, finishes with big margin

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

# ---- per-task isolated case copy (keyed by tag) --------------------
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
MASTER_CASE="taylor_couette_mixing/cases/tc_mixing_case"
RUN_ROOT="${SLURM_TMPDIR:-$SLURM_SUBMIT_DIR/_runs}/${ALGO}_${TAG}"
RUN_CASE="$RUN_ROOT/tc_mixing_case"
rm -rf "$RUN_ROOT"; mkdir -p "$RUN_ROOT"
cp -r "$MASTER_CASE" "$RUN_CASE"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "=== short sweep task $i: tag=$TAG expl=$EXPL_NOISE batch=$BATCH_SIZE ep_len=$MAX_STEPS_PER_EP steps=$MAX_TIMESTEPS ==="
echo "    case copy: $RUN_CASE"
echo "    host: $(hostname)  cores: ${SLURM_CPUS_PER_TASK:-?}"

# ---- run -----------------------------------------------------------
"$PYTHON" train.py \
    --algo "$ALGO" \
    --seed "$SEED" \
    --tag "$TAG" \
    --case_path "$RUN_CASE" \
    --max_timesteps "$MAX_TIMESTEPS" \
    --max_steps_per_ep "$MAX_STEPS_PER_EP" \
    --start_timesteps "$START_TIMESTEPS" \
    --expl_noise "$EXPL_NOISE" \
    --batch_size "$BATCH_SIZE" \
    --save_every "$SAVE_EVERY"

echo "=== short sweep task $i ($TAG) done ==="
