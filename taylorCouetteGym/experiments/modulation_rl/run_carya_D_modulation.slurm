#!/bin/bash
# =====================================================================
# Carya: PARALLELIZED TD3 over the D-ONLY duty-cycle env -- the
# INTERPRETABILITY run (Zhou & Zhu-style one-knob state-action map).
#
# Env: TaylorCouetteDutyEnv (taylor_couette_mixing/envs/taylor_couette_duty.py)
# on side_outlet_grad_case. Everything the v5 design left free is PINNED to
# what the converged v5-s2 policy chose: w_low = 0, T = 5 s, fixed mean
# w_b = 300. ONE action dim: duty D in [0.2, 1] (w_hi = 300/D <= 1500 rpm,
# inside the proven 2500 envelope). TWO obs dims: (X_block, t/tau) with
# tau = 26 s the residence time -- a UNITS choice, not episode information.
#
# CONTINUOUS-MANUFACTURING contract (all three differ from run_carya_modulation):
#   - WARMED episodes: template prep spins 300 rpm x 60 s once per node
#     (~15 min, cached on the node-local worker_root) and caches 0.warmed/;
#     every episode starts from the steady operating state X = 0.353.
#   - RANDOMIZED horizon: 3-7 blocks per episode (mean 5 = the v5 cost), so
#     no t value signals termination and late-t bootstraps stay grounded.
#   - CONTINUING task: the trainer bootstraps THROUGH truncation (done=0) --
#     deliberate opposite of the v5 fixed-horizon handling; see
#     parallel_train_duty.py's docstring before "fixing" either direction.
#
# Reward: conv mode (X_block - P/31.94), benchmark-comparable. The flux mode
# (mass-balance, no credit delay) is one flag away if learning stalls on the
# tau ~ 2.6-block delay: add --reward_mode flux to the python call.
#
# COST: v3 measured 8.5 h for 308 eps / 44 workers with bursts <= 500 rpm.
# Here random-phase episodes burst to 1500 rpm at the duty floor (smaller dt
# while bursting; E[300/D] ~ 600 rpm under uniform D) -> budget ~1.5-2x:
# -t 20 h. Checkpoints + replay_buffer.npz every SAVE_EVERY grad steps make
# an interrupted task a usable dataset.
#
# Submit FROM taylorCouetteGym/ (paths are $SLURM_SUBMIT_DIR-relative; from
# anywhere else every task dies in ~20 s -- learned 2026-07-31):
#     cd /project/.../taylor-couette-td3/taylorCouetteGym
#     sbatch experiments/modulation_rl/run_carya_D_modulation.slurm
# Fewer/more seeds at submit time WITHOUT editing this file:
#     sbatch --array=0 experiments/modulation_rl/run_carya_D_modulation.slurm
# Monitor: squeue -u $USER ; tail -f experiments/modulation_rl/logs/*.out
# =====================================================================

#SBATCH -J tc_td3_duty
#SBATCH -o experiments/modulation_rl/logs/%x_%A_%a.out
#SBATCH -e experiments/modulation_rl/logs/%x_%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48                     # whole Compute node per task: 44 workers + learner
#SBATCH --mem=8G
#SBATCH -t 20:0:0                 # ~8.5 h (v3) x 1.5-2 for the 1500 rpm bursts
#SBATCH --array=0-2               # one task per SEED. Three seeds by default; the
                                  #  1-D action space should converge far more
                                  #  reliably than the 3-D one (surrogate: UTD=64
                                  #  failures were critic-resolution pathologies of
                                  #  the FLAT 3-D landscape). Trim with sbatch --array=0.
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --mail-user=bascomisaiah@gmail.com

set -euo pipefail

# ---- run config -----------------------------------------------------
ALGO=td3
N_WORKERS=44
SEED=${SLURM_ARRAY_TASK_ID:-0}
TAG="duty_v1_s${SEED}"

# ---- the settled D-only design (taylor_couette_duty.py docstring) ---
W_B_RPM=300
BLOCK_DT=10
BLOCKS_MIN=3                      # horizon ~ U[3,7] blocks (mean 5 = v5 cost)
BLOCKS_MAX=7
DUTY_MIN=0.2
DUTY_MAX=1.0
PERIOD=5                          # pinned; divides block_dt -> exact block means
RAMP_TIME=0.05
P_MAX_WATT=31.94
T_SCALE=26                        # tau = V/Q [s]; a units choice for the time obs
X_INIT=0.353                      # warmed steady conversion (measured 2026-07-31)
REWARD_MODE=conv
WARM_DURATION=60                  # one-time template spin-up (~2.3 tau, ~15 min)

# ---- budgets: the v5 surrogate-validated learner recipe carries over ----
MAX_EPISODES=300
MAX_TIMESTEPS=1800
START_TIMESTEPS=500               # ~100 fully-random episodes: the (D -> X, P)
                                  #  landscape probe doubles as sweep data
EXPL_NOISE=0.2                    # raw std 0.2 ~ 0.08 duty
BATCH_SIZE=64
GRAD_PER_STEP=64                  # UTD=64 (surrogate-validated, v5-proven)
SAVE_EVERY=4000

# =====================================================================
# OpenFOAM + python env -- IDENTICAL recipe to the other RL slurms.
# =====================================================================
OPENFOAM_MODULE="OpenFOAM/v2506-foss-2025b"
set +u; module purge; module load "$OPENFOAM_MODULE"; source "$FOAM_BASH"; set -u
unset PYTHONPATH PYTHONHOME
PYTHON=/project/mwang/ibascom/envs/tc/bin/python

cd "$SLURM_SUBMIT_DIR"
EXP_DIR="experiments/modulation_rl"
RESULTS_DIR="$EXP_DIR/results"
mkdir -p "$EXP_DIR/logs"

MASTER_CASE="taylor_couette_mixing/cases/side_outlet_grad_case"
WORKER_ROOT="${TMPDIR:-/tmp/$USER}/duty_workers_${TAG}"

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "=== parallel TD3 (D-only duty, warmed continuous ops) task ${SLURM_ARRAY_TASK_ID:-0}: seed=$SEED workers=$N_WORKERS ==="
echo "    tag=$TAG  master=$MASTER_CASE  worker_root=$WORKER_ROOT  host=$(hostname)  cores=${SLURM_CPUS_PER_TASK:-?}"

"$PYTHON" "$EXP_DIR/parallel_train_duty.py" \
    --algo "$ALGO" \
    --seed "$SEED" \
    --n_workers "$N_WORKERS" \
    --case_path "$MASTER_CASE" \
    --worker_root "$WORKER_ROOT" \
    --w_b_rpm "$W_B_RPM" \
    --block_dt "$BLOCK_DT" \
    --blocks_min "$BLOCKS_MIN" --blocks_max "$BLOCKS_MAX" \
    --duty_min "$DUTY_MIN" --duty_max "$DUTY_MAX" \
    --period "$PERIOD" \
    --ramp_time "$RAMP_TIME" \
    --p_max_watt "$P_MAX_WATT" \
    --t_scale "$T_SCALE" \
    --x_init "$X_INIT" \
    --reward_mode "$REWARD_MODE" \
    --warm_duration "$WARM_DURATION" \
    --max_episodes "$MAX_EPISODES" \
    --max_timesteps "$MAX_TIMESTEPS" \
    --start_timesteps "$START_TIMESTEPS" \
    --expl_noise "$EXPL_NOISE" \
    --batch_size "$BATCH_SIZE" \
    --grad_per_step "$GRAD_PER_STEP" \
    --save_every "$SAVE_EVERY" \
    --tag "$TAG" \
    --results_dir "$RESULTS_DIR"

echo "=== done. checkpoints + logs -> $RESULTS_DIR/$ALGO/$TAG ==="
