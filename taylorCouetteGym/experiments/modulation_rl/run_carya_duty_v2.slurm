#!/bin/bash
# =====================================================================
# Carya: PARALLELIZED TD3 over the (T+, T-) duty env -- duty_v2.
#
# Env: TaylorCouetteDutyV2Env (taylor_couette_mixing/envs/taylor_couette_duty_v2.py)
# on side_outlet_grad_case. TWO action dims put the physics variable on a raw
# axis: T_plus in [1, 5] s (burst), T_minus in [0, 5] s (idle);
# T = T+ + T-, D = T+/T, w_hi solved for an EXACT commanded block mean of
# 300 rpm everywhere (peaks at exactly 1500 rpm in the deep-idle corner --
# v1's proven ceiling, inside the 2500 rpm power-model calibration).
# TWO obs dims: (X_block, t/tau), tau = 26 s. The
# duty_diag idle-duration law predicts the converged policy cares about
# T- (~0.5 s) and ignores T+ -- an axis-aligned target for distillation.
#
# duty_v1 post-mortem fixes baked into parallel_train_duty_v2.py:
#   - reward centering (buffer only; logs raw): baseline 0.2390 = warmed
#     constant-300 R50- (t50_table.py)
#   - random-phase action repeat x3 (sustained waveform values enter buffer)
#   - expl_noise 0.1 (v1's 0.2 blurred the 0.5 s idle optimum into the cliff)
#   - worker seeds offset by seed*1000 (real replications, not shared phases)
# Everything else carries the settled duty_v1 recipe: warmed episodes,
# randomized 3-7-block horizon (mean 5 = the nominal 50 s episode),
# continuing-task bootstrap, UTD=64, 300 episodes.
#
# COST: duty_v1 fit inside 20 h at 44 workers with the same burst ceiling
# (1500 rpm) and block economics; same 20 h budget.
#
# Submit FROM taylorCouetteGym/ (paths are $SLURM_SUBMIT_DIR-relative; from
# anywhere else every task dies in ~20 s):
#     cd /project/.../taylor-couette-td3/taylorCouetteGym
#     sbatch experiments/modulation_rl/run_carya_duty_v2.slurm
# Fewer seeds at submit time: sbatch --array=0-1 experiments/modulation_rl/run_carya_duty_v2.slurm
# Monitor: squeue -u $USER ; tail -f experiments/modulation_rl/logs/tc_td3_duty2*.out
# =====================================================================

#SBATCH -J tc_td3_duty2
#SBATCH -o experiments/modulation_rl/logs/%x_%A_%a.out
#SBATCH -e experiments/modulation_rl/logs/%x_%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48                     # whole Compute node per task: 44 workers + learner
#SBATCH --mem=8G
#SBATCH -t 20:0:0
#SBATCH --array=0-4               # one task per SEED: five decorrelated seeds
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --mail-user=bascomisaiah@gmail.com

set -euo pipefail

# ---- run config -----------------------------------------------------
ALGO=td3
N_WORKERS=44
SEED=${SLURM_ARRAY_TASK_ID:-0}
TAG="duty_v2_s${SEED}"

# ---- the (T+, T-) design (taylor_couette_duty_v2.py docstring) ------
W_B_RPM=300
BLOCK_DT=10
BLOCKS_MIN=3                      # horizon ~ U[3,7] blocks (mean 5 = 50 s episodes)
BLOCKS_MAX=7
T_PLUS_MIN=1.0                    # 1 s floor -> deep-idle corner solves to exactly
                                  #  1500 rpm (proven ceiling; no model extrapolation)
T_PLUS_MAX=5.0
T_MINUS_MIN=0.0
T_MINUS_MAX=5.0
W_HI_CAP_RPM=2500                 # guard rail (calibrated envelope); never engages
RAMP_TIME=0.05
P_MAX_WATT=31.94
T_SCALE=26                        # tau = V/Q [s]; a units choice for the time obs
X_INIT=0.353                      # warmed steady conversion
REWARD_MODE=conv
REWARD_BASELINE=0.2390            # warmed constant-300 R50- (t50_table.py); buffer-only
WARM_DURATION=60                  # one-time template spin-up (~2.3 tau, ~15 min)

# ---- budgets: v5/duty_v1 learner recipe + the v2 robustness fixes ---
MAX_EPISODES=300
MAX_TIMESTEPS=1800
START_TIMESTEPS=500               # ~100 random episodes; with repeat x3 the
                                  #  random phase is a randomized STATIC sweep
RANDOM_ACTION_REPEAT=3
EXPL_NOISE=0.1                    # raw std ~ 0.22 s in T+, 0.25 s in T-
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
WORKER_ROOT="${TMPDIR:-/tmp/$USER}/duty_v2_workers_${TAG}"

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "=== parallel TD3 (duty_v2, (T+,T-) warmed continuous ops) task ${SLURM_ARRAY_TASK_ID:-0}: seed=$SEED workers=$N_WORKERS ==="
echo "    tag=$TAG  master=$MASTER_CASE  worker_root=$WORKER_ROOT  host=$(hostname)  cores=${SLURM_CPUS_PER_TASK:-?}"

"$PYTHON" "$EXP_DIR/parallel_train_duty_v2.py" \
    --algo "$ALGO" \
    --seed "$SEED" \
    --n_workers "$N_WORKERS" \
    --case_path "$MASTER_CASE" \
    --worker_root "$WORKER_ROOT" \
    --w_b_rpm "$W_B_RPM" \
    --block_dt "$BLOCK_DT" \
    --blocks_min "$BLOCKS_MIN" --blocks_max "$BLOCKS_MAX" \
    --t_plus_min "$T_PLUS_MIN" --t_plus_max "$T_PLUS_MAX" \
    --t_minus_min "$T_MINUS_MIN" --t_minus_max "$T_MINUS_MAX" \
    --w_hi_cap_rpm "$W_HI_CAP_RPM" \
    --ramp_time "$RAMP_TIME" \
    --p_max_watt "$P_MAX_WATT" \
    --t_scale "$T_SCALE" \
    --x_init "$X_INIT" \
    --reward_mode "$REWARD_MODE" \
    --reward_baseline "$REWARD_BASELINE" \
    --warm_duration "$WARM_DURATION" \
    --max_episodes "$MAX_EPISODES" \
    --max_timesteps "$MAX_TIMESTEPS" \
    --start_timesteps "$START_TIMESTEPS" \
    --random_action_repeat "$RANDOM_ACTION_REPEAT" \
    --expl_noise "$EXPL_NOISE" \
    --batch_size "$BATCH_SIZE" \
    --grad_per_step "$GRAD_PER_STEP" \
    --save_every "$SAVE_EVERY" \
    --tag "$TAG" \
    --results_dir "$RESULTS_DIR"

echo "=== done. checkpoints + logs -> $RESULTS_DIR/$ALGO/$TAG ==="
