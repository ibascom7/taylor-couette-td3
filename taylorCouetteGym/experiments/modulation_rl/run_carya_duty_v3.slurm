#!/bin/bash
# =====================================================================
# Carya: duty_v3 -- TAU-BLOCK TD3 on the (T+, T-) duty env.
#
# Same env + trainer as duty_v2 (parallel_train_duty_v2.py /
# taylor_couette_duty_v2.py); ONE design change and TWO budget changes:
#
#   BLOCK = ONE RESIDENCE TIME (block_dt = 26 s = tau = V/Q). The v2
#   post-mortem: outlet conversion lags the wall by ~tau, so 10 s blocks
#   billed only ~15-20% of an action's effect to its own block -- the bias
#   that let 3/5 v2 seeds settle into effective-constant. With tau-blocks,
#   ~60-65% lands in-block and the rest in the NEXT block (one-step delay,
#   native to Q-learning) -- near-flux-mode credit while every signal stays
#   plant-measurable. Sampling at the dominant time constant is standard
#   process-control practice; state the granularity choice in the paper.
#   Block-averaging over 26 s also cuts outlet flutter ~sqrt(2.6).
#   T+/T- bounds stay [1,5]/[0,5] s DELIBERATELY UNSCALED: the period range
#   matches the statically-benchmarked waveform family (physics), not the
#   block bookkeeping; a block now holds >= 2.6 full periods.
#
#   RANDOM PHASE = 200 EPISODES (start_timesteps 1000 ~ 200 eps at the
#   mean 5-block horizon; the v5 split). Action repeat x2 (52 s holds = 2 tau).
#
# PRE-REGISTERED SUCCESS CRITERION (2026-08-05, amended same day BEFORE any
# v3 data existed): success = deterministic sustained last-block reward
# matches or beats the best measured static comparator (duty_v3_refs) within
# the measured noise floor. The converged waveform (T+, T-) is REPORTED, not
# preconditioned -- whether it lands at the idle-duration-hypothesis value
# (T- ~ 0.5 s) or somewhere else equally good is itself the finding. (The
# original gate required T- ~ 0.5 s; that baked an n=3 hypothesis into the
# success criterion of the experiment meant to test it independently.)
#
# Horizon stays RANDOMIZED U[3,7] blocks (mean 5 = the nominal episode;
# fixed horizons would leave late-t states ungrounded -- see the v2 docs).
#
# COST: 300 eps x mean 130 s = 2.6x the v2 sim-seconds; v2 ran ~10.5 h
# -> budget ~27-30 h, -t 36 h. Checkpoints every SAVE_EVERY grad steps
# keep an interrupted task usable.
#
# EVAL BASELINES CHANGE: block rewards are X-bar over 26 s windows; the
# constant / champion comparators re-run at 130 s locally
# (duty_v3_refs.py -> results/duty_v3_refs/) -- do NOT reuse the t50 table
# numbers for v3 verdicts. Trainer centering keeps 0.2390 (policy-invariant
# conditioning constant; exact value immaterial).
#
# Submit FROM taylorCouetteGym/ (paths are $SLURM_SUBMIT_DIR-relative):
#     cd /project/.../taylor-couette-td3/taylorCouetteGym
#     sbatch experiments/modulation_rl/run_carya_duty_v3.slurm
# Monitor: squeue -u $USER ; tail -f experiments/modulation_rl/logs/tc_td3_duty3*.out
# =====================================================================

#SBATCH -J tc_td3_duty3
#SBATCH -o experiments/modulation_rl/logs/%x_%A_%a.out
#SBATCH -e experiments/modulation_rl/logs/%x_%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48                     # whole Compute node per task: 44 workers + learner
#SBATCH --mem=8G
#SBATCH -t 36:0:0                 # 2.6x v2 sim-seconds (~27-30 h expected)
#SBATCH --array=0-4               # one task per SEED: five decorrelated seeds
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --mail-user=bascomisaiah@gmail.com

set -euo pipefail

# ---- run config -----------------------------------------------------
ALGO=td3
N_WORKERS=44
SEED=${SLURM_ARRAY_TASK_ID:-0}
TAG="duty_v3_s${SEED}"

# ---- tau-block design (see header) ----------------------------------
W_B_RPM=300
BLOCK_DT=26                       # ONE residence time (tau = V/Q)
BLOCKS_MIN=3                      # horizon ~ U[3,7] blocks (mean 5 = 130 s episodes)
BLOCKS_MAX=7
T_PLUS_MIN=1.0                    # UNSCALED bounds: the benchmarked waveform family
T_PLUS_MAX=5.0
T_MINUS_MIN=0.0
T_MINUS_MAX=5.0
W_HI_CAP_RPM=2500                 # guard rail (calibrated envelope); never engages
RAMP_TIME=0.05
P_MAX_WATT=31.94
T_SCALE=26                        # tau [s]; with tau-blocks the time obs counts blocks
X_INIT=0.353                      # warmed steady conversion
REWARD_MODE=conv
REWARD_BASELINE=0.2390            # centering constant (policy-invariant; see header)
WARM_DURATION=60                  # one-time template spin-up (~2.3 tau, ~15 min)

# ---- budgets: v5-style 200-episode random phase ---------------------
MAX_EPISODES=300
MAX_TIMESTEPS=1800
START_TIMESTEPS=1000              # ~200 random episodes at the mean 5-block horizon
RANDOM_ACTION_REPEAT=2            # 52 s holds = 2 tau of sustained evidence
EXPL_NOISE=0.1
BATCH_SIZE=64
GRAD_PER_STEP=64                  # UTD=64 (surrogate-validated recipe)
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
WORKER_ROOT="${TMPDIR:-/tmp/$USER}/duty_v3_workers_${TAG}"

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "=== parallel TD3 (duty_v3, tau-block (T+,T-) warmed continuous ops) task ${SLURM_ARRAY_TASK_ID:-0}: seed=$SEED workers=$N_WORKERS ==="
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
