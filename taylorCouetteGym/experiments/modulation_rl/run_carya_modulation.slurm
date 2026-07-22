#!/bin/bash
# =====================================================================
# Carya: PARALLELIZED TD3 over block-wise waveform modulation (fixed-mean w_b).
#
# N=44 env workers on a full 48-core Compute node, each with its own OpenFOAM
# case copy, run pimpleFoam rollouts CONCURRENTLY (threads) and feed ONE shared
# replay buffer; a single TD3 learner trains on the pooled experience -- the
# same harness as parallelized_catalysis_rl / parallelized_graded_rl, but the
# env is TaylorCouetteModulationEnv on side_outlet_grad_case (the RL twin of
# side_outlet_case_sc1075_graded: 25 um wall cells, Sc=1075, resolved film):
#
#   - 50 s episodes of 5 x 10 s control blocks, from the PRISTINE pre-filled IC
#     (NO warmup -- episodes are directly comparable to the static-grid
#     benchmarks in experiments/modulation_rl/README.md).
#   - action per block (the ORIGINAL spec box): (duty D in [0.6,1], trough
#     w_low in [0,300] rpm, period T in [0.5,5] s log-mapped); the burst is
#     CONVERTED from the trough via the fixed-mean constraint
#     w_hi = (w_b - (1-D)*w_low)/D, w_b = 300 -> commanded mean = 300 always,
#     w_hi <= 500 rpm (proven stable; the sweep ran 2500 rpm clean).
#   - reward per block: X_block - P_block/31.94 W (block-avg conversion minus
#     normalized motor power on the commanded wave).
#
# HISTORY (compressed -- full trail in the README v5 section): s0/v2/v3/v4 all
# corner-pinned at raw (-1,+1,+1), each ending BELOW its own random baseline.
# Two causes found 2026-07-21: (1) terminal-bootstrap bug -- done stored as 0
# always, critic bootstrapped into ungrounded clock=1 states (probed Q=0.78 vs
# true 0) -- FIXED in parallel_train.py; (2) the fixed-mean design flattened
# the reward landscape to ~0.015, below critic resolving power at ~1.5k
# transitions, so seed-0 critic-init noise steered the actor into tanh
# saturation. DECISION: the action space STAYS fixed-mean w_b=300 (the
# failures are learner pathologies) -- the fix happens in the LEARNER,
# validated on the local surrogate before any more Carya time.
#
# SURROGATE VALIDATION (2026-07-22, experiments/modulation_rl/surrogate/):
# UTD sweep on 10 seeds each -- pass = within 0.01 of the grid-scan optimum:
#   UTD=1 (old): 0/10, all pinned    UTD=32: 5-6/10    UTD=64: 7/10 (+1 at
#   0.950)    UTD=128: 5/10 (worse -- overfits each incoming batch).
# GRAD_PER_STEP=64 chosen. Seed 0 PASSES at UTD=64 on the surrogate (0.971).
# Residual known risk (~2/10): the constant-300 trap (w_low->300 or D->1 in
# the log). It declares itself within the first ~50 policy episodes -- if the
# post-random log shows wlo pinned high with returns sagging below the random
# mean (~0.87), scancel early; the buffer (replay_buffer.npz) and the 200
# random episodes are a keeper dataset either way.
#
# COST: v3 measured 308 episodes / 44 workers = 8.5 h wall PER SEED. Seeds run
# as a JOB ARRAY (one whole node per task, like run_carya_parallel_graded's
# energy-weight sweep) -> all three run IN PARALLEL, ~8.5 h each + ~2 min
# template compile per task. -t 12 h covers a task; replay_buffer.npz +
# SAVE_EVERY checkpoints make even an interrupted task a usable dataset.
#
# Submit:  sbatch experiments/modulation_rl/run_carya_modulation.slurm
# Monitor: squeue -u $USER ; tail -f experiments/modulation_rl/logs/*.out
# =====================================================================

#SBATCH -J tc_td3_modulation
#SBATCH -o experiments/modulation_rl/logs/%x_%A_%a.out
#SBATCH -e experiments/modulation_rl/logs/%x_%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48                     # whole Compute node PER TASK: 44 workers + learner/OS headroom
#SBATCH --mem=8G                  # 44 solvers on the 3330-cell graded mesh are tiny
                                  #  (measured ~1.9 GB peak for 44 comparable solvers)
#SBATCH -t 12:0:0                 # ~8.5 h measured (v3) per seed; margin for stragglers
#SBATCH --array=0-2               # one task per SEED (three independent trainings in
                                  #  parallel, one whole node each -- the same pattern
                                  #  as run_carya_parallel_graded's ew sweep)
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --mail-user=bascomisaiah@gmail.com

set -euo pipefail

# ---- run config -----------------------------------------------------
ALGO=td3
N_WORKERS=44                     # 48-core node: leave ~4 cores for the learner + OS
# array index -> SEED (one whole-node task each). At ~75% surrogate pass-rate
# per seed, three rolls give ~98% that at least one lands; seed 0 passes on
# the surrogate at UTD=64 (0.971). Keep --array = 0-(n_seeds-1).
SEED=${SLURM_ARRAY_TASK_ID:-0}
TAG="mod_wb300_v5_s${SEED}"

# ---- the settled design (experiments/modulation_rl/README.md) -------
# Fixed for this run: w_b=300, 50 s episodes, 10 s blocks, reward weights
# (1,1), P_max=31.94 W. The ORIGINAL spec action box: trough w_low free in
# [0, w_b] (--idle_max_rpm unset -> defaults to w_b).
W_B_RPM=300
EPISODE_DURATION=50
BLOCK_DT=10
DUTY_MIN=0.6
DUTY_MAX=1.0
IDLE_MIN_RPM=0
PERIOD_MIN=0.5                   # T floor: below it vortices low-pass the forcing
PERIOD_MAX=5
RAMP_TIME=0.05
P_MAX_WATT=31.94                 # motor power at 2500 rpm (global reward normalizer)
WALLFLUX_MAX=1.32e-8             # obs normalizer: steady resolved wallFlux at 2500 rpm

# ---- budgets (PLACEHOLDERS -- finalized with the surrogate-validated ------
# ---- learner recipe; see the GATE in the header) --------------------------
# 300 episodes = 200 random (landscape probe of the free-mean box; also the
# surrogate's training data) + 100 policy. The random phase alone is valuable:
# it samples (D, w_hi, T) -> (X, P) across the whole interesting space.
MAX_EPISODES=300
MAX_TIMESTEPS=1800
START_TIMESTEPS=1000             # 200 fully-random episodes before the policy kicks in
EXPL_NOISE=0.2                   # per raw dim; ~30 rpm std on w_low, ~0.04 on duty,
                                 #  ~1/3 octave on T
BATCH_SIZE=64
GRAD_PER_STEP=64                 # UTD: gradient steps per collected transition. THE
                                 #  surrogate-validated fix (2026-07-22): at the old 1.0
                                 #  the critic never fit its data and the actor pinned a
                                 #  bound corner on 10/10 surrogate seeds; UTD>=64 passes
                                 #  most seeds. FREE wall-clock: the learner idles while
                                 #  44 pimpleFoam run; 1500x64=96k grad steps ~ minutes.
SAVE_EVERY=4000                  # checkpoint + log snapshot every N grad steps
                                 #  (at UTD=64 that is every ~62 collected transitions)

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

MASTER_CASE="taylor_couette_mixing/cases/side_outlet_grad_case"   # RL-drivable twin
# NODE-LOCAL SSD for the 44 worker cases: 44 concurrent pimpleFoam do a lot of
# small time-dir I/O, far faster on the node's local SSD than the shared FS.
# Each array task has its own node (and TMPDIR), so no cross-seed sharing.
WORKER_ROOT="${TMPDIR:-/tmp/$USER}/mod_workers_${TAG}"

export OMP_NUM_THREADS=1          # each pimpleFoam is serial; don't let libs grab extra cores
export PYTHONUNBUFFERED=1

echo "=== parallel TD3 (block modulation, fixed-mean w_b=${W_B_RPM}rpm) task ${SLURM_ARRAY_TASK_ID:-0}: seed=$SEED workers=$N_WORKERS ==="
echo "    tag=$TAG  master=$MASTER_CASE  worker_root=$WORKER_ROOT  host=$(hostname)  cores=${SLURM_CPUS_PER_TASK:-?}"

"$PYTHON" "$EXP_DIR/parallel_train.py" \
    --algo "$ALGO" \
    --seed "$SEED" \
    --n_workers "$N_WORKERS" \
    --case_path "$MASTER_CASE" \
    --worker_root "$WORKER_ROOT" \
    --w_b_rpm "$W_B_RPM" \
    --episode_duration "$EPISODE_DURATION" \
    --block_dt "$BLOCK_DT" \
    --duty_min "$DUTY_MIN" --duty_max "$DUTY_MAX" \
    --idle_min_rpm "$IDLE_MIN_RPM" \
    --period_min "$PERIOD_MIN" --period_max "$PERIOD_MAX" \
    --ramp_time "$RAMP_TIME" \
    --p_max_watt "$P_MAX_WATT" \
    --wallflux_max "$WALLFLUX_MAX" \
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
