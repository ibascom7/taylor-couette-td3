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
#   - action per block: (duty D in [0.6,1], idle w_low in [0,300] rpm,
#     period T in [0.5,5] s log-mapped); burst speed SOLVED from the fixed-mean
#     constraint w_hi = (w_b - (1-D)*w_low)/D  ->  commanded mean = w_b always,
#     w_hi <= 500 rpm (proven stable; the sweep ran 2500 rpm clean).
#   - reward per block: X_block - P_block/31.94 W (block-avg conversion minus
#     normalized motor power on the commanded wave).
#
# QUESTION: can state-dependent, block-varying modulation beat the best STATIC
# waveform (D=80%, T=2.5 s: X=0.386, R=0.270 at w_b=300)? R>0.28 sustained =
# genuine discovery; R~0.27 converged = clean validation. Noise floor +-0.01.
#
# COST: 50 s episode ~ 40-53 min on a 7800X3D core at these speeds; Carya cores
# are somewhat slower, call it ~1 h. 1000 episodes / 44 workers ~ 23 episodes
# per worker ~ 23-27 h + ~2 min one-time template compile. -t 48 h covers it
# comfortably; SAVE_EVERY checkpoints make any overrun a usable policy.
#
# Submit:  sbatch experiments/modulation_rl/run_carya_modulation.slurm
# Monitor: squeue -u $USER ; tail -f experiments/modulation_rl/logs/*.out
# =====================================================================

#SBATCH -J tc_td3_modulation
#SBATCH -o experiments/modulation_rl/logs/%x_%j.out
#SBATCH -e experiments/modulation_rl/logs/%x_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48                     # whole Compute node: 44 workers + headroom for learner/OS
#SBATCH --mem=8G                  # 44 solvers on the 3330-cell graded mesh are tiny
                                  #  (measured ~1.9 GB peak for 44 comparable solvers)
#SBATCH -t 48:0:0                 # ~23-27 h expected (see COST above); checkpoints cover overrun
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bascomisaiah@gmail.com

set -euo pipefail

# ---- run config -----------------------------------------------------
ALGO=td3
N_WORKERS=44                     # 48-core node: leave ~4 cores for the learner + OS
SEED=0
TAG="mod_wb300_s${SEED}"

# ---- the settled design (experiments/modulation_rl/README.md) -------
# Fixed for this run: w_b=300, 50 s episodes, 10 s blocks, reward weights (1,1),
# P_max=31.94 W. Future runs change W_B_RPM only (P_max is global, so rewards
# stay comparable across w_b).
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

# ---- budgets --------------------------------------------------------
# PRIMARY stop = 1000 completed episodes (5 transitions each -> 5000 transitions;
# the README budget). MAX_TIMESTEPS is a safety bound (~1200 episodes).
MAX_EPISODES=1000
MAX_TIMESTEPS=6000
START_TIMESTEPS=500              # 100 fully-random episodes before the policy kicks in
EXPL_NOISE=0.1                   # per raw dim; T is log-mapped so noise explores octaves
BATCH_SIZE=64
SAVE_EVERY=500                   # checkpoint + log snapshot every N grad steps

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
WORKER_ROOT="${TMPDIR:-/tmp/$USER}/mod_workers_${TAG}"

export OMP_NUM_THREADS=1          # each pimpleFoam is serial; don't let libs grab extra cores
export PYTHONUNBUFFERED=1

echo "=== parallel TD3 (block modulation, fixed-mean w_b=${W_B_RPM}rpm) seed=$SEED workers=$N_WORKERS ==="
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
    --save_every "$SAVE_EVERY" \
    --tag "$TAG" \
    --results_dir "$RESULTS_DIR"

echo "=== done. checkpoints + logs -> $RESULTS_DIR/$ALGO/$TAG ==="
