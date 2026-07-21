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
#   - action per block: (duty D in [0.6,0.9], idle w_low FROZEN at 0 rpm,
#     period T in [0.5,5] s log-mapped); burst speed SOLVED from the fixed-mean
#     constraint w_hi = (w_b - (1-D)*w_low)/D = w_b/D  ->  commanded mean = w_b
#     always, w_hi <= 500 rpm (proven stable; the sweep ran 2500 rpm clean).
#     The a1 (idle) action dim is DEAD (zero-width range decodes to 0) -- kept
#     so state/action dims match earlier runs' checkpoints and code paths.
#   - reward per block: X_block - P_block/31.94 W (block-avg conversion minus
#     normalized motor power on the commanded wave).
#
# v3 CONTROL RUN. History: s0 collapsed to the degenerate w_low=w_b corner
# (constant-300, R=0.229). v2 capped the box (idle<=200, D<=0.95) -- the cap
# held (no constant wave possible) but the actor STILL raced to the shallowest
# reachable corner (D=0.6, wlo=200, T=5; R~0.235 < its own random baseline
# 0.884-return phase), proving the root cause is CREDIT ASSIGNMENT (same-block
# idle cost corr -0.18 vs NEXT-block renewal payoff corr +0.35 -- a weak critic
# only sees the cost), not the corner itself.
#
# v4 = v3 RERUN WITH THE ROOT-CAUSE FIX. v3 exposed the real bug: episodes
# end by TRUNCATION but the buffer stored done=float(terminated)=0 always, so
# the critic bootstrapped into end-of-episode states (clock=1) that never
# occur as source states -> their Q is UNGROUNDED, drifts (probed: Q=0.78 vs
# true 0), and its gradient dragged the actor to the same raw action corner
# (-1,+1,+1) in s0/v2/v3 alike -- v3's actor even saturated its DEAD idle dim,
# proving the driver is internal. parallel_train.py now stores
# done=(terminated or truncated). The v3 random phase also mapped the reward
# landscape: champion cell D in [0.8,0.9), T in [2.5,5) is genuinely best
# (R 0.275) and same-block corr(r, D) = +0.68 -- the signal is clean; only
# the bootstrap was broken.
#
# v3 QUESTION (unchanged, now a fair test): freeze the runaway idle dimension (w_low = 0 always,
# the same on/off family as the static benchmark grid) and learn ONLY (D, T).
# The box now contains a KNOWN ground truth interior to it: the static champion
# D=80%, T=2.5 s (X=0.386, R=0.270). PASS = policy settles near it (or better);
# FAIL = policy pins D at the 0.9 ceiling (the shallowest wave, the bias
# direction) -> the n-step-return critic fix is required before the long run.
# DUTY_MAX=0.9 (not 0.95) because with w_low=0 a D=0.95 wave is 95% constant --
# too close to the degenerate family this experiment exists to exclude.
#
# COST: 50 s episode ~ 40-53 min on a 7800X3D core at these speeds; Carya cores
# are somewhat slower, call it ~1 h. 300 episodes / 44 workers ~ 7 episodes
# per worker ~ 7 h + ~2 min one-time template compile. -t 12 h covers it
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
#SBATCH -t 12:0:0                 # ~7 h expected (see COST above); checkpoints cover overrun
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bascomisaiah@gmail.com

set -euo pipefail

# ---- run config -----------------------------------------------------
ALGO=td3
N_WORKERS=44                     # 48-core node: leave ~4 cores for the learner + OS
SEED=0
TAG="mod_wb300_v4_s${SEED}"

# ---- the settled design (experiments/modulation_rl/README.md) -------
# Fixed for this run: w_b=300, 50 s episodes, 10 s blocks, reward weights (1,1),
# P_max=31.94 W. Future runs change W_B_RPM only (P_max is global, so rewards
# stay comparable across w_b).
W_B_RPM=300
EPISODE_DURATION=50
BLOCK_DT=10
DUTY_MIN=0.6
DUTY_MAX=0.9                     # v3: 0.9 ceiling; at w_low=0 a D=0.95 wave is nearly
                                 #  constant, and the pass/fail read needs the champion
                                 #  D=0.8 interior with room on both sides
IDLE_MIN_RPM=0
IDLE_MAX_RPM=0                   # v3 CONTROL: zero-width range freezes w_low=0 (full-stop
                                 #  idle, the static-benchmark waveform family); the a1
                                 #  action dim decodes to 0 regardless -> the runaway
                                 #  dimension from s0/v2 no longer exists
PERIOD_MIN=0.5                   # T floor: below it vortices low-pass the forcing
PERIOD_MAX=5
RAMP_TIME=0.05
P_MAX_WATT=31.94                 # motor power at 2500 rpm (global reward normalizer)
WALLFLUX_MAX=1.32e-8             # obs normalizer: steady resolved wallFlux at 2500 rpm

# ---- budgets --------------------------------------------------------
# PRIMARY stop = 300 completed episodes (1500 transitions). MINIMAL SHAKEDOWN:
# 200 random (the critic's clean interior data + landscape probe of the new
# box) + 100 policy episodes. s0 proved the actor expresses its direction
# within ~50-100 policy episodes (~250-500 grad steps -- s0/v2 both fully
# pinned by then), so 100 policy eps answers the one control question:
# does (D,T) head toward the interior champion D=0.8/T=2.5 (pass) or does D
# pin at the 0.9 ceiling (fail -> critic needs n-step returns)? It does NOT
# give a converged R -- judge by the D/T trajectory and R >= the random-phase
# mean, not by distance to the 0.270 champion. Once proven, the real run is a
# FRESH longer run from scratch (bump MAX_EPISODES/MAX_TIMESTEPS/-t, new tag).
# MAX_TIMESTEPS = ~1.2x safety bound.
MAX_EPISODES=300
MAX_TIMESTEPS=1800
START_TIMESTEPS=1000             # v2: 200 fully-random episodes (was 100) -- more interior
                                 #  (s,a) coverage for the critic's delayed-payoff fit
EXPL_NOISE=0.2                   # v2: was 0.1; on w_low that is ~20 rpm std, enough to
                                 #  probe depth even if the actor drifts toward a bound
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
    --idle_max_rpm "$IDLE_MAX_RPM" \
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
