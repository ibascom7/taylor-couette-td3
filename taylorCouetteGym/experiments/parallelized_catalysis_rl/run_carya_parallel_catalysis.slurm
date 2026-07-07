#!/bin/bash
# =====================================================================
# Carya: PARALLELIZED TD3 on the SIDE-OUTLET catalysis case.
#
# N=44 env workers on a full 48-core Compute node, each with its own OpenFOAM case
# copy, run pimpleFoam rollouts CONCURRENTLY (threads) and feed ONE shared replay
# buffer; a single TD3 learner trains on the pooled experience. Off-policy =>
# AlphaZero-style (many games -> one buffer -> one net). Wall-clock ~ serial / N_WORKERS.
#
# The warmup (a ~80 s spin-up so the reactor fills to steady) is done ONCE per task,
# cached as a warmed template on the node's local SSD, and copied into all workers.
# (Node-local worker_root re-warms each job, ~10 min; for a cross-job cache point
# WORKER_ROOT at shared storage.) Pass --rebuild to force a fresh warmup.
#
# SWEEP: a job ARRAY over energy_weight -- each task takes a whole node (44 workers)
# and trains one TD3 agent at one ew. The ew grid walks across the idle<->spin
# crossover (see ENERGY_WEIGHTS below) to locate the interior-optimum window.
#
# Reward (side-outlet, dimensionless indices): conv_weight*wf_index - energy_weight*E_index,
#   wf_index = wallFlux / (Q*c0)         (auto: 2.315e-8 m^3/s, feed-rate ceiling)
#   E_index  = E_step  / E_max(omega_max) (motor energy for 1 step at 2500 rpm)
#
# Submit:  sbatch experiments/parallelized_catalysis_rl/run_carya_parallel_catalysis.slurm
# Monitor: squeue -u $USER ; tail -f experiments/parallelized_catalysis_rl/logs/*.out
# =====================================================================

#SBATCH -J tc_td3_parallel
#SBATCH -o experiments/parallelized_catalysis_rl/logs/%x_%A_%a.out
#SBATCH -e experiments/parallelized_catalysis_rl/logs/%x_%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48                     # whole Compute node PER TASK: 44 workers + headroom for learner/OS
#SBATCH --mem=4G                  # MEASURED: 44 side-outlet solvers peak ~1.9 GB (jobs 7597281/301);
                                  #           8G is ~4x margin, plenty. (was 120G -- 60x over-request.)
#SBATCH -t 8:0:0                  # ~2 h for higher-ew (low-omega) tasks, up to ~6 h for lower-ew tasks that
                                  #           spin to max (~6 h at 150 ep). 8 h covers them; policy converges
                                  #           by ~ep100 so ckpts (SAVE_EVERY) make any overrun a usable policy.
#SBATCH --array=0-4              # one task per energy_weight (5 tasks). Keep = 0-(len(ENERGY_WEIGHTS)-1).
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --mail-user=bascomisaiah@gmail.com

set -euo pipefail

# ---- run config -----------------------------------------------------
ALGO=td3
N_WORKERS=44                     # 48-core node: leave ~4 cores for the learner + OS
CONTROL_MODE=freeform            # the modulation agent (absolute omega each freeform_dt s + clock)

# ---- SWEEP: energy_weight (one task per weight) -----------------------------------
# CORRECTED for the wall-transfer reward (2026-07): with the catalytic-wall mass-
# transfer model, wf_index now RISES with omega (wallFlux ~ omega^0.78) instead of
# sitting flat, so the idle<->spin crossover moved up ~30-100x, from ~0.01-0.1 to
# ~0.7-3.2. This grid spans the interior-optimum window; the best CONSTANT omega*
# runs ~1200 rpm (ew=0.8) down to ~80 rpm (ew=3.0). ew<~0.6 pegs omega_max, ew>~3.3
# idles. (Modulation is cheaper than constant under the motor model, so RL should
# beat these constant-omega rewards and may push the idle boundary higher.)
ENERGY_WEIGHTS=(0.8 1.3 1.8 2.4 3.0)   # 5 values -> --array=0-4
SEED=0
#
# array index -> energy_weight (one task each). Keep --array = 0-(len(ENERGY_WEIGHTS)-1).
i=${SLURM_ARRAY_TASK_ID:-0}
ENERGY_WEIGHT=${ENERGY_WEIGHTS[$i]}
TAG="so_parallel_${CONTROL_MODE}_ew${ENERGY_WEIGHT}_s${SEED}"

# budgets. PRIMARY stop = 150 completed episodes of 120 s each. The policy converges by
# ~ep100 (see the ew0.1 behavior plot), so 150 is enough to CLASSIFY idle vs spin while
# keeping the spin-IC tasks within --time. MAX_TIMESTEPS is just an upper safety bound.
MAX_EPISODES=150               # <- locating/probe target: enough to converge + classify regime
MAX_STEPS_PER_EP=120           # 120 s episodes; MUST equal compare_catalysis EVAL_SECONDS (phase clock)
MAX_TIMESTEPS=30000            # safety cap (~250 eps); MAX_EPISODES governs. Stays < 1e5 buffer.
START_TIMESTEPS=4000           # random-action steps before the policy + learning kick in
FREEFORM_DT=1.0
SAVE_EVERY=2000                # checkpoint + log snapshot every N grad steps

# warmup: ~3 tau (80 s) spins the reactor up to a settled flow, done ONCE per task and
# cached as the 0.warmed IC every episode resets to. WARMUP_OMEGA_RPM sets that single
# flow IC: 500 rpm = a developed supercritical Taylor-vortex field, representative of the
# operating regime. If the warmup ever diverges, lower it or raise ramp_time.
WARMUP_DURATION=80
WARMUP_OMEGA_RPM=500

# reward weights (both terms now ~[0,1]). ENERGY_WEIGHT is set per-task by the sweep above.
CONV_WEIGHT=1.0
ENERGY_MODEL=motor
# wallFlux normalizer: MEASURED steady wallFlux at omega_max (2500 rpm) under the
# catalytic-wall mass-transfer model (side_outlet_cat_case). The wall model changes
# the wallFlux magnitude, so we set the measured value explicitly instead of the
# Q*c0 auto-fallback -> wf_index = wallFlux/WALLFLUX_MAX in [0,1], comparable to E_index.
WALLFLUX_MAX=8.2e-9

# side-outlet geometry / normalizers (r_in/r_out/feed_velocity used for other terms)
R_IN=25.4
R_OUT=31.75
FEED_VELOCITY=1.462e-3            # Umean = Q0(100 mL/min)/annulus area (Lopez, corrected)

# =====================================================================
# OpenFOAM + python env -- IDENTICAL recipe to the other RL slurms.
# =====================================================================
OPENFOAM_MODULE="OpenFOAM/v2506-foss-2025b"
set +u; module purge; module load "$OPENFOAM_MODULE"; source "$FOAM_BASH"; set -u
unset PYTHONPATH PYTHONHOME
PYTHON=/project/mwang/ibascom/envs/tc/bin/python

cd "$SLURM_SUBMIT_DIR"
EXP_DIR="experiments/parallelized_catalysis_rl"
mkdir -p "$EXP_DIR/logs"

MASTER_CASE="taylor_couette_mixing/cases/side_outlet_cat_case"   # pre-configured drop-in
# NODE-LOCAL SSD for the 44 worker cases: 44 concurrent pimpleFoam do a lot of small
# time-dir I/O, far faster on the node's 960 GB local SSD than the shared HDD FS.
# Trade-off: the warmed template is not cached across jobs (re-warms ~10 min per run).
# For a persistent cross-job cache instead, use "$EXP_DIR/_runs/${TAG}/workers".
WORKER_ROOT="${TMPDIR:-/tmp/$USER}/so_workers_${TAG}"

export OMP_NUM_THREADS=1          # each pimpleFoam is serial; don't let libs grab extra cores
export PYTHONUNBUFFERED=1

echo "=== parallel TD3 (side-outlet) task ${i}: mode=$CONTROL_MODE energy_weight=$ENERGY_WEIGHT warmup=${WARMUP_OMEGA_RPM}rpm seed=$SEED workers=$N_WORKERS ==="
echo "    tag=$TAG  master=$MASTER_CASE  worker_root=$WORKER_ROOT  host=$(hostname)  cores=${SLURM_CPUS_PER_TASK:-?}"

"$PYTHON" "$EXP_DIR/parallel_train.py" \
    --algo "$ALGO" \
    --seed "$SEED" \
    --n_workers "$N_WORKERS" \
    --case_path "$MASTER_CASE" \
    --worker_root "$WORKER_ROOT" \
    --control_mode "$CONTROL_MODE" \
    --freeform_dt "$FREEFORM_DT" \
    --max_steps_per_ep "$MAX_STEPS_PER_EP" \
    --max_episodes "$MAX_EPISODES" \
    --max_timesteps "$MAX_TIMESTEPS" \
    --start_timesteps "$START_TIMESTEPS" \
    --save_every "$SAVE_EVERY" \
    --warmup_duration "$WARMUP_DURATION" \
    --warmup_omega_rpm "$WARMUP_OMEGA_RPM" \
    --conv_weight "$CONV_WEIGHT" \
    --energy_weight "$ENERGY_WEIGHT" \
    --energy_model "$ENERGY_MODEL" \
    --wallflux_max "$WALLFLUX_MAX" \
    --r_in "$R_IN" --r_out "$R_OUT" --feed_velocity "$FEED_VELOCITY" \
    --tag "$TAG" \
    --results_dir "$EXP_DIR/results"

echo "=== done. checkpoints + logs -> $EXP_DIR/results/$ALGO/$TAG ==="
