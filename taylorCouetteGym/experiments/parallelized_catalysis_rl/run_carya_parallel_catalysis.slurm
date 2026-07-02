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
# SWEEP: this is a job ARRAY over energy_weight x seed -- each task takes a whole node
# (44 workers) and runs the SAME 500-episode config at a different reward knob, so the
# whole sweep runs at once. Pick the tag whose learned waveform beats the constant
# baseline in the compare_catalysis power-vs-conversion figure.
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
#SBATCH --mem=120G                # node has 189 GB; 44 small solvers use ~15-25 GB, rest is margin
#SBATCH -t 12:0:0                 # warmup (~10 min) + 44x-faster collection; ckpts guard timeout
#SBATCH --array=0-3               # 4 energy_weights x 1 seed = 4 tasks (one node each)
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --mail-user=bascomisaiah@gmail.com

set -euo pipefail

# ---- run config -----------------------------------------------------
ALGO=td3
N_WORKERS=44                     # 48-core node: leave ~4 cores for the learner + OS
CONTROL_MODE=freeform            # the modulation agent (absolute omega each freeform_dt s + clock)

# ---- SWEEP: energy_weight x seed (array index -> one (ew, seed) per node) --------
# energy_weight is THE reward knob (reward = conv_weight*wf_index - energy_weight*E_index,
# both ~[0,1]): too low -> agent spins hard/ignores energy; too high -> idles omega->0.
# The beats-constant regime is intermediate -- this sweep brackets it. If you change the
# list sizes, update "#SBATCH --array" above to 0-(len(EW)*len(SEEDS)-1).
ENERGY_WEIGHTS=(0.1 0.2 0.4 0.8)   # locating band; dropped extremes 0.05 (spins to max, ~12h)
                                   # and 1.6 (hard idle). Multi-seed the winner next.
SEEDS=(0)                        # ONE seed: this is a LOCATING sweep for the knob.
                                 # Multi-seed the winning energy_weight afterward to prove it reproduces.
i=${SLURM_ARRAY_TASK_ID:-0}
NSEEDS=${#SEEDS[@]}
ENERGY_WEIGHT=${ENERGY_WEIGHTS[$((i / NSEEDS))]}
SEED=${SEEDS[$((i % NSEEDS))]}
TAG="so_parallel_${CONTROL_MODE}_ew${ENERGY_WEIGHT}_s${SEED}"

# budgets. PRIMARY stop = 500 completed episodes of 120 s each (~60k env-steps total,
# collected ~N_WORKERS x faster). MAX_TIMESTEPS is just an upper safety bound.
MAX_EPISODES=500               # <- the target: 500 completed episodes
MAX_STEPS_PER_EP=120           # 120 s episodes = ~4.6 residence times (tau~26 s) -> quasi-steady
MAX_TIMESTEPS=80000            # safety cap (~660 eps); MAX_EPISODES governs. Stays < 1e5 buffer.
START_TIMESTEPS=4000           # random-action steps before the policy + learning kick in
FREEFORM_DT=1.0
SAVE_EVERY=2000                # checkpoint + log snapshot every N grad steps

# warmup: shortened side-outlet at 100 mL/min has residence tau~26 s, so ~3 tau
# (80 s) reaches steady state. Done ONCE, then cached (reused unless --rebuild).
WARMUP_DURATION=80
WARMUP_OMEGA_RPM=500

# reward weights (both terms now ~[0,1]). ENERGY_WEIGHT is set per-task by the sweep above.
CONV_WEIGHT=1.0
ENERGY_MODEL=motor

# side-outlet geometry / normalizers (env auto-computes wallflux_max=Q*c0 from these)
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

echo "=== parallel TD3 (side-outlet) task ${i}: mode=$CONTROL_MODE energy_weight=$ENERGY_WEIGHT seed=$SEED workers=$N_WORKERS ==="
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
    --r_in "$R_IN" --r_out "$R_OUT" --feed_velocity "$FEED_VELOCITY" \
    --tag "$TAG" \
    --results_dir "$EXP_DIR/results"

echo "=== done. checkpoints + logs -> $EXP_DIR/results/$ALGO/$TAG ==="
