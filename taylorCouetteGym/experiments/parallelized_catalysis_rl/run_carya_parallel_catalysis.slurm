#!/bin/bash
# =====================================================================
# Carya: PARALLELIZED TD3 on the SIDE-OUTLET catalysis case.
#
# N=16 env workers, each with its own OpenFOAM case copy, run pimpleFoam rollouts
# CONCURRENTLY (threads) and feed ONE shared replay buffer; a single TD3 learner
# trains on the pooled experience. Off-policy => AlphaZero-style (many games ->
# one buffer -> one net). Wall-clock ~ serial / N_WORKERS.
#
# The warmup (a ~130 s spin-up so the reactor fills to steady) is done ONCE, cached
# as a warmed template, and copied into all workers. With a PERSISTENT WORKER_ROOT
# (the default below) the warmed template is REUSED on re-runs -- "saved before the
# run". Pass --rebuild to force a fresh warmup.
#
# Reward (side-outlet, dimensionless indices): conv_weight*wf_index - energy_weight*E_index,
#   wf_index = wallFlux / (Q*c0)         (auto: 9.26e-9 m^3/s, feed-rate ceiling)
#   E_index  = E_step  / E_max(omega_max) (motor energy for 1 step at 2500 rpm)
#
# Submit:  sbatch experiments/parallelized_catalysis_rl/run_carya_parallel_catalysis.slurm
# Monitor: squeue -u $USER ; tail -f experiments/parallelized_catalysis_rl/logs/*.out
# =====================================================================

#SBATCH -J tc_td3_parallel
#SBATCH -o experiments/parallelized_catalysis_rl/logs/%x_%j.out
#SBATCH -e experiments/parallelized_catalysis_rl/logs/%x_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 18                     # 16 pimpleFoam workers + headroom for the learner/torch
#SBATCH --mem=24G
#SBATCH -t 12:0:0                 # warmup (~20 min) + 16x-faster collection; ckpts guard timeout
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bascomisaiah@gmail.com

set -euo pipefail

# ---- run config -----------------------------------------------------
ALGO=td3
SEED=0
N_WORKERS=16
CONTROL_MODE=freeform            # the modulation agent (absolute omega each freeform_dt s + clock)
TAG="so_parallel_${CONTROL_MODE}_s${SEED}"

# budgets (TOTAL env steps across ALL workers; collected ~N_WORKERS x faster)
MAX_TIMESTEPS=6000
START_TIMESTEPS=600
MAX_STEPS_PER_EP=60
FREEFORM_DT=1.0
SAVE_EVERY=500

# warmup (side-outlet needs ~130 s to fill to steady; done ONCE, then cached)
WARMUP_DURATION=130
WARMUP_OMEGA_RPM=500

# reward weights (both terms now ~[0,1], so retune from here)
CONV_WEIGHT=1.0
ENERGY_WEIGHT=0.1
ENERGY_MODEL=motor

# side-outlet geometry / normalizers (env auto-computes wallflux_max=Q*c0 from these)
R_IN=25.4
R_OUT=31.75
FEED_VELOCITY=5.847e-4            # Umean = Q0(40 mL/min)/annulus area

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
# PERSISTENT worker root so the warmed template survives across jobs (reuse = "saved").
# For faster I/O with 16 concurrent solvers you can point this at node-local scratch
# (e.g. WORKER_ROOT="$TMPDIR/so_workers"), at the cost of re-warming each job.
WORKER_ROOT="$EXP_DIR/_runs/${TAG}/workers"

export OMP_NUM_THREADS=1          # each pimpleFoam is serial; don't let libs grab extra cores
export PYTHONUNBUFFERED=1

echo "=== parallel TD3 (side-outlet): mode=$CONTROL_MODE seed=$SEED workers=$N_WORKERS ==="
echo "    master=$MASTER_CASE  worker_root=$WORKER_ROOT  host=$(hostname)  cores=${SLURM_CPUS_PER_TASK:-?}"

"$PYTHON" "$EXP_DIR/parallel_train.py" \
    --algo "$ALGO" \
    --seed "$SEED" \
    --n_workers "$N_WORKERS" \
    --case_path "$MASTER_CASE" \
    --worker_root "$WORKER_ROOT" \
    --control_mode "$CONTROL_MODE" \
    --freeform_dt "$FREEFORM_DT" \
    --max_steps_per_ep "$MAX_STEPS_PER_EP" \
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
