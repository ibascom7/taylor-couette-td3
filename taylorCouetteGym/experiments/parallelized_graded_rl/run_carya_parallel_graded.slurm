#!/bin/bash
# =====================================================================
# Carya: PARALLELIZED TD3 on the GRADED-MESH RESOLVED-GRADIENT side-outlet case.
#
# Same harness as experiments/parallelized_catalysis_rl (N=44 threaded env workers,
# one OpenFOAM case copy each, ONE shared replay buffer, one TD3 learner) but the
# environment is side_outlet_grad_case: the RL-drivable clone of
# side_outlet_case_sc1075_graded. NO Sherwood wall model -- the catalytic outer
# wall is a c=0 sink and the 25 um graded wall cells RESOLVE the Sc=1075
# concentration film, so the reward's wallFlux is real -D*dc/dn physics (the
# question this experiment answers: does TD3 still find pulsed modulation when the
# film response is resolved rather than modeled by the ETW correlation?).
#
# The training script is REUSED from parallelized_catalysis_rl (single source of
# truth for the threading/divergence-recovery logic); only the case, the wallFlux
# normalizer, and the budgets below differ.
#
# COST (measured on the steady_state_conversion graded-Sc=1075 sweep, single core,
# local 7800X3D; Carya cores are somewhat slower):
#     0-750 rpm ~41-51 s/sim-s, 1000 rpm ~66, 2500 rpm ~179 s/sim-s.
# A 120 s episode is therefore ~1.4 h (low omega) to ~6 h (pegged at 2500 rpm);
# random-exploration episodes average ~3 h. 150 episodes / 44 workers ~ 3.4
# episodes per worker -> ~10-20 h + ~1 h one-time warmup, so -t 48 h covers even
# a spin-heavy (low-ew) task; SAVE_EVERY checkpoints make any overrun usable.
#
# SWEEP: a job ARRAY over energy_weight -- each task takes a whole node (44
# workers) and trains one TD3 agent at one ew (see ENERGY_WEIGHTS below).
#
# Reward: conv_weight*wf_index - energy_weight*E_index,
#   wf_index = wallFlux / WALLFLUX_MAX   (resolved outer-wall flux, per unit c0)
#   E_index  = E_step  / E_max(omega_max) (motor energy for 1 step at 2500 rpm)
#
# Submit:  sbatch experiments/parallelized_graded_rl/run_carya_parallel_graded.slurm
# Monitor: squeue -u $USER ; tail -f experiments/parallelized_graded_rl/logs/*.out
# =====================================================================

#SBATCH -J tc_td3_graded
#SBATCH -o experiments/parallelized_graded_rl/logs/%x_%A_%a.out
#SBATCH -e experiments/parallelized_graded_rl/logs/%x_%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48                     # whole Compute node PER TASK: 44 workers + headroom for learner/OS
#SBATCH --mem=8G                  # 44 solvers on the 3330-cell graded mesh are tiny (the cat-case
                                  #  measurement was ~1.9 GB peak for 44 comparable solvers); 8G is ample.
#SBATCH -t 48:0:0                 # see COST above: ~10-20 h typical, ~25 h for a spin-heavy low-ew task
                                  #  on slower Carya cores; checkpoints cover any overrun.
#SBATCH --array=0-4               # one task per energy_weight (5 tasks). Keep = 0-(len(ENERGY_WEIGHTS)-1).
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --mail-user=bascomisaiah@gmail.com

set -euo pipefail

# ---- run config -----------------------------------------------------
ALGO=td3
N_WORKERS=44                     # 48-core node: leave ~4 cores for the learner + OS
CONTROL_MODE=freeform            # the modulation agent (absolute omega each freeform_dt s + clock)

# ---- SWEEP: energy_weight (one task per weight) -----------------------------------
# Calibrated from the steady-state sweep (results_graded_sc1075) + the motor model.
# With WALLFLUX_MAX below, the steady constant-omega operating points are
#   rpm     :    0    250    500    750   1000   2500
#   wf_index: 0.256  0.611  0.732  0.809  0.857  1.000   (= conv/0.568)
#   E_index : 0      0.099  0.191  0.285  0.380  1.000   (motor P/31.9 W)
# The best CONSTANT omega* by regime: ew<0.23 pegs 2500, 0.23-0.51 -> 1000 rpm,
# 0.51-0.82 -> 750, 0.82-1.30 -> 500, 1.30-3.59 -> 250, idle only above ew~3.6
# (wf_index=0.26 at 0 rpm: the resolved film converts 14.6% with no rotation).
# GRID RATIONALE (one task per distinct regime; measured burst transients put the
# pulse-vs-slow-constant breakeven around ew~1.5-2):
#   0.3, 0.6 -> the 1000/750 rpm constant regimes, where burst+regen modulation has
#               the most to gain (these tasks spin fastest -> the slow ones);
#   0.8, 1.3 -> the 500 rpm regime; values SHARED with the wall-model round-2 sweep
#               for cross-case comparison;
#   1.8      -> the 250 rpm regime, at the estimated pulse-vs-constant breakeven --
#               the most discriminating point. (2.4/3.0 dropped: same constant
#               regime as 1.8 here, and idle doesn't win until ew~3.6.)
ENERGY_WEIGHTS=(0.3 0.6 0.8 1.3 1.8)   # 5 values -> --array=0-4
SEED=0
#
# array index -> energy_weight (one task each). Keep --array = 0-(len(ENERGY_WEIGHTS)-1).
i=${SLURM_ARRAY_TASK_ID:-0}
ENERGY_WEIGHT=${ENERGY_WEIGHTS[$i]}
TAG="sog_parallel_${CONTROL_MODE}_ew${ENERGY_WEIGHT}_s${SEED}"

# budgets. PRIMARY stop = 150 completed episodes of 120 s each (converge + classify
# idle vs spin, matching the wall-model sweep). MAX_TIMESTEPS is a safety bound.
MAX_EPISODES=150
MAX_STEPS_PER_EP=120           # 120 s episodes; MUST equal run_carya_compare's EVAL_SECONDS (phase clock)
MAX_TIMESTEPS=30000            # safety cap (~250 eps); MAX_EPISODES governs. Stays < 1e5 buffer.
START_TIMESTEPS=4000           # random-action steps before the policy + learning kick in
FREEFORM_DT=1.0
SAVE_EVERY=2000                # checkpoint + log snapshot every N grad steps

# warmup: ~3 tau (80 s at 100 mL/min) spins the reactor up to a settled flow, done
# ONCE per task and cached as the 0.warmed IC every episode resets to. 500 rpm = a
# developed supercritical Taylor-vortex field, representative of the operating
# regime (~45 s/sim-s -> the warmup costs ~1 h, once).
WARMUP_DURATION=80
WARMUP_OMEGA_RPM=500

# reward weights (both indices ~[0,1]). ENERGY_WEIGHT is set per-task by the sweep above.
CONV_WEIGHT=1.0
ENERGY_MODEL=motor
# wallFlux normalizer: the steady resolved wallFlux at omega_max (2500 rpm), from
# the steady_state_conversion graded-Sc=1075 sweep via the mass balance
#   wallFlux(per unit c0) = Q_wedge * conversion = 2.315e-8 m^3/s * 0.568 = 1.32e-8
# (rlMetrics reports the wedge-slice flux per unit c0, so Q is the 5-degree-wedge
# feed rate the env also uses). -> wf_index = wallFlux/WALLFLUX_MAX ~ 1 at the
# omega_max operating point, same semantics as the wall-model runs' 8.2e-9.
WALLFLUX_MAX=1.32e-8

# side-outlet geometry / normalizers (identical to the wall-model case)
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
EXP_DIR="experiments/parallelized_graded_rl"
# REUSED trainer: the graded experiment deliberately shares parallel_train.py with
# parallelized_catalysis_rl (case-agnostic; everything case-specific is a flag).
TRAIN="experiments/parallelized_catalysis_rl/parallel_train.py"
RESULTS_DIR="$EXP_DIR/results"
mkdir -p "$EXP_DIR/logs"

MASTER_CASE="taylor_couette_mixing/cases/side_outlet_grad_case"   # pre-configured drop-in
# NODE-LOCAL SSD for the 44 worker cases: 44 concurrent pimpleFoam do a lot of small
# time-dir I/O, far faster on the node's local SSD than the shared HDD FS.
# Trade-off: the warmed template is not cached across jobs (re-warms ~1 h per run).
# For a persistent cross-job cache instead, use "$EXP_DIR/_runs/${TAG}/workers".
WORKER_ROOT="${TMPDIR:-/tmp/$USER}/sog_workers_${TAG}"

export OMP_NUM_THREADS=1          # each pimpleFoam is serial; don't let libs grab extra cores
export PYTHONUNBUFFERED=1

echo "=== parallel TD3 (graded resolved-gradient) task ${i}: mode=$CONTROL_MODE energy_weight=$ENERGY_WEIGHT warmup=${WARMUP_OMEGA_RPM}rpm seed=$SEED workers=$N_WORKERS ==="
echo "    tag=$TAG  master=$MASTER_CASE  worker_root=$WORKER_ROOT  host=$(hostname)  cores=${SLURM_CPUS_PER_TASK:-?}"

"$PYTHON" "$TRAIN" \
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
    --results_dir "$RESULTS_DIR"

echo "=== done. checkpoints + logs -> $RESULTS_DIR/$ALGO/$TAG ==="
