#!/bin/bash
# =====================================================================
# Carya: RETRAIN the FREEFORM catalysis agent -- v2, fixing the idle-collapse.
#
# WHY v2 (see results/td3/catalysis_freeform_s0 post-mortem):
# The v1 freeform run collapsed to ~40 rpm ("idle and coast"). Diagnosis from the
# per-step reward logs: reward = wf_norm - energy_weight*E_norm is MONOTONICALLY
# DECREASING in omega -- the energy penalty (up to -0.45 at 2500 rpm) dwarfs the
# tiny wall-flux signal (wf_norm ~0.01-0.05), so idle is the ONLY positive-reward
# band (+0.012). The agent optimized a mis-scaled reward correctly. Conversion
# itself IS healthy vs omega (0.54 below ~500 rpm, 0.98 above -- the SCALAR_D=1e-9
# fix works), so the physics is fine; only the reward balance is wrong.
#
# TWO CHANGES vs run_carya_train_catalysis_td3.slurm:
#   1. energy_weight SWEEP (array) -- primary lever. v1 used 0.1 (energy dominates).
#      The idle->spin crossover is ~0.008-0.02 by back-of-envelope, so sweep across
#      it: 0.02 (expect still idle), 0.008 (expect spin), 0.003 (expect spin hard,
#      maybe pin high). Pick the seed that sustains conversion at the lowest power.
#   2. horizon 60 -> 150 steps -- secondary. v1 episodes were 60 s (~2.3 residence
#      times) starting from the settled 500-rpm warm IC, so idle could COAST on the
#      leftover conversion (start 1.0 -> end 0.03 within an episode). 150 s (~5.8
#      tau_res) makes the settled tail dominate, so idle scores its true low conv.
#
# freeform ONLY (no omega/constant task -- that baseline is already trained as
# catalysis_omega_s0). New TAGs (catalysis_freeform_v2_<label>_s<SEED>) so the v1
# run is preserved for comparison.
#
# Submit:  sbatch experiments/catalysis_rl/run_carya_train_catalysis_td3_v2.slurm
#   (single value instead of the sweep:  sbatch --array=1 ...  -> the 0.008 task)
# =====================================================================

#SBATCH -J tc_td3_cat_v2
#SBATCH -o experiments/catalysis_rl/logs/%x_%A_%a.out
#SBATCH -e experiments/catalysis_rl/logs/%x_%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH -t 50:0:0
#SBATCH --array=0-2               # energy_weight sweep (see EWS below)
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --mail-user=bascomisaiah@gmail.com

set -euo pipefail

# ---- energy_weight sweep: array index -> (weight, label) ------------
ALGO=td3
SEED=0
CONTROL_MODE=freeform
EWS=(0.02  0.008 0.003)          # the lever; brackets the idle->spin crossover
LABELS=(ew020 ew008 ew003)
i=${SLURM_ARRAY_TASK_ID:-0}
ENERGY_WEIGHT=${EWS[$i]}
TAG="catalysis_freeform_v2_${LABELS[$i]}_s${SEED}"

# ---- geometry / normalizer (IDENTICAL to v1) -----------------------
R_IN=25.4
R_OUT=31.75
E_MAX_PER_STEP=0.0011017031875434

# ---- catalysis setup (IDENTICAL to v1) -----------------------------
MEAN_RPM=500
SCALAR_D=1e-9                    # MUST match the compare/eval slurms

# ---- reward weights ------------------------------------------------
CONV_WEIGHT=1.0
# ENERGY_WEIGHT set by the array above (v1 was 0.1 -> idle-collapse)
ENERGY_MODEL=motor

# ---- training hyperparameters --------------------------------------
WARMUP_DURATION=100
WARMUP_OMEGA_RPM=500
FREEFORM_DT=1.0
# v2: longer horizon so the settled tail (not the warm-start transient) drives the
# reward. 150 steps x 1 s = 150 s ~ 5.8 residence times. Same total sim-seconds as
# v1 (6000), so wall time stays ~34 h -- just 40 longer episodes instead of ~100.
MAX_STEPS_PER_EP=150
MAX_TIMESTEPS=6000
START_TIMESTEPS=600
SAVE_EVERY=500
# unused-by-freeform waveform knobs (kept so train.py's args are satisfied)
CONTROL_DT=10; EPISODE_DURATION=120
PERIOD_MIN=5; PERIOD_MAX=30; DUTY_MIN=0.1; DUTY_MAX=1.0
CAPTURE_EPISODES="1,20,final"

# =====================================================================
# OpenFOAM + python env -- IDENTICAL to the other RL slurms.
# =====================================================================
OPENFOAM_MODULE="OpenFOAM/v2506-foss-2025b"
set +u; module purge; module load "$OPENFOAM_MODULE"; source "$FOAM_BASH"; set -u
unset PYTHONPATH PYTHONHOME
PYTHON=/project/mwang/ibascom/envs/tc/bin/python

cd "$SLURM_SUBMIT_DIR"
EXP_DIR="experiments/catalysis_rl"
mkdir -p "$EXP_DIR/logs"
MAKE_CASE="experiments/oscillation_vs_constant/make_case.py"

# ---- per-task isolated case copy (persistent so 0.warmed survives) -
MASTER_CASE="taylor_couette_mixing/cases/tc_mixing_case"
RUN_ROOT="$EXP_DIR/_runs/${TAG}"            # per-task: array tasks must not share a case dir
RUN_CASE="$RUN_ROOT/tc_mixing_case"
rm -rf "$RUN_ROOT"; mkdir -p "$RUN_ROOT"
cp -r "$MASTER_CASE" "$RUN_CASE"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1

echo "=== TD3 catalysis v2 task $i: mode=$CONTROL_MODE energy_weight=$ENERGY_WEIGHT tag=$TAG ==="
echo "    horizon=$MAX_STEPS_PER_EP steps  case copy: $RUN_CASE   host: $(hostname)"

# ---- configure the copy for catalysis (IDENTICAL to v1) ------------
"$PYTHON" "$MAKE_CASE" \
    --case "$RUN_CASE" --geometry wedge --mode constant --catalysis \
    --mean-rpm "$MEAN_RPM" --run-seconds "$MAX_STEPS_PER_EP" --scalar-D "$SCALAR_D" \
    --max-co 1.5
( cd "$RUN_CASE" && blockMesh )
rm -rf "$RUN_CASE/0.warmed" "$RUN_CASE/0.orig"

# ---- train ---------------------------------------------------------
"$PYTHON" train.py \
    --algo "$ALGO" \
    --env catalysis \
    --control_mode "$CONTROL_MODE" \
    --control_dt "$CONTROL_DT" \
    --freeform_dt "$FREEFORM_DT" \
    --episode_duration "$EPISODE_DURATION" \
    --period_min "$PERIOD_MIN" --period_max "$PERIOD_MAX" \
    --duty_min "$DUTY_MIN" --duty_max "$DUTY_MAX" \
    --seed "$SEED" \
    --tag "$TAG" \
    --results_dir "$EXP_DIR/results" \
    --case_path "$RUN_CASE" \
    --r_in "$R_IN" \
    --r_out "$R_OUT" \
    --e_max_per_step "$E_MAX_PER_STEP" \
    --warmup_duration "$WARMUP_DURATION" \
    --warmup_omega_rpm "$WARMUP_OMEGA_RPM" \
    --conv_weight "$CONV_WEIGHT" \
    --energy_weight "$ENERGY_WEIGHT" \
    --energy_model "$ENERGY_MODEL" \
    --max_steps_per_ep "$MAX_STEPS_PER_EP" \
    --max_timesteps "$MAX_TIMESTEPS" \
    --start_timesteps "$START_TIMESTEPS" \
    --save_every "$SAVE_EVERY" \
    --capture_episodes "$CAPTURE_EPISODES"

echo "=== TD3 catalysis v2 task $i done ($TAG). checkpoints + frames -> $EXP_DIR/results/$ALGO/$TAG ==="
echo "    inspect:  python plot_td3_behavior.py $EXP_DIR/results/$ALGO/$TAG"
echo "    the winner is the tag whose learned omega SUSTAINS conversion at the lowest power."
