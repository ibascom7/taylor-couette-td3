#!/bin/bash
# =====================================================================
# Carya: train a TD3 agent on the CATALYSIS wedge.
#
# The catalytic-wall case feeds reactant at c0=1 and consumes it at the outer
# (catalytic) wall; the score is CONVERSION (1 - cup outlet C), not mixing. The
# agent picks an ABSOLUTE inner-cylinder omega in [0, 2500] rpm every simulated
# second (wide enough to reproduce the constant 500 rpm and square-wave 0/2500
# baselines, or invent its own waveform). Reward per step:
#       conv_weight * conversion  -  energy_weight * E_norm
# energy_weight is THE knob (see --energy_weight help / README): too high -> the
# agent idles omega->0; too low -> it pins omega->2500. Start at 0.1 and inspect
# the learned omega(t) in results/.../omega_per_step.npy.
#
# Field snapshots of selected EPISODES are kept for ParaView under
#   results/td3/<tag>/frames/ep<NNNN>/   (one time dir per simulated second),
# exactly like the constant/stepwise RL runs (--capture_episodes below).
#
# After training, compare the agent vs the two baselines on conversion:
#   sbatch run_carya_compare_catalysis.slurm     (drives compare_catalysis.py)
#
# Submit:  sbatch experiments/catalysis_rl/run_carya_train_catalysis_td3.slurm
# Monitor: squeue -u $USER
# =====================================================================

#SBATCH -J tc_td3_catalysis
#SBATCH -o experiments/catalysis_rl/logs/%x_%A_%a.out
#SBATCH -e experiments/catalysis_rl/logs/%x_%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4                      # pimpleFoam runs SERIALLY; 4 covers torch/numpy
#SBATCH --mem=12G
#SBATCH -t 24:0:0                 # cost driven by the 2500 rpm bursts (small dt)
#SBATCH --array=0                 # 1 seed; use --array=0-2 for 3 seeds
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --mail-user=bascomisaiah@gmail.com

set -euo pipefail

# ---- experiment grid: array index -> seed --------------------------
ALGO=td3
SEEDS=(0 1 2)
i=${SLURM_ARRAY_TASK_ID:-0}
SEED=${SEEDS[$i]}

# ---- geometry / normalizer (2D wedge tc_mixing_case, 20x1x150 = 3000 cells) ----
R_IN=25.4
R_OUT=31.75
E_MAX_PER_STEP=0.0011017031875434   # 1 s @ ~500 rpm on the wedge (reward energy scale)

# ---- catalysis setup -----------------------------------------------
MEAN_RPM=500            # placeholder scalar omega make_case writes (env overrides it)
SCALAR_D=1e-8           # species diffusivity (paper ~1e-10; 1e-8 keeps the BL resolvable)

# ---- reward weights ------------------------------------------------
CONV_WEIGHT=1.0
ENERGY_WEIGHT=0.1       # <-- tune this; the whole result hinges on it
ENERGY_MODEL=motor      # motor: paper's electric-motor power (Eqs 18-23), bearing-
                        # dominated + regenerative, so the agent CAN discover modulation.
                        # mechanical: viscous-drag work (convex) -> agent avoids bursts.

# ---- training hyperparameters --------------------------------------
WARMUP_DURATION=20      # spin up the catalytic field (cached as 0.warmed) at MEAN_RPM
WARMUP_OMEGA_RPM=500
MAX_STEPS_PER_EP=60     # 60 s episodes = 2 modulation periods
MAX_TIMESTEPS=1500
START_TIMESTEPS=100
SAVE_EVERY=150
CAPTURE_EPISODES="1,10,final"   # save these episodes' time dirs for ParaView

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
MAKE_CASE="experiments/oscillation_vs_constant/make_case.py"   # reuse the catalysis setup

# ---- per-task isolated case copy (persistent so 0.warmed survives) -
MASTER_CASE="taylor_couette_mixing/cases/tc_mixing_case"
RUN_ROOT="$EXP_DIR/_runs/${ALGO}_catalysis_s${SEED}"
RUN_CASE="$RUN_ROOT/tc_mixing_case"
rm -rf "$RUN_ROOT"; mkdir -p "$RUN_ROOT"
cp -r "$MASTER_CASE" "$RUN_CASE"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1

echo "=== TD3 catalysis task $i: seed=$SEED  (Rin=$R_IN Rout=$R_OUT mm) ==="
echo "    case copy: $RUN_CASE   host: $(hostname)"

# ---- configure the copy for catalysis ------------------------------
# make_case writes the catalytic-wall 0/C, the `catalysis` function object
# (emits the `conv` METRICS field), the adjustTimeStep controlDict, and a SCALAR
# inner_wall.omega the env can set. --mode constant just seeds that scalar; the
# RL env overwrites omega every step.
"$PYTHON" "$MAKE_CASE" \
    --case "$RUN_CASE" --geometry wedge --mode constant --catalysis \
    --mean-rpm "$MEAN_RPM" --run-seconds "$MAX_STEPS_PER_EP" --scalar-D "$SCALAR_D" \
    --max-co 1.5

# Rebuild the mesh from blockMeshDict (axial 300, not the shipped 600 polyMesh).
( cd "$RUN_CASE" && blockMesh )

# Drop the stale MIXING initial-condition snapshots so the env re-warms a true
# CATALYTIC 0.warmed/0.orig from the catalysis 0/ make_case just wrote.
rm -rf "$RUN_CASE/0.warmed" "$RUN_CASE/0.orig"

# ---- train ---------------------------------------------------------
"$PYTHON" train.py \
    --algo "$ALGO" \
    --env catalysis \
    --seed "$SEED" \
    --tag "catalysis_s${SEED}" \
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

echo "=== TD3 catalysis task $i done. checkpoints + frames -> $EXP_DIR/results/$ALGO/catalysis_s${SEED} ==="
echo "    next: sbatch $EXP_DIR/run_carya_compare_catalysis.slurm"
