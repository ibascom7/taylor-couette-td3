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
#SBATCH -t 50:0:0                 # long run; checkpoints every SAVE_EVERY guard against timeout
#SBATCH --array=0-1               # 0=omega (constant baseline)  1=freeform (the waveform agent)
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --mail-user=bascomisaiah@gmail.com

set -euo pipefail

# ---- experiment grid: array index -> CONTROL MODE ------------------
# Two TD3 runs on the catalytic wedge (one slurm task each):
#   0  omega     absolute omega each 1 s -> the best CONSTANT (the baseline to beat)
#   1  freeform  absolute omega every FREEFORM_DT s WITH an episode clock in the state,
#                so the deterministic policy can paint a free-form (messy) omega(t)
#                instead of collapsing to a constant -> THE waveform/modulation agent.
# (waveform_adaptive / waveform_episode are still valid --control_mode values, but the
#  parameterized waveform is better searched by CMA-ES -- run_carya_cma_waveform.slurm.)
ALGO=td3
SEED=0
MODES=(omega freeform)
i=${SLURM_ARRAY_TASK_ID:-0}
CONTROL_MODE=${MODES[$i]}
TAG="catalysis_${CONTROL_MODE}_s${SEED}"

# ---- geometry / normalizer (2D wedge tc_mixing_case, 20x1x150 = 3000 cells) ----
R_IN=25.4
R_OUT=31.75
E_MAX_PER_STEP=0.0011017031875434   # 1 s @ ~500 rpm on the wedge (reward energy scale)

# ---- catalysis setup -----------------------------------------------
MEAN_RPM=500            # placeholder scalar omega make_case writes (env overrides it)
SCALAR_D=1e-9           # species diffusivity. 1e-8 OVER-converted (conv saturated ~0.97 for
                        # all omega -> agent had no gradient -> idle-collapse). 1e-9 thins the
                        # wall boundary layer so conversion DEPENDS on omega again (gives the
                        # agent a real spin-vs-idle trade-off). Paper uses ~1e-10 WITH near-wall
                        # mesh refinement; the thinner the BL, the more our uniform 20-cell radial
                        # mesh under-resolves it, so conv gets numerical-diffusion-tinged at low D.
                        # MUST match the compare slurm's SCALAR_D.

# ---- reward weights ------------------------------------------------
CONV_WEIGHT=1.0
ENERGY_WEIGHT=0.1       # <-- tune this; the whole result hinges on it
ENERGY_MODEL=motor      # motor: paper's electric-motor power (Eqs 18-23), bearing-
                        # dominated + regenerative, so the agent CAN discover modulation.
                        # mechanical: viscous-drag work (convex) -> agent avoids bursts.

# ---- training hyperparameters --------------------------------------
WARMUP_DURATION=100      # spin up the catalytic field (cached as 0.warmed) at WARMUP_OMEGA_RPM.
                        # 100 s ~ 3.8 residence times AND ~27 viscous times (tau_res=26 s at
                        # Gamma=6), so 0.warmed is a SETTLED 500-rpm state -> every episode
                        # starts settled (no spin-up transient). Cached. MUST match the
                        # compare + CMA slurms (warmup IC AND the motor energy normalizer).
WARMUP_OMEGA_RPM=500    # constant-500 reference; 500 (not 1000) keeps the energy normalizer
                        # on the baseline and doesn't bias the starting conversion high.
# Per-mode budgets (CFD cost ~ total SIMULATED seconds; each mode ~similar budget,
# different #decisions). Waveform params apply to modes 1 & 2.
CONTROL_DT=10                  # [waveform_adaptive] s per control update
EPISODE_DURATION=120           # [waveform_episode] s the chosen waveform runs per trial
PERIOD_MIN=5; PERIOD_MAX=30    # [waveform modes] square-wave period range (s)
DUTY_MIN=0.1; DUTY_MAX=1.0     # [waveform modes] duty range; low duty = brief bursts (paper's regime)
FREEFORM_DT=1.0                # [freeform] s of sim per control step (== writeInterval; see below)
case "$CONTROL_MODE" in
  # omega = the CONSTANT baseline. ~5000 transitions at the corrected SETTLED cost
  # (~30 s/sim-second averaged over exploration) ~ 50 h. Checkpoints every 250 steps,
  # so a wall-timeout still leaves a usable policy (compare auto-falls-back).
  omega)             MAX_STEPS_PER_EP=60;  MAX_TIMESTEPS=5000;  START_TIMESTEPS=500;  SAVE_EVERY=250  ;;
  # freeform = THE waveform agent: absolute omega every FREEFORM_DT s + an episode clock,
  # so the policy paints a free-form omega(t). FREEFORM_DT=1 s -> writeInterval=1 (one field
  # write per step, exactly like omega) -> freeform is literally "omega + clock". Each
  # transition is 1 sim-s -> 50 h buys ~6000 transitions (~6x the parameterized waveform;
  # plenty for a 1-D action). 60 steps x 1 s = 60 s episode (2.3 tau_res). (Drop FREEFORM_DT
  # to 0.5 for 2x the transitions + finer control + writeInterval 0.5, if ever wanted.)
  freeform)          MAX_STEPS_PER_EP=60;  MAX_TIMESTEPS=6000;  START_TIMESTEPS=600;  SAVE_EVERY=500  ;;
  # waveform_adaptive = THE run of interest: a closed-loop RL agent that learns a
  # MODULATION strategy (peak,duty,period, now in the state) to beat the best constant.
  # ~1000 control updates x 10 s = 10000 sim-s ~ 50 h. CAVEAT: each transition costs
  # 10 sim-s, so 50 h buys only ~1000 transitions for a 3-D action -> DATA-LIMITED for
  # TD3 (it usually wants 1e5+). Run CMA-ES (run_carya_cma_waveform.slurm) alongside as
  # the sample-efficient modulation finder that produces the beats-constant figure directly.
  waveform_adaptive) MAX_STEPS_PER_EP=12; MAX_TIMESTEPS=1000; START_TIMESTEPS=150; SAVE_EVERY=100 ;;  # 12x10s=120s/ep
  # waveform_episode (3-param bandit) is SUPERSEDED by CMA-ES; kept tiny for a smoke test only.
  waveform_episode)  MAX_STEPS_PER_EP=1;  MAX_TIMESTEPS=15;   START_TIMESTEPS=4;   SAVE_EVERY=3   ;;  # use CMA instead
esac
CAPTURE_EPISODES="1,5,final"   # save these episodes' time dirs for ParaView

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
RUN_ROOT="$EXP_DIR/_runs/${ALGO}_${CONTROL_MODE}_s${SEED}"   # per-mode: array tasks must not share a case dir
RUN_CASE="$RUN_ROOT/tc_mixing_case"
rm -rf "$RUN_ROOT"; mkdir -p "$RUN_ROOT"
cp -r "$MASTER_CASE" "$RUN_CASE"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1

echo "=== TD3 catalysis task $i: mode=$CONTROL_MODE seed=$SEED  (Rin=$R_IN Rout=$R_OUT mm) ==="
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

# Rebuild the mesh from blockMeshDict (axial 150, not the shipped 600 polyMesh).
( cd "$RUN_CASE" && blockMesh )

# Drop the stale MIXING initial-condition snapshots so the env re-warms a true
# CATALYTIC 0.warmed/0.orig from the catalysis 0/ make_case just wrote.
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

echo "=== TD3 catalysis task $i done. checkpoints + frames -> $EXP_DIR/results/$ALGO/$TAG ==="
echo "    next: sbatch $EXP_DIR/run_carya_compare_catalysis.slurm"
