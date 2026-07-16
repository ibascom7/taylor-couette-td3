#!/bin/bash
# =====================================================================
# Carya: EVAL the graded-case parallel-TD3 energy_weight sweep against the
# baselines, producing the beats-constant figure for each learned freeform policy.
#
# Drives experiments/catalysis_rl/compare_catalysis.py on the RESOLVED-GRADIENT
# graded case (side_outlet_grad_case):
#   fig_power_vs_conversion.png  <- THE figure: avg motor power vs conversion;
#                                   the RL point sitting BELOW the constant curve
#                                   at equal conversion == RL beats constant.
#   fig_conversion_vs_speed.png, fig_omega_ts.png (learned waveform), summary.txt
#
# The constant + pulsating BASELINE sweeps are policy-independent, so this job
# computes them ONCE (cached baseline_sweep.npz + frames) and every policy eval
# reuses them -- each of the 5 policies then costs just one extra rollout.
#
# compare_catalysis.py is SERIAL (one pimpleFoam at a time) and this case is
# EXPENSIVE (41-66 s/sim-s below 1000 rpm, ~179 at the 2500 rpm squarewave peaks;
# measured on the steady_state_conversion graded sweep, 7800X3D). At a ~1.5x
# Carya-core factor: baselines ~25 h + 5 policy evals ~17 h = ~42 h expected,
# ~56 h if the cores run 2x slower -> -t 72 h. (A timeout mid-loop is costly:
# the job start wipes the baseline cache, so a resubmit recomputes all ~25 h of
# baselines. If 72 h is ever tight, split: one --baselines-only job, then a
# second job without the rm -rf "$BASE".)
# Small 1-node job, NOT the 48-core training node. Run AFTER the training sweep.
#
# Submit:  sbatch experiments/parallelized_graded_rl/run_carya_compare.slurm
# =====================================================================

#SBATCH -J tc_compare_graded
#SBATCH -o experiments/parallelized_graded_rl/logs/%x_%j.out
#SBATCH -e experiments/parallelized_graded_rl/logs/%x_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2                      # compare is SERIAL; 2 cores cover pimpleFoam + python
#SBATCH --mem=4G
#SBATCH -t 72:0:0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bascomisaiah@gmail.com

set -euo pipefail

# ---- which sweep to eval (MUST match the training run) ---------------
CONTROL_MODE=freeform
SEED=0
ENERGY_WEIGHTS=(0.3 0.6 0.8 1.3 1.8)             # MUST match run_carya_parallel_graded.slurm's ew grid

# ---- eval config ----------------------------------------------------
# EVAL_SECONDS must equal training's MAX_STEPS_PER_EP so the freeform phase clock
# (phase = step/max_steps) sweeps the SAME 0->1 the policy learned -> faithful waveform.
EVAL_SECONDS=120
# Constant means bracket the swept ew grid's optima (~250 rpm at ew3.0 up to
# ~750-1000 rpm at ew0.8; see the training slurm's calibration table). No 2500 rpm
# point -- it alone would cost ~6 h and its steady conversion (0.568) is already
# known from the steady_state_conversion graded-Sc=1075 sweep if an anchor is needed.
CONST_RPM="250,500,750,1000,1200"
PULS_RPM="300,400,500"             # pulsating means (peak = mean/duty <= 2500 at duty 0.2)
DUTY=0.2
PERIOD=30
WARMUP_DURATION=80                 # match training (~3 tau at 100 mL/min)
WARMUP_OMEGA_RPM=500
FEED_VELOCITY=1.462e-3             # match training (100 mL/min)
WALLFLUX_MAX=1.32e-8               # MATCH training: sets the wf_norm scaling in the obs the
                                   # policy conditions on (default would auto = Q*c0 = 2.315e-8,
                                   # a DIFFERENT scale -> policy sees the wrong state -> wrong actions)

# =====================================================================
OPENFOAM_MODULE="OpenFOAM/v2506-foss-2025b"
set +u; module purge; module load "$OPENFOAM_MODULE"; source "$FOAM_BASH"; set -u
unset PYTHONPATH PYTHONHOME
PYTHON=/project/mwang/ibascom/envs/tc/bin/python
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
EXP_DIR="experiments/parallelized_graded_rl"
CMP="experiments/catalysis_rl/compare_catalysis.py"
# MUST match run_carya_parallel_graded.slurm's RESULTS_DIR: read the checkpoints
# from, and write the comparison into, the same tree.
RESULTS_DIR="$EXP_DIR/results"
TRAIN_RESULTS="$RESULTS_DIR/td3"
OUTROOT="$RESULTS_DIR/comparison"
BASE="$OUTROOT/_baselines"
mkdir -p "$EXP_DIR/logs" "$OUTROOT"

# node-local case, warmed ONCE by the first compare call and reused by all rollouts
CASE="${TMPDIR:-/tmp/$USER}/sog_compare_case"
rm -rf "$CASE"; cp -r taylor_couette_mixing/cases/side_outlet_grad_case "$CASE"
( cd "$CASE" && blockMesh > log.blockMesh 2>&1 )

COMMON=(--case "$CASE" --control_mode "$CONTROL_MODE" --eval-seconds "$EVAL_SECONDS"
        --const-rpm "$CONST_RPM" --puls-rpm "$PULS_RPM" --duty "$DUTY" --period "$PERIOD"
        --warmup_duration "$WARMUP_DURATION" --warmup_omega_rpm "$WARMUP_OMEGA_RPM"
        --feed_velocity "$FEED_VELOCITY" --wallflux_max "$WALLFLUX_MAX" --seed "$SEED")

# ---- 1) baseline sweeps ONCE (warms the case; caches baseline_sweep.npz + frames) ----
# Wipe any stale cache first: cache_valid() only checks params, not values, so a
# leftover tree from an earlier/aborted run would silently be reused.
echo "=== baselines (constant + pulsating sweeps), computed once ==="
rm -rf "$BASE"
"$PYTHON" "$CMP" --baselines-only --refresh-baselines --out "$BASE" "${COMMON[@]}"

# ---- 2) per-policy eval, reusing the cached baselines + frames -------
for EW in "${ENERGY_WEIGHTS[@]}"; do
  TAG="sog_parallel_${CONTROL_MODE}_ew${EW}_s${SEED}"                   # match training's output tag
  POLICY="$TRAIN_RESULTS/$TAG/td3_tc_final"
  if [ ! -f "${POLICY}_actor" ]; then   # timeout fallback: newest td3_tc_t<N>
    POLICY=$(ls -d "$TRAIN_RESULTS/$TAG"/td3_tc_t*_actor 2>/dev/null | sort -V | tail -1 | sed 's/_actor$//' || true)
  fi
  if [ -z "${POLICY:-}" ] || [ ! -f "${POLICY}_actor" ]; then
    echo "SKIP $TAG: no checkpoint found"; continue
  fi
  OUT="$OUTROOT/$TAG"
  rm -rf "$OUT"; mkdir -p "$OUT"                                        # clean: drop any stale figures/frames
  cp -f "$BASE/baseline_sweep.npz" "$OUT/"                              # reuse fresh sweep
  ln -sfn "$(readlink -f "$BASE/frames_constant")"   "$OUT/frames_constant"   # skip re-CFD
  ln -sfn "$(readlink -f "$BASE/frames_squarewave")" "$OUT/frames_squarewave"
  echo "=== compare $TAG  (policy=$POLICY) ==="
  "$PYTHON" "$CMP" --policy "$POLICY" --out "$OUT" "${COMMON[@]}"
done

echo "=== done. beats-constant figures -> $OUTROOT/<tag>/fig_power_vs_conversion.png ==="
echo "    verdicts:  grep -H 'vs constant at equal conversion' $OUTROOT/*/summary.txt"
