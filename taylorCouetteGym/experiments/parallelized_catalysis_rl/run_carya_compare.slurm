#!/bin/bash
# =====================================================================
# Carya: EVAL the parallel-TD3 energy_weight sweep against the baselines,
# producing the beats-constant figure for each learned freeform policy.
#
# Drives experiments/catalysis_rl/compare_catalysis.py on the SIDE-OUTLET case:
#   fig_power_vs_conversion.png  <- THE figure: avg motor power vs conversion;
#                                   the RL point sitting BELOW the constant curve
#                                   at equal conversion == RL beats constant.
#   fig_conversion_vs_speed.png, fig_omega_ts.png (learned waveform), summary.txt
#
# The constant + pulsating BASELINE sweeps are policy-independent, so this job
# computes them ONCE (cached baseline_sweep.npz + frames) and every policy eval
# reuses them -- so each of the 6 policies costs just one extra rollout.
#
# compare_catalysis.py is SERIAL (one pimpleFoam at a time), so this is a small
# 1-node job, NOT the 48-core training node. Run AFTER the training sweep finishes.
#
# Submit:  sbatch experiments/parallelized_catalysis_rl/run_carya_compare.slurm
# =====================================================================

#SBATCH -J tc_compare_parallel
#SBATCH -o experiments/parallelized_catalysis_rl/logs/%x_%j.out
#SBATCH -e experiments/parallelized_catalysis_rl/logs/%x_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2                      # compare is SERIAL; 2 cores cover pimpleFoam + python
#SBATCH --mem=4G
#SBATCH -t 12:0:0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bascomisaiah@gmail.com

set -euo pipefail

# ---- which sweep to eval (MUST match the training run) ---------------
CONTROL_MODE=freeform
SEED=0
ENERGY_WEIGHTS=(0.01 0.03 0.05 0.07 0.09)        # match the training array's ew list
IC_NAMES=(idle spin)                             # match the training array's two-IC probe

# ---- eval config ----------------------------------------------------
# EVAL_SECONDS must equal training's MAX_STEPS_PER_EP so the freeform phase clock
# (phase = step/max_steps) sweeps the SAME 0->1 the policy learned -> faithful waveform.
EVAL_SECONDS=120
CONST_RPM="300,500,700,900,1200"   # constant mean speeds to sweep (rpm)
PULS_RPM="300,400,500"             # pulsating means (peak = mean/duty, keep <= ~2500)
DUTY=0.2
PERIOD=30
WARMUP_DURATION=80                 # match training (shortened side-outlet ~3 tau)
WARMUP_OMEGA_RPM=500
FEED_VELOCITY=1.462e-3             # match training (100 mL/min)

# =====================================================================
OPENFOAM_MODULE="OpenFOAM/v2506-foss-2025b"
set +u; module purge; module load "$OPENFOAM_MODULE"; source "$FOAM_BASH"; set -u
unset PYTHONPATH PYTHONHOME
PYTHON=/project/mwang/ibascom/envs/tc/bin/python
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
EXP_DIR="experiments/parallelized_catalysis_rl"
CMP="experiments/catalysis_rl/compare_catalysis.py"
TRAIN_RESULTS="$EXP_DIR/results/td3"
OUTROOT="$EXP_DIR/results/comparison"
BASE="$OUTROOT/_baselines"
mkdir -p "$EXP_DIR/logs" "$OUTROOT"

# node-local case, warmed ONCE by the first compare call and reused by all rollouts
CASE="${TMPDIR:-/tmp/$USER}/so_compare_case"
rm -rf "$CASE"; cp -r taylor_couette_mixing/cases/side_outlet_cat_case "$CASE"
( cd "$CASE" && blockMesh > log.blockMesh 2>&1 )

COMMON=(--case "$CASE" --control_mode "$CONTROL_MODE" --eval-seconds "$EVAL_SECONDS"
        --const-rpm "$CONST_RPM" --puls-rpm "$PULS_RPM" --duty "$DUTY" --period "$PERIOD"
        --warmup_duration "$WARMUP_DURATION" --warmup_omega_rpm "$WARMUP_OMEGA_RPM"
        --feed_velocity "$FEED_VELOCITY" --seed "$SEED")

# ---- 1) baseline sweeps ONCE (warms the case; caches baseline_sweep.npz + frames) ----
echo "=== baselines (constant + pulsating sweeps), computed once ==="
"$PYTHON" "$CMP" --baselines-only --out "$BASE" "${COMMON[@]}"

# ---- 2) per-policy eval, reusing the cached baselines + frames -------
for EW in "${ENERGY_WEIGHTS[@]}"; do
 for IC in "${IC_NAMES[@]}"; do                                        # two-IC probe: idle & spin tags
  TAG="so_parallel_${CONTROL_MODE}_ew${EW}_ic${IC}_s${SEED}"
  POLICY="$TRAIN_RESULTS/$TAG/td3_tc_final"
  if [ ! -f "${POLICY}_actor" ]; then   # timeout fallback: newest td3_tc_t<N>
    POLICY=$(ls -d "$TRAIN_RESULTS/$TAG"/td3_tc_t*_actor 2>/dev/null | sort -V | tail -1 | sed 's/_actor$//' || true)
  fi
  if [ -z "${POLICY:-}" ] || [ ! -f "${POLICY}_actor" ]; then
    echo "SKIP $TAG: no checkpoint found"; continue
  fi
  OUT="$OUTROOT/$TAG"
  mkdir -p "$OUT"
  cp -f "$BASE/baseline_sweep.npz" "$OUT/"                              # reuse cached sweep
  ln -sfn "$(readlink -f "$BASE/frames_constant")"   "$OUT/frames_constant"   # skip re-CFD
  ln -sfn "$(readlink -f "$BASE/frames_squarewave")" "$OUT/frames_squarewave"
  echo "=== compare $TAG  (policy=$POLICY) ==="
  "$PYTHON" "$CMP" --policy "$POLICY" --out "$OUT" "${COMMON[@]}"
 done
done

echo "=== done. beats-constant figures -> $OUTROOT/<tag>/fig_power_vs_conversion.png ==="
echo "    verdicts:  grep -H 'vs constant at equal conversion' $OUTROOT/*/summary.txt"
