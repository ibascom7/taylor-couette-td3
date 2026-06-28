#!/bin/bash
# =====================================================================
# Carya: compare the trained TD3 agent against the constant and pulsating
# baselines the PAPER's way -- at EQUAL CONVERSION, using the electric-motor power
# model (Eqs. 18-23). Runs a constant-speed sweep and a pulsating-speed sweep,
# then the agent, producing Fig.7 (conversion vs mean speed), Fig.11 (power vs
# conversion + equal-conversion gap), the omega(t) waveforms (Fig.3 style), and
# ParaView frames, under experiments/catalysis_rl/results/comparison_s<SEED>/.
#
# The baseline sweeps are CACHED (baseline_sweep.npz), so re-running after
# retraining the agent only re-runs the agent. Add --refresh-baselines to force.
#
# Run AFTER run_carya_train_catalysis_td3.slurm has produced a td3_tc_final
# checkpoint. Edit SEED to match the trained seed you want to evaluate.
#
# Submit:  sbatch experiments/catalysis_rl/run_carya_compare_catalysis.slurm
# =====================================================================

#SBATCH -J tc_cmp_catalysis
#SBATCH -o experiments/catalysis_rl/logs/%x_%A.out
#SBATCH -e experiments/catalysis_rl/logs/%x_%A.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH -t 16:0:0          # ~9 baseline rollouts + the agent; pulsating peaks are slow
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bascomisaiah@gmail.com

set -euo pipefail

SEED=0                  # which trained seed to evaluate
EVAL_SECONDS=60         # 2 modulation periods
DUTY=0.2; PERIOD=30     # paper's best pulsating setting (Fig. 6: D~20% wins)
SCALAR_D=1e-9           # MUST match the training slurm's SCALAR_D (env flags must match);
                        # 1e-9 un-saturates conversion vs the old 1e-8 (see training slurm note)
# Speed sweeps (rpm). Pulsating peak = mean/DUTY, so keep PULS_RPM means <= ~500
# (peak <= 2500, the validated range). CONST_RPM should bracket the pulsating
# conversions so the equal-conversion interpolation stays in range.
CONST_RPM="300,500,700,900,1200"
PULS_RPM="300,400,500"
DUTY_SWEEP=""           # set e.g. "0.2,0.4,0.6,0.8" for a Fig.6 conversion-vs-duty plot
MEAN_RPM=500            # scalar omega make_case seeds (env overrides it)

# env params -- MUST match training (see run_carya_train_catalysis_td3.slurm)
R_IN=25.4; R_OUT=31.75
E_MAX_PER_STEP=0.0011017031875434
WARMUP_DURATION=100; WARMUP_OMEGA_RPM=500; OMEGA_MAX=2500  # warmup MUST match training (100/500)

OPENFOAM_MODULE="OpenFOAM/v2506-foss-2025b"
set +u; module purge; module load "$OPENFOAM_MODULE"; source "$FOAM_BASH"; set -u
unset PYTHONPATH PYTHONHOME
PYTHON=/project/mwang/ibascom/envs/tc/bin/python

cd "$SLURM_SUBMIT_DIR"
EXP_DIR="experiments/catalysis_rl"
mkdir -p "$EXP_DIR/logs"
MAKE_CASE="experiments/oscillation_vs_constant/make_case.py"

# Set BASELINES_ONLY=1 to run ONLY the constant+pulsating sweeps -- the paper's
# Fig 7 / Fig 11 reproduction -- with no TD3 agent (e.g. before any policy exists):
#     BASELINES_ONLY=1 sbatch experiments/catalysis_rl/run_carya_compare_catalysis.slurm
BASELINES_ONLY=${BASELINES_ONLY:-0}
# Training tags runs catalysis_<control_mode>_s<seed>; the omega mode is the one
# compare can evaluate (action_dim=1). (Waveform policies are action_dim=3 -> not
# loadable here; evaluate those via run_carya_cma_waveform.slurm / a waveform eval.)
POLICY_DIR="$EXP_DIR/results/td3/catalysis_omega_s${SEED}"
POLICY="$POLICY_DIR/td3_tc_final"
POLICY_ARGS="--baselines-only"
if [ "$BASELINES_ONLY" = "1" ]; then
    echo "BASELINES_ONLY=1 -> constant+pulsating sweep only (no TD3 agent)"
elif [ -f "${POLICY}_actor" ]; then
    POLICY_ARGS="--policy $POLICY"
else
    # No td3_tc_final (e.g. training timed out). Prefer the newest periodic
    # checkpoint (td3_tc_t<N>); if none exists yet, run baselines-only so the
    # paper sweep still completes instead of erroring out.
    LATEST=$(ls -1 "$POLICY_DIR"/td3_tc_t*_actor 2>/dev/null \
             | sed -E 's#.*/td3_tc_t([0-9]+)_actor#\1 &#' | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -n "${LATEST:-}" ]; then
        POLICY="${LATEST%_actor}"
        echo "WARNING: no td3_tc_final; using newest checkpoint $(basename "$POLICY")" >&2
        POLICY_ARGS="--policy $POLICY"
    else
        echo "no trained policy yet -> baselines-only (sweep reproduces the paper)" >&2
    fi
fi

# Fresh catalysis case copy for the evaluation (independent of the training copy).
MASTER_CASE="taylor_couette_mixing/cases/tc_mixing_case"
RUN_ROOT="$EXP_DIR/_runs/compare_s${SEED}"
RUN_CASE="$RUN_ROOT/tc_mixing_case"
rm -rf "$RUN_ROOT"; mkdir -p "$RUN_ROOT"
cp -r "$MASTER_CASE" "$RUN_CASE"

"$PYTHON" "$MAKE_CASE" \
    --case "$RUN_CASE" --geometry wedge --mode constant --catalysis \
    --mean-rpm "$MEAN_RPM" --run-seconds "$EVAL_SECONDS" --scalar-D "$SCALAR_D" \
    --max-co 1.5
# Rebuild the mesh from blockMeshDict (axial 150, not the shipped 600 polyMesh).
( cd "$RUN_CASE" && blockMesh )
rm -rf "$RUN_CASE/0.warmed" "$RUN_CASE/0.orig"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1

OUT="$EXP_DIR/results/comparison_s${SEED}"
echo "=== compare catalysis seed=$SEED -> $OUT ==="

"$PYTHON" "$EXP_DIR/compare_catalysis.py" \
    --case "$RUN_CASE" \
    $POLICY_ARGS \
    --out "$OUT" \
    --eval-seconds "$EVAL_SECONDS" \
    --const-rpm "$CONST_RPM" --puls-rpm "$PULS_RPM" \
    --duty "$DUTY" --period "$PERIOD" --duty-sweep "$DUTY_SWEEP" \
    --r_in "$R_IN" --r_out "$R_OUT" --e_max_per_step "$E_MAX_PER_STEP" \
    --warmup_duration "$WARMUP_DURATION" --warmup_omega_rpm "$WARMUP_OMEGA_RPM" \
    --omega_max "$OMEGA_MAX" --ramp_time 0.05 --seed "$SEED"

echo "=== compare done. figures + ParaView frames in $OUT ==="
ls -1 "$OUT"
