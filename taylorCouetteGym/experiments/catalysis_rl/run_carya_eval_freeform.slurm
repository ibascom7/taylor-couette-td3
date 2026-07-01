#!/bin/bash
# =====================================================================
# Carya: evaluate the trained FREEFORM waveform agent (catalysis_freeform_s<SEED>)
# and produce the three report timeseries Wang asked for:
#   omega(t), motor power(t), mixing-index/conversion(t)
# for the CONVERGED deterministic policy -- physically self-consistent, because
# they all come from ONE rollout through the same env the agent trained in.
#
# Why this and not run_carya_compare_catalysis.slurm: compare_catalysis.py builds
# an action_dim=1, NO-clock env, so it can only load the `omega` (constant) agent.
# The freeform agent has a phase clock in its state -> use eval_waveform_policy.py
# (--agent freeform), which is what this slurm drives.
#
# The training npy already give you omega(t) and conversion(t) with NO CFD (see
# plot_report_timeseries.py --from-npy), but energy/power is NOT logged there
# (the reward is wall-flux-driven, not conversion-driven), so the energy panel
# requires this rollout.
#
# Run AFTER run_carya_train_catalysis_td3.slurm produced catalysis_freeform_s<SEED>/
# td3_tc_final_*. Edit SEED to match.
#
# Submit:  sbatch experiments/catalysis_rl/run_carya_eval_freeform.slurm
# =====================================================================

#SBATCH -J tc_eval_freeform
#SBATCH -o experiments/catalysis_rl/logs/%x_%A.out
#SBATCH -e experiments/catalysis_rl/logs/%x_%A.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH -t 4:0:0          # 1 policy rollout + a small constant sweep (~8 rollouts)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bascomisaiah@gmail.com

set -euo pipefail

# ---- MUST MATCH run_carya_train_catalysis_td3.slurm (freeform task) ----------
SEED=0
FREEFORM_DT=1.0                 # s per control step (== training FREEFORM_DT)
EVAL_STEPS=60                   # == training MAX_STEPS_PER_EP (keeps energy-norm matched)
SCALAR_D=1e-9
R_IN=25.4; R_OUT=31.75
E_MAX_PER_STEP=0.0011017031875434
WARMUP_DURATION=100; WARMUP_OMEGA_RPM=500
ENERGY_MODEL=motor
MEAN_RPM=500                    # scalar omega make_case seeds (env overrides it)
CONST_SWEEP="0,300,500,800,1200,1600,2000,2500"   # constant-speed reference points

OPENFOAM_MODULE="OpenFOAM/v2506-foss-2025b"
set +u; module purge; module load "$OPENFOAM_MODULE"; source "$FOAM_BASH"; set -u
unset PYTHONPATH PYTHONHOME
PYTHON=/project/mwang/ibascom/envs/tc/bin/python

cd "$SLURM_SUBMIT_DIR"
EXP_DIR="experiments/catalysis_rl"
mkdir -p "$EXP_DIR/logs"
MAKE_CASE="experiments/oscillation_vs_constant/make_case.py"

POLICY_DIR="$EXP_DIR/results/td3/catalysis_freeform_s${SEED}"
POLICY="$POLICY_DIR/td3_tc_final"
if [ ! -f "${POLICY}_actor" ]; then
    # training timed out before final -> newest periodic checkpoint
    LATEST=$(ls -1 "$POLICY_DIR"/td3_tc_t*_actor 2>/dev/null \
             | sed -E 's#.*/td3_tc_t([0-9]+)_actor#\1 &#' | sort -n | tail -1 | cut -d' ' -f2-)
    [ -n "${LATEST:-}" ] || { echo "no freeform policy in $POLICY_DIR" >&2; exit 1; }
    POLICY="${LATEST%_actor}"
    echo "WARNING: no td3_tc_final; using newest checkpoint $(basename "$POLICY")" >&2
fi

# Fresh catalysis case for eval (independent of the training copy).
MASTER_CASE="taylor_couette_mixing/cases/tc_mixing_case"
RUN_ROOT="$EXP_DIR/_runs/eval_freeform_s${SEED}"
RUN_CASE="$RUN_ROOT/tc_mixing_case"
rm -rf "$RUN_ROOT"; mkdir -p "$RUN_ROOT"
cp -r "$MASTER_CASE" "$RUN_CASE"

"$PYTHON" "$MAKE_CASE" \
    --case "$RUN_CASE" --geometry wedge --mode constant --catalysis \
    --mean-rpm "$MEAN_RPM" --run-seconds "$EVAL_STEPS" --scalar-D "$SCALAR_D" \
    --max-co 1.5
( cd "$RUN_CASE" && blockMesh )
rm -rf "$RUN_CASE/0.warmed" "$RUN_CASE/0.orig"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1

OUT="$EXP_DIR/results/comparison_s${SEED}/freeform_eval"
echo "=== eval freeform seed=$SEED  policy=$(basename "$POLICY") -> $OUT ==="

"$PYTHON" "$EXP_DIR/eval_waveform_policy.py" \
    --agent freeform \
    --case_path "$RUN_CASE" \
    --policy "$POLICY" \
    --out "$OUT" \
    --freeform_dt "$FREEFORM_DT" \
    --eval_steps "$EVAL_STEPS" \
    --r_in "$R_IN" --r_out "$R_OUT" \
    --e_max_per_step "$E_MAX_PER_STEP" \
    --warmup_duration "$WARMUP_DURATION" --warmup_omega_rpm "$WARMUP_OMEGA_RPM" \
    --energy_model "$ENERGY_MODEL" \
    --const_sweep "$CONST_SWEEP" \
    --seed "$SEED"

# Wang's 3-panel report figure from the rollout CSV.
"$PYTHON" "$EXP_DIR/plot_report_timeseries.py" \
    --csv "$OUT/policy_trajectory.csv" --dt "$FREEFORM_DT" \
    --title "Learned freeform policy (converged, seed $SEED)"

echo "=== eval done. summary.json / policy_trajectory.csv / waveform_vs_constant.png"
echo "    + report_timeseries.png in $OUT ==="
ls -1 "$OUT"
