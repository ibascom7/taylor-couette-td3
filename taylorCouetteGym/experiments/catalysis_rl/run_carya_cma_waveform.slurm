#!/bin/bash
# =====================================================================
# Carya: CMA-ES search for the best burst-and-idle waveform on the CATALYSIS
# wedge -- the sample-efficient alternative to TD3 for the 3-parameter waveform
# (peak omega, duty D, period T). See cma_waveform.py for the why.
#
# Reuses the SAME catalysis env + case setup as the TD3 slurms, so the physics is
# identical and the result is directly comparable. The driver:
#   1. sweeps constant speeds (duty=1) -> constant power-vs-conversion curve + sets
#      the conversion target = constant 500 rpm's conversion (EQUAL-conversion test),
#   2. runs CMA-ES to MINIMIZE motor power subject to conv >= that target,
#   3. reports the power saving of the modulated waveform vs constant at equal
#      conversion (results/cma_s<SEED>/summary.json + power_vs_conversion.png).
#
# Cost ~ (#const refs + popsize*generations) waveform episodes, each
# --episode_duration s of pimpleFoam. Default ~105 evals x 120 s.
#
# Submit:  sbatch experiments/catalysis_rl/run_carya_cma_waveform.slurm
# =====================================================================

#SBATCH -J tc_cma_catalysis
#SBATCH -o experiments/catalysis_rl/logs/%x_%A.out
#SBATCH -e experiments/catalysis_rl/logs/%x_%A.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4                      # pimpleFoam runs SERIALLY; 4 covers numpy/cma
#SBATCH --mem=12G
#SBATCH -t 48:0:0                 # ~105 waveform episodes; 2500 rpm bursts are slow
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bascomisaiah@gmail.com

set -euo pipefail

SEED=0

# ---- geometry / env params -- MUST MATCH run_carya_train_catalysis_td3.slurm ----
R_IN=25.4
R_OUT=31.75
E_MAX_PER_STEP=0.0011017031875434
SCALAR_D=1e-9                  # MUST match the training slurm's SCALAR_D
MEAN_RPM=500                   # placeholder scalar omega make_case writes (env overrides)
WARMUP_DURATION=100; WARMUP_OMEGA_RPM=500   # MUST match training (warmup IC + energy norm)
ENERGY_MODEL=motor

# ---- waveform / CMA params -----------------------------------------
EPISODE_DURATION=90            # s per trial = 3.5 tau_res (tau_res=26 s) -> quasi-steady
PERIOD_MIN=5; PERIOD_MAX=30    # square-wave period range (s)
DUTY_MIN=0.1; DUTY_MAX=1.0     # DUTY_MAX must be 1.0 so duty=1 gives a clean constant ref
POPSIZE=8                      # CMA candidates per generation
MAX_EVALS=40                   # ~5 generations of 8; (9 const refs + 40) x 90 s ~ 43 h < 48 h.
                               # CMA has no mid-run checkpoint, so this is sized to FINISH;
                               # the driver logs each eval incrementally in case of timeout.
SIGMA0=0.35
OBJECTIVE=constrained          # min power s.t. conv >= target (the thesis)
REF_RPM=500                    # target conversion = this constant speed's conversion

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

# CMA-ES library (idempotent; no-op if already installed in the venv).
"$PYTHON" -m pip install --quiet cma || true

# ---- fresh per-run catalysis case copy ------------------------------
MASTER_CASE="taylor_couette_mixing/cases/tc_mixing_case"
RUN_ROOT="$EXP_DIR/_runs/cma_s${SEED}"
RUN_CASE="$RUN_ROOT/tc_mixing_case"
rm -rf "$RUN_ROOT"; mkdir -p "$RUN_ROOT"
cp -r "$MASTER_CASE" "$RUN_CASE"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1

echo "=== CMA-ES catalysis seed=$SEED  (Rin=$R_IN Rout=$R_OUT mm)  host: $(hostname) ==="

# Configure the copy for catalysis (same as the TD3 slurms), rebuild the mesh,
# and drop the stale MIXING ICs so the env re-warms a true CATALYTIC 0.warmed.
"$PYTHON" "$MAKE_CASE" \
    --case "$RUN_CASE" --geometry wedge --mode constant --catalysis \
    --mean-rpm "$MEAN_RPM" --run-seconds "$EPISODE_DURATION" --scalar-D "$SCALAR_D" \
    --max-co 1.5
( cd "$RUN_CASE" && blockMesh )
rm -rf "$RUN_CASE/0.warmed" "$RUN_CASE/0.orig"

OUT="$EXP_DIR/results/cma_s${SEED}"
echo "=== run CMA-ES -> $OUT ==="

"$PYTHON" "$EXP_DIR/cma_waveform.py" \
    --case_path "$RUN_CASE" \
    --out "$OUT" \
    --r_in "$R_IN" --r_out "$R_OUT" \
    --e_max_per_step "$E_MAX_PER_STEP" \
    --warmup_duration "$WARMUP_DURATION" --warmup_omega_rpm "$WARMUP_OMEGA_RPM" \
    --energy_model "$ENERGY_MODEL" \
    --episode_duration "$EPISODE_DURATION" \
    --period_min "$PERIOD_MIN" --period_max "$PERIOD_MAX" \
    --duty_min "$DUTY_MIN" --duty_max "$DUTY_MAX" \
    --popsize "$POPSIZE" --max_evals "$MAX_EVALS" --sigma0 "$SIGMA0" \
    --objective "$OBJECTIVE" --ref_rpm "$REF_RPM" \
    --seed "$SEED"

echo "=== CMA-ES done. results in $OUT ==="
ls -1 "$OUT"
