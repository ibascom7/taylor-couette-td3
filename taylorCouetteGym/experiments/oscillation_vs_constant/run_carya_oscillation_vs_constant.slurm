#!/bin/bash
# =====================================================================
# Carya: oscillating (square-wave) vs constant inner-cylinder omega on the
# 2D-axisymmetric wedge (tc_mixing_case), paper geometry (Lopez-Guajardo
# CEJ 489 (2024) 151174). This is a STRAIGHT pimpleFoam study -- NO RL, no
# train.py. Each array task prescribes omega(t) on the inner wall and runs
# pimpleFoam from a cold start for --run-seconds, logging the outlet radial
# dye profile (METRICS lines) that analyze.py scores mixing on.
#
#   task 0  constant     omega = 500 rpm, steady
#   task 1  squarewave   omega = 0/2500 rpm, D=0.2, T=30 s, mean 500 rpm
#
# Both share geometry, mean speed and dye injection, so any mixing difference
# is due to the modulation -- the paper's core claim.
#
# Submit:   sbatch run_carya_oscillation_vs_constant.slurm
# After both finish:  python analyze.py results
# Monitor:  squeue -u $USER ;  tail -f logs/tc_osc_*_*.out
# =====================================================================

#SBATCH -J tc_osc_vs_const
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 8:0:0
#SBATCH --array=0-1
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --mail-user=bascomisaiah@gmail.com
# #SBATCH -A <your_allocation>
# #SBATCH -p <partition>

set -euo pipefail

# ---- experiment parameters (shared) --------------------------------
MEAN_RPM=500
DUTY=0.2
PERIOD=30
RUN_SECONDS=120          # 4 modulation periods; see README for cost vs this knob
MAX_CO=0.8               # raise toward ~2 (with PIMPLE correctors) to go faster

# ---- task -> mode --------------------------------------------------
MODES=( constant squarewave )
i=${SLURM_ARRAY_TASK_ID:-0}
MODE=${MODES[$i]}

# ---- OpenFOAM + python env (see carya-environment memory) ----------
OPENFOAM_MODULE="OpenFOAM/v2506-foss-2025b"
set +u
module purge
module load "$OPENFOAM_MODULE"
source "$FOAM_BASH"        # puts blockMesh/pimpleFoam on PATH
set -u
unset PYTHONPATH PYTHONHOME
PYTHON=/project/mwang/ibascom/envs/tc/bin/python

cd "$SLURM_SUBMIT_DIR"
EXP_DIR="experiments/oscillation_vs_constant"
mkdir -p "$EXP_DIR/logs" "$EXP_DIR/results"

# ---- isolated per-task case copy on node-local scratch -------------
MASTER_CASE="taylor_couette_mixing/cases/tc_mixing_case"
RUN_ROOT="${SLURM_TMPDIR:-$SLURM_SUBMIT_DIR/_runs}/osc_${MODE}"
RUN_CASE="$RUN_ROOT/tc_mixing_case"
rm -rf "$RUN_ROOT"; mkdir -p "$RUN_ROOT"
cp -r "$MASTER_CASE" "$RUN_CASE"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1

echo "=== osc-vs-const task $i: mode=$MODE  run=${RUN_SECONDS}s  mean=${MEAN_RPM}rpm ==="
echo "    host=$(hostname)  case=$RUN_CASE"

# ---- configure omega(t) + controlDict (no OpenFOAM needed here) ----
"$PYTHON" "$EXP_DIR/make_case.py" \
    --case "$RUN_CASE" \
    --geometry wedge \
    --mode "$MODE" \
    --mean-rpm "$MEAN_RPM" \
    --duty "$DUTY" \
    --period "$PERIOD" \
    --run-seconds "$RUN_SECONDS" \
    --max-co "$MAX_CO"

# ---- mesh (paper geometry lives in blockMeshDict) ------------------
( cd "$RUN_CASE" && blockMesh )

# ---- solve; tee METRICS log to the persistent results dir ----------
LOG="$EXP_DIR/results/${MODE}.log"
echo "    solving -> $LOG"
( cd "$RUN_CASE" && pimpleFoam ) | tee "$LOG" | grep --line-buffered "^METRICS" || true

# keep the field time-dirs for ParaView (optional; comment out to save space)
cp -r "$RUN_CASE" "$EXP_DIR/results/case_${MODE}" 2>/dev/null || true

echo "=== task $i ($MODE) done -> $LOG ==="
echo "    when both tasks finish:  $PYTHON $EXP_DIR/analyze.py $EXP_DIR/results"
