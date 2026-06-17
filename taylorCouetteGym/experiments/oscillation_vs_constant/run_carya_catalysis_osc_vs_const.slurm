#!/bin/bash
# =====================================================================
# Carya: oscillating vs constant omega on the CATALYSIS wedge -- the paper's
# ACTUAL experiment. Reactant fed at c0=1, consumed at the catalytic outer wall
# (C=0 sink); we measure CONVERSION (1 - cup outlet C) and wall flux, not mixing
# uniformity. This is the metric where modulation is expected to win (it thins
# the wall concentration boundary layer -> more flux -> higher conversion).
#
#   task 0  constant     omega = 500 rpm steady
#   task 1  squarewave   omega = 0/2500 rpm, D=0.2, T=30 s, mean 500 rpm
#
# Same geometry/omega machinery as run_carya_oscillation_vs_constant.slurm; the
# only change is --catalysis (wall sink + conversion FO). Logs -> results_catalysis/.
#
# Submit:  sbatch run_carya_catalysis_osc_vs_const.slurm
# After both finish:  python analyze_catalysis.py results_catalysis
# =====================================================================

#SBATCH -J tc_cat_osc_vs_const
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 12:0:0
#SBATCH --array=0-1
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --mail-user=bascomisaiah@gmail.com

set -euo pipefail

# ---- experiment parameters -----------------------------------------
MEAN_RPM=500; DUTY=0.2; PERIOD=30
RUN_SECONDS=60           # 2 modulation periods past startup
MAX_CO=0.8
SCALAR_D=1e-8            # species diffusivity (paper ~1e-10; 1e-8 keeps BL resolvable)

MODES=( constant squarewave )
i=${SLURM_ARRAY_TASK_ID:-0}
MODE=${MODES[$i]}

# ---- OpenFOAM + python env -----------------------------------------
OPENFOAM_MODULE="OpenFOAM/v2506-foss-2025b"
set +u; module purge; module load "$OPENFOAM_MODULE"; source "$FOAM_BASH"; set -u
unset PYTHONPATH PYTHONHOME
PYTHON=/project/mwang/ibascom/envs/tc/bin/python

cd "$SLURM_SUBMIT_DIR"
EXP_DIR="experiments/oscillation_vs_constant"
mkdir -p "$EXP_DIR/logs" "$EXP_DIR/results_catalysis"

MASTER_CASE="taylor_couette_mixing/cases/tc_mixing_case"
# Run in a PERSISTENT folder (not node-local $SLURM_TMPDIR) so the per-second
# field time-dirs survive for ParaView -- even if the job times out.
RUN_ROOT="$EXP_DIR/results_catalysis/case_${MODE}"
RUN_CASE="$RUN_ROOT/tc_mixing_case"
rm -rf "$RUN_ROOT"; mkdir -p "$RUN_ROOT"
cp -r "$MASTER_CASE" "$RUN_CASE"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1

echo "=== catalysis $MODE: run=${RUN_SECONDS}s mean=${MEAN_RPM}rpm D=${SCALAR_D} ==="

"$PYTHON" "$EXP_DIR/make_case.py" \
    --case "$RUN_CASE" --geometry wedge --mode "$MODE" --catalysis \
    --mean-rpm "$MEAN_RPM" --duty "$DUTY" --period "$PERIOD" \
    --run-seconds "$RUN_SECONDS" --max-co "$MAX_CO" --scalar-D "$SCALAR_D"

( cd "$RUN_CASE" && blockMesh )

LOG="$EXP_DIR/results_catalysis/${MODE}.log"
# field snapshots (writeInterval=1 -> one per simulated second) land straight in
# $RUN_CASE on persistent disk as it solves
( cd "$RUN_CASE" && pimpleFoam ) | tee "$LOG" | grep --line-buffered "^METRICS" || true

touch "$RUN_CASE/case.foam"        # ParaView entry point
echo "=== catalysis $MODE done -> $LOG ==="
echo "    time dirs for ParaView: $RUN_CASE/case.foam"
echo "    frames: $(ls -d "$RUN_CASE"/[0-9]* 2>/dev/null | wc -l)"
echo "    when both finish:  $PYTHON $EXP_DIR/analyze_catalysis.py $EXP_DIR/results_catalysis"
