#!/bin/bash
# =====================================================================
# Carya: oscillating (square-wave) vs constant inner-cylinder omega on the
# FULL 3D annulus (full_tc_mixing_case), paper geometry (Lopez-Guajardo
# CEJ 489 (2024) 151174, r_i=25.4 r_o=31.75 H=190.5 mm, 87552 cells).
#
# Same study as run_carya_oscillation_vs_constant.slurm but in 3D -- it resolves
# the azimuthal (wavy-vortex) structure the 2D wedge cannot. Straight pimpleFoam,
# NO RL/train.py. Each task prescribes omega(t) on the inner wall and logs the
# outlet radial dye profile (METRICS) that analyze.py scores mixing on.
#
#   task 0  constant     omega = 500 rpm steady
#   task 1  squarewave   omega = 0/2500 rpm, D=0.2, T=30 s, mean 500 rpm
#
# COST (SERIAL, measured ~208 ms/step on the 87552-cell mesh via smoke run):
# active phase holds dt~3.6e-4 s (Co 0.8) at 2500 rpm; idle phase ~5x cheaper;
# maxDeltaT caps the constant case. At RUN_SECONDS=60 (2 periods): constant ~3.5 h,
# squarewave ~4.5-5 h; concurrent as a 2-task array so wall ~5 h. -t is 24 h (lots
# of margin). **Check the first minute's ExecutionTime to confirm on Carya.**
# Lower RUN_SECONDS or parallelize (see note below) to go faster.
#
# Submit:   sbatch run_carya_oscillation_vs_constant_3d.slurm
# After both:  python analyze.py results_3d
# =====================================================================

#SBATCH -J tc_osc_vs_const_3d
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 24:0:0
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
RUN_SECONDS=60           # 2 modulation periods; raise/lower to trade cost
MAX_CO=0.8

# ---- task -> mode --------------------------------------------------
MODES=( constant squarewave )
i=${SLURM_ARRAY_TASK_ID:-0}
MODE=${MODES[$i]}

# ---- OpenFOAM + python env (see carya-environment memory) ----------
OPENFOAM_MODULE="OpenFOAM/v2506-foss-2025b"
set +u
module purge
module load "$OPENFOAM_MODULE"
source "$FOAM_BASH"
set -u
unset PYTHONPATH PYTHONHOME
PYTHON=/project/mwang/ibascom/envs/tc/bin/python

cd "$SLURM_SUBMIT_DIR"
EXP_DIR="experiments/oscillation_vs_constant"
mkdir -p "$EXP_DIR/logs" "$EXP_DIR/results_3d"

# ---- isolated per-task case copy on node-local scratch -------------
MASTER_CASE="taylor_couette_mixing/cases/full_tc_mixing_case"   # PAPER geometry
RUN_ROOT="${SLURM_TMPDIR:-$SLURM_SUBMIT_DIR/_runs}/osc3d_${MODE}"
RUN_CASE="$RUN_ROOT/full_tc_mixing_case"
rm -rf "$RUN_ROOT"; mkdir -p "$RUN_ROOT"
cp -r "$MASTER_CASE" "$RUN_CASE"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1

echo "=== osc-vs-const 3D task $i: mode=$MODE  run=${RUN_SECONDS}s  mean=${MEAN_RPM}rpm ==="
echo "    host=$(hostname)  case=$RUN_CASE"

# ---- configure omega(t) + controlDict (full3d patch layout) --------
"$PYTHON" "$EXP_DIR/make_case.py" \
    --case "$RUN_CASE" \
    --geometry full3d \
    --mode "$MODE" \
    --mean-rpm "$MEAN_RPM" \
    --duty "$DUTY" \
    --period "$PERIOD" \
    --run-seconds "$RUN_SECONDS" \
    --max-co "$MAX_CO"

# ---- mesh (paper geometry, 87552 cells) ----------------------------
( cd "$RUN_CASE" && blockMesh )

# ---- solve serially; tee METRICS to the persistent results dir -----
# To PARALLELIZE (recommended if you have cores to spare): write a scotch
# decomposeParDict with N subdomains, then
#     ( cd "$RUN_CASE" && decomposePar -force && \
#       mpirun -np N pimpleFoam -parallel )
# The rlMetrics/scalarTransport FOs are parallel-safe (reduce()); METRICS still
# print once on the master. Pre-compile the coded FOs with a 1-step serial run
# first to avoid a parallel dynamicCode compile race.
LOG="$EXP_DIR/results_3d/${MODE}.log"
echo "    solving (serial) -> $LOG"
( cd "$RUN_CASE" && pimpleFoam ) | tee "$LOG" | grep --line-buffered "^METRICS" || true

echo "=== 3D task $i ($MODE) done -> $LOG ==="
echo "    when both finish:  $PYTHON $EXP_DIR/analyze.py $EXP_DIR/results_3d"
