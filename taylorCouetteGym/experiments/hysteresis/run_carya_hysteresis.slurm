#!/bin/bash
# =====================================================================
# Carya: hysteresis / bistability sweep on the CATALYSIS wedge.
#
# Ramps inner-cylinder omega UP a Reynolds ladder and back DOWN, settling at every
# level, to look for a LOOP in the conversion-vs-power plane -- the mechanism by
# which a TIME-MODULATED speed can beat a CONSTANT one at the same mean power.
#
# The array runs the SAME up/down loop at several SETTLE TIMES. Loop area vs settle
# time is the key diagnostic (analyze_hysteresis.py):
#     area -> nonzero as settle grows  => TRUE (equilibrium) hysteresis / bistability
#     area -> 0       as settle grows  => DYNAMIC hysteresis (transient lag)
# Both support modulation>constant; only the first is genuine bistability. The
# longest-settle task ALSO runs the fixed-Re two-IC probe (cold-start vs hot-start).
#
# Submit:  sbatch experiments/hysteresis/run_carya_hysteresis.slurm
# Then:    python experiments/hysteresis/analyze_hysteresis.py \
#                 experiments/hysteresis/results/h*    (loop-area-vs-rate + overlay)
# =====================================================================

#SBATCH -J tc_hysteresis
#SBATCH -o experiments/hysteresis/logs/%x_%A_%a.out
#SBATCH -e experiments/hysteresis/logs/%x_%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 24:0:0
#SBATCH --array=0-2              # 0,1,2 -> SETTLE_TIMES below (the rate-dependence test)
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --mail-user=bascomisaiah@gmail.com

set -euo pipefail

# ---- rate-dependence grid: array index -> settle seconds -----------
SETTLE_TIMES=(15 40 90)
i=${SLURM_ARRAY_TASK_ID:-0}
SETTLE=${SETTLE_TIMES[$i]}
TAG="h${SETTLE}"

# Speed ladder (rpm); Re ~= 1.571*rpm. Dense near the Taylor-vortex onset (~76 rpm).
RPM_LADDER="40,60,80,100,120,160,220,300,500,800,1200,1800,2500"
RAMP_TIME=0.10

# Fixed-Re two-IC bistability probe -- only on the longest-settle task (cheapest to
# trust). Empty on the others. Pick speeds in the band where the loop looks widest.
PROBE_RPM=""
[ "$SETTLE" = "90" ] && PROBE_RPM="100,160,300,500"

# ---- geometry / fluid (match transportProperties + blockMesh) ------
R_IN=25.4
R_OUT=31.75
NU=1.075e-5
SCALAR_D=1e-9

# =====================================================================
OPENFOAM_MODULE="OpenFOAM/v2506-foss-2025b"
set +u; module purge; module load "$OPENFOAM_MODULE"; source "$FOAM_BASH"; set -u
unset PYTHONPATH PYTHONHOME
PYTHON=/project/mwang/ibascom/envs/tc/bin/python

cd "$SLURM_SUBMIT_DIR"
EXP_DIR="experiments/hysteresis"
mkdir -p "$EXP_DIR/logs"
MAKE_CASE="experiments/oscillation_vs_constant/make_case.py"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1

# ---- isolated case copy (per array task) ---------------------------
MASTER_CASE="taylor_couette_mixing/cases/tc_mixing_case"
RUN_ROOT="$EXP_DIR/_runs/${TAG}"
RUN_CASE="$RUN_ROOT/tc_mixing_case"
rm -rf "$RUN_ROOT"; mkdir -p "$RUN_ROOT"
cp -r "$MASTER_CASE" "$RUN_CASE"

echo "=== hysteresis $TAG: settle=${SETTLE}s  probe='${PROBE_RPM}'  case=$RUN_CASE  host=$(hostname) ==="

# ---- configure for catalysis (same as the training slurm) ----------
"$PYTHON" "$MAKE_CASE" \
    --case "$RUN_CASE" --geometry wedge --mode constant --catalysis \
    --mean-rpm 100 --run-seconds "$SETTLE" --scalar-D "$SCALAR_D" --max-co 1.5
( cd "$RUN_CASE" && blockMesh )
rm -rf "$RUN_CASE/0.warmed" "$RUN_CASE/0.orig"

# ---- run the up/down loop ------------------------------------------
OUT="$EXP_DIR/results/$TAG"
"$PYTHON" "$EXP_DIR/hysteresis_sweep.py" \
    --case "$RUN_CASE" --out "$OUT" \
    --rpm "$RPM_LADDER" --settle-seconds "$SETTLE" --ramp-time "$RAMP_TIME" \
    --direction updown \
    --r-in "$R_IN" --r-out "$R_OUT" --nu "$NU" \
    --probe-rpm "$PROBE_RPM"

echo "=== hysteresis $TAG done -> $OUT ==="
echo "    after all tasks: python $EXP_DIR/analyze_hysteresis.py $EXP_DIR/results/h*"
