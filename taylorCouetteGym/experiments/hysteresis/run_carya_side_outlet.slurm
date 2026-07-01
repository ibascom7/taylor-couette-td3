#!/bin/bash
# =====================================================================
# Carya: hysteresis / bistability sweep on the SIDE-OUTLET catalysis case.
#
# Same up/down Reynolds-ladder sweep as run_carya_hysteresis.slurm, but on
# `side_outlet_cat_case` -- Yuhe's shortened flow-through reactor (top inlet,
# closed bottom, bottom outer-wall SIDE outlet, catalytic outer wall). The case
# ships PRE-CONFIGURED for the env/sweep interface (rotatingWallVelocity inner
# wall with a settable omega, scalar C fed at c0=1, and an rlMetrics function
# object that logs `METRICS ... Mz_kin= conv= wallFlux=`), so unlike the wedge
# job this one does NOT call make_case.py -- it just copies the case and meshes.
#
# Looks for a LOOP in the conversion-vs-power plane (the mechanism by which a
# TIME-MODULATED speed can beat a CONSTANT one at the same mean power). The array
# runs the SAME loop at several SETTLE TIMES; loop area vs settle time is the key
# diagnostic (analyze_hysteresis.py):
#     area -> nonzero as settle grows  => TRUE (equilibrium) hysteresis / bistability
#     area -> 0       as settle grows  => DYNAMIC hysteresis (transient lag)
#
# NOTE on settle times: this reactor's mean residence time is tau = H/U ~ 65 s
# (H=0.0381 m, U=Q/A=5.85e-4 m/s), much longer than the closed-cell wedge. The
# IC is a full reservoir (C=1), so conversion needs ~2-3 tau to reach steady
# state at each level. Settles below are scaled accordingly (30 / 90 / 180 s);
# the 30 s task is the deliberately-fast end of the rate ladder.
#
# Submit:  sbatch experiments/hysteresis/run_carya_side_outlet.slurm
# Then:    python experiments/hysteresis/analyze_hysteresis.py \
#                 experiments/hysteresis/results_side_outlet/so*    (loop-area-vs-rate + overlay)
# =====================================================================

#SBATCH -J tc_side_outlet
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
SETTLE_TIMES=(30 90 180)
i=${SLURM_ARRAY_TASK_ID:-0}
SETTLE=${SETTLE_TIMES[$i]}
TAG="so${SETTLE}"

# Speed ladder (rpm); Re ~= 1.571*rpm. Dense near the Taylor-vortex onset (~76 rpm).
RPM_LADDER="40,60,80,100,120,160,220,300,500,800,1200,1800,2500"
RAMP_TIME=0.10

# Fixed-Re two-IC bistability probe -- only on the longest-settle task (cheapest to
# trust). Empty on the others. Pick speeds in the band where the loop looks widest.
PROBE_RPM=""
[ "$SETTLE" = "180" ] && PROBE_RPM="100,160,300,500"

# ---- geometry / fluid (match transportProperties + blockMesh) ------
R_IN=25.4
R_OUT=31.75
NU=1.075e-5
SCALAR_D=2.8e-8                  # this case's diffusivity (Yuhe's transportProperties)
FEED_VELOCITY=5.847e-4           # U_mean = Q0/A_ann = (40 mL/min)/(pi(Ro^2-Ri^2)); sets wallflux_ref

# =====================================================================
OPENFOAM_MODULE="OpenFOAM/v2506-foss-2025b"
set +u; module purge; module load "$OPENFOAM_MODULE"; source "$FOAM_BASH"; set -u
unset PYTHONPATH PYTHONHOME
PYTHON=/project/mwang/ibascom/envs/tc/bin/python

cd "$SLURM_SUBMIT_DIR"
EXP_DIR="experiments/hysteresis"
mkdir -p "$EXP_DIR/logs"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1

# ---- isolated case copy (per array task) ---------------------------
# The case is already configured for the sweep (rotatingWallVelocity omega,
# C scalar, METRICS function object), so we just copy + mesh -- no make_case.py.
MASTER_CASE="taylor_couette_mixing/cases/side_outlet_cat_case"
RUN_ROOT="$EXP_DIR/_runs/${TAG}"
RUN_CASE="$RUN_ROOT/side_outlet_cat_case"
rm -rf "$RUN_ROOT"; mkdir -p "$RUN_ROOT"
cp -r "$MASTER_CASE" "$RUN_CASE"

echo "=== side-outlet $TAG: settle=${SETTLE}s  probe='${PROBE_RPM}'  case=$RUN_CASE  host=$(hostname) ==="

# fresh cold start from the pristine IC; keep 0.orig as the template, drop any warm cache
rm -rf "$RUN_CASE/0.warmed" "$RUN_CASE/0.cold"
rm -rf "$RUN_CASE"/[1-9]* "$RUN_CASE"/postProcessing "$RUN_CASE"/dynamicCode
rm -rf "$RUN_CASE/0"; cp -r "$RUN_CASE/0.orig" "$RUN_CASE/0"

# diffusivity: keep transportProperties.D and the scalarTransport.D entry in sync
foamDictionary -entry D -set "D [0 2 -1 0 0 0 0] $SCALAR_D" \
    "$RUN_CASE/constant/transportProperties" >/dev/null
foamDictionary -entry "functions.scalarTransport.D" -set "$SCALAR_D" \
    "$RUN_CASE/system/controlDict" >/dev/null

( cd "$RUN_CASE" && blockMesh )

# ---- run the up/down loop ------------------------------------------
OUT="$EXP_DIR/results_side_outlet/$TAG"
"$PYTHON" "$EXP_DIR/hysteresis_sweep.py" \
    --case "$RUN_CASE" --out "$OUT" \
    --rpm "$RPM_LADDER" --settle-seconds "$SETTLE" --ramp-time "$RAMP_TIME" \
    --direction updown \
    --r-in "$R_IN" --r-out "$R_OUT" --nu "$NU" \
    --feed-velocity "$FEED_VELOCITY" --azimuthal-fraction 0.0138888889 \
    --probe-rpm "$PROBE_RPM"

echo "=== side-outlet $TAG done -> $OUT ==="
echo "    after all tasks: python $EXP_DIR/analyze_hysteresis.py $EXP_DIR/results_side_outlet/so*"
