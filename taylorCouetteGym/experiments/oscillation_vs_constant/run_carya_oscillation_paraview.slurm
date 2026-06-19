#!/bin/bash
# =====================================================================
# Carya: wedge runs whose field snapshots are kept for ParaView AND whose METRICS
# logs drive the figures -- this is now the ONLY job for the wedge study
# (run_carya_oscillation_vs_constant.slurm is retired).
#
#   * runs the case IN A PERSISTENT folder (experiments/.../paraview/<mode>),
#     NOT $SLURM_TMPDIR -- so the 1/ 2/ 3/... time dirs survive even on timeout,
#   * writes results/<mode>.log (what make_figures.py / analyze.py read) so the
#     same run produces both the time series and the ParaView fields,
#   * drops a case.foam file you point ParaView at.
#
# Open in ParaView:  paraFoam -case <dir>   OR open <dir>/case.foam
# You'll see the dye field C and velocity U on the r-z wedge face -- the dye
# plume descending and the Taylor vortices folding it, animatable over time.
#
# Submit:  sbatch run_carya_oscillation_paraview.slurm
# =====================================================================

#SBATCH -J tc_osc_paraview
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

RUN_SECONDS=120           # dye breakthrough + ~1.5 modulation periods; bump if you want more frames
MEAN_RPM=500; DUTY=0.2; PERIOD=30; MAX_CO=0.8

MODES=( constant squarewave )
i=${SLURM_ARRAY_TASK_ID:-0}
MODE=${MODES[$i]}

OPENFOAM_MODULE="OpenFOAM/v2506-foss-2025b"
set +u; module purge; module load "$OPENFOAM_MODULE"; source "$FOAM_BASH"; set -u
unset PYTHONPATH PYTHONHOME
PYTHON=/project/mwang/ibascom/envs/tc/bin/python

cd "$SLURM_SUBMIT_DIR"
EXP_DIR="experiments/oscillation_vs_constant"
mkdir -p "$EXP_DIR/logs" "$EXP_DIR/results"

# ---- case lives on PERSISTENT disk so fields survive ----------------
MASTER_CASE="taylor_couette_mixing/cases/tc_mixing_case"
RUN_CASE="$EXP_DIR/paraview/${MODE}/tc_mixing_case"
rm -rf "$EXP_DIR/paraview/${MODE}"; mkdir -p "$EXP_DIR/paraview/${MODE}"
cp -r "$MASTER_CASE" "$RUN_CASE"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1

echo "=== paraview $MODE: run=${RUN_SECONDS}s -> $RUN_CASE ==="

"$PYTHON" "$EXP_DIR/make_case.py" \
    --case "$RUN_CASE" --geometry wedge --mode "$MODE" \
    --mean-rpm "$MEAN_RPM" --duty "$DUTY" --period "$PERIOD" \
    --run-seconds "$RUN_SECONDS" --max-co "$MAX_CO"

( cd "$RUN_CASE" && blockMesh )
# solve in place: every snapshot is written straight to persistent disk.
# Tee the METRICS lines to BOTH (a) results/<mode>.log -- what make_figures.py /
# analyze.py read (they glob *.log) -- and (b) paraview/<mode>.metrics, so this one
# job drives the figures AND keeps the ParaView fields. (run_carya_oscillation_
# vs_constant.slurm is no longer needed.)
LOG="$EXP_DIR/results/${MODE}.log"
( cd "$RUN_CASE" && pimpleFoam ) | grep --line-buffered "^METRICS" \
    | tee "$LOG" > "$EXP_DIR/paraview/${MODE}.metrics" || true

touch "$RUN_CASE/case.foam"        # ParaView entry point
echo "=== $MODE done. log -> $LOG ;  ParaView: $RUN_CASE/case.foam ==="
echo "    when BOTH tasks finish:  $PYTHON $EXP_DIR/make_figures.py $EXP_DIR/results"
ls -d "$RUN_CASE"/[0-9]* | tail