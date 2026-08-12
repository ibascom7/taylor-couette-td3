#!/bin/bash
# =====================================================================
# Carya: FINALS baseline grid, SHORT WEDGE cell (Gamma=6).
#
# Nine regime-placed episodes (constant / in-band pulse / crossing pulse
# in TVF, WVF, MWVF -- see run_short_wedge_baselines.py docstring) on
# side_outlet_case_sc1075_graded, all in one task, 9 concurrent
# single-core pimpleFoam runs (the proven fig7 pattern).
#
# TIME BASE: pulse period T = tau = 26 s (the TD3 block), episodes
# 5*tau = 130 s, X window = last full period [104, 130] s.
#
# COST: episode wall time scales ~ mean omega (Courant dt). Mean-300
# runs ~1.5-3 h, mean-750 ~4-7.5 h, mean-1500 ~8-15 h. Wall clock =
# slowest episode; -t 30 h is margin.
#
# Submit FROM taylorCouetteGym/ (paths are $SLURM_SUBMIT_DIR-relative):
#     cd /project/.../taylor-couette-td3/taylorCouetteGym
#     sbatch experiments/finals_short_wedge_baselines/run_carya_short_wedge_baselines.slurm
# Monitor: tail -f experiments/finals_short_wedge_baselines/logs/*.out
# =====================================================================

#SBATCH -J finals_swb
#SBATCH -o experiments/finals_short_wedge_baselines/logs/%x_%j.out
#SBATCH -e experiments/finals_short_wedge_baselines/logs/%x_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --mem=16G
#SBATCH -t 30:0:0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bascomisaiah@gmail.com

set -euo pipefail

# ---- OpenFOAM + python env: IDENTICAL recipe to the RL slurms -------
OPENFOAM_MODULE="OpenFOAM/v2506-foss-2025b"
set +u; module purge; module load "$OPENFOAM_MODULE"; source "$FOAM_BASH"; set -u
unset PYTHONPATH PYTHONHOME
PYTHON=/project/mwang/ibascom/envs/tc/bin/python

cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "=== finals short-wedge baselines: 9 episodes, 9-wide, host=$(hostname) ==="
"$PYTHON" experiments/finals_short_wedge_baselines/run_short_wedge_baselines.py \
    --workers 9

echo "=== done. results -> experiments/finals_short_wedge_baselines/results/ ==="
