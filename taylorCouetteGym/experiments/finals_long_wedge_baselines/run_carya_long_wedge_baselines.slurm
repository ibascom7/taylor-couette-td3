#!/bin/bash
# =====================================================================
# Carya: FINALS baseline grid, LONG WEDGE cell (Gamma=30).
#
# Same nine regime-placed waveforms as the short-wedge grid, on
# full_tc_cat_case (the results_full_tc reference case). TIME BASE:
# pulse period T = tau = 130 s (the TD3 block on this cell), episodes
# 5*tau = 650 s, X window = last full period [520, 650] s.
# ONE EPISODE PER ARRAY TASK -- costs span ~45 h to ~230 h
# (dt Courant-scales with omega), so tasks must finish independently:
#
#   index  tag            mean rpm   est. wall (650 s episode)
#   0-2    constant_{tvf,wvf,mwvf}   300/750/1500   ~45 h / ~110 h / ~225 h
#   3-5    inband_{tvf,wvf,mwvf}     300/750/1500   ~45 h / ~110 h / ~225 h
#   6-8    cross_{tvf,wvf,mwvf}      300/750/1500   ~45 h / ~110 h / ~225 h
#
# The runner is invoked with --resume, so a WALLTIME KILL IS RECOVERABLE:
# resubmit the killed indices with the same script and each episode
# continues from its latest saved time folder (the mwvf tasks 2,5,8 WILL
# need 1-2 resubmits at -t 160 h):
#     sbatch --array=2,5,8 experiments/finals_long_wedge_baselines/run_carya_long_wedge_baselines.slurm
# If the partition caps walltime lower, lower -t and resubmit more often.
#
# After ALL tasks finish, build the combined table/figures (cheap, run
# anywhere):
#     python3 experiments/finals_long_wedge_baselines/run_long_wedge_baselines.py --analyze-only
#
# Submit FROM taylorCouetteGym/ (paths are $SLURM_SUBMIT_DIR-relative):
#     cd /project/.../taylor-couette-td3/taylorCouetteGym
#     sbatch experiments/finals_long_wedge_baselines/run_carya_long_wedge_baselines.slurm
# Monitor: tail -f experiments/finals_long_wedge_baselines/logs/*.out
# =====================================================================

#SBATCH -J finals_lwb
#SBATCH -o experiments/finals_long_wedge_baselines/logs/%x_%A_%a.out
#SBATCH -e experiments/finals_long_wedge_baselines/logs/%x_%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 160:0:0
#SBATCH --array=0-8
#SBATCH --mail-type=ALL,ARRAY_TASKS
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

IDX=${SLURM_ARRAY_TASK_ID:-0}
echo "=== finals long-wedge baseline: array index $IDX, host=$(hostname) ==="
"$PYTHON" experiments/finals_long_wedge_baselines/run_long_wedge_baselines.py \
    --index "$IDX" --resume

echo "=== task $IDX done. results -> experiments/finals_long_wedge_baselines/results/ ==="
