#!/bin/bash
# =====================================================================
# Carya: ENDGAME E -- Lopez-Guajardo's optimum on OUR Gamma=30 reactor,
# at our Sc=1075 and at their Sc=1e4 (D = nu/Sc set in transportProperties
# + the scalarTransport FO). full_tc_cat_case, 300 s, pristine IC,
# ONE EPISODE PER ARRAY TASK:
#   0  lg_opt_D20_T30_sc1075   0->2500 rpm D=0.20 T=30 s  (their Fig. 6 optimum)
#   1  lg_min_D80_T5_sc1075    0->625  rpm D=0.80 T=5 s   (their local minimum)
#   2  constant_500_sc1075
#   3  lg_opt_D20_T30_sc1e4
#   4  lg_min_D80_T5_sc1e4
#   5  constant_500_sc1e4
# COST: ~22-26 h at <= 625 rpm; the 2500 rpm-burst rows (0, 3) Courant-
# shrink dt -> budget ~40-60 h. --resume makes walltime kills recoverable:
#     sbatch --array=0,3 experiments/endgame/run_carya_E_lopez_optimum.slurm
# After all tasks: python3 experiments/endgame/gamma30_statics.py --group E --analyze-only
#
# Submit FROM taylorCouetteGym/:
#     sbatch experiments/endgame/run_carya_E_lopez_optimum.slurm
# =====================================================================

#SBATCH -J endgame_E
#SBATCH -o experiments/endgame/logs/%x_%A_%a.out
#SBATCH -e experiments/endgame/logs/%x_%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 96:0:0
#SBATCH --array=0-5
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --mail-user=bascomisaiah@gmail.com

set -euo pipefail
OPENFOAM_MODULE="OpenFOAM/v2506-foss-2025b"
set +u; module purge; module load "$OPENFOAM_MODULE"; source "$FOAM_BASH"; set -u
unset PYTHONPATH PYTHONHOME
PYTHON=/project/mwang/ibascom/envs/tc/bin/python
cd "$SLURM_SUBMIT_DIR"
mkdir -p experiments/endgame/logs
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

IDX=${SLURM_ARRAY_TASK_ID:-0}
echo "=== ENDGAME E: Lopez-Guajardo optimum on Gamma=30, index $IDX, host=$(hostname) ==="
"$PYTHON" experiments/endgame/gamma30_statics.py --group E --index "$IDX" --resume
echo "=== E task $IDX done -> experiments/endgame/results/E ==="
