#!/bin/bash
# =====================================================================
# Carya: ENDGAME B -- champion-family statics on the Gamma=30 reactor.
#
# The transfer eval's comparators (constant-300, pulsed D80/T10) never
# included the short-reactor champion family. Three fixed-mean-300
# waveforms on full_tc_cat_case, 300 s from the pristine IC (the
# results_full_tc / eval_full_tc convention), ONE EPISODE PER ARRAY TASK:
#   0  champ_D90_T5     0->333 rpm  D=0.90 T=5 s
#   1  champ_D80_T2p5   0->375 rpm  D=0.80 T=2.5 s
#   2  shimmy_D63_T5    0->476 rpm  D=0.63 T=5 s   (the policy's plateau)
# COST ~22-30 h each on one core. --resume makes a walltime kill
# recoverable (resubmit the killed indices: sbatch --array=2 ...).
# After all tasks: python3 experiments/endgame/gamma30_statics.py --group B --analyze-only
#
# Submit FROM taylorCouetteGym/:
#     sbatch experiments/endgame/run_carya_B_gamma30_champion.slurm
# =====================================================================

#SBATCH -J endgame_B
#SBATCH -o experiments/endgame/logs/%x_%A_%a.out
#SBATCH -e experiments/endgame/logs/%x_%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 48:0:0
#SBATCH --array=0-2
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
echo "=== ENDGAME B: Gamma=30 champion statics, index $IDX, host=$(hostname) ==="
"$PYTHON" experiments/endgame/gamma30_statics.py --group B --index "$IDX" --resume
echo "=== B task $IDX done -> experiments/endgame/results/B ==="
