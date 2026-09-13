#!/bin/bash
# =====================================================================
# Carya: ENDGAME C -- continue the Gamma=30 transfer eval past its
# trained 300 s horizon (3 more residence times = 390 s) with the v5-s2
# policy still in the loop. Tests whether the end-of-run gain over the
# static pulse (X 0.9015 vs 0.876) survives continuous operation.
#   0  source s2_stretch, clock HELD at the last value the policy saw
#   1  source s2_wrap,    clock keeps WRAPPING (cyclic 50 s program)
# The kept 300 s case states are git-tracked under
# experiments/modulation_rl/results/full_tc_eval/<source>/case, so a
# fresh clone has them after git pull. If they are ever missing,
# --ensure_source re-runs eval_full_tc.py --keep_case first (~22 h more;
# hence -t 96 h). COST ~27 h per task (39 x 10 s blocks, 1 core).
# --resume continues from the task's own blocks.csv + latest time folder.
# After: python3 experiments/endgame/c_continue_transfer.py --source s2_stretch --clock hold --analyze-only
#
# Submit FROM taylorCouetteGym/:
#     sbatch experiments/endgame/run_carya_C_continue_transfer.slurm
# =====================================================================

#SBATCH -J endgame_C
#SBATCH -o experiments/endgame/logs/%x_%A_%a.out
#SBATCH -e experiments/endgame/logs/%x_%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 96:0:0
#SBATCH --array=0-1
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
if [ "$IDX" -eq 0 ]; then SRC=s2_stretch; CLOCK=hold; else SRC=s2_wrap; CLOCK=wrap; fi
echo "=== ENDGAME C: continue $SRC ($CLOCK clock), host=$(hostname) ==="
"$PYTHON" experiments/endgame/c_continue_transfer.py \
    --source "$SRC" --clock "$CLOCK" --mode policy --extra_duration 390 \
    --ensure_source --resume
echo "=== C task $IDX done -> experiments/endgame/results/C ==="
