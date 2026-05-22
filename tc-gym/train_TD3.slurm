#!/bin/bash
#SBATCH -J td3_tc
#SBATCH -o logs/td3_tc.o%j
#SBATCH --mail-user=ibascom@uh.edu
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=4 -N 1
#SBATCH -t 8:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=4GB
# Set your allocation (PI account). Replace with the name RCDC assigned to your group.
#SBATCH -A mwang

set -euo pipefail

# ---- environment --------------------------------------------------------- #
# 1. Load OpenFOAM (provides pimpleFoam + foamDictionary).
module load OpenFOAM/v2506-foss-2025b
module load PyTorch/2.11.0-foss-2025b-CUDA-12.9.1
# OpenFOAM EasyBuild modules typically set FOAM_BASH but don't auto-source
# it. Source it here so pimpleFoam lands on PATH. If `which pimpleFoam`
# already works after `module load` alone, this line is a no-op.
if [[ -n "${FOAM_BASH:-}" && -f "${FOAM_BASH}" ]]; then
    # shellcheck disable=SC1090
    source "${FOAM_BASH}"
fi

# 2. Activate the venv (must come *after* sourcing OpenFOAM so the venv's
#    python is first on PATH instead of OpenFOAM's bundled Python).
PROJECT_ROOT="/project/mwang/ibascom/taylor-couette-td3"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/.venv/bin/activate"

# ---- sanity checks (cheap; fail fast if env is wrong) -------------------- #
which pimpleFoam
which python3
python3 -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"
python3 -c "import sys; sys.path.insert(0, '${PROJECT_ROOT}/tc-gym/src'); import tc_gym; print('tc_gym ok')"

# ---- run ----------------------------------------------------------------- #
cd "${PROJECT_ROOT}"
mkdir -p logs

# Tag the save_dir with the SLURM job id so multiple runs don't collide.
SAVE_DIR="runs/td3_${SLURM_JOB_ID}"

python3 tc-gym/train_TD3.py \
    --case-path tc-gym/cases/tc_mixing_case \
    --save-dir "${SAVE_DIR}" \
    --total-timesteps 3000 \
    --learning-starts 120 \
    --max-steps-per-ep 60 \
    --seed 0

echo "DONE. Artifacts in ${PROJECT_ROOT}/${SAVE_DIR}"
