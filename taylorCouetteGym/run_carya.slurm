#!/bin/bash
# =====================================================================
# Carya (UH RCDC) job-array script: TD3 vs DDPG on TaylorCouetteMixing.
#
# Launches one independent run per (algo, seed). Each array task gets its
# OWN copy of the OpenFOAM case, because the env mutates the case dir in
# place (writes/clears time dirs every step) -- sharing one case across
# concurrent tasks would corrupt them.
#
# Submit:   sbatch run_carya.slurm
# Monitor:  squeue -u $USER        Cancel: scancel <jobid>
# Plot:     python plot_comparison.py   (after the array finishes)
#
# pimpleFoam here is SERIAL (the env never calls mpirun/-parallel), so each
# task is a single-process CFD + a tiny RL net -> a few CPU cores, no GPU.
# =====================================================================

#SBATCH -J tc_td3_ddpg            # job name
#SBATCH -o logs/%x_%A_%a.out      # stdout: <name>_<jobid>_<arrayidx>.out
#SBATCH -e logs/%x_%A_%a.err      # stderr
#SBATCH -N 1                      # one node per task
#SBATCH -n 1                      # one (serial) task
#SBATCH -c 48                      # CPU cores per task (OpenFOAM/numpy threads)
#SBATCH --mem=8G                  # case is tiny; pimpleFoam is modest
#SBATCH -t 24:0:0               # walltime -- TUNE to your step timing & queue cap
#SBATCH --array=0-5               # 6 tasks = 2 algos x 3 seeds (see mapping below)
# #SBATCH -A <your_allocation>    # TODO: uncomment/set if you have multiple allocations
# #SBATCH -p <partition>          # TODO: set if a non-default partition is required (check: sinfo)

set -euo pipefail

# ---- experiment grid: array index -> (algo, seed) ------------------
ALGOS=(td3 ddpg)
SEEDS=(0 1 2)
N_SEEDS=${#SEEDS[@]}
i=${SLURM_ARRAY_TASK_ID:-0}
ALGO=${ALGOS[$(( i / N_SEEDS ))]}
SEED=${SEEDS[$(( i % N_SEEDS ))]}

# ---- training hyperparameters --------------------------------------
MAX_TIMESTEPS=3000
START_TIMESTEPS=120
SAVE_EVERY=500

# =====================================================================
# TODO #1 -- OpenFOAM environment. Confirm the exact name with:
#   module avail openfoam
# Most OpenFOAM modules auto-source the foam bashrc; if yours does not,
# uncomment the `source` line.
# ---------------------------------------------------------------------
OPENFOAM_MODULE="OpenFOAM/v2506-foss-2025b"
set +u                                # OpenFOAM's bashrc references unset vars (e.g. WM_PROJECT_SITE); set -u would abort
module purge
module load "$OPENFOAM_MODULE"
source "$FOAM_BASH"                    # EasyBuild only stages deps on load; this activates pimpleFoam/wmake
set -u
unset PYTHONPATH PYTHONHOME            # OpenFOAM's SciPy-bundle dep leaks EasyBuild py3.13 paths that break conda numpy

# =====================================================================
# Python: call the conda env's interpreter by ABSOLUTE PATH -- no activation.
# `module load OpenFOAM` prepends EasyBuild's Python to PATH, so a bare `python`
# (or `source activate`, which can also abort the job under `set -e`) is
# unreliable here. An absolute path is immune: Python loads packages from its
# own location, not PATH. Env was built once with:
#   module add Miniforge3/py3.10
#   export CONDA_PKGS_DIRS=/project/mwang/ibascom/conda_cache_dir
#   conda create -p /project/mwang/ibascom/envs/tc python==3.11
#   conda activate /project/mwang/ibascom/envs/tc
#   pip install -r ../requirements-carya.txt
# ---------------------------------------------------------------------
PYTHON=/project/mwang/ibascom/envs/tc/bin/python   # <-- adjust if you named the env differently

# ---- per-task isolated case copy -----------------------------------
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
MASTER_CASE="taylor_couette_mixing/cases/tc_mixing_case"
# Node-local scratch is best for OpenFOAM's many-small-files I/O if Carya
# exposes one (e.g. $TMPDIR or $SLURM_TMPDIR); fall back to a submit-dir copy.
RUN_ROOT="${SLURM_TMPDIR:-$SLURM_SUBMIT_DIR/_runs}/${ALGO}_s${SEED}"
RUN_CASE="$RUN_ROOT/tc_mixing_case"
rm -rf "$RUN_ROOT"; mkdir -p "$RUN_ROOT"
cp -r "$MASTER_CASE" "$RUN_CASE"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "=== task $i: algo=$ALGO seed=$SEED ==="
echo "    case copy: $RUN_CASE"
echo "    host: $(hostname)  cores: ${SLURM_CPUS_PER_TASK:-?}"

# ---- run -----------------------------------------------------------
# Results land in results/<algo>/seed<seed>/ next to train.py (persistent),
# regardless of where the scratch case lives.
"$PYTHON" train.py \
    --algo "$ALGO" \
    --seed "$SEED" \
    --case_path "$RUN_CASE" \
    --max_timesteps "$MAX_TIMESTEPS" \
    --start_timesteps "$START_TIMESTEPS" \
    --save_every "$SAVE_EVERY"

echo "=== task $i done ==="
