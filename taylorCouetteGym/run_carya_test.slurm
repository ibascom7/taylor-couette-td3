#!/bin/bash
# =====================================================================
# SMOKE TEST for the Carya TD3/DDPG pipeline. Run this BEFORE the full
# run_carya.slurm to confirm a config change (e.g. controlDict
# writeInterval) didn't break the env's step loop -- in minutes, not hours.
#
# Submit:  sbatch run_carya_test.slurm
# Check:   sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed
#
# Single task, tiny budget, short walltime. Uses a throwaway SEED=99 so it
# writes to results/td3/seed99 and _runs/td3_s99 -- it will NOT overwrite
# any real run's outputs. Delete those two dirs when you're done.
#
# PASS criteria:
#   - sacct State = COMPLETED, ExitCode = 0:0   (loop didn't desync / crash)
#   - results/td3/seed99/ has td3_tc_final* + the four .npy logs
#   - the case dir holds a HANDFUL of time dirs, not tens of thousands:
#       ls _runs/td3_s99/tc_mixing_case | grep -cE '^[0-9]'
# =====================================================================

#SBATCH -J tc_smoketest           # job name (distinct from the real run)
#SBATCH -o logs/%x_%j.out         # stdout: tc_smoketest_<jobid>.out
#SBATCH -e logs/%x_%j.err         # stderr
#SBATCH -N 1                      # one node
#SBATCH -n 1                      # one (serial) task
#SBATCH -c 48                     # CPU cores (match the real run)
#SBATCH --mem=16G                 # match the real run
#SBATCH -t 0:30:0                 # 30 min is plenty for a 10-step run
# #SBATCH -A <your_allocation>    # TODO: uncomment/set if you have multiple allocations
# #SBATCH -p <partition>          # TODO: set if a non-default partition is required (check: sinfo)

set -euo pipefail

# ---- fixed test config (throwaway seed, tiny budget) ---------------
ALGO=td3
SEED=99                 # throwaway -> results/td3/seed99, isolated from real runs
MAX_TIMESTEPS=10        # exercises the env step loop + final save
START_TIMESTEPS=5       # <MAX so the policy.train() path also runs
SAVE_EVERY=5            # <MAX so a periodic checkpoint save also fires

# =====================================================================
# OpenFOAM environment (identical to run_carya.slurm).
# ---------------------------------------------------------------------
OPENFOAM_MODULE="OpenFOAM/v2506-foss-2025b"
set +u                                # OpenFOAM's bashrc references unset vars; set -u would abort
module purge
module load "$OPENFOAM_MODULE"
source "$FOAM_BASH"                    # activates pimpleFoam/wmake
set -u
unset PYTHONPATH PYTHONHOME            # avoid EasyBuild py3.13 paths breaking conda numpy

PYTHON=/project/mwang/ibascom/envs/tc/bin/python   # <-- match run_carya.slurm

# ---- isolated case copy --------------------------------------------
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
MASTER_CASE="taylor_couette_mixing/cases/tc_mixing_case"
RUN_ROOT="${SLURM_TMPDIR:-$SLURM_SUBMIT_DIR/_runs}/${ALGO}_s${SEED}"
RUN_CASE="$RUN_ROOT/tc_mixing_case"
rm -rf "$RUN_ROOT"; mkdir -p "$RUN_ROOT"
cp -r "$MASTER_CASE" "$RUN_CASE"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "=== SMOKE TEST: algo=$ALGO seed=$SEED steps=$MAX_TIMESTEPS ==="
echo "    case copy: $RUN_CASE"
echo "    host: $(hostname)  cores: ${SLURM_CPUS_PER_TASK:-?}"

# ---- run -----------------------------------------------------------
"$PYTHON" train.py \
    --algo "$ALGO" \
    --seed "$SEED" \
    --case_path "$RUN_CASE" \
    --max_timesteps "$MAX_TIMESTEPS" \
    --start_timesteps "$START_TIMESTEPS" \
    --save_every "$SAVE_EVERY"

echo "=== smoke test done ==="
echo "--- time dirs written (want a handful, not thousands): ---"
ls "$RUN_CASE" | grep -cE '^[0-9]' || true
echo "--- results produced: ---"
ls -la results/${ALGO}/seed${SEED}/ || true
