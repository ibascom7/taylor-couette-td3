#!/bin/bash
# =====================================================================
# SMOKE TEST for the CONSTANT-OMEGA pipeline (run_carya_sweep_constant.slurm).
# Run this BEFORE the sweep to confirm the constant env, its per-episode
# logging, and the ParaView frame snapshots all work -- in minutes.
#
# Uses train.py --smoke: 3 short episodes (5 s sim each), captures the time
# dirs of episode 1 and the final episode.
#
# Submit:  sbatch run_carya_smoke_constant.slurm
# Check:   sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed
#
# Throwaway TAG=const_smoke -> results/td3/const_smoke and _runs/td3_const_smoke,
# isolated from real runs. Delete those two dirs when done.
#
# PASS criteria:
#   - sacct State = COMPLETED, ExitCode = 0:0
#   - results/td3/const_smoke/ has td3_tc_final* + the four .npy logs
#   - results/td3/const_smoke/frames/ has ep0001_* and ep0003_* dirs, each
#     with a handful of numeric time dirs + a .foam file (open in ParaView)
# =====================================================================

#SBATCH -J tc_const_smoke         # job name (distinct from the real run)
#SBATCH -o logs/%x_%j.out         # stdout: tc_const_smoke_<jobid>.out
#SBATCH -e logs/%x_%j.err         # stderr
#SBATCH -N 1                      # one node
#SBATCH -n 1                      # one (serial) task
#SBATCH -c 48                     # CPU cores (match the real run)
#SBATCH --mem=16G                 # match the real run
#SBATCH -t 0:30:0                 # 30 min is plenty for a 3-episode run
# #SBATCH -A <your_allocation>    # TODO: uncomment/set if you have multiple allocations
# #SBATCH -p <partition>          # TODO: set if a non-default partition is required (check: sinfo)

set -euo pipefail

# ---- fixed test config (throwaway tag, tiny budget via --smoke) ----
ALGO=td3
TAG=const_smoke         # throwaway -> results/td3/const_smoke, isolated

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
RUN_ROOT="${SLURM_TMPDIR:-$SLURM_SUBMIT_DIR/_runs}/${ALGO}_${TAG}"
RUN_CASE="$RUN_ROOT/tc_mixing_case"
rm -rf "$RUN_ROOT"; mkdir -p "$RUN_ROOT"
cp -r "$MASTER_CASE" "$RUN_CASE"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "=== CONSTANT SMOKE TEST: algo=$ALGO tag=$TAG ==="
echo "    case copy: $RUN_CASE"
echo "    host: $(hostname)  cores: ${SLURM_CPUS_PER_TASK:-?}"

# ---- run -----------------------------------------------------------
"$PYTHON" train.py \
    --algo "$ALGO" \
    --env constant \
    --tag "$TAG" \
    --case_path "$RUN_CASE" \
    --smoke

echo "=== constant smoke test done ==="
echo "--- results produced: ---"
ls -la results/${ALGO}/${TAG}/ || true
echo "--- captured frame dirs (want ep0001_* and ep0003_*): ---"
ls -la results/${ALGO}/${TAG}/frames/ || true
