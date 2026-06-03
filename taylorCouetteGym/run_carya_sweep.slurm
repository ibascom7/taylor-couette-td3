#!/bin/bash
# =====================================================================
# Carya (UH RCDC) job-array script: TD3 HYPERPARAMETER SWEEP on
# TaylorCouetteMixing (no DDPG).
#
# Each array task is the SAME algorithm (TD3) at the SAME seed, differing
# in ONE hyperparameter from a common baseline ("base" is the control).
# That makes the comparison clean: any difference vs. base is attributable
# to the single knob that changed.
#
# Outputs go to results/td3/<tag>/ (one dir per config, via train.py --tag),
# so concurrent same-seed runs do NOT overwrite each other.
#
# Like run_carya.slurm, every task gets its OWN copy of the OpenFOAM case,
# because the env mutates the case dir in place.
#
# Submit:   sbatch run_carya_sweep.slurm
# Monitor:  squeue -u $USER        Cancel: scancel <jobid>
# Plot:     python plot_sweep.py   (after / during the array -- logs are
#           re-written every SAVE_EVERY steps, so partial curves are usable)
#
# pimpleFoam here is SERIAL (the env never calls mpirun/-parallel), so each
# task is a single-process CFD + a tiny RL net -> a few CPU cores, no GPU.
# =====================================================================

#SBATCH -J tc_td3_sweep           # job name (distinct from the real run)
#SBATCH -o logs/%x_%A_%a.out      # stdout: <name>_<jobid>_<arrayidx>.out
#SBATCH -e logs/%x_%A_%a.err      # stderr
#SBATCH -N 1                      # one node per task
#SBATCH -n 1                      # one (serial) task
#SBATCH -c 48                     # CPU cores per task (OpenFOAM/numpy threads)
#SBATCH --mem=16G                 # match run_carya.slurm
#SBATCH -t 40:0:0                 # walltime -- the REAL step ceiling (see MAX_TIMESTEPS note)
#SBATCH --array=0-3               # 4 TD3 configs (see mapping below)
#SBATCH --mail-type=ALL,ARRAY_TASKS   # one mail PER array task
#SBATCH --mail-user=bascomisaiah@gmail.com
# #SBATCH -A <your_allocation>    # TODO: uncomment/set if you have multiple allocations
# #SBATCH -p <partition>          # TODO: set if a non-default partition is required (check: sinfo)

set -euo pipefail

# ---- sweep grid: array index -> (tag, hyperparameters) -------------
# Baseline ("base") is the control. Each other cell changes exactly ONE knob:
#   index 0  base      -- expl 0.1, batch 256, episode len 60
#   index 1  expl0.3   -- more exploration noise
#   index 2  batch512  -- larger, more stable critic/actor updates
#   index 3  ep120     -- longer horizon per episode (mixing has more time to finish)
TAGS=(   base   expl0.3  batch512  ep120 )
EXPL=(   0.1    0.3      0.1       0.1   )
BATCH=(  256    256      512       256   )
EP_LEN=( 60     60       60        120   )

i=${SLURM_ARRAY_TASK_ID:-0}
TAG=${TAGS[$i]}
EXPL_NOISE=${EXPL[$i]}
BATCH_SIZE=${BATCH[$i]}
MAX_STEPS_PER_EP=${EP_LEN[$i]}

# ---- training hyperparameters (shared across the sweep) ------------
ALGO=td3
SEED=0                  # fixed across configs -> differences are due to the knob, not the seed
START_TIMESTEPS=1000    # more random warmup than the old default (120) before learning starts
SAVE_EVERY=500          # checkpoints + .npy logs re-written this often -> partial curves survive a walltime kill
# MAX_TIMESTEPS is set HIGH on purpose: per-step wall time on Carya is the
# unknown, so we let the -t walltime above be the real limiter. The job runs
# as many steps as fit in 40 h; if it's killed at the wall, the latest
# SAVE_EVERY checkpoint + logs are already on disk. Raise -t (and this) if your
# QOS allows a longer wall and you still want more steps.
MAX_TIMESTEPS=12000     # 4x the old 3000-step run (= up to 200 episodes at ep_len 60)

# =====================================================================
# OpenFOAM environment -- IDENTICAL to run_carya.slurm.
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
# (Same rationale as run_carya.slurm.)
# ---------------------------------------------------------------------
PYTHON=/project/mwang/ibascom/envs/tc/bin/python   # <-- match run_carya.slurm

# ---- per-task isolated case copy (keyed by tag, not seed) ----------
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
MASTER_CASE="taylor_couette_mixing/cases/tc_mixing_case"
RUN_ROOT="${SLURM_TMPDIR:-$SLURM_SUBMIT_DIR/_runs}/${ALGO}_${TAG}"
RUN_CASE="$RUN_ROOT/tc_mixing_case"
rm -rf "$RUN_ROOT"; mkdir -p "$RUN_ROOT"
cp -r "$MASTER_CASE" "$RUN_CASE"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "=== sweep task $i: tag=$TAG  expl=$EXPL_NOISE batch=$BATCH_SIZE ep_len=$MAX_STEPS_PER_EP ==="
echo "    case copy: $RUN_CASE"
echo "    host: $(hostname)  cores: ${SLURM_CPUS_PER_TASK:-?}"

# ---- run -----------------------------------------------------------
# Results land in results/td3/<tag>/ (persistent), regardless of scratch case.
"$PYTHON" train.py \
    --algo "$ALGO" \
    --seed "$SEED" \
    --tag "$TAG" \
    --case_path "$RUN_CASE" \
    --max_timesteps "$MAX_TIMESTEPS" \
    --max_steps_per_ep "$MAX_STEPS_PER_EP" \
    --start_timesteps "$START_TIMESTEPS" \
    --expl_noise "$EXPL_NOISE" \
    --batch_size "$BATCH_SIZE" \
    --save_every "$SAVE_EVERY"

echo "=== sweep task $i ($TAG) done ==="
